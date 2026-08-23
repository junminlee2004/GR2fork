// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "common/logging/log.h"
#include "common/rdtsc.h"
#include "video_core/skipcache/skipcache.h"

namespace VideoCore::Skipcache {

// Controller thresholds. Engineering priors, tuned by field data - each is a
// named constant so the SESSION reports can be correlated against them.
namespace {
constexpr u64 WindowTargetNs = 250'000'000; // 250 ms
constexpr u32 WindowTargetFlips = 32;
constexpr u32 WarmupMinWindows = 12;
constexpr u32 StableWindowsNeeded = 3;
constexpr u32 LearningMinWindows = 8;
constexpr u32 LearningLifetimeBudget = 40;
constexpr f64 PromoteFloorPct = 0.10;
constexpr f64 ShadowExitFloorPct = 0.06;
constexpr f64 DemoteFloorPct = 0.05;
constexpr f64 MinHitRate = 0.20;
constexpr u32 PromoteStreak = 4;
constexpr u32 DemoteStreakLen = 8;
constexpr u64 ShadowCleanHitsGate = 4096;
constexpr u32 ShadowFramesGate = 600;
constexpr u32 ShadowMaxExtensions = 3;
constexpr u64 ShadowResidencyBudgetNs = 30'000'000'000ull; // 30 s
constexpr u32 AbortSlotInvalidate = 8;
constexpr u32 ForcedVerifyWindowCap = 32;
constexpr u64 QuarantineWindowSpan = 32;
constexpr u32 MaxInvariantViolations = 3;
} // namespace

Framework& Framework::Instance() {
    static Framework instance;
    return instance;
}

void Framework::Init(Mode mode) {
    mode_ = mode;
    requested_mode_.store(static_cast<u8>(mode), std::memory_order_relaxed);
#ifdef _DEBUG
    gpu_thread_ = std::this_thread::get_id();
#endif
    if (mode_ == Mode::Disabled) {
        return;
    }
    // Startup calibration on the boot path, never lazily on a draw.
    tsc_hz_ = Common::EstimateRDTSCFrequency();
    if (tsc_hz_ >= 100'000'000) {
        u64 best = ~0ull;
        for (int i = 0; i < 4096; ++i) {
            const u64 a = Common::FencedRDTSC();
            const u64 b = Common::FencedRDTSC();
            best = std::min(best, b - a);
        }
        tsc_pair_cost_ = best;
        ns_per_cycle_ = 1.0e9 / static_cast<f64>(tsc_hz_);
        pair_ns_ = static_cast<u64>(static_cast<f64>(best) * ns_per_cycle_);
        timing_enabled_ = true;
    } else {
        // Sub-100MHz timestamp source: counters only; enablement then requires
        // hit-rate evidence alone.
        timing_enabled_ = false;
    }
    window_start_ns_ = Now();
    last_session_log_ns_ = window_start_ns_;
    if (!dedup_invalidate_registered_) {
        dedup_invalidate_registered_ = true;
        RegisterInvalidate([](void* self) { static_cast<Framework*>(self)->DedupInvalidateAll(); },
                           this);
    }
    for (auto& cs : caches_) {
        cs.state = State::Learning;
        cs.timer_phase = static_cast<u8>(xorshift_state_ & 0xFF);
        cs.tripwire_phase = static_cast<u8>((xorshift_state_ >> 8) & 0xFF);
    }
    LOG_INFO(Render_Skipcache, "[SkipCache] init mode={} timing={} tsc={}MHz pair_cost={}cy",
             mode_ == Mode::ValidateOnly ? "ValidateOnly" : "Adaptive",
             timing_enabled_ ? "on" : "counters-only", tsc_hz_ / 1'000'000, tsc_pair_cost_);
}

u64 Framework::Now() const {
    if (!timing_enabled_) {
        return 0;
    }
    // Multiply in floating point: cycles * 1e9 overflows u64 after ~5 seconds
    // of TSC at modern clock rates. A double is exact past 2^53 ns (~104 days).
    return static_cast<u64>(static_cast<f64>(Common::FencedRDTSC()) * ns_per_cycle_);
}

bool Framework::ShouldVerify(CacheId id) {
    auto& cs = caches_[static_cast<size_t>(id)];
    if (cs.state == State::Shadow) {
        return true; // predict-then-execute on 100% of would-hits
    }
    if (cs.state != State::Enabled) {
        return false;
    }
    // Forced verifies: first hit after every populate and >=1 per window,
    // capped so pass-interleaved repopulation cannot convert every hit into
    // a verify.
    if (cs.force_verify_next_hit && cs.forced_verifies_this_window < ForcedVerifyWindowCap) {
        cs.force_verify_next_hit = false;
        ++cs.forced_verifies_this_window;
        return true;
    }
    return (++cs.tripwire_decim & 0xFF) == cs.tripwire_phase; // 1/256 tripwire
}

void Framework::RecordVerifyClean(CacheId id) {
    auto& cs = caches_[static_cast<size_t>(id)];
    ++cs.counters.verify_clean;
    cs.consecutive_aborts = 0;
    if (cs.state == State::Shadow) {
        ++cs.shadow_clean_hits_total;
    }
}

void Framework::RecordVerifyAborted(CacheId id) {
    // A racing cross-thread invalidation between hit-decision and compare:
    // not evidence for anything; K consecutive aborts invalidate the slot so
    // the authoritative path runs until a clean verdict lands.
    auto& cs = caches_[static_cast<size_t>(id)];
    ++cs.counters.verify_aborted;
    if (++cs.consecutive_aborts >= AbortSlotInvalidate) {
        cs.consecutive_aborts = 0;
        InvalidateAll();
    }
}

void Framework::RecordDivergence(CacheId id, const char* detail) {
    auto& cs = caches_[static_cast<size_t>(id)];
    ++cs.counters.verify_diverged;
    ++cs.divergences_total;
    cs.consecutive_aborts = 0;
    LOG_ERROR(Render_Skipcache, "[SkipCache] DIVERGENCE cache={} state={} win={} n={} {}",
              CacheName(id), StateName(cs.state), window_id_, cs.divergences_total, detail);
    if (cs.divergences_total >= 2 &&
        window_id_ - cs.last_divergence_window <= QuarantineWindowSpan) {
        Transition(id, cs, State::Quarantined, "second confirmed divergence");
    } else {
        // Output was correct (the slow-path result was served); drop to Shadow
        // to re-earn correctness capital.
        if (cs.state == State::Enabled) {
            Transition(id, cs, State::Shadow, "first confirmed divergence");
            cs.shadow_clean_hits_total = 0;
            cs.shadow_frames_total = 0;
        }
    }
    cs.last_divergence_window = window_id_;
    InvalidateAll();
}

void Framework::DedupCommit(u32 image_index, const DrawToken& entry_token) {
    // Populate re-check: the cross-thread lanes are re-read here; if any moved
    // since slow-path entry the entry stays invalid (seqlock consumer side).
    DedupEntry& e = dedup_[image_index & 0xFF];
    if (gens_.mem_gen.load(std::memory_order_acquire) != entry_token.mem_gen ||
        gens_.tex_gen.load(std::memory_order_acquire) != entry_token.tex_gen) {
        e.valid = false; // mid-build invalidation: entry stays invalid
        return;
    }
    e = DedupEntry{
        .image_index = image_index,
        .tick = entry_token.tick,
        .mem_gen = entry_token.mem_gen,
        .tex_gen = entry_token.tex_gen,
        .img_dirty_gen = entry_token.img_dirty_gen,
        .valid = true,
    };
    NotifyPopulated(CacheId::UpdateImageDedup);
}

f64 Framework::ProjectedNetPct(const CacheState& cs, u64 window_ns) const {
    if (window_ns == 0) {
        return 0.0;
    }
    const CacheCounters& c = cs.counters;
    const auto avg = [](u64 ns, u64 n) { return n ? f64(ns) / f64(n) : 0.0; };
    const f64 avg_hit = avg(c.hit_ns, c.hit_samples);
    const f64 avg_miss = avg(c.miss_ns, c.miss_samples);
    const f64 avg_guard = avg(c.guard_ns, c.guard_samples);
    // benefit = hits*(miss-hit) - eligible*guard; shadow burden is baseline
    // work that would run anyway and is excluded by construction.
    const f64 net_ns = f64(c.hits) * (avg_miss - avg_hit) - f64(c.eligible) * avg_guard;
    return 100.0 * net_ns / f64(window_ns);
}

u64 Framework::RollingMedianEligible(const CacheState& cs) const {
    if (cs.eligible_ring_n == 0) {
        return 0;
    }
    std::array<u64, 8> tmp = cs.eligible_ring;
    const u32 n = std::min<u32>(cs.eligible_ring_n, 8);
    std::sort(tmp.begin(), tmp.begin() + n);
    return tmp[n / 2];
}

void Framework::MaybeCloseWindow(bool from_draw_fallback) {
    const u64 now = Now();
    const u64 dur = now - window_start_ns_;
    if (!timing_enabled_) {
        // Counters-only build: close purely on the draw fallback cadence.
        if (from_draw_fallback) {
            CloseWindow();
        }
        return;
    }
    if (dur >= WindowTargetNs) {
        CloseWindow();
    }
}

void Framework::OnSubmit(u32 frame_num, bool guest_paused) {
    if (!Active()) {
        // Mode may still be enabled from the UI while we sit Disabled.
        const Mode req = static_cast<Mode>(requested_mode_.load(std::memory_order_relaxed));
        if (req != mode_) {
            Init(req);
        }
        return;
    }
    if (guest_paused) {
        window_poisoned_ = true;
    }
    const u64 now = Now();
    const bool time_up = timing_enabled_ && now - window_start_ns_ >= WindowTargetNs;
    const bool flips_up = frame_num - window_start_frame_ >= WindowTargetFlips;
    if (time_up || flips_up) {
        CloseWindow();
        window_start_frame_ = frame_num;
    }
}

void Framework::CloseWindow() {
    const u64 now = Now();
    const u64 dur = timing_enabled_ ? now - window_start_ns_ : 1;
    const bool overlong = timing_enabled_ && dur > 4 * WindowTargetNs;
    ++window_id_;

    // Warmup stability: draws/window EWMA variance settled.
    const f64 draws = f64(window_draws_);
    const f64 alpha = 0.3;
    const f64 delta = draws - draws_ewma_;
    draws_ewma_ += alpha * delta;
    draws_ewma_var_ = (1.0 - alpha) * (draws_ewma_var_ + alpha * delta * delta);
    const bool stable = draws_ewma_ > 0 && draws_ewma_var_ < 0.25 * draws_ewma_ * draws_ewma_;
    stable_windows_ = stable ? stable_windows_ + 1 : 0;
    if (!warmed_up_) {
        ++warmup_windows_;
        if (warmup_windows_ >= WarmupMinWindows && stable_windows_ >= StableWindowsNeeded) {
            warmed_up_ = true;
            // Cost models are discarded at warmup exit so shader-compile-storm
            // samples never seed steady-state decisions.
            for (auto& cs : caches_) {
                cs.counters.guard_ns = cs.counters.guard_samples = 0;
                cs.counters.hit_ns = cs.counters.hit_samples = 0;
                cs.counters.miss_ns = cs.counters.miss_samples = 0;
            }
            LOG_INFO(Render_Skipcache, "[SkipCache] warmup complete at win={}", window_id_);
        }
    }

    for (size_t i = 0; i < NumCaches; ++i) {
        const CacheId id = static_cast<CacheId>(i);
        CacheState& cs = caches_[i];
        CacheCounters& c = cs.counters;

        WindowSummary w{};
        w.window_id = window_id_;
        w.state = cs.state;
        w.c = c;
        w.duration_ns = dur;
        w.poisoned = window_poisoned_ || overlong;
        if (!c.AccountingHolds()) {
            w.poisoned = true;
            ++invariant_violations_;
            LOG_WARNING(Render_Skipcache,
                        "[SkipCache] {} accounting violation win={} elig={} hits={} misses={} "
                        "abandoned={}",
                        CacheName(id), window_id_, c.eligible, c.hits, c.Misses(), c.abandoned);
        }
        const u64 med = RollingMedianEligible(cs);
        w.low_signal = med > 0 && c.eligible * 4 < med; // eligible < 25% of median
        w.net_pct = ProjectedNetPct(cs, dur);

        cs.eligible_ring[cs.eligible_ring_n % 8] = c.eligible;
        ++cs.eligible_ring_n;

        if (cs.state == State::Shadow && timing_enabled_) {
            cs.shadow_residency_ns += dur;
        }

        if (invariant_violations_ >= MaxInvariantViolations) {
            // Broken instrumentation means trust nothing.
            LOG_ERROR(Render_Skipcache,
                      "[SkipCache] {} invariant violations - framework off for session",
                      invariant_violations_);
            for (auto& c2 : caches_) {
                c2.state = State::Off;
            }
            mode_ = Mode::Disabled;
            InvalidateAll();
            return;
        }
        if (!w.poisoned) {
            StepController(cs, id, w);
        }
        if (id == CacheId::BindingSkipProbe && (window_id_ & 63) == 0 && c.eligible > 0) {
            LOG_INFO(Render_Skipcache,
                     "[SkipCache] BSPROBE report win={} elig={} wouldhit%={:.1f} "
                     "veto(pipe/tick/stages/pgm/ud)={},{},{},{},{}",
                     window_id_, c.eligible, 100.0 * f64(c.hits) / f64(c.eligible), c.veto[0],
                     c.veto[1], c.veto[2], c.veto[3], c.veto[4]);
        }
        last_window_[i] = w;

        // Full reset per window.
        c = CacheCounters{};
        cs.forced_verifies_this_window = 0;
        // Re-randomize sampling phases so decimation cannot alias frames.
        xorshift_state_ ^= xorshift_state_ << 13;
        xorshift_state_ ^= xorshift_state_ >> 17;
        xorshift_state_ ^= xorshift_state_ << 5;
        cs.timer_phase = static_cast<u8>(xorshift_state_ & 0xFF);
        cs.tripwire_phase = static_cast<u8>((xorshift_state_ >> 8) & 0xFF);
        ++cs.windows_in_state;
    }

    // Apply UI mode changes at the window boundary.
    const Mode req = static_cast<Mode>(requested_mode_.load(std::memory_order_relaxed));
    if (req != mode_) {
        mode_ = req;
        InvalidateAll();
        if (mode_ == Mode::Disabled) {
            for (auto& cs : caches_) {
                cs.state = State::Off;
            }
            LOG_INFO(Render_Skipcache, "[SkipCache] disabled via settings");
        } else {
            for (auto& cs : caches_) {
                if (cs.state == State::Off) {
                    cs.state = State::Learning;
                }
            }
            LOG_INFO(Render_Skipcache, "[SkipCache] enabled via settings");
        }
    }

    // SESSION block every ~10 minutes.
    if (timing_enabled_ && now - last_session_log_ns_ > 120'000'000'000ull) {
        last_session_log_ns_ = now;
        LogSessionSummary();
    }

    window_poisoned_ = false;
    window_draws_ = 0;
    window_start_ns_ = now;
}

void Framework::Transition(CacheId id, CacheState& cs, State next, const char* reason) {
    if (cs.state == next) {
        return;
    }
    const CacheCounters& tc = cs.counters;
    const f64 thr = tc.eligible ? 100.0 * f64(tc.hits) / f64(tc.eligible) : 0.0;
    LOG_INFO(Render_Skipcache,
             "[SkipCache] {} {}->{} win={} reason={} elig={} hit%={:.1f} samples={}/{}/{} "
             "vfy={}/{}/{}",
             CacheName(id), StateName(cs.state), StateName(next), window_id_, reason, tc.eligible,
             thr, tc.guard_samples, tc.hit_samples, tc.miss_samples, tc.verify_clean,
             tc.verify_diverged, tc.verify_aborted);
    // Cold-start is required only when entering a state that consumes cached
    // results.
    if (next == State::Enabled || next == State::Quarantined) {
        InvalidateAll();
    }
    cs.state = next;
    cs.windows_in_state = 0;
}

void Framework::StepController(CacheState& cs, CacheId id, const WindowSummary& w) {
    const CacheCounters& c = w.c;
    const f64 hit_rate = c.eligible ? f64(c.hits) / f64(c.eligible) : 0.0;

    switch (cs.state) {
    case State::Off: {
        if (cs.reprobe_countdown > 0 && --cs.reprobe_countdown == 0) {
            cs.reprobe_backoff = std::min<u32>(cs.reprobe_backoff * 2, 1024);
            Transition(id, cs, State::Learning, "re-probe");
            cs.promote_streak = 0;
        }
        break;
    }
    case State::Learning: {
        if (id == CacheId::BindingSkipProbe) {
            break; // measurement-only: never promotes, never exhausts a budget
        }
        if (!warmed_up_ || w.low_signal) {
            break;
        }
        ++cs.learning_windows_total;
        // A window with too few timing samples cannot price the trade; promote
        // on strong hit-rate evidence alone and let Shadow (which probes at
        // full rate and consumes nothing) price it properly.
        const bool timing_starved = c.miss_samples < 8;
        const bool qualifies = cs.windows_in_state >= 1 &&
                               (w.net_pct >= PromoteFloorPct ? hit_rate >= MinHitRate
                                                             : timing_starved && hit_rate >= 0.40);
        cs.promote_streak = qualifies ? cs.promote_streak + 1 : 0;
        if (cs.promote_streak >= PromoteStreak && cs.windows_in_state >= LearningMinWindows) {
            // Shadow entry does not cold-clear: nothing was being consumed.
            Transition(id, cs, State::Shadow, w.net_pct >= PromoteFloorPct ? "profit" : "hit-rate");
            cs.shadow_clean_hits_total = 0;
            cs.shadow_frames_total = 0;
            cs.shadow_extensions = 0;
            cs.shadow_residency_ns = 0;
        } else if (cs.learning_windows_total >= LearningLifetimeBudget) {
            Transition(id, cs, State::Off, "learning budget exhausted");
            cs.reprobe_countdown = cs.reprobe_backoff;
            cs.learning_windows_total = 0;
        }
        break;
    }
    case State::Shadow: {
        cs.shadow_frames_total += WindowTargetFlips; // approximation: frames per window
        if (w.net_pct < ShadowExitFloorPct) {
            cs.demote_streak++;
            if (cs.demote_streak >= 4) {
                Transition(id, cs, State::Learning, "below shadow floor");
                cs.demote_streak = 0;
                break;
            }
        } else {
            cs.demote_streak = 0;
        }
        // Promotion gate: absolute-cumulative, never per-window-rate.
        const bool gate = cs.shadow_clean_hits_total >= ShadowCleanHitsGate &&
                          cs.shadow_frames_total >= ShadowFramesGate;
        if (gate && c.verify_diverged == 0) {
            Transition(id, cs, State::Enabled, "shadow burn-in clean");
            cs.demote_streak = 0;
        } else if (cs.shadow_frames_total >= ShadowFramesGate * (cs.shadow_extensions + 1)) {
            if (++cs.shadow_extensions > ShadowMaxExtensions) {
                Transition(id, cs, State::Learning, "insufficient evidence rate");
            }
        }
        if (cs.shadow_residency_ns > ShadowResidencyBudgetNs) {
            Transition(id, cs, State::Learning, "shadow residency budget");
        }
        break;
    }
    case State::Enabled: {
        if (w.low_signal) {
            break;
        }
        cs.demote_streak = w.net_pct < DemoteFloorPct ? cs.demote_streak + 1 : 0;
        if (cs.demote_streak >= DemoteStreakLen) {
            Transition(id, cs, State::Learning, "profit below demote floor");
            cs.demote_streak = 0;
            cs.promote_streak = 0;
        }
        break;
    }
    case State::Quarantined:
        break; // session-sticky
    }
}

void Framework::LogSessionSummary() {
    for (size_t i = 0; i < NumCaches; ++i) {
        const WindowSummary& w = last_window_[i];
        const CacheCounters& c = w.c;
        const f64 hr = c.eligible ? 100.0 * f64(c.hits) / f64(c.eligible) : 0.0;
        LOG_INFO(Render_Skipcache,
                 "[SkipCache] SESSION {} state={} elig={} hit%={:.1f} cold={} key={} "
                 "gen={},{},{},{},{},{} veto={},{},{},{},{},{},{},{} vfy={}/{}/{} net%={:.3f}",
                 CacheName(static_cast<CacheId>(i)), StateName(w.state), c.eligible, hr,
                 c.miss_cold, c.miss_key, c.miss_gen[0], c.miss_gen[1], c.miss_gen[2],
                 c.miss_gen[3], c.miss_gen[4], c.miss_gen[5], c.veto[0], c.veto[1], c.veto[2],
                 c.veto[3], c.veto[4], c.veto[5], c.veto[6], c.veto[7], c.verify_clean,
                 c.verify_diverged, c.verify_aborted, w.net_pct);
    }
}

} // namespace VideoCore::Skipcache
