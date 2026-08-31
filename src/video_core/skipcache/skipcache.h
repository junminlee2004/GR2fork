// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <thread>
#include "common/types.h"

namespace VideoCore::Skipcache {

// =============================================================================
// Runtime-adaptive skip caches: shared core.
//
// A skip cache elides a provably redundant recomputation on the GPU thread.
// Whether each cache is worth running is decided at runtime by measurement,
// never hardcoded per title. The controller decides PROFITABILITY only; it
// never elides, weakens, or reorders a correctness guard - a cache with
// expensive guards gets disabled, not de-guarded.
//
// THE VERIFY CONTRACT (normative; every cache cites this, none restates it):
//  1. Predict-then-execute. On a verify draw the unmodified authoritative slow
//     path executes exactly once and its result is what renders. Never
//     recompute-before-serve (consumes armed clears: instrument-injected
//     artifacts) and never recompute-after-serve (compares post-consumption
//     state and double-advances barrier/stream machinery).
//  2. Prediction purity. A prediction is built only from cached bytes plus a
//     read-only whitelist (IsMetaCleared const read, image uid, backing state,
//     flag reads). Forbidden in any predict/verify path: TouchMeta, ClearMeta,
//     ObtainBuffer, GetBarriers/Transit, UpdateImage/RefreshImage,
//     FindRenderTarget, any push/bind call.
//  3. Prediction isolation. Predictions never write live members; live members
//     are written only by a clean slow-path rebuild or by a consumed hit in
//     the Enabled non-verify flow.
//  4. Compare RAW before any transform. A fresh clear flag on a would-hit IS
//     the caught bug and must report divergence.
//  5. Unknowns fail toward the slow path. A cache failure mode may cost
//     performance, never pixels.
//
// Memory-ordering rule for the validity generations: cross-thread producers
// mutate their structure first, then fetch_add(1, release) as the LAST
// statement of the hook (seqlock discipline - bump-first would create a
// permanently valid stale entry). Consumers load(acquire) at hit-check and
// re-check at populate commit; a mid-build invalidation leaves the entry
// invalid. GPU-thread-confined lanes are plain u64 - single writer, same
// thread reader, asserted not narrated.
// =============================================================================

// Forced pins every cache with a real consumer to Enabled once at boot: no
// learning windows, no verify tripwire, no pricing, no telemetry, no timer
// sampling - OnSubmit returns immediately, so the framework runs zero
// per-frame code. The probe-only and dead ids stay Off. There is no
// divergence safety net in this mode; the setting itself is the valve.
enum class Mode : u8 { Disabled = 0, Adaptive = 1, Forced = 2, ValidateOnly = 3 };
enum class CacheId : u8 {
    UpdateImageDedup = 0,
    BeginRendering = 1,
    BindingSkipProbe = 2,
    FindImage = 3,
    PrepareRt = 4,
    Pipeline = 5,
    Sampler = 6,
    DescDelta = 7,
    StreamMirror = 8,
    DynState = 9,
    Count
};
enum class State : u8 { Off = 0, Learning, Shadow, Enabled, Quarantined };

constexpr size_t NumCaches = static_cast<size_t>(CacheId::Count);
constexpr const char* CacheName(CacheId id) {
    switch (id) {
    case CacheId::UpdateImageDedup:
        return "DEDUP";
    case CacheId::BeginRendering:
        return "BR";
    case CacheId::BindingSkipProbe:
        return "BSPROBE";
    case CacheId::FindImage:
        return "FINDIMG";
    case CacheId::PrepareRt:
        return "RTMEMO";
    case CacheId::Pipeline:
        return "PIPEKEY";
    case CacheId::Sampler:
        return "SAMPLER";
    case CacheId::StreamMirror:
        return "MIRROR";
    case CacheId::DescDelta:
        return "DESCDELTA";
    case CacheId::DynState:
        return "DYNSTATE";
    default:
        return "?";
    }
}
constexpr const char* StateName(State s) {
    switch (s) {
    case State::Off:
        return "OFF";
    case State::Learning:
        return "LEARNING";
    case State::Shadow:
        return "SHADOW";
    case State::Enabled:
        return "ENABLED";
    case State::Quarantined:
        return "QUARANTINED";
    default:
        return "?";
    }
}

// Lanes that certify classes of external mutation when unchanged. reg_stamp
// (GPU-thread, lives in the Liverpool parser) joins via the DrawToken.
struct ValidityGens {
    std::atomic<u64> mem_gen{1};  // host-guest-memory writes incl. emulator-internal downloads
    std::atomic<u64> tex_gen{1};  // texture-cache identity/lifecycle
    std::atomic<u64> pipe_gen{1}; // pipeline object destroyed/replaced
    u64 img_dirty_gen{1};         // GPU-thread confined: GPU-side image dirtying
    u64 layout_gen{1};            // GPU-thread confined: any image layout/backing state write
    u64 meta_gen{1};              // GPU-thread confined: real CMASK/HTILE arm/disarm changes
};

struct DrawToken {
    u64 reg_stamp;
    u64 tick;
    u64 mem_gen;
    u64 tex_gen;
    u64 pipe_gen;
    u64 img_dirty_gen;
};

enum MissLane : u8 { LaneReg = 0, LaneTick, LaneMem, LaneTex, LanePipe, LaneImgDirty, LaneCount };

// Plain u64 instance members - GPU-thread single writer, no thread_local
// (TLS silently forks counters if a probe ever runs off-thread and broke
// pthread_create in a prior life; banned in this directory).
struct CacheCounters {
    u64 eligible{};
    u64 hits{};
    u64 miss_cold{};
    u64 miss_key{};
    std::array<u64, LaneCount> miss_gen{};
    std::array<u64, 8> veto{};
    u64 populated{};
    u64 populate_refused{};
    u64 abandoned{};
    u64 verify_clean{};
    u64 verify_diverged{};
    u64 verify_aborted{};
    // Sampled timing (ns, decimated); shadow burden is attributed separately
    // and never feeds the hit/miss cost models.
    u64 guard_ns{};
    u64 guard_samples{};
    u64 hit_ns{};
    u64 hit_samples{};
    u64 miss_ns{};
    u64 miss_samples{};
    u64 shadow_ns{};

    u64 Misses() const {
        u64 g = 0;
        for (u64 v : miss_gen) {
            g += v;
        }
        u64 vt = 0;
        for (u64 v : veto) {
            vt += v;
        }
        return miss_cold + miss_key + g + vt;
    }
    bool AccountingHolds() const {
        return eligible == hits + Misses() + abandoned;
    }
};

struct WindowSummary {
    u64 window_id{};
    State state{State::Off};
    CacheCounters c{};
    u64 duration_ns{};
    bool poisoned{};
    bool low_signal{};
    f64 net_pct{}; // projected steady-state net as % of window
};

class Framework {
public:
    static Framework& Instance();

    // Called once from the Rasterizer constructor with the settings snapshot.
    // EmulatorSettings is never read on the hot path.
    void Init(Mode mode);
    // Settings setter mirrors mode changes here (UI thread); applied at the
    // next window close on the GPU thread.
    void SetRequestedMode(Mode mode) {
        requested_mode_.store(static_cast<u8>(mode), std::memory_order_relaxed);
    }
    Mode ActiveMode() const {
        return mode_;
    }
    bool Active() const {
        return mode_ != Mode::Disabled;
    }

    State GetState(CacheId id) const {
        return caches_[static_cast<size_t>(id)].state;
    }
    CacheCounters& Counters(CacheId id) {
        return caches_[static_cast<size_t>(id)].counters;
    }
    ValidityGens& Gens() {
        return gens_;
    }

    // Captured once per draw/dispatch entry when active (~6 loads). Cmdbuf-
    // scoped late checks must re-read live CurrentTick() instead - a draw-entry
    // snapshot cannot see a mid-draw flush.
    DrawToken Capture(u64 reg_stamp, u64 tick) const {
        return DrawToken{
            .reg_stamp = reg_stamp,
            .tick = tick,
            .mem_gen = gens_.mem_gen.load(std::memory_order_acquire),
            .tex_gen = gens_.tex_gen.load(std::memory_order_acquire),
            .pipe_gen = gens_.pipe_gen.load(std::memory_order_acquire),
            .img_dirty_gen = gens_.img_dirty_gen,
        };
    }

    // ---- Probe gating -------------------------------------------------------
    // Learning runs the predicate in randomized contiguous 64-call bursts every
    // 512 eligible calls (would-hit rates depend on call adjacency; stride
    // decimation biases them low). Shadow/Enabled probe every call.
    bool ShouldProbe(CacheId id) {
        auto& cs = caches_[static_cast<size_t>(id)];
        switch (cs.state) {
        case State::Off:
        case State::Quarantined:
            return false;
        case State::Learning: {
            if (cs.burst_remaining > 0) {
                --cs.burst_remaining;
                return true;
            }
            if (++cs.burst_countdown >= cs.burst_next) {
                cs.burst_countdown = 0;
                cs.burst_next = 512;
                cs.burst_remaining = 63;
                return true;
            }
            return false;
        }
        default:
            return true;
        }
    }

    // True when a would-hit must be verified this draw (Shadow: always;
    // Enabled: 1/256 tripwire + forced verifies, capped per window).
    bool ShouldVerify(CacheId id);

    // A consumed hit is only legal in Enabled non-verify flow.
    bool MayConsume(CacheId id) const {
        return caches_[static_cast<size_t>(id)].state == State::Enabled &&
               mode_ != Mode::ValidateOnly;
    }

    // ---- Verify/divergence bookkeeping (drives the state machine) ----------
    void RecordVerifyClean(CacheId id);
    void RecordVerifyAborted(CacheId id);
    // detail is a short single-line description for the DIVERGENCE record.
    void RecordDivergence(CacheId id, const char* detail);
    // A populate happened; the first hit after it is force-verified.
    void NotifyPopulated(CacheId id) {
        auto& cs = caches_[static_cast<size_t>(id)];
        ++cs.counters.populated;
        cs.force_verify_next_hit = true;
    }

    // ---- Timing (sampled, N=256 with per-window xorshift phase) ------------
    bool TimingEnabled() const {
        return timing_enabled_;
    }
    bool SampleTimer(CacheId id) {
        auto& cs = caches_[static_cast<size_t>(id)];
        // Learning probes only ~12.5% of calls in bursts; at 1/256 on top the
        // cost model would starve. Sample 1/16 inside bursts, 1/256 elsewhere.
        const u32 mask = cs.state == State::Learning ? 0xF : 0xFF;
        return timing_enabled_ && ((++cs.timer_decim & mask) == (cs.timer_phase & mask));
    }
    u64 Now() const; // FencedRDTSC in ns (calibrated at Init)
    // Remove the fenced-pair overhead from a timed interval so tiny sections
    // are not systematically inflated.
    u64 CorrectSample(u64 ns) const {
        return ns > pair_ns_ ? ns - pair_ns_ : 0;
    }

    // ---- Window driver ------------------------------------------------------
    // Runs on the GPU thread at submit-done. Closes the window when >=250 ms
    // elapsed or >=32 flips, whichever first.
    void OnSubmit(u32 frame_num, bool guest_paused);
    // Masked fallback for titles that rarely reach submit-done. Adaptive-only:
    // in Forced mode states are pinned at boot, and letting this drive the
    // window controller prices every cache at zero (timing is off) and demotes
    // it, silently turning Forced into all-caches-Off within minutes.
    void OnDraw() {
        if (mode_ != Mode::Adaptive) {
            return;
        }
        ++window_draws_;
        if ((++draw_counter_ & 0x1FFF) == 0) {
            MaybeCloseWindow(true);
        }
    }

    // Registered caches invalidate through this (state kept by the caches
    // themselves; they register a callback at Init time).
    using InvalidateFn = void (*)(void*);
    void RegisterInvalidate(InvalidateFn fn, void* user) {
        if (num_invalidate_fns_ < std::size(invalidate_fns_)) {
            invalidate_fns_[num_invalidate_fns_].fn = fn;
            invalidate_fns_[num_invalidate_fns_].user = user;
            ++num_invalidate_fns_;
        }
    }
    void InvalidateAll() {
        for (size_t i = 0; i < num_invalidate_fns_; ++i) {
            invalidate_fns_[i].fn(invalidate_fns_[i].user);
        }
    }

    // Session summary (called from shutdown path; reads the last snapshot).
    void LogSessionSummary();

    // ---- Descriptor delta storage: the serialized form of the last
    // descriptor push per bind point (graphics, compute). Fixed capacity;
    // an oversized serialization fails closed as a veto. ----
    struct DescDeltaSlot {
        bool valid{};
        u64 tick{};
        u64 layout{};
        u64 foreign_gen{};
        u32 size{};
        std::array<u8, 16384> blob{};
    };
    DescDeltaSlot& DescDeltaState(size_t bind_point_index) {
        return desc_delta_[bind_point_index];
    }
    std::array<u8, 16384>& DescDeltaScratch() {
        return desc_delta_scratch_;
    }
    // Any descriptor push issued OUTSIDE Pipeline::BindResources (fault
    // processing, host passes) overwrites command buffer state behind the
    // delta cache's back; those sites bump this so the next probe misses.
    void BumpForeignPushGen(size_t bind_point_index) {
        ++foreign_push_gen_[bind_point_index];
    }
    u64 ForeignPushGen(size_t bind_point_index) const {
        return foreign_push_gen_[bind_point_index];
    }

    // ---- UpdateImageDedup storage (framework-owned: its consumer is the
    // texture cache, which has no rasterizer pointer). 256-slot direct-mapped;
    // collisions overwrite - losing dedup, never skipping required work. ----
    struct DedupEntry {
        u32 image_index{};
        u64 tick{};
        u64 mem_gen{};
        u64 tex_gen{};
        u64 img_dirty_gen{};
        bool valid{};
    };
    // Returns true when UpdateImage may be skipped (all lanes equal). The
    // caller must have already consulted ShouldProbe/MayConsume/ShouldVerify.
    bool DedupProbe(u32 image_index, const DrawToken& t) {
        const DedupEntry& e = dedup_[image_index & 0xFF];
        return e.valid && e.image_index == image_index && e.tick == t.tick &&
               e.mem_gen == t.mem_gen && e.tex_gen == t.tex_gen &&
               e.img_dirty_gen == t.img_dirty_gen;
    }
    void DedupCommit(u32 image_index, const DrawToken& entry_token);
    void DedupInvalidateAll() {
        for (auto& e : dedup_) {
            e.valid = false;
        }
    }

    // Convenience gen bumps (hooks call these; mutate-then-bump discipline is
    // the CALLER's obligation - the bump must be the last statement).
    void BumpMemGen() {
        gens_.mem_gen.fetch_add(1, std::memory_order_release);
    }
    void BumpTexGen() {
        gens_.tex_gen.fetch_add(1, std::memory_order_release);
    }
    void BumpPipeGen() {
        gens_.pipe_gen.fetch_add(1, std::memory_order_release);
    }
    void BumpImgDirtyGen() {
        ++gens_.img_dirty_gen;
    }
    void BumpLayoutGen() {
        ++gens_.layout_gen;
    }
    void BumpMetaGen() {
        ++gens_.meta_gen;
    }

private:
    struct CacheState {
        State state{State::Off};
        CacheCounters counters{};
        // Learning burst sampler
        u32 burst_countdown{};
        u32 burst_next{512};
        u32 burst_remaining{};
        // Verify machinery
        u32 tripwire_decim{};
        u8 tripwire_phase{};
        u32 forced_verifies_this_window{};
        bool force_verify_next_hit{};
        u32 consecutive_aborts{};
        // Timing decimation
        u32 timer_decim{};
        u8 timer_phase{};
        // Controller bookkeeping
        u32 windows_in_state{};
        u32 learning_windows_total{};
        u32 promote_streak{};
        u32 demote_streak{};
        u64 shadow_clean_hits_total{};
        u32 shadow_frames_total{};
        u32 shadow_extensions{};
        u64 shadow_residency_ns{};
        bool lane_bump_survived{};
        bool shadow_priced_low{};
        u32 divergences_total{};
        u64 last_divergence_window{};
        u32 reprobe_countdown{};
        u32 reprobe_backoff{16};
        // Rolling eligible median approximation (window eligibles ring)
        std::array<u64, 8> eligible_ring{};
        u32 eligible_ring_n{};
    };

    void MaybeCloseWindow(bool from_draw_fallback);
    void CloseWindow();
    void StepController(CacheState& cs, CacheId id, const WindowSummary& w);
    void Transition(CacheId id, CacheState& cs, State next, const char* reason);
    f64 ProjectedNetPct(const CacheState& cs, u64 window_ns) const;
    u64 RollingMedianEligible(const CacheState& cs) const;

    ValidityGens gens_{};
    Mode mode_{Mode::Disabled};
    std::atomic<u8> requested_mode_{0};
    std::array<CacheState, NumCaches> caches_{};
    std::array<DedupEntry, 256> dedup_{};
    std::array<DescDeltaSlot, 2> desc_delta_{};
    std::array<u64, 2> foreign_push_gen_{};
    std::array<u8, 16384> desc_delta_scratch_{};

    struct {
        InvalidateFn fn;
        void* user;
    } invalidate_fns_[8]{};
    size_t num_invalidate_fns_{};

    // Window state (GPU thread)
    u64 window_id_{};
    u64 window_start_ns_{};
    u32 window_start_frame_{};
    u64 window_draws_{};
    bool window_poisoned_{};
    u32 invariant_violations_{};
    u64 draw_counter_{};
    u64 last_session_log_ns_{};
    u32 warmup_windows_{};
    bool warmed_up_{};
    // Draws/window EWMA for the warmup stability predicate
    f64 draws_ewma_{};
    f64 draws_ewma_var_{};
    u32 stable_windows_{};

    std::array<WindowSummary, NumCaches> last_window_{};

    bool dedup_invalidate_registered_{};
    bool timing_enabled_{};
    u64 tsc_hz_{};
    u64 tsc_pair_cost_{};
    u64 pair_ns_{};
    f64 ns_per_cycle_{};
    u32 xorshift_state_{0x9E3779B9u};

#ifdef _DEBUG
    std::thread::id gpu_thread_{};
#endif
};

} // namespace VideoCore::Skipcache
