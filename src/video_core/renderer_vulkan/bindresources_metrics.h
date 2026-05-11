// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// =============================================================================
// bindresources_metrics.h — v1.40 diagnostic-only.
//
// PURPOSE
//   v1.39 cut Rasterizer::BindResources from 4.77% to 4.35% of GpuComm by
//   gating dead atomic loads under `if (pipeline_uses_push)`. The remaining
//   4.35% is dominated by the inlined BindBuffers + BindTextures bodies on
//   the slow path, but the per-subsection share is unmeasured. Without
//   that breakdown, v1.40+ optimization choices (descriptor-set reuse vs
//   per-call cache rework vs FindImageWithView restructuring) are guesses.
//
//   This header adds per-thread cycle accumulators that attribute every
//   BindResources call to one of seven sections:
//       total            entire function (RAII at top of body)
//       pre_loop         vector clears + delta cache + MakeUserData
//       skip_cache       the pipeline_uses_push gated block (RADV: ~0)
//       slow_loop        the for-loop over stages (1 entry per slow call)
//         slow_pushud      PushUd portion (1 entry per stage)
//         slow_bindbuf     BindBuffers portion (1 entry per stage)
//         slow_bindtex     BindTextures portion (1 entry per stage)
//       tail             skip-cache update + DMA sync block
//
//   Plus three early-return counters:
//       fast_replay_returns   stamp-matched cached-push_data replay
//       fast_pushud_returns   identical-bindings PushUd-only return
//       (slow path is total.calls - fast_replay - fast_pushud)
//
//   Every 200K calls the GpuComm thread emits one LOG_INFO line via
//   Render_Vulkan and resets the counters so each window is independent.
//   At ~5000 draws/frame × 60 Hz that fires roughly every 0.7 s.
//
// COST
//   Each timer is two FencedRDTSC() reads (lfence + rdtsc + lfence ≈ 30-40
//   cycles each). Six timers per slow-path call × 4 stages × 3 inner timers
//   ≈ 12 timers in scope simultaneously across nesting depth, but each
//   call site is two rdtsc's. Estimated overhead ≈ 700-1000 cycles per
//   BindResources call, i.e. ~0.6-0.9% extra GpuComm. Tractable for
//   measurement; STRIP BEFORE SHIPPING by deleting this header and the
//   BR_TIMER macro uses in vk_rasterizer.cpp.
//
// DESIGN NOTES
//   - Per-thread (TLS) accumulators avoid atomic contention.
//   - Common::FencedRDTSC is used for consistency with the existing rdtsc
//     helper in common/rdtsc.h. Plain rdtsc would be ~2x cheaper but adds
//     ~200-cycle reorder noise that contaminates sub-microsecond sections.
//   - The total timer's destructor doubles as the LogIfDue trigger so
//     that fast-path returns (which exit before reaching bottom-of-function)
//     still advance the log counter.
//   - Resetting the accumulators at log time is intentional: each report
//     is a window-snapshot, not a cumulative average. Independent windows
//     let us see whether the breakdown shifts during a session (loading
//     screen vs gameplay vs cutscene).
// =============================================================================

#include <cstdint>

#include "common/logging/log.h"
#include "common/rdtsc.h"

namespace Vulkan::detail {

struct SectionStat {
    std::uint64_t cycles{0};
    std::uint64_t calls{0};
};

struct BrThreadStats {
    SectionStat total;
    SectionStat pre_loop;
    SectionStat skip_cache;
    SectionStat slow_loop;
    SectionStat slow_pushud;
    SectionStat slow_bindbuf;
    SectionStat slow_bindtex;
    SectionStat tail;

    // v1.41 BindTextures sub-section drill-down. These are fired from
    // inside the inlined BindTextures body (i.e. they're NESTED under
    // slow_bindtex). Sum of (bt_first_pass + bt_second_pass + bt_samplers)
    // should approximate slow_bindtex; the residual is BindTextures'
    // local setup cost (epoch bump, lambda capture, image_bindings.clear).
    // bt_findimage is NESTED under bt_first_pass — it times the
    // texture_cache.FindImageWithView callee specifically, which only
    // fires on per-call + persistent cache double-miss.
    SectionStat bt_first_pass;
    SectionStat bt_findimage;
    SectionStat bt_second_pass;
    SectionStat bt_samplers;

    // v1.42 first-pass cache miss attribution. v1.41 surfaced a 91%
    // bt_findimage fire rate, which means the two caches in the first
    // pass (intra-call find_image_cache + cross-call find_image_pcache_)
    // are missing on essentially every iteration. We need to know WHY:
    //
    //   icache_hits          — find_image_cache hit path (Step B)
    //                          intra-call de-dup (rare unless one shader
    //                          binds the same image twice in same stage)
    //
    //   pcache_invalid_miss  — pe.valid == false (slot empty / cold cache)
    //   pcache_pkey_miss     — pe.valid && pe.key != key  (collision: slot
    //                          carries a different image's data)
    //   pcache_base_miss     — pe.key matches but pe.base != base_desc
    //                          (image_desc_cache_ rotated; same logical
    //                          image but base_desc pointer changed)
    //   pcache_validate_miss — pe.key + pe.base both match but
    //                          ValidateCachedFindImage rejected (the cached
    //                          image_id is genuinely stale per TextureCache
    //                          generation)
    //   pcache_hits          — all four conditions satisfied, fast path
    //
    // Sum-check invariant:
    //   icache_hits + pcache_invalid + pcache_pkey + pcache_base
    //     + pcache_validate + pcache_hits + bt_findimage.calls
    //     == bt_first_pass.calls × avg_images_per_stage
    //
    // The dominant miss bucket dictates v1.43's fix:
    //   pkey dominates    → collision pressure, increase pcache slots
    //   base dominates    → key needs to decouple from base_desc pointer
    //   validate dominates → cached image_id stale; switch to TextureCache-
    //                       version-keyed approach or invalidate-on-event
    std::uint64_t icache_hits{0};
    std::uint64_t pcache_invalid_miss{0};
    std::uint64_t pcache_pkey_miss{0};
    std::uint64_t pcache_base_miss{0};
    std::uint64_t pcache_validate_miss{0};
    std::uint64_t pcache_hits{0};

    // v1.43 diagnostic sub-attribution of pcache_validate_miss. The v1.42
    // data showed validate-fail at 33.66% of all iterations (~52% of all
    // pcache lookups). One of the seven checks inside ValidateCachedFindImage
    // is responsible. These counters localize the failure mechanism so v1.44
    // targets the right fix:
    //   unallocated/unregistered  → texture cache GC eviction; fix is event-
    //                               driven invalidation or generation stamps
    //   address/size/geometry     → same image_id slot now holds a different
    //                               image; fix is rekey or stronger identity
    //   format_incompat           → format compatibility check too strict
    //   null_id                   → cached zero image_id; should never happen
    std::uint64_t validate_null_id{0};
    std::uint64_t validate_unallocated{0};
    std::uint64_t validate_unregistered{0};
    std::uint64_t validate_address_mismatch{0};
    std::uint64_t validate_size_mismatch{0};
    std::uint64_t validate_geometry_mismatch{0};
    std::uint64_t validate_format_incompat{0};

    std::uint64_t fast_replay_returns{0};
    std::uint64_t fast_pushud_returns{0};
    std::uint64_t draws_since_log{0};
};

// One stats block per thread. The GpuComm thread is the dominant writer;
// other threads (compute dispatch in shadPS4 occasionally) have their
// own block and log independently.
inline thread_local BrThreadStats g_br_stats{};

// Plain section timer. RAII: starts on construction, accumulates on
// destruction. Move/copy disabled so the compiler can't accidentally
// duplicate a timer.
class ScopedSectionTimer {
public:
    explicit ScopedSectionTimer(SectionStat& target) noexcept
        : target_(&target), start_(::Common::FencedRDTSC()) {}
    ~ScopedSectionTimer() noexcept {
        const std::uint64_t end = ::Common::FencedRDTSC();
        target_->cycles += end - start_;
        ++target_->calls;
    }
    ScopedSectionTimer(const ScopedSectionTimer&) = delete;
    ScopedSectionTimer& operator=(const ScopedSectionTimer&) = delete;
    ScopedSectionTimer(ScopedSectionTimer&&) = delete;
    ScopedSectionTimer& operator=(ScopedSectionTimer&&) = delete;

private:
    SectionStat* target_;
    std::uint64_t start_;
};

inline void LogAndReset() {
    auto& s = g_br_stats;
    auto pct = [&](const SectionStat& sec) -> double {
        if (s.total.cycles == 0) {
            return 0.0;
        }
        return 100.0 * static_cast<double>(sec.cycles) /
               static_cast<double>(s.total.cycles);
    };
    auto avg = [](const SectionStat& sec) -> double {
        if (sec.calls == 0) {
            return 0.0;
        }
        return static_cast<double>(sec.cycles) /
               static_cast<double>(sec.calls);
    };

    // Loop overhead = slow_loop_total - sum(inner). Captures iteration
    // boilerplate (GetStages range advance, !stage check, uses_dma OR).
    const std::uint64_t inner_sum = s.slow_pushud.cycles +
                                    s.slow_bindbuf.cycles +
                                    s.slow_bindtex.cycles;
    const double loop_overhead_pct =
        s.total.cycles == 0 ? 0.0
            : 100.0 * static_cast<double>(s.slow_loop.cycles - inner_sum) /
              static_cast<double>(s.total.cycles);

    const std::uint64_t slow_calls = s.total.calls - s.fast_replay_returns -
                                     s.fast_pushud_returns;

    LOG_INFO(Render_Vulkan,
        "[BR_METRICS] N={} avg_cyc/call={:.0f} | "
        "pre_loop={:.2f}% skip_cache={:.2f}% slow_loop={:.2f}% tail={:.2f}% | "
        "in slow_loop: pushud={:.2f}% bindbuf={:.2f}% bindtex={:.2f}% loop_oh={:.2f}% | "
        "calls slow={} fast_replay={} fast_pushud={} | "
        "stage-avg cyc: pushud={:.0f} bindbuf={:.0f} bindtex={:.0f}",
        s.total.calls, avg(s.total),
        pct(s.pre_loop), pct(s.skip_cache), pct(s.slow_loop), pct(s.tail),
        pct(s.slow_pushud), pct(s.slow_bindbuf), pct(s.slow_bindtex),
        loop_overhead_pct,
        slow_calls, s.fast_replay_returns, s.fast_pushud_returns,
        avg(s.slow_pushud), avg(s.slow_bindbuf), avg(s.slow_bindtex));

    // v1.41 second line: BindTextures sub-section drill-down. bt_first_pass,
    // bt_second_pass, bt_samplers are reported as fractions of slow_bindtex
    // (the parent scope) so they sum to ~100% modulo the local setup cost
    // residual. bt_findimage is reported separately as a fraction of
    // bt_first_pass since it's nested inside it.
    auto pct_of = [&](const SectionStat& sec, const SectionStat& base) -> double {
        if (base.cycles == 0) {
            return 0.0;
        }
        return 100.0 * static_cast<double>(sec.cycles) /
               static_cast<double>(base.cycles);
    };
    const std::uint64_t bt_sub_sum = s.bt_first_pass.cycles +
                                     s.bt_second_pass.cycles +
                                     s.bt_samplers.cycles;
    const double bt_setup_residual_pct =
        s.slow_bindtex.cycles == 0 ? 0.0
            : 100.0 * static_cast<double>(s.slow_bindtex.cycles - bt_sub_sum) /
              static_cast<double>(s.slow_bindtex.cycles);

    LOG_INFO(Render_Vulkan,
        "[BR_METRICS_BT] in bindtex: first_pass={:.2f}% second_pass={:.2f}% "
        "samplers={:.2f}% setup_resid={:.2f}% | "
        "in first_pass: findimage_callee={:.2f}% (fires {}/{}) | "
        "stage-avg cyc: first_pass={:.0f} second_pass={:.0f} samplers={:.0f} findimage={:.0f}",
        pct_of(s.bt_first_pass, s.slow_bindtex),
        pct_of(s.bt_second_pass, s.slow_bindtex),
        pct_of(s.bt_samplers, s.slow_bindtex),
        bt_setup_residual_pct,
        pct_of(s.bt_findimage, s.bt_first_pass),
        s.bt_findimage.calls, s.bt_first_pass.calls,
        avg(s.bt_first_pass), avg(s.bt_second_pass),
        avg(s.bt_samplers), avg(s.bt_findimage));

    // v1.42 third line: per-iteration cache attribution. Sum of all
    // counters below should equal the total first-pass image iterations
    // (i.e. bt_first_pass.calls × avg_images_per_stage). Each miss bucket
    // localizes the failure mechanism for v1.43 to target.
    const std::uint64_t cache_total = s.icache_hits + s.pcache_invalid_miss +
                                       s.pcache_pkey_miss + s.pcache_base_miss +
                                       s.pcache_validate_miss + s.pcache_hits +
                                       s.bt_findimage.calls;
    auto frac = [&](std::uint64_t n) -> double {
        if (cache_total == 0) {
            return 0.0;
        }
        return 100.0 * static_cast<double>(n) / static_cast<double>(cache_total);
    };
    LOG_INFO(Render_Vulkan,
        "[BR_METRICS_CACHE] N_iters={} | icache_hits={:.2f}% | "
        "pcache: invalid={:.2f}% pkey_collision={:.2f}% base_mismatch={:.2f}% "
        "validate_fail={:.2f}% hits={:.2f}% | findimage_fires={:.2f}% | "
        "raw counts: ich={} pinv={} ppk={} pbs={} pvf={} phits={} fimg={}",
        cache_total, frac(s.icache_hits),
        frac(s.pcache_invalid_miss), frac(s.pcache_pkey_miss),
        frac(s.pcache_base_miss), frac(s.pcache_validate_miss),
        frac(s.pcache_hits), frac(s.bt_findimage.calls),
        s.icache_hits, s.pcache_invalid_miss, s.pcache_pkey_miss,
        s.pcache_base_miss, s.pcache_validate_miss, s.pcache_hits,
        s.bt_findimage.calls);

    // v1.43 fourth line: validate-fail sub-attribution. The pcache_validate_miss
    // counter from line 3 splits across these seven buckets — one per check
    // inside ValidateCachedFindImage. Sum should equal pcache_validate_miss
    // exactly. Dominant bucket dictates v1.44's fix.
    const std::uint64_t validate_total = s.validate_null_id +
                                          s.validate_unallocated +
                                          s.validate_unregistered +
                                          s.validate_address_mismatch +
                                          s.validate_size_mismatch +
                                          s.validate_geometry_mismatch +
                                          s.validate_format_incompat;
    auto vfrac = [&](std::uint64_t n) -> double {
        if (validate_total == 0) {
            return 0.0;
        }
        return 100.0 * static_cast<double>(n) / static_cast<double>(validate_total);
    };
    LOG_INFO(Render_Vulkan,
        "[BR_METRICS_VALIDATE] validate_fails={} (sum-check vs pvf={}) | "
        "null_id={:.2f}% unalloc={:.2f}% unreg={:.2f}% addr_mismatch={:.2f}% "
        "size_mismatch={:.2f}% geom_mismatch={:.2f}% format_incompat={:.2f}% | "
        "raw: nid={} una={} unr={} adm={} szm={} gem={} fmt={}",
        validate_total, s.pcache_validate_miss,
        vfrac(s.validate_null_id), vfrac(s.validate_unallocated),
        vfrac(s.validate_unregistered), vfrac(s.validate_address_mismatch),
        vfrac(s.validate_size_mismatch), vfrac(s.validate_geometry_mismatch),
        vfrac(s.validate_format_incompat),
        s.validate_null_id, s.validate_unallocated, s.validate_unregistered,
        s.validate_address_mismatch, s.validate_size_mismatch,
        s.validate_geometry_mismatch, s.validate_format_incompat);

    s = BrThreadStats{};
}

// Total-timer variant: same as ScopedSectionTimer but its destructor also
// advances the log counter and triggers LogAndReset every kLogEvery calls.
// Kept separate from ScopedSectionTimer so only the outermost timer drives
// logging — inner-section timers don't.
class ScopedTotalTimer {
public:
    ScopedTotalTimer() noexcept : start_(::Common::FencedRDTSC()) {}
    ~ScopedTotalTimer() noexcept {
        const std::uint64_t end = ::Common::FencedRDTSC();
        auto& s = g_br_stats;
        s.total.cycles += end - start_;
        ++s.total.calls;
        ++s.draws_since_log;
        constexpr std::uint64_t kLogEvery = 200000;
        if (s.draws_since_log >= kLogEvery) [[unlikely]] {
            LogAndReset();
        }
    }
    ScopedTotalTimer(const ScopedTotalTimer&) = delete;
    ScopedTotalTimer& operator=(const ScopedTotalTimer&) = delete;
    ScopedTotalTimer(ScopedTotalTimer&&) = delete;
    ScopedTotalTimer& operator=(ScopedTotalTimer&&) = delete;

private:
    std::uint64_t start_;
};

} // namespace Vulkan::detail

// Macros: stack-local timer scoped to the surrounding block. Identifier is
// derived from the section name to avoid collisions when multiple timers
// nest in the same scope.
#define BR_TIMER_TOTAL() \
    ::Vulkan::detail::ScopedTotalTimer _br_total_timer_
#define BR_TIMER(section) \
    ::Vulkan::detail::ScopedSectionTimer _br_timer_##section( \
        ::Vulkan::detail::g_br_stats.section)
#define BR_COUNT(field) (++::Vulkan::detail::g_br_stats.field)
