// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Diagnostic-only. Per-thread (TLS) cycle accumulators attribute each Rasterizer::BindResources
// call to its sub-sections; one LOG_INFO per 200K calls, counters reset per window. Overhead is
// ~700-1000 cycles/call (~0.6-0.9% GpuComm); remove this header and the BR_TIMER uses to strip.

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

    // BindTextures drill-down, nested under slow_bindtex: bt_first_pass + bt_second_pass +
    // bt_samplers approximate slow_bindtex, the residual being local setup cost. bt_findimage
    // nests under bt_first_pass, timing FindImageWithView on per-call + pcache double-miss only.
    SectionStat bt_first_pass;
    SectionStat bt_findimage;
    SectionStat bt_second_pass;
    SectionStat bt_samplers;

    // First-pass cache miss attribution: icache_hits, the four pcache miss causes (empty slot,
    // key collision, base_desc rotation, validate reject), pcache_hits and bt_findimage.calls
    // sum to bt_first_pass.calls x avg images per stage; the dominant bucket picks the fix.
    std::uint64_t icache_hits{0};
    std::uint64_t pcache_invalid_miss{0};
    std::uint64_t pcache_pkey_miss{0};
    std::uint64_t pcache_base_miss{0};
    std::uint64_t pcache_validate_miss{0};
    std::uint64_t pcache_hits{0};

    // Sub-attribution of pcache_validate_miss, one counter per check in ValidateCachedFindImage:
    // unallocated/unregistered = GC eviction, address/size/geometry = slot reused by a different
    // image, format_incompat = over-strict format check, null_id = cached zero id (never expected).
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

    // Second line: BindTextures drill-down. bt_first_pass / bt_second_pass / bt_samplers are
    // fractions of slow_bindtex and sum to ~100% minus local setup; bt_findimage is reported as
    // a fraction of bt_first_pass since it nests inside it.
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

    // Third line: per-iteration cache attribution. The counters below sum to the total
    // first-pass image iterations (bt_first_pass.calls x avg images per stage).
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

    // Fourth line: validate-fail sub-attribution. pcache_validate_miss from line 3 splits
    // exactly across these seven buckets, one per check inside ValidateCachedFindImage.
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

// Total-timer variant of ScopedSectionTimer: its destructor also advances the log counter and
// runs LogAndReset every kLogEvery calls, so only the outermost timer drives logging.
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
