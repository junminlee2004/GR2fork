// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// GpuAsse probe harness (gr2fork, GR2 CUSA04943): header-only, env-gated, zero-cost-when-off
// measurement scaffold. Each probe has its own env gate (GR2FORK_DYN_PROBE, GR2FORK_BR_PROBE,
// GR2FORK_VBB_PROBE, GR2FORK_SYNC_PROBE); when unset the residual cost is one
// predicted-not-taken branch per hook site. Percentages are authoritative - the ns figures are
// timer-inflated and only useful relatively; one LOG_INFO per window. Diagnostic only: the
// #include and every GR2_PHDE_* call site must be removed before a release build.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include "common/logging/log.h"
#include "common/rdtsc.h"
#include "common/types.h"

namespace GR2Fork::PhDE {

// One-time tsc->ns calibration: sample the wall clock across a short FencedRDTSC interval on
// first use. Good enough for relative reporting; the ns numbers are inflated anyway.
inline double TscToNs() noexcept {
    static const double factor = [] {
        const auto w0 = std::chrono::steady_clock::now();
        const u64 t0 = Common::FencedRDTSC();
        // Busy spin ~2ms so the ratio is stable.
        while (std::chrono::steady_clock::now() - w0 < std::chrono::milliseconds(2)) {
        }
        const u64 t1 = Common::FencedRDTSC();
        const auto w1 = std::chrono::steady_clock::now();
        const double ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count();
        const double ticks = static_cast<double>(t1 - t0);
        return ticks > 0.0 ? ns / ticks : 0.0;
    }();
    return factor;
}

// ---- env gate (read once) --------------------------------------------------
inline bool EnvGate(const char* name) noexcept {
    const char* v = std::getenv(name);
    return v && v[0] == '1';
}

// GR2FORK_DYN_PROBE: dynamic_dirty suppression on user_data writes. The producer side (SetShReg
// key-dirty site) counts keydirty_total and the IsUserDataOnlyShReg subset as userdata_only;
// the consumer side (UpdateDynamicState entry) counts dyn_dirty_true vs dyn_dirty_false.
// A high userdata_only fraction means the dirty-true sub-rebuilds (~0.48%) are spurious.
struct DynProbe {
    static bool On() noexcept {
        static const bool on = EnvGate("GR2FORK_DYN_PROBE");
        return on;
    }
    static inline std::atomic<u64> keydirty_total{0};
    static inline std::atomic<u64> userdata_only{0};
    static inline std::atomic<u64> dyn_true{0};
    static inline std::atomic<u64> dyn_false{0};

    // Window: report every N SetShReg key-dirty events.
    static constexpr u64 kWindow = 200000;

    static void OnKeyDirty(bool is_userdata_only) noexcept {
        userdata_only.fetch_add(is_userdata_only ? 1 : 0, std::memory_order_relaxed);
        const u64 n = keydirty_total.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n % kWindow == 0) {
            Report(n);
        }
    }
    static void OnDynEntry(bool dirty) noexcept {
        (dirty ? dyn_true : dyn_false).fetch_add(1, std::memory_order_relaxed);
    }
    static void Report(u64 kd_total) noexcept {
        const u64 ud = userdata_only.load(std::memory_order_relaxed);
        const u64 dt = dyn_true.load(std::memory_order_relaxed);
        const u64 df = dyn_false.load(std::memory_order_relaxed);
        const u64 dtot = dt + df;
        const double ud_pct = kd_total ? 100.0 * double(ud) / double(kd_total) : 0.0;
        const double dyn_pct = dtot ? 100.0 * double(dt) / double(dtot) : 0.0;
        LOG_INFO(Render_Vulkan,
                 "[DYN_PROBE] keydirty_total={} userdata_only={} ({:.2f}%) | "
                 "UpdateDynamicState dirty={}/{} ({:.2f}% true) -> D3' recovers "
                 "the dirty=true sub-rebuilds on the userdata_only fraction",
                 kd_total, ud, ud_pct, dt, dtot, dyn_pct);
    }
};

// GR2FORK_BR_PROBE: BeginRendering stamp -> rt_hash rekey sizing. Classifies the would-be exit
// per entry: stamp_hit (br_cache_.stamp matches and validations pass), rtkey_would_hit
// (br_cache_.rt_hash == last_rt_hash_ plus pipeline match and validations pass), or miss.
// The rekey pays off when rtkey_would_hit is high (>~70%).
struct BrProbe {
    static bool On() noexcept {
        static const bool on = EnvGate("GR2FORK_BR_PROBE");
        return on;
    }
    static inline std::atomic<u64> total{0};
    static inline std::atomic<u64> stamp_hit{0};
    static inline std::atomic<u64> rtkey_hit{0};
    static inline std::atomic<u64> miss{0};

    static constexpr u64 kWindow = 36000;

    // stamp_match / rt_match are the raw key comparisons; validations_pass is
    // the result of the per-attachment revalidation the function already does.
    static void OnEntry(bool stamp_match, bool rt_match, bool validations_pass) noexcept {
        const bool s_hit = stamp_match && validations_pass;
        const bool r_hit = rt_match && validations_pass;
        if (s_hit) {
            stamp_hit.fetch_add(1, std::memory_order_relaxed);
        } else if (r_hit) {
            rtkey_hit.fetch_add(1, std::memory_order_relaxed);
        } else {
            miss.fetch_add(1, std::memory_order_relaxed);
        }
        const u64 n = total.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n % kWindow == 0) {
            Report(n);
        }
    }
    static void Report(u64 n) noexcept {
        const u64 s = stamp_hit.load(std::memory_order_relaxed);
        const u64 r = rtkey_hit.load(std::memory_order_relaxed);
        const u64 m = miss.load(std::memory_order_relaxed);
        const auto pct = [n](u64 x) { return n ? 100.0 * double(x) / double(n) : 0.0; };
        LOG_INFO(Render_Vulkan,
                 "[BR_PROBE] draws={} stamp_hit={} ({:.2f}%) rtkey_would_hit={} "
                 "({:.2f}%) miss={} ({:.2f}%) -> ship D1 if rtkey_would_hit >~ 70%",
                 n, s, pct(s), r, pct(r), m, pct(m));
    }
};

// GR2FORK_VBB_PROBE: BindVertexBuffersLegacy input-sig split sizing. Classifies each exit as
// stamp_skip, bindsig_skip, inputsig_stable_but_bindsig_miss (the reclaimable bucket - the
// GetVertexInputs walk, ~0.33% / 193 Mcyc), or full. A safe split still needs a key capturing
// sharp format/stride, not just {pipeline, step_rate}; this probe only sizes the opportunity.
struct VbbProbe {
    static bool On() noexcept {
        static const bool on = EnvGate("GR2FORK_VBB_PROBE");
        return on;
    }
    static inline std::atomic<u64> total{0};
    static inline std::atomic<u64> stamp_skip{0};
    static inline std::atomic<u64> bindsig_skip{0};
    static inline std::atomic<u64> inputsig_stable{0};
    static inline std::atomic<u64> full{0};

    static constexpr u64 kWindow = 36000;

    enum class Exit { StampSkip, BindSigSkip, InputSigStable, Full };

    static void OnExit(Exit e) noexcept {
        switch (e) {
        case Exit::StampSkip:
            stamp_skip.fetch_add(1, std::memory_order_relaxed);
            break;
        case Exit::BindSigSkip:
            bindsig_skip.fetch_add(1, std::memory_order_relaxed);
            break;
        case Exit::InputSigStable:
            inputsig_stable.fetch_add(1, std::memory_order_relaxed);
            break;
        case Exit::Full:
            full.fetch_add(1, std::memory_order_relaxed);
            break;
        }
        const u64 n = total.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n % kWindow == 0) {
            Report(n);
        }
    }
    static void Report(u64 n) noexcept {
        const u64 ss = stamp_skip.load(std::memory_order_relaxed);
        const u64 bs = bindsig_skip.load(std::memory_order_relaxed);
        const u64 is = inputsig_stable.load(std::memory_order_relaxed);
        const u64 f = full.load(std::memory_order_relaxed);
        const auto pct = [n](u64 x) { return n ? 100.0 * double(x) / double(n) : 0.0; };
        LOG_INFO(Render_Vulkan,
                 "[VBB_PROBE] vbb={} stamp_skip={} ({:.2f}%) bindsig_skip={} "
                 "({:.2f}%) inputsig_stable_but_bindsig_miss={} ({:.2f}%) full={} "
                 "({:.2f}%) -> E1 worth it iff inputsig_stable bucket is large",
                 n, ss, pct(ss), bs, pct(bs), is, pct(is), f, pct(f));
    }
};

// GR2FORK_SYNC_PROBE: per-buffer DMA-sync redundant-scan sizing. Counts SynchronizeBuffer calls
// as had_dirty (emitted at least one copy) vs clean, plus the walked range size. Clean
// dominating means the scan is mostly redundant; common had_dirty means it is irreducible.
struct SyncProbe {
    static bool On() noexcept {
        static const bool on = EnvGate("GR2FORK_SYNC_PROBE");
        return on;
    }
    static inline std::atomic<u64> total{0};
    static inline std::atomic<u64> had_dirty{0};
    static inline std::atomic<u64> clean{0};
    static inline std::atomic<u64> walked_bytes{0};

    static constexpr u64 kWindow = 100000;

    static void OnCall(bool emitted_copy, u64 range_size) noexcept {
        (emitted_copy ? had_dirty : clean).fetch_add(1, std::memory_order_relaxed);
        walked_bytes.fetch_add(range_size, std::memory_order_relaxed);
        const u64 n = total.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n % kWindow == 0) {
            Report(n);
        }
    }
    static void Report(u64 n) noexcept {
        const u64 hd = had_dirty.load(std::memory_order_relaxed);
        const u64 cl = clean.load(std::memory_order_relaxed);
        const u64 wb = walked_bytes.load(std::memory_order_relaxed);
        const auto pct = [n](u64 x) { return n ? 100.0 * double(x) / double(n) : 0.0; };
        LOG_INFO(Render_Vulkan,
                 "[SYNC_PROBE] sync={} had_dirty={} ({:.2f}%) clean={} ({:.2f}%) "
                 "avg_walk={}B -> E2 has a target iff clean dominates",
                 n, hd, pct(hd), cl, pct(cl), n ? wb / n : 0);
    }
};

} // namespace GR2Fork::PhDE

// ---- call-site macros (inert unless the matching env var is set) ------------
#define GR2_PHDE_DYN_KEYDIRTY(is_ud)                                                                \
    do {                                                                                            \
        if (::GR2Fork::PhDE::DynProbe::On()) [[unlikely]]                                           \
            ::GR2Fork::PhDE::DynProbe::OnKeyDirty(is_ud);                                            \
    } while (0)

#define GR2_PHDE_DYN_ENTRY(dirty)                                                                  \
    do {                                                                                            \
        if (::GR2Fork::PhDE::DynProbe::On()) [[unlikely]]                                           \
            ::GR2Fork::PhDE::DynProbe::OnDynEntry(dirty);                                            \
    } while (0)

#define GR2_PHDE_BR_ENTRY(stamp_match, rt_match, validations_pass)                                  \
    do {                                                                                            \
        if (::GR2Fork::PhDE::BrProbe::On()) [[unlikely]]                                            \
            ::GR2Fork::PhDE::BrProbe::OnEntry(stamp_match, rt_match, validations_pass);             \
    } while (0)

#define GR2_PHDE_VBB_EXIT(exit_kind)                                                                \
    do {                                                                                            \
        if (::GR2Fork::PhDE::VbbProbe::On()) [[unlikely]]                                           \
            ::GR2Fork::PhDE::VbbProbe::OnExit(exit_kind);                                            \
    } while (0)

#define GR2_PHDE_SYNC_CALL(emitted, range)                                                          \
    do {                                                                                            \
        if (::GR2Fork::PhDE::SyncProbe::On()) [[unlikely]]                                          \
            ::GR2Fork::PhDE::SyncProbe::OnCall(emitted, range);                                      \
    } while (0)
