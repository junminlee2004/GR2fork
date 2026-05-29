// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// =============================================================================
// GpuAsse Phase D+E probe harness  (gr2fork, GR2 CUSA04943)
// =============================================================================
// Header-only, env-gated, zero-cost-when-off measurement scaffold for the four
// Phase D+E levers described in GpuAsse_PhaseDE_Handoff.md. Each probe is
// gated by its own environment variable so a lever can be measured in
// isolation; when the var is unset the only residual cost is one
// predicted-not-taken branch per hook site.
//
//   GR2FORK_DYN_PROBE   -> D3'  (dynamic_dirty suppression on user_data writes)
//   GR2FORK_BR_PROBE    -> D1   (BeginRendering.br_cache_ stamp -> rt_hash rekey)
//   GR2FORK_VBB_PROBE    -> E1   (BindVertexBuffersLegacy input-sig split)
//   GR2FORK_SYNC_PROBE  -> E2   (per-buffer DMA-sync redundant-scan gate)
//
// Conventions (mirrors the A.1/A.2 report discipline):
//   * Percentages are authoritative; the ns figures are timer-inflated and are
//     reported only for relative comparison. Discard window #1 (cold caches).
//   * One LOG_INFO per window. Windows are counted in domain-specific events.
//   * Common::FencedRDTSC for the cheap timestamp; tsc->ns factor is
//     self-calibrated once against std::chrono on first use.
//
// STRIP-BEFORE-SHIP: this header and every GR2_PHDE_* call site is diagnostic.
// Remove the #include and the macro sites before a release build. With all
// four env vars unset the harness is already inert, but the call sites are not
// part of the shipped optimization and should not survive into main.
// =============================================================================
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include "common/logging/log.h"
#include "common/rdtsc.h"
#include "common/types.h"

namespace GR2Fork::PhDE {

// ---- one-time tsc->ns calibration -----------------------------------------
// Self-calibrating: sample the wall clock across a short FencedRDTSC interval
// the first time a window reports. Good enough for relative reporting; the
// handoff treats the ns numbers as inflated/illustrative anyway.
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

// =============================================================================
// D3' — GR2FORK_DYN_PROBE
// =============================================================================
// Producer side (SetShReg key-dirty site): every key-dirty SH write is counted
// as keydirty_total; the subset classified user_data-only by
// IsUserDataOnlyShReg is counted as userdata_only.
// Consumer side (UpdateDynamicState entry): dyn_dirty_true vs dyn_dirty_false.
//
// Expect: dyn_dirty_true ~= 100% (matches A.1 keydirty=0% from the consumer
// side) and userdata_only high -> confirms ~0.48% is spurious and recoverable
// by D3'. The decision rule lives in the handoff (§4 D3').
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

// =============================================================================
// D1 — GR2FORK_BR_PROBE
// =============================================================================
// At BeginRendering entry, classify the would-be exit under each key:
//   stamp_hit       : current key (br_cache_.stamp == cur_stamp) AND validations pass
//   rtkey_would_hit : br_cache_.rt_hash == last_rt_hash_ AND pipeline match AND
//                     all the existing validations (meta-clear / layout /
//                     needs_rebind) would pass
//   miss            : neither
// Expect: stamp_hit ~= 0%, rtkey_would_hit high (mirrors rt_cache_'s rate).
// Decision: rtkey_would_hit >~ 70% -> ship D1.
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

// =============================================================================
// E1 — GR2FORK_VBB_PROBE
// =============================================================================
// At BindVertexBuffersLegacy, classify the exit:
//   stamp_skip                       : level-1 ultra-fast skip taken
//   bindsig_skip                     : bind_sig == last_vertex_bind_sig
//   inputsig_stable_but_bindsig_miss : bind_sig miss but input_sig unchanged
//                                       (the bucket E1 would reclaim)
//   full                             : input_sig also changed
// If inputsig_stable_but_bindsig_miss is large -> E1 has a target (~0.33% /
// 193 Mcyc, the GetVertexInputs walk). If full is rare -> near-floor, skip E1.
//
// NOTE: a safe E1 still needs a key that captures sharp format/stride, not just
// {pipeline, step_rate}; this probe only sizes the opportunity.
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

// =============================================================================
// E2 — GR2FORK_SYNC_PROBE
// =============================================================================
// Per SynchronizeBuffer call: had_dirty (emitted >=1 copy) vs clean (zero
// copies), plus the walked range size. If clean dominates -> the scan is mostly
// redundant -> E2 has a target. If had_dirty is common -> buffers genuinely
// stream and the scan is irreducible. Measure before any code.
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
