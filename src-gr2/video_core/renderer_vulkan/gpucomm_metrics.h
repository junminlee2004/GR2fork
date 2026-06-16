// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// =============================================================================
// gpucomm_metrics.h — STUB for v1.13 baseline.
//
// The full GpuComm metrics infrastructure (CSV recording, atomic counters,
// per-frame timer ladder) was removed in the v1.13 lean baseline along with
// the rest of the instrumentation. vk_rasterizer.cpp still references the
// GR2_INSTR_* macros at ~19 call sites, so this stub provides no-op versions
// to keep the source compiling without the metrics runtime.
//
// All macros here expand to either ((void)0) or a degenerate ((void)(arg))
// pattern so unused-parameter and unused-expression warnings stay quiet.
// The TIMER_DECL macro declares a tiny struct with an ElapsedNs() method
// because vk_rasterizer.cpp calls .ElapsedNs() on the declared timer value.
//
// To re-enable real metrics, replace this header with the v1.20 version
// (which has the full GpuCommMetrics class + atomic counters + CSV writer).
// =============================================================================

#include <cstdint>

namespace Vulkan::detail {

// No-op timer. Same interface as v1.20's SlowPathTimer (ElapsedNs() returns
// 0 here). Marked [[maybe_unused]] at declaration via the macro to suppress
// the unused-variable warning when the timer's elapsed value is never read
// (in some call paths the timer is declared but the macro that consumes it
// is also a no-op).
struct DisabledSlowPathTimer {
    DisabledSlowPathTimer() noexcept = default;
    DisabledSlowPathTimer(const DisabledSlowPathTimer&) = delete;
    DisabledSlowPathTimer& operator=(const DisabledSlowPathTimer&) = delete;

    [[nodiscard]] std::uint64_t ElapsedNs() const noexcept { return 0; }
};

} // namespace Vulkan::detail

// Per-draw counters
#define GR2_INSTR_ON_DRAW(pipeline)                ((void)(pipeline))
#define GR2_INSTR_ON_DRAW_INDIRECT()               ((void)0)
#define GR2_INSTR_ON_SAME_PIPELINE()               ((void)0)

// BindResources entry / fast path / slow path
#define GR2_INSTR_ON_BR_ENTER()                    ((void)0)
#define GR2_INSTR_ON_BR_FAST_REPLAY()              ((void)0)
#define GR2_INSTR_ON_BR_FAST_PUSHUD()              ((void)0)
#define GR2_INSTR_ON_BR_REPLAY()                   ((void)0)
#define GR2_INSTR_ON_BR_SLOW(ns, stages)           ((void)(ns)), ((void)(stages))

// Per-binding match counters
#define GR2_INSTR_ON_IMAGE_SHARP_MATCH()           ((void)0)
#define GR2_INSTR_ON_BUFFER_SHARP_MATCH()          ((void)0)
#define GR2_INSTR_ON_BT_REPLAY()                   ((void)0)

// Timing ladder
#define GR2_INSTR_ON_BIND_VBUF(ns)                 ((void)(ns))
#define GR2_INSTR_ON_DRAW_DISPATCH(ns)             ((void)(ns))
#define GR2_INSTR_ON_SYNC_BUFFER(ns)               ((void)(ns))
#define GR2_INSTR_ON_UPLOAD_COPIES(ns, count, bytes)                                              \
    ((void)(ns)), ((void)(count)), ((void)(bytes))

// Frame boundary
#define GR2_INSTR_ON_FLIP()                        ((void)0)
#define GR2_INSTR_FLUSH()                          ((void)0)

// Timer declaration. Expands to a stack-local DisabledSlowPathTimer with the
// requested name. The [[maybe_unused]] attribute prevents warnings on call
// paths where the timer is declared but never read (because the consuming
// macro is also a no-op). The declared value still has .ElapsedNs() so
// existing call sites like `GR2_INSTR_ON_BR_SLOW(slow_timer_.ElapsedNs(), ...)`
// continue to compile.
#define GR2_INSTR_TIMER_DECL(name)                                                                \
    [[maybe_unused]] ::Vulkan::detail::DisabledSlowPathTimer name
