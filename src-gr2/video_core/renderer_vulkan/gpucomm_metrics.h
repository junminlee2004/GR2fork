// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// No-op stubs for the GR2_INSTR_* metrics macros still referenced by vk_rasterizer.cpp at ~19
// call sites. Each macro expands to ((void)0) or ((void)(arg)) so unused-parameter and
// unused-expression warnings stay quiet without the metrics runtime.

#include <cstdint>

namespace Vulkan::detail {

// No-op timer; ElapsedNs() returns 0. Call sites still invoke .ElapsedNs() on the declared
// value, so the interface must exist.
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

// Declares a stack-local DisabledSlowPathTimer. [[maybe_unused]] silences warnings on call
// paths where the timer is declared but the consuming macro is also a no-op.
#define GR2_INSTR_TIMER_DECL(name)                                                                \
    [[maybe_unused]] ::Vulkan::detail::DisabledSlowPathTimer name
