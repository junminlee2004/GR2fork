// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// FIX(GR2FORK): continuous stall/hang diagnostic. Runs a dedicated thread
// that samples subsystem health counters every second. The last lines in
// the log before a hang show exactly which subsystem stopped advancing
// (scheduler-tick-stuck = GPU thread hung; num_submits climbing = main
// thread queuing but GPU not draining; VRAM climbing = leak or working-
// set blow-up; everything frozen = whole process stalled). Zero cost in
// the steady state beyond one locked atomic load per sampled counter.

#pragma once

#include <functional>
#include "common/types.h"

namespace Common {

struct HangWatchdogCallbacks {
    std::function<u64()>    scheduler_tick;  // monotonic submit tick
    std::function<u32()>    num_submits;     // pending GPU submits
    std::function<u64()>    vram_used;       // bytes; 0 if unsupported
    std::function<u64()>    vram_budget;     // bytes total; 0 if unsupported
    std::function<u64()>    texture_mem;     // TextureCache::total_used_memory
    std::function<size_t()> num_images;      // optional; 0 if not wired
    std::function<size_t()> num_buffers;     // optional; 0 if not wired
};

class HangWatchdog {
public:
    /// Start the watchdog thread. Safe to call once. No-op on subsequent calls.
    static void Start(HangWatchdogCallbacks cb);

    /// Stop and join the watchdog thread. Call during shutdown. Idempotent.
    static void Stop();

    /// Called by render code right before the main GPU submit to reset the
    /// stall timer. Optional — without it, we rely on scheduler_tick advance
    /// as the liveness signal.
    static void KickMainThread() noexcept;
};

} // namespace Common
