// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <optional>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace Vulkan {

class Instance;
class MasterSemaphore;

/**
 * ComputeScheduler — minimal async-compute submission lane (Phase MQ-3).
 *
 * Pairs with the existing graphics Scheduler. Records compute work into a
 * cmdbuf bound to the device's compute queue family, submits to that queue.
 * Shares the graphics Scheduler's MasterSemaphore so signal ticks come from
 * one monotonic timeline; cross-queue dependencies (graphics→compute or
 * compute→graphics) are expressed by waiting on a tick value from that
 * timeline.
 *
 * When Instance::IsAsyncComputeAvailable() is false (no dedicated compute
 * family on this device), the underlying compute_queue handle aliases the
 * graphics_queue. Submissions still work — they just serialize on the same
 * VkQueue and gain no GPU-side overlap. The shared `Scheduler::submit_mutex`
 * is held during submit to handle this aliasing case correctly.
 *
 * MQ-3 ships the class with no callers wired up. MQ-4 routes the first
 * compute dispatch to it.
 *
 * Threading: per-pool external sync rules apply — only one thread should
 * be calling CommandBuffer()/Submit() at a time. Initially the GpuComm
 * thread, the same one driving the graphics Scheduler.
 */
class ComputeScheduler {
public:
    explicit ComputeScheduler(const Instance& instance, MasterSemaphore* master_semaphore);
    ~ComputeScheduler();

    ComputeScheduler(const ComputeScheduler&) = delete;
    ComputeScheduler& operator=(const ComputeScheduler&) = delete;

    /// Returns a command buffer ready to record compute work into.
    /// Auto-allocates and Begin()s a fresh cmdbuf if no recording is in
    /// progress. Subsequent calls return the same handle until Submit().
    vk::CommandBuffer CommandBuffer();

    /// Submits the current command buffer (if any) to the compute queue and
    /// signals the next tick on the shared timeline.
    ///
    /// `wait_tick`: optional tick on the shared MasterSemaphore that must
    ///              complete before this compute work begins. Use when
    ///              compute reads data produced by graphics (post-process,
    ///              detile of a freshly-rendered RT, etc.). Wait stage is
    ///              eAllCommands to be conservative until callers are
    ///              audited per-dispatch.
    ///
    /// Returns the tick this submission will signal upon completion. When
    /// no work has been recorded, returns the current tick unchanged and
    /// performs no submit.
    u64 Submit(std::optional<u64> wait_tick = std::nullopt);

    /// Waits for a specific tick to complete on the GPU. Same shared
    /// timeline as the graphics Scheduler.
    void Wait(u64 tick);

    /// Returns the current logical tick on the shared timeline.
    [[nodiscard]] u64 CurrentTick() const noexcept;

    /// True if this scheduler is bound to a compute queue family distinct
    /// from the graphics family (genuine async). When false, submissions
    /// still work but serialize against graphics on the same VkQueue.
    [[nodiscard]] bool IsAsyncCapable() const noexcept;

private:
    void BeginIfNeeded();

    const Instance& instance;
    MasterSemaphore* master_semaphore;
    CommandPool command_pool;
    vk::CommandBuffer current_cmdbuf{};
    bool recording{false};
};

} // namespace Vulkan
