// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <span>
#include <thread>
#include <queue>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

class MasterSemaphore {
public:
    explicit MasterSemaphore(const Instance& instance_);
    ~MasterSemaphore();

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return KnownGpuTick() >= tick;
    }

    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] vk::Semaphore Handle() const noexcept {
        return semaphore.get();
    }

    /// Refresh the known GPU tick
    void Refresh();

    /// Waits for a tick to be hit on the GPU
    void Wait(u64 tick);

    /// Waits for a tick with a timeout; returns true when the tick was reached.
    bool WaitFor(u64 tick, u64 timeout_ns);

protected:
    const Instance& instance;
    vk::UniqueSemaphore semaphore;    ///< Timeline semaphore.
    std::atomic<u64> gpu_tick{0};     ///< Current known GPU tick.
    std::atomic<u64> current_tick{1}; ///< Current logical tick.
};

/// readback_offload: a second queue with its own timeline. A readback copy
/// submitted here waits only for the master tick that wrote its source, so it
/// runs beside the batches the GPU thread has since run ahead and submitted
/// instead of behind them. The GPU command thread submits; any thread waits.
class TransferQueue {
public:
    explicit TransferQueue(const Instance& instance, MasterSemaphore& master);
    ~TransferQueue();

    /// Submits one buffer copy after master tick `wait_master_tick`, which must
    /// belong to an already submitted batch; returns the tick of this queue's
    /// timeline that signals its completion.
    u64 SubmitCopy(u64 wait_master_tick, vk::Buffer src, vk::Buffer dst,
                   std::span<const vk::BufferCopy> copies);

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return gpu_tick.load(std::memory_order_acquire) >= tick;
    }

    void Refresh();
    void Wait(u64 tick);
    bool WaitFor(u64 tick, u64 timeout_ns);

private:
    static constexpr size_t NumSlots = 8;
    const Instance& instance;
    MasterSemaphore& master;
    vk::UniqueCommandPool command_pool;
    std::array<vk::CommandBuffer, NumSlots> cmdbufs{};
    std::array<u64, NumSlots> slot_ticks{}; ///< Tick each slot's last submit signals.
    size_t next_slot{};
    vk::UniqueSemaphore semaphore; ///< Timeline semaphore.
    std::atomic<u64> gpu_tick{0};
    u64 current_tick{1};
};

} // namespace Vulkan
