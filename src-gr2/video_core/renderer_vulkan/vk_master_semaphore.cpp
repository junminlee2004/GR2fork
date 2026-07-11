// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include <chrono>
#include "video_core/renderer_vulkan/vk_device_recovery.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

#include "common/assert.h"

namespace Vulkan {

constexpr u64 WAIT_TIMEOUT = std::numeric_limits<u64>::max();

MasterSemaphore::MasterSemaphore(const Instance& instance_) : instance{instance_} {
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);
}

MasterSemaphore::~MasterSemaphore() = default;

void MasterSemaphore::Refresh() {
    // Rate-limit expensive getSemaphoreCounterValue() calls (often an ioctl on RADV).
    // This reduces CPU overhead when Refresh() is called frequently in tight loops.
    constexpr u64 kMinRefreshIntervalNs = 100'000; // 100us
    const u64 now_ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    const u64 last_ns = last_refresh_ns.load(std::memory_order_relaxed);
    // GR2FORK PERF: Refresh() is called from hot loops (CommitResource,
    // Wait, scheduler tick checks) and the 100us throttle catches most of
    // those calls, so hint the early return.
    if (now_ns - last_ns < kMinRefreshIntervalNs) [[likely]] {
        return;
    }
    last_refresh_ns.store(now_ns, std::memory_order_relaxed);
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        // GR2FORK: on a lost device with recovery enabled, bail quietly instead of aborting - the
        // rebuild handles it. Default (recovery off) keeps the fatal assert below.
        if (counter_result != vk::Result::eSuccess && DeviceRecoveryEnabled() &&
            g_device_lost.load(std::memory_order_acquire)) [[unlikely]] {
            return;
        }
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        // GR2FORK: the counter regresses below this_tick only when another
        // thread bumps gpu_tick concurrently between the load and the
        // getSemaphoreCounterValue call - a rare race.
        if (counter < this_tick) [[unlikely]] {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void MasterSemaphore::Wait(u64 tick) {
    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        return;
    }
    // Update the GPU tick and try again
    Refresh();
    if (IsFree(tick)) {
        return;
    }

    // If none of the above is hit, fallback to a regular wait
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    // GR2FORK: recovery opt-in - finite timeout + device-lost break so a lost device does not
    // busy-spin forever. Default (recovery off) keeps the original infinite wait.
    const u64 wait_ns = DeviceRecoveryEnabled() ? 1'000'000'000ull : WAIT_TIMEOUT;
    while (instance.GetDevice().waitSemaphores(&wait_info, wait_ns) != vk::Result::eSuccess) {
        if (DeviceRecoveryEnabled() && g_device_lost.load(std::memory_order_acquire)) {
            return;
        }
    }
    Refresh();
}

} // namespace Vulkan
