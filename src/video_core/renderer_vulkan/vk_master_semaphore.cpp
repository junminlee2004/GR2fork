// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

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
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
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

    while (instance.GetDevice().waitSemaphores(&wait_info, WAIT_TIMEOUT) != vk::Result::eSuccess) {
    }
    Refresh();
}

bool MasterSemaphore::WaitFor(u64 tick, u64 timeout_ns) {
    if (IsFree(tick)) {
        return true;
    }
    Refresh();
    if (IsFree(tick)) {
        return true;
    }

    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    if (instance.GetDevice().waitSemaphores(&wait_info, timeout_ns) != vk::Result::eSuccess) {
        return false;
    }
    Refresh();
    return true;
}

TransferQueue::TransferQueue(const Instance& instance_, MasterSemaphore& master_, bool on_graphics)
    : instance{instance_}, master{master_}, on_graphics_{on_graphics} {
    const auto device = instance.GetDevice();
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] = device.createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
               "Failed to create copy queue semaphore: {}", vk::to_string(semaphore_result));
    semaphore = std::move(sem);

    const vk::CommandPoolCreateInfo pool_info = {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        // The pool's family has to match the queue the copies are submitted to.
        .queueFamilyIndex = on_graphics_ ? instance.GetGraphicsQueueFamilyIndex()
                                         : instance.GetTransferQueueFamilyIndex(),
    };
    auto [pool_result, pool] = device.createCommandPoolUnique(pool_info);
    ASSERT_MSG(pool_result == vk::Result::eSuccess, "Failed to create copy queue command pool: {}",
               vk::to_string(pool_result));
    command_pool = std::move(pool);

    const vk::CommandBufferAllocateInfo alloc_info = {
        .commandPool = *command_pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = static_cast<u32>(NumSlots),
    };
    const auto alloc_result = device.allocateCommandBuffers(&alloc_info, cmdbufs.data());
    ASSERT_MSG(alloc_result == vk::Result::eSuccess,
               "Failed to allocate copy queue command buffers: {}", vk::to_string(alloc_result));
}

TransferQueue::~TransferQueue() {
    // Every submit retires before its command buffer and the timeline go.
    Wait(current_tick - 1);
}

u64 TransferQueue::SubmitCopy(u64 wait_master_tick, vk::Buffer src, vk::Buffer dst,
                              std::span<const vk::BufferCopy> copies) {
    const size_t slot = next_slot;
    next_slot = (next_slot + 1) % NumSlots;
    // A slot is reused only once its previous submit has retired.
    Wait(slot_ticks[slot]);
    const auto cmdbuf = cmdbufs[slot];
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    const auto begin_result = cmdbuf.begin(begin_info);
    ASSERT_MSG(begin_result == vk::Result::eSuccess,
               "Failed to begin copy queue command buffer: {}", vk::to_string(begin_result));
    if (on_graphics_) {
        // Redundant with the master wait below (already a full memory
        // dependency); mirrors the in-batch pre_barrier in buffer_cache.cpp
        // PrepareFaultDownload.
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src,
            .offset = 0,
            .size = vk::WholeSize,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });
    }
    // The master wait below is a full memory dependency for the writer's
    // batch, so no barrier precedes the copy on the transfer route.
    cmdbuf.copyBuffer(src, dst, copies);
    const auto end_result = cmdbuf.end();
    ASSERT_MSG(end_result == vk::Result::eSuccess, "Failed to end copy queue command buffer: {}",
               vk::to_string(end_result));

    const u64 signal_value = current_tick++;
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &wait_master_tick,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &signal_value,
    };
    const vk::Semaphore wait_sema = master.Handle();
    const vk::Semaphore signal_sema = *semaphore;
    static constexpr vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_sema,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdbuf,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_sema,
    };
    // Graphics route: legal only because the caller guarantees the signalling
    // batch is already submitted (writer_tick < scheduler.CurrentTick(), in
    // buffer_cache.cpp PrepareFaultDownload); on the in-order graphics ring a
    // wait on a not-yet-submitted tick hangs the ring. Never relax that. The
    // queue is shared with the scheduler's batches (GPU command thread and
    // presenter) and with present, so the submit takes the same static
    // Scheduler::submit_mutex; the presenter's next batch can still win it.
    vk::Result submit_result;
    if (on_graphics_) {
        std::scoped_lock lk{Scheduler::submit_mutex};
        submit_result = instance.GetGraphicsQueue().submit(submit_info);
    } else {
        submit_result = instance.GetTransferQueue().submit(submit_info);
    }
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost,
               "Device lost during copy queue submit");
    slot_ticks[slot] = signal_value;
    return signal_value;
}

void TransferQueue::Refresh() {
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get copy queue semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void TransferQueue::Wait(u64 tick) {
    if (IsFree(tick)) {
        return;
    }
    while (!WaitFor(tick, WAIT_TIMEOUT)) {
    }
}

bool TransferQueue::WaitFor(u64 tick, u64 timeout_ns) {
    if (IsFree(tick)) {
        return true;
    }
    Refresh();
    if (IsFree(tick)) {
        return true;
    }
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };
    if (instance.GetDevice().waitSemaphores(&wait_info, timeout_ns) != vk::Result::eSuccess) {
        return false;
    }
    Refresh();
    return true;
}

} // namespace Vulkan
