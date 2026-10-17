// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include "common/logging/log.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_lane.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

namespace VideoCore {

struct ReadbackLanes::Lane {
    Lane(const Vulkan::Instance& instance, vk::Queue queue_, u32 family)
        : queue{queue_}, timeline{instance}, staging{instance, 0, BufferCache::READBACK_WINDOW_SIZE,
                                                     MemoryType::HostCached, "Readback lane"} {
        const vk::Device device = instance.GetDevice();
        const vk::CommandPoolCreateInfo pool_ci = {
            .flags = vk::CommandPoolCreateFlagBits::eTransient |
                     vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = family,
        };
        pool = Vulkan::Check(device.createCommandPoolUnique(pool_ci));
        const vk::CommandBufferAllocateInfo alloc_info = {
            .commandPool = *pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };
        Vulkan::Check(device.allocateCommandBuffers(&alloc_info, &cmdbuf));
    }

    vk::Queue queue;
    Vulkan::Semaphore timeline;
    vk::UniqueCommandPool pool;
    vk::CommandBuffer cmdbuf;
    Buffer staging;
    BufferCache::ReadbackCopies copies;
    std::jthread thread;
};

ReadbackLanes::ReadbackLanes(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                             BufferCache& buffer_cache_, vk::Semaphore bind_semaphore_)
    : instance{instance_}, scheduler{scheduler_}, buffer_cache{buffer_cache_},
      bind_semaphore{bind_semaphore_} {
    const auto copy_queues = instance.GetCopyQueues();
    const bool shared = copy_queues.empty();
    const u32 count = shared ? 2 : static_cast<u32>(copy_queues.size());
    for (u32 i = 0; i < count; ++i) {
        const vk::Queue queue = shared ? instance.GetGraphicsQueue() : copy_queues[i];
        const u32 family =
            shared ? instance.GetGraphicsQueueFamilyIndex() : instance.GetCopyQueueFamilyIndex();
        auto& lane = lanes.emplace_back(std::make_unique<Lane>(instance, queue, family));
        lane->thread =
            std::jthread([this, lane = lane.get()](std::stop_token stop) { Run(*lane, stop); });
    }
    LOG_INFO(Render_Vulkan, "Readback copy lanes: {} on {}", count,
             shared ? "the graphics queue" : "a transfer queue family");
}

ReadbackLanes::~ReadbackLanes() {
    {
        std::scoped_lock lk{mutex};
        stopping = true;
    }
    lanes.clear();
}

std::optional<u64> ReadbackLanes::Copy(const Request& request) {
    Job job{request, 0, false};
    std::unique_lock lk{mutex};
    if (stopping) {
        return std::nullopt;
    }
    pending.push_back(&job);
    job_cv.notify_one();
    done_cv.wait(lk, [&] { return job.done; });
    return job.copied;
}

void ReadbackLanes::Run(Lane& lane, std::stop_token stop) {
    Common::SetCurrentThreadName("shadPS4:Readback");
    while (true) {
        Job* job{};
        {
            std::unique_lock lk{mutex};
            Common::CondvarWait(job_cv, lk, stop, [this] { return !pending.empty(); });
            if (pending.empty()) {
                return;
            }
            job = pending.front();
            pending.erase(pending.begin());
        }
        Process(lane, *job);
        {
            std::scoped_lock lk{mutex};
            job->done = true;
        }
        done_cv.notify_all();
    }
}

void ReadbackLanes::Process(Lane& lane, Job& job) {
    const Request& request = job.request;
    lane.copies.clear();
    const u64 total_size =
        buffer_cache.CollectWrites(request.arena, request.window, BufferCache::READBACK_WINDOW_SIZE,
                                   request.copy_tick, lane.copies);
    job.copied = total_size;
    if (total_size == 0) {
        return;
    }
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    Vulkan::Check(lane.cmdbuf.begin(begin_info));
    lane.cmdbuf.copyBuffer(request.arena->Handle(), lane.staging.Handle(), lane.copies);
    const vk::MemoryBarrier2 barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    lane.cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
    Vulkan::Check(lane.cmdbuf.end());

    // The GPU reached wait_tick before the owner handed the window over, so that wait is already
    // satisfied and legal on the graphics queue as well. The arena's sparse binds signal the bind
    // semaphore from the graphics queue, which a lane on another queue waits for explicitly; before
    // the first bind submission the value is 0, which the semaphore satisfied at creation.
    const std::array wait_semaphores = {scheduler.GetWorkSemaphore()->Handle(), bind_semaphore};
    const std::array wait_values = {request.wait_tick, request.bind_tick};
    const std::array<vk::PipelineStageFlags, 2> wait_stages = {
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer};
    const vk::Semaphore signal_semaphore = lane.timeline.Handle();
    const u64 signal_tick = lane.timeline.NextTick();
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = 2,
        .pWaitSemaphoreValues = wait_values.data(),
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &signal_tick,
    };
    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = 2,
        .pWaitSemaphores = wait_semaphores.data(),
        .pWaitDstStageMask = wait_stages.data(),
        .commandBufferCount = 1,
        .pCommandBuffers = &lane.cmdbuf,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_semaphore,
    };
    {
        // Without a transfer family the lanes share the scheduler's graphics queue, whose submits
        // take the scheduler's submit lock.
        std::unique_lock lk{Vulkan::Scheduler::submit_mutex, std::defer_lock};
        if (instance.GetCopyQueues().empty()) {
            lk.lock();
        }
        Vulkan::Check(lane.queue.submit(submit_info));
    }
    lane.timeline.Wait(signal_tick);

    lane.staging.Invalidate(0, total_size);
    buffer_cache.WriteBack(request.arena->CpuAddr(), lane.copies, lane.staging.mapped_data.data(),
                           0);
}

} // namespace VideoCore
