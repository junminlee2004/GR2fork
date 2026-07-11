// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <boost/container/static_vector.hpp>

#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_compute_scheduler.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

ComputeScheduler::ComputeScheduler(const Instance& instance_,
                                   MasterSemaphore* master_semaphore_)
    : instance{instance_}, master_semaphore{master_semaphore_},
      command_pool{instance_, master_semaphore_,
                   instance_.GetComputeQueueFamilyIndex()} {
    ASSERT_MSG(master_semaphore != nullptr,
               "ComputeScheduler requires a shared MasterSemaphore");
}

ComputeScheduler::~ComputeScheduler() {
    // A cmdbuf left recording needs no explicit end/discard - destroying the underlying
    // VkCommandPool frees all its cmdbufs.
}

void ComputeScheduler::BeginIfNeeded() {
    if (recording) {
        return;
    }
    current_cmdbuf = command_pool.Commit();
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    const auto begin_result = current_cmdbuf.begin(begin_info);
    ASSERT_MSG(begin_result == vk::Result::eSuccess,
               "Failed to begin compute cmdbuf: {}", vk::to_string(begin_result));
    recording = true;
}

vk::CommandBuffer ComputeScheduler::CommandBuffer() {
    BeginIfNeeded();
    return current_cmdbuf;
}

u64 ComputeScheduler::Submit(std::optional<u64> wait_tick) {
    if (!recording) {
        return master_semaphore->CurrentTick();
    }

    const auto end_result = current_cmdbuf.end();
    ASSERT_MSG(end_result == vk::Result::eSuccess,
               "Failed to end compute cmdbuf: {}", vk::to_string(end_result));

    const u64 signal_value = master_semaphore->NextTick();
    const vk::Semaphore timeline = master_semaphore->Handle();

    // Wait/signal arrays for the timeline submit info - kept small since
    // there is only the one shared timeline. Per-Scheduler semaphores would
    // expand these; the shared-timeline architecture keeps them at 1 each.
    boost::container::static_vector<u64, 2> wait_values;
    boost::container::static_vector<vk::Semaphore, 2> wait_semas;
    boost::container::static_vector<vk::PipelineStageFlags, 2> wait_stage_masks;
    if (wait_tick.has_value()) {
        wait_values.push_back(*wait_tick);
        wait_semas.push_back(timeline);
        // eAllCommands is conservative but covers any compute access pattern; per-dispatch
        // analysis could narrow this (e.g. eComputeShader for pure shader reads).
        wait_stage_masks.push_back(vk::PipelineStageFlagBits::eAllCommands);
    }

    const std::array signal_values = {signal_value};
    const std::array signal_semas = {timeline};

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = static_cast<u32>(wait_values.size()),
        .pWaitSemaphoreValues = wait_values.data(),
        .signalSemaphoreValueCount = static_cast<u32>(signal_values.size()),
        .pSignalSemaphoreValues = signal_values.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = static_cast<u32>(wait_semas.size()),
        .pWaitSemaphores = wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = 1u,
        .pCommandBuffers = &current_cmdbuf,
        .signalSemaphoreCount = static_cast<u32>(signal_semas.size()),
        .pSignalSemaphores = signal_semas.data(),
    };

    {
        // Share Scheduler::submit_mutex: when async compute is unavailable GetComputeQueue()
        // aliases the graphics queue, and vkQueueSubmit requires external sync per VkQueue.
        std::scoped_lock lk{Scheduler::submit_mutex};
        const auto submit_result =
            instance.GetComputeQueue().submit(submit_info, vk::Fence{});
        // GR2FORK FIX: dump VK_EXT_device_fault info before the assert below runs
        // assert_fail_impl() and Common::Log::Stop(), mirroring the graphics-side dump.
        if (submit_result == vk::Result::eErrorDeviceLost) [[unlikely]] {
            Scheduler::DumpDeviceFaultInfo(instance, "compute submit");
        }
        ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost,
                   "Device lost during compute submit");
        ASSERT_MSG(submit_result == vk::Result::eSuccess,
                   "Compute queue submit failed: {}", vk::to_string(submit_result));
    }
    master_semaphore->Refresh();

    recording = false;
    current_cmdbuf = vk::CommandBuffer{};
    return signal_value;
}

void ComputeScheduler::Wait(u64 tick) {
    master_semaphore->Wait(tick);
}

u64 ComputeScheduler::CurrentTick() const noexcept {
    return master_semaphore->CurrentTick();
}

bool ComputeScheduler::IsAsyncCapable() const noexcept {
    return instance.IsAsyncComputeAvailable();
}

} // namespace Vulkan
