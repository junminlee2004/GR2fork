// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

#include "common/assert.h"
#include "common/debug.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_device_recovery.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, master_semaphore{instance}, command_pool{instance, &master_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    AllocateWorkerCommandBuffers();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    // GR2FORK PERF: trimmed commit copy - active span only. Stale bytes in inactive slots are
    // never read (colorAttachmentCount below) and the active-field operator== above never
    // compares them. Full assignment restored under GR2_NORSTRIM=1.
    if (RStrimEnabled()) [[likely]] {
        render_state.CopyActiveFrom(new_state);
    } else {
        render_state = new_state;
    }

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = render_state.num_color_attachments > 0
                                 ? render_state.color_attachments.data()
                                 : nullptr,
        .pDepthAttachment = render_state.has_depth ? &render_state.depth_attachment : nullptr,
        .pStencilAttachment = render_state.has_stencil ? &render_state.stencil_attachment : nullptr,
    };

    current_cmdbuf.beginRendering(rendering_info);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    current_cmdbuf.endRendering();
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick) {
    if (tick >= master_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    master_semaphore.Wait(tick);
}

void Scheduler::PopPendingOperations() {
    if (pending_ops.empty()) [[likely]] {
        return;
    }
    // GR2FORK PERF: consult the already-known GPU tick before paying Refresh's counter
    // query (a drmSyncobjQuery ioctl on RADV; this call site was measured issuing ~6k/s =
    // 1.36% of the assembler's wall). If the oldest pending op's tick is not yet known
    // complete, skip the query - the GPU may have just passed it, but discovering that can
    // wait for the next call or any other Refresh path. The cost is bounded reclaim
    // latency; deferred ops are frees, so late is the safe direction. Refresh() itself is
    // untouched. GR2_NOPOPGATE=1 restores the unconditional query.
    static const bool pop_gate_enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOPOPGATE");
        return !(e && e[0] == '1');
    }();
    if (pop_gate_enabled && !master_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        return;
    }
    master_semaphore.Refresh();
    while (!pending_ops.empty() && master_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_cmdbuf = command_pool.Commit();
    Check(current_cmdbuf.begin(begin_info));

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

    // Cross-pipeline cmdbuf state caches are scoped to the current cmdbuf - a fresh one has no
    // pushed/bound state, so bumping the gens makes every Pipeline's cached_*_gen comparison
    // fail on the first BindResources call against this cmdbuf.
    ResetCmdbufStateGens();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = master_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    EndRendering();
    Check(current_cmdbuf.end());

    const vk::Semaphore timeline = master_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    // GR2FORK: per-wait stage masks travel in SubmitInfo (see vk_scheduler.h).

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = info.wait_masks.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &current_cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    // GR2FORK FIX: on VK_ERROR_DEVICE_LOST, query VK_EXT_device_fault BEFORE the ASSERT fires -
    // assert_fail_impl() stops logging and traps with no GPU-side detail otherwise. The ASSERT
    // still fires; the log now carries fault addresses, access type, and vendor data.
    if (submit_result == vk::Result::eErrorDeviceLost) [[unlikely]] {
        DumpDeviceFaultInfo(instance, "graphics submit");
        if (DeviceRecoveryEnabled()) {
            // GR2FORK: recovery opt-in (GR2_DEVICE_RECOVERY=1) - flag the loss and skip the fatal
            // assert. The rest of this submit would only fail on the lost device; the main thread
            // observes g_device_lost and rebuilds (Stage 2). Default keeps the assert below.
            g_device_lost.store(true, std::memory_order_release);
            return;
        }
    }
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    master_semaphore.Refresh();
    AllocateWorkerCommandBuffers();

    // Apply pending operations
    PopPendingOperations();
}

// GR2FORK FIX: VK_EXT_device_fault diagnostic helper; two-pass query per the spec (counts, then
// per-count storage). The vendor binary blob is not requested (device created with
// deviceFaultVendorBinary = false); log lines are LOG_CRITICAL to survive the assert's log stop.
void Scheduler::DumpDeviceFaultInfo(const Instance& instance, const char* context) noexcept {
    if (!instance.IsDeviceFaultSupported()) {
        LOG_CRITICAL(Render_Vulkan,
                     "[device-fault] {}: VK_ERROR_DEVICE_LOST. VK_EXT_device_fault is "
                     "NOT enabled on this device — no GPU-side fault info available. "
                     "If this reproduces, run with a driver / device that supports "
                     "VK_EXT_device_fault or with the Vulkan validation layers to "
                     "narrow down the failing draw/dispatch.",
                     context);
        return;
    }

    const vk::Device device = instance.GetDevice();

    // ---- Pass 1: counts ------------------------------------------------------
    vk::DeviceFaultCountsEXT counts{};
    const vk::Result count_result = device.getFaultInfoEXT(&counts, nullptr);
    if (count_result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan,
                     "[device-fault] {}: VK_ERROR_DEVICE_LOST. "
                     "vkGetDeviceFaultInfoEXT (counts pass) returned non-success "
                     "result={} ({}). No GPU-side fault info captured.",
                     context, vk::to_string(count_result),
                     static_cast<int>(count_result));
        return;
    }

    LOG_CRITICAL(Render_Vulkan,
                 "[device-fault] {}: VK_ERROR_DEVICE_LOST. "
                 "addressInfoCount={} vendorInfoCount={} vendorBinarySize={}",
                 context, counts.addressInfoCount, counts.vendorInfoCount,
                 static_cast<u64>(counts.vendorBinarySize));

    // ---- Pass 2: actual info -------------------------------------------------
    // Drop the vendor binary; we don't have a path to consume it (see comment
    // above). Drivers must not write a binary when vendorBinarySize stays 0.
    counts.vendorBinarySize = 0;

    std::vector<vk::DeviceFaultAddressInfoEXT> address_infos(counts.addressInfoCount);
    std::vector<vk::DeviceFaultVendorInfoEXT> vendor_infos(counts.vendorInfoCount);

    vk::DeviceFaultInfoEXT fault_info{};
    fault_info.pAddressInfos = address_infos.empty() ? nullptr : address_infos.data();
    fault_info.pVendorInfos = vendor_infos.empty() ? nullptr : vendor_infos.data();
    fault_info.pVendorBinaryData = nullptr;

    const vk::Result info_result = device.getFaultInfoEXT(&counts, &fault_info);
    if (info_result != vk::Result::eSuccess && info_result != vk::Result::eIncomplete) {
        LOG_CRITICAL(Render_Vulkan,
                     "[device-fault] {}: vkGetDeviceFaultInfoEXT (info pass) "
                     "returned non-success result={} ({}).",
                     context, vk::to_string(info_result),
                     static_cast<int>(info_result));
        return;
    }

    // `description` is a fixed-size char buffer; cap by ::strnlen so a missing NUL cannot run
    // off the end, and use .data() explicitly - ArrayWrapper1D's implicit pointer conversion is
    // not consistent across vulkan-hpp revisions.
    const std::size_t desc_max = sizeof(fault_info.description);
    const std::size_t desc_len = ::strnlen(fault_info.description.data(), desc_max);
    LOG_CRITICAL(Render_Vulkan, "[device-fault] {}: description=\"{}\"", context,
                 std::string_view(fault_info.description.data(), desc_len));

    // VkDeviceFaultAddressTypeEXT values from the spec; we print the int
    // alongside a short tag so log readers don't need to chase the enum.
    auto address_type_name = [](vk::DeviceFaultAddressTypeEXT t) -> const char* {
        switch (static_cast<int>(t)) {
        case 0: return "NONE";
        case 1: return "READ_INVALID";
        case 2: return "WRITE_INVALID";
        case 3: return "EXECUTE_INVALID";
        case 4: return "INSTRUCTION_POINTER_UNKNOWN";
        case 5: return "INSTRUCTION_POINTER_INVALID";
        case 6: return "INSTRUCTION_POINTER_FAULT";
        default: return "UNKNOWN";
        }
    };

    for (u32 i = 0; i < counts.addressInfoCount; ++i) {
        const auto& a = address_infos[i];
        LOG_CRITICAL(Render_Vulkan,
                     "[device-fault] {}: addressInfo[{}] type={} ({}) "
                     "reportedAddress=0x{:016x} addressPrecision=0x{:x}",
                     context, i, address_type_name(a.addressType),
                     static_cast<int>(a.addressType),
                     static_cast<u64>(a.reportedAddress),
                     static_cast<u64>(a.addressPrecision));
    }

    for (u32 i = 0; i < counts.vendorInfoCount; ++i) {
        const auto& v = vendor_infos[i];
        const std::size_t vd_max = sizeof(v.description);
        const std::size_t vd_len = ::strnlen(v.description.data(), vd_max);
        LOG_CRITICAL(Render_Vulkan,
                     "[device-fault] {}: vendorInfo[{}] code=0x{:016x} "
                     "data=0x{:016x} desc=\"{}\"",
                     context, i, static_cast<u64>(v.vendorFaultCode),
                     static_cast<u64>(v.vendorFaultData),
                     std::string_view(v.description.data(), vd_len));
    }
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        master_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    // OPT: Fast early-exit when nothing is dirty. Most consecutive draws with the
    // same pipeline and state produce zero dirty bits. A single comparison avoids
    // all ~25 individual flag checks.
    {
        // Bitfield struct may have padding, so compare all bytes to zero.
        static constexpr decltype(dirty_state) zero_state{};
        if (std::memcmp(&dirty_state, &zero_state, sizeof(dirty_state)) == 0) [[likely]] {
            return;
        }
    }

    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        // OPT: Batch stencil ops/refs/masks. Check front+back together to reduce branching.
        const bool front_ops_dirty = dirty_state.stencil_front_ops;
        const bool back_ops_dirty = dirty_state.stencil_back_ops;
        if (front_ops_dirty || back_ops_dirty) {
            if (front_ops_dirty && back_ops_dirty && stencil_front_ops == stencil_back_ops) {
                dirty_state.stencil_front_ops = false;
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack,
                                    stencil_front_ops.fail_op, stencil_front_ops.pass_op,
                                    stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            } else {
                if (front_ops_dirty) {
                    dirty_state.stencil_front_ops = false;
                    cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront,
                                        stencil_front_ops.fail_op, stencil_front_ops.pass_op,
                                        stencil_front_ops.depth_fail_op,
                                        stencil_front_ops.compare_op);
                }
                if (back_ops_dirty) {
                    dirty_state.stencil_back_ops = false;
                    cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack,
                                        stencil_back_ops.fail_op, stencil_back_ops.pass_op,
                                        stencil_back_ops.depth_fail_op,
                                        stencil_back_ops.compare_op);
                }
            }
        }
        const bool front_ref_dirty = dirty_state.stencil_front_reference;
        const bool back_ref_dirty = dirty_state.stencil_back_reference;
        if (front_ref_dirty || back_ref_dirty) {
            if (front_ref_dirty && back_ref_dirty &&
                stencil_front_reference == stencil_back_reference) {
                dirty_state.stencil_front_reference = false;
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                           stencil_front_reference);
            } else {
                if (front_ref_dirty) {
                    dirty_state.stencil_front_reference = false;
                    cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                               stencil_front_reference);
                }
                if (back_ref_dirty) {
                    dirty_state.stencil_back_reference = false;
                    cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack,
                                               stencil_back_reference);
                }
            }
        }
        const bool front_wm_dirty = dirty_state.stencil_front_write_mask;
        const bool back_wm_dirty = dirty_state.stencil_back_write_mask;
        if (front_wm_dirty || back_wm_dirty) {
            if (front_wm_dirty && back_wm_dirty &&
                stencil_front_write_mask == stencil_back_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                           stencil_front_write_mask);
            } else {
                if (front_wm_dirty) {
                    dirty_state.stencil_front_write_mask = false;
                    cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                               stencil_front_write_mask);
                }
                if (back_wm_dirty) {
                    dirty_state.stencil_back_write_mask = false;
                    cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack,
                                               stencil_back_write_mask);
                }
            }
        }
        const bool front_cm_dirty = dirty_state.stencil_front_compare_mask;
        const bool back_cm_dirty = dirty_state.stencil_back_compare_mask;
        if (front_cm_dirty || back_cm_dirty) {
            if (front_cm_dirty && back_cm_dirty &&
                stencil_front_compare_mask == stencil_back_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                             stencil_front_compare_mask);
            } else {
                if (front_cm_dirty) {
                    dirty_state.stencil_front_compare_mask = false;
                    cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                                 stencil_front_compare_mask);
                }
                if (back_cm_dirty) {
                    dirty_state.stencil_back_compare_mask = false;
                    cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                                 stencil_back_compare_mask);
                }
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
    }
}


} // namespace Vulkan
