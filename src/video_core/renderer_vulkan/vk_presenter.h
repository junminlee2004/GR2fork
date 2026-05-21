// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <vector>

#include "core/libraries/videoout/buffer.h"
#include "imgui/imgui_texture.h"
#include "video_core/renderer_vulkan/host_passes/fsr_pass.h"
#include "video_core/renderer_vulkan/host_passes/pp_pass.h"
#include "video_core/renderer_vulkan/vk_compute_scheduler.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Frontend {
class WindowSDL;
}

namespace AmdGpu {
struct Liverpool;
}

namespace Vulkan {

struct Frame {
    u32 width;
    u32 height;
    VmaAllocation allocation;
    vk::Image image;
    vk::ImageView image_view;
    vk::Fence present_done;
    vk::Semaphore ready_semaphore;
    u64 ready_tick;
    bool is_hdr{false};
    u8 id{};

    ImTextureID imgui_texture;
};

enum SchedulerType {
    Draw,
    Present,
    CpuFlip,
};

class Rasterizer;

class Presenter {
public:
    Presenter(Frontend::WindowSDL& window, AmdGpu::Liverpool* liverpool);
    ~Presenter();

    HostPasses::PostProcessingPass::Settings& GetPPSettingsRef() {
        return pp_settings;
    }

    HostPasses::FsrPass::Settings& GetFsrSettingsRef() {
        return fsr_settings;
    }

    Frontend::WindowSDL& GetWindow() const {
        return window;
    }

    Rasterizer& GetRasterizer() const {
        return *rasterizer.get();
    }

    bool IsHDRSupported() const {
        return swapchain.HasHDR();
    }

    void SetHDR(bool enable) {
        if (!IsHDRSupported()) {
            return;
        }
        swapchain.SetHDR(enable);
        pp_settings.hdr = enable ? 1 : 0;
    }

    VideoCore::Image& RegisterVideoOutSurface(
        const Libraries::VideoOut::BufferAttributeGroup& attribute, VAddr cpu_address) {
        vo_buffers_addr.emplace_back(cpu_address);
        auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
        const auto image_id = texture_cache.FindImage(desc);
        auto& image = texture_cache.GetImage(image_id);
        image.usage.vo_surface = 1u;
        return image;
    }

    bool IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const;

    /// FIX(GR2FORK): drain the GPU to completion. Used by AvPlayerSource::Stop()
    /// to prevent the Vulkan command stream from referencing guest-memory
    /// frame buffers that are about to be freed. CUSA03694 (GR2 US) triggers
    /// a device-lost at vk_presenter.cpp:770 during newspaper comic cutscenes
    /// because Stop() lands while the GPU still has in-flight draws sampling
    /// the decoded NV12 frame.
    ///
    /// NOTE(GR2FORK queue-race): the Khronos validation layer flags this as
    /// UNASSIGNED-Threading-MultipleThreads-Write because vkDeviceWaitIdle
    /// is equivalent to vkQueueWaitIdle on every queue (Vulkan §3.6.3) and
    /// AvPlayerSource::Stop runs on a guest thread that doesn't hold
    /// Scheduler::submit_mutex. Two race-free alternatives were tried and
    /// both regressed performance:
    ///   v1: wrap waitIdle in Scheduler::submit_mutex — held the lock for
    ///       the full drain, stalling the renderer at every AvPlayer Stop.
    ///       GR2 cycles Stop rapidly during comic panels, producing
    ///       visible per-transition stutter.
    ///   v2: snapshot scheduler ticks under brief lock, then wait via
    ///       timeline semaphores (vkWaitSemaphores — device op, not queue
    ///       op). Eliminated per-transition stutter but degraded overall
    ///       steady-state performance, likely because Stop() fires often
    ///       enough that the per-call drain cost accumulates.
    /// On RADV/Mesa with the Z1 Extreme this race does not produce real
    /// driver corruption (no DEVICE_LOST observed across thousands of
    /// validation-on intents). Accept the per-spec race in exchange for
    /// the perf baseline.
    void WaitIdle() {
        (void)instance.GetDevice().waitIdle();
    }

    Frame* PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                        VAddr cpu_address);

    Frame* PrepareBlankFrame(bool present_thread);

    /// Drives the on-disk shader cache deserialization (PipelineCache::WarmUp)
    /// while painting a "LOADING SHADERS" overlay onto the swapchain at a
    /// throttled cadence. Called from gnmdriver::RegisterLib *after* this
    /// Presenter is fully constructed and *before* videoout::RegisterLib
    /// starts the PresentThread, so this method is the only thing driving
    /// the swapchain during warmup. Safe to call when the cache is empty
    /// or disabled — degenerates into an immediate WarmUp() with no frames
    /// pumped.
    void WarmUpPipelineCache();

    void Present(Frame* frame, bool is_reusing_frame = false);
    Frame* PrepareLastFrame();

    /// Captures the last presented frame to CPU memory as RGBA pixels.
    /// Returns true if successful. Output vector will contain width*height*4 bytes.
    bool CaptureScreenshot(std::vector<u8>& out_pixels, u32& out_width, u32& out_height);

private:
    Frame* GetRenderFrame();

    void RecreateFrame(Frame* frame, u32 width, u32 height);

    void SetExpectedGameSize(s32 width, s32 height);

    /// One-shot helper used by WarmUpPipelineCache. Acquires a blank frame
    /// and runs it through Present() so the LoadingScreenLayer overlay is
    /// painted onto the swapchain. Safe to call only on the thread that
    /// constructed the Presenter (gnmdriver registration thread); the
    /// PresentThread does not yet exist when this runs.
    void PumpLoadingFrame();

    /// Phase 1D-pre-E: scheduler-touching chunk of PrepareFrame, extracted
    /// so the producer-side wrapper can route it through
    /// `Rasterizer::PushPresenterRecord`. Runs on the assembler thread
    /// under future async; on the producer thread under sync.
    void DoPrepareFrameRecord(Frame* frame,
                              const Libraries::VideoOut::BufferAttributeGroup& attribute,
                              VideoCore::ImageId image_id,
                              vk::Extent2D image_size);

    /// Phase 1D-pre-E: scheduler-touching chunk of PrepareBlankFrame.
    /// `use_present_scheduler` selects between present_scheduler (called
    /// inline from the present-thread path) and draw_scheduler (called via
    /// PushPresenterRecord from the warmup path).
    void DoPrepareBlankFrameRecord(Frame* frame, bool use_present_scheduler);

private:
    float expected_ratio{1920.0 / 1080.0f};
    u32 expected_frame_width{1920};
    u32 expected_frame_height{1080};

    HostPasses::FsrPass fsr_pass;
    HostPasses::FsrPass::Settings fsr_settings{};
    HostPasses::PostProcessingPass::Settings pp_settings{};
    HostPasses::PostProcessingPass pp_pass;
    Frontend::WindowSDL& window;
    AmdGpu::Liverpool* liverpool;
    Instance instance;
    Scheduler draw_scheduler;
    Scheduler present_scheduler;
    Scheduler flip_scheduler;
    // Phase MQ-3: async-compute submission lane. Shares draw_scheduler's
    // MasterSemaphore (single timeline). Constructed but unused in MQ-3 —
    // first wire-up lands in MQ-4 with the first compute dispatch routing.
    ComputeScheduler compute_scheduler;
    Swapchain swapchain;
    std::unique_ptr<Rasterizer> rasterizer;
    VideoCore::TextureCache& texture_cache;
    vk::UniqueCommandPool command_pool;
    std::vector<Frame> present_frames;
    std::queue<Frame*> free_queue;
    Frame* last_submit_frame;
    std::mutex free_mutex;
    std::condition_variable free_cv;
    std::condition_variable_any frame_cv;
    std::optional<ImGui::RefCountedTexture> splash_img;
    std::vector<VAddr> vo_buffers_addr;
};

} // namespace Vulkan
