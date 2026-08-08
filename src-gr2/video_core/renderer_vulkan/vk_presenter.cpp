// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/config.h"
#include "common/debug.h"
#include "common/elf_info.h"
#include "common/frame_time_recorder.h"
#include "common/singleton.h"
#include "core/libraries/aspect_patches/aspect_patches.h"
#include "core/debug_state.h"
#include "core/devtools/layer.h"
#include "core/libraries/system/systemservice.h"
#include "imgui/renderer/imgui_core.h"
#include "imgui/renderer/imgui_impl_vulkan.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/vk_device_recovery.h"
#include "video_core/renderer_vulkan/vk_loading_screen.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/image.h"

#include <chrono>
#include <imgui.h>
#include <vk_mem_alloc.h>

namespace Vulkan {

bool CanBlitToSwapchain(const vk::PhysicalDevice physical_device, vk::Format format) {
    const vk::FormatProperties props{physical_device.getFormatProperties(format)};
    return static_cast<bool>(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst);
}

[[nodiscard]] vk::ImageSubresourceLayers MakeImageSubresourceLayers() {
    return vk::ImageSubresourceLayers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlit(s32 frame_width, s32 frame_height, s32 dst_width,
                                          s32 dst_height, s32 offset_x, s32 offset_y) {
    return vk::ImageBlit{
        .srcSubresource = MakeImageSubresourceLayers(),
        .srcOffsets =
            std::array{
                vk::Offset3D{
                    .x = 0,
                    .y = 0,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = frame_width,
                    .y = frame_height,
                    .z = 1,
                },
            },
        .dstSubresource = MakeImageSubresourceLayers(),
        .dstOffsets =
            std::array{
                vk::Offset3D{
                    .x = offset_x,
                    .y = offset_y,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = offset_x + dst_width,
                    .y = offset_y + dst_height,
                    .z = 1,
                },
            },
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlitStretch(s32 frame_width, s32 frame_height,
                                                 s32 swapchain_width, s32 swapchain_height) {
    return MakeImageBlit(frame_width, frame_height, swapchain_width, swapchain_height, 0, 0);
}

static vk::Rect2D FitImage(s32 frame_width, s32 frame_height, s32 swapchain_width,
                           s32 swapchain_height) {
    float frame_aspect = static_cast<float>(frame_width) / frame_height;
    float swapchain_aspect = static_cast<float>(swapchain_width) / swapchain_height;

    u32 dst_width = swapchain_width;
    u32 dst_height = swapchain_height;

    if (frame_aspect > swapchain_aspect) {
        dst_height = static_cast<s32>(swapchain_width / frame_aspect);
    } else {
        dst_width = static_cast<s32>(swapchain_height * frame_aspect);
    }

    const s32 offset_x = (swapchain_width - dst_width) / 2;
    const s32 offset_y = (swapchain_height - dst_height) / 2;

    return vk::Rect2D{{offset_x, offset_y}, {dst_width, dst_height}};
}

[[nodiscard]] vk::ImageBlit MakeImageBlitFit(s32 frame_width, s32 frame_height, s32 swapchain_width,
                                             s32 swapchain_height) {
    const auto& dst_rect = FitImage(frame_width, frame_height, swapchain_width, swapchain_height);

    return MakeImageBlit(frame_width, frame_height, dst_rect.extent.width, dst_rect.extent.height,
                         dst_rect.offset.x, dst_rect.offset.y);
}

Presenter::Presenter(Frontend::WindowSDL& window_, AmdGpu::Liverpool* liverpool_)
    : window{window_}, liverpool{liverpool_},
      instance{window, Config::getGpuId(), Config::vkValidationEnabled(),
               Config::getVkCrashDiagnosticEnabled()},
      draw_scheduler{instance}, present_scheduler{instance}, flip_scheduler{instance},
      // Shares draw_scheduler's MasterSemaphore as the unified timeline.
      // Must be constructed AFTER draw_scheduler so GetMasterSemaphore() is valid.
      compute_scheduler{instance, draw_scheduler.GetMasterSemaphore()},
      swapchain{instance, window},
      rasterizer{std::make_unique<Rasterizer>(instance, draw_scheduler, liverpool)},
      texture_cache{rasterizer->GetTextureCache()} {
    const u32 num_images = swapchain.GetImageCount();
    const vk::Device device = instance.GetDevice();

    // Create presentation frames.
    present_frames.resize(num_images);
    for (u32 i = 0; i < num_images; i++) {
        Frame& frame = present_frames[i];
        frame.id = i;
        auto fence = Check<"create present done fence">(
            device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        frame.present_done = fence;
        free_queue.push(&frame);
    }

    fsr_settings.enable = Config::getFsrEnabled();
    fsr_settings.use_rcas = Config::getRcasEnabled();
    fsr_settings.rcas_attenuation = static_cast<float>(Config::getRcasAttenuation() / 1000.f);

    fsr_pass.Create(device, instance.GetAllocator(), num_images);
    pp_pass.Create(device, swapchain.GetSurfaceFormat().format);

    ImGui::Layer::AddLayer(Common::Singleton<Core::Devtools::Layer>::Instance());
}

Presenter::~Presenter() {
    ImGui::Layer::RemoveLayer(Common::Singleton<Core::Devtools::Layer>::Instance());
    // Route draw_scheduler.Finish() through the BundleAssembler so under async the assembler
    // thread (single writer of draw_scheduler) executes it. Blocking Push + WaitFor: the device
    // must be idle before the destructor's Vulkan teardown (vmaDestroyImage etc.) proceeds.
    const u32 seq = rasterizer->PushPresenterRecord([this] {
        draw_scheduler.Finish();
    });
    rasterizer->WaitForAssembler(seq);
    const vk::Device device = instance.GetDevice();
    // GR2FORK FIX: the final flip/present submits can still be writing these frame images;
    // drain the whole device before destruction (write-into-freed on strict drivers otherwise).
    (void)device.waitIdle();
    for (auto& frame : present_frames) {
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        device.destroyImageView(frame.image_view);
        device.destroyFence(frame.present_done);
    }
    ImGui::Core::Shutdown(device);
}

bool Presenter::IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const {
    return std::ranges::find(vo_buffers_addr, color_buffer.Address()) != vo_buffers_addr.cend();
}

void Presenter::RecreateFrame(Frame* frame, u32 width, u32 height) {
    const vk::Device device = instance.GetDevice();
    if (frame->imgui_texture) {
        ImGui::Vulkan::RemoveTexture(frame->imgui_texture);
    }
    if (frame->image_view) {
        device.destroyImageView(frame->image_view);
    }
    if (frame->image) {
        vmaDestroyImage(instance.GetAllocator(), frame->image, frame->allocation);
    }

    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const vk::ImageCreateInfo image_info = {
        .flags = vk::ImageCreateFlagBits::eMutableFormat,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &frame->allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}",
                     vk::to_string(vk::Result{result}));
        UNREACHABLE();
    }
    frame->image = vk::Image{unsafe_image};
    SetObjectName(device, frame->image, "Frame image #{}", frame->id);

    const vk::ImageViewCreateInfo view_info = {
        .image = frame->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    auto view = Check<"create frame image view">(device.createImageView(view_info));
    frame->image_view = view;
    frame->width = width;
    frame->height = height;

    frame->imgui_texture = ImGui::Vulkan::AddTexture(view, vk::ImageLayout::eShaderReadOnlyOptimal);
    frame->is_hdr = swapchain.GetHDR();
}

VideoCore::Image& Presenter::RegisterVideoOutSurface(
    const Libraries::VideoOut::BufferAttributeGroup& attribute, VAddr cpu_address) {
    vo_buffers_addr.emplace_back(cpu_address);
    auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
    // Blocking hop: the caller is a guest syscall, so the round-trip latency is acceptable and
    // the cache mutation happens on its single writer (the assembler).
    VideoCore::ImageId image_id{};
    const u32 seq = rasterizer->PushPresenterRecord([this, &desc, &image_id] {
        image_id = texture_cache.FindImage(desc);
        texture_cache.GetImageUntouched(image_id).usage.vo_surface = 1u;
    });
    rasterizer->WaitForAssembler(seq);
    return texture_cache.GetImageUntouched(image_id);
}

Frame* Presenter::PrepareLastFrame() {
    std::scoped_lock last_lk{last_frame_mutex};
    if (last_submit_frame == nullptr) {
        return nullptr;
    }

    Frame* frame = last_submit_frame;

    while (true) {
        vk::Result result = instance.GetDevice().waitForFences(frame->present_done, false,
                                                               std::numeric_limits<u64>::max());
        if (result == vk::Result::eSuccess) {
            break;
        }
        if (result == vk::Result::eTimeout) {
            continue;
        }
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
    }

    auto& scheduler = flip_scheduler;
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.PrimaryCommandBuffer();

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier =
        vk::ImageMemoryBarrier2{.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
                                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                                .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                                .newLayout = vk::ImageLayout::eGeneral,
                                .image = frame->image,
                                .subresourceRange{frame_subresources}};

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    // Flush frame creation commands.
    frame->ready_semaphore = scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return frame;
}

bool Presenter::CaptureScreenshot(std::vector<u8>& out_pixels, u32& out_width, u32& out_height) {
    std::scoped_lock last_lk{last_frame_mutex};
    if (last_submit_frame == nullptr) {
        LOG_WARNING(Render_Vulkan, "CaptureScreenshot: no frame available");
        return false;
    }

    Frame* frame = last_submit_frame;
    out_width = frame->width;
    out_height = frame->height;
    const u64 pixel_count = static_cast<u64>(out_width) * out_height;
    const u64 buffer_size = pixel_count * 4;

    const vk::Device device = instance.GetDevice();

    // Wait for the frame to finish presenting
    {
        vk::Result result = device.waitForFences(frame->present_done, true,
                                                  std::numeric_limits<u64>::max());
        if (result != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "CaptureScreenshot: fence wait failed");
            return false;
        }
    }

    // Create staging buffer for GPU->CPU transfer
    VkBufferCreateInfo buf_ci{};
    buf_ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_ci.size = buffer_size;
    buf_ci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer staging_buffer{};
    VmaAllocation staging_alloc{};
    VmaAllocationInfo staging_info{};

    if (vmaCreateBuffer(instance.GetAllocator(), &buf_ci, &alloc_ci,
                        &staging_buffer, &staging_alloc, &staging_info) != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "CaptureScreenshot: failed to create staging buffer");
        return false;
    }

    // Record copy commands using flip_scheduler
    auto& sched = flip_scheduler;
    sched.EndRendering();
    const auto cmdbuf = sched.PrimaryCommandBuffer();

    const auto subresource_range = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    // Transition frame image to transfer src
    const auto to_transfer = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .image = frame->image,
        .subresourceRange = subresource_range,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_transfer,
    });

    // Copy image to buffer
    const vk::BufferImageCopy copy_region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {out_width, out_height, 1},
    };
    cmdbuf.copyImageToBuffer(frame->image, vk::ImageLayout::eTransferSrcOptimal,
                             vk::Buffer{staging_buffer}, 1, &copy_region);

    // Transition back to general
    const auto to_general = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = frame->image,
        .subresourceRange = subresource_range,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_general,
    });

    // Submit and wait for completion
    SubmitInfo submit_info{};
    sched.Flush(submit_info);
    sched.Finish();

    // Read pixels from mapped staging buffer
    out_pixels.resize(buffer_size);
    std::memcpy(out_pixels.data(), staging_info.pMappedData, buffer_size);

    // Cleanup
    vmaDestroyBuffer(instance.GetAllocator(), staging_buffer, staging_alloc);

    LOG_INFO(Render_Vulkan, "CaptureScreenshot: captured {}x{} frame ({} bytes)",
             out_width, out_height, buffer_size);
    return true;
}

static vk::Format GetFrameViewFormat(const Libraries::VideoOut::PixelFormat format) {
    switch (format) {
    case Libraries::VideoOut::PixelFormat::A8B8G8R8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A8R8G8B8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A2R10G10B10:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2R10G10B10UnormPack32;
    default:
        break;
    }
    UNREACHABLE_MSG("Unknown format={}", static_cast<u32>(format));
    return {};
}

Frame* Presenter::PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                               VAddr cpu_address) {
    // Presenter-thread prelude: touches the texture cache and free-frame queue but not the
    // scheduler, so it runs on the caller's thread. The aspect-ratio computation stays here too -
    // expected_ratio is read by OnResize on this thread; writing it in the closure would race.
    auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
    // GR2FORK FIX: FindImage/UpdateImage on a CPU-dirty video-out image record barriers and
    // uploads into draw_scheduler's OPEN command buffer; doing that from this (liverpool) thread
    // races the assembler recording draws into the same buffer - two-thread VkCommandBuffer
    // recording corrupts the stream (device loss during video playback, when the image is dirty
    // every frame). Route the prelude through the assembler and block until it ran.
    VideoCore::ImageId image_id{};
    vk::Extent2D image_size{};
    {
        // GR2FORK FIX: the extent is captured INSIDE the hop so no presenter-thread slot
        // dereference remains - the old GetImageUntouched here could resolve a slot freed and
        // reused between the hop and this read.
        const u32 prelude_seq = rasterizer->PushPresenterRecord([this, &desc, &image_id,
                                                                 &image_size] {
            image_id = texture_cache.FindImage(desc);
            texture_cache.UpdateImage(image_id);
            const auto& img = texture_cache.GetImageUntouched(image_id);
            image_size = vk::Extent2D{img.info.size.width, img.info.size.height};
        });
        rasterizer->WaitForAssembler(prelude_seq);
    }

    Frame* frame = GetRenderFrame();

    // [ASPECT OVERRIDE] A non-16:9 aspectRatioOverride forces expected_ratio to match, so the
    // presenter stops letterboxing the game's native 16:9 render; 16:9/Off falls through to the
    // original computation.
    {
        // Resolved (not just parsed): 21:9 may have latched to 2.39:1 at patch time
        // against the reconciled window - the presenter must agree with the eboot patch.
        const auto _aspect_target = Libraries::AspectPatches::ResolveTargetAspect(
            Libraries::AspectPatches::ParseAspectFromConfig(Config::getAspectRatioOverride()));
        if (_aspect_target == Libraries::AspectPatches::TargetAspect::Off) {
            expected_ratio =
                static_cast<float>(image_size.width) / static_cast<float>(image_size.height);
        } else {
            expected_ratio = Libraries::AspectPatches::TargetAspectToRatio(_aspect_target);
        }
    }

    DebugState.game_resolution = {image_size.width, image_size.height};
    DebugState.output_resolution = {frame->width, frame->height};

    // Scheduler chunk routed through the BundleAssembler; the closure captures frame*, attribute,
    // image_id and image_size by value and reaches the schedulers/passes via `this`. Blocking
    // Push + WaitFor: the caller reads frame->ready_semaphore / ready_tick after this returns.
    const u32 seq = rasterizer->PushPresenterRecord(
        [this, frame, attribute, cpu_address, image_id, image_size] {
            DoPrepareFrameRecord(frame, attribute, cpu_address, image_id, image_size);
        });
    rasterizer->WaitForAssembler(seq);
    return frame;
}

void Presenter::DoPrepareFrameRecord(
        Frame* frame,
        const Libraries::VideoOut::BufferAttributeGroup& attribute,
        VAddr cpu_address,
        VideoCore::ImageId image_id,
        vk::Extent2D image_size) {
    // GR2FORK FIX: revalidate the id resolved in the earlier hop - a deletion queued between the
    // two hops can free (and slot-reuse) the video-out image, and the blit below WRITES through
    // it. This runs on the assembler, sequenced with all deletion, so the re-resolve is race-free.
    {
        auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
        const bool live =
            texture_cache.IsImageSlotAllocated(image_id) &&
            True(texture_cache.GetImageUntouched(image_id).flags &
                 VideoCore::ImageFlagBits::Registered) &&
            texture_cache.GetImageUntouched(image_id).info.guest_address ==
                desc.info.guest_address;
        if (!live) [[unlikely]] {
            image_id = texture_cache.FindImage(desc);
            texture_cache.UpdateImage(image_id);
        }
    }
    // The chunk that touches draw_scheduler: under sync it runs on the producer (Presenter)
    // thread, under async on the dedicated assembler thread - the assembler stays sole writer.
    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange{frame_subresources},
    };

    draw_scheduler.EndRendering();
    const auto cmdbuf = draw_scheduler.PrimaryCommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    VideoCore::ImageViewInfo view_info{};
    view_info.format = GetFrameViewFormat(attribute.attrib.pixel_format);
    // Exclude alpha from output frame to avoid blending with UI.
    view_info.mapping.a = vk::ComponentSwizzle::eOne;

    // GR2FORK FIX: GetImageUntouched, not GetImage - GetImage's TouchImage does an unlocked LRU
    // Detach+Attach that races RunGarbageCollector's locked iteration of the same list and can
    // leave it cyclic, spinning the GC forever. The rendering thread touches this image anyway.
    auto& image = texture_cache.GetImageUntouched(image_id);
    auto image_view = *image.FindView(view_info).image_view;
    image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {});

    image_view = fsr_pass.Render(cmdbuf, image_view, image_size, {frame->width, frame->height},
                                 fsr_settings, frame->is_hdr);
    pp_pass.Render(cmdbuf, image_view, image_size, *frame, pp_settings);

    // Flush frame creation commands.
    frame->ready_semaphore = draw_scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = draw_scheduler.CurrentTick();
    SubmitInfo info{};
    draw_scheduler.Flush(info);
}

Frame* Presenter::PrepareBlankFrame(bool present_thread) {
    // Request a free presentation frame.
    Frame* frame = GetRenderFrame();

    // present_thread == true (Present's reuse path) touches present_scheduler, which the
    // assembler never writes - run inline. false (PumpLoadingFrame warmup) touches
    // draw_scheduler: route through the assembler, blocking so frame->ready_* is set on return.
    if (present_thread) {
        DoPrepareBlankFrameRecord(frame, /*use_present_scheduler=*/true);
    } else {
        const u32 seq = rasterizer->PushPresenterRecord([this, frame] {
            DoPrepareBlankFrameRecord(frame, /*use_present_scheduler=*/false);
        });
        rasterizer->WaitForAssembler(seq);
    }
    return frame;
}

void Presenter::DoPrepareBlankFrameRecord(Frame* frame, bool use_present_scheduler) {
    auto& scheduler = use_present_scheduler ? present_scheduler : draw_scheduler;
    scheduler.EndRendering();

    const auto cmdbuf = scheduler.PrimaryCommandBuffer();

    constexpr vk::ImageSubresourceRange simple_subresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const auto post_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const vk::RenderingAttachmentInfo attachment = {
        .imageView = frame->image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .extent = {frame->width, frame->height},
            },
        .layerCount = 1,
        .colorAttachmentCount = 1u,
        .pColorAttachments = &attachment,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    cmdbuf.beginRendering(rendering_info);
    cmdbuf.endRendering();

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &post_barrier,
    });

    // Flush frame creation commands.
    frame->ready_semaphore = scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
}

void Presenter::WarmUpPipelineCache() {
    // Drive WarmUp through here even when the cache is disabled or empty - WarmUp short-circuits
    // cheaply, and the uniform path keeps the gnmdriver::RegisterLib call site branch-free.
    auto& loading = GetLoadingScreenLayer();
    loading.SetVisible(true);
    loading.SetProgress(0, 0);
    // Push the swapchain's actual extent: ImGui's io.DisplaySize is the SDL window's logical size
    // (often 720p) while the swapchain matches the game's output, so the overlay would otherwise
    // paint only the top-left rectangle. Re-pushed each pump to cover mid-warmup recreation.
    {
        const auto extent = swapchain.GetExtent();
        loading.SetSwapchainExtent(extent.width, extent.height);
    }
    ImGui::Layer::AddLayer(&loading);

    // Pump one frame immediately so the overlay shows before the first blob is read; a fast
    // ForEachBlob could otherwise finish before any tick fires, leaving the screen black.
    PumpLoadingFrame();

    // Throttle swapchain redraws while shaders deserialize - a Present() costs an order of
    // magnitude more than decoding one blob. ~60 ms per pump is ~16 fps, enough for a live
    // screen; the counter still advances per blob via SetProgress (two atomic stores).
    using clock = std::chrono::steady_clock;
    auto last_present = clock::now();
    constexpr auto kFrameInterval = std::chrono::milliseconds(60);

    rasterizer->GetPipelineCache().WarmUp([&](u32 loaded, u32 total) {
        loading.SetProgress(loaded, total);
        const auto now = clock::now();
        if (now - last_present >= kFrameInterval) {
            last_present = now;
            PumpLoadingFrame();
        }
    });

    // One last frame in the READY state so the hand-off to the game's first real frame is not a
    // hard cut; the overlay is removed right after and videoout's PresentThread takes over.
    loading.MarkFinished();
    PumpLoadingFrame();

    loading.SetVisible(false);
    ImGui::Layer::RemoveLayer(&loading);
    // The remove queues into ImGui::Core's change_layers; it gets applied
    // on the next NewFrame, which is fine - once visible_ flips to false
    // the layer's Draw is a no-op anyway.
}

void Presenter::PumpLoadingFrame() {
    // Manual single-frame pump on the gnmdriver registration thread, the only thread alive
    // during warmup (videoout's PresentThread does not exist yet). PrepareBlankFrame builds a
    // black frame, Present runs the normal ImGui+swapchain path, LoadingScreenLayer paints on top.

    // Refresh the layer's swapchain extent every pump in case the swapchain was recreated since
    // AddLayer; PrepareBlankFrame does not recreate, but Present can.
    {
        auto& loading = GetLoadingScreenLayer();
        const auto extent = swapchain.GetExtent();
        loading.SetSwapchainExtent(extent.width, extent.height);
    }

    Frame* frame = PrepareBlankFrame(false);
    if (frame != nullptr) {
        Present(frame);
    }
}

void Presenter::Present(Frame* frame, bool is_reusing_frame) {
    // Free the frame for reuse
    const auto free_frame = [&] {
        if (!is_reusing_frame) {
            last_submit_frame = frame;
            std::scoped_lock fl{free_mutex};
            free_queue.push(frame);
            free_cv.notify_one();
        }
    };

    // Recreate the swapchain if the window was resized.
    if (window.GetWidth() != swapchain.GetWidth() || window.GetHeight() != swapchain.GetHeight()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }

    if (!swapchain.AcquireNextImage()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
        if (!swapchain.AcquireNextImage()) {
            // User resizes the window too fast and GPU can't keep up. Skip this frame.
            LOG_WARNING(Render_Vulkan, "Skipping frame!");
            free_frame();
            return;
        }
    }

    // Reset fence for queue submission. Do it here instead of GetRenderFrame() because we may
    // skip frame because of slow swapchain recreation. If a frame skip occurs, we skip signal
    // the frame's present fence and future GetRenderFrame() call will hang waiting for this frame.
    const auto reset_result = instance.GetDevice().resetFences(frame->present_done);
    ASSERT_MSG(reset_result == vk::Result::eSuccess,
               "Unexpected error resetting present done fence: {}", vk::to_string(reset_result));

    ImGuiID dockId = ImGui::Core::NewFrame(is_reusing_frame);

    const vk::Image swapchain_image = swapchain.Image();
    const vk::ImageView swapchain_image_view = swapchain.ImageView();

    auto& scheduler = present_scheduler;
    const auto cmdbuf = scheduler.PrimaryCommandBuffer();

    if (Config::getVkHostMarkersEnabled()) {
        cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
            .pLabelName = "Present",
        });
    }

    {
        auto* profiler_ctx = instance.GetProfilerContext();
        TracyVkNamedZoneC(profiler_ctx, renderer_gpu_zone, cmdbuf, "Host frame",
                          MarkersPalette::GpuMarkerColor, profiler_ctx != nullptr);

        const vk::Extent2D extent = swapchain.GetExtent();
        const std::array pre_barriers{
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = frame->image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
        };

        const vk::ImageMemoryBarrier post_barrier{
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eMemoryRead,
            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);

        { // Draw the game
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0f});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
            ImGui::SetNextWindowDockID(dockId, ImGuiCond_Once);
            if (ImGui::Begin("Display##game_display", nullptr, ImGuiWindowFlags_NoNav)) {
                auto game_texture = frame->imgui_texture;
                auto game_width = frame->width;
                auto game_height = frame->height;

                if (Libraries::SystemService::IsSplashVisible()) { // draw splash
                    if (!splash_img.has_value()) {
                        splash_img.emplace();
                        auto splash_path = Common::ElfInfo::Instance().GetSplashPath();
                        if (!splash_path.empty()) {
                            splash_img = ImGui::RefCountedTexture::DecodePngFile(splash_path);
                        }
                    }
                    if (auto& splash_image = this->splash_img.value()) {
                        auto [im_id, width, height] = splash_image.GetTexture();
                        game_texture = im_id;
                        game_width = width;
                        game_height = height;
                    }
                }

                ImVec2 contentArea = ImGui::GetContentRegionAvail();
                SetExpectedGameSize((s32)contentArea.x, (s32)contentArea.y);

                const auto imgRect =
                    FitImage(game_width, game_height, (s32)contentArea.x, (s32)contentArea.y);
                ImVec2 offset{
                    static_cast<float>(imgRect.offset.x),
                    static_cast<float>(imgRect.offset.y),
                };
                ImVec2 size{
                    static_cast<float>(imgRect.extent.width),
                    static_cast<float>(imgRect.extent.height),
                };

                ImGui::SetCursorPos(ImGui::GetCursorStartPos() + offset);
                ImGui::Image(game_texture, size);

                if (Config::nullGpu()) {
                    Core::Devtools::Layer::DrawNullGpuNotice();
                }
            }
            ImGui::End();
            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor();
        }
        ImGui::Core::Render(cmdbuf, swapchain_image_view, swapchain.GetExtent());

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);

        if (profiler_ctx) {
            TracyVkCollect(profiler_ctx, cmdbuf);
        }
    }

    if (Config::getVkHostMarkersEnabled()) {
        cmdbuf.endDebugUtilsLabelEXT();
    }

    // Flush vulkan commands.
    SubmitInfo info{};
    // T5.B3: acquire only gates swapchain-image WRITES (color-attachment
    // output); frame-ready must also gate ImGui's fragment-stage SAMPLING of
    // the frame image - the old positional masks had these inverted.
    info.AddWait(swapchain.GetImageAcquiredSemaphore(), 1,
                 vk::PipelineStageFlagBits::eColorAttachmentOutput);
    info.AddWait(frame->ready_semaphore, frame->ready_tick,
                 vk::PipelineStageFlagBits::eFragmentShader |
                     vk::PipelineStageFlagBits::eColorAttachmentOutput);
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    info.AddSignal(frame->present_done);
    scheduler.Flush(info);

    // Present to swapchain.
    std::scoped_lock submit_lock{Scheduler::submit_mutex};
    if (!swapchain.Present()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }

    // GR2FORK PERF: optional per-flip wall-clock recorder for FPS measurement; env contract
    // (GR2_FPS_LOG_EVERY / GR2_FPS_OUT / GR2_FPS_DISABLE) in common/frame_time_recorder.h. Hooked
    // after swapchain.Present() so AcquireNextImage-skipped frames stay out of the percentiles.
    // Common::FrameTime::RecordFlip();

    free_frame();
    if (!is_reusing_frame) {
        DebugState.IncFlipFrameNum();
    }
}

Frame* Presenter::GetRenderFrame() {
    // Wait for free presentation frames
    Frame* frame;
    {
        std::unique_lock lock{free_mutex};
        free_cv.wait(lock, [this] { return !free_queue.empty(); });
        LOG_DEBUG(Render_Vulkan, "Got render frame, remaining {}", free_queue.size() - 1);

        // Take the frame from the queue
        frame = free_queue.front();
        free_queue.pop();
    }

    const vk::Device device = instance.GetDevice();
    vk::Result result{};

    const auto wait = [&]() {
        result = device.waitForFences(frame->present_done, false, std::numeric_limits<u64>::max());
        return result;
    };

    // Wait for the presentation to be finished so all frame resources are free
    while (wait() != vk::Result::eSuccess) {
        // GR2FORK: recovery opt-in - flag a lost device and stop waiting instead of aborting; the
        // main thread rebuilds. Also a device-lost detection point. Default keeps the fatal assert.
        if (result == vk::Result::eErrorDeviceLost && DeviceRecoveryEnabled()) {
            g_device_lost.store(true, std::memory_order_release);
            break;
        }
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
        // Retry if the waiting times out
        if (result == vk::Result::eTimeout) {
            continue;
        }
    }

    if (frame->width != expected_frame_width || frame->height != expected_frame_height ||
        frame->is_hdr != swapchain.GetHDR()) {
        // GR2FORK FIX: the present fence can pass instantly for a frame requeued by the
        // frame-skip path while this cycle's FSR/PP writes are still executing; destroying the
        // image under them is a device-losing write. Drain the frame's timeline first - zero
        // cost on the non-recreate steady path.
        if (frame->ready_semaphore) {
            const vk::SemaphoreWaitInfo wait_info{
                .semaphoreCount = 1,
                .pSemaphores = &frame->ready_semaphore,
                .pValues = &frame->ready_tick,
            };
            (void)device.waitSemaphores(wait_info, std::numeric_limits<u64>::max());
        }
        RecreateFrame(frame, expected_frame_width, expected_frame_height);
    }

    return frame;
}

void Presenter::SetExpectedGameSize(s32 width, s32 height) {
    const float ratio = (float)width / (float)height;

    expected_frame_height = height;
    expected_frame_width = width;
    if (ratio > expected_ratio) {
        expected_frame_width = static_cast<s32>(height * expected_ratio);
    } else {
        expected_frame_height = static_cast<s32>(width / expected_ratio);
    }
}

} // namespace Vulkan
