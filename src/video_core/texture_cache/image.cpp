// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <ranges>
#include "common/assert.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/skipcache/skipcache.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

using namespace Vulkan;

Common::IncrementalIdProvider<u64> Image::global_image_uid{};

// Access bits that make a subresource's current state a write that must be
// ordered even when layout and access already match.
constexpr vk::AccessFlags2 kWriteFlags = vk::AccessFlagBits2::eTransferWrite |
                                         vk::AccessFlagBits2::eShaderWrite |
                                         vk::AccessFlagBits2::eMemoryWrite;

static vk::ImageUsageFlags ImageUsageFlags(const Vulkan::Instance& instance,
                                           const ImageInfo& info) {
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc |
                                vk::ImageUsageFlagBits::eTransferDst |
                                vk::ImageUsageFlagBits::eSampled;
    if (!info.props.is_block) {
        if (info.props.is_depth) {
            usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
        } else {
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
            if (instance.IsAttachmentFeedbackLoopLayoutSupported()) {
                usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
            }
            // Always create images with storage flag to avoid needing re-creation in case of e.g
            // compute clears This sacrifices a bit of performance but is less work. ExtendedUsage
            // flag is also used.
            usage |= vk::ImageUsageFlagBits::eStorage;
        }
    } else {
        // Similarly to above, we specify storage usage. This is typically not supported by
        // compressed formats, but may be used for uncompressed views. In order to satisfy this,
        // we will also specify the extended usage bit.
        usage |= vk::ImageUsageFlagBits::eStorage;
    }

    return usage;
}

static vk::ImageType ConvertImageType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
        return vk::ImageType::e1D;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DArray:
        return vk::ImageType::e2D;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE();
    }
}

static vk::FormatFeatureFlags2 FormatFeatureFlags(const vk::ImageUsageFlags usage_flags) {
    vk::FormatFeatureFlags2 feature_flags{};
    if (usage_flags & vk::ImageUsageFlagBits::eTransferSrc) {
        feature_flags |= vk::FormatFeatureFlagBits2::eTransferSrc;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eTransferDst) {
        feature_flags |= vk::FormatFeatureFlagBits2::eTransferDst;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eSampled) {
        feature_flags |= vk::FormatFeatureFlagBits2::eSampledImage;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eColorAttachment) {
        feature_flags |= vk::FormatFeatureFlagBits2::eColorAttachment;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
        feature_flags |= vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
    }
    // Note: StorageImage is intentionally ignored for now since it is always set, and can mess up
    // compatibility checks.
    return feature_flags;
}

UniqueImage::~UniqueImage() {
    if (image) {
        vmaDestroyImage(allocator, image, allocation);
    }
}

void UniqueImage::Destroy() {
    if (image) {
        vmaDestroyImage(allocator, image, allocation);
        image = vk::Image{};
        allocation = {};
    }
}

void UniqueImage::Create(const vk::ImageCreateInfo& image_ci) {
    this->image_ci = image_ci;
    ASSERT(!image);
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    const VkImageCreateInfo image_ci_unsafe = static_cast<VkImageCreateInfo>(image_ci);
    VkImage unsafe_image{};
    VmaAllocationInfo alloc_info{};
    VkResult result = vmaCreateImage(allocator, &image_ci_unsafe, &alloc_ci, &unsafe_image,
                                     &allocation, &alloc_info);
    ASSERT_MSG(result == VK_SUCCESS, "Failed allocating image with error {}",
               vk::to_string(vk::Result{result}));
    image = vk::Image{unsafe_image};
    size_bytes = alloc_info.size;
}

Image::Image(const Vulkan::Instance& instance, Vulkan::Runtime& runtime_,
             Common::SlotVector<ImageView>& slot_image_views_, const ImageInfo& info_)
    : runtime{&runtime_}, slot_image_views{&slot_image_views_}, info{info_} {
    if (info.pixel_format == vk::Format::eUndefined) {
        return;
    }

    image_uid = global_image_uid.Next();
    mip_hashes.resize(info.resources.levels);
    vk::ImageCreateFlags flags{vk::ImageCreateFlagBits::eMutableFormat |
                               vk::ImageCreateFlagBits::eExtendedUsage};
    if (info.props.is_volume) {
        flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
        if (instance.Is2dViewOf3dSupported()) {
            flags |= vk::ImageCreateFlagBits::e2DViewCompatibleEXT;
        }
    }
    if (info.props.is_block && instance.IsBlockTexelViewSupported()) {
        flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
    }

    usage_flags = ImageUsageFlags(instance, info);
    format_features = FormatFeatureFlags(usage_flags);
    if (info.props.is_depth) {
        aspect_mask = vk::ImageAspectFlagBits::eDepth;
        if (info.props.has_stencil) {
            aspect_mask |= vk::ImageAspectFlagBits::eStencil;
        }
    }

    constexpr auto tiling = vk::ImageTiling::eOptimal;
    const auto supported_format = instance.GetSupportedFormat(info.pixel_format, format_features);
    const vk::PhysicalDeviceImageFormatInfo2 format_info{
        .format = supported_format,
        .type = ConvertImageType(info.type),
        .tiling = tiling,
        .usage = usage_flags,
        .flags = flags,
    };
    const auto image_format_properties =
        instance.GetPhysicalDevice().getImageFormatProperties2(format_info);
    if (image_format_properties.result == vk::Result::eErrorFormatNotSupported) {
        LOG_ERROR(Render_Vulkan, "image format {} type {} is not supported (flags {}, usage {})",
                  vk::to_string(supported_format), vk::to_string(format_info.type),
                  vk::to_string(format_info.flags), vk::to_string(format_info.usage));
    }
    supported_samples = image_format_properties.result == vk::Result::eSuccess
                            ? image_format_properties.value.imageFormatProperties.sampleCounts
                            : vk::SampleCountFlagBits::e1;

    const vk::ImageCreateInfo image_ci = {
        .flags = flags,
        .imageType = ConvertImageType(info.type),
        .format = supported_format,
        .extent{
            .width = info.size.width,
            .height = info.size.height,
            .depth = info.size.depth,
        },
        .mipLevels = static_cast<u32>(info.resources.levels),
        .arrayLayers = static_cast<u32>(info.resources.layers),
        .samples = LiverpoolToVK::NumSamples(info.num_samples, supported_samples),
        .tiling = tiling,
        .usage = usage_flags,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    backing = &backing_images.emplace_back();
    backing->num_samples = info.num_samples;
    backing_num_samples = info.num_samples;
    backing_epoch = backing->state_epoch;
    backing->image = UniqueImage{instance.GetDevice(), instance.GetAllocator()};
    backing->image.Create(image_ci);

    Vulkan::SetObjectName(instance.GetDevice(), GetImage(),
                          "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{}", info.size.width,
                          info.size.height, info.size.depth, AmdGpu::NameOf(info.tile_mode),
                          vk::to_string(info.pixel_format), info.guest_address, info.guest_size,
                          info.resources.layers, info.resources.levels, info.num_samples);
}

Image::~Image() = default;

ImageView& Image::FindView(const ImageViewInfo& view_info, bool ensure_guest_samples) {
    if (ensure_guest_samples && backing_num_samples > 1 != info.num_samples > 1) {
        runtime->SetBackingSamples(this, info.num_samples);
    }
    const auto& records = backing->view_records;
    for (size_t i = 0; i < records.size(); ++i) {
        if (records[i].info == view_info) {
            return (*slot_image_views)[backing->image_view_ids[i]];
        }
    }
    return (*slot_image_views)[InsertView(view_info)];
}

vk::ImageView Image::FindViewHandle(const ImageViewInfo& view_info, bool ensure_guest_samples) {
    if (ensure_guest_samples && backing_num_samples > 1 != info.num_samples > 1) {
        runtime->SetBackingSamples(this, info.num_samples);
    }
    const auto& records = backing->view_records;
    for (size_t i = 0; i < records.size(); ++i) {
        if (records[i].info == view_info) {
            return records[i].handle;
        }
    }
    InsertView(view_info);
    return backing->view_records.back().handle;
}

SHAD_NO_INLINE ImageViewId Image::InsertView(const ImageViewInfo& view_info) {
    const auto view_id = slot_image_views->insert(runtime->GetInstance(), view_info, *this);
    // Handle read through the slot AFTER the insert: a reserve inside it can
    // move every ImageView. The two pushes stay adjacent (lockstep contract).
    const vk::ImageView handle = *(*slot_image_views)[view_id].image_view;
    backing->view_records.push_back(BackingImage::ViewRecord{view_info, handle});
    backing->image_view_ids.emplace_back(view_id);
    return view_id;
}

void Image::RecordNoopBarrier(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                              vk::PipelineStageFlags2 dst_stage,
                              const std::optional<SubresourceRange>& subres_range) {
    backing->noop_epoch = backing->state_epoch;
    backing->noop_layout = dst_layout;
    backing->noop_access = dst_mask;
    backing->noop_stage = dst_stage;
    backing->noop_range = RangeKey(subres_range);
}

void Image::GetBarriersSlow(Barriers& barriers, vk::ImageLayout dst_layout,
                            vk::AccessFlags2 dst_mask, vk::PipelineStageFlags2 dst_stage,
                            std::optional<SubresourceRange> subres_range) {
    // The runtime accumulates barriers across images, so this call's own
    // output is what it appends past the entry size.
    const size_t num_barriers_in = barriers.size();
    auto& last_state = backing->state;
    auto& subresource_states = backing->subresource_states;

    const bool needs_partial_transition =
        subres_range &&
        (subres_range->base != SubresourceBase{} || subres_range->extent != info.resources);
    bool partially_transited = !subresource_states.empty();

    // A vector with zero divergent entries is equivalent to empty: collapse it
    // instead of scanning every mip x layer on each call. Entries may still
    // differ in pipeline stage; the union keeps the next full-resource
    // barrier's source stage a superset of every entry's.
    if (partially_transited && backing->subres_divergent == 0) {
        last_state.pl_stage |= backing->subres_stage_union;
        subresource_states.clear();
        backing->subres_stage_union = {};
        BumpStateEpoch();
        partially_transited = false;
    }
    // A partial transition into the state the whole image is already in would
    // materialize the vector only to fill it with identical values.
    if (needs_partial_transition && !partially_transited && last_state.layout == dst_layout &&
        last_state.access_mask == dst_mask && !(last_state.access_mask & kWriteFlags)) {
        RecordNoopBarrier(dst_layout, dst_mask, dst_stage, subres_range);
        return;
    }
    // The inline GetBarriers has already probed BarriersNoop; the collapse
    // above only bumps state_epoch.
    const bool had_subres = !subresource_states.empty();

    if (needs_partial_transition || partially_transited) {
        if (!partially_transited) {
            subresource_states.resize(info.resources.levels * info.resources.layers);
            std::fill(subresource_states.begin(), subresource_states.end(), last_state);
            backing->subres_divergent = 0;
            backing->subres_stage_union = last_state.pl_stage;
        }
        const State base_state = last_state;
        const bool dst_divergent = dst_layout != base_state.layout ||
                                   dst_mask != base_state.access_mask ||
                                   static_cast<bool>(dst_mask & kWriteFlags);

        // In case of partial transition, we need to change the specified subresources only.
        // Otherwise all subresources need to be set to the same state so we can use a full
        // resource transition for the next time.
        const auto mips =
            needs_partial_transition
                ? std::ranges::views::iota(subres_range->base.level,
                                           subres_range->base.level + subres_range->extent.levels)
                : std::views::iota(0u, info.resources.levels);
        const auto layers =
            needs_partial_transition
                ? std::ranges::views::iota(subres_range->base.layer,
                                           subres_range->base.layer + subres_range->extent.layers)
                : std::views::iota(0u, info.resources.layers);

        for (u32 mip : mips) {
            for (u32 layer : layers) {
                // NOTE: these loops may produce a lot of small barriers.
                // If this becomes a problem, we can optimize it by merging adjacent barriers.
                const auto subres_idx = mip * info.resources.layers + layer;
                ASSERT(subres_idx < subresource_states.size());
                auto& state = subresource_states[subres_idx];

                const bool is_write = static_cast<bool>(state.access_mask & kWriteFlags);
                if (state.layout != dst_layout || state.access_mask != dst_mask || is_write) {
                    barriers.emplace_back(vk::ImageMemoryBarrier2{
                        .srcStageMask = state.pl_stage,
                        .srcAccessMask = state.access_mask,
                        .dstStageMask = dst_stage,
                        .dstAccessMask = dst_mask,
                        .oldLayout = state.layout,
                        .newLayout = dst_layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = GetImage(),
                        .subresourceRange{
                            .aspectMask = aspect_mask,
                            .baseMipLevel = mip,
                            .levelCount = 1,
                            .baseArrayLayer = layer,
                            .layerCount = 1,
                        },
                    });
                    const bool was_divergent = state.layout != base_state.layout ||
                                               state.access_mask != base_state.access_mask ||
                                               static_cast<bool>(state.access_mask & kWriteFlags);
                    backing->subres_divergent +=
                        static_cast<u32>(dst_divergent) - static_cast<u32>(was_divergent);
                    state.layout = dst_layout;
                    state.access_mask = dst_mask;
                    state.pl_stage = dst_stage;
                }
            }
        }

        if (barriers.size() != num_barriers_in) {
            backing->subres_stage_union |= dst_stage;
            BumpStateEpoch();
            Skipcache::Framework::Instance().BumpLayoutGen();
        }

        if (!needs_partial_transition) {
            // The loop unified every entry; the tail writes last_state to that state.
            subresource_states.clear();
            backing->subres_divergent = 0;
            backing->subres_stage_union = {};
            BumpStateEpoch();
        }
    } else { // Full resource transition
        const bool is_write = static_cast<bool>(last_state.access_mask & kWriteFlags);
        if (last_state.layout == dst_layout && last_state.access_mask == dst_mask && !is_write) {
            RecordNoopBarrier(dst_layout, dst_mask, dst_stage, subres_range);
            return;
        }

        barriers.emplace_back(vk::ImageMemoryBarrier2{
            .srcStageMask = last_state.pl_stage,
            .srcAccessMask = last_state.access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_mask,
            .oldLayout = last_state.layout,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = GetImage(),
            .subresourceRange{
                .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        });
    }

    if (barriers.size() == num_barriers_in && had_subres == !subresource_states.empty()) {
        // No work was needed and the structure is as it was: the answer is
        // reproducible until the state epoch moves.
        RecordNoopBarrier(dst_layout, dst_mask, dst_stage, subres_range);
        return;
    }
    if (last_state.layout != dst_layout || last_state.access_mask != dst_mask ||
        last_state.pl_stage != dst_stage) {
        BumpStateEpoch();
    }
    last_state.layout = dst_layout;
    last_state.access_mask = dst_mask;
    Skipcache::Framework::Instance().BumpLayoutGen();
    last_state.pl_stage = dst_stage;
}

} // namespace VideoCore
