// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/enum.h"
#include "common/incremental_id.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"

#include <atomic>
#include <deque>
#include <optional>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

namespace VideoCore {

enum ImageFlagBits : u32 {
    Empty = 0,
    MaybeCpuDirty = 1 << 0, ///< The page this image is in was touched before the image address
    CpuDirty = 1 << 1,      ///< Contents have been modified from the CPU
    GpuDirty = 1 << 2, ///< Contents have been modified from the GPU (valid data in buffer cache)
    Dirty = MaybeCpuDirty | CpuDirty | GpuDirty,
    GpuModified = 1 << 3, ///< Contents have been modified from the GPU
    Registered = 1 << 6,  ///< True when the image is registered
    Picked = 1 << 7,      ///< Temporary flag to mark the image as picked
};
DECLARE_ENUM_FLAG_OPERATORS(ImageFlagBits)

struct UniqueImage {
    explicit UniqueImage() = default;
    explicit UniqueImage(vk::Device device, VmaAllocator allocator)
        : device{device}, allocator{allocator} {}
    ~UniqueImage();

    UniqueImage(const UniqueImage&) = delete;
    UniqueImage& operator=(const UniqueImage&) = delete;

    UniqueImage(UniqueImage&& other)
        : allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          image{std::exchange(other.image, VK_NULL_HANDLE)}, image_ci{std::move(other.image_ci)} {}
    UniqueImage& operator=(UniqueImage&& other) {
        image = std::exchange(other.image, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        image_ci = std::move(other.image_ci);
        return *this;
    }

    void Create(const vk::ImageCreateInfo& image_ci);

    void Destroy();

    operator vk::Image() const {
        return image;
    }

    operator bool() const {
        return image;
    }

public:
    vk::Device device{};
    VmaAllocator allocator{};
    VmaAllocation allocation{};
    vk::Image image{};
    vk::ImageCreateInfo image_ci{};
};

class BlitHelper;

struct Image {
    Image(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler, BlitHelper& blit_helper,
          Common::SlotVector<ImageView>& slot_image_views, const ImageInfo& info);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    bool Overlaps(VAddr overlap_cpu_addr, size_t overlap_size) const noexcept {
        const VAddr overlap_end = overlap_cpu_addr + overlap_size;
        const auto image_addr = info.guest_address;
        const auto image_end = info.guest_address + info.guest_size;
        return image_addr < overlap_end && overlap_cpu_addr < image_end;
    }

    vk::Image GetImage() const {
        return backing->image.image;
    }

    bool IsTracked() {
        return track_addr != 0 && track_addr_end != 0;
    }

    bool SafeToDownload() const {
        return True(flags & ImageFlagBits::GpuModified) && False(flags & (ImageFlagBits::Dirty));
    }

    void AssociateDepth(ImageId depth_image_id, u64 depth_image_uid) {
        depth_id = depth_image_id;
        depth_uid = depth_image_uid;
    }

    void DisassociateDepth() {
        depth_id = {};
        depth_uid = {};
    }

    ImageView& FindView(const ImageViewInfo& view_info, bool ensure_guest_samples = true);
    vk::ImageView FindViewHandle(const ImageViewInfo& view_info, bool ensure_guest_samples = true);
    ImageViewId InsertView(const ImageViewInfo& view_info);

    // Inline capacity 2: nearly every call emits 0-1 barriers, and the old
    // 32-slot inline buffer made every GetBarriers reserve a 3.5KB frame.
    using Barriers = boost::container::small_vector<vk::ImageMemoryBarrier2, 2>;
    /// Records that the given query needed no barriers, valid until the
    /// backing's state epoch changes.
    void RecordNoopBarrier(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                           vk::PipelineStageFlags2 dst_stage,
                           std::optional<SubresourceRange> subres_range);

    /// Header fast path for GetBarriers' repeat no-op answer: the memo the
    /// outlined body maintains proves an identical query under an unchanged
    /// state epoch emits nothing. Probing it BEFORE the body's divergent
    /// collapse is sound because every path that makes the collapse
    /// applicable bumps state_epoch, which misses this memo.
    bool BarriersNoop(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                      vk::PipelineStageFlags2 dst_stage,
                      const std::optional<SubresourceRange>& subres_range) const {
        const u64 range_key =
            subres_range ? (u64{subres_range->base.level} | (u64{subres_range->base.layer} << 16) |
                            (u64{subres_range->extent.levels} << 32) |
                            (u64{subres_range->extent.layers} << 48))
                         : ~u64{0};
        return backing->noop_epoch == backing->state_epoch && backing->noop_layout == dst_layout &&
               backing->noop_access == dst_mask && backing->noop_stage == dst_stage &&
               backing->noop_range == range_key;
    }

    Barriers GetBarriers(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                         vk::PipelineStageFlags2 dst_stage,
                         std::optional<SubresourceRange> subres_range) {
        if (BarriersNoop(dst_layout, dst_mask, dst_stage, subres_range)) {
            return {};
        }
        return GetBarriersSlow(dst_layout, dst_mask, dst_stage, subres_range);
    }
    Barriers GetBarriersSlow(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                             vk::PipelineStageFlags2 dst_stage,
                             std::optional<SubresourceRange> subres_range);
    void Transit(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                 std::optional<SubresourceRange> range, vk::CommandBuffer cmdbuf = {}) {
        const vk::PipelineStageFlags2 dst_pl_stage =
            (dst_mask == vk::AccessFlagBits2::eTransferRead ||
             dst_mask == vk::AccessFlagBits2::eTransferWrite)
                ? vk::PipelineStageFlagBits2::eTransfer
                : vk::PipelineStageFlagBits2::eAllGraphics |
                      vk::PipelineStageFlagBits2::eComputeShader;
        if (BarriersNoop(dst_layout, dst_mask, dst_pl_stage, range)) {
            return;
        }
        TransitSlow(dst_layout, dst_mask, range, cmdbuf);
    }
    void TransitSlow(vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                     std::optional<SubresourceRange> range, vk::CommandBuffer cmdbuf = {});
    void Upload(std::span<const vk::BufferImageCopy> upload_copies, vk::Buffer buffer, u64 offset);
    void Download(std::span<const vk::BufferImageCopy> download_copies, vk::Buffer buffer,
                  u64 offset, u64 download_size);

    void CopyImage(Image& src_image);
    void CopyImageWithBuffer(Image& src_image, vk::Buffer buffer, u64 offset);
    void CopyMip(Image& src_image, u32 mip, u32 slice);

    void Resolve(Image& src_image, const VideoCore::SubresourceRange& mrt0_range,
                 const VideoCore::SubresourceRange& mrt1_range);
    void Clear(const vk::ClearValue& clear_value, const VideoCore::SubresourceRange& range);

    void SetBackingSamples(u32 num_samples, bool copy_backing = true);

public:
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    BlitHelper* blit_helper;
    Common::SlotVector<ImageView>* slot_image_views;
    ImageInfo info;
    vk::ImageAspectFlags aspect_mask = vk::ImageAspectFlagBits::eColor;
    vk::SampleCountFlags supported_samples = vk::SampleCountFlagBits::e1;
    ImageFlagBits flags = ImageFlagBits::Dirty;
    VAddr track_addr = 0;
    VAddr track_addr_end = 0;

    // Atomic fast state for the lock-free UpdateImage fast path: most calls
    // find the image clean, tracked and recently touched, yet pay the shared
    // lock. {dirty, tracked, last_touch_tick} pack into one u64. Wrapped so
    // the defaulted Image moves keep working (slot storage moves on growth).
    static constexpr u64 kFastStateDirty = 1ULL << 0;
    static constexpr u64 kFastStateTracked = 1ULL << 1;
    static constexpr u64 kFastStateTouchShift = 2;
    struct MovableAtomicU64 {
        std::atomic<u64> v;
        MovableAtomicU64(u64 init) : v(init) {}
        MovableAtomicU64(MovableAtomicU64&& o) noexcept : v(o.v.load(std::memory_order_relaxed)) {}
        MovableAtomicU64& operator=(MovableAtomicU64&& o) noexcept {
            v.store(o.v.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
    };
    MovableAtomicU64 fast_update_state{kFastStateDirty};

    void MarkFastStateDirty() noexcept {
        fast_update_state.v.fetch_or(kFastStateDirty, std::memory_order_release);
    }
    void UpdateFastState(u64 tick, bool is_tracked) noexcept {
        u64 state = tick << kFastStateTouchShift;
        if (is_tracked) {
            state |= kFastStateTracked;
        }
        fast_update_state.v.store(state, std::memory_order_release);
    }
    u64 ReadFastState() const noexcept {
        return fast_update_state.v.load(std::memory_order_acquire);
    }
    ImageId depth_id{};
    u64 depth_uid{};

    // Resource state tracking
    vk::ImageUsageFlags usage_flags;
    vk::FormatFeatureFlags2 format_features;
    struct State {
        vk::PipelineStageFlags2 pl_stage = vk::PipelineStageFlagBits2::eAllCommands;
        vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eNone;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };
    struct BackingImage {
        UniqueImage image;
        // The draw path's reads cluster below, aligned so the barrier-noop
        // memo probe, the state compare and the descriptor write's layout pay
        // for one line instead of three scattered ones.
        alignas(64) State state;

        // Negative-result memo for GetBarriers. Transitioning an image to a
        // state it already holds is by far the common case (the same textures
        // are re-bound every draw), but proving it costs a scan over every
        // mip x layer. state_epoch changes whenever any tracked state does, so
        // a memo recorded under the current epoch for the same query is exactly
        // reproducible without the scan.
        u64 state_epoch{1};
        u64 noop_epoch{}; // 0 = no memo
        vk::ImageLayout noop_layout{};
        vk::AccessFlags2 noop_access{};
        vk::PipelineStageFlags2 noop_stage{};
        u64 noop_range{};
        std::vector<State> subresource_states;
        // Number of subresource_states entries whose layout or access differ
        // from `state`, or that carry write access. While the vector is alive
        // `state` is frozen, so this is stable; when it is zero every entry
        // would no-op any matching-state scan and the vector is equivalent to
        // empty. Stages are deliberately excluded (they do not affect the
        // scan's skip condition); stage_union tracks them so a collapse can
        // widen the source stage mask conservatively instead.
        u32 subres_divergent{};
        vk::PipelineStageFlags2 subres_stage_union{};
        // The handle rides beside its key so a view hit ends at this record
        // instead of chasing the id through the slot vector into the
        // ImageView object. image_view_ids stays, pushed in lockstep with
        // view_records: FreeImage's deferred reclaim walks it to erase views,
        // and an entry present in one vector but not the other leaks the view
        // for the process lifetime.
        struct ViewRecord {
            ImageViewInfo info;
            vk::ImageView handle{};
        };
        boost::container::small_vector<ViewRecord, 4> view_records;
        boost::container::small_vector<ImageViewId, 4> image_view_ids;
        u32 num_samples;
    };
    std::deque<BackingImage> backing_images;
    BackingImage* backing{};
    // Mirror of backing->num_samples: FindView's per-bind sample check read
    // the LAST field of the 528-byte backing, a line nothing else on the draw
    // path touches. Updated wherever backing or its sample count changes.
    u32 backing_num_samples{};
    boost::container::static_vector<u64, 16> mip_hashes{};
    u64 image_uid{};
    u64 lru_id{};
    mutable u64 lru_touch_tick{~u64{0}};
    u64 tick_accessed_last{};
    u64 hash{};

    struct {
        u32 texture : 1;
        u32 storage : 1;
        u32 render_target : 1;
        u32 depth_target : 1;
        u32 vo_surface : 1;
    } usage{};

    struct {
        u32 is_bound : 1;
        u32 is_target : 1;
        u32 needs_rebind : 1;
        u32 force_general : 1;
    } binding{};

private:
    static Common::IncrementalIdProvider<u64> global_image_uid;
};

} // namespace VideoCore
