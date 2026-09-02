// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <boost/container/small_vector.hpp>
#include <queue>
#include <tsl/robin_map.h>

#include "common/assert.h"
#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "shader_recompiler/resource.h"
#include "video_core/multi_level_page_table.h"
#include "video_core/skipcache/skipcache.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/sampler.h"
#include "video_core/texture_cache/tile_manager.h"

namespace AmdGpu {
struct Liverpool;
}

namespace VideoCore {

class BufferCache;
class PageManager;

class TextureCache {
    // Default values for garbage collection
    static constexpr s64 DEFAULT_PRESSURE_GC_MEMORY = 1_GB + 512_MB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 3_GB;
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;

    using ImageIds = boost::container::small_vector<ImageId, 16>;

    // Page-table bucket entry. The guest range is copied in at registration
    // (it is immutable while registered), so overlap filtering reads the
    // bucket's contiguous memory instead of chasing each candidate's cold
    // 752-byte Image struct - the dominant cost of FindImage was that
    // dependent load missing cache once per candidate.
    struct PageImageRef {
        ImageId id;
        u32 size; // RegisterImage asserts guest_size fits
        VAddr addr;
    };
    static_assert(sizeof(PageImageRef) == 16);
    using PageRefs = boost::container::small_vector<PageImageRef, 4>;

    struct Traits {
        using Entry = PageRefs;
        static constexpr size_t AddressSpaceBits = 40;
        static constexpr size_t FirstLevelBits = 10;
        static constexpr size_t PageBits = 20;
    };
    using PageTable = MultiLevelPageTable<Traits>;

public:
    enum class BindingType : u32 {
        Texture,
        Storage,
        RenderTarget,
        DepthTarget,
        VideoOut,
    };

    struct ImageDesc {
        // Lazy for shader-resource bindings: a FINDIMG memo hit reads only
        // type and view_info afterwards, so the 376-byte build (NSDMI
        // prologue, TLS memo probe, copy-out) is deferred until a route that
        // actually reaches FindImage materializes it via Info(). Target
        // ctors (CB/DB/VideoOut) stay eager: FindRenderTarget and
        // FindDepthTarget read the engaged value through the const accessor.
        std::optional<ImageInfo> info;
        ImageViewInfo view_info;
        BindingType type{BindingType::Texture};
        AmdGpu::Image deferred_tsharp{};
        bool deferred_is_depth{};

        // Deferred view: a FINDIMG memo hit copies the post-rebase view info
        // out of the entry, so the 56-byte build (SurfaceFormat, swizzle, view
        // type) runs only on the routes that reach FindImage. Bindings that
        // mutate the view before the probe (mip fallback) build it eagerly.
        bool view_ready{true};
        bool deferred_is_array{};

        // View memo: the handle and backing a consumed FINDIMG hit carried,
        // and the memo slot FindTexture writes its resolved handle back to.
        static constexpr u16 NoMemoSlot = 0xFFFF;
        vk::ImageView memo_view{};
        const Image::BackingImage* memo_backing{};
        u16 memo_slot{NoMemoSlot};

        ImageDesc() = default;
        ImageDesc(const AmdGpu::Image& image, const Shader::ImageResource& desc,
                  bool defer_view = false)
            : type{desc.is_written ? BindingType::Storage : BindingType::Texture},
              deferred_tsharp{image}, deferred_is_depth{desc.is_depth}, view_ready{!defer_view},
              deferred_is_array{desc.is_array} {
            if (!defer_view) {
                view_info = ImageViewInfo{image, desc};
            }
        }
        ImageDesc(const AmdGpu::ColorBuffer& buffer, AmdGpu::CbDbExtent hint)
            : info{std::in_place, buffer, hint}, view_info{buffer},
              type{BindingType::RenderTarget} {}
        ImageDesc(const AmdGpu::DepthBuffer& buffer, AmdGpu::DepthView view,
                  AmdGpu::DepthControl ctl, VAddr htile_address, AmdGpu::CbDbExtent hint,
                  bool write_buffer = false)
            : info{std::in_place, buffer, view.NumSlices(), htile_address, hint, write_buffer},
              view_info{buffer, view, ctl}, type{BindingType::DepthTarget} {}
        ImageDesc(const Libraries::VideoOut::BufferAttributeGroup& group, VAddr cpu_address)
            : info{std::in_place, group, cpu_address}, type{BindingType::VideoOut} {}

        // Emplace, never assign: assignment would build a 376-byte temporary
        // and move it, reintroducing the copy this deferral deletes.
        ImageInfo& Info() {
            if (!info) {
                info.emplace(deferred_tsharp, deferred_is_depth);
            }
            return *info;
        }
        const ImageInfo& Info() const {
            return *info;
        }
        void EnsureViewInfo() {
            if (!view_ready) {
                view_info = ImageViewInfo{deferred_tsharp, type == BindingType::Storage,
                                          deferred_is_depth, deferred_is_array};
                view_ready = true;
            }
        }
    };
    // Rasterizer target descs are rebuilt with construct_at over an engaged
    // object; that stays legal only while nothing here needs a destructor.
    static_assert(std::is_trivially_destructible_v<ImageDesc>);

public:
    TextureCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                 AmdGpu::Liverpool* liverpool, BufferCache& buffer_cache, PageManager& tracker);
    ~TextureCache();

    TileManager& GetTileManager() noexcept {
        return tile_manager;
    }

    /// Invalidates any image in the logical page range.
    void InvalidateMemory(VAddr addr, size_t size);

    /// Marks an image as dirty if it exists at the provided address.
    void InvalidateMemoryFromGPU(VAddr address, size_t max_size);

    /// Evicts any images that overlap the unmapped range.
    void UnmapMemory(VAddr cpu_addr, size_t size);

    /// Schedules a copy of pending images for download back to CPU memory.
    void ProcessDownloadImages();

    /// Retrieves the image handle of the image with the provided attributes.
    [[nodiscard]] ImageId FindImage(ImageDesc& desc, bool exact_fmt = false);

    /// Retrieves image whose address matches provided
    [[nodiscard]] ImageId FindImageFromRange(VAddr address, size_t size, bool ensure_valid = true);

    /// Retrieves an image view with the properties of the specified image id.
    [[nodiscard]] vk::ImageView FindTexture(ImageId image_id, const ImageDesc& desc);

    /// Retrieves the render target with specified properties
    [[nodiscard]] ImageView& FindRenderTarget(ImageId image_id, const ImageDesc& desc);

    /// Retrieves the depth target with specified properties
    [[nodiscard]] ImageView& FindDepthTarget(ImageId image_id, const ImageDesc& desc);

    /// FindImage with the adaptive memo skip cache in front, for the shader
    /// texture binding path. A hit skips the page-table walk and match loops
    /// but still touches the LRU and re-applies any overlap view rebase.
    [[nodiscard]] ImageId FindImageMemoized(ImageDesc& desc, const AmdGpu::Image& tsharp);

    /// UpdateImage with the adaptive dedup skip cache in front. Only sampled
    /// texture bindings and render-target reuse go through here; storage
    /// images keep the full path.
    void MaybeUpdateImage(ImageId image_id);

    /// Updates image contents if it was modified by CPU.
    void UpdateImage(ImageId image_id);

    /// Resolves overlap between existing cache image and pending merged image
    [[nodiscard]] std::tuple<ImageId, int, int> ResolveOverlap(const ImageInfo& info,
                                                               BindingType binding,
                                                               ImageId cache_img_id,
                                                               ImageId merged_image_id);

    /// Resolves depth overlap and either re-creates the image or returns existing one
    [[nodiscard]] ImageId ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                              ImageId cache_img_id);

    /// Creates a new image with provided image info and copies subresources from image_id
    [[nodiscard]] ImageId ExpandImage(const ImageInfo& info, ImageId image_id);

    /// Reuploads image contents.
    void RefreshImage(Image& image);

    /// Retrieves the sampler that matches the provided S# descriptor.
    [[nodiscard]] vk::Sampler GetSampler(const AmdGpu::Sampler& sampler,
                                         AmdGpu::BorderColorBuffer border_color_base);

    /// Retrieves the image with the specified id.
    [[nodiscard]] Image& GetImage(ImageId id) {
        auto& image = slot_images[id];
        TouchImage(image);
        return image;
    }

    /// Retrieves the image view with the specified id.
    [[nodiscard]] ImageView& GetImageView(ImageId id) {
        return slot_image_views[id];
    }

    /// Get the associated depth stencil image if it is still valid.
    ImageId GetAssociatedDepth(Image& image) {
        if (!image.depth_id) {
            return {};
        }
        if (slot_images.is_allocated(image.depth_id)) {
            auto& depth_image = slot_images[image.depth_id];
            if (depth_image.image_uid == image.depth_uid &&
                depth_image.flags & ImageFlagBits::Registered) {
                return image.depth_id;
            }
        }
        // The linked depth image is no longer valid, disassociate it.
        image.DisassociateDepth();
        return {};
    }

    enum class MetaType {
        CMask,
        FMask,
        HTile,
    };

    /// Returns meta type if the specified address is a metadata surface.
    /// A single-cache-line Bloom filter answers the dominant negative case
    /// without probing the map. Insert-only: erases leave stale bits, which
    /// only cause a harmless fall-through to the real map.
    std::optional<MetaType> IsMeta(VAddr address) const {
        if (VideoCore::Skipcache::Framework::Instance().Active()) {
            const u64 h = address * 0x9e3779b97f4a7c15ULL;
            const u32 bit_a = static_cast<u32>(h >> 32) & 511;
            const u32 bit_b = static_cast<u32>(h >> 48) & 511;
            if (((meta_bloom_[bit_a >> 6] >> (bit_a & 63)) & 1) == 0 ||
                ((meta_bloom_[bit_b >> 6] >> (bit_b & 63)) & 1) == 0) {
                return std::nullopt;
            }
        }
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            return it->second.type;
        }
        return std::nullopt;
    }

    void MetaBloomInsert(VAddr address) {
        const u64 h = address * 0x9e3779b97f4a7c15ULL;
        const u32 bit_a = static_cast<u32>(h >> 32) & 511;
        const u32 bit_b = static_cast<u32>(h >> 48) & 511;
        meta_bloom_[bit_a >> 6] |= 1ULL << (bit_a & 63);
        meta_bloom_[bit_b >> 6] |= 1ULL << (bit_b & 63);
    }

    /// Returns true if a slice of the specified metadata surface has been cleared.
    bool IsMetaCleared(VAddr address, u32 slice) const {
        const auto& it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            return it.value().clear_mask & (1u << slice);
        }
        return false;
    }

    /// Clears all slices of the specified metadata surface.
    bool ClearMeta(VAddr address) {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            if (it.value().clear_mask != u32(-1)) {
                it.value().clear_mask = u32(-1);
                VideoCore::Skipcache::Framework::Instance().BumpMetaGen();
            }
            return true;
        }
        return false;
    }

    /// Updates the state of a slice of the specified metadata surface.
    bool TouchMeta(VAddr address, u32 slice, bool is_clear) {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            const u32 mask = it.value().clear_mask;
            const u32 new_mask = is_clear ? mask | (1u << slice) : mask & ~(1u << slice);
            if (new_mask != mask) {
                it.value().clear_mask = new_mask;
                VideoCore::Skipcache::Framework::Instance().BumpMetaGen();
            }
            return true;
        }
        return false;
    }

    /// Runs the garbage collector.
    void RunGarbageCollector();

    template <typename Func>
    void ForEachImageInRegion(VAddr cpu_addr, size_t size, Func&& func) {
        using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        ImageIds images;
        if (image_picked_.size() < slot_images.IndexCapacity()) {
            image_picked_.resize(slot_images.IndexCapacity());
        }
        ForEachPage(cpu_addr, size, [this, &images, cpu_addr, size, func](u64 page) {
            const auto it = page_table.find(page);
            if (it == nullptr) {
                if constexpr (BOOL_BREAK) {
                    return false;
                } else {
                    return;
                }
            }
            for (const PageImageRef& ref : *it) {
                // Mirrors Image::Overlaps exactly, from the bucket copy; the
                // Image itself is only touched for genuine overlaps, so the
                // Picked dedup semantics (including across nested walks) are
                // unchanged.
                if (ref.addr >= cpu_addr + size || cpu_addr >= ref.addr + ref.size) {
                    continue;
                }
                const ImageId image_id = ref.id;
                // Dedup against the dense per-image byte array: one warm
                // cache line covers hundreds of ids, where the old Picked
                // flag cost a cold load into the 752-byte Image slot per
                // duplicate and the local list cost a linear scan. Semantics
                // match the flag exactly, including across nested walks -
                // bits set by an outer walk stay set until its trailing
                // clear.
                if (image_picked_[image_id.index]) {
                    continue;
                }
                image_picked_[image_id.index] = 1;
                images.push_back(image_id);
                Image& image = slot_images[image_id];
                if constexpr (BOOL_BREAK) {
                    if (func(image_id, image)) {
                        return true;
                    }
                } else {
                    func(image_id, image);
                }
            }
            if constexpr (BOOL_BREAK) {
                return false;
            }
        });
        for (const ImageId image_id : images) {
            image_picked_[image_id.index] = 0;
        }
    }

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    static void ForEachPage(PAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> Traits::PageBits;
        for (u64 page = addr >> Traits::PageBits; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    /// Copies image memory back to CPU.
    void DownloadImageMemory(ImageId image_id, bool sync = false);

    /// Thread function for copying downloaded images out to CPU memory.
    void DownloadedImagesThread(const std::stop_token& token);

    /// Create an image from the given parameters
    [[nodiscard]] ImageId InsertImage(const ImageInfo& info, VAddr cpu_addr);

    /// Register image in the page table
    void RegisterImage(ImageId image);

    /// Unregister image from the page table
    void UnregisterImage(ImageId image);

    /// Track CPU reads and writes for image
    void TrackImage(ImageId image_id);
    void TrackImageHead(ImageId image_id);
    void TrackImageTail(ImageId image_id);

    /// Stop tracking CPU reads and writes for image
    void UntrackImage(ImageId image_id);
    void UntrackImageHead(ImageId image_id);
    void UntrackImageTail(ImageId image_id);

    void MarkAsMaybeDirty(ImageId image_id, Image& image);

    /// Removes the image and any views/surface metas that reference it.
    void DeleteImage(ImageId image_id);

    /// Touch the image in the LRU cache.
    /// Touch is idempotent within one gc tick; the inline mirror compare
    /// spares the call (one per binding per draw) entirely on repeats.
    void TouchImage(const Image& image) {
        if (image.lru_touch_tick == gc_tick &&
            VideoCore::Skipcache::Framework::Instance().Active()) {
            return;
        }
        TouchImageSlow(image);
    }
    void TouchImageSlow(const Image& image);

    /// Overlap resolution, validation, and creation for FindImage when no
    /// accepted perfect match exists. Requires the cache mutex to be held.
    SHAD_NO_INLINE ImageId FindImageSlow(ImageDesc& desc, bool exact_fmt, ImageId image_id,
                                         const ImageIds& image_ids, int& out_view_mip,
                                         int& out_view_slice);

    struct SamplerMemoEntry {
        std::array<u64, 2> key{};
        vk::Sampler handle{};
        u64 lru_id{};
        u64 sampler_gen{};
        u64 touch_tick{};
        bool valid{};
    };
    std::array<SamplerMemoEntry, 256> sampler_memo_{};
    u64 sampler_gen_{1};

    struct FindImageMemoEntry {
        std::array<u64, 4> tsharp_raw{};
        u64 image_uid{};
        u64 tex_gen{};
        u8 type{};
        u8 view_key{}; // is_depth | is_array << 1: the view build's other inputs
        bool valid{};
        u64 access_tick{};
        u64 lru_tick{};
        ImageId image_id{};
        // Post-rebase view info: a consumed hit hands the binding a complete
        // view without rebuilding it, and a verify compares all of it.
        ImageViewInfo view_info{};
        // The view handle FindTexture resolved for this entry on the backing
        // it names; null until a slow pass wrote it back.
        vk::ImageView view_handle{};
        const Image::BackingImage* view_backing{};
    };
    std::array<FindImageMemoEntry, 1024> find_image_memo_{};
    u64 view_memo_hits_{};
    u64 view_memo_slow_{};
    u64 view_memo_writebacks_{};

public:
    struct ViewMemoStats {
        u64 hits;
        u64 slow;
        u64 writebacks;
    };
    ViewMemoStats DrainViewMemoStats() {
        const ViewMemoStats out{view_memo_hits_, view_memo_slow_, view_memo_writebacks_};
        view_memo_hits_ = view_memo_slow_ = view_memo_writebacks_ = 0;
        return out;
    }

private:
    void FreeImage(ImageId image_id) {
        UntrackImage(image_id);
        UnregisterImage(image_id);
        DeleteImage(image_id);
    }

    void GarbageCollectImages();
    void GarbageCollectSamplers();

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    BufferCache& buffer_cache;
    PageManager& tracker;
    BlitHelper blit_helper;
    TileManager tile_manager;
    Common::SlotVector<Image> slot_images;
    Common::SlotVector<ImageView> slot_image_views;
    tsl::robin_map<u64, Sampler> samplers;
    std::unordered_set<ImageId> download_images;
    u64 total_used_memory = 0;
    u64 trigger_gc_memory = 0;
    u64 pressure_gc_memory = 0;
    u64 critical_gc_memory = 0;
    u64 total_used_samplers = 0;
    u64 trigger_gc_samplers = 0;
    u64 pressure_gc_samplers = 0;
    u64 critical_gc_samplers = 0;
    u64 gc_tick = 0;
    Common::LeastRecentlyUsedCache<ImageId, u64> lru_cache;
    Common::LeastRecentlyUsedCache<u64, u64> sampler_lru_cache;
    bool readback_linear_images;
    // Latched once at construction; gates the lock-free UpdateImage fast path.
    bool image_fast_state;
    bool view_memo; // latched once at construction
    PageTable page_table;
    std::mutex mutex;
    std::mutex samplers_mutex;
    std::mutex download_images_mutex;
    struct MetaDataInfo {
        MetaType type;
        s32 clear_mask = -1;
    };
    // Guest addresses are at least 256-byte aligned, and tsl::robin_map
    // masks the hash with a power-of-two bucket count: the default identity
    // hash then reaches only every 2^k-th home bucket and robin-hood
    // displacement builds long clustered probe chains - the chains were
    // ~75% of FindImage. The splitmix64 finalizer pushes entropy into the
    // LOW bits the mask keeps.
    struct MixedVAddrHash {
        size_t operator()(VAddr addr) const noexcept {
            u64 a = addr;
            a ^= a >> 33;
            a *= 0xff51afd7ed558ccdULL;
            a ^= a >> 29;
            return static_cast<size_t>(a);
        }
    };
    tsl::robin_map<VAddr, MetaDataInfo, MixedVAddrHash> surface_metas;
    // Images keyed by their exact guest base address. FindImageFromRange only
    // ever matches on equality, so the page walk it used to do was a range
    // scan answering an exact-match question.
    tsl::robin_map<VAddr, boost::container::small_vector<ImageId, 2>, MixedVAddrHash>
        images_by_addr;
    // Dense dedup bits for ForEachImageInRegion, indexed by ImageId; sized to
    // the slot vector's index capacity at walk start.
    std::vector<u8> image_picked_;
    alignas(64) std::array<u64, 8> meta_bloom_{};
};

} // namespace VideoCore
