// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
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
    // 768-byte Image struct - the dominant cost of FindImage was that
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
        u64 memo_bind_epoch{};
        vk::ImageLayout memo_bind_layout{};

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

    /// Whether a storage image download is queued; read on the GPU thread,
    /// which is the only writer of the queue.
    bool HasPendingDownloads() const noexcept {
        return !download_images.empty();
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
        TouchImageUnlocked(image, id);
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

    /// Walks images oldest-first up to the tick, from the list or the touch
    /// log; the callback may free the current image and may stop the walk by
    /// returning true. Tombstones are skipped, the leading run is dropped.
    template <typename Func>
    void ForEachLruBelow(u64 tick, Func&& func) {
        if (!lru_log) {
            lru_cache.ForEachItemBelow(tick, func);
            return;
        }
        while (lru_head_ < lru_log_.size() && !lru_log_[lru_head_].id) {
            ++lru_head_;
            --lru_dead_;
        }
        for (size_t i = lru_head_; i < lru_log_.size(); ++i) {
            const LruLogEntry e = lru_log_[i]; // func may tombstone, never pushes
            if (static_cast<s64>(tick) - static_cast<s64>(e.tick) < 0) {
                return;
            }
            ++lru_log_walked_;
            if (!e.id) {
                ++lru_log_skipped_;
                continue;
            }
            const size_t size_before = lru_log_.size();
            if constexpr (std::is_same_v<std::invoke_result_t<Func, ImageId>, bool>) {
                const bool stop = func(e.id);
                DEBUG_ASSERT(lru_log_.size() == size_before);
                if (stop) {
                    return;
                }
            } else {
                func(e.id);
                DEBUG_ASSERT(lru_log_.size() == size_before);
            }
        }
    }

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
                // flag cost a cold load into the 768-byte Image slot per
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
    void TouchImage(Image& image, ImageId id) {
        if (image.lru_touch_tick == gc_tick &&
            VideoCore::Skipcache::Framework::Instance().Active()) {
            return;
        }
        TouchImageSlow(image, id);
    }
    void TouchImageSlow(Image& image, ImageId id);
    // FindTexture's two cold arms: the storage binding's mark-and-update, and
    // the view resolve with its memo write-back. The header keeps only the
    // update dispatch and the memoized-handle return.
    SHAD_NO_INLINE void FindTextureStorage(Image& image, ImageId image_id);
    SHAD_NO_INLINE vk::ImageView FindTextureSlow(Image& image, ImageId image_id,
                                                 const ImageDesc& desc);
    /// Touch from a caller that does not hold the cache mutex; the touch log
    /// takes it, the list runs unlocked as it always has.
    void TouchImageUnlocked(Image& image, ImageId id) {
        if (image.lru_touch_tick == gc_tick &&
            VideoCore::Skipcache::Framework::Instance().Active()) {
            return;
        }
        TouchImageSlowUnlocked(image, id);
    }
    SHAD_NO_INLINE void TouchImageSlowUnlocked(Image& image, ImageId id);

    // Lock-free tier of UpdateImage: a clean, tracked image touched within the
    // interval proves the locked pass a no-op. Callers gate on image_fast_state.
    static constexpr u64 kTouchIntervalTicks = 8192;
    bool UpdateImageFast(const Image& image, u64 now_tick) {
        const u64 fast = image.ReadFastState();
        const bool dirty = (fast & Image::kFastStateDirty) != 0;
        const bool tracked = (fast & Image::kFastStateTracked) != 0;
        const u64 last_tick = fast >> Image::kFastStateTouchShift;
        if (dirty || !tracked || now_tick - last_tick > kTouchIntervalTicks) {
            return false;
        }
        update_fast_ += image_update_direct;
        return true;
    }
    void UpdateImage(Image& image, ImageId image_id);
    void UpdateImageSlow(ImageId image_id, u64 now_tick);

    /// Overlap resolution, validation, and creation for FindImage when no
    /// accepted perfect match exists. Requires the cache mutex to be held.
    SHAD_NO_INLINE ImageId FindImageSlow(ImageDesc& desc, bool exact_fmt, ImageId image_id,
                                         const ImageIds& image_ids, int& out_view_mip,
                                         int& out_view_slice);

    // 2-way set-associative, one 64-byte line per set; validity is a non-null
    // handle. GarbageCollectSamplers clears the whole memo whenever it erases,
    // so a live entry's handle, lru_id and map entry are live.
    struct SamplerMemoEntry {
        std::array<u64, 2> key{};
        vk::Sampler handle{};
        u32 lru_id{};
        u32 touch_tick{};
    };
    static_assert(sizeof(SamplerMemoEntry) == 32);
    static constexpr size_t SamplerMemoSets = 256; // 16 KB, L1-resident
    alignas(64) std::array<SamplerMemoEntry, SamplerMemoSets * 2> sampler_memo_{};
    u64 sampler_calls_{};
    u64 sampler_slow_{};
    u64 sampler_touches_{};

public:
    struct SamplerStats {
        u64 calls;
        u64 slow;
        u64 touches;
        u64 map;
    };
    SamplerStats DrainSamplerStats() {
        const SamplerStats out{sampler_calls_, sampler_slow_, sampler_touches_, samplers.size()};
        sampler_calls_ = sampler_slow_ = sampler_touches_ = 0;
        return out;
    }

private:
    // Image memo entry: line 0 holds what a probe and a hit read, line 1 what a
    // consumed hit copies out, line 2 the stamps of the locked touch path and
    // the recency stamp the victim scan reads.
    struct alignas(64) FindImageMemoEntry {
        std::array<u64, 4> tsharp_raw{};
        u64 image_uid{};
        u64 tex_gen{};
        // The view handle FindTexture resolved for this entry on the backing
        // it names; null until a slow pass wrote it back.
        const Image::BackingImage* view_backing{};
        ImageId image_id{};
        u8 type{};
        u8 view_key{}; // is_depth | is_array << 1: the view build's other inputs
        bool valid{};
        // Post-rebase view info: a consumed hit hands the binding a complete
        // view without rebuilding it, and a verify compares all of it.
        alignas(64) ImageViewInfo view_info{};
        vk::ImageView view_handle{};
        alignas(64) u64 access_tick{};
        u64 lru_tick{};
        // Backing epoch at which a shader-read transit of this view was a
        // no-op (0 = none) and the descriptor layout the backing held then.
        u64 bind_epoch{};
        vk::ImageLayout bind_layout{};
        // Last touch; the smallest stamp in a full set is the LRU victim.
        u64 touch_stamp{};
    };
    static_assert(sizeof(FindImageMemoEntry) == 192);
    static_assert(offsetof(FindImageMemoEntry, view_info) == 64);
    static_assert(offsetof(FindImageMemoEntry, access_tick) == 128);
    static_assert(offsetof(FindImageMemoEntry, touch_stamp) == 160);
    static constexpr size_t FindImageMemoEntries = 2048;
    std::array<FindImageMemoEntry, FindImageMemoEntries> find_image_memo_{};
    u32 MemoVictim(const FindImageMemoEntry* set, u32 ways) const;
    // The authoritative arm of FindImageMemoized: the real lookup, the verify
    // and the populate. packed = ways | matched << 8 | would_hit << 9 |
    // deferred << 10 | timed << 11, one register for the probe's verdicts.
    SHAD_NO_INLINE ImageId FindImageMemoizedSlow(ImageDesc& desc, const AmdGpu::Image& tsharp,
                                                 FindImageMemoEntry& e, u64 packed, u64 tex_gen);
    u64 view_memo_hits_{};
    u64 view_memo_slow_{};
    u64 view_memo_writebacks_{};

public:
    struct ViewMemoStats {
        u64 hits;
        u64 slow;
        u64 writebacks;
    };
    struct FindTouchStats {
        u64 consumed;
        u64 locks;
    };
    struct FindImageWayStats {
        u32 ways;
        std::array<u64, 4> hits;
        u64 evictions;
    };
    FindImageWayStats DrainFindImageWayStats() {
        const FindImageWayStats out{memo_ways, findimg_way_hits_, findimg_evictions_};
        findimg_way_hits_ = {};
        findimg_evictions_ = 0;
        return out;
    }
    bool BindNoopMemo() const noexcept {
        return bind_noop;
    }
    struct BindNoopStats {
        u64 records;
        u64 zero;
    };
    BindNoopStats DrainBindNoopStats() {
        const BindNoopStats out{bind_noop_records_, bind_noop_zero_};
        bind_noop_records_ = bind_noop_zero_ = 0;
        return out;
    }
    /// Records, for a binding that just took the slow transit path, whether a
    /// shader-read transit is a no-op under the backing's current epoch.
    void RecordBindNoop(ImageId image_id, const ImageDesc& desc, vk::ImageLayout dst_layout);
    struct ImageUpdateStats {
        u64 fast;
        u64 relock;
        u64 full;
    };
    ImageUpdateStats DrainImageUpdateStats() {
        const ImageUpdateStats out{update_fast_, update_relock_, update_full_};
        update_fast_ = update_relock_ = update_full_ = 0;
        return out;
    }
    struct AddrFilterStats {
        u64 calls;
        u64 cands;
        u64 fast;
        u64 walk;
    };
    AddrFilterStats DrainAddrFilterStats() {
        const AddrFilterStats out{addr_filter_calls_, addr_filter_cands_, addr_filter_fast_,
                                  addr_filter_walk_};
        addr_filter_calls_ = addr_filter_cands_ = addr_filter_fast_ = addr_filter_walk_ = 0;
        return out;
    }
    FindTouchStats DrainFindTouchStats() {
        const FindTouchStats out{findimg_consumed_, findimg_touch_locks_};
        findimg_consumed_ = findimg_touch_locks_ = 0;
        return out;
    }
    struct LruLogStats {
        u64 pushes;
        u64 walked;
        u64 skipped;
        u64 compactions;
        u64 size;
        u64 dead;
    };
    LruLogStats DrainLruLogStats() {
        const LruLogStats out{lru_log_pushes_,      lru_log_walked_, lru_log_skipped_,
                              lru_log_compactions_, lru_log_.size(), lru_dead_};
        lru_log_pushes_ = lru_log_walked_ = lru_log_skipped_ = lru_log_compactions_ = 0;
        return out;
    }
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
    // Touch log: the LRU order as an append-only vector. An image's live entry
    // is the last one pushed for it; a tombstone (null id) marks the superseded
    // or freed ones. Entries move only in the compaction.
    struct LruLogEntry {
        ImageId id;
        u64 tick; // never wraps
    };
    std::vector<LruLogEntry> lru_log_;
    size_t lru_head_{}; // every entry before it is a tombstone
    u64 lru_dead_{};    // tombstones at or after lru_head_
    u64 lru_log_pushes_{};
    u64 lru_log_walked_{};
    u64 lru_log_skipped_{};
    u64 lru_log_compactions_{};
    Common::LeastRecentlyUsedCache<u64, u64> sampler_lru_cache;
    bool readback_linear_images;
    // Latched once at construction; gates the lock-free UpdateImage fast path.
    bool image_fast_state;
    bool view_memo;              // latched once at construction
    bool sampler_lockfree;       // latched once at construction
    bool findimg_touch_lockfree; // latched once at construction
    bool bind_noop;              // latched once at construction; needs view_memo
    bool image_update_direct;    // latched once at construction; needs image_fast_state
    bool lru_log;                // latched once at construction
    bool invalidate_filter;      // latched once at construction
    u64 update_fast_{};
    u64 update_relock_{};
    u64 update_full_{};
    u64 bind_noop_records_{};
    u64 bind_noop_zero_{};
    bool MemoEntryMatches(const FindImageMemoEntry& e, const ImageDesc& desc,
                          ImageId image_id) const;
    u32 memo_ways;      // findimg_memo_ways, clamped to 0/1/2/4 at construction
    u32 memo_set_shift; // 64 - log2(sets): the mixed T# key's top bits index the set
    std::array<u64, 4> findimg_way_hits_{};
    u64 findimg_evictions_{};
    u64 findimg_touch_seq_{};
    u64 findimg_consumed_{};
    u64 findimg_touch_locks_{};
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
    // Exact-address filter fields, dense and indexed by ImageId. The filter
    // read them out of each candidate's Image, where guest_size sits 320 bytes
    // from the extent and format block, so a candidate cost two cold lines of
    // a 768-byte slot. Every field is fixed while the image is registered.
    struct AddrFilter {
        u32 guest_size;
        vk::Format pixel_format;
        u32 type;
        SubresourceExtent resources;
        Extent3D size;
    };
    static_assert(sizeof(AddrFilter) == 32);
    // 64-byte storage so no 32-byte record straddles two lines.
    struct alignas(64) AddrFilterPair {
        std::array<AddrFilter, 2> records;
    };
    std::vector<AddrFilterPair> addr_filter_;
    AddrFilter& AddrFilterOf(u32 index) {
        return addr_filter_[index >> 1].records[index & 1];
    }
    u64 addr_filter_calls_{};
    u64 addr_filter_cands_{};
    u64 addr_filter_fast_{};
    u64 addr_filter_walk_{};
    alignas(64) std::array<u64, 8> meta_bloom_{};

    // Coverage bitmap of the registered images at 64KiB granules over the
    // 40-bit guest space, written under the mutex and probed without it by
    // the fault path. A granule's bit is set before its image reaches the
    // page table and cleared only once no image is left in it, so a clear
    // bit proves the locked walk would visit nothing. Always maintained, so
    // the probe can be audited with the filter off.
public:
    struct InvalidateFilterStats {
        u64 probes;
        u64 skips;
        u64 unsound;
    };
    InvalidateFilterStats DrainInvalidateFilterStats() noexcept {
        return {invfilter_probes_.exchange(0, std::memory_order_relaxed),
                invfilter_skips_.exchange(0, std::memory_order_relaxed),
                invfilter_unsound_.exchange(0, std::memory_order_relaxed)};
    }

private:
    static constexpr u32 CoverGranuleBits = 16;
    static constexpr size_t CoverWords = size_t{1} << (40 - CoverGranuleBits - 6);
    bool CoverAny(VAddr addr, size_t size) const noexcept {
        const u64 first = addr >> CoverGranuleBits;
        const u64 last = (addr + size - 1) >> CoverGranuleBits;
        for (u64 g = first; g <= last; ++g) {
            if ((g >> 6) >= CoverWords) {
                return true;
            }
            if ((invalidate_cover_[g >> 6].load(std::memory_order_acquire) >> (g & 63)) & 1) {
                return true;
            }
        }
        return false;
    }
    void CoverSet(VAddr addr, size_t size) noexcept {
        const u64 first = addr >> CoverGranuleBits;
        const u64 last = (addr + size - 1) >> CoverGranuleBits;
        for (u64 g = first; g <= last && (g >> 6) < CoverWords; ++g) {
            invalidate_cover_[g >> 6].fetch_or(u64{1} << (g & 63), std::memory_order_release);
        }
    }
    void CoverRecompute(VAddr addr, size_t size) {
        const u64 first = addr >> CoverGranuleBits;
        const u64 last = (addr + size - 1) >> CoverGranuleBits;
        for (u64 g = first; g <= last && (g >> 6) < CoverWords; ++g) {
            // Straight off the page table: the picked dedup of the image walk
            // would hide an image an enclosing walk has already visited.
            const VAddr g_addr = g << CoverGranuleBits;
            constexpr size_t g_size = size_t{1} << CoverGranuleBits;
            bool any = false;
            ForEachPage(g_addr, g_size, [&](u64 page) {
                const auto it = page_table.find(page);
                if (it == nullptr) {
                    return;
                }
                for (const PageImageRef& ref : *it) {
                    if (ref.addr < g_addr + g_size && g_addr < ref.addr + ref.size) {
                        any = true;
                        return;
                    }
                }
            });
            if (!any) {
                invalidate_cover_[g >> 6].fetch_and(~(u64{1} << (g & 63)),
                                                    std::memory_order_release);
            }
        }
    }
    std::unique_ptr<std::atomic<u64>[]> invalidate_cover_;
    alignas(64) std::atomic<u64> invfilter_probes_{};
    std::atomic<u64> invfilter_skips_{};
    std::atomic<u64> invfilter_unsound_{};
};

} // namespace VideoCore
