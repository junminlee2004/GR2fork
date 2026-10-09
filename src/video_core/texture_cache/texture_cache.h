// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <type_traits>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include "common/assert.h"
#include "common/lru_cache.h"
#include "common/multi_level_page_table.h"
#include "common/slot_vector.h"
#include "shader_recompiler/resource.h"
#include "video_core/skipcache/skipcache.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/sampler.h"
#include "video_core/texture_cache/tile_manager.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Vulkan {
class Runtime;
}

namespace VideoCore {

class Buffer;
class BufferCache;
class PageManager;

class TextureCache {
    static constexpr s64 DEFAULT_PRESSURE_GC_MEMORY = 1_GB + 512_MB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 2_GB;  // Optimized for 4GB VRAM systems
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;

    struct BucketEntry {
        u32 key;
        u32 size;
        ImageId id;

        bool Overlaps(VAddr addr, size_t size) const noexcept {
            const VAddr base = Address();
            return base < (addr + size) && addr < (base + this->size);
        }

        VAddr Address() const noexcept {
            return VAddr(key) << 8;
        }
    };

    struct alignas(64) Bucket {
        SmallVector<BucketEntry, 4, u32> entries;
    };
    static_assert(sizeof(Bucket) == 64);

    struct Traits {
        using Entry = Bucket;
        static constexpr size_t ADDRESS_SPACE_BITS = 40;
        static constexpr size_t L1_BITS = 10;
        static constexpr size_t PAGE_BITS = 18;
        static constexpr bool NULL_CHECK = true;
    };
    using PageTable = Common::MultiLevelPageTable<Traits>;

public:
    enum class BindingType : u32 {
        Texture,
        Storage,
        RenderTarget,
        DepthTarget,
        VideoOut,
    };

    struct ImageDesc {
        // Lazy for shader-resource bindings, materialized by Info() only on routes that reach
        // FindImage. Target ctors (CB/DB/VideoOut) stay eager: FindRenderTarget and FindDepthTarget
        // read the engaged value through the const accessor.
        std::optional<ImageInfo> info;
        ImageViewInfo view_info;
        BindingType type{BindingType::Texture};
        AmdGpu::Image deferred_tsharp{};
        bool deferred_is_depth{};

        // Deferred view: built only on routes that reach FindImage. Bindings that mutate the view
        // before the probe (mip fallback) build it eagerly.
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

        // Emplace, never assign: assignment would build a temporary ImageInfo
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
        // bind_image_lean: writes only what a deferred probe reads. The memo
        // hit and the slow arm each overwrite every other field before pass
        // two reads it; ClearMemo is the slow arm's half of that contract.
        void PrimeDeferred(const AmdGpu::Image& image, const Shader::ImageResource& desc) {
            info.reset();
            type = desc.is_written ? BindingType::Storage : BindingType::Texture;
            deferred_tsharp = image;
            deferred_is_depth = desc.is_depth;
            view_ready = false;
            deferred_is_array = desc.is_array;
        }
        void ClearMemo() {
            memo_view = vk::ImageView{};
            memo_backing = nullptr;
            memo_slot = NoMemoSlot;
            memo_bind_epoch = 0;
            memo_bind_layout = {};
        }
    };
    // Rasterizer target descs are rebuilt with construct_at over an engaged
    // object; that stays legal only while nothing here needs a destructor.
    static_assert(std::is_trivially_destructible_v<ImageDesc>);

public:
    TextureCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                 Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool, BufferCache& buffer_cache,
                 PageManager& tracker);
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
    [[nodiscard]] ImageId FindImageMemoized(ImageDesc& desc, const AmdGpu::Image& tsharp,
                                            u16* hint = nullptr);

    /// UpdateImage with the adaptive dedup skip cache in front. Only sampled
    /// texture bindings and render-target reuse go through here; storage
    /// images keep the full path.
    void MaybeUpdateImage(ImageId image_id);

    /// Links the stencil plane of a depth surface to its depth image.
    void AssociateStencil(ImageId depth_id, const ImageInfo& depth_info);

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
                                         AmdGpu::BorderColorBuffer border_color_base,
                                         bool is_depth);

    /// Retrieves the image with the specified id.
    [[nodiscard]] Image& GetImage(ImageId id) {
        auto& image = slot_images[id];
        TouchImage(image);
        return image;
    }

    // UpdateImage would take its no-op tier for this image right now. Reads the
    // slot directly so the LRU is not touched.
    [[nodiscard]] bool IsImageUpdateNoop(ImageId id, u64 now_tick) const noexcept {
        return FastStateNoop(slot_images[id].ReadFastState(), now_tick);
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
        if (slot_images.IsAllocated(image.depth_id)) {
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
            return it->second.clear_mask & (1u << slice);
        }
        return false;
    }

    /// Clears all slices of the specified metadata surface.
    bool ClearMeta(VAddr address) {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            if (it->second.clear_mask != u32(-1)) {
                it->second.clear_mask = u32(-1);
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
            const u32 mask = it->second.clear_mask;
            const u32 new_mask = is_clear ? mask | (1u << slice) : mask & ~(1u << slice);
            if (new_mask != mask) {
                it->second.clear_mask = new_mask;
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
        const u64 first_page = cpu_addr >> Traits::PAGE_BITS;
        ForEachPage(cpu_addr, size, [this, first_page, cpu_addr, size, func](u64 page) {
            const auto it = page_table.find(page);
            if (it == nullptr) {
                if constexpr (BOOL_BREAK) {
                    return false;
                } else {
                    return;
                }
            }
            for (const auto& entry : it->entries) {
                const u64 base_page = entry.Address() >> Traits::PAGE_BITS;
                if (page != std::max(first_page, base_page)) {
                    continue;
                }
                if (!entry.Overlaps(cpu_addr, size)) {
                    continue;
                }
                Image& image = slot_images[entry.id];
                if constexpr (BOOL_BREAK) {
                    if (func(entry.id, image)) {
                        return true;
                    }
                } else {
                    func(entry.id, image);
                }
            }
            if constexpr (BOOL_BREAK) {
                return false;
            }
        });
    }

    template <typename Func>
    void ForEachImageWithAddress(VAddr cpu_addr, Func&& func) {
        const auto it = page_table.find(cpu_addr >> Traits::PAGE_BITS);
        if (it == nullptr) {
            return;
        }
        const u32 key = u32(cpu_addr >> 8);
        for (const auto& entry : it->entries) {
            if (entry.key == key) {
                func(entry.id, slot_images[entry.id]);
            }
        }
    }

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    static void ForEachPage(PAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> Traits::PAGE_BITS;
        for (u64 page = addr >> Traits::PAGE_BITS; page <= page_end; ++page) {
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

    /// readback_linear_images_async: records the copy of one queued image into its own staging
    /// buffer; a background thread writes the pixels to guest memory once the GPU finishes.
    /// Returns true when a copy was recorded.
    bool DownloadImageMemoryAsync(ImageId image_id);

    /// Register image in the page table
    void RegisterImage(ImageId image);

    /// Unregister image from the page table
    void UnregisterImage(ImageId image);

    // findimg_range_invalidate: the T# range a populated image memo slot
    // answers for, packed to eight bytes as page numbers. An all-zero pair is
    // the empty range and matches nothing.
    struct MemoRange {
        u32 lo_page;
        u32 hi_page;
    };
    static_assert(sizeof(MemoRange) == 8);

    /// findimg_range_invalidate: the outward-rounded page pair a byte range occupies.
    static MemoRange MemoRangeOf(VAddr addr, u64 size) noexcept;

    /// findimg_range_invalidate: clear image memo entries intersecting [addr, addr + size). GPU
    /// command thread only; queued instead while a batch is open.
    SHAD_NO_INLINE void InvalidateMemoRange(VAddr addr, u64 size);

    /// findimg_range_invalidate: walks on the GPU command thread, else bumps the memo generation.
    void InvalidateMemoForImage(const ImageInfo& info);

    /// Apply every queued range in one pass over the side array.
    SHAD_NO_INLINE void FlushMemoRangeBatch();

    /// One pass over the side array for `count` ranges; `count == 1` is the single-range walk.
    void MemoRangeWalk(const MemoRange* ranges, u32 count);

    /// Track CPU reads and writes for image
    void TrackImage(ImageId image_id);

    /// Stop tracking CPU reads and writes for image
    void UntrackImage(ImageId image_id);
    void UntrackImageHead(ImageId image_id);
    void UntrackImageTail(ImageId image_id);

    void MarkAsMaybeDirty(ImageId image_id, Image& image);

    /// Removes the image and any views/surface metas that reference it.
    void DeleteImage(ImageId image_id);

    /// Touch the image in the LRU cache.
    /// With texture_lru_lazy_touch only the gc tick is stamped; the garbage collector
    /// walk relinks the entry when it meets it.
    void TouchImage(Image& image) {
        if (lru_lazy_touch) {
            image.gc_tick_accessed_last = gc_tick;
            return;
        }
        image_lru_cache.Touch(image, gc_tick);
    }
    // FindTexture's two cold arms: the storage binding's mark-and-update, and
    // the view resolve with its memo write-back.
    SHAD_NO_INLINE void FindTextureStorage(Image& image, ImageId image_id);
    SHAD_NO_INLINE vk::ImageView FindTextureSlow(Image& image, ImageId image_id,
                                                 const ImageDesc& desc);

    // Fast tier of UpdateImage: a clean, tracked image touched within the
    // interval proves the full pass a no-op. Callers gate on image_fast_state.
    static constexpr u64 kTouchIntervalTicks = 8192;
    static bool FastStateNoop(u64 fast, u64 now_tick) noexcept {
        return (fast & (Image::kFastStateDirty | Image::kFastStateTracked)) ==
                   Image::kFastStateTracked &&
               now_tick - (fast >> Image::kFastStateTouchShift) <= kTouchIntervalTicks;
    }
    bool UpdateImageFast(const Image& image, u64 now_tick) {
        if (!FastStateNoop(image.ReadFastState(), now_tick)) {
            return false;
        }
        update_fast_ += image_update_direct;
        return true;
    }
    void UpdateImage(Image& image, ImageId image_id);
    void UpdateImageSlow(ImageId image_id, u64 now_tick);

    /// Overlap resolution, validation, and creation for FindImage when no
    /// accepted perfect match exists.
    SHAD_NO_INLINE ImageId FindImageSlow(ImageDesc& desc, bool exact_fmt, ImageId image_id,
                                         const SmallVector<ImageId, 8>& image_ids,
                                         int& out_view_mip, int& out_view_slice);

    // 2-way set-associative, one 64-byte line per set; validity is a non-null
    // handle. GarbageCollectSamplers clears the whole memo whenever it erases,
    // so a live entry's handle, slot id and map entry are live.
    struct SamplerMemoEntry {
        std::array<u64, 2> key{};
        vk::Sampler handle{};
        SamplerId id{};
        u32 touch_tick{};
    };
    static_assert(sizeof(SamplerMemoEntry) == 32);
    static constexpr size_t SamplerMemoSets = 256; // 16 KB, L1-resident
    alignas(64) std::array<SamplerMemoEntry, SamplerMemoSets * 2> sampler_memo_{};
    u64 sampler_calls_{};
    u64 sampler_slow_{};
    u64 sampler_touches_{};

    // Image memo entry: line 0 holds what a probe and a hit read, line 1 what a
    // consumed hit copies out, line 2 the touch stamps and the recency stamp
    // the victim scan reads.
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
        alignas(64) u64 lru_tick{};
        // Backing epoch at which a shader-read transit of this view was a
        // no-op (0 = none) and the descriptor layout the backing held then.
        u64 bind_epoch{};
        vk::ImageLayout bind_layout{};
        // Last touch; the smallest stamp in a full set is the LRU victim.
        u64 touch_stamp{};
    };
    static_assert(sizeof(FindImageMemoEntry) == 192);
    static_assert(offsetof(FindImageMemoEntry, view_info) == 64);
    static_assert(offsetof(FindImageMemoEntry, lru_tick) == 128);
    static_assert(offsetof(FindImageMemoEntry, touch_stamp) == 152);
    // Sized once at construction from findimg_memo_entries; a power of two,
    // so the set index is the mixed key's top bits.
    std::vector<FindImageMemoEntry> find_image_memo_;
    u32 MemoVictim(const FindImageMemoEntry* set, u32 ways) const;
    // The authoritative arm of FindImageMemoized: the real lookup, the verify
    // and the populate. packed = ways | matched << 8 | would_hit << 9 |
    // deferred << 10 | timed << 11, one register for the probe's verdicts.
    SHAD_NO_INLINE ImageId FindImageMemoizedSlow(ImageDesc& desc, const AmdGpu::Image& tsharp,
                                                 FindImageMemoEntry& e, u64 packed, u64 tex_gen);
    // findimg_memo_first: T# validation. A pure function of the T# bytes; every populate follows a
    // gated FindImage, so a consumed hit needs no gate, and a new populate site must be gated too.
    SHAD_NO_INLINE bool GateTsharp(const AmdGpu::Image& tsharp);
    bool MemoEntryMatches(const FindImageMemoEntry& e, const ImageDesc& desc,
                          ImageId image_id) const;
    u64 tsgate_calls_{};
    u64 tsgate_rejects_{};
    u64 view_memo_hits_{};
    u64 view_memo_slow_{};
    u64 view_memo_writebacks_{};

public:
    struct ViewMemoStats {
        u64 hits;
        u64 slow;
        u64 writebacks;
    };
    struct FindImageWayStats {
        u32 ways;
        u64 entries;
        std::array<u64, 4> hits;
        u64 evictions;
    };
    struct FindImageHintStats {
        u64 probes;
        u64 hits;
        u64 none;
    };
    FindImageHintStats DrainFindImageHintStats() {
        const FindImageHintStats out{findimg_hint_probes_, findimg_hint_hits_, findimg_hint_none_};
        findimg_hint_probes_ = findimg_hint_hits_ = findimg_hint_none_ = 0;
        return out;
    }
    FindImageWayStats DrainFindImageWayStats() {
        const FindImageWayStats out{memo_ways, find_image_memo_.size(), findimg_way_hits_,
                                    findimg_evictions_};
        findimg_way_hits_ = {};
        findimg_evictions_ = 0;
        return out;
    }
    struct MemoRangeStats {
        bool enabled;
        u64 walks;
        u64 inval;
        u64 bumps;
    };
    MemoRangeStats DrainMemoRangeStats() {
        // The bump counter is written off the GPU command thread, so it is
        // drained with one exchange: a load/store pair would drop a concurrent
        // bump from the counter the go/no-go read rests on.
        const MemoRangeStats out{findimg_range_inval, memo_range_walks_, memo_range_inval_,
                                 memo_gen_bumps_.exchange(0, std::memory_order_relaxed)};
        memo_range_walks_ = memo_range_inval_ = 0;
        return out;
    }
    bool BindNoopMemo() const noexcept {
        return bind_noop;
    }
    bool MemoFirst() const noexcept {
        return memo_first;
    }
    struct TsGateStats {
        u64 calls;
        u64 rejects;
    };
    TsGateStats DrainTsGateStats() {
        const TsGateStats out{tsgate_calls_, tsgate_rejects_};
        tsgate_calls_ = tsgate_rejects_ = 0;
        return out;
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
    struct LruLazyStats {
        bool enabled;
        u64 gc_runs;
        u64 hard;
        u64 visits;
        u64 maxvisit;
        u64 relinks;
        u64 frees;
    };
    LruLazyStats DrainLruLazyStats() {
        const LruLazyStats out{lru_lazy_touch,   lru_lazy_gc_runs_,  lru_lazy_hard_,
                               lru_lazy_visits_, lru_lazy_maxvisit_, lru_lazy_relinks_,
                               lru_lazy_frees_};
        lru_lazy_gc_runs_ = lru_lazy_hard_ = lru_lazy_visits_ = lru_lazy_maxvisit_ =
            lru_lazy_relinks_ = lru_lazy_frees_ = 0;
        return out;
    }
    ViewMemoStats DrainViewMemoStats() {
        const ViewMemoStats out{view_memo_hits_, view_memo_slow_, view_memo_writebacks_};
        view_memo_hits_ = view_memo_slow_ = view_memo_writebacks_ = 0;
        return out;
    }

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

    /// Copies the tracked image out of slot_images.
    friend class PhotoReadback;

private:
    /// Removes image from the cache and schedules it for deletion.
    void FreeImage(ImageId image_id) {
        {
            std::scoped_lock lk{slot_images[image_id].mutex};
            UntrackImage(image_id);
        }
        UnregisterImage(image_id);
        DeleteImage(image_id);
    }

    void GarbageCollectImages();
    void GarbageCollectSamplers();

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    AmdGpu::Liverpool* liverpool;
    BufferCache& buffer_cache;
    PageManager& tracker;
    PageTable page_table;
    Common::SlotVector<Image> slot_images;
    Common::SlotVector<ImageView> slot_image_views;
    Common::SlotVector<Sampler> slot_samplers;
    BlitHelper blit_helper;
    TileManager tile_manager;
    absl::flat_hash_map<u64, SamplerId> samplers;
    absl::flat_hash_set<ImageId> download_images;
    u64 total_used_memory = 0;
    u64 trigger_gc_memory = 0;
    u64 pressure_gc_memory = 0;
    u64 critical_gc_memory = 0;
    u64 total_used_samplers = 0;
    u64 trigger_gc_samplers = 0;
    u64 pressure_gc_samplers = 0;
    u64 critical_gc_samplers = 0;
    u64 gc_tick = 0;
    Common::LRUCache<Image> image_lru_cache;
    // Lazy-touch GC-walk accounting; only written with lru_lazy_touch latched on.
    u64 lru_lazy_gc_runs_{};
    u64 lru_lazy_hard_{};     // passes configured pressured or aggressive
    u64 lru_lazy_visits_{};   // clean_up calls
    u64 lru_lazy_maxvisit_{}; // most clean_up calls in one GarbageCollectImages pass
    u64 lru_lazy_relinks_{};
    u64 lru_lazy_frees_{};
    Common::LRUCache<Sampler> sampler_lru_cache;
    const bool readback_linear_images;
    bool readback_linear_images_async{};
    // Staging buffers of readback_linear_images_async, handed back by the background writer. Its
    // own pool, because the runtime's staging pool is not safe to free into from that thread.
    std::mutex async_staging_mutex;
    std::vector<std::unique_ptr<Buffer>> async_staging_pool;
    // All latched once at construction; image_fast_state gates the UpdateImage
    // fast path.
    bool image_fast_state;
    bool view_memo;
    bool findimg_trust_gen;
    bool findimg_range_inval; // needs findimg_trust_gen
    bool memo_first;
    bool bind_noop;           // needs view_memo
    bool image_update_direct; // needs image_fast_state
    bool lru_lazy_touch;
    bool invalidate_filter;
    u64 update_fast_{};
    u64 update_relock_{};
    u64 update_full_{};
    u64 bind_noop_records_{};
    u64 bind_noop_zero_{};
    u32 memo_ways;      // findimg_memo_ways, clamped to 0/1/2/4 at construction
    u32 memo_set_shift; // 64 - log2(sets): the mixed T# key's top bits index the set
    std::array<u64, 4> findimg_way_hits_{};
    u64 findimg_hint_probes_{};
    u64 findimg_hint_hits_{};
    u64 findimg_hint_none_{};
    u64 findimg_evictions_{};
    u64 findimg_touch_seq_{};
    std::mutex download_images_mutex;
    struct MetaDataInfo {
        MetaType type;
        s32 clear_mask = -1;
    };
    absl::flat_hash_map<VAddr, MetaDataInfo> surface_metas;
    // Exact-address filter fields, dense and indexed by ImageId: a copy of what
    // the filter otherwise read from the cold lines of the candidate's Image
    // slot. Every field is fixed while the image is registered.
    struct alignas(32) AddrFilter {
        u32 guest_size;
        vk::Format pixel_format;
        u32 type;
        SubresourceExtent resources;
        Extent3D size;
    };
    // 32-byte alignment keeps every record inside one 64-byte line.
    static_assert(sizeof(AddrFilter) == 32);
    // Sized at construction for every index the slot vector can hand out, so
    // the storage stays put under the unlocked readers.
    std::vector<AddrFilter> addr_filter_;
    AddrFilter& AddrFilterOf(u32 index) {
        return addr_filter_[index];
    }
    u64 addr_filter_calls_{};
    u64 addr_filter_cands_{};
    u64 addr_filter_fast_{};
    u64 addr_filter_walk_{};
    alignas(64) std::array<u64, 8> meta_bloom_{};

    // Coverage bitmap of the registered images at 64KiB granules over the
    // 40-bit guest space, written under cover_mutex_ and probed without it by
    // the fault path. A granule's bit is set before its image reaches the
    // page table and cleared only once no image is left in it, so a clear
    // bit proves the page table walk would visit nothing. Always maintained,
    // so the probe can be audited with the filter off.
    static constexpr u32 CoverGranuleBits = 16;
    static constexpr size_t CoverWords = size_t{1} << (40 - CoverGranuleBits - 6);
    static constexpr u64 CoverLastGranule = (u64{CoverWords} << 6) - 1;
    bool CoverAny(VAddr addr, size_t size) const noexcept {
        const u64 first = addr >> CoverGranuleBits;
        const u64 last = (addr + size - 1) >> CoverGranuleBits;
        if (last > CoverLastGranule) {
            // Past the bitmap: nothing is certified, so the walk must run.
            return true;
        }
        for (u64 g = first; g <= last; ++g) {
            if ((invalidate_cover_[g >> 6].load(std::memory_order_acquire) >> (g & 63)) & 1) {
                return true;
            }
        }
        return false;
    }
    void CoverSet(VAddr addr, size_t size) noexcept {
        const u64 first = addr >> CoverGranuleBits;
        const u64 last = (addr + size - 1) >> CoverGranuleBits;
        const u64 lim = last < CoverLastGranule ? last : CoverLastGranule;
        for (u64 g = first; g <= lim; ++g) {
            invalidate_cover_[g >> 6].fetch_or(u64{1} << (g & 63), std::memory_order_release);
        }
    }
    void CoverRecompute(VAddr addr, size_t size) {
        const u64 first = addr >> CoverGranuleBits;
        const u64 last = (addr + size - 1) >> CoverGranuleBits;
        const u64 lim = last < CoverLastGranule ? last : CoverLastGranule;
        for (u64 g = first; g <= lim; ++g) {
            // Reads only the bucket entries, which the held cover_mutex_ keeps stable.
            const VAddr g_addr = g << CoverGranuleBits;
            constexpr size_t g_size = size_t{1} << CoverGranuleBits;
            bool any = false;
            ForEachPage(g_addr, g_size, [&](u64 page) {
                const auto it = page_table.find(page);
                if (it == nullptr) {
                    return;
                }
                for (const BucketEntry& entry : it->entries) {
                    if (entry.Overlaps(g_addr, g_size)) {
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
    // Serializes the bitmap writers with the page table edits they bracket.
    std::mutex cover_mutex_;
    alignas(64) std::atomic<u64> invfilter_probes_{};
    std::atomic<u64> invfilter_skips_{};
    std::atomic<u64> invfilter_unsound_{};
    // findimg_range_invalidate, declared last so the default-off arm keeps the
    // layout of every member the hot paths reach above.
    // Parallel to find_image_memo_: the range each populated slot answers for,
    // sized only when the setting is on, so the default arm allocates nothing.
    std::vector<MemoRange> memo_range_;
    // The generation the memo certifies with under the setting. Bumped only
    // where the walk cannot run or is not provably on the GPU command thread:
    // the guest-thread unmap route, the video-out registration route and the
    // two rebind arms.
    std::atomic<u64> img_memo_gen_{1};
    // Garbage collection frees up to forty images per pass; their ranges are
    // queued here and applied in one walk when the pass ends.
    static constexpr u32 MemoRangeBatchMax = 64;
    std::array<MemoRange, MemoRangeBatchMax> memo_range_batch_{};
    u32 memo_range_batch_count_{};
    bool memo_range_batching_{};
    u64 memo_range_walks_{};
    u64 memo_range_inval_{};
    // Bumped off the GPU command thread on the unmap and video-out routes,
    // drained on the GPU command thread: atomic for that read alone.
    std::atomic<u64> memo_gen_bumps_{};

    // The memo generation the probe certifies with, plus its drain counter.
    void BumpImgMemoGen() {
        img_memo_gen_.fetch_add(1, std::memory_order_release);
        memo_gen_bumps_.fetch_add(1, std::memory_order_relaxed);
    }
    // A rebind changes what a binding must resolve to beyond the T# range, so
    // the range-scoped arm still invalidates globally.
    void NoteRebind() {
        Skipcache::Framework::Instance().BumpTexGen();
        if (findimg_range_inval) {
            BumpImgMemoGen();
        }
    }
};

} // namespace VideoCore
