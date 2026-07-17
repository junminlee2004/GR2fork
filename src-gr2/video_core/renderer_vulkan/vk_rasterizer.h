// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/amdgpu/regs_snapshot.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/bundle_assembler.h"
#include "video_core/renderer_vulkan/draw_intent.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class Scheduler;
class GraphicsPipeline;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    [[nodiscard]] const Instance& GetInstance() const noexcept {
        return instance;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    // Intent bodies for the public Draw/Dispatch entry points, which only push intents to
    // BundleAssembler. The per-draw work runs here via BundleAssembler::ProcessOne, on the
    // producer thread or the GpuAssembler jthread depending on the assembler mode.
    void DoDrawFromIntent(const DrawIntent& intent);
    void DoDrawIndirectFromIntent(const DrawIntent& intent);
    void DoDispatchDirectFromIntent(const DrawIntent& intent);
    void DoDispatchIndirectFromIntent(const DrawIntent& intent);

    // Intent bodies for the scheduler-touching entry points (ScopeMarker*, FillBuffer,
    // CopyBuffer, CpSync, Flush, Finish, OnSubmit); called by BundleAssembler::ProcessOne
    // in dispatch order.
    void DoFlushFromIntent(const DrawIntent& intent);
    void DoFinishFromIntent(const DrawIntent& intent);
    void DoCpSyncFromIntent(const DrawIntent& intent);
    void DoOnSubmitFromIntent(const DrawIntent& intent);
    void DoFillBufferFromIntent(const DrawIntent& intent);
    void DoCopyBufferFromIntent(const DrawIntent& intent);
    void DoScopeMarkerBeginFromIntent(const DrawIntent& intent);
    void DoScopeMarkerEndFromIntent(const DrawIntent& intent);
    void DoScopedMarkerInsertFromIntent(const DrawIntent& intent);

    // Routes Presenter work that touches `draw_scheduler` through the assembler: the closure is
    // heap-owned by a PresenterRecord intent and self-destructs when processed. Returns the
    // packet_seq; blocking callers pair it with WaitForAssembler, fire-and-forget callers drop it.
    template <typename F>
    [[nodiscard]] u32 PushPresenterRecord(F&& f) {
        using FT = std::decay_t<F>;
        // Heap-allocating the closure costs ~60 allocs/sec at 60 fps - negligible.
        auto* state = new FT(std::forward<F>(f));
        DrawIntent intent;
        intent.type = DrawIntent::Type::PresenterRecord;
        intent.presenter_record.state = state;
        intent.presenter_record.invoke_and_destroy = +[](void* s) noexcept {
            auto* p = static_cast<FT*>(s);
            (*p)();
            delete p;
        };
        return bundle_assembler_.Push(intent);
    }

    void DoPresenterRecordFromIntent(const DrawIntent& intent);

    // Blocks until the assembler has processed the intent identified by `seq` (from
    // PushPresenterRecord or BundleAssembler::Push). Must not be called from the assembler
    // thread itself.
    void WaitForAssembler(u32 seq) {
        bundle_assembler_.WaitFor(seq);
    }

    // In-line ScopeMarker bodies. Rasterizer-internal callers (Resolve, EliminateFastClear,
    // DepthStencilCopy, the FilterDraw inserts) need markers to bracket the surrounding
    // data-plane work synchronously; the DoScopeMarker*FromIntent dispatchers also forward here.
    void DoScopeMarkerBeginInline(const std::string_view& str, bool from_guest);
    void DoScopeMarkerEndInline(bool from_guest);
    void DoScopedMarkerInsertInline(const std::string_view& str, u32 color,
                                    bool from_guest);

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size);
    bool ReadMemory(VAddr addr, u64 size);
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    void CpSync();
    u64 Flush();
    void Finish();
    void OnSubmit();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

private:
    // These helpers read register state from the snapshot (which mirrors `Regs`'s field
    // interface), not from `liverpool->regs`: with the assembler on its own thread the snapshot
    // is the only consistent reg view, as the live regs may have moved on to the next packet.
    void PrepareRenderState(const GraphicsPipeline* pipeline,
                            const AmdGpu::LiverpoolRegsSnapshot& regs);
    const RenderState& BeginRendering(const GraphicsPipeline* pipeline,
                                      const AmdGpu::LiverpoolRegsSnapshot& regs);
    void Resolve(const AmdGpu::LiverpoolRegsSnapshot& regs);
    void DepthStencilCopy(bool is_depth, bool is_stencil,
                          const AmdGpu::LiverpoolRegsSnapshot& regs);
    void EliminateFastClear(const AmdGpu::LiverpoolRegsSnapshot& regs);

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed,
                            const AmdGpu::LiverpoolRegsSnapshot& regs) const;
    void UpdateViewportScissorState(const AmdGpu::LiverpoolRegsSnapshot& regs) const;
    void UpdateDepthStencilState(const AmdGpu::LiverpoolRegsSnapshot& regs) const;
    void UpdatePrimitiveState(bool is_indexed,
                              const AmdGpu::LiverpoolRegsSnapshot& regs) const;
    void UpdateRasterizationState(const AmdGpu::LiverpoolRegsSnapshot& regs) const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline,
                                  const AmdGpu::LiverpoolRegsSnapshot& regs) const;

    bool FilterDraw(const AmdGpu::LiverpoolRegsSnapshot& regs);

    void BindBuffers(const Shader::Info& stage,
                     const Shader::ResolvedStageResources* resolved,
                     Shader::Backend::Bindings& binding, Shader::PushData& push_data,
                     const AmdGpu::LiverpoolRegsSnapshot& regs);
    void BindTextures(const Shader::Info& stage,
                      const Shader::ResolvedStageResources* resolved,
                      Shader::Backend::Bindings& binding,
                      const AmdGpu::LiverpoolRegsSnapshot& regs);
    bool BindResources(const Pipeline* pipeline,
                       const AmdGpu::LiverpoolRegsSnapshot& regs);

    void ResetBindings() {
        for (auto& image_id : bound_images) {
            // OPT: Use GetImageUntouched to avoid LRU touch overhead.
            // ResetBindings only clears transient binding flags and doesn't constitute
            // a "real" image access that should keep the image alive in cache.
            texture_cache.GetImageUntouched(image_id).binding = {};
        }
        bound_images.clear();
    }

    bool IsComputeMetaClear(const Pipeline* pipeline,
                            const AmdGpu::LiverpoolRegsSnapshot& regs);
    bool IsComputeImageCopy(const Pipeline* pipeline,
                            const AmdGpu::LiverpoolRegsSnapshot& regs);
    bool IsComputeImageClear(const Pipeline* pipeline,
                             const AmdGpu::LiverpoolRegsSnapshot& regs);

    /// Bind pipeline with deduplication - skips vkCmdBindPipeline when already bound.
    void BindPipelineCached(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    // GR2FORK PERF: generation counter for IsMapped's TLS cache (defined in vk_rasterizer.cpp).
    // MapMemory/UnmapMemory bump it under release order after mutating mapped_ranges; the
    // per-thread cache invalidates when it changes.
    std::atomic<u64> mapped_ranges_gen_{0};
    VideoCore::BufferCache::DmaSyncState last_dma_sync_state_{};
    u64 last_dma_mapped_ranges_gen_{};
    bool dma_sync_state_valid_{};
    PipelineCache pipeline_cache;

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;
    // GR2FORK PERF: descriptor writes assemble into this persistent array with
    // a register-held count; a per-element emplace_back pays a size RMW, a
    // capacity branch, and a zero-init the field writes discard. sType is set
    // once by the constructor and never rewritten.
    static constexpr u32 kMaxDescWrites = 256;
    std::array<vk::WriteDescriptorSet, kMaxDescWrites> set_writes{};
    u32 set_write_count{0};
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;

    // Descriptor delta cache (skips redundant push-descriptor writes).
    struct DescCacheEntry {
        u64 a{};           // buffer/sampler handle (packed as u64)
        u64 b{};           // offset/imageView handle
        u64 c{};           // range/layout
        u32 epoch{};       // validity epoch
        vk::DescriptorType type{};
    };

    // Push-descriptor delta cache key. Invalidates on tick rotation
    // (each new primary cmdbuf gets a fresh tick) or on Pipeline change.
    mutable u64 desc_cache_tick{};
    mutable vk::PipelineLayout desc_cache_layout{};
    mutable const Pipeline* desc_cache_pipeline{};
    mutable u32 desc_cache_epoch{1};
    // GR2FORK: probe instrumentation classifying descriptor-cache epoch invalidations for the
    // layout-sharing evaluation; desc_cache_shape_hash_ mirrors the bound pipeline's layout shape
    // hash so the match check never dereferences the possibly stale desc_cache_pipeline pointer.
    // TODO: remove once the layout-sharing evaluation concludes.
    mutable u64 desc_cache_shape_hash_{};
    mutable u64 l2probe_inv_total_{};
    mutable u64 l2probe_inv_tick_rot_{};
    mutable u64 l2probe_inv_pipe_same_tick_{};
    mutable u64 l2probe_inv_shape_match_{};
    mutable u64 l2probe_next_log_{1u << 18};
    // GR2FORK PERF: RADV's descriptor-set path always allocates a fresh VkDescriptorSet, so
    // ShouldWriteDescriptor must always return true; the inline wrapper short-circuits without
    // touching desc_cache, eliminating ~10-30 dead cache lookups + writes per draw on RADV.
    mutable bool desc_cache_force_write_{false};
    // GR2FORK PERF: sized to cover multi-stage unified descriptor bindings; a smaller cap lets
    // higher bindings bypass ShouldWriteDescriptor() and be re-emitted every draw (extra
    // push-descriptor CPU + __memmove in RADV). The tick/pipeline keying makes enlarging safe.
    static constexpr u32 DescCacheBindingsCap = 1024;

    // Indexed by dstBinding (binding.unified index). Fixed-capacity avoids resize/zero-fill on hot path.
    mutable std::array<DescCacheEntry, DescCacheBindingsCap> desc_cache{};

    void PrepareDescriptorDeltaCache(const Pipeline* pipeline);

    // GR2FORK: per-binding descriptor delta dedup. On push-descriptor
    // deployments force_write is the exception - it fires only for set-path
    // pipelines, GR2_NODESCDELTA=1, or a null pipeline - hence [[unlikely]].
    [[gnu::always_inline]] inline bool ShouldWriteDescriptor(
        u32 binding, vk::DescriptorType type, u64 a, u64 b, u64 c) {
        if (desc_cache_force_write_) [[unlikely]] {
            return true;
        }
        return ShouldWriteDescriptorImpl(binding, type, a, b, c);
    }
    bool ShouldWriteDescriptorImpl(u32 binding, vk::DescriptorType type, u64 a, u64 b, u64 c);


    // OPT: buffer_bindings removed - BindBuffers is now single-pass.
    struct ImageBindingInfo {
        VideoCore::ImageId image_id{};
        const VideoCore::TextureCache::ImageDesc* desc{}; // Cached base descriptor (info+view_info+type)
        s16 view_mip{-1};
        s16 view_slice{-1};
        // GR2FORK FIX: mip-fallback descriptor arrays (port of upstream #4075). DynamicIndex
        // images reserve one layout slot per mip level; all must be emitted or a lod>=1 dispatch
        // reads a null descriptor -> RADV device-lost (VUID-vkCmdDispatch-None-08114).
        u8 num_bindings{1};
        // GR2FORK FIX: mirror of ImageResource::is_written, carried independently of desc (the
        // garbage-T# bails leave desc null). The pipeline layout types slots eStorageImage from
        // the same flag, so descriptor typing and null-write-sink routing must follow it.
        bool is_written{false};
        bool single_mip{false};
    };
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;

    // GR2FORK PERF: hot-line entry layout. The table is 4096 x ~512B (~2MB, beyond any L2);
    // everything the hit path compares - valid, key, the 32B T#, a guest_address mirror equal to
    // desc.info.guest_address - fits in the first 64B, so a probe drags one line, not three.
    struct alignas(64) CachedImageDescEntry {
        // -------- hot probe line (bytes 0..0x35) --------
        u64 key{};
        AmdGpu::Image image{};
        // Mirror of desc.info.guest_address so the pcache-key computation
        // reads the line the probe just loaded instead of +0x180.
        u64 guest_address{};
        // GR2FORK PERF: bumped whenever `desc` below is rebuilt in place. Together with the
        // texture-cache registry generation this lets the find_image_pcache_ hit path skip
        // ValidateCachedFindImage (see PersistentFindImageCacheEntry). Lives on the hot line.
        u32 desc_gen{};
        bool valid{false};
        // -------- cold payload (next line onward) --------
        VideoCore::TextureCache::ImageDesc desc{};
    };
    // GR2FORK PERF: 256 direct-mapped slots collide heavily across pipelines/stages.
    // A larger cache cuts ImageDesc rebuilds (ImageInfo::UpdateSize etc.).
    std::array<CachedImageDescEntry, 4096> image_desc_cache_{};

    // No intra-call dedup cache: multi-stage shaders effectively never
    // bind the same image to multiple slots within one BindTextures call.

    // GR2FORK PERF: generation-stamped entries. Matching registry_gen + desc_gen proves the
    // stored result by determinism, skipping ValidateCachedFindImage - which subresource-view
    // entries (res.view_mip/view_slice >= 0) can never pass. Kill switch: GR2_NOPCACHEGEN=1.
    struct PersistentFindImageCacheEntry {
        u64 key{};
        const VideoCore::TextureCache::ImageDesc* base{};
        VideoCore::TextureCache::FindImageWithViewResult res{};
        u64 registry_gen{};
        u32 desc_gen{};
        bool valid{false};
    };
    // Cross-call cache to skip repeated FindImageWithView lock/page-table scans for stable descriptors.
    // GR2FORK PERF: the key folds in the texture VA so different
    // textures bound to the same shader binding get distinct slots.
    std::array<PersistentFindImageCacheEntry, 4096> find_image_pcache_{};

    // GR2FORK PERF: epoch-based validity for the BindTextures caches - entries are valid iff
    // their stamp matches the current epoch, avoiding a 640+ byte stamp-array memset per call.
    u32 bind_textures_epoch_{};
    std::array<u32, 128> find_texture_cache_stamp_{}; // avoids a local u8[128] zeroed every call
    struct FindTextureCacheEntry {
        VideoCore::ImageId image_id{};
        VideoCore::TextureCache::BindingType type{};
        VideoCore::ImageView* view{};
    };
    std::array<FindTextureCacheEntry, 128> find_texture_cache_{};

    // Render target identity cache. PrepareRenderState calls texture_cache.FindImage() (page
    // table walks + hash lookups) per CB/DB every draw, but render targets only change on render
    // pass switches; cache the results keyed on CB/DB addresses.
    struct RenderTargetCache {
        u64 hash{};
        std::array<VideoCore::ImageId, AmdGpu::NUM_COLOR_BUFFERS> cb_image_ids{};
        VideoCore::ImageId db_image_id{};
        // GR-Remastered validated fast path. `full_key` hashes every input the ImageInfo ctors +
        // active-gating consume (unlike `hash`, which omits format/samples/tile/view); a full_key
        // + registry_generation match implies FindImage returns identical ids, reused as-is.
        u64 full_key{};
        u64 registry_generation{};
        bool valid{};
    } rt_cache_{};

    // GR2FORK PERF: BeginRendering stamp-skip cache (~0.74% of GpuComm self-time). A hit needs
    // stamp + pipeline match, no armed clear (compute can arm CMASK without bumping the gfx
    // stamp), and unchanged layouts; cached loadOps are forced to eLoad to avoid re-clearing.
    struct BeginRenderingCacheEntry {
        VAddr meta_addr{};                   // CMASK or HTILE address; 0 = no metadata
        u32 slice{};
        VideoCore::ImageId image_id{};
        vk::ImageLayout expected_layout{};   // backing->state.layout at cache populate
    };
    struct BeginRenderingCache {
        bool valid{false};
        bool attachment_feedback_loop{false};
        bool has_db_attachment{false};
        u32 cb_count{};
        const GraphicsPipeline* pipeline{};
        u64 stamp{};
        // Scheduler tick at populate time. FreeImage slot reclamation runs via DeferOperation
        // only after a submit, and the tick advances every submission, so bounding validity to
        // the same tick guarantees no slot reuse between populate and hit.
        u64 tick{};
        // GR2FORK PERF: generation stamps taken at populate (refreshed after a successful
        // verify). Matching image-registry + meta-clear + layout/backing gens skips the
        // per-attachment probes otherwise run every draw. Kill switch: GR2_NOBRGEN=1.
        u64 registry_gen{};
        u64 meta_gen{};
        u64 layout_gen{};
        RenderState state{};                 // loadOps forced to eLoad, clearValues zeroed
        std::array<BeginRenderingCacheEntry, AmdGpu::NUM_COLOR_BUFFERS> cb_data{};
        BeginRenderingCacheEntry db_data{};
    } br_cache_{};

    // GR2FORK PERF: Config::isBeginRenderingCacheEnabled() is a getenv-once process constant
    // (GR2_NOBRCACHE; no config-file disable). Latched at construction to avoid a cross-TU call
    // + magic-static guard on every draw.
    const bool br_cache_enabled_;

    // Persistent build target for the BeginRendering slow path so it can return
    // `const RenderState&` (zero-copy on the br_cache_ hit path). The state is fully consumed
    // within the same draw before the next BeginRendering overwrites it - no lifetime escape.
    RenderState cur_render_state_{};

    bool fault_process_pending{};
    bool attachment_feedback_loop{};

    // PopPendingOperations batching: only check every 16th draw.
    mutable u32 draw_counter_{};

    // FilterDraw cache: skip re-checking when pipeline hasn't changed.
    mutable const GraphicsPipeline* last_filter_pipeline_{};
    mutable bool last_filter_result_{true};

    // OPT: Pipeline bind deduplication. Consecutive draws with the same pipeline
    // skip the redundant vkCmdBindPipeline call (significant driver overhead on RADV/ANV).
    // Keyed on tick (advances on every primary cmdbuf rotation).
    mutable vk::Pipeline last_bound_pipeline_{};
    mutable u64 last_bound_pipeline_tick_{};

    // GR2FORK PERF: persistent UpdateImage de-dup cache. UpdateImage takes a shared_lock even to
    // discover "nothing to do" (the common case); caching {image_id, tick} skips the lock for
    // images already checked this command buffer tick.
    struct UpdateImageCacheEntry {
        VideoCore::ImageId image_id{};
        u64 tick{};
    };
    std::array<UpdateImageCacheEntry, 256> update_image_cache_{};

    // Sampler fast path: GR2 reuses 4-8 unique samplers across hundreds of draws, so a 64-slot
    // direct-mapped cache (>90% hit rate) skips GetSampler's XXH3 + map lookup on hit.
    // GR2FORK PERF: keyed on the raw 2x u64 S# words, so a hit cannot false-match on a collision.
    struct SamplerCacheEntry {
        u64 raw0{};
        u64 raw1{};
        vk::Sampler sampler{};
        bool valid{false};
    };
    static constexpr u32 SamplerCacheSize = 64;
    std::array<SamplerCacheEntry, SamplerCacheSize> sampler_cache_{};

    // BundleAssembler intent-queue dispatcher; in async mode its jthread drains the queue as the
    // sole caller of vkCmd* on the primary cmdbuf. Declared after liverpool/scheduler: Push uses
    // liverpool->CaptureSnapshot() and reverse dtor order must destroy the assembler first.
    BundleAssembler bundle_assembler_;

    // Measured high-water marks: intent queue 195/256 entries, snapshot
    // pool 64/64.
};

} // namespace Vulkan
