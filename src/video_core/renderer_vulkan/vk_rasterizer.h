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

    // =========================================================================
    // Phase 1D-0 (Turn 2A): intent-driven dispatch entry points.
    //
    // Public Draw/DrawIndirect/DispatchDirect/DispatchIndirect become thin
    // intent-pushers that hand control to BundleAssembler. The actual
    // per-draw work — pipeline lookup, BindResources, BeginRendering,
    // descriptor build, bundle emission — runs out of these "DoXFromIntent"
    // methods, called from BundleAssembler::ProcessOne. In 2A the assembler
    // dispatches synchronously on the same thread; in 2B it dispatches on
    // the GpuAssembler jthread. The bodies are unchanged from the pre-Turn
    // 2A Draw/etc. — the only refactor is the call shape.
    // =========================================================================
    void DoDrawFromIntent(const DrawIntent& intent);
    void DoDrawIndirectFromIntent(const DrawIntent& intent);
    void DoDispatchDirectFromIntent(const DrawIntent& intent);
    void DoDispatchIndirectFromIntent(const DrawIntent& intent);

    // Phase 1D-pre-D: Rasterizer scheduler-toucher markers. The public
    // entry points below (ScopeMarkerBegin / End / Insert(Color),
    // FillBuffer, CopyBuffer, CpSync, Flush, Finish, OnSubmit) become thin
    // marker pushers; the bodies move here. Called by
    // BundleAssembler::ProcessOne in dispatch order.
    void DoFlushFromIntent(const DrawIntent& intent);
    void DoFinishFromIntent(const DrawIntent& intent);
    void DoCpSyncFromIntent(const DrawIntent& intent);
    void DoOnSubmitFromIntent(const DrawIntent& intent);
    void DoFillBufferFromIntent(const DrawIntent& intent);
    void DoCopyBufferFromIntent(const DrawIntent& intent);
    void DoScopeMarkerBeginFromIntent(const DrawIntent& intent);
    void DoScopeMarkerEndFromIntent(const DrawIntent& intent);
    void DoScopedMarkerInsertFromIntent(const DrawIntent& intent);

    // Phase 1D-pre-E: producer-side helper for Presenter scheduler-toucher
    // routing. The Presenter (which lives on its own thread distinct from
    // PM4 and from the future async assembler) wraps any work that touches
    // `draw_scheduler` in a closure and pushes a PresenterRecord intent.
    // The assembler invokes the closure when processing the intent and
    // the closure self-destructs.
    //
    // Phase 1D-1 (Phase G): returns the assigned packet_seq for use with
    // WaitForAssembler. BLOCKING wrappers (~Presenter, PrepareFrame,
    // PrepareBlankFrame(false), the BufferCache::ReadMemory SendCommand
    // outer lambda) pair this with WaitForAssembler(seq) so the producer
    // observes the body's side effects before returning. Fire-and-forget
    // callers discard the seq.
    //
    // Lifetime contract: `state` is heap-allocated and owned by the intent
    // until invoke_and_destroy runs; if the intent is dropped without
    // running (impossible by construction with the SPSC ring's FIFO
    // contract), `state` would leak — there is no cancellation path.
    template <typename F>
    [[nodiscard]] u32 PushPresenterRecord(F&& f) {
        using FT = std::decay_t<F>;
        // Heap-allocate the closure. ~60 allocs/sec at 60 fps + occasional
        // ~Presenter / blank-frame markers — far below any sensible
        // overhead bar; if profiling later flags it, swap in a per-Presenter
        // closure pool. Plain new/delete is the simplest correct thing.
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

    // Phase 1D-1 (Phase G): block until the assembler has processed the
    // intent identified by `seq` (the value returned from
    // PushPresenterRecord or BundleAssembler::Push). Used by BLOCKING
    // wrappers to observe completion before returning to their caller.
    // Must NOT be called from the assembler thread itself.
    void WaitForAssembler(u32 seq) {
        bundle_assembler_.WaitFor(seq);
    }

    // Phase 1D-pre-D: in-line ScopeMarker helpers. The public ScopeMarker
    // methods marker-route, but Rasterizer-internal callers (e.g. Resolve,
    // EliminateFastClear, DepthStencilCopy, the "FmaskDecompress" /
    // "PrimitiveTypeNone" inserts in FilterDraw) need the markers to
    // execute in-line so they bracket the surrounding data-plane work
    // synchronously. Internal callers invoke these directly; the
    // DoScopeMarker*FromIntent dispatchers also forward to them after
    // unpacking the intent.
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
    // Phase 1D-0b (Turn 2B-1): every helper that previously read
    // `liverpool->regs` directly now takes a `const AmdGpu::LiverpoolRegsSnapshot&`
    // parameter and reads from the snapshot. The snapshot mirrors `Regs`'s
    // field interface (designed in Turn 1 for exactly this), so the bodies
    // are unchanged from pre-2B-1 — only the data source switches.
    //
    // 2B-1: synchronous dispatch (assembler runs on the producer thread),
    // so the snapshot at intent-capture time matches `liverpool->regs` at
    // helper-execution time. Behavior is bit-identical.
    // 2B-2: assembler runs on its own thread; the snapshot is then the
    // *only* consistent reg view from the assembler's side, because
    // `liverpool->regs` may have moved on to the next packet.
    void PrepareRenderState(const GraphicsPipeline* pipeline,
                            const AmdGpu::LiverpoolRegsSnapshot& regs);
    RenderState BeginRendering(const GraphicsPipeline* pipeline,
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

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data,
                     const AmdGpu::LiverpoolRegsSnapshot& regs);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding,
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

    /// Bind pipeline with deduplication — skips vkCmdBindPipeline when already bound.
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
    // PERF(GR2FORK release_v2_5): generation counter for IsMapped's TLS
    // cache (definition in vk_rasterizer.cpp). Bumped under release order
    // by every MapMemory/UnmapMemory after they mutate mapped_ranges.
    // IsMapped's per-thread cache invalidates when this changes.
    std::atomic<u64> mapped_ranges_gen_{0};
    PipelineCache pipeline_cache;

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;
    Pipeline::DescriptorWrites set_writes;
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;

    // OPT#4: Descriptor delta cache (skip redundant push-descriptor writes).
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
    // PERF(GR2 v1.22): RADV descriptor-set path always re-allocates a fresh
    // VkDescriptorSet, so ShouldWriteDescriptor must always return true (no
    // prior state on the new set). Set in PrepareDescriptorDeltaCache; checked
    // by callers via the inline ShouldWriteDescriptor wrapper, which short-
    // circuits to `true` without touching the desc_cache. Eliminates the
    // ~10-30 dead cache lookups + writes per draw on RADV.
    mutable bool desc_cache_force_write_{false};
    // PERF(GR2): Cover multi-stage unified descriptor bindings.
    // The previous cap only covered ~single-stage bindings, so higher unified bindings bypassed
    // Rasterizer::ShouldWriteDescriptor() and were re-emitted every draw (extra push-descriptor
    // CPU + __memmove in RADV). This cache is keyed by command buffer tick + pipeline/layout and
    // epoch-invalidated, so enlarging the coverage is safe and avoids those uncached writes.
    static constexpr u32 DescCacheBindingsCap = 1024;

    // Indexed by dstBinding (binding.unified index). Fixed-capacity avoids resize/zero-fill on hot path.
    mutable std::array<DescCacheEntry, DescCacheBindingsCap> desc_cache{};
    // (ULTRA MAX) Removed OPT#5 guest-signature cache. It double-worked the hot path and could be unsafe.

    void PrepareDescriptorDeltaCache(const Pipeline* pipeline);

    // PERF(GR2 v1.22): Inline wrapper short-circuits the cache check on
    // RADV's descriptor-set path. The cold body
    // (ShouldWriteDescriptorImpl) handles push-descriptor cache hits.
    [[gnu::always_inline]] inline bool ShouldWriteDescriptor(
        u32 binding, vk::DescriptorType type, u64 a, u64 b, u64 c) {
        // RADV path: cache state is always stale (epoch bumps every draw).
        // Skip the cache lookup + comparison + writeback; the caller will
        // emit the descriptor unconditionally. This is the *common* case on
        // RADV deployments — likely on Z1E / SteamOS / Bazzite.
        // PERF(GR2FORK v1.27): the flag is set in
        // PrepareDescriptorDeltaCache (v1.18 hoisted) for every RADV
        // BindResources call. ShouldWriteDescriptor fires once per
        // binding (10-30 calls per BindResources). The hint pushes the
        // ShouldWriteDescriptorImpl call (push-descriptor cache lookup,
        // dead on RADV) into the function tail.
        if (desc_cache_force_write_) [[likely]] {
            return true;
        }
        return ShouldWriteDescriptorImpl(binding, type, a, b, c);
    }
    bool ShouldWriteDescriptorImpl(u32 binding, vk::DescriptorType type, u64 a, u64 b, u64 c);


    // OPT: buffer_bindings removed — BindBuffers is now single-pass.
    struct ImageBindingInfo {
        VideoCore::ImageId image_id{};
        const VideoCore::TextureCache::ImageDesc* desc{}; // Cached base descriptor (info+view_info+type)
        s16 view_mip{-1};
        s16 view_slice{-1};
        // FIX(GR2FORK): PORT(upstream #4075) mip-fallback descriptor arrays.
        // When the shader's ImageResource has MipStorageFallbackMode::DynamicIndex,
        // the descriptor set layout reserves N slots (one per mip level). The
        // rasterizer must emit N image views + N descriptor array entries
        // here, otherwise Index>=1 stays uninitialized and a compute dispatch
        // that samples lod=1 reads a null descriptor -> RADV device-lost.
        // Validation flags this as VUID-vkCmdDispatch-None-08114 on cs_img16.
        // num_bindings is 1 for the common case; >1 only for DynamicIndex images.
        u8 num_bindings{1};
    };
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;

    // OPT(v14.1 hotfix): BindTextures caches stored on the Rasterizer instance (not TLS).
    struct CachedImageDescEntry {
        u64 key{};
        AmdGpu::Image image{};
        VideoCore::TextureCache::ImageDesc desc{};
        bool valid{false};
    };
    // PERF(GR2): 256 direct-mapped slots collides heavily across pipelines/stages.
    // A larger cache cuts ImageDesc rebuilds (ImageInfo::UpdateSize etc.).
    std::array<CachedImageDescEntry, 4096> image_desc_cache_{};

    // PERF(GR2FORK v1.43): LocalFindImageCacheEntry / find_image_cache_
    // (intra-call dedup) removed — diagnostic measurement showed 0 hits
    // across millions of iterations. Multi-stage shaders effectively never
    // bind the same image to multiple slots within one BindTextures call.

    struct PersistentFindImageCacheEntry {
        u64 key{};
        const VideoCore::TextureCache::ImageDesc* base{};
        VideoCore::TextureCache::FindImageWithViewResult res{};
        bool valid{false};
    };
    // Cross-call cache to skip repeated FindImageWithView lock/page-table scans for stable descriptors.
    // PERF(GR2FORK v1.44): key now folds in the texture VA so different
    // textures bound to the same shader binding get distinct slots.
    // PERF(GR2FORK cache-audit): 16384 was TESTED and REVERTED to 4096. The 4×
    // bump did collapse the conflict (pkey 20.68%->7.40%, FindImageWithView
    // fires 25.58%->12.03%, hits 74%->88%) — so the miss WAS real conflict, not
    // compulsory. BUT bindtex cyc/stage stayed flat (~1940->~2061) and avg_cyc
    // nudged UP: the pcache only caches image_id resolution, while the bulk of
    // bindtex (per-iteration view lookup + descriptor construction) is NOT
    // gated by it. Hit-rate win, zero cost win, +480 KB & worse locality.
    // Keep 4096. (One-line re-bump to 16384 if a future fps read ever favors it.)
    std::array<PersistentFindImageCacheEntry, 4096> find_image_pcache_{};

    // PERF(GR2FORK v1.43 + cache-audit): the BindTextures intra-call caches are
    // gone. find_image_cache_(_stamp_) was culled in v1.43 (0 hits); the texture
    // view cache (find_texture_cache_ / _stamp_, bind_textures_epoch_,
    // FindTextureCacheEntry, the epoch counter) was culled in the cache-audit —
    // 0.00% hit over 38,839 lookups on the mip-fallback path. The one caller now
    // goes straight to texture_cache.FindTexture (behavior-identical at runtime).

    // =========================================================================
    // Render target identity cache
    // =========================================================================
    // PrepareRenderState calls texture_cache.FindImage() per CB/DB every draw.
    // FindImage involves page table walks + hash lookups. But render targets
    // rarely change between draws (they change on render pass switches).
    // Cache the FindImage results keyed on CB/DB addresses.
    struct RenderTargetCache {
        u64 hash{};
        std::array<VideoCore::ImageId, AmdGpu::NUM_COLOR_BUFFERS> cb_image_ids{};
        VideoCore::ImageId db_image_id{};
        bool valid{};
    } rt_cache_{};

    // =========================================================================
    // PERF(GR2FORK): BeginRendering stamp-skip cache (v3 §3.1).
    // =========================================================================
    // Rasterizer::BeginRendering self-time is ~0.74% of GpuComm — mostly the
    // inline state-building (per-attachment image-view lookups via
    // FindRenderTarget/FindDepthTarget, ImageView struct copies, layout
    // computation, ComputeHash). The function inputs are derived from
    // liverpool->regs (CB/DB descriptors) + cb_descs/db_desc (set by
    // PrepareRenderState, also stamp-driven). Stamp-equal calls would build
    // bit-identical state EXCEPT for the consumed-on-read CMASK clear flag
    // (TouchMeta) and externally-driven image layout transitions.
    //
    // Trap (memory entry #26): caching state with loadOp=eClear and replaying
    // it would re-clear the RT every draw → magenta blobs. We avoid this by
    // storing the cached state with all loadOps FORCED to eLoad and clearValues
    // zeroed. The first call (which actually consumes the clear flag) returns
    // the as-built state with eClear; only subsequent stamp-matched calls hit
    // the cache and return the eLoad version.
    //
    // Hit predicate (all must hold):
    //   - stamp match          (regs unchanged → same CB/DB/key/feedback_loop)
    //   - pipeline match       (defensive — implied by stamp but cheap to check)
    //   - no clear armed       (per-attachment IsMetaCleared returns false; a
    //                           compute clear shader can arm CMASK between
    //                           gfx draws WITHOUT bumping the gfx stamp)
    //   - layouts unchanged    (per-attachment image's current layout still
    //                           matches what the cached state declares; a
    //                           compute dispatch using one of these images as
    //                           an input would have transitioned it away)
    //
    // On hit: set attachment_feedback_loop to the cached value, return the
    // eLoad-version of the state. Skips all FindRenderTarget/FindDepthTarget
    // calls, all per-attachment struct field writes, ComputeHash, and the
    // image transit (because layout already matches).
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
        // Scheduler tick at populate time. Tick advances every submission;
        // image slots freed via FreeImage are reclaimed by DeferOperation
        // which only fires after the GPU catches up — i.e., after a submit.
        // Bounding cache validity to the same tick guarantees no slot reuse
        // could have happened between populate and hit.
        u64 tick{};
        RenderState state{};                 // loadOps forced to eLoad, clearValues zeroed
        std::array<BeginRenderingCacheEntry, AmdGpu::NUM_COLOR_BUFFERS> cb_data{};
        BeginRenderingCacheEntry db_data{};
    } br_cache_{};

    // =========================================================================
    // Per-stage binding skip cache
    // =========================================================================
    // When the same pipeline draws consecutive geometry with identical user_data
    // (same textures, same buffers — only transforms/push constants differ),
    // we can skip the entire BindBuffers + BindTextures iteration (~500 lines
    // of T# reads, cache lookups, descriptor construction) since
    // ShouldWriteDescriptor would filter out all writes anyway.
    //
    // Keyed on {pipeline pointer, user_data hash, cmdbuf tick} per stage.
    struct StageBindingCacheEntry {
        u64 pgm_hash{};
        // PERF(GR2FORK v1.58): replaced `u64 ud_hash` with the raw user_data
        // snapshot. The previous design hashed user_data with XXH3 on every
        // stamp-mismatch verification AND every slow-path cache update —
        // ~10-20 ns per stage on 64-byte input × 3-6 stages × thousands of
        // draws/frame. memcmp of 64 bytes on Zen 4 is one AVX-512 load +
        // compare (~3-5 ns), and it's strictly more reliable (no hash
        // collisions, however astronomical). The valid-byte count is
        // determined by the stage's pgm_hash (the shader's user_data
        // layout is fixed for a given program), so the pgm_hash check
        // upstream covers any size change without an explicit length
        // field. The +56 bytes per stage entry is negligible (× 6 stages
        // = ~336 bytes total in binding_skip_cache_).
        std::array<u32, Shader::ShaderParams::NumShaderUserData> cached_user_data{};
    };
    struct BindingSkipState {
        const Pipeline* pipeline{};
        u64 cmdbuf_tick{};
        // PERF(GR2): gfx pipeline stamp at the time the per-stage ud_hashes
        // were stored. When the stamp matches on a subsequent call, every
        // stage's user_data is provably bit-identical (any SetShReg path
        // bumps the stamp under the same lock that mutates user_data), so
        // the per-stage XXH3 loop below can be skipped entirely. Cuts the
        // bulk of the 0.44% XXH3_64bits cost on the BindResources hot path.
        u64 stamp{};
        bool uses_dma{};
        std::array<StageBindingCacheEntry, 6> stages{}; // MaxShaderStages
        bool valid{};
        // PERF(GR2): cached PushData snapshot from the last successful
        // BindResources call. When stamp_match holds AND the cache fast
        // path runs, the per-stage Info::PushUd loop produces a result
        // bit-identical to this snapshot (PushUd reads only stage->user_data
        // and writes ud_regs[]; user_data is unchanged when stamp matches).
        // Replaying the snapshot via single memcpy skips the per-stage
        // PushUd loop (~80-200 ns per draw on typical 4-stage pipelines)
        // plus the upstream MakeUserData regs reads (~10-20 ns).
        Shader::PushData cached_push_data{};
        bool push_data_valid{};
    } binding_skip_cache_{};

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

    // PERF(GR2 v17): Persistent UpdateImage de-dup cache.
    // UpdateImage takes a shared_lock even to discover "nothing to do" (most common case).
    // By caching {image_id, tick} across draws within the same command buffer tick,
    // we skip redundant lock acquire/release for images that were already checked this tick.
    struct UpdateImageCacheEntry {
        VideoCore::ImageId image_id{};
        u64 tick{};
    };
    std::array<UpdateImageCacheEntry, 256> update_image_cache_{};

    // =========================================================================
    // Sampler Resolution Fast-Path Cache
    // =========================================================================
    // GetSampler() calls XXH3_64bits on the 64-byte Sampler struct + does an
    // unordered_map lookup every time. GR2 typically has 4-8 unique samplers
    // that repeat across hundreds of draws. A 64-slot direct-mapped cache here
    // avoids both the hash and the map lookup on hit (>90% hit rate in steady
    // state). Samplers are immutable Vulkan objects so caching handles is safe.
    // Cache miss falls through to the existing GetSampler() path.
    struct SamplerCacheEntry {
        u64 hash{};
        vk::Sampler sampler{};
        bool valid{false};
    };
    static constexpr u32 SamplerCacheSize = 64;
    std::array<SamplerCacheEntry, SamplerCacheSize> sampler_cache_{};

    // =========================================================================
    // Y-1: BundleAssembler intent-queue dispatcher.
    //
    // Public Draw/DrawIndirect/DispatchDirect/DispatchIndirect forward into
    // bundle_assembler_.Push(intent), which captures a reg snapshot. Under
    // Y-1.ASYNC the dedicated GpuAssembler jthread drains the queue and
    // is the sole writer of `draw_scheduler` (i.e., the sole caller of
    // vkCmd* on the primary cmdbuf).
    //
    // Construction order: must be initialized AFTER liverpool/scheduler
    // because Push calls liverpool->CaptureSnapshot() and the DoXFromIntent
    // paths use scheduler. Declared here so the destructor runs before
    // liverpool's destructor (member dtor order is reverse-declaration;
    // bundle_assembler_ is declared after liverpool member).
    // =========================================================================
    BundleAssembler bundle_assembler_;

    // =========================================================================
    // Y-1: HWM instrumentation. Tracks intent-queue depth and snapshot-pool
    // depth. Static (process-lifetime) so std::at_quick_exit can flush
    // counters from a free function — required because shadPS4's normal
    // termination path is std::quick_exit.
    // =========================================================================
    static std::atomic<u32> max_intent_queue_depth_;
    static std::atomic<u32> max_snapshot_pool_depth_;
    static std::atomic<u64> total_intents_;

    // Static HWM bumpers — called from BundleAssembler::Push and
    // Liverpool::CaptureSnapshot. Both are on the producer (PM4) thread,
    // but the atomics handle any future producer arrangement.
public:
    static void BumpIntentQueueHwm(u32 in_flight) noexcept {
        AtomicMaxUpdate(max_intent_queue_depth_, in_flight);
        total_intents_.fetch_add(1, std::memory_order_relaxed);
    }
    static void BumpSnapshotPoolHwm(u32 in_flight) noexcept {
        AtomicMaxUpdate(max_snapshot_pool_depth_, in_flight);
    }
private:

    static void LogHwm() noexcept;

    static void AtomicMaxUpdate(std::atomic<u32>& slot, u32 v) noexcept {
        u32 cur = slot.load(std::memory_order_relaxed);
        while (v > cur && !slot.compare_exchange_weak(cur, v,
                                                       std::memory_order_relaxed,
                                                       std::memory_order_relaxed)) {
        }
    }
};

} // namespace Vulkan
