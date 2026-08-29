// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <bit>
#include <memory>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/skipcache/skipcache.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

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

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

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
    void ProcessDownloadImages();
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
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();
    bool FilterDrawSlow();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);

    void ResetBindings() {
        for (auto& image_id : bound_images) {
            texture_cache.GetImage(image_id).binding = {};
        }
        bound_images.clear();
    }

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    // =========================================================================
    // BeginRendering skip cache (adaptive framework; see SKIPCACHE_DESIGN.md
    // and the verify contract in skipcache.h). The snapshot is always a
    // clear-free state: populate is refused whenever the freshly built state
    // carries any clear flag, which structurally subsumes both the
    // consumed-on-read CMASK trap and the level-triggered register-clear trap.
    // Guards are read-only and run BEFORE the slow path can consume meta.
    // =========================================================================
    struct BrAttachmentGuard {
        VAddr meta_addr{}; // 0 = no metadata
        u32 slice{};
        VideoCore::ImageId image_id{};
        u64 image_uid{};
        const void* backing{};
        vk::ImageLayout expected_layout{};
        vk::AccessFlags2 expected_access{};
        u32 base_level{}, base_layer{}, num_levels{}, num_layers{};
    };
    struct BeginRenderingCache {
        bool valid{};
        bool attachment_feedback_loop{};
        bool has_db{};
        u32 cb_count{};
        const GraphicsPipeline* pipeline{}; // compared, never dereferenced
        VideoCore::Skipcache::DrawToken token{};
        u64 meta_gen{};
        u64 layout_gen{};
        RenderState state{}; // clear-free by populate refusal
        std::array<BrAttachmentGuard, AmdGpu::NUM_COLOR_BUFFERS> cb_guard{};
        BrAttachmentGuard db_guard{};
    };
    bool BrProbe(const VideoCore::Skipcache::DrawToken& token, const GraphicsPipeline* pipeline);
    bool BrGuardAttachment(const BrAttachmentGuard& g, VideoCore::Skipcache::CacheCounters& ctr);
    RenderState BrReplay(const GraphicsPipeline* pipeline);
    void BrVerify(const RenderState& fresh, const VideoCore::Skipcache::DrawToken& token);
    void BrPopulate(const RenderState& fresh, const VideoCore::Skipcache::DrawToken& token,
                    const GraphicsPipeline* pipeline);
    static void BrInvalidateThunk(void* self) {
        static_cast<Rasterizer*>(self)->br_cache_.valid = false;
    }
    BeginRenderingCache br_cache_{};
    bool br_readback_gate_{}; // readbackLinearImages snapshot: cache off when set
    u64 tsc_hz_{};            // measured once at construction; the estimator sleeps ~101ms

    // BindingSkip LEARNING probe state (observation snapshot, not a cache).
    struct BindingSkipProbeState {
        struct StageSnap {
            u64 pgm_hash{};
            std::array<u32, 16> user_data{};
        };
        const Pipeline* pipeline{};
        u64 tick{};
        u32 num_stages{};
        std::array<StageSnap, 8> stages{};
        bool valid{};
    };
    void BindingSkipProbe(const Pipeline* pipeline);
    BindingSkipProbeState bs_probe_{};

    // PrepareRenderState memo: with the CB/DB registers, extent hints and
    // texture-cache structure unchanged, the render-target resolution from the
    // previous draw replays: cb_descs/db_desc still hold the identically
    // constructed (and identically rebased) descriptors, so only the image
    // ids, is_target marks and bound_images entries need re-establishing.
    struct PrepareRtMemo {
        bool valid{};
        u64 reg_stamp{};
        u64 tex_gen{};
        u64 pipe_gen{};
        const GraphicsPipeline* pipeline{}; // compared, never dereferenced
        u32 cb_count{};
        std::array<VideoCore::ImageId, AmdGpu::NUM_COLOR_BUFFERS> cb_id{};
        std::array<u64, AmdGpu::NUM_COLOR_BUFFERS> cb_uid{};
        VideoCore::ImageId db_id{};
        u64 db_uid{};
    };
    bool RtMemoProbe(const GraphicsPipeline* pipeline, u64 reg_stamp, u64 tex_gen, u64 pipe_gen);
    void RtMemoReplay();
    void RtMemoVerifyPopulate(bool would_hit, const GraphicsPipeline* pipeline, u64 reg_stamp,
                              u64 tex_gen, u64 pipe_gen);
    PrepareRtMemo rt_memo_{};

    // =========================================================================
    // Global descriptor set cache (DESCSET) draw-path state. The per-draw key
    // is folded incrementally at the descriptor stores themselves - the values
    // are already in registers there - rather than re-hashing the assembled
    // arrays, which would add a second full content pass to the hottest thread.
    // =========================================================================
    struct DescEpochs {
        u64 view{}, sampler{}, buffer{};
    };
    /// Sampled at the TOP of BindResources and re-checked after the gather: a handle destroyed
    /// mid-gather and recycled onto another object must produce a miss, never a valid-looking
    /// stale entry.
    DescEpochs SampleDescEpochs();

    static constexpr u64 kDescFoldMul = 0x9E3779B97F4A7C15ull;
    void FoldDesc(u64 v) {
        desc_fp_ = std::rotl((desc_fp_ ^ v) * kDescFoldMul, 29);
    }
    /// Mandatory fold routing: every descriptor store in BindBuffers/BindTextures goes through
    /// these, so a missed fold site is impossible by construction.
    void PushBufferInfo(vk::Buffer b, u64 off, u64 range) {
        buffer_infos.emplace_back(b, off, range);
        if (desc_fp_active_) {
            FoldDesc(std::bit_cast<u64>(b));
            FoldDesc(off);
            FoldDesc(range);
        }
    }
    void PushImageInfo(vk::Sampler s, vk::ImageView v, vk::ImageLayout l) {
        image_infos.emplace_back(s, v, l);
        if (desc_fp_active_) {
            FoldDesc(std::bit_cast<u64>(s));
            FoldDesc(std::bit_cast<u64>(v));
            FoldDesc(static_cast<u64>(l));
        }
    }
    /// The per-write shape is NOT structurally fixed by the layout class: the invalid/rejected-T#
    /// paths emit a default-constructed desc.type, so the type must be part of the key.
    void FoldWriteShape(u32 dst_binding, vk::DescriptorType t, u32 count) {
        if (desc_fp_active_) {
            FoldDesc((u64(dst_binding) << 32) | (u64(count) << 8) | u64(u32(t) & 0xFFu));
        }
    }

    u32 desc_cache_mode_{}; // latched in the ctor, never read from the setting per draw
    bool desc_fp_active_{};
    u64 desc_fp_{};
    DescEpochs desc_epochs_pre_{};
    DescSetProbe desc_probe_{};

    // =========================================================================
    // Stream-copy memo (Flatbuf half from mode 3, ClipPlanes half from mode 2). StreamBuffer::Copy
    // always Maps a fresh,
    // monotonically advancing ring region with no dedup, so the Flatbuf and ClipPlanes
    // descriptors change every draw and their pipelines could never hit a content-keyed cache.
    // The memo reuses the previous (buffer, offset) when the payload bytes are identical AND the
    // region is provably still ours.
    //
    // BOTH GUARDS ARE LOAD BEARING. `tick` pins the memo to the command buffer that recorded the
    // original copy - a wrap that could reuse the memoized region forces a flush and therefore a
    // tick change - and `wrap_gen` is the direct belt-and-braces check on the same hazard. Drop
    // either and a later copy can overwrite the memoized bytes, leaving the draw reading another
    // shader's constants.
    // =========================================================================
    struct StreamCopyMemo {
        u64 pgm_hash{};
        u64 tick{};
        u64 wrap_gen{};
        u32 size{};
        vk::Buffer buf{};
        u64 offset{};
        bool valid{};
        std::array<u8, 2048> bytes{};
    };
    /// Returns true and fills out_buf/out_off when `m` still describes a live copy of the first
    /// `size` bytes at `src`. `pgm_hash` is compared as part of the key; pass 0 for keyless slots.
    bool StreamMemoProbe(const StreamCopyMemo& m, u64 pgm_hash, const void* src, u32 size,
                         const VideoCore::StreamBuffer& ring, vk::Buffer& out_buf, u64& out_off);
    void StreamMemoRecord(StreamCopyMemo& m, u64 pgm_hash, const void* src, u32 size,
                          const VideoCore::StreamBuffer& ring, vk::Buffer buf, u64 offset);

    // ~19KB, so it lives behind a pointer allocated only when a memo mode is
    // active: modes 0-2 must not stretch the Rasterizer's hot-member layout.
    struct StreamMemoStore {
        std::array<StreamCopyMemo, 8> flat{}; // index = pgm_hash & 7
        StreamCopyMemo clip{};                // single slot, 96-byte payload
    };
    std::unique_ptr<StreamMemoStore> stream_memo_;
    bool desc_flat_memo_{}; // latched: desc_cache_mode_ >= 3
    /// Latched: desc_cache_mode_ >= 2. The ClipPlanes half of the memo runs one mode earlier than
    /// the Flatbuf half on purpose - see the note in the constructor.
    bool desc_clip_memo_{};
    u64 stream_memo_probes_{}; // reported on the DESCSET line, reset with it
    u64 stream_memo_hits_{};

    // Pipeline bind dedup: {handle, bind point} last issued on this cmdbuf.
    void BindPipelineDedup(vk::PipelineBindPoint point, vk::Pipeline handle);
    std::array<vk::Pipeline, 2> last_bound_pipeline_{};
    u64 last_bound_tick_{};
    u64 filter_true_stamp_{};

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
    // Generation for IsMapped's per-thread interval cache.
    std::atomic<u64> mapped_ranges_gen_{0};
    PipelineCache pipeline_cache;

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;

    using BufferBindingInfo = std::tuple<VideoCore::BufferId, AmdGpu::Buffer, u64>;
    boost::container::static_vector<BufferBindingInfo, Shader::NUM_BUFFERS> buffer_bindings;
    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;
    bool fault_process_pending{};
    bool attachment_feedback_loop{};
};

} // namespace Vulkan
