// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/assert.h"
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

    // Scopes a guest-copy hold to one packet run: the caller yields to guest
    // threads between runs and the submit loop sleeps, so the hold must not
    // outlive this object.
    class PacketRunGuard {
    public:
        explicit PacketRunGuard(Rasterizer* r) : r_{r} {
            if (r_) {
                r_->BeginPacketRun();
            }
        }
        ~PacketRunGuard() {
            if (r_) {
                r_->EndPacketRun();
            }
        }
        PacketRunGuard(const PacketRunGuard&) = delete;
        PacketRunGuard& operator=(const PacketRunGuard&) = delete;

    private:
        Rasterizer* r_;
    };
    /// Arms every read watcher a written bind left pending. GPU command thread.
    void DrainPendingReadArms(VideoCore::ReadArmSite site) {
        if (deferred_read_arm_) {
            buffer_cache.DrainPendingReadArms(site);
        }
    }
    static void PreSubmitThunk(void* self) {
        static_cast<Rasterizer*>(self)->DrainPendingReadArms(VideoCore::ReadArmSite::Submit);
    }
    void BeginPacketRun();
    void EndPacketRun();
    /// The command drain runs fault-download hops inline; none may run under the hold.
    void DropCopyHoldForCommands();

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

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed);
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
    /// The three compute shortcuts, out of the graphics bind path's lines.
    SHAD_NO_INLINE bool TakeComputeShortcut(const Pipeline* pipeline);

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
    bool batch_copy_lock_{};  // settings snapshot, read at draw rate
    // Interval flush: a readback fence otherwise waits on the whole body
    // recorded since the last submit. Boot-latched; 0 = off.
    u32 flush_draw_interval_{};
    u32 draws_since_flush_{};
    u64 flush_tick_{};
    u64 interval_flushes_{};
    // Snapshot of the framework's FindImage counters at the last report; the
    // per-window line prints the deltas (Forced mode never resets them).
    VideoCore::Skipcache::CacheCounters findimg_last_{};
    VideoCore::Skipcache::CacheCounters br_last_{};
    VideoCore::Skipcache::CacheCounters rt_last_{};
    // How often the register stamp the render target memo keys on moves
    // between probes: its miss floor, and the sizing input for a wider key.
    u64 rt_stamp_last_{};
    u64 rt_stamp_moves_{};
    // Last drained framework counters of the descriptor delta cache; the
    // DESCDELTA line prints the window's miss lanes from the difference.
    VideoCore::Skipcache::CacheCounters desc_last_{};
    VideoCore::Skipcache::CacheCounters dynstate_last_{};
    u64 dyn_stamp_last_{};
    u64 gfx_stamp_last_{};
    void MaybeIntervalFlush();
    bool elide_findbuffer_{};
    bool bind_prefetch_{};
    // One guest-copy shared hold per packet run (guest_copy_hold_segment).
    // The hold may cover GPU waits, never a wait on a guest thread; every
    // path that can block on one drops it first.
    std::optional<Core::MemoryManager::GuestCopyScope> run_copy_hold_;
    bool in_packet_run_{};
    bool segment_copy_hold_{};
    u64 hold_arms_{};
    u64 hold_draws_covered_{};
    u64 hold_drops_run_{};
    u64 hold_drops_flush_{};
    u64 hold_drops_wait_{};
    u64 hold_drops_cmd_{};
    u64 hold_compiles_{};
    void ArmCopyHold();
    void DropCopyHold(u64& counter);
    static void PreCompileThunk(void* self) {
        auto* r = static_cast<Rasterizer*>(self);
        r->DropCopyHold(r->hold_compiles_);
    }
    u64 bindpf_img_{};
    u64 bindpf_backing_{};
    bool bind_noop_{};
    u64 bindnoop_hits_{};
    u64 bindnoop_slow_{};
    u64 tsc_hz_{}; // measured once at construction; the estimator sleeps ~101ms

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
    SHAD_NO_INLINE void BindingSkipProbeBody(const Pipeline* pipeline);
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

    // UpdateDynamicState memo: every value the five updaters compute is a pure
    // function of the stamped graphics registers, the pipeline write masks, the
    // feedback-loop flag and immutable device caps, and each setter marks a
    // dirty bit exactly when it writes one - so an unchanged key marks nothing
    // and Commit emits nothing. dyn_gen covers the three Invalidate() sites,
    // which re-arm every bit without changing a value.
    struct DynStateMemo {
        u64 reg_stamp{};
        u64 dyn_gen{};
        u64 pipe_gen{};
        const GraphicsPipeline* pipeline{}; // compared, never dereferenced
        u32 flags{}; // 0 invalid; bit0 valid, bit1 feedback loop, bit2 indexed
        // The only pipeline field the updaters read; the stamp-lane keying
        // compares it instead of the pipeline identity.
        std::array<vk::ColorComponentFlags, AmdGpu::NUM_COLOR_BUFFERS> write_masks{};
    };
    bool DynMemoProbe(const GraphicsPipeline* pipeline, u32 flags, u64 reg_stamp, u64 dyn_gen,
                      u64 pipe_gen);
    DynStateMemo dyn_memo_{};
    bool dyn_memo_enabled_{};
    bool dyn_class_stamp_{};
    bool deferred_read_arm_{};

    // Pipeline bind dedup: {handle, bind point} last issued on this cmdbuf.
    void BindPipelineDedup(vk::PipelineBindPoint point, vk::Pipeline handle);
    std::array<vk::Pipeline, 2> last_bound_pipeline_{};
    std::array<u64, 2> last_bound_pipeline_gen_{};
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
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    Pipeline::DescriptorWrites set_writes;
    Pipeline::BufferBarriers buffer_barriers;
    // 120 bytes: unaligned it straddles three cache lines, so every draw's
    // rebuild touches a third line for eight bytes of it. The value-init is a
    // no-op today and the base case for a prefix-clear memo later.
    alignas(64) Shader::PushData push_data{};

    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;
    // Constructed and destroyed two or three times per draw as a local.
    boost::container::static_vector<u32, Shader::NUM_IMAGES> image_descriptor_array_sizes;
    bool fault_process_pending{};
    bool attachment_feedback_loop{};

    // Pass 1 hands pass 2 the guest base, the clamped size and the id it
    // resolved; the rest of the V# is dead after pass 1, so a new pass-2
    // consumer must re-read the sharp. The guest/special discriminant rides in
    // a caller-held bitmask: with FindBuffer elision a guest binding can carry
    // a null id, and a clamped guest size can legitimately be zero, so neither
    // field can double as the sentinel. Declared last so no hot member moves.
    struct BufferBindingInfo {
        VAddr base;
        u64 size;
        u32 id;
    };
    static_assert(sizeof(BufferBindingInfo) == 24);
    static_assert(Shader::NUM_BUFFERS <= 64, "the guest mask is one u64");
    std::array<BufferBindingInfo, Shader::NUM_BUFFERS> buffer_bindings{};
    std::array<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos{};
    u32 buffer_info_n_{};

    // Buffer bind scratch census. Five stores per stage, none per binding:
    // the per-stage binding count is the multiplier every estimate of this
    // path's cost rests on and it has never been measured. Declared last so
    // no hot member's offset moves.
    u64 bindscratch_calls_{};
    u64 bindscratch_binds_{};
    u64 bindscratch_bindmax_{};
    u64 bindscratch_infos_{};
    u64 bindscratch_infomax_{};
};

} // namespace Vulkan
