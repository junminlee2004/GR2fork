// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>

#include "common/config.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_primitive.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace tracy {
class VkCtxScope;
}

namespace Vulkan {

class Instance;

// =============================================================================
// PERF(GR2FORK R1, 2026-06-10): RenderState copy diet — kill switch.
//
// GR2_NORSTRIM=1 restores the verbatim pre-R1 behavior at EVERY R1 site as a
// coherent set: full-struct copies at all three hot sites (Rasterizer slow-
// path reset, br_cache_ eLoad populate, Scheduler::BeginRendering commit) AND
// the full-struct memcmp tail of operator==. The set must flip together: the
// trimmed copies leave stale bytes in inactive attachment slots, which only
// the active-field equality is licensed to ignore — full memcmp over trimmed
// copies would return false on genuinely-equal states (conservative but
// silently defeats the BeginRendering dedup), and trimmed equality over full
// copies is merely redundant. One process-constant switch, read once,
// keeps the pairing impossible to misconfigure. Default ON. FORCED ON — no
// longer config-controlled; disable only via GR2_NORSTRIM=1. First call is in
// the Rasterizer ctor, post-config-load.
// =============================================================================
inline bool RStrimEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NORSTRIM");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

// GR2FORK L5 toggle: depth-bounds bitwise compare in SetDepthBounds (see the
// comment at the call site). Off = the pre-L5 float compare verbatim, i.e.
// NaN re-arms the dirty bit every draw. FORCED ON — no longer config-controlled;
// disable only via GR2_NODEPTHBITCMP=1. Read once, first call post-config-load.
inline bool DepthBoundsBitCmpEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NODEPTHBITCMP");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

struct RenderState {
    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    vk::RenderingAttachmentInfo depth_attachment;
    vk::RenderingAttachmentInfo stencil_attachment;
    u32 num_color_attachments;
    u32 num_layers;
    bool has_depth;
    bool has_stencil;
    u32 width;
    u32 height;
    // OPT: Lightweight hash for fast equality rejection.
    // Updated by ComputeHash() after all fields are set.
    u64 state_hash{};

    RenderState() {
        std::memset(this, 0, sizeof(*this));
        color_attachments.fill(vk::RenderingAttachmentInfo{});
        depth_attachment = vk::RenderingAttachmentInfo{};
        stencil_attachment = vk::RenderingAttachmentInfo{};
        num_layers = 1;
    }

    /// Call after all fields are populated to enable fast equality checks.
    void ComputeHash() noexcept {
        // Hash the fields most likely to differ between draws.
        // This covers ~98% of state changes without touching the full 700 bytes.
        auto mix = [](u64 h, u64 v) noexcept {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        };
        u64 h = 0x84222325cbf29ce4ULL;
        h = mix(h, (static_cast<u64>(width) << 32) | static_cast<u64>(height));
        h = mix(h, (static_cast<u64>(num_color_attachments) << 32) | static_cast<u64>(num_layers));
        h = mix(h, (static_cast<u64>(has_depth) << 1) | static_cast<u64>(has_stencil));
        // Hash attachment imageViews and loadOps (most frequently changing parts).
        for (u32 i = 0; i < num_color_attachments; ++i) {
            h = mix(h, reinterpret_cast<u64>(
                           static_cast<VkImageView>(color_attachments[i].imageView)));
            h = mix(h, static_cast<u64>(static_cast<u32>(color_attachments[i].loadOp)));
        }
        if (has_depth) {
            h = mix(h, reinterpret_cast<u64>(
                           static_cast<VkImageView>(depth_attachment.imageView)));
            h = mix(h, static_cast<u64>(static_cast<u32>(depth_attachment.loadOp)));
        }
        if (has_stencil) {
            h = mix(h, reinterpret_cast<u64>(
                           static_cast<VkImageView>(stencil_attachment.imageView)));
        }
        state_hash = h;
    }

    bool operator==(const RenderState& other) const noexcept {
        // OPT: Fast reject on hash before expensive memcmp.
        if (state_hash != other.state_hash) [[likely]] {
            return false;
        }
        // PERF(GR2FORK R1): active-field confirm. The trimmed copies (see
        // RStrimEnabled block above) leave stale bytes in inactive
        // attachment slots, so the confirm step must only read what the
        // hash hashed and the driver reads: the scalar header, the active
        // color span, and depth/stencil when flagged. Under GR2_NORSTRIM=1
        // copies are full again and the verbatim full-struct memcmp is
        // restored.
        if (RStrimEnabled()) [[likely]] {
            return ActiveFieldsEqual(other);
        }
        return std::memcmp(this, &other, sizeof(RenderState)) == 0;
    }

    // PERF(GR2FORK R1): equality over exactly the fields ComputeHash covers
    // plus the full bytes of every ACTIVE attachment. Mirrors ComputeHash's
    // active-only iteration directly above; never reads color slots
    // >= num_color_attachments, nor depth/stencil when their flags are off.
    bool ActiveFieldsEqual(const RenderState& other) const noexcept {
        if (num_color_attachments != other.num_color_attachments ||
            num_layers != other.num_layers || has_depth != other.has_depth ||
            has_stencil != other.has_stencil || width != other.width ||
            height != other.height) {
            return false;
        }
        if (num_color_attachments != 0 &&
            std::memcmp(color_attachments.data(), other.color_attachments.data(),
                        num_color_attachments * sizeof(vk::RenderingAttachmentInfo)) != 0) {
            return false;
        }
        if (has_depth && std::memcmp(&depth_attachment, &other.depth_attachment,
                                     sizeof(depth_attachment)) != 0) {
            return false;
        }
        if (has_stencil && std::memcmp(&stencil_attachment, &other.stencil_attachment,
                                       sizeof(stencil_attachment)) != 0) {
            return false;
        }
        return true;
    }

    // PERF(GR2FORK R1): trimmed copy — active color span + depth/stencil
    // when flagged + the scalar tail (including state_hash). Inactive color
    // slots in the destination keep stale bytes; every reader (the driver
    // via colorAttachmentCount, ComputeHash, ActiveFieldsEqual) is
    // active-gated, so they are never observed. Only call when
    // RStrimEnabled(); callers keep the verbatim full assignment otherwise.
    void CopyActiveFrom(const RenderState& src) noexcept {
        if (src.num_color_attachments != 0) {
            std::memcpy(color_attachments.data(), src.color_attachments.data(),
                        src.num_color_attachments * sizeof(vk::RenderingAttachmentInfo));
        }
        if (src.has_depth) {
            std::memcpy(&depth_attachment, &src.depth_attachment, sizeof(depth_attachment));
        }
        if (src.has_stencil) {
            std::memcpy(&stencil_attachment, &src.stencil_attachment,
                        sizeof(stencil_attachment));
        }
        num_color_attachments = src.num_color_attachments;
        num_layers = src.num_layers;
        has_depth = src.has_depth;
        has_stencil = src.has_stencil;
        width = src.width;
        height = src.height;
        state_hash = src.state_hash;
    }

    // PERF(GR2FORK R1): in-place replacement for `*this = RenderState{}` on
    // the BeginRendering slow path. Reproduces the constructor's observable
    // state for everything the slow path can leave unwritten: color slots
    // [0, num_cbs) get the value-initialized vk::RenderingAttachmentInfo
    // (correct sType, null view) because the cb loop's `continue` for
    // masked-out buffers leaves them as-is and the driver still reads them
    // within colorAttachmentCount; the scalar tail is zeroed (num_layers=1
    // per the ctor). depth/stencil attachments are intentionally untouched:
    // they are wholesale-assigned whenever has_depth/has_stencil are set and
    // never read otherwise. Only call when RStrimEnabled().
    void ResetActivePrefix(u32 num_cbs) noexcept {
        const u32 n = std::min<u32>(num_cbs, static_cast<u32>(color_attachments.size()));
        for (u32 i = 0; i < n; ++i) {
            color_attachments[i] = vk::RenderingAttachmentInfo{};
        }
        num_color_attachments = 0;
        num_layers = 1;
        has_depth = false;
        has_stencil = false;
        width = 0;
        height = 0;
        state_hash = 0;
    }
};

struct SubmitInfo {
    std::array<vk::Semaphore, 3> wait_semas;
    std::array<u64, 3> wait_ticks;
    std::array<vk::Semaphore, 3> signal_semas;
    std::array<u64, 3> signal_ticks;
    vk::Fence fence;
    u32 num_wait_semas;
    u32 num_signal_semas;

    void AddWait(vk::Semaphore semaphore, u64 tick = 1) {
        wait_semas[num_wait_semas] = semaphore;
        wait_ticks[num_wait_semas++] = tick;
    }

    void AddSignal(vk::Semaphore semaphore, u64 tick = 1) {
        signal_semas[num_signal_semas] = semaphore;
        signal_ticks[num_signal_semas++] = tick;
    }

    void AddSignal(vk::Fence fence) {
        this->fence = fence;
    }
};

using Viewports = boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS>;
using Scissors = boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS>;
using ColorWriteMasks = std::array<vk::ColorComponentFlags, AmdGpu::NUM_COLOR_BUFFERS>;
struct StencilOps {
    vk::StencilOp fail_op{};
    vk::StencilOp pass_op{};
    vk::StencilOp depth_fail_op{};
    vk::CompareOp compare_op{};

    bool operator==(const StencilOps& other) const {
        return fail_op == other.fail_op && pass_op == other.pass_op &&
               depth_fail_op == other.depth_fail_op && compare_op == other.compare_op;
    }
};
struct DynamicState {
    struct {
        bool viewports : 1;
        bool scissors : 1;

        bool depth_test_enabled : 1;
        bool depth_write_enabled : 1;
        bool depth_compare_op : 1;

        bool depth_bounds_test_enabled : 1;
        bool depth_bounds : 1;

        bool depth_bias_enabled : 1;
        bool depth_bias : 1;

        bool stencil_test_enabled : 1;
        bool stencil_front_ops : 1;
        bool stencil_front_reference : 1;
        bool stencil_front_write_mask : 1;
        bool stencil_front_compare_mask : 1;
        bool stencil_back_ops : 1;
        bool stencil_back_reference : 1;
        bool stencil_back_write_mask : 1;
        bool stencil_back_compare_mask : 1;

        bool primitive_restart_enable : 1;
        bool rasterizer_discard_enable : 1;
        bool cull_mode : 1;
        bool front_face : 1;

        bool blend_constants : 1;
        bool color_write_masks : 1;
        bool line_width : 1;
        bool feedback_loop_enabled : 1;
    } dirty_state{};

    Viewports viewports{};
    Scissors scissors{};

    bool depth_test_enabled{};
    bool depth_write_enabled{};
    vk::CompareOp depth_compare_op{};

    bool depth_bounds_test_enabled{};
    float depth_bounds_min{};
    float depth_bounds_max{};

    bool depth_bias_enabled{};
    float depth_bias_constant{};
    float depth_bias_clamp{};
    float depth_bias_slope{};

    bool stencil_test_enabled{};
    StencilOps stencil_front_ops{};
    u32 stencil_front_reference{};
    u32 stencil_front_write_mask{};
    u32 stencil_front_compare_mask{};
    StencilOps stencil_back_ops{};
    u32 stencil_back_reference{};
    u32 stencil_back_write_mask{};
    u32 stencil_back_compare_mask{};

    bool primitive_restart_enable{};
    bool rasterizer_discard_enable{};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};

    std::array<float, 4> blend_constants{};
    ColorWriteMasks color_write_masks{};
    float line_width{};
    bool feedback_loop_enabled{};

    /// Commits the dynamic state to the provided command buffer.
    void Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf);

    /// Invalidates all dynamic state to be flushed into the next command buffer.
    void Invalidate() {
        std::memset(&dirty_state, 0xFF, sizeof(dirty_state));
    }

    /// Clear dirty flags without issuing any commands (for threaded recording).
    void ClearDirty() {
        std::memset(&dirty_state, 0, sizeof(dirty_state));
    }

    void SetViewports(const Viewports& viewports_) {
        if (!std::ranges::equal(viewports, viewports_)) {
            viewports = viewports_;
            dirty_state.viewports = true;
        }
    }

    void SetScissors(const Scissors& scissors_) {
        if (!std::ranges::equal(scissors, scissors_)) {
            scissors = scissors_;
            dirty_state.scissors = true;
        }
    }

    void SetDepthTestEnabled(const bool enabled) {
        if (depth_test_enabled != enabled) {
            depth_test_enabled = enabled;
            dirty_state.depth_test_enabled = true;
        }
    }

    void SetDepthWriteEnabled(const bool enabled) {
        if (depth_write_enabled != enabled) {
            depth_write_enabled = enabled;
            dirty_state.depth_write_enabled = true;
        }
    }

    void SetDepthCompareOp(const vk::CompareOp compare_op) {
        if (depth_compare_op != compare_op) {
            depth_compare_op = compare_op;
            dirty_state.depth_compare_op = true;
        }
    }

    void SetDepthBoundsTestEnabled(const bool enabled) {
        if (depth_bounds_test_enabled != enabled) {
            depth_bounds_test_enabled = enabled;
            dirty_state.depth_bounds_test_enabled = true;
        }
    }

    void SetDepthBounds(const float min, const float max) {
        // GR2FORK L5 (permanent; from the #47 dynamic_state sub-probe): GR2
        // leaves the depth-bounds regs non-finite, and NaN != NaN is true —
        // so a float compare here re-armed dirty_state.depth_bounds on EVERY
        // draw and Commit re-emitted vkCmdSetDepthBounds each time for an
        // unchanged (garbage) value. Compare bit patterns instead: bitwise-
        // identical re-emits (NaN included) are redundant; genuine value
        // changes still mark dirty. Finite behavior is unchanged except
        // -0.0f vs 0.0f now counts as a change — strictly more conservative.
        // Kill switch GR2_NODEPTHBITCMP=1
        // restores the v4.4 float compare verbatim.
        const bool changed =
            DepthBoundsBitCmpEnabled()
                ? (std::bit_cast<u32>(depth_bounds_min) != std::bit_cast<u32>(min) ||
                   std::bit_cast<u32>(depth_bounds_max) != std::bit_cast<u32>(max))
                : (depth_bounds_min != min || depth_bounds_max != max);
        if (changed) {
            depth_bounds_min = min;
            depth_bounds_max = max;
            dirty_state.depth_bounds = true;
        }
    }

    void SetDepthBiasEnabled(const bool enabled) {
        if (depth_bias_enabled != enabled) {
            depth_bias_enabled = enabled;
            dirty_state.depth_bias_enabled = true;
        }
    }

    void SetDepthBias(const float constant, const float clamp, const float slope) {
        if (depth_bias_constant != constant || depth_bias_clamp != clamp ||
            depth_bias_slope != slope) {
            depth_bias_constant = constant;
            depth_bias_clamp = clamp;
            depth_bias_slope = slope;
            dirty_state.depth_bias = true;
        }
    }

    void SetStencilTestEnabled(const bool enabled) {
        if (stencil_test_enabled != enabled) {
            stencil_test_enabled = enabled;
            dirty_state.stencil_test_enabled = true;
        }
    }

    void SetStencilOps(const StencilOps& front_ops, const StencilOps& back_ops) {
        if (stencil_front_ops != front_ops) {
            stencil_front_ops = front_ops;
            dirty_state.stencil_front_ops = true;
        }
        if (stencil_back_ops != back_ops) {
            stencil_back_ops = back_ops;
            dirty_state.stencil_back_ops = true;
        }
    }

    void SetStencilReferences(const u32 front_reference, const u32 back_reference) {
        if (stencil_front_reference != front_reference) {
            stencil_front_reference = front_reference;
            dirty_state.stencil_front_reference = true;
        }
        if (stencil_back_reference != back_reference) {
            stencil_back_reference = back_reference;
            dirty_state.stencil_back_reference = true;
        }
    }

    void SetStencilWriteMasks(const u32 front_write_mask, const u32 back_write_mask) {
        if (stencil_front_write_mask != front_write_mask) {
            stencil_front_write_mask = front_write_mask;
            dirty_state.stencil_front_write_mask = true;
        }
        if (stencil_back_write_mask != back_write_mask) {
            stencil_back_write_mask = back_write_mask;
            dirty_state.stencil_back_write_mask = true;
        }
    }

    void SetStencilCompareMasks(const u32 front_compare_mask, const u32 back_compare_mask) {
        if (stencil_front_compare_mask != front_compare_mask) {
            stencil_front_compare_mask = front_compare_mask;
            dirty_state.stencil_front_compare_mask = true;
        }
        if (stencil_back_compare_mask != back_compare_mask) {
            stencil_back_compare_mask = back_compare_mask;
            dirty_state.stencil_back_compare_mask = true;
        }
    }

    void SetPrimitiveRestartEnabled(const bool enabled) {
        if (primitive_restart_enable != enabled) {
            primitive_restart_enable = enabled;
            dirty_state.primitive_restart_enable = true;
        }
    }

    void SetCullMode(const vk::CullModeFlags cull_mode_) {
        if (cull_mode != cull_mode_) {
            cull_mode = cull_mode_;
            dirty_state.cull_mode = true;
        }
    }

    void SetFrontFace(const vk::FrontFace front_face_) {
        if (front_face != front_face_) {
            front_face = front_face_;
            dirty_state.front_face = true;
        }
    }

    void SetBlendConstants(const std::array<float, 4> blend_constants_) {
        if (blend_constants != blend_constants_) {
            blend_constants = blend_constants_;
            dirty_state.blend_constants = true;
        }
    }

    void SetRasterizerDiscardEnabled(const bool enabled) {
        if (rasterizer_discard_enable != enabled) {
            rasterizer_discard_enable = enabled;
            dirty_state.rasterizer_discard_enable = true;
        }
    }

    void SetColorWriteMasks(const ColorWriteMasks& color_write_masks_) {
        if (!std::ranges::equal(color_write_masks, color_write_masks_)) {
            color_write_masks = color_write_masks_;
            dirty_state.color_write_masks = true;
        }
    }

    void SetLineWidth(const float width) {
        if (line_width != width) {
            line_width = width;
            dirty_state.line_width = true;
        }
    }

    void SetAttachmentFeedbackLoopEnabled(const bool enabled) {
        if (feedback_loop_enabled != enabled) {
            feedback_loop_enabled = enabled;
            dirty_state.feedback_loop_enabled = true;
        }
    }
};

class Scheduler {
public:
    explicit Scheduler(const Instance& instance);
    ~Scheduler();

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush(SubmitInfo& info);

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush();

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish();

    /// Waits for the given tick to trigger on the GPU.
    void Wait(u64 tick);

    /// Attempts to execute operations whose tick the GPU has caught up with.
    void PopPendingOperations();

    /// Starts a new rendering scope with provided state.
    void BeginRendering(const RenderState& new_state);

    /// Ends current rendering scope.
    void EndRendering();

    /// Returns the current render state.
    const RenderState& GetRenderState() const {
        return render_state;
    }

    /// Returns the current pipeline dynamic state tracking.
    DynamicState& GetDynamicState() {
        return dynamic_state;
    }

    /// Returns the current command buffer.
    /// DEPRECATED ALIAS for PrimaryCommandBuffer.
    /// All recording — resource ops, barriers, rendering passes, draws,
    /// compute dispatches — goes to the same primary command buffer.
    vk::CommandBuffer CommandBuffer() const {
        return current_cmdbuf;
    }

    /// Returns the primary cmdbuf. All vkCmd* recording (resource ops,
    /// barriers, BeginRendering/EndRendering, transfers, image transitions,
    /// compute dispatches, draws inside a rendering pass) targets this
    /// single primary cmdbuf. Secondary command buffers are permanently
    /// banned in this codebase — see HANDOFF_y_series_assembler_only §2.
    vk::CommandBuffer PrimaryCommandBuffer() const {
        return current_cmdbuf;
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore.CurrentTick();
    }

    // =========================================================================
    // Cross-pipeline cmdbuf state generation counters.
    //
    // The per-Pipeline push-descriptor / push-constant / bind-descriptor-set
    // caches are scoped to ONE Pipeline instance, but the cmdbuf state they
    // track is SHARED. When Pipeline X has cached "I pushed sig S to cmdbuf C
    // at tick T with layout L_X" and Pipeline Y subsequently pushes its own
    // descriptors to set 0 (with layout L_Y), the cmdbuf set 0 shadow is now
    // L_Y's data, not L_X's. X's cache will still cache-hit on the next call
    // (cmdbuf, tick, sig all unchanged from X's view) and skip the push,
    // leaving the draw to fire with set 0 holding L_Y data while bindPipeline
    // re-binds X (layout L_X). RADV reports VUID-08600 ("descriptor set 0
    // not compatible with the layout used in pushDescriptorSetKHR") and may
    // DEVICE_LOST.
    //
    // Fix: each push/bind that touches cmdbuf state bumps a per-bind-point
    // generation counter. Pipeline caches store the gen at the time of
    // their update; on read, the gen must still match. Any cross-pipeline
    // push/bind invalidates everyone's caches automatically.
    //
    // Reset to 0 in AllocateWorkerCommandBuffers — fresh cmdbuf, no state.
    [[nodiscard]] u64 GetPushDescGen(vk::PipelineBindPoint bp) const noexcept {
        return bp == vk::PipelineBindPoint::eCompute ? cmdbuf_gen_push_desc_compute_
                                                     : cmdbuf_gen_push_desc_gfx_;
    }
    void BumpPushDescGen(vk::PipelineBindPoint bp) noexcept {
        if (bp == vk::PipelineBindPoint::eCompute) {
            ++cmdbuf_gen_push_desc_compute_;
        } else {
            ++cmdbuf_gen_push_desc_gfx_;
        }
    }
    [[nodiscard]] u64 GetPushConstGen(vk::PipelineBindPoint bp) const noexcept {
        return bp == vk::PipelineBindPoint::eCompute ? cmdbuf_gen_push_const_compute_
                                                     : cmdbuf_gen_push_const_gfx_;
    }
    void BumpPushConstGen(vk::PipelineBindPoint bp) noexcept {
        if (bp == vk::PipelineBindPoint::eCompute) {
            ++cmdbuf_gen_push_const_compute_;
        } else {
            ++cmdbuf_gen_push_const_gfx_;
        }
    }
    [[nodiscard]] u64 GetBoundDescGen(vk::PipelineBindPoint bp) const noexcept {
        return bp == vk::PipelineBindPoint::eCompute ? cmdbuf_gen_bound_desc_compute_
                                                     : cmdbuf_gen_bound_desc_gfx_;
    }
    void BumpBoundDescGen(vk::PipelineBindPoint bp) noexcept {
        if (bp == vk::PipelineBindPoint::eCompute) {
            ++cmdbuf_gen_bound_desc_compute_;
        } else {
            ++cmdbuf_gen_bound_desc_gfx_;
        }
    }
    void ResetCmdbufStateGens() noexcept {
        cmdbuf_gen_push_desc_gfx_ = 0;
        cmdbuf_gen_push_desc_compute_ = 0;
        cmdbuf_gen_push_const_gfx_ = 0;
        cmdbuf_gen_push_const_compute_ = 0;
        cmdbuf_gen_bound_desc_gfx_ = 0;
        cmdbuf_gen_bound_desc_compute_ = 0;
    }

    // =========================================================================
    /// Returns true when a tick has been triggered by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) noexcept {
        if (master_semaphore.IsFree(tick)) {
            return true;
        }
        master_semaphore.Refresh();
        return master_semaphore.IsFree(tick);
    }

    /// Returns the master timeline semaphore.
    [[nodiscard]] MasterSemaphore* GetMasterSemaphore() noexcept {
        return &master_semaphore;
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Will be run when submitting or calling PopPendingOperations.
    void DeferOperation(Common::UniqueFunction<void>&& func) {
        pending_ops.emplace(std::move(func), CurrentTick());
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Runs as soon as possible in another thread.
    void DeferPriorityOperation(Common::UniqueFunction<void>&& func) {
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops.emplace(std::move(func), CurrentTick());
        }
        priority_pending_ops_cv.notify_one();
    }

    static std::mutex submit_mutex;

    // FIX(GR2FORK): VK_ERROR_DEVICE_LOST diagnostics. When a queue.submit()
    // returns DEVICE_LOST (NVIDIA Windows TDR is the dominant trigger on this
    // fork — see hang_watchdog "STALL DETECTED" 5s before), the device is
    // unrecoverable and the only path forward is process termination via
    // ASSERT. The crash log itself only ever said "Device lost during submit"
    // with no GPU-side detail, because vk_instance.cpp:340 enables
    // VK_EXT_device_fault but no caller queried it. This helper closes that
    // gap: on DEVICE_LOST we run the two-pass vkGetDeviceFaultInfoEXT query
    // (counts pass + info pass), log every VkDeviceFaultAddressInfoEXT and
    // VkDeviceFaultVendorInfoEXT entry, and only THEN fall through to the
    // ASSERT. Result: future DEVICE_LOST crashes ship the fault address,
    // fault type (Read / Write / Execution / None), driver-side description,
    // and vendor-specific fault code/data alongside the existing crash dump.
    //
    // Static / does not touch any Scheduler instance state, so it is also
    // callable from ComputeScheduler::Submit on its symmetric DEVICE_LOST
    // assert. `context` is a short literal ("graphics submit" or
    // "compute submit") inserted into the log lines so the side that lost
    // the device is unambiguous.
    static void DumpDeviceFaultInfo(const Instance& instance, const char* context) noexcept;

private:
    void AllocateWorkerCommandBuffers();

    void SubmitExecution(SubmitInfo& info);

    void PriorityPendingOpsThread(std::stop_token stoken);

private:
    const Instance& instance;
    MasterSemaphore master_semaphore;
    CommandPool command_pool;
    DynamicState dynamic_state;
    vk::CommandBuffer current_cmdbuf;
    std::condition_variable_any event_cv;
    struct PendingOp {
        Common::UniqueFunction<void> callback;
        u64 gpu_tick;
    };
    std::queue<PendingOp> pending_ops;
    std::queue<PendingOp> priority_pending_ops;
    std::mutex priority_pending_ops_mutex;
    std::condition_variable_any priority_pending_ops_cv;
    std::jthread priority_pending_ops_thread;
    RenderState render_state;
    bool is_rendering = false;
    tracy::VkCtxScope* profiler_scope{};

    // --- Cross-pipeline cmdbuf state generation counters ---
    // Bumped by Pipeline::BindResources whenever it pushes/binds; read by
    // Pipeline cache hit checks to detect that another Pipeline has touched
    // the same cmdbuf state since the last cache update.
    // See public GetPushDescGen/BumpPushDescGen for rationale.
    u64 cmdbuf_gen_push_desc_gfx_{0};
    u64 cmdbuf_gen_push_desc_compute_{0};
    u64 cmdbuf_gen_push_const_gfx_{0};
    u64 cmdbuf_gen_push_const_compute_{0};
    u64 cmdbuf_gen_bound_desc_gfx_{0};
    u64 cmdbuf_gen_bound_desc_compute_{0};
};

} // namespace Vulkan
