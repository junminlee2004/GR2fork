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

// GR2FORK PERF: RenderState copy-diet kill switch, read once post-config-load. GR2_NORSTRIM=1
// restores full-struct copies at the three trimmed sites AND operator=='s full memcmp tail as one
// set: trimmed copies leave stale bytes in inactive slots only active-field equality may ignore.
inline bool RStrimEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NORSTRIM");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

// GR2FORK toggle: depth-bounds bitwise compare in SetDepthBounds (see the call-site comment).
// Forced on; disable only via GR2_NODEPTHBITCMP=1. Read once, first call post-config-load.
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
        // GR2FORK PERF: active-field confirm. Trimmed copies leave stale bytes in inactive
        // attachment slots, so the confirm reads only what the hash hashed and the driver reads:
        // scalar header, active color span, flagged depth/stencil. GR2_NORSTRIM=1 restores memcmp.
        if (RStrimEnabled()) [[likely]] {
            return ActiveFieldsEqual(other);
        }
        return std::memcmp(this, &other, sizeof(RenderState)) == 0;
    }

    // GR2FORK PERF: equality over exactly the fields ComputeHash covers plus the full bytes of
    // every active attachment; never reads inactive color slots or unflagged depth/stencil.
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

    // GR2FORK PERF: trimmed copy - active color span + flagged depth/stencil + scalar tail
    // (including state_hash). Inactive destination slots keep stale bytes; every reader is
    // active-gated, so never observed. RStrimEnabled() only; otherwise keep the full assignment.
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

    // GR2FORK PERF: in-place `*this = RenderState{}` for the BeginRendering slow path, called only
    // under RStrimEnabled(). Scalar tail zeroed (num_layers=1); depth/stencil set when flagged;
    // color slots [0, num_cbs) are value-initialized; the driver reads inside colorAttachmentCount.
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
    // GR2FORK FIX: callers declare the pipeline stage each wait must block (default eAllCommands).
    // A fixed 2-entry stage array under-sizes the 3 wait slots (latent OOB read) and lands frame-
    // ready on eColorAttachmentOutput, which does not order ImGui's fragment-stage sampling.
    std::array<vk::PipelineStageFlags, 3> wait_masks;
    std::array<vk::Semaphore, 3> signal_semas;
    std::array<u64, 3> signal_ticks;
    vk::Fence fence;
    u32 num_wait_semas;
    u32 num_signal_semas;

    void AddWait(vk::Semaphore semaphore, u64 tick = 1,
                 vk::PipelineStageFlags stage_mask = vk::PipelineStageFlagBits::eAllCommands) {
        wait_semas[num_wait_semas] = semaphore;
        wait_masks[num_wait_semas] = stage_mask;
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
        // GR2FORK PERF: GR2 leaves the depth-bounds regs non-finite; NaN != NaN makes a float
        // compare re-arm the dirty bit and re-emit vkCmdSetDepthBounds every draw. Compare bit
        // patterns (-0.0f vs 0.0f now counts as a change). Kill switch: GR2_NODEPTHBITCMP=1.
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
    /// All recording - resource ops, barriers, rendering passes, draws,
    /// compute dispatches - goes to the same primary command buffer.
    vk::CommandBuffer CommandBuffer() const {
        return current_cmdbuf;
    }

    /// Returns the primary cmdbuf. All vkCmd* recording (resource ops,
    /// barriers, BeginRendering/EndRendering, transfers, image transitions,
    /// compute dispatches, draws inside a rendering pass) targets this
    /// single primary cmdbuf. Secondary command buffers are permanently
    /// banned in this codebase.
    vk::CommandBuffer PrimaryCommandBuffer() const {
        return current_cmdbuf;
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore.CurrentTick();
    }

    // Cross-pipeline cmdbuf state generation counters. Per-Pipeline push/bind caches track shared
    // cmdbuf state; a stale hit would draw another pipeline's set 0 data under this layout (RADV
    // VUID-08600, possible DEVICE_LOST), so every push/bind bumps a gen caches must match on read.
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

    // GR2FORK FIX: DEVICE_LOST diagnostics - on queue.submit() DEVICE_LOST (NVIDIA Windows TDR is
    // the dominant trigger), run the two-pass vkGetDeviceFaultInfoEXT query and log every address/
    // vendor fault record before the ASSERT. Static; ComputeScheduler::Submit calls it too.
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
    // Bumped by Pipeline::BindResources on every push/bind; Pipeline cache-hit checks read them to
    // detect another Pipeline touching the same cmdbuf state. See GetPushDescGen for rationale.
    u64 cmdbuf_gen_push_desc_gfx_{0};
    u64 cmdbuf_gen_push_desc_compute_{0};
    u64 cmdbuf_gen_push_const_gfx_{0};
    u64 cmdbuf_gen_push_const_compute_{0};
    u64 cmdbuf_gen_bound_desc_gfx_{0};
    u64 cmdbuf_gen_bound_desc_compute_{0};
};

} // namespace Vulkan
