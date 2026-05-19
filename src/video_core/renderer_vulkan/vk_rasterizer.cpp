// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/config.h"
#include "common/debug.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/gpucomm_metrics.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"


#include <array>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <xxhash.h>

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif


namespace Vulkan {


// =============================================================================
// Y-1: BundleAssembler intent-queue infrastructure. Secondary cmdbufs and
// the recorder thread are permanently banned (HANDOFF §2). The data plane
// runs on the BundleAssembler jthread and writes vkCmd* directly to the
// primary cmdbuf.
// =============================================================================

// Y-1: HWM static members. Tracks intent-queue depth and snapshot-pool depth.
std::atomic<u32> Rasterizer::max_intent_queue_depth_{0};
std::atomic<u32> Rasterizer::max_snapshot_pool_depth_{0};
std::atomic<u64> Rasterizer::total_intents_{0};

void Rasterizer::LogHwm() noexcept {
    LOG_INFO(Render_Vulkan,
             "[GR2FORK Y-1] intent-queue HWM (this run): "
             "intent_queue_hwm={}/{} snapshot_pool_hwm={}/{} total_intents={}",
             max_intent_queue_depth_.load(std::memory_order_relaxed),
             BundleAssembler::kQueueSize,
             max_snapshot_pool_depth_.load(std::memory_order_relaxed),
             AmdGpu::LiverpoolRegsSnapshotPool::kNumSlots,
             total_intents_.load(std::memory_order_relaxed));
}


static void MakeUserData(Shader::PushData& push_data,
                         const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // PERF(v7): Zero only buf_offsets (40 bytes) with a single memset instead of
    // value-initializing the full ~120-byte PushData every draw. ud_regs are
    // always overwritten by Info::PushUd for every register the shader consumes.
    std::memset(push_data.buf_offsets.data(), 0, push_data.buf_offsets.size());

    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool},
      bundle_assembler_{*this, liverpool_} {
    LOG_INFO(Render_Vulkan,
             "[GR2FORK Y-1.ASYNC] async flip: BundleAssembler runs on "
             "dedicated jthread; BLOCKING wrappers paired with WaitFor; "
             "assembler is sole writer of draw_scheduler; primary-CB "
             "direct recording (no recorder, no secondaries)");

    // HWM verification fix: shadPS4's normal termination path is
    // std::quick_exit, which per C++ spec skips every non-trivial
    // destructor — including ~Rasterizer. Without this hook, the HWM log
    // line never fires on a clean shutdown.
    static bool s_at_quick_exit_registered = false;
    if (!s_at_quick_exit_registered) {
        s_at_quick_exit_registered = true;
        std::at_quick_exit(&Rasterizer::LogHwm);
    }

    if (!Config::nullGpu()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    // Phase 1D-pre-F: BufferCache needs a back-ref so its ReadMemory's
    // outer SendCommand lambda can route through PushPresenterRecord +
    // WaitForAssembler. Set after bundle_assembler_ is alive.
    buffer_cache.SetRasterizer(this);
}

Rasterizer::~Rasterizer() {
    // Y-1: BundleAssembler's jthread is joined automatically by its
    // destructor (declared last in vk_rasterizer.h, destructed first).
    // Normal-shutdown HWM log. Mirror of the at_quick_exit fallback.
    LogHwm();
}

void Rasterizer::CpSync() {
    // Phase 1D-pre-D: marker-route. Body moves to DoCpSyncFromIntent.
    DrawIntent intent;
    intent.type = DrawIntent::Type::CpSync;
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoCpSyncFromIntent(const DrawIntent& intent) {
    (void)intent;  // CpSync carries no payload.
    scheduler.EndRendering();
    auto cmdbuf = scheduler.PrimaryCommandBuffer();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlags{}, ib_barrier, {}, {});
}

bool Rasterizer::FilterDraw(const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // PERF(GR2FORK v1.16): Each of the four special-mode branches below is a
    // rare guest-driver-managed pass (FCE, FmaskDecompress, Resolve,
    // primitive-type=None). The dominant exit is the trailing `return true`
    // for normal triangle-list draws. Hint each rare-mode branch so they sit
    // in the cold tail and the hot exit is straight-line.
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) [[unlikely]] {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear(regs);
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) [[unlikely]] {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        // Phase 1D-pre-D: internal marker → inline helper. Marker-routing
        // would defer this past the surrounding FilterDraw return path.
        DoScopedMarkerInsertInline("FmaskDecompress", 0, /*from_guest=*/false);
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) [[unlikely]] {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve(regs);
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) [[unlikely]] {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        // Phase 1D-pre-D: internal marker → inline helper.
        DoScopedMarkerInsertInline("PrimitiveTypeNone", 0, /*from_guest=*/false);
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) [[unlikely]] {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy, regs);
        return false;
    }

    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline,
                                    const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    if (regs.color_control.degamma_enable) [[unlikely]] {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // PERF(GR2FORK v1.15): Hoist bit_width(key.mrt_mask) once. The value is
    // loop-invariant (key is a const ref to the pipeline's stable key) and
    // is consumed by all three loops below. Compilers usually already do
    // this LICM, but materializing it here costs nothing and guarantees the
    // single-evaluation in worst case.
    const s32 num_cbs = std::bit_width(key.mrt_mask);

    // =========================================================================
    // Render target identity hash: skip FindImage when CB/DB addresses unchanged.
    // FindImage is the most expensive per-draw call (~page table walk + hash lookup).
    // Render targets change on pass switches, not between draws within a pass.
    // =========================================================================
    auto mix = [](u64 h, u64 v) noexcept -> u64 {
        return h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    };
    u64 rt_hash = 0x84222325cbf29ce4ULL;
    // PERF(GR2FORK v1.22): combine the mrt_mask + skip_cb_binding scalars
    // into one packed mix call instead of two. mrt_mask uses at most 8
    // bits (NUM_COLOR_BUFFERS=8), skip_cb_binding is a single bool —
    // packing into bits [0..7] + bit [8] never overflows. Saves one mix
    // iteration of serial dependency on rt_hash per PrepareRenderState
    // call (every draw).
    rt_hash = mix(rt_hash, static_cast<u64>(key.mrt_mask) |
                            (skip_cb_binding ? (1ULL << 8) : 0ULL));
    for (s32 cb = 0; cb < num_cbs; ++cb) {
        // PERF(GR2FORK v1.22): the [[likely]] mirror of the slow-path
        // [[unlikely]] check below — for typical contiguous mrt_mask the
        // cb is active and contributes to rt_hash.
        if (!skip_cb_binding && regs.color_buffers[cb] &&
            regs.color_target_mask.GetMask(cb) &&
            (key.mrt_mask & (1 << cb))) [[likely]] {
            rt_hash = mix(rt_hash, regs.color_buffers[cb].Address());
            // Phase 1D-pre-C: read CB extent hint from the snapshot.
            rt_hash = mix(rt_hash, regs.last_cb_extent[cb].raw);
        }
    }
    // PERF(GR2FORK v1.22): most graphics draws have depth or stencil
    // testing enabled — the depth/stencil-disabled case (no Z-buffer at
    // all) is the minority configuration in 3D titles.
    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) [[likely]] {
        rt_hash = mix(rt_hash, regs.depth_buffer.DepthAddress());
        // Phase 1D-pre-C: read DB extent hint from the snapshot.
        rt_hash = mix(rt_hash, regs.last_db_extent.raw);
        rt_hash = mix(rt_hash, regs.depth_htile_data_base.GetAddress());
    }

    // PERF(GR2FORK v1.15): Hint the rt_cache hit path. Render targets change
    // on render-pass switches, not within a pass — so consecutive draws
    // overwhelmingly hit this branch. The slow path falls through naturally.
    if (rt_cache_.valid && rt_cache_.hash == rt_hash) [[likely]] {
        // Fast path: render targets unchanged. Reuse cached image_ids.
        // Must still add to bound_images and re-mark is_target (cleared by ResetBindings).
        for (s32 cb = 0; cb < num_cbs; ++cb) {
            auto& [image_id, desc] = cb_descs[cb];
            image_id = rt_cache_.cb_image_ids[cb];
            // PERF(GR2FORK v1.27): mirrors the slow-path skip-cb [[unlikely]]
            // — for contiguous mrt_mask, every cb in [0, num_cbs) is active
            // and carries an image_id. Null entries appear only for sparse
            // mrt_mask / cb-disabled / target-mask-zero, all uncommon.
            if (!image_id) [[unlikely]] {
                continue;
            }
            bound_images.emplace_back(image_id);
            texture_cache.GetImage(image_id).binding.is_target = 1u;
        }
        {
            auto& [image_id, desc] = db_desc;
            image_id = rt_cache_.db_image_id;
            // PERF(GR2FORK v1.27): if rt_hash matched, the cached db
            // image_id is whatever the slow path stored last time; for
            // 3D draws (the dominant case) that's a real id, mirroring
            // the v1.22 [[likely]] on depth/stencil enable above.
            if (image_id) [[likely]] {
                bound_images.emplace_back(image_id);
                texture_cache.GetImage(image_id).binding.is_target = 1u;
            }
        }
        return;
    }

    // Full path: resolve render targets via FindImage.
    for (s32 cb = 0; cb < num_cbs; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        // PERF(GR2FORK v1.22): num_cbs = bit_width(mrt_mask), so cb is in
        // [0, highest set bit + 1). For typical contiguous mrt_mask
        // (single-target / sequential MRT draws — the dominant case),
        // every cb in this range is active. The skip body fires only
        // for sparse mrt_mask, color_control.mode == Disable (skip_cb_binding),
        // unconfigured col_buf, or zero target_mask — all uncommon at
        // steady state.
        if (skip_cb_binding || !col_buf || !target_mask ||
            (key.mrt_mask & (1 << cb)) == 0) [[unlikely]] {
            image_id = {};
            rt_cache_.cb_image_ids[cb] = {};
            continue;
        }
        const auto& hint = regs.last_cb_extent[cb];
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
        rt_cache_.cb_image_ids[cb] = image_id;
    }

    // PERF(GR2FORK v1.27): mirrors the v1.22 [[likely]] on the rt_hash
    // compute above — most 3D draws enable depth or stencil testing; the
    // disabled-Z fall-through is the minority configuration.
    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) [[likely]] {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = regs.last_db_extent;
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
        rt_cache_.db_image_id = image_id;
    } else {
        db_desc.first = {};
        rt_cache_.db_image_id = {};
    }

    rt_cache_.hash = rt_hash;
    rt_cache_.valid = true;
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::LiverpoolRegsSnapshot& regs, const Shader::Info& info,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear(const AmdGpu::LiverpoolRegsSnapshot& regs) {
    auto& col_buf = regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, regs.last_cb_extent[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    // Phase 1D-pre-D: internal markers → inline helpers so they bracket
    // image.Clear in the same dispatch order on both producer and assembler
    // threads. Marker-routing would defer the markers past the Clear.
    DoScopeMarkerBeginInline(
        fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                    col_buf.CmaskAddress()),
        /*from_guest=*/false);
    image.Clear(clear_value, desc.view_info.range);
    DoScopeMarkerEndInline(/*from_guest=*/false);
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    // Phase 1D-0/0b: Stage 1 — build a DrawIntent and hand off to
    // BundleAssembler. The actual per-draw work runs out of
    // DoDrawFromIntent below, called synchronously from
    // BundleAssembler::ProcessOne in 2A/2B-1 (asynchronously in 2B-2).
    DrawIntent intent;
    intent.type = is_indexed ? DrawIntent::Type::DrawIndexed
                             : DrawIntent::Type::Draw;
    // Pre-fill draw payload from the live regs. 2B-1: DoDrawFromIntent
    // reads from the snapshot captured in PushAndProcess, so these
    // intent.draw fields are inert today; they exist for forward-
    // compatibility with the 2B-2 async path's intent shape.
    const auto& regs = liverpool->regs;
    intent.draw.num_indices = regs.num_indices;
    intent.draw.num_instances = regs.num_instances.NumInstances();
    intent.draw.vertex_offset = 0; // resolved per-pipeline in DoDrawFromIntent
    intent.draw.instance_offset = 0;
    intent.draw.index_offset = index_offset;
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoDrawFromIntent(const DrawIntent& intent) {
    // Phase 1D-0b (Turn 2B-1): body reads from
    // `liverpool->GetSnapshot(intent.snapshot_idx)`, NOT `liverpool->regs`.
    // Synchronous in 2B-1 (regs == snapshot at this instant), but the
    // migration is what makes 2B-2's async dispatch correct.
    const bool is_indexed = (intent.type == DrawIntent::Type::DrawIndexed);
    const u32 index_offset = intent.draw.index_offset;

    RENDERER_TRACE;

    // Batch PopPendingOperations: only check every 16th draw.
    // Pending ops are deferred GPU-side completions; 16-draw latency is imperceptible.
    //
    // PERF(GR2FORK v1.17): The bitmask gate fires on 1 in 16 draws by
    // construction; hint the body so the call instruction sequence sits in
    // the cold tail and the per-draw straight-line code stays compact.
    if ((draw_counter_++ & 15) == 0) [[unlikely]] {
        scheduler.PopPendingOperations();
    }

    const auto& regs = liverpool->GetSnapshot(intent.snapshot_idx);
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(regs);
    // PERF(GR2FORK v1.21): pipeline is null only on cache failure / unsupported
    // shader stage combinations — an error path. Steady-state draws have a
    // valid pipeline.
    if (!pipeline) [[unlikely]] {
        return;
    }

    // Skip FilterDraw when pipeline hasn't changed — FilterDraw only reads
    // key-affecting regs (color_control.mode, primitive_type, depth overrides)
    // which are stable within the same pipeline.
    //
    // PERF(GR2FORK v1.15): The dominant case is consecutive draws sharing the
    // same pipeline (the pre-existing same_pipeline metric quantifies this).
    // Hint the rare-path branch so the optimizer keeps the FilterDraw call
    // out of the straight-line dispatch sequence.
    if (pipeline != last_filter_pipeline_) [[unlikely]] {
        last_filter_result_ = FilterDraw(regs);
        last_filter_pipeline_ = pipeline;
    } else {
        // PERF(GR2 v1.22): wire up the same_pipeline counter that's been
        // declared but unused. Tracks the FilterDraw skip rate, which
        // bounds how much of br_slow time is "consecutive same-pipeline
        // draws not getting skipped further upstream."
        GR2_INSTR_ON_SAME_PIPELINE();
    }
    // PERF(GR2FORK v1.21): FilterDraw rejects only the special cb-disabled
    // depth-copy / FmaskDecompress / EliminateFastClear / Resolve / NoPrim
    // cases — the dominant draws pass through.
    if (!last_filter_result_) [[unlikely]] {
        return;
    }

    // GpuComm v4 instrumentation: count this draw + track pipeline change for
    // unique-pipeline-per-frame upper bound. See gpucomm_metrics.h header.
    GR2_INSTR_ON_DRAW(pipeline);

    PrepareRenderState(pipeline, regs);
    // PERF(GR2FORK v1.21): BindResources returns false only on rare
    // resource-collision / pipeline-layout error paths; steady-state
    // draws successfully bind.
    if (!BindResources(pipeline, regs)) [[unlikely]] {
        return;
    }
    const auto state = BeginRendering(pipeline, regs);

    {
        // GpuComm v1.14 instrumentation: vertex/index buffer binding cost.
        // Dirty VBO pages trigger SynchronizeBuffer cascades INSIDE the
        // buffer cache, so this bucket captures both the bind work and any
        // synchronous upload that fires from inside it.
        GR2_INSTR_TIMER_DECL(vbuf_timer_);
        buffer_cache.BindVertexBuffers(*pipeline, regs);
        if (is_indexed) {
            buffer_cache.BindIndexBuffer(index_offset, regs);
        }
        GR2_INSTR_ON_BIND_VBUF(vbuf_timer_.ElapsedNs());
    }

    {
        // GpuComm v1.14 instrumentation: Vulkan-command-submission ladder —
        // pushDescriptorSetKHR (pipeline->BindResources), dynamic state,
        // BeginRendering, BindPipelineCached, and the actual cmdbuf.draw[Indexed].
        GR2_INSTR_TIMER_DECL(dispatch_timer_);
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
        UpdateDynamicState(pipeline, is_indexed, regs);
        scheduler.BeginRendering(state);

        const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
        const auto& fetch_shader = pipeline->GetFetchShader();
        const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

        const auto cmdbuf = scheduler.PrimaryCommandBuffer();
        BindPipelineCached(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

        if (is_indexed) {
            cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                               s32(vertex_offset), instance_offset);
        } else {
            cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                        instance_offset);
        }
        GR2_INSTR_ON_DRAW_DISPATCH(dispatch_timer_.ElapsedNs());
    }

    ResetBindings();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    // Phase 1D-0 (Turn 2A): Stage 1 — build a DrawIntent and hand off.
    DrawIntent intent;
    intent.type = is_indexed ? DrawIntent::Type::DrawIndexedIndirect
                             : DrawIntent::Type::DrawIndirect;
    intent.indirect.address = arg_address;
    intent.indirect.offset = offset;
    // Section 8 distinguishes `size` and `stride`; the legacy public API
    // only supplies one (the per-element stride). Fill both with the same
    // value so the shape is forward-compatible without changing the
    // contract observed by liverpool.cpp's PM4 handler.
    intent.indirect.size = stride;
    intent.indirect.stride = stride;
    intent.indirect.max_count = max_count;
    intent.indirect.count_address = count_address;
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoDrawIndirectFromIntent(const DrawIntent& intent) {
    // Body identical to the pre-Turn 2A Rasterizer::DrawIndirect.
    const bool is_indexed = (intent.type == DrawIntent::Type::DrawIndexedIndirect);
    const VAddr arg_address = intent.indirect.address;
    const u32 offset = intent.indirect.offset;
    const u32 stride = intent.indirect.stride;
    const u32 max_count = intent.indirect.max_count;
    const VAddr count_address = intent.indirect.count_address;

    RENDERER_TRACE;

    const auto& regs = liverpool->GetSnapshot(intent.snapshot_idx);

    scheduler.PopPendingOperations();

    if (!FilterDraw(regs)) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(regs);
    if (!pipeline) {
        return;
    }

    // GpuComm v4 instrumentation.
    GR2_INSTR_ON_DRAW_INDIRECT();

    PrepareRenderState(pipeline, regs);
    if (!BindResources(pipeline, regs)) [[unlikely]] {
        return;
    }
    const auto state = BeginRendering(pipeline, regs);

    {
        // GpuComm v1.14 instrumentation: vbuf cost (mirrors Draw).
        GR2_INSTR_TIMER_DECL(vbuf_timer_);
        buffer_cache.BindVertexBuffers(*pipeline, regs);
        if (is_indexed) {
            buffer_cache.BindIndexBuffer(0, regs);
        }
        GR2_INSTR_ON_BIND_VBUF(vbuf_timer_.ElapsedNs());
    }

    // ObtainBuffer for indirect args is buffer-cache work but conceptually
    // an argument-buffer fetch, not a vertex bind. Left untimed (counted
    // toward the residual). DrawIndirect runs ~10/frame so the noise is
    // negligible regardless.
    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    {
        // GpuComm v1.14 instrumentation: dispatch ladder (mirrors Draw).
        GR2_INSTR_TIMER_DECL(dispatch_timer_);
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
        UpdateDynamicState(pipeline, is_indexed, regs);
        scheduler.BeginRendering(state);

        // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
        // instance offsets will be automatically applied by Vulkan from indirect args buffer.

        const auto cmdbuf = scheduler.PrimaryCommandBuffer();
        BindPipelineCached(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

        if (is_indexed) {
            ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

            if (count_address != 0) {
                cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                                count_base, max_count, stride);
            } else {
                cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
            }
        } else {
            ASSERT(sizeof(VkDrawIndirectCommand) == stride);

            if (count_address != 0) {
                cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_base,
                                         max_count, stride);
            } else {
                cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
            }
        }
        GR2_INSTR_ON_DRAW_DISPATCH(dispatch_timer_.ElapsedNs());
    }

    ResetBindings();
}

void Rasterizer::DispatchDirect() {
    // Phase 1D-0 (Turn 2A): Stage 1 — build a DrawIntent and hand off. The
    // dispatch dimensions are pulled from cs_state at intent-build time;
    // DoDispatchDirectFromIntent re-reads them from cs_state when it runs
    // in 2A (synchronous, identical values). Carrying them in the intent
    // makes 2B's async path correct without further code changes.
    const auto& cs_program = liverpool->GetCsRegs();
    DrawIntent intent;
    intent.type = DrawIntent::Type::Dispatch;
    intent.dispatch.dim_x = cs_program.dim_x;
    intent.dispatch.dim_y = cs_program.dim_y;
    intent.dispatch.dim_z = cs_program.dim_z;
    intent.dispatch.indirect_address = 0;
    intent.dispatch.indirect_offset = 0;
    intent.dispatch.indirect_size = 0;
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoDispatchDirectFromIntent(const DrawIntent& intent) {
    // Body identical to the pre-Turn 2A Rasterizer::DispatchDirect.
    (void)intent; // dispatch dims re-read from cs_state in 2A; intent fields are inert here.

    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& regs = liverpool->GetSnapshot(intent.snapshot_idx);
    const auto& cs_program = regs.cs_program;
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline(regs);
    // PERF(GR2FORK v1.26): null pipeline is a cache failure / unsupported
    // shader stage path — error case. Steady-state dispatches resolve a
    // valid pipeline.
    if (!pipeline) [[unlikely]] {
        return;
    }


    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    // PERF(GR2FORK v1.26): ExecuteShaderHLE handles a small set of
    // recognised compute kernels (specific clear/copy patterns) — most
    // game compute dispatches fall through to the regular path.
    // Note: HLE body does not read `regs`; pass live liverpool->regs to
    // satisfy the AmdGpu::Regs& signature without coupling the snapshot.
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) [[unlikely]] {
        return;
    }

    // PERF(GR2FORK v1.26): BindResources fail-return is the rare
    // resource-collision / pipeline-layout error path; steady-state
    // dispatches successfully bind.
    if (!BindResources(pipeline, regs)) [[unlikely]] {
        return;
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.PrimaryCommandBuffer();
    BindPipelineCached(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);

    ResetBindings();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    // Phase 1D-0 (Turn 2A): Stage 1 — build a DrawIntent and hand off.
    DrawIntent intent;
    intent.type = DrawIntent::Type::DispatchIndirect;
    intent.dispatch.dim_x = 0;
    intent.dispatch.dim_y = 0;
    intent.dispatch.dim_z = 0;
    intent.dispatch.indirect_address = address;
    intent.dispatch.indirect_offset = offset;
    intent.dispatch.indirect_size = size;
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoDispatchIndirectFromIntent(const DrawIntent& intent) {
    // Body identical to the pre-Turn 2A Rasterizer::DispatchIndirect.
    const VAddr address = intent.dispatch.indirect_address;
    const u32 offset = intent.dispatch.indirect_offset;
    const u32 size = intent.dispatch.indirect_size;

    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& regs = liverpool->GetSnapshot(intent.snapshot_idx);
    (void)regs.cs_program; // captured but unread for indirect dispatch (dims live in the indirect buffer)
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline(regs);
    // PERF(GR2FORK v1.26): mirrors DispatchDirect — null pipeline /
    // BindResources fail are error paths.
    if (!pipeline) [[unlikely]] {
        return;
    }

    if (!BindResources(pipeline, regs)) [[unlikely]] {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.PrimaryCommandBuffer();
    BindPipelineCached(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);

    ResetBindings();
}

u64 Rasterizer::Flush() {
    // Phase 1D-pre-D: marker-route. The producer captures the pre-flush
    // tick BEFORE pushing the marker so the value reflects the state at
    // entry, regardless of when the assembler thread (under future async)
    // actually runs the body. Under sync PushAndProcess drains inline, so
    // the body has already executed when this function returns.
    //
    // Phase 1D-1 (Phase G): BLOCKING — Push + WaitFor. The producer thread
    // (PM4 via liverpool.cpp:398) waits for the assembler to actually
    // submit. The captured tick is approximate under async — the only
    // current caller (liverpool.cpp:398) discards the return value, so
    // the approximation is observably equivalent.
    const u64 current_tick = scheduler.CurrentTick();
    DrawIntent intent;
    intent.type = DrawIntent::Type::Flush;
    const u32 seq = bundle_assembler_.Push(intent);
    bundle_assembler_.WaitFor(seq);
    return current_tick;
}

void Rasterizer::DoFlushFromIntent(const DrawIntent& intent) {
    (void)intent;  // Flush carries no payload.
    SubmitInfo info{};
    scheduler.Flush(info);
}

void Rasterizer::Finish() {
    // Phase 1D-1 (Phase G): BLOCKING — Push + WaitFor. The producer
    // observes that scheduler.Finish() has completed (GPU idle) before
    // returning to liverpool.cpp:1006.
    DrawIntent intent;
    intent.type = DrawIntent::Type::Finish;
    const u32 seq = bundle_assembler_.Push(intent);
    bundle_assembler_.WaitFor(seq);
}

void Rasterizer::DoFinishFromIntent(const DrawIntent& intent) {
    (void)intent;  // Finish carries no payload.
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    // Phase 1D-1 (Phase G): BLOCKING — Push + WaitFor. OnSubmit's body
    // (texture_cache.ProcessDownloadImages, GC) writes to draw_scheduler;
    // those writes must be observable by subsequent producer-thread reads
    // before the producer continues processing the next PM4 packet. Wait
    // ensures the assembler has completed the cache-state mutations.
    DrawIntent intent;
    intent.type = DrawIntent::Type::OnSubmit;
    const u32 seq = bundle_assembler_.Push(intent);
    bundle_assembler_.WaitFor(seq);
}

void Rasterizer::DoOnSubmitFromIntent(const DrawIntent& intent) {
    (void)intent;  // OnSubmit carries no payload.
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    buffer_cache.RunGarbageCollector();
}

void Rasterizer::PrepareDescriptorDeltaCache(const Pipeline* pipeline) {
    // [GR2FORK fix4 R8 interaction] Push descriptors ARE active on current
    // RADV/Mesa: VK_KHR_push_descriptor is exposed and BuildDescSetLayout
    // selects the push path. The desc-cache delta filter then correctly
    // skipped re-emitting unchanged push-descriptor bindings (its design
    // intent for push-descriptor performance). But R8 / EnsureLayoutCoverage
    // misreads "skipped because cached" as "missing" and overwrites the
    // (still-valid) cached push state with VK_NULL_HANDLE — producing
    // flickering meshes.
    //
    // Fix: force the cache to invalidate every call so set_writes always
    // contains the full layout. Push descriptors stay enabled for the
    // binding mechanism (Pipeline::BindResources still calls
    // vkCmdPushDescriptorSetKHR); only the per-binding emit-dedup is
    // killed. EnsureLayoutCoverage's null-injection becomes unreachable in
    // steady state and stays as a safety net.
    //
    // The push-descriptor cache bookkeeping below (gen/tick/layout/epoch)
    // is now dead — ShouldWriteDescriptorImpl is unreachable behind the
    // unconditional `desc_cache_force_write_ = true` gate in the inline
    // ShouldWriteDescriptor wrapper. Excised for clarity.
    (void)pipeline;
    desc_cache_force_write_ = true;
}

bool Rasterizer::ShouldWriteDescriptorImpl(u32 binding, vk::DescriptorType type, u64 a, u64 b, u64 c) {
    // PERF(GR2 v1.22): The RADV-path short-circuit lives in the inline
    // wrapper in vk_rasterizer.h. By the time we reach this body, we're
    // on the push-descriptor path with a valid cache.
    if (binding >= desc_cache.size()) [[unlikely]] {
        return true;
    }
    auto& e = desc_cache[binding];
    if (e.epoch == desc_cache_epoch && e.type == type && e.a == a && e.b == b && e.c == c) [[likely]] {
        return false;
    }
    e.epoch = desc_cache_epoch;
    e.type = type;
    e.a = a;
    e.b = b;
    e.c = c;
    return true;
}

bool Rasterizer::BindResources(const Pipeline* pipeline,
                               const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // PERF(GR2FORK v1.25): the IsComputeImageCopy / IsComputeMetaClear /
    // IsComputeImageClear trio short-circuits via the existing
    // [[likely]] !IsCompute() fast paths in each function (v1.16). For
    // graphics draws — the dominant case — all three return false in
    // 1-2 cycles each, so the OR-chain body fires only on the rare
    // meta-compute dispatch.
    if (IsComputeImageCopy(pipeline, regs) || IsComputeMetaClear(pipeline, regs) ||
        IsComputeImageClear(pipeline, regs)) [[unlikely]] {
        return false;
    }

    // GpuComm v4 instrumentation: count gfx-only BindResources entries (the
    // compute meta paths above bypass everything we care about for Candidate A).
    GR2_INSTR_ON_BR_ENTER();

    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();

    PrepareDescriptorDeltaCache(pipeline);

    Shader::Backend::Bindings binding{};
    MakeUserData(push_data, regs);

    // =========================================================================
    // Stage binding skip cache. The entire machinery below (stamp/tick check,
    // per-stage XXH3 verification, fast-path returns) is dead on RADV
    // (Jun's deployment) because the gating predicate `pipeline_uses_push`
    // is statically false there: RADV doesn't advertise the push-descriptor
    // extension, so every pipeline takes the descriptor-set path. The v1.25
    // [[unlikely]] hints made the BLOCKS cold, but the unconditional atomic
    // loads of CurrentTick() and GetGfxPipelineStamp() above them ran on
    // every BindResources call regardless — the compiler can't hoist
    // observable atomic loads past dead branches. v1.39 nests both loads,
    // along with the all_stages_cached / stamp_match evaluation and both
    // fast-path early-return blocks, under the `if (pipeline_uses_push)`
    // gate. On RADV the dominant path now flows straight from MakeUserData
    // to the bindings for-loop with zero stamp-cache overhead.
    //
    // FIX(GR2FORK): the all_stages_cached fast path emits an empty set_writes
    // and relies on push-descriptor state persisting across draws. Push
    // descriptors support that; descriptor sets do not — each newly-committed
    // VkDescriptorSet starts uninitialized, so an empty set_writes would
    // produce an empty set. The pipeline_uses_push gate is what keeps this
    // fast path safe.
    //
    // On push-descriptor platforms `tick` and `cur_stamp` are loaded once on
    // entry to the gated block and reused by both the fast-path-return AND
    // the skip-cache-update at function tail (both ifs share the same
    // gating predicate, so when one fires the other does too).
    // =========================================================================
    const bool pipeline_uses_push = pipeline && pipeline->UsesPushDescriptors();
    u64 tick = 0;
    u64 cur_stamp = 0;

    if (pipeline_uses_push) [[unlikely]] {
        tick = scheduler.CurrentTick();
        // Phase 1D-pre-C: read pipeline stamp from the snapshot rather
        // than `liverpool->GetGfxPipelineStamp()`. The binding skip cache
        // wants the stamp value AT the consuming intent, which is exactly
        // what the snapshot froze.
        cur_stamp = regs.gfx_pipeline_stamp;

        bool all_stages_cached = !pipeline->IsCompute() &&
                                  binding_skip_cache_.valid &&
                                  binding_skip_cache_.pipeline == pipeline &&
                                  binding_skip_cache_.cmdbuf_tick == tick;
        const bool stamp_match = all_stages_cached &&
                                 binding_skip_cache_.stamp == cur_stamp;

        // Stamp-mismatch fallback: the GFX pipeline stamp moved, but
        // user_data may not have. Per-stage byte-level verification covers
        // the rare case where a SetContextReg moved without touching the
        // SH user-data registers. On stamp match this loop is skipped and
        // we trust the cache transitively.
        //
        // PERF(GR2FORK v1.58): memcmp the cached user_data snapshot
        // instead of computing XXH3 on each call. memcmp on 64 bytes is
        // one AVX-512 load + cmp on Zen 4 (~3-5 ns) — faster than XXH3
        // setup + finalization (~10-20 ns), and strictly more reliable
        // (no hash collisions). The pgm_hash check below short-circuits
        // before the memcmp when the shader at this stage slot changed.
        if (all_stages_cached && !stamp_match) {
            for (const auto* stage : pipeline->GetStages()) {
                if (!stage) continue;
                const u32 si = static_cast<u32>(stage->l_stage);
                if (binding_skip_cache_.stages[si].pgm_hash != stage->pgm_hash ||
                    std::memcmp(binding_skip_cache_.stages[si].cached_user_data.data(),
                                stage->user_data.data(),
                                stage->user_data.size_bytes()) != 0) {
                    all_stages_cached = false;
                    break;
                }
            }
        }

        if (all_stages_cached && !binding_skip_cache_.uses_dma) {
            // Stamp-matched replay: reuse cached push_data wholesale,
            // advance binding counters from cached stage sizes. PushUd
            // reads stage->user_data and writes push_data.ud_regs[]; with
            // stamp match user_data is unchanged since the snapshot was
            // taken, so the snapshot is bit-identical to what PushUd would
            // produce now. MakeUserData's writes (buf_offsets memset +
            // viewport scalars) are also captured in the snapshot, so
            // overwriting push_data here is correct.
            if (stamp_match && binding_skip_cache_.push_data_valid) {
                push_data = binding_skip_cache_.cached_push_data;
                for (const auto* stage : pipeline->GetStages()) {
                    if (!stage) continue;
                    binding.buffer += stage->buffers.size();
                    binding.unified += stage->buffers.size() +
                                       stage->images.size() +
                                       stage->samplers.size();
                }
                GR2_INSTR_ON_BR_FAST_REPLAY();
                return true;
            }
            // Identical-bindings fast path: rerun PushUd, skip Bind* work.
            // PERF(GR2FORK v1.19): cache the three .size() calls per stage.
            // Each is called from a different += target, so without
            // explicit caching the compiler must re-prove no-aliasing
            // across each binding member write.
            for (const auto* stage : pipeline->GetStages()) {
                if (!stage) continue;
                stage->PushUd(binding, push_data);
                const auto bufs = stage->buffers.size();
                const auto imgs = stage->images.size();
                const auto samps = stage->samplers.size();
                binding.buffer += bufs;
                binding.unified += bufs + imgs + samps;
            }
            // Snapshot push_data so the next stamp-matched call can skip
            // the PushUd loop above.
            binding_skip_cache_.cached_push_data = push_data;
            binding_skip_cache_.push_data_valid = true;
            GR2_INSTR_ON_BR_FAST_PUSHUD();
            return true;
        }
    }

    // Full binding path.
    // GpuComm v4 instrumentation: time the descriptor-construction work that
    // Candidate A would move to a worker thread (per-stage BindBuffers +
    // BindTextures + cache-update bookkeeping). Deliberately stops BEFORE
    // the uses_dma SynchronizeBuffersInRange block — that's Candidate C
    // territory and including it would contaminate Candidate A's signal.
    GR2_INSTR_TIMER_DECL(slow_timer_);
    uint32_t slow_stages_bound_ = 0;

    bool uses_dma = false;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) continue;
        ++slow_stages_bound_;
        uses_dma |= stage->uses_dma;
        stage->PushUd(binding, push_data);
        BindBuffers(*stage, binding, push_data, regs);
        BindTextures(*stage, binding, regs);
    }

    // Y-1 (Y-0 strip): the post-iteration EnsureLayoutCoverage / null-descriptor
    // injection was a 1B fix specifically for the secondary-cmdbuf shape
    // (each secondary opens with fresh descriptor state, so layout-declared
    // bindings the per-stage loop didn't touch had to be filled with null
    // writes to avoid GPUVM faults). Under primary-CB direct recording the
    // descriptor set retains prior draws' values, so unwritten layout-declared
    // bindings are tolerated by the driver — same behavior as pre-1B v1.58.
    // Calls removed; helper bodies deleted at end of file.

    // Update skip cache for next draw.
    //
    // PERF(GR2 v1.22): On RADV (pipeline_uses_push=false), the fast path at
    // line ~621 can never enter — its `all_stages_cached` gate requires
    // `pipeline_uses_push`. Skip the bookkeeping entirely on that path:
    // computing per-stage XXH3 hashes, copying push_data, and storing
    // pipeline/stamp/uses_dma is pure dead work. Saves ~0.5-1.5 ms/frame
    // at 5000 draws.
    //
    // On push-descriptor deployments the fast path remains live and the
    // cache must continue to be maintained.
    //
    // PERF(GR2FORK v1.22): Hint as cold for RADV (Jun's target). The
    // existing comment block above already documents that this body is
    // dead work on RADV — the hint just gives the codegen the same signal.
    if (pipeline_uses_push) [[unlikely]] {
        binding_skip_cache_.pipeline = pipeline;
        binding_skip_cache_.cmdbuf_tick = tick;
        binding_skip_cache_.uses_dma = uses_dma;
        // PERF(GR2): record the stamp seen at the start of this call. cur_stamp
        // is stable through BindResources because the gfx_pipeline_stamp is
        // only mutated by Set*Reg paths in the PM4 parser, which has already
        // handed off to the draw path before BindResources is called.
        binding_skip_cache_.stamp = cur_stamp;
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) continue;
            const u32 si = static_cast<u32>(stage->l_stage);
            binding_skip_cache_.stages[si].pgm_hash = stage->pgm_hash;
            // PERF(GR2FORK v1.58): snapshot user_data bytes instead of
            // hashing. The verification loop above does memcmp against
            // this snapshot — same correctness as the XXH3 path, faster
            // per-call. Copying size_bytes() (≤ 64) is comparable in cost
            // to XXH3_64bits's setup work, but eliminates the cost from
            // the hot verification path.
            std::memcpy(binding_skip_cache_.stages[si].cached_user_data.data(),
                        stage->user_data.data(),
                        stage->user_data.size_bytes());
        }
        binding_skip_cache_.valid = true;
        // PERF(GR2): snapshot push_data on the slow path too, so a subsequent
        // stamp-matched draw can skip the PushUd loop. Slow path runs PushUd
        // (per stage) followed by BindBuffers (which also writes to push_data,
        // e.g. flatbuf offsets); capturing after both ensures the snapshot
        // reflects everything the fast path's PushUd loop would reproduce.
        binding_skip_cache_.cached_push_data = push_data;
        binding_skip_cache_.push_data_valid = true;
    } else {
        // Mark cache invalid on RADV so any future code path that does
        // read it sees stale state and falls back to slow path. The valid
        // bit is already false on first BindResources call, so this only
        // matters if a previous push-deployment session left it valid;
        // safe in all cases.
        binding_skip_cache_.valid = false;
        binding_skip_cache_.push_data_valid = false;
    }

    // GpuComm v4 instrumentation: fire the slow-path metric BEFORE the
    // uses_dma block. The DMA sync is buffer-upload work tracked separately
    // (Candidate C), not the descriptor-construction work targeted here.
    GR2_INSTR_ON_BR_SLOW(slow_timer_.ElapsedNs(), slow_stages_bound_);

    if (uses_dma) {
        // GpuComm v1.14 instrumentation: this is the Candidate C target —
        // SynchronizeBuffersInRange traversal of the mapped-ranges interval
        // set, where any CPU-dirty pages get uploaded to staging buffers.
        GR2_INSTR_TIMER_DECL(sync_timer_);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (auto& range : mapped_ranges) {
            buffer_cache.SynchronizeBuffersInRange(range.lower(), range.upper() - range.lower());
        }
        fault_process_pending = true;
        GR2_INSTR_ON_SYNC_BUFFER(sync_timer_.ElapsedNs());
    }

    return true;
}


void Rasterizer::BindPipelineCached(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) {
    const auto cmdbuf = scheduler.PrimaryCommandBuffer();
    const u64 tick = scheduler.CurrentTick();
    // PERF(GR2FORK v1.16): Pipeline rebinds within the same cmdbuf tick are
    // dominated by the same-pipeline case (the pre-existing FilterDraw
    // same_pipeline counter quantifies this). Hint the cache-hit return.
    if (last_bound_pipeline_ == pipeline &&
        last_bound_pipeline_tick_ == tick) [[likely]] {
        return;
    }
    cmdbuf.bindPipeline(bind_point, pipeline);
    last_bound_pipeline_ = pipeline;
    last_bound_pipeline_tick_ = tick;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline,
                                    const AmdGpu::LiverpoolRegsSnapshot& regs) {
    (void)regs;
    // PERF(GR2FORK v1.16): BindResources calls this trio (IsComputeMetaClear,
    // IsComputeImageCopy, IsComputeImageClear) on every Draw / Dispatch entry.
    // In normal gameplay graphics draws far outnumber compute dispatches, so
    // !IsCompute() is the dominant early-return for all three. Hint it.
    if (!pipeline->IsCompute()) [[likely]] {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline,
                                    const AmdGpu::LiverpoolRegsSnapshot& regs) {
    if (!pipeline->IsCompute()) [[likely]] {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = regs.cs_program;
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    if (instance.IsMaintenance8Supported() ||
        src_image.info.props.is_depth == dst_image.info.props.is_depth) {
        dst_image.CopyImage(src_image);
    } else {
        const auto& copy_buffer =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::DeviceLocal);
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0);
    }
    dst_image.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst_image.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline,
                                     const AmdGpu::LiverpoolRegsSnapshot& regs) {
    if (!pipeline->IsCompute()) [[likely]] {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = regs.cs_program;
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    image1.Clear(clear, range);
    image1.flags |= VideoCore::ImageFlagBits::GpuModified;
    image1.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data,
                             const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // PERF(GR2FORK v1.55): intra-call FindBuffer / ObtainBuffer dedup caches
    // culled. v1.54-D BrDiag measured 0 hits / 32,496,743 lookups for both
    // FindBufferCached and ObtainBufferCached across 11.4M slow-path
    // BindResources calls. The 32-entry static thread_local arrays + lambda
    // wrappers + per-iteration probe loops were doing pure overhead work
    // (linear search, populate, struct construction) with zero payoff —
    // shaders effectively never bind the same buffer to multiple slots
    // within a single BindBuffers call, mirroring the v1.43 finding for
    // BindTextures' find_image_cache. Direct calls to buffer_cache.FindBuffer
    // and buffer_cache.ObtainBuffer behave identically, since the underlying
    // calls fired on every iteration anyway (cache always missed).
    //
    // The cross-stage / cross-call buffer cache lookups handled inside
    // VideoCore::BufferCache itself are unaffected — those carry the real
    // load and stay in place.

    // Single fused pass: resolve buffer_id and bind in one iteration.
    // Eliminates the buffer_bindings intermediate vector (clear + emplace_back + read-back).
    for (u32 i = 0; i < stage.buffers.size(); i++) {
        const auto& desc = stage.buffers[i];
        const auto vsharp = desc.GetSharp(stage);

        VideoCore::BufferId buffer_id{};
        u64 size = 0;
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
        }

        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        if (!buffer_id) {
            // PERF(GR2FORK v1.28): GdsBuffer (Global Data Share) is a hardware
            // feature essentially unused by retail PS4 titles — the rarest
            // branch in this chain.
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) [[unlikely]] {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) [[likely]] {
                // PERF(GR2FORK v1.28): Flatbuf carries the flattened
                // user-data buffer used by ReadConst — virtually every
                // compiled shader has at least one. Among "special"
                // buffer types (the !buffer_id branch), this is the
                // dominant case.
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);

                struct FlatbufCacheEntry {
                    u64 stage_key = 0;
                    vk::CommandBuffer cmdbuf{};
                    u64 hash = 0;
                    u64 offset = 0;
                    u32 size = 0;
                    bool valid = false;
                };
                static thread_local std::array<FlatbufCacheEntry, 32> flatbuf_cache{};

                const auto cmdbuf = scheduler.PrimaryCommandBuffer();
                const u64 stage_key = static_cast<u64>(stage.pgm_hash) ^ (static_cast<u64>(desc.sharp_idx) << 1);

                FlatbufCacheEntry& e = flatbuf_cache[stage_key & (flatbuf_cache.size() - 1)];

                // PERF(GR2 v1.22): Defer the XXH3 hash until the cheap
                // identity checks (slot validity, stage key, cmdbuf, size)
                // pass. On a slot collision (different stage hashes to the
                // same slot) the hash work is unused — but the previous
                // code computed it unconditionally. The reordered check
                // skips XXH3_64bits when any cheap validator fails,
                // saving ~80-300 cycles per BindBuffers Flatbuf binding.
                const bool identity_match =
                    e.valid && e.stage_key == stage_key &&
                    e.cmdbuf == cmdbuf && e.size == ubo_size;

                u64 offset;
                // PERF(GR2FORK v1.27): once the thread-local 32-entry cache
                // is warmed for a stage, identity_match hits dominantly —
                // shadPS4's stage_key hash distributes ~uniformly so slot
                // collisions are rare in steady state.
                if (identity_match) [[likely]] {
                    // Same shader, same cmdbuf, same payload size — content
                    // *might* be unchanged. Hash to verify.
                    const u64 payload_hash = (ubo_size == 0) ? 0
                        : XXH3_64bits(stage.flattened_ud_buf.data(), ubo_size);
                    // PERF(GR2FORK v1.27): UD buffer content typically
                    // changes once per shader invocation parameter set
                    // (camera/view constants on frame change), not once
                    // per draw. Hashes match across consecutive same-
                    // pipeline draws sharing the same flat-buf payload.
                    if (e.hash == payload_hash) [[likely]] {
                        // Full hit — reuse the prior staging-buffer offset.
                        offset = e.offset;
                    } else {
                        // Content changed — re-stage and update hash only.
                        offset = vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                        e.hash = payload_hash;
                        e.offset = offset;
                    }
                } else {
                    // Slot miss — full populate. Hash on the way in.
                    offset = vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                    e.stage_key = stage_key;
                    e.cmdbuf = cmdbuf;
                    e.hash = (ubo_size == 0) ? 0
                        : XXH3_64bits(stage.flattened_ud_buf.data(), ubo_size);
                    e.offset = offset;
                    e.size = ubo_size;
                    e.valid = true;
                }

                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) [[unlikely]] {
                // PERF(GR2FORK v1.28): BdaPagetable is used only by
                // shaders performing buffer-device-address pointer-walks —
                // a small subset.
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) [[unlikely]] {
                // PERF(GR2FORK v1.28): FaultBuffer is the shadPS4 fault-
                // recovery sideband — only present when fault-process is
                // wired up.
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) [[unlikely]] {
                // PERF(GR2FORK v1.28): SharedMemory (LDS) is compute-only;
                // graphics draws never reach this branch.
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = regs.cs_program;
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);

                constexpr u64 kGpuFillThreshold = 256;
                if (lds_size >= kGpuFillThreshold) {
                    (void)data;
                    auto cmdbuf = scheduler.PrimaryCommandBuffer();
                    cmdbuf.fillBuffer(lds_buffer.Handle(), offset, lds_size, 0);

                    buffer_barriers.push_back(vk::BufferMemoryBarrier2{
                        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .buffer = lds_buffer.Handle(),
                        .offset = offset,
                        .size = lds_size,
                    });
                } else {
                    std::memset(data, 0, lds_size);
                }

                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else if (instance.IsNullDescriptorSupported()) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                auto& null_buffer = buffer_cache.GetBuffer(VideoCore::NULL_BUFFER_ID);
                buffer_infos.emplace_back(null_buffer.Handle(), 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id);
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            ASSERT(adjust % 4 == 0);
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            if (auto barrier =
                    vk_buffer->GetBarrier(desc.is_written ? vk::AccessFlagBits2::eShaderWrite
                                                          : vk::AccessFlagBits2::eShaderRead,
                                          // OPT: Use specific shader stage instead of eAllCommands.
                                          // eAllCommands forces a full pipeline drain which prevents
                                          // the GPU from overlapping independent work.
                                          vk::PipelineStageFlagBits2::eAllGraphics |
                                              vk::PipelineStageFlagBits2::eComputeShader)) [[unlikely]] {
                // PERF(GR2FORK v1.21): GetBarrier returns nullopt when the
                // buffer's access mode/stage is unchanged from the last
                // draw — the steady-state case. The barrier-emit body
                // fires only on first touch / access transitions.
                buffer_barriers.emplace_back(*barrier);
            }
            // PERF(GR2FORK v1.21): is_written && is_formatted requires both
            // SSBO write + typed buffer access — uncommon combination in
            // most shaders.
            if (desc.is_written && desc.is_formatted) [[unlikely]] {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
        }

        const u32 dst_binding = binding.unified++;
        const vk::DescriptorType dtype =
        is_storage ? vk::DescriptorType::eStorageBuffer : vk::DescriptorType::eUniformBuffer;

        const auto bi_handle = reinterpret_cast<u64>(static_cast<VkBuffer>(buffer_infos.back().buffer));
        const u64 bi_offset = static_cast<u64>(buffer_infos.back().offset);
        const u64 bi_range  = static_cast<u64>(buffer_infos.back().range);

        if (ShouldWriteDescriptor(dst_binding, dtype, bi_handle, bi_offset, bi_range)) {
            auto& w = set_writes.emplace_back();
            w.dstSet = VK_NULL_HANDLE;
            w.dstBinding = dst_binding;
            w.dstArrayElement = 0;
            w.descriptorCount = 1;
            w.descriptorType = dtype;
            w.pNext = nullptr;
            w.pImageInfo = nullptr;
            w.pBufferInfo = &buffer_infos.back();
            w.pTexelBufferView = nullptr;
        }

        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                              const AmdGpu::LiverpoolRegsSnapshot& regs) {
    image_bindings.clear();
    bool any_needs_rebind = false; // OPT(v18): Track during first pass instead of separate scan

    // OPT(v14.1 hotfix): Keep caches off TLS.
    //
    // v14 used large thread_local std::array<...> caches inside BindTextures(). That inflates TLS,
    // and with shadPS4's custom pthread stacks, some systems hit pthread_create() EINVAL (22)
    // during early game startup. Store the caches on the Rasterizer instance instead.
    //
    // PERF(GR2FORK v1.43): find_image_cache (intra-call dedup) culled —
    // diagnostic metrics across many windows showed 0 hits across millions
    // of iterations. Shaders effectively never bind the same image to
    // multiple slots within one BindTextures call. The members
    // find_image_cache_ and find_image_cache_stamp_ are removed from the
    // header alongside this cull. The cross-call find_image_pcache_ is
    // retained — that one carries real load.
    const bool null_descriptors_supported = instance.IsNullDescriptorSupported();
    const bool attachment_feedback_loop_layout_supported =
    instance.IsAttachmentFeedbackLoopLayoutSupported();

    auto make_flags = [](const Shader::ImageResource& r) noexcept -> u32 {
        // Pack the bits that affect ImageInfo/ImageViewInfo construction.
        return (u32(r.is_depth) << 0) | (u32(r.is_atomic) << 1) | (u32(r.is_array) << 2) |
        (u32(r.is_written) << 3) | (u32(r.is_r128) << 4);
    };

    auto make_key = [&](const Shader::ImageResource& r, u32 flags) noexcept -> u64 {
        // Stage + slot + flags. We still memcmp the raw Image descriptor for full match.
        u64 k = stage.pgm_hash;
        k ^= (static_cast<u64>(r.sharp_idx) << 1);
        k ^= (static_cast<u64>(flags) << 32);
        k ^= (static_cast<u64>(static_cast<u32>(stage.l_stage)) << 56);
        return k;
    };

    // PERF(GR2 v17): Epoch-based validity — avoids zeroing 640+ bytes of stamp arrays per call.
    const u32 epoch = ++bind_textures_epoch_;
    // Handle epoch wrap (extremely unlikely, ~every 4 billion BindTextures calls)
    if (epoch == 0) {
        bind_textures_epoch_ = 1;
        // find_image_cache_stamp_ removed; find_texture_cache_stamp_ still
        // active for the FindTextureCached lambda below.
        find_texture_cache_stamp_.fill(0);
    }

    // OPT(v15): De-dup FindTexture() within this BindTextures() call.
    // GR2 frequently binds the same {image_id,type,view_info} multiple times across stages.
    // FindTexture() takes a lock (UpdateImage) and does view lookup, so avoiding duplicates
    // reduces GpuComm overhead (rwlock + Image::FindView + barriers).
    auto& find_texture_cache = find_texture_cache_;
    auto& find_texture_cache_stamp = find_texture_cache_stamp_;
    auto hash_view_info = [](const VideoCore::ImageViewInfo& v) noexcept -> u64 {
        // PERF(GR2 v8): Pack fields into 2-3 mix calls instead of 11.
        // This saves ~8 multiply+xor cycles per image binding in the hot path.
        auto mix = [](u64 h, u64 x) noexcept {
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        };
        u64 h = 0x84222325cbf29ce4ULL;
        // Pack type(8) + format(8) + level(8) + layer(8) + levels(8) + layers(8) + storage(1) = fits in 64 bits
        const u64 packed0 =
            (static_cast<u64>(static_cast<u32>(v.type))) |
            (static_cast<u64>(static_cast<u32>(v.format)) << 8) |
            (static_cast<u64>(v.range.base.level) << 16) |
            (static_cast<u64>(v.range.base.layer) << 24) |
            (static_cast<u64>(v.range.extent.levels) << 32) |
            (static_cast<u64>(v.range.extent.layers) << 40) |
            (static_cast<u64>(v.is_storage ? 1u : 0u) << 48);
        h = mix(h, packed0);
        // Pack swizzle components into one u64
        const u64 packed1 =
            (static_cast<u64>(v.mapping.r)) |
            (static_cast<u64>(v.mapping.g) << 8) |
            (static_cast<u64>(v.mapping.b) << 16) |
            (static_cast<u64>(v.mapping.a) << 24);
        h = mix(h, packed1);
        return h;
    };

    auto FindTextureCached = [&](VideoCore::ImageId image_id,
                                 VideoCore::TextureCache::BindingType type,
                                 const VideoCore::ImageViewInfo& view_info) -> VideoCore::ImageView& {
                                     const u64 k = (static_cast<u64>(image_id.index) * 0x9e3779b97f4a7c15ULL) ^
                                     (static_cast<u64>(static_cast<u32>(type)) << 48) ^
                                     hash_view_info(view_info);
                                     const u32 slot = static_cast<u32>(k) & (find_texture_cache_stamp.size() - 1);
                                     auto& e = find_texture_cache[slot];
                                     // PERF(GR2FORK v1.23): in steady state
                                     // (consecutive same-shader draws binding
                                     // the same images) this cache hits on
                                     // most calls. Hint the warmed-cache
                                     // return.
                                     if (find_texture_cache_stamp[slot] == epoch &&
                                         e.image_id == image_id &&
                                         e.type == type &&
                                         e.view &&
                                         e.view->info == view_info) [[likely]] {
                                         return *e.view;
                                         }
                                         auto& v = texture_cache.FindTexture(image_id, type, view_info);
                                     e = FindTextureCacheEntry{image_id, type, &v};
                                     find_texture_cache_stamp[slot] = epoch;
                                     return v;
                                 };

                                 for (const auto& image_res : stage.images) {
                                     auto tsharp = image_res.GetSharp(stage);

                                     if (texture_cache.IsMeta(tsharp.Address())) [[unlikely]] {
            // PERF(GR2FORK v1.16): defensive sanity log; never expected to
            // fire on shipping titles. Hint the body cold so the surrounding
            // hot binding work stays straight-line.
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }


        if (tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) [[unlikely]] {
            ImageBindingInfo& nb = image_bindings.emplace_back();
            // FIX(GR2FORK): the layout always reserves NumBindings() slots for
            // this ImageResource regardless of whether the sharp resolved to
            // something real. Preserve num_bindings here so the 2nd-pass null
            // emission writes the full array.
            nb.num_bindings =
                static_cast<u8>(std::min<u32>(255u, image_res.NumBindings(stage)));
            continue;
        }

        // FIX(GR2FORK): garbage T# — block-compressed data format combined
        // with a macro-tiled 2D array mode is impossible (the macro-tile
        // size calculator has no path for block textures; ImageInfo's
        // UpdateSize would otherwise hit ASSERT(!props.is_block)). The
        // upstream layer-count clamp catches one shape of garbage T# but
        // not this one — same descriptor often has both. Bail here to
        // keep the layout symmetric with FormatInvalid above.
        {
            const auto bail_array_mode =
                AmdGpu::GetArrayMode(tsharp.GetTileMode());
            if ((bail_array_mode == AmdGpu::ArrayMode::Array2DTiledThin1 ||
                 bail_array_mode == AmdGpu::ArrayMode::Array2DTiledThick) &&
                AmdGpu::IsBlockCoded(tsharp.GetDataFmt())) [[unlikely]] {
                LOG_WARNING(Render_Vulkan,
                            "T# garbage: macro-tiled 2D + block fmt — "
                            "treating as null binding (slot {})",
                            image_bindings.size());
                ImageBindingInfo& nb = image_bindings.emplace_back();
                nb.num_bindings =
                    static_cast<u8>(std::min<u32>(255u, image_res.NumBindings(stage)));
                continue;
            }
        }

        const u32 flags = make_flags(image_res);
        const u64 key = make_key(image_res, flags);
        // PERF(GR2 v16): Better cache slot selection to reduce collisions.
        // The old key & (size-1) used low bits which cluster when pgm_hash is stable.
        // Mix the key first with a multiplicative hash for better distribution.
        const u64 mixed_key = (key ^ (key >> 16)) * 0x9e3779b97f4a7c15ULL;
        CachedImageDescEntry& ce = image_desc_cache_[static_cast<u32>(mixed_key >> 20) & (image_desc_cache_.size() - 1)];

        // PERF(GR2FORK v1.16): The image_desc_cache hit rate is high in
        // steady state (consecutive same-pipeline draws keep submitting the
        // same T# bytes for each slot). Hint the miss-path body so the cache
        // lookup + memcmp lay out as the straight-line path and the rebuild
        // sits in the cold tail.
        if (!(ce.valid && ce.key == key && std::memcmp(&ce.image, &tsharp, sizeof(tsharp)) == 0)) [[unlikely]] {
            ce.key = key;
            ce.image = tsharp;
            ce.desc = VideoCore::TextureCache::ImageDesc(tsharp, image_res);
            ce.valid = true;
        }
        const auto* base_desc = &ce.desc;

        // De-dup FindImage within this call (common when multiple stages alias).
        VideoCore::TextureCache::FindImageWithViewResult found{};
        bool found_cache_hit = false;
        // PERF(GR2FORK v1.43): Step B (find_image_cache intra-call dedup)
        // culled — it never hit (0 hits across millions of iterations of
        // diagnostic measurement). Save the slot computation, branch, and
        // on-miss struct write per iteration.
        {
            // PERF(GR2FORK v1.44): Cross-call pcache key now includes the
            // texture's guest_address. Without this, the SAME shader binding
            // (pgm_hash, sharp_idx, flags, stage) collides on the same
            // pslot regardless of which texture is bound — so when the game
            // rebinds different textures to the same shader slot,
            // image_desc_cache_ rebuilds ce.desc IN PLACE (same pointer,
            // new content) and the cached image_id falls behind. Diagnostic
            // metrics showed 33.66% of iterations addr_mismatched on
            // ValidateCachedFindImage for exactly this reason — 99.98% of
            // validate-fails were pure address mismatches with image_id
            // still allocated and registered. Folding the texture VA into
            // pkey gives each (binding, texture) tuple its own pslot.
            const u64 pkey = key ^ (base_desc->info.guest_address >> 8);
            const u64 pkey_mixed = (pkey ^ (pkey >> 16)) * 0xbf58476d1ce4e5b9ULL;
            const auto pslot = static_cast<u32>(pkey_mixed >> 20) &
            static_cast<u32>(find_image_pcache_.size() - 1);
            auto& pe = find_image_pcache_[pslot];
            // PERF(GR2FORK v1.44): the pcache resolves the iteration ~66%
            // of the time post-fix (~32% direct hit before, plus the recovered
            // 33% formerly-addr_mismatched iterations). [[unlikely]] removed
            // because the Step B cull made every iteration enter this block
            // and the underlying hit rate is no longer rare.
            if (pe.valid && pe.key == pkey && pe.base == base_desc &&
                texture_cache.ValidateCachedFindImage(*base_desc, pe.res.image_id, false)) [[likely]] {
                found = pe.res;
                found_cache_hit = true;
            }
        }
        // PERF(GR2FORK v1.44): full page-table walk fires when the pcache
        // miss. With the key fix, this is no longer ~33% but should be much
        // rarer in steady state. [[unlikely]] retained — the body is the
        // rarest path through this block.
        if (!found_cache_hit) [[unlikely]] {
            found = texture_cache.FindImageWithView(*base_desc, false, false);
        }
        {
            // PERF(GR2FORK v1.43): icache update culled along with the
            // icache lookup above. Only the persistent pcache update remains.
            // PERF(GR2FORK v1.44): pkey identical to lookup-side computation
            // (key XOR'd with the texture's guest_address) so we insert
            // into the pslot that the next lookup will probe.
            const u64 pkey2 = key ^ (base_desc->info.guest_address >> 8);
            const u64 pkey_mixed2 = (pkey2 ^ (pkey2 >> 16)) * 0xbf58476d1ce4e5b9ULL;
            const auto pslot = static_cast<u32>(pkey_mixed2 >> 20) &
            static_cast<u32>(find_image_pcache_.size() - 1);
            find_image_pcache_[pslot] = PersistentFindImageCacheEntry{
                .key = pkey2,
                .base = base_desc,
                .res = found,
                .valid = true,
            };
        }

        auto image_id = found.image_id;

        auto* image = &texture_cache.GetImage(image_id);
        // PERF(GR2FORK v1.24): depth_id is set only for color images that
        // act as the colour half of a depth-stencil aliased pair (used by
        // stencil-attachment redirect logic). Sampled textures and plain
        // colour attachments don't carry one — the redirect is the
        // minority case.
        if (image->depth_id) [[unlikely]] {
            // If this image has an associated depth image, it's a stencil attachment.
            // Redirect the access to the actual depth-stencil buffer.
            image_id = image->depth_id;
            image = &texture_cache.GetImage(image_id);
        }
        // OPT(v18): Track needs_rebind here instead of a separate O(N) scan loop.
        any_needs_rebind |= image->binding.needs_rebind;
        if (image->binding.is_bound) {
            // The image is already bound. In case if it is about to be used as storage we need
            // to force general layout on it.
            image->binding.force_general |= image_res.is_written;
        }
        image->binding.is_bound = 1u;

        ImageBindingInfo& b = image_bindings.emplace_back();
        b.image_id = image_id;
        b.desc = base_desc;
        b.view_mip = static_cast<s16>(found.view_mip);
        b.view_slice = static_cast<s16>(found.view_slice);
        // FIX(GR2FORK): capture NumBindings for the 2nd-loop mip-fallback
        // expansion. DynamicIndex images want >1 consecutive descriptor slots.
        b.num_bindings = static_cast<u8>(std::min<u32>(255u, image_res.NumBindings(stage)));
    }

    // Second pass to re-bind images that were updated after binding.
    //
    // PERF(GR2 v16 + v18): Track whether any image needs rebinding during first pass.
    // Eliminates the O(N) scan loop that previously iterated all image_bindings.
    //
    // PERF(GR2): FindTexture() always calls TextureCache::UpdateImage(), which takes a shared_lock even
    // when the image is clean. GR2 often touches the same image multiple times with different view_info
    // (mip/slice variants), which causes repeated lock traffic. De-dup UpdateImage() per ImageId within
    // this BindTextures() call, and for sampled images call Image::FindView() directly (no extra UpdateImage).
    //
    // PERF(GR2 v17): Use PERSISTENT tick-based cache instead of per-call stamp arrays.
    // This de-dups UpdateImage across multiple draws within the same command buffer tick,
    // avoiding redundant shared_lock acquire/release for images that were already checked this tick.
    const u64 current_tick = scheduler.CurrentTick();
    auto UpdateImageOnce = [&](VideoCore::ImageId id) {
        const u64 k = (static_cast<u64>(id.index) * 0x9e3779b97f4a7c15ULL);
        const u32 slot = static_cast<u32>(k) & (update_image_cache_.size() - 1);
        auto& e = update_image_cache_[slot];
        if (e.image_id == id && e.tick == current_tick) {
            return;
        }
        texture_cache.UpdateImage(id);
        e.image_id = id;
        e.tick = current_tick;
    };
    for (auto& b : image_bindings) {
        const auto* base_desc = b.desc;
        const bool is_storage = base_desc && base_desc->type == VideoCore::TextureCache::BindingType::Storage;

        if (!base_desc || !b.image_id) {
            // FIX(GR2FORK): honor mip-fallback array slot count on null path.
            // Layout still reserves num_bindings descriptors even when the
            // sharp was invalid, so we must emit that many image_infos to
            // keep binding.unified and the descriptorCount-N write aligned.
            const u32 null_count = b.num_bindings ? b.num_bindings : 1u;
            for (u32 null_i = 0; null_i < null_count; ++null_i) {
                if (null_descriptors_supported) {
                    image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
                } else {
                    VideoCore::ImageViewInfo view_info{};
                    if (base_desc) {
                        view_info = base_desc->view_info;
                        if (b.view_mip > 0) {
                            view_info.range.base.level = b.view_mip;
                        }
                        if (b.view_slice > 0) {
                            view_info.range.base.layer = b.view_slice;
                        }
                    }
                    auto& null_image_view =
                    texture_cache.FindTexture(VideoCore::NULL_IMAGE_ID,
                                              VideoCore::TextureCache::BindingType::Texture,
                                              view_info);
                    image_infos.emplace_back(VK_NULL_HANDLE, *null_image_view.image_view,
                                             vk::ImageLayout::eGeneral);
                }
            }
        } else {
            // PERF(GR2 v16): Only check needs_rebind when we know at least one image needs it.
            if (any_needs_rebind) {
                if (auto& old_image = texture_cache.GetImage(b.image_id); old_image.binding.needs_rebind) {
                    old_image.binding = {};
                    const auto rebound = texture_cache.FindImageWithView(*base_desc, false, false);
                    b.image_id = rebound.image_id;
                    b.view_mip = static_cast<s16>(rebound.view_mip);
                    b.view_slice = static_cast<s16>(rebound.view_slice);
                }
            }

            bound_images.emplace_back(b.image_id);

            auto& image = texture_cache.GetImage(b.image_id);

            VideoCore::ImageViewInfo view_info = base_desc->view_info;
            if (b.view_mip > 0) {
                view_info.range.base.level = b.view_mip;
            }
            if (b.view_slice > 0) {
                view_info.range.base.layer = b.view_slice;
            }

            // FIX(GR2FORK): PORT(upstream #4075). For DynamicIndex mip-fallback
            // images, the layout declares descriptorCount = num_mips (one slot
            // per mip). Emit N views + N image_infos here so every array slot
            // gets populated, then emit a single descriptor write below with
            // descriptorCount = num_bindings. Without this, elements >= 1 stay
            // uninitialized and dispatches that index them (e.g. cs_img16 lod=1)
            // fault on RADV -> VK_ERROR_DEVICE_LOST.
            //
            // For the common num_bindings==1 case this loop runs once and is
            // identical to the prior single-emission path.
            const u32 num_bindings = b.num_bindings ? b.num_bindings : 1u;

            for (u32 mip_offset = 0; mip_offset < num_bindings; ++mip_offset) {
                VideoCore::ImageViewInfo mip_view_info = view_info;
                if (num_bindings > 1) {
                    // FIX(GR2FORK): When expanding a mip-fallback array, each
                    // descriptor slot must view exactly one mip level — that's
                    // the whole point of binding-per-mip. The original view_info
                    // may carry extent.levels covering the full chain (e.g. 6),
                    // which combined with a shifted baseMipLevel violates
                    // VUID-VkImageViewCreateInfo-subresourceRange-01718 (base +
                    // count exceeds image mipLevels). Shader indexes per slot,
                    // so single-level views are the correct geometry here.
                    const u32 desired_level = view_info.range.base.level + mip_offset;
                    const u32 max_level = image.info.resources.levels > 0
                                              ? image.info.resources.levels - 1
                                              : 0;
                    mip_view_info.range.base.level =
                        std::min<u32>(desired_level, max_level);
                    mip_view_info.range.extent.levels = 1;
                }

                VideoCore::ImageView* view_ptr = nullptr;
                if (is_storage) {
                    // Keep the original FindTexture() path for storage images (it tags GpuModified + readback).
                    auto& image_view = FindTextureCached(b.image_id, base_desc->type, mip_view_info);
                    view_ptr = &image_view;
                } else {
                    // Sampled images: UpdateImage once per ImageId, then find view without re-taking the lock.
                    UpdateImageOnce(b.image_id);
                    view_ptr = &image.FindView(mip_view_info);
                }

                // Layout transitions only need to happen once per image, but
                // doing them inside the loop is harmless (Transit is a no-op
                // when already in the target state thanks to ARCH-7).
                // Keep transitions driven by the full requested view range
                // (base mip + all expanded mips) so the barrier covers every
                // slot we're about to sample.
                if ((image.binding.force_general || image.binding.is_target) && !image.info.props.is_depth) {
                    image.Transit(attachment_feedback_loop_layout_supported && image.binding.is_target
                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                    : vk::ImageLayout::eGeneral,
                    vk::AccessFlagBits2::eShaderRead |
                                      (image.info.props.is_depth
                                           ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                           : vk::AccessFlagBits2::eColorAttachmentWrite),
                                  {});
                } else {
                    if (is_storage) {
                        // ARCH-7: Skip Transit when already in target state.
                        const auto storage_access = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
                        if (!image.IsInState(vk::ImageLayout::eGeneral, storage_access)) {
                            image.Transit(vk::ImageLayout::eGeneral, storage_access, mip_view_info.range);
                        }
                    } else {
                        const auto new_layout = image.info.props.is_depth
                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                        : vk::ImageLayout::eShaderReadOnlyOptimal;
                        // ARCH-7: Skip Transit for sampled images already in correct state.
                        // In steady-state rendering, most sampled textures stay in ShaderReadOnlyOptimal
                        // between draws. This avoids Transit → GetBarriers function call overhead.
                        if (!image.IsInState(new_layout, vk::AccessFlagBits2::eShaderRead)) {
                            image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead, mip_view_info.range);
                        }
                    }
                }
                image.usage.storage |= is_storage;
                image.usage.texture |= !is_storage;

                image_infos.emplace_back(VK_NULL_HANDLE, *view_ptr->image_view,
                                         image.backing->state.layout);
            }
        }

        // FIX(GR2FORK): The layout always reserves num_bindings slots for
        // this ImageResource (regardless of whether the sharp was valid). Both
        // branches above (null path and real-image path) now emit exactly
        // num_bindings image_infos, so advance binding.unified and emit a
        // single write with descriptorCount = num_bindings. For the common
        // num_bindings == 1 case, behavior is unchanged from upstream.
        const u32 slot_count = b.num_bindings ? b.num_bindings : 1u;

        const u32 dst_binding = binding.unified;
        binding.unified += slot_count;
        const vk::DescriptorType dtype = is_storage ? vk::DescriptorType::eStorageImage
        : vk::DescriptorType::eSampledImage;

        const u32 write_count = slot_count;
        const size_t array_first = image_infos.size() - slot_count;

        const auto ii_sampler =
        reinterpret_cast<u64>(static_cast<VkSampler>(image_infos[array_first].sampler));
        const auto ii_view =
        reinterpret_cast<u64>(static_cast<VkImageView>(image_infos[array_first].imageView));
        const u64 ii_layout = static_cast<u64>(static_cast<u32>(image_infos[array_first].imageLayout));

        // ShouldWriteDescriptor tracks (binding, type, handles) — for multi-slot
        // arrays we only hash the first slot. In practice the tail slots are
        // derived from the same image (consecutive mips of one texture) so
        // they change together; if a more robust invalidation is ever needed,
        // fold write_count into the signature.
        if (ShouldWriteDescriptor(dst_binding, dtype, ii_sampler, ii_view, ii_layout)) {
            auto& w = set_writes.emplace_back();
            w.dstSet = VK_NULL_HANDLE;
            w.dstBinding = dst_binding;
            w.dstArrayElement = 0;
            w.descriptorCount = write_count;
            w.descriptorType = dtype;
            w.pNext = nullptr;
            w.pImageInfo = &image_infos[array_first];
            w.pBufferInfo = nullptr;
            w.pTexelBufferView = nullptr;
        }
    }

    // Sampler fast-path cache. GetSampler() hashes 64 bytes + does a map
    // lookup every call. A direct-mapped cache on the Rasterizer skips both
    // on hit (>90% in steady-state GR2). Samplers are immutable Vulkan
    // objects so caching handles is always safe.
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(stage);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }

        const u64 samp_hash = XXH3_64bits(&ssharp, sizeof(ssharp));
        const u32 samp_slot = static_cast<u32>(samp_hash) & (SamplerCacheSize - 1);
        auto& sc = sampler_cache_[samp_slot];
        vk::Sampler vk_sampler;
        if (sc.valid && sc.hash == samp_hash) {
            vk_sampler = sc.sampler;
        } else {
            vk_sampler = texture_cache.GetSampler(ssharp, regs.ta_bc_base);
            sc.hash = samp_hash;
            sc.sampler = vk_sampler;
            sc.valid = true;
        }

        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        const u32 dst_binding = binding.unified++;
        const vk::DescriptorType dtype = vk::DescriptorType::eSampler;

        const auto ii_sampler =
        reinterpret_cast<u64>(static_cast<VkSampler>(image_infos.back().sampler));
        const auto ii_view =
        reinterpret_cast<u64>(static_cast<VkImageView>(image_infos.back().imageView));
        const u64 ii_layout = static_cast<u64>(static_cast<u32>(image_infos.back().imageLayout));

        if (ShouldWriteDescriptor(dst_binding, dtype, ii_sampler, ii_view, ii_layout)) {
            auto& w = set_writes.emplace_back();
            w.dstSet = VK_NULL_HANDLE;
            w.dstBinding = dst_binding;
            w.dstArrayElement = 0;
            w.descriptorCount = 1;
            w.descriptorType = dtype;
            w.pNext = nullptr;
            w.pImageInfo = &image_infos.back();
            w.pBufferInfo = nullptr;
            w.pTexelBufferView = nullptr;
        }
    }
}


RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline,
                                       const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // PERF(GR2FORK): stamp-skip fast path. See BeginRenderingCache doc in
    // vk_rasterizer.h. Verifies pipeline+stamp match, no CMASK clear flag
    // armed since populate, image layouts still match, and no image has
    // needs_rebind set (texture cache merge/expand can free the cached
    // image_id and assign a new one — must fall through to the rebind
    // path in that case).
    //
    // Gated on Config::isBeginRenderingCacheEnabled() — on by default; set
    // beginRenderingCacheEnable=false in [Vulkan] to disable if shadow flicker
    // or other rendering artifacts return.
    const bool br_cache_enabled = Config::isBeginRenderingCacheEnabled();
    if (br_cache_enabled) {
        // Phase 1D-pre-C: read pipeline stamp from the snapshot.
        const u64 cur_stamp = regs.gfx_pipeline_stamp;
        const u64 cur_tick = scheduler.CurrentTick();
        if (br_cache_.valid && br_cache_.pipeline == pipeline &&
            br_cache_.stamp == cur_stamp && br_cache_.tick == cur_tick) {
            bool ok = true;
            // Check clear flags + layouts + needs_rebind on color attachments.
            for (u32 i = 0; i < br_cache_.cb_count && ok; ++i) {
                const auto& e = br_cache_.cb_data[i];
                // PERF(GR2FORK v1.20): meta_addr is set only for surfaces
                // with HTILE/CMASK/FMASK active — the majority of cached
                // CB attachments don't carry one.
                if (e.meta_addr &&
                    texture_cache.IsMetaCleared(e.meta_addr, e.slice)) [[unlikely]] {
                    ok = false;
                    break;
                }
                // PERF(GR2FORK v1.20): cached entries almost always carry
                // an image_id — empty slots happen only for masked-out
                // colour buffers, which are uncommon at steady state.
                if (e.image_id) [[likely]] {
                    auto& img = texture_cache.GetImage(e.image_id);
                    // PERF(GR2FORK v1.20): within a single render pass the
                    // attachment image's layout and rebind flag are stable;
                    // a mismatch fires on cache-merge / layout transition,
                    // which is the slow path.
                    if (img.binding.needs_rebind ||
                        img.backing->state.layout != e.expected_layout) [[unlikely]] {
                        ok = false;
                        break;
                    }
                }
            }
            // Check depth/stencil attachment if present.
            if (ok && br_cache_.has_db_attachment) {
                const auto& e = br_cache_.db_data;
                if (e.meta_addr &&
                    texture_cache.IsMetaCleared(e.meta_addr, e.slice)) [[unlikely]] {
                    ok = false;
                }
                if (ok && e.image_id) [[likely]] {
                    auto& img = texture_cache.GetImage(e.image_id);
                    if (img.binding.needs_rebind ||
                        img.backing->state.layout != e.expected_layout) [[unlikely]] {
                        ok = false;
                    }
                }
            }
            if (ok) [[likely]] {
                // PERF(GR2 v1.22): wire up the bt_replay counter that's been
                // declared but unused. Counts BeginRendering cache hits —
                // when this is high, the renderpass setup work is being
                // skipped successfully across consecutive same-pipeline
                // draws. (Counter named "bt_replay" historically but
                // measures BeginRendering replay rate.)
                //
                // PERF(GR2FORK v1.15): The bt_replay counter confirms this
                // is the dominant exit; hint it for code-layout.
                GR2_INSTR_ON_BT_REPLAY();
                attachment_feedback_loop = br_cache_.attachment_feedback_loop;
                return br_cache_.state;
            }
        }
    }

    attachment_feedback_loop = false;
    // PERF(GR2FORK): slow path running — invalidate br_cache_ until repopulated
    // at end of function. Inline writes during the cb/db loops below will fill
    // br_cache_.cb_data / br_cache_.db_data; the trailing populate block will
    // build the eLoad-forced state copy and re-set valid.
    br_cache_.valid = false;
    br_cache_.cb_count = 0;
    br_cache_.has_db_attachment = false;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u32>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);

    // OPT: Reuse the tick-based UpdateImage de-dup cache so that render targets
    // already validated by BindTextures() during this draw don't re-acquire the
    // texture cache shared_lock. This saves ~0.3-0.5% of GpuComm time.
    const u64 current_tick = scheduler.CurrentTick();
    auto UpdateImageDedup = [&](VideoCore::ImageId id) {
        const u64 k = (static_cast<u64>(id.index) * 0x9e3779b97f4a7c15ULL);
        const u32 slot = static_cast<u32>(k) & (update_image_cache_.size() - 1);
        auto& e = update_image_cache_[slot];
        if (e.image_id == id && e.tick == current_tick) {
            return;
        }
        texture_cache.UpdateImage(id);
        e.image_id = id;
        e.tick = current_tick;
    };

    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        // PERF(GR2FORK): pre-zero this cb's cache slot so the `continue` path
        // below leaves a well-defined empty entry (image_id=0, meta_addr=0,
        // expected_layout=eUndefined). The hit-path checks gate on these.
        br_cache_.cb_data[cb] = {};
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        // OPT: Use tick-dedup to skip re-acquiring shared_lock if BindTextures
        // already validated this image during the same draw call.
        UpdateImageDedup(image_id);
        image->SetBackingSamples(key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            image->Transit(instance.IsAttachmentFeedbackLoopLayoutSupported()
                               ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                               : vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eColorAttachmentWrite, {});
            attachment_feedback_loop = true;
        } else {
            // ARCH-5: Skip Transit when render target is already in correct state.
            // Consecutive draws with the same render targets hit this ~95% of the time.
            const auto ca_access = vk::AccessFlagBits2::eColorAttachmentWrite |
                                   vk::AccessFlagBits2::eColorAttachmentRead;
            if (!image->IsInState(vk::ImageLayout::eColorAttachmentOptimal, ca_access)) {
                image->Transit(vk::ImageLayout::eColorAttachmentOptimal, ca_access,
                               desc.view_info.range);
            }
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);
        state.color_attachments[cb] = {
            .imageView = *image_view.image_view,
            .imageLayout = image->backing->state.layout,
            .loadOp = is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue =
                is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{},
        };
        image->usage.render_target = 1u;
        // PERF(GR2FORK): capture for br_cache_ population. expected_layout
        // is read AFTER Transit so it reflects the post-transition layout
        // that subsequent stamp-matched calls must verify is still in place.
        {
            auto& e = br_cache_.cb_data[cb];
            e.image_id = image_id;
            e.slice = slice;
            e.meta_addr = col_buf.CmaskAddress();
            e.expected_layout = image->backing->state.layout;
        }
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear = regs.depth_render_control.depth_clear_enable ||
                                    texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;

        // FIX(GR2FORK): US disc GR2 (CUSA03694) crashes in Nevi Hand encounters
        // with VK_ERROR_DEVICE_LOST. Validation layer (VUID-06886/06887) shows
        // the same depth+stencil image bound in DEPTH_STENCIL_READ_ONLY_OPTIMAL
        // while the current pipeline has depthWriteEnable=VK_TRUE and/or
        // stencilTestEnable=VK_TRUE with non-zero writeMask and non-KEEP ops.
        //
        // Prior logic picked the layout purely from view_info.is_storage,
        // ignoring whether the current draw actually writes depth or stencil.
        // Nevi deferred-shading Z-prepass + forward stencil masking reuses the
        // same image: an earlier draw binds it read-only for sampling, a later
        // draw writes it — but because is_storage was false on the later bind,
        // the layout stayed DEPTH_STENCIL_READ_ONLY_OPTIMAL. RADV hangs on the
        // mismatch; other drivers may silently corrupt depth/stencil.
        //
        // Correct decision tree: need writable if is_storage, OR if depth
        // writes are on, OR if stencil writes are on. The stencil writemask
        // lives in stencil_ref_front (mirrored to stencil_ref_back).
        const bool stencil_test_enabled = regs.depth_control.stencil_enable;
        const bool stencil_writes_enabled =
            stencil_test_enabled && regs.stencil_ref_front.stencil_write_mask != 0;
        const bool depth_writes_enabled =
            regs.depth_control.depth_enable && regs.depth_control.depth_write_enable;
        const bool needs_writable_layout =
            desc.view_info.is_storage || depth_writes_enabled || stencil_writes_enabled;

        const auto new_layout = needs_writable_layout
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        const auto ds_access = vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                               vk::AccessFlagBits2::eDepthStencilAttachmentRead;
        // ARCH-5: Skip Transit when depth target is already in correct state.
        if (!image.IsInState(new_layout, ds_access)) {
            image.Transit(new_layout, ds_access, desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.has_depth = regs.depth_buffer.DepthValid();
        state.has_stencil = regs.depth_buffer.StencilValid();
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);
        if (state.has_depth) {
            state.depth_attachment = {
                .imageView = *image_view.image_view,
                .imageLayout = image.backing->state.layout,
                .loadOp =
                    is_depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{.depthStencil = {.depth = regs.depth_clear}},
            };
        }
        if (state.has_stencil) {
            state.stencil_attachment = {
                .imageView = *image_view.image_view,
                .imageLayout = image.backing->state.layout,
                .loadOp =
                    is_stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{.depthStencil = {.stencil = regs.stencil_clear}},
            };
        }

        image.usage.depth_target = true;
        // PERF(GR2FORK): capture for br_cache_ population.
        {
            auto& e = br_cache_.db_data;
            e.image_id = image_id;
            e.slice = slice;
            e.meta_addr = htile_address;
            e.expected_layout = image.backing->state.layout;
            br_cache_.has_db_attachment = true;
        }
    }

    if (state.num_layers == std::numeric_limits<u32>::max()) {
        state.num_layers = 1;
    }

    // OPT: Pre-compute hash for fast equality rejection in BeginRendering.
    state.ComputeHash();

    // PERF(GR2FORK): populate br_cache_. Build the eLoad-forced state copy that
    // subsequent stamp-matched calls will return — ensures we never replay a
    // loadOp=eClear (memory entry #26 / magenta-blob trap). The current call
    // returns the as-built `state` (which may have eClear on the first draw
    // after a clear); only the cached COPY has loadOps forced to eLoad.
    //
    // BUG-FIX: refuse to cache when register-driven depth/stencil clear is
    // active. CMASK clears (color) and HTILE clears (depth/stencil) are
    // consumed-on-read via TouchMeta, so subsequent stamp-equal draws would
    // naturally emit eLoad — eLoad-forcing matches what the slow path would
    // produce. But `regs.depth_render_control.{depth,stencil}_clear_enable`
    // are REGISTER-DRIVEN flags that persist until the game changes them.
    // If we cache an eLoad version while these are set, subsequent stamp-equal
    // draws would (slow-path) emit eClear but our cache returns eLoad —
    // wiping out the clear and preserving stale depth from the previous draw.
    // This was the cause of permanently flickering shadows in GR2: shadow
    // cascade passes hold depth_clear_enable=true across stamp-equal draws.
    //
    // Gated on Config::isBeginRenderingCacheEnabled() — when off, valid stays
    // false (set at slow-path entry) so subsequent calls never hit and the
    // function always runs the slow path.
    if (br_cache_enabled) {
        const bool reg_clear_active =
            regs.depth_render_control.depth_clear_enable ||
            regs.depth_render_control.stencil_clear_enable;
        if (!reg_clear_active) {
            br_cache_.cb_count = state.num_color_attachments;
            br_cache_.attachment_feedback_loop = attachment_feedback_loop;
            br_cache_.pipeline = pipeline;
            // Phase 1D-pre-C: stamp from snapshot; matches the same value
            // the fast-path comparison above used.
            br_cache_.stamp = regs.gfx_pipeline_stamp;
            br_cache_.tick = scheduler.CurrentTick();

            RenderState eload_state = state;
            for (u32 cb = 0; cb < eload_state.num_color_attachments; ++cb) {
                eload_state.color_attachments[cb].loadOp = vk::AttachmentLoadOp::eLoad;
                eload_state.color_attachments[cb].clearValue = vk::ClearValue{};
            }
            if (eload_state.has_depth) {
                eload_state.depth_attachment.loadOp = vk::AttachmentLoadOp::eLoad;
                eload_state.depth_attachment.clearValue = vk::ClearValue{};
            }
            if (eload_state.has_stencil) {
                eload_state.stencil_attachment.loadOp = vk::AttachmentLoadOp::eLoad;
                eload_state.stencil_attachment.clearValue = vk::ClearValue{};
            }
            eload_state.ComputeHash();
            br_cache_.state = eload_state;
            br_cache_.valid = true;
        }
        // else: leave br_cache_.valid = false (set at slow-path entry) — no
        // cache hit possible until a draw with no register-driven clear runs.
    }

    return state;
}

void Rasterizer::Resolve(const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // Phase 1D-pre-C: extent hints from snapshot.
    const auto& mrt0_hint = regs.last_cb_extent[0];
    const auto& mrt1_hint = regs.last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    // Phase 1D-pre-D: internal markers → inline helpers.
    DoScopeMarkerBeginInline(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                          regs.color_buffers[0].Address(),
                                          regs.color_buffers[1].Address()),
                              /*from_guest=*/false);
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    DoScopeMarkerEndInline(/*from_guest=*/false);
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil,
                                  const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // Phase 1D-pre-C: extent hint from snapshot.
    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), regs.last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), regs.last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = regs.depth_view.slice_start;
    sub_range.extent.layers = regs.depth_view.NumSlices() - sub_range.base.layer;

    // Phase 1D-pre-D: internal markers → inline helpers.
    DoScopeMarkerBeginInline(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()),
        /*from_guest=*/false);

    read_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                       sub_range);
    write_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        sub_range);

    auto aspect_mask = vk::ImageAspectFlags(0);
    if (is_depth) {
        aspect_mask |= vk::ImageAspectFlagBits::eDepth;
    }
    if (is_stencil) {
        aspect_mask |= vk::ImageAspectFlagBits::eStencil;
    }

    vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .dstOffset = {0, 0, 0},
        .extent = {write_image.info.size.width, write_image.info.size.height, 1},
    };
    scheduler.PrimaryCommandBuffer().copyImage(read_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        write_image.GetImage(),
                                        vk::ImageLayout::eTransferDstOptimal, region);

    DoScopeMarkerEndInline(/*from_guest=*/false);
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    // Phase 1D-pre-D: marker-route. Body moves to DoFillBufferFromIntent.
    // FIFO order with surrounding draws is preserved — the assembler
    // processes intents in PM4-emit order.
    DrawIntent intent;
    intent.type = DrawIntent::Type::FillBuffer;
    intent.fill_buffer.address = address;
    intent.fill_buffer.num_bytes = num_bytes;
    intent.fill_buffer.value = value;
    intent.fill_buffer.is_gds = static_cast<u8>(is_gds ? 1 : 0);
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoFillBufferFromIntent(const DrawIntent& intent) {
    buffer_cache.FillBuffer(intent.fill_buffer.address, intent.fill_buffer.num_bytes,
                            intent.fill_buffer.value,
                            intent.fill_buffer.is_gds != 0);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    // Phase 1D-pre-D: marker-route. Body moves to DoCopyBufferFromIntent.
    DrawIntent intent;
    intent.type = DrawIntent::Type::CopyBuffer;
    intent.copy_buffer.dst = dst;
    intent.copy_buffer.src = src;
    intent.copy_buffer.num_bytes = num_bytes;
    intent.copy_buffer.dst_gds = static_cast<u8>(dst_gds ? 1 : 0);
    intent.copy_buffer.src_gds = static_cast<u8>(src_gds ? 1 : 0);
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoCopyBufferFromIntent(const DrawIntent& intent) {
    buffer_cache.CopyBuffer(intent.copy_buffer.dst, intent.copy_buffer.src,
                            intent.copy_buffer.num_bytes,
                            intent.copy_buffer.dst_gds != 0,
                            intent.copy_buffer.src_gds != 0);
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.InvalidateMemory(addr, size);
    // Underlying texture/buffer data changed — caches are stale.
    binding_skip_cache_.valid = false;
    rt_cache_.valid = false;
    br_cache_.valid = false;
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size);
    return true;
}

// PERF(GR2FORK release_v2_5): IsMapped is called once per page-fault on
// the SIGNAL HANDLER path (Rasterizer::InvalidateMemory and ReadMemory,
// both via PageManager::Impl::GuestFaultSignalHandler on
// GAME_MainThread). At ~35K faults/sec on GR2 each call cost ~100–230
// cycles before this patch (RecursiveSharedLock CAS + boost::icl
// contains traversal). Page faults cluster on adjacent pages of the
// same tracked GPU buffer — a 1 MB buffer is 256 pages, of which the
// first fault triggers IsMapped's full lookup and the next 255 are
// expected to fall within the same interval. A small per-thread cache
// of the most-recent positive intervals turns those 255 follow-on
// queries into a generation-counter check + linear scan over <kCacheSize>
// (addr, limit) pairs — ~30–45 cycles, bypassing the lock entirely.
//
// Correctness:
//  - Generation counter mapped_ranges_gen_ is bumped (release) by
//    Map/UnmapMemory AFTER they mutate mapped_ranges under the lock.
//  - IsMapped loads gen (acquire) before consulting cache. If gen
//    advanced beyond cached_gen, cache is dropped and we fall through
//    to the locked path. Release-acquire pairing ensures that any
//    cache hit at gen=G corresponds to an interval that was in
//    mapped_ranges at gen=G; since gen is monotonic and cache_gen==G
//    means we haven't observed any mutation, mapped_ranges still
//    contains that interval. Cache hit returns are therefore correct.
//  - Cache MISS still goes through RecursiveSharedLock + boost::icl,
//    so the slow path is bit-identical to pre-v2_5.
//  - tls_cache is per-thread; only the owning thread reads or writes
//    it. No cross-thread synchronization needed for the cache itself.
//
// Signal-safety: thread_local POD storage, no allocation, no
// non-async-signal-safe calls on the cache-hit path. Falls through to
// the same RecursiveSharedLock path the pre-v2_5 code used in signal
// context, so no new signal-safety burden.
bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }

    struct CacheEntry {
        VAddr base;   // [base, limit) — 0,0 means empty slot
        VAddr limit;
    };
    static constexpr size_t kCacheSize = 4;
    thread_local std::array<CacheEntry, kCacheSize> tls_cache{};
    thread_local u64 tls_gen = ~u64{0};

    const VAddr query_end = addr + size;
    const u64 cur_gen = mapped_ranges_gen_.load(std::memory_order_acquire);

    if (cur_gen == tls_gen) [[likely]] {
        // Cache valid for this generation. Linear scan over kCacheSize
        // entries — branch-friendly, fits in one cacheline.
        for (const auto& e : tls_cache) {
            if (addr >= e.base && query_end <= e.limit) {
                return true;
            }
        }
    } else {
        // Generation advanced: drop cache before falling through. We
        // intentionally don't memset to 0 — overwriting on insert is
        // fine and avoids the cost when most entries will be replaced
        // anyway. Reset tls_gen so future hits compare against the
        // current epoch.
        tls_cache.fill({0, 0});
        tls_gen = cur_gen;
    }

    // Cache miss — full lookup. Do find(addr) instead of contains(range)
    // because the iterator gives us the containing interval bounds for
    // free, which we cache for future hits in the same neighborhood.
    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    auto it = mapped_ranges.find(addr);
    if (it == mapped_ranges.end()) {
        return false;
    }
    const VAddr lo = it->lower();
    const VAddr hi = it->upper();
    if (query_end > hi) {
        // Address is in a tracked interval but the requested range
        // straddles its upper bound. Don't cache (cache entries
        // represent ranges that ARE fully contained).
        return false;
    }

    // Insert at slot 0, shift older entries down, evict slot kCacheSize-1.
    // Round-robin replacement is good enough — most workloads have only
    // a few simultaneously hot intervals.
    for (size_t i = kCacheSize - 1; i > 0; --i) {
        tls_cache[i] = tls_cache[i - 1];
    }
    tls_cache[0] = CacheEntry{lo, hi};
    return true;
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    // PERF(GR2FORK release_v2_5): bump after the mutation is committed
    // and visible. Release order pairs with IsMapped's acquire load so
    // any thread observing the new gen value also observes the new
    // interval.
    mapped_ranges_gen_.fetch_add(1, std::memory_order_release);
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    binding_skip_cache_.valid = false;
    rt_cache_.valid = false;
    br_cache_.valid = false;
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    // PERF(GR2FORK release_v2_5): bump after the mutation is committed.
    // See MapMemory for ordering rationale.
    mapped_ranges_gen_.fetch_add(1, std::memory_order_release);
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed,
                                    const AmdGpu::LiverpoolRegsSnapshot& regs) const {
    auto& dynamic_state = scheduler.GetDynamicState();

    // Skip all 6 sub-functions when no dynamic-state register changed since last draw.
    // Each sub-function reads dozens of regs and does comparison/computation that Commit
    // would then discover produced zero dirty flags. ~236 lines of work saved per hit.
    //
    // PERF(GR2FORK v1.15): Within a render pass, viewport/scissor/blend/depth state
    // is essentially never re-emitted between draws — the dirty flag stays false
    // for the bulk of consecutive same-pipeline draws. Hint the optimizer.

    // Phase 1D-pre-C: read the dirty bit from the snapshot. PM4 cleared
    // the live `liverpool->dynamic_dirty_` inside `Liverpool::CaptureSnapshot()`
    // under sole-writer ownership, so the prior `liverpool->ClearDynamicDirty()`
    // call below is gone — the snapshot's `regs.dynamic_dirty` is a frozen
    // "was dirty AT capture" view; consuming it doesn't need to clear
    // anything.
    if (!regs.dynamic_dirty) [[likely]] {
        dynamic_state.Commit(instance, scheduler.PrimaryCommandBuffer());
        return;
    }

    UpdateViewportScissorState(regs);
    UpdateDepthStencilState(regs);
    UpdatePrimitiveState(is_indexed, regs);
    UpdateRasterizationState(regs);
    UpdateColorBlendingState(pipeline, regs);

    dynamic_state.Commit(instance, scheduler.PrimaryCommandBuffer());
}

void Rasterizer::UpdateViewportScissorState(const AmdGpu::LiverpoolRegsSnapshot& regs) const {
    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState(const AmdGpu::LiverpoolRegsSnapshot& regs) const {
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        dynamic_state.SetStencilReferences(front.stencil_test_val, back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed,
                                      const AmdGpu::LiverpoolRegsSnapshot& regs) const {
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto prim_restart = (regs.enable_primitive_restart & 1) != 0;
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState(const AmdGpu::LiverpoolRegsSnapshot& regs) const {
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline,
                                          const AmdGpu::LiverpoolRegsSnapshot& regs) const {
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

// Phase 1D-pre-D: in-line helpers. Same body as the legacy direct
// ScopeMarker* methods. Internal Rasterizer-side callers use these so the
// markers bracket their surrounding data-plane work in-line, without a
// trip through the bundle queue.
void Rasterizer::DoScopeMarkerBeginInline(const std::string_view& str, bool from_guest) {
    if ((from_guest && !Config::getVkGuestMarkersEnabled()) ||
        (!from_guest && !Config::getVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.PrimaryCommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::DoScopeMarkerEndInline(bool from_guest) {
    if ((from_guest && !Config::getVkGuestMarkersEnabled()) ||
        (!from_guest && !Config::getVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.PrimaryCommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::DoScopedMarkerInsertInline(const std::string_view& str, u32 color,
                                            bool from_guest) {
    if ((from_guest && !Config::getVkGuestMarkersEnabled()) ||
        (!from_guest && !Config::getVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.PrimaryCommandBuffer();
    // ScopedMarkerInsert (no color) packs color=0 here; default
    // vk::DebugUtilsLabelEXT::color is {0,0,0,0}, so color=0 produces an
    // observable-identical Vulkan call to the legacy non-color path. The
    // ScopedMarkerInsertColor caller passes the real RGBA-packed u32.
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

namespace {
// Phase 1D-pre-D: pack a string_view into the intent's inline char buffer.
// Truncates to (DrawIntent::kScopeMarkerInlineLen - 1) chars; writes a
// null terminator at the end. Producer-side helper — runs on whichever
// thread calls Rasterizer::ScopeMarker*.
void PackInlineString(char (&dst)[DrawIntent::kScopeMarkerInlineLen],
                       const std::string_view& src) noexcept {
    constexpr std::size_t kMax = DrawIntent::kScopeMarkerInlineLen - 1;
    const std::size_t n = std::min(src.size(), kMax);
    if (n > 0) {
        std::memcpy(dst, src.data(), n);
    }
    dst[n] = '\0';
}
} // namespace

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    // Phase 1D-pre-D: marker-route. The string is copied (truncated) into
    // the intent's inline buffer at packing time so the consumer doesn't
    // need to reach back to PM4-packet memory. Body moves to
    // DoScopeMarkerBeginInline (also reachable directly by internal
    // callers).
    DrawIntent intent;
    intent.type = DrawIntent::Type::ScopeMarkerBegin;
    PackInlineString(intent.scope_marker.str, str);
    intent.scope_marker.color = 0;
    intent.scope_marker.from_guest = static_cast<u8>(from_guest ? 1 : 0);
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoScopeMarkerBeginFromIntent(const DrawIntent& intent) {
    DoScopeMarkerBeginInline(std::string_view{intent.scope_marker.str},
                              intent.scope_marker.from_guest != 0);
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    // Phase 1D-pre-D: marker-route.
    DrawIntent intent;
    intent.type = DrawIntent::Type::ScopeMarkerEnd;
    intent.scope_marker_end.from_guest = static_cast<u8>(from_guest ? 1 : 0);
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoScopeMarkerEndFromIntent(const DrawIntent& intent) {
    DoScopeMarkerEndInline(intent.scope_marker_end.from_guest != 0);
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    // Phase 1D-pre-D: marker-route. Unified with the (color) variant via
    // color=0 — the Vulkan call is observable-equivalent.
    DrawIntent intent;
    intent.type = DrawIntent::Type::ScopedMarkerInsert;
    PackInlineString(intent.scope_marker.str, str);
    intent.scope_marker.color = 0;
    intent.scope_marker.from_guest = static_cast<u8>(from_guest ? 1 : 0);
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    // Phase 1D-pre-D: marker-route. Same intent type as no-color; color
    // is carried in the payload.
    DrawIntent intent;
    intent.type = DrawIntent::Type::ScopedMarkerInsert;
    PackInlineString(intent.scope_marker.str, str);
    intent.scope_marker.color = color;
    intent.scope_marker.from_guest = static_cast<u8>(from_guest ? 1 : 0);
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoScopedMarkerInsertFromIntent(const DrawIntent& intent) {
    DoScopedMarkerInsertInline(std::string_view{intent.scope_marker.str},
                                intent.scope_marker.color,
                                intent.scope_marker.from_guest != 0);
}

// Phase 1D-pre-E: PresenterRecord consumer. Calls the type-erased closure
// stored in the intent and lets the closure self-destruct. The state
// pointer + invoke_and_destroy function pointer were synthesized by the
// templated producer-side `PushPresenterRecord<F>`. After this returns
// the heap allocation is released.
void Rasterizer::DoPresenterRecordFromIntent(const DrawIntent& intent) {
    intent.presenter_record.invoke_and_destroy(intent.presenter_record.state);
}


} // namespace Vulkan
