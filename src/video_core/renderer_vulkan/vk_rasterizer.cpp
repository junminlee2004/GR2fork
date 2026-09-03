// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>

#include <xxhash.h>

#include "common/rdtsc.h"

#include "common/debug.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/skipcache/skipcache.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Vulkan {

namespace Skipcache = VideoCore::Skipcache;

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    auto& skipcache = Skipcache::Framework::Instance();
    skipcache.Init(static_cast<Skipcache::Mode>(EmulatorSettings.GetAdaptiveSkipCachesMode()));
    skipcache.RegisterInvalidate(&Rasterizer::BrInvalidateThunk, this);
    br_readback_gate_ = EmulatorSettings.IsReadbackLinearImagesEnabled();
    batch_copy_lock_ = EmulatorSettings.IsGuestCopyLockBatch();
    elide_findbuffer_ = EmulatorSettings.IsStreamFindBufferElide();
    bind_prefetch_ = EmulatorSettings.IsBindLinePrefetch();
    bind_noop_ = texture_cache.BindNoopMemo();
    if (EmulatorSettings.IsGuestCopyHoldSegment()) {
        segment_copy_hold_ = true;
        pipeline_cache.SetPreCompileHook(&Rasterizer::PreCompileThunk, this);
    }
    deferred_read_arm_ = EmulatorSettings.IsDeferredReadArm();
    if (deferred_read_arm_) {
        scheduler.SetSubmitHook(&Rasterizer::PreSubmitThunk, this);
    }
    if (const u32 interval = EmulatorSettings.GetFlushDrawInterval(); interval != 0) {
        flush_draw_interval_ = std::max<u32>(interval, 64);
    }
    // The register stamp is armed once from the boot value of the skip-cache
    // mode; enabling the framework later from the settings dialog leaves it
    // pinned, and a frozen stamp compares every draw register-identical.
    dyn_memo_enabled_ =
        EmulatorSettings.IsDynStateMemo() &&
        EmulatorSettings.GetAdaptiveSkipCachesMode() != AdaptiveSkipCachesMode::SkipCachesDisabled;
    dyn_class_stamp_ = dyn_memo_enabled_ && EmulatorSettings.IsDynStateStamp();
    // Calibrated once on the boot path: EstimateRDTSCFrequency sleeps ~101ms
    // to measure, and calling it from the per-300-frame telemetry block put a
    // deterministic two-frame stall on the GPU command thread every ~17s.
    tsc_hz_ = Common::EstimateRDTSCFrequency();
    // Mode, not a worker count: 0 disables the lane, 1 is the unsafe fast
    // path (titles that never unmap mid-play), 2 and above the hardened one.
    // Both modes run two workers, the measured knee.
    const u32 lane_mode = EmulatorSettings.GetStreamCopyWorkers();
    if (lane_mode != 0) {
        VideoCore::StreamCopyLane::Instance().Init(2, lane_mode >= 2);
        Core::MemoryManager::RegisterUnmapDrain(
            [](void*) { VideoCore::StreamCopyLane::Instance().DrainRemote(); }, nullptr);
    }
}

Rasterizer::~Rasterizer() {
    VideoCore::StreamCopyLane::Instance().Shutdown();
}

void Rasterizer::CpSync() {
    scheduler.EndRendering();
    auto cmdbuf = scheduler.CommandBuffer();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlagBits::eByRegion, ib_barrier, {}, {});
}

void Rasterizer::BindPipelineDedup(vk::PipelineBindPoint point, vk::Pipeline handle) {
    const u64 tick = scheduler.CurrentTick();
    const size_t idx = point == vk::PipelineBindPoint::eCompute ? 1 : 0;
    if (!Skipcache::Framework::Instance().Active()) {
        scheduler.CommandBuffer().bindPipeline(point, handle);
        return;
    }
    const u64 fgen = Skipcache::Framework::Instance().ForeignPipelineGen(idx);
    if (tick == last_bound_tick_ && last_bound_pipeline_[idx] == handle &&
        last_bound_pipeline_gen_[idx] == fgen) {
        return; // same handle already bound on this command buffer
    }
    if (tick != last_bound_tick_) {
        last_bound_pipeline_ = {};
        last_bound_tick_ = tick;
    }
    last_bound_pipeline_[idx] = handle;
    last_bound_pipeline_gen_[idx] = fgen;
    scheduler.CommandBuffer().bindPipeline(point, handle);
}

bool Rasterizer::FilterDraw() {
    // The true verdict is a pure function of the registers; the false paths
    // perform real work (fast clear elimination, resolves, depth copies) and
    // must always re-execute, so only true is memoized.
    if (Skipcache::Framework::Instance().Active()) {
        const u64 stamp = liverpool->GetGfxStateStamp();
        if (filter_true_stamp_ == stamp) {
            return true;
        }
        const bool result = FilterDrawSlow();
        filter_true_stamp_ = result ? stamp : 0;
        return result;
    }
    return FilterDrawSlow();
}

bool Rasterizer::FilterDrawSlow() {
    const auto& regs = liverpool->regs;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
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
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

bool Rasterizer::RtMemoProbe(const GraphicsPipeline* pipeline, u64 reg_stamp, u64 tex_gen,
                             u64 pipe_gen) {
    using namespace VideoCore::Skipcache;
    auto& ctr = Skipcache::Framework::Instance().Counters(CacheId::PrepareRt);
    const auto& m = rt_memo_;
    if (!m.valid) {
        ++ctr.miss_cold;
        return false;
    }
    if (m.pipeline != pipeline) {
        ++ctr.miss_key;
        return false;
    }
    if (m.reg_stamp != reg_stamp) {
        ++ctr.miss_gen[LaneReg];
        return false;
    }
    if (m.tex_gen != tex_gen) {
        ++ctr.miss_gen[LaneTex];
        return false;
    }
    if (m.pipe_gen != pipe_gen) {
        ++ctr.miss_gen[LanePipe];
        return false;
    }
    for (u32 cb = 0; cb < m.cb_count; ++cb) {
        if (!m.cb_id[cb]) {
            continue;
        }
        const auto& image = texture_cache.GetImage(m.cb_id[cb]);
        if (image.image_uid != m.cb_uid[cb] ||
            False(image.flags & VideoCore::ImageFlagBits::Registered)) {
            ++ctr.veto[0];
            return false;
        }
    }
    if (m.db_id) {
        const auto& image = texture_cache.GetImage(m.db_id);
        if (image.image_uid != m.db_uid ||
            False(image.flags & VideoCore::ImageFlagBits::Registered)) {
            ++ctr.veto[0];
            return false;
        }
    }
    return true;
}

void Rasterizer::RtMemoReplay() {
    // cb_descs/db_desc still hold the previous identical construction,
    // including any overlap view rebase FindImage applied to them. Only the
    // per-draw marking is re-established.
    const auto& m = rt_memo_;
    for (u32 cb = 0; cb < m.cb_count; ++cb) {
        cb_descs[cb].first = m.cb_id[cb];
        if (m.cb_id[cb]) {
            bound_images.emplace_back(m.cb_id[cb]);
            texture_cache.GetImage(m.cb_id[cb]).binding.is_target = 1u;
        }
    }
    db_desc.first = m.db_id;
    if (m.db_id) {
        bound_images.emplace_back(m.db_id);
        texture_cache.GetImage(m.db_id).binding.is_target = 1u;
    }
}

void Rasterizer::RtMemoVerifyPopulate(bool would_hit, const GraphicsPipeline* pipeline,
                                      u64 reg_stamp, u64 tex_gen, u64 pipe_gen) {
    using namespace VideoCore::Skipcache;
    auto& sc = Skipcache::Framework::Instance();
    constexpr auto kCache = CacheId::PrepareRt;
    const u32 cb_count = std::bit_width(pipeline->GetGraphicsKey().mrt_mask);
    if (would_hit && sc.GetState(kCache) != State::Learning) {
        const auto& m = rt_memo_;
        bool same = m.cb_count == cb_count && m.db_id == db_desc.first;
        for (u32 cb = 0; same && cb < cb_count; ++cb) {
            same = m.cb_id[cb] == cb_descs[cb].first;
        }
        if (same) {
            sc.RecordVerifyClean(kCache);
        } else if (sc.Gens().tex_gen.load(std::memory_order_acquire) != tex_gen) {
            sc.RecordVerifyAborted(kCache);
            rt_memo_.valid = false;
        } else {
            sc.RecordDivergence(kCache, "render target resolution mismatch");
            rt_memo_.valid = false;
        }
        return;
    }
    if (would_hit) {
        return; // Learning observe-only
    }
    auto& m = rt_memo_;
    m.valid = false;
    m.cb_count = cb_count;
    for (u32 cb = 0; cb < cb_count; ++cb) {
        m.cb_id[cb] = cb_descs[cb].first;
        m.cb_uid[cb] = m.cb_id[cb] ? texture_cache.GetImage(m.cb_id[cb]).image_uid : 0;
    }
    m.db_id = db_desc.first;
    m.db_uid = m.db_id ? texture_cache.GetImage(m.db_id).image_uid : 0;
    m.pipeline = pipeline;
    m.reg_stamp = reg_stamp;
    m.pipe_gen = pipe_gen;
    m.tex_gen = tex_gen;
    if (sc.Gens().tex_gen.load(std::memory_order_acquire) == tex_gen) {
        m.valid = true;
        sc.NotifyPopulated(kCache);
    }
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    using VideoCore::Skipcache::CacheId;
    auto& skipcache = Skipcache::Framework::Instance();
    const bool probing = skipcache.Active() && skipcache.ShouldProbe(CacheId::PrepareRt);
    u64 rt_stamp{}, rt_tex_gen{}, rt_pipe_gen{};
    bool rt_would_hit = false;
    if (probing) {
        auto& ctr = skipcache.Counters(CacheId::PrepareRt);
        ++ctr.eligible;
        const bool timed = skipcache.SampleTimer(CacheId::PrepareRt);
        const u64 t0 = timed ? skipcache.Now() : 0;
        rt_stamp = liverpool->GetGfxStateStamp();
        rt_tex_gen = skipcache.Gens().tex_gen.load(std::memory_order_acquire);
        rt_pipe_gen = skipcache.Gens().pipe_gen.load(std::memory_order_acquire);
        rt_would_hit = RtMemoProbe(pipeline, rt_stamp, rt_tex_gen, rt_pipe_gen);
        if (timed) {
            ctr.guard_ns += skipcache.CorrectSample(skipcache.Now() - t0);
            ++ctr.guard_samples;
        }
        if (rt_would_hit) {
            ++ctr.hits;
            if (skipcache.MayConsume(CacheId::PrepareRt) &&
                !skipcache.ShouldVerify(CacheId::PrepareRt)) {
                RtMemoReplay();
                return;
            }
        }
    }
    const bool rt_timed_miss =
        probing && !rt_would_hit && skipcache.SampleTimer(CacheId::PrepareRt);
    const u64 rt_miss_t0 = rt_timed_miss ? skipcache.Now() : 0;

    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->last_cb_extent[cb];
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->last_db_extent;
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }

    if (probing) {
        auto& ctr = skipcache.Counters(CacheId::PrepareRt);
        if (rt_timed_miss) {
            ctr.miss_ns += skipcache.CorrectSample(skipcache.Now() - rt_miss_t0);
            ++ctr.miss_samples;
        }
        RtMemoVerifyPopulate(rt_would_hit, pipeline, rt_stamp, rt_tex_gen, rt_pipe_gen);
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::Info& info,
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

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
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

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    image.Clear(clear_value, desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;

    Skipcache::Framework::Instance().OnDraw();
    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const auto& regs = liverpool->regs;
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }

    // One shared-lock hold covers every guest copy of the whole draw setup;
    // the per-function scopes inside become TLS-flag no-ops. Placed after the
    // filter and pipeline resolution so filtered draws pay nothing and a
    // pipeline compile never holds the memory map open.
    std::optional<Core::MemoryManager::GuestCopyScope> copy_scope;
    if (segment_copy_hold_ && in_packet_run_) {
        ArmCopyHold();
    } else if (batch_copy_lock_) {
        copy_scope.emplace(Core::Memory::Instance());
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(index_offset, buffer_barriers);
    }

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const auto cmdbuf = scheduler.CommandBuffer();
    BindPipelineDedup(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
    DebugState.IncDrawCall();

    ResetBindings();
    if (flush_draw_interval_ != 0) {
        // The flush submits; it must not run under the guest-copy shared lock.
        copy_scope.reset();
        MaybeIntervalFlush();
    }
}

void Rasterizer::MaybeIntervalFlush() {
    const u64 tick = scheduler.CurrentTick();
    if (tick != flush_tick_) {
        flush_tick_ = tick;
        draws_since_flush_ = 0;
    }
    if (++draws_since_flush_ < flush_draw_interval_) {
        return;
    }
    // A pending depth or stencil clear belongs to the open render scope:
    // BeginRendering re-derives the clear load op, so a scope re-begun after
    // a flush would clear the attachment a second time.
    const auto& ds = scheduler.GetRenderState().depth_stencil_attachment;
    if (ds.depth_clear || ds.stencil_clear) {
        return;
    }
    DropCopyHold(hold_drops_flush_);
    scheduler.Flush();
    draws_since_flush_ = 0;
    ++interval_flushes_;
}

void Rasterizer::BeginPacketRun() {
    ASSERT(!in_packet_run_);
    in_packet_run_ = true;
}

void Rasterizer::EndPacketRun() {
    in_packet_run_ = false;
    // The arms run under the guest-copy hold, where the per-bind arms ran.
    DrainPendingReadArms(VideoCore::ReadArmSite::Run);
    DropCopyHold(hold_drops_run_);
}

void Rasterizer::DropCopyHoldForCommands() {
    DropCopyHold(hold_drops_cmd_);
}

void Rasterizer::ArmCopyHold() {
    if (!run_copy_hold_) {
        run_copy_hold_.emplace(memory);
        ++hold_arms_;
    } else {
        ++hold_draws_covered_;
    }
}

void Rasterizer::DropCopyHold(u64& counter) {
    if (run_copy_hold_) {
        run_copy_hold_.reset();
        ++counter;
    }
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }

    // One shared-lock hold covers every guest copy of the whole draw setup;
    // the per-function scopes inside become TLS-flag no-ops. Placed after the
    // filter and pipeline resolution so filtered draws pay nothing and a
    // pipeline compile never holds the memory map open.
    std::optional<Core::MemoryManager::GuestCopyScope> copy_scope;
    if (segment_copy_hold_ && in_packet_run_) {
        ArmCopyHold();
    } else if (batch_copy_lock_) {
        copy_scope.emplace(Core::Memory::Instance());
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(0, buffer_barriers);
    }

    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }
    if (count_buffer) {
        if (auto barrier = count_buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                    vk::PipelineStageFlagBits2::eDrawIndirect)) {
            buffer_barriers.emplace_back(*barrier);
        }
    }

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    const auto cmdbuf = scheduler.CommandBuffer();
    BindPipelineDedup(vk::PipelineBindPoint::eGraphics, pipeline->Handle());

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_base, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_base,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
        DebugState.IncDrawCall();
    }

    ResetBindings();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return;
    }

    // One shared-lock hold covers every guest copy of the whole draw setup;
    // the per-function scopes inside become TLS-flag no-ops. Placed after the
    // filter and pipeline resolution so filtered draws pay nothing and a
    // pipeline compile never holds the memory map open.
    std::optional<Core::MemoryManager::GuestCopyScope> copy_scope;
    if (segment_copy_hold_ && in_packet_run_) {
        ArmCopyHold();
    } else if (batch_copy_lock_) {
        copy_scope.emplace(Core::Memory::Instance());
    }

    if (!BindResources(pipeline)) {
        return;
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
    DebugState.IncDispatch();

    ResetBindings();
    if (flush_draw_interval_ != 0) {
        copy_scope.reset();
        MaybeIntervalFlush();
    }
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    if (segment_copy_hold_ && in_packet_run_) {
        ArmCopyHold();
    }

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);
    DebugState.IncDispatch();

    ResetBindings();
}

u64 Rasterizer::Flush() {
    DropCopyHold(hold_drops_wait_);
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    DropCopyHold(hold_drops_wait_);
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    {
        // Guest packet census. Deliberately outside the skipcache gate below:
        // it describes what the title submits, not how this fork caches, so it
        // stays available for surveying any game on stock settings. A
        // non-zero occl or setpred means the title drives the occlusion path,
        // which shadPS4 currently answers with a fixed "visible" result.
        static u64 last_packet_frame = 0;
        const u64 frame = DebugState.GetFrameNum();
        if (frame - last_packet_frame >= 300) {
            last_packet_frame = frame;
            auto& pk = liverpool->packet_stats;
            LOG_INFO(Render,
                     "PACKETS draws={} predicated={} dispatch={} occl={} setpred={} per300f",
                     pk.draws, pk.predicated_draws, pk.dispatches, pk.occlusion_events,
                     pk.set_predication);
            pk = {};
        }
    }
    auto& skipcache = Skipcache::Framework::Instance();
    if (skipcache.Active()) {
        // Ring pressure report: wraps are what block this thread, so surface
        // them next to the cache accounting rather than in a separate channel.
        static u64 last_report_frame = 0;
        const u64 frame = DebugState.GetFrameNum();
        if (frame - last_report_frame >= 300) {
            last_report_frame = frame;
            auto& st = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream).Stats();
            const u64 hz = tsc_hz_;
            LOG_INFO(Render_Skipcache,
                     "[SkipCache] RING maps={} MiB={} wraps={} blocked_ms={} per300f", st.maps,
                     st.bytes >> 20, st.wraps, hz ? st.blocked_ns * 1000 / hz : 0);
            st = VideoCore::StreamBuffer::RingStats{};
            auto& ws = scheduler.WaitStats();
            const auto ms = [hz](u64 ns) { return hz ? ns * 1000 / hz : 0; };
            LOG_INFO(Render_Skipcache,
                     "[SkipCache] WAITS finish={}/{}ms ring={}/{}ms fault={}/{}ms "
                     "dlbuf={}/{}ms dlimg={}/{}ms per300f",
                     ws[0].count, ms(ws[0].ns), ws[1].count, ms(ws[1].ns), ws[2].count,
                     ms(ws[2].ns), ws[3].count, ms(ws[3].ns), ws[4].count, ms(ws[4].ns));
            ws = {};
            const auto off = buffer_cache.DrainOffloadStats();
            if (off.jobs || off.fallbacks) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] OFFLOAD jobs={} vetoes={} fallbacks={} wait_ms={} per300f",
                         off.jobs, off.vetoes, off.fallbacks, ms(off.wait_ns));
            }
            if (const auto wb = buffer_cache.DrainWritebackStats(); wb.islands) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] WRITEBACK loops={} islands={} KiB={} per300f", wb.loops,
                         wb.islands, wb.bytes >> 10);
            }
            if (const auto wo = buffer_cache.DrainWriteBackOffloadStats();
                wo.guest + wo.prio + wo.gpucomm) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] WBOFF guest={} prio={} gpucomm={} excluded={} copy_ms={} "
                         "per300f",
                         wo.guest, wo.prio, wo.gpucomm, wo.excluded, ms(wo.copy_ns));
            }
            const auto sc = buffer_cache.DrainStreamCopyStats();
            if (sc.probes) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] STREAMCOPY hits={} probes={} fast={} idxfast={} "
                         "genwalk={}/{}/{} per300f",
                         sc.hits, sc.probes, sc.fast, sc.idxfast, sc.stream_genwalk,
                         sc.vertex_genwalk, sc.index_genwalk);
            }
            if (const auto vi = buffer_cache.DrainVertexInputStats(); vi.calls) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] VINPUT calls={} built={} binds={} chain={} layout={} bind={} "
                         "per300f",
                         vi.calls, vi.built, vi.binds, vi.chain, vi.layout, vi.bind);
            }
            if (flush_draw_interval_ != 0) {
                LOG_INFO(Render_Skipcache, "[SkipCache] IFLUSH count={} per300f",
                         interval_flushes_);
                interval_flushes_ = 0;
            }
            if (segment_copy_hold_) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] COPYHOLD arms={} covered={} drops_run={} drops_flush={} "
                         "drops_wait={} drops_cmd={} compiles={} per300f",
                         hold_arms_, hold_draws_covered_, hold_drops_run_, hold_drops_flush_,
                         hold_drops_wait_, hold_drops_cmd_, hold_compiles_);
                hold_arms_ = hold_draws_covered_ = hold_drops_run_ = hold_drops_flush_ =
                    hold_drops_wait_ = hold_drops_cmd_ = hold_compiles_ = 0;
            }
            if (const auto bw = Core::MemoryManager::DrainBackingWriteStats(); bw.calls) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] BACKWRITE calls={} hits={} hitKiB={} missKiB={} multi={} "
                         "per300f",
                         bw.calls, bw.hits, bw.hit_bytes >> 10, bw.miss_bytes >> 10, bw.multi);
            }
            pipeline_cache.DumpColorMaskStats(
                scheduler.GetDynamicState().DrainColorWriteMaskSkips());
            pipeline_cache.DumpKeyReuseStats();
            pipeline_cache.DumpProgramIdentityStats();
            pipeline_cache.DumpSpecFpStats();
            pipeline_cache.DumpRuntimeInfoMemoStats();
            pipeline_cache.DumpLayoutStats();
            const auto vm = texture_cache.DrainViewMemoStats();
            if (vm.hits || vm.slow) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] VIEWMEMO hits={} slow={} writebacks={} per300f", vm.hits,
                         vm.slow, vm.writebacks);
            }
            if (skipcache.ActiveMode() == Skipcache::Mode::Forced) {
                const auto& fc = skipcache.Counters(Skipcache::CacheId::FindImage);
                if (fc.eligible != findimg_last_.eligible) {
                    const auto d = [&](u64 Skipcache::CacheCounters::* f) {
                        return fc.*f - findimg_last_.*f;
                    };
                    LOG_INFO(Render_Skipcache,
                             "[SkipCache] FINDIMG probes={} hits={} cold={} key={} gen={} view={} "
                             "veto={} vfy={}/{}/{} per300f",
                             d(&Skipcache::CacheCounters::eligible),
                             d(&Skipcache::CacheCounters::hits),
                             d(&Skipcache::CacheCounters::miss_cold),
                             d(&Skipcache::CacheCounters::miss_key),
                             fc.miss_gen[Skipcache::LaneTex] -
                                 findimg_last_.miss_gen[Skipcache::LaneTex],
                             fc.veto[1] - findimg_last_.veto[1], fc.veto[0] - findimg_last_.veto[0],
                             d(&Skipcache::CacheCounters::verify_clean),
                             d(&Skipcache::CacheCounters::verify_diverged),
                             d(&Skipcache::CacheCounters::verify_aborted));
                    findimg_last_ = fc;
                }
            }
            if (const auto& dc = skipcache.Counters(Skipcache::CacheId::DynState);
                dc.eligible != dynstate_last_.eligible) {
                const auto d = [&](u64 Skipcache::CacheCounters::* f) {
                    return dc.*f - dynstate_last_.*f;
                };
                const u64 gfx_stamp = liverpool->GetGfxStateStamp();
                const u64 dyn_stamp = liverpool->GetDynStateStamp();
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] DYNSTATE probes={} hits={} reg={} key={} gen={} stamp={} "
                         "dyn={} per300f",
                         d(&Skipcache::CacheCounters::eligible), d(&Skipcache::CacheCounters::hits),
                         dc.miss_gen[Skipcache::LaneReg] -
                             dynstate_last_.miss_gen[Skipcache::LaneReg],
                         d(&Skipcache::CacheCounters::miss_key),
                         dc.miss_gen[Skipcache::LaneTick] + dc.miss_gen[Skipcache::LanePipe] -
                             dynstate_last_.miss_gen[Skipcache::LaneTick] -
                             dynstate_last_.miss_gen[Skipcache::LanePipe],
                         gfx_stamp - gfx_stamp_last_, dyn_stamp - dyn_stamp_last_);
                dynstate_last_ = dc;
                gfx_stamp_last_ = gfx_stamp;
                dyn_stamp_last_ = dyn_stamp;
            }
            if (bindpf_img_) {
                LOG_INFO(Render_Skipcache, "[SkipCache] BINDPF img={} backing={} per300f",
                         bindpf_img_, bindpf_backing_);
                bindpf_img_ = bindpf_backing_ = 0;
            }
            if (bind_noop_) {
                const auto bn = texture_cache.DrainBindNoopStats();
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] BINDNOOP hits={} slow={} records={} zero={} per300f",
                         bindnoop_hits_, bindnoop_slow_, bn.records, bn.zero);
                bindnoop_hits_ = bindnoop_slow_ = 0;
            }
            const auto ft = texture_cache.DrainFindTouchStats();
            if (ft.consumed) {
                LOG_INFO(Render_Skipcache, "[SkipCache] FINDTOUCH consumed={} locks={} per300f",
                         ft.consumed, ft.locks);
            }
            if (const auto iu = texture_cache.DrainImageUpdateStats();
                iu.fast || iu.relock || iu.full) {
                LOG_INFO(Render_Skipcache, "[SkipCache] IMGUPD fast={} relock={} full={} per300f",
                         iu.fast, iu.relock, iu.full);
            }
            if (const auto ll = texture_cache.DrainLruLogStats(); ll.pushes || ll.walked) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] LRULOG pushes={} walked={} skipped={} compact={} size={} "
                         "dead={} per300f",
                         ll.pushes, ll.walked, ll.skipped, ll.compactions, ll.size, ll.dead);
            }
            if (const auto fw = texture_cache.DrainFindImageWayStats(); fw.ways) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] FINDIMGWAYS ways={} hits={}/{}/{}/{} evict={} per300f",
                         fw.ways, fw.hits[0], fw.hits[1], fw.hits[2], fw.hits[3], fw.evictions);
            }
            const auto ss = texture_cache.DrainSamplerStats();
            if (ss.calls) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] SAMPLER calls={} slow={} touches={} map={} per300f", ss.calls,
                         ss.slow, ss.touches, ss.map);
            }
            const auto dd = skipcache.DrainDescDeltaStats();
            if (dd.probes) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] DESCDELTA probes={} hits={} partial={} descs={} pushed={} "
                         "split={} per300f",
                         dd.probes, dd.hits, dd.partial, dd.descs, dd.pushed, dd.split);
            }
            const auto us = buffer_cache.DrainUploadCopyStats();
            if (us.ro_calls || us.w_calls) {
                LOG_INFO(Render_Skipcache, "[SkipCache] UPLOAD ro={} roMiB={} w={} wMiB={} per300f",
                         us.ro_calls, us.ro_bytes >> 20, us.w_calls, us.w_bytes >> 20);
            }
            if (const auto ra = buffer_cache.DrainReadArmStats();
                ra.drains[0] + ra.drains[1] + ra.drains[2] + ra.drains[3] + ra.drains[4] != 0) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] RARM run={} submit={} fence={} wait={} idle={} regions={} "
                         "pages={} calls={} per300f",
                         ra.drains[0], ra.drains[1], ra.drains[2], ra.drains[3], ra.drains[4],
                         ra.regions, ra.pages, ra.calls);
            }
            if (const auto tn = buffer_cache.DrainTexelNoopStats(); tn.probes) {
                LOG_INFO(Render_Skipcache, "[SkipCache] TEXELNOOP hits={} probes={} per300f",
                         tn.hits, tn.probes);
            }
            const auto wr = buffer_cache.DrainWrittenRangeStats();
            if (wr.binds) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] WRANGE binds={} fresh={} hits={} adds={} shrinks={} folds={} "
                         "folded={} full={} direct={} lockskips={} per300f",
                         wr.binds, wr.fresh, wr.hits, wr.adds, wr.shrinks, wr.folds, wr.folded,
                         wr.full, wr.direct, wr.lockskips);
            }
            const auto ds = buffer_cache.DrainDmaSyncStats();
            if (ds.calls) {
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] DMASYNC calls={} buffers={} MiB={} maxMiB={} per300f",
                         ds.calls, ds.buffers, ds.bytes >> 20, ds.max_bytes >> 20);
            }
            if (auto& lane = VideoCore::StreamCopyLane::Instance(); lane.Enabled()) {
                const auto ls = lane.DrainStats();
                LOG_INFO(Render_Skipcache,
                         "[SkipCache] LANE jobs={} MiB={} unres={} full={} barriers={} "
                         "wait_ms={} per300f",
                         ls.jobs, ls.bytes >> 20, ls.inline_unresolved, ls.inline_full, ls.barriers,
                         ls.barrier_wait_ns / 1000000);
            }
            buffer_cache.EmitMirrorTelemetry();
        }
    }
    skipcache.OnSubmit(DebugState.GetFrameNum(), DebugState.IsGuestThreadsPaused());
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    buffer_cache.RunGarbageCollector();
}

// ---- BindingSkip LEARNING probe (observe-only) ---------------------------
// Measures whether a per-stage binding replay cache would pay: would-hit =
// same pipeline, same cmdbuf tick, every stage's pgm_hash and user_data words
// bit-identical (memcmp, never hash). Data-gates the wave-2 replay build; by
// the instrument-contamination rule this probe writes nothing and returns
// nothing.
enum BsVeto : u8 {
    BsVetoPipeline = 0,
    BsVetoTick = 1,
    BsVetoStageCount = 2,
    BsVetoPgmHash = 3,
    BsVetoUserData = 4,
};

void Rasterizer::BindingSkipProbe(const Pipeline* pipeline) {
    using namespace VideoCore::Skipcache;
    auto& sc = Skipcache::Framework::Instance();
    constexpr auto kBS = CacheId::BindingSkipProbe;
    if (!sc.Active() || !sc.ShouldProbe(kBS)) {
        return;
    }
    auto& ctr = sc.Counters(kBS);
    ++ctr.eligible;
    auto& p = bs_probe_;
    const u64 tick = scheduler.CurrentTick();
    const auto stages = pipeline->GetStages();

    bool would_hit = false;
    if (!p.valid) {
        ++ctr.miss_cold;
    } else if (p.pipeline != pipeline) {
        ++ctr.veto[BsVetoPipeline];
    } else if (p.tick != tick) {
        ++ctr.veto[BsVetoTick];
    } else {
        would_hit = true;
        u32 idx = 0;
        for (const auto* stage : stages) {
            if (!stage) {
                continue;
            }
            if (idx >= p.num_stages) {
                ++ctr.veto[BsVetoStageCount];
                would_hit = false;
                break;
            }
            auto& snap = p.stages[idx];
            if (snap.pgm_hash != stage->pgm_hash) {
                ++ctr.veto[BsVetoPgmHash];
                would_hit = false;
                break;
            }
            const size_t words = std::min<size_t>(stage->user_data.size(), snap.user_data.size());
            if (std::memcmp(snap.user_data.data(), stage->user_data.data(), words * sizeof(u32)) !=
                0) {
                ++ctr.veto[BsVetoUserData];
                would_hit = false;
                break;
            }
            ++idx;
        }
        if (would_hit && idx != p.num_stages) {
            ++ctr.veto[BsVetoStageCount];
            would_hit = false;
        }
    }
    if (would_hit) {
        ++ctr.hits;
        return;
    }
    // Refresh the observation snapshot (probe-internal state, not a cache).
    p.pipeline = pipeline;
    p.tick = tick;
    p.num_stages = 0;
    for (const auto* stage : stages) {
        if (!stage || p.num_stages >= p.stages.size()) {
            continue;
        }
        auto& snap = p.stages[p.num_stages++];
        snap.pgm_hash = stage->pgm_hash;
        const size_t words = std::min<size_t>(stage->user_data.size(), snap.user_data.size());
        std::memcpy(snap.user_data.data(), stage->user_data.data(), words * sizeof(u32));
        if (words < snap.user_data.size()) {
            std::memset(snap.user_data.data() + words, 0,
                        (snap.user_data.size() - words) * sizeof(u32));
        }
    }
    p.valid = true;
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    BindingSkipProbe(pipeline);

    set_write_index = 0;
    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        // A stage's buffers and its samplers each collapse into one write; the
        // image loop still emits one per descriptor array. Exact, so
        // set_write_index ends on set_writes.size().
        set_writes.resize(set_writes.size() + !stage->buffers.empty() + stage->images.size() +
                          !stage->samplers.empty());
        stage->PushUd(binding, push_data);
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        // We only use fault buffer for DMA right now.
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (auto& range : mapped_ranges) {
            buffer_cache.SynchronizeBuffersInRange(range.lower(), range.upper() - range.lower());
        }
        fault_process_pending = true;
    }

    return true;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
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

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
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

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
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
                             Shader::PushData& push_data) {
    // One shared-lock hold covers every guest copy this stage stages
    // (flatbuf, clip planes, stream uploads); see GuestCopyScope.
    std::optional<Core::MemoryManager::GuestCopyScope> copy_scope;
    if (batch_copy_lock_) {
        copy_scope.emplace(Core::Memory::Instance());
    }
    buffer_bindings.clear();

    const bool elide_findbuffer = elide_findbuffer_;
    for (const auto& desc : stage.buffers) {
        const auto vsharp = desc.GetSharp(stage);
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            // Stream-eligible read-only bindings never dereference the id:
            // ObtainBuffer's stream path returns before touching it and
            // re-resolves on its own when it falls through to the slot path.
            // FindBuffer here is three dependent cache-missing loads. DMA
            // stages keep the eager call so registration still feeds the BDA
            // page table. The predicate is a hint: a mismatch with
            // ObtainBuffer's guard costs one late FindBuffer, never
            // correctness.
            const bool defer = elide_findbuffer && !desc.is_written &&
                               size <= VideoCore::BufferCache::CACHING_PAGESIZE && !stage.uses_dma;
            const auto buffer_id =
                defer ? VideoCore::BufferId{} : buffer_cache.FindBuffer(vsharp.base_address, size);
            buffer_bindings.emplace_back(buffer_id, vsharp, size, true);
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0, false);
        }
    }

    // Second pass to re-bind buffers that were updated after binding
    const u32 first_info = static_cast<u32>(buffer_infos.size());
    const u32 first_binding = binding.unified;
    for (u32 i = 0; i < buffer_bindings.size(); i++) {
        const auto& [buffer_id, vsharp, size, is_guest] = buffer_bindings[i];
        const auto& desc = stage.buffers[i];
        const u32 alignment = instance.StorageMinAlignment();
        // Buffer is not from the cache, either a special buffer or unbound.
        if (!is_guest) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = stage.srt_info.flattened_bufsize_dw * sizeof(u32);
                const u64 offset = vk_buffer.Copy(stage.flat_ud, ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->regs.clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = liverpool->GetCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id);
            // Power-of-two Vulkan alignment: the generic AlignDown emitted a
            // hardware divide here at binding rate.
            const u32 offset_aligned = static_cast<u32>(offset & ~u64{alignment - 1});
            const u32 adjust = offset - offset_aligned;
            if (adjust % 4 != 0) {
                LOG_WARNING(Render_Vulkan, "Buffer binding {} in shader {:#x} isn't dword aligned",
                            i, stage.pgm_hash);
            }
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            if (auto barrier =
                    vk_buffer->GetBarrier(desc.is_written ? vk::AccessFlagBits2::eShaderWrite
                                                          : vk::AccessFlagBits2::eShaderRead,
                                          vk::PipelineStageFlagBits2::eAllCommands)) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (desc.is_written && desc.is_formatted) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
        }

        ++binding.buffer;
    }

    // Every pass-2 arm appends exactly one buffer_info, so the stage's infos
    // are contiguous and in binding order, and each spanned binding is a
    // descriptorCount-1 eStorageBuffer carrying this stage's flags - the
    // identity a consecutive-binding update requires. descriptorCount 0 is
    // illegal, hence the guard.
    binding.unified += static_cast<u32>(stage.buffers.size());
    if (!stage.buffers.empty()) {
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = first_binding;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = static_cast<u32>(stage.buffers.size());
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = buffer_infos.data() + first_info;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // For loading/storing to explicit mip levels, when no native instruction support, bind an array
    // of descriptors consecutively, 1 for each mip level. The shader can index this with LOD
    // operand.
    // This array holds the size of each consecutive array with the number of bindings consumed.
    // This is currently always 1 for anything other than mip fallback arrays.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    AmdGpu::Image tsharp_scratch{};
    for (const auto& image_desc : stage.images) {
        const AmdGpu::Image& tsharp = image_desc.GetSharpRef(stage, tsharp_scratch);
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        // The reject paths below must consume the descriptor count the layout
        // declared for this image, or every later binding number in this stage
        // and in the stages after it drifts off the layout.
        const u32 num_bindings = image_desc.NumBindings(tsharp);
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            for (u32 i = 0; i < num_bindings; ++i) {
                image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            }
            image_descriptor_array_sizes.push_back(num_bindings);
            continue;
        }

        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
            for (u32 i = 0; i < num_bindings; ++i) {
                image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            }
            image_descriptor_array_sizes.push_back(num_bindings);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;

        for (auto i = 0; i < num_bindings; i++) {
            // Mip fallback rewrites the view range before the memo probe, so
            // only fallback-free bindings defer the view build to a memo miss.
            auto& [image_id, desc] = image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{},
                std::tuple<const AmdGpu::Image&, const Shader::ImageResource&, bool>{
                    tsharp, image_desc, mip_fallback_mode == Shader::MipStorageFallbackMode::None});

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            image_id = texture_cache.FindImageMemoized(desc, tsharp);
            auto* image = &texture_cache.GetImage(image_id);
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
            if (bind_prefetch_) {
                // Pass two reads these lines first: props for a texture
                // binding's layout choice, the backing pointer for the view
                // memo compare, and the backing's state line for the barrier
                // probe. A storage binding never reads props there. After a
                // stencil redirect the memo backing belongs to the pre-redirect
                // image; its line is warmed for nothing.
                if (!image_desc.is_written) {
                    __builtin_prefetch(&image->info.props, 0, 3);
                }
                __builtin_prefetch(&image->backing, 0, 3);
                ++bindpf_img_;
                // Under the bind no-op memo the state line is read only on a
                // memo miss, so it is not warmed.
                if (!bind_noop_ && desc.memo_backing) {
                    __builtin_prefetch(&desc.memo_backing->state, 0, 3);
                    ++bindpf_backing_;
                }
            }
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // Second pass to re-bind images that were updated after binding
    for (auto& [image_id, desc] : image_bindings) {
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            const vk::ImageView image_view = texture_cache.FindTexture(image_id, desc);
            vk::ImageLayout bound_layout;

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
                image.Transit(instance.IsAttachmentFeedbackLoopLayoutSupported() &&
                                      image.binding.is_target
                                  ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                  : vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead |
                                  (image.info.props.is_depth
                                       ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                       : vk::AccessFlagBits2::eColorAttachmentWrite |
                                             vk::AccessFlagBits2::eColorAttachmentRead),
                              {});
                bound_layout = image.backing->state.layout;
            } else if (is_storage) {
                image.Transit(vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                              desc.view_info.range);
                bound_layout = image.backing->state.layout;
            } else {
                const auto new_layout = image.info.props.is_depth
                                            ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                            : vk::ImageLayout::eShaderReadOnlyOptimal;
                // A no-op recorded under the backing's current epoch repeats,
                // and the layout recorded with it is the one the backing still
                // holds. The compare is against the live image: a rebind
                // re-resolve leaves the memo backing on another image.
                if (bind_noop_ && desc.memo_bind_epoch != 0 &&
                    desc.memo_bind_epoch == image.backing_epoch &&
                    desc.memo_backing == image.backing) {
                    bound_layout = desc.memo_bind_layout;
                    ++bindnoop_hits_;
                } else {
                    image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead,
                                  desc.view_info.range);
                    bound_layout = image.backing->state.layout;
                    if (bind_noop_) {
                        if (desc.memo_slot != VideoCore::TextureCache::ImageDesc::NoMemoSlot) {
                            texture_cache.RecordBindNoop(image_id, desc, new_layout);
                        }
                        ++bindnoop_slow_;
                    }
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, image_view, bound_layout);
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    // Both taken after the image writes have finished advancing binding.unified
    // and appending to image_infos.
    const u32 first_sampler_info = static_cast<u32>(image_infos.size());
    const u32 first_sampler_binding = binding.unified;
    AmdGpu::Image aniso_scratch{};
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (sampler.disable_aniso) {
            const AmdGpu::Image& tsharp =
                stage.images[sampler.associated_image].GetSharpRef(stage, aniso_scratch);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        const auto vk_sampler = texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
    }

    // Each spanned binding is a descriptorCount-1 eSampler with this stage's
    // flags and no immutable sampler, which is what a consecutive-binding
    // update requires. descriptorCount 0 is illegal, hence the guard.
    binding.unified += static_cast<u32>(stage.samplers.size());
    if (!stage.samplers.empty()) {
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = first_sampler_binding;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = static_cast<u32>(stage.samplers.size());
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = image_infos.data() + first_sampler_info;
    }
}

// ---- BeginRendering skip cache -------------------------------------------

// Veto buckets for the BR guard chain (named per cache, reported in SESSION).
enum BrVeto : u8 {
    BrVetoMetaClear = 0,
    BrVetoUid = 1,
    BrVetoRegistered = 2,
    BrVetoRebind = 3,
    BrVetoBacking = 4,
    BrVetoLayout = 5,
    BrVetoSubres = 6,
};

bool Rasterizer::BrGuardAttachment(const BrAttachmentGuard& g,
                                   VideoCore::Skipcache::CacheCounters& ctr) {
    // 1. Armed meta clear: compute clear shaders and FillBuffer arm CMASK/HTILE
    //    without any register write; the stamp cannot see them (read-only).
    if (g.meta_addr && texture_cache.IsMetaCleared(g.meta_addr, g.slice)) {
        ++ctr.veto[BrVetoMetaClear];
        return false;
    }
    if (!g.image_id) {
        return true; // masked slot
    }
    auto& image = texture_cache.GetImage(g.image_id);
    if (image.image_uid != g.image_uid) {
        ++ctr.veto[BrVetoUid];
        return false;
    }
    if (False(image.flags & VideoCore::ImageFlagBits::Registered)) {
        // Deleted-but-slot-not-yet-erased (the chance-overlap FreeImage path).
        ++ctr.veto[BrVetoRegistered];
        return false;
    }
    if (image.binding.needs_rebind) {
        ++ctr.veto[BrVetoRebind];
        return false;
    }
    if (static_cast<const void*>(image.backing) != g.backing) {
        // Backing swaps from copy/download paths that pipeline equality never
        // covered.
        ++ctr.veto[BrVetoBacking];
        return false;
    }
    if (image.backing->state.layout != g.expected_layout ||
        image.backing->state.access_mask != g.expected_access) {
        ++ctr.veto[BrVetoLayout];
        return false;
    }
    if (!image.backing->subresource_states.empty()) {
        // Ranged verify: every [mip][layer] in the cached view range must match
        // (partial-view render targets, e.g. cascade shadow slices).
        const u32 layers = image.info.resources.layers;
        for (u32 mip = g.base_level; mip < g.base_level + g.num_levels; ++mip) {
            for (u32 layer = g.base_layer; layer < g.base_layer + g.num_layers; ++layer) {
                const size_t idx = size_t(mip) * layers + layer;
                if (idx >= image.backing->subresource_states.size()) {
                    ++ctr.veto[BrVetoSubres];
                    return false;
                }
                const auto& st = image.backing->subresource_states[idx];
                if (st.layout != g.expected_layout || st.access_mask != g.expected_access) {
                    ++ctr.veto[BrVetoSubres];
                    return false;
                }
            }
        }
    }
    return true;
}

bool Rasterizer::BrProbe(const VideoCore::Skipcache::DrawToken& token,
                         const GraphicsPipeline* pipeline) {
    using namespace VideoCore::Skipcache;
    auto& ctr = Skipcache::Framework::Instance().Counters(CacheId::BeginRendering);
    const auto& c = br_cache_;
    if (!c.valid) {
        ++ctr.miss_cold;
        return false;
    }
    if (c.pipeline != pipeline) { // compared, never dereferenced
        ++ctr.miss_key;
        return false;
    }
    if (c.token.reg_stamp != token.reg_stamp) {
        ++ctr.miss_gen[LaneReg];
        return false;
    }
    if (c.token.tick != token.tick) {
        ++ctr.miss_gen[LaneTick];
        return false;
    }
    if (c.token.mem_gen != token.mem_gen) {
        ++ctr.miss_gen[LaneMem];
        return false;
    }
    if (c.token.tex_gen != token.tex_gen) {
        ++ctr.miss_gen[LaneTex];
        return false;
    }
    if (c.token.pipe_gen != token.pipe_gen) {
        ++ctr.miss_gen[LanePipe];
        return false;
    }
    if (c.token.img_dirty_gen != token.img_dirty_gen) {
        ++ctr.miss_gen[LaneImgDirty];
        return false;
    }
    // Generation short-circuit: with the meta and layout lanes unchanged
    // since the last populate or verified pass, every per-attachment guard
    // below would pass identically (arming is meta_gen; uid/Registered/
    // rebind/backing identity are tex_gen in the token; layout, access,
    // subresource states and backing swaps are layout_gen). Three compares
    // replace the guard loop.
    auto& gens = Skipcache::Framework::Instance().Gens();
    if (c.meta_gen == gens.meta_gen && c.layout_gen == gens.layout_gen) {
        return true;
    }
    for (u32 cb = 0; cb < c.cb_count; ++cb) {
        if (!BrGuardAttachment(c.cb_guard[cb], ctr)) {
            return false;
        }
    }
    if (c.has_db && !BrGuardAttachment(c.db_guard, ctr)) {
        return false;
    }
    // The guard loop passed against the live world: re-stamp so subsequent
    // stamp-equal draws take the three-compare exit.
    br_cache_.meta_gen = gens.meta_gen;
    br_cache_.layout_gen = gens.layout_gen;
    return true;
}

RenderState Rasterizer::BrReplay(const GraphicsPipeline* pipeline) {
    // Consumed hit: replay the clear-free snapshot. SetBackingSamples is
    // cheap and idempotent; re-running it preserves the slow path's every-draw
    // re-assertion.
    const auto& key = pipeline->GetGraphicsKey();
    for (u32 cb = 0; cb < br_cache_.cb_count; ++cb) {
        const auto& g = br_cache_.cb_guard[cb];
        if (g.image_id) {
            texture_cache.GetImage(g.image_id).SetBackingSamples(key.color_samples[cb]);
        }
    }
    attachment_feedback_loop = br_cache_.attachment_feedback_loop;
    return br_cache_.state;
}

void Rasterizer::BrVerify(const RenderState& fresh, const VideoCore::Skipcache::DrawToken& token) {
    using namespace VideoCore::Skipcache;
    auto& sc = Skipcache::Framework::Instance();
    constexpr auto kBR = CacheId::BeginRendering;
    // Attributed sub-checks first (they name the leaked invalidation channel),
    // then the whole-struct closer. The snapshot is clear-free by refusal, so
    // a fresh clear flag on a would-hit IS the caught bug (raw compare).
    const char* diff = nullptr;
    for (u32 cb = 0; cb < fresh.num_color_attachments && !diff; ++cb) {
        if (fresh.color_attachments[cb].is_clear) {
            diff = "fresh color clear under would-hit (leaked arming channel)";
        }
    }
    if (!diff && (fresh.depth_stencil_attachment.depth_clear ||
                  fresh.depth_stencil_attachment.stencil_clear)) {
        diff = "fresh depth/stencil clear under would-hit";
    }
    if (!diff && (fresh.width != br_cache_.state.width || fresh.height != br_cache_.state.height ||
                  fresh.num_layers != br_cache_.state.num_layers ||
                  fresh.num_color_attachments != br_cache_.state.num_color_attachments)) {
        diff = "dims/counts";
    }
    if (!diff && !(fresh == br_cache_.state)) {
        diff = "whole-struct";
    }
    if (!diff && attachment_feedback_loop != br_cache_.attachment_feedback_loop) {
        diff = "attachment_feedback_loop";
    }
    if (!diff) {
        sc.RecordVerifyClean(kBR);
        return;
    }
    // Racing cross-thread invalidation between hit-check and compare is an
    // abort, not a divergence.
    const DrawToken t2 = sc.Capture(liverpool->GetGfxStateStamp(), scheduler.CurrentTick());
    if (t2.mem_gen != token.mem_gen || t2.tex_gen != token.tex_gen ||
        t2.pipe_gen != token.pipe_gen) {
        sc.RecordVerifyAborted(kBR);
        br_cache_.valid = false;
        return;
    }
    sc.RecordDivergence(kBR, diff);
    br_cache_.valid = false;
}

void Rasterizer::BrPopulate(const RenderState& fresh, const VideoCore::Skipcache::DrawToken& token,
                            const GraphicsPipeline* pipeline) {
    using namespace VideoCore::Skipcache;
    auto& sc = Skipcache::Framework::Instance();
    auto& ctr = sc.Counters(CacheId::BeginRendering);
    // Self-invalidate first: any early exit leaves the cache off.
    br_cache_.valid = false;
    // Canonicalize-by-refusal: never store a snapshot carrying any clear flag.
    // The consumed-CMASK populating draw is skipped (one draw of lost populate
    // per pass); level-triggered register clears keep the flags set every draw
    // and are therefore never populated while armed.
    for (u32 cb = 0; cb < fresh.num_color_attachments; ++cb) {
        if (fresh.color_attachments[cb].is_clear) {
            ++ctr.populate_refused;
            return;
        }
    }
    if (fresh.depth_stencil_attachment.depth_clear ||
        fresh.depth_stencil_attachment.stencil_clear) {
        ++ctr.populate_refused;
        return;
    }
    // Capture per-attachment guards from post-Transit live state.
    br_cache_.cb_count = fresh.num_color_attachments;
    const auto& regs = liverpool->regs;
    for (u32 cb = 0; cb < fresh.num_color_attachments; ++cb) {
        auto& g = br_cache_.cb_guard[cb];
        g = {};
        const auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            continue;
        }
        auto& image = texture_cache.GetImage(image_id);
        const auto& col_buf = regs.color_buffers[cb];
        g.meta_addr = col_buf.CmaskAddress();
        g.image_id = image_id;
        g.image_uid = image.image_uid;
        g.backing = image.backing;
        g.expected_layout = image.backing->state.layout;
        g.expected_access = image.backing->state.access_mask;
        g.base_level = desc.view_info.range.base.level;
        g.base_layer = desc.view_info.range.base.layer;
        g.num_levels = desc.view_info.range.extent.levels;
        g.num_layers = desc.view_info.range.extent.layers;
        g.slice = g.base_layer;
    }
    br_cache_.has_db = db_desc.first.operator bool();
    if (br_cache_.has_db) {
        auto& g = br_cache_.db_guard;
        g = {};
        const auto& [image_id, desc] = db_desc;
        auto& image = texture_cache.GetImage(image_id);
        g.meta_addr = regs.depth_htile_data_base.GetAddress();
        g.image_id = image_id;
        g.image_uid = image.image_uid;
        g.backing = image.backing;
        g.expected_layout = image.backing->state.layout;
        g.expected_access = image.backing->state.access_mask;
        g.base_level = desc.view_info.range.base.level;
        g.base_layer = desc.view_info.range.base.layer;
        g.num_levels = desc.view_info.range.extent.levels;
        g.num_layers = desc.view_info.range.extent.layers;
        g.slice = g.base_layer;
    }
    br_cache_.pipeline = pipeline;
    br_cache_.state = fresh;
    br_cache_.attachment_feedback_loop = attachment_feedback_loop;
    {
        const auto& gens = sc.Gens();
        br_cache_.meta_gen = gens.meta_gen;
        br_cache_.layout_gen = gens.layout_gen;
    }
    // Commit re-check: a cross-thread invalidation landing mid-build forces
    // the next probe to miss (seqlock consumer side).
    const DrawToken t2 = sc.Capture(liverpool->GetGfxStateStamp(), scheduler.CurrentTick());
    if (t2.mem_gen != token.mem_gen || t2.tex_gen != token.tex_gen ||
        t2.pipe_gen != token.pipe_gen) {
        return; // br_cache_.valid stays false
    }
    br_cache_.token = token;
    br_cache_.valid = true;
    sc.NotifyPopulated(VideoCore::Skipcache::CacheId::BeginRendering);
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    using VideoCore::Skipcache::CacheId;
    using VideoCore::Skipcache::DrawToken;
    auto& skipcache = VideoCore::Skipcache::Framework::Instance();
    const bool br_probing =
        skipcache.Active() && !br_readback_gate_ && skipcache.ShouldProbe(CacheId::BeginRendering);
    DrawToken br_token{};
    bool br_would_hit = false;
    if (br_probing) {
        auto& ctr = skipcache.Counters(CacheId::BeginRendering);
        ++ctr.eligible;
        const bool timed = skipcache.SampleTimer(CacheId::BeginRendering);
        const u64 t0 = timed ? skipcache.Now() : 0;
        br_token = skipcache.Capture(liverpool->GetGfxStateStamp(), scheduler.CurrentTick());
        br_would_hit = BrProbe(br_token, pipeline);
        if (timed) {
            ctr.guard_ns += skipcache.CorrectSample(skipcache.Now() - t0);
            ++ctr.guard_samples;
        }
        if (br_would_hit) {
            ++ctr.hits;
            if (skipcache.MayConsume(CacheId::BeginRendering) &&
                !skipcache.ShouldVerify(CacheId::BeginRendering)) {
                return BrReplay(pipeline);
            }
        }
    }
    const bool br_timed_miss =
        br_probing && !br_would_hit && skipcache.SampleTimer(CacheId::BeginRendering);
    const u64 br_miss_t0 = br_timed_miss ? skipcache.Now() : 0;

    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state{};
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.MaybeUpdateImage(image_id);
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
            image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentRead,
                           desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        image.Transit(new_layout,
                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                      desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    if (br_probing) {
        auto& ctr = skipcache.Counters(CacheId::BeginRendering);
        if (br_timed_miss) {
            ctr.miss_ns += skipcache.CorrectSample(skipcache.Now() - br_miss_t0);
            ++ctr.miss_samples;
        }
        if (br_would_hit) {
            if (skipcache.GetState(CacheId::BeginRendering) !=
                VideoCore::Skipcache::State::Learning) {
                BrVerify(state, br_token);
            }
        } else {
            BrPopulate(state, br_token, pipeline);
        }
    }
    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

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
    scheduler.CommandBuffer().copyImage(read_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        write_image.GetImage(),
                                        vk::ImageLayout::eTransferDstOptimal, region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    buffer_cache.CopyBuffer(dst, src, num_bytes, dst_gds, src_gds);
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
    Skipcache::Framework::Instance().BumpMemGen();
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size);
    Skipcache::Framework::Instance().BumpMemGen();
    return true;
}

void Rasterizer::ProcessDownloadImages() {
    if (texture_cache.HasPendingDownloads()) {
        DropCopyHold(hold_drops_wait_);
    }
    texture_cache.ProcessDownloadImages();
}

// Async-signal-safe per-thread cache of recent positive IsMapped intervals:
// the fault handler and per-binding validation both hammer this at high rate,
// and the shared lock costs far more than the lookup. Map/Unmap bump the
// generation (release); an acquire load drops stale hits.
bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    struct CacheEntry {
        VAddr base; // [base, limit) - 0,0 means empty slot
        VAddr limit;
    };
    static constexpr size_t kCacheSize = 4;
    thread_local std::array<CacheEntry, kCacheSize> tls_cache{};
    thread_local u64 tls_gen = ~u64{0};

    const VAddr query_end = addr + size;
    if (query_end < addr) [[unlikely]] {
        // Wrapped the address space (a failed upstream resolve can pass -1);
        // a wrapped end would defeat the limit and straddle checks below.
        return false;
    }
    const bool cache_active = Skipcache::Framework::Instance().Active();
    const u64 cur_gen = mapped_ranges_gen_.load(std::memory_order_acquire);
    if (cache_active) {
        if (cur_gen == tls_gen) [[likely]] {
            for (const auto& e : tls_cache) {
                if (addr >= e.base && query_end <= e.limit) {
                    return true;
                }
            }
        } else {
            tls_cache.fill({0, 0});
            tls_gen = cur_gen;
        }
    }

    // Miss: find(addr) instead of contains(range) - the iterator returns the
    // containing interval bounds for free, which seed future hits in the same
    // neighborhood.
    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    const auto it = mapped_ranges.find(addr);
    if (it == mapped_ranges.end()) {
        return false;
    }
    const VAddr lo = it->lower();
    const VAddr hi = it->upper();
    if (query_end > hi) {
        // In a tracked interval but straddling its upper bound; entries cache
        // only fully contained ranges.
        return false;
    }
    if (cache_active) {
        for (size_t i = kCacheSize - 1; i > 0; --i) {
            tls_cache[i] = tls_cache[i - 1];
        }
        tls_cache[0] = CacheEntry{lo, hi};
    }
    return true;
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    // Bump after the mutation is committed; release pairs with IsMapped's
    // acquire load so a thread observing the new generation observes the
    // new interval.
    mapped_ranges_gen_.fetch_add(1, std::memory_order_release);
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.InvalidateMemory(addr, size);
    if (deferred_read_arm_) {
        // Runs before the range leaves the map, so no later drain protects
        // memory the guest has given back.
        buffer_cache.DropPendingReadArms(addr, size);
    }
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    mapped_ranges_gen_.fetch_add(1, std::memory_order_release);
    Skipcache::Framework::Instance().BumpMemGen();
}

bool Rasterizer::DynMemoProbe(const GraphicsPipeline* pipeline, u32 flags, u64 reg_stamp,
                              u64 dyn_gen, u64 pipe_gen) {
    using namespace VideoCore::Skipcache;
    auto& ctr = Skipcache::Framework::Instance().Counters(CacheId::DynState);
    const auto& m = dyn_memo_;
    if (m.flags == 0) {
        ++ctr.miss_cold;
        return false;
    }
    if (m.reg_stamp != reg_stamp) {
        ++ctr.miss_gen[LaneReg];
        return false;
    }
    if (m.flags != flags ||
        (dyn_class_stamp_ ? m.write_masks != pipeline->GetGraphicsKey().write_masks
                          : m.pipeline != pipeline)) {
        ++ctr.miss_key;
        return false;
    }
    if (m.dyn_gen != dyn_gen) {
        ++ctr.miss_gen[LaneTick];
        return false;
    }
    if (m.pipe_gen != pipe_gen) {
        ++ctr.miss_gen[LanePipe];
        return false;
    }
    return true;
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) {
    using VideoCore::Skipcache::CacheId;
    // Null is the disabled path's stand-in: every dereference below sits under
    // probing or verifying, both false whenever it is null.
    auto* const skipcache = dyn_memo_enabled_ ? &Skipcache::Framework::Instance() : nullptr;
    auto& dynamic_state = scheduler.GetDynamicState();
    const bool probing =
        skipcache != nullptr && skipcache->Active() && skipcache->ShouldProbe(CacheId::DynState);
    // is_indexed stays keyed: the primitive-restart ASSERT_MSG is live in
    // release builds, so an indexed transition must re-evaluate it.
    const u32 flags = 1u | (u32(attachment_feedback_loop) << 1) | (u32(is_indexed) << 2);
    u64 dyn_stamp{}, dyn_gen{}, dyn_pipe_gen{};
    bool dyn_would_hit = false, verifying = false;
    if (probing) {
        auto& ctr = skipcache->Counters(CacheId::DynState);
        ++ctr.eligible;
        const bool timed = skipcache->SampleTimer(CacheId::DynState);
        const u64 t0 = timed ? skipcache->Now() : 0;
        dyn_stamp =
            dyn_class_stamp_ ? liverpool->GetDynStateStamp() : liverpool->GetGfxStateStamp();
        // Read live, never from a draw-entry token: the binds that run before
        // this can flush the command buffer and re-arm every dirty bit.
        dyn_gen = dynamic_state.InvalidateGen();
        dyn_pipe_gen = skipcache->Gens().pipe_gen.load(std::memory_order_acquire);
        dyn_would_hit = DynMemoProbe(pipeline, flags, dyn_stamp, dyn_gen, dyn_pipe_gen);
        if (timed) {
            ctr.guard_ns += skipcache->CorrectSample(skipcache->Now() - t0);
            ++ctr.guard_samples;
        }
        if (dyn_would_hit) {
            ++ctr.hits;
            verifying = skipcache->ShouldVerify(CacheId::DynState);
            if (skipcache->MayConsume(CacheId::DynState) && !verifying) {
                return;
            }
        }
    }
    const bool timed_miss = probing && !dyn_would_hit && skipcache->SampleTimer(CacheId::DynState);
    const u64 miss_t0 = timed_miss ? skipcache->Now() : 0;
    const u32 before = verifying ? dynamic_state.DirtyBits() : 0u;

    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    // Sampled before Commit, which clears every bit it emits.
    if (verifying) {
        if (dynamic_state.DirtyBits() != before) {
            skipcache->RecordDivergence(CacheId::DynState, "dirty bit set under would-hit");
            dyn_memo_.flags = 0;
        } else {
            skipcache->RecordVerifyClean(CacheId::DynState);
        }
    }
    dynamic_state.Commit(instance, scheduler.CommandBuffer());

    if (probing && !dyn_would_hit) {
        auto& ctr = skipcache->Counters(CacheId::DynState);
        if (timed_miss) {
            ctr.miss_ns += skipcache->CorrectSample(skipcache->Now() - miss_t0);
            ++ctr.miss_samples;
        }
        // The body writes no register and bumps neither generation, so all
        // three lanes still hold their probe-time values.
        dyn_memo_ = {dyn_stamp, dyn_gen, dyn_pipe_gen,
                     pipeline,  flags,   pipeline->GetGraphicsKey().write_masks};
        skipcache->NotifyPopulated(CacheId::DynState);
    }
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->regs;

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

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
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
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp ||
                   zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op =
            uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
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

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

} // namespace Vulkan
