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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif


namespace Vulkan {


// GR2FORK: BundleAssembler intent-queue infrastructure. No secondary command buffers and no
// recorder thread: the data plane runs on the BundleAssembler jthread and writes vkCmd*
// directly to the primary command buffer.

// GR2FORK: the intent queue carries no high-water-mark instrumentation (the snapshot pool is
// the throttle); Bump*Hwm call sites in bundle_assembler.cpp hit empty inline stubs.


static void MakeUserData(Shader::PushData& push_data,
                         const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // PERF: Zero only buf_offsets (40 bytes) with a single memset instead of
    // value-initializing the full ~120-byte PushData every draw. ud_regs are
    // always overwritten by Info::PushUd for every register the shader consumes.
    std::memset(push_data.buf_offsets.data(), 0, push_data.buf_offsets.size());

    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
}

// GR2FORK PERF: image-view reference fast path in BindTextures' mip-expansion loop.
// GR2_NOVIEWREF=1 forces the copy/storage path for every binding; read once after config load.
static bool ViewRefFastPathEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOVIEWREF");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

// GR2FORK FIX: a V# with base_address == 0 binds the null buffer descriptor instead of
// falling through to ObtainBuffer(0, ...). Kill switch GR2_NONULLSHARP=1; read once after
// config load.
static bool NullSharpBindEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NONULLSHARP");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

// GR2FORK FIX: writable null-descriptor binds route to the real NULL_BUFFER/NULL_IMAGE sink;
// NVIDIA and the AMD Windows driver turn null-descriptor stores into writes to GPU VA 0x0 ->
// DEVICE_LOST (the CUSA03694 boot crash; RADV drops them). Kill switch: GR2_NONULLWRITESINK=1.
static bool NullWriteSinkEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NONULLWRITESINK");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

// GR2FORK FIX: render-target readiness gate. A torn (60fps-patch desync) color/depth target
// base can be unmapped, making the GPU store to ~VA 0 -> DEVICE_LOST on NVIDIA / AMD Windows;
// unmapped-base attachments are dropped (1-byte base probe). Kill switch: GR2_NORTREADY=1.
static bool RenderTargetReadyEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NORTREADY");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

// GR2FORK FIX: bogus-buffer sink. A torn writable storage V# with a garbage non-zero base can
// resolve to a real-but-wrong VkBuffer whose store corrupts memory or faults at VA 0 ->
// DEVICE_LOST; unmapped-base writable bindings route to the NULL_BUFFER sink. GR2_NOBOGUSSINK=1.
static bool BogusBufferSinkEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOBOGUSSINK");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool},
      br_cache_enabled_{Config::isBeginRenderingCacheEnabled()},
      bundle_assembler_{*this, liverpool_} {
    LOG_INFO(Render_Vulkan,
             "[GR2FORK Y-1.ASYNC] async flip: BundleAssembler runs on "
             "dedicated jthread; BLOCKING wrappers paired with WaitFor; "
             "assembler is sole writer of draw_scheduler; primary-CB "
             "direct recording (no recorder, no secondaries)");
    // GR2FORK PERF: boot echo for the RenderState copy diet.
    LOG_INFO(Render_Vulkan,
             "[GR2FORK R1] RenderState copy diet (trimmed copies + active-field equality): {} "
             "(forced on; GR2_NORSTRIM)",
             RStrimEnabled());
    // GR2FORK PERF: boot echo recording that the BindingSkipCache machinery
    // is compiled out (a code removal has no env switch).
    LOG_INFO(Render_Vulkan,
             "[GR2FORK BSC-STRIP] BindingSkipCache machinery removed "
             "(build-level A/B; no env switch)");
    // GR2FORK PERF: boot echo for the image-view reference fast path.
    // The hoisted-flag fold-in rides this switch: GR2_NOVIEWREF=1 forces
    // slow_view true for every binding.
    LOG_INFO(Render_Vulkan,
             "[GR2FORK B2] image-view reference fast path in BindTextures: {} "
             "(forced on; GR2_NOVIEWREF; B2.1 hoisted-flag fold-in active)",
             ViewRefFastPathEnabled());
    // GR2FORK PERF: boot echo for the hot-line CachedImageDescEntry layout
    // (a struct reorder cannot toggle at runtime).
    LOG_INFO(Render_Vulkan,
             "[GR2FORK B3] image_desc_cache hot-line entry layout active "
             "(build-level A/B; no env switch)");

    if (!Config::nullGpu()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    // BufferCache needs a back-reference so its ReadMemory SendCommand
    // lambda can route through PushPresenterRecord + WaitForAssembler.
    // Set after bundle_assembler_ is alive.
    buffer_cache.SetRasterizer(this);
}

Rasterizer::~Rasterizer() {
    // BundleAssembler's jthread is joined automatically by its destructor
    // (declared last in vk_rasterizer.h, destructed first).
}

void Rasterizer::CpSync() {
    // Routed through the intent queue; the body runs in DoCpSyncFromIntent.
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
    // GR2FORK PERF: the four special-mode branches (FCE, FmaskDecompress, Resolve, primitive-type
    // None) are rare guest-driver passes; hint them cold so the trailing `return true` for normal
    // triangle-list draws is straight-line.
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) [[unlikely]] {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear(regs);
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) [[unlikely]] {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        // Internal marker uses the inline helper; routing it through the
        // intent queue would defer it past the surrounding FilterDraw
        // return path.
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
        // Internal marker uses the inline helper.
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

    // GR2FORK PERF: hoist bit_width(key.mrt_mask) once; loop-invariant and consumed by all three
    // loops below.
    const s32 num_cbs = std::bit_width(key.mrt_mask);

    // Render-target identity hash: skip FindImage when CB/DB addresses are unchanged. FindImage
    // is the most expensive per-draw call; render targets change on pass switches, not between
    // draws within a pass.
    auto mix = [](u64 h, u64 v) noexcept -> u64 {
        return h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    };
    u64 rt_hash = 0x84222325cbf29ce4ULL;
    // GR2FORK PERF: pack mrt_mask (<= 8 bits) and skip_cb_binding (one bool) into a single mix
    // call instead of two, saving one serial mix iteration on rt_hash per draw.
    rt_hash = mix(rt_hash, static_cast<u64>(key.mrt_mask) |
                            (skip_cb_binding ? (1ULL << 8) : 0ULL));
    for (s32 cb = 0; cb < num_cbs; ++cb) {
        // GR2FORK PERF: the [[likely]] mirror of the slow-path
        // [[unlikely]] check below - for typical contiguous mrt_mask the
        // cb is active and contributes to rt_hash.
        if (!skip_cb_binding && regs.color_buffers[cb] &&
            regs.color_target_mask.GetMask(cb) &&
            (key.mrt_mask & (1 << cb))) [[likely]] {
            rt_hash = mix(rt_hash, regs.color_buffers[cb].Address());
            // The CB extent hint comes from the snapshot.
            rt_hash = mix(rt_hash, regs.last_cb_extent[cb].raw);
        }
    }
    // GR2FORK PERF: most graphics draws have depth or stencil
    // testing enabled - the depth/stencil-disabled case (no Z-buffer at
    // all) is the minority configuration in 3D titles.
    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) [[likely]] {
        rt_hash = mix(rt_hash, regs.depth_buffer.DepthAddress());
        // The DB extent hint comes from the snapshot.
        rt_hash = mix(rt_hash, regs.last_db_extent.raw);
        rt_hash = mix(rt_hash, regs.depth_htile_data_base.GetAddress());
    }

    // GR2FORK PERF: render targets change on pass switches, so consecutive draws hit the cache.
    // GR2FORK FIX: rt_cache_ hashes content, not ImageId - a recreate at the same VAddr leaves a
    // stale id (shadow flicker); accurateRenderTargetCacheEnabled() gates (forced true on GRR).
    const bool accurate = Config::accurateRenderTargetCacheEnabled();
    const bool rt_cache_active = !accurate;

    // GR-Remastered validated fast path: rt_hash mixes only addresses + extents, enough for GR2
    // but not GRR, whose shadow/cascade passes vary format/samples/tile/view at a stable VAddr;
    // full_key folds in the raw register structs the ImageInfo ctors read. Accurate mode only.
    u64 full_key = 0;
    if (accurate) {
        auto mix_bytes = [&mix](u64 h, const void* p, size_t n) noexcept -> u64 {
            const auto* b = reinterpret_cast<const u8*>(p);
            size_t i = 0;
            for (; i + sizeof(u64) <= n; i += sizeof(u64)) {
                u64 v;
                std::memcpy(&v, b + i, sizeof(u64));
                h = mix(h, v);
            }
            if (i < n) {
                u64 v = 0;
                std::memcpy(&v, b + i, n - i);
                h = mix(h, v);
            }
            return h;
        };
        full_key = rt_hash; // start from the address/extent hash
        full_key = mix_bytes(full_key, &regs.color_control, sizeof(regs.color_control));
        for (s32 cb = 0; cb < num_cbs; ++cb) {
            full_key = mix(full_key, static_cast<u64>(regs.color_target_mask.GetMask(cb)));
            full_key =
                mix_bytes(full_key, &regs.color_buffers[cb], sizeof(regs.color_buffers[cb]));
            full_key = mix(full_key, static_cast<u64>(regs.last_cb_extent[cb].raw));
        }
        full_key = mix_bytes(full_key, &regs.depth_control, sizeof(regs.depth_control));
        full_key = mix_bytes(full_key, &regs.depth_buffer, sizeof(regs.depth_buffer));
        full_key = mix_bytes(full_key, &regs.depth_view, sizeof(regs.depth_view));
        full_key = mix(full_key, regs.depth_htile_data_base.GetAddress());
        full_key = mix(full_key, static_cast<u64>(regs.last_db_extent.raw));
    }
    // GR2FORK FIX: the non-accurate hit must also match the image-registry generation. The hash
    // covers only guest addresses + extents, so an RT freed (GC/overlap) and recreated at the
    // same VA+extent across rapid avplayer stop/start hashed identically and the DEAD slot id was
    // re-recorded as a write target - the deferred vmaDestroyImage then freed the memory under
    // batches still writing it (the WRITE_INVALID image-heap device loss). Register/Unregister
    // already bump the generation, so every deletion path invalidates this hit for free.
    if (rt_cache_active && rt_cache_.valid && rt_cache_.hash == rt_hash &&
        rt_cache_.registry_generation == texture_cache.ImageRegistryGeneration()) [[likely]] {
        // Fast path: render targets unchanged. Reuse cached image_ids.
        // Must still add to bound_images and re-mark is_target (cleared by ResetBindings).
        for (s32 cb = 0; cb < num_cbs; ++cb) {
            auto& [image_id, desc] = cb_descs[cb];
            image_id = rt_cache_.cb_image_ids[cb];
            // GR2FORK PERF: for contiguous mrt_mask every cb in [0, num_cbs) is active and
            // carries an image_id; null entries appear only for sparse masks / cb-disabled /
            // zero target-mask, all uncommon.
            if (!image_id) [[unlikely]] {
                continue;
            }
            bound_images.emplace_back(image_id);
            texture_cache.GetImage(image_id).binding.is_target = 1u;
        }
        {
            auto& [image_id, desc] = db_desc;
            image_id = rt_cache_.db_image_id;
            // GR2FORK PERF: on an rt_hash match the cached db image_id is a real id for 3D
            // draws (the dominant case), mirroring the [[likely]] on depth/stencil enable.
            if (image_id) [[likely]] {
                bound_images.emplace_back(image_id);
                texture_cache.GetImage(image_id).binding.is_target = 1u;
            }
        }
        return;
    }

    // GR-Remastered validated fast path: full_key and ImageRegistryGeneration() both unchanged
    // means FindImage would return the identical ids, so cached ids and descs are reused; any
    // format/view switch, new overlap, GC free, or expand flips one and re-resolves fully.
    if (accurate && rt_cache_.valid && rt_cache_.full_key == full_key &&
        rt_cache_.registry_generation == texture_cache.ImageRegistryGeneration()) [[likely]] {
        for (s32 cb = 0; cb < num_cbs; ++cb) {
            auto& [image_id, desc] = cb_descs[cb];
            image_id = rt_cache_.cb_image_ids[cb];
            if (!image_id) [[unlikely]] {
                continue;
            }
            bound_images.emplace_back(image_id);
            texture_cache.GetImage(image_id).binding.is_target = 1u;
        }
        {
            auto& [image_id, desc] = db_desc;
            image_id = rt_cache_.db_image_id;
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
        // GR2FORK PERF: num_cbs = bit_width(mrt_mask), so for typical contiguous masks every cb in
        // range is active; the skip body fires only for sparse masks, disabled color control,
        // unconfigured col_buf, or zero target_mask - all uncommon at steady state.
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

    // GR2FORK PERF: mirrors the [[likely]] on the rt_hash compute above -
    // most 3D draws enable depth or stencil testing; the disabled-Z
    // fall-through is the minority configuration.
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
    // GR2FORK FIX: stamp the registry generation on EVERY fill (was accurate-only) - the
    // non-accurate hit now validates against it; one relaxed atomic load per cache miss.
    rt_cache_.registry_generation = texture_cache.ImageRegistryGeneration();
    if (accurate) {
        // Captured after every FindImage above so the next draw's fast-path compare sees the image
        // set exactly as left here.
        rt_cache_.full_key = full_key;
    }
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

    // Internal markers use the inline helpers so they bracket image.Clear
    // in the same dispatch order on both producer and assembler threads;
    // routing them through the intent queue would defer them past the Clear.
    DoScopeMarkerBeginInline(
        fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                    col_buf.CmaskAddress()),
        /*from_guest=*/false);
    image.Clear(clear_value, desc.view_info.range);
    DoScopeMarkerEndInline(/*from_guest=*/false);
}

namespace {
// GR2FORK: GR2's fullscreen motion-blur fragment-shader hash. Gated by
// Config::disableMotionBlur(); dropping the draw removes the motion-blur
// pass entirely.
constexpr u64 GR2_MOTION_BLUR_PS_HASH = 0xf696fe23ULL;
} // namespace

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    // Build a DrawIntent and hand off to BundleAssembler; the per-draw
    // work runs in DoDrawFromIntent below, called from
    // BundleAssembler::ProcessOne.
    DrawIntent intent;
    intent.type = is_indexed ? DrawIntent::Type::DrawIndexed
                             : DrawIntent::Type::Draw;
    // DoDrawFromIntent reads from the snapshot captured in PushAndProcess, so these intent.draw
    // fields are inert; they exist for forward compatibility with async dispatch.
    const auto& regs = liverpool->regs;
    intent.draw.num_indices = regs.num_indices;
    intent.draw.num_instances = regs.num_instances.NumInstances();
    intent.draw.vertex_offset = 0; // resolved per-pipeline in DoDrawFromIntent
    intent.draw.instance_offset = 0;
    intent.draw.index_offset = index_offset;
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoDrawFromIntent(const DrawIntent& intent) {
    // The body reads from `liverpool->GetSnapshot(intent.snapshot_idx)`,
    // NOT `liverpool->regs` - required for async dispatch correctness.
    const bool is_indexed = (intent.type == DrawIntent::Type::DrawIndexed);
    const u32 index_offset = intent.draw.index_offset;

    RENDERER_TRACE;

    // Check PopPendingOperations only every 16th draw - pending ops are deferred GPU-side
    // completions, so 16-draw latency is imperceptible; hint the 1-in-16 body cold.
    if ((draw_counter_++ & 15) == 0) [[unlikely]] {
        scheduler.PopPendingOperations();
    }

    const auto& regs = liverpool->GetSnapshot(intent.snapshot_idx);
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(regs);
    // GR2FORK PERF: pipeline is null only on cache failure / unsupported
    // shader stage combinations - an error path. Steady-state draws have a
    // valid pipeline.
    if (!pipeline) [[unlikely]] {
        return;
    }

    // FilterDraw reads only key-affecting regs (color_control.mode, primitive_type, depth
    // overrides), stable within a pipeline - skip it when the pipeline is unchanged, the dominant
    // case for consecutive draws; hint the rare path cold.
    if (pipeline != last_filter_pipeline_) [[unlikely]] {
        last_filter_result_ = FilterDraw(regs);
        last_filter_pipeline_ = pipeline;
    } else {
        // GR2FORK PERF: the same_pipeline counter tracks the FilterDraw
        // skip rate, which bounds how much br_slow time comes from
        // consecutive same-pipeline draws not skipped further upstream.
        GR2_INSTR_ON_SAME_PIPELINE();
    }
    // GR2FORK PERF: FilterDraw rejects only the special cb-disabled
    // depth-copy / FmaskDecompress / EliminateFastClear / Resolve / NoPrim
    // cases - the dominant draws pass through.
    if (!last_filter_result_) [[unlikely]] {
        return;
    }

    // GpuComm instrumentation: count this draw and track pipeline changes
    // for the unique-pipeline-per-frame upper bound. See gpucomm_metrics.h.
    GR2_INSTR_ON_DRAW(pipeline);

    PrepareRenderState(pipeline, regs);
    // GR2FORK PERF: BindResources returns false only on rare
    // resource-collision / pipeline-layout error paths; steady-state
    // draws successfully bind.
    if (!BindResources(pipeline, regs)) [[unlikely]] {
        return;
    }
    const auto& state = BeginRendering(pipeline, regs); // bound by const-ref (zero-copy hit path)

    // GR2FORK: drop GR2's fullscreen motion-blur draw before vertex-buffer bind / render
    // (ResetBindings mirrors the normal draw exit). The hash check runs first: disableMotionBlur()
    // is runtime-settable, so the cross-TU config call fires only on a hash match.
    {
        const auto* fs_hash = pipeline->TryGetStage(Shader::LogicalStage::Fragment);
        if (fs_hash && fs_hash->pgm_hash == GR2_MOTION_BLUR_PS_HASH) [[unlikely]] {
            if (Config::disableMotionBlur()) {
                ResetBindings();
                return;
            }
        }
    }

    {
        // GpuComm instrumentation: vertex/index buffer binding cost. Dirty VBO pages trigger
        // SynchronizeBuffer cascades inside the buffer cache, so this bucket also captures any
        // synchronous upload fired from within.
        GR2_INSTR_TIMER_DECL(vbuf_timer_);
        buffer_cache.BindVertexBuffers(*pipeline, regs);
        if (is_indexed) {
            buffer_cache.BindIndexBuffer(index_offset, regs);
        }
        GR2_INSTR_ON_BIND_VBUF(vbuf_timer_.ElapsedNs());
    }

    {
        // GpuComm instrumentation: Vulkan-command-submission ladder -
        // pushDescriptorSetKHR (pipeline->BindResources), dynamic state,
        // BeginRendering, BindPipelineCached, and the actual cmdbuf.draw[Indexed].
        GR2_INSTR_TIMER_DECL(dispatch_timer_);
        pipeline->BindResources({set_writes.data(), set_write_count}, buffer_barriers, push_data);
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
    // Build a DrawIntent and hand off to BundleAssembler.
    DrawIntent intent;
    intent.type = is_indexed ? DrawIntent::Type::DrawIndexedIndirect
                             : DrawIntent::Type::DrawIndirect;
    intent.indirect.address = arg_address;
    intent.indirect.offset = offset;
    // The intent shape distinguishes `size` and `stride` but the public API supplies only the
    // per-element stride; fill both so the contract seen by liverpool.cpp's PM4 handler is stable.
    intent.indirect.size = stride;
    intent.indirect.stride = stride;
    intent.indirect.max_count = max_count;
    intent.indirect.count_address = count_address;
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoDrawIndirectFromIntent(const DrawIntent& intent) {
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

    // GpuComm instrumentation.
    GR2_INSTR_ON_DRAW_INDIRECT();

    PrepareRenderState(pipeline, regs);
    if (!BindResources(pipeline, regs)) [[unlikely]] {
        return;
    }
    const auto& state = BeginRendering(pipeline, regs); // bound by const-ref (zero-copy hit path)

    // GR2FORK: motion-blur removal (mirrors DoDrawFromIntent); an
    // indirect draw can carry the fullscreen motion-blur pass too.
    // Hash-first ordering - see the DoDrawFromIntent site.
    {
        const auto* fs_hash = pipeline->TryGetStage(Shader::LogicalStage::Fragment);
        if (fs_hash && fs_hash->pgm_hash == GR2_MOTION_BLUR_PS_HASH) [[unlikely]] {
            if (Config::disableMotionBlur()) {
                ResetBindings();
                return;
            }
        }
    }

    {
        // GpuComm instrumentation: vbuf cost (mirrors Draw).
        GR2_INSTR_TIMER_DECL(vbuf_timer_);
        buffer_cache.BindVertexBuffers(*pipeline, regs);
        if (is_indexed) {
            buffer_cache.BindIndexBuffer(0, regs);
        }
        GR2_INSTR_ON_BIND_VBUF(vbuf_timer_.ElapsedNs());
    }

    // ObtainBuffer for indirect args is an argument-buffer fetch, not a vertex bind; left
    // untimed (counted toward the residual). DrawIndirect runs ~10/frame, so noise is negligible.
    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    {
        // GpuComm instrumentation: dispatch ladder (mirrors Draw).
        GR2_INSTR_TIMER_DECL(dispatch_timer_);
        pipeline->BindResources({set_writes.data(), set_write_count}, buffer_barriers, push_data);
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
    // Dispatch dimensions are captured from cs_state at intent-build time and re-read from the
    // snapshot by DoDispatchDirectFromIntent; carrying them in the intent keeps async correct.
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
    (void)intent; // dispatch dims are re-read from the snapshot; intent fields are inert here.

    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& regs = liverpool->GetSnapshot(intent.snapshot_idx);
    const auto& cs_program = regs.cs_program;
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline(regs);
    // GR2FORK PERF: null pipeline is a cache failure / unsupported
    // shader stage path - error case. Steady-state dispatches resolve a
    // valid pipeline.
    if (!pipeline) [[unlikely]] {
        return;
    }


    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);

    // GR2FORK PERF: ExecuteShaderHLE recognizes only a few clear/copy compute kernels; most
    // dispatches fall through to the regular path. The HLE body does not read `regs`; live
    // liverpool->regs merely satisfies the AmdGpu::Regs& signature without coupling the snapshot.
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) [[unlikely]] {
        return;
    }

    // GR2FORK PERF: BindResources fail-return is the rare
    // resource-collision / pipeline-layout error path; steady-state
    // dispatches successfully bind.
    if (!BindResources(pipeline, regs)) [[unlikely]] {
        return;
    }

    scheduler.EndRendering();
    pipeline->BindResources({set_writes.data(), set_write_count}, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.PrimaryCommandBuffer();
    BindPipelineCached(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);

    ResetBindings();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    // Build a DrawIntent and hand off to BundleAssembler.
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
    const VAddr address = intent.dispatch.indirect_address;
    const u32 offset = intent.dispatch.indirect_offset;
    const u32 size = intent.dispatch.indirect_size;

    RENDERER_TRACE;

    scheduler.PopPendingOperations();

    const auto& regs = liverpool->GetSnapshot(intent.snapshot_idx);
    (void)regs.cs_program; // captured but unread for indirect dispatch (dims live in the indirect buffer)
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline(regs);
    // GR2FORK PERF: mirrors DispatchDirect - null pipeline /
    // BindResources fail are error paths.
    if (!pipeline) [[unlikely]] {
        return;
    }

    if (!BindResources(pipeline, regs)) [[unlikely]] {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    scheduler.EndRendering();
    pipeline->BindResources({set_writes.data(), set_write_count}, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.PrimaryCommandBuffer();
    BindPipelineCached(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);

    ResetBindings();
}

u64 Rasterizer::Flush() {
    // Blocking Push + WaitFor: the PM4 producer waits for the assembler to submit. The tick is
    // captured before the push; under async execution it is approximate, and the only caller
    // discards the return value, so the approximation is observably equivalent.
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
    // Blocking: Push + WaitFor. The producer observes that
    // scheduler.Finish() has completed (GPU idle) before returning to the
    // PM4 handler.
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
    // Blocking Push + WaitFor: OnSubmit's writes to draw_scheduler must be observable by
    // subsequent producer-thread reads before the next PM4 packet is processed.
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
    // GR2FORK PERF: fused submit-done flush. Separate OnSubmit() and Flush() calls cost two
    // blocking Push+WaitFor round-trips per guest submit (the second pure fixed overhead); the
    // flush body runs inside the OnSubmit round-trip with ordering unchanged (GC then Flush).
    SubmitInfo info{};
    scheduler.Flush(info);
}

void Rasterizer::PrepareDescriptorDeltaCache(const Pipeline* pipeline) {
    // GR2FORK PERF: per-binding push-descriptor emit dedup (~44% skip rate; GR2_NODESCDELTA=1
    // forces every write; key/epoch design in vk_rasterizer.h). Set-path pipelines keep
    // force_write: a fresh VkDescriptorSet per draw makes skipped bindings a VUID-08114 hazard.
    static const bool kDeltaDisabled = []() noexcept {
        const char* e = std::getenv("GR2_NODESCDELTA");
        return (e && e[0] == '1');
    }();

    if (kDeltaDisabled || !pipeline || !pipeline->UsesPushDescriptors()) {
        desc_cache_force_write_ = true;
        return;
    }

    const u64 tick = scheduler.CurrentTick();
    const vk::PipelineLayout layout = pipeline->GetLayout();
    if (tick != desc_cache_tick || pipeline != desc_cache_pipeline ||
        layout != desc_cache_layout) {
        // Invalidations are dominated by same-tick pipeline switches;
        // tick rotation is a small minority.
        if (++desc_cache_epoch == 0) [[unlikely]] {
            desc_cache = {};
            desc_cache_epoch = 1;
        }
        desc_cache_tick = tick;
        desc_cache_pipeline = pipeline;
        desc_cache_layout = layout;
    }
    desc_cache_force_write_ = false;
}

bool Rasterizer::ShouldWriteDescriptorImpl(u32 binding, vk::DescriptorType type, u64 a, u64 b, u64 c) {
    // GR2FORK PERF: The set-path short-circuit lives in the inline
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
    // GR2FORK PERF: the IsComputeImageCopy / IsComputeMetaClear / IsComputeImageClear trio
    // short-circuits on the [[likely]] !IsCompute() fast path (1-2 cycles each for graphics
    // draws), so the OR-chain body fires only on rare meta-compute dispatches.
    if (IsComputeImageCopy(pipeline, regs) || IsComputeMetaClear(pipeline, regs) ||
        IsComputeImageClear(pipeline, regs)) [[unlikely]] {
        return false;
    }

    // GpuComm instrumentation: count graphics-only BindResources entries
    // (the compute meta paths above bypass the descriptor-construction work
    // being measured).
    GR2_INSTR_ON_BR_ENTER();

    set_write_count = 0;
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();

    PrepareDescriptorDeltaCache(pipeline);

    Shader::Backend::Bindings binding{};
    MakeUserData(push_data, regs);

    // GR2FORK: no per-stage binding-skip cache - GR2 rewrites user_data between essentially every
    // draw, so identical-binding runs almost never exist (fast-path hits ~0.03% vs ~0.5% sys
    // maintenance); the descSetBindingSkipCache TOML getter is left unreferenced.

    // Full binding path. GpuComm instrumentation times descriptor construction (per-stage
    // BindBuffers + BindTextures + cache bookkeeping) and stops before the uses_dma
    // SynchronizeBuffersInRange block - that is buffer-upload work tracked separately.
    GR2_INSTR_TIMER_DECL(slow_timer_);
    uint32_t slow_stages_bound_ = 0;

    bool uses_dma = false;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) continue;
        ++slow_stages_bound_;
        uses_dma |= stage->uses_dma;
        stage->PushUd(binding, push_data);
        const auto* resolved = pipeline_cache.GetResolvedStageResources(*stage);
        BindBuffers(*stage, resolved, binding, push_data, regs);
        BindTextures(*stage, resolved, binding, regs);
    }

    // No post-iteration null-descriptor coverage pass: under primary-CB direct recording the
    // descriptor set retains prior draws' values, so unwritten layout-declared bindings are
    // tolerated (secondary command buffers would need null fills to avoid GPUVM faults).

    // GpuComm instrumentation: fire the slow-path metric BEFORE the
    // uses_dma block. The DMA sync is buffer-upload work tracked
    // separately, not the descriptor-construction work targeted here.
    GR2_INSTR_ON_BR_SLOW(slow_timer_.ElapsedNs(), slow_stages_bound_);

    if (uses_dma) {
        const auto dma_sync_state = buffer_cache.GetDmaSyncState();
        const u64 mapped_ranges_gen = mapped_ranges_gen_.load(std::memory_order_acquire);
        const bool needs_sync =
            !dma_sync_state_valid_ ||
            dma_sync_state.cpu_dirty_generation != last_dma_sync_state_.cpu_dirty_generation ||
            dma_sync_state.buffer_registry_generation !=
                last_dma_sync_state_.buffer_registry_generation ||
            mapped_ranges_gen != last_dma_mapped_ranges_gen_;
        if (needs_sync) {
            // A pre-sweep snapshot leaves mutations that race the traversal visible to the next
            // draw instead of treating them as synchronized.
            GR2_INSTR_TIMER_DECL(sync_timer_);
            Common::RecursiveSharedLock lock{mapped_ranges_mutex};
            for (auto& range : mapped_ranges) {
                buffer_cache.SynchronizeBuffersInRange(range.lower(),
                                                       range.upper() - range.lower());
            }
            last_dma_sync_state_ = dma_sync_state;
            last_dma_mapped_ranges_gen_ = mapped_ranges_gen;
            dma_sync_state_valid_ = true;
            GR2_INSTR_ON_SYNC_BUFFER(sync_timer_.ElapsedNs());
        }
        fault_process_pending = true;
    }

    return true;
}


void Rasterizer::BindPipelineCached(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) {
    const auto cmdbuf = scheduler.PrimaryCommandBuffer();
    const u64 tick = scheduler.CurrentTick();
    // GR2FORK PERF: Pipeline rebinds within the same cmdbuf tick are
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
    // GR2FORK PERF: BindResources calls this trio on every Draw / Dispatch entry and graphics
    // draws far outnumber compute dispatches, so !IsCompute() is the dominant early return.
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

void Rasterizer::BindBuffers(const Shader::Info& stage,
                             const Shader::ResolvedStageResources* resolved,
                             Shader::Backend::Bindings& binding, Shader::PushData& push_data,
                             const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // GR2FORK PERF: register-held write count; stored back once at exit.
    u32 dw_n = set_write_count;
    // GR2FORK PERF: no intra-call FindBuffer/ObtainBuffer dedup cache - measured hit rate was
    // zero across tens of millions of lookups (shaders never bind one buffer to multiple slots in
    // a single call); the cross-call lookups inside VideoCore::BufferCache carry the real load.

    // GR2FORK PERF: no eager per-binding FindBuffer. An eager lookup walks the page table on
    // every guest binding while the dominant UBO stream path never reads the result, and first
    // touch runs a needless CreateBuffer + generation bump. Kill switch: GR2_NOFBSKIP=1.
    static const bool fbskip_enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOFBSKIP");
        return !(e && e[0] == '1');
    }();

    // Single fused pass: resolve buffer_id and bind in one iteration.
    // Eliminates the buffer_bindings intermediate vector (clear + emplace_back + read-back).
    for (u32 i = 0; i < stage.buffers.size(); i++) {
        const auto& desc = stage.buffers[i];
        const auto vsharp = resolved ? resolved->buffers[i] : desc.GetSharp(stage);

        VideoCore::BufferId buffer_id{};
        u64 size = 0;
        const bool is_guest_buffer =
            !desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0;
        if (is_guest_buffer) {
            size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            if (!fbskip_enabled) {
                buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
            }
        }

        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        // Dispatch on the predicate, not `!buffer_id`: with the eager lookup skipped, a guest
        // buffer legitimately carries an empty id into ObtainBuffer, and FindBuffer never returns
        // a null id for a nonzero base, so routing is unchanged.
        if (!is_guest_buffer) {
            // GR2FORK PERF: GdsBuffer (Global Data Share) is a hardware
            // feature essentially unused by retail PS4 titles - the rarest
            // branch in this chain.
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) [[unlikely]] {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) [[likely]] {
                // GR2FORK PERF: Flatbuf carries the flattened user-data buffer read by
                // ReadConst - virtually every compiled shader has one, making it the dominant
                // special-buffer case.
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);

                struct FlatbufCacheEntry {
                    u64 stage_key = 0;
                    u64 tick = 0;
                    u64 offset = 0;
                    u32 size = 0;
                    u32 alignment = 0;
                    vk::Buffer buffer{};
                    bool valid = false;
                };
                static thread_local std::array<FlatbufCacheEntry, 32> flatbuf_cache{};

                const u64 tick = scheduler.CurrentTick();
                const u64 stage_key = static_cast<u64>(stage.pgm_hash) ^
                                      (static_cast<u64>(desc.sharp_idx) << 1);

                FlatbufCacheEntry& e = flatbuf_cache[stage_key & (flatbuf_cache.size() - 1)];

                const bool identity_match =
                    e.valid && e.stage_key == stage_key &&
                    e.tick == tick && e.size == ubo_size && e.alignment == alignment &&
                    e.buffer == vk_buffer.Handle();

                u64 offset;
                // GR2FORK PERF: the cached stream bytes are the exact prior payload. memcmp
                // exits on the first changed dword and avoids hashing the full flat buffer.
                if (identity_match &&
                    (ubo_size == 0 ||
                     std::memcmp(vk_buffer.mapped_data.data() + e.offset,
                                 stage.flattened_ud_buf.data(), ubo_size) == 0)) [[likely]] {
                    offset = e.offset;
                } else {
                    offset = vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                    e.stage_key = stage_key;
                    e.tick = scheduler.CurrentTick();
                    e.offset = offset;
                    e.size = ubo_size;
                    e.alignment = alignment;
                    e.buffer = vk_buffer.Handle();
                    e.valid = true;
                }

                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) [[unlikely]] {
                // GR2FORK PERF: BdaPagetable is used only by
                // shaders performing buffer-device-address pointer-walks -
                // a small subset.
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) [[unlikely]] {
                // GR2FORK PERF: FaultBuffer is the shadPS4 fault-
                // recovery sideband - only present when fault-process is
                // wired up.
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) [[unlikely]] {
                // GR2FORK PERF: SharedMemory (LDS) is compute-only;
                // graphics draws never reach this branch.
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = regs.cs_program;
                // GR2FORK FIX: guest-derived and previously unclamped (u32 wrap); a failed Map
                // (over-ring size) used to record a fillBuffer at offset 0 anyway - an OOB
                // fixed-function GPU write. Null-descriptor sink on any bad case instead.
                const u64 lds_size =
                    u64(cs_program.SharedMemSize()) * u64(cs_program.NumWorkgroups());
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                if (data == nullptr || lds_size == 0 ||
                    offset + lds_size > lds_buffer.SizeBytes()) [[unlikely]] {
                    LOG_CRITICAL(Render_Vulkan,
                                 "GR2 write-gate: LDS request {}B unmappable, null-bound",
                                 lds_size);
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    lds_buffer.Commit();

                    constexpr u64 kGpuFillThreshold = 256;
                    if (lds_size >= kGpuFillThreshold) {
                        auto cmdbuf = scheduler.PrimaryCommandBuffer();
                        cmdbuf.fillBuffer(lds_buffer.Handle(), offset, lds_size, 0);

                        buffer_barriers.push_back(vk::BufferMemoryBarrier2{
                            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                            .dstAccessMask = vk::AccessFlagBits2::eShaderRead |
                                             vk::AccessFlagBits2::eShaderWrite,
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
                }
            } else if (instance.IsNullDescriptorSupported()) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                auto& null_buffer = buffer_cache.GetBuffer(VideoCore::NULL_BUFFER_ID);
                buffer_infos.emplace_back(null_buffer.Handle(), 0, VK_WHOLE_SIZE);
            }
        } else if (vsharp.base_address == 0 && NullSharpBindEnabled()) [[unlikely]] {
            // GR2FORK FIX: a V# resolving to base_address 0 is a stale/zeroed sharp;
            // ObtainBuffer(0, ...) lets an NVIDIA shader write fault at VA 0x0 -> DEVICE_LOST
            // (CUSA04943; RADV absorbs it). Bind the null sink; GR2_NONULLSHARP=1 restores.
            if (instance.IsNullDescriptorSupported()) {
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
                // GR2FORK PERF: GetBarrier returns nullopt when the buffer's access mode/stage
                // is unchanged from the last draw - the steady-state case; the barrier-emit
                // body fires only on first touch or access transitions.
                buffer_barriers.emplace_back(*barrier);
            }
            // GR2FORK PERF: is_written && is_formatted requires both
            // SSBO write + typed buffer access - uncommon combination in
            // most shaders.
            if (desc.is_written && desc.is_formatted) [[unlikely]] {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }

        }

        const u32 dst_binding = binding.unified++;
        const vk::DescriptorType dtype =
        is_storage ? vk::DescriptorType::eStorageBuffer : vk::DescriptorType::eUniformBuffer;

        // GR2FORK FIX: two torn writable shapes route to the real NULL_BUFFER sink (see
        // NullWriteSinkEnabled / BogusBufferSinkEnabled): a null descriptor left in
        // buffer_infos.back(), or a bogus buffer whose base is unmapped. Reads stay as resolved.
        if (is_storage && desc.is_written) [[likely]] {
            const bool null_handle =
                static_cast<VkBuffer>(buffer_infos.back().buffer) == VK_NULL_HANDLE;
            const bool sink_null = null_handle && NullWriteSinkEnabled();
            const bool sink_bogus = !null_handle && BogusBufferSinkEnabled() &&
                                    !desc.IsSpecial() && vsharp.base_address != 0 &&
                                    !IsMapped(vsharp.base_address, 1);
            if (sink_null || sink_bogus) [[unlikely]] {
                auto& sink = buffer_cache.GetBuffer(VideoCore::NULL_BUFFER_ID);
                buffer_infos.back() = vk::DescriptorBufferInfo{sink.Handle(), 0, VK_WHOLE_SIZE};
                static std::atomic<bool> logged_null{false};
                static std::atomic<bool> logged_bogus{false};
                if (sink_null && !logged_null.exchange(true, std::memory_order_relaxed)) {
                    LOG_INFO(Render_Vulkan,
                             "GR2 null-write-sink: absorbed a writable null STORAGE BUFFER "
                             "bind (ps={:#018x} slot={}) into the real NULL_BUFFER — prevents "
                             "WRITE_INVALID @ 0x0 device-lost. Disable: GR2_NONULLWRITESINK=1",
                             static_cast<u64>(stage.pgm_hash), dst_binding);
                } else if (sink_bogus &&
                           !logged_bogus.exchange(true, std::memory_order_relaxed)) {
                    LOG_INFO(Render_Vulkan,
                             "GR2 bogus-buffer-sink: absorbed a writable STORAGE BUFFER bind "
                             "over an UNMAPPED base (ps={:#018x} slot={} base={:#x}) into the "
                             "real NULL_BUFFER — prevents bogus-buffer write / WRITE_INVALID @ "
                             "0x0 (60fps-desync torn descriptor). Disable: GR2_NOBOGUSSINK=1",
                             static_cast<u64>(stage.pgm_hash), dst_binding,
                             vsharp.base_address);
                }
            }
        }

        const auto bi_handle = reinterpret_cast<u64>(static_cast<VkBuffer>(buffer_infos.back().buffer));
        const u64 bi_offset = static_cast<u64>(buffer_infos.back().offset);
        const u64 bi_range  = static_cast<u64>(buffer_infos.back().range);

        if (ShouldWriteDescriptor(dst_binding, dtype, bi_handle, bi_offset, bi_range)) {
            auto& w = set_writes[dw_n++];
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
    set_write_count = dw_n;
}

void Rasterizer::BindTextures(const Shader::Info& stage,
                              const Shader::ResolvedStageResources* resolved,
                              Shader::Backend::Bindings& binding,
                              const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // GR2FORK PERF: register-held write count; stored back once at exit.
    u32 dw_n = set_write_count;
    image_bindings.clear();
    bool any_needs_rebind = false; // OPT: Track during first pass instead of separate scan

    // GR2FORK PERF: caches live on the Rasterizer instance - large thread_local arrays inflate
    // TLS and can hit pthread_create() EINVAL (22) on shadPS4's custom stacks. No intra-call
    // FindImage dedup (measured hit rate zero); the cross-call find_image_pcache_ stays.
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

    // GR2FORK PERF: Epoch-based validity - avoids zeroing 640+ bytes of stamp arrays per call.
    const u32 epoch = ++bind_textures_epoch_;
    // Handle epoch wrap (extremely unlikely, ~every 4 billion BindTextures calls)
    if (epoch == 0) {
        bind_textures_epoch_ = 1;
        // find_texture_cache_stamp_ feeds the FindTextureCached lambda below.
        find_texture_cache_stamp_.fill(0);
    }

    // De-dup FindTexture() within one BindTextures() call: GR2 binds the same
    // {image_id,type,view_info} across stages, and FindTexture takes a lock (UpdateImage) plus
    // view lookup, so avoiding duplicates cuts rwlock + Image::FindView + barrier traffic.
    auto& find_texture_cache = find_texture_cache_;
    auto& find_texture_cache_stamp = find_texture_cache_stamp_;
    auto hash_view_info = [](const VideoCore::ImageViewInfo& v) noexcept -> u64 {
        // GR2FORK PERF: Pack fields into 2-3 mix calls instead of 11.
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
                                     // GR2FORK PERF: consecutive same-shader
                                     // draws bind the same images, so the
                                     // warmed cache hits on most calls.
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

                                 for (u32 image_index = 0; image_index < stage.images.size();
                                      ++image_index) {
                                     const auto& image_res = stage.images[image_index];
                                     auto tsharp = resolved ? resolved->images[image_index]
                                                            : image_res.GetSharp(stage);
                                     const u8 num_bindings = static_cast<u8>(
                                         std::min<u32>(255u, image_res.NumBindings(tsharp)));
                                     const bool single_mip =
                                         tsharp.base_level == 0 && tsharp.last_level == 0;

                                     if (texture_cache.IsMeta(tsharp.Address())) [[unlikely]] {
            // GR2FORK PERF: defensive sanity log; never expected to
            // fire on shipping titles. Hint the body cold so the surrounding
            // hot binding work stays straight-line.
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }


        if (tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) [[unlikely]] {
            ImageBindingInfo& nb = image_bindings.emplace_back();
            // GR2FORK FIX: the layout always reserves NumBindings() slots for this ImageResource;
            // preserve num_bindings so the second-pass null emission writes the full array.
            nb.num_bindings = num_bindings;
            nb.is_written = image_res.is_written;
            nb.single_mip = single_mip;
            continue;
        }

        // GR2FORK FIX: block-compressed format + macro-tiled 2D array mode is a garbage T# -
        // ImageInfo::UpdateSize would hit ASSERT(!props.is_block), and the upstream layer-count
        // clamp misses this shape. Bail to keep the layout symmetric with FormatInvalid above.
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
                nb.num_bindings = num_bindings;
                nb.is_written = image_res.is_written;
                nb.single_mip = single_mip;
                continue;
            }
        }

        const u32 flags = make_flags(image_res);
        const u64 key = make_key(image_res, flags);
        // GR2FORK PERF: Mix the key with a multiplicative hash before slot
        // selection - plain key & (size-1) uses low bits, which cluster
        // when pgm_hash is stable.
        const u64 mixed_key = (key ^ (key >> 16)) * 0x9e3779b97f4a7c15ULL;
        CachedImageDescEntry& ce = image_desc_cache_[static_cast<u32>(mixed_key >> 20) & (image_desc_cache_.size() - 1)];

        // GR2FORK PERF: image_desc_cache hits are dominant in steady state (same-pipeline
        // draws resubmit the same T# bytes per slot); hint the miss body cold so lookup +
        // memcmp lay out straight-line.
        if (!(ce.valid && ce.key == key && std::memcmp(&ce.image, &tsharp, sizeof(tsharp)) == 0)) [[unlikely]] {
            ce.key = key;
            ce.image = tsharp;
            ce.desc = VideoCore::TextureCache::ImageDesc(tsharp, image_res);
            // GR2FORK PERF: keep the hot-line mirror in lockstep with the
            // payload - desc only mutates here, so mirror == desc.info.
            // guest_address is an invariant.
            ce.guest_address = ce.desc.info.guest_address;
            // Desc content changed - any pcache entry stamped with
            // the old desc_gen must re-validate.
            ++ce.desc_gen;
            ce.valid = true;
        }
        const auto* base_desc = &ce.desc;

        // GR2FORK FIX: host-side GNF readiness gate (shares GR2_NORTREADY with the
        // render-target gate). GR2's own texture-descriptor validator (eboot
        // 0x1218ca0) traps on a not-yet-streamed descriptor; the >30fps eboot patch
        // guards that trap so the game forwards not-ready descriptors instead of
        // crashing. Their backing range is unmapped/partial, so binding one makes
        // the GPU sample/store an unmapped address -> WRITE_INVALID @ 0x0 / stale
        // -> DEVICE_LOST (the 60fps-desync crash). Do here what the game's validator
        // used to do, but gracefully: if the texture's full backing range is not
        // mapped, drop to a null binding (same shape as the FormatInvalid/garbage-T#
        // bailouts above). Base + last-byte probe (streaming fills contiguously;
        // avoids a full-range interval scan per texture). Covers sampled AND storage
        // images - a null storage descriptor's writes are discarded by
        // robustBufferAccess2/nullDescriptor instead of faulting.
        if (RenderTargetReadyEnabled()) [[likely]] {
            const VAddr tex_base = base_desc->info.guest_address;
            const u64 tex_size = base_desc->info.guest_size;
            if (tex_base != 0 && tex_size != 0 &&
                (!IsMapped(tex_base, 1) || !IsMapped(tex_base + tex_size - 1, 1))) [[unlikely]] {
                static std::atomic<bool> logged_once{false};
                if (!logged_once.exchange(true, std::memory_order_relaxed)) {
                    LOG_INFO(Render_Vulkan,
                             "GR2 tex-ready: dropped a not-streamed texture (addr={:#x} "
                             "size={:#x}) for one draw - 60fps GNF desync; prevents "
                             "WRITE_INVALID device-lost. Disable: GR2_NORTREADY=1",
                             tex_base, tex_size);
                }
                ImageBindingInfo& nb = image_bindings.emplace_back();
                nb.num_bindings = num_bindings;
                nb.is_written = image_res.is_written;
                nb.single_mip = single_mip;
                continue;
            }
        }

        // De-dup FindImage within this call (common when multiple stages alias).
        VideoCore::TextureCache::FindImageWithViewResult found{};
        bool found_cache_hit = false;
        {
            // GR2FORK PERF: the pcache key folds in the texture guest_address; without it,
            // rebinding textures to one shader slot collides on the same pslot and the cached
            // image_id goes stale (~34% Validate failures). GR2_NOPCACHEGEN=1 restores Validate.
            static const bool pcachegen_enabled = []() noexcept {
                const char* e = std::getenv("GR2_NOPCACHEGEN");
                return !(e && e[0] == '1');
            }();

            const u64 pkey = key ^ (ce.guest_address >> 8);
            const u64 pkey_mixed = (pkey ^ (pkey >> 16)) * 0xbf58476d1ce4e5b9ULL;
            const auto pslot = static_cast<u32>(pkey_mixed >> 20) &
            static_cast<u32>(find_image_pcache_.size() - 1);
            auto& pe = find_image_pcache_[pslot];
            // GR2FORK PERF: the pcache resolves the iteration ~66% of the
            // time in steady state.
            if (pe.valid && pe.key == pkey && pe.base == base_desc) [[likely]] {
                // Matching stamps prove the stored result by determinism (see the
                // PersistentFindImageCacheEntry doc), covering subresource results Validate can
                // never pass; stale stamps fall back to Validate and re-stamp.
                if (pcachegen_enabled &&
                    pe.registry_gen == texture_cache.ImageRegistryGeneration() &&
                    pe.desc_gen == ce.desc_gen) [[likely]] {
                    found = pe.res;
                    found_cache_hit = true;
                } else if (texture_cache.ValidateCachedFindImage(*base_desc, pe.res.image_id,
                                                                 false)) {
                    found = pe.res;
                    found_cache_hit = true;
                    pe.registry_gen = texture_cache.ImageRegistryGeneration();
                    pe.desc_gen = ce.desc_gen;
                }
            }
        }
        // GR2FORK PERF: the full page-table walk fires on pcache miss -
        // the rarest path through this block.
        if (!found_cache_hit) [[unlikely]] {
            found = texture_cache.FindImageWithView(*base_desc, false, false);
            // GR2FORK PERF: pkey matches the lookup-side computation (key XOR guest_address)
            // so the insert lands in the probed pslot. Store only on miss: a hit-path store
            // would rewrite identical bytes, dirtying a pcache line per binding per draw.
            const u64 pkey2 = key ^ (ce.guest_address >> 8);
            const u64 pkey_mixed2 = (pkey2 ^ (pkey2 >> 16)) * 0xbf58476d1ce4e5b9ULL;
            const auto pslot = static_cast<u32>(pkey_mixed2 >> 20) &
            static_cast<u32>(find_image_pcache_.size() - 1);
            find_image_pcache_[pslot] = PersistentFindImageCacheEntry{
                .key = pkey2,
                .base = base_desc,
                .res = found,
                // The walk just produced `found` under the CURRENT
                // registry/desc state - stamp both generations.
                .registry_gen = texture_cache.ImageRegistryGeneration(),
                .desc_gen = ce.desc_gen,
                .valid = true,
            };
        }

        auto image_id = found.image_id;

        auto* image = &texture_cache.GetImage(image_id);
        // GR2FORK PERF: depth_id is set only for color images acting as the color half of a
        // depth-stencil aliased pair (stencil-attachment redirect) - the minority case.
        if (image->depth_id) [[unlikely]] {
            // If this image has an associated depth image, it's a stencil attachment.
            // Redirect the access to the actual depth-stencil buffer.
            // GR2FORK FIX: no deletion path ever cleared depth_id, so the redirect could resolve
            // a freed/reused slot and record it as a write target. Validate; self-heal if dead.
            if (texture_cache.IsImageSlotAllocated(image->depth_id) &&
                True(texture_cache.GetImageUntouched(image->depth_id).flags &
                     VideoCore::ImageFlagBits::Registered)) {
                image_id = image->depth_id;
                image = &texture_cache.GetImage(image_id);
            } else {
                image->depth_id = {};
            }
        }
        // OPT: Track needs_rebind here instead of a separate O(N) scan loop.
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
        b.is_written = image_res.is_written;
        b.single_mip = single_mip;
        b.view_mip = static_cast<s16>(found.view_mip);
        b.view_slice = static_cast<s16>(found.view_slice);
        // GR2FORK FIX: capture NumBindings for the 2nd-loop mip-fallback
        // expansion. DynamicIndex images want >1 consecutive descriptor slots.
        b.num_bindings = num_bindings;
    }

    // GR2FORK PERF: second pass rebinds images updated after binding; the first pass tracks
    // whether any rebind is needed (no O(N) scan), and UpdateImage is deduped per ImageId via a
    // persistent tick-based cache, skipping repeat shared_locks within a command-buffer tick.
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
    // GR2FORK PERF: hoisted storage for the override/mip-expansion path
    // inside the binding loop. The common path binds views by reference and
    // never touches this; the slow path fully reassigns it before use.
    VideoCore::ImageViewInfo view_override_storage;
    // GR2FORK PERF: hoist the magic-static consult out of the mip loop -
    // ViewRefFastPathEnabled() costs a guard-variable acquire load + branch per call (~0.25% sys
    // measured); one consult per call collapses the per-binding test to a precomputed byte.
    const bool view_ref_ok = ViewRefFastPathEnabled();
    const bool null_write_sink_ok = NullWriteSinkEnabled();
    for (auto& b : image_bindings) {
        const auto* base_desc = b.desc;
        // GR2FORK FIX: storage-ness comes from the carried is_written flag, never from desc: the
        // garbage-T# bails leave desc null while the layout slot is still eStorageImage, so typing
        // by desc mistypes the write and skips the null-write sink (store at VA 0x0, DEVICE_LOST).
        const bool is_storage = b.is_written;

        if (!base_desc || !b.image_id) {
            // GR2FORK FIX: the layout reserves num_bindings descriptors even for an invalid
            // sharp; emit that many image_infos to keep binding.unified and the
            // descriptorCount-N write aligned.
            const u32 null_count = b.num_bindings ? b.num_bindings : 1u;
            for (u32 null_i = 0; null_i < null_count; ++null_i) {
                // GR2FORK FIX: a writable storage image that didn't resolve takes the real
                // NULL_IMAGE sink (robustImageAccess2 discards the store) instead of faulting
                // at GPU VA 0x0 on NVIDIA / AMD Windows; sampled nulls keep the null descriptor.
                const bool redirect_storage_null = is_storage && null_write_sink_ok;
                if (null_descriptors_supported && !redirect_storage_null) {
                    image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
                } else {
                    if (redirect_storage_null) [[unlikely]] {
                        static std::atomic<bool> logged_once{false};
                        if (!logged_once.exchange(true, std::memory_order_relaxed)) {
                            LOG_INFO(Render_Vulkan,
                                     "GR2 null-write-sink: absorbed a writable null STORAGE "
                                     "IMAGE bind (ps={:#018x}) into the real NULL_IMAGE — "
                                     "prevents WRITE_INVALID @ 0x0 device-lost. Disable: "
                                     "GR2_NONULLWRITESINK=1",
                                     static_cast<u64>(stage.pgm_hash));
                        }
                    }
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
                    // GR2FORK FIX: a sink view bound as eStorageImage must keep eStorage usage;
                    // ImageView creation strips it when the view info is not marked storage.
                    if (redirect_storage_null) {
                        view_info.is_storage = true;
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
            auto* image_ptr = &texture_cache.GetImageUntouched(b.image_id);
            // GR2FORK PERF: Only check needs_rebind when we know at least one image needs it.
            if (any_needs_rebind && image_ptr->binding.needs_rebind) {
                image_ptr->binding = {};
                const auto replacement =
                    texture_cache.FindImageWithView(*base_desc, false, false);
                b.image_id = replacement.image_id;
                b.view_mip = static_cast<s16>(replacement.view_mip);
                b.view_slice = static_cast<s16>(replacement.view_slice);
                image_ptr = &texture_cache.GetImage(b.image_id);
            }

            bound_images.emplace_back(b.image_id);

            // GR2FORK PERF: the first pass already touched this image. Only a replacement id
            // needs the LRU update above.
            auto& image = *image_ptr;

            // GR2FORK PERF: the common path binds base_desc->view_info by reference - copying
            // through two stack structs stalls store-to-load forwarding (~0.9% sys); overrides
            // assign into hoisted view_override_storage. GR2_NOVIEWREF=1 forces the copy path.

            // GR2FORK FIX (upstream #4075): DynamicIndex mip-fallback layouts declare
            // descriptorCount = num_mips, so emit one view + image_info per slot; unpopulated
            // tail slots fault on RADV (DEVICE_LOST); for num_bindings == 1 the loop runs once.
            const VideoCore::ImageViewInfo& base_view = base_desc->view_info;
            const bool view_overridden = (b.view_mip > 0) || (b.view_slice > 0);
            const u32 num_bindings = b.num_bindings ? b.num_bindings : 1u;
            // GR2FORK PERF: fold the three fast-path disqualifiers into one
            // byte before the mip loop (bitwise | - all operands already
            // computed, no short-circuit branches).
            const bool slow_view =
                view_overridden | (num_bindings > 1u) | !view_ref_ok;

            for (u32 mip_offset = 0; mip_offset < num_bindings; ++mip_offset) {
                const VideoCore::ImageViewInfo* effective_view = &base_view;
                if (slow_view) [[unlikely]] {
                    view_override_storage = base_view;
                    if (b.view_mip > 0) {
                        view_override_storage.range.base.level = b.view_mip;
                    }
                    if (b.view_slice > 0) {
                        view_override_storage.range.base.layer = b.view_slice;
                    }
                    if (num_bindings > 1) {
                        // GR2FORK FIX: each expanded descriptor slot must view exactly one mip
                        // level; a base view covering the full chain with a shifted baseMipLevel
                        // violates VUID-VkImageViewCreateInfo-subresourceRange-01718.
                        const u32 desired_level =
                            view_override_storage.range.base.level + mip_offset;
                        const u32 max_level = image.info.resources.levels > 0
                                                  ? image.info.resources.levels - 1
                                                  : 0;
                        view_override_storage.range.base.level =
                            std::min<u32>(desired_level, max_level);
                        view_override_storage.range.extent.levels = 1;
                    }
                    effective_view = &view_override_storage;
                }
                const VideoCore::ImageViewInfo& mip_view_info = *effective_view;

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

                // Layout transitions are needed once per image but are harmless in the loop
                // (Transit no-ops when already in the target state); drive them from the full
                // requested view range so the barrier covers every slot about to be sampled.
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
                        // Skip Transit when already in target state.
                        const auto storage_access = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
                        if (!image.IsInState(vk::ImageLayout::eGeneral, storage_access)) {
                            image.Transit(vk::ImageLayout::eGeneral, storage_access, mip_view_info.range);
                        }
                    } else {
                        const auto new_layout = image.info.props.is_depth
                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                        : vk::ImageLayout::eShaderReadOnlyOptimal;
                        // Skip Transit for sampled images already in correct state.
                        // In steady-state rendering, most sampled textures stay in ShaderReadOnlyOptimal
                        // between draws. This avoids Transit -> GetBarriers function call overhead.
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

        // GR2FORK FIX: the layout always reserves num_bindings slots for this ImageResource and
        // both branches above emit exactly that many image_infos; advance binding.unified and emit
        // one write with descriptorCount = num_bindings (num_bindings == 1 is unchanged).
        const u32 slot_count = b.num_bindings ? b.num_bindings : 1u;

        const u32 dst_binding = binding.unified;
        binding.unified += slot_count;
        const vk::DescriptorType dtype = is_storage ? vk::DescriptorType::eStorageImage
        : vk::DescriptorType::eSampledImage;

        const u32 write_count = slot_count;
        const size_t array_first = image_infos.size() - slot_count;

        // GR2FORK FIX: last-line net for the null-write sink - no storage slot may reach
        // descriptor emission with a null view (NVIDIA and the AMD Windows driver turn
        // null-descriptor stores into writes to GPU VA 0x0 -> DEVICE_LOST). Runs before the
        // delta-cache signature reads so a rewritten slot hashes as its final content.
        if (is_storage && null_write_sink_ok) {
            for (size_t i = array_first; i < image_infos.size(); ++i) {
                if (!image_infos[i].imageView) [[unlikely]] {
                    // The view must keep eStorage usage to be legal in an eStorageImage slot;
                    // ImageView creation strips it when the view info is not marked storage.
                    VideoCore::ImageViewInfo sink_view_info{};
                    sink_view_info.is_storage = true;
                    auto& sink_view = texture_cache.FindTexture(
                        VideoCore::NULL_IMAGE_ID, VideoCore::TextureCache::BindingType::Texture,
                        sink_view_info);
                    image_infos[i] = vk::DescriptorImageInfo{
                        VK_NULL_HANDLE, *sink_view.image_view, vk::ImageLayout::eGeneral};
                    static std::atomic<bool> logged_once{false};
                    if (!logged_once.exchange(true, std::memory_order_relaxed)) {
                        LOG_WARNING(Render_Vulkan,
                                    "GR2 null-write-sink: storage image slot {} (ps={:#018x}) "
                                    "reached descriptor emission with a null view; rebound to "
                                    "the real NULL_IMAGE - prevents WRITE_INVALID @ 0x0 "
                                    "device-lost. Disable: GR2_NONULLWRITESINK=1",
                                    dst_binding, static_cast<u64>(stage.pgm_hash));
                    }
                }
            }
        }

        const auto ii_sampler =
        reinterpret_cast<u64>(static_cast<VkSampler>(image_infos[array_first].sampler));
        const auto ii_view =
        reinterpret_cast<u64>(static_cast<VkImageView>(image_infos[array_first].imageView));
        const u64 ii_layout = static_cast<u64>(static_cast<u32>(image_infos[array_first].imageLayout));

        // ShouldWriteDescriptor hashes only the first slot of a multi-slot array; tail slots are
        // consecutive mips of the same image and change together. Fold write_count into the
        // signature if more robust invalidation is ever needed.
        if (ShouldWriteDescriptor(dst_binding, dtype, ii_sampler, ii_view, ii_layout)) {
            auto& w = set_writes[dw_n++];
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

    // GR2FORK PERF: sampler fast-path cache - samplers are immutable, so a direct-mapped cache
    // safely skips GetSampler()'s S# hash + map lookup on hit (>90% steady-state GR2). The 2x-u64
    // S# probes via raw mul-xor + exact compare - no XXH3 call, no collision false-hits.
    for (u32 sampler_index = 0; sampler_index < stage.samplers.size(); ++sampler_index) {
        const auto& sampler = stage.samplers[sampler_index];
        auto ssharp = resolved ? resolved->samplers[sampler_index] : sampler.GetSharp(stage);
        if (sampler.disable_aniso) {
            if (image_bindings[sampler.associated_image].single_mip) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }

        const u64 samp_mix =
            (ssharp.raw0 ^ (ssharp.raw1 * 0x9e3779b97f4a7c15ULL)) * 0xbf58476d1ce4e5b9ULL;
        const u32 samp_slot = static_cast<u32>(samp_mix >> 32) & (SamplerCacheSize - 1);
        auto& sc = sampler_cache_[samp_slot];
        vk::Sampler vk_sampler;
        if (sc.valid && sc.raw0 == ssharp.raw0 && sc.raw1 == ssharp.raw1) {
            vk_sampler = sc.sampler;
        } else {
            vk_sampler = texture_cache.GetSampler(ssharp, regs.ta_bc_base);
            sc.raw0 = ssharp.raw0;
            sc.raw1 = ssharp.raw1;
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
            auto& w = set_writes[dw_n++];
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
    set_write_count = dw_n;
}


const RenderState& Rasterizer::BeginRendering(const GraphicsPipeline* pipeline,
                                              const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // GR2FORK PERF: stamp-skip fast path (see BeginRenderingCache in vk_rasterizer.h): requires
    // pipeline+stamp match, no armed CMASK clear, stable image layouts, and no needs_rebind (a
    // cache merge/expand can reassign image_ids). GR2_NOBRCACHE=1 only; latched at construction.
    const bool br_cache_enabled = br_cache_enabled_;
    if (br_cache_enabled) {
        // The pipeline stamp comes from the snapshot.
        const u64 cur_stamp = regs.gfx_pipeline_stamp;
        const u64 cur_tick = scheduler.CurrentTick();
        if (br_cache_.valid && br_cache_.pipeline == pipeline &&
            br_cache_.stamp == cur_stamp && br_cache_.tick == cur_tick) {
            // GR2FORK PERF: generation fast path - three u64 compares replace the
            // per-attachment verification loop when nothing it checks can have changed (gen
            // fields in vk_rasterizer.h). GR2_NOBRGEN=1 falls through to the loop on every hit.
            static const bool brgen_enabled = []() noexcept {
                const char* e = std::getenv("GR2_NOBRGEN");
                return !(e && e[0] == '1');
            }();
            if (brgen_enabled &&
                br_cache_.registry_gen == texture_cache.ImageRegistryGeneration() &&
                br_cache_.meta_gen == texture_cache.MetaClearGeneration() &&
                br_cache_.layout_gen == VideoCore::Image::LayoutGeneration()) [[likely]] {
                GR2_INSTR_ON_BT_REPLAY();
                attachment_feedback_loop = br_cache_.attachment_feedback_loop;
                return br_cache_.state;
            }
            bool ok = true;
            // Check clear flags + layouts + needs_rebind on color attachments.
            for (u32 i = 0; i < br_cache_.cb_count && ok; ++i) {
                const auto& e = br_cache_.cb_data[i];
                // GR2FORK PERF: meta_addr is set only for surfaces
                // with HTILE/CMASK/FMASK active - the majority of cached
                // CB attachments don't carry one.
                if (e.meta_addr &&
                    texture_cache.IsMetaCleared(e.meta_addr, e.slice)) [[unlikely]] {
                    ok = false;
                    break;
                }
                // GR2FORK PERF: cached entries almost always carry
                // an image_id - empty slots happen only for masked-out
                // colour buffers, which are uncommon at steady state.
                if (e.image_id) [[likely]] {
                    auto& img = texture_cache.GetImage(e.image_id);
                    // GR2FORK PERF: within a render pass the attachment's layout and rebind
                    // flag are stable; a mismatch fires on cache-merge / layout transition,
                    // the slow path.
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
                // GR2FORK PERF: bt_replay counts BeginRendering cache hits - the dominant
                // exit; hint it for code layout.
                GR2_INSTR_ON_BT_REPLAY();
                // The loop just verified everything the gens guard
                // (for this cache's attachments) under the CURRENT state -
                // re-stamp so the next hit takes the gen fast path.
                br_cache_.registry_gen = texture_cache.ImageRegistryGeneration();
                br_cache_.meta_gen = texture_cache.MetaClearGeneration();
                br_cache_.layout_gen = VideoCore::Image::LayoutGeneration();
                attachment_feedback_loop = br_cache_.attachment_feedback_loop;
                return br_cache_.state;
            }
        }
    }

    attachment_feedback_loop = false;
    // GR2FORK PERF: slow path - invalidate br_cache_ until repopulated at end of function; the
    // cb/db loops below fill cb_data / db_data and the trailing populate block re-sets valid.
    br_cache_.valid = false;
    br_cache_.cb_count = 0;
    br_cache_.has_db_attachment = false;
    const auto& key = pipeline->GetGraphicsKey();
    // Build into the persistent member so the function can return `const RenderState&`.
    // GR2FORK PERF: reset only the active prefix instead of the full `= RenderState{}` (ctor
    // memset + ~830B blit); the mrt count is known from the key. GR2_NORSTRIM=1 restores.
    const u32 gr2_mrt_n = std::bit_width(key.mrt_mask);
    if (RStrimEnabled()) [[likely]] {
        cur_render_state_.ResetActivePrefix(gr2_mrt_n);
    } else {
        cur_render_state_ = RenderState{};
    }
    RenderState& state = cur_render_state_;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u32>::max();
    state.num_color_attachments = gr2_mrt_n;

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

    // GR2FORK PERF: batch this pass switch's attachment transitions into one
    // vkCmdPipelineBarrier2; inline Transit costs 2-6 driver barrier calls where one suffices.
    // The flush below precedes the consuming renderpass. Kill switch: GR2_NOBARBATCH=1.
    static const bool barbatch_enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOBARBATCH");
        return !(e && e[0] == '1');
    }();
    VideoCore::Image::Barriers pending_att_barriers;
    const auto transit_attachment = [&](VideoCore::Image& image, vk::ImageLayout layout,
                                        vk::AccessFlags2 access,
                                        std::optional<VideoCore::SubresourceRange> range) {
        if (!barbatch_enabled) {
            image.Transit(layout, access, range);
            return;
        }
        constexpr vk::PipelineStageFlags2 dst_stage =
            vk::PipelineStageFlagBits2::eAllGraphics |
            vk::PipelineStageFlagBits2::eComputeShader;
        const auto barriers = image.GetBarriers(layout, access, dst_stage, range);
        pending_att_barriers.insert(pending_att_barriers.end(), barriers.begin(),
                                    barriers.end());
    };

    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        // GR2FORK PERF: pre-zero this cb's cache slot so the `continue` path
        // below leaves a well-defined empty entry (image_id=0, meta_addr=0,
        // expected_layout=eUndefined). The hit-path checks gate on these.
        br_cache_.cb_data[cb] = {};
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            continue;
        }
        // GR2FORK FIX: render-target readiness gate (see RenderTargetReadyEnabled()). A torn
        // color descriptor can resolve to a stale image_id with an unmapped base; drop it like
        // an unresolvable target so the GPU never stores to VA 0. 1-byte base probe.
        if (RenderTargetReadyEnabled()) [[likely]] {
            const auto& cbr = regs.color_buffers[cb];
            if (!IsMapped(cbr.Address(), 1)) [[unlikely]] {
                static std::atomic<bool> logged_once{false};
                if (!logged_once.exchange(true, std::memory_order_relaxed)) {
                    LOG_INFO(Render_Vulkan,
                             "GR2 rt-ready: dropped a torn/unmapped COLOR target "
                             "(addr={:#x}) for one draw — prevents WRITE_INVALID @ 0x0 "
                             "device-lost (60fps-desync torn descriptor). Disable: "
                             "GR2_NORTREADY=1",
                             cbr.Address());
                }
                continue;
            }
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
            transit_attachment(*image,
                               instance.IsAttachmentFeedbackLoopLayoutSupported()
                                   ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                   : vk::ImageLayout::eGeneral,
                               vk::AccessFlagBits2::eColorAttachmentWrite, {});
            attachment_feedback_loop = true;
        } else {
            // Skip Transit when render target is already in correct state.
            // Consecutive draws with the same render targets hit this ~95% of the time.
            const auto ca_access = vk::AccessFlagBits2::eColorAttachmentWrite |
                                   vk::AccessFlagBits2::eColorAttachmentRead;
            if (!image->IsInState(vk::ImageLayout::eColorAttachmentOptimal, ca_access)) {
                transit_attachment(*image, vk::ImageLayout::eColorAttachmentOptimal, ca_access,
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
        // GR2FORK PERF: capture for br_cache_ population. expected_layout
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

    // GR2FORK FIX: render-target readiness gate (see RenderTargetReadyEnabled()) for the DEPTH
    // target: an unmapped base means no depth target (has_depth stays false) so the GPU never
    // writes depth/stencil through a torn pointer. 1-byte base probe.
    const bool depth_unmapped =
        RenderTargetReadyEnabled() && db_desc.first &&
        !IsMapped(regs.depth_buffer.DepthAddress(), 1);
    if (depth_unmapped) [[unlikely]] {
        static std::atomic<bool> logged_once{false};
        if (!logged_once.exchange(true, std::memory_order_relaxed)) {
            LOG_INFO(Render_Vulkan,
                     "GR2 rt-ready: dropped a torn/unmapped DEPTH target (addr={:#x}) "
                     "for one draw — prevents WRITE_INVALID @ 0x0 device-lost "
                     "(60fps-desync torn descriptor). Disable: GR2_NORTREADY=1",
                     regs.depth_buffer.DepthAddress());
        }
    }
    if (auto image_id = db_desc.first; image_id && !depth_unmapped) {
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

        // GR2FORK FIX: picking the depth layout from is_storage alone binds READ_ONLY_OPTIMAL on
        // draws that write depth/stencil - VUID-06886/06887, RADV hangs (CUSA03694 Nevi Hand
        // DEVICE_LOST). Writable if is_storage OR depth writes OR the stencil_ref_front writemask.
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
        // Skip Transit when depth target is already in correct state.
        if (!image.IsInState(new_layout, ds_access)) {
            transit_attachment(image, new_layout, ds_access, desc.view_info.range);
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
        // GR2FORK PERF: capture for br_cache_ population.
        {
            auto& e = br_cache_.db_data;
            e.image_id = image_id;
            e.slice = slice;
            e.meta_addr = htile_address;
            e.expected_layout = image.backing->state.layout;
            br_cache_.has_db_attachment = true;
        }
    }

    // Flush the batched attachment transitions - one EndRendering + one pipelineBarrier2 per
    // pass switch - before returning, so the barriers precede the renderpass that consumes the
    // new layouts.
    if (!pending_att_barriers.empty()) {
        scheduler.EndRendering();
        scheduler.PrimaryCommandBuffer().pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<u32>(pending_att_barriers.size()),
            .pImageMemoryBarriers = pending_att_barriers.data(),
        });
    }

    if (state.num_layers == std::numeric_limits<u32>::max()) {
        state.num_layers = 1;
    }

    // OPT: Pre-compute hash for fast equality rejection in BeginRendering.
    state.ComputeHash();

    // GR2FORK PERF: populate br_cache_ with an eLoad-forced copy so stamp-matched replays never
    // re-run a loadOp=eClear. Never cache while depth_render_control depth/stencil clear_enable
    // is set - the flags persist and a cached eLoad suppresses the clear (shadow flicker).
    if (br_cache_enabled) {
        const bool reg_clear_active =
            regs.depth_render_control.depth_clear_enable ||
            regs.depth_render_control.stencil_clear_enable;
        if (!reg_clear_active) {
            br_cache_.cb_count = state.num_color_attachments;
            br_cache_.attachment_feedback_loop = attachment_feedback_loop;
            br_cache_.pipeline = pipeline;
            // Stamp from the snapshot - the same value the fast-path
            // comparison above used.
            br_cache_.stamp = regs.gfx_pipeline_stamp;
            br_cache_.tick = scheduler.CurrentTick();
            // Stamp the generations under which this populate ran -
            // the slow path above just resolved/transitioned everything
            // under the current registry/meta/layout state.
            br_cache_.registry_gen = texture_cache.ImageRegistryGeneration();
            br_cache_.meta_gen = texture_cache.MetaClearGeneration();
            br_cache_.layout_gen = VideoCore::Image::LayoutGeneration();

            // GR2FORK PERF: eLoad-forcing applies identically on both paths; GR2_NORSTRIM=1 keeps
            // the full copy -> mutate -> hash -> assign sequence while the default builds straight
            // into br_cache_.state - one active-span copy instead of two full ~830B copies.
            const auto gr2_force_eload = [](RenderState& s) {
                for (u32 cb = 0; cb < s.num_color_attachments; ++cb) {
                    s.color_attachments[cb].loadOp = vk::AttachmentLoadOp::eLoad;
                    s.color_attachments[cb].clearValue = vk::ClearValue{};
                }
                if (s.has_depth) {
                    s.depth_attachment.loadOp = vk::AttachmentLoadOp::eLoad;
                    s.depth_attachment.clearValue = vk::ClearValue{};
                }
                if (s.has_stencil) {
                    s.stencil_attachment.loadOp = vk::AttachmentLoadOp::eLoad;
                    s.stencil_attachment.clearValue = vk::ClearValue{};
                }
                s.ComputeHash();
            };
            if (RStrimEnabled()) [[likely]] {
                br_cache_.state.CopyActiveFrom(state);
                gr2_force_eload(br_cache_.state);
            } else {
                RenderState eload_state = state;
                gr2_force_eload(eload_state);
                br_cache_.state = eload_state;
            }
            br_cache_.valid = true;
        }
        // else: leave br_cache_.valid = false (set at slow-path entry) - no
        // cache hit possible until a draw with no register-driven clear runs.
    }

    return state;
}

void Rasterizer::Resolve(const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // Extent hints come from the snapshot.
    const auto& mrt0_hint = regs.last_cb_extent[0];
    const auto& mrt1_hint = regs.last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    // Internal markers use the inline helpers.
    DoScopeMarkerBeginInline(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                          regs.color_buffers[0].Address(),
                                          regs.color_buffers[1].Address()),
                              /*from_guest=*/false);
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    DoScopeMarkerEndInline(/*from_guest=*/false);
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil,
                                  const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // The extent hint comes from the snapshot.
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

    // Internal markers use the inline helpers.
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
    // Routed through the intent queue; the body runs in
    // DoFillBufferFromIntent. FIFO order with surrounding draws is
    // preserved - the assembler processes intents in PM4-emit order.
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
    // Routed through the intent queue; the body runs in DoCopyBufferFromIntent.
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
    // GR2FORK FIX: the 16-bit PM4 gds_index can address past the 64KB GDS mapping; clamp.
    if (gds_offset + sizeof(u32) > gds_buf->mapped_data.size()) [[unlikely]] {
        LOG_WARNING(Render_Vulkan, "GR2 write-gate: OOB GDS read at {:#x} dropped", gds_offset);
        return 0;
    }
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
    // Underlying texture/buffer data changed - caches are stale.
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

// GR2FORK PERF: async-signal-safe per-thread (POD TLS) cache of recent positive IsMapped
// intervals on the signal-handler page-fault path (~35K faults/sec; ~100-230 locked cycles ->
// ~30-45 cached). Map/Unmap bump mapped_ranges_gen_ (release); an acquire load drops stale hits.
bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }

    struct CacheEntry {
        VAddr base;   // [base, limit) - 0,0 means empty slot
        VAddr limit;
    };
    static constexpr size_t kCacheSize = 4;
    thread_local std::array<CacheEntry, kCacheSize> tls_cache{};
    thread_local u64 tls_gen = ~u64{0};

    const VAddr query_end = addr + size;
    // GR2FORK FIX: a failed upstream resolve can pass addr = u64(-1); query_end then wraps below
    // addr, defeating the limit and straddle checks, so the readback path would index page_table
    // far OOB (an AV on Windows). One predicted-not-taken branch before query_end is consumed.
    if (query_end < addr) [[unlikely]] {
        return false;
    }
    const u64 cur_gen = mapped_ranges_gen_.load(std::memory_order_acquire);

    if (cur_gen == tls_gen) [[likely]] {
        // Cache valid for this generation. Linear scan over kCacheSize
        // entries - branch-friendly, fits in one cacheline.
        for (const auto& e : tls_cache) {
            if (addr >= e.base && query_end <= e.limit) {
                return true;
            }
        }
    } else {
        // Generation advanced: drop the cache without memsetting - overwriting on insert is
        // fine and most entries get replaced anyway; reset tls_gen so future hits compare
        // against the current epoch.
        tls_cache.fill({0, 0});
        tls_gen = cur_gen;
    }

    // Cache miss - full lookup. Do find(addr) instead of contains(range)
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
    // Round-robin replacement is good enough - most workloads have only
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
    // GR2FORK PERF: bump after the mutation is committed; release pairs with IsMapped's acquire
    // load so any thread observing the new gen also observes the new interval.
    mapped_ranges_gen_.fetch_add(1, std::memory_order_release);
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    // GR2FORK FIX: guest pages are imported as GPU buffer backing (external_memory_host), so the
    // pages the caller releases right after this hook can still be WRITTEN by queued GPU work -
    // deferred handle destruction does not keep the pages themselves alive. GR2 frees its video
    // buffers on the avplayer StateStop event while frames referencing them are still in flight,
    // and the resulting GPU write into freed pages is the WRITE_INVALID device loss. Drain the
    // GPU before the pages go away when the range carries GPU-side content; such unmaps happen
    // only at content teardown (video stop, level unload), so gameplay never pays this wait.
    // GR2FORK FIX: Finish()/cache eviction must run on the assembler thread - it is the single
    // writer of draw_scheduler's open command buffer (Finish from this guest thread ends a
    // command buffer mid-record = corrupted stream), and DeferOperation pushes race the
    // assembler's pop otherwise. Blocking hop: the caller releases the pages only after this
    // returns, which is the whole point of the drain.
    // GR2FORK FIX: the gpu_content probe (FindImageFromRange mutates Picked flags) and the
    // rt/br cache clears also belong on the assembler - computing/writing them from this guest
    // thread raced Register/UnregisterImage and the per-draw cache reads.
    const u32 unmap_seq = PushPresenterRecord([this, addr, size] {
        const bool gpu_content = buffer_cache.IsRegionGpuModified(addr, size) ||
                                 texture_cache.FindImageFromRange(addr, size);
        if (gpu_content) {
            scheduler.Finish();
        }
        buffer_cache.InvalidateMemory(addr, size);
        texture_cache.UnmapMemory(addr, size);
        rt_cache_.valid = false;
        br_cache_.valid = false;
    });
    WaitForAssembler(unmap_seq);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    // GR2FORK PERF: bump after the mutation is committed.
    // See MapMemory for ordering rationale.
    mapped_ranges_gen_.fetch_add(1, std::memory_order_release);
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed,
                                    const AmdGpu::LiverpoolRegsSnapshot& regs) const {
    auto& dynamic_state = scheduler.GetDynamicState();

    // Skip all 6 dynamic-state sub-functions when no register changed since the last draw - each
    // reads dozens of regs only for Commit to find zero dirty flags (~236 lines of work saved per
    // hit); within a render pass the dirty flag stays false for most consecutive draws.

    // The dirty bit comes from the snapshot: PM4 clears the live dirty flag inside
    // Liverpool::CaptureSnapshot() under sole-writer ownership, so regs.dynamic_dirty is a
    // frozen "was dirty at capture" view and needs no clear here.
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

// Inline marker helpers. Internal Rasterizer-side callers use these so the
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
    // ScopedMarkerInsert (no color) packs color=0; vk::DebugUtilsLabelEXT's default color is
    // {0,0,0,0}, so the Vulkan call is identical to a no-color insert. ScopedMarkerInsertColor
    // passes the real RGBA-packed u32.
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

namespace {
// Packs a string_view into the intent's inline char buffer, truncating to
// (DrawIntent::kScopeMarkerInlineLen - 1) chars plus a null terminator; runs on whichever
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
    // Routed through the intent queue; the string is copied (truncated) into the intent's
    // inline buffer at packing time so the consumer never reaches back to PM4-packet memory.
    // The body runs in DoScopeMarkerBeginInline.
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
    // Routed through the intent queue.
    DrawIntent intent;
    intent.type = DrawIntent::Type::ScopeMarkerEnd;
    intent.scope_marker_end.from_guest = static_cast<u8>(from_guest ? 1 : 0);
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::DoScopeMarkerEndFromIntent(const DrawIntent& intent) {
    DoScopeMarkerEndInline(intent.scope_marker_end.from_guest != 0);
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    // Routed through the intent queue. Unified with the color variant via
    // color=0 - the Vulkan call is observably equivalent.
    DrawIntent intent;
    intent.type = DrawIntent::Type::ScopedMarkerInsert;
    PackInlineString(intent.scope_marker.str, str);
    intent.scope_marker.color = 0;
    intent.scope_marker.from_guest = static_cast<u8>(from_guest ? 1 : 0);
    bundle_assembler_.PushAndProcess(intent);
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    // Routed through the intent queue. Same intent type as the no-color
    // variant; color is carried in the payload.
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

// PresenterRecord consumer: invokes the type-erased closure synthesized by the templated
// producer-side PushPresenterRecord<F> and lets it self-destruct, releasing the heap allocation.
void Rasterizer::DoPresenterRecordFromIntent(const DrawIntent& intent) {
    intent.presenter_record.invoke_and_destroy(intent.presenter_record.state);
}


} // namespace Vulkan
