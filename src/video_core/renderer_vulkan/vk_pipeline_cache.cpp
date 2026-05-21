// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <ranges>
#include <limits>
#include <cstring>
#include <mutex>
#include <vector>
#include <xxhash.h>

#include "common/config.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "core/debug_state.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {

// =========================================================================
// OPT(GR2 v78): Pipeline compile graveyard.
// =========================================================================
// std::future<T>'s destructor joins the worker thread. When a Vulkan driver
// hangs inside vkCreateGraphicsPipelines, that thread never returns, and
// any future destruction (PipelineCache dtor, pending-map erase) blocks
// forever. We dump "abandoned" futures here on permafail and on PipelineCache
// teardown; this storage is intentionally never freed so futures never get
// destructed. On process exit the OS reaps the hung threads via _exit.
struct PipelineCompileGraveyard {
    std::mutex mu;
    // Heap-allocated vector we never delete — this is the leak-on-purpose.
    std::vector<std::future<std::unique_ptr<GraphicsPipeline>>>* graves = nullptr;
    void Bury(std::future<std::unique_ptr<GraphicsPipeline>> f) {
        std::lock_guard lk{mu};
        if (!graves) {
            graves = new std::vector<std::future<std::unique_ptr<GraphicsPipeline>>>();
        }
        graves->push_back(std::move(f));
    }
};

PipelineCompileGraveyard& Graveyard() {
    // Leaked Meyers-style singleton — heap allocated + never deleted, so its
    // dtor never runs (which is exactly what we want; see comment above).
    static auto* g = new PipelineCompileGraveyard();
    return *g;
}

} // namespace

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

// PERF(GR2 v16): Hash RuntimeInfo by stage, matching operator== semantics exactly.
// Cannot hash raw bytes because the union has padding and some stages use custom
// equality (e.g. FragmentRuntimeInfo only compares inputs[0..num_inputs]).
static u64 HashRuntimeInfoForStage(const Shader::RuntimeInfo& ri) {
    // Combine helper: boost::hash_combine style
    auto mix = [](u64 seed, u64 v) -> u64 {
        return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
    };
    u64 h = static_cast<u64>(ri.stage);
    switch (ri.stage) {
    case Shader::Stage::Local:
        h = mix(h, ri.ls_info.ls_stride);
        break;
    case Shader::Stage::Export:
        h = mix(h, ri.es_info.vertex_data_size);
        break;
    case Shader::Stage::Vertex: {
        const auto& v = ri.vs_info;
        h = mix(h, XXH3_64bits(v.outputs.data(), sizeof(v.outputs)));
        // PERF(GR2FORK v1.18): pack the eight scalar fields into three
        // packed mix calls. Each scalar pair fits naturally in a u64:
        //   - step_rate_0 | step_rate_1   (two u32s, exact 64-bit pack)
        //   - num_outputs | hs_output_cp_stride (two u32s, exact 64-bit pack)
        //   - tess bools (3 bits) + tess_type/topology/partitioning (small enums)
        // Cuts the per-Vertex mix-chain length from 9 to 4 iterations,
        // saving ~25 cycles of serial dependency per call. Each packed u64
        // still diffuses through mix() with the same quality as separate
        // calls would have produced.
        h = mix(h, (static_cast<u64>(v.step_rate_0)) |
                    (static_cast<u64>(v.step_rate_1) << 32));
        h = mix(h, (static_cast<u64>(v.num_outputs)) |
                    (static_cast<u64>(v.hs_output_cp_stride) << 32));
        h = mix(h, static_cast<u64>(v.tess_emulated_primitive ? 1u : 0u) |
                    (static_cast<u64>(v.emulate_depth_negative_one_to_one ? 1u : 0u) << 1) |
                    (static_cast<u64>(v.clip_disable ? 1u : 0u) << 2) |
                    (static_cast<u64>(static_cast<u32>(v.tess_type)) << 8) |
                    (static_cast<u64>(static_cast<u32>(v.tess_topology)) << 16) |
                    (static_cast<u64>(static_cast<u32>(v.tess_partitioning)) << 24));
        break;
    }
    case Shader::Stage::Hull:
        // Uses default operator==, so hash all fields.
        h = mix(h, XXH3_64bits(&ri.hs_info, sizeof(ri.hs_info)));
        break;
    case Shader::Stage::Geometry: {
        const auto& g = ri.gs_info;
        // PERF(GR2FORK v1.18): pack four small scalar fields into a single
        // packed mix instead of four separate ones. num_outputs/num_invocations/
        // output_vertices are bounded by GS spec limits (< 256 each in any
        // realistic pipeline), in_primitive is a small AmdGpu enum. All
        // comfortably fit in a u64 with 16-bit slots.
        const u64 packed_scalars =
            static_cast<u64>(g.num_outputs) |
            (static_cast<u64>(g.num_invocations) << 16) |
            (static_cast<u64>(g.output_vertices) << 32) |
            (static_cast<u64>(static_cast<u32>(g.in_primitive)) << 48);
        h = mix(h, packed_scalars);
        // PERF(GR2FORK v1.18): g.outputs is std::array<OutputMap,3> = 12B.
        // XXH3 function-call overhead dominates the actual hashing for this
        // size. Read as 8B + 4B inline (Output is u8, OutputMap is 4B array).
        // Two mix iterations replace 1 XXH3 + 1 mix — same chain length, but
        // the XXH3 function call is gone (~15-20 cycles saved).
        static_assert(sizeof(g.outputs) == 12);
        u64 outputs_lo;
        u32 outputs_hi;
        std::memcpy(&outputs_lo, g.outputs.data(), 8);
        std::memcpy(&outputs_hi,
                    reinterpret_cast<const u8*>(g.outputs.data()) + 8, 4);
        h = mix(h, outputs_lo);
        h = mix(h, static_cast<u64>(outputs_hi));
        // PERF(GR2FORK v1.18): out_primitive is std::array<GsOutputPrimitiveType,4>
        // = 16 bytes, but each enum value occupies only 6 bits in the source
        // hardware (regs_vertex.h: outprim_type : 6). Pack all 4 into a single
        // u64 byte-aligned for safety. Saves the XXH3 function call (~15-20
        // cycles) plus 1 mix iteration.
        static_assert(sizeof(g.out_primitive) == 16);
        const auto& op = g.out_primitive;
        const u64 op_packed =
            static_cast<u64>(static_cast<u32>(op[0])) |
            (static_cast<u64>(static_cast<u32>(op[1])) << 8) |
            (static_cast<u64>(static_cast<u32>(op[2])) << 16) |
            (static_cast<u64>(static_cast<u32>(op[3])) << 24);
        h = mix(h, op_packed);
        h = mix(h, g.vs_copy_hash); // Not the span pointer!
        break;
    }
    case Shader::Stage::Fragment: {
        const auto& f = ri.fs_info;
        h = mix(h, XXH3_64bits(f.color_buffers.data(), sizeof(f.color_buffers)));
        // PERF(GR2FORK v1.18): en_flags and addr_flags are AmdGpu::PsInput,
        // which is a 4-byte u32 bitfield register. XXH3_64bits has ~10-15
        // cycles of plain function-call + setup overhead before doing any
        // work, which is dwarfing the actual hashing for a 4-byte input.
        // Read both as u32 (they're trivially memcpyable bitfield types)
        // and pack into a single mix call. Saves 2 XXH3 function calls and
        // 1 mix iteration per Fragment hash.
        static_assert(sizeof(f.en_flags) == 4);
        static_assert(sizeof(f.addr_flags) == 4);
        u32 en32, addr32;
        std::memcpy(&en32, &f.en_flags, sizeof(en32));
        std::memcpy(&addr32, &f.addr_flags, sizeof(addr32));
        h = mix(h, (static_cast<u64>(en32) << 32) | static_cast<u64>(addr32));
        // PERF(GR2FORK v1.18): pack four small scalars into a single mix
        // call instead of three separate ones. num_inputs <= 32 (array bound),
        // z_export_format is a small enum, mrtz_mask is u8, dual_source_blending
        // is bool — all comfortably fit into a single u64 with room to spare.
        // Cuts 2 mix iterations off the chain per Fragment hash.
        const u64 packed_scalars =
            static_cast<u64>(f.num_inputs) |
            (static_cast<u64>(static_cast<u32>(f.z_export_format)) << 8) |
            (static_cast<u64>(f.mrtz_mask) << 16) |
            (static_cast<u64>(f.dual_source_blending ? 1u : 0u) << 24);
        h = mix(h, packed_scalars);
        if (f.num_inputs > 0) {
            h = mix(h, XXH3_64bits(f.inputs.data(),
                                    f.num_inputs * sizeof(f.inputs[0])));
        }
        break;
    }
    case Shader::Stage::Compute: {
        const auto& c = ri.cs_info;
        // PERF(GR2FORK v1.18): workgroup_size is std::array<u32,3> = 12 bytes.
        // XXH3 function-call overhead dominates for this size. Pack all three
        // axes (each well under 2^20 by Vulkan/AMD hardware limits — typical
        // values 8..1024) plus the three tgid_enable bools into a single u64.
        // Replaces 1 XXH3 + 2 mix calls with 1 mix — saves ~25 cycles per
        // Compute hash (function call elision + 1 mix iteration off chain).
        // Bit layout:
        //   bits  0-19: workgroup_size[0]   (20 bits, max 2^20 = 1,048,576)
        //   bits 20-39: workgroup_size[1]   (20 bits)
        //   bits 40-59: workgroup_size[2]   (20 bits)
        //   bit  60   : tgid_enable[0]
        //   bit  61   : tgid_enable[1]
        //   bit  62   : tgid_enable[2]
        //   bit  63   : free
        const auto& wg = c.workgroup_size;
        const u64 packed =
            (static_cast<u64>(wg[0]) & 0xFFFFFu) |
            ((static_cast<u64>(wg[1]) & 0xFFFFFu) << 20) |
            ((static_cast<u64>(wg[2]) & 0xFFFFFu) << 40) |
            (static_cast<u64>(c.tgid_enable[0] ? 1u : 0u) << 60) |
            (static_cast<u64>(c.tgid_enable[1] ? 1u : 0u) << 61) |
            (static_cast<u64>(c.tgid_enable[2] ? 1u : 0u) << 62);
        h = mix(h, packed);
        break;
    }
    default:
        break;
    }
    return h;
}

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(Stage stage, LogicalStage l_stage,
                                                           const AmdGpu::LiverpoolRegsSnapshot& regs) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto BuildCommon = [&](const auto& program) {
        info.num_user_data = program.settings.num_user_regs;
        info.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.num_allocated_vgprs = program.NumVgprs();
        info.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.fp_round_mode32 = program.settings.fp_round_mode32;
    };
    info.Initialize(stage);
    switch (stage) {
    case Stage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants;
        const auto* hull_info = infos[u32(Shader::LogicalStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.ls_info.ls_stride = tess_constants.ls_stride;
        break;
    }
    case Stage::Hull: {
        BuildCommon(regs.hs_program);
        info.hs_info.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.hs_info.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.hs_info.tess_type = regs.tess_config.type;
        info.hs_info.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;

        // We need to initialize most hs_info fields after finding the V# with tess constants
        break;
    }
    case Stage::Export: {
        BuildCommon(regs.es_program);
        if (l_stage == LogicalStage::TessellationEval) {
            // Combined LS+HS+ES+GS pipeline: ES acts as domain shader.
            info.vs_info.num_outputs = regs.vgt_esgs_ring_itemsize;
            info.vs_info.tess_type = regs.tess_config.type;
            info.vs_info.tess_topology = regs.tess_config.topology;
            info.vs_info.tess_partitioning = regs.tess_config.partitioning;
        } else {
            info.es_info.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        }
        break;
    }
    case Stage::Vertex: {
        BuildCommon(regs.vs_program);
        info.vs_info.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.vs_info.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.vs_info.num_outputs = MapOutputs(info.vs_info.outputs, regs.vs_output_control);
        info.vs_info.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.vs_info.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            regs.primitive_type == AmdGpu::PrimitiveType::QuadList;
        info.vs_info.clip_disable = regs.IsClipDisabled();
        if (l_stage == LogicalStage::TessellationEval) {
            info.vs_info.tess_type = regs.tess_config.type;
            info.vs_info.tess_topology = regs.tess_config.topology;
            info.vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Geometry: {
        BuildCommon(regs.gs_program);
        auto& gs_info = info.gs_info;
        gs_info.num_outputs = MapOutputs(gs_info.outputs, regs.vs_output_control);
        gs_info.output_vertices = regs.vgt_gs_max_vert_out;
        gs_info.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        gs_info.in_primitive = regs.primitive_type;
        // In combined tess+GS pipelines, primitive_type is PatchPrimitive which isn't
        // meaningful for GS input. Resolve to the actual post-tessellation output type.
        if (gs_info.in_primitive == AmdGpu::PrimitiveType::PatchPrimitive) {
            switch (regs.tess_config.type) {
            case AmdGpu::TessellationType::Isoline:
                gs_info.in_primitive = AmdGpu::PrimitiveType::LineList;
                break;
            case AmdGpu::TessellationType::Triangle:
            case AmdGpu::TessellationType::Quad:
                gs_info.in_primitive = AmdGpu::PrimitiveType::TriangleList;
                break;
            }
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            gs_info.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        gs_info.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        gs_info.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        gs_info.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        gs_info.vs_copy = params_vc.code;
        gs_info.vs_copy_hash = params_vc.hash;
        DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, 0, "copy.bin");
        break;
    }
    case Stage::Fragment: {
        BuildCommon(regs.ps_program);
        info.fs_info.en_flags = regs.ps_input_ena;
        info.fs_info.addr_flags = regs.ps_input_addr;
        info.fs_info.num_inputs = regs.num_interp;
        info.fs_info.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.fs_info.mrtz_mask = regs.depth_shader_control.z_export_enable |
                                 (stencil_ref_export_enable << 1) |
                                 (regs.depth_shader_control.mask_export_enable << 2) |
                                 (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.fs_info.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.fs_info.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.fs_info.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.fs_info.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.fs_info.color_buffers[i] = graphics_key.color_buffers[i];
        }
        break;
    }
    case Stage::Compute: {
        const auto& cs_pgm = regs.cs_program;
        info.num_user_data = cs_pgm.settings.num_user_regs;
        info.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.cs_info.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                       cs_pgm.num_thread_z.full};
        info.cs_info.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                    cs_pgm.IsTgidEnabled(2)};
        info.cs_info.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes} {
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        // When binding a UBO, we calculate its size considering the offset in the larger buffer
        // cache underlying resource. In some cases, it may produce sizes exceeding the system
        // maximum allowed UBO range, so we need to reduce the threshold to prevent issues.
        .max_ubo_size = instance.UniformMaxSize() - instance.UniformMinAlignment(),
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_legacy_vertex_attributes = instance_.IsLegacyVertexAttributesSupported(),
        // PORT(upstream #4075, IMAGE_STORE_MIP fallback, commit — merged Mar 17 2026):
        // upstream hardcodes this to false with a // TEST marker. The fallback path
        // is preferred on both AMD and NVIDIA for GR2 per compat issue #1429
        // (mipmap-only-level-0 bug affects both vendors). Small perf cost on AMD
        // where native load-store-lod works — each IMAGE_STORE_MIP image uses
        // N descriptor slots instead of 1.
        .supports_image_load_store_lod = /*instance_.IsImageLoadStoreLodSupported()*/ false, // TEST
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        // TODO: Emitted bounds checks cause problems with phi control flow; needs to be fixed.
        .supports_robust_buffer_access = true, // instance_.IsRobustBufferAccess2Supported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .has_incomplete_fragment_shader_barycentric =
            instance_.IsFragmentShaderBarycentricSupported() &&
            instance.GetDriverID() == vk::DriverId::eMoltenvk,
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMoltenvk,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMoltenvk,
    };

    // NOTE: WarmUp() used to be invoked from here. It was moved out to
    // Presenter::WarmUpPipelineCache(), called from gnmdriver::RegisterLib
    // *after* the Presenter is fully constructed, so the swapchain and
    // schedulers are available and we can drive a "LOADING SHADERS"
    // overlay on the screen instead of presenting a black window for the
    // 30+ seconds a large pipeline cache can take to deserialize.
    //
    // The vk::PipelineCache below is created here so it is available when
    // WarmUp runs externally — LoadGraphicsPipeline / LoadComputePipeline
    // dereference *pipeline_cache to construct GraphicsPipeline /
    // ComputePipeline objects.

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
}

PipelineCache::~PipelineCache() {
    // OPT(GR2 v78): Dump any still-in-flight async compiles to the graveyard.
    // The map's own destructor would destruct each std::future, which joins
    // its worker thread — and if any of those threads are hung inside the
    // Vulkan driver, the emulator would freeze on its way out. Move futures
    // out first so map destruction sees already-empty PendingGraphicsPipeline
    // objects.
    for (auto& [key, pending] : pending_graphics_pipelines) {
        if (pending && pending->future.valid()) {
            Graveyard().Bury(std::move(pending->future));
        }
    }
    pending_graphics_pipelines.clear();
}


const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(
    const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // Phase 1D-pre-C: read stamp + dirty bit from the captured snapshot
    // rather than the live Liverpool. Under sync this is identical; under
    // async (Phase G) it eliminates the v1 hotfix1 race where a relaxed
    // load of `liverpool->gfx_pipeline_stamp` plus a plain bool read of
    // `liverpool->gfx_key_dirty_` could disagree with the regs we
    // captured for THIS intent.
    const u64 stamp = regs.gfx_pipeline_stamp;
    // PERF(GR2FORK v1.15): The gfx-stamp early-out is the dominant exit —
    // consecutive draws share a pipeline and the stamp only bumps on register
    // writes that affect pipeline state. The metrics + the FilterDraw same-
    // pipeline cache directly track this rate. Hint it.
    if (stamp == last_gfx_stamp && last_gfx_pipeline) [[likely]] {
        return last_gfx_pipeline;
    }
    // Level 2: stamp bumped but only dynamic-state regs changed (viewport, scissor, etc.)
    // Skip the entire RefreshGraphicsKey rebuild — pipeline key cannot have changed.
    // PERF(GR2FORK v1.23): once we've passed the v1.15 stamp-equal early
    // exit, the next-most-common case is dynamic-state-only changes
    // (viewport scrolls, scissor moves, blend constants) — those bump the
    // stamp but don't dirty the key. Hint as the dominant exit among the
    // post-stamp-mismatch cases.
    if (!regs.gfx_key_dirty && last_gfx_pipeline) [[likely]] {
        last_gfx_stamp = stamp;
        return last_gfx_pipeline;
    }
    // Phase 1D-pre-C: the prior `liverpool->ClearGfxKeyDirty()` call here
    // is gone — PM4 cleared the live `gfx_key_dirty_` flag inside
    // `Liverpool::CaptureSnapshot()` under sole-writer ownership. The
    // snapshot's `regs.gfx_key_dirty` is a frozen "was dirty AT capture"
    // view; consuming it doesn't need to clear anything.
    if (!RefreshGraphicsKey(regs)) [[unlikely]] {
        return nullptr;
    }
    // Key-level dedup: when only dynamic state changed (viewport, scissor, blend constants),
    // the stamp bumps but the pipeline key is byte-identical. Skip the hash + map lookup.
    if (last_gfx_pipeline &&
        std::memcmp(&graphics_key, &prev_graphics_key_,
                    offsetof(GraphicsPipelineKey, cached_hash_)) == 0) {
        last_gfx_stamp = stamp;
        return last_gfx_pipeline;
    }

    // =========================================================================
    // OPT(GR2 v78): Pending-async check first.
    // =========================================================================
    // If a previous call for this key launched an async compile that didn't
    // finish within the sync budget, the future lives in pending_graphics_pipelines.
    // Poll non-blockingly; finalize into graphics_pipelines on ready.
    // PERF(GR2FORK v1.23): post-warmup pipelines are all already compiled
    // and live in graphics_pipelines, not pending — the find() returning
    // end() is the dominant case.
    if (auto pit = pending_graphics_pipelines.find(graphics_key);
        pit != pending_graphics_pipelines.end()) [[unlikely]] {
        if (TryFinalizePending(*pit->second, graphics_key)) {
            // Result moved into graphics_pipelines[graphics_key]. Erase pending.
            pending_graphics_pipelines.erase(pit);
            // Fall through to the main-path update below.
        } else {
            // Still compiling (or permafailed). Skip this draw.
            return nullptr;
        }
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    // PERF(GR2FORK v1.23): post-warmup is_new is the rare case — most
    // distinct pipelines have already been compiled during early-game.
    if (is_new) [[unlikely]] {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        auto pending = LaunchAsyncPipelineCompile(graphics_key, pipeline_hash);

        // Synchronous fast-path wait. Most compiles complete in <50ms; waiting
        // kInitialSyncBudget catches them without triggering frame-skip.
        if (pending->future.wait_for(kInitialSyncBudget) == std::future_status::ready) {
            std::unique_ptr<GraphicsPipeline> pipeline;
            try {
                pipeline = pending->future.get();
            } catch (const std::exception& e) {
                LOG_ERROR(Render_Vulkan, "Async pipeline compile threw: {}", e.what());
            }
            if (!pipeline) {
                // Compile failed. Drop the empty map slot so we can retry next tick.
                graphics_pipelines.erase(it);
                return nullptr;
            }
            // Move result into the cache slot.
            it.value() = std::move(pipeline);
            // Finalize side effects — these need post-ctor state (sdata, modules).
            RegisterPipelineData(graphics_key, pipeline_hash, pending->sdata);
            ++num_new_pipelines;
            if (Config::collectShadersForDebug()) {
                for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                    if (pending->infos_copy[stage]) {
                        auto& m = pending->modules_copy[stage];
                        module_related_pipelines[m].emplace_back(graphics_key);
                    }
                }
            }
            fetch_shader.reset();
        } else {
            // Slow path: compile hasn't finished. Stash the pending entry, leave
            // graphics_pipelines[key] as null (indicates "in-flight"), return
            // null so the Rasterizer skips this draw. Subsequent draws for the
            // same key hit the TryFinalizePending branch above.
            pending_graphics_pipelines.emplace(graphics_key, std::move(pending));
            fetch_shader.reset();
            return nullptr;
        }
    } else if (!it->second) {
        // Defensive: is_new=false but slot is null. This shouldn't happen if
        // invariants hold (TryFinalizePending always writes non-null on success,
        // and we already checked pending above). Treat as "still compiling."
        return nullptr;
    }

    last_gfx_stamp = stamp;
    last_gfx_pipeline = it->second.get();
    std::memcpy(&prev_graphics_key_, &graphics_key, sizeof(GraphicsPipelineKey));
    return last_gfx_pipeline;
}

bool PipelineCache::TryFinalizePending(PendingGraphicsPipeline& pending,
                                       const GraphicsPipelineKey& key) {
    if (pending.permafailed) {
        return false;
    }
    if (pending.future.wait_for(std::chrono::milliseconds{0}) !=
        std::future_status::ready) {
        // Still compiling. Check thresholds for escalation.
        const auto elapsed = std::chrono::steady_clock::now() - pending.started_at;
        if (elapsed >= kPermaFailThreshold) {
            LOG_CRITICAL(Render_Vulkan,
                         "Pipeline {:#x} stuck >{}s — permafailed. Moving to graveyard; "
                         "this pipeline's draws will be skipped for the rest of the session. "
                         "This is almost certainly a Vulkan driver hang "
                         "(Mesa/RADV). Try updating your GPU driver.",
                         pending.pipeline_hash,
                         std::chrono::duration_cast<std::chrono::seconds>(kPermaFailThreshold)
                             .count());
            Graveyard().Bury(std::move(pending.future));
            pending.permafailed = true;
        } else if (elapsed >= kHangLogThreshold && !pending.hang_warned) {
            LOG_WARNING(Render_Vulkan,
                        "Pipeline {:#x} compile exceeded {}s — likely driver hang. "
                        "Draws using this pipeline are being skipped. Will permafail at {}s.",
                        pending.pipeline_hash,
                        std::chrono::duration_cast<std::chrono::seconds>(kHangLogThreshold)
                            .count(),
                        std::chrono::duration_cast<std::chrono::seconds>(kPermaFailThreshold)
                            .count());
            pending.hang_warned = true;
        }
        return false;
    }
    // Ready. Collect the result.
    std::unique_ptr<GraphicsPipeline> pipeline;
    try {
        pipeline = pending.future.get();
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "Async pipeline {:#x} threw: {}", pending.pipeline_hash,
                  e.what());
    }
    if (!pipeline) {
        // Compile failed. Don't leave a permafail marker — let the outer code
        // retry next tick by erasing the pending entry (caller does this).
        return false;
    }
    // Install into the main map (create slot if missing; it usually exists as null).
    auto [it, is_new] = graphics_pipelines.try_emplace(key);
    it.value() = std::move(pipeline);
    // Finalize side effects.
    RegisterPipelineData(key, pending.pipeline_hash, pending.sdata);
    ++num_new_pipelines;
    if (Config::collectShadersForDebug()) {
        for (auto stage = 0; stage < MaxShaderStages; ++stage) {
            if (pending.infos_copy[stage]) {
                auto& m = pending.modules_copy[stage];
                module_related_pipelines[m].emplace_back(key);
            }
        }
    }
    LOG_INFO(Render_Vulkan, "Pipeline {:#x} compile finished after {} ms",
             pending.pipeline_hash,
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - pending.started_at)
                 .count());
    return true;
}

std::unique_ptr<PipelineCache::PendingGraphicsPipeline>
PipelineCache::LaunchAsyncPipelineCompile(const GraphicsPipelineKey& key, u64 pipeline_hash) {
    auto pending = std::make_unique<PendingGraphicsPipeline>();
    pending->started_at = std::chrono::steady_clock::now();
    pending->pipeline_hash = pipeline_hash;

    // Deep-copy stage data. The PipelineCache::infos/runtime_infos/modules
    // members are span-targets and get overwritten by the next RefreshGraphicsStages.
    // The async task must not alias them.
    pending->infos_copy = infos;
    pending->runtime_infos_copy = runtime_infos;
    pending->modules_copy = modules;
    pending->fetch_shader_copy = fetch_shader;

    // Capture by raw pointer into the pending entry. The pending entry lives
    // in the map owned by PipelineCache until finalize/permafail, so pointers
    // into it are stable for the lifetime of the worker's execution (worker
    // result is harvested before erase in TryFinalizePending, or moved to the
    // graveyard where it's never touched again).
    PendingGraphicsPipeline* raw = pending.get();

    // Snapshot fields the ctor needs but that reference PipelineCache members.
    const Instance* instance_ptr = &instance;
    Scheduler* scheduler_ptr = &scheduler;
    DescriptorHeap* desc_heap_ptr = &desc_heap;
    const Shader::Profile* profile_ptr = &profile;
    vk::PipelineCache cache_handle = *pipeline_cache;
    GraphicsPipelineKey key_copy = key;

    pending->future = std::async(
        std::launch::async,
        [raw, instance_ptr, scheduler_ptr, desc_heap_ptr, profile_ptr, cache_handle,
         key_copy]() -> std::unique_ptr<GraphicsPipeline> {
            // Spans over the pending-owned copies — stable for the async task's lifetime.
            std::span<const Shader::Info*, MaxShaderStages> infos_span{raw->infos_copy};
            std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_span{
                raw->runtime_infos_copy};
            std::span<const vk::ShaderModule> modules_span{raw->modules_copy};
            // Note: GraphicsPipeline ctor ASSERTs on driver-return failure, which
            // kills the process. We can't soften that; it's only the hang case
            // (no return at all) that this whole mechanism addresses.
            return std::make_unique<GraphicsPipeline>(
                *instance_ptr, *scheduler_ptr, *desc_heap_ptr, *profile_ptr, key_copy,
                cache_handle, infos_span, runtime_span, raw->fetch_shader_copy, modules_span,
                raw->sdata, false);
        });
    return pending;
}

const ComputePipeline* PipelineCache::GetComputePipeline(
    const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // PERF(GR2FORK v1.26): RefreshComputeKey returns false only on
    // unrecognised / invalid compute state — error path. Steady-state
    // dispatches refresh successfully.
    if (!RefreshComputeKey(regs)) [[unlikely]] {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    // PERF(GR2FORK v1.26): post-warmup is_new is rare — every distinct
    // compute pipeline has been compiled once during early game.
    if (is_new) [[unlikely]] {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        // PERF(GR2FORK v1.26): debug-only config for the shader-collector
        // workflow — disabled in release.
        if (Config::collectShadersForDebug()) [[unlikely]] {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey(const AmdGpu::LiverpoolRegsSnapshot& regs) {
    auto& key = graphics_key;
    // PERF(GR2FORK B1, post-v1.57): selective field-range zeroing replaces
    // the whole-struct memset. The original cleared ~390 bytes; this clears
    // only the conditionally-written ranges plus the anonymous-bitfield
    // padding. The skipped fields (patch_control_points, num_color_attachments,
    // cb_shader_mask, logic_op, num_samples, depth_samples, mrt_mask, and
    // the bitfield member bits themselves) are unconditionally rewritten by
    // this function or by RefreshGraphicsStages, so pre-zeroing them is
    // wasted work. cached_hash_/hash_valid_ are excluded from the comparison
    // hash range and don't need pre-zeroing — hash_valid_ is reset below.
    //
    // Risk: the v1.57 rollback established that some "dead" stores act as
    // load-bearing cache prefetch for nearby hot code. The conditional
    // arrays still get zeroed here so their cache lines stay warm; only the
    // scalar lines lose their prefetch. FPS-harness validation (Phase 0) is
    // mandatory before promoting this change past initial A/B.
    //
    // The static_asserts encode the field-order assumption — a future struct
    // reorder will fail the build instead of silently corrupting keys.
    static_assert(offsetof(GraphicsPipelineKey, stage_hashes) <
                      offsetof(GraphicsPipelineKey, vertex_buffer_formats) &&
                  offsetof(GraphicsPipelineKey, vertex_buffer_formats) <
                      offsetof(GraphicsPipelineKey, patch_control_points),
                  "stage_hashes/vertex_buffer_formats must precede patch_control_points");
    static_assert(offsetof(GraphicsPipelineKey, num_color_attachments) <
                      offsetof(GraphicsPipelineKey, color_buffers) &&
                  offsetof(GraphicsPipelineKey, color_buffers) <
                      offsetof(GraphicsPipelineKey, blend_controls) &&
                  offsetof(GraphicsPipelineKey, blend_controls) <
                      offsetof(GraphicsPipelineKey, write_masks) &&
                  offsetof(GraphicsPipelineKey, write_masks) <
                      offsetof(GraphicsPipelineKey, cb_shader_mask),
                  "color_buffers/blend_controls/write_masks must be contiguous");
    static_assert(offsetof(GraphicsPipelineKey, depth_samples) <
                      offsetof(GraphicsPipelineKey, color_samples) &&
                  offsetof(GraphicsPipelineKey, color_samples) <
                      offsetof(GraphicsPipelineKey, mrt_mask) &&
                  offsetof(GraphicsPipelineKey, mrt_mask) <
                      offsetof(GraphicsPipelineKey, cached_hash_),
                  "color_samples/mrt_mask/bitfields must precede cached_hash_");

    auto* const key_bytes = reinterpret_cast<char*>(&key);
    // Range A: stage_hashes + vertex_buffer_formats (conditional fills, unused
    // slots must be 0 for memcmp determinism).
    std::memset(key_bytes, 0, offsetof(GraphicsPipelineKey, patch_control_points));
    // Range B: color_buffers + blend_controls + write_masks (conditional
    // fills; PsColorBuffer/BlendControl are bit-field structs whose padding
    // bits would not be cleared by member-by-member assignment).
    std::memset(key_bytes + offsetof(GraphicsPipelineKey, color_buffers), 0,
                offsetof(GraphicsPipelineKey, cb_shader_mask) -
                    offsetof(GraphicsPipelineKey, color_buffers));
    // Range C: color_samples + mrt_mask + both anonymous bitfield words.
    // mrt_mask is unconditionally rewritten so zeroing it is wasted but
    // harmless; including it keeps the range contiguous and clears the
    // bitfield padding bits in the same store stream.
    std::memset(key_bytes + offsetof(GraphicsPipelineKey, color_samples), 0,
                offsetof(GraphicsPipelineKey, cached_hash_) -
                    offsetof(GraphicsPipelineKey, color_samples));
    // hash_valid_ is excluded from the comparison range but must be reset
    // so GetHash() rebuilds the cached hash on the next call.
    key.hash_valid_ = false;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages(regs)) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages(const AmdGpu::LiverpoolRegsSnapshot& regs) {
    auto& key = graphics_key;
    fetch_shader = std::nullopt;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in, Shader::LogicalStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        // PERF(GR2FORK v1.28): bind_stage is called only for stages the
        // surrounding switch already knows are active for the current
        // pipeline configuration (Fragment unconditionally, plus the
        // VgtStages-specific subset). The IsStageEnabled re-check is
        // defensive and almost always passes.
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) [[unlikely]] {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        // PERF(GR2FORK v1.28): null program / null pgm address is a
        // defensive guard against malformed PM4; well-formed shader
        // emission has both populated.
        if (!pgm || !pgm->Address<u32*>()) [[unlikely]] {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        // PERF(GR2FORK v1.60): GetProgram now returns a const-pointer to
        // the program-owned optional<FetchShaderData> instead of a copy.
        // Eliminates one heap-allocating copy through std::tie. The
        // single deep copy below (only when the pointer is non-null and
        // the optional has a value) writes into PipelineCache's own
        // member, which is the only persistent storage that needs it.
        const std::optional<Shader::Gcn::FetchShaderData>* fetch_shader_ptr = nullptr;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_ptr,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding, regs);
        if (fetch_shader_ptr && fetch_shader_ptr->has_value()) {
            fetch_shader = *fetch_shader_ptr;
        }
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);
    bind_stage(Stage::Fragment, LogicalStage::Fragment);

    const auto* fs_info = infos[static_cast<u32>(LogicalStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        // PERF(GR2FORK v1.28): RADV/Mesa with VK_EXT_extended_dynamic_state3
        // exposes geometry shader support — the unsupported warning path
        // is for outlier configurations.
        if (!instance.IsGeometryStageSupported()) [[unlikely]] {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        // PERF(GR2FORK v1.28): stream output is unimplemented in shadPS4;
        // games that use it hit this warning path. The vast majority of
        // GS-using shaders don't enable stream output.
        if (regs.vgt_strmout_config.raw) [[unlikely]] {
            LOG_WARNING(Render_Vulkan, "Stream output unsupported, skipping");
            return false;
        }
        // PERF(GR2FORK v1.28): on the EsGs path both ES and GS stages
        // are required and stage_enable already gates entry — bind_stage
        // failures here are defensive recovery paths.
        if (!bind_stage(Stage::Export, LogicalStage::Vertex)) [[unlikely]] {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) [[unlikely]] {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported() ||
            (regs.tess_config.type == AmdGpu::TessellationType::Isoline &&
             !instance.IsTessellationIsolinesSupported())) [[unlikely]] {
            return false;
        }
        // PERF(GR2FORK v1.28): bind_stage failures on the active LsHs path
        // are defensive recovery — well-formed PM4 has all three stages
        // populated.
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) [[unlikely]] {
            return false;
        }
        if (!bind_stage(Stage::Vertex, LogicalStage::TessellationEval)) [[unlikely]] {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) [[unlikely]] {
            return false;
        }
        break;
    default:
        // PERF(GR2FORK v1.28): the combined LS+HS+ES+GS pipeline (foliage
        // with tessellation + geometry) is rare in shadPS4 workloads —
        // the dominant default path is plain Vertex+Pixel handled by the
        // simple `else` branch.
        if (regs.stage_enable.hs_en && regs.stage_enable.gs_en) [[unlikely]] {
            // Combined LS+HS+ES+GS pipeline (e.g. foliage with tessellation + geometry).
            if (!instance.IsTessellationSupported() || !instance.IsGeometryStageSupported()) {
                LOG_WARNING(Render_Vulkan,
                            "Combined tessellation+geometry pipeline unsupported, skipping");
                return false;
            }
            if (regs.tess_config.type == AmdGpu::TessellationType::Isoline &&
                !instance.IsTessellationIsolinesSupported()) {
                return false;
            }
            if (regs.vgt_strmout_config.raw) {
                LOG_WARNING(Render_Vulkan, "Stream output unsupported, skipping");
                return false;
            }
            if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
                return false;
            }
            if (!bind_stage(Stage::Export, LogicalStage::TessellationEval)) {
                return false;
            }
            if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
                return false;
            }
            if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
                return false;
            }
        } else {
            bind_stage(Stage::Vertex, LogicalStage::Vertex);
        }
        break;
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT(vertex_binding < MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey(const AmdGpu::LiverpoolRegsSnapshot& regs) {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = regs.cs_program;
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    // PERF(GR2FORK v1.60): pointer return — see comment in
    // RefreshGraphicsStages bind_stage above. Compute stages don't have a
    // fetch shader (the returned pointer points to a nullopt or is null
    // itself), so the assign below typically resets fetch_shader.
    const std::optional<Shader::Gcn::FetchShaderData>* fetch_shader_ptr = nullptr;
    std::tie(infos[0], modules[0], fetch_shader_ptr, compute_key.value) =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding, regs);
    if (fetch_shader_ptr && fetch_shader_ptr->has_value()) {
        fetch_shader = *fetch_shader_ptr;
    } else {
        fetch_shader.reset();
    }
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");

    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    DumpShader(spv, info.pgm_hash, info.stage, perm_idx, "spv");

    // GR2FORK Fix A v7: SPIR-V-level patcher for the grass leader-write idiom.
    //
    // GR2's grass compute shaders (cs_0xc282b105 = gating CS, cs_0x3f0a3c48 =
    // per-patch gen) end with a post-barrier "leader-write" pattern, where
    // GCN does S_NOT_B64 s[106:107], s[0:1] then S_AND_SAVEEXEC_B64
    // s[106:107], s[106:107]. shadPS4's IR translation produces
    // OpLogicalAnd(GetExec, OpLogicalNot(GetThreadBitScalarReg(s106))), but
    // both operands end up resolving to the SAME OpUndef bool (FlagTag SSA
    // variables that were never written before the leader-write block).
    // The resulting SPIR-V structure is OpLogicalAnd(%U, OpLogicalNot(%U)) -
    // which is provably ALWAYS FALSE regardless of what %U is, so the
    // leader-write block becomes unreachable and grass vertexCount stays 0.
    //
    // The fix here cannot change what %U resolves to (substitution doesn't
    // help because both operands are the same %U). Instead, it REDIRECTS
    // the consuming OpBranchConditional to use a different predicate that
    // already exists in the shader: the entry-block's `lane == 0` check
    // (OpLogicalNot(OpINotEqual(0, gl_LocalInvocationID.x))). This makes
    // lane 0 write the result (matching the GCN's intent), and no other
    // lanes race against it.
    //
    // History: an earlier attempt (Fix A v5/v6) tried to fix this at the
    // SSA-rewrite level by substituting (scalar >> LaneId) & 1 for the
    // UndefU1. That correctly removed the OpUndef bool from the output,
    // but the LogicalAnd structure remained, and `chain AND NOT chain`
    // is still always false. Reverted to this SPIR-V-level approach.
    {
        const u64 h = info.pgm_hash;
        const bool is_grass_cs = (h == 0xc282b105ULL) ||
                                 (h == 0x3f0a3c48ULL) ||
                                 (h == 0x83048c53ULL);
        if (is_grass_cs) [[unlikely]] {
            const auto try_patch = [&](std::vector<u32>& code) -> bool {
                if (code.size() < 5) return false;
                // SPIR-V opcodes we care about
                constexpr u32 kOpUndef = 1;
                constexpr u32 kOpTypeBool = 20;
                constexpr u32 kOpCompositeExtract = 81;
                constexpr u32 kOpINotEqual = 171;
                constexpr u32 kOpLogicalNot = 168;
                constexpr u32 kOpLogicalAnd = 167;
                constexpr u32 kOpBranchConditional = 250;

                u32 bool_type_id = 0;
                u32 undef_bool_id = 0;
                u32 last_composite_extract_with_zero_idx = 0;
                u32 lane_inotequal_id = 0;  // INotEqual involving the extract
                u32 lane_eq_zero_id = 0;    // LogicalNot of the INotEqual
                u32 lognot_undef_id = 0;
                u32 logand_buggy_id = 0;
                size_t branch_cond_word_pos = 0;

                size_t i = 5;  // skip header
                while (i < code.size()) {
                    const u32 word = code[i];
                    const u32 word_count = word >> 16;
                    const u32 opcode = word & 0xFFFF;
                    if (word_count == 0 || i + word_count > code.size()) break;

                    switch (opcode) {
                    case kOpTypeBool:
                        if (word_count >= 2 && bool_type_id == 0) {
                            bool_type_id = code[i + 1];
                        }
                        break;
                    case kOpUndef:
                        // word[1]=result_type, word[2]=result_id
                        if (word_count >= 3 && undef_bool_id == 0 &&
                            code[i + 1] == bool_type_id) {
                            undef_bool_id = code[i + 2];
                        }
                        break;
                    case kOpCompositeExtract:
                        // word[1]=result_type, word[2]=result_id, word[3]=composite,
                        // word[4..]=indices. We want the first one extracting index 0.
                        // This will be the gl_LocalInvocationID.x extraction.
                        if (word_count >= 5 && code[i + 4] == 0 &&
                            last_composite_extract_with_zero_idx == 0) {
                            last_composite_extract_with_zero_idx = code[i + 2];
                        }
                        break;
                    case kOpINotEqual:
                        // word[1]=result_type, word[2]=result_id, word[3]=op1, word[4]=op2.
                        // Match any INotEqual where one operand is our extract result -
                        // SPIR-V type rules guarantee the other is the corresponding
                        // u32_zero constant we don't need to identify separately.
                        if (word_count >= 5 && lane_inotequal_id == 0 &&
                            last_composite_extract_with_zero_idx != 0 &&
                            (code[i + 3] == last_composite_extract_with_zero_idx ||
                             code[i + 4] == last_composite_extract_with_zero_idx)) {
                            lane_inotequal_id = code[i + 2];
                        }
                        break;
                    case kOpLogicalNot:
                        // word[1]=result_type, word[2]=result_id, word[3]=operand
                        if (word_count >= 4) {
                            if (lane_eq_zero_id == 0 && lane_inotequal_id != 0 &&
                                code[i + 3] == lane_inotequal_id) {
                                lane_eq_zero_id = code[i + 2];
                            }
                            if (lognot_undef_id == 0 && undef_bool_id != 0 &&
                                code[i + 3] == undef_bool_id) {
                                lognot_undef_id = code[i + 2];
                            }
                        }
                        break;
                    case kOpLogicalAnd:
                        if (word_count >= 5 && logand_buggy_id == 0 &&
                            undef_bool_id != 0 && lognot_undef_id != 0) {
                            const bool buggy_a = (code[i + 3] == undef_bool_id &&
                                                  code[i + 4] == lognot_undef_id);
                            const bool buggy_b = (code[i + 4] == undef_bool_id &&
                                                  code[i + 3] == lognot_undef_id);
                            if (buggy_a || buggy_b) {
                                logand_buggy_id = code[i + 2];
                            }
                        }
                        break;
                    case kOpBranchConditional:
                        // word[1]=cond, word[2]=true_label, word[3]=false_label
                        if (word_count >= 4 && logand_buggy_id != 0 &&
                            code[i + 1] == logand_buggy_id &&
                            branch_cond_word_pos == 0) {
                            branch_cond_word_pos = i + 1;
                        }
                        break;
                    default:
                        break;
                    }
                    i += word_count;
                }

                if (lane_eq_zero_id == 0 || branch_cond_word_pos == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "[GR2FORK Fix A v7] cs={:#x}: pattern not found "
                                "(bool_type={} undef_bool={} extract={} "
                                "inotequal={} lane_eq_zero={} lognot_undef={} "
                                "logand_buggy={} branch_pos={})",
                                h, bool_type_id, undef_bool_id,
                                last_composite_extract_with_zero_idx,
                                lane_inotequal_id, lane_eq_zero_id,
                                lognot_undef_id, logand_buggy_id,
                                branch_cond_word_pos);
                    return false;
                }
                code[branch_cond_word_pos] = lane_eq_zero_id;
                LOG_INFO(Render_Vulkan,
                         "[GR2FORK Fix A v7] cs={:#x}: patched OpBranchConditional "
                         "cond from %{} (LogicalAnd(undef, NOT undef)) to %{} "
                         "(lane == 0)",
                         h, logand_buggy_id, lane_eq_zero_id);
                return true;
            };
            try_patch(spv);
        }
    }

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.stage, perm_idx, "spv");
    const bool is_patched = patch && Config::patchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (Config::collectShadersForDebug()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding,
                                                const AmdGpu::LiverpoolRegsSnapshot& regs) {
    Shader::RuntimeInfo runtime_info = BuildRuntimeInfo(stage, l_stage, regs);
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    // PERF(GR2FORK v1.25): after early-game warmup every distinct shader
    // hash has been compiled once and lives in program_cache. New-program
    // insertion is the rare path.
    if (new_program) [[unlikely]] {
        it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
        auto& program = it_pgm.value();
        auto start = binding;
        const auto module = CompileModule(program->info, runtime_info, params.code, 0, binding);
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, start);
        const auto perm_hash = HashCombine(params.hash, 0);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        // PERF(GR2FORK v1.60): return pointer into program-owned spec
        // storage instead of copying the optional<FetchShaderData>. The
        // FetchShaderData wraps a std::vector<VertexAttribute>; the
        // previous by-value return forced a heap-alloc + memcpy here AND
        // a second copy in std::tie's assignment at the call site. With
        // the new pointer-typed Result, RefreshGraphicsStages performs
        // exactly one deep copy at the call site (into PipelineCache's
        // own member). Lifetime: program is owned by program_cache (a
        // tsl::robin_map<size_t, std::unique_ptr<Program>>); the unique_ptr
        // keeps Program at a stable heap address, so &spec.fetch_shader_data
        // remains valid for the duration of the GetProgram call's return
        // path (the caller copies before any subsequent map mutation).
        return std::make_tuple(&program->info, module,
                               &program->modules[0].spec.fetch_shader_data, perm_hash);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    info.RefreshFlatBuf();

    // PERF(GR2 v16): Fast-path to skip StageSpecialization construction.
    // Hash (user_data, runtime_info by stage, full binding) and compare to cached result.
    // When only context regs change (viewport, scissor, blend) but SH regs stay the same,
    // this avoids constructing StageSpecialization (~2.26% of GpuComm).
    //
    // ud_hash_lru is the 32-slot per-Program LRU below the last_result fast
    // path. Gated on Config::isPipelineUdHashLruEnabled() — on by default;
    // set pipelineUdHashLruEnable=false in [Vulkan] to disable if warmup
    // flicker returns. The line-1225 last_result fast-path is NOT gated
    // (its correctness invariant is separately maintained and it has not
    // produced visual issues).
    const bool ud_hash_lru_enabled = Config::isPipelineUdHashLruEnabled();
    u64 ud_hash;
    u64 ri_bind_hash;
    {
        ud_hash = XXH3_64bits(params.user_data.data(), params.user_data.size_bytes());
        // Mix in stage-aware runtime_info hash (handles union padding + custom operator==)
        ri_bind_hash = HashRuntimeInfoForStage(runtime_info);
        // PERF(GR2FORK v1.21): pack the three binding counters into a
        // single u64 and fold them into ri_bind_hash with one mix step
        // instead of two. Each Backend::Bindings field is u32 but bounded
        // by shadPS4 resource constants (NUM_BUFFERS=40 + NUM_IMAGES=64 +
        // NUM_SAMPLERS=16 + NUM_FMASKS=8 = 128 max for `unified` per stage,
        // 40 max for `buffer`, 16 max for `user_data` — even accumulated
        // across all 5 graphics stages they stay well under 16 bits each).
        // The 16-bit slot allocation has ~50× headroom over realistic
        // values; impossible to overflow. Saves one mix-chain iteration
        // (~5-6 cycles) per GetProgram call.
        const u64 binding_packed =
            static_cast<u64>(binding.unified) |
            (static_cast<u64>(binding.buffer) << 16) |
            (static_cast<u64>(binding.user_data) << 32);
        ri_bind_hash ^= binding_packed +
                    0x9e3779b97f4a7c15ULL + (ri_bind_hash << 6) + (ri_bind_hash >> 2);

        ud_hash ^= ri_bind_hash + 0x9e3779b97f4a7c15ULL + (ud_hash << 6) + (ud_hash >> 2);

        // PERF(GR2FORK v1.15): When user_data is stable across consecutive
        // calls (the dominant case in steady state — same shader bound, same
        // resource layout), this branch returns the cached module without
        // touching ud_hash_lru, sig lookup, or perm_index. Hint it.
        if (program->last_result.valid && program->last_result.ud_hash == ud_hash) [[likely]] {
            const auto perm_idx = program->last_result.perm_idx;
            if (perm_idx < program->modules.size()) {
                info.AddBindings(binding);
                // PERF(GR2FORK v1.60): pointer return — see comment at the
                // new-program return site above.
                return std::make_tuple(&program->info, program->last_result.module,
                                       &program->modules[perm_idx].spec.fetch_shader_data,
                                       program->last_result.perm_hash);
            }
        }

        // PERF(GR2FORK): ud_hash → perm_idx LRU. Skips StageSpecialization
        // construction (~0.69% of GpuComm in the trace) when this ud_hash
        // has been resolved for this program before. Distinct from the
        // banned v17 stable-shortcut — see Program::UdHashCacheEntry doc
        // comment in vk_pipeline_cache.h.
        if (ud_hash_lru_enabled) {
            const u32 slot = static_cast<u32>(ud_hash) &
                             (Program::kUdHashCacheSize - 1);
            const auto& e = program->ud_hash_lru[slot];
            // PERF(GR2FORK v1.25): once the v1.15 last_result fast path
            // misses, the ud_hash_lru is the next-best resolver — it
            // keeps recently-used permutations warm for shaders that
            // rotate between a small set (UI vs effect, multiple LOD
            // variants). After warmup the LRU hit is the dominant exit
            // among the post-last_result paths.
            if (e.valid && e.ud_hash == ud_hash &&
                e.perm_idx < program->modules.size()) [[likely]] {
                const size_t perm_idx = e.perm_idx;
                const auto& m = program->modules[perm_idx];
                const u64 perm_hash = HashCombine(params.hash, perm_idx);
                info.AddBindings(binding);
                // Promote into last_result so the next call hits the
                // cheaper line-978 fast-path instead of running this
                // lookup again.
                program->last_result.ud_hash = ud_hash;
                program->last_result.ri_bind_hash = ri_bind_hash;
                program->last_result.perm_idx = perm_idx;
                program->last_result.perm_hash = perm_hash;
                program->last_result.module = m.module;
                program->last_result.valid = true;
                // PERF(GR2FORK v1.60): pointer return — see comment at the
                // new-program return site above.
                return std::make_tuple(&program->info, m.module,
                                       &m.spec.fetch_shader_data, perm_hash);
            }
        }

        // FIX(GR2FORK): PERF(GR2 v17) "Stable single-permutation shortcut" removed.
        //
        // The removed shortcut assumed that if `ri_bind_hash` (hash of
        // runtime_info + binding offsets) matched the cached value for 64+
        // consecutive calls, the cached shader module remained valid even
        // when `user_data` (the SGPRs) differed — on the rationale that
        // "stride/format/etc. are extremely unlikely to change" once a
        // program has been stable.
        //
        // That assumption is wrong. `user_data` values ARE the guest
        // pointers (or inline encodings) to the image/buffer sharps that
        // StageSpecialization codegens against — see Info::ReadUdReg in
        // shader_recompiler/info.h, which dereferences user_data[i] to
        // reach sharp memory. Different user_data → different sharp
        // address → potentially different image type / dst_select /
        // num_conversion / srgb / storage / array-ness / fetch-shader
        // layout. A module specialized against sharp set A is NOT safe to
        // serve for sharp set B, even if both have the same slot layout.
        //
        // GR2 (CUSA03694) exposes this. The same fragment shader is used
        // both for comic/UI panels (RGBA8_SRGB 2D textures) and for
        // in-world effect draws (different image types / num_conversion).
        // Ditto vertex shaders used for UI quads and for high-velocity
        // character animation. After ~64 UI draws the shortcut locks the
        // cache to the UI spec; subsequent effect/fall draws then read
        // the stale module and produce green garbled effect textures and
        // vertex explosions. The shortcut also rewrote last_result.ud_hash
        // to the new user_data, so Fast-path 1 above would keep firing on
        // the stale module for the remainder of the 512-call revalidate
        // window — self-reinforcing.
        //
        // Fast-path 1 (ud_hash-keyed) above remains correct: ud_hash is
        // XXH3 of the user_data BYTES, so any SGPR change — which is what
        // changes when sharp pointers change — invalidates the hit. We
        // lose the v17 perf win for programs that receive genuinely new
        // user_data every frame but point to structurally identical
        // sharps; that case pays one StageSpecialization construct per
        // call, which is what the sig-based lookup below is optimized for.
        //
        // Store hashes for later cache update (after spec construction + lookup)
        program->last_result.ud_hash = ud_hash;
        program->last_result.ri_bind_hash = ri_bind_hash;
    }

    const std::optional<Shader::Gcn::FetchShaderData>* cached_fetch = nullptr;
    if (stage == Stage::Vertex && !program->modules.empty()) {
        cached_fetch = &program->modules.front().spec.fetch_shader_data;
    }
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding, cached_fetch);

    // Fast path: look up by specialization signature.
    // We use a *pair* of signatures (sig + sig2) so we can avoid expensive deep comparisons.
    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    bool found = false;
    if (const auto it_sig = program->perm_index_by_sig.find(spec.sig);
        it_sig != program->perm_index_by_sig.end() && it_sig->second < program->modules.size()) {
        const auto& ms = program->modules[it_sig->second].spec;
        // PERF(GR2FORK v1.25): perm_index_by_sig just returned a hit on
        // spec.sig, so the indexed module's stored sig matches except
        // under a 64-bit hash collision (~2^-64 per pair, negligible).
        // sig2 is computed from the same StageSpecialization fields and
        // confirms the match.
        if (ms.sig == spec.sig && ms.sig2 == spec.sig2) [[likely]] {
            info.AddBindings(binding);
            perm_idx = it_sig->second;
            perm_hash = HashCombine(params.hash, perm_idx);
            module = program->modules[perm_idx].module;
            found = true;
            // Update per-program result cache.
            program->last_result.perm_idx = perm_idx;
            program->last_result.perm_hash = perm_hash;
            program->last_result.module = module;
            program->last_result.valid = true;
            // PERF(GR2FORK): also populate ud_hash_lru so the next call
            // with this ud_hash skips the StageSpecialization ctor above.
            if (ud_hash_lru_enabled) {
                const u32 slot = static_cast<u32>(ud_hash) &
                                 (Program::kUdHashCacheSize - 1);
                program->ud_hash_lru[slot] = Program::UdHashCacheEntry{
                    .ud_hash = ud_hash,
                    .perm_idx = static_cast<u32>(perm_idx),
                    .valid = true,
                };
            }
        }
    }

    if (!found) {
        // Fallback: linear scan by (sig,sig2) without deep comparisons.
        size_t found_idx = std::numeric_limits<size_t>::max();
        for (size_t i = 0; i < program->modules.size(); ++i) {
            const auto& ms = program->modules[i].spec;
            if (ms.sig == spec.sig && ms.sig2 == spec.sig2) {
                found_idx = i;
                break;
            }
        }

        if (found_idx == std::numeric_limits<size_t>::max()) {
            auto new_info = Shader::Info(stage, l_stage, params);
            module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);

            RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
            program->AddPermut(module, std::move(spec));
            // FIX(GR2FORK): update last_result on the new-permutation path.
            //
            // Pre-existing latent bug: lines 1057-1058 above unconditionally
            // overwrote last_result.ud_hash with the new ud_hash before the
            // spec ctor ran. If last_result.valid was already true from a
            // prior call (perm_idx/module pointing to the prior result), and
            // the new permutation path was taken (this branch), last_result
            // ended up with NEW ud_hash but STALE perm_idx/module. The next
            // call with the same ud_hash would hit the line-978 fast-path
            // and return the stale module. Setting last_result fully here
            // makes it self-consistent — same shape as the sig-hit and
            // linear-scan-hit branches below.
            program->last_result.perm_idx = perm_idx;
            program->last_result.perm_hash = perm_hash;
            program->last_result.module = module;
            program->last_result.valid = true;
            // PERF(GR2FORK): populate ud_hash_lru so the second call with
            // this ud_hash skips the StageSpecialization ctor.
            if (ud_hash_lru_enabled) {
                const u32 slot = static_cast<u32>(ud_hash) &
                                 (Program::kUdHashCacheSize - 1);
                program->ud_hash_lru[slot] = Program::UdHashCacheEntry{
                    .ud_hash = ud_hash,
                    .perm_idx = static_cast<u32>(perm_idx),
                    .valid = true,
                };
            }
        } else {
            info.AddBindings(binding);
            module = program->modules[found_idx].module;
            perm_idx = found_idx;
            perm_hash = HashCombine(params.hash, perm_idx);
            // Keep the map warm for future lookups.
            program->perm_index_by_sig.try_emplace(spec.sig, perm_idx);
            // Update per-program result cache.
            program->last_result.perm_idx = perm_idx;
            program->last_result.perm_hash = perm_hash;
            program->last_result.module = module;
            program->last_result.valid = true;
            // PERF(GR2FORK): also populate ud_hash_lru.
            if (ud_hash_lru_enabled) {
                const u32 slot = static_cast<u32>(ud_hash) &
                                 (Program::kUdHashCacheSize - 1);
                program->ud_hash_lru[slot] = Program::UdHashCacheEntry{
                    .ud_hash = ud_hash,
                    .perm_idx = static_cast<u32>(perm_idx),
                    .valid = true,
                };
            }
        }
    }
    // PERF(GR2FORK v1.60): pointer return — see comment at the
    // new-program return site above.
    return std::make_tuple(&program->info, module,
                           &program->modules[perm_idx].spec.fetch_shader_data, perm_hash);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
                // FIX(GR2FORK): last_result caches a vk::ShaderModule HANDLE
                // by value (not a reference into program->modules). Replacing
                // the module above leaves last_result.module pointing at the
                // destroyed handle. Next GetProgram call hitting the
                // line-978 fast-path would return a use-after-destroy module.
                // Invalidate last_result for any program whose cached module
                // has been swapped.
                //
                // ud_hash_lru does NOT need invalidation: it stores perm_idx
                // and reads program->modules[perm_idx].module fresh on every
                // lookup, so the new module is picked up automatically.
                //
                // Note: this is the devtools live-shader-replace path
                // (core/devtools/widget/shader_list.cpp). It runs on the UI
                // thread, not GpuComm. The pre-existing m.module = ...
                // assignment already races against GpuComm's reads in that
                // case; this invalidation is a strict improvement, not a
                // new race.
                program->last_result.valid = false;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::Stage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!Config::dumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::Stage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
                                                              }
} // namespace Vulkan
