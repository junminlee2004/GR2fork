// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <limits>
#include <ranges>
#include <xxhash.h>

#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
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
#include "video_core/skipcache/skipcache.h"

namespace Vulkan {

namespace Skipcache = VideoCore::Skipcache;

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

namespace {

// Address-independent specialization fingerprint - hashes the specialization's per-draw inputs
// (runtime_info, binding start, every bound sharp) with base_address zeroed so pointer re-emits
// hash identically. A superset of the spec identity, it can only over-discriminate, never
// wrongly reuse.
//
// GR2 measured 0 collisions over 12.2M samples. Callers exclude HS/DS (their spec folds tess
// constant-buffer contents read from guest memory). ri_bytes_hash is the raw-byte hash of the
// stage's persistent RuntimeInfo member.
// Hash only the header plus the stage-active union member. RuntimeInfo's
// operator== switches on stage and compares only the active member, so
// header+active is a superset of what equality consults and can only
// over-discriminate. The full struct is 8 bytes past XXH3's midsize cutoff,
// which forced every GetProgram through the hashLong path for union bytes
// equality never reads. Sizes come from sizeof only: MSVC packs the
// bitfields of FragmentRuntimeInfo differently, so literals would be wrong
// there. A garbage stage falls back to the whole union extent, never past it.
u64 RuntimeInfoProxyHash(const Shader::RuntimeInfo& ri) noexcept {
    // Not offsetof: the union members inherit, making RuntimeInfo
    // non-standard-layout, where offsetof is only conditionally supported.
    // The pointer difference folds to the same constant.
    const size_t header = static_cast<size_t>(reinterpret_cast<const char*>(&ri.ls_info) -
                                              reinterpret_cast<const char*>(&ri));
    const size_t union_extent = sizeof(Shader::RuntimeInfo) - header;
    size_t active = union_extent;
    switch (ri.stage) {
    case Shader::Stage::Local:
        active = sizeof(Shader::LocalRuntimeInfo);
        break;
    case Shader::Stage::Export:
        active = sizeof(Shader::ExportRuntimeInfo);
        break;
    case Shader::Stage::Vertex:
        active = sizeof(Shader::VertexRuntimeInfo);
        break;
    case Shader::Stage::Hull:
        active = sizeof(Shader::HullRuntimeInfo);
        break;
    case Shader::Stage::Geometry:
        active = sizeof(Shader::GeometryRuntimeInfo);
        break;
    case Shader::Stage::Fragment:
        active = sizeof(Shader::FragmentRuntimeInfo);
        break;
    case Shader::Stage::Compute:
        active = sizeof(Shader::ComputeRuntimeInfo);
        break;
    default:
        break;
    }
    // The header hash covers stage, so equal active bytes under different
    // stages (and therefore different lengths) cannot collide.
    const u64 h_header = XXH3_64bits(&ri, header);
    const u64 h_active =
        XXH3_64bits(reinterpret_cast<const u8*>(&ri) + header, std::min(active, union_extent));
    return h_header ^ (h_active * 0x9E3779B97F4A7C15ull);
}

u64 ComputeSpecProxyFp(const Shader::Info& info,
                       const std::optional<Shader::Gcn::FetchShaderData>& fetch_data,
                       u64 ri_bytes_hash, const Shader::Backend::Bindings& start) noexcept {
    u64 h = 0x84222325cbf29ce4ULL;
    const auto mix = [&](u64 v) noexcept { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    // Batched gather: one XXH3 over every sharp instead of one call per sharp.
    // Worst case: 12 B bindings + 40*16 + 64*32 + 8*32 + 16*16 sharps + VS attribute sharps.
    // 4096 covers it with headroom for 32 attrs.
    alignas(16) u8 buf[4096];
    size_t len = 0;
    const auto put = [&](const void* p, size_t n) noexcept {
        std::memcpy(buf + len, p, n);
        len += n;
    };
    const size_t attrib_bytes = (info.stage == Shader::Stage::Vertex && fetch_data)
                                    ? fetch_data->attributes.size() * sizeof(AmdGpu::Buffer)
                                    : 0;
    const size_t needed = sizeof(start) + info.buffers.size() * sizeof(AmdGpu::Buffer) +
                          info.images.size() * sizeof(AmdGpu::Image) +
                          info.fmasks.size() * sizeof(AmdGpu::Image) +
                          info.samplers.size() * sizeof(AmdGpu::Sampler) + attrib_bytes;
    // Image/fmask validity IS base_address != 0, and the spec bitset branches
    // on it, so zeroing the address must not erase it: fold a per-sharp
    // validity bit or two T#s differing only in valid-vs-null alias to one fp
    // and a hit returns the wrong permutation. Buffer validity is num_records,
    // which stays in the hashed bytes.
    u64 vmask = 0;
    u32 vidx = 0;
    const auto mix_valid = [&](bool valid) noexcept {
        vmask |= static_cast<u64>(valid) << (vidx++ & 63);
    };
    if (needed <= sizeof(buf)) [[likely]] {
        put(&start, sizeof(start));
        for (const auto& d : info.buffers) {
            AmdGpu::Buffer s = d.GetSharp(info);
            s.base_address = 0;
            put(&s, sizeof(s));
        }
        for (const auto& d : info.images) {
            AmdGpu::Image s = d.GetSharp(info);
            mix_valid(s.base_address != 0);
            s.base_address = 0;
            put(&s, sizeof(s));
        }
        for (const auto& d : info.fmasks) {
            AmdGpu::Image s = d.GetSharp(info);
            mix_valid(s.base_address != 0);
            s.base_address = 0;
            put(&s, sizeof(s));
        }
        for (const auto& d : info.samplers) {
            AmdGpu::Sampler s = d.GetSharp(info);
            put(&s, sizeof(s));
        }
        // vs_attribs are specialized only for the Vertex stage (see StageSpecialization);
        // fold the vertex-buffer sharps that feed them.
        if (attrib_bytes != 0) {
            for (const auto& a : fetch_data->attributes) {
                AmdGpu::Buffer s = a.GetSharp(info);
                s.base_address = 0;
                put(&s, sizeof(s));
            }
        }
        mix(ri_bytes_hash);
        mix(vmask);
        mix(XXH3_64bits(buf, len));
        return h ? h : 1ULL;
    }
    // Per-sharp overflow fallback (an absurd attribute count); counts are Program-static, so
    // the chosen form stays consistent for this Program.
    mix(ri_bytes_hash);
    mix(XXH3_64bits(&start, sizeof(start)));
    for (const auto& d : info.buffers) {
        AmdGpu::Buffer s = d.GetSharp(info);
        s.base_address = 0;
        mix(XXH3_64bits(&s, sizeof(s)));
    }
    for (const auto& d : info.images) {
        AmdGpu::Image s = d.GetSharp(info);
        mix_valid(s.base_address != 0);
        s.base_address = 0;
        mix(XXH3_64bits(&s, sizeof(s)));
    }
    for (const auto& d : info.fmasks) {
        AmdGpu::Image s = d.GetSharp(info);
        mix_valid(s.base_address != 0);
        s.base_address = 0;
        mix(XXH3_64bits(&s, sizeof(s)));
    }
    for (const auto& d : info.samplers) {
        AmdGpu::Sampler s = d.GetSharp(info);
        mix(XXH3_64bits(&s, sizeof(s)));
    }
    if (info.stage == Shader::Stage::Vertex && fetch_data) {
        for (const auto& a : fetch_data->attributes) {
            AmdGpu::Buffer s = a.GetSharp(info);
            s.base_address = 0;
            mix(XXH3_64bits(&s, sizeof(s)));
        }
    }
    mix(vmask);
    return h ? h : 1ULL;
}

} // namespace

constexpr static auto SpirvVersion1_6 = 0x00010600U;

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

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(Stage stage, LogicalStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto& regs = liverpool->regs;
    const auto BuildCommon = [&](const auto& program) {
        info.num_user_data = program.settings.num_user_regs;
        info.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.num_allocated_vgprs = program.NumVgprs();
        info.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.fp_round_mode32 = program.settings.fp_round_mode32;
        info.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage);
    switch (stage) {
    case Stage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
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
        info.es_info.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Vertex: {
        BuildCommon(regs.vs_program);
        info.vs_info.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
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
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
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
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            gs_info.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            gs_info.in_primitive = regs.primitive_type;
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
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Geometry));
        info.fs_info.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    case Stage::Compute: {
        const auto& cs_pgm = liverpool->GetCsRegs();
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
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
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
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };
    spec_mru_perm_probe = EmulatorSettings.IsSpecMruPermProbe();
    // Latched before WarmUp so deserialized permutations get signatures computed.
    spec_fp_cache = EmulatorSettings.IsSpecFpCache();
    WarmUp();

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
}

PipelineCache::~PipelineCache() = default;

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline() {
    if (!RefreshGraphicsKey()) {
        return nullptr;
    }
    // A repeated key returns the previous pipeline without hashing or probing the
    // map. pipe_gen invalidates the cached pair when ReplaceShader erases entries.
    const u64 pipe_gen =
        Skipcache::Framework::Instance().Gens().pipe_gen.load(std::memory_order_acquire);
    if (last_graphics_pipeline && pipe_gen == last_graphics_pipe_gen &&
        graphics_key == last_graphics_key) {
        return last_graphics_pipeline;
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos,
            fetch_shader_ref
                ? std::optional<const Shader::Gcn::FetchShaderData>{*fetch_shader_ref.Get()}
                : std::optional<const Shader::Gcn::FetchShaderData>{},
            modules, sdata, false);

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
        fetch_shader_ref = {};
    }
    // memcpy keeps the padding bytes deterministic for the memcmp-based compare.
    std::memcpy(&last_graphics_key, &graphics_key, sizeof(graphics_key));
    last_graphics_pipeline = it->second.get();
    last_graphics_pipe_gen = pipe_gen;
    return last_graphics_pipeline;
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

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

        const auto& bc = regs.blend_control[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_func == AmdGpu::BlendControl::BlendFunc::Min ||
             bc.color_func == AmdGpu::BlendControl::BlendFunc::Max) &&
            bc.color_src_factor == AmdGpu::BlendControl::BlendFactor::SrcColor &&
            bc.color_dst_factor == AmdGpu::BlendControl::BlendFactor::DstColor;
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
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

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader_ref = {};

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in, Shader::LogicalStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        FetchShaderRef fetch_ref{};
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_ref,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        if (fetch_ref) {
            fetch_shader_ref = fetch_ref;
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
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Vertex, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
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
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(Stage::Vertex, LogicalStage::Vertex);
        break;
    default:
        UNREACHABLE_MSG("unhandled stage_en: {}", (u32)regs.stage_enable.raw);
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader_ref && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader_ref.Get()->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    // Compute stages carry no fetch shader; the reference is discarded.
    FetchShaderRef fetch_ref{};
    std::tie(infos[0], modules[0], fetch_ref, compute_key.value) =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding);
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

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    // Reference, not copy: the member persists until the next
    // BuildRuntimeInfo call, and copying the struct per resolve was measured
    // per-draw-stage cost. Compile branches make their own mutable copy
    // because CompileModule rewrites tess/fragment fields through its
    // reference - and the new-program spec must be built from that SAME
    // mutated copy to keep stored-spec bytes identical to before.
    const auto& runtime_info = BuildRuntimeInfo(stage, l_stage);
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
        auto& program = it_pgm.value();
        auto start = binding;
        auto ri_compile = runtime_info;
        const auto module = CompileModule(program->info, ri_compile, params.code, 0, binding);
        auto spec = Shader::StageSpecialization(program->info, ri_compile, profile, start);
        const auto perm_hash = HashCombine(params.hash, 0);

        if (spec_fp_cache) {
            spec.ComputeSig();
        }
        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        return std::make_tuple(&program->info, module, FetchShaderRef{program.get(), 0u},
                               perm_hash);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    info.RefreshFlatBuf();

    // Spec-fingerprint tier: catches the per-draw UBO-pointer churn a raw user_data key cannot
    // (the raw bytes change every draw, the spec result does not). A hit skips the
    // StageSpecialization rebuild below entirely. HS/DS are excluded - their spec folds tess
    // constant-buffer contents this key cannot cover. spec_fp is reused at the populate site
    // after resolution; it stays 0 when the tier did not run.
    u64 spec_fp = 0;
    const bool spec_fp_eligible = spec_fp_cache && l_stage != LogicalStage::TessellationControl &&
                                  l_stage != LogicalStage::TessellationEval;
    if (spec_fp_eligible && !program->modules.empty()) {
        // Hash the persistent member, not the local copy: the member's padding bytes are stable
        // across calls, so the raw-byte hash repeats; a padding mismatch could only
        // over-discriminate into a miss, never wrongly hit.
        const auto& ri_member = runtime_infos[static_cast<u32>(l_stage)];
        const u64 ri_fp_hash = RuntimeInfoProxyHash(ri_member);
        spec_fp = ComputeSpecProxyFp(info, program->modules[0].spec.fetch_shader_data, ri_fp_hash,
                                     binding);
        // MRU front: consecutive draws overwhelmingly repeat the fingerprint,
        // and this compare reads a line the probe already has hot.
        if (program->mru_fp == spec_fp && program->mru_perm_idx < program->modules.size())
            [[likely]] {
            const size_t hit_idx = program->mru_perm_idx;
            const u64 hit_hash = HashCombine(params.hash, hit_idx);
            info.AddBindings(binding);
            program->last_hit_perm = static_cast<u32>(hit_idx);
            return std::make_tuple(&program->info, program->modules[hit_idx].module,
                                   FetchShaderRef{program.get(), static_cast<u32>(hit_idx)},
                                   hit_hash);
        }
        if (!program->spec_fp_lru) {
            program->spec_fp_lru = std::make_unique<
                std::array<Program::SpecFpCacheEntry, Program::kSpecFpCacheSize>>();
        }
        const u32 fp_slot = static_cast<u32>(spec_fp) & (Program::kSpecFpCacheSize - 1);
        const auto& fe = (*program->spec_fp_lru)[fp_slot];
        if (fe.valid && fe.fp == spec_fp && fe.perm_idx < program->modules.size()) [[likely]] {
            const size_t hit_idx = fe.perm_idx;
            const u64 hit_hash = HashCombine(params.hash, hit_idx);
            info.AddBindings(binding);
            program->mru_fp = spec_fp;
            program->mru_perm_idx = fe.perm_idx;
            program->last_hit_perm = static_cast<u32>(hit_idx);
            return std::make_tuple(&program->info, program->modules[hit_idx].module,
                                   FetchShaderRef{program.get(), static_cast<u32>(hit_idx)},
                                   hit_hash);
        }
    }

    auto& spec = spec_scratch;
    spec.Rebuild(info, runtime_info, profile, binding);
#ifdef _DEBUG
    {
        // A fresh construction must match the rebuilt scratch member for member.
        const Shader::StageSpecialization ref_spec(info, runtime_info, profile, binding);
        DEBUG_ASSERT(ref_spec.info == spec.info && ref_spec.runtime_info == spec.runtime_info &&
                     ref_spec.bitset == spec.bitset &&
                     ref_spec.fetch_shader_data == spec.fetch_shader_data &&
                     ref_spec.vs_attribs == spec.vs_attribs && ref_spec.buffers == spec.buffers &&
                     ref_spec.images == spec.images && ref_spec.fmasks == spec.fmasks &&
                     ref_spec.samplers == spec.samplers && ref_spec.start == spec.start);
    }
#endif

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    if (spec_fp_cache) {
        // Fast path: look up by specialization signature. A *pair* of signatures (sig + sig2)
        // stands in for the deep comparisons the legacy branch below runs.
        spec.ComputeSig();
        bool found = false;
        if (const auto it_sig = program->perm_index_by_sig.find(spec.sig);
            it_sig != program->perm_index_by_sig.end() &&
            it_sig->second < program->modules.size()) {
            const auto& ms = program->modules[it_sig->second].spec;
            // The sig-map hit already matches spec.sig up to a ~2^-64 hash collision; sig2,
            // computed from the same StageSpecialization fields, confirms the match.
            if (ms.sig == spec.sig && ms.sig2 == spec.sig2) [[likely]] {
                info.AddBindings(binding);
                perm_idx = it_sig->second;
                perm_hash = HashCombine(params.hash, perm_idx);
                module = program->modules[perm_idx].module;
                found = true;
            }
        }
        if (!found) {
            // Fallback: linear scan by (sig, sig2) without deep comparisons.
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
                auto ri_compile = runtime_info;
                module = CompileModule(new_info, ri_compile, params.code, perm_idx, binding);

                RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
                program->AddPermut(module, std::move(spec));
            } else {
                info.AddBindings(binding);
                module = program->modules[found_idx].module;
                perm_idx = found_idx;
                perm_hash = HashCombine(params.hash, perm_idx);
                // Keep the map warm for future lookups.
                program->perm_index_by_sig.try_emplace(spec.sig, perm_idx);
            }
        }
        // Record spec fingerprint -> perm_idx so the next draw with structurally identical
        // (address-masked) sharps skips the StageSpecialization rebuild; spec_fp is nonzero
        // only when the eligible tier above computed it (HS/DS leave it 0).
        if (spec_fp_eligible && spec_fp != 0 && perm_idx < program->modules.size()) {
            const u32 fp_slot = static_cast<u32>(spec_fp) & (Program::kSpecFpCacheSize - 1);
            (*program->spec_fp_lru)[fp_slot] = Program::SpecFpCacheEntry{
                .fp = spec_fp,
                .perm_idx = static_cast<u32>(perm_idx),
                .valid = true,
            };
            program->mru_fp = spec_fp;
            program->mru_perm_idx = static_cast<u32>(perm_idx);
        }
        program->last_hit_perm = static_cast<u32>(perm_idx);
        return std::make_tuple(&program->info, module,
                               FetchShaderRef{program.get(), static_cast<u32>(perm_idx)},
                               perm_hash);
    }

    auto it = program->modules.end();
    if (spec_mru_perm_probe) {
        // Probes the previously matched permutation with the same predicate and
        // orientation the linear search below uses.
        const u32 mru = program->last_hit_perm;
        if (mru < program->modules.size() && program->modules[mru].spec == spec) {
            it = program->modules.begin() + mru;
        }
    }
    if (it == program->modules.end()) {
        it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    }
    if (it == program->modules.end()) {
        auto new_info = Shader::Info(stage, l_stage, params);
        auto ri_compile = runtime_info;
        module = CompileModule(new_info, ri_compile, params.code, perm_idx, binding);

        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(module, std::move(spec));
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
    }
    program->last_hit_perm = static_cast<u32>(perm_idx);
    return std::make_tuple(&program->info, module,
                           FetchShaderRef{program.get(), static_cast<u32>(perm_idx)}, perm_hash);
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
                // spec_fp_lru and perm_index_by_sig store permutation indices and read the
                // module fresh on every hit, so this in-place swap needs no invalidation
                // there; the pipe_gen bump below covers the pipeline-level memo.
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
    Skipcache::Framework::Instance().BumpPipeGen();
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
    if (!EmulatorSettings.IsDumpShaders()) {
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
