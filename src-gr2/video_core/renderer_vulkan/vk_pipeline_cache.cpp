// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <ranges>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <xxhash.h>

#include "common/config.h"
#include "core/libraries/resolution_patches/resolution_patches.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "common/arch.h"
#include "core/debug_state.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/fetch_shader.h"
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

// GR2FORK: address-independent specialization fingerprint - hashes ComputeSig's per-draw inputs
// (runtime_info, binding start, every bound sharp) with base_address zeroed so pointer re-emits
// hash identically. A superset of ComputeSig, it can only over-discriminate, never wrongly reuse.

// GR2FORK PERF: 0 collisions over 12.2M GR2 samples. Callers exclude HS/DS (their spec folds
// tess constant-buffer contents read from guest memory). ri_bytes_hash (~0.44% of sys time) is
// memoized per stage by GetProgram; GR2_NOFPBATCH=1 swaps the batched hash for the per-sharp form.
u64 ComputeSpecProxyFp(
    const Shader::Info& info,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_data,
    u64 ri_bytes_hash,
    const Shader::Backend::Bindings& start) noexcept {
    static const bool batch_enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOFPBATCH");
        return !(e && e[0] == '1');
    }();
    u64 h = 0x84222325cbf29ce4ULL;
    const auto mix = [&](u64 v) noexcept {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    if (batch_enabled) [[likely]] {
        // Worst case: 12 B bindings + 40*16 + 64*32 + 8*32 + 16*16 sharps
        // + VS attribute sharps. 4096 covers it with headroom for 32 attrs.
        alignas(16) u8 buf[4096];
        size_t len = 0;
        const auto put = [&](const void* p, size_t n) noexcept {
            std::memcpy(buf + len, p, n);
            len += n;
        };
        const size_t attrib_bytes =
            (info.stage == Shader::Stage::Vertex && fetch_data)
                ? fetch_data->attributes.size() * sizeof(AmdGpu::Buffer)
                : 0;
        const size_t needed = sizeof(start) +
                              info.buffers.size() * sizeof(AmdGpu::Buffer) +
                              info.images.size() * sizeof(AmdGpu::Image) +
                              info.fmasks.size() * sizeof(AmdGpu::Image) +
                              info.samplers.size() * sizeof(AmdGpu::Sampler) +
                              attrib_bytes;
        if (needed <= sizeof(buf)) [[likely]] {
            put(&start, sizeof(start));
            for (const auto& d : info.buffers) {
                AmdGpu::Buffer s = d.GetSharp(info);
                s.base_address = 0;
                put(&s, sizeof(s));
            }
            for (const auto& d : info.images) {
                AmdGpu::Image s = d.GetSharp(info);
                s.base_address = 0;
                put(&s, sizeof(s));
            }
            for (const auto& d : info.fmasks) {
                AmdGpu::Image s = d.GetSharp(info);
                s.base_address = 0;
                put(&s, sizeof(s));
            }
            for (const auto& d : info.samplers) {
                AmdGpu::Sampler s = d.GetSharp(info);
                put(&s, sizeof(s));
            }
            // vs_attribs are specialized only for the Vertex stage (see the
            // StageSpecialization ctor); fold the vertex-buffer sharps that
            // feed them.
            if (attrib_bytes != 0) {
                for (const auto& a : fetch_data->attributes) {
                    AmdGpu::Buffer s = a.GetSharp(info);
                    s.base_address = 0;
                    put(&s, sizeof(s));
                }
            }
            mix(ri_bytes_hash);
            mix(XXH3_64bits(buf, len));
            return h ? h : 1ULL;
        }
        // Overflow (an absurd attribute count) falls through to the per-sharp form; counts are
        // Program-static, so the choice stays consistent for this Program.
    }
    // Per-sharp form (GR2_NOFPBATCH=1, or gather-overflow fallback).
    mix(ri_bytes_hash);
    mix(XXH3_64bits(&start, sizeof(start)));
    for (const auto& d : info.buffers) {
        AmdGpu::Buffer s = d.GetSharp(info);
        s.base_address = 0;
        mix(XXH3_64bits(&s, sizeof(s)));
    }
    for (const auto& d : info.images) {
        AmdGpu::Image s = d.GetSharp(info);
        s.base_address = 0;
        mix(XXH3_64bits(&s, sizeof(s)));
    }
    for (const auto& d : info.fmasks) {
        AmdGpu::Image s = d.GetSharp(info);
        s.base_address = 0;
        mix(XXH3_64bits(&s, sizeof(s)));
    }
    for (const auto& d : info.samplers) {
        AmdGpu::Sampler s = d.GetSharp(info);
        mix(XXH3_64bits(&s, sizeof(s)));
    }
    // vs_attribs are specialized only for the Vertex stage (see the
    // StageSpecialization ctor); fold the vertex-buffer sharps that feed them.
    if (info.stage == Shader::Stage::Vertex && fetch_data) {
        for (const auto& a : fetch_data->attributes) {
            AmdGpu::Buffer s = a.GetSharp(info);
            s.base_address = 0;
            mix(XXH3_64bits(&s, sizeof(s)));
        }
    }
    return h ? h : 1ULL;
}

// GR2FORK: stage-memo kill switch. GR2_NOSTAGEMEMO=1 disables the Level-2.5
// per-stage resolve memo consult (populate is unaffected). The env is read
// once per process; enabled by default and not config-controlled.
bool StageMemoEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOSTAGEMEMO");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

// GR2FORK: stage-compare kill switch. GR2_NOSTAGECMP=1 restores the full-key memcmp at the
// Level-2.5 reuse check; the narrowed stage-region compare is the default. Read once per process.
bool StageCmpNarrowEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOSTAGECMP");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

// GR2FORK: stage-memo eligibility - only pairs whose BuildRuntimeInfo reads nothing but snapshot
// registers frozen under !gfx_key_ctx_dirty (see StageResolveMemo in vk_pipeline_cache.h).
// Geometry/Local read guest memory; Export/Hull/TES are rare; Compute never passes ctx_stable.
constexpr bool StageMemoEligible(Shader::Stage stage, Shader::LogicalStage l_stage) noexcept {
    return (stage == Shader::Stage::Vertex && l_stage == Shader::LogicalStage::Vertex) ||
           (stage == Shader::Stage::Fragment && l_stage == Shader::LogicalStage::Fragment);
}

// GR2FORK: pipeline compile graveyard. std::future's destructor joins its worker thread, so a
// driver hung inside vkCreateGraphicsPipelines would block any future destruction forever;
// abandoned futures are parked here, never freed, and _exit reaps the hung threads on exit.
struct PipelineCompileGraveyard {
    std::mutex mu;
    // Heap-allocated vector we never delete - this is the leak-on-purpose.
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
    // Leaked Meyers-style singleton - heap allocated + never deleted, so its
    // dtor never runs (which is exactly what we want; see comment above).
    static auto* g = new PipelineCompileGraveyard();
    return *g;
}

} // namespace

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

// GR2FORK PERF: hash RuntimeInfo by stage, matching operator== semantics exactly.
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
        // GR2FORK PERF: pack the eight scalar fields into three mix calls (two u32 pairs, plus
        // the tess bools and small enums in one u64), cutting the per-Vertex mix chain from 9 to
        // 4 iterations - ~25 cycles of serial dependency per call.
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
        // GR2FORK PERF: pack four small scalars into one mix call; the counts are bounded by GS
        // spec limits and in_primitive is a small enum, so 16-bit slots in a u64 suffice.
        const u64 packed_scalars =
            static_cast<u64>(g.num_outputs) |
            (static_cast<u64>(g.num_invocations) << 16) |
            (static_cast<u64>(g.output_vertices) << 32) |
            (static_cast<u64>(static_cast<u32>(g.in_primitive)) << 48);
        h = mix(h, packed_scalars);
        // GR2FORK PERF: g.outputs is 12 bytes, where XXH3 call overhead dominates the hashing;
        // reading it inline as 8B + 4B mixes saves the function call (~15-20 cycles).
        static_assert(sizeof(g.outputs) == 12);
        u64 outputs_lo;
        u32 outputs_hi;
        std::memcpy(&outputs_lo, g.outputs.data(), 8);
        std::memcpy(&outputs_hi,
                    reinterpret_cast<const u8*>(g.outputs.data()) + 8, 4);
        h = mix(h, outputs_lo);
        h = mix(h, static_cast<u64>(outputs_hi));
        // GR2FORK PERF: out_primitive holds four enums of 6 hardware bits each (regs_vertex.h:
        // outprim_type : 6); packing them byte-aligned into one u64 saves the XXH3 call
        // (~15-20 cycles) plus one mix iteration.
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
        // GR2FORK PERF: en_flags/addr_flags are 4-byte bitfield registers; XXH3's ~10-15 cycles
        // of call overhead dwarfs hashing 4 bytes, so read both as u32 and pack into one mix
        // call (saves 2 XXH3 calls and 1 mix iteration per Fragment hash).
        static_assert(sizeof(f.en_flags) == 4);
        static_assert(sizeof(f.addr_flags) == 4);
        u32 en32, addr32;
        std::memcpy(&en32, &f.en_flags, sizeof(en32));
        std::memcpy(&addr32, &f.addr_flags, sizeof(addr32));
        h = mix(h, (static_cast<u64>(en32) << 32) | static_cast<u64>(addr32));
        // GR2FORK PERF: num_inputs (<= 32), z_export_format, mrtz_mask, and dual_source_blending
        // fit one u64; packing cuts 2 mix iterations off the chain per Fragment hash.
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
        // GR2FORK PERF: each workgroup_size axis is under 2^20 by Vulkan/AMD limits, so pack the
        // three axes (20 bits each) plus the tgid_enable bools (bits 60-62) into one u64;
        // replaces 1 XXH3 + 2 mix calls with 1 mix (~25 cycles per Compute hash).
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
        // GR2FORK: skip-safe - leave the copy shader empty rather than abort if
        // the GS copy (vs_program) has no locatable binary info.
        if (const auto params_vc = AmdGpu::TryGetParams(regs.vs_program)) {
            gs_info.vs_copy = params_vc->code;
            gs_info.vs_copy_hash = params_vc->hash;
            DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, 0, "copy.bin");
        } else {
            LOG_WARNING(Render_Vulkan,
                        "GS copy program @{:#x} has no shader binary info (no OrbShdr "
                        "within scan window); leaving copy shader empty.",
                        reinterpret_cast<uintptr_t>(regs.vs_program.Address<u32*>()));
            gs_info.vs_copy = {};
            gs_info.vs_copy_hash = 0;
        }
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
      desc_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes},
      // GR2FORK PERF: latch the five process-constant gates once -
      // see the member doc block in vk_pipeline_cache.h.
      key_ctx_skip_enabled_{Config::isPipelineGfxKeyCtxSkipEnabled()},
      stage_cmp_narrow_{StageCmpNarrowEnabled()},
      stage_memo_enabled_{StageMemoEnabled()},
      ud_hash_lru_enabled_{Config::isPipelineUdHashLruEnabled()},
      spec_fp_lru_enabled_{Config::isPipelineSpecFpLruEnabled()} {
    LOG_INFO(Render_Vulkan,
             "[GR2FORK K1] Level-2.5 stage-region compare (vs full-key memcmp): {} "
             "(forced on; GR2_NOSTAGECMP)",
             stage_cmp_narrow_);
    // GR2FORK PERF: the RGS stage-array-reset boot echo lives here rather
    // than as a magic-static inside RefreshGraphicsStages, which would cost
    // a guard load on every call.
    LOG_INFO(Render_Vulkan,
             "[GR2FORK R2] RGS stage-array reset: scalar stores "
             "(vectorized fill + vzeroupper removed; A/B by build)");
    const auto& vk12_props = instance.GetVk12Properties();
    // GR2FORK: FragCoordResolutionScalePass divides FragCoord by the resolution-patch pixel
    // density (target_vertical / 1080). GRR-only: GRR's patches keep the inverse-resolution shader
    // constants native, while GR2's D1 rewrites them to 1/target and the pass would double-correct.
    float gr2fork_res_density = 1.0f;
    if (Config::isGravityRushRemastered()) {
        const auto rp_target = Libraries::ResolutionPatches::ParseResolutionFromConfig(
            Config::getResolutionOverride());
        if (rp_target != Libraries::ResolutionPatches::TargetResolution::Off) {
            const auto rp_base = Libraries::ResolutionPatches::TargetResolutionToBaseSize(rp_target);
            if (rp_base.height > 0) {
                gr2fork_res_density = static_cast<float>(rp_base.height) / 1080.0f;
            }
        }
    }

    profile = Shader::Profile{
        // When binding a UBO, we calculate its size considering the offset in the larger buffer
        // cache underlying resource. In some cases, it may produce sizes exceeding the system
        // maximum allowed UBO range, so we need to reduce the threshold to prevent issues.
        .max_ubo_size = instance.UniformMaxSize() - instance.UniformMinAlignment(),
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .internal_resolution_scale = gr2fork_res_density,
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
        // GR2FORK: IMAGE_STORE_MIP fallback ported from upstream #4075, preferred on both AMD
        // and NVIDIA for GR2 per compat issue #1429 (mipmap-only-level-0 bug). Costs N descriptor
        // slots instead of 1 per IMAGE_STORE_MIP image where native load-store-lod works.
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

    // WarmUp() is not invoked here: Presenter::WarmUpPipelineCache() runs it after the Presenter
    // is fully constructed so a "LOADING SHADERS" overlay can be shown instead of a black window.
    // The vk::PipelineCache is still created here for LoadGraphicsPipeline/LoadComputePipeline.

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
}

PipelineCache::~PipelineCache() {
    // GR2FORK: move still-in-flight compile futures to the graveyard before the map destructs
    // them - destructing a std::future joins its worker thread, and a thread hung inside the
    // Vulkan driver would freeze the emulator on the way out.
    for (auto& [key, pending] : pending_graphics_pipelines) {
        if (pending && pending->future.valid()) {
            Graveyard().Bury(std::move(pending->future));
        }
    }
    pending_graphics_pipelines.clear();
}


const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(
    const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // GR2FORK: read the stamp + dirty bit from the captured snapshot, not the live Liverpool;
    // under async execution separate loads could disagree with the regs captured for this intent.
    const u64 stamp = regs.gfx_pipeline_stamp;
    // GR2FORK PERF: the gfx-stamp early-out is the dominant exit -
    // consecutive draws share a pipeline and the stamp only bumps on register
    // writes that affect pipeline state.
    if (stamp == last_gfx_stamp && last_gfx_pipeline) [[likely]] {
        return last_gfx_pipeline;
    }
    // Level 2: the stamp bumped but only dynamic-state regs changed (viewport, scissor, blend
    // constants) - the pipeline key cannot have changed, so skip the RefreshGraphicsKey rebuild.
    // GR2FORK PERF: this is the dominant exit among the post-stamp-mismatch cases.
    if (!regs.gfx_key_dirty && last_gfx_pipeline) [[likely]] {
        last_gfx_stamp = stamp;
        return last_gfx_pipeline;
    }
    // Level 2.5 (GR2FORK, pipelineGfxKeyCtxSkipEnable, default-on): the key is dirty but no
    // key-affecting context register changed, so only the resolved module set can differ. Resolve
    // stages and re-check key equality to skip RefreshGraphicsKey's rebuild (~0.89% on GR2).
    if (key_ctx_skip_enabled_ && !regs.gfx_key_ctx_dirty &&
        last_gfx_pipeline && instance.IsVertexInputDynamicState()) [[likely]] {
        // GR2FORK: the only ctx_stable=true call site - the snapshot proved !gfx_key_ctx_dirty,
        // which licenses the per-stage resolve memo inside bind_stage/GetProgram.

        // GR2FORK PERF: compare only what RefreshGraphicsStages(ctx_stable=true) mutates -
        // stage_hashes, mrt_mask, num_color_attachments - not the ~450 B key memcmp (~0.37% sys;
        // GR2_NOSTAGECMP restores it). A relaxed vertex-input gate must add vertex_buffer_formats.
        if (RefreshGraphicsStages(regs, /*ctx_stable=*/true) &&
            (stage_cmp_narrow_
                 ? (std::memcmp(graphics_key.stage_hashes.data(),
                                prev_key_narrow_.stage_hashes.data(),
                                sizeof(graphics_key.stage_hashes)) == 0 &&
                    graphics_key.mrt_mask == prev_key_narrow_.mrt_mask &&
                    graphics_key.num_color_attachments ==
                        prev_key_narrow_.num_color_attachments)
                 : (std::memcmp(&graphics_key, &prev_graphics_key_,
                                offsetof(GraphicsPipelineKey, cached_hash_)) == 0))) {
            last_gfx_stamp = stamp;
            return last_gfx_pipeline;
        }
    }
    // GR2FORK: no ClearGfxKeyDirty() needed - PM4 clears the live flag inside
    // Liverpool::CaptureSnapshot() under sole-writer ownership; the snapshot's gfx_key_dirty is
    // a frozen "was dirty at capture" view.
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

    // GR2FORK: poll (non-blockingly) for an async compile a previous call for this key launched
    // past the sync budget; finalize into graphics_pipelines when ready.
    // GR2FORK PERF: post-warmup, pipelines live in graphics_pipelines - find() misses dominate.
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
    // GR2FORK PERF: post-warmup is_new is the rare case - most
    // distinct pipelines have already been compiled during early-game.
    if (is_new) [[unlikely]] {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        auto pending = LaunchAsyncPipelineCompile(graphics_key, pipeline_hash);

        // Most compiles finish in <50ms, so waiting the configured gameplaySyncBudgetMs catches
        // them without frame-skip; on timeout the future is stashed in pending_graphics_pipelines
        // and the draw is skipped. Config is read fresh on this cold path (launcher slider).
        const std::chrono::milliseconds sync_budget{Config::getGameplaySyncBudgetMs()};
        if (pending->future.wait_for(sync_budget) == std::future_status::ready) {
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
            // Finalize side effects - these need post-ctor state (sdata, modules).
            RegisterPipelineData(graphics_key, pipeline_hash, pending->sdata);
            ++num_new_pipelines;
            if (Config::collectShadersForDebug()) {
                for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                    if (pending->live_infos[stage]) {
                        auto& m = pending->modules_copy[stage];
                        module_related_pipelines[m].emplace_back(graphics_key);
                    }
                }
            }
            fetch_shader.reset();
        } else {
            // Compile not finished: stash the pending entry, leave graphics_pipelines[key] null
            // ("in-flight"), and return null so the Rasterizer skips this draw; later draws for
            // the key hit the TryFinalizePending branch above.
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
    static_assert(sizeof(prev_key_narrow_.stage_hashes) == sizeof(graphics_key.stage_hashes));
    prev_key_narrow_.stage_hashes = graphics_key.stage_hashes;
    prev_key_narrow_.mrt_mask = graphics_key.mrt_mask;
    prev_key_narrow_.num_color_attachments = graphics_key.num_color_attachments;
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
        // Compile failed. Don't leave a permafail marker - let the outer code
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
            if (pending.live_infos[stage]) {
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

    // Deep-copy stage data: the infos/runtime_infos/modules members are span-targets overwritten
    // by the next RefreshGraphicsStages, so the async task must not alias them. The worker reads
    // the immutable Info snapshots; the finished pipeline is repointed at live_infos afterwards.
    pending->live_infos = infos;
    for (size_t i = 0; i < MaxShaderStages; ++i) {
        if (infos[i]) {
            pending->info_snapshot[i].emplace(*infos[i]);
            pending->snapshot_infos[i] = &*pending->info_snapshot[i];
        } else {
            pending->snapshot_infos[i] = nullptr;
        }
    }
    pending->runtime_infos_copy = runtime_infos;
    pending->modules_copy = modules;
    pending->fetch_shader_copy = fetch_shader;

    // Capture by raw pointer: the pending entry lives in the PipelineCache-owned map until
    // finalize/permafail (or moves to the graveyard, never touched again), so pointers into it
    // stay stable for the worker's execution.
    PendingGraphicsPipeline* raw = pending.get();

    // Snapshot fields the ctor needs but that reference PipelineCache members.
    const Instance* instance_ptr = &instance;
    Scheduler* scheduler_ptr = &scheduler;
    DescriptorHeap* desc_heap_ptr = &desc_heap;
    const Shader::Profile* profile_ptr = &profile;
    vk::PipelineCache cache_handle = *pipeline_cache;
    GraphicsPipelineKey key_copy = key;

    // GR2FORK PERF: keep async pipeline compiles off the pinned GpuComm/GpuAssembler cores - a
    // std::async worker inherits all-CPU affinity, so a compile burst could land a heavy
    // GraphicsPipeline ctor on a hot core and stall the frame. The worker pins itself first.
    static const u64 compile_affinity_mask = []() -> u64 {
        // GR2FORK: GetHotCoreFenceMask returns 0 in non-exclusive mode, leaving the worker
        // unconstrained - with no reserved cores there is nothing to protect.
        const u64 forbidden = Common::GetHotCoreFenceMask();
        if (forbidden == 0) {
            return 0;
        }
        const unsigned hw = std::thread::hardware_concurrency();
        const u64 all = (hw == 0 || hw >= 64) ? ~0ull : ((1ull << hw) - 1);
        const u64 allowed = all & ~forbidden;
        return allowed; // if somehow empty, the guard below skips application
    }();

    pending->future = std::async(
        std::launch::async,
        [raw, instance_ptr, scheduler_ptr, desc_heap_ptr, profile_ptr, cache_handle,
         key_copy]() -> std::unique_ptr<GraphicsPipeline> {
            // GR2FORK affinity fence: applied before any real work so the GraphicsPipeline ctor
            // below runs on an allowed core only.
            if (compile_affinity_mask != 0) {
                Common::SetCurrentThreadAffinityMask(compile_affinity_mask);
            }
            // Spans over the pending-owned copies, built from the immutable snapshot_infos, not
            // the live Infos: BuildDescSetLayout reads flattened_ud_buf via GetSharp, and a
            // concurrent GetProgram->RefreshFlatBuf() could realloc that buffer under the read.
            std::span<const Shader::Info*, MaxShaderStages> infos_span{raw->snapshot_infos};
            std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_span{
                raw->runtime_infos_copy};
            std::span<const vk::ShaderModule> modules_span{raw->modules_copy};
            // Note: GraphicsPipeline ctor ASSERTs on driver-return failure, which
            // kills the process. We can't soften that; it's only the hang case
            // (no return at all) that this whole mechanism addresses.
            auto pipeline = std::make_unique<GraphicsPipeline>(
                *instance_ptr, *scheduler_ptr, *desc_heap_ptr, *profile_ptr, key_copy,
                cache_handle, infos_span, runtime_span, raw->fetch_shader_copy, modules_span,
                raw->sdata, false);
            // Repoint stages at the live Infos: per-draw BindResources needs the live
            // flattened_ud_buf, and the ctor read the snapshot only for layout construction.
            pipeline->RebindInfoStages(raw->live_infos);
            return pipeline;
        });
    return pending;
}

const ComputePipeline* PipelineCache::GetComputePipeline(
    const AmdGpu::LiverpoolRegsSnapshot& regs) {
    // GR2FORK PERF: RefreshComputeKey returns false only on
    // unrecognised / invalid compute state - error path. Steady-state
    // dispatches refresh successfully.
    if (!RefreshComputeKey(regs)) [[unlikely]] {
        return nullptr;
    }
    // GR2FORK PERF: same-key memo skips the map probe on consecutive dispatches of one pipeline;
    // guarding on the pipeline pointer keeps a legitimate zero key from false-hitting the memo.
    if (last_compute_pipeline_ && compute_key.value == last_compute_key_) [[likely]] {
        return last_compute_pipeline_;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    // GR2FORK PERF: post-warmup is_new is rare - every distinct
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

        // GR2FORK PERF: debug-only config for the shader-collector
        // workflow - disabled in release.
        if (Config::collectShadersForDebug()) [[unlikely]] {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    // GR2FORK: refresh the memo on the normal exit.
    last_compute_key_ = compute_key.value;
    last_compute_pipeline_ = it->second.get();
    return it->second.get();
}

bool PipelineCache::RefreshGraphicsKey(const AmdGpu::LiverpoolRegsSnapshot& regs) {
    auto& key = graphics_key;
    // GR2FORK PERF: selective field-range zeroing replaces a whole-struct memset (~390 bytes);
    // the skipped fields are unconditionally rewritten by this function or RefreshGraphicsStages.
    // The static_asserts pin the field order so a reorder fails the build, not corrupts keys.
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
    // Range C: color_samples + mrt_mask + both anonymous bitfield words. mrt_mask is rewritten
    // anyway, but including it keeps the range contiguous and clears the bitfield padding bits.
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
    // GR2FORK: full-rebuild path - ctx may have changed, so the
    // per-stage memo must not be trusted (it is repopulated by this call).
    if (!RefreshGraphicsStages(regs, /*ctx_stable=*/false)) {
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

bool PipelineCache::RefreshGraphicsStages(const AmdGpu::LiverpoolRegsSnapshot& regs,
                                          bool ctx_stable) {
    auto& key = graphics_key;
    // GR2FORK PERF: skip the fetch_shader reset (and the deep copies at the bind_stage return
    // sites) on the ctx_stable path (~66.6% of calls) - no consumer reads the member there, and
    // a stage-compare failure reruns with ctx_stable=false. Saves a ~150 B copy per draw.
    if (!ctx_stable) {
        fetch_shader = std::nullopt;
    }

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in, Shader::LogicalStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        // GR2FORK PERF: the surrounding switch already knows the stage is active; the
        // IsStageEnabled re-check is defensive and almost always passes.
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) [[unlikely]] {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        // GR2FORK PERF: null program / null pgm address is a
        // defensive guard against malformed PM4; well-formed shader
        // emission has both populated.
        if (!pgm || !pgm->Address<u32*>()) [[unlikely]] {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        // GR2FORK PERF: under ctx_stable the program-address SH words are value-frozen, so when
        // the memo's code pointer equals the live address TryGetParams would re-derive the same
        // {code span, hash}; rebuild from the memo (proof on StageResolveMemo in the header).
        StageResolveMemo& memo = stage_memo_[stage_out_idx];
        if (ctx_stable && stage_memo_enabled_ && memo.valid &&
            memo.code_data == pgm->Address<u32*>()) [[likely]] {
            const Shader::ShaderParams params{
                .user_data = pgm->user_data,
                .code = std::span<const u32>{memo.code_data, memo.code_words},
                .hash = memo.params_hash,
            };
            const std::optional<Shader::Gcn::FetchShaderData>* fetch_shader_ptr = nullptr;
            std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_ptr,
                     key.stage_hashes[stage_out_idx]) =
                GetProgram(stage_in, stage_out, params, binding, regs, /*ctx_stable=*/true);
            // GR2FORK: this branch runs only under ctx_stable, where the
            // fetch_shader member is never read (see the function-top
            // comment), so no fetch-shader deep copy is needed.
            (void)fetch_shader_ptr;
            return true;
        }

        const auto params_opt = AmdGpu::TryGetParams(*pgm);
        // GR2FORK: a non-null program pointer can still reference mapped-but-non-shader memory
        // (a stale pointer, or memory the guest streamed over); the recompiler's binary-info
        // search would hit a fatal UNREACHABLE, so skip the stage and log the address.
        if (!params_opt) [[unlikely]] {
            LOG_WARNING(Render_Vulkan,
                        "Stage {} program @{:#x} has no shader binary info (no OrbShdr "
                        "within scan window); skipping stage.",
                        stage_in_idx, reinterpret_cast<uintptr_t>(pgm->Address<u32*>()));
            // GR2FORK: the code at this address became unreadable
            // - a memo entry pointing into it must not be replayed.
            memo.valid = false;
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }
        const auto& params = *params_opt;
        // GR2FORK PERF: GetProgram returns a pointer to the program-owned fetch-shader optional
        // rather than a copy, eliminating one heap-allocating copy through std::tie; the single
        // deep copy below targets PipelineCache's member, the only storage that needs it.
        const std::optional<Shader::Gcn::FetchShaderData>* fetch_shader_ptr = nullptr;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_ptr,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding, regs, ctx_stable);
        // GR2FORK: copy only when the member can be read downstream (full
        // path). Under ctx_stable the member is dead - see the function-top
        // comment.
        if (!ctx_stable && fetch_shader_ptr && fetch_shader_ptr->has_value()) {
            fetch_shader = *fetch_shader_ptr;
        }
        return true;
    };

    // GR2FORK PERF: plain vectorized clears; instruction-precise sampling
    // attributes ~2% of the symbol here, and the scalar-store variant with
    // per-iteration barriers costs ~10x that in issued stores (Zen 4 has no
    // vzeroupper stall).
    infos.fill(nullptr);
    modules.fill(vk::ShaderModule{});
    bind_stage(Stage::Fragment, LogicalStage::Fragment);

    const auto* fs_info = infos[static_cast<u32>(LogicalStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        // GR2FORK PERF: RADV/Mesa with VK_EXT_extended_dynamic_state3
        // exposes geometry shader support - the unsupported warning path
        // is for outlier configurations.
        if (!instance.IsGeometryStageSupported()) [[unlikely]] {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        // GR2FORK PERF: stream output is unimplemented in shadPS4;
        // games that use it hit this warning path. The vast majority of
        // GS-using shaders don't enable stream output.
        if (regs.vgt_strmout_config.raw) [[unlikely]] {
            LOG_WARNING(Render_Vulkan, "Stream output unsupported, skipping");
            return false;
        }
        // GR2FORK PERF: on the EsGs path both ES and GS stages
        // are required and stage_enable already gates entry - bind_stage
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
        // GR2FORK PERF: bind_stage failures on the active LsHs path
        // are defensive recovery - well-formed PM4 has all three stages
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
        // GR2FORK PERF: the combined LS+HS+ES+GS pipeline is rare; the dominant default path is
        // plain Vertex+Pixel in the `else` branch.
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
    const auto cs_params_opt = AmdGpu::TryGetParams(cs_pgm);
    // GR2FORK: skip (rather than abort) a compute program whose pointer is
    // non-null but carries no shader binary info - GetComputePipeline turns a
    // false return into a skipped dispatch.
    if (!cs_params_opt) [[unlikely]] {
        LOG_WARNING(Render_Vulkan,
                    "Compute program @{:#x} has no shader binary info (no OrbShdr "
                    "within scan window); skipping dispatch.",
                    reinterpret_cast<uintptr_t>(cs_pgm.Address<u32*>()));
        infos[0] = nullptr;
        modules[0] = nullptr;
        compute_key.value = 0;
        fetch_shader.reset();
        return false;
    }
    const auto& cs_params = *cs_params_opt;
    // GR2FORK PERF: pointer return, as in bind_stage above. Compute stages have no fetch shader,
    // so the assign below typically resets fetch_shader.
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
                                                const AmdGpu::LiverpoolRegsSnapshot& regs,
                                                bool ctx_stable) {
    // GR2FORK PERF: per-stage resolve memo. ctx_stable covers 66.6% of GetProgram calls and the
    // memo hits on 100% of them (15.2M-call sample), skipping BuildRuntimeInfo (the member is
    // still byte-identical; proof on StageResolveMemo), its hash, and the program_cache probe.
    StageResolveMemo& memo = stage_memo_[static_cast<u32>(l_stage)];
    const bool memo_hit = ctx_stable && stage_memo_enabled_ && memo.valid &&
                          memo.params_hash == params.hash;
    // Both ternary arms reference the SAME member object - the ternary only
    // controls whether the rebuild work runs before we read it.
    const Shader::RuntimeInfo& runtime_info =
        memo_hit ? runtime_infos[static_cast<u32>(l_stage)]
                 : BuildRuntimeInfo(stage, l_stage, regs);

    Program* program;
    u64 ri_hash_raw;
    if (memo_hit) {
        program = memo.program;
        ri_hash_raw = memo.ri_hash_raw;
    } else {
        // GR2FORK PERF: probe memo before the robin_map find - hash equality alone resolves the
        // Program on this append-only map (soundness in the vk_pipeline_cache.h member doc).
        auto& pmemo = pgm_probe_memo_[static_cast<u32>(l_stage)];
        if (pmemo.program && pmemo.hash == params.hash) [[likely]] {
            program = pmemo.program;
        } else {
            auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
            // GR2FORK PERF: after early-game warmup every distinct shader
            // hash has been compiled once and lives in program_cache. New-program
            // insertion is the rare path.
            if (new_program) [[unlikely]] {
                it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
                auto& new_pgm = it_pgm.value();
                auto start = binding;
                // GR2FORK PERF: CompileModule mutates its RuntimeInfo& for tess stages
                // (TessellationPreprocess writes hs_info/vs_info), so compile against a local
                // copy; the spec built next sees the mutation, the member stays pristine.
                Shader::RuntimeInfo ri_compile = runtime_info;
                const auto module = CompileModule(new_pgm->info, ri_compile, params.code, 0, binding);
                auto spec = Shader::StageSpecialization(new_pgm->info, ri_compile, profile, start);
                const auto perm_hash = HashCombine(params.hash, 0);

                RegisterShaderMeta(new_pgm->info, spec.fetch_shader_data, spec, perm_hash, 0);
                new_pgm->AddPermut(module, std::move(spec));
                // GR2FORK PERF: populate so the next ctx-stable draw of a fresh shader already
                // hits; ri_hash_raw hashes the member (pre-mutation), what a Level-2.5 hit reuses.
                if (StageMemoEligible(stage, l_stage)) {
                    memo.code_data = params.code.data();
                    memo.params_hash = params.hash;
                    memo.ri_hash_raw = HashRuntimeInfoForStage(runtime_info);
                    memo.ri_fp_hash = 0; // reset - runtime_info was just rebuilt
                    memo.program = new_pgm.get();
                    memo.code_words = static_cast<u32>(params.code.size());
                    memo.valid = true;
                }
                // GR2FORK PERF: return a pointer into program-owned spec storage instead of
                // copying the optional<FetchShaderData> here and again in std::tie. program_cache
                // holds Programs by unique_ptr (stable address); the caller copies promptly.
                return std::make_tuple(&new_pgm->info, module,
                                       &new_pgm->modules[0].spec.fetch_shader_data, perm_hash);
            }

            program = it_pgm.value().get();
            pmemo.hash = params.hash;
            pmemo.program = program;
        }
        ri_hash_raw = HashRuntimeInfoForStage(runtime_info);
        // GR2FORK PERF: single steady-state populate site, before the tier ladder's early returns
        // so every exit leaves the memo current; on memo_hit the values are identical anyway.
        if (StageMemoEligible(stage, l_stage)) {
            memo.code_data = params.code.data();
            memo.params_hash = params.hash;
            memo.ri_hash_raw = ri_hash_raw;
            memo.ri_fp_hash = 0; // reset - runtime_info was just rebuilt
            memo.program = program;
            memo.code_words = static_cast<u32>(params.code.size());
            memo.valid = true;
        }
    }

    auto& info = program->info;
    info.pgm_base = params.Base(); // Needs to be actualized for inline cbuffer address fixup
    info.user_data = params.user_data;
    info.RefreshFlatBuf();

    // GR2FORK PERF: hash (user_data, stage-aware runtime_info, binding) and compare to cached
    // results to skip StageSpecialization construction (~2.26% of GpuComm). ud_hash_lru is gated
    // on pipelineUdHashLruEnable (default on; off if warmup flicker appears); last_result is not.
    const bool ud_hash_lru_enabled = ud_hash_lru_enabled_;
    const bool spec_fp_lru_enabled = spec_fp_lru_enabled_;
    u64 ud_hash;
    u64 ri_bind_hash;
    // GR2FORK: spec_fp is computed in the FP tier below and reused at the populate site. HS/DS
    // are excluded - their spec folds tess constant-buffer contents this key cannot cover.
    u64 spec_fp = 0;
    const bool spec_fp_eligible = spec_fp_lru_enabled &&
                                  l_stage != LogicalStage::TessellationControl &&
                                  l_stage != LogicalStage::TessellationEval;
    {
        ud_hash = XXH3_64bits(params.user_data.data(), params.user_data.size_bytes());
        // Mix in the stage-aware runtime_info hash (handles union padding + custom operator==).
        // GR2FORK PERF: ri_hash_raw is served from the memo on a hit; the binding fold below
        // stays live either way, so divergent binding state self-invalidates the tiers.
        ri_bind_hash = ri_hash_raw;
        // GR2FORK PERF: pack the three binding counters into one u64 and fold with a single mix
        // step; each counter stays far below 16 bits by shadPS4 resource constants. Saves one
        // mix-chain iteration (~5-6 cycles) per GetProgram call.
        const u64 binding_packed =
            static_cast<u64>(binding.unified) |
            (static_cast<u64>(binding.buffer) << 16) |
            (static_cast<u64>(binding.user_data) << 32);
        ri_bind_hash ^= binding_packed +
                    0x9e3779b97f4a7c15ULL + (ri_bind_hash << 6) + (ri_bind_hash >> 2);

        ud_hash ^= ri_bind_hash + 0x9e3779b97f4a7c15ULL + (ud_hash << 6) + (ud_hash >> 2);

        // GR2FORK PERF: stable user_data across consecutive calls dominates in steady state;
        // return the cached module without touching ud_hash_lru or the sig lookup.
        if (program->last_result.valid && program->last_result.ud_hash == ud_hash) [[likely]] {
            const auto perm_idx = program->last_result.perm_idx;
            if (perm_idx < program->modules.size()) {
                info.AddBindings(binding);
                // GR2FORK PERF: pointer return - see comment at the
                // new-program return site above.
                return std::make_tuple(&program->info, program->last_result.module,
                                       &program->modules[perm_idx].spec.fetch_shader_data,
                                       program->last_result.perm_hash);
            }
        }

        // GR2FORK PERF: ud_hash -> perm_idx LRU skips StageSpecialization construction (~0.69%
        // of GpuComm) for previously resolved ud_hashes; see Program::UdHashCacheEntry.
        if (ud_hash_lru_enabled) {
            const u32 slot = static_cast<u32>(ud_hash) &
                             (Program::kUdHashCacheSize - 1);
            const auto& e = program->ud_hash_lru[slot];
            // GR2FORK PERF: after a last_result miss the LRU keeps recently used permutations
            // warm (shaders rotating between a small set); post-warmup this hit dominates.
            if (e.valid && e.ud_hash == ud_hash &&
                e.perm_idx < program->modules.size()) [[likely]] {
                const size_t perm_idx = e.perm_idx;
                const auto& m = program->modules[perm_idx];
                const u64 perm_hash = HashCombine(params.hash, perm_idx);
                info.AddBindings(binding);
                // Promote into last_result so the next call hits the
                // cheaper last_result fast-path instead of running this
                // lookup again.
                program->last_result.ud_hash = ud_hash;
                program->last_result.ri_bind_hash = ri_bind_hash;
                program->last_result.perm_idx = perm_idx;
                program->last_result.perm_hash = perm_hash;
                program->last_result.module = m.module;
                program->last_result.valid = true;
                // GR2FORK PERF: pointer return - see comment at the
                // new-program return site above.
                return std::make_tuple(&program->info, m.module,
                                       &m.spec.fetch_shader_data, perm_hash);
            }
        }

        // GR2FORK PERF: address-independent spec-fingerprint tier - catches the per-draw
        // UBO-pointer churn ud_hash_lru cannot (raw user_data bytes change every draw, the spec
        // result does not). Safety and 0-collision/12.2M validation on Program::spec_fp_lru.
        if (spec_fp_eligible && !program->modules.empty()) {
            // GR2FORK PERF: reuse the memoized raw-byte RuntimeInfo hash when the member is
            // provably untouched (memo_hit); populate sites reset ri_fp_hash to 0 on rebuild,
            // so a nonzero memo value always matches the current member bytes.
            u64 ri_fp_hash;
            if (memo_hit && memo.ri_fp_hash != 0) {
                ri_fp_hash = memo.ri_fp_hash;
            } else {
                ri_fp_hash = XXH3_64bits(&runtime_info, sizeof(runtime_info));
                if (StageMemoEligible(stage, l_stage)) {
                    memo.ri_fp_hash = ri_fp_hash;
                }
            }
            spec_fp = ComputeSpecProxyFp(info, program->modules[0].spec.fetch_shader_data,
                                         ri_fp_hash, binding);
            const u32 fp_slot = static_cast<u32>(spec_fp) & (Program::kSpecFpCacheSize - 1);
            const auto& fe = program->spec_fp_lru[fp_slot];
            if (fe.valid && fe.fp == spec_fp &&
                fe.perm_idx < program->modules.size()) [[likely]] {
                const size_t perm_idx = fe.perm_idx;
                const auto& m = program->modules[perm_idx];
                const u64 perm_hash = HashCombine(params.hash, perm_idx);
                info.AddBindings(binding);
                // Promote into the cheaper tiers so an immediately-repeated
                // identical-user_data draw resolves above this point.
                if (ud_hash_lru_enabled) {
                    const u32 ud_slot =
                        static_cast<u32>(ud_hash) & (Program::kUdHashCacheSize - 1);
                    program->ud_hash_lru[ud_slot] = Program::UdHashCacheEntry{
                        .ud_hash = ud_hash,
                        .perm_idx = static_cast<u32>(perm_idx),
                        .valid = true,
                    };
                }
                program->last_result.ud_hash = ud_hash;
                program->last_result.ri_bind_hash = ri_bind_hash;
                program->last_result.perm_idx = perm_idx;
                program->last_result.perm_hash = perm_hash;
                program->last_result.module = m.module;
                program->last_result.valid = true;
                return std::make_tuple(&program->info, m.module,
                                       &m.spec.fetch_shader_data, perm_hash);
            }
        }

        // GR2FORK FIX: no ri_bind_hash-keyed "stable shortcut" is provided: user_data holds the
        // guest sharp pointers StageSpecialization codegens against, so a module built for one
        // sharp set is unsafe for another (GR2 CUSA03694: garbled effects, vertex explosions).

        // Store hashes for the cache update after spec construction + lookup.
        program->last_result.ud_hash = ud_hash;
        program->last_result.ri_bind_hash = ri_bind_hash;
    }

    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    // Fast path: look up by specialization signature.
    // We use a *pair* of signatures (sig + sig2) so we can avoid expensive deep comparisons.
    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    bool found = false;
    if (const auto it_sig = program->perm_index_by_sig.find(spec.sig);
        it_sig != program->perm_index_by_sig.end() && it_sig->second < program->modules.size()) {
        const auto& ms = program->modules[it_sig->second].spec;
        // GR2FORK PERF: the sig-map hit already matches spec.sig up to a ~2^-64 hash collision;
        // sig2, computed from the same StageSpecialization fields, confirms the match.
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
            // GR2FORK PERF: also populate ud_hash_lru so the next call
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
            // GR2FORK PERF: CompileModule may mutate its RuntimeInfo& (tess preprocess), so feed
            // it a discardable copy; the spec was built from the pre-mutation state and nothing
            // after this call reads runtime_info.
            Shader::RuntimeInfo ri_compile = runtime_info;
            module = CompileModule(new_info, ri_compile, params.code, perm_idx, binding);

            RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
            program->AddPermut(module, std::move(spec));
            // GR2FORK FIX: last_result.ud_hash was already overwritten above; leaving a stale
            // perm_idx/module beside the new ud_hash would let the next call's last_result
            // fast-path return the stale module, so set last_result fully here.
            program->last_result.perm_idx = perm_idx;
            program->last_result.perm_hash = perm_hash;
            program->last_result.module = module;
            program->last_result.valid = true;
            // GR2FORK PERF: populate ud_hash_lru so the second call with
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
            // GR2FORK PERF: also populate ud_hash_lru.
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
    // GR2FORK PERF: record spec fingerprint -> perm_idx so the next draw with structurally
    // identical (address-masked) sharps skips the StageSpecialization construct; spec_fp is
    // nonzero only when the eligible FP tier computed it (HS/DS and the disabled gate leave 0).
    if (spec_fp_eligible && spec_fp != 0 && perm_idx < program->modules.size()) {
        const u32 fp_slot = static_cast<u32>(spec_fp) & (Program::kSpecFpCacheSize - 1);
        program->spec_fp_lru[fp_slot] = Program::SpecFpCacheEntry{
            .fp = spec_fp,
            .perm_idx = static_cast<u32>(perm_idx),
            .valid = true,
        };
    }
    // GR2FORK PERF: pointer return - see comment at the
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
                // GR2FORK FIX: last_result caches the module handle by value, so the replaced
                // module would be served use-after-destroy by the fast path - invalidate it.
                // ud_hash_lru stores perm_idx and reads the module fresh, so it needs nothing.
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
        // GR2FORK FIX: the erases above can free the pipelines the last-pipeline memos point at;
        // invalidate both. This devtools path runs on the UI thread and already races GpuComm's
        // reads, so the invalidation is a strict improvement, not a new race.
        last_gfx_pipeline = nullptr;
        last_compute_pipeline_ = nullptr;
        last_compute_key_ = 0;
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
