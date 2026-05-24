// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// LiverpoolRegsSnapshot — versioned per-draw subset of AmdGpu::Regs.
//
// Turn 1 of the GpuComm 4-stage pipeline split (HANDOFF_gpucomm_4stage_split,
// Section 5 Turn 1 + Section 16 Quick-start). This header is INERT — the
// snapshot type and pool exist but are not yet wired into the draw path.
//
// Field set is the audited union of every `regs.X` access reachable from
// Draw / DispatchDirect / DispatchIndirect in:
//   - video_core/renderer_vulkan/vk_rasterizer.cpp
//   - video_core/renderer_vulkan/vk_pipeline_cache.cpp
//   - video_core/renderer_vulkan/vk_pipeline_common.cpp
//   - video_core/renderer_vulkan/vk_graphics_pipeline.cpp
//   - video_core/renderer_vulkan/vk_compute_pipeline.cpp
//   - video_core/buffer_cache/
//   - video_core/texture_cache/
// 67 unique fields. See zip-level audit confirmation.
//
// Snapshot is POD-copy-trivial: every member is a value type defined in
// regs_color.h / regs_depth.h / regs_primitive.h / regs_shader.h /
// regs_texture.h / regs_vertex.h, all aggregate-assignable.

#pragma once

#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"

namespace AmdGpu {

struct LiverpoolRegsSnapshot {
    // === Shader programs (graphics) ===
    ShaderProgram ps_program;
    ShaderProgram vs_program;
    ShaderProgram gs_program;
    ShaderProgram es_program;
    ShaderProgram hs_program;
    ShaderProgram ls_program;

    // === Depth / stencil / clear ===
    DepthRenderControl depth_render_control;
    DepthView depth_view;
    DepthRenderOverride depth_render_override;
    Address depth_htile_data_base;
    float depth_bounds_min;
    float depth_bounds_max;
    u32 stencil_clear;
    float depth_clear;
    Scissor screen_scissor;
    DepthBuffer depth_buffer;
    DepthControl depth_control;
    DepthShaderControl depth_shader_control;
    StencilControl stencil_control;
    StencilRefMask stencil_ref_front;
    StencilRefMask stencil_ref_back;

    // === Texture sampler border colors ===
    BorderColorBuffer ta_bc_base;

    // === Window / scissor ===
    WindowOffset window_offset;
    ViewportScissor window_scissor;
    ViewportScissor generic_scissor;
    std::array<ViewportScissor, NUM_VIEWPORTS> viewport_scissors;
    std::array<ViewportDepth, NUM_VIEWPORTS> viewport_depths;
    std::array<ViewportBounds, NUM_VIEWPORTS> viewports;

    // === Color buffers / blend ===
    ColorBuffer color_buffers[NUM_COLOR_BUFFERS];
    ColorBufferMask color_target_mask;
    ColorBufferMask color_shader_mask;
    ColorControl color_control;
    ColorExportFormat color_export_format;
    BlendConstants blend_constants;
    std::array<BlendControl, NUM_COLOR_BUFFERS> blend_control;

    // === Pixel-shader inputs ===
    std::array<PsInputControl, 32> ps_inputs;
    PsInput ps_input_ena;
    PsInput ps_input_addr;
    u32 num_interp : 6;

    // === Shader export formats ===
    ShaderExportFormat z_export_format;

    // === Index buffer ===
    IndexBufferBase index_base_address;
    IndexBufferType index_buffer_type;
    u32 index_offset;
    u32 num_indices;
    VgtNumInstances num_instances;
    u32 primitive_restart_index;
    u32 enable_primitive_restart;

    // === Primitive / clip / polygon / line ===
    PrimitiveType primitive_type;
    ClipperControl clipper_control;
    PolygonControl polygon_control;
    PolygonOffset poly_offset;
    LineControl line_control;
    ModeControl mode_control;
    ViewportControl viewport_control;
    VsOutputControl vs_output_control;

    // === Geometry / tessellation / streamout ===
    GsMode vgt_gs_mode;
    GsOutPrimitiveType vgt_gs_out_prim_type;
    u32 vgt_esgs_ring_itemsize;
    u32 vgt_gs_max_vert_out : 11;
    GsInstances vgt_gs_instance_cnt;
    std::array<u32, 4> vgt_gs_vert_itemsize;
    ShaderStageEnable stage_enable;
    LsHsConfig ls_hs_config;
    TessellationConfig tess_config;
    StreamOutConfig vgt_strmout_config;

    // === Vertex instancing ===
    u32 vgt_instance_step_rate_0;
    u32 vgt_instance_step_rate_1;

    // === Compute program ===
    // Phase 1D-0b (Turn 2B-1): captured alongside the graphics regs so a
    // single intent's snapshot covers both Draw and Dispatch consumers. The
    // value source is `liverpool->mapped_queues[curr_qid].cs_state` —
    // already populated by the PM4 dispatch packet handler before
    // CaptureSnapshot fires, so this captures what the dispatch is about
    // to consume. Harmless on Draw intents (compute consumers don't run).
    ComputeProgram cs_program;

    // === Phase 1D-pre-C: hot-path Liverpool state ===
    // These fields live OUTSIDE `Regs` proper on the live Liverpool but are
    // read by the data plane on every draw / pipeline lookup. v1's hotfix1
    // crash (DEVICE_LOST) was caused by the assembler thread doing plain
    // bool reads of `liverpool->IsGfxKeyDirty()` / `IsDynamicDirty()` and
    // a relaxed atomic load of `gfx_pipeline_stamp` while PM4 was racing
    // them — the cache returned a pipeline for state PM4 had just dirtied.
    // Phase C captures all five at PM4-side `CaptureSnapshot()` time so
    // every data-plane consumer reads from the snapshot's frozen view.
    //
    // Capture-and-clear semantics for the dirty bits: PM4 is the sole
    // writer of `gfx_key_dirty_` / `dynamic_dirty_` on Liverpool, and
    // `Liverpool::CaptureSnapshot()` reads-and-clears them under that
    // sole-writer ownership. After capture, the live Liverpool flags are
    // false; the next PM4 register write that warrants it will set them.
    // The data plane reads `regs.gfx_key_dirty` / `regs.dynamic_dirty`
    // here — a frozen snapshot of "was this dirty AT capture?" — which is
    // exactly what the pipeline-cache and dynamic-state-skip predicates
    // need.
    //
    // `gfx_pipeline_stamp` is a relaxed-atomic read at capture; PM4
    // bumps it via `Liverpool::BumpGfxPipelineStamp` and consumers compare
    // their cached stamp against `regs.gfx_pipeline_stamp` to decide if
    // their cached lookup result is still valid.
    //
    // `last_cb_extent[]` and `last_db_extent` are PM4-written extent
    // hints (from CB/DB setup NOP packets) that BeginRendering/Resolve/
    // EliminateFastClear consume to size the texture-cache image lookup.
    u64 gfx_pipeline_stamp;
    bool gfx_key_dirty;
    bool dynamic_dirty;
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent;
    CbDbExtent last_db_extent;

    // === API: graphics capture — every audited field a Draw consumer reads. ===
    // Intentionally a member function (not a free function) so the captured
    // field set is co-located with the struct definition. Adding/removing a
    // field touches both the struct and CaptureGfxFrom in one place.
    //
    // Inline by design: the snapshot strategy is header-only on the capture
    // side. The pool's .cpp keeps the storage; the per-field copy lives here
    // so the struct definition and its capture path are auditable as one unit.
    //
    // Tier-1 split (graphics/compute snapshots): this graphics path deliberately
    // OMITS `cs_program`. Audit (see CaptureComputeFrom below) confirms the Draw
    // / DrawIndexed / DrawIndirect / DrawIndexedIndirect consumer chain never
    // reads `cs_program` — the only reads are compute-gated (RefreshComputeKey,
    // BuildRuntimeInfo's Stage::Compute branch, the IsCompute* meta helpers, and
    // BindBuffers's SharedMemory/LDS branch, which is marked "graphics draws
    // never reach this branch"). Skipping the 320-byte ComputeProgram copy here
    // is the per-draw write saving — and since draws never read cs_program, the
    // slot's stale cs_program bytes are inert on the graphics path.
    //
    // Phase 1D-pre-C: also captures `gfx_pipeline_stamp`, `gfx_key_dirty`,
    // `dynamic_dirty`, `last_cb_extent[]`, and `last_db_extent` — five
    // PM4-side fields that live on Liverpool (not on Regs). The producer
    // (Liverpool::CaptureGfxSnapshot) reads-and-clears the two dirty bits
    // outside this method as part of its sole-writer-ownership protocol;
    // here we just copy the values it passes in.
    void CaptureGfxFrom(const Regs& regs,
                        u64 gfx_pipeline_stamp_in, bool gfx_key_dirty_in,
                        bool dynamic_dirty_in,
                        const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_in,
                        CbDbExtent last_db_in) noexcept {
        // Shader programs
        ps_program = regs.ps_program;
        vs_program = regs.vs_program;
        gs_program = regs.gs_program;
        es_program = regs.es_program;
        hs_program = regs.hs_program;
        ls_program = regs.ls_program;

        // Depth / stencil / clear
        depth_render_control = regs.depth_render_control;
        depth_view = regs.depth_view;
        depth_render_override = regs.depth_render_override;
        depth_htile_data_base = regs.depth_htile_data_base;
        depth_bounds_min = regs.depth_bounds_min;
        depth_bounds_max = regs.depth_bounds_max;
        stencil_clear = regs.stencil_clear;
        depth_clear = regs.depth_clear;
        screen_scissor = regs.screen_scissor;
        depth_buffer = regs.depth_buffer;
        depth_control = regs.depth_control;
        depth_shader_control = regs.depth_shader_control;
        stencil_control = regs.stencil_control;
        stencil_ref_front = regs.stencil_ref_front;
        stencil_ref_back = regs.stencil_ref_back;

        // Texture sampler border colors
        ta_bc_base = regs.ta_bc_base;

        // Window / scissor / viewports
        window_offset = regs.window_offset;
        window_scissor = regs.window_scissor;
        generic_scissor = regs.generic_scissor;
        viewport_scissors = regs.viewport_scissors;
        viewport_depths = regs.viewport_depths;
        viewports = regs.viewports;

        // Color buffers / blend
        for (u32 i = 0; i < NUM_COLOR_BUFFERS; ++i) {
            color_buffers[i] = regs.color_buffers[i];
        }
        color_target_mask = regs.color_target_mask;
        color_shader_mask = regs.color_shader_mask;
        color_control = regs.color_control;
        color_export_format = regs.color_export_format;
        blend_constants = regs.blend_constants;
        blend_control = regs.blend_control;

        // PS inputs
        ps_inputs = regs.ps_inputs;
        ps_input_ena = regs.ps_input_ena;
        ps_input_addr = regs.ps_input_addr;
        num_interp = regs.num_interp;

        // Shader export formats
        z_export_format = regs.z_export_format;

        // Index buffer / draw
        index_base_address = regs.index_base_address;
        index_buffer_type = regs.index_buffer_type;
        index_offset = regs.index_offset;
        num_indices = regs.num_indices;
        num_instances = regs.num_instances;
        primitive_restart_index = regs.primitive_restart_index;
        enable_primitive_restart = regs.enable_primitive_restart;

        // Primitive / clip / polygon / line / mode / viewport ctl
        primitive_type = regs.primitive_type;
        clipper_control = regs.clipper_control;
        polygon_control = regs.polygon_control;
        poly_offset = regs.poly_offset;
        line_control = regs.line_control;
        mode_control = regs.mode_control;
        viewport_control = regs.viewport_control;
        vs_output_control = regs.vs_output_control;

        // Geometry / tessellation / streamout
        vgt_gs_mode = regs.vgt_gs_mode;
        vgt_gs_out_prim_type = regs.vgt_gs_out_prim_type;
        vgt_esgs_ring_itemsize = regs.vgt_esgs_ring_itemsize;
        vgt_gs_max_vert_out = regs.vgt_gs_max_vert_out;
        vgt_gs_instance_cnt = regs.vgt_gs_instance_cnt;
        for (u32 i = 0; i < 4; ++i) {
            vgt_gs_vert_itemsize[i] = regs.vgt_gs_vert_itemsize[i];
        }
        stage_enable = regs.stage_enable;
        ls_hs_config = regs.ls_hs_config;
        tess_config = regs.tess_config;
        vgt_strmout_config = regs.vgt_strmout_config;

        // Vertex instancing
        vgt_instance_step_rate_0 = regs.vgt_instance_step_rate_0;
        vgt_instance_step_rate_1 = regs.vgt_instance_step_rate_1;

        // Tier-1 split: cs_program is NOT captured on the graphics path.
        // Whatever stale ComputeProgram bytes occupy this slot are inert here
        // because no Draw consumer reads cs_program (see method doc above).

        // Phase 1D-pre-C: hot-path Liverpool state. These come from the
        // Liverpool object's own members (not from Regs). The producer
        // reads them from the live Liverpool and passes them in as args;
        // `gfx_key_dirty_in` / `dynamic_dirty_in` are the values BEFORE
        // Liverpool::CaptureGfxSnapshot clears them (so the snapshot reflects
        // "what the assembler would have observed had it read live").
        gfx_pipeline_stamp = gfx_pipeline_stamp_in;
        gfx_key_dirty = gfx_key_dirty_in;
        dynamic_dirty = dynamic_dirty_in;
        last_cb_extent = last_cb_in;
        last_db_extent = last_db_in;
    }

    // === API: compute capture — the minimal audited set a Dispatch reads. ===
    //
    // Tier-1 split: a Dispatch / DispatchIndirect consumer touches only a tiny
    // subset of the snapshot. The full audited compute read-set is:
    //
    //   * `cs_program`               — DoDispatch*FromIntent (dims), GetComputePipeline
    //                                  → RefreshComputeKey → GetProgram →
    //                                  BuildRuntimeInfo(Stage::Compute), the three
    //                                  IsCompute* meta helpers, and BindBuffers's
    //                                  SharedMemory/LDS branch.
    //   * `viewport_control`         — MakeUserData (runs inside BindResources on the
    //   * `viewports[0]`               dispatch path too); the four push-constant
    //                                  viewport floats are derived from these. A
    //                                  compute shader's push range generally excludes
    //                                  them, but capturing the two fields keeps
    //                                  push_data BIT-IDENTICAL to the pre-split path,
    //                                  so this is correctness-preserving, not a guess.
    //
    // Everything else is zero-initialized via `*this = {}` FIRST. This is the
    // deliberate safety property of the split: if a future code change adds a
    // `regs.<field>` read on the compute path that this audit didn't cover, it
    // reads a DETERMINISTIC ZERO rather than stale ring-slot garbage — a clean,
    // reproducible wrong value instead of nondeterministic corruption. The memset
    // is a streaming store (no source read) and is cheaper than the old full
    // per-field copy from live Regs; the consumer also reads back far fewer lines.
    //
    // NOTE: this does not shrink sizeof(LiverpoolRegsSnapshot) (the type is shared
    // with the graphics pool so consumer signatures are unchanged). The compute
    // win is reduced write traffic + smaller consumer read-set, not a smaller
    // struct. A dedicated compact compute-snapshot type would cut the 156 KB pool
    // footprint further but requires threading a new type through the pipeline
    // cache + bind path — deferred as a follow-up.
    void CaptureComputeFrom(const Regs& regs, const ComputeProgram& cs_state,
                            u64 gfx_pipeline_stamp_in, bool gfx_key_dirty_in,
                            bool dynamic_dirty_in,
                            const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_in,
                            CbDbExtent last_db_in) noexcept {
        // Zero the whole slot first so every un-captured field is a determinate 0.
        *this = LiverpoolRegsSnapshot{};

        // Compute program — the primary payload of a dispatch snapshot.
        cs_program = cs_state;

        // MakeUserData reads these on the dispatch path; capture verbatim so the
        // derived push-constant viewport transform is unchanged from pre-split.
        viewport_control = regs.viewport_control;
        viewports[0] = regs.viewports[0];

        // Phase 1D-pre-C hot-path Liverpool state (cheap; preserved for parity
        // with the graphics snapshot's contract even though the compute pipeline
        // cache path keys off cs_program rather than the gfx pipeline stamp).
        gfx_pipeline_stamp = gfx_pipeline_stamp_in;
        gfx_key_dirty = gfx_key_dirty_in;
        dynamic_dirty = dynamic_dirty_in;
        last_cb_extent = last_cb_in;
        last_db_extent = last_db_in;
    }

    // === API: accessor methods mirroring AmdGpu::Regs. ===
    // Snapshot is meant to drop into call sites that today read `regs.X`,
    // including those that call `regs.IsClipDisabled()` / `regs.ProgramForStage(i)`.
    [[nodiscard]] bool IsClipDisabled() const noexcept {
        return clipper_control.clip_disable || primitive_type == PrimitiveType::RectList;
    }

    [[nodiscard]] const ShaderProgram* ProgramForStage(u32 index) const noexcept {
        switch (index) {
        case 0:
            return &ps_program;
        case 1:
            return &vs_program;
        case 2:
            return &gs_program;
        case 3:
            return &es_program;
        case 4:
            return &hs_program;
        case 5:
            return &ls_program;
        }
        return nullptr;
    }
};

} // namespace AmdGpu
