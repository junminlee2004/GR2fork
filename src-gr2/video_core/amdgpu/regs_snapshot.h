// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// LiverpoolRegsSnapshot - per-draw subset of AmdGpu::Regs for the GpuComm pipeline split. The
// field set is the audited union of every `regs.X` access reachable from Draw / DispatchDirect /
// DispatchIndirect (67 unique fields); every member is a POD-copy-trivial value type.

#pragma once

#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"

namespace AmdGpu {

// GR2FORK PERF: alignas(64) pads each 2440 B snapshot to 2496 B. Without it consecutive pool
// slots share a cache line holding this struct's tail (the first fields every draw reads) and
// the next slot's head, which the producer overwrites while the consumer drains - false sharing.
struct alignas(64) LiverpoolRegsSnapshot {
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
    // Captured from `liverpool->mapped_queues[curr_qid].cs_state` (populated by the PM4 dispatch
    // handler before CaptureSnapshot fires) so one snapshot covers Draw and Dispatch consumers
    // alike; harmless on Draw intents, whose compute consumers never run.
    ComputeProgram cs_program;

    // === Hot-path Liverpool state ===
    // These five live outside `Regs` on Liverpool but are read by the data plane every draw;
    // live reads from the assembler thread race PM4 (a DEVICE_LOST hazard), so all are frozen at
    // CaptureSnapshot() time, which also reads-and-clears the PM4-owned dirty bits.
    u64 gfx_pipeline_stamp;
    bool gfx_key_dirty;
    // GR2FORK: a key-affecting CONTEXT register changed since the previous
    // capture (vs a user_data-only SH re-emit). See Liverpool::gfx_key_ctx_dirty_.
    bool gfx_key_ctx_dirty;
    bool dynamic_dirty;
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent;
    CbDbExtent last_db_extent;

    // Captures every audited field from a live Regs, the compute program from the queue's
    // cs_state, and the five PM4-side Liverpool fields (passed in by Liverpool::CaptureSnapshot);
    // kept a member function so the struct and its capture path are auditable as one unit.
    // GR2FORK PERF: `capture_cs` skips the 320 B ComputeProgram copy (13% of the snapshot) on
    // Draw intents - every cs_program consumer runs only on dispatch paths, so the stale bytes
    // are provably never read. Kill switch: GR2_NOCSSKIP=1 (checked in BundleAssembler::Push).
    void CaptureFrom(const Regs& regs, const ComputeProgram& cs_state,
                     u64 gfx_pipeline_stamp_in, bool gfx_key_dirty_in,
                     bool gfx_key_ctx_dirty_in, bool dynamic_dirty_in,
                     const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_in,
                     CbDbExtent last_db_in, bool capture_cs) noexcept {
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

        // Compute program (separate state, owned by the queue's cs_state slot,
        // not by Regs). Skipped for Draw intents - see the capture_cs note.
        if (capture_cs) {
            cs_program = cs_state;
        }

        // Hot-path Liverpool state, passed in by the producer from the live Liverpool object;
        // `gfx_key_dirty_in` / `dynamic_dirty_in` are the values BEFORE Liverpool::CaptureSnapshot
        // clears them, so the snapshot reflects what a live read would have observed.
        gfx_pipeline_stamp = gfx_pipeline_stamp_in;
        gfx_key_dirty = gfx_key_dirty_in;
        gfx_key_ctx_dirty = gfx_key_ctx_dirty_in;
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

// GR2FORK PERF: the pool's slot array derives per-slot cache-line isolation from this; alignas
// keeps the invariant if a future field grows the struct, and the assert documents that the
// invariant is load-bearing.
static_assert(alignof(LiverpoolRegsSnapshot) == 64 &&
                  sizeof(LiverpoolRegsSnapshot) % 64 == 0,
              "snapshot slots must be cache-line aligned (see T1.1)");

} // namespace AmdGpu
