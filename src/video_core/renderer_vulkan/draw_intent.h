// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// DrawIntent — Stage 1 → Stage 2 queue payload for the GpuComm 4-stage split.
// Layout matches HANDOFF_gpucomm_4stage_split Section 8 verbatim.
//
// Phase 1D-0 (Turn 2A) note: in the synchronous-dispatcher build, the
// producer (PM4 parser) and consumer (BundleAssembler::Drain) run on the
// same thread, but the intent shape is the one Turn 2B's async assembler
// will inherit. Fields the synchronous path doesn't read are still filled
// at intent build time so the shape is exercised end-to-end.

#pragma once

#include "common/types.h"

namespace Vulkan {

struct DrawIntent {
    // Phase 1D-pre-D: ScopeMarker* string is copied into a per-intent inline
    // buffer (default #2). The producer truncates to (kScopeMarkerInlineLen
    // - 1) chars and writes a null terminator at the end so the consumer
    // can hand `scope_marker.str` directly to vk::DebugUtilsLabelEXT::pLabelName.
    // 32 bytes was chosen to fit the existing 64-byte DrawIntent envelope
    // alongside the largest pre-D union member (32 B `indirect`/`dispatch`)
    // — see the static_assert at the bottom.
    static constexpr u32 kScopeMarkerInlineLen = 32;

    enum class Type : u8 {
        Draw,
        DrawIndexed,
        DrawIndirect,
        DrawIndexedIndirect,
        Dispatch,
        DispatchIndirect,
        EndRendering,    // marker — drains current secondary at Stage 4
        DrainMarker,     // pre-compute drain (HANDOFF Barrier 3)
        EopWrite,        // EOP timestamp packet
        EosWait,         // EOS sync packet
        // === Phase 1D-pre-D: Rasterizer scheduler-toucher markers. ===
        // Behavior under sync: PushAndProcess drains inline, so the body
        // runs synchronously on the producer thread the same way it did
        // before D. Under future async (Phase G) the BLOCKING ones
        // (Flush/Finish/OnSubmit) will be paired with a WaitFor(packet_seq)
        // at the producer wrapper site; non-blocking ones return as soon
        // as the intent is enqueued.
        Flush,            // BLOCKING — captures pre-flush tick at producer
        Finish,           // BLOCKING
        CpSync,           // non-blocking
        OnSubmit,         // BLOCKING — runs cache GC + image-download flush
        FillBuffer,       // non-blocking — payload: fill_buffer
        CopyBuffer,       // non-blocking — payload: copy_buffer
        ScopeMarkerBegin, // non-blocking — payload: scope_marker
        ScopeMarkerEnd,   // non-blocking — payload: scope_marker_end
        // ScopedMarkerInsert subsumes the (color) variant: the no-color
        // wrapper packs color=0 into the intent, which produces the exact
        // same vk::DebugUtilsLabelEXT (default-constructed .color is
        // {0,0,0,0}) as the legacy non-color path emitted.
        ScopedMarkerInsert, // non-blocking — payload: scope_marker (color used)
        // === Phase 1D-pre-E: Presenter scheduler-toucher marker. ===
        // Carries an opaque heap-allocated closure that captures the
        // Presenter-thread-side Frame*/attributes/etc. needed by the
        // body. The assembler invokes the closure when processing the
        // intent and the closure is responsible for self-destruction
        // (invoke_and_destroy runs the body then deletes the state).
        // Used for the three draw_scheduler touch sites in vk_presenter.cpp:
        // ~Presenter (Finish), PrepareFrame, PrepareBlankFrame(false).
        // The other Presenter scheduler touches use flip_scheduler /
        // present_scheduler which the assembler doesn't write to, so they
        // stay direct.
        PresenterRecord,    // non-blocking under sync; under future async
                            // the Presenter-side wrapper adds WaitFor(seq)
                            // before returning when it needs the body's
                            // side effects (e.g. frame->ready_tick) before
                            // proceeding.
    };

    Type type;
    u16 snapshot_idx;        // index into LiverpoolRegsSnapshotPool; valid for Draw/Dispatch
    u32 packet_seq;          // monotonic sequence number; ordering / debug

    union {
        // Draw / DrawIndexed
        struct {
            u32 num_indices;
            u32 num_instances;
            s32 vertex_offset;
            u32 instance_offset;
            u32 index_offset;       // first-index for DrawIndexed; 0 for Draw
        } draw;
        // DrawIndirect / DrawIndexedIndirect
        struct {
            VAddr address;
            u32 offset;
            u32 size;
            u32 max_count;
            u32 stride;
            VAddr count_address;
        } indirect;
        // Dispatch / DispatchIndirect
        struct {
            u32 dim_x;
            u32 dim_y;
            u32 dim_z;
            VAddr indirect_address;
            u32 indirect_offset;
            u32 indirect_size;
        } dispatch;
        // EopWrite
        struct {
            u64 eop_value;
            VAddr eop_address;
        } eop;
        // EosWait — fits in the union via the EOP shape; explicit alias for clarity at call sites
        struct {
            u64 eos_compare_value;
            VAddr eos_address;
        } eos;
        // Phase 1D-pre-D: FillBuffer (DMA write)
        struct {
            VAddr address;
            u32 num_bytes;
            u32 value;
            u8 is_gds;
        } fill_buffer;
        // Phase 1D-pre-D: CopyBuffer (DMA copy)
        struct {
            VAddr dst;
            VAddr src;
            u32 num_bytes;
            u8 dst_gds;
            u8 src_gds;
        } copy_buffer;
        // Phase 1D-pre-D: ScopeMarkerBegin / ScopedMarkerInsert payload.
        // `color` is consumed only by ScopedMarkerInsert (the legacy
        // ScopedMarkerInsert with no color packs 0 here, observable-equivalent
        // to the default vk::DebugUtilsLabelEXT). `str` is null-terminated;
        // the producer truncates to fit (kScopeMarkerInlineLen - 1) chars.
        struct {
            char str[kScopeMarkerInlineLen];
            u32 color;
            u8 from_guest;
        } scope_marker;
        // Phase 1D-pre-D: ScopeMarkerEnd carries only from_guest.
        struct {
            u8 from_guest;
        } scope_marker_end;
        // Phase 1D-pre-E: PresenterRecord. `state` is a heap pointer to a
        // type-erased callable (the producer's `new FT(closure)` block);
        // `invoke_and_destroy` is a function pointer that downcasts state
        // to FT*, invokes operator(), and `delete`s the FT*. The producer
        // template `Rasterizer::PushPresenterRecord<F>(F&&)` synthesizes
        // both pointers from the closure type. The intent owns the state
        // until invoke_and_destroy runs.
        struct {
            void* state;
            void (*invoke_and_destroy)(void* state);
        } presenter_record;
    };
};

// HANDOFF Section 8 sizing constraint — DrawIntent must fit in 64 bytes for
// cache-friendly SPSC ring traversal.
static_assert(sizeof(DrawIntent) <= 64,
              "DrawIntent must fit in 64 bytes for cache friendliness");

} // namespace Vulkan
