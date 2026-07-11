// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// DrawIntent - queue payload between the PM4 parser and BundleAssembler::Drain. In the
// synchronous-dispatcher build producer and consumer run on the same thread, but every field is
// filled at intent build time so the shape an async assembler would inherit is exercised.

#pragma once

#include "common/types.h"

namespace Vulkan {

struct DrawIntent {
    // ScopeMarker* strings are copied into a per-intent inline buffer: the producer truncates to
    // (kScopeMarkerInlineLen - 1) chars and null-terminates so `scope_marker.str` feeds
    // vk::DebugUtilsLabelEXT::pLabelName directly. 32 bytes keeps DrawIntent within 64 bytes.
    static constexpr u32 kScopeMarkerInlineLen = 32;

    enum class Type : u8 {
        Draw,
        DrawIndexed,
        DrawIndirect,
        DrawIndexedIndirect,
        Dispatch,
        DispatchIndirect,
        EndRendering,    // marker - drains current secondary at Stage 4
        DrainMarker,     // pre-compute drain marker
        EopWrite,        // EOP timestamp packet
        EosWait,         // EOS sync packet
        // Rasterizer scheduler-toucher markers. The synchronous dispatcher drains inline, so
        // bodies run on the producer thread; an async assembler pairs the BLOCKING ones with
        // WaitFor(packet_seq) at the producer wrapper, non-blocking ones return once enqueued.
        Flush,            // BLOCKING - captures pre-flush tick at producer
        Finish,           // BLOCKING
        CpSync,           // non-blocking
        OnSubmit,         // BLOCKING - runs cache GC + image-download flush
        FillBuffer,       // non-blocking - payload: fill_buffer
        CopyBuffer,       // non-blocking - payload: copy_buffer
        ScopeMarkerBegin, // non-blocking - payload: scope_marker
        ScopeMarkerEnd,   // non-blocking - payload: scope_marker_end
        // ScopedMarkerInsert covers both color variants: the no-color wrapper packs color=0,
        // matching a default-constructed vk::DebugUtilsLabelEXT (.color is {0,0,0,0}).
        ScopedMarkerInsert, // non-blocking - payload: scope_marker (color used)
        // Presenter scheduler-toucher marker: carries a heap-allocated type-erased closure that
        // the assembler invokes (invoke_and_destroy runs the body, then frees it). Used for the
        // draw_scheduler touch sites in vk_presenter.cpp; flip/present_scheduler stay direct.
        PresenterRecord,    // non-blocking under sync; an async wrapper adds WaitFor(seq) when it
                            // needs the body's side effects (e.g. frame->ready_tick).
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
        // EosWait - fits in the union via the EOP shape; explicit alias for clarity at call sites
        struct {
            u64 eos_compare_value;
            VAddr eos_address;
        } eos;
        // FillBuffer (DMA write)
        struct {
            VAddr address;
            u32 num_bytes;
            u32 value;
            u8 is_gds;
        } fill_buffer;
        // CopyBuffer (DMA copy)
        struct {
            VAddr dst;
            VAddr src;
            u32 num_bytes;
            u8 dst_gds;
            u8 src_gds;
        } copy_buffer;
        // ScopeMarkerBegin / ScopedMarkerInsert payload. `color` is consumed only by
        // ScopedMarkerInsert; `str` is null-terminated and truncated by the producer.
        struct {
            char str[kScopeMarkerInlineLen];
            u32 color;
            u8 from_guest;
        } scope_marker;
        // ScopeMarkerEnd carries only from_guest.
        struct {
            u8 from_guest;
        } scope_marker_end;
        // PresenterRecord: `state` points to a heap-allocated type-erased callable;
        // `invoke_and_destroy` downcasts, invokes, and deletes it. Rasterizer::PushPresenterRecord
        // synthesizes both pointers; the intent owns `state` until invoke_and_destroy runs.
        struct {
            void* state;
            void (*invoke_and_destroy)(void* state);
        } presenter_record;
    };
};

// Sizing constraint - DrawIntent must fit in 64 bytes for
// cache-friendly SPSC ring traversal.
static_assert(sizeof(DrawIntent) <= 64,
              "DrawIntent must fit in 64 bytes for cache friendliness");

} // namespace Vulkan
