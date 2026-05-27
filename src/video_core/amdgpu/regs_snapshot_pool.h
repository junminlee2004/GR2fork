// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// LiverpoolRegsSnapshotPool — 16-slot ring of LiverpoolRegsSnapshot.
//
// Turn 1 of the GpuComm 4-stage pipeline split (HANDOFF Section 5 + 16).
// INERT this turn — built and linked, but no call site yet.
//
// Threading model (post-Turn 2):
//   * Capture() is called from the PM4 parser thread (Stage 1) only.
//   * Release() is called from the bundle assembler / recorder thread
//     (Stage 3 in 3-stage, Stage 4 in 4-stage) only.
//   * Slot() is read-only and may be called from any consumer stage.
//   The producer/consumer asymmetry makes this an SPSC ring; head and tail
//   are atomic so the producer doesn't tear under a slow consumer.
//
// FIFO release is assumed: every stage processes intents in PM4 order, and
// every intent that captured a snapshot eventually releases it before the
// next-but-15 intent is captured. If that invariant breaks, Capture() will
// stall with kPoolFull spin (visible in profiler as a hot std::this_thread::yield).
//
// Pool size is 16 — covers the queue depth between Stage 1 and the slowest
// downstream consumer with margin for spikes (HANDOFF §4 Barrier 1).

#pragma once

#include <atomic>

#include "common/types.h"
#include "video_core/amdgpu/regs_snapshot.h"

namespace AmdGpu {

union Regs;

class LiverpoolRegsSnapshotPool {
public:
    static constexpr u32 kNumSlots = 64;
    static constexpr u32 kSlotMask = kNumSlots - 1;
    static_assert((kNumSlots & kSlotMask) == 0, "kNumSlots must be a power of two");

    // Sentinel returned by Capture() when the caller asks for a non-blocking
    // try_capture and the pool is full (currently unused — Capture() blocks).
    static constexpr u16 kInvalidSlot = 0xFFFFu;

    LiverpoolRegsSnapshotPool() noexcept;
    ~LiverpoolRegsSnapshotPool() = default;

    LiverpoolRegsSnapshotPool(const LiverpoolRegsSnapshotPool&) = delete;
    LiverpoolRegsSnapshotPool& operator=(const LiverpoolRegsSnapshotPool&) = delete;
    LiverpoolRegsSnapshotPool(LiverpoolRegsSnapshotPool&&) = delete;
    LiverpoolRegsSnapshotPool& operator=(LiverpoolRegsSnapshotPool&&) = delete;

    // Capture every audited field from `regs` plus the queue's compute
    // program (`cs_state`) into the next free slot and return its index.
    // Spins (yield) if the pool is full waiting for the consumer to call
    // Release(). Single-producer.
    //
    // Phase 1D-pre-C: also forwards five PM4-side fields that live on
    // Liverpool (not on Regs) — `gfx_pipeline_stamp`, `gfx_key_dirty`,
    // `dynamic_dirty`, `last_cb_extent[]`, `last_db_extent` — so the
    // data plane can read them from the snapshot instead of racing PM4.
    u16 Capture(const Regs& regs, const ComputeProgram& cs_state,
                u64 gfx_pipeline_stamp, bool gfx_key_dirty,
                bool dynamic_dirty,
                const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_extent,
                CbDbExtent last_db_extent) noexcept;

    // Const view of the snapshot at `idx`. Index must be one previously
    // returned by Capture() and not yet released. The returned reference is
    // valid until Release(idx) is called.
    [[nodiscard]] const LiverpoolRegsSnapshot& Slot(u16 idx) const noexcept {
        return slots_[idx & kSlotMask];
    }

    // Release the slot at `idx` back to the pool. FIFO ordering required:
    // idx must equal the current tail. Single-consumer.
    void Release(u16 idx) noexcept;

    // Diagnostic — number of slots currently in flight (head - tail).
    [[nodiscard]] u32 InFlight() const noexcept {
        const u32 h = head_.load(std::memory_order_acquire);
        const u32 t = tail_.load(std::memory_order_acquire);
        return h - t;
    }

    // Diagnostic — true if no captured slots are outstanding. Used by drains.
    [[nodiscard]] bool Empty() const noexcept {
        return InFlight() == 0;
    }

private:
    // Storage. Aligned to a cache line so adjacent slots don't false-share
    // when consumer reads slot N while producer writes slot N+1.
    alignas(64) std::array<LiverpoolRegsSnapshot, kNumSlots> slots_;

    // Producer cursor (next-to-write). Wraps past UINT32_MAX naturally.
    alignas(64) std::atomic<u32> head_;
    // Consumer cursor (next-to-release).
    alignas(64) std::atomic<u32> tail_;
};

} // namespace AmdGpu
