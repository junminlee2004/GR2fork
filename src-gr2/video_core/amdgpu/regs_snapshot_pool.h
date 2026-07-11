// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// LiverpoolRegsSnapshotPool - SPSC ring for the GpuComm pipeline split: Capture() only on the
// PM4 parser thread, Release() only on the assembler/recorder thread, FIFO release assumed (a
// break shows as a hot Capture() yield spin). Depth is per-title: 64 default, 128 on GRR.

#pragma once

#include <atomic>

#include "common/types.h"
#include "video_core/amdgpu/regs_snapshot.h"

namespace AmdGpu {

union Regs;

class LiverpoolRegsSnapshotPool {
public:
    // Ring depth: GRR uses 128 slots, every other title 64, chosen once at construction from
    // boot-time title detection (Config::isGravityRushRemastered(), see ActiveNumSlots()).
    // Storage is sized to kNumSlotsStorage; a session uses only the first ActiveNumSlots() entries.
    static constexpr u32 kNumSlotsDefault = 64;
    static constexpr u32 kNumSlotsGrr = 128;

    // Storage is sized to the LARGEST active depth across configurations (see
    // slots_ below); a session indexes only the first ActiveNumSlots() entries
    // via slot_mask_. With GRR now deeper than the default this is kNumSlotsGrr.
    static constexpr u32 kNumSlotsStorage =
        kNumSlotsGrr > kNumSlotsDefault ? kNumSlotsGrr : kNumSlotsDefault;

    static_assert((kNumSlotsDefault & (kNumSlotsDefault - 1)) == 0,
                  "kNumSlotsDefault must be a power of two");
    static_assert((kNumSlotsGrr & (kNumSlotsGrr - 1)) == 0,
                  "kNumSlotsGrr must be a power of two");
    static_assert(kNumSlotsGrr <= kNumSlotsStorage && kNumSlotsDefault <= kNumSlotsStorage,
                  "snapshot storage must cover every configuration's active depth");

    // Active ring depth for this session: kNumSlotsGrr on Remastered, else
    // kNumSlotsDefault. Static so telemetry (Rasterizer::LogHwm) can report the
    // active capacity without holding a pool instance.
    [[nodiscard]] static u32 ActiveNumSlots() noexcept;

    // Sentinel returned by TryCapture() when the pool is full. Liverpool::CaptureSnapshot uses
    // TryCapture so it can service host commands between retries instead of blind-yielding
    // (the pool is the measured system throttle, hwm 64/64).
    static constexpr u16 kInvalidSlot = 0xFFFFu;

    LiverpoolRegsSnapshotPool() noexcept;
    ~LiverpoolRegsSnapshotPool() = default;

    LiverpoolRegsSnapshotPool(const LiverpoolRegsSnapshotPool&) = delete;
    LiverpoolRegsSnapshotPool& operator=(const LiverpoolRegsSnapshotPool&) = delete;
    LiverpoolRegsSnapshotPool(LiverpoolRegsSnapshotPool&&) = delete;
    LiverpoolRegsSnapshotPool& operator=(LiverpoolRegsSnapshotPool&&) = delete;

    // Capture every audited field from `regs` plus `cs_state` into the next free slot and return
    // its index; spins (yield) while full. Single-producer. Liverpool-side fields are passed in so
    // the data plane never races PM4; `capture_cs` is false for Draw intents (no cs_program read).
    u16 Capture(const Regs& regs, const ComputeProgram& cs_state,
                u64 gfx_pipeline_stamp, bool gfx_key_dirty,
                bool gfx_key_ctx_dirty, bool dynamic_dirty,
                const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_extent,
                CbDbExtent last_db_extent, bool capture_cs) noexcept;

    // Non-blocking Capture. Returns kInvalidSlot when the pool is full, checked before the 2.4 KB
    // copy so a failed try is two atomic loads. Capture is a yield-retry wrapper around this.
    u16 TryCapture(const Regs& regs, const ComputeProgram& cs_state,
                   u64 gfx_pipeline_stamp, bool gfx_key_dirty,
                   bool gfx_key_ctx_dirty, bool dynamic_dirty,
                   const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_extent,
                   CbDbExtent last_db_extent, bool capture_cs) noexcept;

    // Const view of the snapshot at `idx`. Index must be one previously
    // returned by Capture() and not yet released. The returned reference is
    // valid until Release(idx) is called.
    [[nodiscard]] const LiverpoolRegsSnapshot& Slot(u16 idx) const noexcept {
        return slots_[idx & slot_mask_];
    }

    // Release the slot at `idx` back to the pool. FIFO ordering required:
    // idx must equal the current tail. Single-consumer.
    void Release(u16 idx) noexcept;

    // Diagnostic - number of slots currently in flight (head - tail).
    [[nodiscard]] u32 InFlight() const noexcept {
        const u32 h = head_.load(std::memory_order_acquire);
        const u32 t = tail_.load(std::memory_order_acquire);
        return h - t;
    }

    // Diagnostic - true if no captured slots are outstanding. Used by drains.
    [[nodiscard]] bool Empty() const noexcept {
        return InFlight() == 0;
    }

private:
    // Active ring depth and wrap mask, chosen once at construction (see ActiveNumSlots); const so
    // the SPSC indexing math is fixed for the pool's lifetime. Declared before slots_ so the init
    // list (which derives slot_mask_ from num_slots_) runs in declaration order.
    const u32 num_slots_;
    const u32 slot_mask_;

    // Storage, sized to kNumSlotsStorage. Per-slot cache-line isolation comes from alignas(64) on
    // LiverpoolRegsSnapshot (static_assert in regs_snapshot.h); at the raw 2440 B size, 7 of 8
    // slot boundaries would share a line with the neighbor Capture overwrites during a drain.
    alignas(64) std::array<LiverpoolRegsSnapshot, kNumSlotsStorage> slots_;

    // Producer cursor (next-to-write). Wraps past UINT32_MAX naturally.
    alignas(64) std::atomic<u32> head_;
    // GR2FORK PERF: producer-private stale view of tail_ for TryCapture's
    // full check and prefetch guard; shares the producer-owned head_ line.
    u32 tail_cache_{0};
    // Consumer cursor (next-to-release).
    alignas(64) std::atomic<u32> tail_;
};

} // namespace AmdGpu
