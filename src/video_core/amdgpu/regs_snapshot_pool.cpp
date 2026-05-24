// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>

#include "common/assert.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs_snapshot_pool.h"

namespace AmdGpu {

LiverpoolRegsSnapshotPool::LiverpoolRegsSnapshotPool() noexcept
    : slots_{}, head_{0}, tail_{0} {
    // Default-construct the snapshot storage. Members are POD; zero-init is
    // fine — every slot is overwritten by CaptureFrom before it's read.
}

u32 LiverpoolRegsSnapshotPool::ReserveHead() noexcept {
    // Single-producer: only the PM4 parser thread reserves. Relaxed read of
    // head_ is safe because we are the sole writer; the publish (release) in
    // PublishHead synchronizes the slot writes with the consumer's acquire.
    const u32 my_head = head_.load(std::memory_order_relaxed);

    // Wait until there's a free slot. Pool is full when head - tail == kNumSlots.
    // Yielding (rather than spinning hot) is the standard SPSC backpressure
    // pattern when the consumer falls behind.
    while (my_head - tail_.load(std::memory_order_acquire) >= kNumSlots) {
        std::this_thread::yield();
    }
    return my_head;
}

void LiverpoolRegsSnapshotPool::PublishHead(u32 my_head) noexcept {
    // Publish: release ensures the slot writes are visible to any consumer
    // that acquire-loads head_.
    head_.store(my_head + 1, std::memory_order_release);
}

u16 LiverpoolRegsSnapshotPool::CaptureGfx(
    const Regs& regs, u64 gfx_pipeline_stamp, bool gfx_key_dirty,
    bool dynamic_dirty,
    const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_extent,
    CbDbExtent last_db_extent) noexcept {
    const u32 my_head = ReserveHead();
    const u32 slot_idx = my_head & kSlotMask;
    slots_[slot_idx].CaptureGfxFrom(regs, gfx_pipeline_stamp, gfx_key_dirty,
                                    dynamic_dirty, last_cb_extent, last_db_extent);
    PublishHead(my_head);
    return static_cast<u16>(slot_idx);
}

u16 LiverpoolRegsSnapshotPool::CaptureCompute(
    const Regs& regs, const ComputeProgram& cs_state,
    u64 gfx_pipeline_stamp, bool gfx_key_dirty, bool dynamic_dirty,
    const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_extent,
    CbDbExtent last_db_extent) noexcept {
    const u32 my_head = ReserveHead();
    const u32 slot_idx = my_head & kSlotMask;
    slots_[slot_idx].CaptureComputeFrom(regs, cs_state, gfx_pipeline_stamp,
                                        gfx_key_dirty, dynamic_dirty,
                                        last_cb_extent, last_db_extent);
    PublishHead(my_head);
    return static_cast<u16>(slot_idx);
}

void LiverpoolRegsSnapshotPool::Release(u16 idx) noexcept {
    // Single-consumer FIFO release. The caller must release in capture order.
    const u32 my_tail = tail_.load(std::memory_order_relaxed);
    DEBUG_ASSERT_MSG((my_tail & kSlotMask) == (idx & kSlotMask),
                     "LiverpoolRegsSnapshotPool: out-of-order Release "
                     "(expected slot {}, got {})",
                     my_tail & kSlotMask, idx & kSlotMask);
    (void)idx;

    // Release the slot. Release ordering pairs with the producer's acquire
    // on tail_ in the full-pool wait loop.
    tail_.store(my_tail + 1, std::memory_order_release);
}

} // namespace AmdGpu
