// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>

#include "common/assert.h"
#include "common/config.h"
#include "common/logging/log.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs_snapshot_pool.h"

namespace AmdGpu {

u32 LiverpoolRegsSnapshotPool::ActiveNumSlots() noexcept {
    // GRR runs a shallower ring; every other title uses the default depth. The
    // single source of truth is the boot-time SKU detection recorded in Config.
    return Config::isGravityRushRemastered() ? kNumSlotsGrr : kNumSlotsDefault;
}

LiverpoolRegsSnapshotPool::LiverpoolRegsSnapshotPool() noexcept
    : num_slots_{ActiveNumSlots()}, slot_mask_{num_slots_ - 1}, slots_{},
      head_{0}, tail_{0} {
    // Default-construct the snapshot storage. Members are POD; zero-init is
    // fine — every slot is overwritten by CaptureFrom before it's read. Only
    // the first num_slots_ entries are ever indexed (via slot_mask_).
    LOG_INFO(Render_Vulkan, "LiverpoolRegsSnapshotPool: ring depth = {} slots ({})",
             num_slots_, num_slots_ == kNumSlotsGrr ? "GR-Remastered" : "default");
}

u16 LiverpoolRegsSnapshotPool::Capture(
    const Regs& regs, const ComputeProgram& cs_state,
    u64 gfx_pipeline_stamp, bool gfx_key_dirty, bool gfx_key_ctx_dirty, bool dynamic_dirty,
    const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_extent,
    CbDbExtent last_db_extent) noexcept {
    // Single-producer: only the PM4 parser thread calls Capture().
    // Reservation is non-atomic (relaxed read of head_) because we are the
    // only writer; the publish at the bottom uses release ordering so the
    // consumer's acquire-load on head_ is correctly synchronized.
    const u32 my_head = head_.load(std::memory_order_relaxed);

    // Wait until there's a free slot. Pool is full when head - tail == num_slots_.
    // Yielding (rather than spinning hot) is the standard SPSC backpressure
    // pattern when the consumer falls behind.
    while (my_head - tail_.load(std::memory_order_acquire) >= num_slots_) {
        std::this_thread::yield();
    }

    const u32 slot_idx = my_head & slot_mask_;
    slots_[slot_idx].CaptureFrom(regs, cs_state, gfx_pipeline_stamp,
                                  gfx_key_dirty, gfx_key_ctx_dirty, dynamic_dirty,
                                  last_cb_extent, last_db_extent);

    // Publish: release ensures the slot writes are visible to any consumer
    // that acquire-loads head_.
    head_.store(my_head + 1, std::memory_order_release);

    return static_cast<u16>(slot_idx);
}

void LiverpoolRegsSnapshotPool::Release(u16 idx) noexcept {
    // Single-consumer FIFO release. The caller must release in capture order.
    const u32 my_tail = tail_.load(std::memory_order_relaxed);
    DEBUG_ASSERT_MSG((my_tail & slot_mask_) == (idx & slot_mask_),
                     "LiverpoolRegsSnapshotPool: out-of-order Release "
                     "(expected slot {}, got {})",
                     my_tail & slot_mask_, idx & slot_mask_);
    (void)idx;

    // Release the slot. Release ordering pairs with the producer's acquire
    // on tail_ in the full-pool wait loop.
    tail_.store(my_tail + 1, std::memory_order_release);
}

} // namespace AmdGpu
