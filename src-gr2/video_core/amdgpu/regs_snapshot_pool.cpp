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
    // fine - every slot is overwritten by CaptureFrom before it's read. Only
    // the first num_slots_ entries are ever indexed (via slot_mask_).
    LOG_INFO(Render_Vulkan, "LiverpoolRegsSnapshotPool: ring depth = {} slots ({})",
             num_slots_, num_slots_ == kNumSlotsGrr ? "GR-Remastered" : "default");
}

u16 LiverpoolRegsSnapshotPool::Capture(
    const Regs& regs, const ComputeProgram& cs_state,
    u64 gfx_pipeline_stamp, bool gfx_key_dirty, bool gfx_key_ctx_dirty, bool dynamic_dirty,
    const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_extent,
    CbDbExtent last_db_extent, bool capture_cs) noexcept {
    // GR2FORK PERF: blocking wrapper - yield-retry over TryCapture, behaviorally identical to a
    // plain spin (the GR2_NOTRYCAP leg). The default path in Liverpool::CaptureSnapshot uses
    // TryCapture directly so it can service host commands between retries.
    u16 idx;
    while ((idx = TryCapture(regs, cs_state, gfx_pipeline_stamp, gfx_key_dirty,
                             gfx_key_ctx_dirty, dynamic_dirty, last_cb_extent,
                             last_db_extent, capture_cs)) == kInvalidSlot) {
        // Yielding (rather than spinning hot) is the standard SPSC
        // backpressure pattern when the consumer falls behind.
        std::this_thread::yield();
    }
    return idx;
}

u16 LiverpoolRegsSnapshotPool::TryCapture(
    const Regs& regs, const ComputeProgram& cs_state,
    u64 gfx_pipeline_stamp, bool gfx_key_dirty, bool gfx_key_ctx_dirty, bool dynamic_dirty,
    const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_extent,
    CbDbExtent last_db_extent, bool capture_cs) noexcept {
    // Single-producer: only the PM4 parser thread calls this, so the head_ reservation is a
    // relaxed read; the publish at the bottom uses release ordering to pair with the consumer's
    // acquire-load on head_.
    const u32 my_head = head_.load(std::memory_order_relaxed);

    // GR2FORK PERF: full-check against a producer-private cached tail; the
    // shared consumer-written tail_ line is reloaded only when the pool looks
    // full, instead of paying a cross-core acquire load per capture.
    if (my_head - tail_cache_ >= num_slots_) {
        tail_cache_ = tail_.load(std::memory_order_acquire);
        if (my_head - tail_cache_ >= num_slots_) {
            return kInvalidSlot;
        }
    }

    const u32 slot_idx = my_head & slot_mask_;
    slots_[slot_idx].CaptureFrom(regs, cs_state, gfx_pipeline_stamp,
                                  gfx_key_dirty, gfx_key_ctx_dirty, dynamic_dirty,
                                  last_cb_extent, last_db_extent, capture_cs);

    // Publish: release ensures the slot writes are visible to any consumer
    // that acquire-loads head_.
    head_.store(my_head + 1, std::memory_order_release);

    // GR2FORK PERF: warm the next slot for ownership. The consumer just read every line of it
    // on another core, so the next capture's 39 store lines each pay a cross-core RFO;
    // write-intent prefetch overlaps those with the packet work between draws. Guarded (using
    // the stale-conservative tail above) so a slot the consumer may still read is never
    // prefetched. Pure hint.
    if (my_head + 1 - tail_cache_ < num_slots_) [[likely]] {
        const auto* next = reinterpret_cast<const char*>(&slots_[(my_head + 1) & slot_mask_]);
        for (size_t off = 0; off < sizeof(LiverpoolRegsSnapshot); off += 64) {
            __builtin_prefetch(next + off, 1, 1);
        }
    }

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
