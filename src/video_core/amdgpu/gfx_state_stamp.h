// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include "common/types.h"

namespace AmdGpu {

// Monotonic stamp over the graphics register file, bumped only when a covered
// register actually changed value. GPU-parser-thread confined (the graphics
// and compute queues are coroutines on one thread); no atomics by design.
//
// Coverage exclusions are contracts, not accidents: cs_state (compute program
// blocks), the draw-packet-embedded index/instance words written directly by
// draw handlers, and the per-stage user_data words when no consumer needs
// them. Any future cache reading an excluded word must extend coverage in the
// same change.
//
// Dormant (active == false) the funnel degrades to a plain memcpy - zero
// compares, zero bumps - so a disabled framework costs nothing here.
//
// Two lanes: value moves on any covered change; dyn_value moves only when a
// changed block touches a word of dyn_mask, the read set of the dynamic-state
// updaters in vk_rasterizer.cpp. Every bump site feeds both lanes; config
// register writes and the draw-packet words carry no mask and move neither.
// The mask is only as complete as that read set: an updater that starts
// reading a new register extends the block table in liverpool.cpp.
struct GfxStateStamp {
    u64 value{1};
    u64 dyn_value{1};
    bool pending{};
    bool pending_dyn{};
    bool active{};
    bool classify{};
    const u64* dyn_mask{};
    u32 dyn_mask_base{};
    u32 dyn_mask_words{};
    // Funnel census, counted only while active: covered writes and the
    // classifications the pending flag did not already answer.
    u64 funnel_calls{};
    u64 funnel_classified{};

    // True when any word of the block [word, word + n) is in the dyn mask,
    // tested a mask word at a time.
    bool Touches(u32 word, u32 n) const noexcept {
        if (n == 0 || word < dyn_mask_base) {
            return false;
        }
        const u32 first = word - dyn_mask_base;
        if (first >= dyn_mask_words) {
            return false;
        }
        const u32 end = std::min(first + n, dyn_mask_words);
        const u32 last = end - 1;
        const u32 c0 = first >> 6;
        const u32 c1 = last >> 6;
        const u64 lo_mask = ~u64{0} << (first & 63);
        const u64 hi_mask = ~u64{0} >> (63 - (last & 63));
        if (c0 == c1) {
            return (dyn_mask[c0] & lo_mask & hi_mask) != 0;
        }
        if ((dyn_mask[c0] & lo_mask) != 0) {
            return true;
        }
        for (u32 c = c0 + 1; c < c1; ++c) {
            if (dyn_mask[c] != 0) {
                return true;
            }
        }
        return (dyn_mask[c1] & hi_mask) != 0;
    }

    // Change-detected copy: compare before copy, mark pending on change.
    // Deferred bump - the parser coalesces all reg writes between draws into
    // at most one bump, flushed at each draw/dispatch handler.
    void WriteRegs(void* dst, const void* src, size_t bytes, u32 word) {
        if (!active) {
            std::memcpy(dst, src, bytes);
            return;
        }
        ++funnel_calls;
        if (FusedCompareStore(dst, src, bytes)) {
            pending = true;
            // The lane flag only rises between draws, so once it is set the
            // remaining writes of the interval need no classification.
            if (classify && !pending_dyn) {
                ++funnel_classified;
                if (Touches(word, static_cast<u32>(bytes / sizeof(u32)))) {
                    pending_dyn = true;
                }
            }
        }
    }

    /// One fused compare-and-store pass for the small register blocks this
    /// path sees, instead of a libc memcmp plus memcpy call pair. Returns
    /// exactly memcmp's changed verdict: any differing byte.
    static bool FusedCompareStore(void* dst, const void* src, size_t bytes) {
        if (bytes <= 64 && (bytes & 3) == 0) {
            u8* d = static_cast<u8*>(dst);
            const u8* c = static_cast<const u8*>(src);
            bool changed = false;
            size_t i = 0;
            for (; i + 8 <= bytes; i += 8) {
                u64 a, b;
                std::memcpy(&a, d + i, 8);
                std::memcpy(&b, c + i, 8);
                if (a != b) {
                    changed = true;
                    std::memcpy(d + i, c + i, 8);
                }
            }
            for (; i < bytes; i += 4) {
                u32 a, b;
                std::memcpy(&a, d + i, 4);
                std::memcpy(&b, c + i, 4);
                if (a != b) {
                    changed = true;
                    std::memcpy(d + i, c + i, 4);
                }
            }
            return changed;
        }
        if (std::memcmp(dst, src, bytes) != 0) {
            std::memcpy(dst, src, bytes);
            return true;
        }
        return false;
    }

    // Immediate change-detected write for the async-compute queue's writes
    // into the shared gfx range: the ACB has no draw boundary to defer to.
    void WriteRegsImmediate(void* dst, const void* src, size_t bytes, u32 word) {
        if (!active) {
            std::memcpy(dst, src, bytes);
            return;
        }
        if (FusedCompareStore(dst, src, bytes)) {
            ++value;
            if (classify && Touches(word, static_cast<u32>(bytes / sizeof(u32)))) {
                ++dyn_value;
            }
        }
    }

    // For bulk state resets (ClearState) where compare is pointless.
    void MarkDirty() {
        if (active) {
            pending = true;
            pending_dyn = true;
        }
    }

    // Called at every draw/dispatch handler entry. Pending state is never
    // reset anywhere else - marks survive submit boundaries and nested
    // indirect buffers by construction.
    void FlushAtDraw() {
        if (pending) {
            ++value;
            pending = false;
            if (pending_dyn) {
                ++dyn_value;
                pending_dyn = false;
            }
        }
    }
};

} // namespace AmdGpu
