// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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
struct GfxStateStamp {
    u64 value{1};
    bool pending{};
    bool active{};

    // Change-detected copy: compare before copy, mark pending on change.
    // Deferred bump - the parser coalesces all reg writes between draws into
    // at most one bump, flushed at each draw/dispatch handler.
    void WriteRegs(void* dst, const void* src, size_t bytes) {
        if (!active) {
            std::memcpy(dst, src, bytes);
            return;
        }
        pending |= FusedCompareStore(dst, src, bytes);
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
    void WriteRegsImmediate(void* dst, const void* src, size_t bytes) {
        if (!active) {
            std::memcpy(dst, src, bytes);
            return;
        }
        if (FusedCompareStore(dst, src, bytes)) {
            ++value;
        }
    }

    // For bulk state resets (ClearState) where compare is pointless.
    void MarkDirty() {
        if (active) {
            pending = true;
        }
    }

    // Called at every draw/dispatch handler entry. Pending state is never
    // reset anywhere else - marks survive submit boundaries and nested
    // indirect buffers by construction.
    void FlushAtDraw() {
        if (pending) {
            ++value;
            pending = false;
        }
    }
};

} // namespace AmdGpu
