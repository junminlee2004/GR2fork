// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include "common/types.h"

namespace VideoCore {

// Stream-cache epoch invalidation primitive.
//
// The BufferCache stream-buffer fast path (read-only UBOs under
// CACHING_PAGESIZE) caches `(addr, size, cmdbuf, tick) → offset` entries
// with Path D ("trust within cmdbuf") for hit-time perf. Path D assumes
// the source content at `addr` hasn't been mutated since insert — true
// for rolling-allocator UBOs (most gameplay draws) but FALSE for fixed-
// binding UBOs that the game updates in-place every frame (the video
// compositor being the canonical case: YUV→RGB matrix + UV rect + frame
// index land at the same guest VAddr each video frame).
//
// External producers of in-place UBO updates bump the epoch BEFORE the
// game's update lands. Any subsequent ObtainBuffer call observes an
// epoch mismatch on hit and falls through to the miss path → fresh Copy
// → correct content. Producers that don't bump (the dominant gameplay
// shape) pay zero perf cost beyond one extra u32 compare and a relaxed
// atomic load per hit.
//
// Current bumper: sceAvPlayerGetVideoData / sceAvPlayerGetVideoDataEx
// (after a successful frame retrieval, NOT suppressed by tutorial-wins
// serialization). The epoch covers the brief init window where two
// streams are concurrent before tutorial detection kicks in.

extern std::atomic<u32> g_stream_cache_epoch;

inline void BumpStreamCacheEpoch() noexcept {
    g_stream_cache_epoch.fetch_add(1, std::memory_order_release);
}

} // namespace VideoCore
