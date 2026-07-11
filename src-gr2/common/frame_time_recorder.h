// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Frame-time recorder: every flip is streamed to <LogDir>/frame_times.csv (fresh file per run,
// flushed every kFlushEveryFrames flips) and kept in a 4096-entry ring for percentile snapshots.
// Env overrides: GR2_FPS_OUT (CSV name), GR2_FPS_LOG_EVERY, GR2_FPS_FLUSH_EVERY, GR2_FPS_DISABLE.

#include <cstddef>
#include <string>

#include "common/types.h"

namespace Common::FrameTime {

struct Stats {
    u64 count{};
    double mean_ns{};
    double p50_ns{};
    double p95_ns{};
    double p99_ns{};
    u64 min_ns{};
    u64 max_ns{};
    double fps_p50{};
};

// Hot path. Records the ns delta from the previous flip into the ring and streams one CSV line
// ("<index>,<delta_ns>"); the first call after Reset()/startup only primes the prev-timepoint.
void RecordFlip();

// Compute statistics over the most recent `window` samples, or over the full
// retained ring if `window` is 0. Touches the snapshot mutex briefly.
Stats Snapshot(std::size_t window = 0);

// Snapshot the current ring contents into an additional CSV at `path`. Useful
// for ad-hoc extraction; the always-on streaming file is the primary record.
bool DumpCsv(const std::string& path);

// Clear the ring and forget the previous timepoint. The streaming CSV is
// not reopened.
void Reset();

// True iff recording is enabled (i.e. GR2_FPS_DISABLE is not set).
bool RecordingEnabled();

} // namespace Common::FrameTime
