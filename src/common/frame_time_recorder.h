// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// =============================================================================
// frame_time_recorder.h — Phase 0 FPS instrumentation harness.
//
// Purpose: close the methodology gap exposed by the v1.57 rollback. Cycles%
// attribution on perf traces is only a *predictor* of FPS impact; v1.57 saved
// 1pp on cycles% but caused player-perceived frame-delivery regression that
// was invisible to perf-symbol analysis. Per the post-v1.57 handoff, Phase 0
// is the gating prerequisite — no further hot-path edits to BindResources,
// BeginRendering, ObtainBuffer, or ProcessGraphics until direct frame-time
// instrumentation exists with A/B harness support.
//
// Behavior (default-on, no env flags required):
//   * Every flip is streamed to <LogDir>/frame_times.csv next to shad_log.txt.
//     The file is opened on the first flip and overwritten per run, matching
//     the logger's own "fresh file per run" convention.
//   * fflush is called every kFlushEveryFrames flips (~2s @ 60fps), so a
//     SIGKILL or hard force-quit loses at most ~2s of trailing data.
//   * A 4096-entry in-memory ring is also kept for on-demand percentile
//     snapshots and the optional periodic console log line.
//
// Env-driven overrides (no TOML schema changes for an A/B run):
//   GR2_FPS_OUT=foo.csv         CSV filename. Absolute paths used verbatim;
//                               relative paths resolved against <LogDir>.
//                               Default: "frame_times.csv".
//   GR2_FPS_LOG_EVERY=N         Log p50/p95/p99 every N flips (default off).
//   GR2_FPS_FLUSH_EVERY=N       Override flush cadence (default 120).
//   GR2_FPS_DISABLE=1           Hard-disable both the ring write and the CSV
//                               stream (relief valve only).
//
// Acceptance gate from the handoff:
//   candidate must show <= p95 frame-time vs baseline AND no p99 regression
//   above the variance floor.
// =============================================================================

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

// Hot path. Records ns delta from the previous flip into the in-memory ring
// AND streams a single CSV line ("<index>,<delta_ns>") to the on-disk file.
// First call after Reset()/startup primes the prev-timepoint and returns
// without recording (no delta available yet).
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
