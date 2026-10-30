// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

struct SDL_AudioStream;

// A once-a-second trace of audio output and movie timing: per audio port, the blocks its output
// thread took from the guest, the ticks that found none, the longest gap between ticks and how
// much audio waited in the host queue; per movie, how late the game took its video frames; per AJM
// instance, the samples it decoded. Written to gr2_av_trace.tsv in the log folder.
namespace Gr2AvTrace {

void OnAudioTick(const void* backend, s32 port_type, u32 sample_rate, u32 block_frames,
                 bool block_ready);

void OnSdlQueue(const void* backend, SDL_AudioStream* stream, u32 bytes_per_frame);

void OnOpenALQueue(const void* backend, u64 queued_frames);

void OnVideoFrame(const void* source, u64 timestamp_ms, u64 clock_ms, u64 frames_left);

void OnAjmJob(u32 instance_id, u64 total_samples);

} // namespace Gr2AvTrace
