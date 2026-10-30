// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>

#include "common/types.h"

struct SDL_AudioStream;

// A once-a-second trace of audio output and movie timing: per audio port, the blocks its output
// thread took from the guest, the ticks that found none, the longest gap between ticks split into
// timer sleep, port lock wait, output call and backend pacing sleep, and how much audio waited in
// the host queue; for the guest, how long sceAudioOutOutput(s) took to lock and wait on the ports;
// two canary threads (normal and time-critical priority) that report how late a 1 ms sleep woke;
// frame submit and present gaps; per movie, how late the game took its video frames; per AJM
// instance, the samples it decoded. Each longest time is written as ms@seconds since the trace
// started. Written to gr2_av_trace.tsv in the log folder.
namespace Gr2AvTrace {

// Output thread, before it locks the port.
void OnTickBegin(const void* backend);

// Output thread, with the port locked.
void OnAudioTick(const void* backend, s32 port_type, u32 sample_rate, u32 block_frames,
                 bool block_ready);

// Output thread, after the output call, with the port still locked.
void OnTickEnd(const void* backend);

void OnSdlQueue(const void* backend, SDL_AudioStream* stream, u32 bytes_per_frame);

void OnOpenALQueue(const void* backend, u64 queued_frames);

// Records one backend pacing sleep: the requested length now, the actual length when destroyed.
class BackendSleep {
public:
    BackendSleep(const void* backend, u64 requested_us);
    ~BackendSleep();

private:
    const void* backend;
    u64 requested_us;
    std::chrono::steady_clock::time_point begin;
};

// Guest thread in sceAudioOutOutput(s): before it locks the ports, once it holds them, and once
// every port is free.
void OnGuestBegin();
void OnGuestLocked();
void OnGuestWaited();

enum class FlipStage { Submit, Present };

void OnFlip(FlipStage stage);

void OnVideoFrame(const void* source, u64 timestamp_ms, u64 clock_ms, u64 frames_left);

void OnAjmJob(u32 instance_id, u64 total_samples);

} // namespace Gr2AvTrace
