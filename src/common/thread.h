// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include "common/types.h"

namespace Common {

enum class ThreadPriority : u32 {
    Low = 0,
    Normal = 1,
    High = 2,
    VeryHigh = 3,
    Critical = 4,
};

void SetCurrentThreadRealtime(std::chrono::nanoseconds period_ns);

void SetCurrentThreadPriority(ThreadPriority new_priority);

void SetCurrentThreadName(const char* name);

void SetThreadName(void* thread, const char* name);

/// Hard-pins the calling thread to the given logical CPUs (bit N = CPU N, first 64 only).
/// Linux and Windows; a no-op returning false elsewhere.
bool SetCurrentThreadAffinityMask(u64 mask);

/// Turns the GPU command thread core reservation on (gpu_thread_core_reserve). While it is off
/// every mask below reads 0 and the walk does nothing.
void SetCoreReservationEnabled(bool enabled);

/// Both logical CPUs of the physical core reserved for the GPU command thread, or 0.
u64 GetReservedCoreMask();

/// The mask other threads must stay off, or 0 when no dedicated core could be reserved.
/// Threads apply ~mask to themselves at birth instead of waiting for the periodic walk.
u64 GetExclusionStripMask();

/// Strips the reserved core from every other thread of the process. Returns how many changed.
unsigned ExcludeReservedCoresFromAllOtherThreads();

/// Re-runs the walk every 5 s: Windows threads do not inherit their creator's affinity.
void StartPeriodicAffinityRewalk();

/// one_thread_per_core: Windows only, a no-op elsewhere. Restricts the process to the first
/// logical CPU of every physical core. Call before the emulator's threads exist.
void RestrictProcessToOneThreadPerCore();

bool AccurateSleep(std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   bool interruptible);

class AccurateTimer {
    std::chrono::nanoseconds target_interval{};
    std::chrono::nanoseconds total_wait{};

    std::chrono::high_resolution_clock::time_point start_time;

public:
    explicit AccurateTimer(std::chrono::nanoseconds target_interval);

    void Start();

    void End();

    std::chrono::nanoseconds GetTotalWait() const {
        return total_wait;
    }
};

std::string GetCurrentThreadName();

} // namespace Common
