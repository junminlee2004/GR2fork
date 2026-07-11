// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <string>
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

// GR2FORK PERF: pin the calling thread to the host CPUs in `mask` (bit N = may run on CPU N; bits
// beyond the host CPU count are clipped, and a clipped mask of 0 is a no-op returning false).
// Syscall failures log at WARNING; on macOS / *BSD this returns false without effect.
bool SetCurrentThreadAffinityMask(u64 mask);

// GR2FORK PERF: physical core count (SMT not counted: 4c/8t and 4c/4t both return 4; 0 =
// unknown), from the same /sys / GetLogicalProcessorInformationEx enumeration that backs the
// core-reservation logic below and Config::isLegacyMonolithicGpuComm's 4-core gate.
unsigned GetPhysicalCoreCount();

// GR2FORK PERF: mask of host CPUs to pin a hot emulator thread onto. Guest games confine
// themselves to PS4 logical cores 0..6, so >= 8 physical cores yields CPU 7's whole physical
// core, >= 8 logical CPUs yields every host CPU outside 0..6, and smaller hosts yield 0.
u64 GetReservedCoreMask();

// GR2FORK PERF: pin mask for the GpuAssembler (BundleAssembler jthread): the next-highest
// physical core below the GpuComm pin (e.g. 8c/16t Z1 Extreme Linux 0x4040, Windows 0x3000,
// 12c/24t Ryzen 9 Linux 0x800800). Requires >= 7 physical cores; 0 on smaller hosts (floats).
u64 GetGpuAssemblerCoreMask();

// GR2FORK: cores that secondary workers (async pipeline-compile, cache-storage compression I/O)
// must avoid so they never contend with GpuComm or the GpuAssembler - normally both hot-core
// pins OR'd together. 0 in non-exclusive mode (Config::isGpuCoresNonExclusive); apply as ~mask.
u64 GetHotCoreFenceMask();

// GR2FORK PERF: bits Pthread::SetAffinity AND-strips from guest affinity masks so the reserved
// core is exclusive to GpuComm. Non-zero only when the core is stolen from the guest range (hosts
// under 8 physical cores: 6c/12t 0x820, 6c/6t 0x20, 4c/8t 0x88); the guest still sees 0..6.
u64 GetGuestExcludedCoreMask();

// GR2FORK PERF: bits OR'd into every guest affinity mask so guest threads may use otherwise-idle
// host CPUs: all host CPUs minus the PS4 logical range 0..6 minus the GpuComm core (8c/16t Z1
// Linux 0x7F00, Windows 0x3F80; 6c/12t Linux 0x780; 0 when no headroom). Guest view unchanged.
u64 GetGuestExpansionMask();

// GR2FORK PERF: back the caller's reserved-core pin into hard exclusion by AND-stripping the
// reserved bits from every other thread's affinity mask. No-ops (returns 0) when the reserved
// set is an SMT fallback pool shared with guest cores: exclusion would crowd threads onto 0..6.
unsigned ExcludeReservedCoresFromAllOtherThreads();

// GR2FORK: the mask the exclusion walk would strip, under the same gating (else 0); lets a
// thread fence itself at birth (RunThread trampoline, hang watchdog) - critical on Windows,
// where affinity is not inherited. KEEP IN SYNC with ExcludeReservedCoresFromAllOtherThreads.
u64 GetExclusionStripMask();

// GR2FORK PERF: idempotently start a detached thread re-running the exclusion walk every 5 s.
// Windows CreateThread does not inherit thread affinity, so game threads spawned after the
// initial walk land on the reserved core; on Linux this is a cheap safety net.
void StartPeriodicAffinityRewalk();

// Returns the name registered via SetCurrentThreadName / SetThreadName for an OS thread id, or
// an empty string. Process-local, independent of the OS-level thread description - a fallback
// where GetThreadDescription / pthread_getname_np returns empty (e.g. hang watchdog dumps).
std::string GetThreadNameByTid(u32 tid);

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

} // namespace Common
