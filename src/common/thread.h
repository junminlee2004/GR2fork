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

// PERF(GR2FORK): pin the calling thread to a set of host CPUs.
//
// `mask` is interpreted as a bitfield where bit N == "thread may run on host
// CPU N". Bits beyond the host CPU count are silently clipped. A mask of 0
// after clipping is a no-op (returns false; no system call made).
//
// Returns true on success, false on failure or no-op. Failures (syscall
// errors) are logged at WARNING. On macOS / *BSD this currently returns
// false without effect — Darwin's THREAD_AFFINITY_POLICY is advisory-only
// and FreeBSD's cpuset_setaffinity is out of scope for the initial GR2FORK
// affinity work.
bool SetCurrentThreadAffinityMask(u64 mask);

// PERF(GR2FORK): mask of host CPUs to pin a hot emulator thread onto.
//
// PS4 firmware reserves logical core 7 of the Jaguar 8-core for OS use, and
// well-behaved guest games confine their threads to logical cores 0..6 via
// scePthreadSetaffinity. Now that Pthread::SetAffinity actually honors
// those masks, host CPU 7 (and possibly the rest of the host's logical
// CPUs above 6) sit unclaimed by guest work — but emulator-internal threads
// (GpuComm, Present, AjmWork, ...) still float onto whichever host CPUs the
// OS scheduler picks, which on a fully-loaded handheld means SMT siblings of
// overloaded guest cores 0..6. That's the structural source of the "GpuComm
// shares a physical core with JoBwOrKeR" observation.
//
// The mask returned depends on host topology, dynamically discovered via
// /sys (Linux) / GetLogicalProcessorInformationEx (Windows):
//
//   * Host >= 8 *physical* cores (e.g. 8c/16t Z1 Extreme, 8c/16t Ryzen 7,
//     12c/24t Ryzen 9, 16c/32t Threadripper, 8c/8t with SMT off):
//     CPU 7 is its own physical core with no overlap into the guest range.
//     Returns {CPU 7} OR'd with its SMT siblings — a tight pin to a single
//     genuinely-free physical core.
//
//   * Host < 8 physical cores but >= 8 logical CPUs via SMT (e.g. 6c/12t
//     Ryzen 5 3600/5600/7600, i7-8700; 4c/8t Steam Deck Aerith): CPU 7's
//     physical core is partly claimed by the guest (sibling list contains
//     a CPU in 0..6). Returns the broader mask of all host CPUs NOT in
//     0..6, which on 6c/12t is {7,8,9,10,11} and on 4c/8t is {7}. The
//     emulator thread can then float within that pool, still SMT-fighting
//     guest threads but at least never co-tenanting a logical CPU directly.
//
//   * Host < 8 logical CPUs (e.g. 6c/6t i5-9400, 4c/4t i3-9100): no PS4-
//     reserved core to reclaim. Returns 0 — caller treats as no-op.
//
// Suitable for direct passing to SetCurrentThreadAffinityMask().
u64 GetReservedCoreMask();

// Y-2 (PERF/GR2FORK): mask for the GpuAssembler (BundleAssembler jthread)
// to pin to. Computed only on hosts with >= 7 enumerable physical cores;
// returns 0 on smaller hosts (assembler floats on the OS scheduler).
//
// On hosts where it returns non-zero, this is the next-highest physical
// core below the GpuComm pin, picked symmetrically with GetReservedCoreMask:
//
//   * 8c/16t Z1 Extreme Linux:    0x4040  (CPUs {6, 14}, phys core 6)
//   * 8c/16t Z1 Extreme Windows:  0x3000  (CPUs {12, 13}, phys core 6)
//   * 12c/24t Ryzen 9 Linux:      0x800800 (CPUs {11, 23}, phys core 11)
//
// Caller is BundleAssembler::DrainLoop on the assembler jthread itself.
// Suitable for direct passing to SetCurrentThreadAffinityMask().
u64 GetGpuAssemblerCoreMask();

// PERF(GR2FORK): bits to STRIP from guest-thread affinity masks when
// honoring scePthreadSetaffinity, so the stolen physical core (returned by
// GetReservedCoreMask) becomes truly exclusive to GpuComm.
//
// On hosts with >= 8 physical cores (8c/16t Z1 Extreme, 8c/8t SMT-off,
// 12c/24t, 16c/32t etc.) returns 0 — CPU 7's physical core was already
// outside the guest's claimed range, so no theft is necessary.
//
// On hosts where the reserved core had to be stolen from the guest's range
// to give GpuComm exclusive ownership (4c/8t Steam Deck Aerith, 6c/6t,
// 6c/12t — host has fewer than 8 physical cores), returns the bits the
// stolen physical core occupies in the host CPU space:
//
//   * 6c/12t: 0x820       — host CPUs {5, 11}, physical core 5
//   * 6c/6t:  0x20        — host CPU {5}, physical core 5
//   * 4c/8t:  0x88        — host CPUs {3, 7}, physical core 3
//
// When non-zero, Pthread::SetAffinity AND-strips this from the guest's
// requested mask before applying. The game's view of its own affinity (via
// scePthreadGetaffinity) is unchanged — only the host-side mask is
// narrowed; the game still believes it owns 0..6.
u64 GetGuestExcludedCoreMask();

// PERF(GR2FORK): bits to OR into every guest-thread affinity mask, in
// addition to honoring the game's request, so the OS scheduler can place
// guest threads on host CPUs that are otherwise idle. Computed as: all
// host CPUs, minus the guest's natural PS4 logical range (0..6), minus
// the reserved-for-GpuComm core.
//
// Works correctly regardless of how the host OS lays out SMT pairs.
// Linux on Zen 4 lays out siblings as {N, N+8} so on Z1 Extreme the
// "extra" CPUs are 8..14 (the SMT halves). Windows on the same Zen 4
// lays out siblings as {2N, 2N+1} so the GpuComm-pin lands on physical
// core 7 = {14, 15} via theft, and the "extra" CPUs are 7..13 (entire
// physical cores 4..6 plus CPU 7 = SMT half of physical core 3).
//
// Per-host outcomes:
//
//   * 8c/16t Z1 Extreme Linux:    0x7F00     (CPUs 8..14)
//   * 8c/16t Z1 Extreme Windows:  0x3F80     (CPUs 7..13)
//   * 12c/24t Ryzen 9 Linux:      0xF7FF00   (CPUs 8..18, 20..23)
//   * 16c/32t Threadripper Linux: 0xFF7FFF00 (CPUs 8..22, 24..31)
//   * 6c/12t Ryzen 5 Linux:       0x780      (CPUs 7..10)
//   * 6c/12t Ryzen 5 Windows:     0x380      (CPUs 7..9)
//   * 4c/8t Steam Deck:           0          (no headroom after steal)
//   * 6c/6t / 8c/8t / 4c/4t:      0
//
// Pthread::SetAffinity OR-includes this in every guest mask. The game's
// internal view of its own affinity is unchanged.
u64 GetGuestExpansionMask();

// PERF(GR2FORK): grant the reserved cores EXCLUSIVELY to the calling thread,
// when (and ONLY when) the host topology supports it.
//
// On hosts where GetReservedCoreMask returns a *dedicated free physical core*
// (8c/16t+, 8c/8t SMT-off, 12c/24t, 16c/32t — anywhere CPU 7's physical core
// is entirely outside the guest's 0..6 range), this walks every other thread
// in the current process and AND-strips the reserved-core bits out of each
// thread's affinity mask. After it returns, no thread other than the caller
// is scheduling-eligible for the reserved cores. The calling thread is
// presumed to already be pinned (e.g. via SetCurrentThreadAffinityMask) onto
// exactly the reserved cores; this routine is what backs that pin into hard
// exclusion.
//
// On hosts where GetReservedCoreMask returns a *fallback pool of SMT
// siblings* (6c/12t Ryzen 5, 4c/8t Steam Deck — anywhere CPU 7 is the SMT
// alternate of a guest-claimed core), this no-ops and returns 0. On those
// topologies the "reserved" set is shared with guest-claimed physical cores
// via SMT, so excluding it from every other thread would crowd them onto
// CPUs 0..6 alongside the saturated guest threads — strictly worse than no
// exclusion. The caller's pin to the pool is still beneficial (it keeps the
// hot thread off directly-contested logical CPUs); only the exclusion is
// skipped. The skip is logged at INFO so it's visible in the log.
//
// Inheritance handles forward coverage on Linux: pthread_create copies the
// spawning thread's affinity to the child, so once the main thread (and
// every currently-existing emulator-internal thread) is restricted, new
// threads spawned afterward automatically inherit the restriction.
//
// On Windows, CreateThread does NOT inherit thread-level affinity — new
// threads inherit the *process* default, which is the full host range.
// Threads spawned after this walk runs (Havok workers, AvDemuxer/Av*Decoder,
// dynamically-spawned audio threads, etc.) will end up with the full mask
// and can land on the GpuComm-reserved core. The remedy on Windows is the
// periodic re-walk thread started by StartPeriodicAffinityRewalk; see
// below.
//
// Skips the calling thread (its mask is left intact). Skips threads where
// the resulting mask would be 0 (e.g. the rare thread already pinned to
// exactly the reserved set — only the caller, in normal operation). Returns
// the count of threads whose masks were narrowed; returns 0 if the topology
// doesn't justify the walk, the host has no reserved core to protect, or
// the /proc walk / Toolhelp32 enumeration fails outright.
unsigned ExcludeReservedCoresFromAllOtherThreads();

// PERF(GR2FORK): start (once, idempotently) a detached background thread
// that re-runs ExcludeReservedCoresFromAllOtherThreads every 5 seconds.
//
// Necessary on Windows because CreateThread doesn't inherit thread-level
// affinity from the spawning thread — every thread the game spawns after
// the initial walk arrives with the full host mask and is free to land
// on the GpuComm-reserved core. The periodic walk catches them within
// ~5 seconds of spawn.
//
// Mostly redundant on Linux because pthread_create inherits affinity, but
// cheap (microseconds per pass) and acts as a safety net for the rare
// edge cases (e.g. shadps4:disk$0 threads that spawn from a parent that
// wasn't itself narrowed yet).
//
// Calling this multiple times is a no-op; the std::call_once flag inside
// guarantees only the first invocation actually spawns the thread. The
// thread is detached and runs until process exit; no shutdown machinery
// is provided because the walk is a read/AND-only operation that's safe
// to interrupt at any point.
void StartPeriodicAffinityRewalk();

// Returns the name set on a given OS thread id via SetCurrentThreadName /
// SetThreadName, or an empty string if no name was registered. Lookup is
// process-local and works regardless of whether the OS-level thread
// description (Windows GetThreadDescription / pthread_getname_np) is
// available — useful as a fallback for diagnostic tools (e.g. the hang
// watchdog dumps where the OS API has been observed to return empty).
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
