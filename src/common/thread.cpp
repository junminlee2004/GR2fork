// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/error.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "ntapi.h"
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread.h>
#elif defined(_WIN32)
#include <windows.h>
#include <tlhelp32.h>
#include "common/string_util.h"
#else
#if defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#include <pthread_np.h>
#else
#include <pthread.h>
#endif
#include <dirent.h>
#include <sched.h>
#include <sys/syscall.h>
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __FreeBSD__
#define cpu_set_t cpuset_t
#endif

namespace Common {

namespace {

// Process-local registry that mirrors what is set via SetCurrentThreadName /
// SetThreadName, indexed by the OS thread id. The hang watchdog falls back
// to this registry when the OS-level name lookup (Windows GetThreadDescription)
// returns empty — which we have observed in the wild despite SetThreadDescription
// having been called. Population is gated on _WIN32 because that's where the
// fallback is used; on other platforms the registry remains empty and Get()
// returns "".
class ThreadNameRegistry {
public:
    void Set(u32 tid, std::string name) {
        std::unique_lock lk(mutex_);
        if (name.empty()) {
            names_.erase(tid);
        } else {
            names_.insert_or_assign(tid, std::move(name));
        }
    }
    std::string Get(u32 tid) const {
        std::shared_lock lk(mutex_);
        const auto it = names_.find(tid);
        return it == names_.end() ? std::string{} : it->second;
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<u32, std::string> names_;
};

ThreadNameRegistry& Registry() {
    static ThreadNameRegistry r;
    return r;
}

inline void RegisterCurrentThreadName(const char* name) {
#ifdef _WIN32
    Registry().Set(static_cast<u32>(::GetCurrentThreadId()), name ? name : "");
#else
    (void)name;
#endif
}

inline void RegisterThreadHandleName(void* handle, const char* name) {
#ifdef _WIN32
    if (handle) {
        const DWORD tid = ::GetThreadId(static_cast<HANDLE>(handle));
        if (tid != 0) {
            Registry().Set(static_cast<u32>(tid), name ? name : "");
        }
    }
#else
    (void)handle;
    (void)name;
#endif
}

#ifdef _WIN32
// FIX(GR2FORK): per-thread stack reservation for crash capture.
//
// CrashHandler::Install() calls SetThreadStackGuarantee(32 KiB) once, but
// that API only configures the calling thread (the main thread). Every
// worker thread spawned later (scheduler, vk-recorder, async resource prep,
// log backend, watchdog, AvPlayer demuxer/video/audio decoders, etc.)
// runs with NO guard reservation. A stack overflow on any of those threads
// goes straight to STATUS_STACK_OVERFLOW with insufficient room left for
// the UEF/VEH frame, and the kernel hard-kills the process — silent death,
// no crash_dump.txt, no log output. This is exactly the failure pattern
// the OnVectoredException comment block in crash_handler.cpp warns about.
//
// Universal chokepoint: every thread we spawn calls SetCurrentThreadName
// shortly after start (verified across vk_scheduler, hang_watchdog, log
// backend, async_resource_prep, avplayer threads). Idempotent via
// thread_local sentinel — second call is a single-byte branch.
//
// Cost: one Windows syscall per thread, exactly once. Zero per-frame
// overhead. Zero hot-path overhead.
//
// FIX(GR2FORK v3.1): guest-pthread guard. The first version of this
// function called SetThreadStackGuarantee unconditionally, which broke
// shadPS4's PS4 pthread emulation. Guest pthreads are created via
// NtCreateThread with INITIAL_TEB pointing into game-allocated memory
// (see core/thread.cpp InitializeTeb): StackBase=alloc+size,
// StackAllocationBase=alloc, **StackLimit=nullptr** explicitly. The
// host thread runs ON the guest's allocated stack — there is no
// separate Windows-managed stack. Calling SetThreadStackGuarantee on
// such a thread makes Windows install PAGE_GUARD pages at the bottom
// of what it thinks is "the stack" — which is actually guest memory
// the PS4 MemoryManager owns. The page-protection mutation breaks the
// guest's stack-overflow expectations and silently wedges worker
// threads. Visible symptom: game appears alive (main render loop ticks,
// pad input processes, scheduler tick advances) but never progresses
// past initial loading — the GR2 boot-into-black-screen regression
// that shipped with v3.
//
// Guest-pthread signature: TEB->NtTib.StackLimit == nullptr at
// SetCurrentThreadName-call time. RunThread (pthread.cpp:203) calls
// SetName immediately after `g_curthread = curthread`, before
// `native_thr.Initialize()` and before any guest code runs that could
// page-fault and cause the kernel to populate StackLimit. Normal
// Windows-managed threads (std::jthread, CreateThread, NtCreateThread
// with valid INITIAL_TEB.StackLimit) always have StackLimit populated
// by thread entry, so this check cleanly distinguishes the two
// without a layering dependency on pthread internals.
void EnsureCrashReadyThread() {
    thread_local bool done = false;
    if (done) [[likely]] return;
    done = true;

    // Skip guest pthreads — see the v3.1 comment above.
    //
    // Access note: TEB's first member is an NT_TIB at offset 0. The
    // public Windows SDK (and especially with WIN32_LEAN_AND_MEAN)
    // opacity-hides the named NtTib field behind Reserved1[12], so
    // `NtCurrentTeb()->NtTib.StackLimit` does not compile under
    // clang-cl/MSVC with the project's defines. NT_TIB itself is a
    // public, stable struct in winnt.h, so cast through it: the layout
    // is guaranteed because TEB is documented as starting with NT_TIB
    // and every Windows runtime relies on that contract.
    const auto* tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
    if (tib->StackLimit == nullptr) {
        return;
    }

    // 32 KiB matches what CrashHandler::Install reserves for the main
    // thread. Failure (e.g., guarantee larger than thread's existing
    // stack) is non-fatal — the thread simply runs without the guard,
    // which is the pre-fix status quo.
    ULONG guaranteed = 32 * 1024;
    ::SetThreadStackGuarantee(&guaranteed);
}
#else
inline void EnsureCrashReadyThread() {}
#endif

} // anonymous namespace

std::string GetThreadNameByTid(u32 tid) {
    return Registry().Get(tid);
}

} // namespace Common

namespace Common {

#ifdef __APPLE__

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    // CPU time to grant.
    const std::chrono::nanoseconds computation_ns = period_ns / 2;

    // Determine the timebase for converting time to ticks.
    struct mach_timebase_info timebase{};
    mach_timebase_info(&timebase);
    const auto ticks_per_ns =
        static_cast<double>(timebase.denom) / static_cast<double>(timebase.numer);

    const auto period_ticks =
        static_cast<u32>(static_cast<double>(period_ns.count()) * ticks_per_ns);
    const auto computation_ticks =
        static_cast<u32>(static_cast<double>(computation_ns.count()) * ticks_per_ns);

    thread_time_constraint_policy policy = {
        .period = period_ticks,
        .computation = computation_ticks,
        // Should not matter since preemptible is false, but needs to be >= computation regardless.
        .constraint = computation_ticks,
        .preemptible = false,
    };

    int ret = thread_policy_set(
        pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
        reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    if (ret != KERN_SUCCESS) {
        LOG_ERROR(Common, "Could not set thread to real-time with period {} ns: {}",
                  period_ns.count(), ret);
    }
}

#else

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    // Not implemented
}

#endif

#ifdef _WIN32

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    auto handle = GetCurrentThread();
    int windows_priority = 0;
    switch (new_priority) {
    case ThreadPriority::Low:
        windows_priority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
    case ThreadPriority::Normal:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    case ThreadPriority::High:
        windows_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
    case ThreadPriority::VeryHigh:
        windows_priority = THREAD_PRIORITY_HIGHEST;
        break;
    case ThreadPriority::Critical:
        windows_priority = THREAD_PRIORITY_TIME_CRITICAL;
        break;
    default:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    }
    SetThreadPriority(handle, windows_priority);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();

    LARGE_INTEGER interval{
        .QuadPart = -1 * (duration.count() / 100u),
    };
    HANDLE timer = ::CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &interval, 0, NULL, NULL, 0);
    const auto ret = WaitForSingleObjectEx(timer, INFINITE, interruptible);
    ::CloseHandle(timer);

    if (remaining) {
        const auto end_sleep = std::chrono::high_resolution_clock::now();
        const auto sleep_time = end_sleep - begin_sleep;
        *remaining = duration > sleep_time ? duration - sleep_time : std::chrono::nanoseconds(0);
    }
    return ret == WAIT_OBJECT_0;
}

#else

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    pthread_t this_thread = pthread_self();

    const auto scheduling_type = SCHED_OTHER;
    s32 max_prio = sched_get_priority_max(scheduling_type);
    s32 min_prio = sched_get_priority_min(scheduling_type);
    u32 level = std::max(static_cast<u32>(new_priority) + 1, 4U);

    struct sched_param params;
    if (max_prio > min_prio) {
        params.sched_priority = min_prio + ((max_prio - min_prio) * level) / 4;
    } else {
        params.sched_priority = min_prio - ((min_prio - max_prio) * level) / 4;
    }

    pthread_setschedparam(this_thread, scheduling_type, &params);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    timespec request = {
        .tv_sec = duration.count() / 1'000'000'000,
        .tv_nsec = duration.count() % 1'000'000'000,
    };
    timespec remain;
    int ret;
    while ((ret = nanosleep(&request, &remain)) < 0 && errno == EINTR) {
        if (interruptible) {
            break;
        }
        request = remain;
    }
    if (remaining) {
        *remaining = std::chrono::nanoseconds(remain.tv_sec * 1'000'000'000 + remain.tv_nsec);
    }
    return ret == 0 || errno != EINTR;
}

#endif

#ifdef _MSC_VER

// Sets the debugger-visible name of the current thread.
void SetCurrentThreadName(const char* name) {
    EnsureCrashReadyThread();
    RegisterCurrentThreadName(name);
    SetThreadDescription(GetCurrentThread(), UTF8ToUTF16W(name).data());
}

void SetThreadName(void* thread, const char* name) {
    RegisterThreadHandleName(thread, name);
    SetThreadDescription(thread, UTF8ToUTF16W(name).data());
}

#else // !MSVC_VER, so must be POSIX threads

// MinGW with the POSIX threading model does not support pthread_setname_np
#if !defined(_WIN32) || defined(_MSC_VER)
void SetCurrentThreadName(const char* name) {
    EnsureCrashReadyThread();
    RegisterCurrentThreadName(name);
#ifdef __APPLE__
    pthread_setname_np(name);
#elif defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    pthread_set_name_np(pthread_self(), name);
#elif defined(__NetBSD__)
    pthread_setname_np(pthread_self(), "%s", (void*)name);
#elif defined(__linux__)
    // Linux limits thread names to 15 characters and will outright reject any
    // attempt to set a longer name with ERANGE.
    std::string truncated(name, std::min(strlen(name), static_cast<std::size_t>(15)));
    if (int e = pthread_setname_np(pthread_self(), truncated.c_str())) {
        errno = e;
        LOG_ERROR(Common, "Failed to set thread name to '{}': {}", truncated, GetLastErrorMsg());
    }
#else
    pthread_setname_np(pthread_self(), name);
#endif
}

void SetThreadName(void* thread, const char* name) {
    RegisterThreadHandleName(thread, name);
    // TODO
}
#endif

#if defined(_WIN32)
// MinGW path. Previously a no-op, which silently broke every diagnostic that
// reads thread names (notably the hang watchdog dumps, which were showing
// name="" for every thread despite SetCurrentThreadName being called all
// over the codebase). SetThreadDescription is exported from kernel32.dll on
// Windows 10 1607+ regardless of compiler, so call it directly. The registry
// is also populated as a defensive fallback in case GetThreadDescription
// fails to read back what we just wrote.
void SetCurrentThreadName(const char* name) {
    EnsureCrashReadyThread();
    RegisterCurrentThreadName(name);
    if (name) {
        SetThreadDescription(GetCurrentThread(), UTF8ToUTF16W(name).data());
    }
}

void SetThreadName(void* thread, const char* name) {
    RegisterThreadHandleName(thread, name);
    if (thread && name) {
        SetThreadDescription(static_cast<HANDLE>(thread), UTF8ToUTF16W(name).data());
    }
}
#endif

#endif

AccurateTimer::AccurateTimer(std::chrono::nanoseconds target_interval)
    : target_interval(target_interval) {}

void AccurateTimer::Start() {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();
    if (total_wait.count() > 0) {
        AccurateSleep(total_wait, nullptr, false);
    }
    start_time = std::chrono::high_resolution_clock::now();
    total_wait -= std::chrono::duration_cast<std::chrono::nanoseconds>(start_time - begin_sleep);
}

void AccurateTimer::End() {
    auto now = std::chrono::high_resolution_clock::now();
    total_wait +=
        target_interval - std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time);
}

// =============================================================================
// PERF(GR2FORK): per-thread affinity helpers + reserved-core discovery.
//
// Why this exists (recap of the wider story so this file is self-explanatory):
//   1. Until recently shadPS4's guest-side Pthread::SetAffinity stub returned
//      success without applying the mask. With that fix in place, GR2 actually
//      partitions its 30+ guest threads across host CPUs 0..6.
//   2. PS4 itself reserves logical core 7 for OS-only work, which guest games
//      respect. With (1) honored, host CPU 7 sits unclaimed by ANY guest
//      thread — a free physical core that emulator-internal threads can use
//      without contending with the running game.
//   3. Emulator-internal threads (GpuComm, Present, AjmWork, ...) currently
//      have no affinity. They float onto whichever logical CPU the OS picks,
//      which on a heavily-loaded handheld means SMT siblings of game-claimed
//      cores 0..6 — i.e. they fight game threads via SMT instead of using
//      the free physical core 7.
//
// SetCurrentThreadAffinityMask() is the per-thread pin primitive. The
// thread that wants to be pinned calls it from its own entry function. Self-
// pinning sidesteps a class of races (target thread might not have
// initialized its native handle yet at the time a controller-thread tries to
// pin it).
//
// GetReservedCoreMask() is the topology query. It is shaped to match exactly
// one consumer: pin-to-the-reserved-PS4-core. It is NOT a general topology
// API; if the codebase ever needs richer topology info, that should grow
// separately so this helper can stay narrowly scoped.
// =============================================================================

namespace {

#if defined(__linux__)
// Parse a Linux cpuset list: "0,8" or "0-3" or "0-3,8-11" or "0,2,4,6".
// Returns a bitmask, or 0 on parse error.
u64 ParseCpuListLinux(const std::string& s) {
    u64 mask = 0;
    std::size_t i = 0;
    while (i < s.size()) {
        // skip whitespace and commas
        while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) {
            ++i;
        }
        if (i >= s.size()) break;
        // parse first integer
        u64 lo = 0;
        bool got_digit = false;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            lo = lo * 10 + static_cast<u64>(s[i] - '0');
            ++i;
            got_digit = true;
        }
        if (!got_digit) {
            return 0; // malformed
        }
        u64 hi = lo;
        if (i < s.size() && s[i] == '-') {
            ++i;
            hi = 0;
            got_digit = false;
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
                hi = hi * 10 + static_cast<u64>(s[i] - '0');
                ++i;
                got_digit = true;
            }
            if (!got_digit) {
                return 0; // malformed range
            }
        }
        if (lo >= 64 || hi >= 64 || hi < lo) {
            return 0; // out of range or inverted
        }
        for (u64 c = lo; c <= hi; ++c) {
            mask |= (1ULL << c);
        }
    }
    return mask;
}
#endif

} // anonymous namespace

bool SetCurrentThreadAffinityMask(u64 mask) {
    // Clip to host CPU count so the syscall won't reject masks that mention
    // CPUs that don't exist (e.g. running on a smaller host than expected).
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        // hardware_concurrency may legally return 0 ("not computable") — bail.
        return false;
    }
    const u64 host_cpus = (hw >= 64) ? ~0ULL : ((1ULL << hw) - 1ULL);
    mask &= host_cpus;
    if (mask == 0) {
        return false;
    }

#ifdef _WIN32
    const DWORD_PTR win_mask = static_cast<DWORD_PTR>(mask);
    if (SetThreadAffinityMask(GetCurrentThread(), win_mask) == 0) {
        LOG_WARNING(Common, "SetThreadAffinityMask({:#x}) failed: {}", mask, GetLastError());
        return false;
    }
    return true;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (unsigned c = 0; c < std::min(64u, static_cast<unsigned>(CPU_SETSIZE)); ++c) {
        if (mask & (1ULL << c)) {
            CPU_SET(c, &set);
        }
    }
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        LOG_WARNING(Common, "pthread_setaffinity_np({:#x}) failed: errno={}", mask, rc);
        return false;
    }
    return true;
#else
    // macOS / *BSD: see thread.h docstring.
    (void)mask;
    return false;
#endif
}

namespace {

// Internal: the source-of-truth topology decision.
//
// Returns three pieces of information:
//
//   - mask: bits to pin GpuComm to. 0 means no pin available on this host
//     (host too small to give one core to GpuComm without crippling guest).
//
//   - is_dedicated_physical_core: true if `mask` refers to a free physical
//     core that GpuComm can own exclusively (ExcludeReservedCoresFromAllOther-
//     Threads will narrow other threads off it). false in the unusual case
//     where we couldn't get a dedicated core (host between 6 logical CPUs
//     and 4 physical cores — currently no real x86 host hits this branch
//     but it's defensive).
//
//   - steal_from_guest: bits to strip from guest-thread affinity masks via
//     Pthread::SetAffinity. Non-zero on hosts where we steal a physical
//     core from the guest's claimed range to give to GpuComm. 0 on hosts
//     where the reserved core was already naturally free (CPU 7 physical
//     core had no overlap with guest mask 0..6 — i.e. host has >= 8
//     physical cores).
//
// Per-host outcomes (verified by unit test):
//
//   8c/16t Z1 Extreme, 8c/16t Ryzen 7, 12c/24t Ryzen 9, 16c/32t Threadripper:
//       CPU 7's physical core naturally free (PS4-reserved, guest never
//       requests it). mask = {7, 7's SMT sibling}. No steal.
//
//   8c/8t SMT-off:  mask = {7}. No steal. Same reasoning.
//
//   6c/12t Ryzen 5 3600/5600/7600, i7-8700: CPU 7's physical core overlaps
//       guest range (it's the SMT alternate of physical core 1). Steal the
//       highest physical core (#5, CPUs {5, 11}). Strip bit 5 from guest.
//       Guest goes from 6 -> 5 physical cores; GpuComm gets one private.
//
//   6c/6t i5-9400 (no SMT): same reasoning, smaller scale. Steal physical
//       core 5 (CPU 5). Strip bit 5 from guest.
//
//   4c/8t Steam Deck Aerith: GpuComm is the documented single-threaded
//       bottleneck on this host (drops fps into the teens without isolation).
//       Steal physical core 3 (CPUs {3, 7}). Guest 4 -> 3 cores; GpuComm
//       gets a private one. Net gain because guest has enough headroom on
//       3 cores to reach 30 fps while GpuComm benefits from no contention.
//
//   4c/4t and below: not enough physical cores to safely steal one. mask = 0.
struct ReservedCoreDecision {
    u64 mask;
    bool is_dedicated_physical_core;
    u64 steal_from_guest;
    // Y-2: assembler reservation. Computed only when the host has >= 7
    // physical cores so the guest keeps >= 4 phys cores after taking up
    // to 3 (GpuComm + assembler + natural-host overhead). On smaller
    // hosts these stay zero/false and the assembler floats unpinned.
    u64 assembler_mask;
    bool assembler_is_dedicated;
    u64 assembler_steal_from_guest;
};

#if defined(__linux__)
// Enumerate physical cores via /sys topology. Returns one mask per physical
// core, in ascending order of lowest-bit-set (so .back() is the highest-
// numbered physical core). Empty vector on /sys unavailability.
//
// Walks CPUs 0..hw-1; for each CPU not yet seen, reads
// /sys/devices/system/cpu/cpuN/topology/thread_siblings_list and pushes the
// resulting mask. Marks every CPU in the siblings list as seen so we don't
// push the same physical core twice.
std::vector<u64> EnumeratePhysicalCoresLinux(unsigned hw) {
    std::vector<u64> cores;
    if (hw == 0 || hw > 64) return cores;
    std::vector<bool> visited(hw, false);
    for (unsigned cpu = 0; cpu < hw; ++cpu) {
        if (visited[cpu]) continue;
        char path[160];
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%u/topology/thread_siblings_list", cpu);
        std::ifstream in(path);
        u64 sibs = 0;
        if (in.is_open()) {
            std::string line;
            if (std::getline(in, line)) {
                sibs = ParseCpuListLinux(line);
            }
        }
        if (sibs == 0) {
            // /sys unavailable for this CPU — treat as single-thread core.
            sibs = 1ULL << cpu;
        }
        cores.push_back(sibs);
        for (unsigned c = 0; c < hw; ++c) {
            if (sibs & (1ULL << c)) visited[c] = true;
        }
    }
    return cores;
}
#endif

#if defined(_WIN32)
// Windows analogue: walk GetLogicalProcessorInformationEx with
// RelationProcessorCore. Sort by lowest-bit-set so .back() is the highest-
// numbered physical core, matching the Linux ordering.
std::vector<u64> EnumeratePhysicalCoresWindows() {
    std::vector<u64> cores;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return cores;
    std::vector<std::byte> buf(len);
    auto* base = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, base, &len)) {
        return cores;
    }
    auto* p = base;
    while (reinterpret_cast<std::byte*>(p) < buf.data() + len) {
        if (p->Relationship == RelationProcessorCore && p->Processor.GroupCount > 0) {
            const u64 m = static_cast<u64>(p->Processor.GroupMask[0].Mask);
            if (m != 0) cores.push_back(m);
        }
        p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            reinterpret_cast<std::byte*>(p) + p->Size);
    }
    std::sort(cores.begin(), cores.end(), [](u64 a, u64 b) {
        // Sort by lowest set bit (i.e. by the lowest CPU index in the core).
        // std::countr_zero is C++20, portable across clang-cl / MSVC / GCC,
        // and avoids the <intrin.h> _BitScanForward64 vs __builtin_ctzll fork.
        // Both args are non-zero by the filter above, so countr_zero is well-
        // defined here (countr_zero(0) is undefined per the standard).
        return std::countr_zero(a) < std::countr_zero(b);
    });
    return cores;
}
#endif

ReservedCoreDecision DecideReservedCores() {
    ReservedCoreDecision result{0, false, 0, 0, false, 0};
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw < 6) {
        // 4c/4t and below: stealing a physical core would leave the guest
        // with too few. Skip.
        return result;
    }
    const u64 host_cpus = (hw >= 64) ? ~0ULL : ((1ULL << hw) - 1ULL);
    constexpr u64 GUEST_MASK = (1ULL << 7) - 1; // bits 0..6

    // Y-2: hoist physical-core enumeration so both Phase 1/2 and the
    // assembler decision below share one walk. /sys reads on Linux are
    // microseconds-scale; the function is called rarely (init paths +
    // SetAffinity calls) so the hoist is harmless.
    std::vector<u64> cores;
#ifdef _WIN32
    cores = EnumeratePhysicalCoresWindows();
#elif defined(__linux__)
    cores = EnumeratePhysicalCoresLinux(hw);
#endif

    // ----- Phase 1: is CPU 7's physical core naturally free? -----
    // On hosts with >= 8 physical cores (8c/16t Z1 Extreme, 8c/8t, 12c/24t,
    // 16c/32t...), CPU 7 sits on its own physical core that no guest mask
    // ever requests. Use it directly, no theft required.
    u64 cpu7_phys = 0;
#ifdef _WIN32
    {
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len != 0) {
            std::vector<std::byte> buf(len);
            auto* base = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
            if (GetLogicalProcessorInformationEx(RelationProcessorCore, base, &len)) {
                auto* p = base;
                const u64 cpu7_bit = 1ULL << 7;
                while (reinterpret_cast<std::byte*>(p) < buf.data() + len) {
                    if (p->Relationship == RelationProcessorCore && p->Processor.GroupCount > 0) {
                        const u64 gm = static_cast<u64>(p->Processor.GroupMask[0].Mask);
                        if (gm & cpu7_bit) {
                            cpu7_phys = gm;
                            break;
                        }
                    }
                    p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                        reinterpret_cast<std::byte*>(p) + p->Size);
                }
            }
        }
    }
#elif defined(__linux__)
    {
        std::ifstream in("/sys/devices/system/cpu/cpu7/topology/thread_siblings_list");
        if (in.is_open()) {
            std::string line;
            if (std::getline(in, line)) {
                cpu7_phys = ParseCpuListLinux(line);
            }
        }
    }
#endif

    if (cpu7_phys != 0 && (cpu7_phys & (1ULL << 7)) != 0
        && (cpu7_phys & GUEST_MASK) == 0) {
        // CPU 7's physical core is entirely outside guest range. Tight pin
        // available; full exclusion is sensible; no theft needed.
        result.mask = cpu7_phys;
        result.is_dedicated_physical_core = true;
        result.steal_from_guest = 0;
    } else if (cores.size() < 4) {
        // ----- Phase 2 fallback: rare topologies -----
        // Need at least 4 physical cores total to safely steal one. Hosts
        // below this (3c/6t and rarer) get no pin.
        if (cpu7_phys != 0 && (cpu7_phys & (1ULL << 7)) != 0) {
            // Salvage: at least pin to the partially-overlapping CPU 7
            // physical core, no theft. Defensive fallback for 3c/6t and
            // equally rare layouts; the steal would be too costly.
            result.mask = cpu7_phys;
            result.is_dedicated_physical_core = false;
            result.steal_from_guest = 0;
        }
        // No assembler pin on these tiny hosts either.
        return result;
    } else {
        // ----- Phase 2: enumerate physical cores and steal the highest -----
        // We get here on hosts where CPU 7's physical core overlaps the
        // guest's claimed range — guest threads can land on it via SMT.
        // Examples:
        //
        //   * 6c/12t Zen 2/3 (Ryzen 5 3600/5600/7600): CPU 7 = SMT(CPU 1)
        //   * 6c/6t (no SMT, host CPUs 0..5):           cpu7_phys path missing
        //   * 4c/8t Steam Deck Aerith:                  CPU 7 = SMT(CPU 3)
        //
        // To give GpuComm a TRULY private physical core we steal the
        // highest-numbered one. The guest's affinity mask (0..6) gets that
        // core's bits stripped via Pthread::SetAffinity, so guest threads
        // can't land there.
        const u64 highest = cores.back();
        if (highest == 0) {
            return result;
        }
        result.mask = highest;
        result.is_dedicated_physical_core = true;
        result.steal_from_guest = highest;
    }

    // ----- Y-2: assembler reservation -----
    // Gate: host must have >= 4 enumerable physical cores so we can reserve
    // 2 (GpuComm + assembler) and still leave the guest >= 2 physical cores
    // for everything else. Below 4 phys cores DecideReservedCores already
    // bails earlier (cores.size() < 4 fallback above), so by the time we
    // reach this point the gate is reduced to "is there a non-overlapping
    // physical core left to give the assembler?".
    //
    // Per-host outcomes after this gate (all give GpuComm one full physical
    // core + assembler one full physical core; "rest" pool is what's left
    // for game/audio/helper threads after the exclusion walk strips both
    // reserved cores):
    //
    //   * 8c/16t Z1 Extreme Linux:    GpuComm={7,15}, assembler={6,14},
    //                                 rest = phys cores 0..5 (12 logical)
    //   * 8c/16t Z1 Extreme Windows:  GpuComm={14,15}, assembler={12,13},
    //                                 rest = phys cores 0..5 (12 logical)
    //   * 12c/24t Ryzen 9 Linux:      GpuComm={7,15}, assembler={11,23},
    //                                 rest = phys cores 0..6,8..10 (10 phys,
    //                                 20 logical)
    //   * 6c/12t Ryzen 5 Linux:       GpuComm={5,11}, assembler={4,10},
    //                                 rest = phys cores 0..3 (8 logical)
    //   * 4c/8t Steam Deck/Aerith:    GpuComm={3,7}, assembler={2,6},
    //                                 rest = phys cores 0..1 (4 logical)
    //   * 6c/6t no-SMT:               GpuComm={5}, assembler={4},
    //                                 rest = phys cores 0..3 (4 logical)
    //   * 4c/4t no-SMT:               hw < 6 early-return, no pin at all
    //
    // Tradeoff on the smaller hosts (4c/8t, 6c/12t): the guest's 7 PS4-
    // logical cores' worth of work has to fit into 2 or 4 host physical
    // cores. That's tighter than upstream shadPS4 — upstream does no
    // pinning at all and lets the OS scheduler distribute everything. The
    // win is that GpuComm and the assembler get exclusive cores with no
    // guest-thread SMT contention on their physical cores; the cost is
    // that guest threads get crammed into the "rest" pool. On 4c/8t this
    // is the most aggressive deployment topology supported.
    if (cores.size() >= 4 && result.mask != 0) {
        // Walk physical cores from highest to lowest; pick the first that
        // doesn't overlap GpuComm's mask. This naturally selects the
        // second-highest physical core on every host topology.
        for (auto it = cores.rbegin(); it != cores.rend(); ++it) {
            const u64 c = *it;
            if (c == 0) continue;
            if ((c & result.mask) != 0) continue;  // overlaps GpuComm
            result.assembler_mask = c;
            result.assembler_is_dedicated = true;
            // If the picked core has any bits in guest range, we steal
            // those — Pthread::SetAffinity will AND-strip them from
            // guest masks. On Z1 Linux, phys core 6 = {6,14} and bit 6
            // IS in GUEST_MASK (0x7F), so we must steal CPU 6 from the
            // guest's range to keep the pin truly exclusive.
            result.assembler_steal_from_guest = (c & GUEST_MASK);
            break;
        }
    }

    return result;
}

} // anonymous namespace

u64 GetReservedCoreMask() {
    return DecideReservedCores().mask;
}

u64 GetGpuAssemblerCoreMask() {
    return DecideReservedCores().assembler_mask;
}

u64 GetGuestExcludedCoreMask() {
    const auto d = DecideReservedCores();
    return d.steal_from_guest | d.assembler_steal_from_guest;
}

u64 GetGuestExpansionMask() {
    // The set of host CPUs that should be OR'd into every guest-thread
    // affinity mask, in addition to honoring the game's request, so the
    // OS scheduler can place guest threads on host CPUs that are otherwise
    // idle. Computed as: all host CPUs, minus the guest's PS4 logical
    // range (0..6 — those are in the base mask), minus the reserved-for-
    // GpuComm core.
    //
    // This is broader than v7's "SMT siblings of guest cores" formula and
    // works correctly regardless of how the host OS lays out SMT pairs:
    //
    //   - Linux on Zen 4 lays out SMT siblings as {N, N+8}, so CPU 7's
    //     sibling is CPU 15. The natural-free-physical-core path (Phase
    //     1 of DecideReservedCores) pins GpuComm to {7, 15}, leaving
    //     CPUs 8..14 idle. This pool returns 8..14 in that case.
    //
    //   - Windows on the same Zen 4 lays out SMT siblings as {2N, 2N+1},
    //     so CPU 7's sibling is CPU 6 — which OVERLAPS the guest range.
    //     Phase 1 of DecideReservedCores rejects that and Phase 2 steals
    //     the highest physical core ({14, 15}) for GpuComm. CPUs 7..13
    //     are then unused — physical cores 3 (second thread), 4, 5, 6 in
    //     full. This pool returns 7..13 in that case.
    //
    //   - On hosts with more cores than 7 (12c/24t, 16c/32t etc.), the
    //     pool naturally extends to include the upper cores too.
    //
    //   - On hosts where GpuComm steals from the guest range (4c/8t Steam
    //     Deck, 6c/12t with overlap, 6c/6t), the pool is computed against
    //     the stolen mask and naturally evaluates to 0 or near-0.
    //
    // Per-host outcomes:
    //
    //   * 8c/16t Z1 Extreme Linux:    pool = 0x7F00     (CPUs 8..14)
    //   * 8c/16t Z1 Extreme Windows:  pool = 0x3F80     (CPUs 7..13)
    //   * 12c/24t Ryzen 9 Linux:      pool = 0xF7FF00   (CPUs 8..18, 20..23)
    //   * 16c/32t Threadripper Linux: pool = 0xFF7FFF00 (CPUs 8..22, 24..31)
    //   * 6c/12t Ryzen 5 Linux:       pool = 0x780      (CPUs 7..10)
    //   * 6c/12t Ryzen 5 Windows:     pool = 0x380      (CPUs 7..9)
    //   * 4c/8t Steam Deck:           pool = 0          (no headroom after steal)
    //   * 6c/6t / 8c/8t / 4c/4t:      pool = 0
    //
    // Pthread::SetAffinity OR-includes this in every guest mask. The game's
    // internal view of its own affinity is unchanged.
    static const u64 cached = []() -> u64 {
        constexpr u64 GUEST_MASK = (1ULL << 7) - 1; // bits 0..6
        const auto d = DecideReservedCores();
        const u64 reserved = d.mask | d.assembler_mask;
        const unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) {
            return 0;
        }
        const u64 host_cpus = (hw >= 64) ? ~0ULL : ((1ULL << hw) - 1ULL);
        return host_cpus & ~GUEST_MASK & ~reserved;
    }();
    return cached;
}

unsigned ExcludeReservedCoresFromAllOtherThreads() {
    const auto decision = DecideReservedCores();
    if (decision.mask == 0) {
        // Host has no reserved core (hw < 6, or rare topologies that fell
        // out of every branch in DecideReservedCores). Nothing to protect.
        return 0;
    }
    if (!decision.is_dedicated_physical_core) {
        // Defensive: the only path that produces is_dedicated_physical_core=
        // false is the rare-topology salvage in DecideReservedCores (host
        // with < 4 enumerable physical cores where CPU 7 still has a partial
        // overlap — exotic, possibly never reached on real x86 layouts).
        // On such a host we don't have a free core to give exclusively, so
        // walking would force everyone else onto guest-claimed CPUs.
        LOG_INFO(Common,
                 "ExcludeReservedCores: skipping exclusion walk on this host "
                 "(reserved=0x{:x} is not a dedicated free physical core; "
                 "full exclusion would over-constrain other emulator threads "
                 "onto guest-claimed CPUs)",
                 decision.mask);
        return 0;
    }

    // Y-2: strip both the GpuComm pin and the assembler pin from every
    // other thread's mask. Caller is the GpuComm thread; the assembler
    // thread spawns later (after Rasterizer ctor) and inherits the parent's
    // already-narrowed mask, then DrainLoop sets the assembler's own pin
    // explicitly. So this walk OR'ing both masks is correct ordering.
    const u64 reserved = decision.mask | decision.assembler_mask;
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 0;
    }
    const u64 host_cpus = (hw >= 64) ? ~0ULL : ((1ULL << hw) - 1ULL);

#if defined(__linux__)
    // Walk /proc/self/task/. Each entry is a TID owned by this process;
    // sched_setaffinity(tid, ...) on a TID in our own process is unprivileged
    // and always permitted (CAP_SYS_NICE is only required for raising priority
    // or affecting tasks in other processes).
    const pid_t self_tid = static_cast<pid_t>(syscall(SYS_gettid));
    DIR* d = opendir("/proc/self/task");
    if (d == nullptr) {
        LOG_WARNING(Common, "ExcludeReservedCores: opendir(/proc/self/task) failed: errno={}", errno);
        return 0;
    }

    unsigned narrowed = 0;
    unsigned skipped_already_narrow = 0;
    unsigned skipped_zero_result = 0;
    unsigned getaffinity_failed = 0;

    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        if (ent->d_name[0] == '.') {
            continue; // "." and ".."
        }
        // Parse the directory name as a TID. atoi is fine here — directory
        // names under /proc/<pid>/task/ are always pure integers.
        const pid_t tid = static_cast<pid_t>(std::atoi(ent->d_name));
        if (tid <= 0 || tid == self_tid) {
            continue; // skip ourselves; we're already pinned to {reserved}
        }

        cpu_set_t cur_set;
        CPU_ZERO(&cur_set);
        if (sched_getaffinity(tid, sizeof(cur_set), &cur_set) != 0) {
            // Thread may have exited between readdir and getaffinity (ESRCH);
            // or in extremely rare cases EPERM. Either way, skip and move on.
            ++getaffinity_failed;
            continue;
        }

        u64 cur_mask = 0;
        for (unsigned c = 0; c < std::min(64u, static_cast<unsigned>(CPU_SETSIZE)); ++c) {
            if (CPU_ISSET(c, &cur_set)) {
                cur_mask |= (1ULL << c);
            }
        }

        // Fast path: thread already excludes the reserved cores. No syscall
        // needed. Catches all guest pthreads that already had a 0..6 mask
        // applied via Pthread::SetAffinity, plus any thread whose inherited
        // mask happened not to cover the reserved range.
        if ((cur_mask & reserved) == 0) {
            ++skipped_already_narrow;
            continue;
        }

        const u64 next_mask = cur_mask & ~reserved;
        if (next_mask == 0) {
            // Thread is currently pinned to a strict subset of the reserved
            // cores. Stripping them would strand it on no CPU. Don't touch.
            // In normal operation this shouldn't happen because the only
            // intentionally-reserved-pinned thread is the caller (skipped
            // above by self_tid). Defensive in case of future code paths.
            ++skipped_zero_result;
            continue;
        }

        cpu_set_t new_set;
        CPU_ZERO(&new_set);
        for (unsigned c = 0; c < std::min(64u, static_cast<unsigned>(CPU_SETSIZE)); ++c) {
            if (next_mask & (1ULL << c)) {
                CPU_SET(c, &new_set);
            }
        }
        if (sched_setaffinity(tid, sizeof(new_set), &new_set) == 0) {
            ++narrowed;
        }
        // If sched_setaffinity fails (e.g. thread exited between getaffinity
        // and setaffinity), silently skip. Logging every transient ESRCH
        // would be noisy; the per-thread loss is harmless.
    }
    closedir(d);

    LOG_INFO(Common,
             "ExcludeReservedCores: reserved=0x{:x} host_cpus=0x{:x}: "
             "narrowed={} already_narrow={} would_be_zero={} getaffinity_failed={}",
             reserved, host_cpus, narrowed, skipped_already_narrow,
             skipped_zero_result, getaffinity_failed);
    return narrowed;
#elif defined(_WIN32)
    // Walk every thread of the current process via CreateToolhelp32Snapshot.
    // The snapshot is a copy at call time; threads spawned after the snapshot
    // is taken are not visible — but that's fine, because pthread_create /
    // CreateThread copies the spawning thread's affinity to the child, so
    // future threads inherit the restriction once their parents have it.
    const DWORD self_pid = GetCurrentProcessId();
    const DWORD self_tid = GetCurrentThreadId();

    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        LOG_WARNING(Common, "ExcludeReservedCores: CreateToolhelp32Snapshot failed: {}",
                    GetLastError());
        return 0;
    }

    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    if (!Thread32First(snap, &te)) {
        CloseHandle(snap);
        return 0;
    }

    const u64 default_mask = host_cpus & ~reserved;
    if (default_mask == 0) {
        CloseHandle(snap);
        return 0;
    }

    unsigned narrowed = 0;
    do {
        if (te.th32OwnerProcessID != self_pid) {
            continue;
        }
        if (te.th32ThreadID == self_tid) {
            continue;
        }
        HANDLE h = OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE,
                              te.th32ThreadID);
        if (h == nullptr) {
            // Thread may have exited; OpenThread also fails for protected
            // system threads we don't own. Skip silently.
            continue;
        }

        // Read current group affinity. If reading fails, fall back to the
        // default-mask AND with host_cpus (we'd like to AND with the current
        // mask but can't read it).
        GROUP_AFFINITY ga{};
        u64 cur_mask = host_cpus;
        if (GetThreadGroupAffinity(h, &ga)) {
            cur_mask = static_cast<u64>(ga.Mask);
        } else {
            ga.Group = 0;
            ga.Mask = static_cast<KAFFINITY>(host_cpus);
        }
        if ((cur_mask & reserved) == 0) {
            CloseHandle(h);
            continue; // already excluding the reserved cores
        }
        const u64 next_mask = cur_mask & ~reserved;
        if (next_mask == 0) {
            CloseHandle(h);
            continue;
        }
        ga.Mask = static_cast<KAFFINITY>(next_mask);
        if (SetThreadGroupAffinity(h, &ga, nullptr)) {
            ++narrowed;
        }
        CloseHandle(h);
    } while (Thread32Next(snap, &te));

    CloseHandle(snap);
    LOG_INFO(Common,
             "ExcludeReservedCores: reserved=0x{:x} host_cpus=0x{:x}: narrowed={}",
             reserved, host_cpus, narrowed);
    return narrowed;
#else
    // macOS / *BSD: stub. Same rationale as the SetCurrentThreadAffinityMask
    // stub in this file.
    (void)host_cpus;
    return 0;
#endif
}

void StartPeriodicAffinityRewalk() {
    // PERF(GR2FORK): start a detached background thread that periodically
    // re-runs ExcludeReservedCoresFromAllOtherThreads.
    //
    // The motivation is fundamentally different on Linux vs Windows:
    //
    //   - On Linux, pthread_create copies the spawning thread's affinity
    //     to its child. The initial walk done at GpuComm startup catches
    //     every thread that exists at that moment, and forward coverage
    //     for new threads is automatic via inheritance. The periodic
    //     re-walk is essentially redundant on Linux — there are some
    //     edge-case threads (e.g. shadps4:disk$0) that are spawned by
    //     a parent that wasn't itself narrowed yet, but they're rare
    //     and not perf-critical.
    //
    //   - On Windows, CreateThread does NOT inherit the parent thread's
    //     affinity. New threads inherit the *process* default affinity,
    //     which is the full host range. After the initial walk, every
    //     thread the game spawns later (Havok, AvDemuxer, AvVideoDecoder,
    //     audio threads, dynamically-spawned GpuSched workers, etc.)
    //     comes back with mask = 0..15 and can land on the GpuComm-
    //     reserved physical core. Without the periodic re-walk, the
    //     entire isolation guarantee leaks over time as the game spawns
    //     more threads.
    //
    // Cost: one Toolhelp32Snapshot enumeration + per-thread mask check
    // every 5 seconds. Walking ~150 threads with simple mask comparisons
    // is microseconds of CPU time per pass — invisible on the perf trace.
    //
    // Started exactly once via std::call_once. Subsequent calls no-op.
    // The thread is detached; it runs until process exit, at which point
    // the OS reclaims it. No clean-shutdown machinery needed because the
    // walk itself is read/AND-only and safe to interrupt at any point.
    static std::once_flag started;
    std::call_once(started, []() {
        std::thread([]() {
            SetCurrentThreadName("shadPS4:AffinityW");
            using namespace std::chrono_literals;
            for (;;) {
                std::this_thread::sleep_for(5s);
                // The walk itself logs only on first execution at INFO.
                // Subsequent invocations re-run the same path and log
                // again, which is intentional — it provides a heartbeat
                // visible in the log.
                ExcludeReservedCoresFromAllOtherThreads();
            }
        }).detach();
    });
}

} // namespace Common
