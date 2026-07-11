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

#include "common/config.h"
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

// Process-local mirror of names set via SetCurrentThreadName/SetThreadName, keyed by OS thread
// id. The hang watchdog falls back to it when Windows GetThreadDescription returns empty despite
// SetThreadDescription having been called; population is _WIN32-only, else Get() returns "".
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
// GR2FORK FIX: SetThreadStackGuarantee(32 KiB) in CrashHandler::Install only configures the main
// thread; a worker stack overflow then leaves no room for the UEF/VEH frame and the process dies
// with no crash dump. Every spawned thread names itself early, so the guard is applied here.
void EnsureCrashReadyThread() {
    thread_local bool done = false;
    if (done) [[likely]] return;
    done = true;

    // GR2FORK FIX: skip guest pthreads - they run on game-allocated stacks (TEB StackLimit ==
    // nullptr here) where SetThreadStackGuarantee plants PAGE_GUARD pages in guest-owned memory
    // and wedges worker threads. WIN32_LEAN_AND_MEAN hides TEB.NtTib, so cast through NT_TIB.
    const auto* tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
    if (tib->StackLimit == nullptr) {
        return;
    }

    // 32 KiB matches what CrashHandler::Install reserves for the main
    // thread. Failure (e.g., guarantee larger than thread's existing
    // stack) is non-fatal - the thread simply runs without the guard.
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
// MinGW path: SetThreadDescription is exported from kernel32.dll on Windows 10 1607+ regardless
// of compiler, so call it directly; the registry doubles as a fallback in case
// GetThreadDescription cannot read back what was written.
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

// GR2FORK PERF: per-thread affinity helpers + reserved-core discovery. With guest SetAffinity
// honored the game claims host CPUs 0..6, leaving the PS4-reserved core 7 free; emulator threads
// self-pin there (from their own entry, avoiding init races) instead of fighting the game on SMT.

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
        // hardware_concurrency may legally return 0 ("not computable") - bail.
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

// Source-of-truth topology decision: returns the GpuComm pin mask (0 = host too small), whether
// it is a dedicated physical core the exclusion walk may isolate, and the bits to strip from
// guest affinity masks when the core is stolen from the guest's 0..6 range. Unit-tested per host.
struct ReservedCoreDecision {
    u64 mask;
    bool is_dedicated_physical_core;
    u64 steal_from_guest;
    // Assembler reservation is computed only when the host has >= 7 physical cores, so the guest
    // keeps >= 4 after up to 3 are taken; smaller hosts leave these zero and the assembler floats.
    u64 assembler_mask;
    bool assembler_is_dedicated;
    u64 assembler_steal_from_guest;
};

// GR2FORK FIX: on hybrid CPUs (Intel 12th+ gen, ARM big.LITTLE, Snapdragon X) the highest-
// indexed physical cores are E-cores; pinning CPU-bound GpuComm there caps throughput at E-core
// IPC. Sorting by (efficiency class asc, lowest-bit-set asc) keeps cores.back() the top P-core.
struct CoreInfo {
    u64 mask;
    u8 efficiency_class;  // 0 = least performant (E-core), higher = P/Prime.
};

static bool CoreInfoLess(const CoreInfo& a, const CoreInfo& b) {
    if (a.efficiency_class != b.efficiency_class) {
        return a.efficiency_class < b.efficiency_class;
    }
    // Within a single efficiency tier, sort by lowest-set-bit ascending so
    // .back() of that tier is the highest-indexed core. countr_zero(0) is
    // UB per the standard, but mask is non-zero by enumeration filter.
    return std::countr_zero(a.mask) < std::countr_zero(b.mask);
}

#if defined(__linux__)
// Reads the Linux efficiency-class hint from /sys/devices/system/cpu/cpuN/topology/core_type
// ("intel_atom" / "intel_core", Intel hybrid kernels 5.18+). Returns -1 when no hybrid signal
// is available; callers treat that as class 0, collapsing the sort to lowest-bit-set order.
int ReadCoreTypeLinux(unsigned cpu) {
    char path[160];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%u/topology/core_type", cpu);
    std::ifstream in(path);
    if (!in.is_open()) return -1;
    std::string s;
    if (!std::getline(in, s)) return -1;
    // Trim trailing whitespace/newline that getline doesn't strip.
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' ||
                          s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    if (s == "intel_atom") return 0;
    if (s == "intel_core") return 1;
    return -1;
}

// Enumerates physical cores from /sys thread_siblings_list, one CoreInfo per core, sorted by
// (efficiency class asc, lowest-bit-set asc) so .back() is the highest-indexed core of the most
// performant class. Returns an empty vector when /sys is unavailable.
std::vector<CoreInfo> EnumeratePhysicalCoresLinux(unsigned hw) {
    std::vector<CoreInfo> cores;
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
            // /sys unavailable for this CPU - treat as single-thread core.
            sibs = 1ULL << cpu;
        }
        const int ec = ReadCoreTypeLinux(cpu);
        const u8 efficiency = (ec < 0) ? 0 : static_cast<u8>(ec);
        cores.push_back(CoreInfo{sibs, efficiency});
        for (unsigned c = 0; c < hw; ++c) {
            if (sibs & (1ULL << c)) visited[c] = true;
        }
    }
    std::sort(cores.begin(), cores.end(), CoreInfoLess);
    return cores;
}
#endif

#if defined(_WIN32)
// Windows analogue: walks GetLogicalProcessorInformationEx RelationProcessorCore entries,
// capturing EfficiencyClass directly; the sort matches the Linux side.
std::vector<CoreInfo> EnumeratePhysicalCoresWindows() {
    std::vector<CoreInfo> cores;
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
            if (m != 0) {
                cores.push_back(CoreInfo{
                    m, static_cast<u8>(p->Processor.EfficiencyClass)});
            }
        }
        p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            reinterpret_cast<std::byte*>(p) + p->Size);
    }
    std::sort(cores.begin(), cores.end(), CoreInfoLess);
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

    // Physical-core enumeration is hoisted so the phase checks and the assembler decision share
    // one walk (init/SetAffinity paths only, microseconds each). GR2FORK FIX: the sort makes
    // cores.back() the highest-indexed P-core on hybrid hardware, plain highest core on uniform.
    std::vector<CoreInfo> cores;
#ifdef _WIN32
    cores = EnumeratePhysicalCoresWindows();
#elif defined(__linux__)
    cores = EnumeratePhysicalCoresLinux(hw);
#endif

    // Highest efficiency class observed: 0 on uniform CPUs (the top-tier check then passes every
    // core), 1 for Intel P-cores, 2+ for Snapdragon X Prime cores.
    const u8 top_efficiency = cores.empty() ? 0 : cores.back().efficiency_class;

    // Log the hybrid-topology decision once so the log records whether P/E-core detection worked;
    // the process-static cache keeps the many DecideReservedCores calls from spamming.
    {
        static std::once_flag log_once;
        std::call_once(log_once, [&]() {
            unsigned p_cores = 0, e_cores = 0;
            u64 p_mask = 0, e_mask = 0;
            for (const auto& ci : cores) {
                if (ci.efficiency_class >= top_efficiency) {
                    ++p_cores;
                    p_mask |= ci.mask;
                } else {
                    ++e_cores;
                    e_mask |= ci.mask;
                }
            }
            if (e_cores > 0) {
                LOG_INFO(Common,
                         "CPU topology: hybrid detected, {} performant cores "
                         "(mask=0x{:x}, efficiency_class={}), {} efficient cores "
                         "(mask=0x{:x})",
                         p_cores, p_mask, top_efficiency, e_cores, e_mask);
            } else {
                LOG_INFO(Common,
                         "CPU topology: uniform, {} physical cores (mask=0x{:x})",
                         p_cores, p_mask);
            }
        });
    }

    // First choice: use CPU 7's physical core when it is naturally free (hosts with >= 8 physical
    // cores; no guest mask requests it). GR2FORK FIX: the core must also be top efficiency tier -
    // a CPU 7 on an E-core is rejected and the steal path below picks a P-core instead.
    u64 cpu7_phys = 0;
    u8 cpu7_efficiency = 0;
    {
        const u64 cpu7_bit = 1ULL << 7;
        for (const auto& ci : cores) {
            if (ci.mask & cpu7_bit) {
                cpu7_phys = ci.mask;
                cpu7_efficiency = ci.efficiency_class;
                break;
            }
        }
    }

    if (cpu7_phys != 0 && (cpu7_phys & (1ULL << 7)) != 0
        && (cpu7_phys & GUEST_MASK) == 0
        && cpu7_efficiency >= top_efficiency) {
        // CPU 7's physical core is entirely outside guest range AND it's a
        // P-core (or, on uniform hardware, the only class there is). Tight
        // pin available; full exclusion is sensible; no theft needed.
        result.mask = cpu7_phys;
        result.is_dedicated_physical_core = true;
        result.steal_from_guest = 0;
    } else if (cores.size() < 4) {
        // Fallback for rare topologies.
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
        // Steal path: CPU 7's physical core overlaps the guest range (SMT sibling) or sits on an
        // E-core, so steal the highest physical core for GpuComm and strip its bits from guest
        // masks. GR2FORK FIX: the sort makes this the top P-core on hybrid CPUs, not an E-core.
        const u64 highest = cores.back().mask;
        if (highest == 0) {
            return result;
        }
        result.mask = highest;
        result.is_dedicated_physical_core = true;
        result.steal_from_guest = highest;
    }

    // Assembler reservation: requires >= 4 enumerable physical cores so GpuComm and the
    // assembler each own a physical core and the guest still keeps >= 2 for everything else.

    // GR2FORK: the legacy monolithic GpuComm path (Config::isLegacyMonolithicGpuComm,
    // 4-physical-core hosts running GRR) has no GpuAssembler thread to pin; skipping this
    // block leaves the zero defaults, so only GpuComm's core is stripped from guest threads.
    if (cores.size() >= 4 && result.mask != 0 &&
        !Config::isLegacyMonolithicGpuComm()) {
        // Walk physical cores from highest to lowest and pick the first that does not overlap
        // GpuComm's mask - the second-highest core on every topology. GR2FORK FIX: the sort walks
        // P-cores first; only a single-P-core host falls through to an E-core pin.
        for (auto it = cores.rbegin(); it != cores.rend(); ++it) {
            const u64 c = it->mask;
            if (c == 0) continue;
            if ((c & result.mask) != 0) continue;  // overlaps GpuComm
            result.assembler_mask = c;
            result.assembler_is_dedicated = true;
            // Bits of the picked core inside the guest range are stolen - Pthread::SetAffinity
            // AND-strips them from guest masks (on Z1 Linux phys core 6 = {6,14}, and bit 6 is in
            // GUEST_MASK 0x7F).
            result.assembler_steal_from_guest = (c & GUEST_MASK);
            break;
        }
    }

    return result;
}

} // anonymous namespace

unsigned GetPhysicalCoreCount() {
#if defined(_WIN32)
    return static_cast<unsigned>(EnumeratePhysicalCoresWindows().size());
#elif defined(__linux__)
    const unsigned hw = std::thread::hardware_concurrency();
    return static_cast<unsigned>(EnumeratePhysicalCoresLinux(hw).size());
#else
    // macOS / *BSD: no enumerator (the affinity code no-ops here too). Report
    // "unknown" so callers don't force topology-dependent behavior.
    return 0;
#endif
}

u64 GetReservedCoreMask() {
    return DecideReservedCores().mask;
}

u64 GetGpuAssemblerCoreMask() {
    return DecideReservedCores().assembler_mask;
}

u64 GetHotCoreFenceMask() {
    // Cores that secondary workers (async pipeline-compile workers, the cache-storage
    // compression I/O worker) must avoid so they never land on the GpuComm or GpuAssembler
    // core while those are busy - normally both hot-core pins OR'd together.

    // GR2FORK: non-exclusive mode (Config::isGpuCoresNonExclusive, 4-physical-core GRR hosts)
    // returns 0, so the scheduler may place a compile on the hot cores exactly when those
    // threads idle during a compile burst - reclaiming them frees the burst on a starved host.
    if (Config::isGpuCoresNonExclusive()) {
        return 0;
    }
    return GetReservedCoreMask() | GetGpuAssemblerCoreMask();
}

u64 GetGuestExcludedCoreMask() {
    const auto d = DecideReservedCores();
    return d.steal_from_guest | d.assembler_steal_from_guest;
}

u64 GetGuestExpansionMask() {
    // Host CPUs OR'd into every guest-thread affinity mask so guest threads can use otherwise-
    // idle cores: all host CPUs minus the guest 0..6 range minus the reserved cores. Holds for
    // any SMT layout (Linux {N, N+8}, Windows {2N, 2N+1}); the game's affinity view is unchanged.
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

u64 GetExclusionStripMask() {
    // GR2FORK: mirrors the gate sequence of ExcludeReservedCoresFromAllOtherThreads below - keep
    // the two in sync. Returns the strip mask only when the walk would run (else 0); callers
    // self-apply ~mask at thread start instead of waiting up to 5 s for the periodic re-walk.
    if (Config::isGpuCoresNonExclusive()) {
        return 0;
    }
    const auto decision = DecideReservedCores();
    if (decision.mask == 0) {
        return 0;
    }
    if (!decision.is_dedicated_physical_core) {
        return 0;
    }
    return decision.mask | decision.assembler_mask;
}

unsigned ExcludeReservedCoresFromAllOtherThreads() {
    // GR2FORK: non-exclusive mode (Config::isGpuCoresNonExclusive, 4-physical-core GRR hosts)
    // keeps the GpuComm/GpuAssembler self-pins but skips the strip walk - isolating two of four
    // cores starves GRR's compile bursts. Gate mirrors GetExclusionStripMask - keep in sync.
    if (Config::isGpuCoresNonExclusive()) {
        return 0;
    }
    const auto decision = DecideReservedCores();
    if (decision.mask == 0) {
        // Host has no reserved core (hw < 6, or rare topologies that fell
        // out of every branch in DecideReservedCores). Nothing to protect.
        return 0;
    }
    if (!decision.is_dedicated_physical_core) {
        // Defensive: is_dedicated_physical_core is false only on the rare-topology salvage path
        // in DecideReservedCores (< 4 enumerable physical cores). No exclusively-free core exists
        // there, so walking would force every other thread onto guest-claimed CPUs.
        LOG_INFO(Common,
                 "ExcludeReservedCores: skipping exclusion walk on this host "
                 "(reserved=0x{:x} is not a dedicated free physical core; "
                 "full exclusion would over-constrain other emulator threads "
                 "onto guest-claimed CPUs)",
                 decision.mask);
        return 0;
    }

    // Strip both the GpuComm and assembler pins from every other thread. The caller is GpuComm;
    // the assembler spawns later, inherits the narrowed mask, and applies its own pin in
    // DrainLoop, so OR'ing both masks here is correct ordering.
    const u64 reserved = decision.mask | decision.assembler_mask;
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 0;
    }
    const u64 host_cpus = (hw >= 64) ? ~0ULL : ((1ULL << hw) - 1ULL);

#if defined(__linux__)
    // Walk /proc/self/task/: sched_setaffinity on a TID inside the calling process is
    // unprivileged (CAP_SYS_NICE only gates priority raises and other processes).
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
        // Parse the directory name as a TID. atoi is fine here - directory
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

        // Fast path, no syscall: the thread already excludes the reserved cores (guest pthreads
        // with a 0..6 mask from Pthread::SetAffinity, or an inherited mask missing the range).
        if ((cur_mask & reserved) == 0) {
            ++skipped_already_narrow;
            continue;
        }

        const u64 next_mask = cur_mask & ~reserved;
        if (next_mask == 0) {
            // Pinned to a strict subset of the reserved cores - stripping would strand it with no
            // CPU. Only the caller is intentionally reserved-pinned (skipped via self_tid);
            // defensive against future code paths.
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
    // Walk every thread of the process via CreateToolhelp32Snapshot. Threads spawned after the
    // snapshot are invisible, but children copy the spawner's affinity, so they inherit the
    // restriction once their parents carry it.
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
    // GR2FORK PERF: detached thread re-runs ExcludeReservedCoresFromAllOtherThreads every 5 s -
    // Windows CreateThread does not inherit thread affinity, so later game-spawned threads arrive
    // with the full process mask and would leak onto the reserved cores; a pass takes microseconds.
    static std::once_flag started;
    std::call_once(started, []() {
        std::thread([]() {
            SetCurrentThreadName("shadPS4:AffinityW");
            // GR2FORK FIX: narrow this walker thread itself - Linux births it on GpuComm's pin,
            // Windows gives it the full process mask, and the walk skips self_tid. The raw pins
            // (not GetExclusionStripMask) keep it off the hot cores in every mode.
            if (const u64 hot = GetReservedCoreMask() | GetGpuAssemblerCoreMask()) {
                SetCurrentThreadAffinityMask(~hot);
            }
            using namespace std::chrono_literals;
            for (;;) {
                std::this_thread::sleep_for(5s);
                // The walk logs on every invocation by design - the repeats provide a heartbeat
                // in the log.
                ExcludeReservedCoresFromAllOtherThreads();
            }
        }).detach();
    });
}

} // namespace Common
