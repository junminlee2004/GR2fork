// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/libraries/fiber/fiber.h"
#include "core/libraries/kernel/threads/pthread.h"

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
// windows.h first
#include <tlhelp32.h>
#include "common/string_util.h"
#else
#if defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#include <pthread_np.h>
#else
#include <pthread.h>
#endif
#include <sched.h>
#endif
#ifdef __linux__
#include <dirent.h>
#include <sys/syscall.h>
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __FreeBSD__
#define cpu_set_t cpuset_t
#endif

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

#ifdef _WIN32

// Sets the debugger-visible name of the current thread.
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    SetThreadDescription(GetCurrentThread(), UTF8ToUTF16W(name).data());
}

void SetThreadName(void* thread, const char* name) {
    SetThreadDescription(thread, UTF8ToUTF16W(name).data());
}

#else // !_WIN32, so must be POSIX threads

// MinGW with the POSIX threading model does not support pthread_setname_np
#if !defined(_WIN32) || defined(_MSC_VER)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
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
    // TODO
}
#endif

#if defined(_WIN32)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    // Do Nothing on MinGW
}

void SetThreadName(void* thread, const char* name) {
    // Do Nothing on MinGW
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

std::string GetCurrentThreadName() {
    using namespace Libraries::Kernel;
    if (g_curthread && !g_curthread->name.empty()) {
        if (g_curthread->tcb->tcb_fiber) {
            return fmt::format("{}@@{}", g_curthread->name,
                               g_curthread->tcb->tcb_fiber->current_fiber->name);
        }
        return g_curthread->name;
    }
#ifdef _WIN32
    PWSTR name{};
    if (FAILED(GetThreadDescription(GetCurrentThread(), &name)) || name == nullptr) {
        return "<unknown name>";
    }
    const auto result = Common::UTF16ToUTF8(name);
    LocalFree(name);
    return result;
#else
    char name[256];
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0) {
        return "<unknown name>";
    }
    return std::string{name};
#endif
}

namespace {

std::atomic<bool> g_core_reservation_enabled{false};

#if defined(__linux__)
// Parses a Linux cpu list ("0,8", "0-3", "0-3,8-11"). Returns 0 on a parse error.
u64 ParseCpuListLinux(const std::string& s) {
    u64 mask = 0;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) {
            ++i;
        }
        if (i >= s.size()) {
            break;
        }
        u64 lo = 0;
        bool got_digit = false;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            lo = lo * 10 + static_cast<u64>(s[i] - '0');
            ++i;
            got_digit = true;
        }
        if (!got_digit) {
            return 0;
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
                return 0;
            }
        }
        if (lo >= 64 || hi >= 64 || hi < lo) {
            return 0;
        }
        for (u64 cpu = lo; cpu <= hi; ++cpu) {
            mask |= 1ULL << cpu;
        }
    }
    return mask;
}
#endif

u64 HostCpuMask(unsigned hw) {
    return hw >= 64 ? ~0ULL : (1ULL << hw) - 1ULL;
}

} // Anonymous namespace

bool SetCurrentThreadAffinityMask(u64 mask) {
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return false;
    }
    mask &= HostCpuMask(hw);
    if (mask == 0) {
        return false;
    }
#ifdef _WIN32
    if (SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(mask)) == 0) {
        LOG_WARNING(Common, "SetThreadAffinityMask({:#x}) failed: {}", mask, GetLastErrorMsg());
        return false;
    }
    return true;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    for (unsigned cpu = 0; cpu < std::min(64u, static_cast<unsigned>(CPU_SETSIZE)); ++cpu) {
        if (mask & (1ULL << cpu)) {
            CPU_SET(cpu, &set);
        }
    }
    const int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        LOG_WARNING(Common, "pthread_setaffinity_np({:#x}) failed: errno={}", mask, rc);
        return false;
    }
    return true;
#else
    return false;
#endif
}

namespace {

// The pin mask (0 = host too small), whether it is a dedicated physical core the walk may
// isolate, and the bits taken from the guest's CPU 0..6 range.
struct ReservedCoreDecision {
    u64 mask;
    bool is_dedicated_physical_core;
    u64 steal_from_guest;
};

// Sorted by (efficiency class, lowest logical CPU) so back() is the highest-indexed core of the
// fastest class: on hybrid CPUs the highest-indexed cores are the slow ones.
struct CoreInfo {
    u64 mask;
    u8 efficiency_class; // 0 = least performant
};

bool CoreInfoLess(const CoreInfo& a, const CoreInfo& b) {
    if (a.efficiency_class != b.efficiency_class) {
        return a.efficiency_class < b.efficiency_class;
    }
    return std::countr_zero(a.mask) < std::countr_zero(b.mask);
}

#if defined(__linux__)
// "intel_atom" / "intel_core" on hybrid kernels 5.18+; -1 when there is no hybrid signal.
int ReadCoreTypeLinux(unsigned cpu) {
    char path[160];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/topology/core_type", cpu);
    std::ifstream in(path);
    if (!in.is_open()) {
        return -1;
    }
    std::string s;
    if (!std::getline(in, s)) {
        return -1;
    }
    while (!s.empty() &&
           (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    if (s == "intel_atom") {
        return 0;
    }
    if (s == "intel_core") {
        return 1;
    }
    return -1;
}

std::vector<CoreInfo> EnumeratePhysicalCores(unsigned hw) {
    std::vector<CoreInfo> cores;
    if (hw == 0 || hw > 64) {
        return cores;
    }
    std::vector<bool> visited(hw, false);
    for (unsigned cpu = 0; cpu < hw; ++cpu) {
        if (visited[cpu]) {
            continue;
        }
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
            sibs = 1ULL << cpu;
        }
        const int ec = ReadCoreTypeLinux(cpu);
        cores.push_back(CoreInfo{sibs, ec < 0 ? u8{0} : static_cast<u8>(ec)});
        for (unsigned c = 0; c < hw; ++c) {
            if (sibs & (1ULL << c)) {
                visited[c] = true;
            }
        }
    }
    std::sort(cores.begin(), cores.end(), CoreInfoLess);
    return cores;
}
#elif defined(_WIN32)
std::vector<CoreInfo> EnumeratePhysicalCores(unsigned) {
    std::vector<CoreInfo> cores;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return cores;
    }
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
                cores.push_back(CoreInfo{m, static_cast<u8>(p->Processor.EfficiencyClass)});
            }
        }
        p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
            reinterpret_cast<std::byte*>(p) + p->Size);
    }
    std::sort(cores.begin(), cores.end(), CoreInfoLess);
    return cores;
}
#else
std::vector<CoreInfo> EnumeratePhysicalCores(unsigned) {
    return {};
}
#endif

ReservedCoreDecision DecideReservedCores() {
    ReservedCoreDecision result{0, false, 0};
    if (!g_core_reservation_enabled.load(std::memory_order_acquire)) {
        return result;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw < 6) {
        // 4c/4t and below: taking a physical core would leave the guest too few.
        return result;
    }
    constexpr u64 GUEST_MASK = (1ULL << 7) - 1; // the guest's CPUs 0..6
    const std::vector<CoreInfo> cores = EnumeratePhysicalCores(hw);
    const u8 top_efficiency = cores.empty() ? u8{0} : cores.back().efficiency_class;

    {
        static std::once_flag log_once;
        std::call_once(log_once, [&] {
            unsigned fast = 0, slow = 0;
            u64 fast_mask = 0, slow_mask = 0;
            for (const auto& ci : cores) {
                if (ci.efficiency_class >= top_efficiency) {
                    ++fast;
                    fast_mask |= ci.mask;
                } else {
                    ++slow;
                    slow_mask |= ci.mask;
                }
            }
            LOG_INFO(Common,
                     "CPU topology: {} performant cores (mask={:#x}, class {}), {} efficient "
                     "cores (mask={:#x})",
                     fast, fast_mask, top_efficiency, slow, slow_mask);
        });
    }

    // First choice: CPU 7's physical core when it lies wholly outside the guest range (Linux
    // numbering {7, 15}) and is of the fastest class.
    u64 cpu7_phys = 0;
    u8 cpu7_efficiency = 0;
    for (const auto& ci : cores) {
        if (ci.mask & (1ULL << 7)) {
            cpu7_phys = ci.mask;
            cpu7_efficiency = ci.efficiency_class;
            break;
        }
    }

    if (cpu7_phys != 0 && (cpu7_phys & GUEST_MASK) == 0 && cpu7_efficiency >= top_efficiency) {
        result.mask = cpu7_phys;
        result.is_dedicated_physical_core = true;
    } else if (cores.size() < 4) {
        // Too few physical cores to take one: pin without isolating, if CPU 7 exists at all.
        if (cpu7_phys != 0) {
            result.mask = cpu7_phys;
        }
    } else {
        // CPU 7 shares a core with the guest range (Windows numbering {6, 7}) or is a slow
        // core: take the highest fast core and strip it from the guest.
        const u64 highest = cores.back().mask;
        if (highest != 0) {
            result.mask = highest;
            result.is_dedicated_physical_core = true;
            result.steal_from_guest = highest & GUEST_MASK;
        }
    }
    return result;
}

} // Anonymous namespace

void SetCoreReservationEnabled(bool enabled) {
    g_core_reservation_enabled.store(enabled, std::memory_order_release);
}

u64 GetReservedCoreMask() {
    return DecideReservedCores().mask;
}

u64 GetExclusionStripMask() {
    // Same gates as the walk below; keep the two in step.
    const auto decision = DecideReservedCores();
    if (decision.mask == 0 || !decision.is_dedicated_physical_core) {
        return 0;
    }
    return decision.mask;
}

unsigned ExcludeReservedCoresFromAllOtherThreads() {
    const auto decision = DecideReservedCores();
    if (decision.mask == 0) {
        return 0;
    }
    if (!decision.is_dedicated_physical_core) {
        LOG_INFO(Common,
                 "ExcludeReservedCores: reserved={:#x} is not a dedicated physical core, no walk",
                 decision.mask);
        return 0;
    }
    const u64 reserved = decision.mask;
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 0;
    }
    const u64 host_cpus = HostCpuMask(hw);

#if defined(__linux__)
    // sched_setaffinity on a thread of the calling process needs no privilege.
    const pid_t self_tid = static_cast<pid_t>(syscall(SYS_gettid));
    DIR* d = opendir("/proc/self/task");
    if (d == nullptr) {
        LOG_WARNING(Common, "ExcludeReservedCores: opendir(/proc/self/task) failed: {}", errno);
        return 0;
    }
    unsigned narrowed = 0;
    unsigned already_narrow = 0;
    while (const dirent* ent = readdir(d)) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        const pid_t tid = static_cast<pid_t>(std::atoi(ent->d_name));
        if (tid <= 0 || tid == self_tid) {
            continue;
        }
        cpu_set_t cur_set;
        CPU_ZERO(&cur_set);
        if (sched_getaffinity(tid, sizeof(cur_set), &cur_set) != 0) {
            continue; // the thread exited between readdir and here
        }
        u64 cur_mask = 0;
        for (unsigned c = 0; c < std::min(64u, static_cast<unsigned>(CPU_SETSIZE)); ++c) {
            if (CPU_ISSET(c, &cur_set)) {
                cur_mask |= 1ULL << c;
            }
        }
        if ((cur_mask & reserved) == 0) {
            ++already_narrow;
            continue;
        }
        const u64 next_mask = cur_mask & ~reserved;
        if (next_mask == 0) {
            continue; // stripping would strand it with no CPU
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
    }
    closedir(d);
    if (narrowed != 0) {
        LOG_INFO(Common, "ExcludeReservedCores: reserved={:#x} host={:#x} narrowed={} already={}",
                 reserved, host_cpus, narrowed, already_narrow);
    }
    return narrowed;
#elif defined(_WIN32)
    // Threads created after the snapshot are caught by the periodic walk.
    const DWORD self_pid = GetCurrentProcessId();
    const DWORD self_tid = GetCurrentThreadId();
    if ((host_cpus & ~reserved) == 0) {
        return 0;
    }
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        LOG_WARNING(Common, "ExcludeReservedCores: CreateToolhelp32Snapshot failed: {}",
                    GetLastErrorMsg());
        return 0;
    }
    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    if (!Thread32First(snap, &te)) {
        CloseHandle(snap);
        return 0;
    }
    unsigned narrowed = 0;
    do {
        if (te.th32OwnerProcessID != self_pid || te.th32ThreadID == self_tid) {
            continue;
        }
        HANDLE h =
            OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
        if (h == nullptr) {
            continue;
        }
        GROUP_AFFINITY ga{};
        u64 cur_mask = host_cpus;
        if (GetThreadGroupAffinity(h, &ga)) {
            cur_mask = static_cast<u64>(ga.Mask);
        } else {
            ga.Group = 0;
            ga.Mask = static_cast<KAFFINITY>(host_cpus);
        }
        const u64 next_mask = cur_mask & ~reserved;
        if ((cur_mask & reserved) != 0 && next_mask != 0) {
            ga.Mask = static_cast<KAFFINITY>(next_mask);
            if (SetThreadGroupAffinity(h, &ga, nullptr)) {
                ++narrowed;
            }
        }
        CloseHandle(h);
    } while (Thread32Next(snap, &te));
    CloseHandle(snap);
    if (narrowed != 0) {
        LOG_INFO(Common, "ExcludeReservedCores: reserved={:#x} host={:#x} narrowed={}", reserved,
                 host_cpus, narrowed);
    }
    return narrowed;
#else
    (void)host_cpus;
    return 0;
#endif
}

void StartPeriodicAffinityRewalk() {
    static std::once_flag started;
    std::call_once(started, [] {
        std::thread([] {
            SetCurrentThreadName("shadPS4:AffinityW");
            // Linux births this thread on the caller's pin and the walk skips its own thread.
            if (const u64 hot = GetReservedCoreMask()) {
                SetCurrentThreadAffinityMask(~hot);
            }
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                ExcludeReservedCoresFromAllOtherThreads();
            }
        }).detach();
    });
}

} // namespace Common
