// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/hang_watchdog.h"
#include "common/sync_trace.h"
#include "common/thread.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include "common/logging/log.h"

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#include <psapi.h>
#include <tlhelp32.h>
#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "psapi.lib")
#endif

#ifdef __linux__
#include <unistd.h>
#endif

namespace Common {

namespace {

constexpr auto kSamplePeriod            = std::chrono::seconds{1};
constexpr auto kStallWarnThreshold      = std::chrono::seconds{5};
constexpr auto kStallCriticalThreshold  = std::chrono::seconds{15};

struct State {
    std::atomic<bool> running{false};
    std::thread thread;
    HangWatchdogCallbacks cb;
    std::mutex start_mu;
    std::atomic<u64> main_kick_count{0};
};

static State& S() {
    static State s;
    return s;
}

// GR2FORK FIX: sample host RAM on every 1 Hz watchdog tick. Memory exhaustion kills silently
// (the Linux OOM killer SIGKILLs uncatchably; Windows commit exhaustion __fastfails past
// VEH/SEH), so a log ending with sysFree collapsing toward zero identifies the death as OOM.

struct RamStats {
    u64 proc_rss = 0;   // process working set / resident set, bytes
    u64 proc_cmt = 0;   // process private commit, bytes (Windows only)
    u64 sys_avail = 0;  // system available physical RAM, bytes
    u64 sys_total = 0;  // system total physical RAM, bytes
    u64 cmt_avail = 0;  // Windows: remaining system commit headroom, bytes
    u64 cmt_total = 0;  // Windows: system commit limit, bytes
    u64 swap_free = 0;  // Linux: free swap (incl. zram), bytes
    u64 swap_total = 0; // Linux: total swap, bytes
};

#ifdef _WIN32
static RamStats SampleRam() {
    RamStats r{};
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        r.proc_rss = pmc.WorkingSetSize;
        r.proc_cmt = pmc.PrivateUsage;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        r.sys_avail = ms.ullAvailPhys;
        r.sys_total = ms.ullTotalPhys;
        // "PageFile" fields are the system commit limit / remaining commit
        // charge - the number that actually runs out on Windows (VirtualAlloc
        // fails on commit exhaustion even with free physical RAM).
        r.cmt_avail = ms.ullAvailPageFile;
        r.cmt_total = ms.ullTotalPageFile;
    }
    return r;
}
#elif defined(__linux__)
static RamStats SampleRam() {
    RamStats r{};
    // /proc/self/statm field 2 = resident pages - the cheapest RSS source.
    // (VmSize is useless here: the guest address-space reservation makes it
    // enormous regardless of real usage, so only RSS is reported on Linux.)
    if (FILE* f = std::fopen("/proc/self/statm", "r")) {
        unsigned long long size_pages = 0, resident_pages = 0;
        if (std::fscanf(f, "%llu %llu", &size_pages, &resident_pages) == 2) {
            static const u64 page_size = static_cast<u64>(sysconf(_SC_PAGESIZE));
            r.proc_rss = static_cast<u64>(resident_pages) * page_size;
        }
        std::fclose(f);
    }
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[160];
        while (std::fgets(line, sizeof(line), f)) {
            unsigned long long kb = 0;
            if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) {
                r.sys_total = static_cast<u64>(kb) << 10;
            } else if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
                // MemAvailable is THE OOM-relevant number (already accounts
                // for reclaimable caches) - when it hits ~0 with swap
                // exhausted, the OOM killer fires.
                r.sys_avail = static_cast<u64>(kb) << 10;
            } else if (std::sscanf(line, "SwapTotal: %llu kB", &kb) == 1) {
                r.swap_total = static_cast<u64>(kb) << 10;
            } else if (std::sscanf(line, "SwapFree: %llu kB", &kb) == 1) {
                r.swap_free = static_cast<u64>(kb) << 10;
            }
        }
        std::fclose(f);
    }
    return r;
}
#else
static RamStats SampleRam() {
    return {}; // unsupported platform: fields print as 0, matching the
               // "0 if unsupported" convention of the optional callbacks
}
#endif

#ifdef _WIN32
// Thread stack dump on hang (Windows): suspends each other thread, captures register context,
// walks 32 frames, resumes. Symbols resolved via dbghelp.dll.

static std::atomic<bool> sym_initialized{false};

static void EnsureSymbolsInitialized() {
    if (sym_initialized.exchange(true)) return;
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    if (!SymInitialize(GetCurrentProcess(), nullptr, TRUE)) {
        LOG_WARNING(Common, "[HangWatchdog] SymInitialize failed (code={}), "
                            "stack dumps will be partial",
                    GetLastError());
        return;
    }

    // GR2FORK FIX: log which PDBs dbghelp found and matched per module, distinguishing a
    // rejected PDB (sig mismatch, stripped symbols) from one that was never searched.
    HMODULE mods[256]{};
    DWORD needed = 0;
    HANDLE proc = GetCurrentProcess();
    if (EnumProcessModules(proc, mods, sizeof(mods), &needed)) {
        const DWORD count = needed / sizeof(HMODULE);
        LOG_INFO(Common, "[HangWatchdog] Symbol diagnostic — {} modules loaded",
                 count);
        for (DWORD i = 0; i < count && i < 256; ++i) {
            char mod_name[MAX_PATH]{};
            GetModuleFileNameA(mods[i], mod_name, sizeof(mod_name));
            MODULEINFO mi{};
            GetModuleInformation(proc, mods[i], &mi, sizeof(mi));
            const DWORD64 base = reinterpret_cast<DWORD64>(mi.lpBaseOfDll);

            IMAGEHLP_MODULE64 info{};
            info.SizeOfStruct = sizeof(info);
            const BOOL ok = SymGetModuleInfo64(proc, base, &info);

            // Only log the interesting modules - shadps4 + any that have
            // partial/no symbols. Skip silent system-DLL noise.
            const char* sym_type =
                info.SymType == SymNone       ? "SymNone"       :
                info.SymType == SymCoff       ? "SymCoff"       :
                info.SymType == SymCv         ? "SymCv"         :
                info.SymType == SymPdb        ? "SymPdb"        :
                info.SymType == SymExport     ? "SymExport"     :
                info.SymType == SymDeferred   ? "SymDeferred"   :
                info.SymType == SymSym        ? "SymSym"        :
                info.SymType == SymDia        ? "SymDia"        :
                info.SymType == SymVirtual    ? "SymVirtual"    :
                                                "Unknown";

            const bool is_shadps4 =
                strstr(mod_name, "shadps4") != nullptr ||
                strstr(mod_name, "shadPS4") != nullptr;

            if (is_shadps4 || info.SymType == SymNone ||
                info.SymType == SymExport) {
                LOG_INFO(Common,
                         "[HangWatchdog]   module={} base={:#x} size={} "
                         "symType={} ok={} loadedPdb=\"{}\" loadedImage=\"{}\"",
                         mod_name, base, (u32)mi.SizeOfImage,
                         sym_type, ok != FALSE,
                         info.LoadedPdbName, info.LoadedImageName);
            }
        }
    } else {
        LOG_WARNING(Common, "[HangWatchdog] EnumProcessModules failed");
    }
}

static void DumpOneThread(DWORD tid) {
    HANDLE t = OpenThread(THREAD_GET_CONTEXT | THREAD_SUSPEND_RESUME |
                          THREAD_QUERY_INFORMATION,
                          FALSE, tid);
    if (!t) {
        LOG_WARNING(Common, "[HangWatchdog]   tid={} OpenThread failed (code={})",
                    tid, GetLastError());
        return;
    }

    if (SuspendThread(t) == (DWORD)-1) {
        LOG_WARNING(Common, "[HangWatchdog]   tid={} SuspendThread failed", tid);
        CloseHandle(t);
        return;
    }

    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_FULL;
    if (!GetThreadContext(t, &ctx)) {
        LOG_WARNING(Common, "[HangWatchdog]   tid={} GetThreadContext failed", tid);
        ResumeThread(t);
        CloseHandle(t);
        return;
    }

    // Thread name (Win10 1607+). Ignored if unavailable.
    PWSTR thread_name_w = nullptr;
    std::string thread_name;
    if (GetThreadDescription(t, &thread_name_w) == S_OK && thread_name_w) {
        char buf[128]{};
        WideCharToMultiByte(CP_UTF8, 0, thread_name_w, -1, buf, sizeof(buf)-1,
                            nullptr, nullptr);
        thread_name = buf;
        LocalFree(thread_name_w);
    }
    // Fallback: process-local registry populated by SetCurrentThreadName, for builds where
    // SetThreadDescription is a no-op (MinGW) or GetThreadDescription returns an empty string.
    if (thread_name.empty()) {
        thread_name = Common::GetThreadNameByTid(static_cast<u32>(tid));
    }

    LOG_CRITICAL(Common, "[HangWatchdog] --- tid={} name=\"{}\" rip={:#x} rsp={:#x}",
                 tid, thread_name, (u64)ctx.Rip, (u64)ctx.Rsp);

    STACKFRAME64 frame{};
    frame.AddrPC.Offset    = ctx.Rip;   frame.AddrPC.Mode    = AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Rbp;   frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Offset = ctx.Rsp;   frame.AddrStack.Mode = AddrModeFlat;

    HANDLE proc = GetCurrentProcess();
    for (int i = 0; i < 32; ++i) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, t, &frame, &ctx,
                         nullptr, SymFunctionTableAccess64, SymGetModuleBase64,
                         nullptr)) {
            break;
        }
        if (frame.AddrPC.Offset == 0) break;

        alignas(SYMBOL_INFO) char sym_buf[sizeof(SYMBOL_INFO) + 512]{};
        SYMBOL_INFO* sym = reinterpret_cast<SYMBOL_INFO*>(sym_buf);
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen   = 512;

        DWORD64 disp = 0;
        IMAGEHLP_MODULE64 mod{};
        mod.SizeOfStruct = sizeof(mod);
        SymGetModuleInfo64(proc, frame.AddrPC.Offset, &mod);

        if (SymFromAddr(proc, frame.AddrPC.Offset, &disp, sym)) {
            IMAGEHLP_LINE64 line{};
            line.SizeOfStruct = sizeof(line);
            DWORD line_disp = 0;
            if (SymGetLineFromAddr64(proc, frame.AddrPC.Offset, &line_disp, &line)) {
                LOG_CRITICAL(Common, "[HangWatchdog]   #{:02} {:#x} {}!{}+{:#x}  ({}:{})",
                             i, (u64)frame.AddrPC.Offset, mod.ModuleName,
                             sym->Name, disp, line.FileName, line.LineNumber);
            } else {
                LOG_CRITICAL(Common, "[HangWatchdog]   #{:02} {:#x} {}!{}+{:#x}",
                             i, (u64)frame.AddrPC.Offset, mod.ModuleName,
                             sym->Name, disp);
            }
        } else {
            LOG_CRITICAL(Common, "[HangWatchdog]   #{:02} {:#x} {}!<unresolved>",
                         i, (u64)frame.AddrPC.Offset, mod.ModuleName);
        }
    }

    ResumeThread(t);
    CloseHandle(t);
}

static void DumpAllOtherThreads() {
    EnsureSymbolsInitialized();

    const DWORD our_pid = GetCurrentProcessId();
    const DWORD our_tid = GetCurrentThreadId();

    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) {
        LOG_WARNING(Common, "[HangWatchdog] CreateToolhelp32Snapshot failed");
        return;
    }

    THREADENTRY32 te{};
    te.dwSize = sizeof(te);
    int dumped = 0;
    if (Thread32First(snap, &te)) {
        do {
            if (te.th32OwnerProcessID != our_pid) continue;
            if (te.th32ThreadID == our_tid) continue;
            DumpOneThread(te.th32ThreadID);
            ++dumped;
        } while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);

    LOG_CRITICAL(Common, "[HangWatchdog] --- end stack dump ({} threads) ---", dumped);
}
#else
static void DumpAllOtherThreads() {
    LOG_WARNING(Common, "[HangWatchdog] Thread stack dump not implemented on this platform");
}
#endif

static void WatchdogLoop() {
    Common::SetCurrentThreadName("shadPS4:HangWatchdog");
    // GR2FORK PERF: fence the watchdog off the GpuComm/GpuAssembler reserved cores at entry -
    // the exclusion walk only strips this thread once Liverpool::Process runs it.
    // GetExclusionStripMask carries the walk's own gating, so this no-ops wherever the walk does.
    if (const u64 strip = Common::GetExclusionStripMask()) {
        Common::SetCurrentThreadAffinityMask(~strip);
    }
    auto& s = S();
    u64 last_tick    = 0;
    u64 last_kick    = 0;
    u64 last_vram    = 0;
    u32 last_pending = 0;
    auto last_change = std::chrono::steady_clock::now();
    bool warned  = false;
    bool critted = false;
    bool low_ram_warned = false;
    bool low_cmt_warned = false;

    LOG_INFO(Common,
             "[HangWatchdog] started — sampling every {}s, warn at {}s stall, critical at {}s",
             std::chrono::duration_cast<std::chrono::seconds>(kSamplePeriod).count(),
             std::chrono::duration_cast<std::chrono::seconds>(kStallWarnThreshold).count(),
             std::chrono::duration_cast<std::chrono::seconds>(kStallCriticalThreshold).count());

    while (s.running.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(kSamplePeriod);
        if (!s.running.load(std::memory_order_acquire)) break;

        const u64 tick      = s.cb.scheduler_tick ? s.cb.scheduler_tick() : 0;
        const u32 pending   = s.cb.num_submits    ? s.cb.num_submits()    : 0;
        const u64 vram_used = s.cb.vram_used      ? s.cb.vram_used()      : 0;
        const u64 vram_bud  = s.cb.vram_budget    ? s.cb.vram_budget()    : 0;
        const u64 tex_mem   = s.cb.texture_mem    ? s.cb.texture_mem()    : 0;
        const size_t nimg   = s.cb.num_images     ? s.cb.num_images()     : 0;
        const size_t nbuf   = s.cb.num_buffers    ? s.cb.num_buffers()    : 0;
        const u64 kick      = s.main_kick_count.load(std::memory_order_relaxed);

        const double vram_pct = vram_bud ? (100.0 * static_cast<double>(vram_used) /
                                                  static_cast<double>(vram_bud))
                                         : 0.0;

        // Host RAM snapshot for this tick. Formatted into a small suffix
        // string once so the tick line AND the stall snapshot below carry
        // the identical fields without duplicating the platform #ifdefs.
        const RamStats ram = SampleRam();
        char ram_buf[160];
#ifdef _WIN32
        std::snprintf(ram_buf, sizeof(ram_buf),
                      "ws=%llu commit=%llu MiB sysFree=%llu/%llu MiB cmtFree=%llu/%llu MiB",
                      static_cast<unsigned long long>(ram.proc_rss >> 20),
                      static_cast<unsigned long long>(ram.proc_cmt >> 20),
                      static_cast<unsigned long long>(ram.sys_avail >> 20),
                      static_cast<unsigned long long>(ram.sys_total >> 20),
                      static_cast<unsigned long long>(ram.cmt_avail >> 20),
                      static_cast<unsigned long long>(ram.cmt_total >> 20));
#elif defined(__linux__)
        std::snprintf(ram_buf, sizeof(ram_buf),
                      "rss=%llu MiB sysFree=%llu/%llu MiB swapFree=%llu/%llu MiB",
                      static_cast<unsigned long long>(ram.proc_rss >> 20),
                      static_cast<unsigned long long>(ram.sys_avail >> 20),
                      static_cast<unsigned long long>(ram.sys_total >> 20),
                      static_cast<unsigned long long>(ram.swap_free >> 20),
                      static_cast<unsigned long long>(ram.swap_total >> 20));
#else
        std::snprintf(ram_buf, sizeof(ram_buf), "unavailable");
#endif

        LOG_INFO(Common,
                 "[HangWatchdog] tick={} kick={} pending_submits={} "
                 "vram={}/{} MiB ({:.1f}%) tex_mem={} MiB nimg={} nbuf={} ram: {}",
                 tick, kick, pending,
                 vram_used >> 20, vram_bud >> 20, vram_pct,
                 tex_mem >> 20, nimg, nbuf, ram_buf);

        const bool alive = (tick != last_tick) || (kick != last_kick);
        const auto now = std::chrono::steady_clock::now();

        if (alive) {
            last_change = now;
            warned = false;
            critted = false;
        } else {
            const auto stalled_for = now - last_change;
            const auto stalled_s =
                std::chrono::duration_cast<std::chrono::seconds>(stalled_for).count();

            // GR2FORK FIX: dump stacks at the WARN threshold. A driver TDR returns DEVICE_LOST
            // from the presenter's infinite vkWaitForFences and its ASSERT_MSG tears the process
            // down ~10s in (RTX 3050 Ti / NV 580.88) - before the 15s critical dump could fire.
            if (stalled_for >= kStallWarnThreshold && !warned) {
                LOG_CRITICAL(Common,
                             "[HangWatchdog] *** STALL DETECTED *** scheduler tick stuck at {} "
                             "for {}s. pending_submits={} (Δ since stall-start: {}). "
                             "vram={}/{} MiB ({:.1f}%). ram: {}. Dumping all thread stacks now "
                             "(before the presenter ASSERT path can fire on TDR).",
                             tick, stalled_s, pending,
                             static_cast<s32>(pending) - static_cast<s32>(last_pending),
                             vram_used >> 20, vram_bud >> 20, vram_pct, ram_buf);
                DumpAllOtherThreads();
                // Sync-trace ring + active waiters. Shows the signal graph
                // leading into the hang and which tid is blocked on which
                // EF/Sema object.
                Common::SyncTrace::Dump();
                warned = true;
            }

            // Re-dump at the critical threshold: the same threads pinned on the same RIPs after
            // a 10s gap distinguishes a true deadlock from slow forward progress.
            if (stalled_for >= kStallCriticalThreshold && !critted) {
                LOG_CRITICAL(Common,
                             "[HangWatchdog] *** HANG PERSISTS *** scheduler tick still stuck at {} "
                             "after {}s. pending_submits={}. Re-dumping for delta diagnosis.",
                             tick, stalled_s, pending);
                DumpAllOtherThreads();
                Common::SyncTrace::Dump();
                critted = true;
            }
        }

        if (tick == last_tick && pending > last_pending && pending > 4) {
            LOG_WARNING(Common,
                        "[HangWatchdog] GPU thread drain stall suspected — "
                        "pending submits climbing ({} -> {}) with no tick advance.",
                        last_pending, pending);
        }

        if (vram_bud > 0 && vram_pct > 90.0 && vram_used > last_vram + (16ULL << 20)) {
            LOG_WARNING(Common,
                        "[HangWatchdog] VRAM climbing past {:.1f}% — {} -> {} MiB "
                        "(+{} MiB since last sample). Expect OOM soon.",
                        vram_pct, last_vram >> 20, vram_used >> 20,
                        (vram_used - last_vram) >> 20);
        }

        // GR2FORK FIX: host-RAM exhaustion warnings, edge-triggered with 25% re-arm hysteresis
        // so a system hovering at the threshold does not spam at 1 Hz. An OOM kill or commit
        // failure right after one of these lines is memory exhaustion, not a code crash.
        if (ram.sys_total > 0) {
            const u64 low_ram_threshold =
                std::max<u64>(512ULL << 20, ram.sys_total / 20); // 512 MiB or 5%
            if (!low_ram_warned && ram.sys_avail < low_ram_threshold) {
                LOG_WARNING(Common,
                            "[HangWatchdog] *** LOW SYSTEM RAM *** {}/{} MiB available "
                            "({:.1f}%), process rss={} MiB. If the process dies with NO "
                            "crash-handler output shortly after this line, the OS killed it "
                            "(Linux OOM-kill = uncatchable SIGKILL; Windows commit failure = "
                            "__fastfail) — memory exhaustion, not a code crash.",
                            ram.sys_avail >> 20, ram.sys_total >> 20,
                            100.0 * static_cast<double>(ram.sys_avail) /
                                static_cast<double>(ram.sys_total),
                            ram.proc_rss >> 20);
                low_ram_warned = true;
            } else if (low_ram_warned &&
                       ram.sys_avail > low_ram_threshold + (low_ram_threshold >> 2)) {
                low_ram_warned = false;
            }
        }
        if (ram.cmt_total > 0) { // Windows only - fields stay 0 elsewhere
            const u64 low_cmt_threshold =
                std::max<u64>(512ULL << 20, ram.cmt_total / 20);
            if (!low_cmt_warned && ram.cmt_avail < low_cmt_threshold) {
                LOG_WARNING(Common,
                            "[HangWatchdog] *** LOW COMMIT *** {}/{} MiB system commit "
                            "headroom left (process commit={} MiB). Allocation failures are "
                            "imminent; their failure paths can end in a silent __fastfail.",
                            ram.cmt_avail >> 20, ram.cmt_total >> 20, ram.proc_cmt >> 20);
                low_cmt_warned = true;
            } else if (low_cmt_warned &&
                       ram.cmt_avail > low_cmt_threshold + (low_cmt_threshold >> 2)) {
                low_cmt_warned = false;
            }
        }

        last_tick    = tick;
        last_kick    = kick;
        last_vram    = vram_used;
        last_pending = pending;
    }

    LOG_INFO(Common, "[HangWatchdog] stopped cleanly");
}

} // namespace

void HangWatchdog::Start(HangWatchdogCallbacks cb) {
    auto& s = S();
    std::lock_guard lk{s.start_mu};
    if (s.running.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    s.cb = std::move(cb);
    s.thread = std::thread(&WatchdogLoop);
}

void HangWatchdog::Stop() {
    auto& s = S();
    std::lock_guard lk{s.start_mu};
    if (!s.running.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    if (s.thread.joinable()) {
        s.thread.join();
    }
    s.cb = {};
}

void HangWatchdog::KickMainThread() noexcept {
    S().main_kick_count.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Common
