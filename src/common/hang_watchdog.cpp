// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/hang_watchdog.h"
#include "common/sync_trace.h"
#include "common/thread.h"

#include <atomic>
#include <chrono>
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

#ifdef _WIN32
// ---------------------------------------------------------------------------
// Thread stack dump on hang (Windows). Called once when critical threshold
// is crossed. Suspends each other thread, captures register context, walks
// 32 frames, resumes. Symbols resolved via dbghelp.dll.
// ---------------------------------------------------------------------------

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

    // FIX(GR2FORK): diagnostic — enumerate loaded modules and log which PDBs
    // dbghelp actually found and matched. Without this we cannot tell whether
    // a PDB near the exe is being rejected (sig mismatch, stripped symbols)
    // or merely not searched. For the shadps4 module specifically we dump
    // the PDB path and whether symbols loaded.
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

            // Only log the interesting modules — shadps4 + any that have
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
    // Fallback: process-local registry populated by SetCurrentThreadName.
    // Defends against build configs where SetThreadDescription compiles to a
    // no-op (the MinGW path before this fix), and against any future Windows
    // API quirks where GetThreadDescription returns an empty string despite
    // a name having been set.
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
    auto& s = S();
    u64 last_tick    = 0;
    u64 last_kick    = 0;
    u64 last_vram    = 0;
    u32 last_pending = 0;
    auto last_change = std::chrono::steady_clock::now();
    bool warned  = false;
    bool critted = false;

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

        LOG_INFO(Common,
                 "[HangWatchdog] tick={} kick={} pending_submits={} "
                 "vram={}/{} MiB ({:.1f}%) tex_mem={} MiB nimg={} nbuf={}",
                 tick, kick, pending,
                 vram_used >> 20, vram_bud >> 20, vram_pct,
                 tex_mem >> 20, nimg, nbuf);

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

            // FIX(GR2FORK): dump stacks at the WARN threshold, not just at
            // CRITICAL. The presenter's vkWaitForFences uses u64::max timeout,
            // so the only way it returns from the wait is success or
            // VK_ERROR_DEVICE_LOST (driver TDR). Observed on RTX 3050 Ti
            // Laptop / NV 580.88: the GPU hangs, NV TDR fires, the wait
            // returns DEVICE_LOST, and vk_presenter.cpp:903 ASSERT_MSG tears
            // the process down ~10s into the stall — well before the previous
            // 15s critical threshold could fire its dump. Capturing at 5s
            // gets the snapshot in before that cascade. Note: this also
            // produces dumps for transient 5–14s stalls that self-recover,
            // which is desirable: those hitches are also worth diagnosing.
            if (stalled_for >= kStallWarnThreshold && !warned) {
                LOG_CRITICAL(Common,
                             "[HangWatchdog] *** STALL DETECTED *** scheduler tick stuck at {} "
                             "for {}s. pending_submits={} (Δ since stall-start: {}). "
                             "vram={}/{} MiB ({:.1f}%). Dumping all thread stacks now "
                             "(before the presenter ASSERT path can fire on TDR).",
                             tick, stalled_s, pending,
                             static_cast<s32>(pending) - static_cast<s32>(last_pending),
                             vram_used >> 20, vram_bud >> 20, vram_pct);
                DumpAllOtherThreads();
                // Sync-trace ring + active waiters. Shows the signal graph
                // leading into the hang and which tid is blocked on which
                // EF/Sema object.
                Common::SyncTrace::Dump();
                warned = true;
            }

            // Re-dump at the original critical threshold for a delta snapshot.
            // If the same threads are still pinned on the same RIPs after a
            // 10s gap, that's hard evidence of a true deadlock vs. just slow
            // forward progress. If they've moved, the stall is unwinding
            // (and the process is probably already torn down by ASSERT_MSG —
            // but on builds where we choose to survive past device-lost,
            // this gives the second data point).
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
