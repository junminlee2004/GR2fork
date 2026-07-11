// SPDX-FileCopyrightText: Copyright 2025 GR2FORK
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/crash_handler.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>

#include "common/logging/backend.h"
#include "common/logging/log.h"

#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#include <io.h>
#include <psapi.h>
#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "psapi.lib")
#else
// GR2FORK FIX: execinfo-based backtrace for Linux OnTerminate. Printing only the exception type
// leaves static_vector overflow, heap corruption, and real OOM indistinguishable (all
// `boost::container::bad_alloc`) with no trace of the throw site.
#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <cstring>

// GR2FORK FIX: capture the backtrace at throw time - the stack is already unwound when
// std::terminate runs, so backtrace() there only sees invoker plumbing. Override __cxa_throw,
// snapshot frames per thread (__thread avoids dynamic-TLS init that can itself throw), forward on.
namespace Common::CrashHandler {
namespace {
constexpr int kThrowFrameCap = 64;
__thread void* g_throw_frames[kThrowFrameCap];
__thread int   g_throw_nframes = 0;
__thread char  g_throw_type_name[256] = {};
} // namespace
} // namespace Common::CrashHandler

extern "C" {
using cxa_throw_fn = void (*)(void*, std::type_info*, void (*)(void*));

// noreturn: the real __cxa_throw never returns, so neither do we.
__attribute__((noreturn))
void __cxa_throw(void* obj, std::type_info* tinfo, void (*dest)(void*)) {
    using namespace Common::CrashHandler;
    // Snapshot the throw site BEFORE the unwinder runs.
    g_throw_nframes = ::backtrace(g_throw_frames, kThrowFrameCap);
    if (tinfo && tinfo->name()) {
        std::strncpy(g_throw_type_name, tinfo->name(),
                     sizeof(g_throw_type_name) - 1);
        g_throw_type_name[sizeof(g_throw_type_name) - 1] = '\0';
    } else {
        g_throw_type_name[0] = '\0';
    }

    // Resolve the real __cxa_throw once, then cache. RTLD_NEXT walks the
    // dynamic loader's lookup order past us (the executable) and finds
    // libstdc++'s implementation.
    static cxa_throw_fn real = nullptr;
    if (!real) {
        real = reinterpret_cast<cxa_throw_fn>(::dlsym(RTLD_NEXT, "__cxa_throw"));
    }
    real(obj, tinfo, dest);
    __builtin_unreachable();
}
} // extern "C"
#endif

namespace Common::CrashHandler {

namespace {

std::atomic<bool> g_installed{false};

// GR2FORK FIX: per-thread reentry guard - a process-global atomic that never clears (e.g.
// LogCurrentThreadStack blocks on the backend-thread join) silences every later handler on every
// thread. Thread-local still blocks same-thread re-entry (AV inside SymFromAddr re-entering VEH).
thread_local bool t_reentry_guard = false;

// Set by emulator.cpp before an intentional std::quick_exit(0); OnAtQuickExit then logs a short
// "clean shutdown" note instead of recording every routine quit as a crash.
std::atomic<bool> g_clean_shutdown{false};

// Dedicated crash-dump file, opened lazily on first crash and kept open so fflush +
// FlushFileBuffers can force bytes to disk before the process dies.
FILE* g_crash_fp = nullptr;

// Synchronous write + flush. Used from inside crash handlers where we
// cannot trust the async log worker to drain the MPSCQueue before the
// process terminates.
void CrashPrintf(const char* fmt, ...) {
    if (!g_crash_fp) {
        // Put the crash dump next to the exe so it's easy to find.
        // 4096 covers MAX_PATH (260) and PATH_MAX (typically 4096) without
        // needing platform-specific headers here.
        constexpr size_t kPathMax = 4096;
        char exe_path[kPathMax]{};
#ifdef _WIN32
        GetModuleFileNameA(nullptr, exe_path, static_cast<DWORD>(kPathMax));
        // Strip the filename.
        char* last_slash = nullptr;
        for (char* p = exe_path; *p; ++p) {
            if (*p == '\\' || *p == '/') last_slash = p;
        }
        if (last_slash) *(last_slash + 1) = '\0';
        else exe_path[0] = '\0';
#else
        // On Linux/macOS we just write crash_dump.txt to CWD; readlink on
        // /proc/self/exe would also work but CWD is fine for this use.
        exe_path[0] = '\0';
#endif
        char crash_path[kPathMax + 64]{};
        std::snprintf(crash_path, sizeof(crash_path),
                      "%scrash_dump.txt", exe_path);
        g_crash_fp = std::fopen(crash_path, "ab");
        if (!g_crash_fp) {
            // Fall back to CWD.
            g_crash_fp = std::fopen("crash_dump.txt", "ab");
        }
        if (g_crash_fp) {
            // Time-stamped banner.
            std::time_t now = std::time(nullptr);
            std::tm tmv{};
#ifdef _WIN32
            localtime_s(&tmv, &now);
#else
            localtime_r(&now, &tmv);
#endif
            std::fprintf(
                g_crash_fp,
                "\n\n================= CRASH %04d-%02d-%02d %02d:%02d:%02d =================\n",
                tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
        }
    }

    va_list args;
    va_start(args, fmt);
    if (g_crash_fp) {
        va_list copy;
        va_copy(copy, args);
        std::vfprintf(g_crash_fp, fmt, copy);
        va_end(copy);
        std::fflush(g_crash_fp);
#ifdef _WIN32
        // Force the bytes to the disk cache - fflush only moves them from
        // the CRT buffer to the kernel. FlushFileBuffers is what actually
        // pushes them to disk.
        HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(g_crash_fp)));
        if (h != INVALID_HANDLE_VALUE) FlushFileBuffers(h);
#endif
    }
    // Mirror to shad_log.txt, flipping the backend to synchronous mode first - an async
    // LOG_CRITICAL only enqueues, and the consumer thread races the dying process. After
    // StopAsyncAndForceSync(), PushEntry writes inline with fflush.
    Common::Log::StopAsyncAndForceSync();
    char msg[1024]{};
    std::vsnprintf(msg, sizeof(msg), fmt, args);
    // Strip trailing newline for LOG_CRITICAL.
    size_t n = std::strlen(msg);
    while (n > 0 && (msg[n-1] == '\n' || msg[n-1] == '\r')) msg[--n] = '\0';
    LOG_CRITICAL(Common, "[CrashHandler] {}", msg);
    va_end(args);
}

#ifdef _WIN32

// Symbol initialization. The crash handler owns its own symbol context so traces work before
// HangWatchdog starts (a second SymInitialize is a no-op that preserves state). An explicit search
// path (exe dir + CWD + _NT_SYMBOL_PATH) keeps dbghelp off the useless public-export fallback.

static std::atomic<bool> g_sym_inited{false};

static void BuildSymbolSearchPath(char* out, size_t cap) {
    out[0] = '\0';
    auto append = [&](const char* s) {
        if (!s || !*s) return;
        size_t len = std::strlen(out);
        if (len > 0 && len + 1 < cap) {
            out[len++] = ';';
            out[len] = '\0';
        }
        size_t room = (cap > len + 1) ? (cap - len - 1) : 0;
        std::strncat(out, s, room);
    };

    // 1) exe directory
    char exe_path[MAX_PATH]{};
    GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    char* last_slash = nullptr;
    for (char* p = exe_path; *p; ++p) {
        if (*p == '\\' || *p == '/') last_slash = p;
    }
    if (last_slash) { *last_slash = '\0'; append(exe_path); }

    // 2) CWD
    char cwd[MAX_PATH]{};
    if (GetCurrentDirectoryA(MAX_PATH, cwd) > 0) append(cwd);

    // 3) _NT_SYMBOL_PATH
    char env_path[4096]{};
    DWORD got = GetEnvironmentVariableA("_NT_SYMBOL_PATH", env_path,
                                        sizeof(env_path));
    if (got > 0 && got < sizeof(env_path)) append(env_path);
}

static bool SymInitSafe() {
    if (g_sym_inited.exchange(true)) return true;
    const HANDLE proc = GetCurrentProcess();
    SymSetOptions(SymGetOptions() |
                  SYMOPT_LOAD_LINES | SYMOPT_UNDNAME |
                  SYMOPT_DEFERRED_LOADS | SYMOPT_FAIL_CRITICAL_ERRORS);
    char search_path[8192]{};
    BuildSymbolSearchPath(search_path, sizeof(search_path));
    const char* sp = search_path[0] ? search_path : nullptr;
    // fInvadeProcess=TRUE auto-enumerates loaded modules. If another component (e.g. HangWatchdog)
    // already called SymInitialize, the call below is a no-op that keeps the old search path, so
    // force the path via SymSetSearchPath and re-resolve every module regardless of init order.
    SymInitialize(proc, sp, TRUE);
    if (sp) {
        SymSetSearchPath(proc, sp);
    }
    SymRefreshModuleList(proc);
    return true;
}

// Log whether the main module (shadps4.exe) actually got a PDB.
// Run once during Install() so the regular log records whether symbols
// are usable before any crash happens.
static void LogSymbolDiagnostic() {
    HANDLE proc = GetCurrentProcess();
    HMODULE mods[256]{};
    DWORD needed = 0;
    if (!EnumProcessModules(proc, mods, sizeof(mods), &needed)) return;
    const DWORD count = needed / sizeof(HMODULE);
    for (DWORD i = 0; i < count && i < 256; ++i) {
        char mod_name[MAX_PATH]{};
        GetModuleFileNameA(mods[i], mod_name, sizeof(mod_name));
        const bool is_shadps4 =
            std::strstr(mod_name, "shadps4") != nullptr ||
            std::strstr(mod_name, "shadPS4") != nullptr ||
            std::strstr(mod_name, "gr2fork") != nullptr ||
            std::strstr(mod_name, "gr2Fork") != nullptr;
        if (!is_shadps4) continue;

        MODULEINFO mi{};
        GetModuleInformation(proc, mods[i], &mi, sizeof(mi));
        const DWORD64 base = reinterpret_cast<DWORD64>(mi.lpBaseOfDll);

        IMAGEHLP_MODULE64 info{};
        info.SizeOfStruct = sizeof(info);
        if (SymGetModuleInfo64(proc, base, &info)) {
            const char* sym_type =
                info.SymType == SymNone     ? "SymNone"     :
                info.SymType == SymExport   ? "SymExport"   :
                info.SymType == SymPdb      ? "SymPdb"      :
                info.SymType == SymDeferred ? "SymDeferred" :
                                              "Other";
            const bool pdb_ok =
                info.LoadedPdbName[0] != '\0' &&
                (info.SymType == SymPdb || info.SymType == SymDeferred);
            LOG_INFO(Common,
                     "[CrashHandler] shadps4 module PDB status: "
                     "symType={} loadedPdb=\"{}\" (usable_symbols={}). "
                     "If usable_symbols=false, place shadps4.pdb next to "
                     "shadps4.exe for readable crash stacks.",
                     sym_type, info.LoadedPdbName, pdb_ok);
        }
    }
}

// Stack walking. Offsets past the nearest public export > 0x40000 bytes almost certainly mean
// dbghelp matched the wrong symbol (export-table fallback, no PDB); drop the symbol name and
// emit "module!+<rva>" so the output stays actionable.
constexpr DWORD64 kMaxBelievableDisp = 0x40000; // 256 KB

void LogCurrentThreadStack(const char* cause, const CONTEXT* fault_ctx = nullptr) {
    if (t_reentry_guard) {
        return; // a handler is already running on this thread
    }
    t_reentry_guard = true;

    // Make sure symbols are available and fresh (in case DLLs loaded
    // after our Install).
    SymInitSafe();
    SymRefreshModuleList(GetCurrentProcess());

    HANDLE proc = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    const DWORD tid = GetCurrentThreadId();

    CrashPrintf("*** %s *** tid=%lu\n", cause, (unsigned long)tid);

    // GR2FORK FIX: name the faulting thread in the dump. Every emulator thread is named via
    // SetCurrentThreadName (SetThreadDescription), and "which thread was this" has cost multiple
    // device-recovery investigations a full repro round-trip.
    {
        PWSTR tdesc = nullptr;
        if (SUCCEEDED(GetThreadDescription(thread, &tdesc)) && tdesc != nullptr) {
            char narrow[128]{};
            WideCharToMultiByte(CP_UTF8, 0, tdesc, -1, narrow, sizeof(narrow) - 1, nullptr,
                                nullptr);
            LocalFree(tdesc);
            if (narrow[0] != '\0') {
                CrashPrintf("thread name: \"%s\"\n", narrow);
            }
        }
    }

    // GR2FORK FIX: walk the FAULTING context when the caller has one - walking
    // our own captured context prints only these handler frames (constant
    // RVAs), which masked the true crash site across three device-recovery
    // investigations.
    CONTEXT ctx{};
    if (fault_ctx != nullptr) {
        ctx = *fault_ctx;
    } else {
        ctx.ContextFlags = CONTEXT_FULL;
        RtlCaptureContext(&ctx);
    }

    STACKFRAME64 frame{};
    DWORD machine = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset    = ctx.Rip;
    frame.AddrPC.Mode      = AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Rbp;
    frame.AddrFrame.Mode   = AddrModeFlat;
    frame.AddrStack.Offset = ctx.Rsp;
    frame.AddrStack.Mode   = AddrModeFlat;

    alignas(SYMBOL_INFO) char sym_buf[sizeof(SYMBOL_INFO) + 512]{};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(sym_buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen   = 512;

    for (int i = 0; i < 48; ++i) {
        if (!StackWalk64(machine, proc, thread, &frame, &ctx,
                         nullptr, SymFunctionTableAccess64,
                         SymGetModuleBase64, nullptr)) {
            break;
        }
        if (frame.AddrPC.Offset == 0) break;

        const DWORD64 pc = frame.AddrPC.Offset;

        // Resolve module for the frame - lets us fall back to
        // "module!+<rva>" when the symbol offset is unbelievable.
        char mod_short[64] = "?";
        DWORD64 mod_base = 0;
        IMAGEHLP_MODULE64 mi{};
        mi.SizeOfStruct = sizeof(mi);
        if (SymGetModuleInfo64(proc, pc, &mi)) {
            mod_base = mi.BaseOfImage;
            std::snprintf(mod_short, sizeof(mod_short), "%s", mi.ModuleName);
        }

        DWORD64 disp = 0;
        const BOOL ok = SymFromAddr(proc, pc, &disp, sym);
        DWORD line_disp = 0;
        IMAGEHLP_LINE64 line{};
        line.SizeOfStruct = sizeof(line);
        const BOOL have_line = SymGetLineFromAddr64(proc, pc, &line_disp, &line);

        // Treat huge offsets as "nearest-export fallback, not a real name".
        const bool name_believable = ok && disp <= kMaxBelievableDisp;

        if (name_believable && have_line) {
            CrashPrintf("  #%02d 0x%llx %s!%s+0x%llx  (%s:%lu)\n",
                        i, (unsigned long long)pc,
                        mod_short, sym->Name, (unsigned long long)disp,
                        line.FileName, (unsigned long)line.LineNumber);
        } else if (name_believable) {
            CrashPrintf("  #%02d 0x%llx %s!%s+0x%llx\n",
                        i, (unsigned long long)pc,
                        mod_short, sym->Name, (unsigned long long)disp);
        } else if (mod_base) {
            // No usable symbol, but we know the module - emit RVA.
            CrashPrintf("  #%02d 0x%llx %s!+0x%llx\n",
                        i, (unsigned long long)pc,
                        mod_short, (unsigned long long)(pc - mod_base));
        } else {
            CrashPrintf("  #%02d 0x%llx <unresolved>\n",
                        i, (unsigned long long)pc);
        }
    }

    CrashPrintf("==== end stack ====\n");
    // Keep guard set; process is about to die.
}

const char* SehCodeName(DWORD code) {
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:         return "ACCESS_VIOLATION";
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:    return "ARRAY_BOUNDS_EXCEEDED";
    case EXCEPTION_BREAKPOINT:               return "BREAKPOINT (int3 — probably ASSERT_MSG path)";
    case EXCEPTION_DATATYPE_MISALIGNMENT:    return "DATATYPE_MISALIGNMENT";
    case EXCEPTION_FLT_DENORMAL_OPERAND:     return "FLT_DENORMAL_OPERAND";
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:       return "FLT_DIVIDE_BY_ZERO";
    case EXCEPTION_FLT_INEXACT_RESULT:       return "FLT_INEXACT_RESULT";
    case EXCEPTION_FLT_INVALID_OPERATION:    return "FLT_INVALID_OPERATION";
    case EXCEPTION_FLT_OVERFLOW:             return "FLT_OVERFLOW";
    case EXCEPTION_FLT_STACK_CHECK:          return "FLT_STACK_CHECK";
    case EXCEPTION_FLT_UNDERFLOW:            return "FLT_UNDERFLOW";
    case EXCEPTION_ILLEGAL_INSTRUCTION:      return "ILLEGAL_INSTRUCTION";
    case EXCEPTION_IN_PAGE_ERROR:            return "IN_PAGE_ERROR";
    case EXCEPTION_INT_DIVIDE_BY_ZERO:       return "INT_DIVIDE_BY_ZERO";
    case EXCEPTION_INT_OVERFLOW:             return "INT_OVERFLOW";
    case EXCEPTION_INVALID_DISPOSITION:      return "INVALID_DISPOSITION";
    case EXCEPTION_NONCONTINUABLE_EXCEPTION: return "NONCONTINUABLE_EXCEPTION";
    case EXCEPTION_PRIV_INSTRUCTION:         return "PRIV_INSTRUCTION";
    case EXCEPTION_SINGLE_STEP:              return "SINGLE_STEP";
    case EXCEPTION_STACK_OVERFLOW:           return "STACK_OVERFLOW";
    case 0xE06D7363:                         return "CPP_EH_EXCEPTION";
    // __fastfail family - these bypass SEH/UEF and kill the process
    // directly. VEH is the only user-mode catch point before the
    // kernel tears the process down.
    case 0xC0000374:                         return "HEAP_CORRUPTION (fastfail)";
    case 0xC0000409:                         return "STACK_BUFFER_OVERRUN / /GS (fastfail)";
    case 0xC0000417:                         return "INVALID_CRUNTIME_PARAMETER (fastfail)";
    case 0xC0000602:                         return "FAIL_FAST_EXCEPTION (__fastfail)";
    default:                                 return "UNKNOWN";
    }
}

LONG WINAPI OnUnhandledSEH(EXCEPTION_POINTERS* info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    const void* addr = info->ExceptionRecord->ExceptionAddress;

    if (code == 0xE06D7363) {
        return EXCEPTION_CONTINUE_SEARCH; // handled by std::set_terminate
    }

    char detail[256]{};
    if (code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_IN_PAGE_ERROR) {
        const auto type = info->ExceptionRecord->ExceptionInformation[0];
        const auto va   = info->ExceptionRecord->ExceptionInformation[1];
        const char* op =
            type == 0 ? "read" :
            type == 1 ? "write" :
            type == 8 ? "execute" : "?";
        std::snprintf(detail, sizeof(detail),
                      "SEH %s at %p (%s %#llx)",
                      SehCodeName(code), addr, op,
                      static_cast<unsigned long long>(va));
    } else {
        std::snprintf(detail, sizeof(detail), "SEH %s at %p",
                      SehCodeName(code), addr);
    }
    if (info->ContextRecord != nullptr) {
        const CONTEXT fault_ctx_copy = *info->ContextRecord;
        LogCurrentThreadStack(detail, &fault_ctx_copy);
    } else {
        LogCurrentThreadStack(detail);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

// GR2FORK FIX: the vectored handler catches only EXCEPTION_STACK_OVERFLOW (UEF has no stack room
// to run) and the __fastfail family, which the kernel terminates before UEF sees it. Everything
// else stays on SEH/UEF - logging every AV here would flag shadPS4's own guest-memory handlers.
LONG WINAPI OnVectoredException(EXCEPTION_POINTERS* info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    const bool is_fastfail =
        code == 0xC0000374 || // HEAP_CORRUPTION
        code == 0xC0000409 || // STACK_BUFFER_OVERRUN (/GS)
        code == 0xC0000417 || // INVALID_CRUNTIME_PARAMETER (secure CRT)
        code == 0xC0000602;   // FAIL_FAST_EXCEPTION
    if (code == EXCEPTION_STACK_OVERFLOW || is_fastfail) {
        char detail[256]{};
        std::snprintf(detail, sizeof(detail),
                      "VEH-early %s at %p",
                      SehCodeName(code),
                      info->ExceptionRecord->ExceptionAddress);
        // GR2FORK FIX: walk the faulting context, not our own (same artifact fix as the SEH
        // handler). The callee copies it, so the dispatcher's record stays unmodified.
        LogCurrentThreadStack(detail, info->ContextRecord);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

#endif // _WIN32

void OnTerminate() noexcept {
    const char* what = "unknown";
    char what_buf[512]{};
    try {
        if (auto ex = std::current_exception()) {
            std::rethrow_exception(ex);
        }
    } catch (const std::exception& e) {
        std::snprintf(what_buf, sizeof(what_buf),
                      "std::terminate() from std::exception: %s", e.what());
        what = what_buf;
    } catch (...) {
        what = "std::terminate() from unknown exception type";
    }

#ifdef _WIN32
    LogCurrentThreadStack(what);
#else
    CrashPrintf("*** %s ***\n", what);

    // GR2FORK FIX: emit the thread-local frames stashed by the __cxa_throw override - the stack
    // is already unwound here, so backtrace() would only show the std::terminate path.
    auto emit_frames = [&](const char* tag, void** frames, int nframes) {
        CrashPrintf("*** %s (%d frames): ***\n", tag, nframes);
        for (int i = 0; i < nframes; ++i) {
            Dl_info info{};
            const int got = ::dladdr(frames[i], &info);
            const char* obj = (got && info.dli_fname) ? info.dli_fname : "??";
            const uintptr_t file_off = (got && info.dli_fbase)
                ? reinterpret_cast<uintptr_t>(frames[i])
                  - reinterpret_cast<uintptr_t>(info.dli_fbase)
                : 0;
            if (got && info.dli_sname) {
                char demangled_buf[1024];
                const char* display = info.dli_sname;
                int status = 0;
                size_t buflen = sizeof(demangled_buf);
                char* res = abi::__cxa_demangle(info.dli_sname, demangled_buf,
                                                &buflen, &status);
                if (status == 0 && res) {
                    display = res;
                }
                const uintptr_t sym_off = info.dli_saddr
                    ? reinterpret_cast<uintptr_t>(frames[i])
                      - reinterpret_cast<uintptr_t>(info.dli_saddr)
                    : 0;
                CrashPrintf("  #%02d %p %s+0x%lx (%s+0x%lx)\n", i, frames[i],
                            display, static_cast<unsigned long>(sym_off), obj,
                            static_cast<unsigned long>(file_off));
            } else {
                CrashPrintf("  #%02d %p (%s+0x%lx)\n", i, frames[i], obj,
                            static_cast<unsigned long>(file_off));
            }
        }
    };

    if (g_throw_nframes > 0) {
        // Demangle the type name for clarity.
        char demangled[512];
        const char* type_display = g_throw_type_name;
        if (g_throw_type_name[0]) {
            int status = 0;
            size_t buflen = sizeof(demangled);
            char* res = abi::__cxa_demangle(g_throw_type_name, demangled,
                                            &buflen, &status);
            if (status == 0 && res) {
                type_display = res;
            }
        }
        CrashPrintf("*** Exception type at throw site: %s ***\n",
                    type_display[0] ? type_display : "<unknown>");
        emit_frames("Throw-site backtrace", g_throw_frames, g_throw_nframes);
    } else {
        CrashPrintf("*** No throw-site backtrace captured (rethrow without "
                    "original frame or non-throw terminate) ***\n");
    }

    // Also print the current stack (terminate handler context) for
    // completeness - useful if the throw-site capture failed.
    void* now_frames[64];
    const int now_nframes = ::backtrace(now_frames, 64);
    emit_frames("Terminate-handler backtrace", now_frames, now_nframes);
#endif
    std::abort();
}

void OnSigabrt(int /*sig*/) {
#ifdef _WIN32
    LogCurrentThreadStack("SIGABRT (abort() called)");
#else
    CrashPrintf("*** SIGABRT ***\n");
#endif
    std::signal(SIGABRT, SIG_DFL);
    std::raise(SIGABRT);
}

#ifdef _WIN32
// GR2FORK FIX: secure-CRT violation handler. Without one the CRT calls
// __fastfail(FAST_FAIL_INVALID_PARAMETER), bypassing UEF and SEH; log a stack trace first, then
// abort() so termination routes through SIGABRT.
void OnInvalidParameter(const wchar_t* expr, const wchar_t* func,
                        const wchar_t* file, unsigned line,
                        uintptr_t /*reserved*/) {
    char detail[768]{};
    // Narrow the wide-char strings for the crash dump. Tolerate NULLs -
    // retail builds pass NULL for all four in most secure CRT call sites.
    char nexpr[256] = "<none>", nfunc[256] = "<none>", nfile[256] = "<none>";
    if (expr) WideCharToMultiByte(CP_UTF8, 0, expr, -1, nexpr, sizeof(nexpr), nullptr, nullptr);
    if (func) WideCharToMultiByte(CP_UTF8, 0, func, -1, nfunc, sizeof(nfunc), nullptr, nullptr);
    if (file) WideCharToMultiByte(CP_UTF8, 0, file, -1, nfile, sizeof(nfile), nullptr, nullptr);
    std::snprintf(detail, sizeof(detail),
                  "secure-CRT invalid parameter: expr=\"%s\" "
                  "func=\"%s\" file=\"%s\" line=%u",
                  nexpr, nfunc, nfile, line);
    LogCurrentThreadStack(detail);
    std::abort();
}

// GR2FORK FIX: pure-virtual call handler. Without this, a pure
// virtual dispatch ends in __fastfail with no app-level callback.
void OnPureCall() {
    LogCurrentThreadStack("pure virtual call");
    std::abort();
}
#endif

// Catches exit() / std::exit() / main-return / return-from-WinMain.
// Does NOT catch _exit(), _Exit(), quick_exit(), TerminateProcess().
void OnAtExit() {
#ifdef _WIN32
    // Log only if nothing fired on this thread, so a crash is not logged twice via CRT unwinding.
    // The per-thread guard means atexit unwinding on a thread that did not witness the crash
    // still logs a line, which shows the unwind path.
    if (t_reentry_guard) return;
    if (g_clean_shutdown.load()) {
        // Routine quit via exit() / return from main. Record a one-liner
        // so crash_dump.txt timeline is complete, but no stack walk.
        CrashPrintf("--- clean shutdown via atexit (tid=%lu) ---\n",
                    (unsigned long)GetCurrentThreadId());
        return;
    }
    LogCurrentThreadStack("exit()/normal-return (atexit)");
#else
    if (t_reentry_guard) return;
    if (g_clean_shutdown.load()) {
        CrashPrintf("--- clean shutdown via atexit ---\n");
        return;
    }
    CrashPrintf("*** exit()/normal-return (atexit) ***\n");
#endif
}

// Catches quick_exit() / _Exit() where the implementation honors at_quick_exit (MSVC UCRT does).
// GR2FORK FIX: when g_clean_shutdown is set (emulator.cpp routine shutdown), log a one-liner
// instead of the full crash banner so crash_dump.txt gets no spurious entry on every normal quit.
void OnAtQuickExit() {
#ifdef _WIN32
    if (t_reentry_guard) return;
    if (g_clean_shutdown.load()) {
        CrashPrintf("--- clean shutdown via quick_exit (tid=%lu) ---\n",
                    (unsigned long)GetCurrentThreadId());
        return;
    }
    LogCurrentThreadStack("quick_exit() / _Exit() (at_quick_exit)");
#else
    if (t_reentry_guard) return;
    if (g_clean_shutdown.load()) {
        CrashPrintf("--- clean shutdown via quick_exit ---\n");
        return;
    }
    CrashPrintf("*** quick_exit / _Exit ***\n");
#endif
}

#ifdef _WIN32

// GR2FORK FIX: inline hooks for ExitProcess, TerminateProcess, RtlExitUserProcess, and
// ntdll!NtTerminateProcess - silent self-shutdown paths that bypass SEH, atexit, and std::terminate
// (UCRT _Exit exits via RtlExitUserProcess). Hooks log the caller, then resume via a trampoline.

struct InlineHook {
    void* target = nullptr;       // e.g. kernel32!ExitProcess
    uint8_t saved[16]{};          // original first prologue_len bytes
    uint8_t prologue_len = 12;    // 12 for normal funcs, 16 for syscall stubs
    void* trampoline = nullptr;   // executable page: saved bytes + jmp (target+prologue_len)
    bool installed = false;
};

InlineHook g_exitprocess_hook{};
InlineHook g_terminateprocess_hook{};
InlineHook g_rtlexit_hook{};
InlineHook g_ntterminateprocess_hook{};

// Writes a 12-byte absolute-jump prologue at `dst` that transfers to `addr`.
static void WriteAbsoluteJump(void* dst, void* addr) {
    auto* p = reinterpret_cast<uint8_t*>(dst);
    // movabs rax, imm64
    p[0] = 0x48;
    p[1] = 0xB8;
    std::memcpy(p + 2, &addr, 8);
    // jmp rax
    p[10] = 0xFF;
    p[11] = 0xE0;
}

// Allocates an RWX page, fills it with the original prologue bytes followed
// by an absolute jump to (target + prologue_len). Returns a function pointer
// callable to execute the original function as if the hook weren't there.
static void* BuildTrampoline(void* target, const uint8_t* saved,
                             unsigned prologue_len) {
    void* page = VirtualAlloc(nullptr, 64, MEM_COMMIT | MEM_RESERVE,
                              PAGE_EXECUTE_READWRITE);
    if (!page) return nullptr;
    auto* p = reinterpret_cast<uint8_t*>(page);
    std::memcpy(p, saved, prologue_len);
    void* resume = reinterpret_cast<uint8_t*>(target) + prologue_len;
    WriteAbsoluteJump(p + prologue_len, resume);
    FlushInstructionCache(GetCurrentProcess(), page, 64);
    return page;
}

// Returns a human-readable reason on failure (caller writes it to log).
static const char* InstallInlineHookVerbose(InlineHook& h,
                                            const char* module_name,
                                            const char* func_name,
                                            void* our_hook) {
    HMODULE mod = GetModuleHandleA(module_name);
    if (!mod) return "GetModuleHandle failed";
    void* target = reinterpret_cast<void*>(GetProcAddress(mod, func_name));
    if (!target) return "GetProcAddress failed";

    DWORD old_prot = 0;
    if (!VirtualProtect(target, 12, PAGE_EXECUTE_READWRITE, &old_prot)) {
        return "VirtualProtect RWX failed (ACG/CFG?)";
    }

    h.target = target;
    h.prologue_len = 12;
    std::memcpy(h.saved, target, 12);
    h.trampoline = BuildTrampoline(target, h.saved, h.prologue_len);
    if (!h.trampoline) {
        VirtualProtect(target, 12, old_prot, &old_prot);
        return "BuildTrampoline failed";
    }
    WriteAbsoluteJump(target, our_hook);

    VirtualProtect(target, 12, old_prot, &old_prot);
    FlushInstructionCache(GetCurrentProcess(), target, 12);
    h.installed = true;
    return nullptr;
}

// GR2FORK FIX: syscall-stub-aware installer for ntdll Nt* functions. The canonical Win10/11
// stub's first 16 bytes are three whole instructions, so the 12-byte jump + 4 NOPs never land
// mid-instruction; any pattern mismatch returns a reason string and leaves the target untouched.
static const char* InstallSyscallStubHookVerbose(InlineHook& h,
                                                 const char* func_name,
                                                 void* our_hook) {
    HMODULE mod = GetModuleHandleA("ntdll.dll");
    if (!mod) return "GetModuleHandle ntdll failed";
    void* target = reinterpret_cast<void*>(GetProcAddress(mod, func_name));
    if (!target) return "GetProcAddress failed";

    // Pattern-verify before any write.
    const auto* p = reinterpret_cast<const uint8_t*>(target);
    const bool pattern_ok =
        p[0] == 0x4C && p[1] == 0x8B && p[2] == 0xD1 &&  // mov r10, rcx
        p[3] == 0xB8 &&                                   // mov eax, imm32
        p[8] == 0xF6 && p[9] == 0x04 && p[10] == 0x25;    // test byte ptr [imm32], imm8
    if (!pattern_ok) {
        return "syscall stub pattern mismatch (already hooked? old Windows? Wow64?)";
    }

    DWORD old_prot = 0;
    if (!VirtualProtect(target, 16, PAGE_EXECUTE_READWRITE, &old_prot)) {
        return "VirtualProtect RWX failed (ACG/CFG?)";
    }

    h.target = target;
    h.prologue_len = 16;
    std::memcpy(h.saved, target, 16);
    h.trampoline = BuildTrampoline(target, h.saved, h.prologue_len);
    if (!h.trampoline) {
        VirtualProtect(target, 16, old_prot, &old_prot);
        return "BuildTrampoline failed";
    }
    WriteAbsoluteJump(target, our_hook);
    // NOP-pad the 4 bytes after the unconditional jump - never executed, but keeps disassemblers
    // honest and hides partial original-instruction tail bytes at +0x0C..+0x0F.
    auto* dst = reinterpret_cast<uint8_t*>(target);
    dst[12] = dst[13] = dst[14] = dst[15] = 0x90;

    VirtualProtect(target, 16, old_prot, &old_prot);
    FlushInstructionCache(GetCurrentProcess(), target, 16);
    h.installed = true;
    return nullptr;
}

// Our replacement for kernel32!ExitProcess. Logs the caller stack then
// invokes the real function via the trampoline.
using ExitProcessFn = void(WINAPI*)(UINT);
using TerminateProcessFn = BOOL(WINAPI*)(HANDLE, UINT);
using RtlExitUserProcessFn = void(NTAPI*)(ULONG);
using NtTerminateProcessFn = LONG(NTAPI*)(HANDLE, LONG);

#ifdef GR2_PGO_INSTRUMENTED
// GR2FORK: profile-runtime entry points, linked only in the instrumented PGO
// build. Declared at file scope (extern "C" is illegal in a function body).
extern "C" int __llvm_profile_write_file(void);
extern "C" unsigned long long __llvm_profile_get_size_for_buffer(void);
#endif

void WINAPI HookedExitProcess(UINT code) {
#ifdef GR2_PGO_INSTRUMENTED
    // GR2FORK: instrumented PGO writes its .profraw from an atexit handler, but
    // the clean-quit path uses quick_exit()/ExitProcess which skips atexit.
    // This hook is the guaranteed last user-mode code on every exit path, so
    // flush here. DIAGNOSTIC: get_size_for_buffer reports the serialized counter
    // size. A tiny value (~header only) proves -fprofile-generate did NOT
    // instrument the code (only the runtime linked) - the true cause of a
    // 0-byte profraw, which is a SEPARATE problem from the atexit flush. A large
    // size + rc=0 means instrumentation and flush both work. Absent line = the
    // GR2_PGO_INSTRUMENTED define never reached this translation unit.
    const unsigned long long gr2_pgo_sz = __llvm_profile_get_size_for_buffer();
    const int gr2_pgo_rc = __llvm_profile_write_file();
    CrashPrintf("[CrashHandler] GR2 PGO: counters=%llu bytes write_rc=%d\n",
                gr2_pgo_sz, gr2_pgo_rc);
#endif
    char detail[128]{};
    std::snprintf(detail, sizeof(detail),
                  "ExitProcess(%u) [intercepted via inline hook]",
                  (unsigned)code);
    LogCurrentThreadStack(detail);
    if (g_exitprocess_hook.trampoline) {
        reinterpret_cast<ExitProcessFn>(g_exitprocess_hook.trampoline)(code);
    }
    ExitProcess(code);
}

BOOL WINAPI HookedTerminateProcess(HANDLE process, UINT code) {
    // Only log if the target is OUR process. Games (and shadPS4 internals)
    // may legitimately TerminateProcess on child processes and we don't
    // want to log those.
    const bool self =
        process == GetCurrentProcess() ||
        GetProcessId(process) == GetCurrentProcessId();
    if (self) {
        char detail[128]{};
        std::snprintf(detail, sizeof(detail),
                      "TerminateProcess(self, %u) [intercepted via inline hook]",
                      (unsigned)code);
        LogCurrentThreadStack(detail);
    }
    if (g_terminateprocess_hook.trampoline) {
        return reinterpret_cast<TerminateProcessFn>(
            g_terminateprocess_hook.trampoline)(process, code);
    }
    return TerminateProcess(process, code);
}

void NTAPI HookedRtlExitUserProcess(ULONG code) {
    char detail[128]{};
    std::snprintf(detail, sizeof(detail),
                  "RtlExitUserProcess(0x%lx) [intercepted via inline hook — "
                  "catches UCRT _Exit/quick_exit bypass of kernel32]",
                  (unsigned long)code);
    LogCurrentThreadStack(detail);
    if (g_rtlexit_hook.trampoline) {
        reinterpret_cast<RtlExitUserProcessFn>(
            g_rtlexit_hook.trampoline)(code);
    }
    // If trampoline missing, hard-stop through TerminateProcess. Our
    // TerminateProcess hook will re-fire but the reentry guard
    // short-circuits its stack walk.
    TerminateProcess(GetCurrentProcess(), code);
}

// GR2FORK FIX: the final user-mode bottleneck before the kernel tears the process down - direct
// NtTerminateProcess syscalls (injected DLLs, vendor drivers, anything bypassing
// RtlExitUserProcess) flow through here. NTSTATUS = LONG in the typedef avoids ntstatus.h.
LONG NTAPI HookedNtTerminateProcess(HANDLE process, LONG exit_status) {
    // Self-target detection: NULL, the GetCurrentProcess() pseudo-handle (-1), or a real handle
    // with a matching PID all mean self; anything else is a child/foreign process - log briefly
    // without treating it as our own death.
    const bool is_null     = process == nullptr;
    const bool is_pseudo   = process == GetCurrentProcess();
    const bool is_self_pid = !is_null && !is_pseudo &&
                             GetProcessId(process) == GetCurrentProcessId();
    const bool self = is_null || is_pseudo || is_self_pid;

    if (self) {
        char detail[160]{};
        std::snprintf(detail, sizeof(detail),
                      "NtTerminateProcess(self, 0x%lx) [intercepted via "
                      "syscall-stub inline hook — final user-mode exit path]",
                      (unsigned long)exit_status);
        LogCurrentThreadStack(detail);
    } else {
        // Foreign-process terminate: one-line note, no stack walk.
        CrashPrintf("--- NtTerminateProcess(foreign pid=%lu, 0x%lx) "
                    "[not our death] tid=%lu ---\n",
                    (unsigned long)(is_null ? 0 : GetProcessId(process)),
                    (unsigned long)exit_status,
                    (unsigned long)GetCurrentThreadId());
    }

    if (g_ntterminateprocess_hook.trampoline) {
        return reinterpret_cast<NtTerminateProcessFn>(
            g_ntterminateprocess_hook.trampoline)(process, exit_status);
    }
    // Fallback: TerminateProcess for self only; foreign handles get
    // 0xC0000001 (STATUS_UNSUCCESSFUL) so the caller sees something.
    if (self) {
        TerminateProcess(GetCurrentProcess(), (UINT)exit_status);
    }
    return 0xC0000001L;
}

#endif // _WIN32

} // namespace

void SignalCleanShutdown() {
    g_clean_shutdown.store(true, std::memory_order_release);
}

void Install() {
    if (g_installed.exchange(true)) return;

    std::set_terminate(OnTerminate);
    std::signal(SIGABRT, OnSigabrt);
    // Register exit hooks as early as possible. atexit handlers run in
    // LIFO order, so registering first means we run LAST - after all the
    // game's own cleanup - which is what we want for logging purposes.
    std::atexit(OnAtExit);
    std::at_quick_exit(OnAtQuickExit);

#ifdef _WIN32
    // Initialize symbols first so early crashes (before HangWatchdog starts) walk with a real
    // search path - HangWatchdog never extends it. Also log the shadps4.exe PDB status so the
    // log records at a glance whether the PDB is being found.
    SymInitSafe();
    LogSymbolDiagnostic();

    SetUnhandledExceptionFilter(OnUnhandledSEH);
    AddVectoredExceptionHandler(/*FirstHandler=*/0, OnVectoredException);
    // Reserve stack space so the handler can run after a stack overflow.
    ULONG guaranteed = 32 * 1024;
    SetThreadStackGuarantee(&guaranteed);

    // GR2FORK FIX: secure-CRT and pure-virtual callbacks - without them a bad argument (e.g.
    // printf with NULL format) or a pure virtual dispatch goes straight to __fastfail, killing
    // the process before any handler can log a stack trace.
    _set_invalid_parameter_handler(OnInvalidParameter);
    _set_purecall_handler(OnPureCall);

    // Inline hooks. Log success/failure per hook so crash_dump forensics can tell which
    // intercepts are live - a silent failure would hide, for example, a Windows build returning
    // a short forwarder stub for ExitProcess that is unsafe to splat.
    const char* e_err =
        InstallInlineHookVerbose(g_exitprocess_hook, "kernel32.dll",
                                 "ExitProcess",
                                 reinterpret_cast<void*>(&HookedExitProcess));
    const char* t_err =
        InstallInlineHookVerbose(g_terminateprocess_hook, "kernel32.dll",
                                 "TerminateProcess",
                                 reinterpret_cast<void*>(&HookedTerminateProcess));
    // RtlExitUserProcess - this is the one the kernel32 hooks miss.
    const char* r_err =
        InstallInlineHookVerbose(g_rtlexit_hook, "ntdll.dll",
                                 "RtlExitUserProcess",
                                 reinterpret_cast<void*>(&HookedRtlExitUserProcess));
    // GR2FORK FIX: NtTerminateProcess is the final user-mode exit bottleneck. Pattern-verified
    // before splatting; a mismatch (older Windows, hot-patched stub, Wow64) is logged and the
    // hook skipped rather than risking a corrupt splat.
    const char* nt_err =
        InstallSyscallStubHookVerbose(g_ntterminateprocess_hook,
                                      "NtTerminateProcess",
                                      reinterpret_cast<void*>(&HookedNtTerminateProcess));

    LOG_INFO(Common,
             "[CrashHandler] installed: terminate SEH-UEF VEH(+fastfail) "
             "SIGABRT atexit at_quick_exit inv_param purecall ; "
             "hooks: ExitProcess={} TerminateProcess={} "
             "RtlExitUserProcess={} NtTerminateProcess={}",
             e_err ? e_err : "ok",
             t_err ? t_err : "ok",
             r_err ? r_err : "ok",
             nt_err ? nt_err : "ok");
#else
    LOG_INFO(Common,
             "[CrashHandler] installed: terminate SIGABRT atexit at_quick_exit");
#endif
}

} // namespace Common::CrashHandler
