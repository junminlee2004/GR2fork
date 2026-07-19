// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <string>

#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/logging/log.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/signals.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <pthread.h>
#include <ucontext.h>
// GR2FORK: read the CPU GS base directly via arch_prctl, to check the
// per-guest-thread TLS base on a faulting thread.
#if defined(__linux__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#endif

#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif

// GR2FORK: the emulator's per-guest-thread record, declared here (matching
// core/libraries/kernel/threads/pthread.h) so the SIGSEGV handler can read the pointer value
// without the full Pthread definition. Never dereferenced in the handler.
namespace Libraries::Kernel {
struct Pthread;
extern thread_local Pthread* g_curthread;
} // namespace Libraries::Kernel

namespace Core {

// GR2FORK: walk a guest thread stack and log every slot that lands inside the eboot module as a
// return address (runtime VA plus the ghidra offset va - 0x800000000 + 0x107BF0). Shared by the
// SIGSEGV crash-caller dump and the sceKernelPrintBacktraceWithModuleInfo HLE hook. rsp/rbp are
// the guest stack and frame pointers; pass rbp == 0 to scan the stack only. Reads only.
void DumpGuestEbootBacktrace(u64 rsp, u64 rbp, const char* tag, int scan_slots) {
    const auto to_ghidra = [](u64 a) -> u64 {
        return (a >= 0x800000000ull && a < 0x801b80000ull) ? a - 0x800000000ull + 0x107BF0ull : 0;
    };
    // Top of the guest stack: every eboot-range slot is a candidate return address (the call
    // chain). A leaf libc routine pushes no frame, so the first hit is usually the direct caller.
    if (rsp > 0x10000) {
        const u64* s = reinterpret_cast<const u64*>(rsp);
        for (int i = 0; i < scan_slots; ++i) {
            const u64 g = to_ghidra(s[i]);
            if (g) {
                LOG_CRITICAL(Core, "  [{}] stack[{}] = {:#x}  -> EBOOT ghidra {:#x}", tag, i, s[i],
                             g);
            }
        }
    }
    // Follow the rbp frame chain too, in case the immediate caller set a frame.
    u64 fp = rbp;
    for (int depth = 0; depth < 12 && fp > 0x10000; ++depth) {
        const u64 ret = *reinterpret_cast<const u64*>(fp + 8);
        const u64 g = to_ghidra(ret);
        if (g) {
            LOG_CRITICAL(Core, "  [{}] frame[{}] ret={:#x} -> EBOOT ghidra {:#x}", tag, depth, ret,
                         g);
        }
        fp = *reinterpret_cast<const u64*>(fp);
    }
}

#if defined(_WIN32)

// Names come from the process-local registry that SetCurrentThreadName populates, rather than
// GetThreadDescription, whose returned string the caller must LocalFree.
static std::string GetThreadName() {
    const u32 tid = static_cast<u32>(GetCurrentThreadId());
    std::string name = Common::GetThreadNameByTid(tid);
    return name.empty() ? fmt::format("tid {}", tid) : name;
}

#else

static std::string GetThreadName() {
    char name[256];
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0) {
        return "<unknown name>";
    }
    return std::string{name};
}

#endif

static std::string DisassembleInstruction(void* code_address) {
    char buffer[256] = "<unable to decode>";

#ifdef ARCH_X86_64
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const auto status =
        Common::Decoder::Instance()->decodeInstruction(instruction, operands, code_address);
    if (ZYAN_SUCCESS(status)) {
        ZydisFormatter formatter;
        ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);
        ZydisFormatterFormatInstruction(&formatter, &instruction, operands,
                                        instruction.operand_count_visible, buffer, sizeof(buffer),
                                        reinterpret_cast<u64>(code_address), ZYAN_NULL);
    }
#endif

    return buffer;
}

// GR2FORK: null-page absorption telemetry. The atomic increments are async-signal-safe; the
// LOG_WARNING is not, so it is power-of-two throttled via AbsorbLogThrottle - low in-handler
// deadlock risk, never permanently silent (a hard cap would mute the run-up to a real crash).
static std::atomic<u64> g_absorb_entity_render{0}; // RIP in 0xf5b410..0xf5bdd5
static std::atomic<u64> g_absorb_entity_subfn{0};  // RIP in 0xfa5f60..0xfa6157
static std::atomic<u64> g_absorb_generic{0};       // decoded; skipped one insn
// GR2FORK: negative-window twin of g_absorb_generic (faults at 0xFFFF...0000 and above - small
// negative offsets off a null base), counted separately so this class stays distinguishable
// from the positive null-page reads.
static std::atomic<u64> g_absorb_generic_neg{0};
static std::atomic<u64> g_absorb_decode_fail{0};   // decode failed; skipped 1 byte
// GR2FORK: a null-page read absorbed inside guest std::string::assign (entry 0x1228060) means
// the caller assigned a NULL char* (absent field in the rkgDownloadEnvTable blob). assign keeps
// a frame pointer, so [rbp+8] names the calling parser; logged explicitly with its own throttle.
static std::atomic<u64> g_absorb_assign_null{0};   // null READ inside std::string::assign

// Log the 1st, 2nd, 4th, 8th, ... occurrence of a bucket. Always reports the
// first hit, with a cadence that decays as the count grows, so the log can
// never go permanently quiet no matter how many faults are absorbed.
static bool AbsorbLogThrottle(u64 n) {
    return (n & (n - 1)) == 0;
}

// GR2FORK: the faulting thread's general-purpose registers, read out of the platform fault
// context so the absorber and its diagnostics take the same fields on every host.
struct FaultRegs {
    u64 rax, rbx, rcx, rdx, rdi, rsi, rbp, rsp, r14, r15;
};

static bool AbsorbNullPageFaultImpl(void* raw_context, void* fault_address, bool is_write,
                                    const FaultRegs& regs) {
    auto* code_address = Common::GetRip(raw_context);
    const uintptr_t fault = reinterpret_cast<uintptr_t>(fault_address);
    const uintptr_t rip = reinterpret_cast<uintptr_t>(code_address);

    // GR2FORK: the null window covers both sides of address 0 - a null base with a negative
    // displacement faults just below the wrap (observed 0xffffffffffffff98 = null - 0x68); no
    // host allocation lives within 64 KiB of the wrap.
    const bool null_page_pos = fault < 0x10000;
    bool null_page_neg = fault >= static_cast<uintptr_t>(-0x10000);
#if defined(_WIN32)
    // Windows reports an all-ones fault address when the faulting address is unavailable, which
    // marks a wild pointer rather than a null base; skipping an instruction for it is unsound.
    null_page_neg = null_page_neg && fault != static_cast<uintptr_t>(-1);
#endif
    if (!null_page_pos && !null_page_neg) {
        return false;
    }
    // An instruction-fetch fault leaves RIP itself in the null window: there is nothing to decode
    // there, and resuming past it would only land on another unmapped address.
    if (rip < 0x10000 || rip >= static_cast<uintptr_t>(-0x10000)) {
        return false;
    }

    const u32 rip_lo = static_cast<u32>(rip & 0xFFFFFFFFu);
    // Entity render function 0xf5b410-0xf5bdd5 called with NULL entity data would loop forever
    // under per-instruction absorption (r14 never matches rcx=0), so jump to its epilogue
    // (0xf5bdd6), which checks the canary and returns cleanly.
    if (rip_lo >= 0x00f5b410 && rip_lo < 0x00f5bdd6) {
        const u64 n = g_absorb_entity_render.fetch_add(1, std::memory_order_relaxed) + 1;
        if (AbsorbLogThrottle(n)) {
            LOG_WARNING(Core,
                        "Absorbed null-page {} in entity-render (rip={}, addr={}) -> "
                        "jump to epilogue 0xf5bdd6 [entity-render #{}]",
                        is_write ? "write" : "read", fmt::ptr(code_address),
                        fmt::ptr(fault_address), n);
        }
        Common::IncrementRip(raw_context, 0xf5bdd6 - rip_lo);
        return true;
    }
    // Also fast-path the sub-function 0xfa5f60 called by entity render
    // Its epilogue is at 0xfa6158 (add rsp,0x28; pop rbx..r15,rbp; ret)
    if (rip_lo >= 0x00fa5f60 && rip_lo < 0x00fa6158) {
        const u64 n = g_absorb_entity_subfn.fetch_add(1, std::memory_order_relaxed) + 1;
        if (AbsorbLogThrottle(n)) {
            LOG_WARNING(Core,
                        "Absorbed null-page {} in entity-render subfn (rip={}, "
                        "addr={}) -> jump to epilogue 0xfa6158 [entity-subfn #{}]",
                        is_write ? "write" : "read", fmt::ptr(code_address),
                        fmt::ptr(fault_address), n);
        }
        Common::IncrementRip(raw_context, 0xfa6158 - rip_lo);
        return true;
    }
#ifdef ARCH_X86_64
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const auto status =
        Common::Decoder::Instance()->decodeInstruction(instruction, operands, code_address);
    if (ZYAN_SUCCESS(status)) {
        Common::IncrementRip(raw_context, instruction.length);
        // GR2FORK: negative-window faults count and tag separately ("generic-neg") - see the
        // counter documentation above.
        std::atomic<u64>& counter = null_page_neg ? g_absorb_generic_neg : g_absorb_generic;
        const char* tag = null_page_neg ? "generic-neg" : "generic";
        const u64 n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
#if !defined(_WIN32)
        // GR2FORK: absorbed null reads inside guest std::string::assign (0x1228060..0x12280d0)
        // log the caller via [rbp+8] (ghidra address = caller - 0x800000000 + 0x107BF0),
        // throttled separately from boot faults. The frame chain walk dereferences unvalidated
        // guest pointers, so it stays off the Windows path, which has no alternate fault stack.
        if (rip_lo >= 0x01228060u && rip_lo < 0x012280d0u) {
            u64 caller_a = 0, saved_rbp_a = 0, caller2_a = 0;
            if (regs.rbp > 0x10000) {
                saved_rbp_a = *reinterpret_cast<const u64*>(regs.rbp);
                caller_a = *reinterpret_cast<const u64*>(regs.rbp + 8);
                if (saved_rbp_a > 0x10000) {
                    caller2_a = *reinterpret_cast<const u64*>(saved_rbp_a + 8);
                }
            }
            const u64 an = g_absorb_assign_null.fetch_add(1, std::memory_order_relaxed) + 1;
            if (AbsorbLogThrottle(an)) {
                LOG_WARNING(Core,
                            "[ASSIGN-NULL #{}] {}: std::string::assign(dst={:#x}, "
                            "src=NULL) -> PARSER ret_addr={:#x} (ghidra {:#x}); "
                            "caller2={:#x} (ghidra {:#x}); saved_rbp={:#x} src={:#x}",
                            an, GetThreadName(), regs.rbx, caller_a,
                            caller_a > 0x800000000ull ? caller_a - 0x800000000ull + 0x107BF0ull
                                                      : 0,
                            caller2_a,
                            caller2_a > 0x800000000ull ? caller2_a - 0x800000000ull + 0x107BF0ull
                                                       : 0,
                            saved_rbp_a, regs.r14);
            }
        }
#endif
        if (AbsorbLogThrottle(n)) {
            LOG_WARNING(Core, "Absorbed null-page {} in '{}' at {} -> addr {} [{} #{}]: {}",
                        is_write ? "write" : "read", GetThreadName(), fmt::ptr(code_address),
                        fmt::ptr(fault_address), tag, n, DisassembleInstruction(code_address));
            // Dump registers on the first few absorptions to find
            // where the NULL pointer comes from.
            if (n <= 4) {
                // Reads of the guest stack are unvalidated beyond the null-window check, so they
                // stay off the Windows path, which has no alternate fault stack; a fault raised
                // inside the absorber latches that thread out of absorption for good.
                u64 ret_addr = 0;
#if !defined(_WIN32)
                if (regs.rsp > 0x10000) {
                    ret_addr = *reinterpret_cast<const u64*>(regs.rsp);
                }
#endif
                LOG_WARNING(Core,
                            "  regs: rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} "
                            "rdi={:#x} rsi={:#x} r14={:#x} r15={:#x} "
                            "rsp={:#x} ret_addr={:#x}",
                            regs.rax, regs.rbx, regs.rcx, regs.rdx, regs.rdi, regs.rsi, regs.r14,
                            regs.r15, regs.rsp, ret_addr);

#if !defined(_WIN32)
                // Short guest stack walk (raw return-address candidates).
                if (regs.rsp > 0x10000) {
                    const u64* s = reinterpret_cast<const u64*>(regs.rsp);
                    LOG_WARNING(Core, "  stack[0..3]: {:#x} {:#x} {:#x} {:#x}", s[0], s[1], s[2],
                                s[3]);
                }
#endif

#if defined(__linux__)
                // Read the GS base the CPU holds on this thread: gs_base=0 or a stale value means
                // the combined build did not set up the thread's TLS (compare GAME_MainThread
                // against the fiber workers).
                unsigned long gs_base = 0;
                syscall(SYS_arch_prctl, ARCH_GET_GS, &gs_base);
                u64 gs0 = 0;
                const char* gs0_state = "UNSET/low";
                if (gs_base > 0x10000) {
                    gs0 = *reinterpret_cast<const u64*>(gs_base); // gs:[0]
                    gs0_state = "read";
                }
                const void* curthread = static_cast<const void*>(Libraries::Kernel::g_curthread);
                LOG_WARNING(Core, "  TLS[{}]: gs_base={:#x} gs[0]={:#x} ({}) g_curthread={}",
                            GetThreadName(), static_cast<u64>(gs_base), gs0, gs0_state,
                            fmt::ptr(curthread));
#endif

#if !defined(_WIN32)
                // First fault only: raw guest code bytes around the faulting RIP so the
                // instruction that produced rax=0 can be disassembled offline (gs:[disp] read =
                // GS base suspect; [reg+disp] = null object field). The window reaches into the
                // page before RIP, so it stays off the Windows path for the same reason as the
                // frame walk above.
                if (n == 1) {
                    const auto* cp = reinterpret_cast<const unsigned char*>(code_address);
                    static const char hd[] = "0123456789abcdef";
                    constexpr int kBack = 24, kFwd = 16;
                    char hex[(kBack + kFwd) * 4 + 4];
                    int o = 0;
                    for (int i = -kBack; i < kFwd; ++i) {
                        if (i == 0) {
                            hex[o++] = '>';
                        }
                        const unsigned char b = cp[i];
                        hex[o++] = hd[b >> 4];
                        hex[o++] = hd[b & 0x0F];
                        hex[o++] = ' ';
                    }
                    hex[o] = '\0';
                    LOG_WARNING(Core, "  guest code bytes ('>' marks faulting RIP): {}", hex);
                }
#endif
            }
        }
        return true;
    }
#endif
    // Decode failed - skip 1 byte as a last resort. A bad skip here
    // can cascade into mid-instruction faults, so log it (throttled)
    // to make that visible.
    {
        const u64 n = g_absorb_decode_fail.fetch_add(1, std::memory_order_relaxed) + 1;
        if (AbsorbLogThrottle(n)) {
            LOG_WARNING(Core,
                        "Absorbed null-page {} (decode FAILED) at {} -> addr {}; "
                        "skipping 1 byte [decode-fail #{}]",
                        is_write ? "write" : "read", fmt::ptr(code_address),
                        fmt::ptr(fault_address), n);
        }
        Common::IncrementRip(raw_context, 1);
        return true;
    }
}

// GR2FORK: absorb a null-page access that no registered handler claimed, by resuming past the
// faulting instruction. NOP'd fiber workers in GR2 hit NULL pointers regularly, and HLE calls
// that decline an implausible request leave a NULL behind for their caller to store through, so
// crashing on every one makes the game unplayable. Returns true when the context was fixed up.
static bool AbsorbNullPageFault(void* raw_context, void* fault_address, bool is_write,
                                const FaultRegs& regs) {
#if defined(_WIN32)
    // The vectored handler runs on the faulting thread's own stack, so a fault raised inside the
    // absorber would recurse until that stack overflows. A thread that faults here latches and
    // takes the normal crash path from then on.
    static thread_local bool active = false;
    if (active) {
        return false;
    }
    active = true;
    const bool absorbed = AbsorbNullPageFaultImpl(raw_context, fault_address, is_write, regs);
    active = false;
    return absorbed;
#else
    return AbsorbNullPageFaultImpl(raw_context, fault_address, is_write, regs);
#endif
}

#if defined(_WIN32)

static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
    const auto* signals = Signals::Instance();

    bool handled = false;
    switch (pExp->ExceptionRecord->ExceptionCode) {
    case EXCEPTION_ACCESS_VIOLATION: {
        void* const fault_address =
            reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]);
        handled = signals->DispatchAccessViolation(pExp, fault_address);
        if (!handled) {
            FaultRegs regs{};
#if defined(ARCH_X86_64)
            const CONTEXT& c = *pExp->ContextRecord;
            regs = FaultRegs{c.Rax, c.Rbx, c.Rcx, c.Rdx, c.Rdi,
                             c.Rsi, c.Rbp, c.Rsp, c.R14, c.R15};
#endif
            handled = AbsorbNullPageFault(pExp, fault_address, Common::IsWriteError(pExp), regs);
        }
        break;
    }
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    default:
        break;
    }

    return handled ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

#else

static void SignalHandler(int sig, siginfo_t* info, void* raw_context) {
    const auto* signals = Signals::Instance();

    auto* code_address = Common::GetRip(raw_context);

    switch (sig) {
    case SIGSEGV:
    case SIGBUS: {
        const bool is_write = Common::IsWriteError(raw_context);
        if (signals->DispatchAccessViolation(raw_context, info->si_addr)) {
            break;
        }
        {
            FaultRegs regs{};
#if defined(ARCH_X86_64)
            auto& gregs = static_cast<ucontext_t*>(raw_context)->uc_mcontext.gregs;
            regs = FaultRegs{
                static_cast<u64>(gregs[REG_RAX]), static_cast<u64>(gregs[REG_RBX]),
                static_cast<u64>(gregs[REG_RCX]), static_cast<u64>(gregs[REG_RDX]),
                static_cast<u64>(gregs[REG_RDI]), static_cast<u64>(gregs[REG_RSI]),
                static_cast<u64>(gregs[REG_RBP]), static_cast<u64>(gregs[REG_RSP]),
                static_cast<u64>(gregs[REG_R14]), static_cast<u64>(gregs[REG_R15])};
#endif
            if (AbsorbNullPageFault(raw_context, info->si_addr, is_write, regs)) {
                break;
            }
        }
        // GR2FORK: a final unhandled AV is the real crash, but boot-time null faults exhaust
        // the n<=4 reg dump above long before it. Dump guest GP regs + a stack walk once to pin
        // the eboot caller; eboot maps at 0x800000000, ghidra = va - 0x800000000 + 0x107BF0.
#if !defined(_WIN32) && defined(ARCH_X86_64)
        {
            auto* uctx = static_cast<ucontext_t*>(raw_context);
            auto& gregs = uctx->uc_mcontext.gregs;
            const u64 rsp_val = static_cast<u64>(gregs[REG_RSP]);
            const u64 rbp_val = static_cast<u64>(gregs[REG_RBP]);
            LOG_CRITICAL(Core,
                         "[CRASH-CALLER] thread='{}' rip={:#x} fault={:#x} | rax={:#x} rbx={:#x} "
                         "rcx={:#x} rdx={:#x} rdi={:#x} rsi={:#x} rbp={:#x} rsp={:#x}",
                         GetThreadName(), reinterpret_cast<u64>(code_address),
                         reinterpret_cast<u64>(info->si_addr), static_cast<u64>(gregs[REG_RAX]),
                         static_cast<u64>(gregs[REG_RBX]), static_cast<u64>(gregs[REG_RCX]),
                         static_cast<u64>(gregs[REG_RDX]), static_cast<u64>(gregs[REG_RDI]),
                         static_cast<u64>(gregs[REG_RSI]), rbp_val, rsp_val);
            // Top of the guest stack plus the rbp frame chain: log eboot-range return addresses.
            DumpGuestEbootBacktrace(rsp_val, rbp_val, "CRASH-CALLER", 48);
        }
#endif
        UNREACHABLE_MSG(
            "Unhandled access violation in thread '{}' at code address {}: {} address {}",
            GetThreadName(), fmt::ptr(code_address), is_write ? "Write to" : "Read from",
            fmt::ptr(info->si_addr));
        break;
    }
    case SIGILL:
        if (!signals->DispatchIllegalInstruction(raw_context)) {
            UNREACHABLE_MSG("Unhandled illegal instruction in thread '{}' at code address {}: {}",
                            GetThreadName(), fmt::ptr(code_address),
                            DisassembleInstruction(code_address));
        }
        break;
    case SIGUSR1: { // Sleep thread until signal is received
        sigset_t sigset;
        sigemptyset(&sigset);
        sigaddset(&sigset, SIGUSR1);
        sigwait(&sigset, &sig);
    } break;
    default:
        break;
    }
}

#endif

SignalDispatch::SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(handle = AddVectoredExceptionHandler(0, SignalHandler),
               "Failed to register exception handler.");
#else
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to register access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to register illegal instruction signal handler.");
    ASSERT_MSG(sigaction(SIGUSR1, &action, nullptr) == 0,
               "Failed to register sleep signal handler.");
#endif
}

SignalDispatch::~SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(RemoveVectoredExceptionHandler(handle), "Failed to remove exception handler.");
#else
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to remove access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to remove illegal instruction signal handler.");
#endif
}

bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& [handler, _] : access_violation_handlers) {
        if (handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}

bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& [handler, _] : illegal_instruction_handlers) {
        if (handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core