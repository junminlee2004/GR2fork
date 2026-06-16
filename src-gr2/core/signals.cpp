// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>

#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/logging/log.h"
#include "common/signal_context.h"
#include "core/signals.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <pthread.h>
#include <ucontext.h>
// GR2FORK probe (combined-binary TLS diagnosis): read the CPU GS base directly
// via arch_prctl, to check the per-guest-thread TLS base on a faulting thread.
#if defined(__linux__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif
#endif

// GR2FORK probe: the emulator's per-guest-thread record. Declared here (matching
// core/libraries/kernel/threads/pthread.h) so the SIGSEGV handler can read the
// pointer value without pulling in the full Pthread definition. Used read-only;
// never dereferenced in the handler.
namespace Libraries::Kernel {
struct Pthread;
extern thread_local Pthread* g_curthread;
} // namespace Libraries::Kernel

namespace Core {

#if defined(_WIN32)

static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
    const auto* signals = Signals::Instance();

    bool handled = false;
    switch (pExp->ExceptionRecord->ExceptionCode) {
    case EXCEPTION_ACCESS_VIOLATION:
        handled = signals->DispatchAccessViolation(
            pExp, reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]));
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    default:
        break;
    }

    return handled ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

#else

static std::string GetThreadName() {
    char name[256];
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0) {
        return "<unknown name>";
    }
    return std::string{name};
}

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

// GR2FORK: null-page absorption telemetry. The SIGSEGV handler skips faulting
// instructions for guest null-page accesses (NOP'd GR2 fiber workers hit NULL
// constantly); these counters make every absorbed fault observable. The atomic
// increments are async-signal-safe (LOCK XADD — no lock, no allocation). The
// actual LOG_WARNING still routes through the (async-signal-UNSAFE) logging
// backend, so it is power-of-two throttled via AbsorbLogThrottle: this keeps
// the in-handler logging frequency — and thus the deadlock window if a fault
// lands while the faulting thread holds the log mutex — low, while NEVER going
// permanently silent. The previous `<= 50` hard cap meant the handler stopped
// reporting after 50 hits, so a run-up to a real crash produced no output.
static std::atomic<u64> g_absorb_entity_render{0}; // RIP in 0xf5b410..0xf5bdd5
static std::atomic<u64> g_absorb_entity_subfn{0};  // RIP in 0xfa5f60..0xfa6157
static std::atomic<u64> g_absorb_generic{0};       // decoded; skipped one insn
// GR2FORK (2026-06-10): negative-window twin of g_absorb_generic — faults at
// 0xFFFF...0000 and above, i.e. small NEGATIVE offsets off a null base. Kept
// as a separate counter + "[generic-neg #N]" tag so this class stays
// forensically distinguishable from the positive null-page reads.
static std::atomic<u64> g_absorb_generic_neg{0};
static std::atomic<u64> g_absorb_decode_fail{0};   // decode failed; skipped 1 byte

// Log the 1st, 2nd, 4th, 8th, ... occurrence of a bucket. Always reports the
// first hit, with a cadence that decays as the count grows, so the log can
// never go permanently quiet no matter how many faults are absorbed.
static bool AbsorbLogThrottle(u64 n) {
    return (n & (n - 1)) == 0;
}

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
        // Fallback: absorb null-page accesses (0x0..0xFFFF) by skipping the
        // faulting instruction.  NOP'd fiber workers in GR2 hit NULL pointers
        // regularly; crashing on every one makes the game unplayable.
        {
            const uintptr_t fault = reinterpret_cast<uintptr_t>(info->si_addr);
            // GR2FORK (2026-06-10): the null window covers BOTH sides of
            // address 0. The original gate (`fault < 0x10000`) absorbed only
            // POSITIVE null-page offsets; a null-base access with a NEGATIVE
            // displacement — e.g. a negative struct/vtable field or a
            // multiple-inheritance-adjusted `this` — faults just below the
            // address-space wrap and fell through to UNREACHABLE, aborting
            // the whole process for the same guest-bug class the absorber
            // exists for. Observed in the wild 2026-06-10 (ledger build):
            // thread MTTW, rip 0x800d08e1f, fault addr 0xffffffffffffff98
            // (= null − 0x68), moments after [generic #2]/[generic #4]
            // null reads in the same placement-churn window — the familiar
            // NOP'd-fiber null-entity pattern, new (negative) shape.
            // DispatchAccessViolation (tracked guest memory) has already
            // declined above, and no host allocation lives within 64 KiB of
            // the wrap, so absorbing here cannot mask a tracked-write fault.
            const bool null_page_pos = fault < 0x10000;
            const bool null_page_neg = fault >= static_cast<uintptr_t>(-0x10000);
            if (null_page_pos || null_page_neg) {
                const u32 rip_lo = reinterpret_cast<uintptr_t>(code_address) & 0xFFFFFFFFu;
                // Fast-path: entity render function 0xf5b410-0xf5bdd5.
                // When called with NULL entity data, absorbed signals zero registers,
                // causing an infinite iteration loop (r14 increments but never matches
                // rcx=0). Instead of absorbing individual signals, jump to the function's
                // epilogue (0xf5bdd6) which checks canary and returns cleanly.
                if (rip_lo >= 0x00f5b410 && rip_lo < 0x00f5bdd6) {
                    const u64 n =
                        g_absorb_entity_render.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (AbsorbLogThrottle(n)) {
                        LOG_WARNING(Core,
                                    "Absorbed null-page {} in entity-render (rip={}, addr={}) -> "
                                    "jump to epilogue 0xf5bdd6 [entity-render #{}]",
                                    is_write ? "write" : "read", fmt::ptr(code_address),
                                    fmt::ptr(info->si_addr), n);
                    }
                    Common::IncrementRip(raw_context, 0xf5bdd6 - rip_lo);
                    break;
                }
                // Also fast-path the sub-function 0xfa5f60 called by entity render
                // Its epilogue is at 0xfa6158 (add rsp,0x28; pop rbx..r15,rbp; ret)
                if (rip_lo >= 0x00fa5f60 && rip_lo < 0x00fa6158) {
                    const u64 n =
                        g_absorb_entity_subfn.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (AbsorbLogThrottle(n)) {
                        LOG_WARNING(Core,
                                    "Absorbed null-page {} in entity-render subfn (rip={}, "
                                    "addr={}) -> jump to epilogue 0xfa6158 [entity-subfn #{}]",
                                    is_write ? "write" : "read", fmt::ptr(code_address),
                                    fmt::ptr(info->si_addr), n);
                    }
                    Common::IncrementRip(raw_context, 0xfa6158 - rip_lo);
                    break;
                }
#ifdef ARCH_X86_64
                ZydisDecodedInstruction instruction;
                ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
                const auto status = Common::Decoder::Instance()->decodeInstruction(
                    instruction, operands, code_address);
                if (ZYAN_SUCCESS(status)) {
                    Common::IncrementRip(raw_context, instruction.length);
                    // GR2FORK (2026-06-10): negative-window faults count and
                    // tag separately ("generic-neg") — see counter doc above.
                    std::atomic<u64>& counter =
                        null_page_neg ? g_absorb_generic_neg : g_absorb_generic;
                    const char* tag = null_page_neg ? "generic-neg" : "generic";
                    const u64 n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (AbsorbLogThrottle(n)) {
                        LOG_WARNING(Core,
                                    "Absorbed null-page {} in '{}' at {} -> addr {} [{} #{}]: "
                                    "{}",
                                    is_write ? "write" : "read", GetThreadName(),
                                    fmt::ptr(code_address), fmt::ptr(info->si_addr), tag, n,
                                    DisassembleInstruction(code_address));
                        // Dump registers on the first few absorptions to find
                        // where the NULL pointer comes from.
                        if (n <= 4) {
                            auto* uctx = static_cast<ucontext_t*>(raw_context);
                            auto& gregs = uctx->uc_mcontext.gregs;
                            u64 rsp_val = static_cast<u64>(gregs[REG_RSP]);
                            u64 ret_addr = 0;
                            if (rsp_val > 0x10000) {
                                ret_addr = *reinterpret_cast<u64*>(rsp_val);
                            }
                            LOG_WARNING(Core,
                                        "  regs: rax={:#x} rbx={:#x} rcx={:#x} rdx={:#x} "
                                        "rdi={:#x} rsi={:#x} r14={:#x} r15={:#x} "
                                        "rsp={:#x} ret_addr={:#x}",
                                        static_cast<u64>(gregs[REG_RAX]),
                                        static_cast<u64>(gregs[REG_RBX]),
                                        static_cast<u64>(gregs[REG_RCX]),
                                        static_cast<u64>(gregs[REG_RDX]),
                                        static_cast<u64>(gregs[REG_RDI]),
                                        static_cast<u64>(gregs[REG_RSI]),
                                        static_cast<u64>(gregs[REG_R14]),
                                        static_cast<u64>(gregs[REG_R15]), rsp_val, ret_addr);

                            // ===== GR2FORK combined-binary diagnosis probe =====
                            // The standalone exe runs this exact guest code fine;
                            // only the dlopen'd-core (combined) build faults here
                            // with rax=0. Two possibilities: (a) a wrong/unset GS
                            // base on this thread, so guest gs:[] TLS reads return
                            // null; or (b) a genuinely-null object. These dumps
                            // tell them apart. Always-on, throttled with the parent.

                            // Short guest stack walk (raw return-address candidates).
                            if (rsp_val > 0x10000) {
                                const u64* s = reinterpret_cast<const u64*>(rsp_val);
                                LOG_WARNING(Core, "  stack[0..3]: {:#x} {:#x} {:#x} {:#x}",
                                            s[0], s[1], s[2], s[3]);
                            }

#if defined(__linux__)
                            // What GS base does the CPU actually hold on THIS thread?
                            // Compare across threads (GAME_MainThread vs the fiber
                            // workers in the cascade): a thread running guest code
                            // with gs_base=0 or a stale value is the smoking gun for
                            // the combined build not setting up that thread's TLS.
                            unsigned long gs_base = 0;
                            syscall(SYS_arch_prctl, ARCH_GET_GS, &gs_base);
                            u64 gs0 = 0;
                            const char* gs0_state = "UNSET/low";
                            if (gs_base > 0x10000) {
                                gs0 = *reinterpret_cast<const u64*>(gs_base); // gs:[0]
                                gs0_state = "read";
                            }
                            const void* curthread =
                                static_cast<const void*>(Libraries::Kernel::g_curthread);
                            LOG_WARNING(Core,
                                        "  TLS[{}]: gs_base={:#x} gs[0]={:#x} ({}) "
                                        "g_curthread={}",
                                        GetThreadName(), static_cast<u64>(gs_base), gs0,
                                        gs0_state, fmt::ptr(curthread));
#endif

                            // First fault only: raw guest code bytes ending at the
                            // faulting RIP, so the instruction that produced rax=0
                            // can be disassembled offline. 'mov rax, gs:[disp]' =>
                            // TLS read (GS base suspect); 'mov rax,[reg+disp]' =>
                            // null object field; a 'call' just before => null return.
                            if (n == 1) {
                                const auto* cp =
                                    reinterpret_cast<const unsigned char*>(code_address);
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
                                LOG_WARNING(Core,
                                            "  guest code bytes ('>' marks faulting RIP): {}",
                                            hex);
                            }
                            // ===== end GR2FORK probe =====
                        }
                    }
                    break;
                }
#endif
                // Decode failed — skip 1 byte as a last resort. Previously this
                // path was silent; a bad skip here can cascade into mid-instruction
                // faults, so log it (throttled) to make that visible.
                {
                    const u64 n = g_absorb_decode_fail.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (AbsorbLogThrottle(n)) {
                        LOG_WARNING(Core,
                                    "Absorbed null-page {} (decode FAILED) at {} -> addr {}; "
                                    "skipping 1 byte [decode-fail #{}]",
                                    is_write ? "write" : "read", fmt::ptr(code_address),
                                    fmt::ptr(info->si_addr), n);
                    }
                    Common::IncrementRip(raw_context, 1);
                    break;
                }
            }
        }
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