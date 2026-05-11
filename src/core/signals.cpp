// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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
#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif
#endif

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
            if (fault < 0x10000) {
                // Fast-path: entity render function 0xf5b410-0xf5bdd5.
                // When called with NULL entity data, absorbed signals zero registers,
                // causing an infinite iteration loop (r14 increments but never matches
                // rcx=0). Instead of absorbing individual signals, jump to the function's
                // epilogue (0xf5bdd6) which checks canary and returns cleanly.
                {
                    const uintptr_t rip = reinterpret_cast<uintptr_t>(code_address);
                    const u32 rip_lo = rip & 0xFFFFFFFF;
                    if (rip_lo >= 0x00f5b410 && rip_lo < 0x00f5bdd6) {
                        Common::IncrementRip(raw_context, 0xf5bdd6 - rip_lo);
                        break;
                    }
                    // Also fast-path the sub-function 0xfa5f60 called by entity render
                    // Its epilogue is at 0xfa6158 (add rsp,0x28; pop rbx..r15,rbp; ret)
                    if (rip_lo >= 0x00fa5f60 && rip_lo < 0x00fa6158) {
                        Common::IncrementRip(raw_context, 0xfa6158 - rip_lo);
                        break;
                    }
                }
#ifdef ARCH_X86_64
                ZydisDecodedInstruction instruction;
                ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
                const auto status = Common::Decoder::Instance()->decodeInstruction(
                    instruction, operands, code_address);
                if (ZYAN_SUCCESS(status)) {
                    Common::IncrementRip(raw_context, instruction.length);
                    static thread_local int s_null_absorb_count = 0;
                    if (++s_null_absorb_count <= 50) {
                        LOG_WARNING(Core,
                            "Absorbed null-page {} in '{}' at {} -> addr {} (#{}): {}",
                            is_write ? "write" : "read",
                            GetThreadName(), fmt::ptr(code_address),
                            fmt::ptr(info->si_addr), s_null_absorb_count,
                            DisassembleInstruction(code_address));
                        // v122: Dump registers on first few absorptions to find
                        // where the NULL pointer comes from
                        if (s_null_absorb_count <= 10) {
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
                                static_cast<u64>(gregs[REG_R15]),
                                rsp_val, ret_addr);
                        }
                    }
                    break;
                }
#endif
                // Decode failed — skip 1 byte as last resort
                Common::IncrementRip(raw_context, 1);
                break;
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