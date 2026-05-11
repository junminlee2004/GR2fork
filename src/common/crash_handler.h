// SPDX-FileCopyrightText: Copyright 2025 GR2FORK
// SPDX-License-Identifier: GPL-2.0-or-later

// Crash capture for silent-death scenarios.
//
// shadPS4 sets SEM_NOGPFAULTERRORBOX at startup, which causes Windows to
// suppress the WER dialog AND any Event Viewer entry when the process dies
// from an unhandled exception. On Jun's 3050 Ti the emulator sometimes
// vanishes mid-frame with no log marker of any kind. This module installs
// last-resort handlers that capture:
//   * C++ exceptions that reach std::terminate (uncaught throw)
//   * SEH exceptions unhandled by any __try/__except (AV, stack overflow,
//     illegal instruction, etc.)
//   * abort() / assert() via SIGABRT
//   * __fastfail family (HEAP_CORRUPTION, STACK_BUFFER_OVERRUN,
//     INVALID_CRUNTIME_PARAMETER, FAIL_FAST_EXCEPTION) via VEH
//   * secure-CRT invalid parameter via _set_invalid_parameter_handler
//   * pure virtual calls via _set_purecall_handler
//   * kernel32!ExitProcess, kernel32!TerminateProcess, and
//     ntdll!RtlExitUserProcess via inline hook (the last one is what
//     catches UCRT _Exit() / quick_exit() internals that bypass kernel32
//     and call RtlExitUserProcess → NtTerminateProcess directly)
// Each handler logs the cause + a stack walk of the current thread BEFORE
// letting the process die.

#pragma once

namespace Common::CrashHandler {

// Call once, as early in startup as possible. Safe to call multiple times;
// only the first call installs handlers.
void Install();

// Call right before an intentional std::quick_exit(0) in a routine
// shutdown path (e.g. after the main SDL loop exits). Tells the
// at_quick_exit callback that this is not a crash, so crash_dump.txt
// will receive a single-line "clean shutdown" note instead of a full
// stack walk. No effect if a real handler (SEH / fastfail / abort) has
// already fired on this process — those always log as crashes.
void SignalCleanShutdown();

} // namespace Common::CrashHandler
