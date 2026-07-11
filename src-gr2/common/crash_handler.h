// SPDX-FileCopyrightText: Copyright 2025 GR2FORK
// SPDX-License-Identifier: GPL-2.0-or-later

// Crash capture for silent-death scenarios: SEM_NOGPFAULTERRORBOX makes Windows suppress both
// the WER dialog and the Event Viewer entry, so an unhandled exception can kill the process
// with no log marker. Last-resort handlers log the cause + a stack walk before the process dies.

#pragma once

namespace Common::CrashHandler {

// Call once, as early in startup as possible. Safe to call multiple times;
// only the first call installs handlers.
void Install();

// Call right before an intentional std::quick_exit(0) so the at_quick_exit callback writes a
// single-line "clean shutdown" note instead of a full stack walk. No effect once a real
// handler (SEH / fastfail / abort) has fired - those always log as crashes.
void SignalCleanShutdown();

} // namespace Common::CrashHandler
