// SPDX-FileCopyrightText: Copyright 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>
#include "common/logging/filter.h"

namespace Common::Log {

class Filter;

/// Initializes the logging system. This should be the first thing called in main.
void Initialize(std::string_view log_file = "");

bool IsActive();

/// Starts the logging threads.
void Start();

/// Explictily stops the logger thread and flushes the buffers
void Stop();

/// Lightweight, non-destructive flush: pushes every backend's buffered bytes to
/// the OS (fflush) WITHOUT joining the backend thread or draining the async
/// queue. Mirrors v0.16.0's Common::Log::Flush() so crash/checkpoint paths have
/// a cheap "get what's written onto disk" call that is safe to invoke
/// repeatedly. For a full async drain before the process dies use Stop() or
/// StopAsyncAndForceSync() instead; in the default sync config every line is
/// already flushed by the file backend, so this is belt-and-suspenders there.
void Flush();

/// Closes log files and stops the logger
void Denitializer();

/// The global filter will prevent any messages from even being processed if they are filtered.
void SetGlobalFilter(const Filter& filter);

void SetColorConsoleBackendEnabled(bool enabled);

void SetAppend();

/// Drains the async log queue, stops the backend thread, and switches
/// every subsequent LOG_* call to a synchronous write that fflush()es
/// before returning. Idempotent. Used by the crash handler so that
/// LOG_CRITICAL output during crash processing actually lands in the
/// main log file before the process dies, rather than sitting in the
/// MPSC queue when the consumer thread gets killed by process exit.
///
/// Safe to call from any thread, including (defensively) the backend
/// thread itself: a self-thread call sets the sync flag but skips the
/// join to avoid deadlock - in that case the backend thread will exit
/// when it returns from whatever it was doing.
void StopAsyncAndForceSync();

} // namespace Common::Log
