// SPDX-FileCopyrightText: Copyright 2025 GR2FORK
// SPDX-License-Identifier: GPL-2.0-or-later

// In-memory sync-trace for diagnosing HLE sync-primitive deadlocks: a 16384-entry ring of
// recent signal/slow-wait events plus a tid -> WaitInfo map of active waiters, dumped only when
// the hang watchdog calls Dump() - per-event LOG_INFO drowns the log system (~100 MB per 40 min).
// GR2FORK PERF: the rwlock holder/waiter side-table (the Rw* family, called from every
// PthreadRwlock op) is gated behind an atomic bool, default OFF - its global mutex plus map
// churn at 10K-100K ops/sec charges ~5-8% of total CPU; enable via GR2FORK_SYNC_TRACE_RWLOCK=1
// or SetRwlockTracingEnabled(true). Record() (~50 cycles, lock-free) and the slow-path-only
// NoteWaitBegin/End stay always-on. Independent of the rdlock hang fix in threads/rwlock.cpp.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string_view>

#include "common/types.h"

namespace Common::SyncTrace {

enum class Op : u8 {
    // Signalling side - always recorded.
    SET_EF = 1,
    CLEAR_EF,
    CANCEL_EF,
    DELETE_EF,
    SIGNAL_SEMA,
    CANCEL_SEMA,
    DELETE_SEMA,
    SIGNAL_COND,    // pthread_cond_signal / broadcast
    // Slow/error wait exits - recorded only if wait_ms >= threshold.
    WAIT_EF_SLOW,
    WAIT_SEMA_SLOW,
    WAIT_COND_SLOW,   // posix_pthread_cond_(timed)wait
    WAIT_MUTEX_SLOW,  // posix_pthread_mutex_lock contention
};

namespace detail {

// Single source of truth for the rwlock-tier on/off state: relaxed read on every Rw* call,
// seq_cst write from SetRwlockTracingEnabled(), initialized from GR2FORK_SYNC_TRACE_RWLOCK at
// static-init time. Only the Rw* family is gated; Record() and NoteWait*() are unconditional.
extern std::atomic<bool> g_rwlock_enabled;

// Rw* implementations: only invoked when g_rwlock_enabled is true.
void RwRdlockAcquiredImpl(const void* lock);
void RwRdlockReleasedImpl(const void* lock);
void RwWrlockAcquiredImpl(const void* lock);
void RwWrlockReleasedImpl(const void* lock);
void RwWaitBeginImpl(const void* lock, bool want_write);
void RwWaitEndImpl(const void* lock);

inline bool RwlockEnabled() noexcept {
    return g_rwlock_enabled.load(std::memory_order_relaxed);
}

} // namespace detail

// Runtime control for the rwlock-tier holder/waiter map; the default comes from the
// GR2FORK_SYNC_TRACE_RWLOCK env var at static-init time.
inline void SetRwlockTracingEnabled(bool on) noexcept {
    detail::g_rwlock_enabled.store(on, std::memory_order_seq_cst);
}

// ---- Public API ---------------------------------------------------------

// Record a signal/slow-wait event: a lock-free ring write (~50 cycles/call, always on); name is
// truncated to an inline buffer. The ring is the most useful diagnostic output of Dump().
void Record(Op op, const void* obj, std::string_view name,
            u64 arg1, u64 arg2, u32 waiters, s32 result);

// Mark the current thread as entering a wait (one mutex-protected hashmap insert per tid,
// overwriting any previous entry - a thread is in at most one wait). Always on: invoked only on
// the slow path of a mutex/sem/ef/cond wait, so the cost is amortized into the wait itself.
void NoteWaitBegin(Op kind /* WAIT_EF_SLOW or WAIT_SEMA_SLOW */,
                   const void* obj, std::string_view name, u64 arg1);

// Mark the current thread as exiting its wait. Hashmap erase.
void NoteWaitEnd();

// Rwlock tracking: which thread holds which PthreadRwlock and who waits - state
// std::shared_timed_mutex does not expose, without which reader-vs-writer deadlocks are
// invisible in stack dumps. Gated (default OFF, see the file header) because every
// PthreadRwlock op calls these; enable when chasing a rwlock-specific deadlock.
inline void RwRdlockAcquired(const void* lock) {
    if (!detail::RwlockEnabled()) [[likely]] return;
    detail::RwRdlockAcquiredImpl(lock);
}
inline void RwRdlockReleased(const void* lock) {
    if (!detail::RwlockEnabled()) [[likely]] return;
    detail::RwRdlockReleasedImpl(lock);
}
inline void RwWrlockAcquired(const void* lock) {
    if (!detail::RwlockEnabled()) [[likely]] return;
    detail::RwWrlockAcquiredImpl(lock);
}
inline void RwWrlockReleased(const void* lock) {
    if (!detail::RwlockEnabled()) [[likely]] return;
    detail::RwWrlockReleasedImpl(lock);
}
inline void RwWaitBegin(const void* lock, bool want_write) {
    if (!detail::RwlockEnabled()) [[likely]] return;
    detail::RwWaitBeginImpl(lock, want_write);
}
inline void RwWaitEnd(const void* lock) {
    if (!detail::RwlockEnabled()) [[likely]] return;
    detail::RwWaitEndImpl(lock);
}

// Dump ring buffer, active waiter table, and rwlock holder/waiter map to LOG_CRITICAL; called
// by the hang watchdog after the stack dump. Ring and waiters map are always populated; the
// rwlock map only when rwlock-tier tracing is enabled (otherwise the dump notes it is disabled).
void Dump();

} // namespace Common::SyncTrace
