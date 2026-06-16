// SPDX-FileCopyrightText: Copyright 2025 GR2FORK
// SPDX-License-Identifier: GPL-2.0-or-later

// In-memory sync-trace for diagnosing HLE sync-primitive deadlocks.
//
// Design goals:
//   * ZERO log I/O during normal operation. LOG_INFO inside sceKernelSetEventFlag
//     drowned the log system at 100MB/40min. Instead we write into two
//     in-memory structures:
//       (1) a fixed-size ring buffer of recent signal/slow-wait events
//       (2) a tid -> WaitInfo map of currently-active waiters
//   * ONLY dumped at hang time — the watchdog calls Dump() inside its critical
//     handler after the stack dump, producing a single burst of LOG_CRITICAL
//     lines that describe the signal graph leading into the hang.
//
// The ring captures the last kRingSize = 16384 signaling events (roughly a
// few seconds on a busy scene). The waiter map captures EVERY thread that
// entered a sem/ef wait and hasn't exited — the 28 stuck threads we saw in
// prior hang dumps would appear here with their object pointer and wait bits.
//
// PERF(GR2FORK v1.60): tiered gating of the rwlock holder/waiter side-table.
//
// What changed: the Rw* family (RwRdlockAcquired/Released, RwWrlock*,
// RwWaitBegin/End) — called from EVERY PthreadRwlock op via rwlock.cpp —
// is now gated behind a single atomic-bool. Default OFF. The cost driver
// here was a SINGLE GLOBAL mutex `g_rwlock_mtx` plus a nested
// std::unordered_map insert/erase on every rdlock/unlock pair. With Havok,
// GpuComm, kernel, and worker threads all hammering rwlocks at 10K–100K
// ops/sec, the global mutex unlock fired futex_wake into a queue of
// blocked SyncTrace callers; in a captured perf trace this charged
// ~5–8% of total CPU (chains visible at MTTW lines 30–54, JoBwOrKeR
// line 1263, GpuComm lines 658/702 of the cycles report).
//
// What did NOT change: Record() (the lock-free ring write — atomic
// fetch_add + slot store, no contention, ~50 cycles/call) and
// NoteWaitBegin/End (called only on the slow path of mutex/sem/ef/cond
// waits, at most a few hundred per second per thread). Both stay
// always-on, so the hang watchdog's Dump() retains:
//   - the full ring of recent SET_EF/SIGNAL_SEMA/SIGNAL_COND/slow-wait-exit
//     events (the highest-signal-density part of a hang dump — reveals
//     "thread X stopped signalling at t-Nms"),
//   - the live waiters map (which threads are stuck in which
//     sem/ef/mutex/cond wait, with object pointer and wait bits).
//
// What you lose by default: the live rwlock holder/waiter map. To restore
// it for diagnosing a rwlock-specific deadlock, set the environment
// variable GR2FORK_SYNC_TRACE_RWLOCK=1 before launch, or call
// Common::SyncTrace::SetRwlockTracingEnabled(true) at startup.
//
// IMPORTANT: this gating does NOT touch the recursive-rdlock hang fix in
// core/libraries/kernel/threads/rwlock.cpp (the thread_local
// TlsReadCounts machinery that prevents std::shared_timed_mutex from
// deadlocking when a thread re-enters Rdlock with a writer queued).
// That fix lives in rwlock.cpp itself and is independent of this file.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string_view>

#include "common/types.h"

namespace Common::SyncTrace {

enum class Op : u8 {
    // Signalling side — always recorded.
    SET_EF = 1,
    CLEAR_EF,
    CANCEL_EF,
    DELETE_EF,
    SIGNAL_SEMA,
    CANCEL_SEMA,
    DELETE_SEMA,
    SIGNAL_COND,    // pthread_cond_signal / broadcast
    // Slow/error wait exits — recorded only if wait_ms >= threshold.
    WAIT_EF_SLOW,
    WAIT_SEMA_SLOW,
    WAIT_COND_SLOW,   // posix_pthread_cond_(timed)wait
    WAIT_MUTEX_SLOW,  // posix_pthread_mutex_lock contention
};

namespace detail {

// Single source of truth for the rwlock-tier on/off state. Read with
// relaxed memory order on every Rw* call; writes use seq_cst from
// SetRwlockTracingEnabled(). Initialized from GR2FORK_SYNC_TRACE_RWLOCK
// at static-init time. Only the Rw* family is gated by this — Record()
// and NoteWait*() are unconditional (lock-free / low-frequency).
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

// Runtime control for the rwlock-tier holder/waiter map. Default state
// is set from the GR2FORK_SYNC_TRACE_RWLOCK env var on first construction
// of the .cpp's static globals. Pass true to enable the map at runtime
// (e.g. from a debug-build main()).
inline void SetRwlockTracingEnabled(bool on) noexcept {
    detail::g_rwlock_enabled.store(on, std::memory_order_seq_cst);
}

// ---- Public API ---------------------------------------------------------

// Record a signal/slow-wait event. Lock-free write into the ring.
// name may be any string_view — it's truncated to an inline buffer.
//
// ALWAYS ON: lock-free, ~50 cycles/call, doesn't show as a hotspot.
// The ring is the single most useful diagnostic output of Dump().
void Record(Op op, const void* obj, std::string_view name,
            u64 arg1, u64 arg2, u32 waiters, s32 result);

// Mark the current thread as entering a wait. Stored under GetCurrentThreadId().
// Overwrites any previous wait entry for the same tid (a thread can only be in
// one wait at a time). Single mutex-protected hashmap insert.
//
// ALWAYS ON: only invoked on the slow path of a mutex/sem/ef/cond wait
// (after the fast acquire failed), so it's bounded by genuine wait
// frequency. Cost is amortized into the wait itself.
void NoteWaitBegin(Op kind /* WAIT_EF_SLOW or WAIT_SEMA_SLOW */,
                   const void* obj, std::string_view name, u64 arg1);

// Mark the current thread as exiting its wait. Hashmap erase.
void NoteWaitEnd();

// Rwlock tracking: which thread holds which PthreadRwlock, and who is waiting.
// Needed because std::shared_timed_mutex (the backing store for PthreadRwlock)
// doesn't expose its holder/waiter state. Without this, a deadlock between a
// thread holding a read lock + blocked on a sema, and a writer blocked on the
// rwlock, is invisible in stack dumps.
//
// GATED: every PthreadRwlock op calls these. With 10K–100K ops/sec,
// the underlying global mutex + nested unordered_map insert/erase
// dominated CPU in the perf trace. Default OFF; flip on with
// GR2FORK_SYNC_TRACE_RWLOCK=1 or SetRwlockTracingEnabled(true) when
// chasing a rwlock-specific deadlock.
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

// Dump ring buffer, active waiter table, AND rwlock holder/waiter map to
// LOG_CRITICAL. Called by the hang watchdog after the stack dump.
//
// Always callable. Ring + waiters map are always populated (see "ALWAYS
// ON" tier above). The rwlock holder/waiter map is populated only if
// rwlock-tier tracing is enabled; otherwise that section reports
// `rwlocks in map: 0` and the dump notes the disabled state.
void Dump();

} // namespace Common::SyncTrace
