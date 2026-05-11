// SPDX-FileCopyrightText: Copyright 2025 GR2FORK
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/sync_trace.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common/logging/log.h"

#ifdef _WIN32
#include <windows.h>
#endif

// PERF(GR2FORK release_v2_2): two structural fast-path rewrites in this TU.
//
// (1) CurrentTid() — was `std::hash<std::thread::id>{}(std::this_thread::get_id())`
//     on every call. Both `get_id()` (a libpthread call returning an opaque
//     pthread_t-wrapper) and the hash of it (a non-trivial mix of an opaque
//     id type) showed up across the perf trace because Record(),
//     NoteWaitBegin(), and NoteWaitEnd() all call CurrentTid(). Cached
//     once per thread in a `thread_local` initialised on first use; reads
//     then collapse to a single TLS load.
//
// (2) NoteWaitBegin/End — was a `std::mutex g_waiters_mtx` guarding a
//     `std::unordered_map<u64, WaitInfo> g_waiters`. EVERY slow-path
//     mutex/cond/sem/ef wait acquired this single global mutex twice
//     (once at begin, once at end) and performed a hashmap insert/erase.
//     Captured perf chain (MTTW thread):
//
//         0.06%  __GI___pthread_mutex_unlock_usercnt
//                  Common::SyncTrace::NoteWaitEnd()
//         0.04%  Common::SyncTrace::NoteWaitBegin(...)
//
//     plus mirror chains on JoBwOrKeR/Havok. With ~50 emulator threads
//     occasionally hitting cold mutex/cond/sem paths, the global mutex
//     in `g_waiters_mtx` was a serialization point precisely where the
//     guest is already paying for one futex round-trip. Replaced with a
//     per-thread WaitSlot that lives in a lock-free Treiber stack
//     anchored at `g_slot_head`. Each thread's slot has a single
//     `std::atomic<bool> active` flag plus the wait fields; only the
//     owning thread writes the fields, only the owning thread flips
//     `active`. Dump() walks the list lock-free and snapshots each
//     active slot. The old global mutex is gone entirely.
//
//     Race analysis:
//       - Slot registration (first NoteWaitBegin per thread) is a CAS
//         push onto g_slot_head. Multiple threads registering
//         concurrently is the standard Treiber-stack push and is safe.
//         No pop is ever performed, so there is no ABA hazard. Memory
//         ordering: release on push (so a Dump reader sees the next
//         pointer + tid before the head update), acquire on the head
//         load + walk in Dump.
//       - Per-slot writes: only the owning thread writes its slot.
//         Fields are written, THEN `active` is flipped to true with
//         release. A reader that sees `active==true` (acquire) is
//         guaranteed to see consistent fields published before the
//         release.
//       - Tearing in Dump: a reader that catches a slot mid-write
//         (active==true, fields half-updated from a back-to-back
//         Begin call) gets a slightly inconsistent snapshot. This is
//         acceptable for a debug dump — the alternative was a global
//         mutex that serialized every wait-path entry.
//       - Thread death: TlsSlotHolder's destructor flips active=false
//         and tid=0, marking the slot as orphaned. Slots are never
//         removed from the list (intentional — keeps push lock-free).
//         Dump skips orphaned slots. ~96 bytes leaked per dead thread,
//         which over a session is at most a few KB.

namespace Common::SyncTrace {

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kRingSize = 16384; // must be power of two
static_assert((kRingSize & (kRingSize - 1)) == 0, "kRingSize must be pow2");
constexpr size_t kRingMask = kRingSize - 1;
constexpr size_t kNameMax = 40;

struct Event {
    u64 seq;                         // write-order counter; 0 => unused slot
    Clock::time_point when;
    u64 tid;
    Op op;
    const void* obj;
    char name[kNameMax];
    u64 arg1;
    u64 arg2;
    u32 waiters;
    s32 result;
};

std::array<Event, kRingSize> g_ring{};
std::atomic<u64> g_write_idx{1}; // start at 1 so seq=0 means empty

// PERF(GR2FORK release_v2_2): per-thread wait slot replacing the global
// mutex+hashmap. See file header for race analysis.
struct WaitSlot {
    // Owner-only writes; reader uses acquire+release pairing on `active`
    // to publish/read the fields.
    std::atomic<bool> active{false};
    std::atomic<u64> tid{0};            // 0 => orphaned (thread died)
    Clock::time_point when{};
    Op op{};
    const void* obj{nullptr};
    char name[kNameMax]{};
    u64 arg1{0};

    // Treiber-stack singly-linked list. Set once at registration, never
    // mutated again. Loaded by Dump() under acquire.
    WaitSlot* next{nullptr};
};

// Lock-free Treiber stack head. Pushes only — slots are immortal so no
// ABA hazard.
std::atomic<WaitSlot*> g_slot_head{nullptr};

// Rwlock holder/waiter tracking. ONLY accessed when g_rwlock_enabled is
// true (gated at the public API in sync_trace.h). When the gate is off
// these stay empty and the global mutex is uncontended.
struct RwLockInfo {
    u64 writer_tid = 0;                // nonzero tid if held exclusively
    std::chrono::steady_clock::time_point writer_acquired_at{};
    std::unordered_map<u64, std::chrono::steady_clock::time_point> readers;
    std::unordered_map<u64, std::chrono::steady_clock::time_point> waiting_writers;
    std::unordered_map<u64, std::chrono::steady_clock::time_point> waiting_readers;
};
std::mutex g_rwlock_mtx;
std::unordered_map<const void*, RwLockInfo> g_rwlocks;

// PERF(GR2FORK release_v2_2): cached tid per thread. The std::hash of a
// std::thread::id is not trivial (libstdc++ implements it as an
// integer-conversion + hash; libc++ has its own permute) and was on the
// hot path for every Record/NoteWait* call. Caching collapses it to a
// TLS load.
inline u64 CurrentTid() {
    thread_local u64 cached_tid = []() -> u64 {
#ifdef _WIN32
        return static_cast<u64>(GetCurrentThreadId());
#else
        return static_cast<u64>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
#endif
    }();
    return cached_tid;
}

// Per-thread slot holder. Owns a heap-allocated WaitSlot pushed onto
// g_slot_head on first use. On thread destruction, marks the slot
// orphaned (tid=0, active=false) but does NOT unlink — that would
// require a write lock on the list and we want the dump path lock-free.
class TlsSlotHolder {
public:
    WaitSlot* slot{nullptr};

    WaitSlot* GetOrCreate() {
        if (slot) [[likely]] return slot;
        return SlowAcquire();
    }

    ~TlsSlotHolder() {
        if (slot) {
            // Order: drop active first so any in-flight Dump() ignores
            // the slot, then clear tid to mark fully orphaned.
            slot->active.store(false, std::memory_order_release);
            slot->tid.store(0, std::memory_order_release);
        }
    }

private:
    [[gnu::cold, gnu::noinline]] WaitSlot* SlowAcquire() {
        slot = new WaitSlot{};
        slot->tid.store(CurrentTid(), std::memory_order_relaxed);
        // Treiber-stack push.
        WaitSlot* old_head = g_slot_head.load(std::memory_order_relaxed);
        do {
            slot->next = old_head;
        } while (!g_slot_head.compare_exchange_weak(
                     old_head, slot,
                     std::memory_order_release,
                     std::memory_order_relaxed));
        return slot;
    }
};
thread_local TlsSlotHolder tls_slot_holder;

inline void CopyName(char (&dst)[kNameMax], std::string_view src) {
    const size_t n = std::min(src.size(), kNameMax - 1);
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

const char* OpName(Op op) {
    switch (op) {
    case Op::SET_EF:         return "SET_EF";
    case Op::CLEAR_EF:       return "CLEAR_EF";
    case Op::CANCEL_EF:      return "CANCEL_EF";
    case Op::DELETE_EF:      return "DELETE_EF";
    case Op::SIGNAL_SEMA:    return "SIGNAL_SEMA";
    case Op::CANCEL_SEMA:    return "CANCEL_SEMA";
    case Op::DELETE_SEMA:    return "DELETE_SEMA";
    case Op::SIGNAL_COND:    return "SIGNAL_COND";
    case Op::WAIT_EF_SLOW:   return "WAIT_EF_SLOW";
    case Op::WAIT_SEMA_SLOW: return "WAIT_SEMA_SLOW";
    case Op::WAIT_COND_SLOW: return "WAIT_COND_SLOW";
    case Op::WAIT_MUTEX_SLOW:return "WAIT_MUTEX_SLOW";
    }
    return "?";
}

// Read the env var ONCE at static-init for the rwlock-tier gate. Returns
// true if GR2FORK_SYNC_TRACE_RWLOCK is set to a non-empty, non-"0" value.
// Static-init ordering safe: getenv() is reentrant on POSIX, no other
// static init in this TU touches g_rwlock_enabled, and no SyncTrace
// function should be reachable before thread creation (after main).
bool ReadRwlockEnvDefault() noexcept {
    const char* e = std::getenv("GR2FORK_SYNC_TRACE_RWLOCK");
    return e && e[0] && e[0] != '0';
}

} // namespace

namespace detail {

// Rwlock-tier on/off switch. Default from env var; flipped at runtime by
// SetRwlockTracingEnabled(). Only the Rw* forwarders in the header check
// this — Record() and NoteWait*() do not, by design.
std::atomic<bool> g_rwlock_enabled{ReadRwlockEnvDefault()};

void RwRdlockAcquiredImpl(const void* lock) {
    const auto now = Clock::now();
    const u64 tid = CurrentTid();
    std::lock_guard lk(g_rwlock_mtx);
    auto& info = g_rwlocks[lock];
    info.waiting_readers.erase(tid);
    info.readers[tid] = now;
}

void RwRdlockReleasedImpl(const void* lock) {
    const u64 tid = CurrentTid();
    std::lock_guard lk(g_rwlock_mtx);
    auto it = g_rwlocks.find(lock);
    if (it == g_rwlocks.end()) return;
    it->second.readers.erase(tid);
    if (it->second.readers.empty() && it->second.writer_tid == 0 &&
        it->second.waiting_writers.empty() &&
        it->second.waiting_readers.empty()) {
        g_rwlocks.erase(it);
    }
}

void RwWrlockAcquiredImpl(const void* lock) {
    const auto now = Clock::now();
    const u64 tid = CurrentTid();
    std::lock_guard lk(g_rwlock_mtx);
    auto& info = g_rwlocks[lock];
    info.waiting_writers.erase(tid);
    info.writer_tid = tid;
    info.writer_acquired_at = now;
}

void RwWrlockReleasedImpl(const void* lock) {
    std::lock_guard lk(g_rwlock_mtx);
    auto it = g_rwlocks.find(lock);
    if (it == g_rwlocks.end()) return;
    it->second.writer_tid = 0;
    it->second.writer_acquired_at = {};
    if (it->second.readers.empty() && it->second.waiting_writers.empty() &&
        it->second.waiting_readers.empty()) {
        g_rwlocks.erase(it);
    }
}

void RwWaitBeginImpl(const void* lock, bool want_write) {
    const auto now = Clock::now();
    const u64 tid = CurrentTid();
    std::lock_guard lk(g_rwlock_mtx);
    auto& info = g_rwlocks[lock];
    if (want_write) info.waiting_writers[tid] = now;
    else            info.waiting_readers[tid] = now;
}

void RwWaitEndImpl(const void* lock) {
    const u64 tid = CurrentTid();
    std::lock_guard lk(g_rwlock_mtx);
    auto it = g_rwlocks.find(lock);
    if (it == g_rwlocks.end()) return;
    it->second.waiting_writers.erase(tid);
    it->second.waiting_readers.erase(tid);
}

} // namespace detail

// ---- Always-on tier ------------------------------------------------------
// Record() and NoteWait*() are NOT gated. Record is lock-free; NoteWait*
// is now per-thread lock-free (was global-mutex + hashmap pre-v2_2).

void Record(Op op, const void* obj, std::string_view name,
            u64 arg1, u64 arg2, u32 waiters, s32 result) {
    const u64 seq = g_write_idx.fetch_add(1, std::memory_order_relaxed);
    Event& e = g_ring[seq & kRingMask];
    e.seq = seq;
    e.when = Clock::now();
    e.tid = CurrentTid();
    e.op = op;
    e.obj = obj;
    CopyName(e.name, name);
    e.arg1 = arg1;
    e.arg2 = arg2;
    e.waiters = waiters;
    e.result = result;
}

void NoteWaitBegin(Op kind, const void* obj, std::string_view name, u64 arg1) {
    // PERF(GR2FORK release_v2_2): owner-only writes to per-thread slot,
    // then atomic publish of `active`. No global mutex.
    auto* slot = tls_slot_holder.GetOrCreate();
    slot->when = Clock::now();
    slot->op = kind;
    slot->obj = obj;
    CopyName(slot->name, name);
    slot->arg1 = arg1;
    // Release publishes the field stores above to any future acquire
    // load of `active` by Dump().
    slot->active.store(true, std::memory_order_release);
}

void NoteWaitEnd() {
    // PERF(GR2FORK release_v2_2): single atomic store. The slot keeps
    // its stale field values; readers ignore the slot because active
    // is false.
    if (auto* slot = tls_slot_holder.slot) [[likely]] {
        slot->active.store(false, std::memory_order_release);
    }
    // If `slot` is null we never registered, which means NoteWaitBegin
    // was never called on this thread — no-op is correct.
}

void Dump() {
    const auto now = Clock::now();
    const bool rwlock_on = detail::g_rwlock_enabled.load(std::memory_order_relaxed);
    LOG_CRITICAL(Common, "[SyncTrace] ==== begin dump ====");
    LOG_CRITICAL(Common,
                 "[SyncTrace] tier state: ring=ALWAYS_ON, waiters=ALWAYS_ON, "
                 "rwlock_map={}",
                 rwlock_on ? "ENABLED" : "DISABLED (set "
                                         "GR2FORK_SYNC_TRACE_RWLOCK=1 to enable)");

    // 1) Active waiters — who's stuck right now. ALWAYS POPULATED.
    //    Lock-free walk of the per-thread slot list. We snapshot count
    //    and content in two passes so the log header is accurate even
    //    if a thread flips active during iteration.
    {
        // Acquire pairs with the release in the Treiber-stack push, so
        // we see all slots and their `next` pointers consistently.
        WaitSlot* head = g_slot_head.load(std::memory_order_acquire);

        // First pass: count active slots.
        size_t active_count = 0;
        for (WaitSlot* s = head; s != nullptr; s = s->next) {
            // Acquire on `active` so subsequent field reads see the
            // values published by the writing thread's release-store.
            if (s->active.load(std::memory_order_acquire) &&
                s->tid.load(std::memory_order_relaxed) != 0) {
                ++active_count;
            }
        }
        LOG_CRITICAL(Common, "[SyncTrace] active waiters: {}", active_count);

        // Second pass: dump each active slot.
        for (WaitSlot* s = head; s != nullptr; s = s->next) {
            if (!s->active.load(std::memory_order_acquire)) continue;
            const u64 tid = s->tid.load(std::memory_order_relaxed);
            if (tid == 0) continue;
            // Snapshot. Tearing: if the writer is mid-update we may
            // see a half-old/half-new field set. Acceptable for a
            // diagnostic dump.
            const auto when = s->when;
            const Op op = s->op;
            const void* obj = s->obj;
            char name_copy[kNameMax];
            std::memcpy(name_copy, s->name, kNameMax);
            const u64 arg1 = s->arg1;

            const auto age_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - when).count();
            LOG_CRITICAL(
                Common,
                "[SyncTrace]   tid={} {} obj={} name=\"{}\" arg1={:#x} "
                "waiting_ms={}",
                tid, OpName(op), obj, name_copy, arg1, age_ms);
        }
    }

    // 2) Rwlock holder / waiter map — critical for diagnosing thread-holds-
    //    read-lock-while-blocked deadlocks. std::shared_timed_mutex does not
    //    expose its internal state; this side-table is the only way to see
    //    it. POPULATED ONLY IF rwlock-tier tracing is enabled.
    {
        std::lock_guard lk(g_rwlock_mtx);
        LOG_CRITICAL(Common, "[SyncTrace] rwlocks in map: {}{}",
                     g_rwlocks.size(),
                     rwlock_on ? "" : " (rwlock-tier tracing was DISABLED — "
                                       "map is empty by design)");
        for (const auto& [lock, info] : g_rwlocks) {
            const bool has_activity =
                info.writer_tid != 0 || !info.readers.empty() ||
                !info.waiting_writers.empty() || !info.waiting_readers.empty();
            if (!has_activity) continue;
            LOG_CRITICAL(
                Common,
                "[SyncTrace]   rwlock={} writer_tid={} readers={} "
                "waiting_writers={} waiting_readers={}",
                lock, info.writer_tid, info.readers.size(),
                info.waiting_writers.size(), info.waiting_readers.size());
            if (info.writer_tid != 0) {
                const auto age_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - info.writer_acquired_at).count();
                LOG_CRITICAL(Common,
                             "[SyncTrace]     WRITER tid={} held_ms={}",
                             info.writer_tid, age_ms);
            }
            for (const auto& [tid, when] : info.readers) {
                const auto age_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - when).count();
                LOG_CRITICAL(Common,
                             "[SyncTrace]     READER tid={} held_ms={}",
                             tid, age_ms);
            }
            for (const auto& [tid, when] : info.waiting_writers) {
                const auto age_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - when).count();
                LOG_CRITICAL(
                    Common,
                    "[SyncTrace]     WAITING(W) tid={} waiting_ms={}",
                    tid, age_ms);
            }
            for (const auto& [tid, when] : info.waiting_readers) {
                const auto age_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - when).count();
                LOG_CRITICAL(
                    Common,
                    "[SyncTrace]     WAITING(R) tid={} waiting_ms={}",
                    tid, age_ms);
            }
        }
    }

    // 3) Ring buffer — recent signals and slow waits, oldest first.
    //    ALWAYS POPULATED.
    const u64 head = g_write_idx.load(std::memory_order_relaxed);
    const u64 start = (head > kRingSize) ? (head - kRingSize) : 1;
    u64 shown = 0;
    LOG_CRITICAL(Common,
                 "[SyncTrace] ring: head_seq={} window=[{}..{}) capacity={}",
                 head, start, head, kRingSize);
    for (u64 seq = start; seq < head; ++seq) {
        const Event& e = g_ring[seq & kRingMask];
        if (e.seq != seq) continue; // torn / overwritten / empty
        const auto age_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - e.when).count();
        LOG_CRITICAL(
            Common,
            "[SyncTrace]   seq={} t-{}ms tid={} {} obj={} name=\"{}\" "
            "arg1={:#x} arg2={:#x} waiters={} result={:#x}",
            e.seq, age_ms, e.tid, OpName(e.op), e.obj, e.name,
            e.arg1, e.arg2, e.waiters, e.result);
        ++shown;
    }
    LOG_CRITICAL(Common, "[SyncTrace] ==== end dump ({} entries) ====", shown);
}

} // namespace Common::SyncTrace
