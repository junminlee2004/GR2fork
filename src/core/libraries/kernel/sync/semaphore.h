// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <type_traits>

#include "common/assert.h"
#include "common/types.h"

#ifdef _WIN64
#include <windows.h>
#elif defined(__APPLE__)
#include <dispatch/dispatch.h>
#else
#include <semaphore>
// PERF(GR2FORK release_v2_4): direct-futex headers for the Semaphore<1>
// specialization below. Only the binary-semaphore case uses these — the
// generic Semaphore<max> still uses std::counting_semaphore<max>.
#include <cerrno>
#include <ctime>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libraries::Kernel {

template <s64 max>
class Semaphore {
public:
    Semaphore(s32 initialCount)
#if !defined(_WIN64) && !defined(__APPLE__)
        : sem{initialCount}
#endif
    {
#ifdef _WIN64
        sem = CreateSemaphore(nullptr, initialCount, max, nullptr);
        ASSERT(sem);
#elif defined(__APPLE__)
        sem = dispatch_semaphore_create(initialCount);
        ASSERT(sem);
#endif
    }

    ~Semaphore() {
#ifdef _WIN64
        CloseHandle(sem);
#elif defined(__APPLE__)
        dispatch_release(sem);
#endif
    }

    void release() {
#ifdef _WIN64
        ReleaseSemaphore(sem, 1, nullptr);
#elif defined(__APPLE__)
        dispatch_semaphore_signal(sem);
#else
        sem.release();
#endif
    }

    void acquire() {
#ifdef _WIN64
        for (;;) {
            u64 res = WaitForSingleObjectEx(sem, INFINITE, true);
            if (res == WAIT_OBJECT_0) {
                return;
            }
        }
#elif defined(__APPLE__)
        for (;;) {
            const auto res = dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            if (res == 0) {
                return;
            }
        }
#else
        sem.acquire();
#endif
    }

    bool try_acquire() {
#ifdef _WIN64
        return WaitForSingleObjectEx(sem, 0, true) == WAIT_OBJECT_0;
#elif defined(__APPLE__)
        return dispatch_semaphore_wait(sem, DISPATCH_TIME_NOW) == 0;
#else
        return sem.try_acquire();
#endif
    }

    template <class Rep, class Period>
    bool try_acquire_for(const std::chrono::duration<Rep, Period>& rel_time) {
#ifdef _WIN64
        auto rel_time_ms = std::chrono::ceil<std::chrono::milliseconds>(rel_time);
        do {
            const auto start_time = std::chrono::high_resolution_clock::now();
            u64 timeout_ms = static_cast<u64>(rel_time_ms.count());
            u64 res = WaitForSingleObjectEx(sem, timeout_ms, true);
            if (res == WAIT_OBJECT_0) {
                return true;
            } else if (res == WAIT_IO_COMPLETION) {
                auto elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
                rel_time_ms -= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time);
            } else {
                return false;
            }
        } while (rel_time_ms.count() > 0);

        return false;
#elif defined(__APPLE__)
        const auto rel_time_ns = std::chrono::ceil<std::chrono::nanoseconds>(rel_time);
        const auto timeout = dispatch_time(DISPATCH_TIME_NOW, rel_time_ns.count());
        return dispatch_semaphore_wait(sem, timeout) == 0;
#else
        return sem.try_acquire_for(rel_time);
#endif
    }

    template <class Clock, class Duration>
    bool try_acquire_until(const std::chrono::time_point<Clock, Duration>& abs_time) {
        const auto current = Clock::now();
        if (current >= abs_time) {
            return try_acquire();
        }
        return try_acquire_for(abs_time - current);
    }

private:
#ifdef _WIN64
    HANDLE sem;
#elif defined(__APPLE__)
    dispatch_semaphore_t sem;
#else
    std::counting_semaphore<max> sem;
#endif
};

#if !defined(_WIN64) && !defined(__APPLE__)
// PERF(GR2FORK release_v2_4): direct-futex specialization for binary
// semaphores on Linux.
//
// What changed: the Semaphore<1> path (BinarySemaphore — every guest
// thread's wake_sema, every OrbisSem WaitingThread::sem, every condvar
// signaller's deferred-wake target) now goes straight to
// FUTEX_WAIT_PRIVATE on contention instead of through libstdc++'s
// __atomic_semaphore::_M_acquire spin. libstdc++ runs ~16 PAUSE-loop
// iterations on the atomic counter before falling to futex; for every
// genuine block (counter is 0, producer hasn't released) those
// iterations are pure waste, ~800 cycles per acquire at ~50 cyc each
// on Zen4. Pre-v2_4 this showed at 0.05% on MTTW alone in the cycles
// trace as `std::__detail::__atomic_spin<...>`, plus mirror time on
// every other thread that ever blocks (BinarySemaphore is the master
// block-and-wake primitive: Pthread::Sleep → wake_sema.acquire on
// every cv wait, OrbisSem slow path, condvar wakeup deferral).
//
// What did NOT change: Semaphore<max> (CountingSemaphore, used by
// PthreadSem) still goes through std::counting_semaphore<max>. The
// counting case is more involved (multi-permit accounting + N-waiter
// wake fanout) and isn't on the hot path for our workload — the
// observable spin cost is in the binary path. Keeping the counting
// case on libstdc++ minimises the surface area of this patch.
//
// Race / lost-wakeup analysis:
//  - release() does state.exchange(kHasToken, release). If old was
//    kEmpty, a waiter may exist or be about to exist; issue
//    FUTEX_WAKE(1). If old was kHasToken, no waiter could possibly be
//    sleeping on this state (a waiter would have CAS-ed it to kEmpty
//    before sleeping), so skip the syscall.
//  - acquire() CAS-loops kHasToken→kEmpty; on failure (state was
//    kEmpty) it calls FUTEX_WAIT(kEmpty). The kernel atomically checks
//    state against the expected value BEFORE sleeping; if a release()
//    has landed in between our load and the syscall, FUTEX_WAIT
//    returns EAGAIN immediately and the loop retries the CAS. No lost
//    wakeup.
//  - Spurious wakeups (signal interruption, kernel artifacts) are
//    handled by the loop: any FUTEX_WAIT return is followed by a
//    fresh CAS attempt; only a successful CAS exits the loop.
//
// Memory ordering: release() uses release on the exchange so all writes
// before it (e.g., the producer's data publication, OrbisSem's
// `was_signaled = true` flag) are visible to the consumer that
// CAS-acquires. acquire() uses acquire on the successful CAS so the
// consumer sees the producer's published writes.
//
// Saturation: a binary semaphore caps at 1 token. Two consecutive
// release()s produce one acquirable token, not two. All current users
// pair each release with at most one waiter (Pthread::WakeAll iterates
// distinct defer_waiters[]; OrbisSem signals each waiter exactly once;
// condvar signaller releases exactly once per target wake_sema), so
// saturation is not observable.
template <>
class Semaphore<1> {
public:
    explicit Semaphore(s32 initialCount)
        : state_{initialCount > 0 ? kHasToken : kEmpty} {}

    ~Semaphore() = default;

    Semaphore(const Semaphore&) = delete;
    Semaphore& operator=(const Semaphore&) = delete;

    void release() {
        // Publish "token available". exchange returns prior value so we
        // can decide whether a wake is needed.
        const u32 prev = state_.exchange(kHasToken, std::memory_order_release);
        if (prev == kEmpty) {
            // A consumer either already CAS-failed (and will FUTEX_WAIT)
            // or is about to. Wake one if sleeping; the syscall is a
            // no-op if no waiter is queued at kernel level.
            FutexWake(1);
        }
        // prev == kHasToken: nobody could be sleeping on this state
        // (a waiter would have CAS-ed it to kEmpty), so skip syscall.
    }

    bool try_acquire() {
        u32 expected = kHasToken;
        return state_.compare_exchange_strong(
            expected, kEmpty,
            std::memory_order_acquire, std::memory_order_relaxed);
    }

    void acquire() {
        for (;;) {
            if (try_acquire()) [[likely]] return;
            // No token. FUTEX_WAIT atomically checks state_==kEmpty
            // before sleeping; an interleaving release returns EAGAIN
            // and the loop re-CASes without sleeping.
            const int rc = FutexWait(kEmpty, /*timeout=*/nullptr);
            (void)rc;
            // EAGAIN / EINTR / spurious wake: retry CAS. Any other
            // errno (shouldn't occur with our valid pointer + flags)
            // also falls through to retry; worst case is one extra CAS
            // attempt per wake.
        }
    }

    template <class Rep, class Period>
    bool try_acquire_for(const std::chrono::duration<Rep, Period>& rel_time) {
        if (try_acquire()) [[likely]] return true;
        const auto deadline = std::chrono::steady_clock::now() + rel_time;
        return AcquireUntilSteady(deadline);
    }

    template <class Clock, class Duration>
    bool try_acquire_until(const std::chrono::time_point<Clock, Duration>& abs_time) {
        if (try_acquire()) [[likely]] return true;
        // FUTEX_WAIT with our relative-timeout form measures on
        // CLOCK_MONOTONIC, so convert any caller's clock to a
        // steady_clock deadline for a stable countdown across spurious
        // wakeups.
        if constexpr (std::is_same_v<Clock, std::chrono::steady_clock>) {
            return AcquireUntilSteady(abs_time);
        } else {
            const auto now = Clock::now();
            if (now >= abs_time) return false;
            const auto deadline =
                std::chrono::steady_clock::now() + (abs_time - now);
            return AcquireUntilSteady(deadline);
        }
    }

private:
    static constexpr u32 kEmpty = 0;
    static constexpr u32 kHasToken = 1;
    std::atomic<u32> state_;

    bool AcquireUntilSteady(
        const std::chrono::time_point<std::chrono::steady_clock>& deadline) {
        for (;;) {
            if (try_acquire()) return true;
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) return false;
            const auto remaining = deadline - now;
            const auto secs =
                std::chrono::duration_cast<std::chrono::seconds>(remaining);
            const auto nsecs =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    remaining - secs);
            timespec ts;
            ts.tv_sec = static_cast<time_t>(secs.count());
            ts.tv_nsec = static_cast<long>(nsecs.count());
            const int rc = FutexWait(kEmpty, &ts);
            if (rc == -1 && errno == ETIMEDOUT) {
                // Final CAS attempt in case a release raced in within
                // the same slice — extremely unlikely but cheap.
                return try_acquire();
            }
            // EAGAIN / EINTR / 0: loop and retry CAS.
        }
    }

    int FutexWait(u32 expected, const timespec* ts) {
        return static_cast<int>(::syscall(SYS_futex, &state_,
                                          FUTEX_WAIT_PRIVATE, expected, ts,
                                          nullptr, 0));
    }

    void FutexWake(int n) {
        ::syscall(SYS_futex, &state_, FUTEX_WAKE_PRIVATE, n, nullptr,
                  nullptr, 0);
    }
};
#endif // !_WIN64 && !__APPLE__

using BinarySemaphore = Semaphore<1>;
using CountingSemaphore = Semaphore<0x7FFFFFFF /*ORBIS_KERNEL_SEM_VALUE_MAX*/>;

} // namespace Libraries::Kernel
