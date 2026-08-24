// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace Common {

/**
 * Like std::shared_mutex, but reader has priority over writer.
 *
 * Ownership lives in one atomic word: the high bit means a writer holds the
 * lock, the remaining bits count readers. A writer takes ownership only by
 * moving that word from exactly zero to the writer bit, which gives the two
 * properties this type exists for:
 *
 *  - Readers outrank writers. A writer that is merely *waiting* has not set the
 *    bit, so it never holds off an incoming reader. Writers can therefore be
 *    starved by a busy reader, which is the intended trade.
 *  - Shared acquisition is recursive. A thread already holding a read keeps the
 *    word non-zero, so no writer can acquire underneath it, so the same thread
 *    can take another read without deadlocking. Guest read faults re-enter the
 *    memory manager's lock this way, and std::shared_mutex is unusable here for
 *    that reason - MSVC's SRWLock and libc++ both admit a waiting writer ahead
 *    of a new reader and deadlock, while glibc's reader-preferring rwlock does
 *    not, so the failure would appear only off Linux.
 *
 * The mutex and condition variable below are the slow path only: contended
 * acquisition, and waking a writer once the last reader leaves. An uncontended
 * lock_shared is a load and a compare-exchange, which matters because the GPU
 * command thread takes this lock thousands of times per frame.
 */
class SharedFirstMutex {
public:
    void lock() {
        std::unique_lock<std::mutex> lock(mtx);
        // Publishing the intent to write does not block readers; it only asks
        // the last reader out to wake us. Ordering with unlock_shared's check
        // is sequentially consistent so exactly one of the two sees the other.
        waiting_writers.fetch_add(1, std::memory_order_seq_cst);
        std::uint32_t expected = 0;
        while (!state.compare_exchange_weak(expected, WRITER_BIT, std::memory_order_acq_rel,
                                            std::memory_order_relaxed)) {
            expected = 0;
            // Notifications are issued while holding mtx, which we hold here
            // until wait() releases it, so a wakeup cannot be missed.
            cv.wait(lock);
        }
        waiting_writers.fetch_sub(1, std::memory_order_release);
    }

    bool try_lock() {
        std::lock_guard<std::mutex> lock(mtx);
        std::uint32_t expected = 0;
        return state.compare_exchange_strong(expected, WRITER_BIT, std::memory_order_acq_rel,
                                             std::memory_order_relaxed);
    }

    template <typename Clock, typename Duration>
    bool try_lock_until(const std::chrono::time_point<Clock, Duration>& abs_time) {
        std::unique_lock<std::mutex> lock(mtx);
        waiting_writers.fetch_add(1, std::memory_order_seq_cst);
        std::uint32_t expected = 0;
        bool acquired = true;
        while (!state.compare_exchange_weak(expected, WRITER_BIT, std::memory_order_acq_rel,
                                            std::memory_order_relaxed)) {
            expected = 0;
            if (cv.wait_until(lock, abs_time) == std::cv_status::timeout) {
                // The deadline may still coincide with the lock falling free.
                expected = 0;
                acquired = state.compare_exchange_strong(
                    expected, WRITER_BIT, std::memory_order_acq_rel, std::memory_order_relaxed);
                break;
            }
        }
        waiting_writers.fetch_sub(1, std::memory_order_release);
        return acquired;
    }

    void unlock() {
        state.fetch_and(~WRITER_BIT, std::memory_order_release);
        std::lock_guard<std::mutex> lock(mtx);
        cv.notify_all();
    }

    void lock_shared() {
        if (TryAddReader()) {
            return;
        }
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock,
                [this]() { return (state.load(std::memory_order_acquire) & WRITER_BIT) == 0; });
        // A writer only sets its bit while holding mtx, which we hold, so the
        // word cannot gain one between the predicate and this increment.
        state.fetch_add(1, std::memory_order_acq_rel);
    }

    bool try_lock_shared() {
        return TryAddReader();
    }

    template <typename Clock, typename Duration>
    bool try_lock_shared_until(const std::chrono::time_point<Clock, Duration>& abs_time) {
        if (TryAddReader()) {
            return true;
        }
        std::unique_lock<std::mutex> lock(mtx);
        if (!cv.wait_until(lock, abs_time, [this]() {
                return (state.load(std::memory_order_acquire) & WRITER_BIT) == 0;
            })) {
            return false;
        }
        state.fetch_add(1, std::memory_order_acq_rel);
        return true;
    }

    void unlock_shared() {
        const std::uint32_t prev = state.fetch_sub(1, std::memory_order_seq_cst);
        if ((prev & READER_MASK) != 1) {
            return; // other readers remain; no writer can proceed yet
        }
        // Last reader out. Only a writer waits on the drain, so with none
        // pending there is nobody to wake and the mutex can be skipped - the
        // whole point of the fast path, since this runs per guest memory copy.
        if (waiting_writers.load(std::memory_order_seq_cst) != 0) {
            std::lock_guard<std::mutex> lock(mtx);
            cv.notify_all();
        }
    }

private:
    static constexpr std::uint32_t WRITER_BIT = 1u << 31;
    static constexpr std::uint32_t READER_MASK = WRITER_BIT - 1u;

    /// Adds one reader unless a writer currently holds the lock. The exchange
    /// makes the test and the increment one step, so a writer acquiring
    /// concurrently either loses the race or is seen.
    bool TryAddReader() noexcept {
        std::uint32_t cur = state.load(std::memory_order_acquire);
        while ((cur & WRITER_BIT) == 0) {
            if (state.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
                return true;
            }
        }
        return false;
    }

    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<std::uint32_t> state{0};
    std::atomic<std::uint32_t> waiting_writers{0};
};

} // namespace Common
