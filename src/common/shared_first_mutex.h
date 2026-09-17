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
 * Like std::shared_mutex, but a reader outranks a writer: a writer that is only
 * waiting has not taken the word, so it never holds off an incoming reader.
 * Writers can therefore be starved by busy readers - the intended trade.
 *
 * Shared acquisition is consequently recursive, which this codebase needs: guest
 * read faults re-enter the memory manager's lock, and std::shared_mutex deadlocks
 * there on MSVC's SRWLock and on libc++ (both admit a waiting writer ahead of a
 * new reader) while glibc's reader-preferring rwlock does not, so the failure
 * would appear only off Linux.
 *
 * mtx and cv are the contended slow path only; an uncontended lock_shared is a
 * load plus a CAS, which matters because the GPU command thread takes this lock
 * thousands of times per frame.
 */
class SharedFirstMutex {
public:
    void lock() {
        std::unique_lock<std::mutex> lock(mtx);
        // seq_cst pairs with unlock_shared's waiting_writers load: one always sees the other.
        waiting_writers.fetch_add(1, std::memory_order_seq_cst);
        std::uint32_t expected = 0;
        while (!state.compare_exchange_weak(expected, WRITER_BIT, std::memory_order_acq_rel,
                                            std::memory_order_relaxed)) {
            expected = 0;
            // notify_all is issued under mtx, held until wait() releases it: no wakeup is lost.
            cv.wait(lock);
        }
        waiting_writers.fetch_sub(1, std::memory_order_release);
    }

    bool try_lock() {
        std::lock_guard<std::mutex> lock(mtx);
        // The CAS must run under mtx: lock_shared's slow path increments after its predicate.
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
        // A writer only sets its bit under mtx, which we hold.
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
        // Last reader out. Only a writer waits on the drain, so skip mtx when none is pending -
        // this runs per guest memory copy.
        if (waiting_writers.load(std::memory_order_seq_cst) != 0) {
            std::lock_guard<std::mutex> lock(mtx);
            cv.notify_all();
        }
    }

private:
    static constexpr std::uint32_t WRITER_BIT = 1u << 31;
    static constexpr std::uint32_t READER_MASK = WRITER_BIT - 1u;

    /// Adds a reader unless a writer holds the lock; the test and the increment are one CAS.
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
