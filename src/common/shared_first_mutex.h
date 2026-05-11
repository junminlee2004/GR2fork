// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace Common {

// Like std::shared_mutex, but reader has priority over writer.
//
// PERF(GR2FORK release_v2_2): atomic fast path for the reader hot path.
//
// The previous implementation took `std::mutex` + ran a
// `condition_variable::wait` predicate on EVERY `lock_shared` /
// `unlock_shared`, even when uncontended. The Rasterizer's
// `mapped_ranges_mutex` is read-locked from page-fault handlers
// (GAME_MainThread, via GuestFaultSignalHandler →
// BufferCache::InvalidateMemory → ...), from IsMapped checks called by
// arbitrary threads, and from the GpuComm draw loop's
// SynchronizeBuffersInRange traversal. Each of those paid one futex on
// the std::mutex even with zero contention.
//
// Visible in non-GpuComm threads:
//   * GAME_MainThread: 0.04% pthread_mutex_lock@@GLIBC_2.2.5 →
//     Common::SharedFirstMutex::lock_shared()
//   * MTTW: 0.02% Common::SharedFirstMutex::lock_shared()
//
// Design:
//   * Reader fast path: a single atomic CAS on the readers count.
//     Falls back to the mtx+cv slow path only if a writer is currently
//     active (readers == -1).
//   * Writer fast path: still requires mtx+cv because writer waits for
//     readers to drain — there's no purely-atomic way to express that.
//   * unlock_shared: a single atomic fetch_sub. Only takes mtx+notifies
//     if (a) we were the last reader AND (b) a writer is queued. The
//     waiting_writers_ counter lets us skip the mutex in the no-writer
//     case (the most common one).
//
// Reader-priority is preserved: the reader fast path does NOT defer to
// queued writers. It only blocks if a writer has already taken the
// writer slot (readers_ == -1). That matches the original behavior:
// `cv.wait(..., [&]{ return !writer_active; })` — readers ignored
// queued writers.
//
// Memory ordering: SC on the cross-thread synchronization
// (waiting_writers_ + readers_ in the reader-last-out / writer-queued
// pair) to ensure no lost wakeups across architectures. On x86-64
// (Jun's target) SC on RMWs compiles to lock-prefixed instructions
// that are essentially free vs release/acquire.
class SharedFirstMutex {
public:
    void lock_shared() {
        // Fast path: bump readers atomically if no writer is active.
        // Reader-priority — we don't honor queued writers.
        int r = readers_.load(std::memory_order_relaxed);
        for (;;) {
            if (r >= 0) {
                if (readers_.compare_exchange_weak(r, r + 1,
                                                   std::memory_order_acquire,
                                                   std::memory_order_relaxed)) {
                    return;
                }
                continue; // r reloaded by CAS on failure
            }
            // r == -1: writer active. Slow path — wait until writer releases.
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] {
                return readers_.load(std::memory_order_acquire) >= 0;
            });
            r = readers_.load(std::memory_order_relaxed);
            // lk released here; loop and retry CAS lock-free.
        }
    }

    void unlock_shared() {
        // Fast path: just decrement.
        const int prev = readers_.fetch_sub(1, std::memory_order_seq_cst);
        if (prev != 1) {
            // Other readers still hold the lock; nobody to wake.
            return;
        }
        // We were the last reader. Notify any queued writer. The
        // waiting_writers_ load is gated against the writer's seq_cst
        // fetch_add to prevent lost wakeup.
        if (waiting_writers_.load(std::memory_order_seq_cst) == 0) {
            return;
        }
        std::lock_guard<std::mutex> lk(mtx_);
        cv_.notify_all();
    }

    void lock() {
        // Writer announces itself BEFORE taking the mtx so that any
        // reader-last-out coming through unlock_shared sees us.
        waiting_writers_.fetch_add(1, std::memory_order_seq_cst);
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [this] {
            int expected = 0;
            return readers_.compare_exchange_strong(expected, -1,
                                                    std::memory_order_acq_rel,
                                                    std::memory_order_relaxed);
        });
        waiting_writers_.fetch_sub(1, std::memory_order_release);
    }

    void unlock() {
        // Release the writer slot. Both new readers and queued writers
        // may now proceed.
        readers_.store(0, std::memory_order_release);
        std::lock_guard<std::mutex> lk(mtx_);
        cv_.notify_all();
    }

private:
    // readers_ semantics:
    //   >= 0  : count of active readers (0 means free)
    //   == -1 : exclusive writer is active (no readers, no other writer)
    std::atomic<int> readers_{0};
    // Number of writers currently queued in lock(). Reader uses this on
    // the unlock-shared fast path to decide whether to take the mtx.
    std::atomic<int> waiting_writers_{0};
    // Slow-path coordination. Only touched when a reader contends with
    // an active writer, or the last reader needs to wake a queued writer,
    // or a writer is itself queued.
    std::mutex mtx_;
    std::condition_variable cv_;
};

} // namespace Common
