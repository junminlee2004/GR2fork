// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace Common {

// Like std::shared_mutex, but reader has priority over writer.
//
// GR2FORK PERF: atomic reader fast path. A mutex+cv implementation pays a futex on every
// lock_shared/unlock_shared even uncontended, and the Rasterizer's mapped_ranges_mutex is
// read-locked from page-fault handlers, IsMapped checks, and the GpuComm draw loop. Readers CAS
// the count, blocking only while a writer is active (readers_ == -1, preserving reader
// priority); unlock takes the mutex only when the last reader leaves with a writer queued.
// Writers keep mtx+cv. Seq_cst RMWs avoid lost wakeups and are ~free on x86-64.
class SharedFirstMutex {
public:
    void lock_shared() {
        // Fast path: bump readers atomically if no writer is active.
        // Reader-priority - we don't honor queued writers.
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
            // r == -1: writer active. Slow path - wait until writer releases.
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
