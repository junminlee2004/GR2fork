// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/spin_lock.h"
#include <atomic>
#include <thread>
#include <immintrin.h> // For _mm_pause

// Zen 4 Architecture Optimization
// _mm_pause() takes ~140 cycles on Zen 4 (vs ~10 on older CPUs).
// We use fewer pauses to avoid stalling the pipeline too long.

namespace Common {

    void SpinLock::lock() {
        // 1. Fast Path: Try to acquire immediately (Optimistic Lock)
        // Using 'acquire' ensures memory ordering for the critical section.
        if (!lck.test_and_set(std::memory_order_acquire)) {
            return;
        }

        // 2. Contention Path: Exponential Backoff
        // "Test-Test-And-Set" (TTAS) pattern reduces cache coherency traffic.
        int backoff = 1;
        while (true) {
            // A. Read-Only Loop: Wait for lock to become free *before* trying to write.
            // This prevents "cache line bouncing" between CCX complexes on Ryzen.
            while (lck.test(std::memory_order_relaxed)) {
                // Spin with exponential backoff
                for (int i = 0; i < backoff; ++i) {
                    _mm_pause();
                }

                // Increase backoff, capping at a reasonable limit (~2000 cycles)
                if (backoff < 16) {
                    backoff <<= 1;
                } else {
                    // B. Yield/Sleep: If we've spun too long, force a context switch.
                    // This breaks the "busy-wait storm" seen in your perf report.

                    // C++20: If available, .wait() is superior (uses futex/monitor).
                    // If not, yield() is the fallback.
                    #if defined(__cpp_lib_atomic_wait) && (__cpp_lib_atomic_wait >= 201907L)
                    lck.wait(true, std::memory_order_relaxed);
                    #else
                    std::this_thread::yield();
                    #endif

                    backoff = 1; // Reset backoff after waking up
                }
            }

            // C. Retry Acquisition
            // Only attempt to write (CAS) if we saw the lock was free in step A.
            if (!lck.test_and_set(std::memory_order_acquire)) {
                return;
            }
        }
    }

    void SpinLock::unlock() {
        lck.clear(std::memory_order_release);

        // C++20: Notify waiting threads to wake up.
        // This is required if we used .wait() in the lock function.
        #if __cpp_lib_atomic_flag_test >= 201907L
        lck.notify_one();
        #endif
    }

    bool SpinLock::try_lock() {
        return !lck.test_and_set(std::memory_order_acquire);
    }

} // namespace Common
