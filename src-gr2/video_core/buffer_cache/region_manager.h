// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <type_traits>
#include "common/config.h"
#include "common/div_ceil.h"
#include "common/logging/log.h"

#ifdef __linux__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif
#include "common/debug.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace VideoCore {

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
using SeqWriterMutex = Common::AdaptiveMutex;
#else
using SeqWriterMutex = Common::SpinLock;
#endif

// GR2FORK PERF: seqlock around RegionBits, replacing the upstream AdaptiveMutex/SpinLock -
// MemoryTracker's pure bit-read paths spend ~0.88% of total CPU in mutex lock/unlock. Writers
// (GpuComm, SIGSEGV-handler guest threads) hold writer_mu and bump seq; readers never lock.
class RegionSeqLock {
public:
    RegionSeqLock() = default;
    RegionSeqLock(const RegionSeqLock&) = delete;
    RegionSeqLock& operator=(const RegionSeqLock&) = delete;

    // Writer interface - compatible with std::scoped_lock<RegionSeqLock>.
    // Acquires writer mutex, bumps seq to odd. Symmetric unlock restores even.
    void lock() noexcept {
        writer_mu.lock();
        seq.fetch_add(1, std::memory_order_release);
    }
    void unlock() noexcept {
        seq.fetch_add(1, std::memory_order_release);
        writer_mu.unlock();
    }

    // Reader - conservative, no retry: returns `conservative_value` when a writer is in flight
    // or seq changed across the read. Callers must tolerate the fallback as safe - for the
    // IsRegion*Modified users "treat as modified" costs an extra sync, never a skipped one.
    template <typename F>
    bool read_conservative(bool conservative_value, F&& read_fn) const noexcept {
        const u64 s1 = seq.load(std::memory_order_acquire);
        if (s1 & 1ULL) {
            return conservative_value;
        }
        const bool result = read_fn();
        std::atomic_thread_fence(std::memory_order_acquire);
        const u64 s2 = seq.load(std::memory_order_acquire);
        if (s1 != s2) {
            return conservative_value;
        }
        return result;
    }

    // Reader - full retry, for snapshot iteration where partial results are unacceptable.
    // `read_fn` can run multiple times, so it must be side-effect-free.
    template <typename F>
    auto read_retry(F&& read_fn) const -> std::invoke_result_t<F> {
        for (;;) {
            const u64 s1 = seq.load(std::memory_order_acquire);
            if ((s1 & 1ULL) == 0ULL) {
                auto result = read_fn();
                std::atomic_thread_fence(std::memory_order_acquire);
                const u64 s2 = seq.load(std::memory_order_acquire);
                if (s1 == s2) {
                    return result;
                }
            }
            // Writer in flight or raced. Spin briefly, then yield-effective
            // pause. Writer crit sections are O(microseconds) at worst.
            for (int spin = 0; spin < 64; ++spin) {
                if ((seq.load(std::memory_order_relaxed) & 1ULL) == 0ULL) {
                    break;
                }
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                _mm_pause();
#elif defined(__aarch64__)
                __asm__ __volatile__("yield");
#endif
            }
        }
    }

private:
    mutable std::atomic<u64> seq{0};
    SeqWriterMutex writer_mu;
};

using LockType = RegionSeqLock;

/**
 * Allows tracking CPU and GPU modification of pages in a contigious 16MB virtual address region.
 * Information is stored in bitsets for spacial locality and fast update of single pages.
 */
class RegionManager {
public:
    explicit RegionManager(PageManager* tracker_, VAddr cpu_addr_)
        : tracker{tracker_}, cpu_addr{cpu_addr_} {
        cpu.Fill();
        gpu.Clear();
        writeable.Fill();
        readable.Fill();
    }
    explicit RegionManager() = default;

    void SetCpuAddress(VAddr new_cpu_addr) {
        cpu_addr = new_cpu_addr;
    }

    VAddr GetCpuAddr() const {
        return cpu_addr;
    }

    static constexpr size_t SanitizeAddress(size_t address) {
        return static_cast<size_t>(std::max<s64>(static_cast<s64>(address), 0LL));
    }

    template <Type type>
    RegionBits& GetRegionBits() noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    template <Type type>
    const RegionBits& GetRegionBits() const noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    /**
     * Change the state of a range of pages
     *
     * @param dirty_addr    Base address to mark or unmark as modified
     * @param size          Size in bytes to mark or unmark as modified
     */
    template <Type type, bool enable>
    void ChangeRegionState(u64 dirty_addr, u64 size) noexcept(type == Type::GPU) {
        RENDERER_TRACE;
        const size_t offset = dirty_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        RegionBits& bits = GetRegionBits<type>();
        if constexpr (enable) {
            bits.SetRange(start_page, end_page);
        } else {
            bits.UnsetRange(start_page, end_page);
        }
        if constexpr (type == Type::CPU) {
            UpdateProtection<!enable, false>();
        } else if (Config::readbacksMode() == Config::GpuReadbacksMode::Precise) {
            // GR2FORK FIX: eager per-page protection only in Precise mode - Relaxed defers
            // protection to the ForEachModifiedRange path; firing on any non-Disabled mode
            // would make Relaxed behave identically to Precise.
            UpdateProtection<enable, true>();
        }
    }

    /**
     * Loop over each page in the given range, turn off those bits and notify the tracker if
     * needed. Call the given function on each turned off range.
     *
     * @param query_cpu_range Base CPU address to loop over
     * @param size            Size in bytes of the CPU range to loop over
     * @param func            Function to call for each turned off region
     */
    template <Type type, bool clear>
    void ForEachModifiedRange(VAddr query_cpu_range, s64 size, auto&& func) {
        RENDERER_TRACE;
        const size_t offset = query_cpu_range - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        RegionBits& bits = GetRegionBits<type>();
        RegionBits mask(bits, start_page, end_page);

        if constexpr (clear) {
            bits.UnsetRange(start_page, end_page);
            if constexpr (type == Type::CPU) {
                UpdateProtection<true, false>();
            } else if (Config::readbacks()) {
                UpdateProtection<false, true>();
            }
        }

        for (const auto& [start, end] : mask) {
            func(cpu_addr + start * TRACKER_BYTES_PER_PAGE, (end - start) * TRACKER_BYTES_PER_PAGE);
        }
    }

    /**
     * Returns true when a region has been modified
     *
     * @param offset Offset in bytes from the start of the buffer
     * @param size   Size in bytes of the region to query for modifications
     */
    template <Type type>
    [[nodiscard]] bool IsRegionModified(u64 offset, u64 size) noexcept {
        RENDERER_TRACE;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }
        const RegionBits& bits = GetRegionBits<type>();
        return bits.AnyInRange(start_page, end_page);
    }

    /**
     * GR2FORK PERF: true when EVERY page of the region
     * already carries the `type` modified bit. Read-only companion to
     * IsRegionModified, used by ForEachUploadRange's written-bind no-op peek
     * (skip the writer lock when CPU has no dirty bits AND GPU bits are all
     * set - the finalize would re-set already-set bits). Degenerate ranges
     * return false: conservative, the caller then takes the locked path.
     */
    template <Type type>
    [[nodiscard]] bool IsRegionAllMarked(u64 offset, u64 size) noexcept {
        RENDERER_TRACE;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }
        const RegionBits& bits = GetRegionBits<type>();
        return bits.AllInRange(start_page, end_page);
    }

    /**
     * Iterate already-snapshotted bits without touching internal state.
     *
     * Pure read - does NOT modify cpu/gpu/writeable/readable bitsets and does
     * NOT call UpdateProtection. Used by the seqlock read_retry path in
     * MemoryTracker::ForEachDownloadRange<false>: caller takes a snapshot of
     * GetRegionBits<type>() under read_retry, then calls this with the
     * snapshot to walk modified ranges outside the lock. The user `func`
     * therefore runs exactly once per snapshot, never re-runs on retry, and
     * sees a consistent point-in-time view of the bits.
     */
    void IterateRangesFromSnapshot(const RegionBits& snapshot, VAddr query_cpu_range,
                                   s64 size, auto&& func) const {
        RENDERER_TRACE;
        const size_t offset = query_cpu_range - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }
        const RegionBits mask(snapshot, start_page, end_page);
        for (const auto& [start, end] : mask) {
            func(cpu_addr + start * TRACKER_BYTES_PER_PAGE,
                 (end - start) * TRACKER_BYTES_PER_PAGE);
        }
    }

    LockType lock;

private:
    /**
     * Notify tracker about changes in the CPU tracking state of a word in the buffer
     *
     * @param word_index   Index to the word to notify to the tracker
     * @param current_bits Current state of the word
     * @param new_bits     New state of the word
     *
     * @tparam track True when the tracker should start tracking the new pages
     */
    template <bool track, bool is_read>
    void UpdateProtection() {
        RENDERER_TRACE;
        RegionBits mask = is_read ? (~gpu ^ readable) : (cpu ^ writeable);
        if (mask.None()) {
            return;
        }
        if constexpr (is_read) {
            readable = ~gpu;
        } else {
            writeable = cpu;
        }
        tracker->UpdatePageWatchersForRegion<track, is_read>(cpu_addr, mask);
    }

    PageManager* tracker;
    VAddr cpu_addr = 0;
    RegionBits cpu;
    RegionBits gpu;
    RegionBits writeable;
    RegionBits readable;
};

} // namespace VideoCore
