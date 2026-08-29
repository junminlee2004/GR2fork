// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <optional>

#include "common/div_ceil.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"

#ifdef __unix__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif
#include "common/assert.h"
#include "common/debug.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"

namespace VideoCore {

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
using LockType = Common::AdaptiveMutex;
#else
using LockType = Common::SpinLock;
#endif

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

        if constexpr (type == Type::GPU && enable) {
            // GPU bits are only ever mutated on the GPU command thread, so this
            // is a plain counter. It advances on marks alone: an unchanged
            // value between two points on that thread proves no new GPU write
            // was recorded for this region in between, which is the guard the
            // offloaded readback path uses before clearing bits it earlier
            // snapshotted.
            ++gpu_write_seq;
        }
        RegionBits& bits = GetRegionBits<type>();
        // A range already in the target state makes the write below an
        // identity: the bits cannot change, so the protection masks derived
        // from them cannot change either. Skipping it also leaves the sequence
        // count stable for concurrent lock-free readers.
        if constexpr (enable) {
            if (bits.AllInRange(start_page, end_page)) {
                return;
            }
        } else {
            if (!bits.AnyInRange(start_page, end_page)) {
                return;
            }
        }
        WriteScope write_scope{*this};
        if constexpr (enable) {
            bits.SetRange(start_page, end_page);
        } else {
            bits.UnsetRange(start_page, end_page);
        }
        if constexpr (type == Type::CPU) {
            RefreshCpuSummary(start_page, end_page);
            UpdateProtection<!enable, false>();
        } else if (EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise) {
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

        // Only the clearing form mutates the bits; entering the scope
        // conditionally keeps pure iteration off the writers' path.
        std::optional<WriteScope> write_scope;
        if constexpr (clear) {
            write_scope.emplace(*this);
        }
        RegionBits& bits = GetRegionBits<type>();
        RegionBits mask(bits, start_page, end_page);

        if constexpr (clear) {
            bits.UnsetRange(start_page, end_page);
            if constexpr (type == Type::CPU) {
                RefreshCpuSummary(start_page, end_page);
                UpdateProtection<true, false>();
            } else if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled) {
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

        // Summary front: one mask test answers the clean common case without
        // scanning the bit words. A set summary bit only proves the word is
        // worth scanning, so stale-set bits fall through to the exact answer.
        if constexpr (type == Type::CPU) {
            if ((cpu_summary_ & SummaryMask(start_page, end_page)) == 0) {
                return false;
            }
        }
        // Ask the range directly: materialising a masked copy of the whole
        // region's bits to answer a yes/no question was the single largest
        // cost on the bind path.
        return GetRegionBits<type>().AnyInRange(start_page, end_page);
    }

    /**
     * Read whether a range is modified without taking the lock.
     *
     * The dirty bits are read optimistically between two reads of a sequence
     * counter that writers make odd while mutating. An unchanged even counter
     * proves no writer ran during the read, so the answer is exactly what the
     * locked query would have returned. Contended acquisitions of this lock
     * between the GPU thread and the guest fault handler are otherwise the
     * dominant cost of every buffer bind.
     */
    template <Type type>
    [[nodiscard]] bool PeekRegionModified(u64 offset, u64 size) noexcept {
        if constexpr (type == Type::CPU) {
            // Clean-case fast path: half the function's measured cost was
            // call glue executed before the summary test. The degenerate
            // range guards must run before SummaryMask (out-of-range pages
            // are shift-count UB), and the seq/fence ordering here is
            // exactly the outlined protocol's - a stale summary read is
            // caught by the seq re-check and falls through.
            const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
            const size_t end_page =
                Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
            if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
                return false;
            }
            const u32 before = seq.load(std::memory_order_acquire);
            if (!(before & 1u)) {
                const u16 summary = cpu_summary_;
                std::atomic_thread_fence(std::memory_order_acquire);
                if (seq.load(std::memory_order_relaxed) == before &&
                    (summary & SummaryMask(start_page, end_page)) == 0) {
                    return false;
                }
            }
        }
        return PeekRegionModifiedSlow<type>(offset, size);
    }

    template <Type type>
    [[nodiscard]] SHAD_NO_INLINE bool PeekRegionModifiedSlow(u64 offset, u64 size) noexcept {
        for (u32 attempt = 0; attempt < 4; ++attempt) {
            const u32 before = seq.load(std::memory_order_acquire);
            if (before & 1u) {
                continue; // writer in flight
            }
            const bool result = IsRegionModified<type>(offset, size);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq.load(std::memory_order_relaxed) == before) {
                return result;
            }
        }
        std::scoped_lock lk{lock};
        return IsRegionModified<type>(offset, size);
    }

    /// Lock-free counterpart of PeekRegionModified for full coverage.
    template <Type type>
    [[nodiscard]] bool PeekRegionFullySet(u64 offset, u64 size) noexcept {
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }
        for (u32 attempt = 0; attempt < 4; ++attempt) {
            const u32 before = seq.load(std::memory_order_acquire);
            if (before & 1u) {
                continue;
            }
            const bool result = GetRegionBits<type>().AllInRange(start_page, end_page);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq.load(std::memory_order_relaxed) == before) {
                return result;
            }
        }
        std::scoped_lock lk{lock};
        return GetRegionBits<type>().AllInRange(start_page, end_page);
    }

    /// Scope guard marking a mutation of the tracked bits for readers.
    struct WriteScope {
        explicit WriteScope(RegionManager& m) : mgr{m} {
            mgr.seq.fetch_add(1, std::memory_order_acq_rel);
        }
        ~WriteScope() {
            mgr.seq.fetch_add(1, std::memory_order_release);
        }
        RegionManager& mgr;
    };

    std::atomic<u32> seq{0};
    // Counts GPU-bit marks. GPU-command-thread confined; see ChangeRegionState.
    u64 gpu_write_seq{0};
    LockType lock;

    // Word epochs advance whenever guest bytes in a span may change outside
    // the write watchers' sight: write protection loss, guest protection
    // grants, and direct backing writes. A consumer that pairs an epoch sum
    // with a content record gets a cheap it-cannot-have-changed certificate.
    static constexpr u64 EPOCH_WORD_BITS = 18; // 256KB per word
    static constexpr size_t NUM_EPOCH_WORDS = TRACKER_HIGHER_PAGE_SIZE >> EPOCH_WORD_BITS;
    static constexpr u64 EPOCH_SUB_BITS = 16; // 64KB per subword
    static constexpr size_t NUM_EPOCH_SUBS = TRACKER_HIGHER_PAGE_SIZE >> EPOCH_SUB_BITS;

    void BumpWordEpochs(u64 offset, u64 size, u8 cause) noexcept {
        if (size == 0) {
            return;
        }
        const size_t w0 = std::min<u64>(offset >> EPOCH_WORD_BITS, NUM_EPOCH_WORDS - 1);
        const size_t w1 =
            std::min<u64>((offset + size - 1) >> EPOCH_WORD_BITS, NUM_EPOCH_WORDS - 1);
        for (size_t w = w0; w <= w1; ++w) {
            word_epochs[w].fetch_add(1, std::memory_order_release);
            last_bump_cause[w].store(cause, std::memory_order_relaxed);
        }
        const size_t s0 = std::min<u64>(offset >> EPOCH_SUB_BITS, NUM_EPOCH_SUBS - 1);
        const size_t s1 = std::min<u64>((offset + size - 1) >> EPOCH_SUB_BITS, NUM_EPOCH_SUBS - 1);
        for (size_t sub = s0; sub <= s1; ++sub) {
            sub_epochs[sub].fetch_add(1, std::memory_order_release);
        }
    }

    void PoisonEpochWords(u64 offset, u64 size) noexcept {
        if (size == 0) {
            return;
        }
        const size_t w0 = std::min<u64>(offset >> EPOCH_WORD_BITS, NUM_EPOCH_WORDS - 1);
        const size_t w1 =
            std::min<u64>((offset + size - 1) >> EPOCH_WORD_BITS, NUM_EPOCH_WORDS - 1);
        u32 mask = 0;
        for (size_t w = w0; w <= w1; ++w) {
            mask |= 1u << w;
        }
        poison_words.fetch_or(mask, std::memory_order_release);
    }

    // 16-bit summary over the CPU dirty bits: bit k covers pages
    // [k*64, k*64+64) - exactly one storage word of the bitset. Maintained
    // under the same write scope as the bits, so the lock-free peek's seqlock
    // protocol covers it too; the region starts fully dirty, so the summary
    // starts fully set. Readers treat it as a prefilter only: clean is
    // authoritative (a clear bit proves its word is zero), dirty falls
    // through to the exact scan.
    static constexpr size_t PAGES_PER_SUMMARY_BIT = 64;
    static_assert(NUM_PAGES_PER_REGION / PAGES_PER_SUMMARY_BIT <= 16);

    static u16 SummaryMask(size_t start_page, size_t end_page) noexcept {
        const size_t w0 = start_page / PAGES_PER_SUMMARY_BIT;
        const size_t w1 = (end_page - 1) / PAGES_PER_SUMMARY_BIT;
        return static_cast<u16>(((2u << w1) - (1u << w0)) & 0xFFFFu);
    }

    void RefreshCpuSummary(size_t start_page, size_t end_page) noexcept {
        const size_t w0 = start_page / PAGES_PER_SUMMARY_BIT;
        const size_t w1 = (end_page - 1) / PAGES_PER_SUMMARY_BIT;
        for (size_t w = w0; w <= w1; ++w) {
            const size_t p0 = w * PAGES_PER_SUMMARY_BIT;
            if (cpu.AnyInRange(p0, p0 + PAGES_PER_SUMMARY_BIT)) {
                cpu_summary_ |= static_cast<u16>(1u << w);
            } else {
                cpu_summary_ &= static_cast<u16>(~(1u << w));
            }
        }
    }

    std::array<std::atomic<u64>, NUM_EPOCH_WORDS> word_epochs{};
    std::array<std::atomic<u64>, NUM_EPOCH_SUBS> sub_epochs{};
    std::array<std::atomic<u8>, NUM_EPOCH_WORDS> last_bump_cause{};
    std::atomic<u32> poison_words{0};

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
    u16 cpu_summary_{0xFFFF}; // cpu.Fill() in the ctor => fully set
    RegionBits cpu;
    RegionBits gpu;
    RegionBits writeable;
    RegionBits readable;
};

} // namespace VideoCore
