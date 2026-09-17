// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <utility>

#include "common/div_ceil.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"

#ifdef __unix__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif
#include "common/assert.h"
#include "common/cpu_pause.h"
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
 * The region lock: a bounded spin (tracker_lock_spin_rounds, see that setting for why) in
 * front of the blocking acquire.
 *
 * Counter invariant: every increment sits inside the `rounds != 0` gate and
 * must never be hoisted out of it; gpu_spin_rounds is non-zero on the
 * GpuCommandProcessor thread alone (latched once in Liverpool::Process), and
 * Drain() is called from Rasterizer::OnSubmit on that same thread. The
 * counters are therefore GPU-command-thread confined and need no atomics.
 */
class RegionLock {
public:
    void lock() noexcept {
        // Budget first: at 0 this stays the plain blocking acquire, no extra trylock.
        const u32 rounds = gpu_spin_rounds;
        if (rounds == 0) {
            inner_.lock();
            return;
        }
        if (inner_.try_lock()) {
            return;
        }
        ++contended_;
        for (u32 r = 0; r < rounds; ++r) {
            for (int p = 0; p < SPINS_PER_ROUND; ++p) {
                Common::CpuPause();
            }
            if (inner_.try_lock()) {
                rounds_used_ += r + 1;
                ++spun_;
                return;
            }
        }
        rounds_used_ += rounds;
        ++blocked_;
        inner_.lock();
    }

    void unlock() {
        inner_.unlock();
    }

    struct Stats {
        u64 contended;
        u64 spun;
        u64 blocked;
        u64 rounds_used;
    };
    static Stats Drain() {
        return Stats{std::exchange(contended_, u64{0}), std::exchange(spun_, u64{0}),
                     std::exchange(blocked_, u64{0}), std::exchange(rounds_used_, u64{0})};
    }

    // Non-zero on the GPU command thread alone. See the counter invariant above.
    static inline thread_local u32 gpu_spin_rounds{0};

private:
    static constexpr int SPINS_PER_ROUND = 16;

    static inline u64 contended_{};
    static inline u64 spun_{};
    static inline u64 blocked_{};
    static inline u64 rounds_used_{};
    LockType inner_;
};

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
    /// DeferRelease is opt-in per CALL SITE, never a global mode: a pending release keeps the
    /// page unreadable, so an undrained one refaults the guest forever. KeepArmed marks
    /// CPU-dirty without touching protection: the caller already wrote the bytes through the
    /// backing alias, so the page keeps its write watcher and the guest still faults on it.
    /// Returns whether any bit changed.
    template <Type type, bool enable, bool DeferRelease = false, bool KeepArmed = false>
    bool ChangeRegionState(u64 dirty_addr, u64 size) noexcept(type == Type::GPU) {
        RENDERER_TRACE;
        const size_t offset = dirty_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }

        if constexpr (type == Type::GPU && enable) {
            // Marks only, and GPU bits are mutated on the GPU command thread alone, so a plain
            // counter suffices: an unchanged value between two points on that thread proves no
            // new GPU write was recorded for this region, which is the guard the offloaded
            // readback path checks before clearing bits it snapshotted earlier.
            ++gpu_write_seq;
        }
        RegionBits& bits = GetRegionBits<type>();
        // A range already in the target state makes the write below an identity - the bits cannot
        // change, so neither can the protection masks derived from them - and skipping it also
        // leaves the sequence count stable for concurrent lock-free readers. The exception is a
        // KeepArmed mark that left a CPU-dirty page still armed: a guest write fault on it must
        // fall through and release it, or the guest refaults on the same page without end.
        if constexpr (enable) {
            if (bits.AllInRange(start_page, end_page) &&
                (type != Type::CPU || KeepArmed || writeable.AllInRange(start_page, end_page))) {
                return false;
            }
        } else {
            if (!bits.AnyInRange(start_page, end_page)) {
                return false;
            }
        }
        WriteScope write_scope{*this};
        if constexpr (enable) {
            bits.SetRange(start_page, end_page);
        } else {
            bits.UnsetRange(start_page, end_page);
        }
        if constexpr (type == Type::CPU) {
            // A page whose release is still pending is unreadable, and dropping
            // its write watcher below would ask for a write-only mapping, which
            // Protect rejects: settle the region pending mask first.
            if constexpr (enable && !KeepArmed) {
                if (read_release_pending_) {
                    u32 pages = 0;
                    ReleaseReadWatchers(pages);
                }
            }
            if constexpr (enable) {
                MarkCpuSummary(start_page, end_page);
            } else {
                RefreshCpuSummary(start_page, end_page);
            }
            if constexpr (!KeepArmed) {
                if constexpr (enable) {
                    ReleaseWriteWatchers(start_page, end_page);
                } else {
                    ArmWriteWatchers();
                }
            }
        } else if (ReadbacksMode(enable ? &mode_reads_mark_ : &mode_reads_unmark_) ==
                   GpuReadbacksMode::Precise) {
            if constexpr (enable) {
                if (defer_read_arm_) {
                    read_arm_pending_ = true;
                } else {
                    u32 pages = 0;
                    ArmReadWatchers(pages);
                }
            } else if (DeferRelease && defer_read_release_) {
                read_release_pending_ = true;
            } else {
                u32 pages = 0;
                ReleaseReadWatchers(pages);
            }
        }
        return true;
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
            WriteScope write_scope{*this};
            bits.UnsetRange(start_page, end_page);
            if constexpr (type == Type::CPU) {
                RefreshCpuSummary(start_page, end_page);
                ArmWriteWatchers();
            } else if (ReadbacksMode() != GpuReadbacksMode::Disabled) {
                // Never deferred: gated on any readbacks mode, and its caller
                // is not the download completion the drain hangs off.
                u32 pages = 0;
                ReleaseReadWatchers(pages);
            }
        }

        for (const auto& [start, end] : mask) {
            func(cpu_addr + start * TRACKER_BYTES_PER_PAGE, (end - start) * TRACKER_BYTES_PER_PAGE);
        }
    }

    /// Widens a page interval outward to [lo_limit, hi_limit) over pages that
    /// are CPU dirty and hold no pending GPU write, stopping at the first that
    /// is not. Caller holds the lock.
    std::pair<size_t, size_t> WidenCpuDirty(size_t start_page, size_t end_page, size_t lo_limit,
                                            size_t hi_limit) const noexcept {
        while (start_page > lo_limit && cpu.Get(start_page - 1) && !gpu.Get(start_page - 1)) {
            --start_page;
        }
        while (end_page < hi_limit && cpu.Get(end_page) && !gpu.Get(end_page)) {
            ++end_page;
        }
        return {start_page, end_page};
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
            if ((Summary() & SummaryMask(start_page, end_page)) == 0) {
                return false;
            }
        }
        return GetRegionBits<type>().AnyInRange(start_page, end_page);
    }

    /// Whole-region CPU-clean test with no range math: the seqlock read of the
    /// summary word alone. Valid only on a region the query covers completely;
    /// a false answer merely hands the region to the full check.
    [[nodiscard]] bool PeekFullRegionClean() const noexcept {
        return (state.load(std::memory_order_acquire) & CLEAN_MASK) == 0;
    }

    template <Type type>
    [[nodiscard]] bool PeekRegionModified(u64 offset, u64 size) noexcept {
        if constexpr (type == Type::CPU) {
            // The range guards must precede SummaryMask (out-of-range pages
            // are shift-count UB); an even sequence with a zero summary mask
            // in the SAME load proves the range clean, else fall through.
            const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
            const size_t end_page =
                Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
            if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
                return false;
            }
            const u64 w = state.load(std::memory_order_acquire);
            if ((w & (SEQ_ONE | SummaryMask(start_page, end_page))) == 0) {
                return false;
            }
        }
        return PeekRegionModifiedSlow<type>(offset, size);
    }

    template <Type type>
    [[nodiscard]] SHAD_NO_INLINE bool PeekRegionModifiedSlow(u64 offset, u64 size) noexcept {
        return SeqPeek([&] { return IsRegionModified<type>(offset, size); });
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
        return SeqPeek([&] { return GetRegionBits<type>().AllInRange(start_page, end_page); });
    }

    /// Arms the read watcher of every GPU-dirty page that still lacks one and
    /// returns the protection calls issued. The arm and release masks are
    /// disjoint, so a release never touches a pending page. readable is cleared
    /// under this lock before the watcher update: the read watcher count is one
    /// bit, and arming an armed page would wrap it.
    u32 ArmReadWatchers(u32& pages) {
        read_arm_pending_ = false;
        RegionBits mask = gpu & readable;
        if (mask.None()) {
            return 0;
        }
        readable &= ~mask;
        for (const auto& [start, end] : mask) {
            pages += static_cast<u32>(end - start);
        }
        return tracker->UpdatePageWatchersForRegion<true, true>(cpu_addr, mask);
    }

    /// Releases the read watcher of every page that is no longer GPU dirty and
    /// returns the protection calls issued. The mask is consumed by the
    /// readable update; batching the releases lets the gap-merge in
    /// UpdatePageWatchersForRegion fuse their mprotects into few calls.
    u32 ReleaseReadWatchers(u32& pages) {
        read_release_pending_ = false;
        RegionBits mask = ~(gpu | readable);
        if (mask.None()) {
            return 0;
        }
        readable |= mask;
        u32 runs = 0;
        for (const auto& [start, end] : mask) {
            pages += static_cast<u32>(end - start);
            ++runs;
        }
        const u32 calls = tracker->UpdatePageWatchersForRegion<false, true>(cpu_addr, mask);
        release_calls_.fetch_add(calls, std::memory_order_relaxed);
        release_pages_.fetch_add(pages, std::memory_order_relaxed);
        release_runs_.fetch_add(runs, std::memory_order_relaxed);
        release_batches_.fetch_add(1, std::memory_order_relaxed);
        return calls;
    }

    /// Readbacks mode latched once before any region exists (see
    /// MemoryTracker::SetModeLatch), so mark/unmark and fault paths read a word
    /// instead of EmulatorSettings. Written on the ctor thread before readers
    /// exist and constant afterwards; relaxed atomics only to publish it safely.
    static inline std::atomic<bool> mode_latched_{false};
    static inline std::atomic<u32> readbacks_mode_{GpuReadbacksMode::Disabled};

    static u32 ReadbacksMode(std::atomic<u64>* counter = nullptr) noexcept {
        if (!mode_latched_.load(std::memory_order_relaxed)) {
            return EmulatorSettings.GetReadbacksMode();
        }
        if (counter != nullptr) {
            counter->fetch_add(1, std::memory_order_relaxed);
        }
        return readbacks_mode_.load(std::memory_order_relaxed);
    }

    /// Census of the reads the latch removes, counted at the read site and
    /// only while latched, so the unlatched arm keeps the live read alone.
    static inline std::atomic<u64> mode_reads_mark_{};
    static inline std::atomic<u64> mode_reads_unmark_{};
    static inline std::atomic<u64> mode_reads_fault_{};

    /// Read-watcher release census. Static because a release happens from any
    /// region; drained once per telemetry window through the tracker.
    static inline std::atomic<u64> release_calls_{};
    static inline std::atomic<u64> release_pages_{};
    static inline std::atomic<u64> release_runs_{};
    static inline std::atomic<u64> release_batches_{};

    /// Scope guard marking a mutation of the tracked bits for readers.
    struct WriteScope {
        explicit WriteScope(RegionManager& m) : mgr{m} {
            mgr.state.fetch_add(SEQ_ONE, std::memory_order_acq_rel);
        }
        ~WriteScope() {
            mgr.state.fetch_add(SEQ_ONE, std::memory_order_release);
        }
        RegionManager& mgr;
    };

    // The CPU dirty summary (low 16 bits) and the write sequence (bits 16 and
    // up) share one word: a single load is a consistent snapshot of both, and
    // an even sequence in it proves no writer was inside its scope then.
    static constexpr u64 SUMMARY_BITS = 0xFFFF;
    static constexpr u64 SEQ_ONE = u64{1} << 16;
    static constexpr u64 CLEAN_MASK = SUMMARY_BITS | SEQ_ONE; // no dirty word, even sequence
    std::atomic<u64> state{SUMMARY_BITS};                     // cpu.Fill() in the ctor => fully set
    [[nodiscard]] u16 Summary() const noexcept {
        return static_cast<u16>(state.load(std::memory_order_relaxed));
    }
    // Counts GPU-bit marks. GPU-command-thread confined; see ChangeRegionState.
    // GPU bits are set only there; an unmap clears them from a guest thread
    // under the lock, which leaves this count untouched.
    u64 gpu_write_seq{0};
    RegionLock lock;
    // Copied from the tracker when the region is handed out.
    bool defer_read_arm_{false};
    // A mark that recorded its bits and left the arm to the next drain. Read
    // and written under the lock on GPU-command-thread paths only.
    bool read_arm_pending_{false};
    // Copied from the tracker when the region is handed out.
    bool defer_read_release_{false};
    // An unmark that cleared its bits and left the release to the next drain.
    bool read_release_pending_{false};

    // Word epochs advance whenever guest bytes in a span may change outside
    // the write watchers' sight: write protection loss, guest protection
    // grants, and direct backing writes. A consumer that pairs an epoch sum
    // with a content record gets a cheap it-cannot-have-changed certificate.
    static constexpr u64 EPOCH_WORD_BITS = 18; // 256KB per word
    static constexpr size_t NUM_EPOCH_WORDS = TRACKER_HIGHER_PAGE_SIZE >> EPOCH_WORD_BITS;

    /// Clamped epoch-word range covering [offset, offset + size) of this region.
    static constexpr std::pair<size_t, size_t> EpochWordRange(u64 offset, u64 size) noexcept {
        return {std::min<u64>(offset >> EPOCH_WORD_BITS, NUM_EPOCH_WORDS - 1),
                std::min<u64>((offset + size - 1) >> EPOCH_WORD_BITS, NUM_EPOCH_WORDS - 1)};
    }

    void BumpWordEpochs(u64 offset, u64 size) noexcept {
        if (size == 0) {
            return;
        }
        const auto [w0, w1] = EpochWordRange(offset, size);
        for (size_t w = w0; w <= w1; ++w) {
            word_epochs[w].fetch_add(1, std::memory_order_release);
        }
    }

    /// Single-region form of the tracker's word-epoch sum: same loads, same
    /// sum, same poison rule. The caller certifies size != 0 and full coverage
    /// by this region, which is what lets the shifts below skip the
    /// NUM_EPOCH_WORDS clamp Bump/Poison apply. Poison is read before the epoch
    /// words on both arms, so a poison-and-bump pair is seen from one side.
    [[nodiscard]] bool EpochSumResolved(u64 offset, u64 size, u64& sum) const noexcept {
        const size_t w0 = offset >> EPOCH_WORD_BITS;
        const size_t w1 = (offset + size - 1) >> EPOCH_WORD_BITS;
        if (w0 != w1) [[unlikely]] {
            return EpochSumResolvedMultiWord(w0, w1, sum);
        }
        const u32 poison = poison_words.load(std::memory_order_acquire);
        sum = word_epochs[w0].load(std::memory_order_acquire);
        return ((poison >> w0) & 1u) == 0;
    }

    [[nodiscard]] SHAD_NO_INLINE bool EpochSumResolvedMultiWord(size_t w0, size_t w1,
                                                                u64& sum) const noexcept {
        const u32 poison = poison_words.load(std::memory_order_acquire);
        u64 s = 0;
        for (size_t w = w0; w <= w1; ++w) {
            s += word_epochs[w].load(std::memory_order_acquire);
        }
        sum = s;
        return ((poison >> w0) & ((2u << (w1 - w0)) - 1u)) == 0;
    }

    void PoisonEpochWords(u64 offset, u64 size) noexcept {
        if (size == 0) {
            return;
        }
        const auto [w0, w1] = EpochWordRange(offset, size);
        poison_words.fetch_or((2u << w1) - (1u << w0), std::memory_order_release);
    }

    // 16-bit summary over the CPU dirty bits: bit k covers exactly one storage
    // word of the bitset. Maintained under the same write scope as the bits, so
    // the lock-free peek's seqlock covers it too; a prefilter only - clean is
    // authoritative (a clear bit proves its word is zero), dirty rescans exactly.
    static constexpr size_t PAGES_PER_SUMMARY_BIT = 64;
    static_assert(NUM_PAGES_PER_REGION / PAGES_PER_SUMMARY_BIT <= 16);

    static u16 SummaryMask(size_t start_page, size_t end_page) noexcept {
        const size_t w0 = start_page / PAGES_PER_SUMMARY_BIT;
        const size_t w1 = (end_page - 1) / PAGES_PER_SUMMARY_BIT;
        return static_cast<u16>(((2u << w1) - (1u << w0)) & 0xFFFFu);
    }

    // Set form for a range whose pages were just set: every summary word the
    // range touches is now non-empty, so the refresh loop can only OR these
    // bits in. Runs inside a WriteScope, on the low 16 bits only.
    void MarkCpuSummary(size_t start_page, size_t end_page) noexcept {
        if (const u16 add = static_cast<u16>(SummaryMask(start_page, end_page) & ~Summary());
            add != 0) {
            state.fetch_or(add, std::memory_order_relaxed);
        }
    }

    // Runs inside a WriteScope; the xor keeps the update atomic against the
    // scope's own increments on the shared word.
    void RefreshCpuSummary(size_t start_page, size_t end_page) noexcept {
        const size_t w0 = start_page / PAGES_PER_SUMMARY_BIT;
        const size_t w1 = (end_page - 1) / PAGES_PER_SUMMARY_BIT;
        const u16 old = Summary();
        u16 summary = old;
        for (size_t w = w0; w <= w1; ++w) {
            const size_t p0 = w * PAGES_PER_SUMMARY_BIT;
            if (cpu.AnyInRange(p0, p0 + PAGES_PER_SUMMARY_BIT)) {
                summary |= static_cast<u16>(1u << w);
            } else {
                summary &= static_cast<u16>(~(1u << w));
            }
        }
        if (const u64 delta = old ^ summary; delta != 0) {
            state.fetch_xor(delta, std::memory_order_relaxed);
        }
    }

    std::array<std::atomic<u64>, NUM_EPOCH_WORDS> word_epochs{};
    std::atomic<u32> poison_words{0};

private:
    // Seqlock read: the bits are read between two reads of a sequence counter
    // writers make odd, so an unchanged even counter proves no writer ran and
    // the answer equals the locked query's. This lock is contended between the
    // GPU thread and the guest fault handler on every bind.
    template <typename F>
    [[nodiscard]] SHAD_FORCE_INLINE bool SeqPeek(F&& read) noexcept {
        for (u32 attempt = 0; attempt < 4; ++attempt) {
            const u64 before = state.load(std::memory_order_acquire);
            if (before & SEQ_ONE) {
                continue; // writer in flight
            }
            const bool result = read();
            std::atomic_thread_fence(std::memory_order_acquire);
            if (state.load(std::memory_order_relaxed) == before) {
                return result;
            }
        }
        std::scoped_lock lk{lock};
        return read();
    }

    // Takes only pages that lost their CPU bit and are still writable.
    // Directional, not symmetric: a page a CPU mark deliberately left armed
    // (cpu 1, writeable 0) must be skipped here or it gains a second watcher
    // no release ever returns.
    void ArmWriteWatchers() {
        RENDERER_TRACE;
        RegionBits mask = writeable & ~cpu;
        if (mask.None()) {
            return;
        }
        // == writeable & ~mask, without materialising a second bitset.
        writeable &= cpu;
        tracker->UpdatePageWatchersForRegion<true, false>(cpu_addr, mask);
    }

    // Takes only pages that gained the CPU bit and are still armed, limited to
    // [lo, hi) so a fault elsewhere in the region does not un-arm the pages a
    // KeepArmed mark left armed.
    void ReleaseWriteWatchers(size_t lo, size_t hi) {
        RENDERER_TRACE;
        RegionBits mask(cpu & ~writeable, lo, hi);
        if (mask.None()) {
            return;
        }
        writeable |= mask;
        tracker->UpdatePageWatchersForRegion<false, false>(cpu_addr, mask);
    }

    PageManager* tracker;
    VAddr cpu_addr = 0;
    RegionBits cpu;
    RegionBits gpu;
    RegionBits writeable;
    RegionBits readable;
};

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
// The per-uaddr futex parse that proves or disproves tracker_lock_spin_rounds
// identifies these locks by the 0x2D8 = 728 byte stride between them. Keep the
// size pinned so the next trace still resolves.
static_assert(sizeof(RegionManager) == 728);
#endif

} // namespace VideoCore
