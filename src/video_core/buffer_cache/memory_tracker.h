// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <deque>
#include <utility>

#include <mutex>
#include <type_traits>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/debug.h"
#include "common/types.h"
#include "core/emulator_settings.h"
#include "video_core/buffer_cache/region_manager.h"

namespace VideoCore {

class MemoryTracker {
public:
    static constexpr size_t MAX_CPU_PAGE_BITS = 40;
    static constexpr size_t NUM_HIGH_PAGES = 1ULL << (MAX_CPU_PAGE_BITS - TRACKER_HIGHER_PAGE_BITS);
    static constexpr size_t MANAGER_POOL_SIZE = 32;
    // Widest range Sum256ForRange can key: 64 epoch words.
    static constexpr u64 MAX_EPOCH_SUM_SPAN = u64{64} << RegionManager::EPOCH_WORD_BITS;

public:
    explicit MemoryTracker(PageManager& tracker_) : tracker{&tracker_} {}
    ~MemoryTracker() = default;

    // Upload-walk peek baseline; GPU-command-thread confined like the walk.
    u64 peek_fastpath_calls{};
    u64 peek_fastpath_dirty{};
    u64 multi_walks{};
    u64 multi_regions{};
    u64 multi_clean_regions{};

    /// Returns true if a region has been modified from the CPU
    bool IsRegionCpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<true>(
            query_cpu_addr, query_size, [](RegionManager* manager, u64 offset, size_t size) {
                return manager->template PeekRegionModified<Type::CPU>(offset, size);
            });
    }

    /// Returns true if a region has been modified from the GPU
    bool IsRegionGpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<false>(
            query_cpu_addr, query_size, [](RegionManager* manager, u64 offset, size_t size) {
                return manager->template PeekRegionModified<Type::GPU>(offset, size);
            });
    }

    /// Mark region as CPU modified, notifying the device_tracker about this change
    void MarkRegionAsCpuModified(VAddr dirty_cpu_addr, u64 query_size) {
        IteratePages<false>(dirty_cpu_addr, query_size,
                            [](RegionManager* manager, u64 offset, size_t size) {
                                std::scoped_lock lk{manager->lock};
                                manager->template ChangeRegionState<Type::CPU, true>(
                                    manager->GetCpuAddr() + offset, size);
                            });
    }

    /// Unmark region as modified from the host GPU
    void UnmarkRegionAsGpuModified(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<false>(dirty_cpu_addr, query_size,
                            [](RegionManager* manager, u64 offset, size_t size) {
                                std::scoped_lock lk{manager->lock};
                                manager->template ChangeRegionState<Type::GPU, false>(
                                    manager->GetCpuAddr() + offset, size);
                            });
    }

    /// One region's identity and its gpu_write_seq value at snapshot time.
    /// GPU-command-thread confined, like the counter it captures.
    struct GpuSeqSnapshot {
        RegionManager* manager;
        u64 seq;
    };
    using GpuSeqSnapshots = boost::container::small_vector<GpuSeqSnapshot, 4>;

    /**
     * Captures each overlapped region's GPU write sequence. Call on the GPU
     * command thread at the moment download copies are recorded; pass the
     * result to GpuWriteSeqMatches when deciding whether the copied data may
     * be written back.
     */
    void SnapshotGpuWriteSeq(VAddr cpu_addr, u64 size, GpuSeqSnapshots& out) {
        IteratePages<false>(cpu_addr, size, [&out](RegionManager* manager, u64, size_t) {
            out.push_back({manager, manager->gpu_write_seq});
        });
    }

    /**
     * True when every region overlapping the range still carries the GPU write
     * sequence captured in the snapshot - that is, no new GPU write to those
     * regions has been recorded since. A changed sequence means downloaded
     * bytes for the range may be stale and must not be written back or have
     * their bits cleared. GPU command thread only, so the comparison cannot
     * race the writers it guards against.
     */
    bool GpuWriteSeqMatches(VAddr cpu_addr, u64 size, const GpuSeqSnapshots& snap) {
        bool matches = true;
        IteratePages<false>(cpu_addr, size, [&](RegionManager* manager, u64, size_t) {
            const auto it = std::ranges::find(snap, manager, &GpuSeqSnapshot::manager);
            if (it == snap.end() || it->seq != manager->gpu_write_seq) {
                matches = false;
            }
        });
        return matches;
    }

    /// Advances the word epochs of every existing region overlapping the
    /// range. Missing regions have no consumers and are skipped.
    void BumpEpochsForRange(VAddr cpu_addr, u64 size, u8 cause) noexcept {
        IteratePages<false>(cpu_addr, size,
                            [cause](RegionManager* manager, u64 offset, size_t range_size) {
                                manager->BumpWordEpochs(offset, range_size, cause);
                            });
    }

    /// Poisons the word epochs covering the range; a poisoned word never
    /// certifies content stability.
    void PoisonEpochsForRange(VAddr cpu_addr, u64 size) noexcept {
        IteratePages<false>(cpu_addr, size,
                            [](RegionManager* manager, u64 offset, size_t range_size) {
                                manager->PoisonEpochWords(offset, range_size);
                            });
    }

    /// Like IsRegionCpuModified but never creates missing regions; uncovered
    /// spans report modified, matching a fresh region's all-dirty default.
    bool PeekRegionCpuModifiedNoCreate(VAddr query_cpu_addr, u64 query_size) noexcept {
        u64 covered = 0;
        const bool dirty = IteratePages<false>(
            query_cpu_addr, query_size,
            [&covered](RegionManager* manager, u64 offset, size_t size) {
                covered += size;
                return manager->template PeekRegionModified<Type::CPU>(offset, size);
            });
        return dirty || covered != query_size;
    }

    struct EpochSum256 {
        u64 sum;
        bool ok;
    };

    /// Word-epoch sum alone, for consumers that key memos on it. ok is false
    /// when part of the range has no region, when any covered word is
    /// poisoned, or when the span exceeds 64 words; callers then fall back to
    /// their coarse generation key.
    EpochSum256 Sum256ForRange(VAddr cpu_addr, u64 size) noexcept {
        EpochSum256 out{0, true};
        if (size == 0 || size > MAX_EPOCH_SUM_SPAN) {
            out.ok = false;
            return out;
        }
        u64 covered = 0;
        IteratePages<false>(
            cpu_addr, size,
            [&out, &covered](RegionManager* manager, u64 offset, size_t range_size) {
                covered += range_size;
                const size_t w0 = std::min<u64>(offset >> RegionManager::EPOCH_WORD_BITS,
                                                RegionManager::NUM_EPOCH_WORDS - 1);
                const size_t w1 =
                    std::min<u64>((offset + range_size - 1) >> RegionManager::EPOCH_WORD_BITS,
                                  RegionManager::NUM_EPOCH_WORDS - 1);
                const u32 poison = manager->poison_words.load(std::memory_order_acquire);
                for (size_t w = w0; w <= w1; ++w) {
                    out.sum += manager->word_epochs[w].load(std::memory_order_acquire);
                    if ((poison >> w) & 1u) {
                        out.ok = false;
                    }
                }
            });
        out.ok = out.ok && covered == size;
        return out;
    }

    /// Twin of Sum256ForRange that also names the region when one covers the
    /// whole range; the resolved memo probe reads that region directly.
    EpochSum256 Sum256ForRangeResolved(VAddr cpu_addr, u64 size, RegionManager*& region) noexcept {
        EpochSum256 out{0, true};
        region = nullptr;
        if (size == 0 || size > MAX_EPOCH_SUM_SPAN) {
            out.ok = false;
            return out;
        }
        u64 covered = 0;
        u32 regions = 0;
        IteratePages<false>(
            cpu_addr, size, [&](RegionManager* manager, u64 offset, size_t range_size) {
                covered += range_size;
                ++regions;
                region = manager;
                const size_t w0 = std::min<u64>(offset >> RegionManager::EPOCH_WORD_BITS,
                                                RegionManager::NUM_EPOCH_WORDS - 1);
                const size_t w1 =
                    std::min<u64>((offset + range_size - 1) >> RegionManager::EPOCH_WORD_BITS,
                                  RegionManager::NUM_EPOCH_WORDS - 1);
                const u32 poison = manager->poison_words.load(std::memory_order_acquire);
                for (size_t w = w0; w <= w1; ++w) {
                    out.sum += manager->word_epochs[w].load(std::memory_order_acquire);
                    if ((poison >> w) & 1u) {
                        out.ok = false;
                    }
                }
            });
        out.ok = out.ok && covered == size;
        if (regions != 1 || !out.ok) {
            region = nullptr;
        }
        return out;
    }

    struct EpochSums {
        u64 sum256;
        u64 sum64;
        bool poisoned;
        bool ok;
    };

    /// Sums the word and subword epochs covering the range. ok is false when
    /// part of the range has no region yet, since absent epochs prove nothing.
    EpochSums SumEpochsForRange(VAddr cpu_addr, u64 size) noexcept {
        EpochSums out{0, 0, false, true};
        u64 covered = 0;
        IteratePages<false>(
            cpu_addr, size,
            [&out, &covered](RegionManager* manager, u64 offset, size_t range_size) {
                covered += range_size;
                const size_t w0 = std::min<u64>(offset >> RegionManager::EPOCH_WORD_BITS,
                                                RegionManager::NUM_EPOCH_WORDS - 1);
                const size_t w1 =
                    std::min<u64>((offset + range_size - 1) >> RegionManager::EPOCH_WORD_BITS,
                                  RegionManager::NUM_EPOCH_WORDS - 1);
                const u32 poison = manager->poison_words.load(std::memory_order_acquire);
                for (size_t w = w0; w <= w1; ++w) {
                    out.sum256 += manager->word_epochs[w].load(std::memory_order_acquire);
                    out.poisoned |= (poison >> w) & 1u;
                }
                const size_t s0 = std::min<u64>(offset >> RegionManager::EPOCH_SUB_BITS,
                                                RegionManager::NUM_EPOCH_SUBS - 1);
                const size_t s1 =
                    std::min<u64>((offset + range_size - 1) >> RegionManager::EPOCH_SUB_BITS,
                                  RegionManager::NUM_EPOCH_SUBS - 1);
                for (size_t sub = s0; sub <= s1; ++sub) {
                    out.sum64 += manager->sub_epochs[sub].load(std::memory_order_acquire);
                }
            });
        out.ok = covered == size;
        return out;
    }

    /// Removes all protection from a page and ensures GPU data has been flushed if requested
    void InvalidateRegion(VAddr cpu_addr, u64 size, auto&& on_flush) noexcept {
        IteratePages<false>(
            cpu_addr, size, [&on_flush](RegionManager* manager, u64 offset, size_t size) {
                const bool should_flush = [&] {
                    // Perform both the GPU modification check and CPU state change with the lock
                    // in case we are racing with GPU thread trying to mark the page as GPU
                    // modified. If we need to flush the flush function is going to perform CPU
                    // state change.
                    std::scoped_lock lk{manager->lock};
                    if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled &&
                        manager->template IsRegionModified<Type::GPU>(offset, size)) {
                        return true;
                    }
                    manager->template ChangeRegionState<Type::CPU, true>(
                        manager->GetCpuAddr() + offset, size);
                    return false;
                }();
                if (should_flush) {
                    on_flush();
                }
            });
    }

    /// InvalidateRegion with fault widening: the whole widened chunk goes
    /// CPU-dirty when it holds no GPU-modified page - extra dirty pages only
    /// re-upload bytes the guest already owns, and pages never uploaded are
    /// dirty by default so their protection is never touched. Any GPU bit in
    /// the chunk falls back to page-exact semantics for the original range
    /// (identical flush behavior, no spurious readbacks, and never a
    /// CPU-dirty mark over read-tracked pages).
    void InvalidateRegionWidened(VAddr orig_addr, u64 orig_size, VAddr wide_addr, u64 wide_size,
                                 auto&& on_flush) noexcept {
        IteratePages<false>(
            wide_addr, wide_size, [&](RegionManager* manager, u64 offset, size_t size) {
                const VAddr chunk_addr = manager->GetCpuAddr() + offset;
                bool flush = false;
                {
                    std::scoped_lock lk{manager->lock};
                    const bool readbacks =
                        EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled;
                    if (!readbacks ||
                        !manager->template IsRegionModified<Type::GPU>(offset, size)) {
                        manager->template ChangeRegionState<Type::CPU, true>(chunk_addr, size);
                        return;
                    }
                    const VAddr lo = std::max(chunk_addr, orig_addr);
                    const VAddr hi = std::min<VAddr>(chunk_addr + size, orig_addr + orig_size);
                    if (lo >= hi) {
                        return; // pure widening over GPU data: leave untouched
                    }
                    if (manager->template IsRegionModified<Type::GPU>(lo - manager->GetCpuAddr(),
                                                                      hi - lo)) {
                        flush = true;
                    } else {
                        manager->template ChangeRegionState<Type::CPU, true>(lo, hi - lo);
                    }
                }
                if (flush) {
                    on_flush();
                }
            });
    }

    /// Call 'func' for each CPU modified range and unmark those pages as CPU modified
    /// Returns whether the written marking set any GPU-clean page.
    bool ForEachUploadRange(VAddr query_cpu_range, u64 query_size, bool is_written, auto&& func,
                            auto&& on_upload) {
        // Nearly every bind is a few hundred bytes and lands inside a single
        // 4MB region. Resolving the manager once up front runs both passes on
        // it directly, without the second memo probe and the per-region
        // index and skip-set bookkeeping the generic walk below needs.
        const std::size_t first_page = query_cpu_range >> TRACKER_HIGHER_PAGE_BITS;
        if (query_size != 0 && first_page == ((query_cpu_range + query_size - 1) >>
                                              TRACKER_HIGHER_PAGE_BITS)) [[likely]] {
            RENDERER_TRACE;
            RegionManager* manager = LookupRegion(first_page);
            if (manager == nullptr) [[unlikely]] {
                manager = CreateRegion(first_page);
            }
            const u64 offset = query_cpu_range & TRACKER_HIGHER_PAGE_MASK;
            const bool nothing_to_upload =
                !manager->template PeekRegionModified<Type::CPU>(offset, query_size);
            ++peek_fastpath_calls;
            peek_fastpath_dirty += nothing_to_upload ? 0 : 1;
            const bool skippable = nothing_to_upload &&
                                   (!is_written || manager->template PeekRegionFullySet<Type::GPU>(
                                                       offset, query_size));
            if (skippable) {
                if (is_written) {
                    // The bits stay as they are, but this is still a new GPU
                    // write to the region: the write sequence must advance or
                    // a snapshot taken before this bind could not tell that
                    // its downloaded bytes are now stale.
                    ++manager->gpu_write_seq;
                }
                on_upload();
                return false;
            }
            manager->lock.lock();
            manager->template ForEachModifiedRange<Type::CPU, true>(manager->GetCpuAddr() + offset,
                                                                    query_size, func);
            if (!is_written) {
                manager->lock.unlock();
                on_upload();
                return false;
            }
            // A written bind holds the lock from the upload walk until the
            // GPU marking below, so the marking observes the bits it covers.
            on_upload();
            const bool changed = manager->template ChangeRegionState<Type::GPU, true>(
                manager->GetCpuAddr() + offset, query_size);
            manager->lock.unlock();
            return changed;
        }
        // A written bind holds each region's lock from the upload walk until
        // the GPU marking below, so a region skipped in the first pass must be
        // skipped in the second. The skip set is recorded rather than
        // recomputed: without the lock held the bits can change in between.
        bool changed = false;
        u64 skipped = 0;
        u32 index = 0;
        {
            RENDERER_TRACE;
            // Direct walk, shaped like the fast path above: only the first
            // region is a scattered lookup, the rest step top_tier linearly,
            // and the next region's header line is requested one region early
            // so its first-touch miss overlaps the current region's body.
            std::size_t remaining_size = query_size;
            std::size_t page_index = first_page;
            u64 page_offset = query_cpu_range & TRACKER_HIGHER_PAGE_MASK;
            bool scattered = true;
            ++multi_walks;
            u64 clean_regions = 0;
            u64 walked_regions = 0;
            while (remaining_size > 0) {
                if (!is_written && page_offset == 0) {
                    // Register-only scan over the whole regions ahead: the
                    // general body below costs ~44 instructions with ten
                    // stack accesses per region, and ~93% of a read-only
                    // multi-region walk's regions are clean middle regions.
                    // Trip count and cursor stay in locals and fold into the
                    // walk state once, so the loop carries nothing through
                    // memory. A null slot or a dirty region hands the walk
                    // back to the general body at that region.
                    const std::size_t full = remaining_size >> TRACKER_HIGHER_PAGE_BITS;
                    std::size_t clean = 0;
                    while (clean < full) {
                        RegionManager* const ahead = top_tier[page_index + clean];
                        if (ahead == nullptr || !ahead->PeekFullRegionClean()) {
                            break;
                        }
                        ++clean;
                    }
                    if (clean != 0) {
                        page_index += clean;
                        remaining_size -= clean << TRACKER_HIGHER_PAGE_BITS;
                        clean_regions += clean;
                        scattered = false;
                        continue;
                    }
                }
                ++walked_regions;
                const std::size_t copy_amount{
                    std::min<std::size_t>(TRACKER_HIGHER_PAGE_SIZE - page_offset, remaining_size)};
                RegionManager* manager =
                    scattered ? LookupRegion(page_index) : top_tier[page_index];
                scattered = false;
                if (manager == nullptr) {
                    manager = CreateRegion(page_index);
                }
                if (remaining_size > copy_amount) {
                    if (RegionManager* next = top_tier[page_index + 1]) {
                        __builtin_prefetch(next, 0, 3);
                    }
                }
                const u64 offset = page_offset;
                const std::size_t size = copy_amount;
                page_index++;
                page_offset = 0;
                remaining_size -= copy_amount;
                // Read-only binds almost never have anything to upload, and
                // proving it under the lock is the hottest contended site in
                // the frame.
                const bool nothing_to_upload =
                    !manager->template PeekRegionModified<Type::CPU>(offset, size);
                if (!is_written) {
                    // The counter and the skip set it indexes are consumed
                    // only by written binds in the second pass.
                    if (nothing_to_upload) {
                        continue;
                    }
                    manager->lock.lock();
                    manager->template ForEachModifiedRange<Type::CPU, true>(
                        manager->GetCpuAddr() + offset, size, func);
                    manager->lock.unlock();
                    continue;
                }
                // Written binds can skip too when there is nothing to upload
                // and the range is already marked, which also avoids
                // re-applying its protection.
                const u32 i = index++;
                if (nothing_to_upload && i < 64 &&
                    manager->template PeekRegionFullySet<Type::GPU>(offset, size)) {
                    skipped |= u64{1} << i;
                    // The bits stay as they are, but this is still a new GPU
                    // write to the region: the write sequence must advance or
                    // a snapshot taken before this bind could not tell that
                    // its downloaded bytes are now stale. GPU-command-thread
                    // confined, like the counter.
                    ++manager->gpu_write_seq;
                    continue;
                }
                manager->lock.lock();
                manager->template ForEachModifiedRange<Type::CPU, true>(
                    manager->GetCpuAddr() + offset, size, func);
            }
            multi_regions += walked_regions + clean_regions;
            multi_clean_regions += clean_regions;
        }
        on_upload();
        if (!is_written) {
            return false;
        }
        {
            // Second pass mirrors the first walk's region sequence exactly; a
            // region skipped there was never locked here. Regions all exist
            // by now (pass one created them), so a null slot is skipped the
            // way the generic no-create walk skipped it.
            u32 unlock_index = 0;
            std::size_t remaining_size = query_size;
            std::size_t page_index = first_page;
            u64 page_offset = query_cpu_range & TRACKER_HIGHER_PAGE_MASK;
            while (remaining_size > 0) {
                const std::size_t copy_amount{
                    std::min<std::size_t>(TRACKER_HIGHER_PAGE_SIZE - page_offset, remaining_size)};
                RegionManager* const manager = top_tier[page_index];
                const u64 offset = page_offset;
                const std::size_t size = copy_amount;
                page_index++;
                page_offset = 0;
                remaining_size -= copy_amount;
                if (manager == nullptr) {
                    continue;
                }
                const u32 i = unlock_index++;
                if (i < 64 && (skipped & (u64{1} << i)) != 0) {
                    continue; // never locked in the first pass
                }
                changed |= manager->template ChangeRegionState<Type::GPU, true>(
                    manager->GetCpuAddr() + offset, size);
                manager->lock.unlock();
            }
        }
        return changed;
    }

    /// Call 'func' for each GPU modified range and unmark those pages as GPU modified
    template <bool clear>
    void ForEachDownloadRange(VAddr query_cpu_range, u64 query_size, auto&& func) {
        IteratePages<false>(query_cpu_range, query_size,
                            [&func](RegionManager* manager, u64 offset, size_t size) {
                                std::scoped_lock lk{manager->lock};
                                manager->template ForEachModifiedRange<Type::GPU, clear>(
                                    manager->GetCpuAddr() + offset, size, func);
                            });
    }

private:
    /**
     * Resolve a region index to its manager.
     *
     * top_tier spans the whole 40 bit guest address space at 4MB granularity,
     * so it is a 2MB pointer array - larger than the L2 of the handheld parts
     * this matters on. It is also sparse, so consecutive lookups for unrelated
     * buffers land on unrelated lines and the load stalls; profiling put ~60%
     * of SynchronizeBuffer on exactly that load. The live region set is tiny by
     * comparison, so a small direct mapped memo of resolved indices keeps the
     * working set in L1.
     *
     * This is exactly equivalent to indexing top_tier: a slot only ever goes
     * from null to a manager and is never cleared or reassigned, so a resolved
     * mapping stays true for the life of the process. Only non-null results are
     * recorded, since a null means "not created yet" and can still change.
     *
     * Thread local because the tracker is also driven from the guest fault
     * path; a shared table could tear a key against a neighbouring value and
     * hand back the wrong manager.
     */
    static constexpr std::size_t NUM_LOOKUP_SLOTS = 128; // power of two
    // Keys are stored biased by one so that a zeroed table reads as empty.
    // That keeps the memo constant initialized, which lets the thread local
    // be reached directly instead of through an initialization guard;
    // constinit turns a regression of that property into a compile error.
    // A slot carries its key and value together, 16 byte aligned, so a
    // probe touches exactly one cache line.
    struct alignas(16) LookupSlot {
        std::size_t key;
        RegionManager* val;
    };
    static_assert(sizeof(LookupSlot) == 16 && alignof(LookupSlot) == 16,
                  "a slot must fit one cache line");
    struct LookupMemo {
        const MemoryTracker* owner;
        std::array<LookupSlot, NUM_LOOKUP_SLOTS> slots;
    };

    // Outlined: the ~420-byte zeroing body was emitted in line at every
    // inlined lookup while being reachable only when a second tracker
    // instance aliases the thread's memo.
    SHAD_NO_INLINE static void ResetLookupMemo(LookupMemo& memo,
                                               const MemoryTracker* owner) noexcept {
        memo.owner = owner;
        for (LookupSlot& reset_slot : memo.slots) {
            reset_slot.key = 0;
        }
    }

    [[nodiscard]] RegionManager* LookupRegion(std::size_t page_index) noexcept {
        static thread_local constinit LookupMemo memo{};
        if (memo.owner != this) [[unlikely]] {
            ResetLookupMemo(memo, this);
        }
        const std::size_t key = page_index + 1;
        LookupSlot& slot = memo.slots[page_index & (NUM_LOOKUP_SLOTS - 1)];
        if (slot.key == key) {
            return slot.val;
        }
        RegionManager* const manager = top_tier[page_index];
        if (manager != nullptr) {
            slot.key = key;
            slot.val = manager;
        }
        return manager;
    }

    /**
     * @brief IteratePages Iterates L2 word manager page table.
     * @param cpu_address Start byte cpu address
     * @param size Size in bytes of the region of iterate.
     * @param func Callback for each word manager.
     * @return
     */
    template <bool create_region_on_fail, typename Func>
    bool IteratePages(VAddr cpu_address, size_t size, Func&& func) {
        RENDERER_TRACE;
        using FuncReturn = typename std::invoke_result<Func, RegionManager*, u64, size_t>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        std::size_t remaining_size{size};
        std::size_t page_index{cpu_address >> TRACKER_HIGHER_PAGE_BITS};
        u64 page_offset{cpu_address & TRACKER_HIGHER_PAGE_MASK};
        // Only a walk's first region is a scattered lookup; the rest are the
        // next entries of top_tier, a unit stride the prefetcher covers.
        // Indexing directly returns exactly what the memo would, since a slot
        // only ever goes from null to a manager and is never cleared or
        // reassigned. Memoising the rest would evict the entries the
        // single-region callers hit on and still load top_tier on every miss.
        bool scattered = true;
        while (remaining_size > 0) {
            const std::size_t copy_amount{
                std::min<std::size_t>(TRACKER_HIGHER_PAGE_SIZE - page_offset, remaining_size)};
            auto* manager{scattered ? LookupRegion(page_index) : top_tier[page_index]};
            scattered = false;
            if (manager) {
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            } else if constexpr (create_region_on_fail) {
                manager = CreateRegion(page_index);
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        return false;
    }

    RegionManager* CreateRegion(std::size_t page_index) {
        const VAddr base_cpu_addr = page_index << TRACKER_HIGHER_PAGE_BITS;
        if (free_managers.empty()) {
            manager_pool.emplace_back();
            auto& last_pool = manager_pool.back();
            for (size_t i = 0; i < MANAGER_POOL_SIZE; i++) {
                std::construct_at(&last_pool[i], tracker, 0);
                free_managers.push_back(&last_pool[i]);
            }
        }
        // Each manager tracks a 4_MB virtual address space.
        auto* new_manager = free_managers.back();
        new_manager->SetCpuAddress(base_cpu_addr);
        free_managers.pop_back();
        top_tier[page_index] = new_manager;
        // Returned directly: re-probing the lookup memo twenty instructions
        // after it just missed only re-derives this pointer. The memo entry
        // for the fresh page fills on the page's next scattered lookup.
        return new_manager;
    }

    PageManager* tracker;
    std::deque<std::array<RegionManager, MANAGER_POOL_SIZE>> manager_pool;
    std::vector<RegionManager*> free_managers;
    std::array<RegionManager*, NUM_HIGH_PAGES> top_tier{};
};

} // namespace VideoCore
