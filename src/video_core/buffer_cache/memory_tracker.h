// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <deque>
#include <mutex>
#include <type_traits>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/debug.h"
#include "common/scope_exit.h"
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

    /// For GPU-command-thread callers that open a protect-carry scope around a
    /// loop of their own; see PageManager::BeginProtectCarry.
    [[nodiscard]] const PageManager& GetPageManager() const noexcept {
        return *tracker;
    }

    /// Latched once before any region exists; new regions inherit it.
    void SetDeferReadArm(bool value) {
        defer_read_arm_ = value;
    }
    void SetDeferReadRelease(bool value) {
        defer_read_release_ = value;
    }

    /// Latched once before any region exists, on the thread that builds the
    /// buffer cache: the tracker's readbacks-mode reads stop taking the global
    /// settings mutex. A mid-run mode change then no longer reaches the
    /// tracker, matching the cache's own latched readback state.
    void SetModeLatch(bool on) {
        RegionManager::readbacks_mode_.store(EmulatorSettings.GetReadbacksMode(),
                                             std::memory_order_relaxed);
        RegionManager::mode_latched_.store(on, std::memory_order_relaxed);
    }

    struct ReadReleaseDrain {
        u32 regions;
        u32 pages;
        u32 calls;
    };

    [[nodiscard]] bool HasPendingReadReleases() const noexcept {
        return !pending_read_releases_.empty();
    }

    /// Releases the read watchers every unmark since the last drain left
    /// pending, coalescing a download's islands into one masked update per
    /// region. GPU command thread only, same constraint as the arm drain.
    /// carry: the caller certifies it is the GPU command thread (see
    /// PageManager::BeginProtectCarry); every other caller must pass false.
    ReadReleaseDrain ReleasePendingReadWatchers(bool carry) {
        return DrainPendingWatchers<&RegionManager::ReleaseReadWatchers>(pending_read_releases_,
                                                                         carry);
    }

    struct ReadReleaseCensus {
        u64 calls;
        u64 pages;
        u64 runs;
        u64 batches;
    };
    /// The syscall count of the release path in either mode: the number this
    /// whole mechanism exists to collapse.
    static ReadReleaseCensus DrainReadReleaseCensus() {
        return ReadReleaseCensus{
            RegionManager::release_calls_.exchange(0, std::memory_order_relaxed),
            RegionManager::release_pages_.exchange(0, std::memory_order_relaxed),
            RegionManager::release_runs_.exchange(0, std::memory_order_relaxed),
            RegionManager::release_batches_.exchange(0, std::memory_order_relaxed),
        };
    }

    using ReadArmDrain = ReadReleaseDrain;

    /// Arms the read watchers every mark since the last drain left pending.
    /// GPU command thread only. A walk that holds region locks across its
    /// upload defers the drain to the next site rather than deadlocking.
    /// carry: the caller certifies it is the GPU command thread (see
    /// PageManager::BeginProtectCarry); every other caller must pass false.
    ReadArmDrain ArmPendingReadWatchers(bool carry) {
        return DrainPendingWatchers<&RegionManager::ArmReadWatchers>(pending_read_arms_, carry);
    }

    // Upload-walk peek baseline; GPU-command-thread confined like the walk.
    u64 peek_fastpath_calls{};
    u64 peek_fastpath_dirty{};
    u64 multi_walks{};
    u64 multi_regions{};
    u64 multi_clean_regions{};
    // Upload arm chunking census: clearing walks on the single-region path,
    // the walks the chunk widened and the extra pages those uploaded.
    u64 arm_chunk_walks{};
    u64 arm_chunk_widened{};
    u64 arm_chunk_pages{};
    // Chunk granule in pages, a power of two; 0 leaves uploads unwidened.
    const u32 arm_chunk_pages_per{ArmChunkPages()};

    static u32 ArmChunkPages() noexcept {
        const u32 bytes = std::min<u32>(EmulatorSettings.GetUploadArmChunkBytes(), 65536u);
        return bytes < 2 * TRACKER_BYTES_PER_PAGE ? 0u : std::bit_floor(bytes) >> TRACKER_PAGE_BITS;
    }

    struct FastPathDrain {
        u64 sum_fast;
        u64 sum_walk;
        u64 gpu_fast;
        u64 gpu_walk;
        u64 single_null;
        u64 foreign; // of the two above, the ones the fault path contributed
        u64 peek_word;
        u64 peek_split;
        u64 fused;
    };
    FastPathDrain DrainFastPathStats() {
        const u64 foreign_fast = foreign_stats_.gpu_fast.exchange(0, std::memory_order_relaxed);
        const u64 foreign_walk = foreign_stats_.gpu_walk.exchange(0, std::memory_order_relaxed);
        const FastPathDrain out{fast_stats_.sum_fast,
                                fast_stats_.sum_walk,
                                fast_stats_.gpu_fast + foreign_fast,
                                fast_stats_.gpu_walk + foreign_walk,
                                fast_stats_.single_null,
                                foreign_fast + foreign_walk,
                                fast_stats_.peek_word,
                                fast_stats_.gpu_fast - fast_stats_.peek_word,
                                fast_stats_.fused};
        fast_stats_ = {};
        return out;
    }

    /// Returns true if a region has been modified from the CPU
    bool IsRegionCpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<true>(
            query_cpu_addr, query_size, [](RegionManager* manager, u64 offset, size_t size) {
                return manager->template PeekRegionModified<Type::CPU>(offset, size);
            });
    }

    /// Returns true if a region has been modified from the GPU.
    /// Foreign = the readback fault path, which probes from whichever thread
    /// took the fault; it counts on its own line so the bind path, which is
    /// GPU-command-thread confined, needs no lock prefix per probe.
    template <bool Foreign = false>
    bool IsRegionGpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        RegionManager* region;
        if (TrySingleRegion(query_cpu_addr, query_size, region)) [[likely]] {
            RENDERER_TRACE;
            const u64 region_offset = query_cpu_addr & TRACKER_HIGHER_PAGE_MASK;
            if constexpr (Foreign) {
                foreign_stats_.gpu_fast.fetch_add(1, std::memory_order_relaxed);
            } else {
                ++fast_stats_.gpu_fast;
                CountPeekSpan(region_offset, query_size);
            }
            if (region == nullptr) {
                return false;
            }
            return region->template PeekRegionModified<Type::GPU>(region_offset, query_size);
        }
        if constexpr (Foreign) {
            foreign_stats_.gpu_walk.fetch_add(1, std::memory_order_relaxed);
        } else {
            ++fast_stats_.gpu_walk;
        }
        return IteratePages<false>(
            query_cpu_addr, query_size, [](RegionManager* manager, u64 offset, size_t size) {
                return manager->template PeekRegionModified<Type::GPU>(offset, size);
            });
    }

    /// The GPU probe for a range whose single covering region the caller
    /// resolved in this same call, from TrySingleRegion or a memo entry
    /// certified for exactly [addr, addr + size); size != 0. GPU command
    /// thread only: the foreign fault-path probes never reach it.
    [[nodiscard]] bool IsRegionGpuModifiedIn(RegionManager* region, VAddr addr, u64 size) noexcept {
        RENDERER_TRACE;
        ++fast_stats_.gpu_fast;
        ++fast_stats_.fused;
        const u64 region_offset = addr & TRACKER_HIGHER_PAGE_MASK;
        CountPeekSpan(region_offset, size);
        return region->template PeekRegionModified<Type::GPU>(region_offset, size);
    }

    /// Mark region as CPU modified while its write watchers stay in place: the
    /// caller is about to put the bytes there itself, through the backing
    /// alias, so no release and no re-arm is needed and the guest keeps
    /// faulting on the page. Returns false when the range holds a GPU-modified
    /// page or a pending read release - the caller must then store normally
    /// and take the ordinary fault, which runs the readback path unchanged.
    [[nodiscard]] bool MarkRegionAsCpuModifiedKeepArmed(VAddr dirty_cpu_addr, u64 query_size) {
        bool ok = true;
        IteratePages<false>(
            dirty_cpu_addr, query_size, [&ok](RegionManager* manager, u64 offset, size_t size) {
                std::scoped_lock lk{manager->lock};
                if (manager->read_release_pending_ ||
                    (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled &&
                     manager->template IsRegionModified<Type::GPU>(offset, size))) {
                    ok = false;
                    return;
                }
                manager->template ChangeRegionState<Type::CPU, true, false, true>(
                    manager->GetCpuAddr() + offset, size);
            });
        return ok;
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

    /// As above, but under deferred_read_release the read-watcher release is
    /// left to ReleasePendingReadWatchers. ONLY for callers that drain before
    /// returning: an undrained pending release leaves the page unreadable, and
    /// the guest then refaults on it without end.
    void UnmarkRegionAsGpuModifiedDeferred(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<false>(dirty_cpu_addr, query_size,
                            [this](RegionManager* manager, u64 offset, size_t size) {
                                std::scoped_lock lk{manager->lock};
                                const bool was_pending = manager->read_release_pending_;
                                manager->template ChangeRegionState<Type::GPU, false, true>(
                                    manager->GetCpuAddr() + offset, size);
                                if (manager->read_release_pending_ && !was_pending) {
                                    pending_read_releases_.push_back(manager);
                                }
                            });
    }

    /// One region's identity and its gpu_write_seq value at snapshot time.
    /// GPU-command-thread confined, like the counter it captures.
    struct GpuSeqSnapshot {
        RegionManager* manager;
        u64 seq;
    };
    using GpuSeqSnapshots = boost::container::small_vector<GpuSeqSnapshot, 4>;

    /// Captures each overlapped region's GPU write sequence. Call on the GPU
    /// command thread as the download copies are recorded; pass the result to
    /// GpuWriteSeqMatches.
    void SnapshotGpuWriteSeq(VAddr cpu_addr, u64 size, GpuSeqSnapshots& out) {
        IteratePages<false>(cpu_addr, size, [&out](RegionManager* manager, u64, size_t) {
            out.push_back({manager, manager->gpu_write_seq});
        });
    }

    /// True when no new GPU write reached any overlapped region since the
    /// snapshot. A mismatch means the downloaded bytes may be stale: do not
    /// write them back or clear their bits. GPU command thread only, so this
    /// cannot race the writers it guards.
    bool GpuWriteSeqMatches(VAddr cpu_addr, u64 size, const GpuSeqSnapshots& snap) {
        return !IteratePages<false>(cpu_addr, size, [&snap](RegionManager* manager, u64, size_t) {
            const auto it = std::ranges::find(snap, manager, &GpuSeqSnapshot::manager);
            return it == snap.end() || it->seq != manager->gpu_write_seq;
        });
    }

    /// Advances the word epochs of every existing region overlapping the
    /// range. Missing regions have no consumers and are skipped.
    void BumpEpochsForRange(VAddr cpu_addr, u64 size) noexcept {
        IteratePages<false>(cpu_addr, size,
                            [](RegionManager* manager, u64 offset, size_t range_size) {
                                manager->BumpWordEpochs(offset, range_size);
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

    struct EpochSum256 {
        u64 sum;
        bool ok;
    };

    /// Word-epoch sum alone, for consumers that key memos on it. ok is false
    /// when part of the range has no region, when any covered word is
    /// poisoned, or when the span exceeds 64 words; callers then fall back to
    /// their coarse generation key. Nearly every bind lies inside one region,
    /// so that case is answered here and the general walk is the exception.
    EpochSum256 Sum256ForRange(VAddr cpu_addr, u64 size) noexcept {
        RegionManager* region;
        if (size <= MAX_EPOCH_SUM_SPAN && TrySingleRegion(cpu_addr, size, region)) [[likely]] {
            RENDERER_TRACE;
            ++fast_stats_.sum_fast;
            if (region == nullptr) {
                ++fast_stats_.single_null;
                return {0, false};
            }
            EpochSum256 out{0, false};
            out.ok = region->EpochSumResolved(cpu_addr & TRACKER_HIGHER_PAGE_MASK, size, out.sum);
            return out;
        }
        ++fast_stats_.sum_walk;
        return Sum256ForRangeSlow(cpu_addr, size);
    }

    /// Twin of Sum256ForRange that also names the region when one covers the
    /// whole range; the resolved memo probe reads that region directly. Forced
    /// inline: its two callers are the stream-copy and index-bind memo probes,
    /// and the heuristic inliner has declined the neighbouring GPU probe there.
    SHAD_FORCE_INLINE EpochSum256 Sum256ForRangeResolved(VAddr cpu_addr, u64 size,
                                                         RegionManager*& region) noexcept {
        region = nullptr;
        RegionManager* single;
        if (size <= MAX_EPOCH_SUM_SPAN && TrySingleRegion(cpu_addr, size, single)) [[likely]] {
            RENDERER_TRACE;
            ++fast_stats_.sum_fast;
            if (single == nullptr) {
                ++fast_stats_.single_null;
                return {0, false};
            }
            EpochSum256 out{0, false};
            out.ok = single->EpochSumResolved(cpu_addr & TRACKER_HIGHER_PAGE_MASK, size, out.sum);
            region = out.ok ? single : nullptr;
            return out;
        }
        ++fast_stats_.sum_walk;
        return Sum256ForRangeResolvedSlow(cpu_addr, size, region);
    }

    /// The general walk behind Sum256ForRange.
    SHAD_NO_INLINE EpochSum256 Sum256ForRangeSlow(VAddr cpu_addr, u64 size) noexcept {
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

    /// The general walk behind Sum256ForRangeResolved; names the region only
    /// when one covers the whole range.
    SHAD_NO_INLINE EpochSum256 Sum256ForRangeResolvedSlow(VAddr cpu_addr, u64 size,
                                                          RegionManager*& region) noexcept {
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
                    if (RegionManager::ReadbacksMode(&RegionManager::mode_reads_fault_) !=
                            GpuReadbacksMode::Disabled &&
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
                        RegionManager::ReadbacksMode(&RegionManager::mode_reads_fault_) !=
                        GpuReadbacksMode::Disabled;
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
    /// window_size > 0 names the bound buffer, inside which the single-region
    /// walk may widen its upload to the surrounding dirty pages.
    bool ForEachUploadRange(VAddr query_cpu_range, u64 query_size, bool is_written, auto&& func,
                            auto&& on_upload, VAddr window_addr = 0, u64 window_size = 0) {
        // A written bind holds region locks across its upload, which can flush
        // the scheduler; a drain from that flush would take a lock this thread
        // already holds, so it is left to the next drain site.
        if (is_written) {
            ++upload_walk_depth_;
        }
        SCOPE_EXIT {
            if (is_written) {
                --upload_walk_depth_;
            }
        };
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
            if (arm_chunk_pages_per != 0 && window_size != 0) {
                // The chunk is the granule around the query, clamped to the
                // buffer and the region; the widening then takes only the
                // dirty pages adjacent to the query inside it, so the one
                // protection call covers them all and their binds find them
                // clean. The peek above and the GPU marking below keep the
                // original range.
                ++arm_chunk_walks;
                const VAddr region_base = manager->GetCpuAddr();
                const VAddr region_end = region_base + TRACKER_HIGHER_PAGE_SIZE;
                const VAddr win_lo = std::max(window_addr, region_base);
                const VAddr win_hi = std::min<VAddr>(window_addr + window_size, region_end);
                const size_t start_page = offset >> TRACKER_PAGE_BITS;
                const size_t end_page =
                    (offset + query_size + TRACKER_BYTES_PER_PAGE - 1) >> TRACKER_PAGE_BITS;
                const size_t mask = arm_chunk_pages_per - 1;
                const size_t lo_limit = std::max<size_t>(
                    start_page & ~mask,
                    (win_lo - region_base + TRACKER_BYTES_PER_PAGE - 1) >> TRACKER_PAGE_BITS);
                const size_t hi_limit = std::min<size_t>(
                    (end_page + mask) & ~mask, (win_hi - region_base) >> TRACKER_PAGE_BITS);
                auto [lo, hi] = manager->WidenCpuDirty(start_page, end_page, lo_limit, hi_limit);
                if (lo != start_page || hi != end_page) {
                    ++arm_chunk_widened;
                    arm_chunk_pages += (hi - lo) - (end_page - start_page);
                }
                manager->template ForEachModifiedRange<Type::CPU, true>(
                    region_base + (lo << TRACKER_PAGE_BITS), (hi - lo) << TRACKER_PAGE_BITS, func);
            } else {
                manager->template ForEachModifiedRange<Type::CPU, true>(
                    manager->GetCpuAddr() + offset, query_size, func);
            }
            if (!is_written) {
                manager->lock.unlock();
                on_upload();
                return false;
            }
            // A written bind holds the lock from the upload walk until the
            // GPU marking below, so the marking observes the bits it covers.
            on_upload();
            const bool was_pending = manager->read_arm_pending_;
            const bool changed = manager->template ChangeRegionState<Type::GPU, true>(
                manager->GetCpuAddr() + offset, query_size);
            if (manager->read_arm_pending_ && !was_pending) {
                pending_read_arms_.push_back(manager);
            }
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
                    // Register-only scan over the whole clean regions ahead: it
                    // skips the clean middle regions that dominate a read-only
                    // multi-region walk without the general body's per-region
                    // stack traffic. A null slot or a dirty region hands the
                    // walk back to the general body at that region.
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
                    // New GPU write: advance the sequence (see the single-region
                    // path above). GPU-command-thread confined, like the counter.
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
            // Pass two mirrors pass one's region sequence exactly: a region
            // skipped there was never locked, and every region exists because
            // pass one created it (a top_tier slot is never cleared).
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
                const u32 i = unlock_index++;
                if (i < 64 && (skipped & (u64{1} << i)) != 0) {
                    continue; // never locked in the first pass
                }
                const bool was_pending = manager->read_arm_pending_;
                changed |= manager->template ChangeRegionState<Type::GPU, true>(
                    manager->GetCpuAddr() + offset, size);
                if (manager->read_arm_pending_ && !was_pending) {
                    pending_read_arms_.push_back(manager);
                }
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
    /// Drains one pending-watcher list under a single protect-carry scope.
    /// GPU command thread only; a walk holding region locks across its upload
    /// (upload_walk_depth_ != 0) defers to the next site rather than deadlocking.
    /// carry: the caller certifies it is the GPU command thread (see
    /// PageManager::BeginProtectCarry); every other caller passes false.
    template <u32 (RegionManager::*Watchers)(u32&)>
    ReadReleaseDrain DrainPendingWatchers(
        boost::container::small_vector<RegionManager*, 16>& pending, bool carry) {
        ReadReleaseDrain out{};
        if (upload_walk_depth_ != 0) {
            return out;
        }
        // Ascending regions, so a run that ends on a region boundary can be
        // carried into the next call and issued as one protection change.
        if (pending.size() > 1) {
            std::ranges::sort(pending, {}, [](const RegionManager* m) { return m->GetCpuAddr(); });
        }
        const ProtectCarryScope scope{*tracker, carry};
        for (RegionManager* manager : pending) {
            std::scoped_lock lk{manager->lock};
            out.calls += (manager->*Watchers)(out.pages);
            ++out.regions;
        }
        pending.clear();
        return out;
    }

    /// Whether a single-region GPU peek stays inside one bitset word: the
    /// input that decides whether an inline single-word peek is worth its
    /// code size on this workload. size != 0 on both probe paths.
    SHAD_FORCE_INLINE void CountPeekSpan(u64 region_offset, u64 size) noexcept {
        const size_t first = (region_offset / TRACKER_BYTES_PER_PAGE) >> 6;
        const size_t last = ((region_offset + size - 1) / TRACKER_BYTES_PER_PAGE) >> 6;
        fast_stats_.peek_word += first == last;
    }

    /**
     * Resolve a region index to its manager.
     *
     * top_tier spans the 40 bit guest address space at 4MB granularity, so it
     * is a 2MB sparse pointer array whose scattered load stalls; the live
     * region set is tiny, so a small direct mapped memo keeps the probe in L1.
     * This is exactly equivalent to indexing top_tier: a slot only ever goes
     * from null to a manager, is never cleared or reassigned, and only non-null
     * results are memoised. Thread local because the tracker is also driven
     * from the guest fault path, where a shared table could tear a key against
     * its value and hand back the wrong manager.
     */
    static constexpr std::size_t NUM_LOOKUP_SLOTS = 128; // power of two
    // Keys biased by one so a zeroed table reads as empty, keeping the memo
    // constant initialized (constinit enforces it) so no initialization guard
    // is emitted; a 16 byte slot keeps a probe to one cache line.
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

    /// True when [addr, addr + size) lies inside one 4 MB tracker region;
    /// region is that region, or null when it does not exist yet. size == 0
    /// falls through: the addr + size - 1 form underflows for an empty range.
    [[nodiscard]] bool TrySingleRegion(VAddr addr, u64 size, RegionManager*& region) noexcept {
        const std::size_t page_index = addr >> TRACKER_HIGHER_PAGE_BITS;
        if (size == 0 || ((addr + size - 1) >> TRACKER_HIGHER_PAGE_BITS) != page_index) {
            return false;
        }
        region = LookupRegion(page_index);
        return true;
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
            if (manager == nullptr) {
                if constexpr (create_region_on_fail) {
                    manager = CreateRegion(page_index);
                }
            }
            if (manager) {
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
        new_manager->defer_read_arm_ = defer_read_arm_;
        new_manager->defer_read_release_ = defer_read_release_;
        free_managers.pop_back();
        top_tier[page_index] = new_manager;
        // The memo entry for this fresh page fills on its next scattered lookup.
        return new_manager;
    }

    bool defer_read_arm_{};
    bool defer_read_release_{};
    // Regions whose marks await their arm, and the depth of the walk that must
    // not be drained into. GPU-command-thread confined, like gpu_write_seq.
    boost::container::small_vector<RegionManager*, 16> pending_read_arms_;
    boost::container::small_vector<RegionManager*, 16> pending_read_releases_;
    u32 upload_walk_depth_{};
    // Probe telemetry on a line of its own. Every bump here is the GPU
    // command thread's and is drained there, so these are plain adds.
    struct alignas(64) FastPathStats {
        u64 sum_fast{};
        u64 sum_walk{};
        u64 single_null{};
        u64 gpu_fast{};
        u64 gpu_walk{};
        // Single-word vs straddling single-region GPU probes.
        u64 peek_word{};
        // GPU probes that reused the region the caller had already resolved.
        u64 fused{};
    };
    FastPathStats fast_stats_;

    // The readback path probes from the faulting guest thread; those two sites
    // count here so the bind path's own counters need no lock prefix.
    struct alignas(64) ForeignPathStats {
        std::atomic<u64> gpu_fast{};
        std::atomic<u64> gpu_walk{};
    };
    ForeignPathStats foreign_stats_;

    PageManager* tracker;
    std::deque<std::array<RegionManager, MANAGER_POOL_SIZE>> manager_pool;
    std::vector<RegionManager*> free_managers;
    std::array<RegionManager*, NUM_HIGH_PAGES> top_tier{};
};

} // namespace VideoCore
