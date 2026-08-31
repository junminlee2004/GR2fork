// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <chrono>
#include <optional>
#include <thread>
#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>
#include "common/alignment.h"
#include "common/debug.h"
#include "common/logging/log.h"
#include "common/rdtsc.h"
#include "common/scope_exit.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/skipcache/skipcache.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {

static constexpr size_t DataShareBufferSize = 64_KB;
static constexpr size_t StagingBufferSize = 512_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t DeviceBufferSize = 128_MB;

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         AmdGpu::Liverpool* liverpool_, TextureCache& texture_cache_,
                         PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      fault_manager{instance, scheduler, *this, CACHING_PAGEBITS, CACHING_NUMPAGES},
      staging_buffer{instance, scheduler, MemoryUsage::Upload, StagingBufferSize},
      stream_buffer{instance, scheduler, MemoryUsage::Stream,
                    std::max<size_t>(EmulatorSettings.GetStreamBufferSizeMb(), 16) * 1_MB},
      download_buffer{instance, scheduler, MemoryUsage::Download, DownloadBufferSize},
      device_buffer{instance, scheduler, MemoryUsage::DeviceLocal, DeviceBufferSize},
      gds_buffer{instance, scheduler, MemoryUsage::Stream, 0, AllFlags, DataShareBufferSize},
      bda_pagetable_buffer{instance, scheduler, MemoryUsage::DeviceLocal,
                           0,        AllFlags,  BDA_PAGETABLE_SIZE} {
    batch_copy_lock_ = EmulatorSettings.IsGuestCopyLockBatch();
    Vulkan::SetObjectName(instance.GetDevice(), gds_buffer.Handle(), "GDS Buffer");
    Vulkan::SetObjectName(instance.GetDevice(), bda_pagetable_buffer.Handle(),
                          "BDA Page Table Buffer");

    memory_tracker = std::make_unique<MemoryTracker>(tracker);

    std::memset(gds_buffer.mapped_data.data(), 0, DataShareBufferSize);

    // Set up garbage collection parameters
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = DEFAULT_TRIGGER_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    trigger_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_TRIGGER_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
    mirror_mode_ = EmulatorSettings.GetStreamUploadMirrorMode() != 0;
    if (mirror_mode_) {
        Core::SetProtectObserver(&BufferCache::MirrorProtectThunk, this);
        Core::SetBackingWriteObserver(&BufferCache::MirrorBackingThunk, this);
    }
}

BufferCache::~BufferCache() {
    if (mirror_mode_) {
        Core::SetProtectObserver(nullptr, nullptr);
        Core::SetBackingWriteObserver(nullptr, nullptr);
    }
}

void BufferCache::MirrorProtectThunk(void* user, VAddr addr, u64 size, bool write_granted,
                                     bool tracker_origin) {
    // Runs on any thread, including inside the fault handler under the page
    // manager's range locks: only atomics and the constinit lookup memo are
    // touched, never locks or allocation.
    if (!write_granted) {
        return;
    }
    auto* cache = static_cast<BufferCache*>(user);
    cache->memory_tracker->BumpEpochsForRange(addr, size, tracker_origin ? 0 : 1);
    if (tracker_origin) {
        cache->mirror_sink_.bump_tracker.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Guest protection grants can hide later guest writes from every
        // watcher, so the covered words are conservatively poisoned.
        cache->mirror_sink_.bump_guestapi.fetch_add(1, std::memory_order_relaxed);
        cache->memory_tracker->PoisonEpochsForRange(addr, size);
        cache->mirror_sink_.poisoned.fetch_add(1, std::memory_order_relaxed);
    }
}

void BufferCache::MirrorBackingThunk(void* user, VAddr addr, u64 size) {
    auto* cache = static_cast<BufferCache*>(user);
    cache->memory_tracker->BumpEpochsForRange(addr, size, 2);
    cache->mirror_sink_.bump_backing.fetch_add(1, std::memory_order_relaxed);
}

void BufferCache::MirrorOracleProbeSlow(VAddr device_addr, u32 size, bool tick_hit,
                                        bool gpu_dirty) {
    auto& skipcache = VideoCore::Skipcache::Framework::Instance();
    if (!mirror_mode_ || !skipcache.ShouldProbe(VideoCore::Skipcache::CacheId::StreamMirror)) {
        return;
    }
    auto& mo = mirror_oracle_;
    ++mo.elig;
    mo.elig_bytes += size;
    if (!tick_hit) {
        ++mo.tick_miss;
    }
    if (gpu_dirty) {
        ++mo.gpu_dirty;
        return;
    }
    const bool cpu_dirty = memory_tracker->PeekRegionCpuModifiedNoCreate(device_addr, size);
    if (cpu_dirty) {
        ++mo.cpu_dirty;
        mo.cpu_dirty_bytes += size;
    } else {
        ++mo.clean;
        mo.clean_bytes += size;
        if (!tick_hit) {
            ++mo.clean_tm;
            mo.clean_tm_bytes += size;
        }
    }
    const auto sums = memory_tracker->SumEpochsForRange(device_addr, size);
    if (!sums.ok) {
        ++mo.sum_unresolved;
        return;
    }
    // The source is not GPU modified here, so its pages are readable; a racing
    // GPU mark can still fault this read, which resolves through the GPU
    // thread's own synchronous fault path like any guest access.
    const u64 hash = XXH3_64bits(reinterpret_cast<const void*>(device_addr), size);

    struct OracleEntry {
        VAddr addr;
        u32 size;
        u8 clean;
        u64 hash;
        u64 sum256;
        u64 sum64;
    };
    constexpr size_t kOracleSlots = 65536;
    struct OracleTable {
        std::array<OracleEntry, kOracleSlots> slots{};
    };
    // Heap-backed for the same TLS-budget reason as the stream copy cache.
    static thread_local std::unique_ptr<OracleTable> oracle;
    if (!oracle) {
        oracle = std::make_unique<OracleTable>();
    }
    u64 key = device_addr ^ (u64{size} * 0x9e3779b97f4a7c15ULL);
    key ^= key >> 29;
    OracleEntry* entry = nullptr;
    OracleEntry* victim = nullptr;
    for (size_t way = 0; way < 2; ++way) {
        OracleEntry& slot = oracle->slots[(key + way) & (kOracleSlots - 1)];
        if (slot.addr == device_addr && slot.size == size) {
            entry = &slot;
            break;
        }
        if (victim == nullptr || slot.addr == 0) {
            victim = &slot;
        }
    }
    if (entry == nullptr) {
        ++mo.cold;
        if (victim->addr != 0) {
            ++mo.evict;
        }
        if (!cpu_dirty) {
            ++mo.ws_keys;
            mo.ws_bytes += size;
        }
        *victim = {device_addr, size,        static_cast<u8>(cpu_dirty ? 0 : 1),
                   hash,        sums.sum256, sums.sum64};
        return;
    }
    const bool sum256_same = entry->sum256 == sums.sum256;
    const bool sum64_same = entry->sum64 == sums.sum64;
    const bool hash_same = entry->hash == hash;
    if (entry->clean) {
        if (sum256_same && hash_same) {
            ++mo.hit_clean;
            if (!tick_hit) {
                ++mo.hit_clean_tm;
            }
        } else if (sum256_same && !hash_same) {
            // The soundness lane: content changed under a stable, unpoisoned
            // epoch sum. Any nonzero count falsifies the substrate.
            if (!sums.poisoned) {
                ++mo.div;
                skipcache.RecordDivergence(VideoCore::Skipcache::CacheId::StreamMirror,
                                           "epoch stable content changed");
            } else {
                ++mo.changed;
            }
        } else if (!sum256_same && hash_same) {
            ++mo.alias256;
            if (sum64_same) {
                ++mo.alias64;
            }
        } else {
            ++mo.changed;
        }
    } else {
        if (sum256_same && hash_same) {
            ++mo.dirty_stable;
        } else if (sum256_same) {
            ++mo.dirty_stable_chg;
        } else {
            ++mo.dirty_moved;
        }
    }
    *entry = {device_addr, size, static_cast<u8>(cpu_dirty ? 0 : 1), hash, sums.sum256, sums.sum64};
}

void BufferCache::EmitMirrorTelemetry() {
    if (!mirror_mode_) {
        return;
    }
    auto& mo = mirror_oracle_;
    const auto pct = [](u64 part, u64 whole) {
        return whole ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
    };
    LOG_INFO(Render_Skipcache,
             "[SkipCache] MIRROR elig={} tickmiss={} clean%={:.1f} cleanTM%={:.1f} "
             "cleanTM_MiB={:.1f} hitC%={:.1f} hitCTM={} cold={} evict={} unres={} per300f",
             mo.elig, mo.tick_miss, pct(mo.clean, mo.elig), pct(mo.clean_tm, mo.tick_miss),
             static_cast<double>(mo.clean_tm_bytes) / (1024.0 * 1024.0),
             pct(mo.hit_clean, mo.clean), mo.hit_clean_tm, mo.cold, mo.evict, mo.sum_unresolved);
    LOG_INFO(Render_Skipcache,
             "[SkipCache] MIRRORVETO div={} dirty_stable={} dirty_stable_chg={} dirty_moved={} "
             "alias256={} alias64={} changed={} gpudirty={} cpudirty={} cpudirty_MiB={:.1f} "
             "per300f",
             mo.div, mo.dirty_stable, mo.dirty_stable_chg, mo.dirty_moved, mo.alias256, mo.alias64,
             mo.changed, mo.gpu_dirty, mo.cpu_dirty,
             static_cast<double>(mo.cpu_dirty_bytes) / (1024.0 * 1024.0));
    LOG_INFO(Render_Skipcache,
             "[SkipCache] MIRROREPOCH tracker={} guestapi={} backing={} poisoned={} per300f",
             mirror_sink_.bump_tracker.exchange(0, std::memory_order_relaxed),
             mirror_sink_.bump_guestapi.exchange(0, std::memory_order_relaxed),
             mirror_sink_.bump_backing.exchange(0, std::memory_order_relaxed),
             mirror_sink_.poisoned.exchange(0, std::memory_order_relaxed));
    LOG_INFO(Render_Skipcache,
             "[SkipCache] MIRRORTIERA hit%={:.1f} hits={} walks={} elig_walks={} "
             "span_le64%={:.1f} ws_keys={} ws_MiB={:.1f} per300f",
             pct(mo.tierA_hits, mo.tierA_hits + mo.tierA_walks), mo.tierA_hits, mo.tierA_walks,
             mo.tierA_elig_walks, pct(mo.tierA_span_le64, mo.tierA_walks), mo.ws_keys,
             static_cast<double>(mo.ws_bytes) / (1024.0 * 1024.0));
    LOG_INFO(Render_Skipcache, "[SkipCache] PEEKBASE calls={} dirty={} per300f",
             memory_tracker->peek_fastpath_calls, memory_tracker->peek_fastpath_dirty);
    memory_tracker->peek_fastpath_calls = 0;
    memory_tracker->peek_fastpath_dirty = 0;
    mo = MirrorOracleCounters{};
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size) {
    if (!IsRegionRegistered(device_addr, size)) {
        return;
    }
    // Rounded down to a power of two once; the flush callback keeps the
    // original range so readbacks never widen.
    static const u64 widen = std::bit_floor(u64{EmulatorSettings.GetFaultWidenBytes()});
    if (widen >= TRACKER_BYTES_PER_PAGE) {
        const VAddr wide_addr = device_addr & ~(widen - 1);
        const u64 wide_size = ((device_addr + size + widen - 1) & ~(widen - 1)) - wide_addr;
        memory_tracker->InvalidateRegionWidened(
            device_addr, size, wide_addr, wide_size,
            [this, device_addr, size] { ReadMemory(device_addr, size, true); });
        return;
    }
    memory_tracker->InvalidateRegion(
        device_addr, size, [this, device_addr, size] { ReadMemory(device_addr, size, true); });
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write) {
    // Offloaded form: the faulting thread is blocked until its data arrives no
    // matter what, so it - not the GPU command thread - should absorb the
    // semaphore wait. The GPU command thread only records the download and
    // flushes (PrepareFaultDownload), then writes the bytes back once the
    // faulting thread has waited out the fence (FinishFaultDownload); between
    // the two hops it is free to keep translating draws. Measured before this
    // change, that wait held the GPU command thread for a third of its wall
    // clock. When this function is reached from the GPU command thread itself
    // (its own guest-memory read faulting), SendCommand runs the hops inline
    // and the wait lands where it always did - never worse than the sync form.
    //
    // The bounded mode caps the faulting thread's fence wait. Some titles gate
    // further GPU progress on memory the faulting thread writes only after its
    // read returns; an unbounded wait then deadlocks guest against GPU. On
    // timeout the job moves to the priority waiter thread, the write-back runs
    // on the GPU command thread once the fence signals, and the refault loop
    // above resolves the access through the in-flight-window spin.
    const u32 offload_mode = EmulatorSettings.GetReadbackOffloadMode();
    if (offload_mode != GpuReadbackOffloadMode::OffloadDisabled) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            FaultDownloadJob job;
            liverpool->SendCommand<true>(
                [&] { PrepareFaultDownload(job, device_addr, size, is_write); });
            if (!job.has_download) {
                // Nothing pending for this window usually means the fault is
                // resolved, but it also happens when another thread's download
                // owns the ranges and has not written back yet. Waiting here
                // instead of returning keeps that thread from re-faulting in a
                // tight loop and hammering the GPU command thread with empty
                // download requests while the first one is in flight.
                for (int spin = 0;
                     spin < 400 && memory_tracker->IsRegionGpuModified(device_addr, size); ++spin) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                return;
            }
            const u64 t0 = Common::FencedRDTSC();
            if (offload_mode == GpuReadbackOffloadMode::OffloadBounded) {
                constexpr u64 BoundedWaitNs = 100'000'000;
                if (!scheduler.GetMasterSemaphore()->WaitFor(job.wait_tick, BoundedWaitNs)) {
                    offload_wait_ns_.fetch_add(Common::FencedRDTSC() - t0,
                                               std::memory_order_relaxed);
                    auto deferred = std::make_unique<FaultDownloadJob>(std::move(job));
                    scheduler.DeferPriorityOperationAt(
                        deferred->wait_tick,
                        [this, device_addr, size, is_write, job = std::move(deferred)]() mutable {
                            liverpool->SendCommand<false>(
                                [this, device_addr, size, is_write, job = std::move(job)] {
                                    FinishFaultDownload(*job, device_addr, size, is_write);
                                });
                        });
                    return;
                }
            } else {
                scheduler.GetMasterSemaphore()->Wait(job.wait_tick);
            }
            offload_wait_ns_.fetch_add(Common::FencedRDTSC() - t0, std::memory_order_relaxed);
            liverpool->SendCommand<true>(
                [&] { FinishFaultDownload(job, device_addr, size, is_write); });
            // The faulted range itself may sit outside the vetoed regions, in
            // which case its pages are clear and the fault is resolved even if
            // part of the window was not.
            if (job.fully_cleared || !memory_tracker->IsRegionGpuModified(device_addr, size)) {
                return;
            }
        }
        // Newer GPU writes kept landing in the window while it was in flight.
        // Fall through to the synchronous form, which blocks the GPU command
        // thread and therefore cannot be outrun.
        offload_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    }
    liverpool->SendCommand<true>([this, device_addr, size, is_write] {
        Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];
        // GPU-modified ranges come as many small scattered islands, so the download
        // is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr buf_start = buffer.CpuAddr();
        const VAddr buf_end = buf_start + buffer.SizeBytes();
        VAddr window_start = std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), buf_start);
        VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), buf_end);
        if (EmulatorSettings.IsReadbackBatchingEnabled()) {
            // Every readback costs a full GPU drain, so the drain - not the
            // copy - is what to economise on: service the whole buffer at once
            // and later faults in it find their data already downloaded. Only
            // GPU-modified sub-ranges are copied either way, so the number of
            // bytes moved is unchanged.
            window_start = buf_start;
            window_end = buf_end;
        }
        DownloadBufferMemory<false>(buffer, window_start, window_end - window_start);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    });
}

void BufferCache::PrepareFaultDownload(FaultDownloadJob& job, VAddr device_addr, u64 size,
                                       bool is_write) {
    Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];
    // Window widening mirrors the synchronous form above.
    constexpr u64 WindowSize = 512_KB;
    const VAddr buf_start = buffer.CpuAddr();
    const VAddr buf_end = buf_start + buffer.SizeBytes();
    VAddr window_start = std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), buf_start);
    VAddr window_end =
        std::min<VAddr>(std::max<VAddr>(window_start + WindowSize, device_addr + size), buf_end);
    if (EmulatorSettings.IsReadbackBatchingEnabled()) {
        window_start = buf_start;
        window_end = buf_end;
    }

    // Copy collection mirrors DownloadBufferMemory, except destinations index
    // the job's dedicated staging from zero. The shared download ring cannot be
    // used here: its reclamation tracks GPU ticks only, and the bytes must
    // survive on the host until the faulting thread has consumed them.
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        window_start, window_end - window_start, [&](u64 device_addr_out, u64 range_size) {
            const VAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                job.copies.push_back(vk::BufferCopy{
                    .srcOffset = new_offset,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
            };
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        // Nothing pending for the window. Unlike the synchronous form, empty
        // does not imply resolved here: another thread's in-flight job may own
        // the ranges while the bits are still set. Marking CPU-dirty in that
        // state would drop the write watcher while the read watcher is armed,
        // which is an invalid (write-only) protection. Mark only when the
        // faulted range is actually clear; otherwise the caller's damping loop
        // and the refault path converge once the owner finishes.
        if (is_write && !memory_tracker->IsRegionGpuModified(device_addr, size)) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
        return;
    }

    job.staging = AcquireFaultStaging(total_size_bytes);
    memory_tracker->SnapshotGpuWriteSeq(window_start, window_end - window_start, job.snapshots);
    job.buffer_base = buffer.CpuAddr();
    job.window_start = window_start;
    job.window_size = window_end - window_start;

    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    // Synchronize prior GPU writes to this buffer before the transfer read
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = buffer.buffer,
        .offset = 0,
        .size = buffer.SizeBytes(),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(buffer.buffer, job.staging->Handle(), job.copies);
    job.wait_tick = scheduler.CurrentTick();
    scheduler.Flush();
    job.has_download = true;
    offload_jobs_.fetch_add(1, std::memory_order_relaxed);
}

void BufferCache::FinishFaultDownload(FaultDownloadJob& job, VAddr device_addr, u64 size,
                                      bool is_write) {
    job.staging->InvalidateForRead(0, VK_WHOLE_SIZE);
    auto* memory = Core::Memory::Instance();
    const u8* download = job.staging->mapped_data.data();

    // Every copy gets its own verdict. This runs on the GPU command thread, so
    // between the sequence test and the bit clear nothing can interleave; and
    // because the sequence advances on every recorded GPU write - including
    // rebinds of already-dirty regions - a matching sequence proves any other
    // download of the same range holds byte-identical data, making write-back
    // order among matching jobs irrelevant. A mismatched copy is written back
    // by nobody: its bytes may predate the newer write.
    //
    // Verified copies arrive in ascending address order, so page-adjacent runs
    // are merged into one pending span and unmarked in a single call: the page
    // watcher update then coalesces the whole run into one protection change
    // instead of one per copy. Unmarking never advances the write sequence, so
    // deferral cannot change any later verdict. A merge never crosses a page
    // gap - gap pages may be GPU-dirty outside this download - which keeps the
    // unmarked page set exactly the union of the per-copy page sets. The span
    // is flushed before any tracker read that could see its pages: the veto
    // branch below, and the CPU mark at the end (marking with GPU bits still
    // set would ask for a write-only page, which Protect() rejects).
    VAddr pending_start = 0;
    VAddr pending_end = 0;
    const auto flush_pending_unmark = [&] {
        if (pending_end != pending_start) {
            memory_tracker->UnmarkRegionAsGpuModified(pending_start, pending_end - pending_start);
            pending_start = 0;
            pending_end = 0;
        }
    };
    bool vetoed_any = false;
    for (const auto& copy : job.copies) {
        const VAddr copy_device_addr = job.buffer_base + copy.srcOffset;
        if (memory_tracker->GpuWriteSeqMatches(copy_device_addr, copy.size, job.snapshots)) {
            memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr), download + copy.dstOffset,
                                    copy.size);
            // The ordering guard makes an out-of-order copy flush and restart
            // the span instead of moving pending_end backward, so the merged
            // range can never grow beyond the union of the copies.
            const bool adjacent = pending_end != pending_start && copy_device_addr >= pending_end &&
                                  Common::AlignDown(copy_device_addr, TRACKER_BYTES_PER_PAGE) <=
                                      Common::AlignUp(pending_end, TRACKER_BYTES_PER_PAGE);
            if (!adjacent) {
                flush_pending_unmark();
                pending_start = copy_device_addr;
            }
            pending_end = std::max<VAddr>(pending_end, copy_device_addr + copy.size);
            continue;
        }
        vetoed_any = true;
        flush_pending_unmark();
        // The newer write's own bind restored the range set for its span; put
        // back only what is still marked and uncovered, so the tracker bits
        // and the range set stay in lockstep. New dirty coverage expires the
        // GPU-clean epoch, per the protocol at every other Add site.
        if (memory_tracker->IsRegionGpuModified(copy_device_addr, copy.size) &&
            !gpu_modified_ranges.Contains(copy_device_addr, copy.size)) {
            ++gpu_dirty_generation_;
            gpu_modified_ranges.Add(copy_device_addr, copy.size);
        }
    }
    flush_pending_unmark();
    job.fully_cleared = !vetoed_any;
    if (vetoed_any) {
        offload_vetoes_.fetch_add(1, std::memory_order_relaxed);
    }
    // Same write-only-protection hazard as the empty path in Prepare: the mark
    // is only legal once the faulted range's GPU bits are clear. When a veto
    // kept them set, the caller's retry loop resolves the fault instead.
    if (is_write && !memory_tracker->IsRegionGpuModified(device_addr, size)) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
    }
    ReleaseFaultStaging(std::move(job.staging));
}

std::unique_ptr<Buffer> BufferCache::AcquireFaultStaging(u64 size) {
    for (auto it = fault_staging_pool_.begin(); it != fault_staging_pool_.end(); ++it) {
        if ((*it)->SizeBytes() >= size) {
            auto staging = std::move(*it);
            fault_staging_pool_.erase(it);
            return staging;
        }
    }
    constexpr u64 MinStagingSize = 512_KB;
    const u64 wanted = std::max<u64>(std::bit_ceil(size), MinStagingSize);
    return std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Download, 0,
                                    vk::BufferUsageFlagBits::eTransferDst, wanted);
}

void BufferCache::ReleaseFaultStaging(std::unique_ptr<Buffer> staging) {
    // The recorded copy into this staging retired before FinishFaultDownload
    // ran (the faulting thread waited out the tick), so dropping it here is
    // safe when the pool is full or the buffer is an outlier size.
    constexpr size_t MaxPooledBuffers = 4;
    constexpr u64 MaxPooledSize = 16_MB;
    if (fault_staging_pool_.size() >= MaxPooledBuffers || staging->SizeBytes() > MaxPooledSize) {
        return;
    }
    fault_staging_pool_.push_back(std::move(staging));
}

template <bool async>
void BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const VAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(vk::BufferCopy{
                    .srcOffset = new_offset,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
            };
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += offset;
    }
    download_buffer.Commit();
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    // Synchronize prior GPU writes to this buffer before the transfer read
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = buffer.buffer,
        .offset = 0,
        .size = buffer.SizeBytes(),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), copies);
    const auto write_data = [&]() {
        auto* memory = Core::Memory::Instance();
        for (const auto& copy : copies) {
            const VAddr copy_device_addr = buffer.CpuAddr() + copy.srcOffset;
            const u64 dst_offset = copy.dstOffset - offset;
            memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr), download + dst_offset,
                                    copy.size);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
    };
    if constexpr (async) {
        scheduler.DeferOperation(write_data);
    } else {
        const u64 t0 = Common::FencedRDTSC();
        scheduler.Finish();
        scheduler.RecordWait(Vulkan::Scheduler::WaitSite::DownloadBuffer,
                             Common::FencedRDTSC() - t0);
        write_data();
    }
}

void BufferCache::BindVertexBuffers(
    const Vulkan::GraphicsPipeline& pipeline,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const bool batch_copy_lock = batch_copy_lock_;
    std::optional<Core::MemoryManager::GuestCopyScope> copy_scope;
    if (batch_copy_lock) {
        copy_scope.emplace(Core::Memory::Instance());
    }
    const auto& regs = liverpool->regs;
    Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    Vulkan::VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        vk::Buffer vk_buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Signatures over the RESOLVED inputs: the vertex input layout feeds
    // setVertexInputEXT, the full bind signature adds the guest V# contents.
    // Content-keyed, so the user_data churn that defeats raw-register memos
    // does not apply. location, binding, inputRate, offset and the divisor
    // class are all read out of the pipeline's own fetch shader, so the
    // pipeline pointer and the two step rates already key them; only stride
    // and format vary under a fixed pipeline. divisors is empty here because
    // the bindings carry the divisor inline on the 2EXT path.
    auto& skipcache = VideoCore::Skipcache::Framework::Instance();
    const bool memo_active = skipcache.Active();
    const auto mix = [](u64& h, u64 v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    u64 input_sig = 0xcbf29ce484222325ULL;
    u64 bind_sig = 0;
    // Decoded once per binding and reused by the signature, the memo span, the
    // merged range list and the final offsets; on a memo hit those consumers
    // never run and the list is discarded, and the loop cannot be deferred
    // because the key that decides the hit is not complete until it ends. An
    // unbound binding takes {~0, 0} so the span reduction needs no second
    // predicate. Indexing attributes and bindings in parallel is exact only
    // because GetVertexInputs pushes to all three vectors once per fetch
    // shader attribute.
    Vulkan::VertexInputs<BufferRange> spans;
    VAddr span_lo = ~VAddr{0};
    VAddr span_hi = 0;
    for (size_t i = 0; i < guest_buffers.size(); ++i) {
        const auto& gb = guest_buffers[i];
        const VAddr base = gb.base_address;
        const u32 size = gb.GetSize();
        const bool bound = base != 0 && size > 0;
        const VAddr lo = bound ? base : ~VAddr{0};
        const VAddr hi = bound ? base + size : VAddr{0};
        spans.emplace_back(lo, hi);
        if (memo_active) {
            mix(input_sig, (static_cast<u64>(static_cast<u32>(attributes[i].format)) << 32) |
                               static_cast<u64>(bindings[i].stride));
            mix(bind_sig, static_cast<u64>(base));
            mix(bind_sig, static_cast<u64>(size));
            span_lo = std::min<VAddr>(span_lo, lo);
            span_hi = std::max<VAddr>(span_hi, hi);
        }
    }
    if (memo_active) {
        mix(input_sig, static_cast<u64>(regs.vgt_instance_step_rate_0));
        mix(input_sig, static_cast<u64>(regs.vgt_instance_step_rate_1));
        mix(input_sig, reinterpret_cast<u64>(&pipeline));
        mix(bind_sig, input_sig);
        const u64 tick = scheduler.CurrentTick();
        // The memo keys on the bound span's word-epoch sum under the mirror
        // mode, so faults outside the span no longer invalidate it; the
        // certificate is the same one the sync memo relies on.
        u64 mem_key = skipcache.Gens().mem_gen.load(std::memory_order_acquire);
        bool mem_key_ok = true;
        if (mirror_mode_) {
            if (span_hi > span_lo) {
                const auto sum = memory_tracker->Sum256ForRange(span_lo, span_hi - span_lo);
                mem_key = sum.sum;
                mem_key_ok = sum.ok;
            } else {
                mem_key_ok = false;
            }
        }
        if (mem_key_ok && vertex_bind_valid_ && bind_sig == vertex_bind_sig_ &&
            tick == vertex_bind_tick_ && mem_key == vertex_bind_mem_key_) {
            // Identical resolved layout and buffer contents descriptors on the
            // same command buffer with no intervening CPU write. GPU writes do
            // not move the key, so a skip additionally requires that no bound
            // range is GPU modified (else the re-bind's barrier is required).
            // The generation only moves on new GPU-dirty coverage, so an
            // unchanged value reproves the recorded clean answer without the
            // region walk.
            if (vertex_bind_clean_gpu_gen_ == gpu_dirty_generation_) {
                return;
            }
            if (span_hi <= span_lo || !IsRegionGpuModified(span_lo, span_hi - span_lo)) {
                vertex_bind_clean_gpu_gen_ = gpu_dirty_generation_;
                return;
            }
        }
        vertex_bind_sig_ = bind_sig;
        vertex_bind_tick_ = tick;
        vertex_bind_mem_key_ = mem_key;
        vertex_bind_valid_ = mem_key_ok;
        // A fresh stamp invalidates the standing clean proof.
        vertex_bind_clean_gpu_gen_ = 0;
    } else {
        vertex_bind_valid_ = false;
        vertex_input_valid_ = false;
    }

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs, unless the layout is unchanged on
        // this command buffer.
        const u64 input_tick = scheduler.CurrentTick();
        if (!memo_active || !vertex_input_valid_ || input_sig != vertex_input_sig_ ||
            vertex_input_tick_ != input_tick) {
            const auto cmdbuf = scheduler.CommandBuffer();
            cmdbuf.setVertexInputEXT(bindings, attributes);
            vertex_input_sig_ = input_sig;
            vertex_input_tick_ = input_tick;
            vertex_input_valid_ = memo_active;
        }
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    // Coalesce the bound spans into disjoint ranges. Every entry the span
    // touches collapses into it, which is the whole merge: the interleaved
    // attribute layout yields one range, so this costs a compare per binding
    // where sorting the spans first costs a 32-byte move per binding.
    Vulkan::VertexInputs<BufferRange> ranges_merged{};
    for (const auto& span : spans) {
        if (span.end_address == 0) {
            continue;
        }
        VAddr lo = span.base_address;
        VAddr hi = span.end_address;
        size_t kept = 0;
        for (size_t i = 0; i < ranges_merged.size(); ++i) {
            const BufferRange range = ranges_merged[i];
            if (range.end_address < lo || hi < range.base_address) {
                ranges_merged[kept++] = range;
                continue;
            }
            lo = std::min<VAddr>(lo, range.base_address);
            hi = std::max<VAddr>(hi, range.end_address);
        }
        ranges_merged.resize(kept);
        ranges_merged.emplace_back(lo, hi);
    }
    // Ascending order is what the buffer creation path saw before, and the
    // merged list is one entry in the common case.
    if (ranges_merged.size() > 1) {
        std::ranges::sort(ranges_merged, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
    }

    // Map buffers for merged ranges
    bool span_proven_clean = memo_active && vertex_bind_valid_ && ranges_merged.size() == 1;
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        // Resolved once and reused: ObtainBuffer would otherwise walk the same
        // range, and the barrier decision below would walk it a second time.
        const bool gpu_modified = IsRegionGpuModified(range.base_address, size);
        if (size != range.GetSize() || gpu_modified) {
            span_proven_clean = false;
        }
        const auto [buffer, offset] =
            ObtainBuffer(range.base_address, size, false, false, {}, gpu_modified);
        range.vk_buffer = buffer->buffer;
        range.offset = offset;
        if (gpu_modified) {
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers.emplace_back(*barrier);
            }
        }
    }
    // A single merged range spans the whole memo span, so one clean unclamped
    // walk proves it at the current generation. The seed relies on the loop
    // reading guest memory only inside the walked range: a read outside it
    // could fault into a veto re-Add that bumps the generation mid-loop.
    if (span_proven_clean) {
        vertex_bind_clean_gpu_gen_ = gpu_dirty_generation_;
    }

    // Bind vertex buffers
    Vulkan::VertexInputs<vk::Buffer> host_buffers;
    Vulkan::VertexInputs<vk::DeviceSize> host_offsets;
    Vulkan::VertexInputs<vk::DeviceSize> host_sizes;
    Vulkan::VertexInputs<vk::DeviceSize> host_strides;
    // Sizes and strides feed only the bindVertexBuffers2 fallback; on devices
    // with vertex-input dynamic state the fills are dead.
    const bool needs_sizes_strides = !instance.IsVertexInputDynamicState();
    for (size_t i = 0; i < spans.size(); ++i) {
        const VAddr base = spans[i].base_address;
        if (spans[i].end_address != 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return base >= range.base_address && base < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->vk_buffer);
            host_offsets.push_back(host_buffer_info->offset + base -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        if (needs_sizes_strides) {
            host_sizes.push_back(guest_buffers[i].GetSize());
            host_strides.push_back(guest_buffers[i].GetStride());
        }
    }

    const auto cmdbuf = scheduler.CommandBuffer();
    const auto num_buffers = guest_buffers.size();
    if (instance.IsVertexInputDynamicState()) {
        cmdbuf.bindVertexBuffers(0, num_buffers, host_buffers.data(), host_offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, host_buffers.data(), host_offsets.data(),
                                  host_sizes.data(), host_strides.data());
    }
}

void BufferCache::BindIndexBuffer(
    u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const bool batch_copy_lock = batch_copy_lock_;
    std::optional<Core::MemoryManager::GuestCopyScope> copy_scope;
    if (batch_copy_lock) {
        copy_scope.emplace(Core::Memory::Instance());
    }
    const auto& regs = liverpool->regs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    // Resolved lazily: the memo check, the acquire and the barrier decision
    // all ask the same question about the same range, and a memo hit whose
    // range is proven clean at the current GPU-dirty generation needs no walk.
    std::optional<bool> index_gpu_modified{};
    auto& skipcache = VideoCore::Skipcache::Framework::Instance();
    if (skipcache.Active()) {
        const u64 tick = scheduler.CurrentTick();
        u64 mem_key = skipcache.Gens().mem_gen.load(std::memory_order_acquire);
        bool mem_key_ok = true;
        if (mirror_mode_) {
            const auto sum = memory_tracker->Sum256ForRange(index_address, index_buffer_size);
            mem_key = sum.sum;
            mem_key_ok = sum.ok;
        }
        if (mem_key_ok && index_bind_valid_ && index_bind_addr_ == index_address &&
            index_bind_size_ == index_buffer_size &&
            index_bind_type_ == static_cast<u32>(index_type) && index_bind_tick_ == tick &&
            index_bind_mem_key_ == mem_key) {
            // Same resolved index range already bound on this command buffer
            // with no intervening CPU write; an unchanged GPU-dirty generation
            // reproves the recorded clean answer without the region walk.
            if (index_bind_clean_gpu_gen_ == gpu_dirty_generation_) {
                return;
            }
            index_gpu_modified = IsRegionGpuModified(index_address, index_buffer_size);
            if (!*index_gpu_modified) {
                index_bind_clean_gpu_gen_ = gpu_dirty_generation_;
                return;
            }
        }
        index_bind_addr_ = index_address;
        index_bind_size_ = index_buffer_size;
        index_bind_type_ = static_cast<u32>(index_type);
        index_bind_tick_ = tick;
        index_bind_mem_key_ = mem_key;
        index_bind_valid_ = mem_key_ok;
        // A fresh stamp invalidates the standing clean proof.
        index_bind_clean_gpu_gen_ = 0;
    } else {
        index_bind_valid_ = false;
    }
    // has_value distinguishes an unresolved answer from a resolved false.
    if (!index_gpu_modified.has_value()) {
        index_gpu_modified = IsRegionGpuModified(index_address, index_buffer_size);
        if (index_bind_valid_ && !*index_gpu_modified) {
            // The walked range is exactly the stamped memo range, so a fresh
            // clean answer seeds the walk skip for the next hit.
            index_bind_clean_gpu_gen_ = gpu_dirty_generation_;
        }
    }
    const auto [vk_buffer, offset] =
        ObtainBuffer(index_address, index_buffer_size, false, false, {}, *index_gpu_modified);
    if (*index_gpu_modified) {
        if (auto barrier = vk_buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                 vk::PipelineStageFlagBits2::eIndexInput)) {
            barriers.emplace_back(*barrier);
        }
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindIndexBuffer(vk_buffer->Handle(), offset, index_type);
}

void BufferCache::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    ASSERT_MSG(address % 4 == 0, "GDS offset must be dword aligned");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + num_bytes / sizeof(u32), value);
            return;
        }
    }
    Buffer* buffer = [&] {
        if (is_gds) {
            return &gds_buffer;
        }
        const auto [buffer, offset] = ObtainBuffer(address, num_bytes, true);
        return buffer;
    }();
    buffer->Fill(buffer->Offset(address), num_bytes, value);
}

void BufferCache::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds && !IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
        // Without a readback there's nothing we can do with this
        // Fallback to creating dst buffer on GPU to at least have this data there
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    auto& src_buffer = [&] -> const Buffer& {
        if (src_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(src, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, src, num_bytes, false, true);
        return buffer;
    }();
    auto& dst_buffer = [&] -> const Buffer& {
        if (dst_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(dst, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, dst, num_bytes, true, true);
        if (!gpu_modified_ranges.Contains(dst, num_bytes)) {
            ++gpu_dirty_generation_;
            gpu_modified_ranges.Add(dst, num_bytes);
        }
        return buffer;
    }();
    const vk::BufferCopy region = {
        .srcOffset = src_buffer.Offset(src),
        .dstOffset = dst_buffer.Offset(dst),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 buf_barriers_before[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_before,
    });
    cmdbuf.copyBuffer(src_buffer.Handle(), dst_buffer.Handle(), region);
    const vk::BufferMemoryBarrier2 buf_barriers_after[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_after,
    });
}

std::pair<Buffer*, u32> BufferCache::ObtainBuffer(VAddr device_addr, u32 size, bool is_written,
                                                  bool is_texel_buffer, BufferId buffer_id,
                                                  std::optional<bool> gpu_modified) {
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= CACHING_PAGESIZE) {
        // Small read-only uploads dominate this function: the same (addr, size)
        // binds repeat across draws within one command buffer, so a repeat bind
        // can reuse the offset the previous copy landed at.
        //
        // The submission tick is load bearing, not merely a freshness hint. An
        // unchanged memory generation does NOT prove the guest left the source
        // alone: under precise readbacks a read fault can drop a page's
        // protection without setting its CPU dirty bit, after which guest
        // writes land unobserved. Keying on the tick keeps a reused offset
        // confined to a single command buffer, which is what actually bounds
        // the exposure. Widening that window corrupts vertex data.
        //
        // The table is deliberately small enough to stay resident in L1: it is
        // probed on every call with a hashed index, so a large table turns each
        // probe into a cache miss and evicts hot data for a lookup that rarely
        // hits.
        auto& skipcache = VideoCore::Skipcache::Framework::Instance();
        if (skipcache.Active()) {
            struct StreamCopyCacheEntry {
                VAddr addr; // 0 = invalid
                u64 tick;
                // Host-memory generation, or the range's word-epoch sum under
                // the mirror mode: range-local keying survives the constant
                // generation churn of unrelated faults.
                u64 mem_key;
                u64 gpu_gen;
                u32 offset;
                u32 size;
            };
            constexpr size_t kSets = 64;
            constexpr size_t kWays = 2;
            struct StreamCopyCache {
                std::array<std::array<StreamCopyCacheEntry, kWays>, kSets> sets{};
                std::array<u8, kSets> lru{};
            };
            // Heap-backed: only the pointer lives in TLS (a large in-TLS array
            // blows the guest pthread TLS budget at create time). Probes run on the
            // GPU command thread only; the plain stream_copy_* counters share that contract.
            static thread_local std::unique_ptr<StreamCopyCache> storage;
            if (!storage) {
                storage = std::make_unique<StreamCopyCache>();
            }
            auto& cache = *storage;
            u64 key = static_cast<u64>(device_addr);
            key ^= static_cast<u64>(size) * 0x9e3779b97f4a7c15ULL;
            key ^= key >> 33;
            key *= 0xff51afd7ed558ccdULL;
            key ^= key >> 33;
            const size_t set_idx = key & (kSets - 1);
            auto& set = cache.sets[set_idx];
            const u64 tick = scheduler.CurrentTick();
            u64 mem_key = skipcache.Gens().mem_gen.load(std::memory_order_acquire);
            bool mem_key_ok = true;
            if (mirror_mode_) {
                const auto sum = memory_tracker->Sum256ForRange(device_addr, size);
                mem_key = sum.sum;
                mem_key_ok = sum.ok;
            }

            ++stream_copy_probes_;
            StreamCopyCacheEntry* hit = nullptr;
            if (!mem_key_ok) {
                // An uncertifiable range neither hits nor records.
            } else if (set[0].addr == device_addr && set[0].size == size && set[0].tick == tick &&
                       set[0].mem_key == mem_key) {
                hit = &set[0];
            } else if (set[1].addr == device_addr && set[1].size == size && set[1].tick == tick &&
                       set[1].mem_key == mem_key) {
                hit = &set[1];
            }
            if (hit) {
                cache.lru[set_idx] = static_cast<u8>(hit == &set[0] ? 1u : 0u);
                if (hit->gpu_gen == gpu_dirty_generation_) {
                    ++stream_copy_hits_;
                    MirrorOracleProbe(device_addr, size, true, false);
                    return {&stream_buffer, hit->offset};
                }
                if (gpu_modified ? !*gpu_modified : !IsRegionGpuModified(device_addr, size)) {
                    hit->gpu_gen = gpu_dirty_generation_;
                    ++stream_copy_hits_;
                    MirrorOracleProbe(device_addr, size, true, false);
                    return {&stream_buffer, hit->offset};
                }
                hit->addr = 0; // went GPU-dirty: no longer stream-eligible
                MirrorOracleProbe(device_addr, size, true, true);
            } else if (gpu_modified ? !*gpu_modified : !IsRegionGpuModified(device_addr, size)) {
                MirrorOracleProbe(device_addr, size, false, false);
                const u64 offset =
                    stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
                if (mem_key_ok) {
                    const u8 victim = cache.lru[set_idx] & 1u;
                    set[victim] = StreamCopyCacheEntry{
                        .addr = device_addr,
                        .tick = tick,
                        .mem_key = mem_key,
                        .gpu_gen = gpu_dirty_generation_,
                        .offset = static_cast<u32>(offset),
                        .size = size,
                    };
                    cache.lru[set_idx] = static_cast<u8>(victim ^ 1u);
                }
                return {&stream_buffer, static_cast<u32>(offset)};
            }
            // GPU-modified: fall through to the slot path below.
        } else if (gpu_modified ? !*gpu_modified : !IsRegionGpuModified(device_addr, size)) {
            const u64 offset =
                stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
            return {&stream_buffer, offset};
        }
    }
    if (IsBufferInvalid(buffer_id)) {
        buffer_id = FindBuffer(device_addr, size);
    }
    Buffer& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        // Bump the GPU-clean epoch only on new coverage; steady-state
        // re-writes of the same ranges skip both the bump and the no-op
        // interval merge.
        if (!gpu_modified_ranges.Contains(device_addr, size)) {
            ++gpu_dirty_generation_;
            gpu_modified_ranges.Add(device_addr, size);
        }
    }
    return {&buffer, buffer.Offset(device_addr)};
}

std::pair<Buffer*, u32> BufferCache::ObtainBufferForImage(VAddr gpu_addr, u32 size) {
    // Check if any buffer contains the full requested range.
    const BufferId buffer_id = page_table[gpu_addr >> CACHING_PAGEBITS].buffer_id;
    if (buffer_id) {
        if (Buffer& buffer = slot_buffers[buffer_id]; buffer.IsInBounds(gpu_addr, size)) {
            SynchronizeBuffer(buffer, gpu_addr, size, false, false);
            return {&buffer, buffer.Offset(gpu_addr)};
        }
    }
    // If some buffer within was GPU modified create a full buffer to avoid losing GPU data.
    if (IsRegionGpuModified(gpu_addr, size)) {
        return ObtainBuffer(gpu_addr, size, false, false);
    }
    // In all other cases, just do a CPU copy to the staging buffer.
    const auto [data, offset] = staging_buffer.Map(size, instance.StorageMinAlignment());
    memory->CopySparseMemory(gpu_addr, data, size);
    staging_buffer.Commit();
    return {&staging_buffer, offset};
}

bool BufferCache::IsRegionRegistered(VAddr addr, size_t size) {
    // Check if we are missing some edge case here
    return buffer_ranges.Intersects(addr, size);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

BufferId BufferCache::FindBuffer(VAddr device_addr, u32 size) {
    ASSERT(device_addr != 0);
    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page].buffer_id;
    if (!buffer_id) {
        return CreateBuffer(device_addr, size);
    }
    const Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        return buffer_id;
    }
    return CreateBuffer(device_addr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(VAddr device_addr, u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    VAddr begin = device_addr;
    VAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    const auto expand_begin = [&](VAddr add_value) {
        static constexpr VAddr min_page = CACHING_PAGESIZE + DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    const auto expand_end = [&](VAddr add_value) {
        static constexpr VAddr max_page = 1ULL << MemoryTracker::MAX_CPU_PAGE_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.is_picked) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.is_picked = true;
        const VAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const VAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_end(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

void BufferCache::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                              bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    const vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = dst_base_offset,
        .size = overlap.SizeBytes(),
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre_barriers{};
    if (auto src_barrier = overlap.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                              vk::PipelineStageFlagBits2::eTransfer)) {
        pre_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier =
            new_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eTransfer, dst_base_offset)) {
        pre_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    cmdbuf.copyBuffer(overlap.Handle(), new_buffer.Handle(), copy);

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> post_barriers{};
    if (auto src_barrier =
            overlap.GetBarrier(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                               vk::PipelineStageFlagBits2::eAllCommands)) {
        post_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier = new_buffer.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands, dst_base_offset)) {
        post_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
        .pBufferMemoryBarriers = post_barriers.data(),
    });
    DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(VAddr device_addr, u32 wanted_size) {
    const VAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, overlap.begin,
                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
    auto& new_buffer = slot_buffers[new_buffer_id];
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    Register(new_buffer_id);
    return new_buffer_id;
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager.ProcessFaultBuffer();
}

void BufferCache::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

void BufferCache::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    const VAddr device_addr_begin = buffer.CpuAddr();
    const VAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    const u64 size_pages = page_end - page_begin;
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page].buffer_id = buffer_id;
        } else {
            page_table[page].buffer_id = BufferId{};
        }
    }
    if constexpr (insert) {
        total_used_memory += Common::AlignUp(size, CACHING_PAGESIZE);
        buffer.SetLRUId(lru_cache.Insert(buffer_id, gc_tick));
        boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
        bda_addrs.reserve(size_pages);
        for (u64 i = 0; i < size_pages; ++i) {
            vk::DeviceAddress addr = buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS);
            bda_addrs.push_back(addr);
        }
        WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                        bda_addrs.data(), bda_addrs.size() * sizeof(vk::DeviceAddress));
        buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
    } else {
        total_used_memory -= Common::AlignUp(size, CACHING_PAGESIZE);
        lru_cache.Free(buffer.LRUId());
        const u64 offset = bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
        bda_pagetable_buffer.Fill(offset, size_pages * sizeof(vk::DeviceAddress), 0);
        buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
    }
}

// Guest upload sources are written by game threads on other cores and read
// exactly once by the staging copy, so their lines are cold. Requesting the
// head of an island early overlaps the DRAM latency with the tracker walk,
// staging map, and barrier setup. Prefetch never faults, so watched or
// unmapped pages in sparse ranges are safe to request.
static void PrefetchGuestSource(VAddr device_addr, u64 size) {
    constexpr u64 prefetch_bytes = 1024;
    for (u64 i = 0; i < std::min<u64>(size, prefetch_bytes); i += 64) {
        __builtin_prefetch(reinterpret_cast<const void*>(device_addr + i), 0, 3);
    }
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                                    bool is_texel_buffer) {
    // Read-only binds dominate and almost never have anything to upload, but
    // proving it walks every page of the range under the tracker lock. While
    // the host-memory generation is unchanged the guest bytes still equal the
    // device-buffer bytes for a range that had nothing to upload, so eliding
    // the walk and upload is byte-identical. A query contained in a recorded
    // clean range hits, since a clean range has no dirty subrange. Written
    // binds are excluded: their walk also marks the range GPU modified.
    auto& skipcache = VideoCore::Skipcache::Framework::Instance();
    const bool memo_eligible = skipcache.Active() && !is_written && !is_texel_buffer;
    // Under the mirror mode the memo keys on the recorded range's word-epoch
    // sum instead of the global generation, so unrelated faults elsewhere no
    // longer invalidate it. The sum is validated over the recorded range, not
    // the queried subrange, since the stored sum covers exactly that span.
    const bool epoch_keyed = memo_eligible && mirror_mode_;
    const u64 memo_gen =
        memo_eligible ? skipcache.Gens().mem_gen.load(std::memory_order_acquire) : 0;
    if (memo_eligible) {
        for (const auto& noop : buffer.sync_noop) {
            if (noop.size == 0 || device_addr < noop.addr ||
                device_addr + size > noop.addr + noop.size) {
                continue;
            }
            if (epoch_keyed) {
                const auto sum = memory_tracker->Sum256ForRange(noop.addr, noop.size);
                if (sum.ok && noop.mem_key == sum.sum) {
                    ++mirror_oracle_.tierA_hits;
                    return false;
                }
            } else if (noop.mem_key == memo_gen) {
                return false;
            }
        }
    }

    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes = 0;
    VAddr buffer_start = buffer.CpuAddr();
    vk::Buffer src_buffer = VK_NULL_HANDLE;
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 device_addr_out, u64 range_size) {
            PrefetchGuestSource(device_addr_out, range_size);
            copies.emplace_back(total_size_bytes, device_addr_out - buffer_start, range_size);
            total_size_bytes += range_size;
        },
        [&] { src_buffer = UploadCopies(buffer, copies, total_size_bytes); });

    if (src_buffer) [[unlikely]] {
        EmitBufferUpload(buffer, src_buffer, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeBufferFromImage(buffer, device_addr, size);
    }
    if (mirror_mode_ && memo_eligible) {
        ++mirror_oracle_.tierA_elig_walks;
    }
    if (mirror_mode_ && memo_eligible && !src_buffer) {
        ++mirror_oracle_.tierA_walks;
        constexpr u64 word_size = u64{1} << RegionManager::EPOCH_WORD_BITS;
        if (size <= 64 * word_size) {
            ++mirror_oracle_.tierA_span_le64;
        }
    }
    if (memo_eligible && !src_buffer) {
        // Nothing was uploaded: record the exact walked range so the next
        // query contained in it skips the page walk entirely. Prefer to
        // replace a same-key entry the new range covers, then a stale entry,
        // then the round-robin victim.
        u64 record_key = memo_gen;
        bool record_valid = true;
        if (epoch_keyed) {
            const auto sum = memory_tracker->Sum256ForRange(device_addr, size);
            record_key = sum.sum;
            record_valid = sum.ok;
        }
        if (record_valid) {
            size_t victim = buffer.sync_noop.size();
            for (size_t i = 0; i < buffer.sync_noop.size(); ++i) {
                const auto& noop = buffer.sync_noop[i];
                if (noop.mem_key != record_key || noop.size == 0) {
                    victim = i;
                    continue;
                }
                if (noop.addr >= device_addr && noop.addr + noop.size <= device_addr + size) {
                    victim = i;
                    break;
                }
            }
            if (victim == buffer.sync_noop.size()) {
                victim = buffer.sync_noop_next;
            }
            buffer.sync_noop[victim] = {.addr = device_addr, .size = size, .mem_key = record_key};
            buffer.sync_noop_next = static_cast<u32>((victim + 1) % buffer.sync_noop.size());
        }
    }
    return false;
}

void BufferCache::EmitBufferUpload(Buffer& buffer, vk::Buffer src_buffer,
                                   std::span<const vk::BufferCopy> copies) {
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
                         vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = 0,
        .size = buffer.SizeBytes(),
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = 0,
        .size = buffer.SizeBytes(),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(src_buffer, buffer.buffer, copies);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
    TouchBuffer(buffer);
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     size_t total_size_bytes) {
    if (copies.empty()) [[likely]] {
        return VK_NULL_HANDLE;
    }
    const auto [staging, offset] = staging_buffer.Map(total_size_bytes);
    if (staging == nullptr) [[unlikely]] {
        return UploadCopiesFallback(buffer, copies, total_size_bytes);
    }
    for (size_t i = 0; i < copies.size(); ++i) {
        auto& copy = copies[i];
        // Requesting the next island's source overlaps its misses with this
        // island's copy. Walk-time prefetches of early islands may already
        // be evicted on large multi-island syncs, so this repeat is kept.
        if (i + 1 < copies.size()) {
            PrefetchGuestSource(buffer.CpuAddr() + copies[i + 1].dstOffset, copies[i + 1].size);
        }
        u8* const src_pointer = staging + copy.srcOffset;
        const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
        memory->CopySparseMemory(device_addr, src_pointer, copy.size);
        // Apply the staging offset
        copy.srcOffset += offset;
    }
    staging_buffer.Commit();
    return staging_buffer.Handle();
}

vk::Buffer BufferCache::UploadCopiesFallback(Buffer& buffer, std::span<const vk::BufferCopy> copies,
                                             size_t total_size_bytes) {
    // For large one time transfers use a temporary host buffer.
    auto temp_buffer =
        std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                 vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes);
    const vk::Buffer src_buffer = temp_buffer->Handle();
    u8* const staging = temp_buffer->mapped_data.data();
    for (size_t i = 0; i < copies.size(); ++i) {
        const auto& copy = copies[i];
        if (i + 1 < copies.size()) {
            PrefetchGuestSource(buffer.CpuAddr() + copies[i + 1].dstOffset, copies[i + 1].size);
        }
        u8* const src_pointer = staging + copy.srcOffset;
        const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
        memory->CopySparseMemory(device_addr, src_pointer, copy.size);
    }
    scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable { buffer.reset(); });
    return src_buffer;
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            buffer.Fill(buffer.Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u32 buf_offset = buffer.Offset(image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    u32 copy_size = 0;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (buf_offset + mip_info.offset + mip_info.size > buffer.SizeBytes()) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
        copy_size += mip_info.size;
    }
    if (copy_size == 0) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, buffer.Handle(), buf_offset, copy_size);
    return true;
}

void BufferCache::SynchronizeBuffersInRange(VAddr device_addr, u64 size) {
    const VAddr device_addr_end = device_addr + size;
    ++dmasync_calls_;
    ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
        RENDERER_TRACE;
        VAddr start = std::max(buffer.CpuAddr(), device_addr);
        VAddr end = std::min(buffer.CpuAddr() + buffer.SizeBytes(), device_addr_end);
        u32 size = static_cast<u32>(end - start);
        ++dmasync_buffers_;
        dmasync_bytes_ += size;
        dmasync_max_bytes_ = std::max<u64>(dmasync_max_bytes_, size);
        SynchronizeBuffer(buffer, start, size, false, false);
    });
}

void BufferCache::WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes) {
    vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = buffer.Offset(address),
        .size = num_bytes,
    };
    vk::Buffer src_buffer = staging_buffer.Handle();
    if (num_bytes < StagingBufferSize) {
        const auto [staging, offset] = staging_buffer.Map(num_bytes);
        std::memcpy(staging, value, num_bytes);
        copy.srcOffset = offset;
        staging_buffer.Commit();
    } else {
        // For large one time transfers use a temporary host buffer.
        // RenderDoc can lag quite a bit if the stream buffer is too large.
        Buffer temp_buffer{
            instance, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
            num_bytes};
        src_buffer = temp_buffer.Handle();
        u8* const staging = temp_buffer.mapped_data.data();
        std::memcpy(staging, value, num_bytes);
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable {});
    }
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(src_buffer, buffer.Handle(), copy);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

void BufferCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    const bool aggressive = total_used_memory >= critical_gc_memory;
    const u64 ticks_to_destroy = std::min<u64>(aggressive ? 80 : 160, gc_tick);
    int max_deletions = aggressive ? 64 : 32;
    const auto clean_up = [&](BufferId buffer_id) {
        if (max_deletions == 0) {
            return;
        }
        --max_deletions;
        Buffer& buffer = slot_buffers[buffer_id];
        // InvalidateMemory(buffer.CpuAddr(), buffer.SizeBytes());
        DownloadBufferMemory<true>(buffer, buffer.CpuAddr(), buffer.SizeBytes());
        // Nothing invokes clean_up today. Wiring it up requires a skip cache
        // mem_gen bump alongside this CPU-dirty marking, which the sync-noop
        // and bind memos rely on.
        memory_tracker->MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
        DeleteBuffer(buffer_id);
    };
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

void BufferCache::DeleteBuffer(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    Unregister(buffer_id);
    scheduler.DeferOperation([this, buffer_id] { slot_buffers.erase(buffer_id); });
    buffer.is_deleted = true;
}

} // namespace VideoCore
