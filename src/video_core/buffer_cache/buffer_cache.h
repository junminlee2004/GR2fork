// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <vector>

#include <boost/container/small_vector.hpp>
#include "common/assert.h"
#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/multi_level_page_table.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
}

namespace VideoCore {

using BufferId = Common::SlotId;

class TextureCache;
class PageManager;

class BufferCache {
public:
    static constexpr u32 CACHING_PAGEBITS = 14;
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;
    static constexpr u64 DEVICE_PAGESIZE = 16_KB;
    static constexpr u64 CACHING_NUMPAGES = u64{1} << (40 - CACHING_PAGEBITS);
    static constexpr u64 BDA_PAGETABLE_SIZE = CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

    // Default values for garbage collection
    static constexpr s64 DEFAULT_TRIGGER_GC_MEMORY = 1_GB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 2_GB;
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;

    struct PageData {
        BufferId buffer_id{};
    };

    struct Traits {
        using Entry = PageData;
        static constexpr size_t AddressSpaceBits = 40;
        static constexpr size_t FirstLevelBits = 16;
        static constexpr size_t PageBits = CACHING_PAGEBITS;
    };
    using PageTable = MultiLevelPageTable<Traits>;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        VAddr begin;
        VAddr end;
        bool has_stream_leap = false;
    };

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         AmdGpu::Liverpool* liverpool, TextureCache& texture_cache,
                         PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return &bda_pagetable_buffer;
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager.GetFaultBuffer();
    }

    /// Retrieves the buffer with the specified id.
    [[nodiscard]] Buffer& GetBuffer(BufferId id) {
        return slot_buffers[id];
    }

    /// Retrieves a utility buffer optimized for specified memory usage.
    StreamBuffer& GetUtilityBuffer(MemoryUsage usage) noexcept {
        if (usage == MemoryUsage::Stream) {
            return stream_buffer;
        } else if (usage == MemoryUsage::Download) {
            return download_buffer;
        } else if (usage == MemoryUsage::DeviceLocal) {
            return device_buffer;
        } else {
            return staging_buffer;
        }
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false);

    struct OffloadStats {
        u64 jobs;
        u64 vetoes;
        u64 fallbacks;
        u64 wait_ns;
    };

    /// Snapshot and reset the offloaded-readback counters (for periodic logs).
    OffloadStats DrainOffloadStats() {
        return {offload_jobs_.exchange(0, std::memory_order_relaxed),
                offload_vetoes_.exchange(0, std::memory_order_relaxed),
                offload_fallbacks_.exchange(0, std::memory_order_relaxed),
                offload_wait_ns_.exchange(0, std::memory_order_relaxed)};
    }

    struct StreamCopyStats {
        u64 hits;
        u64 probes;
    };

    /// Snapshot and reset the stream copy cache counters (for periodic logs).
    StreamCopyStats DrainStreamCopyStats() {
        const StreamCopyStats stats{stream_copy_hits_, stream_copy_probes_};
        stream_copy_hits_ = 0;
        stream_copy_probes_ = 0;
        return stats;
    }

    /// Binds host vertex buffers for the current draw.
    void BindVertexBuffers(const Vulkan::GraphicsPipeline& pipeline,
                           boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Bind host index buffer for the current draw.
    void BindIndexBuffer(u32 index_offset,
                         boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Writes a value to GPU buffer. (uses command buffer to temporarily store the data)
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);

    /// Performs buffer to buffer data copy on the GPU.
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);

    /// Obtains a buffer for the specified region.
    ///
    /// `gpu_modified` lets a caller that already knows whether the range is GPU
    /// modified hand that answer in rather than have it recomputed. Resolving it
    /// is a chain of dependent loads - a 2 MiB pointer table, then the region
    /// object, then its bitmap - which the out-of-order engine cannot overlap
    /// and the prefetcher cannot predict, so paying for it twice on the same
    /// range is worth avoiding. Only valid for read-only queries: a written bind
    /// marks the range GPU modified as a side effect, so a value sampled
    /// beforehand would be stale by the time it is used.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(VAddr gpu_addr, u32 size, bool is_written,
                                                       bool is_texel_buffer = false,
                                                       BufferId buffer_id = {},
                                                       std::optional<bool> gpu_modified = {});

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBufferForImage(VAddr gpu_addr, u32 size);

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Return buffer id for the specified region
    BufferId FindBuffer(VAddr device_addr, u32 size);

    /// Processes the fault buffer.
    void ProcessFaultBuffer();

    /// Synchronizes all buffers in the specified range.
    void SynchronizeBuffersInRange(VAddr device_addr, u64 size);

    /// Synchronizes all buffers neede for DMA.
    void SynchronizeDmaBuffers();

    /// Runs the garbage collector.
    void RunGarbageCollector();

private:
    template <typename Func>
    void ForEachBufferInRange(VAddr device_addr, u64 size, Func&& func) {
        buffer_ranges.ForEachInRange(device_addr, size,
                                     [&](u64 page_start, u64 page_end, BufferId id) {
                                         Buffer& buffer = slot_buffers[id];
                                         func(id, buffer);
                                     });
    }

    inline bool IsBufferInvalid(BufferId buffer_id) const {
        return !buffer_id || slot_buffers[buffer_id].is_deleted;
    }

    template <bool async>
    void DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size);

    /**
     * One offloaded fault readback in flight. Filled on the GPU command thread
     * (PrepareFaultDownload), the semaphore wait happens on the faulting guest
     * thread, and the writeback runs on the GPU command thread again
     * (FinishFaultDownload). The SendCommand handshake orders every cross-
     * thread access, so no field needs synchronization of its own.
     */
    struct FaultDownloadJob {
        std::unique_ptr<Buffer> staging; // pool buffer holding the copied data
        boost::container::small_vector<vk::BufferCopy, 4> copies;
        MemoryTracker::GpuSeqSnapshots snapshots;
        VAddr buffer_base = 0;  // guest base of the source buffer at record time
        VAddr window_start = 0; // range whose tracker bits the writeback clears
        u64 window_size = 0;
        u64 wait_tick = 0;
        bool has_download = false;
        bool fully_cleared = false; // FinishFaultDownload verdict
    };

    /// Records download copies for an offloaded fault readback and flushes the
    /// submission. GPU command thread only (reached via SendCommand).
    void PrepareFaultDownload(FaultDownloadJob& job, VAddr device_addr, u64 size, bool is_write);

    /// Writes the downloaded bytes back to guest memory and clears tracker
    /// bits for regions with no newer GPU writes. GPU command thread only.
    void FinishFaultDownload(FaultDownloadJob& job, VAddr device_addr, u64 size, bool is_write);

    /// Takes a staging buffer of at least the given size from the fault pool.
    /// GPU command thread only.
    std::unique_ptr<Buffer> AcquireFaultStaging(u64 size);

    /// Returns a staging buffer to the fault pool. GPU command thread only.
    void ReleaseFaultStaging(std::unique_ptr<Buffer> staging);

    [[nodiscard]] OverlapResult ResolveOverlaps(VAddr device_addr, u32 wanted_size);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    BufferId CreateBuffer(VAddr device_addr, u32 wanted_size);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    bool SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                            size_t total_size_bytes);

    /// Records the barriers and copy commands for a completed upload batch.
    /// Out of line so the no-upload walk in SynchronizeBuffer stays compact.
    SHAD_NO_INLINE void EmitBufferUpload(Buffer& buffer, vk::Buffer src_buffer,
                                         std::span<const vk::BufferCopy> copies);

    /// Large one time transfer path for UploadCopies when the staging ring
    /// cannot hold the batch; copies through a temporary host buffer.
    SHAD_NO_INLINE vk::Buffer UploadCopiesFallback(Buffer& buffer,
                                                   std::span<const vk::BufferCopy> copies,
                                                   size_t total_size_bytes);

    bool SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size);

    void WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes);

    void TouchBuffer(const Buffer& buffer);

    void DeleteBuffer(BufferId buffer_id);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    FaultManager fault_manager;
    std::unique_ptr<MemoryTracker> memory_tracker;

    // Vertex/index bind memos (adaptive skip caches, inline form). Validity
    // requires same submission tick (stream ring offsets are only stable
    // within one command buffer) and an unchanged memory key: the host-memory
    // generation, or the bound range's word-epoch sum under the mirror mode. The clean-gen fields
    // record the gpu_dirty_generation_ at which the memoized range was last proven not GPU
    // modified; zero means unproven.
    u64 vertex_bind_sig_{};
    u64 vertex_input_sig_{};
    u64 vertex_bind_tick_{};
    u64 vertex_input_tick_{};
    u64 vertex_bind_mem_key_{};
    u64 vertex_bind_clean_gpu_gen_{};
    bool vertex_bind_valid_{};
    bool vertex_input_valid_{};
    VAddr index_bind_addr_{};
    u32 index_bind_size_{};
    u32 index_bind_type_{};
    u64 index_bind_tick_{};
    u64 index_bind_mem_key_{};
    u64 index_bind_clean_gpu_gen_{};
    bool index_bind_valid_{};
    StreamBuffer staging_buffer;
    StreamBuffer stream_buffer;
    StreamBuffer download_buffer;
    StreamBuffer device_buffer;
    Buffer gds_buffer;
    Buffer bda_pagetable_buffer;
    Common::SlotVector<Buffer> slot_buffers;
    u64 total_used_memory = 0;
    u64 trigger_gc_memory = 0;
    u64 critical_gc_memory = 0;
    u64 gc_tick = 0;
    Common::LeastRecentlyUsedCache<BufferId, u64> lru_cache;
    RangeSet gpu_modified_ranges;
    // Bumped only on clean->dirty coverage transitions; an entry stamped with
    // the current value is proven GPU-clean without walking the range set.
    // Overlapping in-flight fault downloads can leave the range set covering
    // clear-bit pages (veto re-Add in FinishFaultDownload); every consumer of
    // this generation shares that window and is off when skip caches are off.
    u64 gpu_dirty_generation_{1};
    SplitRangeMap<BufferId> buffer_ranges;
    PageTable page_table;
    // Staging pool for offloaded fault readbacks. GPU command thread only.
    std::vector<std::unique_ptr<Buffer>> fault_staging_pool_;
    // Offload counters; wait_ns is written by faulting guest threads.
    std::atomic<u64> offload_jobs_{};
    std::atomic<u64> offload_vetoes_{};
    std::atomic<u64> offload_fallbacks_{};
    std::atomic<u64> offload_wait_ns_{};
    // Stream copy cache counters; hits count probes that return a cached
    // offset. The probes and the telemetry drain both run on the GPU command
    // thread, so plain counters suffice.
    u64 stream_copy_hits_{};
    u64 stream_copy_probes_{};

public:
    /// Emits and resets the phase-1 stream mirror telemetry. Logs nothing when
    /// the mirror mode is off.
    void EmitMirrorTelemetry();

private:
    static void MirrorProtectThunk(void* user, VAddr addr, u64 size, bool write_granted,
                                   bool tracker_origin);
    static void MirrorBackingThunk(void* user, VAddr addr, u64 size);
    void MirrorOracleProbe(VAddr device_addr, u32 size, bool tick_hit, bool gpu_dirty);

    // Phase-1 observe-only stream mirror instrumentation. The sink counters
    // are written from guest threads and the fault path; the oracle counters
    // are GPU command thread only.
    struct MirrorSinkCounters {
        std::atomic<u64> bump_tracker{};
        std::atomic<u64> bump_guestapi{};
        std::atomic<u64> bump_backing{};
        std::atomic<u64> poisoned{};
    };
    struct MirrorOracleCounters {
        u64 elig{};
        u64 elig_bytes{};
        u64 tick_miss{};
        u64 clean{};
        u64 clean_tm{};
        u64 clean_bytes{};
        u64 clean_tm_bytes{};
        u64 hit_clean{};
        u64 hit_clean_tm{};
        u64 div{};
        u64 alias256{};
        u64 alias64{};
        u64 changed{};
        u64 dirty_stable{};
        u64 dirty_stable_chg{};
        u64 dirty_moved{};
        u64 gpu_dirty{};
        u64 cpu_dirty{};
        u64 cpu_dirty_bytes{};
        u64 cold{};
        u64 evict{};
        u64 ws_keys{};
        u64 ws_bytes{};
        u64 sum_unresolved{};
        u64 tierA_walks{};
        u64 tierA_hits{};
        u64 tierA_elig_walks{};
        u64 tierA_span_le64{};
    };
    MirrorSinkCounters mirror_sink_;
    MirrorOracleCounters mirror_oracle_;
    bool mirror_mode_{};

    // Tier B mirror arena: a persistent host-cached device buffer serving
    // recurring clean stream sources across frames. Entries certify with the
    // same word-epoch sums the tier A memos use; published slot bytes are
    // immutable and return to their free list only after the last referencing
    // submission's fence.
    struct alignas(64) MirrorEntry {
        VAddr addr;    // 0 = invalid
        u64 epoch_sum; // certificate recorded at adoption
        u64 gpu_gen;
        u64 last_use_tick;
        u32 arena_offset;
        u16 size;
        u16 size_class;
    };
    static constexpr size_t MIRROR_ENTRIES = 32768;
    static constexpr u32 MIRROR_ARENA_SIZE = 64 * 1024 * 1024;
    static constexpr u32 MIRROR_MIN_CLASS = 256;
    static constexpr u32 MIRROR_NUM_CLASSES = 7; // 256B..16KB

    std::pair<Buffer*, u32> MirrorServe(MirrorEntry& entry);
    void MirrorRetireSlot(MirrorEntry& entry);
    bool MirrorAllocSlot(u32 size, u32& out_offset, u16& out_class);
    void MirrorEvictSome();
    std::optional<std::pair<Buffer*, u32>> MirrorProbe(VAddr device_addr, u32 size, u64 mem_key,
                                                       bool mem_key_ok,
                                                       std::optional<bool> gpu_modified);
    std::optional<std::pair<Buffer*, u32>> MirrorTryAdopt(VAddr device_addr, u32 size, u64 mem_key,
                                                          bool mem_key_ok, bool serve);
    static void MirrorDropThunk(void* user);

public:
    /// Drops every mirror arena entry; live slots recycle after their fences.
    void MirrorDropAll();

private:
    struct MirrorArenaCounters {
        u64 hits{};
        u64 served_bytes{};
        u64 verify_clean{};
        u64 verify_div{};
        u64 adopts{};
        u64 adopt_bytes{};
        u64 adopt_torn{};
        u64 alloc_fail{};
        u64 evicted{};
        u64 drops{};
        u64 gpu_restamp{};
    };
    std::unique_ptr<std::array<MirrorEntry, MIRROR_ENTRIES>> mirror_map_;
    std::unique_ptr<Buffer> mirror_arena_;
    std::array<std::vector<std::pair<u32, u64>>, MIRROR_NUM_CLASSES> mirror_retired_;
    MirrorArenaCounters mirror_b_;
    u32 mirror_bump_{};
    u32 mirror_evict_cursor_{};
    u32 mirror_populate_tick_{};
    bool mirror_arena_mode_{};
};

} // namespace VideoCore
