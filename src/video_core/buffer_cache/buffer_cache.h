// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
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
#include "video_core/skipcache/skipcache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class TransferQueue;
class GraphicsPipeline;
} // namespace Vulkan

namespace VideoCore {

using BufferId = Common::SlotId;

class TextureCache;
class PageManager;

/// Where a pending read-watcher arm was drained, for telemetry only.
enum class ReadArmSite : u8 { Run, Submit, Fence, Wait, Idle, Count };

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
    /// readback_linear_images_lazy: hands a linear image's guest range to the tracker as a GPU
    /// write on a buffer that covers it; nothing is copied until a CPU read faults.
    void MarkRangeForLazyReadback(VAddr addr, u32 size);
    /// Fills the buffer covering a lazily tracked image with the image's content ahead of a
    /// download; the image must still start at addr.
    void SynchronizeLazyImage(VAddr addr, u32 size);
    /// Pulls every lazily tracked image inside a fault's readback window into its buffer.
    void SyncLazyReadbackImagesForFault(VAddr device_addr, u64 size);

    struct OffloadStats {
        u64 jobs;
        u64 vetoes;
        u64 fallbacks;
        u64 wait_ns;
        // Fault windows that found nothing to download (counted regardless of settings).
        u64 empty;
        // readback_offload: copies taken by the second queue, the fallbacks
        // by reason, and the faulting threads' wait on that queue.
        u64 q2_copies;
        u64 q2_open;
        u64 q2_unknown;
        u64 q2_dma;
        u64 q2_wait_ns;
    };

    /// Snapshot and reset the offloaded-readback counters (for periodic logs).
    OffloadStats DrainOffloadStats() {
        return {offload_jobs_.exchange(0, std::memory_order_relaxed),
                offload_vetoes_.exchange(0, std::memory_order_relaxed),
                offload_fallbacks_.exchange(0, std::memory_order_relaxed),
                offload_wait_ns_.exchange(0, std::memory_order_relaxed),
                join_empty_.exchange(0, std::memory_order_relaxed),
                q2_copies_.exchange(0, std::memory_order_relaxed),
                q2_open_.exchange(0, std::memory_order_relaxed),
                q2_unknown_.exchange(0, std::memory_order_relaxed),
                q2_dma_.exchange(0, std::memory_order_relaxed),
                q2_wait_ns_.exchange(0, std::memory_order_relaxed)};
    }

    /// readback_offload: a device-address shader is being recorded, which
    /// can write any buffer; readbacks treat the open batch as its writer.
    void NoteDmaWrite();

    /// readback_offload: whether the draw being recorded wrote a buffer a
    /// readback has already read. Cleared by the call.
    bool TakeProneWrite() {
        const bool pending = prone_write_pending_;
        prone_write_pending_ = false;
        return pending;
    }

    struct StreamCopyStats {
        u64 hits;
        u64 probes;
        u64 fast;
        u64 idxfast;
        // Memo hits whose clean proof was stale and walked the GPU bits.
        u64 stream_genwalk;
        u64 vertex_genwalk;
        u64 index_genwalk;
    };

    /// Snapshot and reset the stream copy cache counters (for periodic logs).
    StreamCopyStats DrainStreamCopyStats() {
        return {std::exchange(stream_copy_hits_, 0), std::exchange(stream_copy_probes_, 0),
                std::exchange(stream_copy_fast_, 0), std::exchange(index_bind_fast_, 0),
                std::exchange(stream_genwalk_, 0),   std::exchange(vertex_genwalk_, 0),
                std::exchange(index_genwalk_, 0)};
    }

    // DMA-draw full-overlap sync: range syncs run, buffers walked, bytes covered.
    struct DmaSyncStats {
        u64 calls;
        u64 buffers;
        u64 bytes;
        u64 max_bytes;
    };
    DmaSyncStats DrainDmaSyncStats() {
        return {std::exchange(dmasync_calls_, 0), std::exchange(dmasync_buffers_, 0),
                std::exchange(dmasync_bytes_, 0), std::exchange(dmasync_max_bytes_, 0)};
    }

    // Byte split of the staging upload path by bind writability: the
    // read-only share is what the upload drain setting can move to the lane.
    struct UploadCopyStats {
        u64 ro_calls;
        u64 ro_bytes;
        u64 w_calls;
        u64 w_bytes;
    };
    UploadCopyStats DrainUploadCopyStats() {
        return {std::exchange(upload_ro_calls_, 0), std::exchange(upload_ro_bytes_, 0),
                std::exchange(upload_w_calls_, 0), std::exchange(upload_w_bytes_, 0)};
    }

    struct TexelNoopStats {
        u64 hits;
        u64 probes;
    };
    TexelNoopStats DrainTexelNoopStats() {
        return {std::exchange(texel_noop_hits_, 0), std::exchange(texel_noop_probes_, 0)};
    }

    MemoryTracker::FastPathDrain DrainTrackerFastStats() {
        return memory_tracker->DrainFastPathStats();
    }

    /// Binds host vertex buffers for the current draw.
    void BindVertexBuffers(const Vulkan::GraphicsPipeline& pipeline,
                           boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers);

    /// Bind host index buffer for the current draw. Returns the draw's
    /// firstIndex: 0 for an exact bind, offset / index size for a whole bind;
    /// a caller whose firstIndex lives in indirect args passes allow_whole false.
    u32 BindIndexBuffer(u32 index_offset,
                        boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers,
                        bool allow_whole);

    /// Writes a value to GPU buffer. (uses command buffer to temporarily store the data)
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);

    /// Performs buffer to buffer data copy on the GPU.
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);

    /// Obtains a buffer for the specified region.
    ///
    /// `gpu_modified` hands in an answer the caller already resolved, so the
    /// dependent-load chain (page table -> region -> bitmap) is not walked
    /// twice. Read-only queries only: a written bind marks the range GPU
    /// modified as a side effect, so a value sampled beforehand is stale.
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

    /// readback_writeback_share: the owner of an in-flight readback publishes
    /// its downloaded islands so the threads parked behind it copy a share.
    /// pieces, download, total and total_bytes are filled by the owner before
    /// ready and read by every copier; next hands islands out, done counts the
    /// finished ones for the owner's tail wait; tick is what a joiner waits on;
    /// coherent lets the priority helper claim before the owner's invalidate.
    /// download is dereferenced only after a successful claim.
    struct WriteBackShare {
        struct Piece {
            VAddr dst;
            u64 src_off;
            u32 size;
        };
        std::vector<Piece> pieces;
        const u8* download = nullptr;
        u64 tick = 0;
        // readback_offload: the copy retires on the second queue at this
        // tick; the master tick above is then the writer's batch.
        u64 copy_queue_tick = 0;
        bool coherent = false;
        u64 total_bytes = 0;
        u32 total = 0;
        alignas(64) std::atomic<u32> next{0};
        std::atomic<u32> done{0};
        std::atomic<bool> ready{false};
    };

    /**
     * One offloaded fault readback in flight. Filled on the GPU command thread
     * (PrepareFaultDownload), the semaphore wait happens on the faulting guest
     * thread, and the verdict and unmark run on the GPU command thread again
     * (FinishFaultDownload). The write-back runs there too, or - behind
     * readback_writeback_offload - on the thread that waited out the fence,
     * before the second hop. The SendCommand handshake orders every cross-
     * thread access, so no field needs synchronization of its own.
     */
    struct FaultDownloadJob {
        std::unique_ptr<Buffer> staging; // pool buffer holding the copied data
        boost::container::small_vector<vk::BufferCopy, 4> copies;
        MemoryTracker::GpuSeqSnapshots snapshots;
        VAddr buffer_base = 0; // guest base of the source buffer at record time
        u64 wait_tick = 0;
        // readback_offload: the copy retires on the second queue's own
        // timeline at this tick; wait_tick then names the writer's batch.
        u64 copy_queue_tick = 0;
        bool on_copy_queue = false;
        u64 inflight_id = 0;     // registry entry owning the copied islands
        u64 written_islands = 0; // filled by the offloaded write-back
        u64 written_bytes = 0;
        u64 copy_ns = 0;
        // 0 guest thread, 2 GPU command thread; 1 (priority thread) is never
        // assigned - the priority helper is counted by prio_helped_/prio_bytes_.
        u8 copier = 0;
        bool copied = false;
        bool has_download = false;
        bool fully_cleared = false; // FinishFaultDownload verdict
        // readback_writeback_share: this job's islands, and the in-flight
        // owners of the faulted range an empty job may help instead of sleeping.
        std::shared_ptr<WriteBackShare> share;
        boost::container::small_vector<std::shared_ptr<WriteBackShare>, 4> joins;
        u64 join_tick = 0;            // newest fence among the joined owners
        u64 join_copy_queue_tick = 0; // newest copy-queue tick among them
    };

    /// The [start, end) of the buffer to read back for a fault on
    /// [device_addr, device_addr + size), widened and clamped to the buffer.
    std::pair<VAddr, VAddr> ComputeReadbackWindow(const Buffer& buffer, VAddr device_addr,
                                                  u64 size) const;

    /// Records download copies for an offloaded fault readback and flushes the
    /// submission. GPU command thread only (reached via SendCommand).
    void PrepareFaultDownload(FaultDownloadJob& job, VAddr device_addr, u64 size, bool is_write);

    /// Writes the downloaded bytes back to guest memory and clears tracker
    /// bits for regions with no newer GPU writes. GPU command thread only.
    void FinishFaultDownload(FaultDownloadJob& job, VAddr device_addr, u64 size, bool is_write);

    /// Copies every downloaded island into guest memory through the backing
    /// view. Any thread; the caller has waited out job.wait_tick. Records the
    /// copying thread in job.copier: 0 guest, 2 GPU command thread.
    void WriteBackFaultDownload(FaultDownloadJob& job);

    /// readback_offload: nothing to download for this window - waits out the owners'
    /// write-back by helping, waiting a notify or sleeping, so the faulting thread does
    /// not re-fault into the GPU command thread. Runs on whichever thread faulted.
    void DampAfterEmptyDownload(FaultDownloadJob& job, VAddr device_addr, u64 size);

    /// Copies islands of another job's share until its cursor is exhausted;
    /// false when none was left. Any thread, once the share is ready. The
    /// bytes it copied are added to copied_bytes when one is given. With
    /// bail_on_pending, liverpool->HasPendingWork() is polled before every claim
    /// past the first and sets *bailed when it stops the copy; a non-zero
    /// max_bytes stops claiming once that many bytes were copied. Either way a
    /// claimed island is always finished.
    bool HelpWriteBack(WriteBackShare& share, u64* copied_bytes = nullptr,
                       bool bail_on_pending = false, bool* bailed = nullptr, u64 max_bytes = 0,
                       u32* max_island = nullptr);

    /// Priority-ops thread, after the share's fence: copies islands until the
    /// cursor is exhausted; gives up as late when a non-coherent owner has not
    /// published yet.
    void HelpAsPriority(WriteBackShare& share);

    /// Runs the per-island verdict (write sequence test, write-back when the
    /// offload is off, unmark or veto re-add) for one span of a job's copies.
    /// Returns true when at least one island was vetoed. GPU command thread
    /// only; 'download' is null when the bytes are already in guest memory.
    bool FinishIslands(std::span<const vk::BufferCopy> copies,
                       const MemoryTracker::GpuSeqSnapshots& snapshots, VAddr buffer_base,
                       const u8* download, bool copied);

    /// finish_release_faulted_first: settles every island FinishFaultDownload
    /// parked after releasing the faulting thread. GPU command thread only.
    void DrainPendingFinish();

    /// Same, from a thread that may not be the GPU command thread: hops over
    /// when it is not, so the parked state stays GPU-command-thread confined.
    void DrainPendingFinishSynced();

    /// GPU command thread, drained while it has nothing else to do: copies
    /// islands of a share up to a byte cap, yielding at the first island
    /// boundary after a submit or command arrives.
    void HelpAsGpuIdle(WriteBackShare& share);

    using OwnedIslands = boost::container::small_vector<std::pair<VAddr, u32>, 16>;
    /// Islands of in-flight readbacks that overlap [start, end), sorted by
    /// address. GPU command thread only.
    void CollectOwnedIslands(VAddr start, VAddr end, OwnedIslands& out) const;

    /// Records the download copies for the GPU-modified islands of
    /// [start, start + size), folds the pending ranges, subtracts what it took
    /// and reports the in-flight islands it left to their owners in 'owned'.
    /// Copy destinations are packed from zero. Returns the staging bytes.
    template <typename Copies>
    u64 CollectDownloadCopies(Buffer& buffer, VAddr start, u64 size, Copies& copies,
                              OwnedIslands& owned);

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
                           bool is_texel_buffer, bool* new_gpu_pages = nullptr);

    vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies, bool is_written,
                            size_t total_size_bytes);

    /// The slot path of ObtainBuffer: taken when the range cannot be served
    /// from the stream ring. Outlined so the ring path, which is most binds,
    /// does not carry its code through the instruction fetcher.
    SHAD_NO_INLINE std::pair<Buffer*, u32> ObtainBufferSlot(VAddr device_addr, u32 size,
                                                            bool is_written, bool is_texel_buffer,
                                                            BufferId buffer_id);

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

    // Vertex/index bind memos (adaptive skip caches, inline form). A memo is valid only on the
    // same submission tick (stream ring offsets are stable only within one command buffer) with
    // an unchanged memory key: mem_gen, or the bound range's word-epoch sum under mirror mode.
    // The clean-gen fields hold the gpu_dirty_generation_ at which the range was last proven not
    // GPU modified; zero means unproven. In VertexBindEntry the layout words key the vertex
    // input state; base and size add the guest V# contents.
    static constexpr u32 MaxVertexBindings = 32;
    struct VertexBindEntry {
        VAddr base;
        u64 layout;
        u32 size;
    };
    std::array<VertexBindEntry, MaxVertexBindings> vertex_bind_entries_{};
    u32 vertex_bind_count_{};
    const Vulkan::GraphicsPipeline* vertex_bind_pipeline_{};
    u32 vertex_bind_step0_{};
    u32 vertex_bind_step1_{};
    u64 vertex_bind_tick_{};
    u64 vertex_input_tick_{};
    u64 vertex_input_gen_{}; // foreign pipeline gen at the last vertex input emit
    u64 vertex_bind_mem_key_{};
    u64 vertex_bind_clean_gpu_gen_{};
    bool vertex_bind_valid_{};
    bool vertex_input_valid_{};
    VAddr index_bind_addr_{};
    u32 index_bind_size_{};
    u32 index_bind_type_{};
    u64 index_bind_tick_{};
    u64 index_bind_mem_key_{};
    RegionManager* index_bind_region_{}; // certifies the mem key without the walk
    u64 index_bind_clean_gpu_gen_{};
    bool index_bind_valid_{};
    // Last whole index bind (offset 0) on this tick and the firstIndex it answered; an exact
    // bind clears the handle.
    VkBuffer index_whole_handle_{};
    u64 index_whole_tick_{};
    u32 index_whole_type_{};
    u32 index_whole_first_{};
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
    GpuModifiedRangeSet gpu_modified_ranges;
    // Bumped only on clean->dirty coverage transitions; an entry stamped with
    // the current value is proven GPU-clean without walking the range set.
    // Overlapping in-flight fault downloads can leave the range set covering
    // clear-bit pages (veto re-Add in FinishFaultDownload); every consumer of
    // this generation shares that window and is off when skip caches are off.
    u64 gpu_dirty_generation_{1};
    // Written-bind containment (written_range_fast): mode 1 adds a range whose
    // mark set a GPU-clean page without probing the set; mode 2 memoizes ranges
    // proven contained, keyed on a counter every subtract advances.
    struct WrittenRangeEntry {
        VAddr addr;
        u32 size;
        u32 shrink_gen;
    };
    static constexpr size_t WrittenRangeSets = 1024;
    std::unique_ptr<std::array<std::array<WrittenRangeEntry, 2>, WrittenRangeSets>>
        written_range_memo_;
    std::array<u8, WrittenRangeSets> written_range_lru_{};
    u32 gpu_range_shrink_gen_{1}; // never 0: a zero entry must not match
    u32 written_range_mode_{};
    u64 written_binds_{};
    u64 written_fresh_{};
    u64 written_hits_{};
    u64 written_adds_{};
    u64 written_shrinks_{};
    void SubtractGpuModifiedRange(VAddr addr, u64 size);
    bool WrittenRangeCovered(VAddr addr, u32 size) const;
    void RecordWrittenRange(VAddr addr, u32 size);
    // Mode 3 appends a written range to the lane of its 4 MB region; a reader
    // folds the lane entries overlapping its range into the set first, so the
    // set is exact as the union of tree and lanes.
    struct PendingRange {
        VAddr addr;
        u64 size;
    };
    static constexpr size_t PendingLanes = 64;         // region index & 63
    static constexpr size_t PendingLaneCapacity = 256; // a full lane folds whole
    std::array<std::vector<PendingRange>, PendingLanes> pending_lanes_;
    std::vector<PendingRange> pending_batch_; // fold scratch
    u64 pending_folds_{};
    u64 pending_folded_{};
    u64 pending_full_{};
    u64 pending_direct_{};
    void AddWrittenRange(VAddr addr, u64 size);
    void FoldLane(std::vector<PendingRange>& lane, VAddr lo, VAddr hi);
    void FoldPendingRanges(VAddr addr, u64 size);
    bool GpuModifiedRangesContain(VAddr addr, u64 size);

public:
    struct VertexInputStats {
        u64 calls;
        u64 built;
        u64 binds;
        u64 chain;
        u64 layout;
        u64 bind;
        u64 fetchskip;
    };
    VertexInputStats DrainVertexInputStats() {
        return std::exchange(vinput_, {});
    }
    struct IndexWholeStats {
        u64 binds;
        u64 skips;
        u64 veto;
    };
    IndexWholeStats DrainIndexWholeStats() {
        return std::exchange(idxwhole_, {});
    }
    struct WritebackStats {
        u64 loops;
        u64 islands;
        u64 bytes;
    };
    WritebackStats DrainWritebackStats() {
        return std::exchange(writeback_, {});
    }
    /// Marks a CP write CPU-dirty without releasing its write watcher; the
    /// caller then writes the bytes through the backing alias. Refuses an
    /// unregistered range (InvalidateMemory returns early on those, so the
    /// tracker must not gain state here either) or a GPU-modified page.
    [[nodiscard]] bool MarkCpuWriteKeepArmed(VAddr addr, u64 size) {
        if (!IsRegionRegistered(addr, size)) {
            return false;
        }
        return memory_tracker->MarkRegionAsCpuModifiedKeepArmed(addr, size);
    }

    /// Arms every read watcher left pending since the last drain.
    void DrainPendingReadArms(ReadArmSite site, bool carry) {
        // finish_release_faulted_first: the islands FinishFaultDownload parked
        // when it released the faulting thread are settled here. Every caller
        // of this function is the GPU command thread, which owns that state.
        if (!pending_finish_.empty()) {
            DrainPendingFinish();
        }
        const auto d = memory_tracker->ArmPendingReadWatchers(carry);
        ++rarm_.drains[static_cast<size_t>(site)];
        rarm_.regions += d.regions;
        rarm_.pages += d.pages;
        rarm_.calls += d.calls;
    }
    /// Releases every read watcher a finished download left pending.
    void DrainPendingReadReleases(bool carry) {
        if (!memory_tracker->HasPendingReadReleases()) {
            return;
        }
        const auto d = memory_tracker->ReleasePendingReadWatchers(carry);
        ++rrel_drains_;
        rrel_regions_ += d.regions;
        rrel_pages_ += d.pages;
        rrel_calls_ += d.calls;
    }
    struct ReadReleaseStats {
        u64 drains;
        u64 regions;
        u64 pages;
        u64 calls;
        u64 census_calls;
        u64 census_pages;
        u64 census_runs;
        u64 census_batches;
    };
    ReadReleaseStats DrainReadReleaseStats() {
        const auto c = MemoryTracker::DrainReadReleaseCensus();
        const ReadReleaseStats out{rrel_drains_, rrel_regions_, rrel_pages_, rrel_calls_,
                                   c.calls,      c.pages,       c.runs,      c.batches};
        rrel_drains_ = rrel_regions_ = rrel_pages_ = rrel_calls_ = 0;
        return out;
    }
    struct FinishSplitStats {
        u64 jobs;
        u64 inline_islands;
        u64 rest_islands;
        u64 inline_ns;
        u64 rest_ns;
        u64 vetoes;
    };
    FinishSplitStats DrainFinishSplitStats() {
        return std::exchange(finsplit_, {});
    }
    /// Clears the GPU bits of a range the guest is unmapping.
    void DropPendingReadArms(VAddr addr, u64 size) {
        // finish_release_faulted_first: unlike every other drain site this one
        // is reached from guest threads (MemoryManager::UnmapMemory ->
        // Rasterizer::UnmapMemory), so the parked islands are settled through the
        // GPU command thread.
        if (pending_finish_any_.load(std::memory_order_acquire)) {
            DrainPendingFinishSynced();
        }
        memory_tracker->UnmarkRegionAsGpuModified(addr, size);
        // The unmark defers its own release under deferred_read_release, and a
        // release left pending would protect memory the guest has already given
        // back, so settle while the range is still mapped; guest thread, no carry.
        DrainPendingReadReleases(false);
    }
    struct ReadArmStats {
        std::array<u64, static_cast<size_t>(ReadArmSite::Count)> drains;
        u64 regions;
        u64 pages;
        u64 calls;
    };
    ReadArmStats DrainReadArmStats() {
        return std::exchange(rarm_, {});
    }
    struct WriteBackOffloadStats {
        u64 guest;
        u64 prio;
        u64 gpucomm;
        u64 excluded;
        u64 copy_ns;
    };
    WriteBackOffloadStats DrainWriteBackOffloadStats() {
        return std::exchange(wboff_, {});
    }
    struct WbIdleStats {
        u64 posted;
        u64 ran;
        u64 skipped;
        u64 bailed;
        u64 late;
        u64 bytes;
        u32 max_island;
    };
    WbIdleStats DrainWbIdleStats() {
        const auto take = [](auto& c) { return c.exchange(0, std::memory_order_relaxed); };
        return WbIdleStats{take(wbidle_posted_),    take(wbidle_ran_),  take(wbidle_skipped_),
                           take(wbidle_bailed_),    take(wbidle_late_), take(wbidle_bytes_),
                           take(wbidle_max_island_)};
    }
    struct WriteBackShareStats {
        u64 shares;
        u64 joins;
        u64 fencewaits;
        u64 helped;
        u64 helped_bytes;
        u64 owner_islands;
        u64 tail_ns;
        u64 prio_posted;
        u64 prio_helped;
        u64 prio_bytes;
        u64 prio_late;
    };
    WriteBackShareStats DrainWriteBackShareStats() {
        const auto take = [](std::atomic<u64>& c) {
            return c.exchange(0, std::memory_order_relaxed);
        };
        return WriteBackShareStats{
            take(share_shares_),  take(share_joins_),        take(share_fencewaits_),
            take(share_helped_),  take(share_helped_bytes_), take(share_owner_islands_),
            take(share_tail_ns_), take(prio_posted_),        take(prio_helped_),
            take(prio_bytes_),    take(prio_late_)};
    }
    struct WrittenRangeStats {
        u64 binds;
        u64 fresh;
        u64 hits;
        u64 adds;
        u64 shrinks;
        u64 folds;
        u64 folded;
        u64 full;
        u64 direct;
        u64 lockskips;
    };
    WrittenRangeStats DrainWrittenRangeStats() {
        const WrittenRangeStats out{written_binds_,         written_fresh_,   written_hits_,
                                    written_adds_,          written_shrinks_, pending_folds_,
                                    pending_folded_,        pending_full_,    pending_direct_,
                                    GpuRangeSetMutex::skips};
        written_binds_ = written_fresh_ = written_hits_ = written_adds_ = written_shrinks_ = 0;
        pending_folds_ = pending_folded_ = pending_full_ = pending_direct_ = 0;
        GpuRangeSetMutex::skips = 0;
        return out;
    }
    struct GpuRangeStats {
        u64 flat;
        u64 live;
        u64 batches;
        u64 batched;
        u64 moved;
        u64 subs;
    };
    /// GPU command thread: the GPU-modified range set's container, its live
    /// interval count and the flat container's counters, read by the GPURANGE line.
    GpuRangeStats DrainGpuRangeStats() {
        const auto fs = gpu_modified_ranges.vec.DrainStats();
        return GpuRangeStats{GpuModifiedRangeSet::flat,
                             gpu_modified_ranges.Size(),
                             fs.batches,
                             fs.batched,
                             fs.moved,
                             fs.subs};
    }

    /// Emits and resets the tracker telemetry lines of this window.
    void EmitTrackerTelemetry();

private:
    SplitRangeMap<BufferId> buffer_ranges;
    PageTable page_table;
    // Staging pool for offloaded fault readbacks. GPU command thread only.
    std::vector<std::unique_ptr<Buffer>> fault_staging_pool_;
    // readback_offload: the second queue, when the setting and the device
    // provide one, and the open tick of the last device-address shader.
    std::unique_ptr<Vulkan::TransferQueue> copy_queue_;
    u64 dma_write_tick_{};
    bool readback_offload_{};
    // readback_offload: set when a written bind touches a buffer a readback
    // has read within the last kProneWindow garbage collector periods, one
    // per guest submit; the rasterizer takes it after recording the draw.
    static constexpr u64 kProneWindow = 4;
    bool prone_write_pending_{};
    // Islands owned by in-flight readbacks; a later download skips them. GPU
    // command thread only.
    struct InflightDownload {
        u64 id;
        VAddr lo;
        VAddr hi;
        std::vector<std::pair<VAddr, u32>> islands;
        std::shared_ptr<WriteBackShare> share; // readback_writeback_share
    };
    std::vector<InflightDownload> inflight_downloads_;
    u64 next_inflight_id_{1};
    // finish_release_faulted_first: the islands of one job that did not cover
    // the faulted range, parked when the guest thread was released and settled
    // at the next drain site. Written and read on the GPU command thread only;
    // the flag lets the guest-thread unmap path test for one without touching
    // the vector. Capped at one entry (the push drains first).
    struct PendingFinish {
        // The job's whole island vector, moved in rather than copied (a batched
        // window carries several hundred), with the partition point kept as an
        // index: the parked span is [first, copies.size()).
        boost::container::small_vector<vk::BufferCopy, 4> copies;
        size_t first;
        MemoryTracker::GpuSeqSnapshots snapshots;
        VAddr buffer_base;
        u64 inflight_id;
        bool vetoed;
        bool copied;
    };
    boost::container::small_vector<PendingFinish, 2> pending_finish_;
    std::atomic<bool> pending_finish_any_{};
    FinishSplitStats finsplit_{};
    // Offload counters; wait_ns is written by faulting guest threads.
    std::atomic<u64> offload_jobs_{};
    std::atomic<u64> offload_vetoes_{};
    std::atomic<u64> offload_fallbacks_{};
    std::atomic<u64> offload_wait_ns_{};
    std::atomic<u64> join_empty_{};
    std::atomic<u64> q2_copies_{};
    std::atomic<u64> q2_open_{};
    std::atomic<u64> q2_unknown_{};
    std::atomic<u64> q2_dma_{};
    std::atomic<u64> q2_wait_ns_{};
    // Stream copy cache counters; hits count probes that return a cached
    // offset. The probes and the telemetry drain both run on the GPU command
    // thread, so plain counters suffice.
    u64 stream_copy_hits_{};
    u64 stream_copy_probes_{};
    u64 stream_copy_fast_{};
    u64 index_bind_fast_{};
    u64 stream_genwalk_{};
    u64 vertex_genwalk_{};
    u64 index_genwalk_{};
    static void MirrorProtectThunk(void* user, VAddr addr, u64 size, bool write_granted,
                                   bool tracker_origin);
    static void MirrorBackingThunk(void* user, VAddr addr, u64 size);
    bool mirror_mode_{};
    bool tracker_mode_latch_{};
    bool stream_copy_resolved_epoch_{};
    bool writeback_hold_{};
    bool writeback_offload_{};
    bool finish_split_{};
    bool writeback_share_{};
    bool writeback_helper_{};
    bool writeback_gpucomm_idle_{};
    bool texel_sync_noop_{};
    bool vertex_lazy_desc_{};
    bool vinput_fetch_key_{};
    bool index_bind_whole_{};
    VertexInputStats vinput_{};
    IndexWholeStats idxwhole_{};
    WritebackStats writeback_{};
    WriteBackOffloadStats wboff_{};
    std::atomic<u64> wbidle_posted_{};
    std::atomic<u64> wbidle_ran_{};
    std::atomic<u64> wbidle_skipped_{};
    std::atomic<u64> wbidle_bailed_{};
    std::atomic<u64> wbidle_late_{};
    std::atomic<u64> wbidle_bytes_{};
    std::atomic<u32> wbidle_max_island_{};
    ReadArmStats rarm_{};
    bool batch_copy_lock_{};
    bool upload_drain_{};
    u64 upload_ro_calls_{};
    u64 upload_ro_bytes_{};
    u64 upload_w_calls_{};
    u64 upload_w_bytes_{};
    u64 texel_noop_hits_{};
    u64 texel_noop_probes_{};
    u64 dmasync_calls_{};
    u64 dmasync_buffers_{};
    u64 dmasync_bytes_{};
    u64 dmasync_max_bytes_{};
    // readback_wait_notify: the fault waiter blocks on the write-back generation
    // instead of polling (a 50 us sleep is not honoured on a loaded core).
    bool wait_notify_{};
    bool defer_read_release_{};
    u64 rrel_drains_{};
    u64 rrel_regions_{};
    u64 rrel_pages_{};
    u64 rrel_calls_{};
    // Fault window, and the widened invalidate span. Both are latched once:
    // a fault reads them on the guest thread's critical path.
    u64 readback_window_{};
    u64 fault_widen_{};
    std::mutex writeback_cv_m_;
    std::condition_variable writeback_cv_;
    std::atomic<u64> writeback_gen_{};
    void NotifyWriteBack() {
        if (!wait_notify_) {
            return;
        }
        {
            std::lock_guard lk{writeback_cv_m_};
            writeback_gen_.fetch_add(1, std::memory_order_release);
        }
        writeback_cv_.notify_all();
    }
    /// Blocks until a write-back moves the generation off gen_snapshot or the
    /// timeout expires. The caller samples the generation before testing its
    /// own predicate, so a write-back landing between the two cannot be lost.
    void WaitWriteBack(u64 gen_snapshot, u64 timeout_us) {
        std::unique_lock lk{writeback_cv_m_};
        writeback_cv_.wait_for(lk, std::chrono::microseconds(timeout_us), [&] {
            return writeback_gen_.load(std::memory_order_acquire) != gen_snapshot;
        });
    }
    // Refault damping census, written by guest threads on the fault path.
    alignas(64) std::atomic<u64> damp_entries_{};
    std::atomic<u64> damp_iters_{};
    std::atomic<u64> damp_stuck_{};
    // readback_writeback_share census: shares and joins from the GPU command
    // thread, the rest from the copying guest threads. Read by the WBSHARE line.
    alignas(64) std::atomic<u64> share_shares_{};
    std::atomic<u64> share_joins_{};
    std::atomic<u64> share_fencewaits_{};
    std::atomic<u64> share_helped_{};
    std::atomic<u64> share_helped_bytes_{};
    std::atomic<u64> share_owner_islands_{};
    std::atomic<u64> share_tail_ns_{};
    std::atomic<u64> prio_posted_{};
    std::atomic<u64> prio_helped_{};
    std::atomic<u64> prio_bytes_{};
    std::atomic<u64> prio_late_{};
};

} // namespace VideoCore
