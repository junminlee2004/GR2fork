// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/enum.h"
#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "video_core/texture_cache/image.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
struct StagingBufferRef;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class PageManager;
class ReadbackLanes;

enum class ObtainBufferFlags {
    None = 0,
    IsWritten = 1 << 0,
    IsTexelBuffer = 1 << 1,
    IgnoreStreamBuffer = 1 << 2,
    InvalidateTextureCache = 1 << 3,
};
DECLARE_ENUM_FLAG_OPERATORS(ObtainBufferFlags)

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

public:
    /// A readback serves the GPU writes of one window at a time.
    static constexpr u64 READBACK_WINDOW_SIZE = 512_KB;

    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    [[nodiscard]] const Buffer* GetNullIndexBuffer() const noexcept {
        return null_index_buffer.get();
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Unmark a region as modified from the GPU.
    void UnmarkRegionAsGpuModified(VAddr device_addr, u64 size, bool is_write = false);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool async = false,
                    bool assume_locks = false);

    /// Flushes GPU modified ranges of the uncovered part of the edge pages of an image. The caller
    /// owns the image's readback windows.
    void ReadEdgeImagePages(const Image& image);

    /// One downloader owns a range at a time: a GPU thread download from its record to its
    /// write-back, a faulting thread from its copy to the consume, the texture cache across an
    /// image download. Acquiring waits for the current owners.
    void AcquireWindows(VAddr device_addr, u64 size);
    void ReleaseWindows(VAddr device_addr, u64 size);

    using ReadbackCopies = boost::container::small_vector<vk::BufferCopy, 2>;

    /// Appends the GPU writes of a range stamped up to tick as copies from the arena into a staging
    /// buffer, packed behind the copies already listed. Returns the packed size of the list.
    u64 CollectWrites(const Buffer* arena, VAddr device_addr, u64 size, u64 tick,
                      ReadbackCopies& copies);

    /// Writes the copies of a staging buffer, whose host mapping starts at staging_offset, back to
    /// guest memory.
    void WriteBack(VAddr arena_base, const ReadbackCopies& copies, const u8* mapped,
                   u64 staging_offset) const;

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(
        VAddr device_addr, u32 size, ObtainBufferFlags flags = ObtainBufferFlags::None);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Mark region as modified from the GPU
    void MarkRegionAsGpuModified(VAddr addr, size_t size);

    /// Records a GPU write the packet being bound will perform.
    void MarkGpuWrite(VAddr device_addr, u64 size);

    /// Stamps and marks the writes of the packet again when a flush moved it into the current
    /// batch. Called right after each draw, dispatch, fill or copy is recorded, and on every exit
    /// of a packet whose record did not happen.
    void CommitGpuWrites();

    /// Moves the writes of a packet still being bound into the batch this flush opens. Called from
    /// the submit callback.
    void MoveInFlightWrites();

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

private:
    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        std::scoped_lock lk{arena_mutex};
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size, bool async);

    /// Serves a fault from the thread that took it, inside one readback window, while the GPU
    /// thread keeps running. Returns false when the request has to take the synchronous download.
    bool FaultReadback(VAddr device_addr, u64 size, bool is_write);

    /// Serves a fault out of the arena's host mapping: waits for the newest GPU write of the
    /// faulting page, submitting the open batch when it holds that write (the GPU thread flushes
    /// in place, a guest thread asks it to), then copies every retired write of the window into
    /// guest memory and consumes it. Returns false when the synchronous download has to serve
    /// it.
    bool DirectReadback(VAddr device_addr);

    /// A fault taken on the GPU thread, served by DirectReadback. Returns false when the
    /// synchronous download has to serve it.
    bool GpuThreadReadback(VAddr device_addr, u64 size, bool is_write);

    /// Streams a GPU written range from the arena's host mapping into guest memory. Returns false
    /// when a block has no host mapping or the guest range has no backing.
    bool DirectRead(VAddr device_addr, u64 size);

    /// The arena holding an address and the bind semaphore value its binds signal, for the copy
    /// lanes; nullptr while binds of the arena are still queued for the next submit.
    const Buffer* ArenaOf(VAddr device_addr, u64& bind_tick);

    /// Requires window_mutex.
    bool WindowsOwned(VAddr device_addr, u64 size) const;

    /// Erases the GPU writes of the range that the GPU had completed at known_tick and clears the
    /// pages left without one; a write marked since keeps its pages GPU modified.
    void ConsumeGpuWrites(VAddr device_addr, u64 size, u64 known_tick);

    /// Stamps a GPU write with the tick of the open batch.
    void StampGpuWrite(VAddr device_addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    std::unique_ptr<Buffer> null_index_buffer;
    /// GPU written ranges with the scheduler tick of the batch holding the write; the newest
    /// write of a byte sets its tick. Guarded by gpu_writes_mutex.
    RangeMap<u64> gpu_writes;
    std::mutex gpu_writes_mutex;
    struct Range {
        VAddr addr;
        u64 size;
    };
    /// Writes marked for the packet being bound, and whether a flush moved them since.
    boost::container::small_vector<Range, 8> in_flight;
    bool in_flight_moved{};

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        /// Host mapping of the allocation; nullptr when its memory type has no host access.
        u8* host;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start), host};
        }
    };
    /// Guards resident_ranges, address_space and pending_binds between the GPU thread, which
    /// grows them, and the faulting threads, which look arenas and blocks up.
    std::mutex arena_mutex;
    IntervalList<Backing> resident_ranges;
    /// memory_semaphore value of the last sparse bind submission. Guarded by arena_mutex.
    u64 last_bind_tick{};

    /// Host visible device memory for the arenas; nullopt without readbacks or without a type
    /// whose heap is the video memory.
    std::optional<u32> arena_host_type_index;
    /// Plain device memory, the arena type without a host visible one and the fallback once the
    /// host visible heap is spent.
    u32 arena_device_type_index{};
    /// Set once a host visible allocation failed; later allocations take device memory.
    bool arena_host_spent{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};

    std::mutex window_mutex;
    std::condition_variable window_cv;
    boost::container::small_vector<Range, 8> window_owners;
    /// Last member: the lane threads stop before anything they use goes away.
    std::unique_ptr<ReadbackLanes> copy_lanes;
};

} // namespace VideoCore
