// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <boost/container/small_vector.hpp>
#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/multi_level_page_table.h"

namespace AmdGpu {
struct Liverpool;
struct LiverpoolRegsSnapshot;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
class Rasterizer;  // back-ref for PushPresenterRecord
}

namespace VideoCore {

using BufferId = Common::SlotId;

static constexpr BufferId NULL_BUFFER_ID{0};

class TextureCache;
class MemoryTracker;
class PageManager;

// GR2FORK FIX: holds the resolved vertex-input descriptors cached across cmdbuf
// rotations. Defined in buffer_cache.cpp so this header need not pull in the
// heavy vk_graphics_pipeline.h (VertexInputs / vk vertex-input types).
struct VbbResolveCache;

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

    struct DmaSyncState {
        u64 cpu_dirty_generation{};
        u64 buffer_registry_generation{};
    };

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         AmdGpu::Liverpool* liverpool, TextureCache& texture_cache,
                         PageManager& tracker);
    ~BufferCache();

    /// Back-ref to the Rasterizer that owns this BufferCache, used so
    /// SendCommand-dispatched scheduler-touchers in ReadMemory can route
    /// through Rasterizer::PushPresenterRecord.
    /// Called from Rasterizer's ctor body once `bundle_assembler_`
    /// is constructed; before SetRasterizer fires `rasterizer_` is nullptr
    /// and the SendCommand path must not be reachable (the page-fault
    /// signal handler is wired only after Rasterizer is fully constructed).
    void SetRasterizer(Vulkan::Rasterizer* rasterizer) noexcept {
        rasterizer_ = rasterizer;
    }

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

    /// Binds host vertex buffers for the current draw.
    void BindVertexBuffers(const Vulkan::GraphicsPipeline& pipeline,
                           const AmdGpu::LiverpoolRegsSnapshot& regs);

    /// Bind host index buffer for the current draw.
    void BindIndexBuffer(u32 index_offset, const AmdGpu::LiverpoolRegsSnapshot& regs);

    /// Writes a value to GPU buffer. (uses command buffer to temporarily store the data)
    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);

    /// Performs buffer to buffer data copy on the GPU.
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);

    /// Obtains a buffer for the specified region.
    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(VAddr gpu_addr, u32 size, bool is_written,
                                                       bool is_texel_buffer = false,
                                                       BufferId buffer_id = {});

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

    /// Returns the generations that invalidate a DMA-wide upload sweep.
    [[nodiscard]] DmaSyncState GetDmaSyncState() const noexcept;

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

    /// Monotonic counter bumped on every buffer (un)registration (the single
    /// ChangeRegister chokepoint behind Register/Unregister, hence behind
    /// CreateBuffer, DeleteBuffer, and overlap join/expand). A BufferId / host
    /// VkBuffer handle / buffer offset resolved while this counter read V stays
    /// valid as long as it still reads V: no buffer was created, freed,
    /// relocated, or had its slot recycled in the interim. Pure CPU-write
    /// content changes do NOT bump it (they aren't a registration event), so a
    /// caller reusing cached host binds across cmdbufs must still re-run
    /// SynchronizeBuffer for content. relaxed: all (un)registration and all
    /// reads happen on the GpuAssembler resolve thread (FindBuffer-driven
    /// creation and GC FreeImage both run there); the atomic only guarantees a
    /// non-torn read.
    [[nodiscard]] u64 BufferRegistryGeneration() const noexcept {
        return buffer_registry_generation_.load(std::memory_order_relaxed);
    }

    template <bool async>
    void DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write);

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

    bool SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size);

    void WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes);

    void TouchBuffer(const Buffer& buffer);

    void DeleteBuffer(BufferId buffer_id);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    // Set by SetRasterizer post-construction. Used to route
    // signal-handler-dispatched scheduler touchers in ReadMemory's
    // SendCommand lambda through Rasterizer::PushPresenterRecord.
    Vulkan::Rasterizer* rasterizer_{nullptr};
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    FaultManager fault_manager;
    std::unique_ptr<MemoryTracker> memory_tracker;
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
    SplitRangeMap<BufferId> buffer_ranges;
    // Per-thread hot-path cache: skip redundant vertex input/buffer binds.
    u64 last_vertex_bind_sig = 0;
    bool last_vertex_bind_sig_valid = false;

    // Vertex input dynamic state cache (does NOT include guest buffer addresses).
    u64 last_vertex_input_sig = 0;
    bool last_vertex_input_sig_valid = false;

    // GR2FORK: BindVertexBuffers dispatches on Config::accurateVertexBufferCacheEnabled() (forced
    // on for Gravity Rush Remastered in emulator.cpp): Fixed = cmdbuf-rotation-aware + resolve
    // cache; Legacy = verbatim upstream. An unchanged pipeline+stamp skip saves ~0.76% of GpuComm.
    void BindVertexBuffersFixed(const Vulkan::GraphicsPipeline& pipeline,
                                const AmdGpu::LiverpoolRegsSnapshot& regs);
    void BindVertexBuffersLegacy(const Vulkan::GraphicsPipeline& pipeline,
                                 const AmdGpu::LiverpoolRegsSnapshot& regs);

    const Vulkan::GraphicsPipeline* last_vbb_pipeline_{};
    u64 last_vbb_stamp_{};

    // GR2FORK FIX: setVertexInputEXT/bindVertexBuffers are per-primary-cmdbuf state, so the skips
    // also key on last_vbb_tick_ - otherwise the first bind of a rotated cmdbuf is skipped (black
    // meshes). vbb_resolve_ caches pipeline+stamp-pure descriptors; ObtainBuffer runs every cmdbuf.
    u64 last_vbb_tick_{};
    bool last_vbb_resolve_valid_{false};
    std::unique_ptr<VbbResolveCache> vbb_resolve_;
    // GR2FORK PERF: bumped by ChangeRegister (Register/Unregister); see
    // BufferRegistryGeneration(). Cached host VkBuffer handles/offsets in the
    // BindVertexBuffersFixed rotation-reuse path are only reused while this counter is unchanged.
    std::atomic<u64> buffer_registry_generation_{0};

    // GR2FORK PERF: GPU-clean epoch, bumped on clean-to-dirty tracker transitions. A stream-copy
    // cache entry with a matching stamp proves no range went GPU-dirty since validation, skipping
    // the per-bind tracker walk (~15-30 ns x 1e4-1e5/frame). Starts at 1. Kill: GR2_NOGPUEPOCH=1.
    u64 gpu_dirty_generation_{1};

    // GR2FORK PERF: Index buffer bind deduplication.
    // Skip redundant vkCmdBindIndexBuffer when the same buffer/offset/type is already bound.
    // Keyed on (stamp, tick) - tick advances on every primary cmdbuf rotation.
    VAddr last_index_address_{};
    u32 last_index_buffer_size_{};
    vk::IndexType last_index_type_{};
    u64 last_index_tick_{};
    // GR2FORK PERF: BindIndexBuffer stamp skip. Every bind input is a context reg covered by the
    // gfx_pipeline_stamp; a (stamp, index_offset, tick) match proves the correct index buffer is
    // already bound, avoiding the regs read + index_address computation every draw.
    u64 last_index_stamp_{};
    u32 last_index_offset_{};

    PageTable page_table;
};

} // namespace VideoCore
