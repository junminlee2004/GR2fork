// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
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
class Rasterizer;  // Phase 1D-pre-F: back-ref for PushPresenterRecord
}

namespace VideoCore {

using BufferId = Common::SlotId;

static constexpr BufferId NULL_BUFFER_ID{0};

class TextureCache;
class MemoryTracker;
class PageManager;

// GR2FORK fix: holds the resolved vertex-input descriptors cached across cmdbuf
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

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         AmdGpu::Liverpool* liverpool, TextureCache& texture_cache,
                         PageManager& tracker);
    ~BufferCache();

    /// Phase 1D-pre-F: back-ref to the Rasterizer that owns this BufferCache,
    /// used so SendCommand-dispatched scheduler-touchers in ReadMemory can
    /// route through Rasterizer::PushPresenterRecord (Phase E's closure
    /// marker). Called from Rasterizer's ctor body once `bundle_assembler_`
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
    // Phase 1D-pre-F: set by SetRasterizer post-construction. Used to
    // route signal-handler-dispatched scheduler touchers in ReadMemory's
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

    // PERF(GR2): BindVertexBuffers ultra-fast skip. When pipeline ptr + gfx
    // pipeline stamp are unchanged since the previous successful call, no
    // input to GetVertexInputs / vertex bind hashing could have moved
    // (fetch_shader is pipeline-fixed; vgt_instance_step_rate_0/1 are
    // context regs in the stamp; buffer sharps are SH regs in the stamp).
    // Returning here saves ~0.76% of GpuComm time spent in
    // GetVertexInputs + dual hash loops.
    // GR2FORK: BindVertexBuffers dispatches to one of these based on
    // Config::accurateVertexBufferCacheEnabled() (force-enabled for Gravity Rush
    // Remastered in emulator.cpp). Fixed = cmdbuf-rotation-aware + resolve-cache
    // path; Legacy = verbatim upstream behavior (no tick guard, no resolve cache).
    void BindVertexBuffersFixed(const Vulkan::GraphicsPipeline& pipeline,
                                const AmdGpu::LiverpoolRegsSnapshot& regs);
    void BindVertexBuffersLegacy(const Vulkan::GraphicsPipeline& pipeline,
                                 const AmdGpu::LiverpoolRegsSnapshot& regs);

    const Vulkan::GraphicsPipeline* last_vbb_pipeline_{};
    u64 last_vbb_stamp_{};

    // GR2FORK fix (cmdbuf-rotation correctness + perf): setVertexInputEXT +
    // bindVertexBuffers are per-PRIMARY-cmdbuf dynamic state, but every skip
    // above keys on pipeline/stamp/bind_sig/input_sig which all persist across
    // cmdbuf rotations. Without a cmdbuf-aware guard the FIRST bind of a freshly
    // rotated cmdbuf was skipped -> undefined vertex-input state -> black mesh
    // (correct silhouette), intermittent.
    //   - last_vbb_tick_ records the primary-cmdbuf tick of the last emit so the
    //     skips become cmdbuf-aware (CurrentTick advances on every rotation, the
    //     same signal the index-buffer skip uses).
    //   - vbb_resolve_ caches the resolved descriptors (attributes/bindings/guest
    //     buffers). They depend ONLY on the pipeline (immutable fetch_shader) and
    //     stamp-tracked regs, so on a rotation with unchanged pipeline+stamp we
    //     reuse them and skip the GetVertexInputs walk + dual hash loops. NO host
    //     buffer handles/offsets are cached, so ObtainBuffer still runs every
    //     cmdbuf (stream offsets / buffer handles / content sync stay correct).
    u64 last_vbb_tick_{};
    bool last_vbb_resolve_valid_{false};
    std::unique_ptr<VbbResolveCache> vbb_resolve_;

    // OPT(v18): Index buffer bind deduplication.
    // Skip redundant vkCmdBindIndexBuffer when the same buffer/offset/type is already bound.
    // Keyed on (stamp, tick) — tick advances on every primary cmdbuf rotation.
    VAddr last_index_address_{};
    u32 last_index_buffer_size_{};
    vk::IndexType last_index_type_{};
    u64 last_index_tick_{};
    // PERF(GR2): BindIndexBuffer stamp ultra-fast skip. Inputs to the bind
    // (regs.index_buffer_type, regs.index_base_address, regs.num_indices)
    // are all context regs in the gfx_pipeline_stamp; combined with
    // index_offset (the function arg) and tick, a stamp match
    // proves the cmdbuf already has the correct index buffer bound.
    // Avoids reading regs + computing index_address every draw.
    u64 last_index_stamp_{};
    u32 last_index_offset_{};

    PageTable page_table;
};

} // namespace VideoCore
