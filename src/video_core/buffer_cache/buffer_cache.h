// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/small_vector.hpp>
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/range_set.h"

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

class TextureCache;
class MemoryTracker;
class PageManager;

using MapRanges = boost::icl::interval_map<VAddr, bool, boost::icl::partial_enricher>;

class BufferCache {
public:
    static constexpr u32 CACHING_PAGEBITS = 14;
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;
    static constexpr u64 DEVICE_PAGESIZE = 16_KB;
    static constexpr u64 CACHING_NUMPAGES = u64{1} << (40 - CACHING_PAGEBITS);
    static constexpr u64 BDA_PAGETABLE_SIZE = CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         AmdGpu::Liverpool* liverpool, TextureCache& texture_cache,
                         PageManager& tracker, MapRanges& mapped_ranges);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer& GetGdsBuffer() const noexcept {
        return gds_buffer;
    }

    [[nodiscard]] const AddressSpaceBuffer& GetAddressSpace() noexcept {
        return address_space;
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

    bool SynchronizeMemory(VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer);

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
    [[nodiscard]] std::pair<Buffer*, u64> ObtainBuffer(VAddr gpu_addr, u32 size, bool is_written,
                                                       bool is_texel_buffer = false);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<Buffer*, u64> ObtainBufferForImage(VAddr gpu_addr, u32 size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

private:
    template <bool async>
    void DownloadBufferMemory(VAddr device_addr, u64 size, bool is_write);

    vk::Buffer UploadCopies(std::span<vk::BufferCopy> copies, size_t total_size_bytes);

    bool SynchronizeBufferFromImage(VAddr device_addr, u32 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;
    AddressSpaceBuffer address_space;
    StreamBuffer staging_buffer;
    StreamBuffer stream_buffer;
    StreamBuffer download_buffer;
    StreamBuffer device_buffer;
    Buffer gds_buffer;
    RangeSet gpu_modified_ranges;
    MapRanges& mapped_ranges;
};

} // namespace VideoCore
