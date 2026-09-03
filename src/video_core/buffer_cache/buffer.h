// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
#include "common/assert.h"
#include "common/types.h"
#include "core/memory.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/buffer_cache/stream_copy_lane.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

struct VmaAllocationInfo;

namespace VideoCore {

/// Hints and requirements for the backing memory type of a commit
enum class MemoryUsage {
    DeviceLocal, ///< Requests device local buffer.
    Upload,      ///< Requires a host visible memory type optimized for CPU to GPU uploads
    Download,    ///< Requires a host visible memory type optimized for GPU to CPU readbacks
    Stream,      ///< Requests device local host visible buffer, falling back host memory.
};

constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

struct UniqueBuffer {
    explicit UniqueBuffer(vk::Device device, VmaAllocator allocator);
    ~UniqueBuffer();

    UniqueBuffer(const UniqueBuffer&) = delete;
    UniqueBuffer& operator=(const UniqueBuffer&) = delete;

    // Moves must carry device and bda_addr: a dropped bda_addr leaves the
    // moved buffer with address zero (the assert in BufferDeviceAddress is
    // compiled out under NDEBUG), the BDA page table gets near-null entries
    // and shader writes through them fault. A Buffer moves whenever
    // slot_buffers grows on new resource creation.
    UniqueBuffer(UniqueBuffer&& other)
        : device{other.device}, allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          buffer{std::exchange(other.buffer, VK_NULL_HANDLE)},
          bda_addr{std::exchange(other.bda_addr, 0)} {}
    UniqueBuffer& operator=(UniqueBuffer&& other) {
        device = other.device;
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        bda_addr = std::exchange(other.bda_addr, 0);
        return *this;
    }

    void Create(const vk::BufferCreateInfo& image_ci, MemoryUsage usage,
                VmaAllocationInfo* out_alloc_info);

    operator vk::Buffer() const {
        return buffer;
    }

    vk::Device device;
    VmaAllocator allocator;
    VmaAllocation allocation;
    vk::Buffer buffer{};
    vk::DeviceAddress bda_addr = 0;
};

class Buffer {
public:
    explicit Buffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    MemoryUsage usage, VAddr cpu_addr_, vk::BufferUsageFlags flags,
                    u64 size_bytes_);

    Buffer& operator=(const Buffer&) = delete;
    Buffer(const Buffer&) = delete;

    Buffer& operator=(Buffer&&) = default;
    Buffer(Buffer&&) = default;

    void IncreaseStreamScore(int score) noexcept {
        stream_score += score;
    }

    [[nodiscard]] int StreamScore() const noexcept {
        return stream_score;
    }

    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= cpu_addr && addr + size <= cpu_addr + SizeBytes();
    }

    [[nodiscard]] VAddr CpuAddr() const noexcept {
        return cpu_addr;
    }

    [[nodiscard]] u64 Offset(VAddr other_cpu_addr) const noexcept {
        return other_cpu_addr - cpu_addr;
    }

    size_t SizeBytes() const {
        return size_bytes;
    }

    void SetLRUId(u64 id) noexcept {
        lru_id = id;
    }

    u64 LRUId() const noexcept {
        return lru_id;
    }

    vk::Buffer Handle() const noexcept {
        return buffer;
    }

    vk::DeviceAddress BufferDeviceAddress() const noexcept {
        ASSERT_MSG(buffer.bda_addr != 0, "Can't get BDA from a non BDA buffer");
        return buffer.bda_addr;
    }

    std::optional<vk::BufferMemoryBarrier2> GetBarrier(vk::AccessFlags2 dst_acess_mask,
                                                       vk::PipelineStageFlagBits2 dst_stage,
                                                       u32 offset = 0) {
        if (dst_acess_mask == access_mask && stage == dst_stage) {
            return {};
        }

        DEBUG_ASSERT(offset < size_bytes);

        const auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = stage,
            .srcAccessMask = access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_acess_mask,
            .buffer = buffer.buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
        access_mask = dst_acess_mask;
        stage = dst_stage;
        return barrier;
    }

    void Fill(u64 offset, u32 num_bytes, u32 value);

    /// Makes device writes visible to host reads on non-coherent memory.
    void InvalidateForRead(u64 offset, u64 num_bytes);

public:
    VAddr cpu_addr = 0;
    bool is_picked{};
    bool is_coherent{};
    bool is_deleted{};
    int stream_score = 0;
    size_t size_bytes = 0;
    u64 lru_id = 0;
    std::span<u8> mapped_data;
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    MemoryUsage usage;
    UniqueBuffer buffer;
    vk::Flags<vk::AccessFlagBits2> access_mask{
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
    vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
    // Memos of recent read-only upload queries that found nothing to upload.
    // While the key is unchanged the guest bytes still equal the device-buffer
    // bytes for a recorded range, so eliding the upload is byte-identical; a
    // query contained in a recorded range hits, since a clean range has no
    // dirty subrange. The key is the host-memory generation, or the range's
    // word-epoch sum under the stream mirror mode, where a clean range keeps
    // every page write-protected and any write bumps the sum through the
    // protection grant. Entries wider than the epoch-sum span key on the
    // generation either way. Zero size = empty.
    struct SyncNoop {
        VAddr addr = 0;
        u32 size = 0;
        u64 mem_key = 0;
    };
    std::array<SyncNoop, 3> sync_noop{};
    u32 sync_noop_next = 0;
};
// The memo array sits last so the fields every bind reads keep the front
// of the object as entries are added.
static_assert(offsetof(Buffer, sync_noop) > offsetof(Buffer, stage));

class StreamBuffer : public Buffer {
public:
    explicit StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                          MemoryUsage usage, u64 size_bytes_);

    /// Reserves a region of memory from the stream buffer.
    std::pair<u8*, u64> Map(u64 size, u64 alignment = 0, bool allow_wait = true);

    /// Ring statistics: wraps and nanoseconds spent blocked waiting for the GPU
    /// to drain the previous lap, plus bytes handed out. Reported per window by
    /// the skip cache framework; zero cost when it is inactive.
    struct RingStats {
        u64 wraps{};
        u64 blocked_ns{};
        u64 bytes{};
        u64 maps{};
    };
    RingStats& Stats() {
        return ring_stats_;
    }

    /// Ensures that reserved bytes of memory are available to the GPU. The
    /// header fast path collapses a coherent same-tick commit into the last
    /// watch without the call or the flush branch - identical to the slow
    /// path's own first branch.
    void Commit() {
        if (is_coherent && current_watch_cursor != 0) {
            auto& last = current_watches[current_watch_cursor - 1];
            if (last.tick == scheduler->CurrentTick()) {
                offset += mapped_size;
                last.upper_bound = offset;
                return;
            }
        }
        CommitSlow();
    }
    void CommitSlow();

    /// Maps and commits a memory region with user provided data
    u64 Copy(auto src, size_t size, size_t alignment = 0) {
        const VAddr src_vaddr = reinterpret_cast<const VAddr>(src);
        // Deferred drain lane: large-enough guest-addressed sources are copied
        // by worker threads reading the physical backing view (never
        // protected, so a worker cannot fault). Host-pointer sources are
        // stack- or heap-locals whose lifetime ends with the caller and must
        // stay inline. The 64-byte alignment keeps a job's write-combining
        // lines private to one core; padding costs are absorbed by the ring.
        if (auto& lane = StreamCopyLane::Instance(); lane.Enabled() && size >= 192) {
            auto* memory = Core::Memory::Instance();
            if (memory->IsValidMapping(src_vaddr)) {
                Core::MemoryManager::BackingSpan spans[2];
                const bool hardened = lane.Hardened();
                const u32 num_spans =
                    memory->ResolveBackingSpans(src_vaddr, size, spans, 2, hardened);
                if (num_spans != 0) {
                    const auto [data, offset] = Map(size, alignment < 64 ? 64 : alignment);
                    u8* dst = data;
                    bool queued = true;
                    for (u32 i = 0; i < num_spans; ++i) {
                        if (queued) {
                            queued = lane.Push(spans[i].ptr, dst, static_cast<u32>(spans[i].size));
                        }
                        if (!queued) {
                            std::memcpy(dst, spans[i].ptr, spans[i].size);
                        }
                        dst += spans[i].size;
                    }
                    if (hardened) {
                        Core::MemoryManager::EndBackingPush();
                    }
                    Commit();
                    return offset;
                }
                lane.NoteInlineUnresolved();
            }
        }
        // Guest sources are written by game threads on other cores and read
        // exactly once here, so their lines are cold. Requesting them before
        // the ring bookkeeping and address resolution overlaps the memory
        // latency with that work. Prefetch never faults, so protected or
        // unmapped pages in sparse ranges are safe to request.
        constexpr size_t prefetch_bytes = 1024;
        for (size_t i = 0; i < std::min<size_t>(size, prefetch_bytes); i += 64) {
            __builtin_prefetch(reinterpret_cast<const void*>(src_vaddr + i), 0, 3);
        }
        const auto [data, offset] = Map(size, alignment);
        auto* memory = Core::Memory::Instance();
        if (memory->IsValidMapping(src_vaddr)) {
            memory->CopySparseMemory(src_vaddr, data, size);
        } else {
            std::memcpy(data, reinterpret_cast<const void*>(src), size);
        }
        Commit();
        return offset;
    }

private:
    struct Watch {
        u64 tick{};
        u64 upper_bound{};
    };

    /// Increases the amount of watches available.
    void ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size);

    /// Starts a new ring lap when a map does not fit in the remaining space
    /// and arms the watch drain for the previous lap.
    SHAD_NO_INLINE void MapWrap();

    /// Waits pending watches until requested upper bound.
    bool WaitPendingOperations(u64 requested_upper_bound, bool allow_wait);

private:
    u64 offset{};
    u64 mapped_size{};
    std::vector<Watch> current_watches;
    std::size_t current_watch_cursor{};
    std::optional<size_t> invalidation_mark;
    std::vector<Watch> previous_watches;
    std::size_t wait_cursor{};
    RingStats ring_stats_{};
    u64 wait_bound{};
};

} // namespace VideoCore
