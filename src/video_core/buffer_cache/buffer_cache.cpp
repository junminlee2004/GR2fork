// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <thread>
#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>

#include "common/alignment.h"
#include "common/scope_exit.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t NULL_INDEX_BUFFER_SIZE = 16;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;
// Every readback window lies inside one tracker region.
static_assert(HIGHER_PAGE_SIZE % BufferCache::READBACK_WINDOW_SIZE == 0);
constexpr std::optional<u32> CmaskColorExpandedValue(u32 num_samples) {
    switch (num_samples) {
    case 1:
        return 0xFFFFFFFF;
    case 2:
        return 0xDDDDDDDD;
    case 4:
        return 0xEEEEEEEE;
    case 8:
        return 0xFFFFFFFF;
    default:
        return 0;
    }
}
static_assert(CmaskColorExpandedValue(1) == 0xFFFFFFFF);
static_assert(CmaskColorExpandedValue(2) == 0xDDDDDDDD);
static_assert(CmaskColorExpandedValue(4) == 0xEEEEEEEE);
static_assert(CmaskColorExpandedValue(8) == 0xFFFFFFFF);

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

// Host visible device memory serves direct readbacks when its heap is the device's video memory
// (integrated GPUs, resizable BAR); a BAR carveout of a few hundred MiB is left alone.
static std::optional<u32> FindHostVisibleArenaType(const Vulkan::Instance& instance,
                                                   u32 memory_type_bits) {
    using Flags = vk::MemoryPropertyFlagBits;
    const auto& props = instance.GetMemoryProperties();
    const auto type = FindMemoryType(
        props, Flags::eDeviceLocal | Flags::eHostVisible | Flags::eHostCoherent, memory_type_bits);
    if (!type || instance.IsIntegrated()) {
        return type;
    }
    u64 largest_local_heap = 0;
    for (u32 i = 0; i < props.memoryHeapCount; ++i) {
        if ((props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) !=
            vk::MemoryHeapFlags{}) {
            largest_local_heap = std::max<u64>(largest_local_heap, props.memoryHeaps[i].size);
        }
    }
    if (props.memoryHeaps[props.memoryTypes[*type].heapIndex].size == largest_local_heap) {
        return type;
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_device_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();
    if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled) {
        arena_host_type_index = FindHostVisibleArenaType(instance, reqs.memoryTypeBits);
    }
    scheduler.SetHostReadBarrier(arena_host_type_index.has_value());
    LOG_INFO(Render_Vulkan, "Readback arena memory type {}, host visible: {}",
             arena_host_type_index.value_or(arena_device_type_index),
             arena_host_type_index.has_value());

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
    if (!instance.IsMaintenance6Supported()) {
        null_index_buffer = std::make_unique<Buffer>(instance, 0, NULL_INDEX_BUFFER_SIZE,
                                                     MemoryType::DeviceLocal, "Null Index Buffer");
        runtime.FillBuffer(null_index_buffer.get(), 0u, NULL_INDEX_BUFFER_SIZE, 0u);
    }
}

BufferCache::~BufferCache() = default;

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, false, assume_locks);
    });
}

void BufferCache::UnmarkRegionAsGpuModified(VAddr device_addr, u64 size, bool is_write) {
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, is_write);
    std::scoped_lock lk{gpu_writes_mutex};
    gpu_writes.Subtract(device_addr, size);
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool async,
                             bool assume_locks) {
    // A fault is served on the thread that took it; the texture cache's downloads and the requests
    // the fault paths hand back take the download below.
    if (!async && (assume_locks ? GpuThreadReadback(device_addr, size, is_write)
                                : FaultReadback(device_addr, size, is_write))) {
        return;
    }
    const auto flush_request = [this, device_addr, size, is_write, async] {
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);
        VAddr start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
        VAddr end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
        if (!async) {
            // GPU-modified ranges come as many small scattered islands, so a fault's download is
            // widened to a window around the request and ends on the page boundary the consume
            // clears up to; an asynchronous download covers exactly the pages of its range.
            const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
            start = std::max<VAddr>(Common::AlignDown(device_addr, READBACK_WINDOW_SIZE),
                                    arena->cpu_addr);
            end = std::min<VAddr>(std::max<VAddr>(start + READBACK_WINDOW_SIZE, end), arena_end);
        }
        DownloadMemory(arena, start, end - start, async);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

u64 BufferCache::CollectWrites(const Buffer* arena, VAddr device_addr, u64 size, u64 tick,
                               ReadbackCopies& copies) {
    u64 total_size = copies.empty() ? 0 : copies.back().dstOffset + copies.back().size;
    std::scoped_lock lk{gpu_writes_mutex};
    gpu_writes.ForEachInRange(device_addr, size, [&](VAddr begin, VAddr end, u64 write_tick) {
        if (write_tick > tick) {
            return;
        }
        copies.push_back(vk::BufferCopy{
            .srcOffset = arena->Offset(begin),
            .dstOffset = total_size,
            .size = end - begin,
        });
        total_size += end - begin;
    });
    return total_size;
}

void BufferCache::WriteBack(VAddr arena_base, const ReadbackCopies& copies, const u8* mapped,
                            u64 staging_offset) const {
    for (const auto& copy : copies) {
        memory->TryWriteBacking(std::bit_cast<u8*>(arena_base + copy.srcOffset),
                                mapped + (copy.dstOffset - staging_offset), copy.size);
    }
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size, bool async) {
    // An asynchronous download's caller owns the range until the deferred write-back has run; a
    // synchronous one owns it here.
    if (!async) {
        AcquireWindows(device_addr, size);
    }
    ReadbackCopies copies;
    const VAddr arena_base = arena->cpu_addr;
    const u64 total_size_bytes =
        CollectWrites(arena, device_addr, size, scheduler.CurrentTick(), copies);
    if (total_size_bytes == 0) {
        // A page GPU modified without a GPU write to copy would fault forever.
        if (memory_tracker->IsRegionGpuModified(device_addr, size)) {
            LOG_WARNING(Render_Vulkan, "GPU modified pages without a GPU write at {:#x}",
                        device_addr);
            memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
        }
        if (!async) {
            ReleaseWindows(device_addr, size);
        }
        return;
    }
    const auto download =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 16, async);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    const u64 record_tick = scheduler.CurrentTick();

    const auto write_data = [this, arena_base, device_addr, size, download, async, record_tick,
                             copies = std::move(copies)] {
        download.Invalidate();
        WriteBack(arena_base, copies, download.mapped, download.offset);
        // After the synchronous wait every write the copy found has executed, including the
        // writes of a packet still being bound, which the commit after its record marks again.
        ConsumeGpuWrites(device_addr, size, async ? record_tick : scheduler.CurrentTick());
        if (async) {
            staging_pool.FreeDeferred(download);
        } else {
            ReleaseWindows(device_addr, size);
        }
    };

    if (async) {
        scheduler.DeferOperation(std::move(write_data));
    } else {
        scheduler.Finish();
        write_data();
    }
}

void BufferCache::ConsumeGpuWrites(VAddr device_addr, u64 size, u64 known_tick) {
    // The pages are cleared whole, so the writes are erased whole as well.
    const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const VAddr page_end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
    memory_tracker->ForEachRegionLocked(
        page_start, page_end - page_start,
        [&](RegionManager* manager, VAddr region_base, VAddr start, u64 len) {
            std::array<u64, NUM_REGION_WORDS> consumed{};
            std::array<u64, NUM_REGION_WORDS> kept{};
            const auto mark_pages = [&](std::array<u64, NUM_REGION_WORDS>& pages, VAddr begin,
                                        VAddr end) {
                RegionManager::IterateWords(
                    RegionManager::GetBounds(begin - region_base, end - begin),
                    [&](u64 index, u64 mask) { pages[index] |= mask; });
            };
            boost::container::small_vector<Range, 32> erased;
            // The map lock stays held through the clear: a write marked meanwhile either shows
            // up here with a newer tick and keeps its pages, or finds the pages clear afterwards
            // and sets them again.
            std::scoped_lock lk{gpu_writes_mutex};
            gpu_writes.ForEachInRange(start, len, [&](VAddr begin, VAddr end, u64 tick) {
                if (tick <= known_tick) {
                    mark_pages(consumed, begin, end);
                    erased.push_back({begin, end - begin});
                } else {
                    mark_pages(kept, begin, end);
                }
            });
            for (const auto& [addr, erased_size] : erased) {
                gpu_writes.Subtract(addr, erased_size);
            }
            for (u64 index = 0; index < NUM_REGION_WORDS; ++index) {
                consumed[index] &= ~kept[index];
            }
            manager->ChangeRegionStateMasked<StateOp::None, StateOp::Clear>(consumed);
        });
}

bool BufferCache::WindowsOwned(VAddr device_addr, u64 size) const {
    return std::ranges::any_of(window_owners, [&](const Range& owner) {
        return owner.addr < device_addr + size && device_addr < owner.addr + owner.size;
    });
}

void BufferCache::AcquireWindows(VAddr device_addr, u64 size) {
    std::unique_lock lk{window_mutex};
    window_cv.wait(lk, [&] { return !WindowsOwned(device_addr, size); });
    window_owners.push_back({device_addr, size});
}

bool BufferCache::TryAcquireWindows(VAddr device_addr, u64 size) {
    std::scoped_lock lk{window_mutex};
    if (WindowsOwned(device_addr, size)) {
        return false;
    }
    window_owners.push_back({device_addr, size});
    return true;
}

void BufferCache::ReleaseWindows(VAddr device_addr, u64 size) {
    {
        std::scoped_lock lk{window_mutex};
        const auto it = std::ranges::find_if(window_owners, [&](const Range& owner) {
            return owner.addr == device_addr && owner.size == size;
        });
        ASSERT(it != window_owners.end());
        window_owners.erase(it);
    }
    window_cv.notify_all();
}

void BufferCache::WaitWindowsFree(VAddr device_addr, u64 size) {
    std::unique_lock lk{window_mutex};
    window_cv.wait(lk, [&] { return !WindowsOwned(device_addr, size); });
}

struct BufferCache::Readback {
    VAddr window{};
    ReadbackCopies copies{};
    Vulkan::StagingBufferRef staging{};
    VAddr arena_base{};
    u64 tick{};
    bool busy{};
};

bool BufferCache::FaultReadback(VAddr device_addr, u64 size, bool is_write) {
    const VAddr window = Common::AlignDown(device_addr, READBACK_WINDOW_SIZE);
    if (device_addr + size > window + READBACK_WINDOW_SIZE) {
        return false;
    }
    // Guest signals stay pending until the readback is done: other threads that fault on the
    // window wait for this one, and a guest handler could wait for them.
    Libraries::Kernel::Sigset all_signals;
    Libraries::Kernel::Sigset old_sigmask{};
    Libraries::Kernel::posix_sigfillset(&all_signals);
    Libraries::Kernel::posix_pthread_sigmask(POSIX_SIG_SETMASK, &all_signals, &old_sigmask);
    SCOPE_EXIT {
        Libraries::Kernel::posix_pthread_sigmask(POSIX_SIG_SETMASK, &old_sigmask, nullptr);
    };
    // A request still GPU modified after two readbacks of its own takes the synchronous download,
    // which copies and unmarks in one GPU thread command.
    u32 attempts = 0;
    while (attempts < 2) {
        if (DirectReadback(device_addr)) {
            ++attempts;
        } else {
            Readback job{.window = window};
            liverpool->SendCommand<true>([this, &job] { RecordReadback(job); });
            if (job.busy) {
                WaitWindowsFree(window, READBACK_WINDOW_SIZE);
            } else {
                ++attempts;
                if (!job.copies.empty()) {
                    scheduler.GetWorkSemaphore()->Wait(job.tick);
                    FinishReadback(job);
                }
            }
        }
        bool resolved = true;
        if (is_write) {
            // Marks the range CPU modified if it is GPU clean.
            memory_tracker->InvalidateRegion(device_addr, size, [&resolved] { resolved = false; });
        } else {
            resolved = !memory_tracker->IsRegionGpuModified(device_addr, size);
        }
        if (resolved) {
            return true;
        }
    }
    return false;
}

void BufferCache::RecordReadback(Readback& job) {
    if (!TryAcquireWindows(job.window, READBACK_WINDOW_SIZE)) {
        job.busy = true;
        return;
    }
    const u64 block = job.window >> block_shift;
    const auto* arena = GetArena(block, block);
    const u64 total_size_bytes =
        CollectWrites(arena, job.window, READBACK_WINDOW_SIZE, scheduler.CurrentTick(), job.copies);
    if (total_size_bytes == 0) {
        ReleaseWindows(job.window, READBACK_WINDOW_SIZE);
        return;
    }
    // Held until FinishReadback releases it; the pool reuses it once the GPU passes that release.
    job.staging = staging_pool.Request(total_size_bytes, MemoryType::HostCached, 0, true);
    for (auto& copy : job.copies) {
        copy.dstOffset += job.staging.offset;
    }
    job.arena_base = arena->cpu_addr;
    runtime.CopyBuffer(arena, job.staging.buffer, job.copies);
    job.tick = scheduler.CurrentTick();
    scheduler.Flush();
}

void BufferCache::FinishReadback(const Readback& job) {
    job.staging.Invalidate();
    // The write-back runs ahead of the consume: the window is owned, so a write marked meanwhile
    // keeps its pages GPU modified and the guest never reads them.
    WriteBack(job.arena_base, job.copies, job.staging.mapped, job.staging.offset);
    ConsumeGpuWrites(job.window, READBACK_WINDOW_SIZE, job.tick);
    ReleaseWindows(job.window, READBACK_WINDOW_SIZE);
    // The staging pool belongs to the GPU thread.
    liverpool->SendCommand([this, staging = job.staging] { staging_pool.FreeDeferred(staging); });
}

bool BufferCache::GpuThreadReadback(VAddr device_addr, u64 size, bool is_write) {
    if (!DirectReadback(device_addr)) {
        return false;
    }
    if (is_write) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
    }
    return true;
}

bool BufferCache::DirectReadback(VAddr device_addr) {
    const VAddr window = Common::AlignDown(device_addr, READBACK_WINDOW_SIZE);
    const VAddr page = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const bool on_gpu_thread =
        std::this_thread::get_id() == liverpool->GetGpuCommandProcessorThread();
    u64 page_tick = 0;
    {
        std::scoped_lock lk{gpu_writes_mutex};
        gpu_writes.ForEachInRange(page, BYTES_PER_PAGE, [&](VAddr, VAddr, u64 tick) {
            page_tick = std::max(page_tick, tick);
        });
    }
    if (page_tick == 0) {
        return false;
    }
    bool flushed = false;
    if (page_tick >= scheduler.CurrentTick()) {
        if (on_gpu_thread) {
            scheduler.Flush();
            flushed = true;
        } else {
            liverpool->SendCommand(
                [this, page_tick] { liverpool->GetRasterizer()->RequestFlush(page_tick); });
        }
    }
    scheduler.GetWorkSemaphore()->Wait(page_tick);
    // Owned only now, so the GPU thread never waits for a thread that waits for it.
    AcquireWindows(window, READBACK_WINDOW_SIZE);
    const u64 known = scheduler.GetWorkSemaphore()->KnownGpuTick();
    // After the GPU thread's own flush and wait every recorded write has executed, and the writes
    // of a packet still being bound were moved into the new batch, which marks them again at its
    // commit; their bytes are the arena's as the GPU left them, so the copy and the consume take
    // the current tick, like the synchronous download does.
    const u64 copy_tick = flushed ? scheduler.CurrentTick() : known;
    const VAddr window_end = window + READBACK_WINDOW_SIZE;
    bool copied = false;
    if (arena_host_type_index) {
        // The islands are collected and copied in batches, the list living on the signal stack.
        static constexpr size_t BatchIslands = 32;
        boost::container::small_vector<Range, BatchIslands> islands;
        copied = true;
        for (VAddr cursor = window; copied && cursor < window_end;) {
            islands.clear();
            VAddr next = window_end;
            {
                std::scoped_lock lk{gpu_writes_mutex};
                gpu_writes.ForEachInRange(cursor, window_end - cursor,
                                          [&](VAddr begin, VAddr end, u64 tick) {
                                              if (islands.size() == BatchIslands) {
                                                  next = std::min(next, begin);
                                                  return;
                                              }
                                              if (tick > copy_tick) {
                                                  return;
                                              }
                                              islands.push_back({begin, end - begin});
                                          });
            }
            for (auto it = islands.begin(); copied && it != islands.end(); ++it) {
                copied = DirectRead(it->addr, it->size);
            }
            cursor = next;
        }
    }
    if (copied) {
        ConsumeGpuWrites(window, READBACK_WINDOW_SIZE, copy_tick);
    }
    ReleaseWindows(window, READBACK_WINDOW_SIZE);
    return copied;
}

bool BufferCache::DirectRead(VAddr device_addr, u64 size) {
    while (size != 0) {
        const u8* src;
        u64 chunk;
        {
            std::scoped_lock lk{arena_mutex};
            const auto it = resident_ranges.Find(device_addr >> block_shift);
            if (it == resident_ranges.end() || !it->host) {
                return false;
            }
            src =
                it->host + (it->offset << block_shift) + (device_addr - (it->start << block_shift));
            chunk = std::min(size, (it->end << block_shift) - device_addr);
        }
        if (!memory->TryStreamBacking(std::bit_cast<u8*>(device_addr), src, chunk)) {
            return false;
        }
        device_addr += chunk;
        size -= chunk;
    }
    return true;
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        ObtainBufferFlags flags) {
    const bool is_written = True(flags & ObtainBufferFlags::IsWritten);
    const bool is_texel_buffer = True(flags & ObtainBufferFlags::IsTexelBuffer);
    const bool skip_stream_buffer = True(flags & ObtainBufferFlags::IgnoreStreamBuffer);
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && !skip_stream_buffer && size <= STREAM_THRESHOLD &&
        !IsRegionGpuModified(device_addr, size)) {
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    if (is_written) {
        texture_cache.MarkAsMaybeGpuDirty(device_addr, size);
        MarkGpuWrite(device_addr, size);
    }
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::MarkRegionAsGpuModified(VAddr addr, size_t size) {
    // The copy is recorded already, so its write belongs to the open batch.
    texture_cache.MarkAsMaybeGpuDirty(addr, size);
    StampGpuWrite(addr, size);
    memory_tracker->MarkRegionAsGpuModified(addr, size);
}

void BufferCache::StampGpuWrite(VAddr device_addr, u64 size) {
    const u64 tick = scheduler.CurrentTick();
    std::scoped_lock lk{gpu_writes_mutex};
    gpu_writes.Set(device_addr, size, tick);
}

void BufferCache::MarkGpuWrite(VAddr device_addr, u64 size) {
    StampGpuWrite(device_addr, size);
    in_flight.push_back({device_addr, size});
}

void BufferCache::CommitGpuWrites() {
    // A flush between the binds and the record moved the packet into the current batch, and a
    // synchronous download on this thread meanwhile consumed the writes it found: they are
    // stamped and marked again.
    if (in_flight_moved) {
        for (const auto& [addr, size] : in_flight) {
            StampGpuWrite(addr, size);
            memory_tracker->MarkRegionAsGpuModified(addr, size);
        }
        in_flight_moved = false;
    }
    in_flight.clear();
}

void BufferCache::MoveInFlightWrites() {
    if (in_flight.empty()) {
        return;
    }
    // The submit took the batch the stamps name; the packet records into the one being opened.
    for (const auto& [addr, size] : in_flight) {
        StampGpuWrite(addr, size);
    }
    in_flight_moved = true;
}

void BufferCache::ReadEdgeImagePages(const Image& image) {
    // May happen that after downloading the image and invalidating region,
    // that there were GPU modified ranges that are lost due to CPU reuploading.
    // This doesn't change tracker state and it is spected to call DownloadImageMemory after this.
    const VAddr image_addr = image.info.guest_address;
    const u64 image_size = image.info.guest_size;
    const VAddr image_end = image_addr + image_size;
    const VAddr page_start = PageManager::GetPageAddr(image_addr);
    const VAddr page_end = PageManager::GetNextPageAddr(image_end - 1);
    const auto [buffer, offset] = ObtainBuffer(page_start, page_end - page_start);
    ReadbackCopies copies;
    const u64 tick = scheduler.CurrentTick();
    CollectWrites(buffer, page_start, image_addr - page_start, tick, copies);
    const u64 total_size_bytes =
        CollectWrites(buffer, image_end, page_end - image_end, tick, copies);
    {
        std::scoped_lock lk{gpu_writes_mutex};
        gpu_writes.Subtract(page_start, page_end - page_start);
    }
    if (total_size_bytes == 0) {
        return;
    }
    const auto download =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 16, true);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += download.offset;
    }
    scheduler.EndRendering();
    runtime.CopyBuffer(buffer, download.buffer, copies);
    scheduler.DeferOperation(
        [this, base = buffer->CpuAddr(), download, copies = std::move(copies)]() {
            download.Invalidate();
            WriteBack(base, copies, download.mapped, download.offset);
            staging_pool.FreeDeferred(download);
        });
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            std::scoped_lock lk{arena_mutex};
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    std::scoped_lock lk{arena_mutex};
    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }

    const vk::MemoryAllocateFlagsInfo alloc_flags = {
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const bool host_type = arena_host_type_index && !arena_host_spent;
    vk::MemoryAllocateInfo alloc_info = {
        .pNext = &alloc_flags,
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = host_type ? *arena_host_type_index : arena_device_type_index,
    };
    const auto device = instance.GetDevice();
    auto allocation = device.allocateMemory(alloc_info);
    if (allocation.result != vk::Result::eSuccess && host_type &&
        *arena_host_type_index != arena_device_type_index) {
        LOG_WARNING(Render_Vulkan,
                    "Host visible arena memory spent, arenas continue in device memory");
        arena_host_spent = true;
        alloc_info.memoryTypeIndex = arena_device_type_index;
        allocation = device.allocateMemory(alloc_info);
    }
    const auto device_memory = Vulkan::Check(allocation);
    u8* host = nullptr;
    if (host_type && !arena_host_spent) {
        const auto [map_result, mapping] = device.mapMemory(device_memory, 0, vk::WholeSize);
        if (map_result == vk::Result::eSuccess) {
            host = static_cast<u8*>(mapping);
        } else {
            LOG_WARNING(Render_Vulkan, "Mapping arena memory failed: {}",
                        vk::to_string(map_result));
        }
    }

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;
        backing.host = host;
        {
            std::scoped_lock lk{arena_mutex};
            resident_ranges.Add(backing);
        }

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 size) {
        copies.emplace_back(total_size_bytes, addr, size);
        total_size_bytes += size;
    });
    if (!copies.empty()) {
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (const auto meta = texture_cache.GetMetaInfo(device_addr)) {
        if (meta->type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else if (meta->type == TextureCache::MetaType::CMask) {
            const auto expanded_value = CmaskColorExpandedValue(meta->num_samples);
            if (!expanded_value) {
                LOG_WARNING(Render_Vulkan,
                            "Unsupported CMask sample count: address={:#x}, size={:#x}, "
                            "samples={}, owner={}:{}",
                            device_addr, size, meta->num_samples, meta->owner_id.index,
                            meta->owner_uid);
                return false;
            }
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, *expanded_value);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}",
                        magic_enum::enum_name(meta->type));
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
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
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
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    std::scoped_lock lk{arena_mutex};
    pending_binds.clear();
}

} // namespace VideoCore
