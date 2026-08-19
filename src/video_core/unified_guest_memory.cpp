// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>

#include "common/alignment.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/unified_guest_memory.h"

namespace VideoCore {

constexpr vk::BufferUsageFlags WindowUsage =
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformTexelBuffer |
    vk::BufferUsageFlagBits::eStorageTexelBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eShaderDeviceAddress;

// Device memory the renderer itself needs from the carve heap for images,
// the swapchain and pipelines.
constexpr u64 RendererCarveReserve = 2_GB;
constexpr u64 OnionHeapHeadroom = 256_MB;
// Direct memory kept out of the garlic pool so the game's own CPU-side
// (WB_ONION) allocations land on cached pages instead of spilling onto
// write-combined ones.
constexpr u64 OnionDirectReserve = 1_GB + 512_MB;

UnifiedGuestMemory::UnifiedGuestMemory(const Vulkan::Instance& instance) {
#ifdef _WIN32
    (void)instance;
    return;
#else
    if (EmulatorSettings.GetNativeUmaMode() == 0) {
        return;
    }
    if (!instance.IsExternalMemoryDmaBufSupported()) {
        LOG_WARNING(Render_Vulkan, "Native UMA requested but dma-buf export unsupported");
        return;
    }
    const auto device = instance.GetDevice();
    const auto physical = instance.GetPhysicalDevice();
    auto* memory_manager = Core::Memory::Instance();
    auto& aspace = memory_manager->GetAddressSpace();

    // The provider is created at Gnm initialization, after the game's
    // memory regions are configured, so these are the real totals the
    // title can address - not the dev-kit maximum.
    const u64 direct_total = Common::AlignUp(memory_manager->GetConfiguredDirectSize(), 64_KB);
    const PAddr flex_base = memory_manager->GetFlexibleBase();
    const u64 flex_size = Common::AlignUp(memory_manager->GetTotalFlexibleSize(), 64_KB);

    // Type policy. Onion: plain host-cached coherent - the CPU reads it
    // constantly. Garlic: device-local carve memory; its CPU view will be
    // write-combined, which is faithful to the console's GPU bus.
    const auto mem_props = physical.getMemoryProperties();
    s32 onion_type = -1;
    u64 onion_heap = 0;
    s32 garlic_type = -1;
    u64 garlic_heap = 0;
    for (u32 i = 0; i < mem_props.memoryTypeCount; ++i) {
        const auto flags = mem_props.memoryTypes[i].propertyFlags;
        if (flags & vk::MemoryPropertyFlagBits::eDeviceCoherentAMD) {
            continue; // GPU-uncached variants of either heap.
        }
        const u64 heap = mem_props.memoryHeaps[mem_props.memoryTypes[i].heapIndex].size;
        constexpr auto onion_wanted = vk::MemoryPropertyFlagBits::eHostVisible |
                                      vk::MemoryPropertyFlagBits::eHostCoherent |
                                      vk::MemoryPropertyFlagBits::eHostCached;
        if ((flags & onion_wanted) == onion_wanted) {
            if (onion_type < 0 || heap > onion_heap) {
                onion_type = static_cast<s32>(i);
                onion_heap = heap;
            }
        } else if ((flags & vk::MemoryPropertyFlagBits::eDeviceLocal) &&
                   (flags & vk::MemoryPropertyFlagBits::eHostVisible) &&
                   !(flags & vk::MemoryPropertyFlagBits::eHostCached)) {
            // Host-visible carve memory: the CPU view is write-combined
            // (garlic semantics) and, critically, the driver keeps the
            // allocation CPU-accessible so its dma-buf can be mapped.
            if (garlic_type < 0 || heap > garlic_heap) {
                garlic_type = static_cast<s32>(i);
                garlic_heap = heap;
            }
        }
    }
    if (onion_type < 0) {
        LOG_WARNING(Render_Vulkan, "Native UMA: no host-cached coherent memory type");
        return;
    }

    // Placement asymmetry decides the default: a garlic-typed allocation
    // served from cached memory is merely stricter than asked, but an
    // onion-typed one served from write-combined memory breaks the guest's
    // ordering expectations - its plain stores sit in write-combining
    // buffers while the GPU reads through. Games size those two classes as
    // they please, so any fixed split can strand onion demand on the wrong
    // pages. Default therefore keeps every guest page cached; the carve
    // split is opt-in (native_uma bit 2) for placement experiments.
    const bool want_garlic = (EmulatorSettings.GetNativeUmaMode() & 4) != 0;
    u64 garlic_pool = 0;
    if (want_garlic && garlic_type >= 0 && garlic_heap > RendererCarveReserve &&
        direct_total > OnionDirectReserve) {
        // Halve direct memory at most: the onion side must be able to hold
        // the title's whole cached-memory demand, which is unknowable here.
        garlic_pool =
            Common::AlignDown(std::min({direct_total / 2, direct_total - OnionDirectReserve,
                                        garlic_heap - RendererCarveReserve}),
                              64_KB);
    }
    const u64 onion_pool = (direct_total - garlic_pool) + flex_size;
    if (onion_heap < onion_pool + OnionHeapHeadroom) {
        LOG_CRITICAL(Render_Vulkan,
                     "Native UMA requested but NOT ACTIVE: host-cached heap {:#x} cannot hold "
                     "the onion pool {:#x} plus headroom (direct {:#x}, garlic split {:#x}, "
                     "flexible {:#x}); running without unified memory.",
                     onion_heap, onion_pool, direct_total, garlic_pool, flex_size);
        return;
    }

    // One export allocation per pool, overlapping 4GB windows at 3GB
    // stride over each so any range up to 1GB resolves inside one window.
    const auto allocate_pool = [&](u64 pool_size, s32 type_index,
                                   vk::UniqueDeviceMemory& out) -> bool {
        const vk::ExportMemoryAllocateInfo export_info{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
        };
        const vk::MemoryAllocateFlagsInfo flags_info{
            .pNext = &export_info,
            .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
        };
        const vk::MemoryAllocateInfo alloc_info{
            .pNext = &flags_info,
            .allocationSize = pool_size,
            .memoryTypeIndex = static_cast<u32>(type_index),
        };
        auto [result, mem] = device.allocateMemoryUnique(alloc_info);
        if (result != vk::Result::eSuccess) {
            LOG_WARNING(Render_Vulkan, "Native UMA: pool allocation of {:#x} failed: {}", pool_size,
                        vk::to_string(result));
            return false;
        }
        out = std::move(mem);
        return true;
    };
    const auto add_windows = [&](vk::DeviceMemory mem, u64 pool_size, u64 linear_base,
                                 s32 type_index) -> bool {
        const vk::ExternalMemoryBufferCreateInfo ext_buf{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
        };
        for (u64 base = 0; base < pool_size; base += WindowStride) {
            const u64 span = std::min(WindowSpan, pool_size - base);
            const vk::BufferCreateInfo buf_info{
                .pNext = &ext_buf,
                .size = span,
                .usage = WindowUsage,
            };
            auto [buf_result, buf] = device.createBufferUnique(buf_info);
            if (buf_result != vk::Result::eSuccess) {
                LOG_WARNING(Render_Vulkan, "Native UMA: window creation failed: {}",
                            vk::to_string(buf_result));
                return false;
            }
            const auto reqs = device.getBufferMemoryRequirements(*buf);
            if (!(reqs.memoryTypeBits & (1u << type_index)) || (base % reqs.alignment) != 0) {
                LOG_WARNING(Render_Vulkan, "Native UMA: window at {:#x} incompatible", base);
                return false;
            }
            if (device.bindBufferMemory(*buf, mem, base) != vk::Result::eSuccess) {
                LOG_WARNING(Render_Vulkan, "Native UMA: window bind at {:#x} failed", base);
                return false;
            }
            windows.push_back({std::move(buf), linear_base + base, span});
            if (base + span >= pool_size) {
                break;
            }
        }
        return true;
    };
    const auto export_fd = [&](vk::DeviceMemory mem, int& out) -> bool {
        const vk::MemoryGetFdInfoKHR fd_info{
            .memory = mem,
            .handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
        };
        auto [result, fd] = device.getMemoryFdKHR(fd_info);
        if (result != vk::Result::eSuccess) {
            LOG_WARNING(Render_Vulkan, "Native UMA: dma-buf export failed: {}",
                        vk::to_string(result));
            return false;
        }
        out = fd;
        return true;
    };

    if (garlic_pool != 0) {
        if (!allocate_pool(garlic_pool, garlic_type, garlic_memory) ||
            !add_windows(*garlic_memory, garlic_pool, 0, garlic_type) ||
            !export_fd(*garlic_memory, garlic_fd)) {
            windows.clear();
            garlic_memory.reset();
            garlic_pool = 0;
        }
    }
    u64 recomputed_onion = (direct_total - garlic_pool) + flex_size;
    if (!allocate_pool(recomputed_onion, onion_type, onion_memory) ||
        !add_windows(*onion_memory, recomputed_onion, garlic_pool, onion_type) ||
        !export_fd(*onion_memory, onion_fd)) {
        if (garlic_pool != 0 || garlic_type < 0 || garlic_heap <= RendererCarveReserve) {
            return;
        }
        // A single allocation spanning all of guest memory exceeded what the
        // driver grants. Fall back to the split: the carve takes the lower
        // half, which the strict placement rule reserves for write-combined
        // requests, and the cached pool covers the rest.
        LOG_WARNING(Render_Vulkan,
                    "Native UMA: single cached pool of {:#x} rejected; falling back to a "
                    "carve split",
                    recomputed_onion);
        windows.clear();
        onion_memory.reset();
        onion_fd = -1;
        garlic_pool = Common::AlignDown(
            std::min(direct_total / 2, garlic_heap - RendererCarveReserve), 64_KB);
        recomputed_onion = (direct_total - garlic_pool) + flex_size;
        if (!allocate_pool(garlic_pool, garlic_type, garlic_memory) ||
            !add_windows(*garlic_memory, garlic_pool, 0, garlic_type) ||
            !export_fd(*garlic_memory, garlic_fd) ||
            !allocate_pool(recomputed_onion, onion_type, onion_memory) ||
            !add_windows(*onion_memory, recomputed_onion, garlic_pool, onion_type) ||
            !export_fd(*onion_memory, onion_fd)) {
            return;
        }
    }

    // Reback the guest physical regions: garlic prefix, the rest of direct
    // memory, and the flexible band - the latter two from the onion pool.
    std::vector<Core::AddressSpace::UnifiedRegion> regions;
    if (garlic_pool != 0) {
        regions.push_back({0, garlic_pool, garlic_fd, 0});
    }
    if (direct_total > garlic_pool) {
        regions.push_back({garlic_pool, direct_total - garlic_pool, onion_fd, 0});
    }
    regions.push_back({flex_base, flex_size, onion_fd, direct_total - garlic_pool});
    if (!aspace.AdoptUnifiedRegions(regions)) {
        if (garlic_pool == 0) {
            LOG_WARNING(Render_Vulkan, "Native UMA: backing adoption failed");
            return;
        }
        // The carve export could not be CPU-mapped on this driver; retry
        // with everything in the onion pool when the heap allows it.
        LOG_WARNING(Render_Vulkan, "Native UMA: garlic adoption failed; retrying onion-only");
        windows.clear();
        garlic_memory.reset();
        onion_memory.reset();
        garlic_fd = -1;
        onion_fd = -1;
        garlic_pool = 0;
        const u64 all_onion = direct_total + flex_size;
        if (onion_heap < all_onion + OnionHeapHeadroom) {
            LOG_CRITICAL(Render_Vulkan,
                         "Native UMA requested but NOT ACTIVE: host-cached heap {:#x} cannot "
                         "hold the full onion pool {:#x}; running without unified memory.",
                         onion_heap, all_onion);
            return;
        }
        if (!allocate_pool(all_onion, onion_type, onion_memory) ||
            !add_windows(*onion_memory, all_onion, 0, onion_type) ||
            !export_fd(*onion_memory, onion_fd)) {
            return;
        }
        regions.clear();
        regions.push_back({0, direct_total, onion_fd, 0});
        regions.push_back({flex_base, flex_size, onion_fd, direct_total});
        if (!aspace.AdoptUnifiedRegions(regions)) {
            LOG_WARNING(Render_Vulkan, "Native UMA: backing adoption failed");
            return;
        }
    }
    memory_manager->SetUnifiedGarlicSplit(garlic_pool);

    garlic_size = garlic_pool;
    direct_begin = garlic_pool;
    direct_end = direct_total;
    flex_begin = flex_base;
    flex_end = flex_base + flex_size;
    total_size = garlic_pool + recomputed_onion;
    active = true;
    LOG_INFO(Render_Vulkan,
             "Native UMA active: garlic {:#x} (carve) + onion {:#x} (cached) cover direct "
             "{:#x} and flexible {:#x} through {} windows",
             garlic_pool, recomputed_onion, direct_total, flex_size, windows.size());
#endif
}

UnifiedGuestMemory::~UnifiedGuestMemory() = default;

} // namespace VideoCore
