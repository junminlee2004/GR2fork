// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

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

UnifiedGuestMemory::UnifiedGuestMemory(const Vulkan::Instance& instance) {
    if (EmulatorSettings.GetNativeUmaMode() == 0) {
        return;
    }
    const auto physical = instance.GetPhysicalDevice();
    auto& aspace = Core::Memory::Instance()->GetAddressSpace();
    const u64 backing_size = Common::AlignUp(aspace.GetBackingSize(), 64_KB);

    // Type policy: plain host-cached coherent memory. The CPU reads guest
    // memory constantly; write-combined mappings are disqualifying.
    const auto mem_props = physical.getMemoryProperties();
    s32 chosen_type = -1;
    u64 chosen_heap = 0;
    for (u32 i = 0; i < mem_props.memoryTypeCount; ++i) {
        const auto flags = mem_props.memoryTypes[i].propertyFlags;
        constexpr auto wanted = vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent |
                                vk::MemoryPropertyFlagBits::eHostCached;
        if ((flags & wanted) != wanted) {
            continue;
        }
        if (flags & vk::MemoryPropertyFlagBits::eDeviceCoherentAMD) {
            continue; // GPU-uncached; keep GPU-side caching by default.
        }
        const u64 heap = mem_props.memoryHeaps[mem_props.memoryTypes[i].heapIndex].size;
        if (chosen_type < 0 || heap > chosen_heap) {
            chosen_type = static_cast<s32>(i);
            chosen_heap = heap;
        }
    }
    if (chosen_type < 0) {
        LOG_WARNING(Render_Vulkan, "Native UMA: no host-cached coherent memory type");
        return;
    }

    // True UMA is all-or-nothing: partial coverage leaves a class of
    // GPU-written data the CPU can never see, which testing showed as
    // stale simulation feedback. The whole backing must fit in the heap.
    constexpr u64 heap_headroom = 1_GB;
    if (chosen_heap < backing_size + heap_headroom) {
        LOG_CRITICAL(Render_Vulkan,
                     "Native UMA requested but NOT ACTIVE: heap {:#x} cannot cover backing "
                     "{:#x} plus headroom. Raise the GTT size (e.g. amdgpu.gttsize=12288 "
                     "kernel argument) to enable; running without unified memory.",
                     chosen_heap, backing_size);
        return;
    }

    // Two mechanisms produce the same windowed geometry. Exporting device
    // memory over the backing keeps GPU-side allocation flags ideal and is
    // preferred where dma-buf exists; importing the backing as host memory
    // needs no allocation-size headroom and no address-space adoption.
    if (instance.IsExternalMemoryDmaBufSupported() &&
        AdoptDmaBuf(instance, backing_size, chosen_type)) {
        active = true;
    } else if (instance.IsExternalMemoryHostSupported() &&
               ImportHostBacking(instance, backing_size)) {
        active = true;
    } else {
        LOG_WARNING(Render_Vulkan, "Native UMA: no usable unified memory mechanism");
        return;
    }
    total_size = backing_size;
    LOG_INFO(Render_Vulkan,
             "Native UMA active ({}): covers all {:#x} guest physical bytes through {} windows",
             dmabuf_fd >= 0 ? "dma-buf" : "import-host", total_size, buffers.size());
}

bool UnifiedGuestMemory::AdoptDmaBuf(const Vulkan::Instance& instance, u64 backing_size,
                                     s32 chosen_type) {
#ifndef _WIN32
    const auto device = instance.GetDevice();
    auto& aspace = Core::Memory::Instance()->GetAddressSpace();

    // One allocation spans the whole backing. maxMemoryAllocationSize is
    // only the guaranteed bound; a larger request fails cleanly if the
    // driver cannot honor it.
    const vk::ExportMemoryAllocateInfo export_info{
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
    };
    const vk::MemoryAllocateFlagsInfo flags_info{
        .pNext = &export_info,
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::MemoryAllocateInfo alloc_info{
        .pNext = &flags_info,
        .allocationSize = backing_size,
        .memoryTypeIndex = static_cast<u32>(chosen_type),
    };
    auto [alloc_result, mem] = device.allocateMemoryUnique(alloc_info);
    if (alloc_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Native UMA: full-backing allocation of {:#x} failed: {}",
                    backing_size, vk::to_string(alloc_result));
        return false;
    }

    // Overlapping buffer windows over the allocation: stride 3GB, span 4GB,
    // so any range up to 1GB fits entirely inside some window.
    const vk::ExternalMemoryBufferCreateInfo ext_buf{
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
    };
    std::vector<vk::UniqueBuffer> windows;
    for (u64 base = 0; base < backing_size; base += WindowStride) {
        const u64 span = std::min(WindowSpan, backing_size - base);
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
        if (!(reqs.memoryTypeBits & (1u << chosen_type)) || (base % reqs.alignment) != 0) {
            LOG_WARNING(Render_Vulkan, "Native UMA: window at {:#x} incompatible", base);
            return false;
        }
        if (device.bindBufferMemory(*buf, *mem, base) != vk::Result::eSuccess) {
            LOG_WARNING(Render_Vulkan, "Native UMA: window bind at {:#x} failed", base);
            return false;
        }
        windows.push_back(std::move(buf));
        if (base + span >= backing_size) {
            break;
        }
    }

    const vk::MemoryGetFdInfoKHR fd_info{
        .memory = *mem,
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
    };
    auto [fd_result, fd] = device.getMemoryFdKHR(fd_info);
    if (fd_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Native UMA: dma-buf export failed: {}",
                    vk::to_string(fd_result));
        return false;
    }
    if (!aspace.AdoptUnifiedBackingPrefix(fd, backing_size)) {
        LOG_WARNING(Render_Vulkan, "Native UMA: backing adoption failed");
        return false;
    }
    memory = std::move(mem);
    buffers = std::move(windows);
    dmabuf_fd = fd;
    return true;
#else
    (void)instance;
    (void)backing_size;
    (void)chosen_type;
    return false;
#endif
}

bool UnifiedGuestMemory::ImportHostBacking(const Vulkan::Instance& instance, u64 backing_size) {
    const auto device = instance.GetDevice();
    auto& aspace = Core::Memory::Instance()->GetAddressSpace();
    u8* const backing_base = aspace.BackingBase();

    // Each window imports its own overlapping slice of the existing guest
    // backing: the memory is already host-owned and CPU-coherent, so no
    // address-space adoption is involved and imports may overlap freely.
    std::vector<vk::UniqueDeviceMemory> memories;
    std::vector<vk::UniqueBuffer> windows;
    for (u64 base = 0; base < backing_size; base += WindowStride) {
        const u64 span = std::min(WindowSpan, backing_size - base);
        const vk::ExternalMemoryBufferCreateInfo ext_buf{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        };
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

        const auto [props_result, host_props] = device.getMemoryHostPointerPropertiesEXT(
            vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, backing_base + base);
        if (props_result != vk::Result::eSuccess) {
            LOG_WARNING(Render_Vulkan, "Native UMA: host pointer query failed: {}",
                        vk::to_string(props_result));
            return false;
        }
        // The imported type must satisfy the buffer and stay CPU-cached;
        // imports never take write-combined host views, so host-visible
        // coherent suffices.
        const auto host_mem_props = instance.GetPhysicalDevice().getMemoryProperties();
        s32 import_type = -1;
        for (u32 i = 0; i < host_mem_props.memoryTypeCount; ++i) {
            constexpr auto wanted = vk::MemoryPropertyFlagBits::eHostVisible |
                                    vk::MemoryPropertyFlagBits::eHostCoherent;
            if (!(host_props.memoryTypeBits & (1u << i)) || !(reqs.memoryTypeBits & (1u << i))) {
                continue;
            }
            if ((host_mem_props.memoryTypes[i].propertyFlags & wanted) != wanted) {
                continue;
            }
            import_type = static_cast<s32>(i);
            if (host_mem_props.memoryTypes[i].propertyFlags &
                vk::MemoryPropertyFlagBits::eHostCached) {
                break;
            }
        }
        if (import_type < 0) {
            LOG_WARNING(Render_Vulkan, "Native UMA: no import type for window at {:#x}", base);
            return false;
        }

        const vk::ImportMemoryHostPointerInfoEXT import_info{
            .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
            .pHostPointer = backing_base + base,
        };
        const vk::MemoryAllocateFlagsInfo flags_info{
            .pNext = &import_info,
            .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
        };
        const vk::MemoryAllocateInfo alloc_info{
            .pNext = &flags_info,
            .allocationSize = span,
            .memoryTypeIndex = static_cast<u32>(import_type),
        };
        auto [alloc_result, mem] = device.allocateMemoryUnique(alloc_info);
        if (alloc_result != vk::Result::eSuccess) {
            LOG_WARNING(Render_Vulkan, "Native UMA: import of window at {:#x} failed: {}", base,
                        vk::to_string(alloc_result));
            return false;
        }
        if (device.bindBufferMemory(*buf, *mem, 0) != vk::Result::eSuccess) {
            LOG_WARNING(Render_Vulkan, "Native UMA: window bind at {:#x} failed", base);
            return false;
        }
        memories.push_back(std::move(mem));
        windows.push_back(std::move(buf));
        if (base + span >= backing_size) {
            break;
        }
    }
    window_memories = std::move(memories);
    buffers = std::move(windows);
    return true;
}

UnifiedGuestMemory::~UnifiedGuestMemory() = default;

} // namespace VideoCore
