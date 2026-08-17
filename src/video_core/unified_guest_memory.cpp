// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/unified_guest_memory.h"

namespace VideoCore {

UnifiedGuestMemory::UnifiedGuestMemory(const Vulkan::Instance& instance) {
#ifndef _WIN32
    if (EmulatorSettings.GetNativeUmaMode() == 0) {
        return;
    }
    if (!instance.IsExternalMemoryDmaBufSupported()) {
        LOG_WARNING(Render_Vulkan, "Native UMA requested but dma-buf export unsupported");
        return;
    }
    const auto device = instance.GetDevice();
    const auto physical = instance.GetPhysicalDevice();
    auto& aspace = Core::Memory::Instance()->GetAddressSpace();
    const u64 backing_size = aspace.GetBackingSize();

    // The buffer is created first so the memory can be dedicated to it.
    const vk::ExternalMemoryBufferCreateInfo ext_buf{
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
    };
    constexpr vk::BufferUsageFlags usage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformTexelBuffer |
        vk::BufferUsageFlagBits::eStorageTexelBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eShaderDeviceAddress;

    // Type policy (design 12.1/13): plain host-cached coherent memory. The
    // CPU reads guest memory constantly; WC mappings are disqualifying.
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

    // Cover as much of the backing as the heap allows, leaving headroom for
    // the driver's other GTT users. The tail stays memfd-backed (legacy path).
    constexpr u64 heap_headroom = 1_GB;
    const u64 heap_budget = chosen_heap > heap_headroom ? chosen_heap - heap_headroom : 0;
    covered_size = Common::AlignDown(std::min(backing_size, heap_budget), 64_KB);
    if (covered_size < 1_GB) {
        LOG_WARNING(Render_Vulkan, "Native UMA: heap too small ({:#x})", chosen_heap);
        covered_size = 0;
        return;
    }

    const vk::BufferCreateInfo buf_info{
        .pNext = &ext_buf,
        .size = covered_size,
        .usage = usage,
    };
    auto [buf_result, buf] = device.createBufferUnique(buf_info);
    if (buf_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Native UMA: buffer creation failed: {}",
                    vk::to_string(buf_result));
        covered_size = 0;
        return;
    }
    const auto reqs = device.getBufferMemoryRequirements(*buf);
    if (!(reqs.memoryTypeBits & (1u << chosen_type))) {
        LOG_WARNING(Render_Vulkan, "Native UMA: chosen type {} unusable for buffer", chosen_type);
        covered_size = 0;
        return;
    }

    const vk::MemoryDedicatedAllocateInfo dedicated{
        .buffer = *buf,
    };
    const vk::ExportMemoryAllocateInfo export_info{
        .pNext = &dedicated,
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
    };
    const vk::MemoryAllocateFlagsInfo flags_info{
        .pNext = &export_info,
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::MemoryAllocateInfo alloc_info{
        .pNext = &flags_info,
        .allocationSize = reqs.size,
        .memoryTypeIndex = static_cast<u32>(chosen_type),
    };
    auto [alloc_result, mem] = device.allocateMemoryUnique(alloc_info);
    if (alloc_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Native UMA: allocation of {:#x} failed: {}", covered_size,
                    vk::to_string(alloc_result));
        covered_size = 0;
        return;
    }
    if (device.bindBufferMemory(*buf, *mem, 0) != vk::Result::eSuccess) {
        covered_size = 0;
        return;
    }

    const vk::MemoryGetFdInfoKHR fd_info{
        .memory = *mem,
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
    };
    auto [fd_result, fd] = device.getMemoryFdKHR(fd_info);
    if (fd_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Native UMA: dma-buf export failed: {}",
                    vk::to_string(fd_result));
        covered_size = 0;
        return;
    }
    if (!aspace.AdoptUnifiedBackingPrefix(fd, covered_size)) {
        LOG_WARNING(Render_Vulkan, "Native UMA: backing adoption failed");
        covered_size = 0;
        return;
    }

    memory = std::move(mem);
    buffer = std::move(buf);
    dmabuf_fd = fd;
    active = true;
    LOG_INFO(Render_Vulkan,
             "Native UMA active: type {} covers {:#x} of {:#x} guest physical bytes", chosen_type,
             covered_size, backing_size);
#else
    (void)instance;
#endif
}

UnifiedGuestMemory::~UnifiedGuestMemory() = default;

} // namespace VideoCore
