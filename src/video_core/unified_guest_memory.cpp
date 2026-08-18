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
        LOG_WARNING(Render_Vulkan,
                    "Native UMA: heap {:#x} cannot cover backing {:#x} plus headroom; "
                    "raise the GTT size (amdgpu.gttsize) to enable",
                    chosen_heap, backing_size);
        return;
    }

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
        return;
    }

    // Overlapping buffer windows over the allocation: stride 3GB, span 4GB,
    // so any range up to 1GB fits entirely inside some window.
    const vk::ExternalMemoryBufferCreateInfo ext_buf{
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
    };
    constexpr vk::BufferUsageFlags usage =
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformTexelBuffer |
        vk::BufferUsageFlagBits::eStorageTexelBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eShaderDeviceAddress;

    std::vector<vk::UniqueBuffer> windows;
    for (u64 base = 0; base < backing_size; base += WindowStride) {
        const u64 span = std::min(WindowSpan, backing_size - base);
        const vk::BufferCreateInfo buf_info{
            .pNext = &ext_buf,
            .size = span,
            .usage = usage,
        };
        auto [buf_result, buf] = device.createBufferUnique(buf_info);
        if (buf_result != vk::Result::eSuccess) {
            LOG_WARNING(Render_Vulkan, "Native UMA: window creation failed: {}",
                        vk::to_string(buf_result));
            return;
        }
        const auto reqs = device.getBufferMemoryRequirements(*buf);
        if (!(reqs.memoryTypeBits & (1u << chosen_type)) || (base % reqs.alignment) != 0) {
            LOG_WARNING(Render_Vulkan, "Native UMA: window at {:#x} incompatible", base);
            return;
        }
        if (device.bindBufferMemory(*buf, *mem, base) != vk::Result::eSuccess) {
            LOG_WARNING(Render_Vulkan, "Native UMA: window bind at {:#x} failed", base);
            return;
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
        return;
    }
    if (!aspace.AdoptUnifiedBackingPrefix(fd, backing_size)) {
        LOG_WARNING(Render_Vulkan, "Native UMA: backing adoption failed");
        return;
    }

    memory = std::move(mem);
    buffers = std::move(windows);
    dmabuf_fd = fd;
    total_size = backing_size;
    active = true;
    LOG_INFO(Render_Vulkan, "Native UMA active: type {} covers all {:#x} guest physical bytes "
                            "through {} windows",
             chosen_type, total_size, buffers.size());
#else
    (void)instance;
#endif
}

UnifiedGuestMemory::~UnifiedGuestMemory() = default;

} // namespace VideoCore
