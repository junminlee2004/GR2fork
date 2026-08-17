// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
}

namespace VideoCore {

/// Native UMA: rebacks a prefix of guest physical memory with exported
/// Vulkan device memory so CPU and GPU share the same bytes. Buffers whose
/// physical range falls inside the covered prefix can be bound directly
/// through Handle() at their physical offset, with hardware coherence.
class UnifiedGuestMemory {
public:
    explicit UnifiedGuestMemory(const Vulkan::Instance& instance);
    ~UnifiedGuestMemory();

    UnifiedGuestMemory(const UnifiedGuestMemory&) = delete;
    UnifiedGuestMemory& operator=(const UnifiedGuestMemory&) = delete;

    [[nodiscard]] bool IsActive() const noexcept {
        return active;
    }

    [[nodiscard]] u64 CoveredSize() const noexcept {
        return covered_size;
    }

    [[nodiscard]] vk::Buffer Handle() const noexcept {
        return *buffer;
    }

private:
    bool active{};
    u64 covered_size{};
    vk::UniqueDeviceMemory memory;
    vk::UniqueBuffer buffer;
    int dmabuf_fd{-1};
};

} // namespace VideoCore
