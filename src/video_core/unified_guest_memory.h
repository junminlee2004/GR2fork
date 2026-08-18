// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
}

namespace VideoCore {

/// Native UMA: rebacks all of guest physical memory with one exported
/// Vulkan device allocation so CPU and GPU share the same bytes. Buffers
/// cannot span the whole allocation (driver buffer size limits), so
/// overlapping windows are bound over it: any physical range up to the
/// overlap size fits entirely inside some window, leaving no seams.
class UnifiedGuestMemory {
public:
    static constexpr u64 WindowStride = 3_GB;
    static constexpr u64 WindowSpan = 4_GB;
    static constexpr u64 MaxRange = WindowSpan - WindowStride;

    explicit UnifiedGuestMemory(const Vulkan::Instance& instance);
    ~UnifiedGuestMemory();

    UnifiedGuestMemory(const UnifiedGuestMemory&) = delete;
    UnifiedGuestMemory& operator=(const UnifiedGuestMemory&) = delete;

    [[nodiscard]] bool IsActive() const noexcept {
        return active;
    }

    /// Total guest physical bytes covered; equals the backing size when active.
    [[nodiscard]] u64 TotalSize() const noexcept {
        return total_size;
    }

    [[nodiscard]] u32 NumWindows() const noexcept {
        return static_cast<u32>(buffers.size());
    }

    [[nodiscard]] vk::Buffer WindowHandle(u32 window) const noexcept {
        return *buffers[window];
    }

    [[nodiscard]] u64 WindowBase(u32 window) const noexcept {
        return window * WindowStride;
    }

    [[nodiscard]] u64 WindowSize(u32 window) const noexcept {
        return std::min(WindowSpan, total_size - WindowBase(window));
    }

    /// Resolves a physical range to (window, offset inside the window).
    /// Ranges up to MaxRange always resolve; larger ones may not and must
    /// be clamped by the caller.
    [[nodiscard]] std::optional<std::pair<u32, u64>> Resolve(PAddr phys, u64 size) const noexcept {
        if (!active || phys + size > total_size) {
            return std::nullopt;
        }
        u32 window = static_cast<u32>(phys / WindowStride);
        if (window >= buffers.size()) {
            window = static_cast<u32>(buffers.size()) - 1;
        }
        if (phys - WindowBase(window) + size <= WindowSize(window)) {
            return std::make_pair(window, phys - WindowBase(window));
        }
        if (window > 0 && phys - WindowBase(window - 1) + size <= WindowSize(window - 1)) {
            return std::make_pair(window - 1, phys - WindowBase(window - 1));
        }
        return std::nullopt;
    }

private:
    bool active{};
    u64 total_size{};
    vk::UniqueDeviceMemory memory;
    std::vector<vk::UniqueBuffer> buffers;
    int dmabuf_fd{-1};
};

} // namespace VideoCore
