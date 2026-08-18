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

/// Native UMA: rebacks the guest physical memory a game can actually
/// address with exported Vulkan device memory, split PS4-faithfully into
/// two pools over the same bytes the game sees:
///  - garlic (device-local, CPU write-combined): the GPU bus. Games place
///    GPU-hot data here by declared memory type and never hot-read it from
///    the CPU - exactly the carve's semantics.
///  - onion (host-cached coherent): the CPU bus, holding CPU-feedback data
///    and the flexible band.
/// Buffers cannot span a whole pool (driver buffer size limits), so
/// overlapping windows are bound per pool: any range up to the overlap
/// size fits entirely inside some window, leaving no seams within a pool.
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

    /// Size of the linear hazard-tracking space (sum of pool sizes).
    [[nodiscard]] u64 TotalSize() const noexcept {
        return total_size;
    }

    [[nodiscard]] u32 NumWindows() const noexcept {
        return static_cast<u32>(windows.size());
    }

    [[nodiscard]] vk::Buffer WindowHandle(u32 window) const noexcept {
        return *windows[window].buffer;
    }

    /// Base of the window inside the linear hazard space; collision-free
    /// across pools, so WindowBase + offset keys the hazard tracker.
    [[nodiscard]] u64 WindowBase(u32 window) const noexcept {
        return windows[window].linear_base;
    }

    [[nodiscard]] u64 WindowSize(u32 window) const noexcept {
        return windows[window].size;
    }

    /// Resolves a physical range to (window, offset inside the window).
    /// Fails for unmapped physical space, ranges crossing the pool split,
    /// and ranges beyond MaxRange that no window can hold.
    [[nodiscard]] std::optional<std::pair<u32, u64>> Resolve(PAddr phys, u64 size) const noexcept {
        // Segment-relative linear offset for the physical address; ranges
        // never span segments (the split and the flexible band are not
        // physically adjacent to their neighbors from the guest's view).
        u64 linear{};
        if (phys + size <= garlic_size) {
            linear = phys;
        } else if (phys >= direct_begin && phys + size <= direct_end) {
            linear = garlic_size + (phys - direct_begin);
        } else if (phys >= flex_begin && phys + size <= flex_end) {
            linear = garlic_size + (direct_end - direct_begin) + (phys - flex_begin);
        } else {
            return std::nullopt;
        }
        for (u32 w = 0; w < windows.size(); ++w) {
            if (linear >= windows[w].linear_base &&
                linear - windows[w].linear_base + size <= windows[w].size) {
                return std::make_pair(w, linear - windows[w].linear_base);
            }
        }
        return std::nullopt;
    }

private:
    struct Window {
        vk::UniqueBuffer buffer;
        u64 linear_base;
        u64 size;
    };

    bool active{};
    u64 total_size{};
    u64 garlic_size{};
    PAddr direct_begin{};
    PAddr direct_end{};
    PAddr flex_begin{};
    PAddr flex_end{};
    vk::UniqueDeviceMemory garlic_memory;
    vk::UniqueDeviceMemory onion_memory;
    std::vector<vk::UniqueDeviceMemory> import_memories;
    std::vector<Window> windows;
    int garlic_fd{-1};
    int onion_fd{-1};
};

} // namespace VideoCore
