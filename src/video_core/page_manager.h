// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include "common/alignment.h"
#include "common/types.h"
#include "video_core/buffer_cache//region_definitions.h"

namespace Vulkan {
class Rasterizer;
}

namespace VideoCore {

class PageManager {
    // PAGE_SIZE and PAGE_BITS conflicts with machine/param.h definitions on freebsd!
    // Use the same page size as the tracker.
    static constexpr size_t PM_PAGE_BITS = TRACKER_PAGE_BITS;
    static constexpr size_t PM_PAGE_SIZE = TRACKER_BYTES_PER_PAGE;

    // Keep the lock granularity the same as region granularity. (since each regions has
    // itself a lock)
    static constexpr size_t PAGES_PER_LOCK = NUM_PAGES_PER_REGION;

public:
    explicit PageManager(Vulkan::Rasterizer* rasterizer);
    ~PageManager();

    /// Register a range of mapped gpu memory.
    void OnGpuMap(VAddr address, size_t size);

    /// Unregister a range of gpu memory that was unmapped.
    void OnGpuUnmap(VAddr address, size_t size);

    /// Updates watches in the pages touching the specified region.
    template <bool track>
    void UpdatePageWatchers(VAddr addr, u64 size) const;

    /// Updates watches in the pages touching the specified region using a
    /// mask; returns the number of protection calls issued.
    template <bool track, bool is_read = false>
    u32 UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) const;

    /// guest_protect_handoff: the protection calls of one region update,
    /// planned under the region's locks and issued after the tracker's lock
    /// is released. The page-manager lock stays held from the plan to the
    /// last call, so the calls of one region still land in tracker order.
    /// Fixed capacity and no allocation: the plan lives on a signal stack.
    struct ProtectPlan {
        static constexpr u32 Capacity = 8;
        struct Range {
            VAddr addr;
            u64 size;
            u32 perms;
        };
        std::array<Range, Capacity> ranges;
        u32 count = 0;
        u32 inline_calls = 0; // issued under both locks: the plan was full
        size_t lock_index = 0;
        bool held = false;
    };
    /// UpdatePageWatchersForRegion that fills a plan instead of protecting.
    template <bool track, bool is_read = false>
    u32 PlanPageWatchersForRegion(VAddr base_addr, RegionBits& mask, ProtectPlan& plan) const;
    /// Issues a plan's calls and releases the lock it holds. Empty plans are
    /// free.
    void IssueProtectPlan(ProtectPlan& plan) const noexcept;
    /// Waits out every plan in flight on the regions of a range.
    void SyncProtect(VAddr addr, u64 size) const;

    /// Returns page aligned address.
    static constexpr VAddr GetPageAddr(VAddr addr) {
        return Common::AlignDown(addr, PM_PAGE_SIZE);
    }

    /// Returns address of the next page.
    static constexpr VAddr GetNextPageAddr(VAddr addr) {
        return Common::AlignUp(addr + 1, PM_PAGE_SIZE);
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace VideoCore
