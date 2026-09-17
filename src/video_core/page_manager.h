// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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

    /// Opens a protect-carry scope on the calling thread (protect_carry_merge).
    /// Inside one, a per-region watcher update whose last run ends exactly at
    /// the region boundary keeps its region lock and defers its mprotect, so
    /// the next region's leading run can be issued as one cross-region call.
    /// ONLY legal on the GPU command thread, and only around loops that call
    /// nothing but UpdatePageWatchersForRegion: see the lock-order note at the
    /// carry site in page_manager.cpp. Callers certify that with an explicit
    /// flag, and the named sites are Rasterizer::DrainPendingReadArms for every
    /// ReadArmSite but Submit, plus BufferCache::FinishFaultDownload's
    /// PendingUnmark and DrainPendingReadReleases; everything else, the
    /// guest-thread DropPendingReadArms unmap route included, passes false.
    /// BeginProtectCarry asserts that only one thread ever opens a scope.
    /// Always paired through ProtectCarryScope.
    void BeginProtectCarry() const;
    void EndProtectCarry() const;

    struct ProtectCarryStats {
        u64 scopes;
        u64 merged;
        u64 flushed;
    };
    ProtectCarryStats DrainProtectCarryStats() const;

    /// Returns true if any page touched by [addr, addr + size) currently holds a
    /// write watcher (i.e. it is mapped PROT_READ and a guest store to it would
    /// fault). An address outside the tracked low 40 bits answers false.
    /// Diagnostic only: raced against concurrent arms/releases, never branched on.
    bool IsWriteWatched(VAddr addr, u64 size) const;

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

/// RAII protect-carry scope. Never open one with two bare calls: the carry
/// holds a page-manager lock, so any path that skipped the close would hang
/// the next guest fault in that region.
class ProtectCarryScope {
public:
    explicit ProtectCarryScope(const PageManager& pm_, bool enable_ = true)
        : pm{pm_}, enable{enable_} {
        if (enable) {
            pm.BeginProtectCarry();
        }
    }
    ~ProtectCarryScope() {
        if (enable) {
            pm.EndProtectCarry();
        }
    }
    ProtectCarryScope(const ProtectCarryScope&) = delete;
    ProtectCarryScope& operator=(const ProtectCarryScope&) = delete;

private:
    const PageManager& pm;
    bool enable;
};

} // namespace VideoCore
