// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <deque>
#include <vector>

#include "common/alignment.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_manager.h"

namespace VideoCore {

class MemoryTracker {
public:
    static constexpr u64 MAX_CPU_PAGE_BITS = 40;
    static constexpr u64 NUM_HIGH_PAGES = 1ULL << (MAX_CPU_PAGE_BITS - HIGHER_PAGE_BITS);
    static constexpr u64 MANAGER_POOL_SIZE = 32;

public:
    explicit MemoryTracker(PageManager& tracker_)
        : tracker{&tracker_}, readbacks_mode{EmulatorSettings.GetReadbacksMode()} {}
    ~MemoryTracker() = default;

    /// Returns true if a region has been modified from the GPU
    bool IsRegionGpuModified(VAddr cpu_addr, u64 size) noexcept {
        return IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            return manager->template IsRegionModified<Type::GPU>(offset, size);
        });
    }

    /// Returns true if a region has been modified from the CPU
    bool IsRegionCpuModified(VAddr cpu_addr, u64 size) noexcept {
        return IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            return manager->template IsRegionModified<Type::CPU>(offset, size);
        });
    }

    /// Mark region as modified from the host GPU without uploading it again
    void MarkRegionAsGpuModified(VAddr cpu_addr, u64 size) noexcept {
        IteratePages<true>(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            manager->template ChangeRegionState<StateOp::None, StateOp::Set>(offset, size);
        });
    }

    /// Unmark region as modified from the host GPU and erase its GPU writes
    void UnmarkRegionAsGpuModified(VAddr cpu_addr, u64 size, bool is_write) noexcept {
        IteratePages(cpu_addr, size, [is_write](RegionManager* manager, u64 offset, u64 size) {
            if (is_write) {
                manager->template UnmarkGpuModified<StateOp::Set>(offset, size);
            } else {
                manager->template UnmarkGpuModified<StateOp::None>(offset, size);
            }
        });
    }

    /// Stamps a GPU written range with the tick of the batch holding the write.
    void StampGpuWrite(VAddr cpu_addr, u64 size, u64 tick) noexcept {
        IteratePages<true>(cpu_addr, size, [tick](RegionManager* manager, u64 offset, u64 size) {
            manager->StampGpuWrite(offset, size, tick);
        });
    }

    /// Calls func(begin, end, tick) for each GPU write of the range, one region lock at a time.
    void ForEachGpuWrite(VAddr cpu_addr, u64 size, auto&& func) {
        IteratePages(cpu_addr, size, [&func](RegionManager* manager, u64 offset, u64 size) {
            manager->ForEachGpuWrite(offset, size, func);
        });
    }

    /// Erases the GPU writes of the range stamped up to known_tick and clears the pages left
    /// without a newer one. The pages are cleared whole, so the writes are erased whole as well.
    void ConsumeGpuWrites(VAddr cpu_addr, u64 size, u64 known_tick) {
        const VAddr page_start = Common::AlignDown(cpu_addr, BYTES_PER_PAGE);
        const VAddr page_end = Common::AlignUp(cpu_addr + size, BYTES_PER_PAGE);
        IteratePages(page_start, page_end - page_start,
                     [known_tick](RegionManager* manager, u64 offset, u64 size) {
                         manager->ConsumeGpuWrites(offset, size, known_tick);
                     });
    }

    /// Erases the GPU writes of the range.
    void SubtractGpuWrites(VAddr cpu_addr, u64 size) {
        IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            manager->SubtractGpuWrites(offset, size);
        });
    }

    /// Holds the readback words of the range, region by region in ascending order. With create
    /// the missing regions are made, which only the GPU thread may do.
    template <bool create>
    void AcquireReadback(VAddr cpu_addr, u64 size) {
        IteratePages<create>(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            manager->AcquireReadback(offset, size);
        });
    }

    void ReleaseReadback(VAddr cpu_addr, u64 size) {
        IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            manager->ReleaseReadback(offset, size);
        });
    }

    /// Mark region as modified from the CPU
    void MarkRegionAsCpuModified(VAddr cpu_addr, u64 size) noexcept {
        IteratePages(cpu_addr, size, [](RegionManager* manager, u64 offset, u64 size) {
            manager->template ChangeRegionState<StateOp::Set, StateOp::None>(offset, size);
        });
    }

    /// Removes all protection from a page and ensures GPU data has been flushed if requested
    void InvalidateRegion(VAddr cpu_addr, u64 size, auto&& on_flush) noexcept {
        if (readbacks_mode == GpuReadbacksMode::Disabled) {
            return MarkRegionAsCpuModified(cpu_addr, size);
        }
        bool should_flush = false;
        IteratePages(cpu_addr, size, [&should_flush](RegionManager* manager, u64 offset, u64 size) {
            manager->Lock();
            const bool modified = manager->template IsRegionModified<Type::GPU>(offset, size);
            if (!modified) {
                manager->template ChangeRegionState<StateOp::Set, StateOp::None, false>(offset,
                                                                                        size);
            }
            should_flush |= modified;
            manager->Unlock();
        });
        if (should_flush) {
            on_flush();
        }
    }

    /// Call 'func' for each CPU modified range and unmark those pages as CPU modified
    void ForEachUploadRange(VAddr cpu_addr, u64 size, bool is_written, auto&& func) {
        IteratePages<true>(
            cpu_addr, size, [&func, is_written](RegionManager* manager, u64 offset, u64 size) {
                if (is_written) {
                    manager->template ForEachModifiedRange<Type::CPU, StateOp::Clear, StateOp::Set>(
                        offset, size, func);
                } else {
                    manager
                        ->template ForEachModifiedRange<Type::CPU, StateOp::Clear, StateOp::None>(
                            offset, size, func);
                }
            });
    }

    /// Call 'func' for each GPU modified range and unmark those pages as GPU modified
    template <bool clear>
    void ForEachDownloadRange(VAddr cpu_addr, u64 size, auto&& func) {
        constexpr auto gpu_op = clear ? StateOp::Clear : StateOp::None;
        IteratePages(cpu_addr, size, [&func](RegionManager* manager, u64 offset, u64 size) {
            manager->template ForEachModifiedRange<Type::GPU, StateOp::None, gpu_op>(offset, size,
                                                                                     func);
        });
    }

private:
    template <bool create_region_on_fail = false, typename Func>
    bool IteratePages(VAddr cpu_address, u64 size, Func&& func) {
        using FuncReturn = typename std::invoke_result<Func, RegionManager*, u64, size_t>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        u64 remaining_size = size;
        u64 page_index = cpu_address >> HIGHER_PAGE_BITS;
        u64 page_offset = cpu_address & HIGHER_PAGE_MASK;
        while (remaining_size > 0) {
            const u64 copy_amount = std::min(HIGHER_PAGE_SIZE - page_offset, remaining_size);
            if (auto* region = top_tier[page_index]; region) {
                if constexpr (BOOL_BREAK) {
                    if (func(region, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(region, page_offset, copy_amount);
                }
            } else if constexpr (create_region_on_fail) {
                region = CreateRegion(page_index);
                if constexpr (BOOL_BREAK) {
                    if (func(region, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(region, page_offset, copy_amount);
                }
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        return false;
    }

    RegionManager* CreateRegion(u64 page_index) {
        const VAddr base_cpu_addr = page_index << HIGHER_PAGE_BITS;
        if (free_managers.empty()) {
            manager_pool.emplace_back();
            auto& last_pool = manager_pool.back();
            for (size_t i = 0; i < MANAGER_POOL_SIZE; i++) {
                // The array default-constructs each manager, whose GPU write map can own storage.
                std::destroy_at(&last_pool[i]);
                std::construct_at(&last_pool[i], tracker, 0);
                free_managers.push_back(&last_pool[i]);
            }
        }
        auto* new_manager = free_managers.back();
        new_manager->SetCpuAddress(base_cpu_addr);
        free_managers.pop_back();
        top_tier[page_index] = new_manager;
        return new_manager;
    }

    PageManager* tracker;
    const u32 readbacks_mode;
    std::deque<std::array<RegionManager, MANAGER_POOL_SIZE>> manager_pool;
    std::vector<RegionManager*> free_managers;
    std::array<RegionManager*, NUM_HIGH_PAGES> top_tier{};
};

} // namespace VideoCore
