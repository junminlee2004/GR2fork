// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <optional>

#include "common/div_ceil.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"

#ifdef __unix__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif
#include "common/debug.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"

namespace VideoCore {

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
using LockType = Common::AdaptiveMutex;
#else
using LockType = Common::SpinLock;
#endif

/**
 * Allows tracking CPU and GPU modification of pages in a contigious 16MB virtual address region.
 * Information is stored in bitsets for spacial locality and fast update of single pages.
 */
class RegionManager {
public:
    explicit RegionManager(PageManager* tracker_, VAddr cpu_addr_)
        : tracker{tracker_}, cpu_addr{cpu_addr_} {
        cpu.Fill();
        gpu.Clear();
        writeable.Fill();
        readable.Fill();
    }
    explicit RegionManager() = default;

    void SetCpuAddress(VAddr new_cpu_addr) {
        cpu_addr = new_cpu_addr;
    }

    VAddr GetCpuAddr() const {
        return cpu_addr;
    }

    static constexpr size_t SanitizeAddress(size_t address) {
        return static_cast<size_t>(std::max<s64>(static_cast<s64>(address), 0LL));
    }

    template <Type type>
    RegionBits& GetRegionBits() noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    template <Type type>
    const RegionBits& GetRegionBits() const noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    /**
     * Change the state of a range of pages
     *
     * @param dirty_addr    Base address to mark or unmark as modified
     * @param size          Size in bytes to mark or unmark as modified
     */
    template <Type type, bool enable>
    void ChangeRegionState(u64 dirty_addr, u64 size) noexcept(type == Type::GPU) {
        RENDERER_TRACE;
        const size_t offset = dirty_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        if constexpr (type == Type::GPU && enable) {
            // GPU bits are only ever mutated on the GPU command thread, so this
            // is a plain counter. It advances on marks alone: an unchanged
            // value between two points on that thread proves no new GPU write
            // was recorded for this region in between, which is the guard the
            // offloaded readback path uses before clearing bits it earlier
            // snapshotted.
            ++gpu_write_seq;
        }
        RegionBits& bits = GetRegionBits<type>();
        // A range already in the target state makes the write below an
        // identity: the bits cannot change, so the protection masks derived
        // from them cannot change either. Skipping it also leaves the sequence
        // count stable for concurrent lock-free readers.
        if constexpr (enable) {
            if (bits.AllInRange(start_page, end_page)) {
                return;
            }
        } else {
            if (!bits.AnyInRange(start_page, end_page)) {
                return;
            }
        }
        WriteScope write_scope{*this};
        if constexpr (enable) {
            bits.SetRange(start_page, end_page);
        } else {
            bits.UnsetRange(start_page, end_page);
        }
        if constexpr (type == Type::CPU) {
            UpdateProtection<!enable, false>();
        } else if (EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise) {
            UpdateProtection<enable, true>();
        }
    }

    /**
     * Loop over each page in the given range, turn off those bits and notify the tracker if
     * needed. Call the given function on each turned off range.
     *
     * @param query_cpu_range Base CPU address to loop over
     * @param size            Size in bytes of the CPU range to loop over
     * @param func            Function to call for each turned off region
     */
    template <Type type, bool clear>
    void ForEachModifiedRange(VAddr query_cpu_range, s64 size, auto&& func) {
        RENDERER_TRACE;
        const size_t offset = query_cpu_range - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        // Only the clearing form mutates the bits; entering the scope
        // conditionally keeps pure iteration off the writers' path.
        std::optional<WriteScope> write_scope;
        if constexpr (clear) {
            write_scope.emplace(*this);
        }
        RegionBits& bits = GetRegionBits<type>();
        RegionBits mask(bits, start_page, end_page);

        if constexpr (clear) {
            bits.UnsetRange(start_page, end_page);
            if constexpr (type == Type::CPU) {
                UpdateProtection<true, false>();
            } else if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled) {
                UpdateProtection<false, true>();
            }
        }

        for (const auto& [start, end] : mask) {
            func(cpu_addr + start * TRACKER_BYTES_PER_PAGE, (end - start) * TRACKER_BYTES_PER_PAGE);
        }
    }

    /**
     * Returns true when a region has been modified
     *
     * @param offset Offset in bytes from the start of the buffer
     * @param size   Size in bytes of the region to query for modifications
     */
    template <Type type>
    [[nodiscard]] bool IsRegionModified(u64 offset, u64 size) noexcept {
        RENDERER_TRACE;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }

        // Ask the range directly: materialising a masked copy of the whole
        // region's bits to answer a yes/no question was the single largest
        // cost on the bind path.
        return GetRegionBits<type>().AnyInRange(start_page, end_page);
    }

    /**
     * Read whether a range is modified without taking the lock.
     *
     * The dirty bits are read optimistically between two reads of a sequence
     * counter that writers make odd while mutating. An unchanged even counter
     * proves no writer ran during the read, so the answer is exactly what the
     * locked query would have returned. Contended acquisitions of this lock
     * between the GPU thread and the guest fault handler are otherwise the
     * dominant cost of every buffer bind.
     */
    template <Type type>
    [[nodiscard]] bool PeekRegionModified(u64 offset, u64 size) noexcept {
        for (u32 attempt = 0; attempt < 4; ++attempt) {
            const u32 before = seq.load(std::memory_order_acquire);
            if (before & 1u) {
                continue; // writer in flight
            }
            const bool result = IsRegionModified<type>(offset, size);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq.load(std::memory_order_relaxed) == before) {
                return result;
            }
        }
        std::scoped_lock lk{lock};
        return IsRegionModified<type>(offset, size);
    }

    /// Lock-free counterpart of PeekRegionModified for full coverage.
    template <Type type>
    [[nodiscard]] bool PeekRegionFullySet(u64 offset, u64 size) noexcept {
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }
        for (u32 attempt = 0; attempt < 4; ++attempt) {
            const u32 before = seq.load(std::memory_order_acquire);
            if (before & 1u) {
                continue;
            }
            const bool result = GetRegionBits<type>().AllInRange(start_page, end_page);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq.load(std::memory_order_relaxed) == before) {
                return result;
            }
        }
        std::scoped_lock lk{lock};
        return GetRegionBits<type>().AllInRange(start_page, end_page);
    }

    /// Scope guard marking a mutation of the tracked bits for readers.
    struct WriteScope {
        explicit WriteScope(RegionManager& m) : mgr{m} {
            mgr.seq.fetch_add(1, std::memory_order_acq_rel);
        }
        ~WriteScope() {
            mgr.seq.fetch_add(1, std::memory_order_release);
        }
        RegionManager& mgr;
    };

    std::atomic<u32> seq{0};
    // Counts GPU-bit marks. GPU-command-thread confined; see ChangeRegionState.
    u64 gpu_write_seq{0};
    LockType lock;

private:
    /**
     * Notify tracker about changes in the CPU tracking state of a word in the buffer
     *
     * @param word_index   Index to the word to notify to the tracker
     * @param current_bits Current state of the word
     * @param new_bits     New state of the word
     *
     * @tparam track True when the tracker should start tracking the new pages
     */
    template <bool track, bool is_read>
    void UpdateProtection() {
        RENDERER_TRACE;
        RegionBits mask = is_read ? (~gpu ^ readable) : (cpu ^ writeable);
        if (mask.None()) {
            return;
        }
        if constexpr (is_read) {
            readable = ~gpu;
        } else {
            writeable = cpu;
        }
        tracker->UpdatePageWatchersForRegion<track, is_read>(cpu_addr, mask);
    }

    PageManager* tracker;
    VAddr cpu_addr = 0;
    RegionBits cpu;
    RegionBits gpu;
    RegionBits writeable;
    RegionBits readable;
};

} // namespace VideoCore
