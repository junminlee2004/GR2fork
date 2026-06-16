// SPDX-FileCopyrightText: 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/object_pool.h"
#include "common/types.h"

namespace VideoCore {

template <class Traits>
class MultiLevelPageTable final {
    using Entry = typename Traits::Entry;

    static constexpr size_t AddressSpaceBits = Traits::AddressSpaceBits;
    static constexpr size_t FirstLevelBits = Traits::FirstLevelBits;
    static constexpr size_t PageBits = Traits::PageBits;
    static constexpr size_t FirstLevelShift = AddressSpaceBits - FirstLevelBits;
    static constexpr size_t SecondLevelBits = FirstLevelShift - PageBits;
    static constexpr size_t NumEntriesPerL1Page = 1ULL << SecondLevelBits;

    using L1Page = std::array<Entry, NumEntriesPerL1Page>;

public:
    // GR2FORK: first_level_map is sized once here and never grows. Its slots are
    // atomic so the lazily-installed L1 page can be read lock-free on the hot path
    // while another thread installs it concurrently (see EntryAt below).
    explicit MultiLevelPageTable() : first_level_map(1ULL << FirstLevelBits) {}

    ~MultiLevelPageTable() noexcept = default;

    [[nodiscard]] Entry* find(size_t page) {
        const size_t l1_page = page >> SecondLevelBits;
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        L1Page* const l1 = first_level_map[l1_page].load(std::memory_order_acquire);
        if (l1 == nullptr) {
            return nullptr;
        }
        return &(*l1)[l2_page];
    }

    [[nodiscard]] const Entry* find(size_t page) const {
        const size_t l1_page = page >> SecondLevelBits;
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        const L1Page* const l1 = first_level_map[l1_page].load(std::memory_order_acquire);
        if (l1 == nullptr) {
            return nullptr;
        }
        return &(*l1)[l2_page];
    }

    [[nodiscard]] const Entry& operator[](size_t page) const {
        return EntryAt(page);
    }

    [[nodiscard]] Entry& operator[](size_t page) {
        return EntryAt(page);
    }

private:
    // GR2FORK race fix.
    //
    // operator[] is reached from BufferCache::ReadMemory on the guest-fault
    // signal-handler thread AND from FindBuffer on the PM4/assembler thread. The
    // previous body called the non-thread-safe Common::ObjectPool::Create()
    // unsynchronized from both. Two concurrent first-touch allocations corrupt the
    // pool's bookkeeping (non-atomic used_objects / raw chunk pointer) and a wild
    // construct_at splatters the page_table header — the torn first_level_map
    // pointer that faulted inside operator[] from the signal handler.
    //
    // We must NOT move the allocation off the fault thread (doing so reintroduces
    // the loading-transition freeze), so instead we serialize the rare install:
    //   - fast path: the L1 page already exists -> one lock-free atomic load.
    //   - slow path: take alloc_mutex, double-check, Create(), publish with a
    //     release store.
    // The lock is held only for a first-touch allocation (~one per 16 MiB of
    // address space; for GR2 the pool's first chunk is never exhausted so Create()
    // never allocates at runtime). It is never held across a guest fault — faults
    // occur on guest memory, never inside this routine — so the handler cannot
    // deadlock on it, and the dead read releases it before the SendCommand
    // round-trip in ReadMemory.
    [[nodiscard]] Entry& EntryAt(size_t page) const {
        const size_t l1_page = page >> SecondLevelBits;
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        L1Page* l1 = first_level_map[l1_page].load(std::memory_order_acquire);
        if (l1 == nullptr) [[unlikely]] {
            std::scoped_lock lock{alloc_mutex};
            l1 = first_level_map[l1_page].load(std::memory_order_relaxed);
            if (l1 == nullptr) {
                l1 = page_alloc.Create();
                first_level_map[l1_page].store(l1, std::memory_order_release);
            }
        }
        return (*l1)[l2_page];
    }

    mutable std::vector<std::atomic<L1Page*>> first_level_map;
    mutable Common::ObjectPool<L1Page> page_alloc;
    mutable std::mutex alloc_mutex;
};

} // namespace VideoCore
