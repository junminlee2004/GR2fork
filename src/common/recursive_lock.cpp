// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_map>
#include "common/assert.h"
#include "common/recursive_lock.h"

namespace Common::Detail {

// PERF(GR2FORK release_v2_2): inline-slot-array replacement for the
// thread_local unordered_map.
//
// Background: every Common::RecursiveScopedLock and RecursiveSharedLock
// constructor / destructor pair issued one unordered_map<void*, ...>
// `operator[]` (which is find-or-insert with possible heap allocation)
// plus `find` + `erase` on release. With Vulkan rasterizer's
// `mapped_ranges_mutex` held under RecursiveSharedLock from
// ForEachMappedRangeInRange / IsMapped / SynchronizeBuffersInRange, the
// page-fault signal handler on GAME_MainThread, and Vulkan's draw path
// on GpuComm, this ran tens-of-thousands of times per second. The hash
// path costs ~30–80 cycles vs ~5–20 cycles for an inline-pointer scan.
//
// What changed: the thread_local store is now a tiny inline array
// `slots[kInlineSlots]` of {mutex_ptr, count, type} tuples. Linear scan
// on mutex pointer match. Empty slot detected by mutex_ptr == nullptr
// (no caller passes null — all current call sites take address-of a
// reference, which is non-null by construction). Overflow falls back to
// a heap unordered_map allocated lazily, only if a single thread holds
// >kInlineSlots distinct recursive locks at once.
//
// What did NOT change: the public Detail::Increment/Decrement API,
// the assertion semantics (matching-type assert on every increment
// after the first, count > 0 assert on every decrement), or the
// outermost-acquire / outermost-release return-value contract used by
// the templates in recursive_lock.h.

namespace {

constexpr size_t kInlineSlots = 12;

struct RecursiveLockSlot {
    void* mutex;       // nullptr => empty slot
    int count;
    RecursiveLockType type;
};

struct RecursiveLockTlsState {
    RecursiveLockSlot slots[kInlineSlots]{};
    // Overflow fallback. Allocated lazily and only if a thread holds
    // >kInlineSlots distinct recursive locks at once. For most threads
    // this stays nullptr forever.
    std::unordered_map<void*, std::pair<RecursiveLockType, int>>* overflow{nullptr};

    ~RecursiveLockTlsState() { delete overflow; }
};

thread_local RecursiveLockTlsState g_state;

} // namespace

bool IncrementRecursiveLock(void* mutex, RecursiveLockType type) {
    auto& s = g_state;

    // Fast path #1: this mutex is already held — bump the count.
    for (auto& slot : s.slots) {
        if (slot.mutex == mutex) {
            ASSERT(slot.type == type);
            slot.count++;
            return false; // not the outermost acquire
        }
    }

    // Fast path #2: claim an empty inline slot. This is the typical
    // first-acquire path.
    for (auto& slot : s.slots) {
        if (slot.mutex == nullptr) {
            slot.mutex = mutex;
            slot.count = 1;
            slot.type = type;
            return true; // outermost acquire
        }
    }

    // Slow path: inline slots are full. Fall back to a heap map.
    if (!s.overflow) [[unlikely]] {
        s.overflow = new std::unordered_map<void*, std::pair<RecursiveLockType, int>>();
    }
    auto& [stored_type, count] = (*s.overflow)[mutex];
    if (count == 0) {
        stored_type = type;
    }
    ASSERT(stored_type == type);
    return count++ == 0;
}

bool DecrementRecursiveLock(void* mutex, RecursiveLockType type) {
    auto& s = g_state;

    // Fast path: scan inline slots.
    for (auto& slot : s.slots) {
        if (slot.mutex == mutex) {
            ASSERT(slot.type == type && slot.count > 0);
            if (--slot.count == 0) {
                slot.mutex = nullptr; // free the slot
                return true;
            }
            return false;
        }
    }

    // Slow path: check overflow map.
    if (s.overflow) [[unlikely]] {
        auto it = s.overflow->find(mutex);
        if (it != s.overflow->end()) {
            ASSERT(it->second.first == type && it->second.second > 0);
            if (--it->second.second == 0) {
                s.overflow->erase(it);
                return true;
            }
            return false;
        }
    }

    UNREACHABLE_MSG("Recursive lock decrement on mutex not held");
    return true;
}

} // namespace Common::Detail
