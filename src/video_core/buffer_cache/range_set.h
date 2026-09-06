// SPDX-FileCopyrightText: 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <mutex>
#include <span>
#include <vector>
#include <boost/icl/discrete_interval.hpp>
#include <boost/icl/interval_map.hpp>
#include <boost/icl/split_interval_map.hpp>
#include <boost/icl/split_interval_set.hpp>
#include <boost/pool/pool.hpp>
#include <boost/pool/pool_alloc.hpp>
#include <boost/pool/poolfwd.hpp>
#include "common/types.h"

namespace VideoCore {

template <class T>
using RangeSetsAllocator =
    boost::fast_pool_allocator<T, boost::default_user_allocator_new_delete,
                               boost::details::pool::default_mutex, 1024, 2048>;

// Node-pool mutex of the GPU-modified range set: every operation on that set
// runs on the GPU command thread, so the lock is skipped once the setting
// latches; the skip count feeds the WRANGE line.
struct GpuRangeSetMutex {
    static inline bool lockfree = false;
    static inline u64 skips = 0;
    void lock() {
        if (lockfree) {
            ++skips;
            return;
        }
        mutex.lock();
    }
    void unlock() {
        if (!lockfree) {
            mutex.unlock();
        }
    }
    std::mutex mutex;
};

template <class T>
using GpuRangeSetAllocator = boost::fast_pool_allocator<T, boost::default_user_allocator_new_delete,
                                                        GpuRangeSetMutex, 1024, 2048>;

template <template <class> class Alloc>
struct BasicRangeSet {
    using IntervalSet = boost::icl::interval_set<
        VAddr, std::less, ICL_INTERVAL_INSTANCE(ICL_INTERVAL_DEFAULT, VAddr, std::less), Alloc>;
    using IntervalType = typename IntervalSet::interval_type;

    explicit BasicRangeSet() = default;
    ~BasicRangeSet() = default;

    void Add(VAddr base_address, size_t size) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_set.add(interval);
    }

    using Iterator = typename IntervalSet::iterator;
    Iterator End() {
        return m_ranges_set.end();
    }
    /// Hinted insert for ascending batches: returns the node holding the
    /// added interval, which is the hint for the next one.
    Iterator Add(Iterator hint, VAddr base_address, size_t size) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return m_ranges_set.add(hint, interval);
    }

    void Subtract(VAddr base_address, size_t size) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_set.subtract(interval);
    }

    void Clear() {
        m_ranges_set.clear();
    }

    bool Contains(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::contains(m_ranges_set, interval);
    }

    bool Intersects(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::intersects(m_ranges_set, interval);
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        if (m_ranges_set.empty()) {
            return;
        }

        for (const auto& set : m_ranges_set) {
            const VAddr inter_addr_end = set.upper();
            const VAddr inter_addr = set.lower();
            func(inter_addr, inter_addr_end);
        }
    }

    template <typename Func>
    void ForEachInRange(VAddr base_addr, size_t size, Func&& func) const {
        if (m_ranges_set.empty()) {
            return;
        }
        const VAddr start_address = base_addr;
        const VAddr end_address = start_address + size;
        const IntervalType search_interval{start_address, end_address};
        auto it = m_ranges_set.lower_bound(search_interval);
        if (it == m_ranges_set.end()) {
            return;
        }
        auto end_it = m_ranges_set.upper_bound(search_interval);
        for (; it != end_it; it++) {
            VAddr inter_addr_end = it->upper();
            VAddr inter_addr = it->lower();
            if (inter_addr_end > end_address) {
                inter_addr_end = end_address;
            }
            if (inter_addr < start_address) {
                inter_addr = start_address;
            }
            func(inter_addr, inter_addr_end);
        }
    }

    template <typename Func>
    void ForEachNotInRange(VAddr base_addr, size_t size, Func&& func) const {
        const VAddr end_addr = base_addr + size;
        ForEachInRange(base_addr, size, [&](VAddr range_addr, VAddr range_end) {
            if (size_t gap_size = range_addr - base_addr; gap_size != 0) {
                func(base_addr, gap_size);
            }
            base_addr = range_end;
        });
        if (base_addr != end_addr) {
            func(base_addr, end_addr - base_addr);
        }
    }

    IntervalSet m_ranges_set;
};

using RangeSet = BasicRangeSet<RangeSetsAllocator>;
using GpuRangeSet = BasicRangeSet<GpuRangeSetAllocator>;

// Sorted, disjoint, non-touching intervals in a vector: the canonical form of
// the joining interval set above, with its edge semantics (a zero-size add or
// subtract is a no-op; a zero-size query is contained and never intersects).
// GPU command thread only, like the set it stands in for.
class FlatRangeSet {
public:
    struct Interval {
        VAddr lo;
        VAddr hi;
    };
    struct Stats {
        u64 batches;
        u64 batched;
        u64 moved;
        u64 subs;
    };

    FlatRangeSet() {
        v_.reserve(4096);
        scratch_.reserve(4096);
        merged_.reserve(4096);
    }

    void Add(VAddr base, size_t size) {
        if (size == 0) {
            return;
        }
        const Interval one{base, base + size};
        MergeRun(std::span<const Interval>{&one, 1});
    }

    /// Ascending batch sorted by (addr, size) and unique on the pair; entries
    /// may nest or overlap, so they are coalesced with a running end first.
    template <class T>
    void AddSortedBatch(std::span<const T> batch) {
        if (batch.empty()) {
            return;
        }
        ++stats_.batches;
        stats_.batched += batch.size();
        scratch_.clear();
        for (const auto& e : batch) {
            if (e.size == 0) {
                continue;
            }
            const VAddr lo = e.addr;
            const VAddr hi = e.addr + e.size;
            if (!scratch_.empty() && lo <= scratch_.back().hi) {
                scratch_.back().hi = std::max<VAddr>(scratch_.back().hi, hi);
            } else {
                scratch_.push_back(Interval{lo, hi});
            }
        }
        if (!scratch_.empty()) {
            MergeRun(std::span<const Interval>{scratch_});
        }
    }

    void Subtract(VAddr base, size_t size) {
        if (size == 0 || v_.empty()) {
            return;
        }
        ++stats_.subs;
        const VAddr s = base;
        const VAddr e = base + size;
        auto it = std::upper_bound(v_.begin(), v_.end(), s,
                                   [](VAddr x, const Interval& i) { return x < i.hi; });
        if (it == v_.end() || it->lo >= e) {
            return;
        }
        if (it->lo < s && it->hi > e) {
            // A strictly covering interval splits in two.
            const VAddr hi = it->hi;
            it->hi = s;
            stats_.moved += static_cast<u64>(v_.end() - it);
            v_.insert(it + 1, Interval{e, hi});
            return;
        }
        auto first_erase = it;
        if (it->lo < s) {
            it->hi = s; // the head keeps [lo, s)
            first_erase = it + 1;
        }
        auto last = first_erase;
        while (last != v_.end() && last->hi <= e) {
            ++last;
        }
        if (last != v_.end() && last->lo < e) {
            last->lo = e; // the tail keeps [e, hi)
        }
        stats_.moved += static_cast<u64>(v_.end() - last);
        v_.erase(first_erase, last);
    }

    void Clear() {
        v_.clear();
    }

    bool Contains(VAddr base, size_t size) const {
        if (size == 0) {
            return true;
        }
        const VAddr e = base + size;
        auto it = std::upper_bound(v_.begin(), v_.end(), base,
                                   [](VAddr x, const Interval& i) { return x < i.lo; });
        if (it == v_.begin()) {
            return false;
        }
        --it;
        return it->hi >= e;
    }

    bool Intersects(VAddr base, size_t size) const {
        if (size == 0 || v_.empty()) {
            return false;
        }
        const VAddr e = base + size;
        auto it = std::upper_bound(v_.begin(), v_.end(), base,
                                   [](VAddr x, const Interval& i) { return x < i.hi; });
        return it != v_.end() && it->lo < e;
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        for (const Interval& i : v_) {
            func(i.lo, i.hi);
        }
    }

    template <typename Func>
    void ForEachInRange(VAddr base_addr, size_t size, Func&& func) const {
        if (size == 0 || v_.empty()) {
            return;
        }
        const VAddr s = base_addr;
        const VAddr e = base_addr + size;
        auto it = std::upper_bound(v_.begin(), v_.end(), s,
                                   [](VAddr x, const Interval& i) { return x < i.hi; });
        for (; it != v_.end() && it->lo < e; ++it) {
            func(std::max<VAddr>(it->lo, s), std::min<VAddr>(it->hi, e));
        }
    }

    template <typename Func>
    void ForEachNotInRange(VAddr base_addr, size_t size, Func&& func) const {
        const VAddr end_addr = base_addr + size;
        ForEachInRange(base_addr, size, [&](VAddr range_addr, VAddr range_end) {
            if (size_t gap_size = range_addr - base_addr; gap_size != 0) {
                func(base_addr, gap_size);
            }
            base_addr = range_end;
        });
        if (base_addr != end_addr) {
            func(base_addr, end_addr - base_addr);
        }
    }

    u64 Size() const {
        return v_.size();
    }

    Stats DrainStats() {
        const Stats out = stats_;
        stats_ = {};
        return out;
    }

private:
    // Merges a sorted, disjoint batch into the run of v_ it touches (every
    // interval with hi >= batch.lo and lo <= batch.hi): the union is built in
    // merged_ and replaces the run with one tail move.
    void MergeRun(std::span<const Interval> in) {
        const VAddr blo = in.front().lo;
        const VAddr bhi = in.back().hi;
        auto first = std::lower_bound(v_.begin(), v_.end(), blo,
                                      [](const Interval& i, VAddr x) { return i.hi < x; });
        auto last = std::upper_bound(first, v_.end(), bhi,
                                     [](VAddr x, const Interval& i) { return x < i.lo; });
        merged_.clear();
        const auto emit = [&](Interval x) {
            if (!merged_.empty() && x.lo <= merged_.back().hi) {
                merged_.back().hi = std::max<VAddr>(merged_.back().hi, x.hi);
            } else {
                merged_.push_back(x);
            }
        };
        auto a = first;
        size_t b = 0;
        while (a != last || b != in.size()) {
            if (b == in.size() || (a != last && a->lo < in[b].lo)) {
                emit(*a++);
            } else {
                emit(in[b++]);
            }
        }
        const size_t pos = static_cast<size_t>(first - v_.begin());
        const size_t old_n = static_cast<size_t>(last - first);
        const size_t new_n = merged_.size();
        const size_t common = std::min(old_n, new_n);
        std::copy_n(merged_.begin(), common, v_.begin() + pos);
        if (new_n > old_n) {
            v_.insert(v_.begin() + pos + common, merged_.begin() + common, merged_.end());
        } else if (old_n > new_n) {
            v_.erase(v_.begin() + pos + common, v_.begin() + pos + old_n);
        }
        if (new_n != old_n) {
            stats_.moved += static_cast<u64>(v_.size() - (pos + new_n));
        }
    }

    std::vector<Interval> v_;
    std::vector<Interval> scratch_;
    std::vector<Interval> merged_;
    Stats stats_{};
};

// The GPU-modified set behind one flag latched before any operation: the
// interval tree today, the flat vector behind gpu_range_set_flat.
struct GpuModifiedRangeSet {
    static inline bool flat = false;
    GpuRangeSet tree;
    FlatRangeSet vec;

    void Add(VAddr base, size_t size) {
        if (flat) {
            vec.Add(base, size);
        } else {
            tree.Add(base, size);
        }
    }
    /// Ascending, (addr, size)-unique batch: one merge in the flat arm, the
    /// hinted insert loop in the tree arm.
    template <class T>
    void AddSortedBatch(std::span<const T> batch) {
        if (flat) {
            vec.AddSortedBatch(batch);
            return;
        }
        auto hint = tree.End();
        for (const auto& e : batch) {
            hint = tree.Add(hint, e.addr, e.size);
        }
    }
    void Subtract(VAddr base, size_t size) {
        if (flat) {
            vec.Subtract(base, size);
        } else {
            tree.Subtract(base, size);
        }
    }
    void Clear() {
        if (flat) {
            vec.Clear();
        } else {
            tree.Clear();
        }
    }
    bool Contains(VAddr base, size_t size) const {
        return flat ? vec.Contains(base, size) : tree.Contains(base, size);
    }
    bool Intersects(VAddr base, size_t size) const {
        return flat ? vec.Intersects(base, size) : tree.Intersects(base, size);
    }
    template <typename Func>
    void ForEach(Func&& func) const {
        if (flat) {
            vec.ForEach(std::forward<Func>(func));
        } else {
            tree.ForEach(std::forward<Func>(func));
        }
    }
    template <typename Func>
    void ForEachInRange(VAddr base_addr, size_t size, Func&& func) const {
        if (flat) {
            vec.ForEachInRange(base_addr, size, std::forward<Func>(func));
        } else {
            tree.ForEachInRange(base_addr, size, std::forward<Func>(func));
        }
    }
    template <typename Func>
    void ForEachNotInRange(VAddr base_addr, size_t size, Func&& func) const {
        if (flat) {
            vec.ForEachNotInRange(base_addr, size, std::forward<Func>(func));
        } else {
            tree.ForEachNotInRange(base_addr, size, std::forward<Func>(func));
        }
    }
    u64 Size() const {
        return flat ? vec.Size() : tree.m_ranges_set.iterative_size();
    }
};

template <typename T>
class RangeMap {
public:
    using IntervalMap =
        boost::icl::interval_map<VAddr, T, boost::icl::total_absorber, std::less,
                                 boost::icl::inplace_identity, boost::icl::inter_section,
                                 ICL_INTERVAL_INSTANCE(ICL_INTERVAL_DEFAULT, VAddr, std::less),
                                 RangeSetsAllocator>;
    using IntervalType = typename IntervalMap::interval_type;

public:
    RangeMap() = default;
    ~RangeMap() = default;

    RangeMap(RangeMap const&) = delete;
    RangeMap& operator=(RangeMap const&) = delete;

    RangeMap(RangeMap&& other);
    RangeMap& operator=(RangeMap&& other);

    void Add(VAddr base_address, size_t size, const T& value) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_map.add({interval, value});
    }

    void Subtract(VAddr base_address, size_t size) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_map -= interval;
    }

    void Clear() {
        m_ranges_map.clear();
    }

    bool Contains(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::contains(m_ranges_map, interval);
    }

    bool Intersects(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::intersects(m_ranges_map, interval);
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        if (m_ranges_map.empty()) {
            return;
        }

        for (const auto& [interval, value] : m_ranges_map) {
            const VAddr inter_addr_end = interval.upper();
            const VAddr inter_addr = interval.lower();
            func(inter_addr, inter_addr_end, value);
        }
    }

    template <typename Func>
    void ForEachInRange(VAddr base_addr, size_t size, Func&& func) const {
        if (m_ranges_map.empty()) {
            return;
        }
        const VAddr start_address = base_addr;
        const VAddr end_address = start_address + size;
        const IntervalType search_interval{start_address, end_address};
        auto it = m_ranges_map.lower_bound(search_interval);
        if (it == m_ranges_map.end()) {
            return;
        }
        auto end_it = m_ranges_map.upper_bound(search_interval);
        for (; it != end_it; it++) {
            VAddr inter_addr_end = it->first.upper();
            VAddr inter_addr = it->first.lower();
            if (inter_addr_end > end_address) {
                inter_addr_end = end_address;
            }
            if (inter_addr < start_address) {
                inter_addr = start_address;
            }
            func(inter_addr, inter_addr_end, it->second);
        }
    }

    template <typename Func>
    void ForEachNotInRange(VAddr base_addr, size_t size, Func&& func) const {
        const VAddr end_addr = base_addr + size;
        ForEachInRange(base_addr, size, [&](VAddr range_addr, VAddr range_end, const T&) {
            if (size_t gap_size = range_addr - base_addr; gap_size != 0) {
                func(base_addr, gap_size);
            }
            base_addr = range_end;
        });
        if (base_addr != end_addr) {
            func(base_addr, end_addr - base_addr);
        }
    }

private:
    IntervalMap m_ranges_map;
};

template <typename T>
class SplitRangeMap {
public:
    using IntervalMap = boost::icl::split_interval_map<
        VAddr, T, boost::icl::total_absorber, std::less, boost::icl::inplace_identity,
        boost::icl::inter_section, ICL_INTERVAL_INSTANCE(ICL_INTERVAL_DEFAULT, VAddr, std::less),
        RangeSetsAllocator>;
    using IntervalType = typename IntervalMap::interval_type;

public:
    SplitRangeMap() = default;
    ~SplitRangeMap() = default;

    SplitRangeMap(SplitRangeMap const&) = delete;
    SplitRangeMap& operator=(SplitRangeMap const&) = delete;

    SplitRangeMap(SplitRangeMap&& other);
    SplitRangeMap& operator=(SplitRangeMap&& other);

    void Add(VAddr base_address, size_t size, const T& value) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_map.add({interval, value});
    }

    void Subtract(VAddr base_address, size_t size) {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        m_ranges_map -= interval;
    }

    void Clear() {
        m_ranges_map.clear();
    }

    bool Contains(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::contains(m_ranges_map, interval);
    }

    bool Intersects(VAddr base_address, size_t size) const {
        const VAddr end_address = base_address + size;
        IntervalType interval{base_address, end_address};
        return boost::icl::intersects(m_ranges_map, interval);
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        if (m_ranges_map.empty()) {
            return;
        }

        for (const auto& [interval, value] : m_ranges_map) {
            const VAddr inter_addr_end = interval.upper();
            const VAddr inter_addr = interval.lower();
            func(inter_addr, inter_addr_end, value);
        }
    }

    template <typename Func>
    void ForEachInRange(VAddr base_addr, size_t size, Func&& func) const {
        if (m_ranges_map.empty()) {
            return;
        }
        const VAddr start_address = base_addr;
        const VAddr end_address = start_address + size;
        const IntervalType search_interval{start_address, end_address};
        auto it = m_ranges_map.lower_bound(search_interval);
        if (it == m_ranges_map.end()) {
            return;
        }
        auto end_it = m_ranges_map.upper_bound(search_interval);
        for (; it != end_it; it++) {
            VAddr inter_addr_end = it->first.upper();
            VAddr inter_addr = it->first.lower();
            if (inter_addr_end > end_address) {
                inter_addr_end = end_address;
            }
            if (inter_addr < start_address) {
                inter_addr = start_address;
            }
            func(inter_addr, inter_addr_end, it->second);
        }
    }

    template <typename Func>
    void ForEachNotInRange(VAddr base_addr, size_t size, Func&& func) const {
        const VAddr end_addr = base_addr + size;
        ForEachInRange(base_addr, size, [&](VAddr range_addr, VAddr range_end, const T&) {
            if (size_t gap_size = range_addr - base_addr; gap_size != 0) {
                func(base_addr, gap_size);
            }
            base_addr = range_end;
        });
        if (base_addr != end_addr) {
            func(base_addr, end_addr - base_addr);
        }
    }

private:
    IntervalMap m_ranges_map;
};

} // namespace VideoCore
