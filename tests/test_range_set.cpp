// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <random>
#include <span>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

#include "video_core/buffer_cache/range_set.h"

namespace {

using Tree = VideoCore::BasicRangeSet<VideoCore::RangeSetsAllocator>;
using Flat = VideoCore::FlatRangeSet;
using Dumped = std::vector<std::pair<VAddr, VAddr>>;

struct Range {
    VAddr addr;
    u64 size;
};

template <class Set>
Dumped Dump(const Set& set) {
    Dumped out;
    set.ForEach([&](VAddr lo, VAddr hi) { out.emplace_back(lo, hi); });
    return out;
}

template <class Set>
Dumped DumpInRange(const Set& set, VAddr addr, u64 size) {
    Dumped out;
    set.ForEachInRange(addr, size, [&](VAddr lo, VAddr hi) { out.emplace_back(lo, hi); });
    return out;
}

} // namespace

TEST(FlatRangeSet, TouchingIntervalsJoin) {
    Flat flat;
    flat.Add(0x100, 0x100);
    flat.Add(0x200, 0x100);
    EXPECT_EQ(Dump(flat), (Dumped{{0x100, 0x300}}));
    EXPECT_TRUE(flat.Contains(0x100, 0x200));
    EXPECT_TRUE(flat.Contains(0x2FF, 0));
    EXPECT_FALSE(flat.Intersects(0x2FF, 0));
    EXPECT_FALSE(flat.Intersects(0x300, 0x10));
    EXPECT_TRUE(flat.Intersects(0x2FF, 0x10));
}

TEST(FlatRangeSet, SubtractSplitsAndClips) {
    Flat flat;
    flat.Add(0x100, 0x400);
    flat.Subtract(0x200, 0x100); // strictly inside: split
    EXPECT_EQ(Dump(flat), (Dumped{{0x100, 0x200}, {0x300, 0x500}}));
    flat.Subtract(0x000, 0x180); // head clip of the first
    flat.Subtract(0x480, 0x100); // tail clip of the second
    EXPECT_EQ(Dump(flat), (Dumped{{0x180, 0x200}, {0x300, 0x480}}));
    flat.Subtract(0x180, 0);      // no-op
    flat.Subtract(0x000, 0x1000); // everything
    EXPECT_EQ(Dump(flat), Dumped{});
    EXPECT_EQ(flat.Size(), 0u);
}

TEST(FlatRangeSet, MatchesIntervalSet) {
    std::mt19937_64 rng{12345};
    Tree tree;
    Flat flat;
    const auto rnd = [&](u64 n) { return static_cast<u64>(rng() % n); };
    // Queries never use an empty interval: the interval set asserts on one in
    // debug builds, and the emulator never asks it. Adds and subtracts do.
    for (int step = 0; step < 20000; ++step) {
        const VAddr a = static_cast<VAddr>(rnd(4096));
        const u64 op = rnd(6);
        const u64 s = op >= 3 ? 1 + rnd(64) : rnd(65);
        switch (op) {
        case 0:
            tree.Add(a, s);
            flat.Add(a, s);
            break;
        case 1: {
            std::vector<Range> batch(1 + rnd(32));
            for (auto& r : batch) {
                r = Range{static_cast<VAddr>(rnd(4096)), rnd(65)};
            }
            std::ranges::sort(batch, {}, [](const Range& r) { return std::pair{r.addr, r.size}; });
            const auto dup = std::ranges::unique(batch, [](const Range& x, const Range& y) {
                return x.addr == y.addr && x.size == y.size;
            });
            batch.erase(dup.begin(), dup.end());
            auto hint = tree.End();
            for (const Range& r : batch) {
                hint = tree.Add(hint, r.addr, r.size);
            }
            flat.AddSortedBatch(std::span<const Range>{batch});
            break;
        }
        case 2:
            tree.Subtract(a, s);
            flat.Subtract(a, s);
            break;
        case 3:
            EXPECT_EQ(tree.Contains(a, s), flat.Contains(a, s)) << "step " << step;
            break;
        case 4:
            EXPECT_EQ(tree.Intersects(a, s), flat.Intersects(a, s)) << "step " << step;
            break;
        default:
            EXPECT_EQ(DumpInRange(tree, a, s), DumpInRange(flat, a, s)) << "step " << step;
            break;
        }
        ASSERT_EQ(Dump(tree), Dump(flat)) << "step " << step;
    }
}
