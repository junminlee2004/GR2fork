// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <deque>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <tsl/robin_map.h>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

/// Per-draw probe handed from the rasterizer to the cache. `fp` is the incremental fold of the
/// descriptor payload accumulated while the descriptor arrays were assembled; the three
/// generations are sampled BEFORE any lookup that could destroy a handle, so a hit is only
/// honored when they still match at emission time (handle-recycling guard).
struct DescSetProbe {
    u64 fp{};
    u64 view_gen{};
    u64 sampler_gen{};
    u64 buffer_gen{};
    bool active{};
    /// The payload is backed by a stream-ring copy that the rasterizer's memo only repeats inside
    /// the tick that recorded it, so the entry must not outlive that submit. Set for Flatbuf
    /// pipelines (mode 3) and, per draw, for a ClipPlanes draw with the clipper register on.
    bool tick_bound{};
};

/// Global, content-keyed descriptor set cache.
///
/// A pipeline registers the *definition* of its descriptor set layout once at creation time and
/// receives a dense class id; the cache owns a canonical layout per class and allocates every set
/// from it, so any two pipelines with an identically defined layout share cached sets. Entries are
/// keyed by a 64-bit fold of the descriptor payload and always verified field-wise before use.
///
/// From mode 2 on, a cacheable pipeline's layout no longer carries ePushDescriptorKHR, so the cache
/// - not the caller - owns its descriptor update. A declined *admission* therefore must not become
/// a declined *bind*: the draw is served from a tick-recycled scratch set of the same class. Only a
/// draw that can get no set at all hands the draw back, because routing ordinary misses into
/// DescriptorHeap::Commit would cycle the shared 1024-set heap and drop every other layout's
/// pre-allocated batch - the a9e97bde stutter mechanism. That state is transient by default (this
/// tick's single permitted vkCreateDescriptorPool is spent, or every pool is momentarily full);
/// degraded_ is reserved for a genuine, unrecoverable driver failure.
///
/// There is no clear(), no descriptor pool reset, no vkFreeDescriptorSets and no deferred mass
/// destroy anywhere in this class.
class DescriptorSetCache {
public:
    explicit DescriptorSetCache(const Instance& instance, Scheduler& scheduler);
    ~DescriptorSetCache();

    DescriptorSetCache(const DescriptorSetCache&) = delete;
    DescriptorSetCache& operator=(const DescriptorSetCache&) = delete;

    /// Latched DescriptorSetCacheMode. 0 means the cache is fully inert.
    u32 Mode() const noexcept {
        return mode_;
    }

    static constexpr u32 kInvalidClass = 0xFFFFFFFFu;
    static constexpr u32 kMaxElems = 48;
    static constexpr u32 kMaxWrites = 48;

    /// Pipeline-creation time. Thread-safe (takes class_mutex_). Returns a dense class id, or
    /// kInvalidClass if the mode is Disabled.
    u32 RegisterClass(std::span<const vk::DescriptorSetLayoutBinding> bindings, u32 num_elems,
                      u32 num_writes);

    /// Pipeline classification telemetry. `veto` is 0 when cacheable, else the failing veto id.
    void NoteClassified(bool cacheable, u8 veto);

    /// Draw path. Returns true only when it actually issued the descriptor set bind. An inactive
    /// probe (clip gate, mid-gather epoch abort) still reaches here in mode >= 2: there is nothing
    /// to look up, but the pipeline has no push descriptors any more, so the cache still owns the
    /// bind and serves it from a scratch set.
    bool Bind(u32 class_id, const DescSetProbe& probe, std::vector<vk::WriteDescriptorSet>& writes,
              vk::PipelineLayout pipeline_layout, vk::CommandBuffer cmdbuf,
              vk::PipelineBindPoint bind_point);

    /// Once per submit, from Rasterizer::OnSubmit, on the GPU thread.
    void OnSubmit(u64 view_gen, u64 sampler_gen, u64 buffer_gen);

    struct Stats {
        u64 eligible;
        u64 hits;
        u64 miss;
        u64 decl_budget;
        u64 decl_noset;
        u64 decl_noslot;
        u64 decl_payload;
        u64 gen_abort;
        /// Draws on a cacheable pipeline whose clip planes rode the stream ring this draw. In
        /// Observe these are still gated out (no memo there); from mode 2 on they are memoized,
        /// stay eligible and are admitted as tick-bound entries.
        u64 clip_stream;
        /// Binds served from a scratch set because the entry could not be admitted (or there was
        /// nothing to admit). These pay updateDescriptorSets + bind instead of a push, but they
        /// never touch DescriptorHeap.
        u64 scratch;
    };
    const Stats& GetStats() const noexcept {
        return stats_;
    }
    void ResetStats();

    /// Draw-path bookkeeping owned by the rasterizer: a cacheable pipeline whose clip planes came
    /// from the stream ring on this draw, and a draw whose epoch snapshot moved mid-gather. Both
    /// are counted here so every DESCSET counter shares one home and one reset.
    void NoteClipStream() noexcept {
        ++stats_.clip_stream;
    }
    void NoteGenAbort() noexcept {
        ++stats_.gen_abort;
    }

    struct PipeStats {
        u64 total;
        u64 cacheable;
        u64 v_flat;
        u64 v_lds;
        u64 v_dynmip;
        u64 v_size;
    };
    const PipeStats& GetPipeStats() const noexcept {
        return pipe_stats_;
    }

    u32 LiveEntries() const noexcept {
        return live_;
    }
    u32 NumClasses() const noexcept;
    u32 NumSets() const noexcept {
        return total_sets_;
    }
    u32 NumPools() const noexcept {
        return static_cast<u32>(pools_.size());
    }
    bool Degraded() const noexcept {
        return degraded_;
    }

private:
    static constexpr u32 kMaxEntries = 1024;
    static constexpr u32 kBuckets = 4096; // power of two
    static constexpr u32 kEvictBudget = 32;
    static constexpr u32 kAdmitBudget = 64;
    static constexpr u64 kMaxAgeTicks = 256;
    static constexpr u32 kHighWater = 896; // kMaxEntries * 7/8
    static constexpr u32 kInitialPools = 2;
    static constexpr u32 kMaxPools = 8;
    static constexpr u32 kPoolMaxSets = 512;
    /// UPPER bound on one vkAllocateDescriptorSets batch, not the batch itself. A class's batch
    /// grows geometrically from one (see AcquireSet): a fixed batch made set consumption scale with
    /// kSetBatch * the number of layout classes rather than with demand, because a set never leaves
    /// the class it was allocated from, so a class touched once stranded kSetBatch - 1 sets for the
    /// process lifetime.
    static constexpr u32 kSetBatch = 16;
    /// Device-wide set ceiling and the fraction of it past which batching is abandoned and sets are
    /// allocated strictly one at a time, after a reclaim attempt.
    static constexpr u32 kMaxSets = kMaxPools * kPoolMaxSets;
    static constexpr u32 kSetSoftCap = kMaxSets - kMaxSets / 4;
    static constexpr u32 kInvalidIdx = 0xFFFFFFFFu;
    /// LRU steps the last-resort set reclaim is allowed to walk. Bounded like every other pass in
    /// this class; it only runs when the alternative is DescriptorHeap::Commit.
    static constexpr u32 kReclaimBudget = 64;

    /// One verified descriptor element. Packed by construction - three u64 and no padding.
    struct DescElem {
        u64 a, b, c;
    };

    struct Entry {
        u64 fp{};
        u64 view_gen{};
        u64 sampler_gen{};
        u64 buffer_gen{};
        u64 last_use_tick{};
        u64 bound_tick{};
        vk::DescriptorSet set{};
        u32 class_id{kInvalidClass};
        u32 hash_next{kInvalidIdx};
        u32 lru_prev{kInvalidIdx};
        u32 lru_next{kInvalidIdx};
        u16 n_elems{};
        u16 n_writes{};
        bool live{};
    };

    struct LayoutClass {
        u64 sig{};
        /// Own dense class id. The set-reclaim pass needs it to tell which LRU entries hold a set
        /// that belongs to this class; a set never crosses classes.
        u32 id{kInvalidClass};
        /// Kept for the exact element-wise compare that guards against a signature collision.
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        /// Cache-owned canonical layout, flags {}. A pipeline's own layout is never borrowed.
        vk::UniqueDescriptorSetLayout layout;
        std::vector<vk::DescriptorSet> free_sets;
        /// Sets handed to a draw the cache could not admit, stamped with the tick that recorded
        /// them. Ticks are handed out monotonically, so this is FIFO by construction and the drain
        /// is a prefix pop. A set returns to free_sets once its command buffer has retired.
        std::deque<std::pair<vk::DescriptorSet, u64>> scratch;
        /// Every set ever allocated for this class: free_sets + entry-held + scratch. Drives the
        /// geometric batch growth, and is never decremented because a set cannot leave its class.
        u32 sets_owned{};
        u32 num_elems{};
        u32 num_writes{};
    };

    bool CreatePool();
    /// GPU-thread only. Returns retired scratch sets of `c` to its free list. O(retired).
    void ReclaimScratch(LayoutClass& c);
    /// GPU-thread only. Bounded LRU pass that evicts already-retired entries holding a set of `c`
    /// and returns those sets to c.free_sets. Runs ONLY on the pool-pressure path - never on the
    /// hit path - and stops at the first entry still in flight. Returns the number recovered.
    u32 ReclaimSetsFor(LayoutClass& c);
    /// GPU-thread only. Returns a set owned by `c`, or a null handle when the cache declines.
    vk::DescriptorSet AcquireSet(LayoutClass& c);
    /// GPU-thread only. Updates and binds a scratch set of `c`. False only when no set exists at
    /// all, which is the one case the caller must handle itself.
    bool BindScratch(LayoutClass& c, std::vector<vk::WriteDescriptorSet>& writes,
                     vk::PipelineLayout pipeline_layout, vk::CommandBuffer cmdbuf,
                     vk::PipelineBindPoint bind_point, u64 tick, size_t bind_point_index);
    /// GPU-thread only. Returns a free entry index, or kInvalidIdx when none can be reclaimed.
    u32 TakeEntrySlot();

    void LruTouch(u32 idx);
    void LruUnlink(u32 idx);
    void LruPushFront(u32 idx);
    void HashInsert(u32 idx);
    void HashRemove(u32 idx);
    void Evict(u32 idx);

    const Instance& instance_;
    Scheduler& scheduler_;
    vk::Device device_{};
    u32 mode_{};

    // Layout classes. A deque never relocates, so a draw may read a class whose id was handed out
    // by a pipeline built on another thread. The draw path never takes class_mutex_.
    std::deque<LayoutClass> classes_;
    tsl::robin_map<u64, boost::container::small_vector<u32, 2>> class_by_sig_;
    std::mutex class_mutex_;

    // Fixed-capacity storage, allocated exactly once in the constructor.
    std::vector<Entry> entries_;
    std::vector<u32> hash_heads_;
    std::vector<u32> free_entries_;
    std::vector<DescElem> elem_arena_;
    std::vector<u32> shape_arena_;
    std::vector<vk::UniqueDescriptorPool> pools_;

    u32 lru_head_{kInvalidIdx};
    u32 lru_tail_{kInvalidIdx};
    u32 live_{};
    u32 total_sets_{};
    u32 admit_budget_{};
    u32 pool_grow_budget_{};
    /// Pool the next set batch is allocated from. Advances only when a pool refuses an allocation,
    /// so every created pool is drained before a new one is asked for - pools_.back() alone would
    /// leave the other initial pool untouched for the whole process.
    u32 pool_cursor_{};
    /// Tick during which every existing pool refused an allocation and the per-submit growth budget
    /// was already spent. That is a RATE LIMIT, not a failure: it must not set degraded_. Remember
    /// it only to keep the rest of this tick from re-issuing an allocation that is known to fail;
    /// the next tick refills pool_grow_budget_ and tries again.
    u64 alloc_stall_tick_{};
    /// The tick the admission and pool-growth budgets were last refilled for. The budgets are
    /// per-tick (i.e. per command buffer), not per guest submit: Rasterizer::OnSubmit fires roughly
    /// once per frame, which would cap the whole cache at kAdmitBudget populates per frame and make
    /// a 1024-entry table take 16 frames to fill.
    u64 budget_tick_{};
    /// Set only by an unrecoverable driver failure, never by the per-tick pool-growth rate limit.
    bool degraded_{};
    bool warned_pool_{};
    bool warned_stall_{};

    Stats stats_{};
    PipeStats pipe_stats_{};
};

} // namespace Vulkan
