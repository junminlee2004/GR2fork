// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "video_core/renderer_vulkan/vk_descriptor_set_cache.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/skipcache/skipcache.h"

namespace Vulkan {

namespace Skipcache = VideoCore::Skipcache;

namespace {

constexpr u64 kFoldMul = 0x9E3779B97F4A7C15ull;

inline u64 Fold(u64 h, u64 v) {
    return std::rotl((h ^ v) * kFoldMul, 29);
}

/// Signature over the four meaningful fields of every binding. Never over the raw struct: it
/// carries padding, so a byte-wise hash would be nondeterministic.
u64 SignatureOf(std::span<const vk::DescriptorSetLayoutBinding> bindings) {
    u64 h = 0x243F6A8885A308D3ull;
    h = Fold(h, static_cast<u64>(bindings.size()));
    for (const auto& b : bindings) {
        h = Fold(h, static_cast<u64>(b.binding));
        h = Fold(h, static_cast<u64>(static_cast<u32>(b.descriptorType)));
        h = Fold(h, static_cast<u64>(b.descriptorCount));
        h = Fold(h, static_cast<u64>(static_cast<u32>(
                        static_cast<vk::ShaderStageFlags::MaskType>(b.stageFlags))));
    }
    return h;
}

/// Exact element-wise compare. A signature collision would otherwise let a set allocated from a
/// differently defined layout be bound - undefined behavior.
bool BindingsEqual(std::span<const vk::DescriptorSetLayoutBinding> a,
                   std::span<const vk::DescriptorSetLayoutBinding> b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].binding != b[i].binding || a[i].descriptorType != b[i].descriptorType ||
            a[i].descriptorCount != b[i].descriptorCount || a[i].stageFlags != b[i].stageFlags) {
            return false;
        }
    }
    return true;
}

/// Walks the emitted writes in order and either fills (Fill == true) or field-wise compares
/// (Fill == false) the packed payload of an entry. This - not the 64-bit fingerprint - is the
/// correctness guarantee: the fold is only an index.
///
/// NEVER memcmp a vk::DescriptorBufferInfo / vk::DescriptorImageInfo array. Those structs carry
/// padding, so a byte-wise compare is nondeterministic; the tree already states this rule at
/// vk_pipeline_common.cpp:26. One templated function serves both compare and fill so the probe and
/// the populate can never drift apart.
///
/// Fails closed: an unknown descriptor type, or an element/write count past the caps, returns
/// false and the draw declines the cache.
///
/// `Elem` is deduced so this stays a plain file-local helper without naming the cache's private
/// element type.
template <bool Fill, typename Elem>
bool WalkPayload(const std::vector<vk::WriteDescriptorSet>& writes, Elem* elems, u32* shape,
                 u32 max_elems, u32 max_writes, u32& out_elems, u32& out_writes) {
    u32 ne = 0;
    u32 nw = 0;
    for (const auto& w : writes) {
        if (nw >= max_writes) {
            return false;
        }
        const u32 shape_word = static_cast<u32>((static_cast<u64>(w.dstBinding) << 16) |
                                                (static_cast<u32>(w.descriptorType) & 0xFFFFu));
        if constexpr (Fill) {
            shape[nw] = shape_word;
        } else if (shape[nw] != shape_word) {
            return false;
        }
        ++nw;

        const u32 count = w.descriptorCount;
        if (count > max_elems || ne > max_elems - count) {
            return false;
        }
        switch (w.descriptorType) {
        case vk::DescriptorType::eStorageBuffer:
        case vk::DescriptorType::eUniformBuffer: {
            if (w.pBufferInfo == nullptr) {
                return false;
            }
            for (u32 i = 0; i < count; ++i) {
                const auto& bi = w.pBufferInfo[i];
                const u64 a = std::bit_cast<u64>(bi.buffer);
                const u64 b = static_cast<u64>(bi.offset);
                const u64 c = static_cast<u64>(bi.range);
                if constexpr (Fill) {
                    elems[ne] = Elem{a, b, c};
                } else {
                    const Elem& p = elems[ne];
                    if (p.a != a || p.b != b || p.c != c) {
                        return false;
                    }
                }
                ++ne;
            }
            break;
        }
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eSampler:
        case vk::DescriptorType::eCombinedImageSampler: {
            if (w.pImageInfo == nullptr) {
                return false;
            }
            for (u32 i = 0; i < count; ++i) {
                const auto& ii = w.pImageInfo[i];
                const u64 a = std::bit_cast<u64>(ii.sampler);
                const u64 b = std::bit_cast<u64>(ii.imageView);
                const u64 c = static_cast<u64>(ii.imageLayout);
                if constexpr (Fill) {
                    elems[ne] = Elem{a, b, c};
                } else {
                    const Elem& p = elems[ne];
                    if (p.a != a || p.b != b || p.c != c) {
                        return false;
                    }
                }
                ++ne;
            }
            break;
        }
        default:
            // Fail closed on anything the payload cannot describe exactly.
            return false;
        }
    }
    out_elems = ne;
    out_writes = nw;
    return true;
}

} // Anonymous namespace

DescriptorSetCache::DescriptorSetCache(const Instance& instance, Scheduler& scheduler)
    : instance_{instance}, scheduler_{scheduler}, device_{instance.GetDevice()} {
    mode_ = EmulatorSettings.GetDescriptorSetCache();
    if (mode_ == DescSetCacheDisabled) {
        // Disabled-path guarantee: no allocation, no Vulkan object, nothing.
        return;
    }

    entries_.resize(kMaxEntries);
    hash_heads_.assign(kBuckets, kInvalidIdx);
    free_entries_.reserve(kMaxEntries);
    for (u32 i = kMaxEntries; i-- > 0;) {
        free_entries_.push_back(i);
    }
    elem_arena_.resize(static_cast<size_t>(kMaxEntries) * kMaxElems);
    shape_arena_.resize(static_cast<size_t>(kMaxEntries) * kMaxWrites);
    lru_head_ = kInvalidIdx;
    lru_tail_ = kInvalidIdx;
    live_ = 0;
    admit_budget_ = kAdmitBudget;
    pool_grow_budget_ = 1;

    // Pools are created up front, never lazily on a pipeline's first draw.
    pools_.reserve(kMaxPools);
    for (u32 i = 0; i < kInitialPools; ++i) {
        if (!CreatePool()) {
            break;
        }
    }
}

DescriptorSetCache::~DescriptorSetCache() = default;

bool DescriptorSetCache::CreatePool() {
    static constexpr std::array<vk::DescriptorPoolSize, 4> pool_sizes = {{
        {vk::DescriptorType::eStorageBuffer, kPoolMaxSets * 8},
        {vk::DescriptorType::eSampledImage, kPoolMaxSets * 10},
        {vk::DescriptorType::eStorageImage, kPoolMaxSets * 2},
        {vk::DescriptorType::eSampler, kPoolMaxSets * 6},
    }};
    // Flags {}: sets are recycled in place, vkFreeDescriptorSets is never called and no binding
    // uses update-after-bind.
    const vk::DescriptorPoolCreateInfo pool_info = {
        .flags = vk::DescriptorPoolCreateFlags{},
        .maxSets = kPoolMaxSets,
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    };
    auto [result, pool] = device_.createDescriptorPoolUnique(pool_info);
    if (result != vk::Result::eSuccess) {
        degraded_ = true;
        if (!warned_pool_) {
            warned_pool_ = true;
            LOG_WARNING(Render_Vulkan,
                        "DESCSET: descriptor pool creation failed ({}); cache runs "
                        "degraded and declines misses",
                        vk::to_string(result));
        }
        return false;
    }
    pools_.push_back(std::move(pool));
    return true;
}

u32 DescriptorSetCache::RegisterClass(std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                      u32 num_elems, u32 num_writes) {
    if (mode_ == DescSetCacheDisabled || bindings.empty()) {
        return kInvalidClass;
    }

    const u64 sig = SignatureOf(bindings);

    std::scoped_lock lock{class_mutex_};
    auto it = class_by_sig_.find(sig);
    if (it != class_by_sig_.end()) {
        for (const u32 candidate : it->second) {
            if (BindingsEqual(classes_[candidate].bindings, bindings)) {
                return candidate;
            }
        }
    }

    // New class: the cache builds and owns its own canonical layout.
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlags{},
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto [layout_result, layout] = device_.createDescriptorSetLayoutUnique(desc_layout_ci);
    if (layout_result != vk::Result::eSuccess) {
        degraded_ = true;
        LOG_WARNING(Render_Vulkan, "DESCSET: canonical descriptor set layout creation failed: {}",
                    vk::to_string(layout_result));
        return kInvalidClass;
    }

    const u32 id = static_cast<u32>(classes_.size());
    LayoutClass& c = classes_.emplace_back();
    c.sig = sig;
    c.id = id;
    c.bindings.assign(bindings.begin(), bindings.end());
    c.layout = std::move(layout);
    c.num_elems = num_elems;
    c.num_writes = num_writes;

    if (it != class_by_sig_.end()) {
        it.value().push_back(id);
    } else {
        boost::container::small_vector<u32, 2> list;
        list.push_back(id);
        class_by_sig_.emplace(sig, std::move(list));
    }
    return id;
}

void DescriptorSetCache::NoteClassified(bool cacheable, u8 veto) {
    if (mode_ == DescSetCacheDisabled) {
        return;
    }
    std::scoped_lock lock{class_mutex_};
    ++pipe_stats_.total;
    if (cacheable) {
        ++pipe_stats_.cacheable;
        return;
    }
    switch (veto) {
    case 1:
        ++pipe_stats_.v_flat;
        break;
    case 2:
        ++pipe_stats_.v_lds;
        break;
    case 3:
        ++pipe_stats_.v_dynmip;
        break;
    case 4:
        ++pipe_stats_.v_size;
        break;
    default:
        break;
    }
}

void DescriptorSetCache::ReclaimScratch(LayoutClass& c) {
    if (c.scratch.empty()) {
        return;
    }
    // One relaxed atomic load, never Scheduler::IsFree / MasterSemaphore::Refresh.
    const u64 known = scheduler_.GetMasterSemaphore()->KnownGpuTick();
    while (!c.scratch.empty() && c.scratch.front().second <= known) {
        c.free_sets.push_back(c.scratch.front().first);
        c.scratch.pop_front();
    }
}

u32 DescriptorSetCache::ReclaimSetsFor(LayoutClass& c) {
    // Last resort before a draw is handed back to the caller, whose only remaining fallback in
    // mode >= 2 is DescriptorHeap::Commit - the shared 1024-set heap whose exhaustion path calls
    // MasterSemaphore::IsFree (a drmSyncobjQuery ioctl), vkCreateDescriptorPool and
    // descriptor_sets.clear() inline on the GpuComm thread. Sets pinned by entries whose command
    // buffer has already retired are free real estate: recovering one is a bounded walk and no
    // allocation, and it keeps that path unreached.
    if (c.id == kInvalidClass) {
        return 0;
    }
    // One relaxed atomic load, never Scheduler::IsFree / MasterSemaphore::Refresh.
    const u64 known = scheduler_.GetMasterSemaphore()->KnownGpuTick();
    u32 recovered = 0;
    u32 idx = lru_tail_;
    for (u32 n = 0; n < kReclaimBudget && idx != kInvalidIdx && recovered < kSetBatch; ++n) {
        Entry& e = entries_[idx];
        const u32 prev = e.lru_prev;
        if (e.last_use_tick > known) {
            // The oldest entry is still in flight, so every newer one is too.
            break;
        }
        if (e.class_id == c.id && e.set) {
            // Evict returns the set to its own class's free list, i.e. to c.
            Evict(idx);
            ++recovered;
        }
        idx = prev;
    }
    return recovered;
}

vk::DescriptorSet DescriptorSetCache::AcquireSet(LayoutClass& c) {
    // A set only reaches free_sets through Evict or ReclaimScratch, and both require
    // last_use_tick / scratch tick <= KnownGpuTick() - i.e. the command buffer that recorded the
    // bind has already retired. That, plus the fact that no descriptor set layout in the tree uses
    // UPDATE_AFTER_BIND binding flags, is what makes rewriting a recycled set in place legal.
    ReclaimScratch(c);
    if (!c.free_sets.empty()) {
        const auto set = c.free_sets.back();
        c.free_sets.pop_back();
        return set;
    }
    if (degraded_ || pools_.empty()) {
        return {};
    }
    // Every pool refused an allocation earlier in this same tick and the tick's single permitted
    // pool creation was already spent, so the driver call is known to fail; skip it for the rest of
    // the tick. The GPU does keep retiring command buffers inside a tick, though, so entry-held
    // sets can still become recoverable - try that before declining, because a decline here sends
    // the draw to DescriptorHeap::Commit for every remaining cacheable draw of this command buffer,
    // which is the burst shape this design exists to avoid.
    if (alloc_stall_tick_ != 0 && alloc_stall_tick_ == budget_tick_) {
        if (ReclaimSetsFor(c) > 0) {
            const auto set = c.free_sets.back();
            c.free_sets.pop_back();
            return set;
        }
        return {};
    }

    // Batch size is proportional to what this class already owns, starting at one, and capped at
    // kSetBatch. A set can never leave the class it was allocated from - vkFreeDescriptorSets is
    // never called and the pools are created without eFreeDescriptorSet - so every set handed to a
    // class is spent for the process lifetime. A fixed batch therefore charged kSetBatch sets to
    // every class on its first admitted miss, stranding kSetBatch - 1 of them forever for a class
    // touched once; consumption scaled with the number of layout classes (a few hundred exhausts
    // kMaxSets) instead of with concurrent demand, and a scene change introducing N classes fired N
    // full-size allocations in one frame. Doubling keeps the allocation count amortized O(1) for a
    // class that really is hot while a cold one costs exactly the one set it used.
    u32 batch = std::min(std::max(c.sets_owned, 1u), kSetBatch);
    if (total_sets_ + batch > kSetSoftCap) {
        // Near the device-wide ceiling: recover this class's already-retired sets before taking
        // more, and stop batching so the remaining headroom is spread across classes by demand.
        if (ReclaimSetsFor(c) > 0) {
            const auto set = c.free_sets.back();
            c.free_sets.pop_back();
            return set;
        }
        batch = 1;
    }

    std::array<vk::DescriptorSetLayout, kSetBatch> layouts;
    layouts.fill(*c.layout);
    std::array<vk::DescriptorSet, kSetBatch> sets{};

    for (;;) {
        if (pool_cursor_ >= pools_.size()) {
            pool_cursor_ = static_cast<u32>(pools_.size()) - 1;
        }
        const vk::DescriptorSetAllocateInfo alloc_info = {
            .descriptorPool = *pools_[pool_cursor_],
            .descriptorSetCount = batch,
            .pSetLayouts = layouts.data(),
        };
        const auto result = device_.allocateDescriptorSets(&alloc_info, sets.data());
        if (result == vk::Result::eSuccess) {
            for (u32 i = 0; i + 1 < batch; ++i) {
                c.free_sets.push_back(sets[i]);
            }
            total_sets_ += batch;
            c.sets_owned += batch;
            return sets[batch - 1];
        }
        if (result != vk::Result::eErrorOutOfPoolMemory &&
            result != vk::Result::eErrorFragmentedPool) {
            // A real driver failure (out of host/device memory). This one is genuinely terminal.
            degraded_ = true;
            if (!warned_pool_) {
                warned_pool_ = true;
                LOG_WARNING(Render_Vulkan,
                            "DESCSET: descriptor set allocation failed ({}); cache runs degraded "
                            "and declines misses",
                            vk::to_string(result));
            }
            return {};
        }
        // Pool pressure, not an error. A pool with room for one set but not for a whole batch must
        // not be abandoned - it never comes back, since pool_cursor_ only moves forward.
        if (batch > 1) {
            batch = 1;
            continue;
        }
        // Drain the pools that already exist before asking for another one; this costs nothing and
        // no budget.
        if (pool_cursor_ + 1 < pools_.size()) {
            ++pool_cursor_;
            continue;
        }
        if (pools_.size() < kMaxPools && pool_grow_budget_ > 0) {
            // At most one vkCreateDescriptorPool per tick: pool_grow_budget_ is refilled on a tick
            // change in Bind and nowhere else.
            if (!CreatePool()) {
                // CreatePool sets degraded_ itself; a failed vkCreateDescriptorPool is terminal.
                return {};
            }
            --pool_grow_budget_;
            pool_cursor_ = static_cast<u32>(pools_.size()) - 1;
            continue;
        }
        // No pool can serve this class right now. Before declining - which costs the caller a
        // DescriptorHeap::Commit - reclaim the sets of this class's already-retired entries.
        if (ReclaimSetsFor(c) > 0) {
            const auto set = c.free_sets.back();
            c.free_sets.pop_back();
            return set;
        }
        // Rate-limited, NOT degraded: either this tick's one permitted pool creation is spent, or
        // all kMaxPools exist and are momentarily full. The draw is declined for now; the next tick
        // refills the growth budget and eviction keeps returning sets to the class free lists.
        alloc_stall_tick_ = budget_tick_;
        if (!warned_stall_) {
            warned_stall_ = true;
            LOG_INFO(Render_Vulkan,
                     "DESCSET: descriptor pools full ({} of {}); declining set allocations until "
                     "the next tick",
                     pools_.size(), kMaxPools);
        }
        return {};
    }
}

bool DescriptorSetCache::BindScratch(LayoutClass& c, std::vector<vk::WriteDescriptorSet>& writes,
                                     vk::PipelineLayout pipeline_layout, vk::CommandBuffer cmdbuf,
                                     vk::PipelineBindPoint bind_point, u64 tick,
                                     size_t bind_point_index) {
    const auto set = AcquireSet(c);
    if (!set) {
        // Genuinely out of sets: only reachable once kMaxPools pools are exhausted or the driver
        // refused an allocation. This is the single case that still falls through to the caller.
        ++stats_.decl_noset;
        return false;
    }
    for (auto& w : writes) {
        w.dstSet = set;
    }
    device_.updateDescriptorSets(writes, {});
    cmdbuf.bindDescriptorSets(bind_point, pipeline_layout, 0, set, {});
    Skipcache::Framework::Instance().BumpForeignPushGen(bind_point_index);
    // The set is live until this command buffer retires; it is not a cache entry and must never be
    // looked up, so it goes straight onto the tick-stamped recycle queue.
    c.scratch.emplace_back(set, tick);
    ++stats_.scratch;
    return true;
}

u32 DescriptorSetCache::TakeEntrySlot() {
    if (!free_entries_.empty()) {
        const u32 idx = free_entries_.back();
        free_entries_.pop_back();
        return idx;
    }
    // One relaxed atomic load, never Scheduler::IsFree / MasterSemaphore::Refresh - those fire a
    // drmSyncobjQuery ioctl on the bottleneck thread. Work recorded at tick T has retired once
    // KnownGpuTick() >= T, because SubmitExecution signals the pre-increment value that
    // CurrentTick() returned while the command buffer was being recorded.
    const u64 known = scheduler_.GetMasterSemaphore()->KnownGpuTick();
    u32 idx = lru_tail_;
    for (u32 n = 0; n < 8 && idx != kInvalidIdx; ++n) {
        Entry& e = entries_[idx];
        const u32 prev = e.lru_prev;
        if (e.last_use_tick <= known) {
            // Evict returns the set to its OWN class's free list, so a cross-class steal still
            // recycles the set correctly.
            Evict(idx);
            if (!free_entries_.empty()) {
                const u32 slot = free_entries_.back();
                free_entries_.pop_back();
                return slot;
            }
            return kInvalidIdx;
        }
        idx = prev;
    }
    return kInvalidIdx;
}

bool DescriptorSetCache::Bind(u32 class_id, const DescSetProbe& probe,
                              std::vector<vk::WriteDescriptorSet>& writes,
                              vk::PipelineLayout pipeline_layout, vk::CommandBuffer cmdbuf,
                              vk::PipelineBindPoint bind_point) {
    if (mode_ == DescSetCacheDisabled || class_id == kInvalidClass) {
        return false;
    }
    // Observe: full bookkeeping, but no Vulkan object is created, updated or bound and Bind always
    // returns false, so the caller takes today's exact path. That is what makes mode 1 zero risk.
    const bool observe = mode_ == DescSetCacheObserve;
    const u64 tick = scheduler_.CurrentTick();
    LayoutClass& c = classes_[class_id];
    // Binding a set at index 0 invalidates whatever pushDescriptorSetKHR previously applied there,
    // so every bind below is a foreign push as far as the DescDelta cache is concerned: without
    // this bump its slot would still claim a push is live and skip a required one.
    const size_t bind_point_index = bind_point == vk::PipelineBindPoint::eCompute ? 1 : 0;

    // Budgets are per tick, i.e. per command buffer. Refilling them from Rasterizer::OnSubmit
    // alone would tie them to the guest submit rate (roughly one per frame) and cap the entire
    // cache at kAdmitBudget populates per frame.
    if (tick != budget_tick_) {
        budget_tick_ = tick;
        admit_budget_ = kAdmitBudget;
        pool_grow_budget_ = 1;
    }

    if (!probe.active) {
        // Clip gate or mid-gather epoch abort. Nothing can be looked up or admitted, but from
        // mode 2 on this pipeline has no push descriptors left, so the cache still owns the bind.
        // Observe mode never touches a Vulkan object and always hands the draw back.
        return observe ? false
                       : BindScratch(c, writes, pipeline_layout, cmdbuf, bind_point, tick,
                                     bind_point_index);
    }
    ++stats_.eligible;

    // --- probe ---
    u32 idx = hash_heads_[probe.fp & (kBuckets - 1)];
    while (idx != kInvalidIdx) {
        Entry& e = entries_[idx];
        if (e.fp == probe.fp && e.class_id == class_id && e.view_gen == probe.view_gen &&
            e.sampler_gen == probe.sampler_gen && e.buffer_gen == probe.buffer_gen &&
            (e.bound_tick == 0 || e.bound_tick == tick)) {
            u32 ne = 0;
            u32 nw = 0;
            if (WalkPayload<false>(writes, &elem_arena_[static_cast<size_t>(idx) * kMaxElems],
                                   &shape_arena_[static_cast<size_t>(idx) * kMaxWrites], e.n_elems,
                                   e.n_writes, ne, nw) &&
                ne == e.n_elems && nw == e.n_writes) {
                ++stats_.hits;
                // A hit still records a vkCmdBindDescriptorSets into the current command buffer.
                e.last_use_tick = tick;
                LruTouch(idx);
                if (observe) {
                    return false;
                }
                cmdbuf.bindDescriptorSets(bind_point, pipeline_layout, 0, e.set, {});
                Skipcache::Framework::Instance().BumpForeignPushGen(bind_point_index);
                return true;
            }
            // Same fingerprint, different content: treat it as a miss.
            break;
        }
        idx = e.hash_next;
    }

    // --- miss ---
    ++stats_.miss;

    // Admission is gated; the BIND is not. Every decline below still has to produce a descriptor
    // set, because a cacheable pipeline in mode >= 2 lost ePushDescriptorKHR at creation time and
    // the caller's only remaining fallback is DescriptorHeap::Commit - which cycles the shared
    // 1024-set heap and drops every other layout's pre-allocated batch on exhaustion. Serving the
    // decline from a scratch set of this class keeps that path unreached.
    u32 slot = kInvalidIdx;
    if (admit_budget_ == 0) {
        ++stats_.decl_budget;
    } else if ((slot = TakeEntrySlot()) == kInvalidIdx) {
        ++stats_.decl_noslot;
    }
    u32 ne = 0;
    u32 nw = 0;
    if (slot != kInvalidIdx &&
        !WalkPayload<true>(writes, &elem_arena_[static_cast<size_t>(slot) * kMaxElems],
                           &shape_arena_[static_cast<size_t>(slot) * kMaxWrites], kMaxElems,
                           kMaxWrites, ne, nw)) {
        free_entries_.push_back(slot);
        slot = kInvalidIdx;
        ++stats_.decl_payload;
    }
    if (slot == kInvalidIdx) {
        return observe ? false
                       : BindScratch(c, writes, pipeline_layout, cmdbuf, bind_point, tick,
                                     bind_point_index);
    }
    vk::DescriptorSet set{};
    if (!observe) {
        set = AcquireSet(c);
        if (!set) {
            // Out of sets entirely: give the slot back and let the caller take the heap path.
            free_entries_.push_back(slot);
            ++stats_.decl_noset;
            return false;
        }
        for (auto& w : writes) {
            w.dstSet = set;
        }
        instance_.GetDevice().updateDescriptorSets(writes, {});
        cmdbuf.bindDescriptorSets(bind_point, pipeline_layout, 0, set, {});
        Skipcache::Framework::Instance().BumpForeignPushGen(bind_point_index);
    }
    Entry& e = entries_[slot];
    e.fp = probe.fp;
    e.class_id = class_id;
    e.view_gen = probe.view_gen;
    e.sampler_gen = probe.sampler_gen;
    e.buffer_gen = probe.buffer_gen;
    e.last_use_tick = tick;
    // Stream-backed payloads only repeat within the tick that recorded the copy: stamping the tick
    // here confines the entry to this submit. The probe above requires bound_tick == 0 || == tick,
    // and OnSubmit treats a bound_tick below KnownGpuTick as stale.
    e.bound_tick = probe.tick_bound ? tick : 0;
    e.set = set;
    e.n_elems = static_cast<u16>(ne);
    e.n_writes = static_cast<u16>(nw);
    e.live = true;
    HashInsert(slot);
    LruPushFront(slot);
    ++live_;
    --admit_budget_;
    return !observe;
}

void DescriptorSetCache::OnSubmit(u64 view_gen, u64 sampler_gen, u64 buffer_gen) {
    if (mode_ == DescSetCacheDisabled) {
        return;
    }
    // The admission and pool-growth budgets are NOT refilled here: this runs once per guest submit
    // (roughly once per frame), and tying admission to that rate caps the cache at kAdmitBudget
    // populates per frame. Bind refills them on every tick change instead.

    // One relaxed atomic load. Never Scheduler::IsFree / MasterSemaphore::Refresh - those fire a
    // drmSyncobjQuery ioctl on the bottleneck thread.
    const u64 known = scheduler_.GetMasterSemaphore()->KnownGpuTick();

    u32 idx = lru_tail_;
    for (u32 n = 0; n < kEvictBudget && idx != kInvalidIdx; ++n) {
        Entry& e = entries_[idx];
        const u32 prev = e.lru_prev;
        if (e.last_use_tick > known) {
            // The oldest entry is still in flight, so every newer one is too.
            break;
        }
        const bool stale = e.view_gen != view_gen || e.sampler_gen != sampler_gen ||
                           e.buffer_gen != buffer_gen ||
                           (e.bound_tick != 0 && e.bound_tick < known);
        const bool aged = known >= e.last_use_tick + kMaxAgeTicks;
        if (stale || aged || live_ > kHighWater) {
            Evict(idx);
        }
        idx = prev;
    }
}

void DescriptorSetCache::ResetStats() {
    stats_ = Stats{};
}

u32 DescriptorSetCache::NumClasses() const noexcept {
    return static_cast<u32>(classes_.size());
}

void DescriptorSetCache::LruUnlink(u32 idx) {
    Entry& e = entries_[idx];
    if (e.lru_prev != kInvalidIdx) {
        entries_[e.lru_prev].lru_next = e.lru_next;
    } else if (lru_head_ == idx) {
        lru_head_ = e.lru_next;
    }
    if (e.lru_next != kInvalidIdx) {
        entries_[e.lru_next].lru_prev = e.lru_prev;
    } else if (lru_tail_ == idx) {
        lru_tail_ = e.lru_prev;
    }
    e.lru_prev = kInvalidIdx;
    e.lru_next = kInvalidIdx;
}

void DescriptorSetCache::LruPushFront(u32 idx) {
    Entry& e = entries_[idx];
    e.lru_prev = kInvalidIdx;
    e.lru_next = lru_head_;
    if (lru_head_ != kInvalidIdx) {
        entries_[lru_head_].lru_prev = idx;
    }
    lru_head_ = idx;
    if (lru_tail_ == kInvalidIdx) {
        lru_tail_ = idx;
    }
}

void DescriptorSetCache::LruTouch(u32 idx) {
    if (lru_head_ == idx) {
        return;
    }
    LruUnlink(idx);
    LruPushFront(idx);
}

void DescriptorSetCache::HashInsert(u32 idx) {
    Entry& e = entries_[idx];
    const u32 bucket = static_cast<u32>(e.fp & (kBuckets - 1));
    e.hash_next = hash_heads_[bucket];
    hash_heads_[bucket] = idx;
}

void DescriptorSetCache::HashRemove(u32 idx) {
    Entry& e = entries_[idx];
    const u32 bucket = static_cast<u32>(e.fp & (kBuckets - 1));
    u32 cur = hash_heads_[bucket];
    if (cur == idx) {
        hash_heads_[bucket] = e.hash_next;
        e.hash_next = kInvalidIdx;
        return;
    }
    while (cur != kInvalidIdx) {
        Entry& c = entries_[cur];
        if (c.hash_next == idx) {
            c.hash_next = e.hash_next;
            break;
        }
        cur = c.hash_next;
    }
    e.hash_next = kInvalidIdx;
}

void DescriptorSetCache::Evict(u32 idx) {
    Entry& e = entries_[idx];
    if (!e.live) {
        return;
    }
    HashRemove(idx);
    LruUnlink(idx);
    if (e.set && e.class_id != kInvalidClass) {
        // A set never crosses classes: it goes back to the class it was allocated from.
        classes_[e.class_id].free_sets.push_back(e.set);
    }
    e.set = vk::DescriptorSet{};
    e.class_id = kInvalidClass;
    e.fp = 0;
    e.n_elems = 0;
    e.n_writes = 0;
    e.live = false;
    free_entries_.push_back(idx);
    --live_;
}

} // namespace Vulkan
