// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <cstddef>
#include <cstring>
#include <limits>

#include <boost/container/static_vector.hpp>
#include "common/assert.h"
#include "common/logging/log.h"

#include "core/emulator_settings.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/skipcache/skipcache.h"

namespace Vulkan {

namespace Skipcache = VideoCore::Skipcache;

namespace {

// The four key header fields are contiguous, and so are the payload fields of
// every info struct, so each run copies as raw bytes. An image info is 24
// deterministic bytes: the rasterizer's AppendImageInfo zeroes the four tail
// padding bytes after construction, so copies and compares take the struct whole.
// Offsets are checked on the C structs because MSVC rejects offsetof on the
// vulkan.hpp wrappers; the wrappers are what gets indexed, so their strides are
// checked against the C ones.
static_assert(offsetof(VkWriteDescriptorSet, dstArrayElement) ==
                  offsetof(VkWriteDescriptorSet, dstBinding) + 4 &&
              offsetof(VkWriteDescriptorSet, descriptorCount) ==
                  offsetof(VkWriteDescriptorSet, dstBinding) + 8 &&
              offsetof(VkWriteDescriptorSet, descriptorType) ==
                  offsetof(VkWriteDescriptorSet, dstBinding) + 12);
static_assert(sizeof(vk::DescriptorBufferInfo) == sizeof(VkDescriptorBufferInfo) &&
              sizeof(VkDescriptorBufferInfo) == 24 &&
              offsetof(VkDescriptorBufferInfo, buffer) == 0 &&
              offsetof(VkDescriptorBufferInfo, offset) == 8 &&
              offsetof(VkDescriptorBufferInfo, range) == 16);
static_assert(sizeof(vk::DescriptorImageInfo) == sizeof(VkDescriptorImageInfo) &&
              sizeof(VkDescriptorImageInfo) == 24 &&
              offsetof(VkDescriptorImageInfo, sampler) == 0 &&
              offsetof(VkDescriptorImageInfo, imageView) == 8 &&
              offsetof(VkDescriptorImageInfo, imageLayout) == 16);
static_assert(sizeof(vk::BufferView) == 8);
static_assert(sizeof(Skipcache::Framework::DescDeltaSlot::write_changed) >=
              Pipeline::NUM_DESCRIPTOR_WRITES);

// Serialize a descriptor write list into a deterministic byte stream, payload
// contents included. Returns 0 on any unknown descriptor type or overflow:
// unknowns fail toward the slow path.
size_t SerializeDescriptorWrites(const Pipeline::DescriptorWrites& writes,
                                 std::array<u8, 16384>& out) {
    u8* cursor = out.data();
    const u8* const limit = out.data() + out.size();
    const auto put = [&](const void* p, size_t n) {
        std::memcpy(cursor, p, n);
        cursor += n;
    };
    for (const auto& w : writes) {
        const u32 count = w.descriptorCount;
        // Bail where the overflow would happen rather than on a whole-list
        // worst case: 24 bounds every payload, so this can only overestimate.
        if (static_cast<size_t>(limit - cursor) < 16 + size_t{count} * 24) {
            return 0;
        }
        const VkWriteDescriptorSet& raw = w;
        put(&raw.dstBinding, 16);
        switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer: {
            const auto* const infos = w.pBufferInfo;
            for (u32 i = 0; i < count; ++i) {
                put(&infos[i], 24);
            }
            break;
        }
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampler: {
            const auto* const infos = w.pImageInfo;
            for (u32 i = 0; i < count; ++i) {
                put(&infos[i], 24);
            }
            break;
        }
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer: {
            const auto* const views = w.pTexelBufferView;
            for (u32 i = 0; i < count; ++i) {
                put(&views[i], 8);
            }
            break;
        }
        default:
            return 0;
        }
    }
    return static_cast<size_t>(cursor - out.data());
}

// XOR-OR of two byte ranges, 8 bytes at a time; bytes is a multiple of 8.
// Accumulating instead of exiting early keeps this a plain vector reduction
// rather than a library compare.
u64 DiffWords(const void* a, const void* b, size_t bytes) noexcept {
    const u8* pa = static_cast<const u8*>(a);
    const u8* pb = static_cast<const u8*>(b);
    u64 acc = 0;
    for (size_t i = 0; i < bytes; i += 8) {
        u64 x, y;
        std::memcpy(&x, pa + i, 8);
        std::memcpy(&y, pb + i, 8);
        acc |= x ^ y;
    }
    return acc;
}

template <size_t N>
u64 DiffFixed(const void* a, const void* b) noexcept {
    static_assert(N % 8 == 0);
    const u8* pa = static_cast<const u8*>(a);
    const u8* pb = static_cast<const u8*>(b);
    u64 acc = 0;
    for (size_t i = 0; i < N; i += 8) {
        u64 x, y;
        std::memcpy(&x, pa + i, 8);
        std::memcpy(&y, pb + i, 8);
        acc |= x ^ y;
    }
    return acc;
}

// One descriptor of a compile-time stride, copied without ever forming the
// runtime-length memcpy the walk must not call on its hit path.
template <size_t N>
void CopyFixed(void* dst, const void* src) noexcept {
#if defined(__clang__)
    __builtin_memcpy_inline(dst, src, N);
#else
    std::memcpy(dst, src, N);
#endif
}

// Walks the write list once, leaving blob equal to the serialized form and
// reporting whether any byte differed. Each write is compared whole first;
// only a changed write pays the per-descriptor verdicts and the copy, so an
// unchanged set stores nothing. Same chunking and bail rules as
// SerializeDescriptorWrites.
// The mapped form also records a per-write verdict and, for changed writes,
// a per-descriptor verdict in the slot, which a partial push consumes; the
// unmapped form is the plain walk.
template <bool kMap>
size_t MatchDescriptorWrites(const Pipeline::DescriptorWrites& writes,
                             Skipcache::Framework::DescDeltaSlot& slot, bool& changed) {
    auto& blob = slot.blob;
    u8* cursor = blob.data();
    const u8* const limit = blob.data() + blob.size();
    u64 diff = 0;
    // The verdict counters live in locals: they share the slot with the blob
    // the cursor stores into, so a slot-resident counter is reloaded after
    // every store. Every exit writes them back, so a bail leaves the slot as
    // before.
    [[maybe_unused]] u8* const verdicts = kMap ? slot.changed.data() : nullptr;
    [[maybe_unused]] u8* const write_verdicts = kMap ? slot.write_changed.data() : nullptr;
    [[maybe_unused]] u32 n = 0;
    [[maybe_unused]] u32 n_changed = 0;
    [[maybe_unused]] u32 w_idx = 0;
    [[maybe_unused]] u64 hdr_diff = 0;
    const auto write_back = [&] {
        if constexpr (kMap) {
            slot.desc_count = n;
            slot.desc_changed = n_changed;
            slot.header_changed = hdr_diff != 0;
            slot.write_count = w_idx;
        }
    };
    const auto sync_payload = [&]<size_t Stride>(const void* payload, u32 count) {
        const size_t bytes = size_t{count} * Stride;
        const u64 d =
            count == 1 ? DiffFixed<Stride>(payload, cursor) : DiffWords(payload, cursor, bytes);
        diff |= d;
        if constexpr (kMap) {
            write_verdicts[w_idx++] = d != 0;
        }
        if (d == 0) {
            if constexpr (kMap) {
                n += count;
            }
            cursor += bytes;
            return;
        }
        const u8* src = static_cast<const u8*>(payload);
        for (u32 i = 0; i < count; ++i) {
            if constexpr (kMap) {
                const u32 c = DiffFixed<Stride>(src, cursor) != 0;
                verdicts[n++] = static_cast<u8>(c);
                n_changed += c;
            }
            CopyFixed<Stride>(cursor, src);
            src += Stride;
            cursor += Stride;
        }
    };
    for (const auto& w : writes) {
        const u32 count = w.descriptorCount;
        if (static_cast<size_t>(limit - cursor) < 16 + size_t{count} * 24) {
            write_back();
            return 0;
        }
        if constexpr (kMap) {
            // Reached only past the size check above, so the u32 sum cannot wrap.
            if (n + count > slot.changed.size()) {
                write_back();
                return 0;
            }
        }
        const VkWriteDescriptorSet& raw = w;
        const u64 header = DiffFixed<16>(&raw.dstBinding, cursor);
        diff |= header;
        if constexpr (kMap) {
            hdr_diff |= header;
        }
        if (header != 0) {
            CopyFixed<16>(cursor, &raw.dstBinding);
        }
        cursor += 16;
        switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer:
            sync_payload.template operator()<24>(w.pBufferInfo, count);
            break;
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampler:
            sync_payload.template operator()<24>(w.pImageInfo, count);
            break;
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer:
            sync_payload.template operator()<8>(w.pTexelBufferView, count);
            break;
        default:
            write_back();
            return 0;
        }
    }
    write_back();
    changed = diff != 0;
    return static_cast<size_t>(cursor - blob.data());
}

// The count of a compaction that would not fit the write list: larger than any
// bound the caller compares against, so the set is pushed whole. Unreachable on
// radv, whose 32 push descriptors bound the descriptor count, but a real hole
// elsewhere, and the write list itself only fails closed by dropping runs.
constexpr size_t kRunsOverflow = std::numeric_limits<size_t>::max();

// One write of a compaction: the identity fields and exactly one payload
// pointer, the others null. A field copy of the source write would hand the
// driver whatever the other pointer slots last held.
vk::WriteDescriptorSet MakeRun(const vk::WriteDescriptorSet& w, u32 first, u32 count) {
    vk::WriteDescriptorSet run{};
    run.dstSet = w.dstSet;
    run.dstBinding = w.dstBinding + first;
    run.dstArrayElement = 0;
    run.descriptorCount = count;
    run.descriptorType = w.descriptorType;
    switch (w.descriptorType) {
    case vk::DescriptorType::eUniformBuffer:
    case vk::DescriptorType::eStorageBuffer:
        run.pBufferInfo = w.pBufferInfo + first;
        break;
    case vk::DescriptorType::eUniformTexelBuffer:
    case vk::DescriptorType::eStorageTexelBuffer:
        run.pTexelBufferView = w.pTexelBufferView + first;
        break;
    default:
        run.pImageInfo = w.pImageInfo + first;
        break;
    }
    return run;
}

// Compacts the write list to the descriptors the mapped walk saw change.
// Buffer and sampler writes span consecutive count-1 bindings, so each
// maximal run of changed descriptors becomes one write on its own binding;
// image writes are one binding with an array, so they go whole or not at all.
// The layouts come from the graphics and compute pipeline builders. Returns
// kRunsOverflow when the runs would not fit the write list.
size_t CompactDescriptorWrites(const Pipeline::DescriptorWrites& in,
                               const Skipcache::Framework::DescDeltaSlot& map,
                               Pipeline::DescriptorWrites& out) {
    out.clear();
    u32 k = 0;
    u32 w_idx = 0;
    for (const auto& w : in) {
        const u32 count = w.descriptorCount;
        ASSERT(k + count <= map.desc_count);
        // A write the mapped walk saw unchanged carries no per-descriptor verdicts.
        if (!map.write_changed[w_idx++]) {
            k += count;
            continue;
        }
        switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer:
        case vk::DescriptorType::eSampler: {
            u32 i = 0;
            while (i < count) {
                if (!map.changed[k + i]) {
                    ++i;
                    continue;
                }
                u32 j = i + 1;
                while (j < count && map.changed[k + j]) {
                    ++j;
                }
                if (out.size() >= Pipeline::NUM_DESCRIPTOR_WRITES) {
                    return kRunsOverflow;
                }
                out.push_back(MakeRun(w, i, j - i));
                i = j;
            }
            break;
        }
        default: {
            bool any = false;
            for (u32 i = 0; i < count; ++i) {
                any |= map.changed[k + i] != 0;
            }
            if (any) {
                if (out.size() >= Pipeline::NUM_DESCRIPTOR_WRITES) {
                    return kRunsOverflow;
                }
                out.push_back(MakeRun(w, 0, count));
            }
            break;
        }
        }
        k += count;
    }
    return out.size();
}

Pipeline::DescriptorWrites partial_scratch;

} // namespace

// Out of line and externally linked: the header's inline Next() calls it from
// every translation unit that fills a descriptor write list.
void DescWriteOverflow() {
    UNREACHABLE_MSG("Descriptor write list is full");
}

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   PipelineLayoutCache* layouts_, bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      layouts{layouts_}, is_compute{is_compute_} {}

void Pipeline::AssignLayouts(std::span<const vk::DescriptorSetLayoutBinding> bindings,
                             vk::DescriptorSetLayoutCreateFlags flags,
                             const vk::PushConstantRange& push_constants,
                             std::string_view debug_name) {
    if (layouts) {
        const auto shared = layouts->Acquire(bindings, flags, push_constants, debug_name);
        desc_layout = shared.set;
        pipeline_layout = shared.pipeline;
        return;
    }
    const auto device = instance.GetDevice();
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto [set_result, set] = device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(set_result == vk::Result::eSuccess, "Failed to create descriptor set layout: {}",
               vk::to_string(set_result));
    owned_desc_layout = std::move(set);
    desc_layout = *owned_desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &desc_layout,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_constants,
    };
    auto [layout_result, layout] = device.createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess, "Failed to create pipeline layout: {}",
               vk::to_string(layout_result));
    owned_pipeline_layout = std::move(layout);
    pipeline_layout = *owned_pipeline_layout;
    SetObjectName(device, pipeline_layout, "{} PipelineLayout {}",
                  is_compute ? "Compute" : "Graphics", debug_name);
}

Pipeline::~Pipeline() = default;

void Pipeline::BindResources(DescriptorWrites& set_writes, const BufferBarriers& buffer_barriers,
                             const Shader::PushData& push_data) const {
    const auto cmdbuf = scheduler.CommandBuffer();
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;

    if (!buffer_barriers.empty()) {
        const auto dependencies = vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = u32(buffer_barriers.size()),
            .pBufferMemoryBarriers = buffer_barriers.data(),
        };
        scheduler.EndRendering();
        cmdbuf.pipelineBarrier2(dependencies);
    }

    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    static const bool push_const_dedup = EmulatorSettings.IsPushConstDedup();
    bool push_same = false;
    if (push_const_dedup) {
        // A push identical to the last one on this command buffer, through
        // the same layout, leaves the same bytes in place. The slot is written
        // either way, so a taken skip stores what the storage already holds.
        using namespace VideoCore::Skipcache;
        auto& sc = Framework::Instance();
        auto& slot = sc.PushConstState();
        if (sc.Active()) [[likely]] {
            static_assert(sizeof(push_data) % sizeof(u64) == 0 &&
                          sizeof(push_data) <= sizeof(slot.words));
            constexpr size_t kWords = sizeof(push_data) / sizeof(u64);
            std::array<u64, kWords> words;
            std::memcpy(words.data(), &push_data, sizeof(push_data));
            const u64 layout_bits =
                std::bit_cast<u64>(static_cast<VkPipelineLayout>(pipeline_layout));
            const u32 flag_bits = static_cast<u32>(static_cast<vk::ShaderStageFlags>(stage_flags));
            u64 diff = 0;
            for (size_t i = 0; i < kWords; ++i) {
                diff |= words[i] ^ slot.words[i];
                slot.words[i] = words[i];
            }
            diff |= slot.layout ^ layout_bits;
            diff |= slot.stage_flags ^ flag_bits;
            const bool same = slot.valid && diff == 0;
            slot.layout = layout_bits;
            slot.stage_flags = flag_bits;
            slot.valid = true;
            sc.CountPushConst(same);
            push_same = same && sc.ActiveMode() != Mode::ValidateOnly;
        } else {
            slot.valid = false;
        }
    }
    if (!push_same) {
        cmdbuf.pushConstants(pipeline_layout, stage_flags, 0u, sizeof(push_data), &push_data);
    }

    // Bind descriptor set.
    if (set_writes.empty()) {
        return;
    }

    if (uses_push_descriptors) {
        // Descriptor delta cache: push descriptors are command buffer state,
        // so re-pushing a byte-identical write set onto the same command
        // buffer and layout is a spec-level no-op; skipping it saves the
        // driver-side descriptor update. Byte equality of the serialized
        // form proves redundancy, so the shadow verdict is structural.
        auto& sc = Skipcache::Framework::Instance();
        constexpr auto kCache = Skipcache::CacheId::DescDelta;
        if (sc.Active() && sc.ShouldProbe(kCache)) {
            auto& ctr = sc.Counters(kCache);
            ++ctr.eligible;
            const bool timed = sc.SampleTimer(kCache);
            const u64 t0 = timed ? sc.Now() : 0;
            const size_t idx = IsCompute() ? 1 : 0;
            auto& slot = sc.DescDeltaState(idx);
            auto& scratch = sc.DescDeltaScratch();
            static const bool inplace = EmulatorSettings.IsDescDeltaInplace();
            static const bool partial_enabled = [] {
                const bool partial = EmulatorSettings.IsDescDeltaPartial();
                if (partial && !EmulatorSettings.IsDescDeltaInplace()) {
                    LOG_WARNING(Render_Vulkan, "partial descriptor pushes need desc_delta_inplace; "
                                               "every content miss pushes the whole set");
                }
                return partial;
            }();
            const u64 tick = scheduler.CurrentTick();
            const u64 layout = std::bit_cast<u64>(static_cast<VkPipelineLayout>(pipeline_layout));
            const u64 foreign = sc.ForeignPushGen(idx);
            bool changed = false;
            const size_t size =
                inplace
                    ? (partial_enabled ? MatchDescriptorWrites<true>(set_writes, slot, changed)
                                       : MatchDescriptorWrites<false>(set_writes, slot, changed))
                    : SerializeDescriptorWrites(set_writes, scratch);
            bool would_hit = false;
            bool partial_ok = false;
            if (size == 0) {
                ++ctr.veto[0]; // unknown type or overflow: fail closed
            } else if (!slot.valid) {
                ++ctr.miss_cold;
            } else if (slot.tick != tick || slot.foreign_gen != foreign) {
                ++ctr.miss_gen[Skipcache::LaneTick];
            } else if (slot.layout != layout) {
                ++ctr.miss_key;
            } else if (slot.size != size ||
                       (inplace ? changed
                                : std::memcmp(slot.blob.data(), scratch.data(), size) != 0)) {
                ++ctr.veto[1]; // content changed
                // Same command buffer, foreign gen and layout: the driver's
                // set still holds the last push, so only the changed
                // descriptors need to go. Near-total changes push whole.
                partial_ok = inplace && partial_enabled && size == slot.size &&
                             !slot.header_changed && slot.desc_changed * 4 <= slot.desc_count * 3;
                if (partial_ok) {
                    ++slot.partial;
                    slot.descs += slot.desc_count;
                    slot.pushed += slot.desc_changed;
                }
            } else {
                would_hit = true;
                ++ctr.hits;
            }
            ++slot.probes;
            slot.hits += would_hit;
            if (timed) {
                ctr.guard_ns += sc.CorrectSample(sc.Now() - t0);
                ++ctr.guard_samples;
            }
            if (would_hit && sc.MayConsume(kCache) && !sc.ShouldVerify(kCache)) {
                return; // identical set already pushed on this command buffer
            }
            const bool timed_miss = timed && !would_hit;
            const u64 m0 = timed_miss ? sc.Now() : 0;
            if (partial_ok && sc.MayConsume(kCache) && !sc.ShouldVerify(kCache)) {
                // Each extra write costs the driver's per-write prologue, so a
                // compaction that saves fewer descriptors than it adds writes
                // is sent whole.
                const size_t runs = CompactDescriptorWrites(set_writes, slot, partial_scratch);
                if (runs <= set_writes.size() + (slot.desc_count - slot.desc_changed) / 2) {
                    slot.runs += runs;
                    cmdbuf.pushDescriptorSetKHR(
                        bind_point, pipeline_layout, 0,
                        vk::ArrayProxy<const vk::WriteDescriptorSet>(
                            static_cast<uint32_t>(partial_scratch.size()), partial_scratch.data()));
                } else {
                    ++slot.split;
                    cmdbuf.pushDescriptorSetKHR(
                        bind_point, pipeline_layout, 0,
                        vk::ArrayProxy<const vk::WriteDescriptorSet>(
                            static_cast<uint32_t>(set_writes.size()), set_writes.data()));
                }
            } else {
                cmdbuf.pushDescriptorSetKHR(
                    bind_point, pipeline_layout, 0,
                    vk::ArrayProxy<const vk::WriteDescriptorSet>(
                        static_cast<uint32_t>(set_writes.size()), set_writes.data()));
            }
            if (timed_miss) {
                ctr.miss_ns += sc.CorrectSample(sc.Now() - m0);
                ++ctr.miss_samples;
            }
            if (would_hit) {
                if (sc.GetState(kCache) != Skipcache::State::Learning) {
                    sc.RecordVerifyClean(kCache);
                }
                return;
            }
            if (size != 0) {
                slot.valid = true;
                slot.tick = tick;
                slot.layout = layout;
                slot.foreign_gen = foreign;
                slot.size = static_cast<u32>(size);
                // The in-place walk already left the blob in its serialized form.
                if (!inplace) {
                    std::memcpy(slot.blob.data(), scratch.data(), size);
                }
                sc.NotifyPopulated(kCache);
            } else {
                // A bailed in-place walk leaves the blob partially rewritten.
                slot.valid = false;
            }
            return;
        }
        cmdbuf.pushDescriptorSetKHR(
            bind_point, pipeline_layout, 0,
            vk::ArrayProxy<const vk::WriteDescriptorSet>(static_cast<uint32_t>(set_writes.size()),
                                                         set_writes.data()));
        return;
    }

    const auto desc_set = desc_heap.Commit(desc_layout);
    {
        // The heap leg is the delta-free stream; its size prices any future
        // second push set. DescDeltaState is ungated.
        auto& hs = VideoCore::Skipcache::Framework::Instance().DescDeltaState(
            bind_point == vk::PipelineBindPoint::eCompute ? 1 : 0);
        ++hs.heap;
        for (const auto& set_write : set_writes) {
            hs.heap_descs += set_write.descriptorCount;
        }
    }
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(
        vk::ArrayProxy<const vk::WriteDescriptorSet>(static_cast<uint32_t>(set_writes.size()),
                                                     set_writes.data()),
        {});
    cmdbuf.bindDescriptorSets(bind_point, pipeline_layout, 0, desc_set, {});
    // The heap set replaces this bind point's set 0 behind the delta cache;
    // the bump makes its next probe miss.
    VideoCore::Skipcache::Framework::Instance().BumpForeignPushGen(
        bind_point == vk::PipelineBindPoint::eCompute ? 1 : 0);
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

} // namespace Vulkan
