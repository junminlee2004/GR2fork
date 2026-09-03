// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <cstddef>
#include <cstring>

#include <boost/container/static_vector.hpp>

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
// every info struct, so each run copies as raw bytes. The image copy is 20 and
// not sizeof: a VkDescriptorImageInfo's four tail padding bytes are never
// initialised, and feeding them to the memcmp would make it nondeterministic.
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
              sizeof(VkDescriptorImageInfo) >= 20 &&
              offsetof(VkDescriptorImageInfo, sampler) == 0 &&
              offsetof(VkDescriptorImageInfo, imageView) == 8 &&
              offsetof(VkDescriptorImageInfo, imageLayout) == 16);
static_assert(sizeof(vk::BufferView) == 8);

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
                put(&infos[i], 20);
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

// Walks the write list once, leaving blob equal to the serialized form and
// reporting whether any byte differed; compares run even when an earlier key
// field already missed - one loop is cheaper than two. Same chunking and
// bail rules as SerializeDescriptorWrites.
size_t MatchDescriptorWrites(const Pipeline::DescriptorWrites& writes, std::array<u8, 16384>& blob,
                             bool& changed) {
    u8* cursor = blob.data();
    const u8* const limit = blob.data() + blob.size();
    u64 diff = 0;
    const auto sync8 = [&](const void* p) {
        u64 a, b;
        std::memcpy(&a, p, 8);
        std::memcpy(&b, cursor, 8);
        diff |= a ^ b;
        std::memcpy(cursor, p, 8);
        cursor += 8;
    };
    const auto sync4 = [&](const void* p) {
        u32 a, b;
        std::memcpy(&a, p, 4);
        std::memcpy(&b, cursor, 4);
        diff |= a ^ b;
        std::memcpy(cursor, p, 4);
        cursor += 4;
    };
    for (const auto& w : writes) {
        const u32 count = w.descriptorCount;
        if (static_cast<size_t>(limit - cursor) < 16 + size_t{count} * 24) {
            return 0;
        }
        const VkWriteDescriptorSet& raw = w;
        sync8(&raw.dstBinding);
        sync8(&raw.descriptorCount);
        switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer: {
            const auto* const infos = w.pBufferInfo;
            for (u32 i = 0; i < count; ++i) {
                const VkDescriptorBufferInfo& info = infos[i];
                sync8(&info.buffer);
                sync8(&info.offset);
                sync8(&info.range);
            }
            break;
        }
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampler: {
            const auto* const infos = w.pImageInfo;
            for (u32 i = 0; i < count; ++i) {
                const VkDescriptorImageInfo& info = infos[i];
                sync8(&info.sampler);
                sync8(&info.imageView);
                sync4(&info.imageLayout);
            }
            break;
        }
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer: {
            const auto* const views = w.pTexelBufferView;
            for (u32 i = 0; i < count; ++i) {
                sync8(&views[i]);
            }
            break;
        }
        default:
            return 0;
        }
    }
    changed = diff != 0;
    return static_cast<size_t>(cursor - blob.data());
}

} // namespace

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
    cmdbuf.pushConstants(pipeline_layout, stage_flags, 0u, sizeof(push_data), &push_data);

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
            bool changed = false;
            const size_t size = inplace ? MatchDescriptorWrites(set_writes, slot.blob, changed)
                                        : SerializeDescriptorWrites(set_writes, scratch);
            const u64 tick = scheduler.CurrentTick();
            const u64 layout = std::bit_cast<u64>(static_cast<VkPipelineLayout>(pipeline_layout));
            const u64 foreign = sc.ForeignPushGen(idx);
            bool would_hit = false;
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
            cmdbuf.pushDescriptorSetKHR(bind_point, pipeline_layout, 0, set_writes);
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
        cmdbuf.pushDescriptorSetKHR(bind_point, pipeline_layout, 0, set_writes);
        return;
    }

    const auto desc_set = desc_heap.Commit(desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
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
