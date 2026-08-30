// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <cstring>

#include <boost/container/static_vector.hpp>

#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/skipcache/skipcache.h"

namespace Vulkan {

namespace Skipcache = VideoCore::Skipcache;

namespace {

// Serialize a descriptor write list into a deterministic byte stream, payload
// contents included (the vk structs carry padding, so field-by-field is the
// only memcmp-safe form). Returns 0 on any unknown descriptor type or
// overflow: unknowns fail toward the slow path.
size_t SerializeDescriptorWrites(const Pipeline::DescriptorWrites& writes,
                                 std::array<u8, 16384>& out) {
    // One up-front capacity check instead of a bounds test per field: the
    // worst case per write is a 16-byte header plus 24 bytes per descriptor.
    // Overestimating (images serialize 20) only ever fails toward the slow
    // path, same as the old mid-stream overflow return.
    size_t need = 0;
    for (const auto& w : writes) {
        need += 16 + size_t{w.descriptorCount} * 24;
    }
    if (need > out.size()) {
        return 0;
    }
    u8* cursor = out.data();
    const auto put = [&](const void* p, size_t n) {
        std::memcpy(cursor, p, n);
        cursor += n;
    };
    const auto put64 = [&](u64 v) { put(&v, sizeof(v)); };
    const auto put32 = [&](u32 v) { put(&v, sizeof(v)); };
    for (const auto& w : writes) {
        put32(w.dstBinding);
        put32(w.dstArrayElement);
        put32(w.descriptorCount);
        put32(static_cast<u32>(w.descriptorType));
        switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer:
            for (u32 i = 0; i < w.descriptorCount; ++i) {
                const auto& b = w.pBufferInfo[i];
                put64(std::bit_cast<u64>(b.buffer));
                put64(b.offset);
                put64(b.range);
            }
            break;
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampler:
            for (u32 i = 0; i < w.descriptorCount; ++i) {
                const auto& im = w.pImageInfo[i];
                put64(std::bit_cast<u64>(im.sampler));
                put64(std::bit_cast<u64>(im.imageView));
                put32(static_cast<u32>(im.imageLayout));
            }
            break;
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer:
            for (u32 i = 0; i < w.descriptorCount; ++i) {
                put64(std::bit_cast<u64>(w.pTexelBufferView[i]));
            }
            break;
        default:
            return 0;
        }
    }
    return static_cast<size_t>(cursor - out.data());
}

} // namespace

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

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
    cmdbuf.pushConstants(*pipeline_layout, stage_flags, 0u, sizeof(push_data), &push_data);

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
            const size_t size = SerializeDescriptorWrites(set_writes, scratch);
            const u64 tick = scheduler.CurrentTick();
            const u64 layout = std::bit_cast<u64>(static_cast<VkPipelineLayout>(*pipeline_layout));
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
                       std::memcmp(slot.blob.data(), scratch.data(), size) != 0) {
                ++ctr.veto[1]; // content changed
            } else {
                would_hit = true;
                ++ctr.hits;
            }
            if (timed) {
                ctr.guard_ns += sc.CorrectSample(sc.Now() - t0);
                ++ctr.guard_samples;
            }
            if (would_hit && sc.MayConsume(kCache) && !sc.ShouldVerify(kCache)) {
                return; // identical set already pushed on this command buffer
            }
            const bool timed_miss = timed && !would_hit;
            const u64 m0 = timed_miss ? sc.Now() : 0;
            cmdbuf.pushDescriptorSetKHR(bind_point, *pipeline_layout, 0, set_writes);
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
                std::memcpy(slot.blob.data(), scratch.data(), size);
                sc.NotifyPopulated(kCache);
            } else {
                slot.valid = false;
            }
            return;
        }
        cmdbuf.pushDescriptorSetKHR(bind_point, *pipeline_layout, 0, set_writes);
        return;
    }

    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
    cmdbuf.bindDescriptorSets(bind_point, *pipeline_layout, 0, desc_set, {});
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
