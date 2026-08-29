// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <cstring>

#include <boost/container/static_vector.hpp>

#include "shader_recompiler/info.h"
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
    size_t off = 0;
    const auto put = [&](const void* p, size_t n) {
        if (off + n > out.size()) {
            return false;
        }
        std::memcpy(out.data() + off, p, n);
        off += n;
        return true;
    };
    const auto put64 = [&](u64 v) { return put(&v, sizeof(v)); };
    const auto put32 = [&](u32 v) { return put(&v, sizeof(v)); };
    for (const auto& w : writes) {
        if (!put32(w.dstBinding) || !put32(w.dstArrayElement) || !put32(w.descriptorCount) ||
            !put32(static_cast<u32>(w.descriptorType))) {
            return 0;
        }
        switch (w.descriptorType) {
        case vk::DescriptorType::eUniformBuffer:
        case vk::DescriptorType::eStorageBuffer:
            for (u32 i = 0; i < w.descriptorCount; ++i) {
                const auto& b = w.pBufferInfo[i];
                if (!put64(std::bit_cast<u64>(b.buffer)) || !put64(b.offset) || !put64(b.range)) {
                    return 0;
                }
            }
            break;
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampler:
            for (u32 i = 0; i < w.descriptorCount; ++i) {
                const auto& im = w.pImageInfo[i];
                if (!put64(std::bit_cast<u64>(im.sampler)) ||
                    !put64(std::bit_cast<u64>(im.imageView)) ||
                    !put32(static_cast<u32>(im.imageLayout))) {
                    return 0;
                }
            }
            break;
        case vk::DescriptorType::eUniformTexelBuffer:
        case vk::DescriptorType::eStorageTexelBuffer:
            for (u32 i = 0; i < w.descriptorCount; ++i) {
                if (!put64(std::bit_cast<u64>(w.pTexelBufferView[i]))) {
                    return 0;
                }
            }
            break;
        default:
            return 0;
        }
    }
    return off;
}

} // namespace

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   DescriptorSetCache& desc_set_cache_, const Shader::Profile& profile_,
                   vk::PipelineCache pipeline_cache, bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_},
      desc_set_cache{desc_set_cache_}, profile{profile_}, is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;

Pipeline::DescClassify Pipeline::ClassifyDescStages() const {
    DescClassify c{};
    // Mode 3 memoizes the Flatbuf and ClipPlanes stream copies in the rasterizer, so their
    // descriptors do repeat within a tick and veto 1 no longer applies. The resulting entries are
    // tick-bound: they only hit inside the command buffer that recorded the copy.
    const bool flat_memo = desc_set_cache.Mode() >= 3; // DescSetCacheEnabledFlatMemo
    u8 veto = 0;
    for (const auto* stage : GetStages()) {
        if (!stage) {
            continue;
        }
        for (const auto& buffer : stage->buffers) {
            ++c.num_elems;
            ++c.num_writes;
            if (buffer.buffer_type == Shader::BufferType::Flatbuf) {
                // StreamBuffer::Copy hands out a fresh ring offset every draw, so without the
                // mode 3 memo the descriptor content can never repeat.
                c.has_flatbuf = true;
                if (!flat_memo) {
                    veto = veto ? veto : 1;
                }
            } else if (buffer.buffer_type == Shader::BufferType::SharedMemory) {
                // Unconditional in every mode, including 3: the LDS path is a Map + memset with
                // no memo, and its offset advances on every dispatch.
                veto = veto ? veto : 2;
            } else if (buffer.buffer_type == Shader::BufferType::ClipPlanes) {
                // Not a veto: LowerUserClipPlanes inserts one for EVERY vertex-stage program
                // before its enabled-mask early return, so vetoing it would make every graphics
                // pipeline ineligible. Gated per draw on clipper_control instead.
                c.has_clip_planes = true;
            }
        }
        for (const auto& image : stage->images) {
            ++c.num_elems;
            ++c.num_writes;
            if (image.mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                // The only mode making NumBindings > 1: a partial write into a recycled set would
                // leave the previous tenant's descriptors in elements 1..N-1.
                veto = veto ? veto : 3;
            }
        }
        for ([[maybe_unused]] const auto& sampler : stage->samplers) {
            ++c.num_elems;
            ++c.num_writes;
        }
    }
    if (veto == 0 && (c.num_writes == 0 || c.num_elems > DescriptorSetCache::kMaxElems ||
                      c.num_writes > DescriptorSetCache::kMaxWrites)) {
        veto = 4;
    }
    c.veto = veto;
    c.cacheable = veto == 0;
    return c;
}

void Pipeline::ClassifyDescSet(std::span<const vk::DescriptorSetLayoutBinding> bindings) {
    const auto c = ClassifyDescStages();
    desc_has_clip_planes_ = c.has_clip_planes;
    // Stream-backed descriptors only repeat inside the tick that recorded the copy, so their
    // entries must not outlive it. Only Flatbuf is unconditional: it rides the ring on every draw
    // of such a pipeline, and only mode 3 makes it eligible at all. ClipPlanes rides the ring only
    // while the clipper register is on, so it is stamped per DRAW in Rasterizer::BindResources -
    // stamping it here would confine every clip-carrying pipeline (i.e. every graphics pipeline)
    // to one submit even when its descriptor is the stable null one.
    desc_tick_bound_ = c.has_flatbuf && desc_set_cache.Mode() >= 3; // DescSetCacheEnabledFlatMemo
    if (desc_set_cache.Mode() == 0) {
        return;
    }
    desc_set_cache.NoteClassified(c.cacheable, c.veto);
    if (!c.cacheable) {
        return;
    }
    desc_class_id_ = desc_set_cache.RegisterClass(bindings, c.num_elems, c.num_writes);
    if (desc_class_id_ == DescriptorSetCache::kInvalidClass) {
        return;
    }
    desc_cacheable_ = true;
    // splitmix64 of (class_id + 1): separates classes in fingerprint space.
    u64 z = static_cast<u64>(desc_class_id_) + 1;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    desc_class_seed_ = z ^ (z >> 31);
}

bool Pipeline::DescDropPush() const noexcept {
    return desc_cacheable_ && desc_set_cache.Mode() >= 2;
}

void Pipeline::BindResources(DescriptorWrites& set_writes, const BufferBarriers& buffer_barriers,
                             const Shader::PushData& push_data,
                             const DescSetProbe& desc_probe) const {
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

    // Global content-keyed set cache. No last-bound-set dedup lives here: bindDescriptorSets is
    // re-issued on every call, which is what makes this immune to the foreign pushDescriptorSetKHR
    // sites at set 0.
    //
    // An inactive probe is passed through rather than short-circuited on purpose. From mode 2 on
    // this pipeline's layout has no ePushDescriptorKHR, so the only fallback left below is
    // desc_heap.Commit; the cache must therefore own every descriptor update for a cacheable
    // pipeline, hit or not. It hands the draw back only when it is completely out of sets.
    if (desc_cacheable_ && desc_set_cache.Bind(desc_class_id_, desc_probe, set_writes,
                                               *pipeline_layout, cmdbuf, bind_point)) {
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
    // Binding at set 0 invalidates any push descriptors previously applied there, so the delta
    // cache's slot must be invalidated the same way a foreign push invalidates it.
    Skipcache::Framework::Instance().BumpForeignPushGen(IsCompute() ? 1 : 0);
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
