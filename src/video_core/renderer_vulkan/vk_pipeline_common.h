// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_descriptor_set_cache.h"

#include <span>
#include <boost/container/small_vector.hpp>

namespace Shader {
struct Info;
struct PushData;
} // namespace Shader

namespace Vulkan {

static constexpr auto AllGraphicsStageBits =
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationControl |
    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eGeometry |
    vk::ShaderStageFlagBits::eFragment;

class Instance;
class Scheduler;
class DescriptorHeap;

class Pipeline {
public:
    Pipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
             DescriptorSetCache& desc_set_cache, const Shader::Profile& profile,
             vk::PipelineCache pipeline_cache, bool is_compute = false);
    virtual ~Pipeline();

    vk::Pipeline Handle() const noexcept {
        return *pipeline;
    }

    vk::PipelineLayout GetLayout() const noexcept {
        return *pipeline_layout;
    }

    auto GetStages() const {
        static_assert(static_cast<u32>(Shader::LogicalStage::Compute) == Shader::MaxStageTypes - 1);
        if (is_compute) {
            return std::span{stages.cend() - 1, stages.cend()};
        } else {
            return std::span{stages.cbegin(), stages.cend() - 1};
        }
    }

    const Shader::Info& GetStage(Shader::LogicalStage stage) const noexcept {
        return *stages[u32(stage)];
    }

    bool IsCompute() const {
        return is_compute;
    }

    using DescriptorWrites = std::vector<vk::WriteDescriptorSet>;
    using BufferBarriers = boost::container::small_vector<vk::BufferMemoryBarrier2, 16>;

    void BindResources(DescriptorWrites& set_writes, const BufferBarriers& buffer_barriers,
                       const Shader::PushData& push_data, const DescSetProbe& desc_probe) const;

    /// True when this pipeline's descriptor set is servable from the global content-keyed cache.
    bool DescCacheable() const noexcept {
        return desc_cacheable_;
    }
    /// True when any stage carries a ClipPlanes buffer. Not a veto: whether the descriptor rides
    /// the stream ring is decided per draw from the clipper_control register.
    bool DescHasClipPlanes() const noexcept {
        return desc_has_clip_planes_;
    }
    /// Fingerprint seed separating layout classes in fold space.
    u64 DescClassSeed() const noexcept {
        return desc_class_seed_;
    }
    /// True when this pipeline's descriptor set is UNCONDITIONALLY backed by a stream-ring copy
    /// (Flatbuf), which only the mode 3 memo makes repeatable. Its cache entries live for exactly
    /// one submit. The conditional ClipPlanes case is stamped on the probe per draw instead.
    bool DescTickBound() const noexcept {
        return desc_tick_bound_;
    }

protected:
    [[nodiscard]] std::string GetDebugString() const;

    struct DescClassify {
        bool cacheable;
        u8 veto;
        u32 num_elems;
        u32 num_writes;
        bool has_clip_planes;
        bool has_flatbuf;
    };
    /// Static cacheability test shared by the graphics and compute layout builders so the two can
    /// never drift. Reads only static Shader::Info fields - never a sharp - so the verdict is
    /// identical when preloading and at runtime.
    DescClassify ClassifyDescStages() const;
    void ClassifyDescSet(std::span<const vk::DescriptorSetLayoutBinding> bindings);
    bool DescDropPush() const noexcept;

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    DescriptorSetCache& desc_set_cache;
    const Shader::Profile& profile;
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniqueDescriptorSetLayout desc_layout;
    std::array<const Shader::Info*, Shader::MaxStageTypes> stages{};
    u32 desc_class_id_{DescriptorSetCache::kInvalidClass};
    u64 desc_class_seed_{};
    bool desc_cacheable_{};
    bool desc_has_clip_planes_{};
    bool desc_tick_bound_{};
    bool uses_push_descriptors{};
    bool is_compute;
};

} // namespace Vulkan
