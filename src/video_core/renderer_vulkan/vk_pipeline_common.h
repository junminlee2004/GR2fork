// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <array>
#include <span>
#include <string_view>
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
class PipelineLayoutCache;

void DescWriteOverflow();

class Pipeline {
public:
    Pipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
             const Shader::Profile& profile, vk::PipelineCache pipeline_cache,
             PipelineLayoutCache* layouts, bool is_compute = false);
    virtual ~Pipeline();

    vk::Pipeline Handle() const noexcept {
        return *pipeline;
    }

    vk::PipelineLayout GetLayout() const noexcept {
        return pipeline_layout;
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

    static constexpr u32 NUM_DESCRIPTOR_WRITES = 128;

    /// Permanent, value-initialised once. The old vector re-value-initialised
    /// every element it grew into on every draw, while the fill sites overwrite
    /// all but sType, pNext and pTexelBufferView, which are constant for the
    /// life of the process. begin/end are bounded by count, never by capacity,
    /// so a stale tail entry is unreachable.
    class DescriptorWriteList {
    public:
        void clear() noexcept {
            count = 0;
        }
        bool empty() const noexcept {
            return count == 0;
        }
        std::size_t size() const noexcept {
            return count;
        }
        vk::WriteDescriptorSet* data() noexcept {
            return writes.data();
        }
        const vk::WriteDescriptorSet* data() const noexcept {
            return writes.data();
        }
        vk::WriteDescriptorSet* begin() noexcept {
            return writes.data();
        }
        vk::WriteDescriptorSet* end() noexcept {
            return writes.data() + count;
        }
        const vk::WriteDescriptorSet* begin() const noexcept {
            return writes.data();
        }
        const vk::WriteDescriptorSet* end() const noexcept {
            return writes.data() + count;
        }
        /// Fails closed: the assert helpers in this tree return, so running
        /// past the array would write into whatever follows it.
        vk::WriteDescriptorSet& Next() noexcept {
            if (count >= NUM_DESCRIPTOR_WRITES) [[unlikely]] {
                DescWriteOverflow();
                return overflow_slot;
            }
            return writes[count++];
        }
        void push_back(const vk::WriteDescriptorSet& w) noexcept {
            Next() = w;
        }

    private:
        std::array<vk::WriteDescriptorSet, NUM_DESCRIPTOR_WRITES> writes{};
        vk::WriteDescriptorSet overflow_slot{};
        u32 count{};
    };
    using DescriptorWrites = DescriptorWriteList;
    using BufferBarriers = boost::container::small_vector<vk::BufferMemoryBarrier2, 16>;

    void BindResources(DescriptorWrites& set_writes, const BufferBarriers& buffer_barriers,
                       const Shader::PushData& push_data) const;

protected:
    [[nodiscard]] std::string GetDebugString() const;
    /// Takes the pipeline's layouts from the shared cache, or creates owned ones.
    void AssignLayouts(std::span<const vk::DescriptorSetLayoutBinding> bindings,
                       vk::DescriptorSetLayoutCreateFlags flags,
                       const vk::PushConstantRange& push_constants, std::string_view debug_name);

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    const Shader::Profile& profile;
    vk::UniquePipeline pipeline;
    PipelineLayoutCache* layouts;
    vk::PipelineLayout pipeline_layout{};
    vk::DescriptorSetLayout desc_layout{};
    vk::UniquePipelineLayout owned_pipeline_layout;
    vk::UniqueDescriptorSetLayout owned_desc_layout;
    std::array<const Shader::Info*, Shader::MaxStageTypes> stages{};
    bool uses_push_descriptors{};
    bool is_compute;
};

} // namespace Vulkan
