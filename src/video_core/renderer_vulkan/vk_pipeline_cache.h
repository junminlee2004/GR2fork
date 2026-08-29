// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <variant>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};
    // Index of the permutation matched by the previous lookup; probed first when
    // the MRU probe setting is enabled. Always bounds-checked before use.
    u32 last_hit_perm{};

    // Fast lookup for shader permutations by specialization signature; a sig-map hit confirmed
    // by sig2 replaces the deep StageSpecialization comparisons on hot paths. Populated only
    // while the spec_fp_cache setting is on (specs carry sig == 0 otherwise). Never
    // invalidated: modules are append-only and entries hold indices, not handles.
    tsl::robin_map<u64, size_t> perm_index_by_sig{};

    // ComputeSpecProxyFp -> perm_idx cache for draws where only resource addresses changed.
    // base_address is masked and not a fingerprint input; the key only over-discriminates
    // (GR2: 93.1% reclaim, 0 collisions over 12.2M samples). HS/DS excluded (their spec folds
    // guest tess constants). Stores perm_idx and reads the module fresh on every hit, so the
    // ReplaceShader in-place module swap needs no invalidation here.
    struct SpecFpCacheEntry {
        u64 fp{};
        u32 perm_idx{};
        bool valid{false};
    };
    // Direct-mapped, indexed by spec_fp & (size-1). Measured reclaim knee on GR2: 64 slots ->
    // 71%, 1024 -> 91.6%, 4096 -> 97.2% (residual ~2.8% is the cold tail, not conflicts); a
    // smaller table only collides fingerprints without shrinking the touched-line working set
    // (16 B/entry). Heap-backed and allocated on first probe so the disabled path keeps the
    // current Program footprint.
    // One-entry MRU in front of the fingerprint table: consecutive draws
    // overwhelmingly repeat the fingerprint, and a hit here touches no extra
    // cache line. Same validity contract as the table (fp match + index
    // bound; modules are append-only, ReplaceShader swaps in place).
    u64 mru_fp{};
    u32 mru_perm_idx{};
    static constexpr size_t kSpecFpCacheSize = 4096;
    std::unique_ptr<std::array<SpecFpCacheEntry, kSpecFpCacheSize>> spec_fp_lru{};

    Program() = default;
    Program(Shader::Stage stage, Shader::LogicalStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        // Only keep the first index for a given sig; multiple permutation indices may map to
        // the same specialization (safe to reuse the same module). sig == 0 means the
        // signature was never computed (spec_fp_cache off), so the map stays empty.
        const u64 sig = spec.sig;
        modules.emplace_back(module, std::move(spec));
        if (sig != 0) {
            perm_index_by_sig.try_emplace(sig, modules.size() - 1);
        }
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        const u64 sig = spec.sig;
        modules[perm_idx] = {module, std::move(spec)};
        if (sig != 0) {
            perm_index_by_sig.try_emplace(sig, perm_idx);
        }
    }
};

// Names a fetch shader by (program, permutation) and resolves it against stored
// module specs at use time, so module vector reallocation cannot dangle it.
struct FetchShaderRef {
    const Program* program{};
    u32 perm_idx{};

    const std::optional<Shader::Gcn::FetchShaderData>& Get() const {
        return program->modules[perm_idx].spec.fetch_shader_data;
    }

    explicit operator bool() const {
        return program != nullptr && Get().has_value();
    }
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage,
                           std::optional<Shader::Gcn::FetchShaderData>& fetch_out);

    const GraphicsPipeline* GetGraphicsPipeline();

    /// The key refreshed by the latest GetGraphicsPipeline call.
    const GraphicsPipelineKey& CurrentGraphicsKey() const noexcept {
        return graphics_key;
    }

    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, FetchShaderRef, u64>;
    Result GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::Stage stage, Shader::LogicalStage l_stage);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    FetchShaderRef fetch_shader_ref{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start
    // Persistent probe object for GetProgram; rebuilt in place every lookup.
    Shader::StageSpecialization spec_scratch{};
    // Result of the previous successful graphics lookup; a repeated key skips the
    // hash and map probe. Validated against pipe_gen, which ReplaceShader bumps.
    GraphicsPipelineKey last_graphics_key{};
    const GraphicsPipeline* last_graphics_pipeline{};
    u64 last_graphics_pipe_gen{};
    // Cached value of the spec_mru_perm_probe setting, read once at construction.
    bool spec_mru_perm_probe{};
    // Cached value of the spec_fp_cache setting, read once at construction (before WarmUp so
    // deserialized permutations get signatures). Gates the spec-fingerprint tier and the
    // (sig, sig2) permutation resolve in GetProgram; off means byte-identical legacy behavior.
    bool spec_fp_cache{};

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
