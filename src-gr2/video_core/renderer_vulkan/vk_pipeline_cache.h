// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
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
struct Liverpool;
struct LiverpoolRegsSnapshot;
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
    static constexpr size_t MaxPermutations = 16;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};

    // Fast lookup for shader permutations by specialization signature.
    // Avoids repeated deep StageSpecialization comparisons on hot paths.
    tsl::robin_map<u64, size_t> perm_index_by_sig{};

    // GR2FORK PERF: last GetProgram result per program. When the pipeline stamp changes but
    // user_data, runtime_info, and binding offsets are unchanged, the StageSpecialization
    // construction (~2.26% of GpuComm) is skipped.
    struct LastResultCache {
        u64 ud_hash{};           // stage-aware hash of user_data + runtime_info + binding
        u64 ri_bind_hash{};      // hash of runtime_info + binding only (for stable program shortcut)
        size_t perm_idx{};
        u64 perm_hash{};
        vk::ShaderModule module{};
        bool valid{false};
    } last_result{};

    // GR2FORK PERF: a program that keeps a single permutation across many consecutive calls is
    // marked stable; stable programs skip StageSpecialization construction when only user_data
    // addresses change (stride/format/etc. are extremely unlikely to change).
    u32 stability_counter{};
    static constexpr u32 kStabilityThreshold = 64;
    static constexpr u32 kStabilityRevalidateInterval = 512;

    // GR2FORK PERF: ud_hash -> perm_idx cache between LastResultCache and the StageSpecialization
    // construct. Keys on XXH3 of the full user_data bytes; keying on ri_bind_hash alone is unsound
    // (same slot layout, different sharps -> stale module). Never invalidated: modules append-only.
    struct UdHashCacheEntry {
        u64 ud_hash{};
        u32 perm_idx{};
        bool valid{false};
    };
    static constexpr size_t kUdHashCacheSize = 32;
    std::array<UdHashCacheEntry, kUdHashCacheSize> ud_hash_lru{};

    // GR2FORK PERF: ComputeSpecProxyFp -> perm_idx cache (opt-in: GR2FORK_SPEC_FP_LRU=1) for draws
    // where only resource addresses changed. base_address is masked and not a ComputeSig input; the
    // key only over-discriminates (93.1% reclaim, 0/12.2M). HS/DS excluded (guest tess constants).
    struct SpecFpCacheEntry {
        u64 fp{};
        u32 perm_idx{};
        bool valid{false};
    };
    // Direct-mapped, indexed by spec_fp & (size-1). Measured reclaim knee on GR2: 64 slots -> 71%,
    // 1024 -> 91.6%, 4096 -> 97.2% (residual ~2.8% is the cold tail, not conflicts); a smaller
    // table only collides fingerprints without shrinking the touched-line working set (16 B/entry).
    static constexpr size_t kSpecFpCacheSize = 4096;
    std::array<SpecFpCacheEntry, kSpecFpCacheSize> spec_fp_lru{};

    Program() = default;
    Program(Shader::Stage stage, Shader::LogicalStage l_stage, Shader::ShaderParams params)
    : info{stage, l_stage, params} {
        modules.reserve(MaxPermutations);
        perm_index_by_sig.reserve(MaxPermutations * 2);
    }

        void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
            const u64 sig = spec.sig;
            modules.emplace_back(module, std::move(spec));
            // Only keep the first index for a given sig; multiple serialized permutation indices
            // may map to the same specialization (safe to reuse the same module).
            perm_index_by_sig.try_emplace(sig, modules.size() - 1);
        }

        void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                          size_t perm_idx) {
            modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
            const u64 sig = spec.sig;
            modules[perm_idx] = {module, std::move(spec)};
            perm_index_by_sig.try_emplace(sig, perm_idx);
                          }
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool);
    ~PipelineCache();

    // Deserializes the on-disk shader cache. Runs post-construction so the Presenter can drive a
    // "LOADING SHADERS" overlay instead of a black window for the 30+ seconds a large cache takes.
    // `tick` is called once per blob with the running tally and the CountBlobs total; {} disables.
    void WarmUp(const std::function<void(u32 loaded, u32 total)>& tick = {});
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    // GR2FORK: the public lookups take the captured reg snapshot instead of reading
    // liverpool->regs. The snapshot mirrors Regs' field interface, so the lookup bodies read
    // regs.X identically - only the source switches.
    const GraphicsPipeline* GetGraphicsPipeline(const AmdGpu::LiverpoolRegsSnapshot& regs);

    const ComputePipeline* GetComputePipeline(const AmdGpu::LiverpoolRegsSnapshot& regs);

    // GR2FORK PERF: the third tuple element is a pointer, not a by-value optional - the by-value
    // form deep-copied the attribute list twice per stage per draw (~0.37% of GpuComm). It points
    // into program-owned storage (stable address); caller copies once; nullptr = no fetch shader.
    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              const std::optional<Shader::Gcn::FetchShaderData>*, u64>;
    // GR2FORK PERF: ctx_stable is true only from the Level-2.5 (!regs.gfx_key_ctx_dirty) path in
    // GetGraphicsPipeline; it licenses the StageResolveMemo to skip BuildRuntimeInfo,
    // HashRuntimeInfoForStage, and the program_cache probe. All other callers pass false.
    Result GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding,
                      const AmdGpu::LiverpoolRegsSnapshot& regs, bool ctx_stable = false);

    const Shader::ResolvedStageResources* GetResolvedStageResources(
        const Shader::Info& info) const noexcept {
        const u32 index = static_cast<u32>(info.l_stage);
        if (index >= resolved_stage_resources_.size() ||
            (resolved_stage_mask_ & (1u << index)) == 0) {
            return nullptr;
        }
        const auto& resolved = resolved_stage_resources_[index];
        return resolved.Matches(info) ? &resolved : nullptr;
    }

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey(const AmdGpu::LiverpoolRegsSnapshot& regs);
    // GR2FORK PERF: ctx_stable=true only from the Level-2.5 call
    // site in GetGraphicsPipeline (the snapshot has !gfx_key_ctx_dirty). The
    // full RefreshGraphicsKey path passes false.
    bool RefreshGraphicsStages(const AmdGpu::LiverpoolRegsSnapshot& regs, bool ctx_stable);
    bool RefreshComputeKey(const AmdGpu::LiverpoolRegsSnapshot& regs);

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::Stage stage, Shader::LogicalStage l_stage,
                                                const AmdGpu::LiverpoolRegsSnapshot& regs);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

    // GR2FORK: async pipeline compile + driver-hang watchdog. vkCreateGraphicsPipelines can hang
    // forever on some (pipeline, RADV version) combos, freezing the submit thread. A compile past
    // the wait budget skips its draw and is polled on later calls; hung workers are never joined.
    struct PendingGraphicsPipeline {
        std::future<std::unique_ptr<GraphicsPipeline>> future;
        std::chrono::steady_clock::time_point started_at;
        u64 pipeline_hash{};
        GraphicsPipeline::SerializationSupport sdata{};
        // Stage-data deep copies: the ctor takes spans into cache members the next
        // RefreshGraphicsStages overwrites. live_infos must stay live for per-draw BindResources;
        // info_snapshot isolates the worker from RefreshFlatBuf realloc tears (RADV DEVICE_LOST).
        std::array<const Shader::Info*, MaxShaderStages> live_infos{};
        std::array<std::optional<Shader::Info>, MaxShaderStages> info_snapshot{};
        std::array<const Shader::Info*, MaxShaderStages> snapshot_infos{};
        std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos_copy{};
        std::array<vk::ShaderModule, MaxShaderStages> modules_copy{};
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_copy{};
        bool hang_warned{false};
        bool permafailed{false};
    };

    // GR2FORK: the inline wait budget is the [GPU] gameplaySyncBudgetMs config key (default 10000,
    // launcher slider 0..50000 ms), read fresh at the cold-compile wait site. A low budget is safe
    // (a skip costs 1-3 frames of pop-in) and keeps the assembler draining draws during loads.
    static constexpr std::chrono::seconds      kHangLogThreshold{5};
    static constexpr std::chrono::seconds      kPermaFailThreshold{30};

    // True if the pending entry's future is ready and the result was moved into
    // graphics_pipelines[key] (caller must then erase from pending map). False
    // if still compiling or permafailed.
    bool TryFinalizePending(PendingGraphicsPipeline& pending,
                            const GraphicsPipelineKey& key);

    std::unique_ptr<PendingGraphicsPipeline> LaunchAsyncPipelineCompile(
        const GraphicsPipelineKey& key, u64 pipeline_hash);

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
    // GR2FORK: In-flight async compiles keyed on graphics_key.
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<PendingGraphicsPipeline>>
        pending_graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};

    // GR2FORK PERF: per-stage resolve memo for the Level-2.5 path, disabled by GR2_NOSTAGEMEMO=1.
    // Under !gfx_key_ctx_dirty all memoized inputs are frozen (the PM4 parser raises ctx_dirty on
    // any write outside a user_data block); Vertex/Fragment only, Geometry/Local read guest memory.
    struct StageResolveMemo {
        const u32* code_data{}; // == pgm->Address<u32*>() == params.code.data() at populate
        u64 params_hash{};      // bininfo shader_hash from the last real TryGetParams
        u64 ri_hash_raw{};      // HashRuntimeInfoForStage(runtime_infos[l]) BEFORE the binding fold
        // GR2FORK PERF: raw-byte XXH3 of runtime_infos[l], lazily filled by the spec-fp tier and
        // reset to 0 at every populate so a value cannot outlive the runtime_info build it hashed
        // (0 doubles as the unset sentinel; a literal-0 XXH3 is a 2^-64 fluke, one recompute).
        u64 ri_fp_hash{};
        Program* program{};     // program_cache entry for params_hash (stable address)
        u32 code_words{};       // params.code.size() at populate
        bool valid{false};
    };
    std::array<StageResolveMemo, MaxShaderStages> stage_memo_{};

    // GR2FORK PERF: per-stage {params.hash -> Program*} memo consulted before the program_cache
    // probe on the non-memo GetProgram path. program_cache is append-only and unique_ptr-owned, so
    // hash equality means the same stable Program*; ReplaceShader mutates contents, not addresses.
    struct ProgramProbeMemo {
        u64 hash{};
        Program* program{};
    };
    std::array<ProgramProbeMemo, MaxShaderStages> pgm_probe_memo_{};

    // GR2FORK PERF: ctor-latched copies of the five process-constant gate getters the per-draw
    // ladder otherwise reads via cross-TU calls + magic-static guards (~6-10 per draw):
    // GR2_NOKEYCTXSKIP, GR2_NOSTAGECMP, GR2_NOSTAGEMEMO, GR2_NOUDHASHLRU, GR2_NOSPECFPLRU.
    const bool key_ctx_skip_enabled_;
    const bool stage_cmp_narrow_;
    const bool stage_memo_enabled_;
    const bool ud_hash_lru_enabled_;
    const bool spec_fp_lru_enabled_;

    // Fast path: if only shader user data changes, the graphics pipeline key does not.
    u32 resolved_stage_mask_{};
    u64 last_gfx_stamp{};
    const GraphicsPipeline* last_gfx_pipeline{};
    // GR2FORK PERF: compact copy of the three narrow-compared key fields,
    // co-located with the stamp/pipeline the fast path already touches;
    // prev_graphics_key_'s compared regions span several cold lines.
    // Refreshed only at the single prev_graphics_key_ write site.
    struct {
        std::array<size_t, MaxShaderStages> stage_hashes;
        u32 mrt_mask;
        u32 num_color_attachments;
    } prev_key_narrow_{};

    // GR2FORK PERF: last-key -> last-pipeline memo for the compute path; key equality (one u64)
    // implies the same stable unique_ptr-owned entry. Devtools ReplaceShader must invalidate this
    // memo AND last_gfx_pipeline or both dangle.
    size_t last_compute_key_{};
    const ComputePipeline* last_compute_pipeline_{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    GraphicsPipelineKey prev_graphics_key_{};  // Key-level dedup: skip map lookup when unchanged
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
    std::array<Shader::ResolvedStageResources, MaxShaderStages> resolved_stage_resources_{};
};

} // namespace Vulkan
