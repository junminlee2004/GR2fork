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
    // The MRU permutation's module, valid while mru_pipe_gen matches the
    // lookup's pipe_gen (ReplaceShader swaps modules in place and bumps it).
    vk::ShaderModule mru_module{};
    u64 mru_pipe_gen{};
    // Second entry under the canonical key: two specializations alternating
    // draw by draw both stay line-hot instead of probing the fingerprint table.
    u64 mru2_fp{};
    u32 mru2_perm_idx{};
    vk::ShaderModule mru2_module{};
    u64 mru2_pipe_gen{};
    void DemoteMru() {
        mru2_fp = mru_fp;
        mru2_perm_idx = mru_perm_idx;
        mru2_module = mru_module;
        mru2_pipe_gen = mru_pipe_gen;
    }
    void SwapMru() {
        std::swap(mru_fp, mru2_fp);
        std::swap(mru_perm_idx, mru2_perm_idx);
        std::swap(mru_module, mru2_module);
        std::swap(mru_pipe_gen, mru2_pipe_gen);
    }
    // Bit i set = modules[i].spec.fetch_shader_data is engaged. Read by
    // FetchShaderRef::operator bool so the per-stage probe stops striding into
    // the 1520-byte Module array for one engaged byte. ASSIGNED, never OR-ed:
    // InsertPermut's resize default-constructs gap Modules with disengaged
    // fetch data, and a twice-written slot must not keep a stale set bit.
    u64 fetch_mask{};
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
        SetFetchBit(modules.size() - 1);
        if (sig != 0) {
            perm_index_by_sig.try_emplace(sig, modules.size() - 1);
        }
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        const u64 sig = spec.sig;
        modules[perm_idx] = {module, std::move(spec)};
        SetFetchBit(perm_idx);
        if (sig != 0) {
            perm_index_by_sig.try_emplace(sig, perm_idx);
        }
    }

    void SetFetchBit(size_t perm_idx) {
        if (perm_idx < 64) {
            const u64 bit = u64{1} << perm_idx;
            fetch_mask = modules[perm_idx].spec.fetch_shader_data.has_value() ? fetch_mask | bit
                                                                              : fetch_mask & ~bit;
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
        // The mask read replaces a 1520-byte-strided probe into the Module
        // array; indices past the mask width fall back to the direct check.
        return program != nullptr &&
               (perm_idx < 64 ? ((program->fetch_mask >> perm_idx) & 1) != 0 : Get().has_value());
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

    /// Per-window telemetry for the stamp-keyed key reuse; silent when it is off.
    void DumpKeyReuseStats();
    /// Per-window telemetry for the program identity memo; silent while nothing ran.
    void DumpProgramIdentityStats();
    /// Per-window telemetry for the canonical specialization key; silent when it is off.
    void DumpSpecFpStats();

private:
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool ReuseGraphicsKey(u64 pipe_gen);
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
    // Stamp-keyed key reuse: the register stamp the last successful key was
    // built at. While it repeats, every register-derived field of that key
    // still holds and only the stage resolve reruns (ReuseGraphicsKey).
    u64 last_key_stamp{};
    bool key_stamp_reuse{};
    bool key_reuse_validate{}; // ValidateOnly mode: every reuse is rebuilt and compared
    u64 key_reuse_hits{};
    u64 key_reuse_rebuilds{}; // stamp repeated, the stage resolve changed the key
    u64 key_reuse_stamp_misses{};
    u64 key_reuse_mismatches{};
    // Cached value of the spec_mru_perm_probe setting, read once at construction.
    bool spec_mru_perm_probe{};
    // Stamp gate for BuildRuntimeInfo: while the graphics register stamp and
    // stage repeat, the rebuilt struct and its fingerprint hash are byte
    // identical, so both are skipped. Vertex and Fragment only - the other
    // arms read guest memory or untracked state. hash_valid is cleared on
    // every restamp: the new-program resolve returns before the hash site,
    // so a stale hash would alias the fingerprint of the PREVIOUS state.
    struct RuntimeInfoStamp {
        u64 stamp{};
        u64 ri_fp_hash{};
        u8 stage{};
        bool valid{};
        bool hash_valid{};
    };
    std::array<RuntimeInfoStamp, MaxShaderStages> ri_stamp{};
    bool ri_stamp_gate{};
    // Per logical stage: the last resolved program. Programs are added to
    // program_cache and never removed or re-seated (unique_ptr values survive
    // rehash), so a remembered (hash, Program*) pair is exactly what
    // try_emplace returns.
    // Under the params memo the slot also keeps the last binary-info
    // location; program_hash is separate from hash because a params miss
    // rewrites hash before GetProgram runs.
    struct StageIdentity {
        const u32* code{};
        const u64* hash_ptr{};
        u64 hash{};
        u32 len_dw{};
        Program* program{};
        u64 program_hash{};
    };
    std::array<StageIdentity, MaxShaderStages> stage_identity{};
    bool shader_params_memo{};
    u64 pgmid_map_hits{};
    u64 pgmid_map_probes{};
    u64 params_hits{};
    u64 params_misses{};
    template <typename Pgm>
    Shader::ShaderParams ResolveParams(Shader::LogicalStage l_stage, const Pgm& pgm);
    // Canonical specialization key: the sharp bytes the specialization reads,
    // masked, plus the runtime-info hash, the start bindings and the fetch
    // shader address. Value 2 keeps the last key per stage so a repeat is a
    // memcmp; the slot names one permutation of one program under one pipe_gen.
    struct GatherSlot {
        const Program* program{};
        u64 pipe_gen{};
        u64 perm_hash{};
        vk::ShaderModule module{};
        u32 perm_idx{};
        u32 len{};
        alignas(16) std::array<u8, 4096> buf{};
    };
    std::array<GatherSlot, MaxShaderStages> gather_slots{};
    u64 lookup_pipe_gen_{}; // pipe_gen of the current pipeline lookup
    u8 spec_fp_canonical{};
    bool spec_fp_validate{}; // ValidateOnly mode: every hit is rebuilt and compared
    u64 specfp_slot_hits{};
    u64 specfp_mru_hits{};
    u64 specfp_mru2_hits{};
    u64 specfp_table_hits{};
    u64 specfp_rebuilds{};
    u64 specfp_validate_misses{};
    void ValidateSpecHit(const Program& program, u32 hit_idx, const Shader::Info& info,
                         const Shader::RuntimeInfo& runtime_info, Shader::Backend::Bindings start);
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
