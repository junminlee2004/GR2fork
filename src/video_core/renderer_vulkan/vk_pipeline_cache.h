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
    // Associative front over the fingerprint table: a program that cycles
    // through more specializations per frame than the MRU pair holds keeps
    // them here with the module carried, so a hit touches neither the 64 KB
    // table nor the 1520-byte-strided Module. fp 0 marks an empty entry; the
    // modules are valid while pipe_gen matches the lookup's, as for mru_module.
    static constexpr u32 kSpecFpFront = 16;
    struct alignas(64) SpecFpFront {
        std::array<u64, kSpecFpFront> fp{};
        std::array<vk::ShaderModule, kSpecFpFront> module{};
        std::array<u32, kSpecFpFront> perm_idx{};
        u64 pipe_gen{};
        u32 next{};
    };
    SpecFpFront front{};
    void FrontInsert(u64 fp, u32 idx, vk::ShaderModule module, u64 gen) {
        if (front.pipe_gen != gen) {
            front = {};
            front.pipe_gen = gen;
        }
        front.fp[front.next] = fp;
        front.module[front.next] = module;
        front.perm_idx[front.next] = idx;
        front.next = (front.next + 1) % kSpecFpFront;
    }

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

    /// Resolves the stage's program and publishes its info and module into
    /// infos[out_slot] and modules[out_slot] (read by the pipeline lookups) and,
    /// for graphics stages, the fetch shader reference (read by the vertex
    /// format walk and the graphics pipeline constructor). Returns the
    /// permutation hash. Compute publishes into slot 0.
    [[nodiscard]] u64 GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                                 const Shader::ShaderParams& params,
                                 Shader::Backend::Bindings& binding, u32 out_slot);

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
    /// Runs before every inline pipeline compile; the rasterizer drops its
    /// guest-copy hold there so a compile never holds the memory map open.
    void SetPreCompileHook(void (*hook)(void*), void* user) {
        pre_compile_hook_ = hook;
        pre_compile_user_ = user;
    }
    /// Per-window telemetry for the static color write mask: pipeline counts and skipped emits.
    void DumpColorMaskStats(u64 emit_skips);
    /// Per-window telemetry for the runtime-info input memo; silent when it is off.
    void DumpRuntimeInfoMemoStats();
    /// Shared layout count against the pipeline count; silent when sharing is off.
    void DumpLayoutStats();
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
    PipelineLayoutCache layouts; // destroyed after every pipeline
    bool share_layouts{};
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
    bool key_is_last{};     // graphics_key is byte-equal to last_graphics_key
    u64 stage_hash_diff{};  // OR of (old ^ new) over every stage hash an armed resolve rewrites
    bool hash_diff_armed{}; // set by the stamp-keyed reuse around its stage resolve
    bool key_reuse_hash_diff{};
    u64 key_reuse_diff_decisions{};
    u64 nonfull_mask_pipelines{};
    u64 graphics_lookups{};
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
    // Register words the Vertex, Fragment and Compute runtime-info arms read,
    // with the struct and fingerprint hash they produced; an equal snapshot
    // restores both. Two entries per logical stage keep an alternating
    // program pair resident.
    static constexpr size_t kRuntimeInputWords = 96;
    struct RuntimeInputMemo {
        u64 ri_fp_hash{};
        std::array<u32, kRuntimeInputWords> words{};
        u8 n_words{};
        bool used{};
        bool hash_valid{};
        Shader::RuntimeInfo ri{};
    };
    std::array<std::array<RuntimeInputMemo, 2>, MaxShaderStages> ri_memo{};
    // The entry whose bytes runtime_infos[l_stage] holds; null after any other rebuild.
    std::array<RuntimeInputMemo*, MaxShaderStages> ri_memo_last{};
    bool ri_input_memo{};
    bool ri_memo_validate{};
    u64 rimemo_hits{};
    u64 rimemo_misses{};
    u64 rimemo_restores{};
    u64 rimemo_vmiss{};
    u32 SnapshotRuntimeInputs(Shader::Stage stage, u32* out) const;
    bool MemoRuntimeInfo(Shader::Stage stage, Shader::LogicalStage l_stage, RuntimeInfoStamp& slot);
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
    // Direct-mapped table behind stage_identity, indexed by the code address:
    // a program a stage returns to resolves from it, validated by the search's
    // own anchors and the hash re-read. Empty unless the entries setting is set.
    struct alignas(64) IdentityEntry {
        StageIdentity id;
    };
    std::vector<IdentityEntry> identity_table{};
    u32 identity_table_mask{};
    u64 params_table_hits{};
    u64 params_table_misses{};
    u64 params_prefetches{};
    u32 IdentitySlot(const u32* code) const noexcept {
        return static_cast<u32>(
                   ((reinterpret_cast<uintptr_t>(code) >> 8) * 0x9E3779B97F4A7C15ull) >> 40) &
               identity_table_mask;
    }
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
    struct alignas(64) GatherSlot {
        const Program* program{};
        u64 pipe_gen{};
        u64 perm_hash{};
        vk::ShaderModule module{};
        u32 perm_idx{};
        u32 len{};
        alignas(64) std::array<u8, 4096> buf{};
    };
    std::array<GatherSlot, MaxShaderStages> gather_slots{};
    // The specialization key under construction. A member, not a 4 KB stack
    // array in the per-draw lookup: sized to GatherSlot::buf because the
    // validate mode copies a whole slot into it. Written only by the parser
    // thread, which is the sole entry to the lookup.
    alignas(64) std::array<u8, 4096> key_scratch{};
    u64 lookup_pipe_gen_{}; // pipe_gen of the current pipeline lookup
    void (*pre_compile_hook_)(void*){};
    void* pre_compile_user_{};
    u8 spec_fp_canonical{};
    bool spec_fp_validate{}; // ValidateOnly mode: every hit is rebuilt and compared
    bool spec_fp_slot_inplace{};
    bool spec_fp_front{};
    bool spec_key_align{};
    bool slot_prefetch{};
    bool spec_key_fused{};
    u64 specfp_slot_hits{};
    u64 specfp_mru_hits{};
    u64 specfp_mru2_hits{};
    u64 specfp_table_hits{};
    u64 specfp_rebuilds{};
    u64 specfp_validate_misses{};
    u64 specfp_inplace_bytes{};
    u64 specfp_ri_rehash{};
    u64 specfp_front_hits{};
    u64 specfp_slot_pf{};
    u64 specfp_fused{};
    u64 specfp_fused_miss{};
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
