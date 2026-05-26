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

    // PERF(GR2 v16): Cache the last GetProgram result per program.
    // When the pipeline stamp changes (e.g. viewport/scissor update) but the shader's
    // user_data, runtime_info, and binding offsets are unchanged, we can skip the expensive
    // StageSpecialization construction (~2.26% of GpuComm) entirely.
    struct LastResultCache {
        u64 ud_hash{};           // stage-aware hash of user_data + runtime_info + binding
        u64 ri_bind_hash{};      // hash of runtime_info + binding only (for stable program shortcut)
        size_t perm_idx{};
        u64 perm_hash{};
        vk::ShaderModule module{};
        bool valid{false};
    } last_result{};

    // PERF(GR2 v17): Stability tracking for single-permutation programs.
    // When a program has had only 1 permutation for many consecutive calls, mark it as "stable".
    // For stable single-permutation programs, skip StageSpecialization construction when only
    // user_data addresses change (stride/format/etc. are extremely unlikely to change).
    u32 stability_counter{};
    static constexpr u32 kStabilityThreshold = 64;
    static constexpr u32 kStabilityRevalidateInterval = 512;

    // PERF(GR2FORK): ud_hash → perm_idx cache. Sits between LastResultCache
    // (single-entry, "exact same call as last time") and the
    // StageSpecialization constructor + sig lookup. Catches the common case
    // where a program is called with one of a small set of distinct ud_hashes
    // in rotation (e.g. UI vs in-world fragment shaders sharing a program).
    //
    // Distinct from the BANNED v17 stable-single-permutation shortcut:
    // - v17 keyed on ri_bind_hash (no SGPR content). Different sharps with
    //   the same slot layout collided → stale module served → green
    //   garbled effects, vertex explosions in GR2.
    // - This keys on ud_hash, which is XXH3 of the full user_data BYTES.
    //   Different SGPRs → different ud_hash → different cache slot or
    //   miss → no cross-contamination. Same correctness invariant the
    //   surviving line-978 last_result fast-path relies on.
    //
    // 32 direct-mapped slots; ~512 bytes per Program. Bumped on every
    // success path that resolves perm_idx after a StageSpecialization
    // construction. Never invalidated proactively — perm_idx is only
    // appended-to in modules, so old indices stay valid. Bounds-checked
    // against modules.size() on lookup as a defensive guard.
    struct UdHashCacheEntry {
        u64 ud_hash{};
        u32 perm_idx{};
        bool valid{false};
    };
    static constexpr size_t kUdHashCacheSize = 32;
    std::array<UdHashCacheEntry, kUdHashCacheSize> ud_hash_lru{};

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

    // Deserializes the on-disk shader cache into program_cache /
    // graphics_pipelines / compute_pipelines. Used to be called from this
    // ctor; was moved out so the Presenter (which is *this object's
    // grand-owner) is fully constructed when WarmUp runs and can drive a
    // "LOADING SHADERS" overlay on the swapchain instead of presenting a
    // black window for the 30+ seconds it can take with a large cache.
    //
    // `tick` is invoked from this thread once per blob processed; `loaded`
    // is the running tally and `total` is the precomputed CountBlobs result
    // (constant for the duration of the call). The callback is optional —
    // passing {} preserves the historical behavior.
    void WarmUp(const std::function<void(u32 loaded, u32 total)>& tick = {});
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    // Phase 1D-0b (Turn 2B-1): the public lookup methods now take a
    // const reference to the captured reg snapshot rather than reading
    // `liverpool->regs` internally. The snapshot mirrors `Regs`'s field
    // interface, so the lookup bodies (RefreshGraphicsKey, RefreshGraphicsStages,
    // RefreshComputeKey, BuildRuntimeInfo) read regs.X identically to the
    // pre-2B-1 path — only the source switches.
    const GraphicsPipeline* GetGraphicsPipeline(const AmdGpu::LiverpoolRegsSnapshot& regs);

    const ComputePipeline* GetComputePipeline(const AmdGpu::LiverpoolRegsSnapshot& regs);

    // PERF(GR2FORK v1.60): the third tuple element used to be a
    // std::optional<Shader::Gcn::FetchShaderData> RETURNED BY VALUE. That
    // value contains a std::vector<VertexAttribute> internally, so every
    // GetProgram call performed a heap-alloc + memcpy of the attribute
    // vector to construct the tuple, then std::tie at the call site
    // assigned it into a local optional, then RefreshGraphicsStages did
    // ANOTHER copy from local into PipelineCache::fetch_shader. Two
    // unnecessary deep copies per stage per draw, ~5 stages × ~5000 draws/s
    // = ~25K extra heap ops/s on the GpuComm thread. perf attributed
    // _Optional_payload_base<FetchShaderData>::_M_copy_assign at 0.37% of
    // GpuComm under nominally-unrelated frames (skid).
    //
    // The pointer points into program->modules[N].spec.fetch_shader_data,
    // which is owned by the program (whose unique_ptr lives in
    // program_cache and has stable address). Lifetime is bounded by the
    // single GetProgram call's return path — the caller dereferences and
    // copies into its own storage exactly once. nullptr means "no fetch
    // shader" (e.g. compute stage, or vertex stage without VS_FETCH).
    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              const std::optional<Shader::Gcn::FetchShaderData>*, u64>;
    Result GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding,
                      const AmdGpu::LiverpoolRegsSnapshot& regs);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey(const AmdGpu::LiverpoolRegsSnapshot& regs);
    bool RefreshGraphicsStages(const AmdGpu::LiverpoolRegsSnapshot& regs);
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

    // =========================================================================
    // OPT(GR2 v78): Async graphics pipeline compile + driver-hang watchdog.
    // =========================================================================
    // shadPS4 has been hanging silently inside `vkCreateGraphicsPipelines` on
    // certain (pipeline, RADV Mesa version) combinations — the synchronous
    // Vulkan call never returns. Since GetGraphicsPipeline runs on the GPU
    // submit thread, a driver hang there freezes the entire emulator.
    //
    // Fix: launch the GraphicsPipeline ctor on a worker via std::async. Wait
    // briefly on the future (most compiles take <50ms — zero behavior change
    // for the common case). If the budget elapses, return nullptr from
    // GetGraphicsPipeline and stash the future in `pending_graphics_pipelines`.
    // Rasterizer::Draw already handles nullptr as "skip this draw" (frame-skip).
    // On every subsequent call for the same key, non-blocking-poll the future.
    // Log loudly once past kHangLogThreshold; mark permafailed + move future
    // to the file-local graveyard past kPermaFailThreshold so we neither block
    // destruction nor join a hung thread.
    struct PendingGraphicsPipeline {
        std::future<std::unique_ptr<GraphicsPipeline>> future;
        std::chrono::steady_clock::time_point started_at;
        u64 pipeline_hash{};
        GraphicsPipeline::SerializationSupport sdata{};
        // Stage-data deep copies. GraphicsPipeline's ctor takes spans into
        // PipelineCache::infos/runtime_infos/modules — those cache members are
        // overwritten on the next RefreshGraphicsStages, so the async task
        // cannot rely on them. Copy once at launch.
        std::array<const Shader::Info*, MaxShaderStages> infos_copy{};
        std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos_copy{};
        std::array<vk::ShaderModule, MaxShaderStages> modules_copy{};
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_copy{};
        bool hang_warned{false};
        bool permafailed{false};
    };

    // Thresholds. Tune via constants here; no config plumbing to avoid scope creep.
    //
    // PERF(post-B1, spike-score targeted): kInitialSyncBudget was 200ms, which
    // blocked the GpuComm thread on first encounter of a new pipeline for up to
    // that duration. With multiple new pipelines per frame (typical at warmup
    // and scene transitions) this stacked into 400-1100ms frame spikes that
    // dominate the spike-frametime score (~75% of every captured baseline run):
    //
    //   warmup spike at frame ~68: ~1086ms ≈ 5 × 200ms (5 pipelines serialized)
    //   scene cluster at frame ~492: ~433ms ≈ 2 × 200ms (2 pipelines serialized)
    //   200ms ceiling spikes at frame ~493: exactly the budget
    //
    // GR2FORK photo-mode-gated budget (replaces the single kInitialSyncBudget):
    //
    //   kGameplaySyncBudget = 0  — normal play. Compile is fully async on first
    //     encounter. wait_for(0) still returns ready for compiles that already
    //     finished (rare, serialized cache hits) so those proceed; anything not
    //     ready is stashed in pending_graphics_pipelines and the draw is skipped
    //     this frame, then installed a few frames later via TryFinalizePending.
    //     Zero blocking on the assembler thread — this is what removes the
    //     steady few-fps regression that a non-zero budget caused.
    //
    //   kPhotoSyncBudget = 10s — a photo capture render is in progress, i.e. the
    //     bound color target is the 1024x1024 LINEAR photo RT (detected from regs
    //     at the wait site). A skipped draw is unacceptable here: the photo RT is
    //     copied out verbatim, so a draw skipped for a not-yet-ready pipeline
    //     yields a blank/black capture. The photo-render pipelines are large and,
    //     COLD (first capture of a session / empty disk cache), take ~2.5s to
    //     compile — far beyond the old 15ms, which is exactly why the first
    //     capture was black. We block long enough to finish even a cold compile
    //     so the capture render never skips. A capture is rare, user-initiated,
    //     and non-realtime, so a one-time multi-second stall while a cold photo
    //     pipeline compiles is the acceptable price of a perfect capture; warm
    //     captures (pipeline already on disk) don't stall at all.
    //
    // The wait site in GetGraphicsPipeline selects between these per-compile by
    // detecting the photo color target from regs (1024x1024 linear) — done there,
    // not via an RT-bind gate, because the draw path returns on a null pipeline
    // before BeginRendering binds the RT, so an RT-bind gate can't arm in time
    // for the first cold capture. Do not raise kGameplaySyncBudget above 0
    // without re-running the spike-score A/B — the warmup/scene-transition spikes
    // and the gameplay fps regression both return with any non-zero value.
    static constexpr std::chrono::milliseconds kGameplaySyncBudget{0};
    static constexpr std::chrono::milliseconds kPhotoSyncBudget{10000}; // cold photo-pipeline compile is ~2.5s; must exceed it so the capture never skips
    // POST-WARMUP STARTUP WINDOW (GR2FORK): for kPostWarmupWindow seconds after
    // WarmUp() finishes, hold a 10s sync budget so pipelines encountered during
    // the first scene load block long enough to finish rather than being stashed
    // as async-pending. This prevents a wave of skipped draws (and a visible
    // frame-skip / geometry-pop) immediately after the loading screen clears.
    // After the window expires the budget reverts to kGameplaySyncBudget (0ms)
    // and the normal fully-async path resumes — zero fps impact at steady state.
    static constexpr std::chrono::milliseconds kPostWarmupSyncBudget{10000};
    static constexpr std::chrono::seconds      kPostWarmupWindow{5};
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
    // OPT(GR2 v78): In-flight async compiles keyed on graphics_key.
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<PendingGraphicsPipeline>>
        pending_graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    // Fast path: if only shader user data changes, the graphics pipeline key does not.
    u64 last_gfx_stamp{};
    const GraphicsPipeline* last_gfx_pipeline{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    GraphicsPipelineKey prev_graphics_key_{};  // Key-level dedup: skip map lookup when unchanged
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    // POST-WARMUP STARTUP WINDOW: set by WarmUp() on completion. The time_point
    // is written before the atomic flag is released, so reading it after an
    // acquire-load of warmup_complete_ is safe by C++ happens-before.
    std::atomic<bool> warmup_complete_{false};
    std::chrono::steady_clock::time_point warmup_complete_time_{};

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
