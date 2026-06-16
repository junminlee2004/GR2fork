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

    // PERF(GR2FORK): spec-fingerprint -> perm_idx cache. Sits BELOW ud_hash_lru
    // and catches what it structurally cannot: draws where only resource
    // ADDRESSES changed. GR2 re-emits a fresh per-draw UBO pointer into a
    // buffer-sharp SGPR every draw, so ud_hash (XXH3 of the raw user_data
    // BYTES) differs every draw and BOTH fast paths above miss -- yet the
    // resulting StageSpecialization is byte-identical (the probe measured a
    // 100% redundant-respec ceiling: ~315 distinct specs rebuilt 12.2M times).
    //
    // Keyed on ComputeSpecProxyFp: a hash of runtime_info + start + every
    // bound sharp with base_address masked out (+ vertex-attribute sharps for
    // the Vertex stage). base_address is provably NOT a
    // StageSpecialization::ComputeSig input -- the *Spec structs extract only
    // format/type/dimension fields -- so the key is a SUPERSET of ComputeSig's
    // per-draw-varying inputs and maps 1:1 to spec.sig. It can therefore only
    // OVER-discriminate (a missed reclaim), never UNDER-discriminate (a wrong
    // reuse). Validated on GR2: ceiling 100%, reclaim 93.1%, COLLISIONS 0 over
    // 12.2M re-spec samples.
    //
    // This is NOT the banned v17 single-permutation shortcut and NOT a
    // CUSA03694-class stamp suppression: re-resolution is not skipped
    // (GetProgram is entered every draw, the key is recomputed every draw,
    // BindResources still writes the live address). Only the StageSpecialization
    // CONSTRUCT is skipped on a hit -- exactly what ud_hash_lru does. HS/DS are
    // excluded at the call site: their spec also folds tessellation-constant
    // CONTENTS read from guest memory, which a sharp-descriptor-only key does
    // not cover. Opt-in: GR2FORK_SPEC_FP_LRU=1.
    struct SpecFpCacheEntry {
        u64 fp{};
        u32 perm_idx{};
        bool valid{false};
    };
    // Direct-mapped, indexed by spec_fp & (size-1). Sizing is governed by the
    // busiest program's distinct-FINGERPRINT working set, which is larger than
    // the ~315 distinct-spec count because the proxy key over-discriminates
    // (the probe measured 93% reclaim vs a 100% sig-ceiling: masked-but-
    // spec-irrelevant sharp fields like num_records produce fresh fingerprints
    // for an already-seen sig). Measured knee on GR2: 64 slots -> 71% reclaim,
    // 1024 -> 91.6%, 4096 -> 97.2% (residual ~2.8% is the cold/new tail, not
    // conflicts, so larger buys ~0% more). Shrinking below the knee is pure
    // loss: the touched-line working set is identical regardless of capacity,
    // so a smaller table does NOT reduce the cache/TLB shadow -- it only
    // collides the same fingerprints, and each resulting miss reruns a full
    // StageSpecialization build that touches far more cache than the slot
    // saved. Lookup cost itself is size-independent (one index + one load);
    // only allocation (16 B/entry -> 64 KB/program here) and the
    // memset-at-construction scale with size, both negligible. If a future
    // title pushes realized reclaim down at 4096, the lever is set-
    // associativity, not a larger direct-mapped table.
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
    // value contains the attribute list internally, so every GetProgram
    // call copied the whole optional to construct the tuple, then std::tie
    // at the call site assigned it into a local optional, then
    // RefreshGraphicsStages did ANOTHER copy from local into
    // PipelineCache::fetch_shader. Two unnecessary deep copies per stage
    // per draw, ~5 stages × ~5000 draws/s = ~25K extra copies/s on the
    // GpuComm thread. perf attributed
    // _Optional_payload_base<FetchShaderData>::_M_copy_assign at 0.37% of
    // GpuComm under nominally-unrelated frames (skid). The pointer return
    // removes the redundant return-path copy; OPT1 (small_vector<...,16>
    // attribute storage) additionally makes the single remaining call-site
    // copy heap-free for the common <=16-attribute case.
    //
    // The pointer points into program->modules[N].spec.fetch_shader_data,
    // which is owned by the program (whose unique_ptr lives in
    // program_cache and has stable address). Lifetime is bounded by the
    // single GetProgram call's return path — the caller dereferences and
    // copies into its own storage exactly once. nullptr means "no fetch
    // shader" (e.g. compute stage, or vertex stage without VS_FETCH).
    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              const std::optional<Shader::Gcn::FetchShaderData>*, u64>;
    // PERF(GR2FORK STAGE-MEMO): `ctx_stable` is true ONLY when the caller is
    // the Level-2.5 (!regs.gfx_key_ctx_dirty) path in GetGraphicsPipeline,
    // routed through RefreshGraphicsStages(regs, /*ctx_stable=*/true). It
    // licenses the per-stage resolve memo (see StageResolveMemo below) to
    // skip BuildRuntimeInfo + HashRuntimeInfoForStage + the program_cache
    // map probe. Every other caller (full RefreshGraphicsKey path,
    // RefreshComputeKey) passes/defaults false and behaves byte-identically
    // to the pre-memo code.
    Result GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding,
                      const AmdGpu::LiverpoolRegsSnapshot& regs, bool ctx_stable = false);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey(const AmdGpu::LiverpoolRegsSnapshot& regs);
    // PERF(GR2FORK STAGE-MEMO): ctx_stable=true only from the Level-2.5 call
    // site in GetGraphicsPipeline (snapshot proved !gfx_key_ctx_dirty). The
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
        //
        // RACE FIX: the Shader::Info pointers serve two masters with different
        // lifetime/mutability needs, so they are split:
        //
        //   live_infos     — the program-owned Shader::Info pointers for THIS
        //                    draw. The finished GraphicsPipeline stores these in
        //                    Pipeline::stages and reads them per-draw in
        //                    BindResources, so they MUST stay live: the
        //                    GpuAssembler refreshes their flattened_ud_buf on
        //                    every draw (GetProgram -> RefreshFlatBuf), on its
        //                    own thread.
        //
        //   info_snapshot  — immutable per-launch DEEP COPIES of those Infos.
        //                    The (fenced) compile worker builds the descriptor-
        //                    set layout from these: BuildDescSetLayout reads
        //                    flattened_ud_buf via GetSharp to choose storage-vs-
        //                    uniform. Copying the Info also deep-copies its
        //                    flattened_ud_buf vector — the exact buffer
        //                    RefreshFlatBuf() reallocates. So when a budget-
        //                    skipped draw recurs next frame and the GpuAssembler
        //                    re-runs RefreshFlatBuf on the LIVE Info, it cannot
        //                    realloc/tear the buffer this worker is mid-read on.
        //                    That tear was the wedge: a torn sharp -> wrong
        //                    descriptor type in the layout -> RADV DEVICE_LOST.
        //
        //   snapshot_infos — pointers into info_snapshot in the ctor's span
        //                    shape (nullptr for absent stages).
        std::array<const Shader::Info*, MaxShaderStages> live_infos{};
        std::array<std::optional<Shader::Info>, MaxShaderStages> info_snapshot{};
        std::array<const Shader::Info*, MaxShaderStages> snapshot_infos{};
        std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos_copy{};
        std::array<vk::ShaderModule, MaxShaderStages> modules_copy{};
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_copy{};
        bool hang_warned{false};
        bool permafailed{false};
    };

    // Thresholds for the async-compile + driver-hang watchdog.
    //
    // GR2FORK: the synchronous sync-budget is now user-tunable via the [GPU]
    // gameplaySyncBudgetMs config key (default 10000; surfaced as a slider in
    // the Qt launcher, range 0..50000 ms). It is read fresh from Config at the
    // wait site in GetGraphicsPipeline — a cold-compile-only ([[unlikely]])
    // path, so the per-read cost is irrelevant and there is no hot-path
    // regression. The wait site blocks the GpuComm thread for up to that many
    // milliseconds on first encounter of a new pipeline; if the compile
    // finishes in that window the pipeline is installed inline, otherwise the
    // future is stashed in pending_graphics_pipelines and the draw is skipped
    // (TryFinalizePending polls non-blockingly on subsequent frames).
    //
    // A low budget is now SAFE. The "near-zero budget -> infinite loading"
    //   symptom (see chat "Infinite loading and shader cache issues",
    //   2026-05-25) was NOT inherent to skipping — it was a data race the skip
    //   path exposed: the fenced compile worker read the live Shader::Info
    //   (flattened_ud_buf, via GetSharp in BuildDescSetLayout) while the
    //   GpuAssembler re-ran RefreshFlatBuf() on it for the same draw recurring
    //   next frame, reallocating the buffer mid-read -> torn sharp -> wrong
    //   descriptor type -> RADV DEVICE_LOST / wedged load. That is fixed at the
    //   source: the worker now builds the layout from an immutable per-launch
    //   Info snapshot (see PendingGraphicsPipeline::info_snapshot), so skips can
    //   no longer corrupt a compile. A LOW budget is therefore preferred for
    //   throughput: it lets the GpuAssembler skip-and-backfill (stash the future,
    //   keep draining draws) instead of blocking on the future, keeping its
    //   dedicated core at full utilization. The only cost of a skip is brief
    //   pop-in (1-3 frames) on a pipeline's first appearance, never corruption.
    //   Default stays 10000ms (catches most cold compiles inline, zero pop-in);
    //   lower it via the slider if you want the assembler kept busy during loads.
    //
    // kHangLogThreshold / kPermaFailThreshold stay compile-time constants: they
    // govern the non-blocking escalation inside TryFinalizePending, not the
    // inline wait, and have no reason to be user-facing.
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

    // =========================================================================
    // PERF(GR2FORK STAGE-MEMO): per-LogicalStage resolve memo for the
    // Level-2.5 (!gfx_key_ctx_dirty) path.
    // =========================================================================
    // The Level-2.5 fast path re-resolves all stages on every ud-dirtied draw,
    // and that resolve repeats work whose inputs are provably frozen:
    //
    //   TryGetParams        2-3 guest-memory reads to re-derive {code, hash}
    //   BuildRuntimeInfo    full struct rebuild from snapshot regs
    //   HashRuntimeInfoForStage   per-stage field hashing (incl. XXH3 of the
    //                       vs outputs array)
    //   program_cache probe robin_map find on params.hash
    //
    // SOUNDNESS (why those four are skippable under ctx_stable):
    // the PM4 parser VALUE-DIFFS every register write
    // (CopyRegWordsFastIfChangedSmall) and classifies SH writes with
    // IsUserDataOnlyShReg: a write entirely inside one program's 16-word
    // user_data block raises gfx_key_dirty only; ANY other SH write
    // (program address, rsrc/settings) and any key-affecting context/uconfig
    // write raises gfx_key_ctx_dirty as well. Snapshot dirty windows tile the
    // capture timeline exactly (CaptureSnapshot clears the live flags), and a
    // ctx-dirty DRAW snapshot can never exit at Level 1/2/2.5 (ctx_dirty
    // implies key_dirty implies a stamp bump), so it always reaches the full
    // Level-3 resolve, which repopulates this memo. Therefore on any draw
    // arriving at Level 2.5 with !gfx_key_ctx_dirty, every memoized input —
    // program address, code span, params hash, all BuildRuntimeInfo register
    // inputs — has the same VALUE it had at the last GetProgram call for this
    // l_stage. Reusing the memo is byte-for-byte equivalent to recomputing;
    // user_data VALUES are the one thing allowed to change, and those are
    // explicitly repointed to the CURRENT snapshot on every hit and flow
    // through the unchanged downstream ladder (RefreshFlatBuf, ud XXH3,
    // last_result / ud_hash_lru / spec_fp tiers, AddBindings).
    //
    // ELIGIBILITY: trusted only for (Vertex->Vertex) and (Fragment->Fragment),
    // the GR2-dominant pairs whose BuildRuntimeInfo reads ONLY snapshot
    // registers (verified field-by-field). Geometry reads guest memory
    // (TryGetParams(regs.vs_program) for the GS copy shader) and Local reads
    // the hull stage's tessellation constants from guest memory — neither is
    // a pure function of frozen regs, so they never populate the memo and
    // always take the full path. Compute never passes ctx_stable.
    //
    // INHERITED EXPOSURE (not new): a PRE-EXISTING flag-swallowing hole lets
    // a Dispatch intent's CaptureSnapshot consume gfx_key_dirty/ctx_dirty
    // raised by a ctx write (liverpool.cpp DispatchDirect ~:1015-1035 and the
    // ASC path ~:1462 both FlushGfxPipelineDirty+capture). A subsequent draw
    // then sees clean flags despite the change. In the no-rearm case the
    // existing Level-2 exit already serves the stale pipeline BEFORE this
    // memo is ever consulted; in the ud-rearm case (ctx write -> dispatch
    // swallow -> ud write -> draw) Level 2.5 fires with a stale-by-swallow
    // ctx, exactly as it does today — the memo additionally reuses the stale
    // runtime_info the old code would have rebuilt, a strict subset-widening
    // WITHIN the same pre-existing exposure class, never a new trigger.
    // GR2's DCB emits ctx/ud writes immediately before their draws, which is
    // why the deployed Level-1/2 exits already survive this class at 30fps.
    //
    // STALE-CODE EXPOSURE (not new): trusting memo.code_data == pgm->Address
    // without re-reading the OrbShdr hash means a guest that overwrites
    // shader code in place WITHOUT touching any register is served the old
    // module. The deployed Level-1/2 exits already accept this exact class
    // (they return last_gfx_pipeline without looking at the program at all);
    // the memo's trust is strictly narrower (address verified + reg freeze).
    //
    // Lifetime: memo.program points at a Program owned by a unique_ptr in
    // program_cache — stable across robin_map rehash (v1.60 precedent).
    // Single-writer: all reads/writes on the assembler thread; plain fields.
    // Invalidation: params_opt failure in bind_stage (code became unreadable)
    // clears the entry; otherwise entries are overwritten by the next full
    // resolve for that l_stage.
    // Kill switch: GR2_NOSTAGEMEMO=1 disables CONSULT (populate may continue;
    // the env is read once per process).
    // Footprint: 6 x 40 B = 240 B — irrelevant to the Steam-Deck cache floor.
    struct StageResolveMemo {
        const u32* code_data{}; // == pgm->Address<u32*>() == params.code.data() at populate
        u64 params_hash{};      // bininfo shader_hash from the last real TryGetParams
        u64 ri_hash_raw{};      // HashRuntimeInfoForStage(runtime_infos[l]) BEFORE the binding fold
        // PERF(GR2FORK STAGE-MEMO ri-fp): raw-byte XXH3 of runtime_infos[l]
        // (sizeof(Shader::RuntimeInfo)) — the input ComputeSpecProxyFp mixes
        // first. Lazily filled the first time the spec-fp tier runs after a
        // resolve; RESET TO 0 at every populate so a value can never outlive
        // the runtime_info build it hashed. 0 doubles as the "unset" sentinel
        // (XXH3 yielding literal 0 is a 2^-64 fluke costing one recompute).
        u64 ri_fp_hash{};
        Program* program{};     // program_cache entry for params_hash (stable address)
        u32 code_words{};       // params.code.size() at populate
        bool valid{false};
    };
    std::array<StageResolveMemo, MaxShaderStages> stage_memo_{};

    // Fast path: if only shader user data changes, the graphics pipeline key does not.
    u64 last_gfx_stamp{};
    const GraphicsPipeline* last_gfx_pipeline{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    GraphicsPipelineKey prev_graphics_key_{};  // Key-level dedup: skip map lookup when unchanged
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
