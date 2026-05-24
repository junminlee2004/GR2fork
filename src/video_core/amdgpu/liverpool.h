// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <atomic>
#include <coroutine>
#include <exception>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>
#include <queue>

#include "common/assert.h"
#include "common/bounded_threadsafe_queue.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs_snapshot_pool.h"

namespace Vulkan {
class Rasterizer;
}

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

struct Liverpool {
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;     // actually 2, but HP is reserved by system software
    static constexpr u32 NumComputePipes = 7u; // actually 8, but #7 is reserved by system software
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u); // need to fit into u64 bitmap for ffs

    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    Regs regs{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};

    // Phase 1D-pre-C: the prior data-plane accessors
    //   GetGfxPipelineStamp() / IsGfxKeyDirty() / IsDynamicDirty() /
    //   ClearGfxKeyDirty()    / ClearDynamicDirty()
    // have been removed. Every consumer now reads `regs.gfx_pipeline_stamp`
    // / `regs.gfx_key_dirty` / `regs.dynamic_dirty` from the captured
    // LiverpoolRegsSnapshot, and the dirty-bit clear is folded into
    // `Liverpool::CaptureSnapshot()` under PM4's sole-writer ownership.
    // Removing the accessors enforces the new contract at compile time —
    // a future code change that re-introduces a live `liverpool->Is*Dirty()`
    // read on the data plane will fail to compile rather than silently
    // re-introduce the v1 hotfix1 race.

public:
    explicit Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);

    void SubmitDone() noexcept {
        mapped_queues[GfxQueueId].ccb_buffer_offset.store(0, std::memory_order_relaxed);
        mapped_queues[GfxQueueId].dcb_buffer_offset.store(0, std::memory_order_relaxed);
        submit_done.store(true, std::memory_order_release);
        NotifyGpu();
    }

    void WaitGpuIdle() noexcept {
        std::unique_lock lk{idle_mutex_};
        idle_cv_.wait(lk, [this] {
            return num_submits.load(std::memory_order_acquire) == 0;
        });
    }

    bool IsGpuIdle() const {
        return num_submits == 0;
    }

    [[nodiscard]] u32 GetNumSubmits() const noexcept {
        return num_submits.load(std::memory_order_acquire);
    }

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    // Capture the audited subset of `regs` plus the current queue's compute
    // program (`cs_state`) into the snapshot pool and return the slot index.
    // Thin wrapper over `LiverpoolRegsSnapshotPool::Capture`.
    //
    // Turn 1: this method is defined and reachable but is NOT yet called from
    // the PM4 draw/dispatch handlers. Wired into BundleAssembler::PushAndProcess
    // in Turn 2A.
    //
    // Turn 2B-1: extended to capture cs_state alongside regs so dispatch-path
    // consumers (compute pipeline lookup, BindBuffers's SharedMemory branch,
    // IsCompute*) can read from the snapshot uniformly with graphics consumers.
    // Reads `mapped_queues[curr_qid].cs_state` — the same value that
    // `GetCsRegs()` would return at this instant.
    //
    // Phase 1D-pre-C: extended to also capture five PM4-side hot-path fields
    // — `gfx_pipeline_stamp`, `gfx_key_dirty_`, `dynamic_dirty_`,
    // `last_cb_extent[]`, and `last_db_extent` — so the data plane reads
    // them from the snapshot rather than racing PM4. The dirty bits are
    // captured-and-then-cleared on the live Liverpool: PM4 is sole writer
    // of these flags, this function is called from PM4-side
    // BundleAssembler::PushAndProcess, so the read+clear pair is safe.
    // After capture, the live flags are false; the next PM4 register write
    // that warrants it will set them via Mark*Dirty / MarkGfxPipelineDirty
    // (which sets dynamic_dirty_) / MarkGfxKeyDirty.
    //
    // The previous data-plane consumer of `liverpool->ClearGfxKeyDirty()` /
    // `liverpool->ClearDynamicDirty()` (vk_pipeline_cache.cpp:543 and
    // vk_rasterizer.cpp:2723) were the only cleared sites; with Phase C
    // those calls are removed, and PM4 now owns the clear at capture time.
    // Tier-1 split: two snapshot pools with independent SPSC ring cursors.
    // Draw* intents capture into the graphics pool (cs_program omitted);
    // Dispatch* intents capture into the compute pool (minimal audited set).
    // Independent pools mean draws and dispatches don't share a ring head/tail
    // and don't false-share each other's slots.
    [[nodiscard]] u16 CaptureGfxSnapshot() noexcept {
        const u16 idx = gfx_snapshot_pool_.CaptureGfx(
            regs,
            gfx_pipeline_stamp.load(std::memory_order_relaxed),
            gfx_key_dirty_, dynamic_dirty_,
            last_cb_extent, last_db_extent);
        gfx_key_dirty_ = false;
        dynamic_dirty_ = false;
        return idx;
    }
    [[nodiscard]] u16 CaptureComputeSnapshot() noexcept {
        const u16 idx = compute_snapshot_pool_.CaptureCompute(
            regs, mapped_queues[curr_qid].cs_state,
            gfx_pipeline_stamp.load(std::memory_order_relaxed),
            gfx_key_dirty_, dynamic_dirty_,
            last_cb_extent, last_db_extent);
        gfx_key_dirty_ = false;
        dynamic_dirty_ = false;
        return idx;
    }

    // Const views of a previously-captured snapshot. Index must be one returned
    // by the matching CaptureXSnapshot() and not yet released.
    [[nodiscard]] const LiverpoolRegsSnapshot& GetGfxSnapshot(u16 idx) const noexcept {
        return gfx_snapshot_pool_.Slot(idx);
    }
    [[nodiscard]] const LiverpoolRegsSnapshot& GetComputeSnapshot(u16 idx) const noexcept {
        return compute_snapshot_pool_.Slot(idx);
    }

    // Release a snapshot slot back to its pool. Single-consumer FIFO per pool.
    void ReleaseGfxSnapshot(u16 idx) noexcept {
        gfx_snapshot_pool_.Release(idx);
    }
    void ReleaseComputeSnapshot(u16 idx) noexcept {
        compute_snapshot_pool_.Release(idx);
    }

    // Y-1 telemetry: live in-flight counts for HWM tracking.
    [[nodiscard]] u32 GfxSnapshotPoolInFlight() const noexcept {
        return gfx_snapshot_pool_.InFlight();
    }
    [[nodiscard]] u32 ComputeSnapshotPoolInFlight() const noexcept {
        return compute_snapshot_pool_.InFlight();
    }

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if (std::this_thread::get_id() == gpu_id) {
            return func();
        }
        if constexpr (wait_done) {
            std::binary_semaphore sem{0};
            command_queue_.EmplaceWait([&sem, &func] {
                func();
                sem.release();
            });
            num_commands.fetch_add(1, std::memory_order_release);
            NotifyGpu();
            sem.acquire();
        } else {
            command_queue_.EmplaceWait(std::move(func));
            num_commands.fetch_add(1, std::memory_order_release);
            NotifyGpu();
        }
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    inline ComputeProgram& GetCsRegs() {
        return mapped_queues[curr_qid].cs_state;
    }

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{};

private:
    struct Task {
        struct promise_type {
            auto get_return_object() {
                Task task{};
                task.handle = std::coroutine_handle<promise_type>::from_promise(*this);
                return task;
            }
            static constexpr std::suspend_always initial_suspend() noexcept {
                // We want the task to be suspended at start
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                    UNREACHABLE_MSG("Unhandled exception: {}", e.what());
                }
            }
            void return_void() {}
            struct empty {};
            std::suspend_always yield_value(empty&&) {
                return {};
            }
        };

        using Handle = std::coroutine_handle<promise_type>;
        Handle handle;
    };

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessCeUpdate(std::span<const u32> ccb);
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid);
    void BumpGfxPipelineStamp() noexcept {
        gfx_pipeline_stamp.fetch_add(1, std::memory_order_relaxed);
    }

    /// Mark pipeline state as potentially changed. Actual stamp bump deferred to draw time.
    void MarkGfxPipelineDirty() noexcept {
        pipeline_dirty_ = true;
        dynamic_dirty_ = true;
    }

    /// Mark that a key-affecting register changed (not just dynamic state).
    void MarkGfxKeyDirty() noexcept {
        pipeline_dirty_ = true;
        gfx_key_dirty_ = true;
        dynamic_dirty_ = true;
    }

    /// Bump the stamp only if dirty (called from draw/dispatch handlers).
    void FlushGfxPipelineDirty() noexcept {
        if (pipeline_dirty_) {
            BumpGfxPipelineStamp();
            pipeline_dirty_ = false;
        }
    }

    /// Returns true if the context register range is dynamic-state-only
    /// (viewport, scissor, blend constants, depth control, etc.)
    /// and does NOT affect the GraphicsPipelineKey.
    bool IsDynamicStateOnlyContextReg(u32 reg_addr) const noexcept {
        // Use pointer arithmetic against reg_array to compute word offsets.
        // This avoids offsetof on anonymous union/struct members (non-portable).
        const u32* base = regs.reg_array.data();
        auto wo = [base](const auto& field) noexcept -> u32 {
            return static_cast<u32>(reinterpret_cast<const u32*>(&field) - base);
        };
        auto in_range = [reg_addr](u32 start, u32 end) noexcept -> bool {
            return reg_addr >= start && reg_addr < end;
        };

        if (in_range(wo(regs.depth_bounds_min),
                      wo(regs.depth_clear) + 1)) return true;
        if (in_range(wo(regs.screen_scissor),
                      wo(regs.screen_scissor) + sizeof(regs.screen_scissor) / 4)) return true;
        if (in_range(wo(regs.window_offset),
                      wo(regs.window_scissor) + sizeof(regs.window_scissor) / 4)) return true;
        if (in_range(wo(regs.generic_scissor),
                      wo(regs.generic_scissor) + sizeof(regs.generic_scissor) / 4)) return true;
        if (in_range(wo(regs.viewport_scissors[0]),
                      wo(regs.viewport_depths[0]) + sizeof(regs.viewport_depths) / 4)) return true;
        if (in_range(wo(regs.index_offset),
                      wo(regs.primitive_restart_index) + 1)) return true;
        if (in_range(wo(regs.blend_constants),
                      wo(regs.stencil_ref_back) + sizeof(regs.stencil_ref_back) / 4)) return true;
        if (in_range(wo(regs.viewports[0]),
                      wo(regs.viewports[0]) + sizeof(regs.viewports) / 4)) return true;
        if (in_range(wo(regs.poly_offset),
                      wo(regs.poly_offset) + sizeof(regs.poly_offset) / 4)) return true;
        if (in_range(wo(regs.depth_control),
                      wo(regs.depth_control) + sizeof(regs.depth_control) / 4)) return true;
        if (in_range(wo(regs.viewport_control),
                      wo(regs.viewport_control) + sizeof(regs.viewport_control) / 4)) return true;
        return false;
    }

    void ProcessCommands();
    void Process(std::stop_token stoken);

    void NotifyGpu() {
        {
            std::lock_guard lk{wake_mutex_};
        }
        wake_cv_.notify_one();
    }

    void NotifyIdle() {
        {
            std::lock_guard lk{idle_mutex_};
        }
        idle_cv_.notify_all();
    }

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> submit_count{0};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        std::queue<Task::Handle> submits{};
        // PERF(GR2FORK v1.65): cached front task. GpuComm-only; written
        // from Liverpool::Process when a new task is fetched and cleared
        // when the task completes. Skips the m_access peek lock on every
        // coroutine yield in the same task — v1.64 [LP_MUTEX_METRICS]
        // showed ~99.88% of mutex acquisitions were redundant peeks of
        // an unchanged front handle. std::queue is FIFO; producer pushes
        // (SubmitGfx/SubmitAsc) don't change front; GpuComm is the only
        // popper. Caching front for the duration of resumption is
        // invariant-safe.
        Task::Handle current_task{};
        ComputeProgram cs_state{};
    };
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    std::atomic<u32> num_mapped_queues{1u};

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};

    struct ConstantEngine {
        void Reset() {
            ce_count = 0;
            de_count = 0;
            ce_compare_count = 0;
        }

        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_compare_count{};
        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    Vulkan::Rasterizer* rasterizer{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    std::jthread process_thread{};
    std::atomic<u64> gfx_pipeline_stamp{1};
    std::atomic<u32> num_submits{};
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};

    Common::MPSCQueue<Common::UniqueFunction<void>, 256> command_queue_;

    std::mutex wake_mutex_;
    std::condition_variable_any wake_cv_;

    std::mutex idle_mutex_;
    std::condition_variable idle_cv_;

    std::thread::id gpu_id;
    s32 curr_qid{-1};
    bool pipeline_dirty_{};
    bool gfx_key_dirty_{};
    bool dynamic_dirty_{};

    // === GR2FORK S-2: all-coroutines-stalled detection ===
    //
    // Bumped at the top of every PM4 dispatch loop iteration (graphics,
    // compute, CE) for each packet entered, AND once per ProcessCommands
    // callback invocation. The semantic is "an actual unit of forward
    // progress just happened on this thread."
    //
    // NOT bumped on the inner spin of wait-style opcodes (WaitRegMem,
    // WaitOnCeCounter, IB completion polling). Those re-enter the case
    // body via coroutine yield/resume without advancing past the
    // packet — the absence of further bumps is the "stalled" signal
    // that Process() reads.
    //
    // All bumps happen on the GpuComm thread:
    //   * PM4 dispatch loops run inside coroutines that Process() resumes
    //     on this thread
    //   * ProcessCommands callbacks execute synchronously on this thread
    //     (callers from other threads only push to command_queue_)
    // so non-atomic increment is correct.
    //
    // Read by Process() in its outer dispatch loop (`gpu_progress_round_`
    // tracks the value at the start of each round; if it doesn't advance
    // for STALLED_ROUNDS_THRESHOLD iterations, Process sleeps the thread
    // briefly). The signals that resolve waits (assembler fence writes
    // post-Vulkan-completion, presenter VO labels) come from other
    // threads, so sleeping doesn't stall its own progress.
    u64 pm4_progress_counter_{0};

    // Versioned per-draw reg snapshot pool. INERT in Turn 1 — the pool is
    // constructed but no captures occur until Stage 1 wiring lands in Turn 2.
    LiverpoolRegsSnapshotPool gfx_snapshot_pool_{};
    LiverpoolRegsSnapshotPool compute_snapshot_pool_{};
};

} // namespace AmdGpu
