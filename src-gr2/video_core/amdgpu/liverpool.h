// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <atomic>
#include <cstdlib>
#include <immintrin.h> // _mm_pause in the pool-full spin
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

    // GR2FORK: deliberately no data-plane accessors for the pipeline stamp and dirty bits -
    // consumers read them from the captured LiverpoolRegsSnapshot, and CaptureSnapshot() clears
    // the dirty bits under PM4's sole-writer ownership. A live read here would race PM4.

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

    // Captures the audited `regs` subset, cs_state, and the PM4 hot fields (stamp, dirty bits,
    // CB/DB extents); dirty bits are captured-then-cleared under PM4's sole-writer ownership.
    // GR2FORK PERF: capture_cs=false skips the 320 B cs copy per draw. Kill: GR2_NOTRYCAP=1.
    [[nodiscard]] u16 CaptureSnapshot(bool capture_cs) noexcept {
        static const bool trycap_enabled = []() noexcept {
            const char* e = std::getenv("GR2_NOTRYCAP");
            return !(e && e[0] == '1');
        }();
        u16 idx;
        if (trycap_enabled) [[likely]] {
            // GR2FORK PERF: spin-then-park replaces a yield spin (2.33% of system cycles,
            // ~1M sched_yield/s while throttled; the pool is the measured system throttle,
            // hwm 64/64). Parking covers only stalls past ~30-60us. Kill: GR2_NOPOOLPARK=1.
            static const bool park_enabled = []() noexcept {
                const char* e = std::getenv("GR2_NOPOOLPARK");
                return !(e && e[0] == '1');
            }();
            const auto try_cap = [&]() noexcept {
                return snapshot_pool_.TryCapture(
                    regs, mapped_queues[curr_qid].cs_state,
                    gfx_pipeline_stamp.load(std::memory_order_relaxed), gfx_key_dirty_,
                    gfx_key_ctx_dirty_, dynamic_dirty_, last_cb_extent, last_db_extent,
                    capture_cs);
            };
            while ((idx = try_cap()) == LiverpoolRegsSnapshotPool::kInvalidSlot) [[unlikely]] {
                ProcessCommands();
                if (!park_enabled) {
                    std::this_thread::yield();
                    continue;
                }
                // GR2FORK PERF: adaptive pre-park spin. A fixed 24-round (~20us) budget burns
                // 18-19% of cycles when stalls outlast it; halve on burnout (floor 1 round,
                // a ~1us probe), double on a catch (cap 24). PM4-private. Kill: GR2_NOPOOLPARK=1.
                bool caught = false;
                for (u32 round = 0; round < pool_spin_budget_; ++round) {
                    for (u32 i = 0; i < 64; ++i) {
                        _mm_pause();
                    }
                    if ((idx = try_cap()) != LiverpoolRegsSnapshotPool::kInvalidSlot) {
                        caught = true;
                        break;
                    }
                }
                pool_spin_budget_ =
                    caught ? std::min(pool_spin_budget_ * 2, 24u)
                           : std::max(pool_spin_budget_ / 2, 1u);
                if (idx != LiverpoolRegsSnapshotPool::kInvalidSlot) {
                    break;
                }
                ProcessCommands();
                // Long stall: park. Register (seq_cst RMW) -> wake counter -> final recheck ->
                // wait; a release or SendCommand landing after the recheck bumps pm4_wake_,
                // failing the futex value compare, so the wait returns immediately.
                pm4_parked_.fetch_add(1, std::memory_order_seq_cst);
                const u32 w = pm4_wake_.load(std::memory_order_acquire);
                if ((idx = try_cap()) != LiverpoolRegsSnapshotPool::kInvalidSlot) {
                    pm4_parked_.fetch_sub(1, std::memory_order_release);
                    break;
                }
                pm4_wake_.wait(w, std::memory_order_acquire);
                pm4_parked_.fetch_sub(1, std::memory_order_release);
                // Loop: re-try capture and re-service commands.
            }
        } else {
            idx = snapshot_pool_.Capture(
                regs, mapped_queues[curr_qid].cs_state,
                gfx_pipeline_stamp.load(std::memory_order_relaxed),
                gfx_key_dirty_, gfx_key_ctx_dirty_, dynamic_dirty_,
                last_cb_extent, last_db_extent, capture_cs);
        }
        gfx_key_dirty_ = false;
        gfx_key_ctx_dirty_ = false;
        dynamic_dirty_ = false;
        return idx;
    }

    // Const view of a previously-captured snapshot. Index must be one returned
    // by CaptureSnapshot() and not yet released.
    [[nodiscard]] const LiverpoolRegsSnapshot& GetSnapshot(u16 idx) const noexcept {
        return snapshot_pool_.Slot(idx);
    }

    // Release a snapshot slot back to the pool. Single-consumer FIFO.
    void ReleaseSnapshot(u16 idx) noexcept {
        snapshot_pool_.Release(idx);
        // Wake the producer only when it is (about to be) parked: the flag is set with a seq_cst
        // RMW before the producer's final recheck, so a release landing in the window is caught
        // by that recheck or by this notify.
        if (pm4_parked_.load(std::memory_order_seq_cst) != 0) [[unlikely]] {
            WakePm4();
        }
    }

    // Telemetry: live in-flight count for HWM tracking.
    [[nodiscard]] u32 SnapshotPoolInFlight() const noexcept {
        return snapshot_pool_.InFlight();
    }

    // Wake a producer parked in CaptureSnapshot's pool-full wait. Must be called on every event a
    // parked producer may wait on: snapshot release (gated, see ReleaseSnapshot), host commands
    // (unconditional - SendCommand must never starve behind a full pool), and shutdown.
    void WakePm4() noexcept {
        pm4_wake_.fetch_add(1, std::memory_order_release);
        pm4_wake_.notify_one();
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
            WakePm4();
            sem.acquire();
        } else {
            command_queue_.EmplaceWait(std::move(func));
            num_commands.fetch_add(1, std::memory_order_release);
            NotifyGpu();
            WakePm4();
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

    // GR2FORK: measured key-dirty raisers: PA_SU_SC_MODE_CNTL 55.6% (98.6% transition rate),
    // SPI_SHADER_PGM_LO_VS 30.9%, PGM_LO_PS 3.9%, VGT_GS_MODE 3.7%, VGT_PRIMITIVE_TYPE 1.4%,
    // rest <= 1.2%. polygon_control is classified in liverpool.cpp's SetContextReg handler.
    /// Mark that a key-affecting register changed (not just dynamic state).
    void MarkGfxKeyDirty() noexcept {
        pipeline_dirty_ = true;
        gfx_key_dirty_ = true;
        gfx_key_ctx_dirty_ = true;
        dynamic_dirty_ = true;
    }

    /// GR2FORK: mark that a key-affecting SH register changed WITHOUT
    /// raising dynamic_dirty_. Used for user_data-only SH writes (the per-draw SRT
    /// root/resource-pointer re-emit). We must keep gfx_key_dirty_+stamp so that
    /// perm_idx/pipeline resolution stays correct (suppressing those breaks
    /// pipeline resolution - a CUSA03694-class hazard - and is NEVER done). But dynamic_dirty_ has exactly
    /// one reader - UpdateDynamicState - whose sub-functions read only context regs
    /// (viewport/scissor/blend/depth/raster). user_data CANNOT affect any of those,
    /// so leaving dynamic_dirty_ alone here is provably correct and saves the ~0.48%
    /// of spurious per-draw dynamic-state sub-rebuilds. See IsUserDataOnlyShReg.
    void MarkGfxKeyDirtyNoDynamic() noexcept {
        pipeline_dirty_ = true;
        gfx_key_dirty_ = true;
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

    /// GR2FORK: returns true iff the SH register write range
    /// [reg_addr, reg_addr+nwords) lies ENTIRELY inside one shader program's
    /// 16-word user_data block. These are the per-draw SRT root + resource
    /// pointer SGPRs that the guest re-emits every draw; they affect the
    /// pipeline key (so we still set gfx_key_dirty_) but cannot affect any
    /// dynamic state (so dynamic_dirty_ can be suppressed - see
    /// MarkGfxKeyDirtyNoDynamic).
    ///
    /// Mirrors IsDynamicStateOnlyContextReg's wo() pointer-arithmetic
    /// pattern. The "entire range in user_data" test is intentionally strict:
    /// a mixed packet that also touches a program-address/settings word (where
    /// perm-relevant sharp structure lives) fails the test and falls back to
    /// the full MarkGfxKeyDirty, so a genuine dynamic/key change riding in the
    /// same packet is never misclassified.
    bool IsUserDataOnlyShReg(u32 reg_addr, u32 nwords) const noexcept {
        if (nwords == 0) {
            return false;
        }
        const u32* base = regs.reg_array.data();
        auto wo = [base](const auto& field) noexcept -> u32 {
            return static_cast<u32>(reinterpret_cast<const u32*>(&field) - base);
        };
        const u32 end = reg_addr + nwords; // exclusive
        auto in_ud = [&](const UserData& ud) noexcept -> bool {
            const u32 s = wo(ud[0]);
            return reg_addr >= s && end <= s + AmdGpu::NUM_USER_DATA;
        };
        return in_ud(regs.ps_program.user_data) || in_ud(regs.vs_program.user_data) ||
               in_ud(regs.gs_program.user_data) || in_ud(regs.es_program.user_data) ||
               in_ud(regs.hs_program.user_data) || in_ud(regs.ls_program.user_data);
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
        // GR2FORK PERF: cached front task (GpuComm-only) skips the m_access peek lock on every
        // coroutine yield in the same task - ~99.88% of those acquisitions re-peek an unchanged
        // front. FIFO pushes never change front and GpuComm is the only popper, so this is safe.
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
    // GR2FORK PERF: 1-in-32 gate counters for the CE/ASC loops'
    // num_commands checks (PM4-thread-private, plain u32).
    u32 ce_gate_counter_{};
    u32 asc_gate_counter_{};
    // GR2FORK PERF: adaptive pre-park spin budget, rounds of 64 pauses, range
    // [1, 24]. PM4-thread-private; lives here because on the pm4_parked_ line
    // its per-stall rewrite invalidated the line ReleaseSnapshot loads per
    // intent.
    u32 pool_spin_budget_{24};
    // GR2FORK PERF: pool-full park plumbing. pm4_wake_ is the producer's park/wake eventcount;
    // pm4_parked_ gates the assembler-side notify so ReleaseSnapshot's steady-state cost is one
    // load. Waiter registers with a seq_cst RMW before its final recheck; wakers bump-then-notify.
    alignas(64) std::atomic<u32> pm4_wake_{0};
    alignas(64) std::atomic<u32> pm4_parked_{0};

    // GR2FORK PERF: cache-line partition. The three atomics below are written by other threads;
    // without the split they share a line with rasterizer/gfx_pipeline_stamp, which GpuComm
    // reads/bumps per draw, so every external write would invalidate GpuComm's hot line.
    alignas(64) std::atomic<u32> num_submits{};
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
    // GR2FORK: set when a key-affecting CONTEXT register changed (gfx_key_dirty_ causes that are
    // not user_data-only SH re-emits). Lets RefreshGraphicsKey skip rebuilding context-derived key
    // fields on per-draw SRT-pointer churn. Set by MarkGfxKeyDirty, not MarkGfxKeyDirtyNoDynamic.
    bool gfx_key_ctx_dirty_{};
    bool dynamic_dirty_{};

    // GR2FORK: all-coroutines-stalled detector. Bumped per PM4 packet and per ProcessCommands
    // callback, never in wait-opcode spins; if it stalls for STALLED_ROUNDS_THRESHOLD rounds,
    // Process() sleeps briefly (wakeups come from other threads). GpuComm-only, so non-atomic.
    u64 pm4_progress_counter_{0};

    // Versioned per-draw reg snapshot pool.
    LiverpoolRegsSnapshotPool snapshot_pool_{};
};

} // namespace AmdGpu
