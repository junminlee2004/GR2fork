// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// AsyncResourcePrep - worker thread that moves detile/tile compute dispatches (and other
// resource-prep work) off the GpuComm thread. The worker owns its own command pool, batches
// queued dispatches into one submit under Scheduler::submit_mutex, and signals a dedicated
// worker timeline that the parser waits on at its next SubmitExecution. The master timeline is
// never touched: a worker NextTick would break Scheduler::DeferOperation's assumption that the
// next parser submit signals the captured tick, freeing scratch buffers still in flight.

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

namespace Vulkan {

class AsyncResourcePrep {
public:
    /// Master toggle. When false the worker is disabled entirely and all
    /// work runs synchronously inline on the caller.
    static constexpr bool kEnabled = true;

    /// Maximum buffer descriptors per dispatch. Tile manager uses 3
    /// (tiled buffer SSBO, linear buffer SSBO, params UBO). Generous slack
    /// keeps the work item POD-copyable without dynamic allocation.
    static constexpr u32 kMaxBufferInfos = 8;

    /// Per-dispatch unit of work. POD-style for cheap queueing.
    struct DispatchWork {
        vk::Pipeline pipeline{};
        vk::PipelineLayout layout{};

        // Up to kMaxBufferInfos buffer descriptors. Each pBufferInfo is fixed up to point into
        // this struct's buffer_infos array right before recording, so the caller's stack-locals
        // do not need to outlive the queue insertion.
        std::array<vk::DescriptorBufferInfo, kMaxBufferInfos> buffer_infos{};
        std::array<vk::WriteDescriptorSet, kMaxBufferInfos> writes{};
        u32 num_writes{};

        // Compute dispatch dimensions.
        u32 dim_x{1};
        u32 dim_y{1};
        u32 dim_z{1};

        // Output buffer: gets a write-to-read barrier at the end of the recorded batch so the
        // parser's subsequent transfer/shader reads observe the dispatch's writes after the
        // semaphore wait. VK_NULL_HANDLE emits no barrier (no parser-side consumer).
        vk::Buffer out_buffer{};
        vk::DeviceSize out_size{0};
    };

    /// Build a DispatchWork for the canonical tile_manager descriptor layout
    /// (3 buffers: 2 SSBOs + 1 UBO at bindings 0/1/2). pushDescriptor writes
    /// in the work item are constructed against the work item's own
    /// buffer_infos array - no stale pointers into caller's stack.
    static DispatchWork MakeTilingDispatch(
        vk::Pipeline pipeline, vk::PipelineLayout layout,
        vk::Buffer tiled_buffer, vk::DeviceSize tiled_offset, vk::DeviceSize tiled_size,
        vk::Buffer linear_buffer, vk::DeviceSize linear_offset, vk::DeviceSize linear_size,
        vk::Buffer params_buffer, vk::DeviceSize params_offset, vk::DeviceSize params_size,
        u32 dim_x, vk::Buffer out_buffer, vk::DeviceSize out_size) {
        DispatchWork w{};
        w.pipeline = pipeline;
        w.layout = layout;
        w.dim_x = dim_x;
        w.out_buffer = out_buffer;
        w.out_size = out_size;
        w.num_writes = 3;
        w.buffer_infos[0] = vk::DescriptorBufferInfo{
            .buffer = tiled_buffer, .offset = tiled_offset, .range = tiled_size};
        w.buffer_infos[1] = vk::DescriptorBufferInfo{
            .buffer = linear_buffer, .offset = linear_offset, .range = linear_size};
        w.buffer_infos[2] = vk::DescriptorBufferInfo{
            .buffer = params_buffer, .offset = params_offset, .range = params_size};
        for (u32 i = 0; i < 3; ++i) {
            w.writes[i] = vk::WriteDescriptorSet{
                .dstSet = VK_NULL_HANDLE,
                .dstBinding = i,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = (i == 2) ? vk::DescriptorType::eUniformBuffer
                                           : vk::DescriptorType::eStorageBuffer,
                .pImageInfo = nullptr,
                .pBufferInfo = nullptr,  // fixed up at record time
                .pTexelBufferView = nullptr,
            };
        }
        return w;
    }

    AsyncResourcePrep(const Instance& instance,
                      MasterSemaphore& master_semaphore,
                      std::mutex& submit_mutex)
        : instance_{instance},
          master_semaphore_{master_semaphore},
          submit_mutex_{submit_mutex} {
        if constexpr (!kEnabled) {
            return;
        }
        const vk::Device device = instance_.GetDevice();

        // Worker timeline. Distinct from master so worker submits never
        // interleave master ticks with parser submits.
        const vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo>
            sem_chain{
                vk::SemaphoreCreateInfo{},
                vk::SemaphoreTypeCreateInfo{
                    .semaphoreType = vk::SemaphoreType::eTimeline,
                    .initialValue = 0,
                },
            };
        auto [sem_res, sem] = device.createSemaphoreUnique(sem_chain.get());
        ASSERT_MSG(sem_res == vk::Result::eSuccess,
                   "AsyncResourcePrep: failed to create timeline semaphore: {}",
                   vk::to_string(sem_res));
        worker_timeline_ = std::move(sem);

        // Worker-private command pool. Vulkan requires per-pool external
        // sync - only the worker thread will ever touch this pool.
        const vk::CommandPoolCreateInfo pool_ci{
            .flags = vk::CommandPoolCreateFlagBits::eTransient |
                     vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = instance_.GetGraphicsQueueFamilyIndex(),
        };
        auto [pool_res, pool] = device.createCommandPoolUnique(pool_ci);
        ASSERT_MSG(pool_res == vk::Result::eSuccess,
                   "AsyncResourcePrep: failed to create command pool: {}",
                   vk::to_string(pool_res));
        cmd_pool_ = std::move(pool);
        SetObjectName(device, *cmd_pool_, "AsyncResourcePrep CmdPool");

        // Pre-allocate a small ring of cmdbufs, one per batch submission, sized so the worker
        // rarely blocks; the worker's own timeline says when one is safe to recycle.
        const vk::CommandBufferAllocateInfo alloc_ci{
            .commandPool = *cmd_pool_,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = kCmdBufRingSize,
        };
        cmd_buffers_.resize(kCmdBufRingSize);
        auto alloc_res = device.allocateCommandBuffers(&alloc_ci, cmd_buffers_.data());
        ASSERT_MSG(alloc_res == vk::Result::eSuccess,
                   "AsyncResourcePrep: failed to allocate cmdbufs: {}",
                   vk::to_string(alloc_res));
        cmd_buffer_ticks_.assign(kCmdBufRingSize, 0);
        for (u32 i = 0; i < kCmdBufRingSize; ++i) {
            SetObjectName(device, cmd_buffers_[i],
                          "AsyncResourcePrep CmdBuf {}", i);
        }

        worker_thread_ = std::jthread(std::bind_front(&AsyncResourcePrep::WorkerLoop, this));
    }

    ~AsyncResourcePrep() {
        if (worker_thread_.joinable()) {
            worker_thread_.request_stop();
            queue_cv_.notify_all();
            worker_thread_.join();
        }
        // Wait for in-flight worker submissions before destroying the cmd pool they reference.
        // Scheduler::Finish()/vkDeviceWaitIdle normally drain worker work, but the worker can
        // submit after those calls (Finish() may run before the parser stops feeding
        // tile_manager), so an explicit local wait closes that window.
        if constexpr (kEnabled) {
            const u64 final_tick = last_submitted_tick_.load(std::memory_order_acquire);
            if (final_tick > 0) {
                CpuWaitWorker(final_tick);
            }
        }
        // worker_timeline_ / cmd_pool_ / cmd_buffers_ destroyed by
        // vk::UniquePool's RAII. cmd_buffers_ are owned by cmd_pool_ and
        // freed when pool is destroyed.
    }

    AsyncResourcePrep(const AsyncResourcePrep&) = delete;
    AsyncResourcePrep& operator=(const AsyncResourcePrep&) = delete;

    /// Queue a dispatch.
    /// Returns true if the work was queued (async path).
    /// Returns false if the worker is disabled (kEnabled=false) - the
    /// caller must then record the synchronous inline equivalent.
    bool Queue(DispatchWork&& work) {
        if constexpr (!kEnabled) {
            return false;
        }
        {
            std::scoped_lock lk{queue_mutex_};
            queue_.push_back(std::move(work));
        }
        queue_cv_.notify_one();
        return true;
    }

    /// Block until: (a) the queue is empty, AND (b) the worker is not
    /// currently processing a batch. After return, last_submitted_tick_
    /// reflects the latest worker submission. Then: if there is anything
    /// the parser hasn't yet AddWaited on, append an AddWait to `info`
    /// using a SubmitInfo-like interface. The caller is the scheduler in
    /// SubmitExecution; this is the only place worker->parser ordering is
    /// established.
    ///
    /// Template on the SubmitInfo type to avoid an include cycle with
    /// vk_scheduler.h. The SubmitInfoT must have an `AddWait(vk::Semaphore,
    /// u64)` method matching ::Vulkan::SubmitInfo.
    template <typename SubmitInfoT>
    void DrainAndAddWaitIfNeeded(SubmitInfoT& info) {
        if constexpr (!kEnabled) {
            return;
        }
        // Block until queue empty + worker idle.
        {
            std::unique_lock lk{queue_mutex_};
            drain_cv_.wait(lk, [this] {
                return queue_.empty() && !worker_busy_;
            });
        }
        const u64 wt = last_submitted_tick_.load(std::memory_order_acquire);
        if (wt > last_waited_tick_) {
            // SubmitInfo has 3 wait slots that the caller's AddWaits may already fill; when
            // full, fall back to a CPU wait (rare - the presenter does not use all 3, see
            // vk_presenter.cpp).
            if (info.num_wait_semas < info.wait_semas.size()) {
                info.AddWait(*worker_timeline_, wt);
            } else {
                LOG_WARNING(Render_Vulkan,
                            "AsyncResourcePrep: SubmitInfo full, falling back to CPU wait");
                CpuWaitWorker(wt);
            }
            last_waited_tick_ = wt;
        }
    }

    /// CPU-block until worker has processed up to and including tick `target`.
    /// Used as a fallback when SubmitInfo's wait slots are full.
    void CpuWaitWorker(u64 target) {
        if constexpr (!kEnabled) {
            return;
        }
        if (target == 0) {
            return;
        }
        const vk::Semaphore sem = *worker_timeline_;
        const vk::SemaphoreWaitInfo wait_info{
            .semaphoreCount = 1,
            .pSemaphores = &sem,
            .pValues = &target,
        };
        while (instance_.GetDevice().waitSemaphores(
                   &wait_info, std::numeric_limits<u64>::max()) !=
               vk::Result::eSuccess) {
        }
    }

    /// Returns the worker's timeline semaphore. Exposed only for diagnostics.
    [[nodiscard]] vk::Semaphore TimelineSemaphore() const noexcept {
        return *worker_timeline_;
    }

    /// True if the worker is enabled and has at least one in-flight or
    /// queued work item. Used by tile_manager to decide whether the async
    /// path is going to be taken (so it can correctly tag DeferOperation
    /// lifetime).
    [[nodiscard]] static constexpr bool IsAsyncAvailable() noexcept {
        return kEnabled;
    }

private:
    static constexpr u32 kCmdBufRingSize = 4;

    void WorkerLoop(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:GpuAsyncRes");
        std::vector<DispatchWork> batch;
        batch.reserve(16);

        while (!stop.stop_requested()) {
            {
                std::unique_lock lk{queue_mutex_};
                queue_cv_.wait(lk, stop, [this] { return !queue_.empty(); });
                if (stop.stop_requested()) {
                    break;
                }
                batch.swap(queue_);
                worker_busy_ = true;
            }

            RecordAndSubmitBatch(batch);
            batch.clear();

            {
                std::scoped_lock lk{queue_mutex_};
                worker_busy_ = false;
            }
            drain_cv_.notify_all();
        }

        // Final drain on shutdown - process whatever's left so destruction
        // doesn't leak GPU work that was already accounted for in
        // DeferOperation tick tracking.
        {
            std::unique_lock lk{queue_mutex_};
            batch.swap(queue_);
        }
        if (!batch.empty()) {
            RecordAndSubmitBatch(batch);
        }
    }

    // Acquire a free cmdbuf from the worker ring. Blocks (CPU wait on the
    // worker timeline) if none are free. Caller is the worker thread only.
    vk::CommandBuffer AcquireCmdBuffer() {
        // Find a cmdbuf whose recorded tick has been signaled. tick=0
        // means "never submitted" (fresh) - always available.
        for (u32 attempt = 0; attempt < 2; ++attempt) {
            for (u32 i = 0; i < kCmdBufRingSize; ++i) {
                if (cmd_buffer_ticks_[i] == 0 ||
                    SemaphoreReached(cmd_buffer_ticks_[i])) {
                    cmd_ring_idx_ = i;
                    return cmd_buffers_[i];
                }
            }
            // None free - wait for the oldest one to finish.
            u64 oldest = std::numeric_limits<u64>::max();
            for (u32 i = 0; i < kCmdBufRingSize; ++i) {
                if (cmd_buffer_ticks_[i] != 0 &&
                    cmd_buffer_ticks_[i] < oldest) {
                    oldest = cmd_buffer_ticks_[i];
                }
            }
            if (oldest == std::numeric_limits<u64>::max()) {
                // All zero - first allocation is always available.
                cmd_ring_idx_ = 0;
                return cmd_buffers_[0];
            }
            CpuWaitWorker(oldest);
        }
        UNREACHABLE_MSG("AsyncResourcePrep: ring exhausted after wait");
    }

    bool SemaphoreReached(u64 tick) const {
        const auto [res, value] = instance_.GetDevice().getSemaphoreCounterValue(
            *worker_timeline_);
        if (res != vk::Result::eSuccess) {
            return false;
        }
        return value >= tick;
    }

    void RecordAndSubmitBatch(std::vector<DispatchWork>& batch) {
        if (batch.empty()) {
            return;
        }
        vk::CommandBuffer cmd = AcquireCmdBuffer();
        const vk::CommandBufferBeginInfo begin_info{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };
        Check(cmd.begin(begin_info));

        // Each work item: bindPipeline, fixed-up pushDescriptorSet, dispatch. Pipeline binds are
        // sticky within a cmdbuf but are issued per-item for simplicity - typically 1-3 items
        // per batch, and the dispatch dominates the cost.
        boost::container::small_vector<vk::BufferMemoryBarrier2, 16> barriers;
        for (auto& w : batch) {
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, w.pipeline);

            // pBufferInfo points at the work item's own buffer_infos array; the item lives in
            // the function-scope `batch` vector and the submit below is synchronous, so the
            // pointers stay valid.
            for (u32 i = 0; i < w.num_writes; ++i) {
                w.writes[i].pBufferInfo = &w.buffer_infos[i];
            }
            cmd.pushDescriptorSetKHR(
                vk::PipelineBindPoint::eCompute, w.layout, 0,
                vk::ArrayProxy<const vk::WriteDescriptorSet>{
                    w.num_writes, w.writes.data()});
            cmd.dispatch(w.dim_x, w.dim_y, w.dim_z);

            if (w.out_buffer && w.out_size > 0) {
                barriers.push_back(vk::BufferMemoryBarrier2{
                    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .dstAccessMask =
                        vk::AccessFlagBits2::eTransferRead |
                        vk::AccessFlagBits2::eShaderRead |
                        vk::AccessFlagBits2::eShaderStorageRead,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .buffer = w.out_buffer,
                    .offset = 0,
                    .size = w.out_size,
                });
            }
        }

        // Single barrier covering all output buffers in the batch.
        if (!barriers.empty()) {
            cmd.pipelineBarrier2(vk::DependencyInfo{
                .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                .bufferMemoryBarrierCount = static_cast<u32>(barriers.size()),
                .pBufferMemoryBarriers = barriers.data(),
            });
        }
        Check(cmd.end());

        // Submit under the shared graphics-queue mutex. The worker timeline
        // is INDEPENDENT of master_semaphore.
        const u64 worker_tick = worker_next_tick_.fetch_add(1, std::memory_order_acq_rel) + 1;
        const vk::Semaphore worker_sem = *worker_timeline_;
        const u64 signal_value = worker_tick;

        const vk::TimelineSemaphoreSubmitInfo timeline_si{
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &signal_value,
        };
        const vk::SubmitInfo submit_info{
            .pNext = &timeline_si,
            .waitSemaphoreCount = 0,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &worker_sem,
        };

        {
            std::scoped_lock lk{submit_mutex_};
            const auto res = instance_.GetGraphicsQueue().submit(submit_info);
            ASSERT_MSG(res == vk::Result::eSuccess,
                       "AsyncResourcePrep: vkQueueSubmit failed: {}",
                       vk::to_string(res));
        }

        cmd_buffer_ticks_[cmd_ring_idx_] = worker_tick;
        last_submitted_tick_.store(worker_tick, std::memory_order_release);
    }

    static void Check(vk::Result r) {
        ASSERT_MSG(r == vk::Result::eSuccess, "Vulkan call failed: {}",
                   vk::to_string(r));
    }

private:
    const Instance& instance_;
    MasterSemaphore& master_semaphore_;  // Reserved for future consumers; unused at present.
    std::mutex& submit_mutex_;

    // Worker timeline - independent of master_semaphore. Worker signals,
    // parser AddWaits this. Never advances master.
    vk::UniqueSemaphore worker_timeline_;
    std::atomic<u64> worker_next_tick_{0};
    std::atomic<u64> last_submitted_tick_{0};
    u64 last_waited_tick_{0};  // Parser-thread-only.

    // Worker-private command pool + ring of primary cmdbufs.
    vk::UniqueCommandPool cmd_pool_;
    std::vector<vk::CommandBuffer> cmd_buffers_;
    std::vector<u64> cmd_buffer_ticks_;  // tick assigned to each cmdbuf (0 = unused)
    u32 cmd_ring_idx_{0};

    // Work queue.
    mutable std::mutex queue_mutex_;
    std::condition_variable_any queue_cv_;
    std::condition_variable_any drain_cv_;
    std::vector<DispatchWork> queue_;
    bool worker_busy_{false};

    std::jthread worker_thread_;
};

} // namespace Vulkan
