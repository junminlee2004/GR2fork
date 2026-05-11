// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/bundle_assembler.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace Vulkan {

BundleAssembler::BundleAssembler(Rasterizer& rasterizer,
                                 AmdGpu::Liverpool* liverpool) noexcept
    : rasterizer_{rasterizer}, liverpool_{liverpool},
      queue_{std::make_unique<DrawIntent[]>(kQueueSize)} {
    // Phase 1D-1 (Phase G): spawn the assembler thread AT THE END of the
    // ctor body so queue_/head_/tail_/wake_counter_/rasterizer_/liverpool_
    // are fully initialized before the thread first runs DrainLoop. The
    // thread parks on wake_counter_ immediately (queue is empty) and
    // wakes on the first producer Push or on stop request.
    LOG_INFO(Render_Vulkan,
             "[GR2FORK Y-1.ASYNC] BundleAssembler constructed: ASYNC dispatcher "
             "(queue={} slots, intent_size={}B, dedicated jthread, "
             "primary-CB direct recording — no recorder, no secondaries)",
             kQueueSize, sizeof(DrawIntent));
    thread_ = std::jthread([this](std::stop_token stop) noexcept {
        DrainLoop(stop);
    });
}

BundleAssembler::~BundleAssembler() {
    // Phase 1D-1 (Phase G): jthread member destructs first (declared last),
    // calling request_stop() and joining. The stop_callback registered in
    // DrainLoop bumps wake_counter_ + notifies so the assembler wakes,
    // sees stop_requested(), and exits the loop. By the time we reach
    // this body the thread has joined and no producer is still pushing
    // (PM4 / Presenter / signal-handler paths have all been shut down by
    // the higher-level lifecycle).
    //
    // Diagnostic: log if the queue has unprocessed intents at shutdown.
    // Under clean shutdown this is 0; non-zero means a producer pushed
    // after request_stop fired (would indicate a lifecycle bug).
    const u32 leftover = InFlightIntents();
    if (leftover != 0) {
        LOG_WARNING(Render_Vulkan,
                    "BundleAssembler dtor: {} intents in queue at shutdown",
                    leftover);
    }
}

u32 BundleAssembler::Push(DrawIntent intent) {
    // Stage 1: capture the reg snapshot and stamp the intent before publish.
    if (IntentUsesSnapshot(intent.type)) {
        intent.snapshot_idx = liverpool_->CaptureSnapshot();
        // Y-1 telemetry: snapshot pool HWM after capture.
        Rasterizer::BumpSnapshotPoolHwm(liverpool_->SnapshotPoolInFlight());
    } else {
        intent.snapshot_idx = kNoSnapshot;
    }
    const u32 my_seq = next_packet_seq_++;
    intent.packet_seq = my_seq;

    // Producer reservation. SPSC: head_ is single-writer (producer thread).
    // Capacity: assert that we don't exceed kQueueSize. Under Phase G this
    // assert can fire if the assembler falls badly behind the producer (e.g.
    // assembler core saturated or stalled). The 256-slot queue gives a
    // generous buffer; the snapshot pool's 16-slot capacity is the more
    // likely backpressure point and is handled by Liverpool::CaptureSnapshot
    // yielding (above).
    const u32 my_head = head_.load(std::memory_order_relaxed);
    const u32 t = tail_.load(std::memory_order_acquire);
    ASSERT_MSG(my_head - t < kQueueSize,
               "BundleAssembler intent queue full: head={} tail={} cap={}",
               my_head, t, kQueueSize);

    // Write slot before advancing head; the head_.store(release) below
    // pairs with the consumer's head_.load(acquire) so the slot writes
    // are visible to the assembler thread when it sees the new head.
    queue_[my_head & (kQueueSize - 1)] = intent;
    head_.store(my_head + 1, std::memory_order_release);

    // Y-1 telemetry: bump intent-queue HWM + total_intents.
    Rasterizer::BumpIntentQueueHwm((my_head + 1) - t);

    // Wake the assembler if it's parked on wake_counter_. The bump+notify
    // is the producer→consumer signal. Always notifying (rather than only
    // when the assembler is known-parked) keeps Push branch-free; notify
    // is cheap when no waiters.
    wake_counter_.fetch_add(1, std::memory_order_release);
    wake_counter_.notify_one();

    return my_seq;
}

void BundleAssembler::WaitFor(u32 seq) {
    // Block until tail_ has advanced past seq. The signed-difference
    // comparison handles u32 wraparound correctly: (t - seq) interpreted
    // as i32 is positive when t is "after" seq in the circular order, and
    // we use the recent-seq invariant (WaitFor is called within
    // microseconds of Push, so the seq is never more than a few hundred
    // pushes old).
    u32 t = tail_.load(std::memory_order_acquire);
    while (static_cast<s32>(t - seq) <= 0) {
        tail_.wait(t, std::memory_order_acquire);
        t = tail_.load(std::memory_order_acquire);
    }
}

void BundleAssembler::DrainLoop(std::stop_token stop) noexcept {
    // Y-2: name + pin the assembler thread to a dedicated physical core
    // (third reservation after GpuComm). On Z1 Extreme Linux this lands
    // on phys core 6 = {6, 14}; the GuestExcludedCoreMask stripped CPU 6
    // from every guest thread's mask before this pin took effect, so the
    // assembler owns its core exclusively. On hosts with < 7 enumerable
    // physical cores GetGpuAssemblerCoreMask returns 0 and the assembler
    // floats on the OS scheduler.
    Common::SetCurrentThreadName("shadPS4:GpuAssembler");
    if (const u64 mask = Common::GetGpuAssemblerCoreMask()) {
        Common::SetCurrentThreadAffinityMask(mask);
        LOG_INFO(Render_Vulkan,
                 "[GR2FORK Y-2] BundleAssembler thread pinned to mask 0x{:x}",
                 mask);
    } else {
        LOG_INFO(Render_Vulkan,
                 "[GR2FORK Y-1.ASYNC] BundleAssembler thread started (unpinned, "
                 "host has < 7 physical cores)");
    }

    // The stop_callback bumps wake_counter_ + notifies so the wait below
    // returns when request_stop is called by the jthread destructor. The
    // outer loop's stop_requested check then exits cleanly. We use
    // wake_counter_ (not head_) for the wakeup so the stop path doesn't
    // race with concurrent producer Push() that does load+store on head_.
    std::stop_callback wake{stop, [this] {
        wake_counter_.fetch_add(1, std::memory_order_release);
        wake_counter_.notify_all();
    }};

    while (!stop.stop_requested()) {
        u32 t = tail_.load(std::memory_order_relaxed);
        u32 h = head_.load(std::memory_order_acquire);
        if (t == h) {
            // Queue empty. Park on wake_counter_ until producer pushes
            // (or stop fires). The "load wake, recheck head, then wait"
            // pattern closes the window where producer pushes between
            // our two checks.
            const u32 w = wake_counter_.load(std::memory_order_acquire);
            h = head_.load(std::memory_order_acquire);
            if (t == h) {
                wake_counter_.wait(w, std::memory_order_acquire);
            }
            continue;
        }
        // Copy the intent out by value before advancing tail. Under
        // async the producer may rewrite the slot once tail moves past
        // it, so the consumer must not hold a reference into queue_
        // after the store.
        const DrawIntent intent = queue_[t & (kQueueSize - 1)];
        ProcessOne(intent);
        // Advance tail (release: pairs with WaitFor's tail_.load(acquire)
        // and with the producer's tail_.load(acquire) capacity check).
        tail_.store(t + 1, std::memory_order_release);
        // Wake any WaitFor'ers parked on tail_.
        tail_.notify_all();
    }
}

void BundleAssembler::ProcessOne(const DrawIntent& intent) {
    // Dispatch first; release the snapshot slot afterwards. The Rasterizer
    // entry points may read from `liverpool->regs` directly in 2A (they're
    // migrated to the snapshot in 2B); the slot's lifetime spans the work
    // either way so the FIFO Release contract on LiverpoolRegsSnapshotPool
    // is observed.
    switch (intent.type) {
    case DrawIntent::Type::Draw:
    case DrawIntent::Type::DrawIndexed:
        rasterizer_.DoDrawFromIntent(intent);
        break;
    case DrawIntent::Type::DrawIndirect:
    case DrawIntent::Type::DrawIndexedIndirect:
        rasterizer_.DoDrawIndirectFromIntent(intent);
        break;
    case DrawIntent::Type::Dispatch:
        rasterizer_.DoDispatchDirectFromIntent(intent);
        break;
    case DrawIntent::Type::DispatchIndirect:
        rasterizer_.DoDispatchIndirectFromIntent(intent);
        break;
    // Phase 1D-pre-D: Rasterizer scheduler-toucher markers. Producer
    // wrappers (Rasterizer::Flush / Finish / CpSync / OnSubmit / FillBuffer
    // / CopyBuffer / ScopeMarker*) build the intent and call
    // PushAndProcess, which under sync drains inline so the body runs on
    // the producer thread before the wrapper returns. Under future async
    // (Phase G) the wrappers add WaitFor(packet_seq) for the BLOCKING
    // ones (Flush/Finish/OnSubmit) so the producer observes the body has
    // completed before returning.
    case DrawIntent::Type::Flush:
        rasterizer_.DoFlushFromIntent(intent);
        break;
    case DrawIntent::Type::Finish:
        rasterizer_.DoFinishFromIntent(intent);
        break;
    case DrawIntent::Type::CpSync:
        rasterizer_.DoCpSyncFromIntent(intent);
        break;
    case DrawIntent::Type::OnSubmit:
        rasterizer_.DoOnSubmitFromIntent(intent);
        break;
    case DrawIntent::Type::FillBuffer:
        rasterizer_.DoFillBufferFromIntent(intent);
        break;
    case DrawIntent::Type::CopyBuffer:
        rasterizer_.DoCopyBufferFromIntent(intent);
        break;
    case DrawIntent::Type::ScopeMarkerBegin:
        rasterizer_.DoScopeMarkerBeginFromIntent(intent);
        break;
    case DrawIntent::Type::ScopeMarkerEnd:
        rasterizer_.DoScopeMarkerEndFromIntent(intent);
        break;
    case DrawIntent::Type::ScopedMarkerInsert:
        rasterizer_.DoScopedMarkerInsertFromIntent(intent);
        break;
    // Phase 1D-pre-E: Presenter scheduler-toucher closure. The assembler
    // invokes the closure and the closure self-destructs via the
    // invoke_and_destroy function pointer.
    case DrawIntent::Type::PresenterRecord:
        rasterizer_.DoPresenterRecordFromIntent(intent);
        break;
    case DrawIntent::Type::EndRendering:
    case DrawIntent::Type::DrainMarker:
    case DrawIntent::Type::EopWrite:
    case DrawIntent::Type::EosWait:
        // These four pre-Phase-D markers are still not pushed by Stage 1
        // — they enter the pipeline in a future turn when their
        // respective producer entry points get intent-queue routing. The
        // existing direct-dispatch paths for them still run alongside
        // the intent queue. Treat as a hard error if one slips through.
        UNREACHABLE_MSG("BundleAssembler: marker intent type {} not yet pushed",
                        static_cast<int>(intent.type));
        break;
    }

    if (intent.snapshot_idx != kNoSnapshot) {
        liverpool_->ReleaseSnapshot(intent.snapshot_idx);
    }
}

} // namespace Vulkan
