// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <thread>
#include <immintrin.h> // _mm_prefetch (one-iteration-ahead snapshot warming)

#include "common/assert.h"
#include "common/config.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/bundle_assembler.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace Vulkan {

u32 BundleAssembler::ActiveQueueSize() noexcept {
    // inFAMOUS Second Son bursts more intents per frame than the 256-slot default holds; GRR
    // runs a shallower queue. The single source of truth is the boot-time SKU detection in
    // Config.
    if (Config::isInfamousSecondSon()) {
        return kQueueSizeIss;
    }
    return Config::isGravityRushRemastered() ? kQueueSizeGrr : kQueueSizeDefault;
}

BundleAssembler::BundleAssembler(Rasterizer& rasterizer,
                                 AmdGpu::Liverpool* liverpool) noexcept
    : rasterizer_{rasterizer}, liverpool_{liverpool},
      monolithic_{Config::isLegacyMonolithicGpuComm()},
      queue_size_{ActiveQueueSize()}, queue_mask_{queue_size_ - 1},
      queue_{std::make_unique<DrawIntent[]>(queue_size_)} {
    if (monolithic_) {
        // GR2FORK monolithic: single-thread GpuComm path (see Config::isLegacyMonolithicGpuComm).
        // No GpuAssembler jthread is spawned - Push() drains each intent inline on the GpuComm
        // thread, freeing the assembler's core; the dtor's implicit jthread join is a no-op.
        LOG_INFO(Render_Vulkan,
                 "[GR2FORK MONOLITHIC] BundleAssembler constructed: SYNCHRONOUS "
                 "monolithic dispatcher (queue={} slots [{}], intent_size={}B, "
                 "NO GpuAssembler jthread — GpuComm records inline; assembler "
                 "core returned to guest)",
                 queue_size_,
                 Config::isInfamousSecondSon() ? "Infamous-Second-Son"
                     : (Config::isGravityRushRemastered() ? "GR-Remastered" : "default"),
                 sizeof(DrawIntent));
        return;
    }
    // Spawn the assembler thread at the end of the ctor body so every member DrainLoop touches
    // is fully initialized; the thread parks on wake_counter_ until the first Push or a stop
    // request.
    LOG_INFO(Render_Vulkan,
             "[GR2FORK Y-1.ASYNC] BundleAssembler constructed: ASYNC dispatcher "
             "(queue={} slots [{}], intent_size={}B, dedicated jthread, "
             "primary-CB direct recording — no recorder, no secondaries)",
             queue_size_,
             Config::isInfamousSecondSon() ? "Infamous-Second-Son"
                 : (Config::isGravityRushRemastered() ? "GR-Remastered" : "default"),
             sizeof(DrawIntent));
    thread_ = std::jthread([this](std::stop_token stop) noexcept {
        DrainLoop(stop);
    });
}

BundleAssembler::~BundleAssembler() {
    // The jthread member (declared last) destructs first: its request_stop wakes DrainLoop via
    // the stop_callback and joins, and no producer is still pushing by the time this body runs.
    // A non-zero leftover count means a producer pushed after request_stop (a lifecycle bug).
    const u32 leftover = InFlightIntents();
    if (leftover != 0) {
        LOG_WARNING(Render_Vulkan,
                    "BundleAssembler dtor: {} intents in queue at shutdown",
                    leftover);
    }
}

u32 BundleAssembler::Push(DrawIntent intent) {
    // Capture the reg snapshot and stamp the intent before publish.
    if (IntentUsesSnapshot(intent.type)) {
        // GR2FORK PERF: Draw-class intents never read the snapshot's cs_program (consumer audit
        // in regs_snapshot.h), so skip its 320 B capture for them. Kill switch: GR2_NOCSSKIP=1
        // restores the unconditional capture (env read once per process).
        static const bool cs_skip_enabled = []() noexcept {
            const char* e = std::getenv("GR2_NOCSSKIP");
            return !(e && e[0] == '1');
        }();
        const bool capture_cs = !cs_skip_enabled ||
                                intent.type == DrawIntent::Type::Dispatch ||
                                intent.type == DrawIntent::Type::DispatchIndirect;
        intent.snapshot_idx = liverpool_->CaptureSnapshot(capture_cs);
        // No snapshot-pool telemetry here: evaluating SnapshotPoolInFlight() costs two atomic
        // loads, one of a consumer-written line - ~1 cross-core miss per snapshot-bearing push.
    } else {
        intent.snapshot_idx = kNoSnapshot;
    }
    const u32 my_seq = next_packet_seq_++;
    intent.packet_seq = my_seq;

    if (monolithic_) {
        // GR2FORK monolithic: ProcessOne runs inline and releases the snapshot it just captured,
        // so at most one snapshot is ever in flight. head_/tail_ advance in lockstep to seq+1 so
        // WaitFor(seq) is satisfied at once; all single-threaded here, so relaxed ordering is ok.
        ProcessOne(intent);
        const u32 done = my_seq + 1;
        head_.store(done, std::memory_order_relaxed);
        tail_.store(done, std::memory_order_relaxed);
        return my_seq;
    }

    // Producer reservation (SPSC: head_ is single-writer). The capacity assert fires only if the
    // assembler falls badly behind; the snapshot pool (64 slots default / 16 GRR) is the likelier
    // backpressure point and is handled by Liverpool::CaptureSnapshot yielding.
    const u32 my_head = head_.load(std::memory_order_relaxed);
    // GR2FORK PERF: a per-push acquire load of the consumer-written tail_
    // makes that line ping-pong between the two hot cores. Check a
    // producer-private cached tail; refresh it only when the queue looks
    // full - the snapshot pool backpressures long before this queue fills.
    // GR2FORK PERF: the assert lives inside the refresh branch so the fast
    // path is a single compare - the SHAD_NO_INLINE assert lambda call
    // otherwise sits in the per-push instruction stream.
    if (my_head - tail_cache_ >= queue_size_) [[unlikely]] {
        tail_cache_ = tail_.load(std::memory_order_acquire);
        ASSERT_MSG(my_head - tail_cache_ < queue_size_,
                   "BundleAssembler intent queue full: head={} tail={} cap={}",
                   my_head, tail_cache_, queue_size_);
    }

    // Write slot before advancing head; the head_.store(release) below
    // pairs with the consumer's head_.load(acquire) so the slot writes
    // are visible to the assembler thread when it sees the new head.
    queue_[my_head & queue_mask_] = intent;
    head_.store(my_head + 1, std::memory_order_release);

    // GR2FORK PERF: warm the next slot for ownership - the consumer read it
    // last, so the next push's slot copy pays a cross-core RFO otherwise.
    // Guarded by the producer-private tail cache so a slot the consumer may
    // still read is never prefetched. Pure hint.
    if (my_head + 1 - tail_cache_ < queue_size_) [[likely]] {
        __builtin_prefetch(&queue_[(my_head + 1) & queue_mask_], 1, 1);
    }

    // Wake the assembler if it is parked on wake_counter_. Always notifying keeps Push
    // branch-free; notify is cheap when there are no waiters.
    wake_counter_.fetch_add(1, std::memory_order_release);
    wake_counter_.notify_one();

    return my_seq;
}

// GR2FORK PERF: waiter-flag gate, latched once. See waiters_ doc in
// bundle_assembler.h. GR2_NOWAITFLAG=1 restores unconditional notify.
static bool WaitFlagEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOWAITFLAG");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

// GR2FORK PERF: bounded spin before the DrainLoop futex park, latched
// once. Enabled only when the assembler owns a dedicated pinned core (mask
// != 0), where burning the wait is free. GR2_NOASMSPIN=1 parks immediately.
static bool AsmSpinEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOASMSPIN");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

void BundleAssembler::WaitFor(u32 seq) {
    // Block until tail_ has advanced past seq. The signed-difference comparison handles u32
    // wraparound because WaitFor runs within microseconds of Push, so seq is never more than a
    // few hundred pushes old.
    u32 t = tail_.load(std::memory_order_acquire);
    if (static_cast<s32>(t - seq) > 0) [[likely]] {
        // Already complete (always true in monolithic mode, and common under
        // async when the assembler is ahead) - no waiter registration.
        return;
    }
    if (WaitFlagEnabled()) [[likely]] {
        // Register BEFORE the recheck+wait so the consumer's
        // store-tail -> load-waiters sequence cannot miss us (see waiters_
        // doc block for the ordering argument).
        waiters_.fetch_add(1, std::memory_order_seq_cst);
        t = tail_.load(std::memory_order_acquire);
        while (static_cast<s32>(t - seq) <= 0) {
            tail_.wait(t, std::memory_order_acquire);
            t = tail_.load(std::memory_order_acquire);
        }
        waiters_.fetch_sub(1, std::memory_order_release);
        return;
    }
    while (static_cast<s32>(t - seq) <= 0) {
        tail_.wait(t, std::memory_order_acquire);
        t = tail_.load(std::memory_order_acquire);
    }
}

void BundleAssembler::DrainLoop(std::stop_token stop) noexcept {
    // Pin the assembler to a dedicated physical core (GuestExcludedCoreMask strips it from every
    // guest thread's mask, so ownership is exclusive). On hosts with < 7 enumerable physical
    // cores GetGpuAssemblerCoreMask returns 0 and the thread floats on the OS scheduler.
    Common::SetCurrentThreadName("shadPS4:GpuAssembler");
    const u64 assembler_mask = Common::GetGpuAssemblerCoreMask();
    if (assembler_mask) {
        Common::SetCurrentThreadAffinityMask(assembler_mask);
        LOG_INFO(Render_Vulkan,
                 "[GR2FORK Y-2] BundleAssembler thread pinned to mask 0x{:x}",
                 assembler_mask);
    } else {
        LOG_INFO(Render_Vulkan,
                 "[GR2FORK Y-1.ASYNC] BundleAssembler thread started (unpinned, "
                 "host has < 7 physical cores)");
    }

    // GR2FORK PERF: spin-before-park (~40 rounds x 64 _mm_pause = tens of us of poll) avoids a
    // futex sleep+wake on every queue empty-to-nonempty transition. Only when pinned - the core
    // is exclusively owned, so no guest work is displaced. Kill switch: GR2_NOASMSPIN=1.
    const bool spin_before_park = assembler_mask != 0 && AsmSpinEnabled();
    constexpr u32 kSpinRounds = 40;
    constexpr u32 kPausesPerRound = 64;

    // The stop_callback bumps wake_counter_ + notifies so the parked wait wakes on request_stop.
    // wake_counter_ (not head_) is used so the stop path cannot race a concurrent producer
    // Push() doing load+store on head_.
    std::stop_callback wake{stop, [this] {
        wake_counter_.fetch_add(1, std::memory_order_release);
        wake_counter_.notify_all();
    }};

    // GR2FORK PERF: one-ahead snapshot prefetch. GpuComm writes the snapshot, so the assembler's
    // first touch of each line is a cross-core miss (~40 cyc on Zen 4); warm the next published
    // intent's snapshot into L2 (HINT_T1) early. Kill switch: GR2FORK_ASSEMBLER_PREFETCH=0.
    static const bool prefetch_enabled = []() noexcept {
        const char* e = std::getenv("GR2FORK_ASSEMBLER_PREFETCH");
        return !e || e[0] != '0';
    }();
    // GR2FORK PERF: warm the full struct, not just the head - the dominant ctx-clean draw reads
    // the middle and tail first (stamp/dirty/extents, color_buffers, viewports, index block).
    // 39 prefetcht1 (~20 cycles of issue); excess hints are dropped, so it degrades gracefully.
    constexpr u32 kSnapshotPrefetchBytes = sizeof(AmdGpu::LiverpoolRegsSnapshot);

    const bool waitflag_enabled = WaitFlagEnabled();

    while (!stop.stop_requested()) {
        u32 t = tail_.load(std::memory_order_relaxed);
        u32 h = head_.load(std::memory_order_acquire);
        if (t == h) {
            // Poll head_ for a bounded window before conceding to
            // the futex (dedicated-core hosts only - see doc above).
            if (spin_before_park) {
                for (u32 round = 0; round < kSpinRounds && t == h; ++round) {
                    for (u32 i = 0; i < kPausesPerRound; ++i) {
                        _mm_pause();
                    }
                    h = head_.load(std::memory_order_acquire);
                }
                if (t != h) {
                    continue;
                }
            }
            // Queue empty: park on wake_counter_. The "load wake, recheck head, then wait"
            // pattern closes the window where the producer pushes between the two checks.
            const u32 w = wake_counter_.load(std::memory_order_acquire);
            h = head_.load(std::memory_order_acquire);
            if (t == h) {
                wake_counter_.wait(w, std::memory_order_acquire);
            }
            continue;
        }
        // GR2FORK PERF: batch-drain [t, h) - a per-intent acquire load of the producer-written
        // head_ is a near-guaranteed cross-core transfer, so head_ is reloaded once per batch
        // (observed queue hwm ~195). Intents pushed mid-batch wait for the next outer iteration.
        do {
            // Warm the NEXT intent's snapshot (if one is already published)
            // into L2 while we process the current intent below.
            if (prefetch_enabled && (t + 1) != h) [[likely]] {
                const DrawIntent& next = queue_[(t + 1) & queue_mask_];
                if (next.snapshot_idx != kNoSnapshot) {
                    const auto& snap = liverpool_->GetSnapshot(next.snapshot_idx);
                    const char* p = reinterpret_cast<const char*>(&snap);
                    for (u32 off = 0; off < kSnapshotPrefetchBytes; off += 64) {
                        _mm_prefetch(p + off, _MM_HINT_T1);
                    }
                }
            }
            // Copy the intent out by value before advancing tail - the producer may rewrite the
            // slot once tail moves past it.
            const DrawIntent intent = queue_[t & queue_mask_];
            ProcessOne(intent);
            // Advance tail (release: pairs with WaitFor's tail_.load(acquire)
            // and with the producer's tail_.load(acquire) capacity check).
            tail_.store(t + 1, std::memory_order_release);
            // Wake WaitFor'ers only when one is registered - an unconditional per-intent notify
            // probes libstdc++'s global waiter-pool bucket (~10.3 M times per run) for waiters
            // that exist only around per-submit round-trips. Ordering: waiters_ doc block.
            if (waitflag_enabled) [[likely]] {
                if (waiters_.load(std::memory_order_seq_cst) != 0) [[unlikely]] {
                    tail_.notify_all();
                }
            } else {
                tail_.notify_all();
            }
            ++t;
        } while (t != h && !stop.stop_requested());
    }
}

void BundleAssembler::ProcessOne(const DrawIntent& intent) {
    // Dispatch first, release the snapshot slot afterwards: the slot's lifetime spans the work
    // so the FIFO Release contract on LiverpoolRegsSnapshotPool is observed.
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
    // Rasterizer scheduler-toucher markers. Producer wrappers build the intent and call
    // PushAndProcess; the blocking ones (Flush/Finish/OnSubmit) add WaitFor(packet_seq) under
    // async so the producer observes completion before returning.
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
    // Presenter scheduler-toucher closure. The assembler
    // invokes the closure and the closure self-destructs via the
    // invoke_and_destroy function pointer.
    case DrawIntent::Type::PresenterRecord:
        rasterizer_.DoPresenterRecordFromIntent(intent);
        break;
    case DrawIntent::Type::EndRendering:
    case DrawIntent::Type::DrainMarker:
    case DrawIntent::Type::EopWrite:
    case DrawIntent::Type::EosWait:
        // These four marker types have no intent-queue routing (their direct-dispatch paths run
        // alongside the queue); one slipping through is a hard error.
        UNREACHABLE_MSG("BundleAssembler: marker intent type {} not yet pushed",
                        static_cast<int>(intent.type));
        break;
    }

    if (intent.snapshot_idx != kNoSnapshot) {
        liverpool_->ReleaseSnapshot(intent.snapshot_idx);
    }
}

} // namespace Vulkan
