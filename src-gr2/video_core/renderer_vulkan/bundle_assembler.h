// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// BundleAssembler — Stage 2/3 of the GpuComm 4-stage split (HANDOFF §3, §8, §9).
//
// Phase 1D-0 (Turn 2A — intent-infra): synchronous dispatcher. The producer
// (PM4 parser thread) builds a DrawIntent, calls PushAndProcess, and the
// assembler immediately drains the queue on the same thread. Behavior is
// bit-identical to the pre-Turn 2A build — the intent shape, snapshot
// capture/release lifecycle, and SPSC queue mechanics are exercised end-to-
// end, but no parallelism is introduced.
//
// Phase 1D-1 (Turn 2B): the body of this class becomes asynchronous. A
// dedicated jthread (`shadPS4:GpuAssembler`) runs Drain() in a loop, and
// PushAndProcess returns immediately after the queue write. Drain logic is
// already factored to support that without changes to call sites.
//
// Threading model:
//   * 2A: producer + consumer = same thread. Atomics are single-threaded
//     loads/stores; the choice is for forward compatibility.
//   * 2B: producer = PM4 parser thread (`shadPS4:GpuComm`); consumer =
//     `shadPS4:GpuAssembler` jthread. SPSC, lock-free.

#pragma once

#include <atomic>
#include <memory>
#include <stop_token>
#include <thread>

#include "common/types.h"
#include "video_core/renderer_vulkan/draw_intent.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Vulkan {

class Rasterizer;

class BundleAssembler {
public:
    // HANDOFF §9: PM4-parser → assembler queue depth. The first queue is
    // deepest because PM4 packets can burst (CE writes hundreds of regs between
    // draws). The same binary runs Gravity Rush 2 (and every other title) as
    // well as Gravity Rush Remastered and inFAMOUS Second Son; GRR uses a
    // shallower 32-slot queue, inFAMOUS Second Son a much deeper one, and every
    // other title uses 256. The active depth is selected once at construction
    // from the boot-time title detection (Config::isInfamousSecondSon() /
    // Config::isGravityRushRemastered()); see ActiveQueueSize().
    //
    // SIZED FROM MEASUREMENT: a 65536-slot measurement run reported a true peak
    // of intent_queue_hwm=1822 for ISS (total_intents=3.6M over ~4 min). 256/512/
    // 1024 all overflowed because real demand is ~1822; 65536 held but is wasteful.
    // 4096 is the next power of two that clears the peak with ~2.2x headroom
    // (4096 * 48B = 192 KB, one up-front alloc, sequential access — negligible).
    // If a heavier scene ever overflows 4096, the producer ASSERT will name the
    // new cap; re-run oversized to re-measure and bump to the next power of two.
    static constexpr u32 kQueueSizeDefault = 256;
    static constexpr u32 kQueueSizeGrr = 32;
    static constexpr u32 kQueueSizeIss = 4096;
    static_assert((kQueueSizeDefault & (kQueueSizeDefault - 1)) == 0,
                  "kQueueSizeDefault must be a power of two");
    static_assert((kQueueSizeGrr & (kQueueSizeGrr - 1)) == 0,
                  "kQueueSizeGrr must be a power of two");
    static_assert((kQueueSizeIss & (kQueueSizeIss - 1)) == 0,
                  "kQueueSizeIss must be a power of two");

    // Active queue depth for this session: kQueueSizeGrr on Remastered, else
    // kQueueSizeDefault. Static so telemetry (Rasterizer::LogHwm) can report the
    // active capacity without holding an assembler instance.
    [[nodiscard]] static u32 ActiveQueueSize() noexcept;

    BundleAssembler(Rasterizer& rasterizer, AmdGpu::Liverpool* liverpool) noexcept;
    ~BundleAssembler();

    BundleAssembler(const BundleAssembler&) = delete;
    BundleAssembler& operator=(const BundleAssembler&) = delete;
    BundleAssembler(BundleAssembler&&) = delete;
    BundleAssembler& operator=(BundleAssembler&&) = delete;

    // === Phase 1D-1 (Phase G) — async Push / WaitFor API. ===
    //
    // Push(intent): single-producer entry. Captures the reg snapshot if the
    // intent type uses one, stamps `intent.packet_seq`, writes the slot,
    // advances head_ (release), and notifies the assembler thread via
    // wake_counter_. Returns the assigned packet_seq for use with WaitFor.
    // Does NOT drain — the assembler thread runs Drain in its own loop.
    //
    // The producer must be a single thread per call (the SPSC head_ load+
    // store sequence is non-atomic-as-a-whole). In practice every Push call
    // chain originates on the PM4 thread under future async (Phase G); the
    // Presenter and signal-handler paths reach Push via SendCommand which
    // also dispatches to PM4. The bundle assembler queue is single-producer
    // by construction.
    [[nodiscard]] u32 Push(DrawIntent intent);

    // PushAndProcess: legacy non-blocking entry. Wraps Push() and discards
    // the seq. Kept for the many call sites that don't care about
    // completion (Draw / Dispatch / FillBuffer / CopyBuffer / ScopeMarker*
    // / CpSync). Replaces the synchronous-drain Pre-G semantics.
    void PushAndProcess(DrawIntent intent) {
        (void)Push(intent);
    }

    // WaitFor: block the producer until the assembler has processed the
    // intent with packet_seq == seq (i.e. tail_ has advanced past seq).
    // Used by the BLOCKING wrappers — Flush, Finish, OnSubmit, the three
    // Presenter sites, and the BufferCache::ReadMemory SendCommand outer
    // lambda — to observe completion before returning to their caller.
    //
    // Implementation uses std::atomic::wait on tail_; the consumer's
    // tail_.notify_all() in DrainLoop wakes the waiter. Robust to u32
    // wraparound (signed-difference comparison via s32) for the common
    // case where seq is recent (microseconds-old).
    //
    // Must NOT be called from the assembler thread itself (would deadlock
    // waiting for itself). All current callers run on the producer side.
    void WaitFor(u32 seq);

    // Diagnostic — number of intents currently queued (head - tail).
    [[nodiscard]] u32 InFlightIntents() const noexcept {
        const u32 h = head_.load(std::memory_order_acquire);
        const u32 t = tail_.load(std::memory_order_acquire);
        return h - t;
    }

    // Diagnostic — true if no intents are queued.
    [[nodiscard]] bool Empty() const noexcept {
        return InFlightIntents() == 0;
    }

    // Sentinel used in `intent.snapshot_idx` for marker intents that don't
    // capture a snapshot.
    static constexpr u16 kNoSnapshot = 0xFFFFu;

private:
    // Assembler-thread main loop. Pops intents in FIFO order and dispatches
    // each via ProcessOne. Parks on wake_counter_ when the queue is empty;
    // wakes on producer notify or stop_token request. The stop_callback in
    // the ctor body bumps wake_counter_ at request_stop time so the wait
    // observes a value change and returns to recheck the stop token.
    void DrainLoop(std::stop_token stop) noexcept;
    // Dispatch a single intent to the matching Rasterizer entry point.
    // Reads the snapshot from the pool (if applicable) and releases the
    // slot when done.
    void ProcessOne(const DrawIntent& intent);

    // True if the intent type captures and uses a reg snapshot.
    [[nodiscard]] static constexpr bool IntentUsesSnapshot(DrawIntent::Type t) noexcept {
        switch (t) {
        case DrawIntent::Type::Draw:
        case DrawIntent::Type::DrawIndexed:
        case DrawIntent::Type::DrawIndirect:
        case DrawIntent::Type::DrawIndexedIndirect:
        case DrawIntent::Type::Dispatch:
        case DrawIntent::Type::DispatchIndirect:
            return true;
        case DrawIntent::Type::EndRendering:
        case DrawIntent::Type::DrainMarker:
        case DrawIntent::Type::EopWrite:
        case DrawIntent::Type::EosWait:
        // Phase 1D-pre-D: scheduler-toucher markers don't read regs from
        // a snapshot — their payloads are self-contained, and the bodies
        // operate on scheduler primary cmdbuf / cache state directly.
        case DrawIntent::Type::Flush:
        case DrawIntent::Type::Finish:
        case DrawIntent::Type::CpSync:
        case DrawIntent::Type::OnSubmit:
        case DrawIntent::Type::FillBuffer:
        case DrawIntent::Type::CopyBuffer:
        case DrawIntent::Type::ScopeMarkerBegin:
        case DrawIntent::Type::ScopeMarkerEnd:
        case DrawIntent::Type::ScopedMarkerInsert:
        // Phase 1D-pre-E: PresenterRecord carries a closure, no snapshot.
        case DrawIntent::Type::PresenterRecord:
            return false;
        }
        return false;
    }

    Rasterizer& rasterizer_;
    AmdGpu::Liverpool* liverpool_;

    // GR2FORK monolithic: latched once at construction from
    // Config::isLegacyMonolithicGpuComm() (forced ON for 4-physical-core hosts
    // running Gravity Rush Remastered — see emulator.cpp). When true this
    // assembler runs as the pre-async (Phase 1D-0) SYNCHRONOUS dispatcher: the
    // GpuAssembler jthread is never spawned, and Push() records each intent
    // inline on the producer (GpuComm) thread before returning. const so the
    // dispatch mode is fixed for the assembler's lifetime. Declared before
    // queue_size_ so the init list runs in declaration order.
    const bool monolithic_;

    // Active queue depth + wrap mask, chosen once at construction (see
    // ActiveQueueSize). const so the SPSC indexing math is fixed for the
    // assembler's lifetime. Declared before queue_ so the init list (which
    // sizes queue_ from queue_size_) runs in declaration order.
    const u32 queue_size_;
    const u32 queue_mask_;

    // SPSC ring storage. Heap-allocated (queue_size_ × 64 B/intent: 16 KiB at
    // the 256-slot default, 2 KiB on GRR's 32-slot queue) to keep the
    // BundleAssembler value type small and to ensure cache-line alignment of
    // the slot array regardless of where the BundleAssembler itself sits in
    // memory.
    std::unique_ptr<DrawIntent[]> queue_;

    // Cache-line-isolated cursors. Producer writes head_, consumer writes
    // tail_; bouncing them on different lines avoids false sharing in 2B.
    alignas(64) std::atomic<u32> head_{0};
    alignas(64) std::atomic<u32> tail_{0};

    // Phase 1D-1 (Phase G): wake counter for producer→consumer notification.
    // Producer Push() bumps + notifies after each enqueue. The DrainLoop
    // parks on this when the queue is empty (recheck pattern: load head,
    // load wake, recheck head, then wait if still empty). The stop_callback
    // also bumps + notifies so the wait wakes and the loop checks the stop
    // token.
    //
    // Decoupling the wake variable from head_ avoids a race where the
    // stop_callback would otherwise need to fake-bump head_ — which would
    // mismatch the slot/intent invariant under a concurrent producer Push.
    alignas(64) std::atomic<u32> wake_counter_{0};

    // Monotonic sequence stamp written into every pushed intent.
    // Producer-only state, no synchronization needed.
    u32 next_packet_seq_{0};

    // Phase 1D-1 (Phase G): the assembler thread. Spawned at the END of
    // the BundleAssembler ctor body so all members it touches (queue_,
    // head_, tail_, wake_counter_, rasterizer_, liverpool_) are fully
    // initialized before the thread starts running. Must be the LAST
    // member declared so it is the FIRST to destruct — its dtor calls
    // request_stop() and joins the thread before any other member is
    // torn down.
    std::jthread thread_;
};

} // namespace Vulkan
