// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// BundleAssembler: the PM4 parser thread (shadPS4:GpuComm) pushes DrawIntents onto a lock-free
// SPSC queue drained by a dedicated jthread (shadPS4:GpuAssembler), so Push returns right after
// the queue write. Monolithic mode (see monolithic_) processes each intent inline in Push.

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
    // Per-SKU queue depth, selected once at construction from boot-time title detection (see
    // ActiveQueueSize). inFAMOUS Second Son measures intent_queue_hwm=1822 (3.6M intents over
    // ~4 min), so 4096 gives ~2.2x headroom; the producer ASSERT names any future overflow.
    static constexpr u32 kQueueSizeDefault = 256;
    static constexpr u32 kQueueSizeGrr = 256;
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

    // Push: single-producer entry. Captures the reg snapshot if the intent type uses one,
    // publishes the slot, and returns the assigned packet_seq for WaitFor. Every call chain
    // originates on the PM4 thread (Presenter/signal paths route via SendCommand): SPSC holds.
    [[nodiscard]] u32 Push(DrawIntent intent);

    // PushAndProcess: non-blocking entry. Wraps Push() and discards the
    // seq. Kept for the many call sites that don't care about completion
    // (Draw / Dispatch / FillBuffer / CopyBuffer / ScopeMarker* / CpSync).
    void PushAndProcess(DrawIntent intent) {
        (void)Push(intent);
    }

    // WaitFor: block the producer until the assembler has processed packet_seq == seq. Used by
    // the blocking wrappers (Flush/Finish/OnSubmit, Presenter, BufferCache::ReadMemory) to
    // observe completion. Must not be called from the assembler thread itself (self-deadlock).
    void WaitFor(u32 seq);

    // Diagnostic - number of intents currently queued (head - tail).
    [[nodiscard]] u32 InFlightIntents() const noexcept {
        const u32 h = head_.load(std::memory_order_acquire);
        const u32 t = tail_.load(std::memory_order_acquire);
        return h - t;
    }

    // Diagnostic - true if no intents are queued.
    [[nodiscard]] bool Empty() const noexcept {
        return InFlightIntents() == 0;
    }

    // Sentinel used in `intent.snapshot_idx` for marker intents that don't
    // capture a snapshot.
    static constexpr u16 kNoSnapshot = 0xFFFFu;

private:
    // Assembler-thread main loop: pops intents FIFO through ProcessOne, parking on wake_counter_
    // when empty. A stop_callback bumps wake_counter_ at request_stop so the wait wakes and the
    // loop rechecks the stop token.
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
        // Scheduler-toucher markers don't read regs from a snapshot -
        // their payloads are self-contained, and the bodies operate on
        // scheduler primary cmdbuf / cache state directly.
        case DrawIntent::Type::Flush:
        case DrawIntent::Type::Finish:
        case DrawIntent::Type::CpSync:
        case DrawIntent::Type::OnSubmit:
        case DrawIntent::Type::FillBuffer:
        case DrawIntent::Type::CopyBuffer:
        case DrawIntent::Type::ScopeMarkerBegin:
        case DrawIntent::Type::ScopeMarkerEnd:
        case DrawIntent::Type::ScopedMarkerInsert:
        // PresenterRecord carries a closure, no snapshot.
        case DrawIntent::Type::PresenterRecord:
            return false;
        }
        return false;
    }

    Rasterizer& rasterizer_;
    AmdGpu::Liverpool* liverpool_;

    // GR2FORK: monolithic mode, latched once from Config::isLegacyMonolithicGpuComm() (forced ON
    // for 4-physical-core hosts running GRR - see emulator.cpp). When true no GpuAssembler
    // jthread is spawned and Push() records inline. Declared before queue_size_ for init order.
    const bool monolithic_;

    // Active queue depth + wrap mask, chosen once at construction (see ActiveQueueSize).
    // Declared before queue_ so the init list, which sizes queue_ from queue_size_, runs in
    // declaration order.
    const u32 queue_size_;
    const u32 queue_mask_;

    // SPSC ring storage. Heap-allocated (queue_size_ x 64 B/intent: 16 KiB default/GRR, 256 KiB
    // on ISS) to keep the BundleAssembler value type small and the slot array cache-line
    // aligned regardless of where the BundleAssembler itself sits.
    std::unique_ptr<DrawIntent[]> queue_;

    // Cache-line-isolated cursors. Producer writes head_, consumer writes
    // tail_; keeping them on different lines avoids false sharing.
    alignas(64) std::atomic<u32> head_{0};
    // GR2FORK PERF: producer-private stale view of tail_ for Push's capacity
    // check; shares the producer-owned head_ line by design. Only Push reads
    // or writes it.
    u32 tail_cache_{0};
    // GR2FORK PERF: monotonic sequence stamp for pushed intents. Producer-only;
    // lives here because on the waiters_ line its per-push increment traded a
    // cross-core miss with DrainLoop's per-intent waiters_ load.
    u32 next_packet_seq_{0};
    alignas(64) std::atomic<u32> tail_{0};

    // Producer-to-consumer wake counter: Push() and the stop_callback bump + notify; DrainLoop
    // parks on it when the queue is empty. Kept separate from head_ so the stop path never has
    // to fake-bump head_, which would break the slot/intent invariant under a concurrent Push.
    alignas(64) std::atomic<u32> wake_counter_{0};

    // GR2FORK PERF: producers parked (or about to park) in WaitFor; DrainLoop notifies tail_
    // only when nonzero (unconditional notify probes libstdc++'s global waiter bucket ~10.3 M
    // times/session). Waiter seq_cst RMW precedes the final recheck+wait. Kill: GR2_NOWAITFLAG=1.
    alignas(64) std::atomic<u32> waiters_{0};

    // The assembler thread, spawned at the end of the ctor body so every member it touches is
    // initialized first. Must stay the last member declared so it destructs first: its dtor
    // request_stops and joins before any other member is torn down.
    std::jthread thread_;
};

} // namespace Vulkan
