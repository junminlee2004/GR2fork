// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include "common/types.h"

namespace VideoCore {

// Off-thread drain lane for guest -> stream-ring uploads.
//
// The GPU command thread allocates ring offsets in order (Map/Commit stay
// untouched) and queues {backing src, ring dst, size} jobs; worker threads
// move the bytes. Sources are read through the physical backing view, which
// is never mprotected, so a worker can never take a guest fault and never
// re-enters emulator machinery. Ordering contract: every queued byte is in
// place before the command buffer that reads it is submitted, enforced by
// DrainProducer() at SubmitExecution and MapWrap, and by DrainRemote() ahead
// of guest unmaps (the backing pointers must outlive the job).
//
// Single producer: only the GPU command thread may call Push/DrainProducer.
// The queue is a bounded MPMC ring with per-slot sequence numbers; a push is
// a handful of stores on producer-owned lines. Workers spin briefly between
// bursts and futex-sleep through idle stretches, so the wake syscall never
// lands on the producer during a frame.
class StreamCopyLane {
public:
    static StreamCopyLane& Instance();

    void Init(u32 num_workers);
    void Shutdown();

    bool Enabled() const {
        return num_workers_.load(std::memory_order_relaxed) != 0;
    }

    /// GPU command thread only. False = ring full; the caller copies inline.
    bool Push(const u8* src, u8* dst, u32 size);

    /// GPU command thread: wait until every pushed job has landed.
    void DrainProducer();

    /// Any other thread (guest unmap path): wait for every published job.
    void DrainRemote();

    void NoteInlineUnresolved() {
        ++inline_unresolved_;
    }

    struct Stats {
        u64 jobs;
        u64 bytes;
        u64 inline_unresolved;
        u64 inline_full;
        u64 barriers;
        u64 barrier_wait_ns;
    };
    /// GPU command thread: returns and resets the counters.
    Stats DrainStats();

private:
    struct Job {
        const u8* src;
        u8* dst;
        u32 size;
    };
    struct alignas(64) Slot {
        std::atomic<u64> seq;
        Job job;
    };
    static constexpr size_t kRingSlots = 8192; // power of two

    void WorkerLoop();
    bool TryDrainOne();

    // Written only by Init/Shutdown. Every worker loads slots_ on each drain
    // attempt and every Copy loads num_workers_, so a per-frame store to any
    // of these four would put the poll line back under the producer's writes.
    alignas(64) std::unique_ptr<Slot[]> slots_;
    std::vector<std::thread> threads_;
    std::atomic<u32> num_workers_{0};
    std::atomic<bool> stop_{false};

    // Producer-owned (GPU command thread); plain because single-writer. These
    // seven fill one line exactly; an eighth would re-share published_'s.
    alignas(64) u64 enqueue_pos_{};
    u64 jobs_{};
    u64 bytes_{};
    u64 inline_unresolved_{};
    u64 inline_full_{};
    u64 barriers_{};
    u64 barrier_wait_ns_{};

    alignas(64) std::atomic<u64> published_{};
    alignas(64) std::atomic<u64> dequeue_pos_{};
    alignas(64) std::atomic<u64> copies_done_{};
    alignas(64) std::atomic<u32> sleepers_{};
};

} // namespace VideoCore
