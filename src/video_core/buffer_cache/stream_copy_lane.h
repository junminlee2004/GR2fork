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

    /// idle_ticks > 0 parks an idle worker in a timed MWAITX on the publish
    /// word for that many TSC ticks per round instead of the pause spin.
    /// Honoured only when the CPU reports MWAITX.
    void Init(u32 num_workers, bool hardened, u32 idle_ticks = 0);

    /// Safe mode: foreign-producer refusal, atomic barrier targets and the
    /// resolver push windows. Unsafe mode is byte-for-byte the original hot
    /// path and must only run for titles that never unmap mid-play.
    bool Hardened() const {
        return hardened_;
    }
    void Shutdown();

    bool Enabled() const {
        return num_workers_.load(std::memory_order_relaxed) != 0;
    }

    /// The lane never runs more workers than this; Init clamps to it, so the
    /// per-consumer cells below are a fixed-size array no caller can index past.
    static constexpr u32 kMaxWorkers = 4;

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
        u64 mwaits;                   // timed monitor waits entered by idle workers
        u64 mwait_wakes;              // of those, ended by a publish rather than the timer
        u64 worker_jobs[kMaxWorkers]; // copies each worker rank completed
        u64 helper_jobs;              // copies drained by a non-worker caller
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

    void WorkerLoop(u32 rank);
    /// Claims one job and copies it. The completion is the CALLER's to record:
    /// workers store into their own cell, everyone else bumps the helper cell.
    bool ClaimAndCopy();
    bool TryDrainShared();
    /// Total copies completed, as a lower bound: every cell is monotone, so a
    /// value read here was true at some point and can only have grown since.
    u64 Completed() const;

    // Written only by Init/Shutdown. Every worker loads slots_ on each drain
    // attempt and every Copy loads num_workers_, so a per-frame store to any
    // of these four would put the poll line back under the producer's writes.
    alignas(64) std::unique_ptr<Slot[]> slots_;
    std::vector<std::thread> threads_;
    std::atomic<u32> num_workers_{0};
    std::atomic<bool> stop_{false};
    bool hardened_{true};
    u32 idle_ticks_{0};

    // Producer-owned (GPU command thread); plain because single-writer. The
    // alignas(64) on published_ keeps the worker-touched atomics off this line.
    alignas(64) u64 enqueue_pos_{};
    u64 jobs_{};
    u64 bytes_{};
    u64 inline_unresolved_{};
    u64 inline_full_{};
    u64 barriers_{};
    u64 barrier_wait_ns_{};

    std::atomic<u64> producer_tid_{0};

    alignas(64) std::atomic<u64> published_{};
    alignas(64) std::atomic<u64> dequeue_pos_{};
    // Read by the producer on every Push.
    alignas(64) std::atomic<u32> sleepers_{};

    // One cell per consumer, last in the class so no offset above it moves.
    // A shared completion counter and a shared wait census cost 557 of the
    // machine's 906 cross-core transfers with four workers; each consumer now
    // owns its line and readers sum. 128 bytes, not 64: the adjacent-line
    // prefetcher pairs cell i's census line with cell i+1's completion line.
    static constexpr u32 kHelperCell = kMaxWorkers;
    static constexpr u32 kCells = kMaxWorkers + 1;
    struct alignas(128) Cell {
        std::atomic<u64> done{0};
        u64 pad0[7];
        std::atomic<u64> mwaits{0};
        std::atomic<u64> mwait_wakes{0};
        u64 pad1[6];
    };
    static_assert(sizeof(Cell) == 128 && alignof(Cell) == 128);
    Cell cells_[kCells]{};

    // Producer-held drain baselines: the cells are cumulative for the object's
    // life, like published_ and dequeue_pos_, so telemetry reports deltas
    // rather than resetting a word a worker is writing.
    u64 done_base_[kCells]{};
    u64 mwaits_base_{};
    u64 wakes_base_{};
    // Monotone lower bound on Completed(), so the common already-drained case
    // reads one line instead of five. Relaxed and racy by design: any value
    // ever stored was a true completion count, and a stale one only costs a
    // re-sum. DrainProducer is reachable from guest threads in both modes.
    std::atomic<u64> completed_seen_{};
};

} // namespace VideoCore
