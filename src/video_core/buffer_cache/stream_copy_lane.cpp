// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstring>
#include "common/arch.h"
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(ARCH_X86_64)
#include <emmintrin.h>
#endif
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#define SHAD_COPY_LANE_MWAITX 1
#endif
#include "common/thread.h"
#include "video_core/buffer_cache/stream_copy_lane.h"

namespace VideoCore {

namespace {

// The spin hint each target actually has, in the ladder common/spin_lock.cpp
// already ships. The lane spins on three paths and MWAITX covers none of them
// off x86.
void CpuPause() {
#if defined(ARCH_X86_64)
    _mm_pause();
#elif defined(ARCH_ARM64) && defined(_MSC_VER)
    __yield();
#elif defined(ARCH_ARM64)
    asm("yield");
#endif
}

bool CpuHasMwaitx() {
#ifdef SHAD_COPY_LANE_MWAITX
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(0x80000001u, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }
    return (ecx & (1u << 29)) != 0;
#else
    return false;
#endif
}

// Timed wait on one line: the core parks until the line is written or the
// TSC timer runs out, whichever comes first. The hint keeps the core in C0
// so a wake costs no state exit.
void MonitorWait(const void* line, u32 ticks) {
#ifdef SHAD_COPY_LANE_MWAITX
    asm volatile("monitorx" : : "a"(line), "c"(0u), "d"(0u) : "memory");
    asm volatile("mwaitx" : : "a"(0xF0u), "b"(ticks), "c"(0x2u) : "memory");
#else
    (void)line;
    (void)ticks;
    CpuPause();
#endif
}

} // namespace

StreamCopyLane& StreamCopyLane::Instance() {
    static StreamCopyLane lane;
    return lane;
}

void StreamCopyLane::Init(u32 num_workers, bool hardened, u32 idle_ticks) {
    if (num_workers == 0 || !threads_.empty()) {
        return;
    }
    // Public entry point: clamp here rather than trust the caller, because the
    // rank indexes a fixed-size cell array.
    num_workers = std::min<u32>(num_workers, kMaxWorkers);
    hardened_ = hardened;
    idle_ticks_ = CpuHasMwaitx() ? idle_ticks : 0;
    slots_ = std::make_unique<Slot[]>(kRingSlots);
    for (size_t i = 0; i < kRingSlots; ++i) {
        slots_[i].seq.store(i, std::memory_order_relaxed);
    }
    stop_.store(false, std::memory_order_relaxed);
    producer_tid_.store(0, std::memory_order_relaxed);
    threads_.reserve(num_workers);
    for (u32 i = 0; i < num_workers; ++i) {
        threads_.emplace_back([this, i] { WorkerLoop(i); });
    }
    num_workers_.store(num_workers, std::memory_order_release);
}

void StreamCopyLane::Shutdown() {
    if (threads_.empty()) {
        return;
    }
    num_workers_.store(0, std::memory_order_release);
    stop_.store(true, std::memory_order_release);
    // The value change is what releases sleepers; a bare notify may be missed.
    published_.fetch_add(1, std::memory_order_release);
    published_.notify_all();
    for (auto& t : threads_) {
        t.join();
    }
    threads_.clear();
    slots_.reset();
    // The cells, published_ and dequeue_pos_ are cumulative for the object's
    // life and must be reset together or not at all: zeroing only the cells
    // would leave the next DrainProducer waiting on a carried-over target.
}

bool StreamCopyLane::Push(const u8* src, u8* dst, u32 size) {
    // Single-producer contract, enforced in safe mode: readback fault paths
    // run emulator code on guest threads, and a second producer would corrupt
    // the plain enqueue cursor. A foreign caller copies inline instead.
    if (hardened_) {
        const u64 tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        if (producer_tid_.load(std::memory_order_relaxed) != tid) {
            u64 expected = 0;
            if (!producer_tid_.compare_exchange_strong(expected, tid, std::memory_order_relaxed)) {
                return false;
            }
        }
    }
    // Only the producer writes enqueue_pos_, so one read serves the whole
    // job; reading the member again would reload it across the slot stores.
    const u64 pos = enqueue_pos_;
    Slot& slot = slots_[pos & (kRingSlots - 1)];
    if (slot.seq.load(std::memory_order_acquire) != pos) {
        ++inline_full_; // a whole lap is still in flight
        return false;
    }
    slot.job = Job{src, dst, size};
    slot.seq.store(pos + 1, std::memory_order_release);
    enqueue_pos_ = pos + 1;
    published_.store(pos + 1, std::memory_order_release);
    if (sleepers_.load(std::memory_order_relaxed) != 0) {
        published_.notify_all();
    }
    ++jobs_;
    bytes_ += size;
    return true;
}

bool StreamCopyLane::ClaimAndCopy() {
    u64 pos = dequeue_pos_.load(std::memory_order_relaxed);
    Slot* slot = nullptr;
    while (true) {
        // The slot must be re-derived every lap: a lost exchange leaves pos
        // holding the winner's cursor. The acquire orders every read of job.
        slot = &slots_[pos & (kRingSlots - 1)];
        if (slot->seq.load(std::memory_order_acquire) != pos + 1) {
            return false;
        }
        if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
            break;
        }
    }
    const Job job = slot->job;
    // Recycle the slot before the copy: the producer only reuses it a full
    // lap later, and the payload bytes are ordered by the caller's completion
    // store, which every drain waiter reads with acquire.
    slot->seq.store(pos + kRingSlots, std::memory_order_release);
    std::memcpy(job.dst, job.src, job.size);
    return true;
}

bool StreamCopyLane::TryDrainShared() {
    if (!ClaimAndCopy()) {
        return false;
    }
    // Unconditional, in both lane modes: guest fault threads reach the drains
    // through Buffer::Copy -> Map -> MapWrap, so this is not the worker path.
    cells_[kHelperCell].done.fetch_add(1, std::memory_order_release);
    return true;
}

u64 StreamCopyLane::Completed() const {
    u64 total = 0;
    for (const Cell& cell : cells_) {
        total += cell.done.load(std::memory_order_acquire);
    }
    return total;
}

void StreamCopyLane::WorkerLoop(u32 rank) {
    Common::SetCurrentThreadName("shadPS4:CopyLane");
    u32 idle_rounds = 0;
    const u32 idle_ticks = idle_ticks_;
    Cell& cell = cells_[rank];
    // Seeded, not zeroed: a rank reused after Shutdown+Init must continue its
    // cell, or the first store regresses a total the drains wait on.
    u64 local_done = cell.done.load(std::memory_order_relaxed);
    u64 local_mwaits = cell.mwaits.load(std::memory_order_relaxed);
    u64 local_wakes = cell.mwait_wakes.load(std::memory_order_relaxed);
    while (true) {
        if (ClaimAndCopy()) {
            // Release: the bytes are in place before the count that publishes
            // them. Only this thread writes the line, so it is a plain store.
            cell.done.store(++local_done, std::memory_order_release);
            idle_rounds = 0;
            continue;
        }
        if (stop_.load(std::memory_order_relaxed)) {
            return;
        }
        if (++idle_rounds < 512) {
            if (idle_ticks != 0) {
                // Arm the monitor, then re-check the cursor: a publish that
                // landed between the drain attempt and the arm is seen here
                // and skips the wait, so the wait never misses it.
                const u64 seen = published_.load(std::memory_order_acquire);
                if (seen != dequeue_pos_.load(std::memory_order_relaxed)) {
                    continue;
                }
                MonitorWait(&published_, idle_ticks);
                cell.mwaits.store(++local_mwaits, std::memory_order_relaxed);
                if (published_.load(std::memory_order_acquire) != seen) {
                    cell.mwait_wakes.store(++local_wakes, std::memory_order_relaxed);
                }
                continue;
            }
            for (int i = 0; i < 64; ++i) {
                CpuPause();
            }
            continue;
        }
        idle_rounds = 0;
        const u64 seen = published_.load(std::memory_order_acquire);
        if (seen != dequeue_pos_.load(std::memory_order_relaxed)) {
            continue; // work appeared between the slot check and here
        }
        sleepers_.fetch_add(1, std::memory_order_relaxed);
        published_.wait(seen, std::memory_order_acquire);
        sleepers_.fetch_sub(1, std::memory_order_relaxed);
    }
}

void StreamCopyLane::DrainProducer() {
    if (!Enabled()) {
        return;
    }
    // Safe mode targets published_ rather than the producer-plain cursor:
    // fault-path flushes reach this from guest threads, and on the producer
    // itself the two are always equal (published_ is stored on every push).
    const u64 target = hardened_ ? published_.load(std::memory_order_acquire) : enqueue_pos_;
    // The hint first: summing five worker-written lines on every already-drained
    // submit would put back the contention this split removes.
    if (completed_seen_.load(std::memory_order_relaxed) >= target) {
        return;
    }
    u64 done = Completed();
    if (done >= target) {
        completed_seen_.store(done, std::memory_order_relaxed);
        return;
    }
    ++barriers_;
    const auto t0 = std::chrono::steady_clock::now();
    while (done < target) {
        // Help instead of stalling: the producer drains jobs itself, so a
        // descheduled worker can never wedge a submit.
        if (!TryDrainShared()) {
            CpuPause();
        }
        done = Completed();
    }
    completed_seen_.store(done, std::memory_order_relaxed);
    barrier_wait_ns_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
            .count();
}

void StreamCopyLane::DrainRemote() {
    if (!Enabled()) {
        return;
    }
    const u64 target = published_.load(std::memory_order_acquire);
    // Same hint as the producer drain: a guest unmap almost always finds the
    // lane idle, and spinning on the sum would read five worker lines a lap.
    if (completed_seen_.load(std::memory_order_relaxed) >= target) {
        return;
    }
    u64 done = Completed();
    while (done < target) {
        CpuPause();
        done = Completed();
    }
    completed_seen_.store(done, std::memory_order_relaxed);
}

StreamCopyLane::Stats StreamCopyLane::DrainStats() {
    // Deltas against producer-held baselines: an exchange here would be a
    // command-thread RMW on a line a worker is writing every job.
    Stats out{};
    out.jobs = jobs_;
    out.bytes = bytes_;
    out.inline_unresolved = inline_unresolved_;
    out.inline_full = inline_full_;
    out.barriers = barriers_;
    out.barrier_wait_ns = barrier_wait_ns_;
    u64 mwaits = 0;
    u64 wakes = 0;
    for (u32 i = 0; i < kCells; ++i) {
        const u64 done = cells_[i].done.load(std::memory_order_relaxed);
        const u64 slice = done - done_base_[i];
        done_base_[i] = done;
        if (i == kHelperCell) {
            out.helper_jobs = slice;
        } else {
            out.worker_jobs[i] = slice;
        }
        mwaits += cells_[i].mwaits.load(std::memory_order_relaxed);
        wakes += cells_[i].mwait_wakes.load(std::memory_order_relaxed);
    }
    out.mwaits = mwaits - mwaits_base_;
    out.mwait_wakes = wakes - wakes_base_;
    mwaits_base_ = mwaits;
    wakes_base_ = wakes;
    jobs_ = 0;
    bytes_ = 0;
    inline_unresolved_ = 0;
    inline_full_ = 0;
    barriers_ = 0;
    barrier_wait_ns_ = 0;
    return out;
}

} // namespace VideoCore
