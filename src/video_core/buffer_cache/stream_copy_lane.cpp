// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstring>
#include <emmintrin.h>
#include "common/thread.h"
#include "video_core/buffer_cache/stream_copy_lane.h"

namespace VideoCore {

StreamCopyLane& StreamCopyLane::Instance() {
    static StreamCopyLane lane;
    return lane;
}

void StreamCopyLane::Init(u32 num_workers) {
    if (num_workers == 0 || !threads_.empty()) {
        return;
    }
    slots_ = std::make_unique<Slot[]>(kRingSlots);
    for (size_t i = 0; i < kRingSlots; ++i) {
        slots_[i].seq.store(i, std::memory_order_relaxed);
    }
    stop_.store(false, std::memory_order_relaxed);
    threads_.reserve(num_workers);
    for (u32 i = 0; i < num_workers; ++i) {
        threads_.emplace_back([this] { WorkerLoop(); });
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
}

bool StreamCopyLane::Push(const u8* src, u8* dst, u32 size) {
    Slot& slot = slots_[enqueue_pos_ & (kRingSlots - 1)];
    if (slot.seq.load(std::memory_order_acquire) != enqueue_pos_) {
        ++inline_full_; // a whole lap is still in flight
        return false;
    }
    slot.job = Job{src, dst, size};
    slot.seq.store(enqueue_pos_ + 1, std::memory_order_release);
    ++enqueue_pos_;
    published_.store(enqueue_pos_, std::memory_order_release);
    if (sleepers_.load(std::memory_order_relaxed) != 0) {
        published_.notify_all();
    }
    ++jobs_;
    bytes_ += size;
    return true;
}

bool StreamCopyLane::TryDrainOne() {
    u64 pos = dequeue_pos_.load(std::memory_order_relaxed);
    Slot& slot = slots_[pos & (kRingSlots - 1)];
    if (slot.seq.load(std::memory_order_acquire) != pos + 1) {
        return false;
    }
    if (!dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
        return true; // another worker took it; there may be more
    }
    const Job job = slot.job;
    // Recycle the slot before the copy: the producer only reuses it a full
    // lap later, and the payload bytes are ordered by copies_done_ below.
    slot.seq.store(pos + kRingSlots, std::memory_order_release);
    std::memcpy(job.dst, job.src, job.size);
    copies_done_.fetch_add(1, std::memory_order_release);
    return true;
}

void StreamCopyLane::WorkerLoop() {
    Common::SetCurrentThreadName("shadPS4:CopyLane");
    u32 idle_rounds = 0;
    while (true) {
        if (TryDrainOne()) {
            idle_rounds = 0;
            continue;
        }
        if (stop_.load(std::memory_order_relaxed)) {
            return;
        }
        if (++idle_rounds < 512) {
            for (int i = 0; i < 64; ++i) {
                _mm_pause();
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
    const u64 target = enqueue_pos_;
    if (copies_done_.load(std::memory_order_acquire) >= target) {
        return;
    }
    ++barriers_;
    const auto t0 = std::chrono::steady_clock::now();
    while (copies_done_.load(std::memory_order_acquire) < target) {
        // Help instead of stalling: the producer drains jobs itself, so a
        // descheduled worker can never wedge a submit.
        if (!TryDrainOne()) {
            _mm_pause();
        }
    }
    barrier_wait_ns_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
            .count();
}

void StreamCopyLane::DrainRemote() {
    if (!Enabled()) {
        return;
    }
    const u64 target = published_.load(std::memory_order_acquire);
    while (copies_done_.load(std::memory_order_acquire) < target) {
        _mm_pause();
    }
}

StreamCopyLane::Stats StreamCopyLane::DrainStats() {
    const Stats out{
        .jobs = jobs_,
        .bytes = bytes_,
        .inline_unresolved = inline_unresolved_,
        .inline_full = inline_full_,
        .barriers = barriers_,
        .barrier_wait_ns = barrier_wait_ns_,
    };
    jobs_ = 0;
    bytes_ = 0;
    inline_unresolved_ = 0;
    inline_full_ = 0;
    barriers_ = 0;
    barrier_wait_ns_ = 0;
    return out;
}

} // namespace VideoCore
