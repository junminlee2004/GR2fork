// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/buffer_cache/stream_copy_lane.h"
#include "video_core/gr2_native/frame_sync.h"
#include "video_core/gr2_native/marker.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Gr2Native {

// The fields of Sony's sce::Gnm::DrawCommandBuffer that a marker is written through.
struct DrawCommandBuffer {
    u32* begin;
    u32* end;
    u32* cursor;
    // Makes room for the given number of words, or returns false if there is none.
    bool(PS4_SYSV_ABI* reserve)(DrawCommandBuffer* dcb, u32 dwords, void* user_data);
    void* user_data;
};
static_assert(offsetof(DrawCommandBuffer, end) == 0x08);
static_assert(offsetof(DrawCommandBuffer, cursor) == 0x10);
static_assert(offsetof(DrawCommandBuffer, reserve) == 0x18);
static_assert(offsetof(DrawCommandBuffer, user_data) == 0x20);

namespace {

using Clock = std::chrono::steady_clock;
using WriteAtEndOfPipeFn = void(PS4_SYSV_ABI*)(DrawCommandBuffer* dcb, u32 event_type,
                                               u32 dst_select, u64* label, u32 src_select,
                                               u64 value, u32 cache_action, u32 cache_policy);

// The EventWriteDest and EventWriteSource of a frame label write: a 64-bit immediate to memory.
constexpr u32 DestMemory = 0;
constexpr u32 Source64BitImmediate = 2;
// Frame statistics are logged every this many frame labels.
constexpr u64 StatsFrames = 600;
constexpr auto WaitWarning = std::chrono::seconds{1};

// Per StatsFrames labels: the time from the command processor reaching a marker to the GPU
// finishing the frame, and the draw thread's waits that found a label not written yet.
struct Stats {
    u64 labels;
    u64 gpu_ns;
    u64 gpu_max_ns;
    u64 waits;
    u64 blocked;
    u64 blocked_ns;
    u64 blocked_max_ns;
};

WriteAtEndOfPipeFn g_write_eop;
std::mutex g_mutex;
std::condition_variable g_written;
Stats g_stats;

u64 Nanoseconds(Clock::duration duration) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

u64 LoadLabel(u64* label) {
    return std::atomic_ref<u64>{*label}.load(std::memory_order_acquire);
}

void LogStats(const Stats& stats) {
    LOG_INFO(Render_Vulkan,
             "GR2 native: {} frames, the GPU finished each {:.2f} ms after the command processor "
             "reached its end (longest {:.2f} ms); "
             "the draw thread waited {} times, {} for an unwritten label, {:.2f} ms each "
             "(longest {:.2f} ms)",
             stats.labels, stats.gpu_ns / 1e6 / std::max<u64>(stats.labels, 1),
             stats.gpu_max_ns / 1e6, stats.waits, stats.blocked,
             stats.blocked_ns / 1e6 / std::max<u64>(stats.blocked, 1), stats.blocked_max_ns / 1e6);
}

// Writes the label as the command processor writes end-of-pipe data, then wakes the draw thread.
void Publish(u64* label, u64 value, Clock::time_point reached) {
    if (!Core::Memory::Instance()->TryWriteBacking(label, &value, sizeof(value))) {
        std::atomic_ref<u64>{*label}.store(value, std::memory_order_release);
    }
    const u64 gpu_ns = Nanoseconds(Clock::now() - reached);
    Stats finished{};
    {
        std::scoped_lock lock{g_mutex};
        ++g_stats.labels;
        g_stats.gpu_ns += gpu_ns;
        g_stats.gpu_max_ns = std::max(g_stats.gpu_max_ns, gpu_ns);
        if (g_stats.labels == StatsFrames) {
            finished = g_stats;
            g_stats = {};
        }
    }
    g_written.notify_all();
    if (finished.labels != 0) {
        LogStats(finished);
    }
}

} // namespace

void SetWriteAtEndOfPipe(VAddr address) {
    g_write_eop = reinterpret_cast<WriteAtEndOfPipeFn>(address);
}

void PS4_SYSV_ABI WriteFrameLabel(DrawCommandBuffer* dcb, u32 event_type, u32 dst_select,
                                  u64* label, u32 src_select, u64 value, u32 cache_action,
                                  u32 cache_policy) {
    if (dst_select != DestMemory || src_select != Source64BitImmediate ||
        reinterpret_cast<uintptr_t>(label) % alignof(u64) != 0) {
        g_write_eop(dcb, event_type, dst_select, label, src_select, value, cache_action,
                    cache_policy);
        return;
    }
    constexpr u32 Words = 1 + FrameLabelPayloadDwords;
    // As in Sony's packet writers, a command buffer that cannot grow drops the packet.
    if (dcb->end - dcb->cursor < Words && !dcb->reserve(dcb, Words, dcb->user_data)) {
        return;
    }
    const u64 address = reinterpret_cast<u64>(label);
    u32* packet = dcb->cursor;
    packet[0] = NopHeader(FrameLabelPayloadDwords);
    packet[1] = MarkerMagic;
    packet[2] = static_cast<u32>(MarkerKind::FrameLabel);
    packet[3] = static_cast<u32>(address);
    packet[4] = static_cast<u32>(address >> 32);
    packet[5] = static_cast<u32>(value);
    packet[6] = static_cast<u32>(value >> 32);
    dcb->cursor = packet + Words;
}

void PS4_SYSV_ABI WaitFrameLabel(u64* label, u64 expected) {
    const auto start = Clock::now();
    std::unique_lock lock{g_mutex};
    ++g_stats.waits;
    if (LoadLabel(label) == expected) {
        return;
    }
    while (!g_written.wait_for(lock, WaitWarning, [&] { return LoadLabel(label) == expected; })) {
        LOG_WARNING(Render_Vulkan,
                    "GR2 native: the frame label at {} holds {} after a second of waiting for {}",
                    fmt::ptr(label), LoadLabel(label), expected);
    }
    const u64 blocked_ns = Nanoseconds(Clock::now() - start);
    ++g_stats.blocked;
    g_stats.blocked_ns += blocked_ns;
    g_stats.blocked_max_ns = std::max(g_stats.blocked_max_ns, blocked_ns);
}

void RunFrameLabel(const u32* payload, Vulkan::Rasterizer* rasterizer) {
    auto* label = reinterpret_cast<u64*>(u64{payload[2]} | u64{payload[3]} << 32);
    const u64 value = u64{payload[4]} | u64{payload[5]} << 32;
    // The command processor's end-of-pipe handling: the copy lanes drain and the downloads are
    // queued before anything guest-visible is written.
    VideoCore::StreamCopyLane::Instance().DrainProducer();
    const auto reached = Clock::now();
    if (rasterizer == nullptr) {
        Publish(label, value, reached);
        return;
    }
    rasterizer->OnFence();
    // The frame's commands are all recorded, so the current tick covers them. The draw thread
    // submits and flips right after this marker, which submits the tick.
    rasterizer->GetScheduler().DeferPriorityOperation(
        [label, value, reached] { Publish(label, value, reached); });
}

} // namespace Gr2Native
