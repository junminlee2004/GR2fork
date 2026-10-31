// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#include <fmt/format.h>

#include "common/alignment.h"
#include "common/gravity_rush.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "core/gr2_clock_trace/gr2_clock_trace.h"

namespace Gr2ClockTrace {

namespace {

using Clock = std::chrono::steady_clock;

// Patch XML addresses (module offset + 0x400000) of the values the fps patches keep in the
// eboot's RW tail, and the game's own frame counter and frame period.
constexpr u64 XmlBase = 0x400000;
constexpr u64 TsF = 0x1f7c600;
constexpr u64 TsSm1 = TsF + 0xd4;
constexpr u64 TsLock = TsF + 0xd8;
constexpr u64 FrameCount = 0x1efb08c;
constexpr u64 PeriodUs = 0x1f585c8;
constexpr u64 LastAddress = TsLock + 4;

// The time service runs a few microseconds after the frame counter moves; the frame's values are
// read one poll later.
constexpr auto PollInterval = std::chrono::microseconds{500};
constexpr auto FlushInterval = std::chrono::seconds{1};

std::mutex mutex;
std::string pending;

double Millis(Clock::time_point t) {
    static const Clock::time_point start = Clock::now();
    return std::chrono::duration<double, std::milli>(t - start).count();
}

template <typename T>
T ReadGuest(u64 xml_address) {
    T value;
    std::memcpy(
        &value,
        reinterpret_cast<const void*>(MemoryPatcher::g_eboot_address + xml_address - XmlBase),
        sizeof(value));
    return value;
}

// The RW tail lies past the last segment's memory size, inside the page-aligned module mapping.
bool GuestReadable() {
    return MemoryPatcher::g_eboot_address != 0 &&
           LastAddress - XmlBase <= Common::AlignUp(MemoryPatcher::g_eboot_image_size, 0x4000);
}

void Append(std::string line) {
    std::scoped_lock lock{mutex};
    pending += line;
}

void PollThread() {
    Common::SetCurrentThreadName("shadPS4:Gr2ClockTrace");
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "gr2_clock_trace.tsv";
    std::ofstream file{path, std::ios::trunc};
    file << "# times in ms since the trace started\n"
            "# frame: time the frame counter moved, frame counter, f (game step in 30 Hz frames),"
            " s (smoothed step in target frames, one-clock build), frames near the target period"
            " (one-clock build), game frame period in us\n"
            "# present: time, vblank count\n";
    file.flush();

    u32 last_frame = ReadGuest<u32>(FrameCount);
    bool frame_pending = false;
    u32 frame = 0;
    double frame_ms = 0.0;
    auto next_flush = Clock::now() + FlushInterval;
    for (;;) {
        Common::AccurateSleep(PollInterval, nullptr, false);
        const auto now = Clock::now();
        if (frame_pending) {
            Append(fmt::format("frame\t{:.3f}\t{}\t{:.6f}\t{:.6f}\t{}\t{}\n", frame_ms, frame,
                               ReadGuest<float>(TsF), ReadGuest<float>(TsSm1) + 1.0f,
                               ReadGuest<u32>(TsLock), ReadGuest<s64>(PeriodUs)));
            frame_pending = false;
        }
        const u32 count = ReadGuest<u32>(FrameCount);
        if (count != last_frame) {
            last_frame = count;
            frame = count;
            frame_ms = Millis(now);
            frame_pending = true;
        }
        if (now >= next_flush) {
            next_flush = now + FlushInterval;
            std::string out;
            {
                std::scoped_lock lock{mutex};
                out.swap(pending);
            }
            file << out;
            file.flush();
        }
    }
}

bool Start() {
    static const bool started = [] {
        if (!Common::IsGravityRush2() || !GuestReadable()) {
            return false;
        }
        std::thread{PollThread}.detach();
        return true;
    }();
    return started;
}

} // namespace

void OnPresent(u64 vblank_count) {
    if (!Start()) {
        return;
    }
    Append(fmt::format("present\t{:.3f}\t{}\n", Millis(Clock::now()), vblank_count));
}

} // namespace Gr2ClockTrace
