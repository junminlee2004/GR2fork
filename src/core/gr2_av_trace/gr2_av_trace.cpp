// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include <SDL3/SDL_audio.h>
#include <fmt/format.h>

#include "common/path_util.h"
#include "common/thread.h"
#include "core/gr2_av_trace/gr2_av_trace.h"

namespace Gr2AvTrace {

namespace {

using Clock = std::chrono::steady_clock;

// A port that took no block for this long starts its output deficit again.
constexpr double IdleResetMs = 1000.0;

struct Port {
    s32 type{};
    u32 sample_rate{};
    double block_ms{};
    const char* host = "-";
    Clock::time_point last_tick{};
    Clock::time_point first_block{};
    Clock::time_point last_block{};
    u64 total_blocks{};
    u32 ticks{};
    u32 blocks{};
    u32 empty_ticks{};
    u32 fast_blocks{};
    double max_gap_ms{};
    u32 queue_samples{};
    u32 dry_blocks{};
    double queue_sum_ms{};
    double queue_min_ms{};
    double queue_max_ms{};
};

struct Video {
    u32 id{};
    u32 frames{};
    u32 none_left{};
    double late_sum_ms{};
    double late_max_ms{};
    u64 last_timestamp_ms{};
};

struct Decoder {
    u64 last_total{};
    u64 samples{};
};

struct State {
    std::mutex mutex;
    std::map<const void*, Port> ports;
    std::map<const void*, Video> videos;
    std::map<u32, Decoder> decoders;
    u32 next_video_id{};
    Clock::time_point start{};
    bool started{};
};

const bool g_enabled = [] {
    const char* enabled = std::getenv("GR2_AV_TRACE");
    return enabled != nullptr && std::string_view{enabled} == "1";
}();

// Never destroyed, so hooks that run while the emulator exits still find it.
State& GetState() {
    static auto* state = new State{};
    return *state;
}

double Millis(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

std::string Report(State& state, Clock::time_point now) {
    std::string out;
    const double t = Millis(now - state.start) / 1000.0;
    for (auto& [backend, port] : state.ports) {
        if (port.ticks == 0) {
            continue;
        }
        const double behind_ms =
            port.total_blocks == 0 ? 0.0
                                   : Millis(port.last_block - port.first_block) -
                                         static_cast<double>(port.total_blocks - 1) * port.block_ms;
        const double queue_avg_ms =
            port.queue_samples == 0 ? 0.0 : port.queue_sum_ms / port.queue_samples;
        out += fmt::format("{:.1f}\tport\t{}\thost={}\tticks={}\tblocks={}\tempty={}\tfast={}\t"
                           "gap_max_ms={:.1f}\tqueue_ms={:.1f}/{:.1f}/{:.1f}\tdry={}\t"
                           "behind_ms={:.1f}\n",
                           t, port.type, port.host, port.ticks, port.blocks, port.empty_ticks,
                           port.fast_blocks, port.max_gap_ms, port.queue_min_ms, queue_avg_ms,
                           port.queue_max_ms, port.dry_blocks, behind_ms);
        port.ticks = port.blocks = port.empty_ticks = port.fast_blocks = 0;
        port.queue_samples = port.dry_blocks = 0;
        port.max_gap_ms = port.queue_sum_ms = port.queue_min_ms = port.queue_max_ms = 0.0;
    }
    for (auto& [source, video] : state.videos) {
        if (video.frames == 0) {
            continue;
        }
        out += fmt::format("{:.1f}\tvideo\t{}\tframes={}\tlate_ms={:.1f}/{:.1f}\tnone_left={}\t"
                           "timestamp_ms={}\n",
                           t, video.id, video.frames, video.late_sum_ms / video.frames,
                           video.late_max_ms, video.none_left, video.last_timestamp_ms);
        video.frames = video.none_left = 0;
        video.late_sum_ms = video.late_max_ms = 0.0;
    }
    std::string decoded;
    for (auto& [id, decoder] : state.decoders) {
        if (decoder.samples != 0) {
            decoded += fmt::format(" {}={}", id, decoder.samples);
            decoder.samples = 0;
        }
    }
    if (!decoded.empty()) {
        out += fmt::format("{:.1f}\tajm\t-\t{}\n", t, decoded.substr(1));
    }
    return out;
}

void ReporterThread() {
    Common::SetCurrentThreadName("shadPS4:Gr2AvTrace");
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "gr2_av_trace.tsv";
    std::ofstream file{path, std::ios::trunc};
    file << "# port: ticks, blocks taken, ticks with no guest block, blocks taken under half a "
            "block after the previous one, longest gap between ticks, host queue min/avg/max, "
            "blocks taken with the host queue empty, wall time minus audio output since the first "
            "block\n"
            "# video: frames taken, player clock minus frame timestamp avg/max, frames taken with "
            "no decoded frame left, last timestamp\n"
            "# ajm: samples per channel each instance decoded\n";
    file.flush();
    auto next = Clock::now();
    for (;;) {
        next += std::chrono::seconds{1};
        std::this_thread::sleep_until(next);
        std::string out;
        {
            auto& state = GetState();
            std::scoped_lock lock{state.mutex};
            out = Report(state, Clock::now());
        }
        if (!out.empty()) {
            file << out;
            file.flush();
        }
    }
}

// Called with the state lock held.
void StartReporter(State& state) {
    if (state.started) {
        return;
    }
    state.started = true;
    state.start = Clock::now();
    std::thread{ReporterThread}.detach();
}

void AddQueue(const void* backend, const char* host, u64 queued_frames) {
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
    auto& port = state.ports[backend];
    if (port.sample_rate == 0) {
        return;
    }
    port.host = host;
    const double queue_ms = 1000.0 * static_cast<double>(queued_frames) / port.sample_rate;
    if (port.queue_samples == 0) {
        port.queue_min_ms = port.queue_max_ms = queue_ms;
    } else {
        port.queue_min_ms = std::min(port.queue_min_ms, queue_ms);
        port.queue_max_ms = std::max(port.queue_max_ms, queue_ms);
    }
    port.queue_sum_ms += queue_ms;
    ++port.queue_samples;
    if (queued_frames == 0) {
        ++port.dry_blocks;
    }
}

} // namespace

void OnAudioTick(const void* backend, s32 port_type, u32 sample_rate, u32 block_frames,
                 bool block_ready) {
    if (!g_enabled || sample_rate == 0) {
        return;
    }
    const auto now = Clock::now();
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
    StartReporter(state);
    auto& port = state.ports[backend];
    port.type = port_type;
    port.sample_rate = sample_rate;
    port.block_ms = 1000.0 * block_frames / sample_rate;
    if (port.last_tick != Clock::time_point{}) {
        port.max_gap_ms = std::max(port.max_gap_ms, Millis(now - port.last_tick));
    }
    port.last_tick = now;
    ++port.ticks;
    if (!block_ready) {
        ++port.empty_ticks;
        return;
    }
    ++port.blocks;
    if (port.total_blocks != 0 && Millis(now - port.last_block) > IdleResetMs) {
        port.total_blocks = 0;
    }
    if (port.total_blocks == 0) {
        port.first_block = now;
    } else if (Millis(now - port.last_block) < port.block_ms / 2) {
        ++port.fast_blocks;
    }
    port.last_block = now;
    ++port.total_blocks;
}

void OnSdlQueue(const void* backend, SDL_AudioStream* stream, u32 bytes_per_frame) {
    if (!g_enabled || bytes_per_frame == 0) {
        return;
    }
    const int queued = SDL_GetAudioStreamQueued(stream);
    if (queued >= 0) {
        AddQueue(backend, "sdl", static_cast<u64>(queued) / bytes_per_frame);
    }
}

void OnOpenALQueue(const void* backend, u64 queued_frames) {
    if (g_enabled) {
        AddQueue(backend, "openal", queued_frames);
    }
}

void OnVideoFrame(const void* source, u64 timestamp_ms, u64 clock_ms, u64 frames_left) {
    if (!g_enabled) {
        return;
    }
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
    StartReporter(state);
    const auto [it, inserted] = state.videos.try_emplace(source);
    auto& video = it->second;
    // A source whose timestamps go back is playing a new movie.
    if (inserted || timestamp_ms < video.last_timestamp_ms) {
        video.id = state.next_video_id++;
    }
    const double late_ms = static_cast<double>(static_cast<s64>(clock_ms - timestamp_ms));
    video.late_max_ms = video.frames == 0 ? late_ms : std::max(video.late_max_ms, late_ms);
    video.late_sum_ms += late_ms;
    ++video.frames;
    if (frames_left == 0) {
        ++video.none_left;
    }
    video.last_timestamp_ms = timestamp_ms;
}

void OnAjmJob(u32 instance_id, u64 total_samples) {
    if (!g_enabled) {
        return;
    }
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
    auto& decoder = state.decoders[instance_id];
    decoder.samples +=
        total_samples >= decoder.last_total ? total_samples - decoder.last_total : total_samples;
    decoder.last_total = total_samples;
}

} // namespace Gr2AvTrace
