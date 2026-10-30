// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <string>
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

constexpr auto CanarySleep = std::chrono::milliseconds{1};

constexpr std::array<const char*, 2> CanaryNames{"normal", "critical"};
constexpr std::array<const char*, 2> FlipNames{"submit", "present"};

double Millis(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

Clock::time_point StartTime() {
    static const Clock::time_point start = Clock::now();
    return start;
}

// The longest of a run of intervals and the trace time it ended at.
struct Peak {
    double ms{};
    double at_s{};
    double extra_ms{};

    void Add(double value_ms, Clock::time_point end, double extra = 0.0) {
        if (value_ms > ms) {
            ms = value_ms;
            at_s = Millis(end - StartTime()) / 1000.0;
            extra_ms = extra;
        }
    }

    std::string Take() {
        auto out = fmt::format("{:.1f}@{:.3f}", ms, at_s);
        *this = {};
        return out;
    }
};

struct Port {
    std::mutex mutex;
    s32 type{};
    u32 sample_rate{};
    double block_ms{};
    const char* host = "-";
    Clock::time_point lock_request{};
    Clock::time_point tick_start{};
    Clock::time_point tick_end{};
    Clock::time_point last_tick{};
    Clock::time_point first_block{};
    Clock::time_point last_block{};
    u64 total_blocks{};
    u32 ticks{};
    u32 blocks{};
    u32 empty_ticks{};
    u32 fast_blocks{};
    Peak gap;
    Peak sleep;
    Peak lock_wait;
    Peak output;
    Peak backend_sleep;
    u32 queue_samples{};
    double queue_sum_ms{};
    double queue_min_ms{};
    double queue_max_ms{};
};

struct Guest {
    std::mutex mutex;
    u32 calls{};
    Peak lock;
    Peak wait;
    Peak between;
};

struct Canary {
    std::mutex mutex;
    u32 wakes{};
    Peak late;
};

struct Flips {
    std::mutex mutex;
    u32 count{};
    Clock::time_point last{};
    Peak gap;
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
    // Guards the maps; each entry has its own lock for the per-tick updates.
    std::mutex mutex;
    std::map<const void*, Port> ports;
    std::map<const void*, Video> videos;
    std::map<u32, Decoder> decoders;
    u32 next_video_id{};
    Guest guest;
    std::array<Canary, 2> canaries;
    std::array<Flips, 2> flips;
};

// Never destroyed, so hooks that run while the emulator exits still find it.
State& GetState() {
    static auto* state = new State{};
    return *state;
}

// Each output thread serves one port, so it keeps the entry it last looked up.
thread_local const void* tl_backend = nullptr;
thread_local Port* tl_port = nullptr;

Port* FindPort(const void* backend) {
    if (backend == nullptr) {
        return nullptr;
    }
    if (backend != tl_backend) {
        auto& state = GetState();
        std::scoped_lock lock{state.mutex};
        tl_port = &state.ports[backend];
        tl_backend = backend;
    }
    return tl_port;
}

std::string Report(State& state, Clock::time_point now) {
    std::string out;
    const double t = Millis(now - StartTime()) / 1000.0;
    {
        std::scoped_lock lock{state.mutex};
        for (auto& [backend, port] : state.ports) {
            std::scoped_lock port_lock{port.mutex};
            if (port.ticks == 0) {
                continue;
            }
            const double behind_ms =
                port.total_blocks == 0
                    ? 0.0
                    : Millis(port.last_block - port.first_block) -
                          static_cast<double>(port.total_blocks - 1) * port.block_ms;
            const double queue_avg_ms =
                port.queue_samples == 0 ? 0.0 : port.queue_sum_ms / port.queue_samples;
            const double requested_ms = port.backend_sleep.extra_ms;
            out += fmt::format("{:.1f}\tport\t{}\thost={}\tticks={}\tblocks={}\tempty={}\tfast={}\t"
                               "gap={}\tsleep={}\tlock={}\toutput={}\t",
                               t, port.type, port.host, port.ticks, port.blocks, port.empty_ticks,
                               port.fast_blocks, port.gap.Take(), port.sleep.Take(),
                               port.lock_wait.Take(), port.output.Take());
            out += fmt::format("backend_sleep={}/req{:.1f}\tqueue_ms={:.1f}/{:.1f}/{:.1f}\t"
                               "behind_ms={:.1f}\n",
                               port.backend_sleep.Take(), requested_ms, port.queue_min_ms,
                               queue_avg_ms, port.queue_max_ms, behind_ms);
            port.ticks = port.blocks = port.empty_ticks = port.fast_blocks = 0;
            port.queue_samples = 0;
            port.queue_sum_ms = port.queue_min_ms = port.queue_max_ms = 0.0;
        }
        for (auto& [source, video] : state.videos) {
            if (video.frames == 0) {
                continue;
            }
            out += fmt::format("{:.1f}\tvideo\t{}\tframes={}\tlate_ms={:.1f}/{:.1f}\t"
                               "none_left={}\ttimestamp_ms={}\n",
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
    }
    {
        auto& guest = state.guest;
        std::scoped_lock lock{guest.mutex};
        if (guest.calls != 0) {
            out += fmt::format("{:.1f}\tguest\t-\tcalls={}\tlock={}\twait={}\tbetween={}\n", t,
                               guest.calls, guest.lock.Take(), guest.wait.Take(),
                               guest.between.Take());
            guest.calls = 0;
        }
    }
    for (std::size_t i = 0; i < state.canaries.size(); ++i) {
        auto& canary = state.canaries[i];
        std::scoped_lock lock{canary.mutex};
        out += fmt::format("{:.1f}\tcanary\t{}\twakes={}\tlate={}\n", t, CanaryNames[i],
                           canary.wakes, canary.late.Take());
        canary.wakes = 0;
    }
    for (std::size_t i = 0; i < state.flips.size(); ++i) {
        auto& flips = state.flips[i];
        std::scoped_lock lock{flips.mutex};
        if (flips.count != 0) {
            out += fmt::format("{:.1f}\tflip\t{}\tcount={}\tgap={}\n", t, FlipNames[i], flips.count,
                               flips.gap.Take());
            flips.count = 0;
        }
    }
    return out;
}

void ReporterThread() {
    Common::SetCurrentThreadName("shadPS4:Gr2AvTrace");
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "gr2_av_trace.tsv";
    std::ofstream file{path, std::ios::trunc};
    file << "# times are longest ms@seconds since the trace started\n"
            "# port: ticks, blocks taken, ticks with no guest block, blocks taken under half a "
            "block after the previous one, longest gap between ticks, longest timer sleep, "
            "longest port lock wait, longest output call, longest backend pacing sleep (actual, "
            "then requested), host queue min/avg/max, wall time minus audio output since the "
            "first block\n"
            "# guest: sceAudioOutOutput(s) calls, longest port lock, longest wait for the ports "
            "to free, longest time between returning and the next call\n"
            "# canary: 1 ms sleeps that woke, longest oversleep\n"
            "# flip: frames submitted by the game or presented, longest gap\n"
            "# video: frames taken, player clock minus frame timestamp avg/max, frames taken with "
            "no decoded frame left, last timestamp\n"
            "# ajm: samples per channel each instance decoded\n";
    file.flush();
    auto next = Clock::now();
    for (;;) {
        next += std::chrono::seconds{1};
        std::this_thread::sleep_until(next);
        const auto out = Report(GetState(), Clock::now());
        if (!out.empty()) {
            file << out;
            file.flush();
        }
    }
}

void CanaryThread(Canary& canary, bool critical) {
    Common::SetCurrentThreadName(critical ? "shadPS4:Gr2AvCanaryCritical"
                                          : "shadPS4:Gr2AvCanaryNormal");
    if (critical) {
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    }
    for (;;) {
        const auto begin = Clock::now();
        Common::AccurateSleep(CanarySleep, nullptr, false);
        const auto end = Clock::now();
        std::scoped_lock lock{canary.mutex};
        ++canary.wakes;
        canary.late.Add(Millis(end - begin - CanarySleep), end);
    }
}

void StartTrace() {
    static std::once_flag started;
    std::call_once(started, [] {
        StartTime();
        auto& state = GetState();
        std::thread{ReporterThread}.detach();
        std::thread{CanaryThread, std::ref(state.canaries[0]), false}.detach();
        std::thread{CanaryThread, std::ref(state.canaries[1]), true}.detach();
    });
}

void AddQueue(const void* backend, const char* host, u64 queued_frames) {
    Port* port = FindPort(backend);
    if (port == nullptr) {
        return;
    }
    std::scoped_lock lock{port->mutex};
    if (port->sample_rate == 0) {
        return;
    }
    port->host = host;
    const double queue_ms = 1000.0 * static_cast<double>(queued_frames) / port->sample_rate;
    if (port->queue_samples == 0) {
        port->queue_min_ms = port->queue_max_ms = queue_ms;
    } else {
        port->queue_min_ms = std::min(port->queue_min_ms, queue_ms);
        port->queue_max_ms = std::max(port->queue_max_ms, queue_ms);
    }
    port->queue_sum_ms += queue_ms;
    ++port->queue_samples;
}

// Per guest thread: when its current call started, locked the ports and last returned.
thread_local Clock::time_point tl_guest_begin{};
thread_local Clock::time_point tl_guest_locked{};
thread_local Clock::time_point tl_guest_done{};

} // namespace

void OnTickBegin(const void* backend) {
    StartTrace();
    const auto now = Clock::now();
    Port* port = FindPort(backend);
    if (port == nullptr) {
        return;
    }
    std::scoped_lock lock{port->mutex};
    if (port->tick_end != Clock::time_point{}) {
        port->sleep.Add(Millis(now - port->tick_end), now);
    }
    port->lock_request = now;
}

void OnAudioTick(const void* backend, s32 port_type, u32 sample_rate, u32 block_frames,
                 bool block_ready) {
    if (sample_rate == 0) {
        return;
    }
    const auto now = Clock::now();
    Port* port = FindPort(backend);
    if (port == nullptr) {
        return;
    }
    std::scoped_lock lock{port->mutex};
    port->type = port_type;
    port->sample_rate = sample_rate;
    port->block_ms = 1000.0 * block_frames / sample_rate;
    if (port->lock_request != Clock::time_point{}) {
        port->lock_wait.Add(Millis(now - port->lock_request), now);
    }
    if (port->last_tick != Clock::time_point{}) {
        port->gap.Add(Millis(now - port->last_tick), now);
    }
    port->last_tick = now;
    port->tick_start = now;
    ++port->ticks;
    if (!block_ready) {
        ++port->empty_ticks;
        return;
    }
    ++port->blocks;
    if (port->total_blocks != 0 && Millis(now - port->last_block) > IdleResetMs) {
        port->total_blocks = 0;
    }
    if (port->total_blocks == 0) {
        port->first_block = now;
    } else if (Millis(now - port->last_block) < port->block_ms / 2) {
        ++port->fast_blocks;
    }
    port->last_block = now;
    ++port->total_blocks;
}

void OnTickEnd(const void* backend) {
    const auto now = Clock::now();
    Port* port = FindPort(backend);
    if (port == nullptr) {
        return;
    }
    std::scoped_lock lock{port->mutex};
    if (port->tick_start != Clock::time_point{}) {
        port->output.Add(Millis(now - port->tick_start), now);
    }
    port->tick_end = now;
}

void OnSdlQueue(const void* backend, SDL_AudioStream* stream, u32 bytes_per_frame) {
    if (bytes_per_frame == 0) {
        return;
    }
    const int queued = SDL_GetAudioStreamQueued(stream);
    if (queued >= 0) {
        AddQueue(backend, "sdl", static_cast<u64>(queued) / bytes_per_frame);
    }
}

void OnOpenALQueue(const void* backend, u64 queued_frames) {
    AddQueue(backend, "openal", queued_frames);
}

BackendSleep::BackendSleep(const void* backend_, u64 requested_us_)
    : backend{backend_}, requested_us{requested_us_}, begin{Clock::now()} {}

BackendSleep::~BackendSleep() {
    const auto now = Clock::now();
    Port* port = FindPort(backend);
    if (port == nullptr) {
        return;
    }
    std::scoped_lock lock{port->mutex};
    port->backend_sleep.Add(Millis(now - begin), now, static_cast<double>(requested_us) / 1000.0);
}

void OnGuestBegin() {
    StartTrace();
    const auto now = Clock::now();
    auto& guest = GetState().guest;
    std::scoped_lock lock{guest.mutex};
    if (tl_guest_done != Clock::time_point{}) {
        guest.between.Add(Millis(now - tl_guest_done), now);
    }
    tl_guest_begin = now;
}

void OnGuestLocked() {
    const auto now = Clock::now();
    auto& guest = GetState().guest;
    std::scoped_lock lock{guest.mutex};
    guest.lock.Add(Millis(now - tl_guest_begin), now);
    tl_guest_locked = now;
}

void OnGuestWaited() {
    const auto now = Clock::now();
    auto& guest = GetState().guest;
    std::scoped_lock lock{guest.mutex};
    guest.wait.Add(Millis(now - tl_guest_locked), now);
    ++guest.calls;
    tl_guest_done = now;
}

void OnFlip(FlipStage stage) {
    StartTrace();
    const auto now = Clock::now();
    auto& flips = GetState().flips[stage == FlipStage::Submit ? 0 : 1];
    std::scoped_lock lock{flips.mutex};
    if (flips.last != Clock::time_point{}) {
        flips.gap.Add(Millis(now - flips.last), now);
    }
    flips.last = now;
    ++flips.count;
}

void OnVideoFrame(const void* source, u64 timestamp_ms, u64 clock_ms, u64 frames_left) {
    StartTrace();
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
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
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
    auto& decoder = state.decoders[instance_id];
    decoder.samples +=
        total_samples >= decoder.last_total ? total_samples - decoder.last_total : total_samples;
    decoder.last_total = total_samples;
}

} // namespace Gr2AvTrace
