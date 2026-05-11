// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/frame_time_recorder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <vector>

#include "common/logging/log.h"
#include "common/path_util.h"

namespace Common::FrameTime {

namespace {

// 4096 x 8B = 32 KiB BSS. ~68s @ 60fps, ~136s @ 30fps of retained history.
constexpr std::size_t kRingCapacity = 4096;
static_assert((kRingCapacity & (kRingCapacity - 1)) == 0,
              "kRingCapacity must be a power of two");
constexpr std::size_t kRingMask = kRingCapacity - 1;

// Default CSV flush cadence. ~2s @ 60fps; bounds SIGKILL data loss.
constexpr u64 kDefaultFlushEvery = 120;

// Default CSV filename, joined onto <LogDir> unless GR2_FPS_OUT overrides.
constexpr const char* kDefaultCsvName = "frame_times.csv";

struct State {
    // ----- Hot-path data (writer = videoout vblank thread, single producer)
    std::array<u64, kRingCapacity> ring{};
    std::atomic<u64> head{0};
    std::chrono::steady_clock::time_point prev_tp{};
    bool have_prev{false};

    // Streaming CSV. Opened lazily on first RecordFlip; never closed
    // explicitly (process-exit closes it). nullptr if open failed.
    std::FILE* csv_file{nullptr};
    u64 flush_every{kDefaultFlushEvery};
    u64 next_flush_at{kDefaultFlushEvery};

    // Periodic-log ladder.
    u64 next_log_at{0};
    u64 log_every{0};

    // ----- Init-once
    std::once_flag init_once;
    bool disabled{false};

    // ----- Snapshot lock — keeps Snapshot()/DumpCsv() callers consistent.
    // Not held on the writer's hot path except by the periodic logger.
    std::mutex snap_mu;

    // Reusable scratch for percentile compute under snap_mu.
    std::vector<u64> scratch;
};

State& Get() {
    static State s;
    return s;
}

std::filesystem::path ResolveCsvPath(const char* env_value) {
    namespace fs = std::filesystem;
    const fs::path log_dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
    if (env_value == nullptr || env_value[0] == '\0') {
        return log_dir / kDefaultCsvName;
    }
    fs::path p{env_value};
    if (p.is_absolute()) {
        return p;
    }
    return log_dir / p;
}

void EnsureInit() {
    auto& s = Get();
    std::call_once(s.init_once, [&] {
        if (const char* v = std::getenv("GR2_FPS_DISABLE"); v && v[0] == '1') {
            s.disabled = true;
        }
        if (const char* v = std::getenv("GR2_FPS_LOG_EVERY"); v) {
            const long n = std::strtol(v, nullptr, 10);
            if (n > 0) {
                s.log_every = static_cast<u64>(n);
                s.next_log_at = s.log_every;
            }
        }
        if (const char* v = std::getenv("GR2_FPS_FLUSH_EVERY"); v) {
            const long n = std::strtol(v, nullptr, 10);
            if (n > 0) {
                s.flush_every = static_cast<u64>(n);
            }
        }
        s.next_flush_at = s.flush_every;
        s.scratch.reserve(kRingCapacity);

        if (!s.disabled) {
            const char* env_out = std::getenv("GR2_FPS_OUT");
            const std::filesystem::path csv_path = ResolveCsvPath(env_out);
            // Ensure parent dir exists (LogDir already does, but a custom
            // GR2_FPS_OUT may point elsewhere).
            std::error_code ec;
            std::filesystem::create_directories(csv_path.parent_path(), ec);
            s.csv_file = std::fopen(csv_path.string().c_str(), "w");
            if (s.csv_file != nullptr) {
                std::fprintf(s.csv_file, "frame_index,delta_ns\n");
                std::fflush(s.csv_file);
                LOG_INFO(Render_Vulkan,
                         "FrameTime recorder streaming to {} (flush_every={}, log_every={})",
                         csv_path.string(), s.flush_every, s.log_every);
            } else {
                LOG_WARNING(Render_Vulkan,
                            "FrameTime recorder: failed to open {} (errno={}). "
                            "In-memory ring still active; CSV stream disabled.",
                            csv_path.string(), errno);
            }
        } else {
            LOG_INFO(Render_Vulkan, "FrameTime recorder disabled via GR2_FPS_DISABLE.");
        }
    });
}

// Compute stats over the most recent `window` entries. Caller holds snap_mu.
Stats ComputeStatsLocked(std::size_t window) {
    auto& s = Get();
    Stats out{};
    const u64 head = s.head.load(std::memory_order_acquire);
    const std::size_t total = static_cast<std::size_t>(std::min<u64>(head, kRingCapacity));
    if (total == 0) {
        return out;
    }
    if (window == 0 || window > total) {
        window = total;
    }

    s.scratch.clear();
    s.scratch.resize(window);
    const u64 first = head - window;
    for (std::size_t i = 0; i < window; ++i) {
        s.scratch[i] = s.ring[(first + i) & kRingMask];
    }

    out.count = window;
    {
        u64 mn = s.scratch[0];
        u64 mx = s.scratch[0];
        long double sum = 0.0L;
        for (u64 v : s.scratch) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += static_cast<long double>(v);
        }
        out.min_ns = mn;
        out.max_ns = mx;
        out.mean_ns = static_cast<double>(sum / static_cast<long double>(window));
    }
    auto pct = [&](double p) -> double {
        const std::size_t k =
            std::min<std::size_t>(window - 1, static_cast<std::size_t>(p * (window - 1)));
        std::nth_element(s.scratch.begin(), s.scratch.begin() + k, s.scratch.end());
        return static_cast<double>(s.scratch[k]);
    };
    out.p50_ns = pct(0.50);
    out.p95_ns = pct(0.95);
    out.p99_ns = pct(0.99);
    out.fps_p50 = (out.p50_ns > 0.0) ? (1e9 / out.p50_ns) : 0.0;
    return out;
}

} // namespace

void RecordFlip() {
    EnsureInit();
    auto& s = Get();
    if (s.disabled) [[unlikely]] {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!s.have_prev) [[unlikely]] {
        s.prev_tp = now;
        s.have_prev = true;
        return;
    }
    const u64 dt_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - s.prev_tp).count());
    s.prev_tp = now;

    const u64 head = s.head.load(std::memory_order_relaxed);
    s.ring[head & kRingMask] = dt_ns;
    const u64 new_head = head + 1;
    s.head.store(new_head, std::memory_order_release);

    // Stream this row to the on-disk CSV. fprintf into a buffered FILE* is
    // ~few hundred ns; the periodic fflush is the only syscall path.
    if (s.csv_file != nullptr) [[likely]] {
        std::fprintf(s.csv_file, "%" PRIu64 ",%" PRIu64 "\n", new_head, dt_ns);
        if (new_head >= s.next_flush_at) {
            s.next_flush_at = new_head + s.flush_every;
            std::fflush(s.csv_file);
        }
    }

    if (s.log_every != 0 && new_head >= s.next_log_at) [[unlikely]] {
        s.next_log_at = new_head + s.log_every;
        std::lock_guard lk{s.snap_mu};
        Stats st = ComputeStatsLocked(s.log_every);
        LOG_INFO(Render_Vulkan,
                 "[fps] window={} p50={:.3f}ms p95={:.3f}ms p99={:.3f}ms "
                 "min={:.3f}ms max={:.3f}ms mean={:.3f}ms fps_p50={:.2f}",
                 st.count, st.p50_ns / 1e6, st.p95_ns / 1e6, st.p99_ns / 1e6,
                 static_cast<double>(st.min_ns) / 1e6,
                 static_cast<double>(st.max_ns) / 1e6, st.mean_ns / 1e6, st.fps_p50);
    }
}

Stats Snapshot(std::size_t window) {
    EnsureInit();
    auto& s = Get();
    std::lock_guard lk{s.snap_mu};
    return ComputeStatsLocked(window);
}

bool DumpCsv(const std::string& path) {
    EnsureInit();
    auto& s = Get();
    std::lock_guard lk{s.snap_mu};
    const u64 head = s.head.load(std::memory_order_acquire);
    const std::size_t total = static_cast<std::size_t>(std::min<u64>(head, kRingCapacity));

    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        LOG_WARNING(Render_Vulkan, "FrameTime DumpCsv: failed to open {}", path);
        return false;
    }
    std::fprintf(f, "frame_index,delta_ns\n");
    if (total > 0) {
        const u64 first = head - total;
        for (std::size_t i = 0; i < total; ++i) {
            const u64 idx = (first + i) & kRingMask;
            std::fprintf(f, "%" PRIu64 ",%" PRIu64 "\n",
                         static_cast<u64>(first + i), s.ring[idx]);
        }
    }
    std::fclose(f);

    Stats st = ComputeStatsLocked(0);
    LOG_INFO(Render_Vulkan,
             "FrameTime DumpCsv: wrote {} ({} samples) p50={:.3f}ms p95={:.3f}ms "
             "p99={:.3f}ms fps_p50={:.2f}",
             path, st.count, st.p50_ns / 1e6, st.p95_ns / 1e6, st.p99_ns / 1e6, st.fps_p50);
    return true;
}

void Reset() {
    EnsureInit();
    auto& s = Get();
    std::lock_guard lk{s.snap_mu};
    s.head.store(0, std::memory_order_release);
    s.have_prev = false;
    s.next_log_at = s.log_every;
    s.next_flush_at = s.flush_every;
    if (s.csv_file != nullptr) {
        std::fprintf(s.csv_file, "# reset\n");
        std::fflush(s.csv_file);
    }
}

bool RecordingEnabled() {
    EnsureInit();
    return !Get().disabled;
}

} // namespace Common::FrameTime
