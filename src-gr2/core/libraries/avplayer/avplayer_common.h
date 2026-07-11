// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <queue>

#include "core/libraries/avplayer/avplayer.h"

#define AVPLAYER_IS_ERROR(x) ((x) < 0)

namespace Libraries::AvPlayer {

enum class AvState {
    Unknown,
    Initial,
    AddingSource,
    Ready,
    Play,
    Stop,
    EndOfFile,
    Pause,
    C0x08,
    Jump,
    TrickMode,
    C0x0B,
    Buffering,
    Starting,
    Error,
};

enum class AvEventType {
    ChangeFlowState = 21,
    WarningId = 22,
    RevertState = 30,
    AddSource = 40,
    Error = 255,
};

union AvPlayerEventData {
    u32 num_frames; // 20
    AvState state;  // AvEventType::ChangeFlowState
    s32 error;      // AvEventType::WarningId
    u32 attempt;    // AvEventType::AddSource
};

struct AvPlayerEvent {
    AvEventType event;
    AvPlayerEventData payload;
};

template <class T>
class AvPlayerQueue {
public:
    // GR2FORK FIX: lock the size check. An unlocked size-then-pop race lets two consumers both
    // enter the pop branch and the loser call front() on an empty queue (UB); this kills the
    // process during avplayer loop reset, where the demuxer and game thread share m_video_frames.
    size_t Size() {
        std::lock_guard guard(m_mutex);
        return m_queue.size();
    }

    void Push(T&& value) {
        std::lock_guard guard(m_mutex);
        m_queue.emplace(std::forward<T>(value));
    }

    // GR2FORK: the returned reference is only valid while no other thread can Pop/Push/Clear
    // this queue; multi-producer/multi-consumer callers use TryPeek(), which copies under the
    // mutex.
    T& Front() {
        return m_queue.front();
    }

    // GR2FORK FIX: invokes `acc(front)` under the queue mutex so the caller can copy fields out
    // without a dangling reference. Returns false if the queue was empty (acc not invoked).
    template <class Acc>
    bool TryPeek(Acc acc) {
        std::lock_guard guard(m_mutex);
        if (m_queue.empty()) {
            return false;
        }
        acc(m_queue.front());
        return true;
    }

    // GR2FORK FIX: empty check + front + pop all under the mutex so concurrent consumers cannot
    // both enter the pop path and have the loser pop an empty queue; returns nullopt when empty.
    std::optional<T> Pop() {
        std::lock_guard guard(m_mutex);
        if (m_queue.empty()) {
            return std::nullopt;
        }
        auto result = std::move(m_queue.front());
        m_queue.pop();
        return result;
    }

    void Clear() {
        std::lock_guard guard(m_mutex);
        m_queue = {};
    }

private:
    std::mutex m_mutex{};
    std::queue<T> m_queue{};
};

AvPlayerSourceType GetSourceType(std::string_view path);

} // namespace Libraries::AvPlayer
