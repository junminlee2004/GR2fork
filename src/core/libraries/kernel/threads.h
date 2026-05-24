// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include "common/polyfill_thread.h"
#include "common/types.h"
#include "core/libraries/kernel/threads/pthread.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::Kernel {

int PS4_SYSV_ABI posix_pthread_attr_init(PthreadAttrT* attr);

int PS4_SYSV_ABI posix_pthread_attr_setstacksize(PthreadAttrT* attr, size_t stacksize);

int PS4_SYSV_ABI posix_pthread_attr_destroy(PthreadAttrT* attr);

int PS4_SYSV_ABI posix_pthread_attr_getaffinity_np(const PthreadAttrT* pattr, size_t cpusetsize,
                                                   Cpuset* cpusetp);

int PS4_SYSV_ABI posix_pthread_attr_setaffinity_np(PthreadAttrT* pattr, size_t cpusetsize,
                                                   const Cpuset* cpusetp);

int PS4_SYSV_ABI posix_pthread_create(PthreadT* thread, const PthreadAttrT* attr,
                                      PthreadEntryFunc start_routine, void* arg);

int PS4_SYSV_ABI posix_pthread_join(PthreadT pthread, void** thread_return);

void RegisterThreads(Core::Loader::SymbolsResolver* sym);

class Thread {
public:
    explicit Thread() = default;
    ~Thread() {
        Stop();
    }

    void Run(std::function<void(std::stop_token)>&& func) {
        // FIX(GR2FORK v4): if a thread is already running against this
        // Thread object, stop+join it BEFORE overwriting the pthread
        // handle and function storage. Previously Run() unconditionally
        // called pthread_create, which:
        //   1. orphaned the old pthread (handle overwritten, never joined)
        //   2. silently overwrote this->func while the old thread might
        //      still be about to read it from RunWrapper — a data race
        //      on std::function whose deferred cost is random vtable /
        //      pure-call dispatches in the orphan
        //   3. reused the same stop_source, which was not reset, so
        //      neither the old nor the new thread got a clean token
        // Visible trigger: avplayer_source.cpp calls Run() on its
        // demuxer/decoder Thread members inside Start(); if the game
        // double-Start()s a source without an intervening Stop() (log
        // evidence: two StatePlay events + two triplets of decoder-
        // thread-started logs in rapid succession), we leaked three
        // orphan threads every time.
        if (Joinable()) {
            Stop();
        }
        this->func = std::move(func);
        PthreadAttrT attr{};
        posix_pthread_attr_init(&attr);
        // FIX(GR2FORK): widen the stack for these emulator-internal worker
        // threads. The default attr uses ThrStackDefault (1 MiB, the guest
        // default), but every thread launched through this wrapper runs HOST
        // C++ HLE on that same stack — and several of them (the AvPlayer
        // controller/demuxer/decoder threads in particular) re-enter GUEST
        // code via Core::ExecuteGuest for event callbacks. GR2's opening-movie
        // AvPlayer callback drives a deep guest chain (video manager -> job
        // system -> sceFiberSwitch) on top of ExecuteGuest's own 12 KB
        // ClearStack alloca and the heavier-than-native host trampoline
        // frames. On 1 MiB that overflows the controller thread's stack: RSP
        // runs past the live fiber structures and smashes them (observed as a
        // corrupt current_fiber with a garbage name and an impossible
        // size_context=0x90 in _sceFiberCheckStackOverflow), then the process
        // dies via fastfail with no clean crash line — the log just ends. The
        // canary handler only reports this after the fact; it cannot prevent
        // the overflow. 8 MiB matches a typical host thread stack and gives the
        // HLE+guest-callback chain ample room.
        //
        // This is a one-time, per-thread reservation made at creation (the
        // stack allocator mmaps it; pages commit on demand), so it costs
        // nothing per frame and does not affect FPS.
        static constexpr size_t kInternalThreadStackSize = 8_MB;
        posix_pthread_attr_setstacksize(&attr, kInternalThreadStackSize);
        posix_pthread_create(&thread, &attr, HOST_CALL(RunWrapper), this);
        posix_pthread_attr_destroy(&attr);
    }

    void Join() {
        if (thread) {
            posix_pthread_join(thread, nullptr);
            thread = nullptr;
        }
    }

    bool Joinable() const {
        return thread != nullptr;
    }

    void Stop() {
        if (Joinable()) {
            stop.request_stop();
            Join();
        }
        thread = nullptr;
        func = nullptr;
        stop = std::stop_source{};
    }

    static void* PS4_SYSV_ABI RunWrapper(void* arg) {
        Thread* thr = (Thread*)arg;
        thr->func(thr->stop.get_token());
        return nullptr;
    }

private:
    PthreadT thread{};
    std::function<void(std::stop_token)> func;
    std::stop_source stop;
};

} // namespace Libraries::Kernel
