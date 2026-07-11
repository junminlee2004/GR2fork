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
        // GR2FORK FIX: stop+join any running thread before overwriting the pthread handle and
        // function storage - an unconditional pthread_create orphans the old pthread, races it on
        // this->func, and reuses the stop_source (avplayer_source.cpp double-Start() shape).
        if (Joinable()) {
            Stop();
        }
        this->func = std::move(func);
        PthreadAttrT attr{};
        posix_pthread_attr_init(&attr);
        // GR2FORK FIX: 8 MiB stacks for these emulator-internal worker threads. The 1 MiB guest
        // default overflows when host HLE re-enters guest code (AvPlayer callbacks in GR2's opening
        // movie ride ExecuteGuest's 12 KB alloca), smashing live fibers; the process fastfails.
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
