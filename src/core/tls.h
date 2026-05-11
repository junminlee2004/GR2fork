// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>
#include "common/types.h"
#ifdef _WIN32
#include <malloc.h>
#endif

namespace Xbyak {
class CodeGenerator;
}

namespace Libraries::Fiber {
struct OrbisFiberContext;
}

namespace Core {

union DtvEntry {
    std::size_t counter;
    u8* pointer;
};

struct Tcb {
    Tcb* tcb_self;
    DtvEntry* tcb_dtv;
    void* tcb_thread;
    ::Libraries::Fiber::OrbisFiberContext* tcb_fiber;
};

#ifdef _WIN32
/// Gets the thread local storage key for the TCB block.
u32 GetTcbKey();
#endif

/// Sets the data pointer to the TCB block.
void SetTcbBase(void* image_address);

/// Retrieves Tcb structure for the calling thread.
Tcb* GetTcbBase();

/// Makes sure TLS is initialized for the thread before entering guest.
void EnsureThreadInitialized();

template <size_t size>
#ifdef __clang__
__attribute__((optnone))
#else
__attribute__((optimize("O0")))
#endif
void ClearStack() {
    volatile void* buf = alloca(size);
    memset(const_cast<void*>(buf), 0, size);
    buf = nullptr;
}

template <class ReturnType, class... FuncArgs, class... CallArgs>
ReturnType ExecuteGuest(PS4_SYSV_ABI ReturnType (*func)(FuncArgs...), CallArgs&&... args) {
    EnsureThreadInitialized();

    // MTTW optimization (fork): only clear stack ONCE per thread.
    // The original comment says this is to clear trash from EnsureThreadInitialized,
    // which is only relevant the first time the thread enters guest.
    //
    // PORT(upstream #4033, GR2-win-crash-fix, commit 0f92285): additional guard
    // to skip the clear when a PS4 fiber is active on this thread. In that case
    // the "stack" is the fiber's stack (not the OS thread stack), and 12 KB of
    // alloca() + memset corrupts fiber state on Windows.
    //
    // Composition: we only clear once per thread AND only when no fiber is
    // active. If the first ExecuteGuest on a thread happens inside a fiber,
    // we defer the clear until the next call where the fiber is inactive.
    // Once cleared, never clear again on this thread.
    static thread_local bool cleared_after_init = false;
    if (!cleared_after_init) {
        auto* tcb = GetTcbBase();
        if (tcb != nullptr && tcb->tcb_fiber == nullptr) {
            ClearStack<12_KB>();
            cleared_after_init = true;
        }
    }

    return func(std::forward<CallArgs>(args)...);
}

template <class F, F f>
struct HostCallWrapperImpl;

template <class ReturnType, class... Args, PS4_SYSV_ABI ReturnType (*func)(Args...)>
struct HostCallWrapperImpl<PS4_SYSV_ABI ReturnType (*)(Args...), func> {
    static ReturnType PS4_SYSV_ABI wrap(Args... args) {
        return func(args...);
    }
};

#define HOST_CALL(func) (Core::HostCallWrapperImpl<decltype(&(func)), func>::wrap)

} // namespace Core
