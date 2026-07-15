// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>

#include "common/alignment.h"
#include "common/logging/log.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "thread.h"
#ifdef _WIN64
#include <windows.h>
#include "common/ntapi.h"
#else
#include <csignal>
#include <pthread.h>
#include <unistd.h>
#include <xmmintrin.h>
#endif

namespace Core {

static constexpr u32 ORBIS_MXCSR = 0x9fc0;
static constexpr u32 ORBIS_FPUCW = 0x037f;

#ifdef _WIN64
#define KGDT64_R3_DATA (0x28)
#define KGDT64_R3_CODE (0x30)
#define KGDT64_R3_CMTEB (0x50)
#define RPL_MASK (0x03)
#define EFLAGS_INTERRUPT_MASK (0x200)

void InitializeTeb(INITIAL_TEB* teb, const ::Libraries::Kernel::PthreadAttr* attr) {
    teb->StackBase = (void*)((u64)attr->stackaddr_attr + attr->stacksize_attr);
    teb->StackLimit = nullptr;
    teb->StackAllocationBase = attr->stackaddr_attr;
}

void InitializeContext(CONTEXT* ctx, ThreadFunc func, void* arg,
                       const ::Libraries::Kernel::PthreadAttr* attr) {
    /* Note: The stack has to be reversed */
    ctx->Rsp = (u64)attr->stackaddr_attr + attr->stacksize_attr;
    ctx->Rbp = (u64)attr->stackaddr_attr + attr->stacksize_attr;
    ctx->Rcx = (u64)arg;
    ctx->Rip = (u64)func;

    ctx->SegGs = KGDT64_R3_DATA | RPL_MASK;
    ctx->SegEs = KGDT64_R3_DATA | RPL_MASK;
    ctx->SegDs = KGDT64_R3_DATA | RPL_MASK;
    ctx->SegCs = KGDT64_R3_CODE | RPL_MASK;
    ctx->SegSs = KGDT64_R3_DATA | RPL_MASK;
    ctx->SegFs = KGDT64_R3_CMTEB | RPL_MASK;

    ctx->EFlags = 0x3000 | EFLAGS_INTERRUPT_MASK;

    ctx->ContextFlags =
        CONTEXT_CONTROL | CONTEXT_INTEGER | CONTEXT_SEGMENTS | CONTEXT_FLOATING_POINT;
}

// GR2FORK FIX: AV/EDR hooks and exploit-protection policies deny the custom-TEB NtCreateThread
// call with STATUS_ACCESS_DENIED (it matches process-hollowing heuristics), which kills every
// guest thread and boots the game to a black screen. The fallback creates a plain CreateThread
// thread whose entry updates the TEB stack bounds to the guest stack and pivots RSP onto it, so
// guest code sees the same stack placement as the NtCreateThread path.
struct FallbackLaunch {
    ThreadFunc func;
    void* arg;
    void* stack_addr;
    u64 stack_size;
};

[[noreturn]] static void PivotToGuestStack(ThreadFunc func, void* arg, void* rsp) {
    asm volatile("movq %[rsp], %%rsp\n\t"
                 "xorl %%ebp, %%ebp\n\t"
                 "callq *%[entry]\n\t"
                 "ud2"
                 :
                 : [entry] "r"(func), [rsp] "r"(rsp), "c"(arg)
                 : "memory");
    __builtin_unreachable();
}

static DWORD WINAPI FallbackThreadEntry(LPVOID param) {
    FallbackLaunch launch = *static_cast<FallbackLaunch*>(param);
    delete static_cast<FallbackLaunch*>(param);

    // Exception dispatch validates RSP against the TEB stack bounds, so they must describe the
    // guest stack before the pivot. DeallocationStack is left pointing at the host stack so the
    // kernel frees it at thread termination (see NativeThread::Exit).
    const u64 top =
        Common::AlignDown(reinterpret_cast<u64>(launch.stack_addr) + launch.stack_size, 16);
    auto* teb = reinterpret_cast<TEB*>(NtCurrentTeb());
    teb->Tib.StackBase = reinterpret_cast<void*>(top);
    teb->Tib.StackLimit = launch.stack_addr;

    // Pivot below the top so the entry's ABI shadow space stays inside the guest stack.
    PivotToGuestStack(launch.func, launch.arg, reinterpret_cast<void*>(top - 0x20));
}
#endif

NativeThread::NativeThread() : native_handle{0} {}

NativeThread::~NativeThread() {}

int NativeThread::Create(ThreadFunc func, void* arg, const ::Libraries::Kernel::PthreadAttr* attr) {
#ifndef _WIN64
    pthread_t* pthr = reinterpret_cast<pthread_t*>(&native_handle);
    pthread_attr_t pattr;
    pthread_attr_init(&pattr);
    pthread_attr_setstack(&pattr, attr->stackaddr_attr, attr->stacksize_attr);
    return pthread_create(pthr, &pattr, (PthreadFunc)func, arg);
#else
    CLIENT_ID clientId{};
    INITIAL_TEB teb{};
    CONTEXT ctx{};

    clientId.UniqueProcess = GetCurrentProcess();
    clientId.UniqueThread = GetCurrentThread();

    InitializeTeb(&teb, attr);
    InitializeContext(&ctx, func, arg, attr);

    const u64 status = NtCreateThread(&native_handle, THREAD_ALL_ACCESS, nullptr,
                                      GetCurrentProcess(), &clientId, &ctx, &teb, false);
    if (status == 0) {
        return 0;
    }

    static std::atomic<bool> fallback_logged{false};
    if (!fallback_logged.exchange(true, std::memory_order_relaxed)) {
        LOG_WARNING(Core,
                    "[GR2FORK thread-fallback] NtCreateThread failed (status 0x{:x}); guest "
                    "threads now run via CreateThread with a guest-stack pivot. Security "
                    "software commonly blocks the custom-TEB path.",
                    status);
    }

    host_stack_thread = true;
    auto* launch = new FallbackLaunch{func, arg, attr->stackaddr_attr, attr->stacksize_attr};
    // The host stack only hosts FallbackThreadEntry up to the pivot; 256 KB reservation is ample.
    native_handle = CreateThread(nullptr, 256_KB, FallbackThreadEntry, launch,
                                 STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (native_handle == nullptr) {
        LOG_ERROR(Core, "[GR2FORK thread-fallback] CreateThread fallback also failed (error {}).",
                  GetLastError());
        delete launch;
        return static_cast<int>(status);
    }
    return 0;
#endif
}

void NativeThread::Exit() {
    if (!native_handle) {
        return;
    }

    tid = 0;

#ifdef _WIN64
    NtClose(native_handle);
    native_handle = nullptr;

    // GR2FORK FIX: fallback threads run on a guest stack but own a CreateThread host stack;
    // DeallocationStack still points at the host stack so terminating here frees it. The guest
    // stack is owned and recycled by ThrState either way.
    if (host_stack_thread) {
        NtTerminateThread(nullptr, 0);
    }

    /* The Windows kernel will free the stack
       given at thread creation via INITIAL_TEB
       (StackAllocationBase) upon thread termination.

       In earlier Windows versions (NT4 to Windows Server 2003),
       you could get around this via disabling FreeStackOnTermination
       on the TEB. This has been removed since then.

       To avoid this, we must forcefully set the TEB
       deallocation stack pointer to NULL so ZwFreeVirtualMemory fails
       in the kernel and our stack is not freed.
     */
    auto* teb = reinterpret_cast<TEB*>(NtCurrentTeb());
    teb->DeallocationStack = nullptr;

    NtTerminateThread(nullptr, 0);
#else
    // Disable and free the signal stack.
    constexpr stack_t sig_stack = {
        .ss_flags = SS_DISABLE,
    };
    sigaltstack(&sig_stack, nullptr);

    if (sig_stack_ptr) {
        free(sig_stack_ptr);
        sig_stack_ptr = nullptr;
    }

    pthread_exit(nullptr);
#endif
}

void NativeThread::Initialize() {
    // Set MXCSR and FPUCW registers to the values used by Orbis.
    _mm_setcsr(ORBIS_MXCSR);
    asm volatile("fldcw %0" : : "m"(ORBIS_FPUCW));
#if _WIN64
    tid = GetCurrentThreadId();
#else
    tid = (u64)pthread_self();

    // Set up an alternate signal handler stack to avoid overflowing small thread stacks.
    const size_t page_size = getpagesize();
    const size_t sig_stack_size = Common::AlignUp(std::max<size_t>(64_KB, MINSIGSTKSZ), page_size);
    ASSERT_MSG(posix_memalign(&sig_stack_ptr, page_size, sig_stack_size) == 0,
               "Failed to allocate signal stack: {}", errno);

    stack_t sig_stack;
    sig_stack.ss_sp = sig_stack_ptr;
    sig_stack.ss_size = sig_stack_size;
    sig_stack.ss_flags = 0;
    ASSERT_MSG(sigaltstack(&sig_stack, nullptr) == 0, "Failed to set signal stack: {}", errno);
#endif
}

} // namespace Core
