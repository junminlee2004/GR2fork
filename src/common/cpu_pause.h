// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include "common/arch.h"
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(ARCH_X86_64)
#include <emmintrin.h>
#endif

namespace Common {

// One spin-wait step: a pause on x86, a yield on ARM, nothing elsewhere.
inline void CpuPause() {
#if defined(ARCH_X86_64)
    _mm_pause();
#elif defined(ARCH_ARM64) && defined(_MSC_VER)
    __yield();
#elif defined(ARCH_ARM64)
    asm("yield");
#endif
}

// Makes every prior store of this thread globally visible, including stores
// to write-combined memory: on x86 a partial WC line otherwise stays in the
// core's WC buffer until a fence, a locked instruction or an interrupt, and a
// plain release store flushes nothing.
inline void StoreFence() {
#if defined(ARCH_X86_64)
    _mm_sfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

} // namespace Common
