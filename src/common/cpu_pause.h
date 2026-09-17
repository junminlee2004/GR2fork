// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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

} // namespace Common
