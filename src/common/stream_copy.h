// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include "common/types.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Common {

/// Copies from write-combined memory, such as a host mapping of video memory, with streaming
/// loads that fetch whole cache lines; plain memcpy where the build has no AVX2.
inline void StreamCopy(void* dst, const void* src, size_t size) {
#if defined(__AVX2__)
    auto* d = static_cast<u8*>(dst);
    const auto* s = static_cast<const u8*>(src);
    // Streaming loads need a 32-byte aligned source.
    const size_t head = std::min<size_t>(size, (-reinterpret_cast<uintptr_t>(s)) & 31);
    std::memcpy(d, s, head);
    d += head;
    s += head;
    size -= head;
    for (; size >= 32; size -= 32, d += 32, s += 32) {
        const __m256i line = _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(s));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d), line);
    }
    std::memcpy(d, s, size);
#else
    std::memcpy(dst, src, size);
#endif
}

} // namespace Common
