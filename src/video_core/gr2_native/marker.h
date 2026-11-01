// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Gr2Native {

// "GR2N", the first payload word of a marker NOP. The NOP payloads the command processor reads
// start with 0x6875xxxx (PM4CmdNop::PayloadType).
constexpr u32 MarkerMagic = 0x4e325247;

enum class MarkerKind : u32 {
    // Writes a 64-bit value to a frame label once the GPU has finished the work before it.
    // Payload: magic, kind, label address low and high words, value low and high words.
    FrameLabel = 1,
};

constexpr u32 FrameLabelPayloadDwords = 6;

// The header of a PM4 type 3 NOP with the given payload length.
constexpr u32 NopHeader(u32 payload_dwords) {
    return 0xc0001000u | ((payload_dwords - 1) << 16);
}

} // namespace Gr2Native
