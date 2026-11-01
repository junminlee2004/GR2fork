// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Vulkan {
class Rasterizer;
}

// A native graphics backend for Gravity Rush 2 01.11, in place of the PM4 command stream that
// Sony's graphics library, linked into the executable, builds for the game. The game's calls are
// replaced one family at a time. Native work is ordered against the rest of the command stream
// by marker packets: PM4 NOPs whose payload starts with MarkerMagic, run by OnNop when the
// command processor reaches them.
//
// It replaces the draw thread's frame synchronization: each frame label is written once the GPU
// has finished that frame, and the draw thread waits for a label on a condition variable.
namespace Gr2Native {

/// Replaces the game's calls when module_base is the Gravity Rush 2 01.11 executable. Runs once
/// the executable is loaded and patched, before any of its code runs.
void Install(VAddr module_base);

/// Runs a marker packet, on the thread that processes the graphics queue and in command stream
/// order. Ignores NOPs that are not markers. dwords is the length of the NOP's payload.
void OnNop(const u32* payload, u32 dwords, Vulkan::Rasterizer* rasterizer);

} // namespace Gr2Native
