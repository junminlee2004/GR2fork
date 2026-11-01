// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Vulkan {
class Rasterizer;
}

// The draw thread's frame synchronization. The draw thread ends each frame's commands by writing
// the frame number to one of three frame labels, and before it records a frame it waits until
// the label of the frame two before holds its number.
namespace Gr2Native {

struct DrawCommandBuffer;

/// Sets the address of Sony's DrawCommandBuffer::writeAtEndOfPipe, which takes the calls that
/// WriteFrameLabel does not cover.
void SetWriteAtEndOfPipe(VAddr address);

/// Takes the draw thread's DrawCommandBuffer::writeAtEndOfPipe call for a frame label and writes
/// a frame label marker in place of the end-of-pipe event.
void PS4_SYSV_ABI WriteFrameLabel(DrawCommandBuffer* dcb, u32 event_type, u32 dst_select,
                                  u64* label, u32 src_select, u64 value, u32 cache_action,
                                  u32 cache_policy);

/// Takes the draw thread's spin on a frame label: returns once the label holds expected.
void PS4_SYSV_ABI WaitFrameLabel(u64* label, u64 expected);

/// Runs a frame label marker.
void RunFrameLabel(const u32* payload, Vulkan::Rasterizer* rasterizer);

} // namespace Gr2Native
