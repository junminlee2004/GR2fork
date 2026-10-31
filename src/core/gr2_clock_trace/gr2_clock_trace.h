// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

// Frame clock trace for the Gravity Rush 2 fps patches. Writes user/log/gr2_clock_trace.tsv with
// one row per game frame (the step the patch's time service chose, read from guest memory) and
// one row per presented flip (host time and vblank count).
namespace Gr2ClockTrace {

// Called by the video out driver for every flip it presents.
void OnPresent(u64 vblank_count);

} // namespace Gr2ClockTrace
