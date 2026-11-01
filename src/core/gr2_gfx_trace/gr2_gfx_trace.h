// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

// A trace of the Sony graphics library that Gravity Rush 2 links into its executable. Every library
// function the game calls is hooked at its entry and counted, with its callers, the threads it runs
// on and the arguments of a few of its calls. The log folder gets two files:
// - gr2_gfx_cpu.tsv: every second, the game and presented frames, the emulator's draws per frame,
//   the library calls, the CPU time of every thread with the part of it spent in the hooks, and
//   on NVIDIA GPUs the GPU utilization and clock.
// - gr2_gfx_trace.tsv: the call counts, call sites and argument samples of the whole run.
namespace Gr2GfxTrace {

/// Hooks the library functions when module_base is the Gravity Rush 2 01.11 executable. Runs once
/// the executable is loaded and patched, before any of its code runs.
void Install(VAddr module_base);

} // namespace Gr2GfxTrace
