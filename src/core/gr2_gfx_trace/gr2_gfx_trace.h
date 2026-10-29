// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

// A trace of the Sony graphics library that Gravity Rush 2 links into its executable. Every library
// function the game calls is hooked at its entry and counted, with its callers, the threads it runs
// on and the arguments of a few of its calls, and the counts are written to gr2_gfx_trace.tsv in
// the log folder. On with the environment variable GR2_GFX_TRACE=1.
namespace Gr2GfxTrace {

/// Hooks the library functions when module_base is the Gravity Rush 2 01.11 executable. Runs once
/// the executable is loaded and patched, before any of its code runs.
void Install(VAddr module_base);

} // namespace Gr2GfxTrace
