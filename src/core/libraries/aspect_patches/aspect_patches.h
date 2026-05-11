// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Runtime aspect-ratio patches for Gravity Rush 2 (CUSA00547 / CUSA03694 /
// CUSA04934 / CUSA04943, eboot v01.11).
//
// Replaces all 45 occurrences of the 16:9 float constant 1.7777778f
// (0x3FE38E39, little-endian bytes 39 8E E3 3F) with the chosen target
// aspect ratio, applied directly to the loaded eboot bytes via memcpy
// — equivalent to an HxD pre-patch.
//
// Must be called AFTER the eboot is mapped (g_eboot_address set) but
// BEFORE the game executes any of its own code. The natural hook point
// is core/module.cpp, immediately after MemoryPatcher::OnGameLoaded().

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace Libraries::AspectPatches {

enum class TargetAspect {
    Off,        // No-op; do not patch.
    R16_10,     // 1.6f       (16:10) — bytes CD CC CC 3F
    R21_9,      // 2.3333333f (21:9)  — bytes 55 55 15 40
    R32_9,      // 3.5555556f (32:9)  — bytes 39 8E 63 40
};

// Apply the chosen aspect override to all 45 sites. Returns the number of
// sites patched. Idempotent — safe to call repeatedly (sites already at
// the target value are skipped). Logs a warning per mismatched site (e.g.
// when the eboot version doesn't match v1.11).
//
// `eboot_base` should be MemoryPatcher::g_eboot_address.
int ApplyGr2AspectPatches(uintptr_t eboot_base, TargetAspect target);

// Parse a config string ("16:9", "16:10", "21:9", "32:9", or "Off") into a
// TargetAspect. Unknown values default to Off (no patches, native 16:9).
TargetAspect ParseAspectFromConfig(std::string_view s);

// Numeric aspect ratio for a TargetAspect (e.g. R16_10 -> 1.6f). Off/default
// returns 16:9 = 1.7777778f. Used by the presenter to override
// expected_ratio when running in a non-16:9 mode.
float TargetAspectToRatio(TargetAspect t);

} // namespace Libraries::AspectPatches
