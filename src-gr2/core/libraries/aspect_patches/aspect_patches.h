// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Runtime aspect-ratio patches: replace the 16:9 constant 1.7777778f (0x3FE38E39) in the loaded
// eboot with the target aspect. Gravity Rush 2 (CUSA00547/03694/04934/04943, v01.11) bakes 45
// sites; Gravity Rush Remastered (CUSA01130/01112/01113/02318) bakes far fewer (the remaster
// derives most projection math from the render resolution). Title-gated by ApplyAspectPatches
// (mirrors ResolutionPatches). Runs after the eboot maps, before any game code.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace Libraries::AspectPatches {

enum class TargetAspect {
    Off,        // No-op; do not patch.
    R16_10,     // 1.6f       (16:10) - bytes CD CC CC 3F
    R21_9,      // 2.3333333f (21:9)  - bytes 55 55 15 40
    R32_9,      // 3.5555556f (32:9)  - bytes 39 8E 63 40
    R21_10,     // 2.1f       (21:10) - bytes 66 66 06 40
    R32_10,     // 3.2f       (32:10) - bytes CD CC 4C 40
    R4_3,       // 1.3333334f (4:3)   - bytes AB AA AA 3F
    R2_39,      // 2.39f      (2.39:1) - bytes C3 F5 18 40. NOT config-selectable: reached
                // only through ResolveTargetAspect's 21:9 window-match compatibility mode.
};

// Outcome of one aspect-patch pass; filled when a non-null stats pointer is passed. The combined
// entry treats (patched + already) > 0 as "this title's constants were present", which drives the
// GR2-to-GRR fallback decision.
struct AspectStats {
    int patched = 0;
    int already = 0;
    int mismatched = 0;
};

// Per-title passes. Idempotent, memcmp-guarded; log a warning per mismatched site. `eboot_base` is
// MemoryPatcher::g_eboot_address. GR2 patches 45 sites; GRR patches the MOVABS camera FOV+aspect
// site plus two rodata aspect constants (the writable-.data occurrence is deliberately skipped).
int ApplyGr2AspectPatches(uintptr_t eboot_base, TargetAspect target, AspectStats* stats = nullptr);
int ApplyGrrAspectPatches(uintptr_t eboot_base, TargetAspect target, AspectStats* stats = nullptr);

// Combined title-gated entry (mirrors ResolutionPatches::ApplyResolutionPatches): tries GR2 first
// unless Config::isGravityRushRemastered() is set, and falls back to GRR when no GR2 site matches.
// Call this from module.cpp. Returns the matched path's patched count.
int ApplyAspectPatches(uintptr_t eboot_base, TargetAspect target);

// Parse a config string ("16:9", "16:10", "21:9", "21:10", "32:9", "32:10", "4:3", or "Off")
// into a TargetAspect. Unknown values default to Off (no patches, native 16:9).
TargetAspect ParseAspectFromConfig(std::string_view s);

// Numeric aspect ratio for a TargetAspect (e.g. R16_10 -> 1.6f). Off/default
// returns 16:9 = 1.7777778f. Used by the presenter to override
// expected_ratio when running in a non-16:9 mode.
float TargetAspectToRatio(TargetAspect t);

// 2.39:1 compatibility mode. "21:9" monitors are rarely 7:3 exactly (2560x1080 = 2.370,
// 3440x1440 = 2.389), so with "21:9" selected in config the EFFECTIVE target is resolved
// against the SDL window's reconciled pixel aspect: nearer 2.39 patches R2_39, nearer
// 7/3 keeps R21_9. The window publishes its size once after SDL setup (before the eboot
// maps); the first resolve latches the choice for the whole session, because the patches
// are baked into the eboot and later window resizes must not change the answer. Non-21:9
// targets pass through unchanged; with no window size published yet (the pre-window
// sizing pass in emulator.cpp) 21:9 stays 21:9 and nothing is latched.
void NoteReconciledWindowSize(int width, int height);
TargetAspect ResolveTargetAspect(TargetAspect parsed);

} // namespace Libraries::AspectPatches
