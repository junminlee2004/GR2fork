// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Runtime resolution patches for Gravity Rush 2 (CUSA00547/03694/04934/04943, eboot v01.11).
// Rewrites the eboot's hard-coded 1920x1080 render-target constants in place to the target
// resolution composed with the aspect ratio; the 1.7777f projection constant is aspect_patches'
// job. The 960/540 cluster at 0x14ce638..0x14d065c is level data, not RT dims - never patch it.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "common/types.h"

namespace Libraries::ResolutionPatches {

// Ordered by ascending pixel density; ApplyGr2ResolutionPatches gates M1 by ordinal
// comparison against R2160p, so new tiers must keep the order.
enum class TargetResolution {
    Off,    // No-op; do not patch resolution constants at all.
    R270p,  // Base 480x270
    R360p,  // Base 640x360
    R480p,  // Base 856x480 (16:9 at 480 rows is 853.33 wide; 856 is the nearest pitch-safe width)
    R540p,  // Base 960x540
    R720p,  // Base 1280x720
    R900p,  // Base 1600x900
    R1080p, // Base 1920x1080  (native - patches are mostly idempotent at 16:9)
    R1440p, // Base 2560x1440
    R2160p, // Base 3840x2160  (4K)
    R2880p, // Base 5120x2880  (5K)
    R3456p, // Base 6144x3456  (6K)
    R4032p, // Base 7168x4032  (7K)
    R4320p, // Base 7680x4320  (8K)
};

struct Resolution {
    int width;
    int height;
};

// Patch groups, toggled via the `groups` config string passed to ApplyGr2ResolutionPatches; see
// resolution_patches.cpp for the per-group classification. Bitmask values are stable and may be
// persisted or composed.
enum class PatchGroupBit : std::uint32_t {
    A1 = 1u << 0,   // float (W,H) struct init in fn @ 0x044c093  - UNCERTAIN
    A2 = 1u << 1,   // float (W,H) struct init in fn @ 0x0fdcb86  - pre-render setup?
    A3 = 1u << 2,   // float (W,H) struct init in fn @ 0x0fdcb86  - pre-render setup?
    A4 = 1u << 3,   // float (W,H) struct init in fn @ 0x110adfd  - secondary RT?
                    // In `recommended`. Corrupts UI text positioning only when patched without
                    // its B1/D1/H1/H2 sibling groups, not in the full recommended set.
    B1 = 1u << 4,   // u32 (W,H) struct init  in fn @ 0x102a5b0   - MAIN RENDER (highest confidence)
    C1 = 1u << 5,   // (W,H,1000.01,H) vec  in fn @ 0x0136d74d    - UNCERTAIN
    C2 = 1u << 6,   // (W,H,1000.01,H) vec  in fn @ 0x0110c110    - secondary RT?
    C3 = 1u << 7,   // (W,H,1000.01,H) vec  in fn @ 0x0110d140    - secondary RT?
    D1 = 1u << 8,   // (W,H,1/W,1/H) vec    in fn @ 0x102a5b0     - MAIN RENDER (paired with B1)
    E1 = 1u << 9,   // (1/W,1/H) reciprocals in fn @ 0x0fe6090    - paired with A2/A3?
                    // UI-canvas corruptor, excluded from `recommended`: the re-normalized
                    // reciprocals mismatch the UI viewport submission, which stays 1920x1080,
                    // distorting UI element sizes. Enumerable for consumers that want it.
    F1 = 1u << 10,  // (halfW,halfH,W,H) tuple in fn @ 0x05fb908  - sub-camera projection (UI?)
    F2 = 1u << 11,  // (halfW,halfH,W,H) tuple in fn @ 0x09236a0  - sub-camera projection (UI?)
    F3 = 1u << 12,  // (halfW,halfH,W,H) tuple in fn @ 0x0c46bf0  - sub-camera projection (UI?)
    G1 = 1u << 13,  // ecx=W,r8d=H args to fn 0xfdf9d0 (per-entity unproject) - 145+ call sites
                    // UI-canvas corruptor, excluded from `recommended`: UI entities compute
                    // unproject math against the target-res screen but render through a viewport
                    // still sized 1920x1080, so UI element sizes come out wrong. Enumerable for
                    // callers that need target-res 3D unproject and accept the UI cost.
    H1 = 1u << 14,  // companion main RT (W,H) pair in fn 0x102a5b0 - paired with B1
    H2 = 1u << 15,  // quarter-res sub-RT inits passed to fn 0x1218370 (sid=3,6) - 8 pairs
    H3 = 1u << 16,  // 4K-selector 1080p-default-branch (W,H) pair - OFF by default (inert)
    H4 = 1u << 17,  // IMUL-imm32 dim pairs at 4K-toggle entry fn - OFF by default
    I1 = 1u << 18,  // UI projection lock - replaces the only 2 (1/W,-1/H,1/W,1/H)
                    // float-projection readers of B1's struct slot with literal 1920/1080 loads.
                    // Does not fix UI element scaling (possibly a postprocess-texel-size path
                    // rather than UI projection); excluded from `recommended`.
    I2 = 1u << 19,  // UI ortho projection scale - patches the rodata pair
                    // (2/1920, 2/1080) at 0x14C565C/0x14C5664. EMPIRICALLY
                    // CONFIRMED USELESS - no visible effect at any res.
                    // fn 0x10d3b60 is dead/niche. Excluded from `recommended`.
    I3 = 1u << 20,  // UI VIEWPORT SCALE - patches (960, 540) at 0x01499EC0
                    // and 0x01499EC4, the X/Y scale slots of the viewport descriptor passed to
                    // fn 0x110C110 (the C2 UI/text viewport-set function). Sole consumer is the
                    // vmovaps at 0x00EABC8D; no visible UI effect. Excluded from `recommended`.
    I4 = 1u << 21,  // Candidate-A UI viewport SCALE row. Patches the
                    // (960.0f, 540.0f) pair at rodata 0x0149A190/0x0149A194, consumed by 8+
                    // vmovaps loads within one function (0xEBD8ED..0xEBDE9A). Does not fix UI
                    // canvas scaling; excluded from `recommended`, enumerable for re-checks.
    J1 = 1u << 22,  // Flip-Y mirror vecs paired with C1/C2/C3. Three
                    // (1920, -1080, -999.99, -1080) vecs at C_VA - 0x20 (0x014985b0, 0x014a0f30,
                    // 0x014a0fe0), component-multiplied with the C vecs to feed per-character
                    // text positioning, not canvas sizing. No UI-canvas effect; excluded from
                    // `recommended`, enumerable for future C/J text-glyph work.
    K1 = 1u << 23,  // SceneParameter.exposureIntensity = 2.0f sites
                    // scaled to 2.0 * min(1.0, (H/1080)^2), the pixel-count ratio - compensates
                    // low-res over-exposure (worst at 540p; linear H/1080 is not enough). Sites
                    // 0x00b3f625/0x004c7d16/0x008abd31, imm32 at instr_va+3; field +0x50 per the
                    // Lua registrar at 0x01056d9e. Internal-only; excluded from `recommended`.
    P3 = 1u << 30,  // **Screen-edge diamond vec4 array at 0x01488a00.**
                    // A 6xvec4 rodata block read by fn ~0x00bc4b00 - a diamond of edge midpoints
                    // inscribed in the 1080p framebuffer, used as a screen-space sample pattern.
                    // Patches the six resolution floats (0x01488a20/24/34/40/50/54) to target
                    // W, H/2, H/2, W/2, W/2, H; basis/padding untouched. Idempotent at 1080p;
                    // in `recommended`.
    P2 = 1u << 29,  // **BSS_FLAG-indexed (1080p, 4K) mode table at
                    // 0x014c7e60.** Two copies of (W, W4K, H, H4K) indexed by the M1 flag byte
                    // at 0x01b586c0 (readers at 0x00feeb8e and 0x00fef0dd). Patching all 8 floats
                    // to target dims makes both readers flag-independent. In `recommended`.
    P1 = 1u << 28,  // **Viewport-center / half-dimension rodata
                    // floats - the comic-scene "offset images" fix.** 150 floats across 68
                    // rodata clusters hold the (960, 540) screen-center constants of NDC-to-
                    // screen transforms (confirmed at cluster 0x014c2a48, loader 0x012f4798);
                    // scaled to (target_W/2, target_H/2), per-draw scales untouched. Idempotent
                    // at 1080p; in `recommended`, opt out via "recommended,~p1".
    O1 = 1u << 27,  // **Swap-chain buffer ALLOCATION size fix.** Pairs
                    // with N1: N1 makes videoout see 4K, but the game sizes each buffer for
                    // 1080p (8 MB), so a 4K RefreshImage reads past the end (driver AV).
                    // Rewrites the four imm16 W/H slots (0x00446825/29/2d/31, imm16 at
                    // instr_va+2) feeding the allocator at 0x1215570 so both cmova branches
                    // yield target dims. Either patch alone is broken; idempotent at 1080p;
                    // in `recommended`.
    N1 = 1u << 26,  // **Real sceVideoOutSetBufferAttribute caller patch.**
                    // The real caller is at 0x00446975; its pitch and packed W|H come from BSS
                    // slots 0x01b31318/0x01b31350 written indirectly, so value patches would be
                    // clobbered. The BSS loads at 0x00446933/0x00446939 become same-length
                    // immediates: (target_W - 8) / 8 and target_W | (target_H << 16).
                    // Idempotent at 1080p; in `recommended`.
    M1 = 1u << 25,  // **PS4-Pro 4K-mode master enable.** GR2 has a
                    // complete 4K path gated by the BSS byte at 0x01b586c0; under base-PS4
                    // emulation the seta at 0x00446852 stores 0, so all 18 readers stay 1080p.
                    // Forces the flag writers 0x00446852/0x004466dd to 1, NOPs the flag-shift
                    // pairs at 0x0102b9a5/0x0102ba8e (they would re-double B1's dims to 7680),
                    // and writes the BSS byte at apply time. Gated to targets >= 2160p (the 4K
                    // branch is hard-coded 3840x2160); opt-in via "recommended,m1".
    Q1 = 1u << 31,  // **Negative viewport-center half-dims - the
                    // sign-complement of P1.** 7 rodata floats (-960.0f/-540.0f), additive
                    // screen-center offsets in fns ~0x135d000/0x136e6e5/0x1364d1c/~0x136494c,
                    // unreachable by P1's positive-only encoders; scaled to (-target_W/2,
                    // -target_H/2), idempotent at 1080p. Hardware-unverified: excluded from both
                    // kGroupMaskAll and kGroupMaskRecommended, opt in via "recommended,q1".
};

// Preset masks, usable by name in the config string ("safe", "geom", "ext", "all") or composed
// per-group ("B1,D1,H1,H2,..."). H1 always pairs with B1 (same function, same call) and H2 fixes
// the quarter-res sub-RTs that otherwise stay 480x270, so both belong in `safe`; D1's
// (W,H,1/W,1/H) vec must travel with B1 - D1 alone changes the frustum without the RT
// (rainbow corruption).
inline constexpr std::uint32_t kGroupMaskSafe =
    static_cast<std::uint32_t>(PatchGroupBit::B1) |
    static_cast<std::uint32_t>(PatchGroupBit::D1) |
    static_cast<std::uint32_t>(PatchGroupBit::H1) |
    static_cast<std::uint32_t>(PatchGroupBit::H2);

inline constexpr std::uint32_t kGroupMaskGeom =
    kGroupMaskSafe |
    static_cast<std::uint32_t>(PatchGroupBit::A4) |
    static_cast<std::uint32_t>(PatchGroupBit::C2) |
    static_cast<std::uint32_t>(PatchGroupBit::C3);

inline constexpr std::uint32_t kGroupMaskExt =
    kGroupMaskGeom |
    static_cast<std::uint32_t>(PatchGroupBit::A2) |
    static_cast<std::uint32_t>(PatchGroupBit::A3) |
    static_cast<std::uint32_t>(PatchGroupBit::E1) |
    static_cast<std::uint32_t>(PatchGroupBit::G1);

// "All" mask covers bits 0..30 (31 groups, A1..P3).
inline constexpr std::uint32_t kGroupMaskAll = 0x7FFFFFFFu;  // bits 0..30 (bit 31 = Q1 is opt-in only; intentionally excluded - see PatchGroupBit::Q1)

// `recommended` = `all` minus the groups below, all empirically determined: E1 (reciprocals at
// 0x14c4d60) and G1 (276 sites feeding fn 0xfdf9d0) break UI sizing at non-1080p; C2/C3 corrupt
// UI text positioning when solo (pairing with J1 fixes the text but not the canvas); I1-I4 and
// J1 tested with no UI-canvas effect; K1 is the opt-in exposure compensation ("recommended,K1").
inline constexpr std::uint32_t kGroupMaskRecommended =
    kGroupMaskAll
    & ~static_cast<std::uint32_t>(PatchGroupBit::C1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::C2)
    & ~static_cast<std::uint32_t>(PatchGroupBit::C3)
    & ~static_cast<std::uint32_t>(PatchGroupBit::P1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::E1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::G1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::I1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::I2)
    & ~static_cast<std::uint32_t>(PatchGroupBit::I3)
    & ~static_cast<std::uint32_t>(PatchGroupBit::I4)
    & ~static_cast<std::uint32_t>(PatchGroupBit::J1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::K1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::M1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::Q1);
// M1 does not change the sceVideoOutSetBufferAttribute dims at 2160p: the call at 0x00d9fa33
// (call 0x1218370, inside the flag-gated cmovne branch) is an internal game registrar, not the
// videoout import - the real caller is patched by N1. M1 stays enumerable as an opt-in toggle.

// Bisection presets - each adds exactly one group on top of the B1+D1 baseline so a test moves
// one variable at a time. B1 alone (or B1+D1, or `all`) lowers the main RT but leaves sub-RTs at
// 1080p (visible scene desync); D1 alone changes the frustum without the RT (rainbow corruption),
// so D1 only makes sense paired with B1.
inline constexpr std::uint32_t kGroupMaskBaseline =
    static_cast<std::uint32_t>(PatchGroupBit::B1) |
    static_cast<std::uint32_t>(PatchGroupBit::D1);

inline constexpr std::uint32_t kGroupMaskAddH1 =
    kGroupMaskBaseline |
    static_cast<std::uint32_t>(PatchGroupBit::H1);

inline constexpr std::uint32_t kGroupMaskAddH2 =
    kGroupMaskBaseline |
    static_cast<std::uint32_t>(PatchGroupBit::H2);

inline constexpr std::uint32_t kGroupMaskAddG1 =
    kGroupMaskBaseline |
    static_cast<std::uint32_t>(PatchGroupBit::G1);

inline constexpr std::uint32_t kGroupMaskAddAllH =
    kGroupMaskBaseline |
    static_cast<std::uint32_t>(PatchGroupBit::H1) |
    static_cast<std::uint32_t>(PatchGroupBit::H2) |
    static_cast<std::uint32_t>(PatchGroupBit::H3) |
    static_cast<std::uint32_t>(PatchGroupBit::H4);

inline constexpr std::uint32_t kGroupMaskWithoutG1 =
    kGroupMaskAll & ~static_cast<std::uint32_t>(PatchGroupBit::G1);

// ApplyGr2ResolutionPatches applies the GR2 override at `eboot_base`
// (MemoryPatcher::g_eboot_address). Patches the groups parsed from `groups_config` (empty =
// kGroupMaskRecommended; config.toml [GPU] resolutionPatchGroups); `aspect_ratio` only composes
// the final (W, H). Idempotent; returns the patched-site count, warning per mismatched site.
//
// ResPatchStats: outcome of one resolution-patch pass, filled when a non-null stats pointer is
// passed. The combined entry point treats (patched + already) > 0 as "this title's constants
// were present in the eboot", which drives the GR2-to-GRR fallback decision.
struct ResPatchStats {
    s32 patched = 0;     // sites freshly written this call
    s32 already = 0;     // sites already at the target value (idempotent re-run)
    s32 mismatched = 0;  // sites whose bytes matched neither stock nor target
    s32 skipped = 0;     // sites belonging to a disabled group
};

int ApplyGr2ResolutionPatches(uintptr_t eboot_base, TargetResolution resolution,
                              float aspect_ratio,
                              std::string_view groups_config,
                              ResPatchStats* stats = nullptr);

// Apply the Gravity Rush Remastered (CUSA01130/01112/01113/02318) override. GRR bakes a
// different constant family than GR2: main RT 1920x1088 (RT heights align up to a multiple of
// 64) plus 960x544 / 480x272 sub-RT floats. Same idempotent, memcmp-guarded contract as the GR2
// entry point; `groups_config` is [GPU] resolutionPatchGroupsGrr. Token grammar and per-group
// classification (R1/U1/C1 UI-only, S1..S7 scene-RT opt-ins at allocator 0x223e0/0x1cac0, T1
// comic text-cull fix in `recommended`, T2 opt-in comic-textbox pin) live in the .cpp.
int ApplyGrrResolutionPatches(uintptr_t eboot_base, TargetResolution resolution,
                              float aspect_ratio,
                              std::string_view groups_config,
                              ResPatchStats* stats = nullptr);

// Combined resolution-patch entry point, called by module.cpp at eboot load. Tries the GR2
// patches first and falls back to GRR when no GR2 site matches; a positively set
// Config::isGravityRushRemastered skips the GR2 attempt. The two group selectors are independent
// and only the one for the path that runs has any effect. Returns the matched path's site count.
int ApplyResolutionPatches(uintptr_t eboot_base, TargetResolution resolution,
                           float aspect_ratio,
                           std::string_view gr2_groups_config,
                           std::string_view grr_groups_config);

// Parse a config string into a TargetResolution. ("Off", "270p", "360p", "480p", "540p",
// "720p", "900p", "1080p", "1440p", "2160p"/"4K", "4320p"/"8K", or "WxH" forms.)
TargetResolution ParseResolutionFromConfig(std::string_view s);

// Return the *base* (16:9) (W, H) for a given TargetResolution.
Resolution TargetResolutionToBaseSize(TargetResolution t);

// Compose the FINAL (W, H) from a base resolution and an aspect ratio.
Resolution ComputeFinalResolution(TargetResolution t, float aspect_ratio);

} // namespace Libraries::ResolutionPatches
