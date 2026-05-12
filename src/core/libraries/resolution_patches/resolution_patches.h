// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// Runtime resolution patches for Gravity Rush 2 (CUSA00547 / CUSA03694 /
// CUSA04934 / CUSA04943, eboot v01.11).
//
// Replaces the hard-coded 1920x1080 render-target dimension constants in
// the eboot with the chosen pixel-density (vertical-resolution) target.
// Aspect-ratio scaling is composed in: this module REQUIRES the numeric
// aspect ratio to be passed alongside the resolution target so that the
// final (W, H) reflects both the chosen pixel density AND the chosen
// aspect.  Aspect changes never stretch — they widen or heighten the
// rendered image while keeping pixel density constant.
//
// v7 (this revision): EXHAUSTIVE re-hunt for new patchable RT-dim
// sites returned EMPTY — v6's 337 sites are the complete imm32-
// patchable set. The 540p corruption-with-all-groups symptom must
// therefore come from EXISTING group(s) being wrong, not from
// missing patches. v7's additions are PURELY diagnostic:
//
//   * NEW PRESETS for bisection:
//       - "minimal"   — B1 only (bare-minimum main-RT struct init)
//       - "no_quart"  — safe minus H2 (test if quarter-res RTs are
//                       what's desynchronising)
//       - "no_invproj"— safe minus D1 (test if D1's (1/W,1/H) part
//                       of the screen-vec uniform corrupts)
//       - "geom_minus_e1" — geom with E1 explicitly excluded
//       - "ext_minus_g1"  — ext with G1 (276 per-entity sites)
//                       excluded (test if G1 is over-patching)
//   * NEW CONFIRMATION: E1 (1/W,1/H at VA 0x14c4d60/0x14c4d64,
//     read once by vmulss at 0xfe60cc/0xfe60d4) IS the PS4 analog
//     of GXP shader reciprocal-of-resolution literals. Single-
//     instruction blast radius. If desync involves UI/HUD
//     coordinate scaling, enable E1.
//
// === v7 HUNT RESULTS (negative findings, documented for future) ==
//
// 1. PSSL shader bytecode (re-scanned all 392 OrbShdr binaries with
//    proper bounds — prologue 0xBEEB03FF .. last s_endpgm 0xBF810000):
//    zero hits for 1/1920, 1/1080, 1920.0f, 1080.0f, 960.0f, 540.0f,
//    480.0f, 270.0f as dword-aligned literals. PS4 PSSL does NOT
//    embed framebuffer dimensions in instruction streams the way
//    Vita GXP does. The CPU-side rodata constants (group E1 / D1)
//    ARE the GXP-analog — they get uploaded as immediate CBs.
//
// 2. Adjacent (W,H,1/W,1/H) and (1/W,1/H) rodata quads:  EXACTLY 2
//    locations site-wide, BOTH already covered:
//      * 0x0149e630 (D1 quad) — main RT screen-vec uniform
//      * 0x014c4d60 (E1 pair) — vmulss-consumer reciprocals
//
// 3. PM4 SET_CONTEXT_REG with packed scissor BR (1080<<16 | 1920 =
//    0x04380780): ZERO occurrences anywhere in the eboot — no
//    static command-buffer packets with baked-in dims.
//
// 4. V# image descriptors with (W-1)<<14|(H-1) or (H-1)<<14|(W-1):
//    ZERO in rodata — V#s are runtime-built from B1's struct.
//
// 5. Backward call-graph trace of all 30 callers of fn 0x1218370:
//      * 12 callers re-read W/H from [r15+0x92c0]/[+0x92c4] —
//        auto-handled by B1's struct store
//      *  7 callers are the H2 quarter-res sites (covered)
//      *  4 callers use small fixed dims (16x16, 64x60, 1xN) for
//        LUTs/cube-faces — NOT screen-related
//      *  4 callers use W*2, H*2 supersample from B1's struct —
//        auto-handled by B1
//      *  3 callers load W/H from a BSS variable or struct slot
//        that has NO discoverable imm32 writer (config-driven)
//
// 6. Mass 960.0f/540.0f cluster at VA 0x14ce638..0x14d065c (~70
//    pairs): NOT screen dims — pattern is mixed with 1/30 frame
//    time, π/2, 1.7999 ≈ 16:9, 100.0, 1000.0, 25.0 — a per-scene
//    or per-camera CONFIG TABLE, NOT a render-target descriptor
//    array. 467 LEA references point in. Patching would corrupt
//    level data. DO NOT enumerate as RT patches.
//
// === v6 carry-over (unchanged in v7) =============================
//
// Groups H1..H4 from v6 (also in v7):
//   * H1 — companion main RT dim pair inside fn 0x102a5b0 (paired
//          with B1's struct stores). Two sites, same call as B1.
//   * H2 — quarter-resolution sub-RT init pairs registered with
//          subsystem-IDs 3 and 6, eight pairs (16 sites), value
//          scaling (Wsel/4, Hsel/4).
//   * H3 — 4K-selector default-branch dim pair. INERT downstream.
//          OFF by default.
//   * H4 — IMUL-imm32 dim pairs at 4K-toggle entry fn. OFF by
//          default.
//
// 337 total sites in 18 disjoint groups (A1..H4). v7 adds NO new
// sites and NO new groups — only new presets for bisection.
//
// Composition with aspect_patches:
//   The aspect_patches module patches the 45 occurrences of the 1.7777f
//   projection-matrix aspect constant. This module patches the W/H/etc.
//   constants that control framebuffer/render-target dimensions and
//   pixel-space coordinate transforms.
//
//   Final dims compose as:
//     aspect ≈ 16:9    → ( base_W,                 base_H )
//     aspect <  16:9   → ( base_W,                 round(base_W / aspect) )
//     aspect >  16:9   → ( round(base_H * aspect), base_H )
//
// Patches are applied directly to the loaded eboot bytes via memcpy —
// equivalent to an HxD pre-patch.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace Libraries::ResolutionPatches {

enum class TargetResolution {
    Off,    // No-op; do not patch resolution constants at all.
    R540p,  // Base 960x540
    R720p,  // Base 1280x720
    R900p,  // Base 1600x900
    R1080p, // Base 1920x1080  (native — patches are mostly idempotent at 16:9)
    R1440p, // Base 2560x1440
    R2160p, // Base 3840x2160  (4K)
    R4320p, // Base 7680x4320  (8K)
};

struct Resolution {
    int width;
    int height;
};

// Patch groups — enable/disable via the `groups` config string passed to
// ApplyGr2ResolutionPatches. See resolution_patches.cpp for the per-group
// classification (consumer function, hypothesis, recommended use).
//
// Bitmask values are stable; you can persist them or compose them.
enum class PatchGroupBit : std::uint32_t {
    A1 = 1u << 0,   // float (W,H) struct init in fn @ 0x044c093  — UNCERTAIN
    A2 = 1u << 1,   // float (W,H) struct init in fn @ 0x0fdcb86  — pre-render setup?
    A3 = 1u << 2,   // float (W,H) struct init in fn @ 0x0fdcb86  — pre-render setup?
    A4 = 1u << 3,   // float (W,H) struct init in fn @ 0x110adfd  — secondary RT?
                    // v13: REINSTATED in `recommended`. Previously
                    // excluded based on a "UI text positioning
                    // corruptor when solo" classification, but Jun
                    // empirically confirmed A4 belongs in the working
                    // resolution set
                    // "A1,A2,A3,A4,B1,C1,D1,F1,F2,F3,H1,H2,H3,H4".
                    // The solo "corruption" was either misattributed
                    // (the same observation was actually caused by E1
                    // or G1, which v11 fixed) or only manifests when
                    // A4 is patched without its B1/D1/H1/H2 sibling
                    // group — i.e. it's only a corruptor in isolation,
                    // not in the full recommended set.
    B1 = 1u << 4,   // u32 (W,H) struct init  in fn @ 0x102a5b0   — MAIN RENDER (highest confidence)
    C1 = 1u << 5,   // (W,H,1000.01,H) vec  in fn @ 0x0136d74d    — UNCERTAIN
    C2 = 1u << 6,   // (W,H,1000.01,H) vec  in fn @ 0x0110c110    — secondary RT?
    C3 = 1u << 7,   // (W,H,1000.01,H) vec  in fn @ 0x0110d140    — secondary RT?
    D1 = 1u << 8,   // (W,H,1/W,1/H) vec    in fn @ 0x102a5b0     — MAIN RENDER (paired with B1)
    E1 = 1u << 9,   // (1/W,1/H) reciprocals in fn @ 0x0fe6090    — paired with A2/A3?
                    // v11: EMPIRICALLY UI-CANVAS-CORRUPTING. Removed from
                    // `recommended`. Patching the reciprocals to (1/target_W,
                    // 1/target_H) re-normalizes UI element coords to the
                    // target framebuffer, but the UI viewport submission
                    // STAYS at 1920×1080 — the mismatched normalization
                    // distorts UI element sizes. Keep enumerable for the
                    // small minority of consumers that actually want the
                    // re-normalization.
    F1 = 1u << 10,  // (halfW,halfH,W,H) tuple in fn @ 0x05fb908  — sub-camera projection (UI?)
    F2 = 1u << 11,  // (halfW,halfH,W,H) tuple in fn @ 0x09236a0  — sub-camera projection (UI?)
    F3 = 1u << 12,  // (halfW,halfH,W,H) tuple in fn @ 0x0c46bf0  — sub-camera projection (UI?)
    G1 = 1u << 13,  // ecx=W,r8d=H args to fn 0xfdf9d0 (per-entity unproject) — 145+ call sites
                    // v11: EMPIRICALLY UI-CANVAS-CORRUPTING. Removed from
                    // `recommended`. Previously thought to "SYNC 3D at
                    // non-1080p" (v7.3 note) but the per-entity unproject
                    // also affects UI element screen-space transforms.
                    // With G1 patched to target res, UI entities compute
                    // unproject math against target-res screen but render
                    // through a viewport still sized 1920×1080, producing
                    // wrong UI element sizes. Keep enumerable for callers
                    // that need 3D-entity unproject tracking the target
                    // resolution AND accept the UI size cost.
    // ── v6 additions ────────────────────────────────────────────────
    H1 = 1u << 14,  // companion main RT (W,H) pair in fn 0x102a5b0 — paired with B1
    H2 = 1u << 15,  // quarter-res sub-RT inits passed to fn 0x1218370 (sid=3,6) — 8 pairs
    H3 = 1u << 16,  // 4K-selector 1080p-default-branch (W,H) pair — OFF by default (handoff: inert)
    H4 = 1u << 17,  // IMUL-imm32 dim pairs at 4K-toggle entry fn — OFF by default
    // ── v7.3 addition ───────────────────────────────────────────────
    I1 = 1u << 18,  // UI projection lock — replaces the only 2 (1/W,-1/H,1/W,1/H)
                    // float-projection readers of B1's struct slot with literal
                    // 1920/1080 loads. NOTE: empirical testing showed this does
                    // NOT fix the UI element scaling issue — left available but
                    // NOT in `recommended`. May be a postprocess-texel-size
                    // path rather than UI projection.
    // ── v7.4 addition ───────────────────────────────────────────────
    I2 = 1u << 19,  // UI ortho projection scale — patches the rodata pair
                    // (2/1920, 2/1080) at 0x14C565C/0x14C5664. EMPIRICALLY
                    // CONFIRMED USELESS — no visible effect at any res.
                    // fn 0x10d3b60 is dead/niche. Excluded from `recommended`.
    // ── v7.5 addition ───────────────────────────────────────────────
    I3 = 1u << 20,  // UI VIEWPORT SCALE — patches (960, 540) at 0x01499EC0
                    // and 0x01499EC4. These are the X/Y scale slots of the
                    // viewport descriptor passed to fn 0x110C110 (the UI/text
                    // viewport-set function, same one C2 patches into).
                    // EMPIRICALLY CONFIRMED USELESS in v7.5 — only one
                    // vmovaps consumer at 0x00EABC8D, patching had no visible
                    // effect on UI scaling. DEMOTED from `recommended` in v8.
                    // Left enumerable so the binding can be opted into for
                    // re-verification or bisection, but the rodata at this
                    // VA is not the UI canvas lever.
    // ── v7.6 / v8 addition ───────────────────────────────────────────
    I4 = 1u << 21,  // Candidate-A UI viewport SCALE row. Patches the
                    // (960.0f, 540.0f) pair at rodata 0x0149A190 /
                    // 0x0149A194 — the second untested heavy-use vec4
                    // identified by the v7.5 hunt. Consumed by ≥8 vmovaps
                    // loads in the 0xEBC8xx-0xEBD9xx fn range (possibly a
                    // SECOND UI/3D viewport descriptor distinct from I3's
                    // 0x01499EC0 site). Single-function blast radius
                    // (0xEBD8ED..0xEBDE9A). Empirically tested by user:
                    // does NOT fix UI canvas scaling. DEMOTED — kept
                    // enumerable for re-verification only.
    // ── v9 addition ──────────────────────────────────────────────────
    J1 = 1u << 22,  // Flip-Y mirror vecs paired with C1/C2/C3. Three
                    // (1920, -1080, -999.99, -1080) vecs at C_VA - 0x20
                    // in rodata (0x014985b0, 0x014a0f30, 0x014a0fe0).
                    // Disasm confirms the C-consumer fns load BOTH
                    // vecs and component-multiply them — but the product
                    // feeds per-character TEXT POSITIONING math, not
                    // canvas/viewport sizing. Empirically tested by
                    // user (v9): does NOT fix the UI canvas problem.
                    // DEMOTED. Retained as enumerable for any future
                    // C-group correctness work involving text glyph
                    // positioning across the C/J pair.
    // ── v14 (was v12) ───────────────────────────────────────────────
    K1 = 1u << 23,  // SceneParameter.exposureIntensity = 2.0f sites
                    // scaled by pixel-density ratio
                    // min(1.0, (target_H/1080)²) — since pixel count
                    // scales as H² at fixed 16:9 aspect, this is the
                    // framebuffer-pixel-count ratio relative to 1080p.
                    // Lower-res targets render with QUADRATICALLY less
                    // exposure (compensates empirical over-exposure
                    // at 540p/720p/900p — 540p is the worst; v12's
                    // linear H/1080 scaling wasn't aggressive enough).
                    //
                    // Three sites identified statically:
                    //   0x00b3f625  (instr 0x00b3f622, scene-param ctor)
                    //   0x004c7d16  (instr 0x004c7d13)
                    //   0x008abd31  (instr 0x008abd2e)
                    // All share the encoding `C7 40 50 00 00 00 40`
                    // (= `mov dword ptr [rax+0x50], 0x40000000`); the
                    // imm32 sits at instr_va+3.
                    //
                    // Field offset +0x50 confirmed via Lua-binding
                    // registrar at 0x01056d9e — the four exposure
                    // params (exposureIntensity, exposureAdaptionRange,
                    // exposureAdaptionSpeed, exposureKeyValue) live at
                    // struct offsets +0x50/0x54/0x58/0x5c.
                    //
                    // Encoded value: 2.0f × min(1.0, (H/1080)²):
                    //   540p   → 0.5    (quarter of 1080p's 2.0, per spec)
                    //   720p   → 0.889  (× 4/9)
                    //   900p   → 1.389  (× 25/36)
                    //   1080p+ → 2.0    (idempotent, clamped at 1.0)
                    //
                    // EXCLUDED from `recommended`. v16: the user-
                    // facing config toggle for individual groups was
                    // removed; K1 is no longer reachable from outside
                    // this module. Retained as an internal-only knob
                    // for any future re-export (e.g. an exposure-
                    // compensation config bool).
    // ── v15 addition ────────────────────────────────────────────────
    L1 = 1u << 24,  // Disable motion blur by NOP'ing two `call
                    // <register-pass>` instructions in the render-
                    // graph builder. MotionBlur is registered from
                    // two sites:
                    //   0x0103e356  call 0x121b040 (special-slot)
                    //   0x0103ea95  call 0x121ae70 (standard)
                    // Both calls' return values are discarded by the
                    // next instruction (`mov eax, [rip + ...]`), so
                    // NOP'ing is safe w.r.t. dataflow — only the
                    // side effect (registering the pass with the
                    // active draw list) is removed.
                    //
                    // The "MotionBlur" string at 0x014fc61a sits in
                    // the same rodata cluster as Tonemap, Bloom,
                    // Antialias, SSAO — these are render-graph node
                    // names passed to the pass-init function by name.
                    // All other passes use only the standard register
                    // call; MotionBlur is the only one with the extra
                    // special-slot register, likely because it needs
                    // a specific position in the draw order.
                    //
                    // RISK: untested at ship time. If either register
                    // function has required side effects beyond list
                    // insertion (refcounts, init state for later
                    // passes), patching could crash. The shared
                    // 0x121ae70 is called by Tonemap/Bloom/etc. from
                    // their OWN sites — NOP'ing here only removes the
                    // MotionBlur-specific call, not the function
                    // pointer itself. Should be clean.
                    //
                    // EXCLUDED from `recommended`. v16: gated by the
                    // `disableMotionBlur` config bool — wired through
                    // ApplyGr2ResolutionPatches's `disable_motion_blur`
                    // parameter. To turn off motion blur, set
                    // `[GPU] disableMotionBlur = true` in config.toml.
};

// Recommended preset masks. Use these by name in the config string ("safe",
// "geom", "ext", "all"), or compose your own e.g. "B1,D1,H1,H2,A4,C2,C3".
//
// v6 promotes H1 + H2 to the safe preset: H1 always pairs with B1
// (same function, same call), and H2 fixes the quarter-res sub-RTs that
// otherwise stay at 1080/4 = 480x270 even when the main RT is lowered.
//
// D1 stays in `safe`: D1's (W,H,1/W,1/H) vec is the frustum/screen-space
// uniform that MUST travel with B1's resolution change. D1 alone (without
// B1) produced rainbow/frustum corruption in earlier testing, but that's
// because D1 changes the frustum without changing the RT — the two must
// be patched together.
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

// "All" mask covers bits 0..24 (25 groups, A1..L1).
inline constexpr std::uint32_t kGroupMaskAll = 0x1FFFFFFu;  // bits 0..24

// ── v13 ─────────────────────────────────────────────────────────────────
// `recommended` = `all` minus ten groups, ALL EMPIRICALLY DETERMINED.
// User-confirmed working resolution set as of v13:
//   "A1,A2,A3,A4,B1,C1,D1,F1,F2,F3,H1,H2,H3,H4"
//
//   * A4 — **v13 reinstated.** Previously excluded based on a "UI text
//          positioning corruptor when solo" classification, but Jun
//          confirmed empirically that A4 belongs in the working
//          resolution set. Earlier exclusion was stale — A4 is needed
//          and the "UI text" classification was either wrong or
//          obsoleted by the v11 E1/G1 removal.
//
//   * E1, G1 — UI-canvas corrupters (v11 finding). E1 (reciprocal pair
//                at 0x14c4d60) and G1 (276 per-entity ecx=1920/r8d=1080
//                mov-imm32 sites feeding fn 0xfdf9d0) broke UI sizing
//                at non-1080p. Removing them from `recommended` (v11)
//                restored correct UI canvas rendering.
//
//   * C2, C3 — UI text positioning corruptors when patched solo.
//                  J1 hypothesis (pair them with flip-Y siblings)
//                  produced consistent text positioning but did NOT
//                  fix the UI canvas centering (now superseded — UI
//                  canvas was fixed by removing E1+G1, not by adding
//                  C+J patches).
//   * I1        — UI proj instr-replace; tested, no visible UI effect.
//   * I2        — ortho 2/W,2/H pair; tested, no effect (dead/niche fn).
//   * I3        — UI viewport scale row at 0x01499EC0; tested, no effect.
//   * I4        — Candidate-A at 0x0149A190; tested, no effect.
//   * J1        — flip-Y mirror of C-vecs; tested, text positioning only.
//   * K1        — v14 exposure-intensity pixel-density compensation
//                 (was v12 linear H/1080; v14 changed to quadratic
//                 (H/1080)² after Jun confirmed linear didn't fully
//                 compensate low-res over-exposure). Opt-in via
//                 "recommended,K1". Targets the empirical over-
//                 exposure at low resolutions ("540p is the worst").
//                 Patches three exposureIntensity = 2.0f sites to
//                 scale by pixel-density ratio min(1.0, (H/1080)²):
//                 540p → 0.5 (quarter of 1080p's 2.0).
//   * L1        — v15 disable motion blur. Two NOP-call patches at
//                 0x0103e356 and 0x0103ea95 in the render-graph
//                 builder skip the MotionBlur pass registration.
//                 Opt-in via "recommended,L1". Untested at v15 ship
//                 time — risk if either register-pass function has
//                 required side effects beyond list insertion.
inline constexpr std::uint32_t kGroupMaskRecommended =
    kGroupMaskAll
    & ~static_cast<std::uint32_t>(PatchGroupBit::C2)
    & ~static_cast<std::uint32_t>(PatchGroupBit::C3)
    & ~static_cast<std::uint32_t>(PatchGroupBit::E1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::G1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::I1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::I2)
    & ~static_cast<std::uint32_t>(PatchGroupBit::I3)
    & ~static_cast<std::uint32_t>(PatchGroupBit::I4)
    & ~static_cast<std::uint32_t>(PatchGroupBit::J1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::K1)
    & ~static_cast<std::uint32_t>(PatchGroupBit::L1);

// ── v7 bisection presets ────────────────────────────────────────────────
// These exist to find which of the NEW-in-v5/v6 groups actually help vs
// hurt the residual non-main-RT desync. Confirmed-already-tested baselines
// (do NOT need re-testing):
//
//   * `none` / Off               → no patches, original 1080p, clean
//   * `B1` alone                 → main RT lowered, sub-RTs at 1080p →
//                                  visible scene desync (chars correct,
//                                  environment wrong scale)
//   * `D1` alone                 → frustum changes without RT — rainbow/
//                                  frustum corruption. D1 ONLY makes sense
//                                  paired with B1.
//   * `B1,D1`   (v3 safe)        → same desync class as B1 alone
//   * `all`     (v5 313 sites)   → same desync class
//
// What's been added since the last test:
//   * H1 (2 sites)  — companion main-RT (W,H) pair, same fn as B1
//   * H2 (16 sites) — quarter-res sub-RT inits (sid=3, sid=6)
//   * H3 (2 sites)  — 4K-selector default branch (off by default)
//   * H4 (4 sites)  — IMUL-imm32 dim pairs near 4K toggle (off by default)
//
// Each preset below adds exactly ONE not-yet-tested group on top of v3's
// known baseline (B1+D1), so the test ratchets one variable at a time:
//
//   baseline   = B1+D1            (v3 known-desync baseline; sanity check
//                                  the build is identical to v3's behavior)
//   add_h1     = B1+D1+H1         (does H1's companion-pair help sync?)
//   add_h2     = B1+D1+H2         (does H2 quarter-res help sync?)
//   add_h1h2   = B1+D1+H1+H2  =safe (v6 default; combined H additions)
//   add_g1     = B1+D1+G1         (does G1's 276 per-entity transforms help?)
//   add_all_h  = B1+D1+H1+H2+H3+H4 (all H-groups together; bypass H3/H4
//                                  "off by default" caveat)
//   without_g1 = all & ~G1        (if all-with-G1 corrupts MORE than v3,
//                                  this isolates the G1 cost)
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

// Apply the GR2 resolution+aspect override. Internally patches the
// `recommended` group set (the empirically-verified working subset —
// see kGroupMaskRecommended above) and, optionally, the L1 motion-blur
// disable patch. Returns the number of sites patched. Idempotent. Logs
// a warning per mismatched site (e.g. when the eboot doesn't match v1.11).
//
// `eboot_base`           — should be MemoryPatcher::g_eboot_address.
// `resolution`           — chosen vertical pixel density (Off skips
//                          patching entirely).
// `aspect_ratio`         — final aspect ratio (typically obtained from
//                          AspectPatches::TargetAspectToRatio). Used to
//                          compose the final (W, H); does NOT patch the
//                          projection float (that's aspect_patches' job).
// `disable_motion_blur`  — if true, also apply the L1 group (two NOP-
//                          call patches that skip the MotionBlur pass
//                          registration in the render-graph builder).
//                          Independent of the resolution override; the
//                          patches are applied even when `resolution`
//                          is Off (so motion blur can be disabled at
//                          native res too).
int ApplyGr2ResolutionPatches(uintptr_t eboot_base, TargetResolution resolution,
                              float aspect_ratio, bool disable_motion_blur);

// Parse a config string into a TargetResolution. ("Off", "540p", "720p",
// "900p", "1080p", "1440p", "2160p"/"4K", "4320p"/"8K", or "WxH" forms.)
TargetResolution ParseResolutionFromConfig(std::string_view s);

// Return the *base* (16:9) (W, H) for a given TargetResolution.
Resolution TargetResolutionToBaseSize(TargetResolution t);

// Compose the FINAL (W, H) from a base resolution and an aspect ratio.
Resolution ComputeFinalResolution(TargetResolution t, float aspect_ratio);

} // namespace Libraries::ResolutionPatches
