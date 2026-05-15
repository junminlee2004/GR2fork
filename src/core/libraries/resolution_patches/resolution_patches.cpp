// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include "common/logging/log.h"
#include "core/libraries/resolution_patches/resolution_patches.h"

namespace Libraries::ResolutionPatches {

namespace {

enum class Kind {
    W_f, H_f, HalfW_f, HalfH_f, InvW_f, InvH_f, W_u32, H_u32,
    // ── v6 additions ────────────────────────────────────────────────
    // Quarter-resolution integer dimensions. Encoded as (final_W / 4)
    // and (final_H / 4) at apply-time; baseline bytes are 0x1E0 (480)
    // and 0x10E (270) at 1080p.
    QuarterW_u32,
    QuarterH_u32,
    // ── v7.4 additions ──────────────────────────────────────────────
    // 2/W, 2/H = orthographic projection X/Y scale for a 1920x1080
    // design canvas. Encoded as 2.0f/W_actual and 2.0f/H_actual. At
    // 1080p these equal 2/1920 = 1/960 and 2/1080 = 1/540 (so the
    // patch is idempotent at native). Used to scale UI ortho projection
    // with framebuffer dim so 1080p-design UI elements scale relative
    // to the actual canvas.
    InvHalfW_f,
    InvHalfH_f,
    // ── v9 additions ────────────────────────────────────────────────
    // Negative-Y height. Encoded as -H_actual (sign bit set). At 1080p
    // the baseline is -1080.0f = 0xC4870000. Used in flip-Y mirror
    // projection vecs (J-group) that sit adjacent to C-group vecs and
    // are component-multiplied with them inside the C-consumer fns.
    NegH_f,
    // ── v14 (was v12) ───────────────────────────────────────────────
    // SceneParameter.exposureIntensity default of 2.0f, scaled by
    // pixel-density ratio min(1.0, (H/1080)²) — at fixed 16:9 aspect
    // the framebuffer pixel count scales as H², so this is the
    // pixel-count ratio relative to 1080p. Lower-res targets get
    // *quadratically* less exposure, compensating Jun's empirical
    // over-exposure observation. 540p → 0.5 (quarter of 1080p's 2.0),
    // 720p → 0.889, 900p → 1.389, 1080p+ → 2.0 (idempotent and
    // clamped above native).
    // Baseline bytes at patch site: 0x40000000 (= 2.0f LE).
    Exposure2_f,
};

// Parallel to PatchGroupBit; small integer index for table compactness.
enum class Group : std::uint8_t {
    A1 = 0, A2, A3, A4, B1, C1, C2, C3, D1, E1, F1, F2, F3, G1,
    // ── v6 additions ────────────────────────────────────────────────
    H1, H2, H3, H4,
    // ── v7.3 addition ───────────────────────────────────────────────
    I1,  // UI projection lock — instruction-replacement patches at
         // 0x01011306/0x0101130C and 0x01018D90/0x01018D96. These were
         // identified by exhaustive scan as the ONLY two readers of
         // [reg+0x92c0]/[+0x92c4] that perform float math (cvtsi2ss
         // + vdivss) — i.e., they compute the screen-space (1/W, -1/H,
         // 1/W, 1/H) normalization quad for UI rendering. Patching
         // them to use literal 0x780/0x438 (1920/1080) instead of
         // reading from B1's struct keeps UI elements at correct
         // relative size across all resolutions.
    // ── v7.4 addition ───────────────────────────────────────────────
    I2,  // UI ortho projection scale — patches the (2/1920, 2/1080)
         // pair at rodata 0x14C565C/0x14C5664. EMPIRICALLY CONFIRMED
         // useless (user test 2026-05-10: no visible effect at any
         // resolution). fn 0x10D3B60 is dead/niche code. Retained for
         // completeness but excluded from `recommended`.
    // ── v7.5 addition ───────────────────────────────────────────────
    I3,  // UI VIEWPORT SCALE — patches the (960, 540, 0, 0) vec4 at
         // rodata 0x01499EC0. Loaded by `vmovaps xmm1, [rip+...]` at
         // VA 0x00EABC8D, then stored into a 4-vec4 viewport descriptor
         // on stack and passed to fn 0x110C110. EMPIRICALLY CONFIRMED
         // USELESS in v7.5 — only one vmovaps consumer of this rodata
         // pair, patching had no visible effect on UI scaling. DEMOTED
         // from `recommended` in v8. Retained as an opt-in for
         // re-verification only.
    // ── v7.6 / v8 addition ──────────────────────────────────────────
    I4,  // Candidate-A UI viewport SCALE row. The (960.0f, 540.0f)
         // pair at rodata 0x0149A190 / 0x0149A194 is the second
         // untested heavy-use vec4 from the v7.5 hunt: 8+ vmovaps
         // consumers in fn 0xEBC8xx-0xEBD9xx range. Static structure:
         //   0x0149A180  (1,1,1,1)         — scale row 0 / identity
         //   0x0149A190  (960,540,0,0)     — this site (X/Y scale)
         //   0x0149A1A0  (0,0,0,0)         — offset row?
         //   0x0149A1B0  (150,150,150,150) — limits row?
         //   0x0149A1C0  (0,0,0,1)         — w-component identity
         // Empirically tested by user: I4 alone does NOT fix UI canvas
         // scaling. DEMOTED. Retained as enumerable for re-verification.
    // ── v9 addition ─────────────────────────────────────────────────
    J1,  // Flip-Y mirror vecs paired with C1/C2/C3. Three
         // (1920, -1080, -999.99, -1080) vecs at C_VA - 0x20
         // in rodata (0x014985b0, 0x014a0f30, 0x014a0fe0).
         // Disasm of C2's consumer fn 0x0110c110 confirmed
         // BOTH vecs loaded: vmovaps J-sibling @ 0x0110c474,
         // then vmulps with C-vec @ 0x0110c4b9 (component-wise
         // product, same fn, 0x45 bytes apart). C3 mirrors the
         // pattern at fn 0x0110d140.
         //
         // EMPIRICALLY CONFIRMED USELESS for the UI canvas
         // problem (user test, v9): the vmulps product is
         // consumed by per-character TEXT POSITIONING math, not
         // canvas/viewport sizing. C2/C3's "UI text corruption
         // when solo" is real but it's text-glyph positioning
         // breakage, not a missing canvas-scale lever. Patching
         // J1+C2+C3 produces consistent text positioning across
         // the C/J pair but does NOT fix the centered-1920x1080
         // UI canvas region. DEMOTED. Retained as enumerable
         // for any future C-group correctness work.
    // ── v14 (was v12) ───────────────────────────────────────────────
    K1,  // SceneParameter.exposureIntensity = 2.0f sites, scaled by
         // pixel-density ratio min(1.0, (target_H/1080)²) so lower-res
         // targets get *quadratically* less exposure (compensates the
         // OVER-exposure observed at 540p/720p/900p; 540p is the
         // worst). Three sites identified via Lua scene-parameter
         // struct field offset map: 'exposureIntensity' lives at
         // struct[+0x50], with a default of 2.0f set at 0x00b3f622,
         // 0x004c7d13, 0x008abd2e — instructions of form
         // `mov dword ptr [rax+0x50], 0x40000000`. Each patch rewrites
         // the imm32 bytes to encode the scaled value. At 1080p+ the
         // patch is idempotent (multiplier clamped to 1.0). EXCLUDED
         // from `recommended` — opt in to test.
    Count
};

constexpr std::uint32_t GroupBit(Group g) {
    return 1u << static_cast<std::uint32_t>(g);
}

struct ResSite {
    std::uint32_t va_offset; // VA offset from eboot base
    Kind          kind;
    Group         group;
    const char*   label;
};

// 37 verified sites, each tagged with the consumer-function group.
//
// ULTRATHINK classification (consumer fn / what the group likely controls):
//
//  ┌──────┬─────────────┬───────────────────────────────────────────────┐
//  │ B1   │ fn 0x102a5b0│ MAIN render-device init (size 0xC94, calls    │
//  │      │             │ a 5-iteration RT setup pattern). u32 W/H      │
//  │      │             │ struct[+0x92C0]/+0x92C4 init followed by a    │
//  │      │             │ vmovaps writing the D1 vec at +0x92D0.        │
//  │      │             │ HIGHEST confidence — patch this for any res   │
//  │      │             │ change.                                       │
//  ├──────┼─────────────┼───────────────────────────────────────────────┤
//  │ D1   │ same fn     │ MAIN GPU screen-vec uniform (W,H,1/W,1/H)     │
//  │      │             │ paired with B1. HIGHEST confidence.           │
//  ├──────┼─────────────┼───────────────────────────────────────────────┤
//  │ A4   │ fn 0x110adfd│ Likely SECONDARY RT struct init (sized 0xD2F, │
//  │ C2   │ fn 0x110c110│ in same code area, immediately after main RT  │
//  │ C3   │ fn 0x110d140│ init). C2/C3 are template-α vec multipliers.  │
//  │      │             │ Recommended for buffer scaling fidelity.      │
//  │      │             │ ── v13 EMPIRICAL ── A4 REINSTATED IN          │
//  │      │             │ `recommended`. User-confirmed working res set │
//  │      │             │ is "A1,A2,A3,A4,B1,C1,D1,F1,F2,F3,H1,H2,H3,   │
//  │      │             │ H4". A4's previous "UI text corruptor when    │
//  │      │             │ solo" classification was stale — either       │
//  │      │             │ misattributed (E1/G1 were the actual UI       │
//  │      │             │ corrupters, fixed in v11) or only manifests   │
//  │      │             │ when A4 is patched without its sibling group  │
//  │      │             │ (B1/D1/H1/H2). C2 and C3 remain excluded —    │
//  │      │             │ they still corrupt UI text positioning solo.  │
//  ├──────┼─────────────┼───────────────────────────────────────────────┤
//  │ A2   │ fn 0x0fdcb86│ Float W/H struct inits earlier in pipeline.   │
//  │ A3   │ same fn     │ Likely shader/descriptor-table setup.         │
//  │ E1   │ fn 0x0fe6090│ Paired (1/W,1/H) reciprocals nearby — same    │
//  │      │             │ subsystem. ── v11 EMPIRICAL ── PATCHING E1    │
//  │      │             │ CORRUPTS UI CANVAS SIZING. Re-normalizing UI  │
//  │      │             │ element coords to target-res reciprocals      │
//  │      │             │ while the UI viewport still submits 1920×1080 │
//  │      │             │ distorts element sizes. Removed from          │
//  │      │             │ `recommended`.                                │
//  ├──────┼─────────────┼───────────────────────────────────────────────┤
//  │ A1   │ fn 0x044c093│ Repeated 12+ calls to ONE callee 0x13f6360,   │
//  │      │             │ likely populating a struct array. Doesn't     │
//  │      │             │ smell like a render target. UNCERTAIN.        │
//  │ C1   │ fn 0x0136d74d│ Standalone vmulps consumer; far in code      │
//  │      │             │ (~0x136e000+). UNCERTAIN.                     │
//  ├──────┼─────────────┼───────────────────────────────────────────────┤
//  │ F1   │ fn 0x05fb908│ (halfW,halfH,W,H) tuple → calls               │
//  │ F2   │ fn 0x09236a0│ 0x10cbc00 + 0x10cf0e0 — the same projection-  │
//  │ F3   │ fn 0x0c46bf0│ matrix builder pair used by Group C of the    │
//  │      │             │ aspect patches. Each F-group is a SEPARATE    │
//  │      │             │ camera/viewport projection setup (photo mode? │
//  │      │             │ minimap? UI canvas? gallery?). Patching them  │
//  │      │             │ rewrites the (W,H) those sub-cameras think    │
//  │      │             │ their viewport is — but their actual sub-RT   │
//  │      │             │ may be fixed-size, causing visual mismatch.   │
//  │      │             │ STRONG SUSPECT for "UI corruption" — try      │
//  │      │             │ disabling these first if visuals are wrong.   │
//  ├──────┼─────────────┼───────────────────────────────────────────────┤
//  │ G1   │ fn 0xfdf9d0 │ Per-entity screen-space transform / unproject.│
//  │      │ (109 sites) │ Heavy SIMD math: takes (W,H) + 4×4 matrix,    │
//  │      │             │ writes transformed 4×4 to [r15+0xC0..0xF0].   │
//  │      │             │ Called from 109 distinct sites, each with     │
//  │      │             │ `mov ecx,0x780; mov r8d,0x438` arg-prep. Each │
//  │      │             │ caller is a separate entity/object computing  │
//  │      │             │ its own screen-relative frustum or pick ray.  │
//  │      │             │ Adds 218 patch sites (109 W + 109 H).         │
//  │      │             │ ── v11 EMPIRICAL ── PATCHING G1 CORRUPTS UI   │
//  │      │             │ CANVAS SIZING. The per-entity unproject also  │
//  │      │             │ runs on UI entities; patching it makes UI     │
//  │      │             │ entities compute unproject against target res │
//  │      │             │ while the UI viewport stays at 1920×1080.     │
//  │      │             │ Removed from `recommended`. The hypothesized  │
//  │      │             │ "3D sync at non-1080p" benefit was either     │
//  │      │             │ illusory or carried by other groups (B1+H2+   │
//  │      │             │ H1+D1 cover the main RT chain).               │
//  └──────┴─────────────┴───────────────────────────────────────────────┘
constexpr std::array<ResSite, 355> kResSites = {{
    // ── Group A: float mov-imm32 paired struct-init writes (8 sites) ────
    {0x0044cad5, Kind::W_f, Group::A1, "A1.W (mov [rsi+rax*4],1920.0f) — fn 0x44c093"},
    {0x0044cadc, Kind::H_f, Group::A1, "A1.H (mov [rdx+rax*4],1080.0f) — fn 0x44c093"},
    {0x00fde098, Kind::W_f, Group::A2, "A2.W (mov [rsi+0x20],1920.0f)  — fn 0xfdcb86"},
    {0x00fde0a0, Kind::H_f, Group::A2, "A2.H (mov [rsi+0x24],1080.0f)  — fn 0xfdcb86"},
    {0x00fde146, Kind::W_f, Group::A3, "A3.W (mov [rsi+rdi+0x18],…)    — fn 0xfdcb86"},
    {0x00fde14f, Kind::H_f, Group::A3, "A3.H (mov [rsi+rdi+0x1c],…)    — fn 0xfdcb86"},
    {0x0110b4a7, Kind::W_f, Group::A4, "A4.W (mov [rsi+rax*4],1920.0f) — fn 0x110adfd — UI TEXT POS (do not enable at non-1080p)"},
    {0x0110b4ae, Kind::H_f, Group::A4, "A4.H (mov [rdx+rax*4],1080.0f) — fn 0x110adfd — UI TEXT POS (do not enable at non-1080p)"},

    // ── Group B: integer mov-imm32 paired struct-init (2 sites) ────────
    {0x0102aa21, Kind::W_u32, Group::B1, "B1.W (mov [rdi+0x92c0],0x780) — fn 0x102a5b0 MAIN"},
    {0x0102aa2c, Kind::H_u32, Group::B1, "B1.H (mov [r15+0x92c4],0x438) — fn 0x102a5b0 MAIN"},

    // ── Group C: (W,H,1000.01,H) vec multipliers (9 sites) ─────────────
    {0x014985d0, Kind::W_f, Group::C1, "C1.W — vmulps in fn 0x136d74d"},
    {0x014985d4, Kind::H_f, Group::C1, "C1.H — vmulps in fn 0x136d74d"},
    {0x014985dc, Kind::H_f, Group::C1, "C1.H_again(+12) — vmulps in fn 0x136d74d"},
    {0x014a0f50, Kind::W_f, Group::C2, "C2.W — vmulps in fn 0x110c110 — UI TEXT POS (do not enable at non-1080p)"},
    {0x014a0f54, Kind::H_f, Group::C2, "C2.H — vmulps in fn 0x110c110 — UI TEXT POS"},
    {0x014a0f5c, Kind::H_f, Group::C2, "C2.H_again(+12) — vmulps in fn 0x110c110 — UI TEXT POS"},
    {0x014a1000, Kind::W_f, Group::C3, "C3.W — vmulps in fn 0x110d140 — UI TEXT POS (do not enable at non-1080p)"},
    {0x014a1004, Kind::H_f, Group::C3, "C3.H — vmulps in fn 0x110d140 — UI TEXT POS"},
    {0x014a100c, Kind::H_f, Group::C3, "C3.H_again(+12) — vmulps in fn 0x110d140 — UI TEXT POS"},

    // ── Group D: (W,H,1/W,1/H) projection vec (4 sites) ────────────────
    {0x0149e630, Kind::W_f,    Group::D1, "D1.W   — vmovaps in fn 0x102a5b0 MAIN"},
    {0x0149e634, Kind::H_f,    Group::D1, "D1.H   — vmovaps in fn 0x102a5b0 MAIN"},
    {0x0149e638, Kind::InvW_f, Group::D1, "D1.invW— vmovaps in fn 0x102a5b0 MAIN"},
    {0x0149e63c, Kind::InvH_f, Group::D1, "D1.invH— vmovaps in fn 0x102a5b0 MAIN"},

    // ── Group E: (1/W,1/H) reciprocal pair (2 sites) ───────────────────
    {0x014c4d60, Kind::InvW_f, Group::E1, "E1.invW — vmulss in fn 0x0fe6090"},
    {0x014c4d64, Kind::InvH_f, Group::E1, "E1.invH — vmulss in fn 0x0fe6090"},

    // ── Group F: (halfW,halfH,W,H) sub-camera projection tuples (12) ───
    //  Each F-group is consumed by a single function that calls
    //  0x10cbc00 + 0x10cf0e0 (sub-camera/projection build). Strongest
    //  suspect for UI/sub-viewport corruption when patched.
    {0x014b975c, Kind::HalfW_f, Group::F1, "F1.halfW — fn 0x05fb908 (sub-camera)"},
    {0x014b9760, Kind::HalfH_f, Group::F1, "F1.halfH — fn 0x05fb908"},
    {0x014b9764, Kind::W_f,     Group::F1, "F1.W     — fn 0x05fb908"},
    {0x014b9768, Kind::H_f,     Group::F1, "F1.H     — fn 0x05fb908"},
    {0x014bdf08, Kind::HalfW_f, Group::F2, "F2.halfW — fn 0x09236a0 (sub-camera)"},
    {0x014bdf0c, Kind::HalfH_f, Group::F2, "F2.halfH — fn 0x09236a0"},
    {0x014bdf10, Kind::W_f,     Group::F2, "F2.W     — fn 0x09236a0"},
    {0x014bdf14, Kind::H_f,     Group::F2, "F2.H     — fn 0x09236a0"},
    {0x014c1388, Kind::HalfW_f, Group::F3, "F3.halfW — fn 0x0c46bf0 (sub-camera)"},
    {0x014c138c, Kind::HalfH_f, Group::F3, "F3.halfH — fn 0x0c46bf0"},
    {0x014c1390, Kind::W_f,     Group::F3, "F3.W     — fn 0x0c46bf0"},
    {0x014c1394, Kind::H_f,     Group::F3, "F3.H     — fn 0x0c46bf0"},

    // ── Group G: per-entity screen-space transform — fn 0xfdf9d0 callers ──
    //  145 unique call sites of fn 0xfdf9d0 (the per-entity unprojection /
    //  pick / frustum-corner builder), each setting `mov ecx, 0x780` (W)
    //  and `mov r8d, 0x438` (H) before calling. After patching, each entity
    //  computes its screen-space math against the new resolution, eliminating
    //  the per-entity desync that all-37 patches alone could not fix.
    //  v5: 276 sites (was 218 in v4) — call-search window extended from
    //  64 to 256 bytes to catch sites where vpermilps/vmulps math interleaves
    //  between the W/H prep and the call.
    {0x00dc90be, Kind::W_u32, Group::G1, "G1.W (instr 0x00dc90bd)"},
    {0x00dc90c4, Kind::H_u32, Group::G1, "G1.H (instr 0x00dc90c2)"},
    {0x00dd1a40, Kind::W_u32, Group::G1, "G1.W (instr 0x00dd1a3f)"},
    {0x00dd1a46, Kind::H_u32, Group::G1, "G1.H (instr 0x00dd1a44)"},
    {0x00df08be, Kind::W_u32, Group::G1, "G1.W (instr 0x00df08bd)"},
    {0x00df08c4, Kind::H_u32, Group::G1, "G1.H (instr 0x00df08c2)"},
    {0x00df0c20, Kind::W_u32, Group::G1, "G1.W (instr 0x00df0c1f)"},
    {0x00df0c26, Kind::H_u32, Group::G1, "G1.H (instr 0x00df0c24)"},
    {0x00e1c5e4, Kind::W_u32, Group::G1, "G1.W (instr 0x00e1c5e3)"},
    {0x00e1c5ea, Kind::H_u32, Group::G1, "G1.H (instr 0x00e1c5e8)"},
    {0x00e1caf4, Kind::W_u32, Group::G1, "G1.W (instr 0x00e1caf3)"},
    {0x00e1cafa, Kind::H_u32, Group::G1, "G1.H (instr 0x00e1caf8)"},
    {0x00e1cf04, Kind::W_u32, Group::G1, "G1.W (instr 0x00e1cf03)"},
    {0x00e1cf0a, Kind::H_u32, Group::G1, "G1.H (instr 0x00e1cf08)"},
    {0x00e1d9a4, Kind::W_u32, Group::G1, "G1.W (instr 0x00e1d9a3)"},
    {0x00e1d9aa, Kind::H_u32, Group::G1, "G1.H (instr 0x00e1d9a8)"},
    {0x00e1ed24, Kind::W_u32, Group::G1, "G1.W (instr 0x00e1ed23)"},
    {0x00e1ed2a, Kind::H_u32, Group::G1, "G1.H (instr 0x00e1ed28)"},
    {0x00e20914, Kind::W_u32, Group::G1, "G1.W (instr 0x00e20913)"},
    {0x00e2091a, Kind::H_u32, Group::G1, "G1.H (instr 0x00e20918)"},
    {0x00e225e2, Kind::W_u32, Group::G1, "G1.W (instr 0x00e225e1)"},
    {0x00e225e8, Kind::H_u32, Group::G1, "G1.H (instr 0x00e225e6)"},
    {0x00e23374, Kind::W_u32, Group::G1, "G1.W (instr 0x00e23373)"},
    {0x00e2337a, Kind::H_u32, Group::G1, "G1.H (instr 0x00e23378)"},
    {0x00e25ee3, Kind::W_u32, Group::G1, "G1.W (instr 0x00e25ee2)"},
    {0x00e25ee9, Kind::H_u32, Group::G1, "G1.H (instr 0x00e25ee7)"},
    {0x00e2803c, Kind::W_u32, Group::G1, "G1.W (instr 0x00e2803b)"},
    {0x00e28042, Kind::H_u32, Group::G1, "G1.H (instr 0x00e28040)"},
    {0x00e295cf, Kind::W_u32, Group::G1, "G1.W (instr 0x00e295ce)"},
    {0x00e295d5, Kind::H_u32, Group::G1, "G1.H (instr 0x00e295d3)"},
    {0x00e2bd90, Kind::W_u32, Group::G1, "G1.W (instr 0x00e2bd8f)"},
    {0x00e2bd96, Kind::H_u32, Group::G1, "G1.H (instr 0x00e2bd94)"},
    {0x00e2cc47, Kind::W_u32, Group::G1, "G1.W (instr 0x00e2cc46)"},
    {0x00e2cc4d, Kind::H_u32, Group::G1, "G1.H (instr 0x00e2cc4b)"},
    {0x00e302ed, Kind::W_u32, Group::G1, "G1.W (instr 0x00e302ec)"},
    {0x00e302f3, Kind::H_u32, Group::G1, "G1.H (instr 0x00e302f1)"},
    {0x00e312ee, Kind::W_u32, Group::G1, "G1.W (instr 0x00e312ed)"},
    {0x00e312f4, Kind::H_u32, Group::G1, "G1.H (instr 0x00e312f2)"},
    {0x00e44163, Kind::W_u32, Group::G1, "G1.W (instr 0x00e44162)"},
    {0x00e44169, Kind::H_u32, Group::G1, "G1.H (instr 0x00e44167)"},
    {0x00e455d7, Kind::W_u32, Group::G1, "G1.W (instr 0x00e455d6)"},
    {0x00e455dd, Kind::H_u32, Group::G1, "G1.H (instr 0x00e455db)"},
    {0x00e4c734, Kind::W_u32, Group::G1, "G1.W (instr 0x00e4c733)"},
    {0x00e4c73a, Kind::H_u32, Group::G1, "G1.H (instr 0x00e4c738)"},
    {0x00e5beea, Kind::W_u32, Group::G1, "G1.W (instr 0x00e5bee9)"},
    {0x00e5bef0, Kind::H_u32, Group::G1, "G1.H (instr 0x00e5beee)"},
    {0x00e5d5c0, Kind::W_u32, Group::G1, "G1.W (instr 0x00e5d5bf)"},
    {0x00e5d5c6, Kind::H_u32, Group::G1, "G1.H (instr 0x00e5d5c4)"},
    {0x00e5d5e5, Kind::W_u32, Group::G1, "G1.W (instr 0x00e5d5e4)"},
    {0x00e5d5eb, Kind::H_u32, Group::G1, "G1.H (instr 0x00e5d5e9)"},
    {0x00e5d60a, Kind::W_u32, Group::G1, "G1.W (instr 0x00e5d609)"},
    {0x00e6e384, Kind::W_u32, Group::G1, "G1.W (instr 0x00e6e383)"},
    {0x00e6e38a, Kind::H_u32, Group::G1, "G1.H (instr 0x00e6e388)"},
    {0x00e6e834, Kind::W_u32, Group::G1, "G1.W (instr 0x00e6e833)"},
    {0x00e6e83a, Kind::H_u32, Group::G1, "G1.H (instr 0x00e6e838)"},
    {0x00e6ecd4, Kind::W_u32, Group::G1, "G1.W (instr 0x00e6ecd3)"},
    {0x00e6ecda, Kind::H_u32, Group::G1, "G1.H (instr 0x00e6ecd8)"},
    {0x00e93251, Kind::W_u32, Group::G1, "G1.W (instr 0x00e93250)"},
    {0x00e93257, Kind::H_u32, Group::G1, "G1.H (instr 0x00e93255)"},
    {0x00e93276, Kind::W_u32, Group::G1, "G1.W (instr 0x00e93275)"},
    {0x00e9327c, Kind::H_u32, Group::G1, "G1.H (instr 0x00e9327a)"},
    {0x00e9329b, Kind::W_u32, Group::G1, "G1.W (instr 0x00e9329a)"},
    {0x00e932a1, Kind::H_u32, Group::G1, "G1.H (instr 0x00e9329f)"},
    {0x00e932c0, Kind::W_u32, Group::G1, "G1.W (instr 0x00e932bf)"},
    {0x00e932e9, Kind::W_u32, Group::G1, "G1.W (instr 0x00e932e8)"},
    {0x00e932ef, Kind::H_u32, Group::G1, "G1.H (instr 0x00e932ed)"},
    {0x00e934c8, Kind::W_u32, Group::G1, "G1.W (instr 0x00e934c7)"},
    {0x00e934ce, Kind::H_u32, Group::G1, "G1.H (instr 0x00e934cc)"},
    {0x00e934ea, Kind::W_u32, Group::G1, "G1.W (instr 0x00e934e9)"},
    {0x00e934f0, Kind::H_u32, Group::G1, "G1.H (instr 0x00e934ee)"},
    {0x00e9350c, Kind::W_u32, Group::G1, "G1.W (instr 0x00e9350b)"},
    {0x00e97ac0, Kind::W_u32, Group::G1, "G1.W (instr 0x00e97abf)"},
    {0x00e97ac6, Kind::H_u32, Group::G1, "G1.H (instr 0x00e97ac4)"},
    {0x00e99d6d, Kind::W_u32, Group::G1, "G1.W (instr 0x00e99d6c)"},
    {0x00e99d73, Kind::H_u32, Group::G1, "G1.H (instr 0x00e99d71)"},
    {0x00e99d92, Kind::W_u32, Group::G1, "G1.W (instr 0x00e99d91)"},
    {0x00e99dbc, Kind::W_u32, Group::G1, "G1.W (instr 0x00e99dbb)"},
    {0x00e99dc2, Kind::H_u32, Group::G1, "G1.H (instr 0x00e99dc0)"},
    {0x00e99de2, Kind::W_u32, Group::G1, "G1.W (instr 0x00e99de1)"},
    {0x00e99de8, Kind::H_u32, Group::G1, "G1.H (instr 0x00e99de6)"},
    {0x00e9a895, Kind::W_u32, Group::G1, "G1.W (instr 0x00e9a894)"},
    {0x00e9a89b, Kind::H_u32, Group::G1, "G1.H (instr 0x00e9a899)"},
    {0x00e9adb2, Kind::W_u32, Group::G1, "G1.W (instr 0x00e9adb1)"},
    {0x00e9adb8, Kind::H_u32, Group::G1, "G1.H (instr 0x00e9adb6)"},
    {0x00e9b2b2, Kind::W_u32, Group::G1, "G1.W (instr 0x00e9b2b1)"},
    {0x00e9b2b8, Kind::H_u32, Group::G1, "G1.H (instr 0x00e9b2b6)"},
    {0x00e9b41d, Kind::W_u32, Group::G1, "G1.W (instr 0x00e9b41c)"},
    {0x00e9b423, Kind::H_u32, Group::G1, "G1.H (instr 0x00e9b421)"},
    {0x00e9b5c6, Kind::W_u32, Group::G1, "G1.W (instr 0x00e9b5c5)"},
    {0x00e9b5cc, Kind::H_u32, Group::G1, "G1.H (instr 0x00e9b5ca)"},
    {0x00eb1f90, Kind::W_u32, Group::G1, "G1.W (instr 0x00eb1f8f)"},
    {0x00eb1f96, Kind::H_u32, Group::G1, "G1.H (instr 0x00eb1f94)"},
    {0x00ec8ef4, Kind::W_u32, Group::G1, "G1.W (instr 0x00ec8ef3)"},
    {0x00ec8efa, Kind::H_u32, Group::G1, "G1.H (instr 0x00ec8ef8)"},
    {0x00ee0224, Kind::W_u32, Group::G1, "G1.W (instr 0x00ee0223)"},
    {0x00ee022a, Kind::H_u32, Group::G1, "G1.H (instr 0x00ee0228)"},
    {0x00eea754, Kind::W_u32, Group::G1, "G1.W (instr 0x00eea753)"},
    {0x00eea75a, Kind::H_u32, Group::G1, "G1.H (instr 0x00eea758)"},
    {0x00ef0a90, Kind::W_u32, Group::G1, "G1.W (instr 0x00ef0a8f)"},
    {0x00ef0a96, Kind::H_u32, Group::G1, "G1.H (instr 0x00ef0a94)"},
    {0x00ef0ab5, Kind::W_u32, Group::G1, "G1.W (instr 0x00ef0ab4)"},
    {0x00ef0abb, Kind::H_u32, Group::G1, "G1.H (instr 0x00ef0ab9)"},
    {0x00ef0ada, Kind::W_u32, Group::G1, "G1.W (instr 0x00ef0ad9)"},
    {0x00ef0b05, Kind::W_u32, Group::G1, "G1.W (instr 0x00ef0b04)"},
    {0x00ef0b0b, Kind::H_u32, Group::G1, "G1.H (instr 0x00ef0b09)"},
    {0x00f0d254, Kind::W_u32, Group::G1, "G1.W (instr 0x00f0d253)"},
    {0x00f0d25a, Kind::H_u32, Group::G1, "G1.H (instr 0x00f0d258)"},
    {0x00f0fed2, Kind::W_u32, Group::G1, "G1.W (instr 0x00f0fed1)"},
    {0x00f0fed8, Kind::H_u32, Group::G1, "G1.H (instr 0x00f0fed6)"},
    {0x00f4b6f8, Kind::W_u32, Group::G1, "G1.W (instr 0x00f4b6f7)"},
    {0x00f4b6fe, Kind::H_u32, Group::G1, "G1.H (instr 0x00f4b6fc)"},
    {0x00f4b8da, Kind::W_u32, Group::G1, "G1.W (instr 0x00f4b8d9)"},
    {0x00f4b8e0, Kind::H_u32, Group::G1, "G1.H (instr 0x00f4b8de)"},
    {0x00f4b900, Kind::W_u32, Group::G1, "G1.W (instr 0x00f4b8ff)"},
    {0x00f4b906, Kind::H_u32, Group::G1, "G1.H (instr 0x00f4b904)"},
    {0x00f4b926, Kind::W_u32, Group::G1, "G1.W (instr 0x00f4b925)"},
    {0x00f4b92c, Kind::H_u32, Group::G1, "G1.H (instr 0x00f4b92a)"},
    {0x00f4db54, Kind::W_u32, Group::G1, "G1.W (instr 0x00f4db53)"},
    {0x00f4db5a, Kind::H_u32, Group::G1, "G1.H (instr 0x00f4db58)"},
    {0x00f4de30, Kind::W_u32, Group::G1, "G1.W (instr 0x00f4de2f)"},
    {0x00f4de36, Kind::H_u32, Group::G1, "G1.H (instr 0x00f4de34)"},
    {0x00f525bd, Kind::W_u32, Group::G1, "G1.W (instr 0x00f525bc)"},
    {0x00f525c3, Kind::H_u32, Group::G1, "G1.H (instr 0x00f525c1)"},
    {0x00f58583, Kind::W_u32, Group::G1, "G1.W (instr 0x00f58582)"},
    {0x00f58589, Kind::H_u32, Group::G1, "G1.H (instr 0x00f58587)"},
    {0x00f58775, Kind::W_u32, Group::G1, "G1.W (instr 0x00f58774)"},
    {0x00f5877b, Kind::H_u32, Group::G1, "G1.H (instr 0x00f58779)"},
    {0x00f5879a, Kind::W_u32, Group::G1, "G1.W (instr 0x00f58799)"},
    {0x00f587a0, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5879e)"},
    {0x00f587bf, Kind::W_u32, Group::G1, "G1.W (instr 0x00f587be)"},
    {0x00f587f3, Kind::W_u32, Group::G1, "G1.W (instr 0x00f587f2)"},
    {0x00f587f9, Kind::H_u32, Group::G1, "G1.H (instr 0x00f587f7)"},
    {0x00f58818, Kind::W_u32, Group::G1, "G1.W (instr 0x00f58817)"},
    {0x00f5881e, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5881c)"},
    {0x00f5883d, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5883c)"},
    {0x00f58843, Kind::H_u32, Group::G1, "G1.H (instr 0x00f58841)"},
    {0x00f58862, Kind::W_u32, Group::G1, "G1.W (instr 0x00f58861)"},
    {0x00f58868, Kind::H_u32, Group::G1, "G1.H (instr 0x00f58866)"},
    {0x00f58887, Kind::W_u32, Group::G1, "G1.W (instr 0x00f58886)"},
    {0x00f5888d, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5888b)"},
    {0x00f588ac, Kind::W_u32, Group::G1, "G1.W (instr 0x00f588ab)"},
    {0x00f588b2, Kind::H_u32, Group::G1, "G1.H (instr 0x00f588b0)"},
    {0x00f588d1, Kind::W_u32, Group::G1, "G1.W (instr 0x00f588d0)"},
    {0x00f588d7, Kind::H_u32, Group::G1, "G1.H (instr 0x00f588d5)"},
    {0x00f588f6, Kind::W_u32, Group::G1, "G1.W (instr 0x00f588f5)"},
    {0x00f588fc, Kind::H_u32, Group::G1, "G1.H (instr 0x00f588fa)"},
    {0x00f5891b, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5891a)"},
    {0x00f58921, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5891f)"},
    {0x00f58940, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5893f)"},
    {0x00f5acdd, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5acdc)"},
    {0x00f5ace3, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5ace1)"},
    {0x00f5ad07, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5ad06)"},
    {0x00f5ad0d, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5ad0b)"},
    {0x00f5ad32, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5ad31)"},
    {0x00f5ad38, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5ad36)"},
    {0x00f5ad5d, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5ad5c)"},
    {0x00f5ad63, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5ad61)"},
    {0x00f5bb58, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5bb57)"},
    {0x00f5bb5e, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5bb5c)"},
    {0x00f5c34b, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c34a)"},
    {0x00f5c351, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c34f)"},
    {0x00f5c377, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c376)"},
    {0x00f5c37d, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c37b)"},
    {0x00f5c556, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c555)"},
    {0x00f5c55c, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c55a)"},
    {0x00f5c578, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c577)"},
    {0x00f5c57e, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c57c)"},
    {0x00f5c59a, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c599)"},
    {0x00f5c5a0, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c59e)"},
    {0x00f5c5bc, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c5bb)"},
    {0x00f5c5c2, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c5c0)"},
    {0x00f5c5de, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c5dd)"},
    {0x00f5c5e4, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c5e2)"},
    {0x00f5c603, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c602)"},
    {0x00f5c62f, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c62e)"},
    {0x00f5c635, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c633)"},
    {0x00f5c820, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c81f)"},
    {0x00f5c826, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c824)"},
    {0x00f5c848, Kind::W_u32, Group::G1, "G1.W (instr 0x00f5c847)"},
    {0x00f5c84e, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c84c)"},
    {0x00f652ff, Kind::W_u32, Group::G1, "G1.W (instr 0x00f652fe)"},
    {0x00f65305, Kind::H_u32, Group::G1, "G1.H (instr 0x00f65303)"},
    {0x00f6532a, Kind::W_u32, Group::G1, "G1.W (instr 0x00f65329)"},
    {0x00f65330, Kind::H_u32, Group::G1, "G1.H (instr 0x00f6532e)"},
    {0x00f65355, Kind::W_u32, Group::G1, "G1.W (instr 0x00f65354)"},
    {0x00f6535b, Kind::H_u32, Group::G1, "G1.H (instr 0x00f65359)"},
    {0x00f6537c, Kind::W_u32, Group::G1, "G1.W (instr 0x00f6537b)"},
    {0x00f65382, Kind::H_u32, Group::G1, "G1.H (instr 0x00f65380)"},
    {0x00f653a3, Kind::W_u32, Group::G1, "G1.W (instr 0x00f653a2)"},
    {0x00f653a9, Kind::H_u32, Group::G1, "G1.H (instr 0x00f653a7)"},
    {0x00f653ca, Kind::W_u32, Group::G1, "G1.W (instr 0x00f653c9)"},
    {0x00f653d0, Kind::H_u32, Group::G1, "G1.H (instr 0x00f653ce)"},
    {0x00f65997, Kind::W_u32, Group::G1, "G1.W (instr 0x00f65996)"},
    {0x00f6599d, Kind::H_u32, Group::G1, "G1.H (instr 0x00f6599b)"},
    {0x00f659c3, Kind::W_u32, Group::G1, "G1.W (instr 0x00f659c2)"},
    {0x00f659c9, Kind::H_u32, Group::G1, "G1.H (instr 0x00f659c7)"},
    {0x00f65ba2, Kind::W_u32, Group::G1, "G1.W (instr 0x00f65ba1)"},
    {0x00f65ba8, Kind::H_u32, Group::G1, "G1.H (instr 0x00f65ba6)"},
    {0x00f65bc4, Kind::W_u32, Group::G1, "G1.W (instr 0x00f65bc3)"},
    {0x00f65bca, Kind::H_u32, Group::G1, "G1.H (instr 0x00f65bc8)"},
    {0x00f65be6, Kind::W_u32, Group::G1, "G1.W (instr 0x00f65be5)"},
    {0x00f66f98, Kind::W_u32, Group::G1, "G1.W (instr 0x00f66f97)"},
    {0x00f66f9e, Kind::H_u32, Group::G1, "G1.H (instr 0x00f66f9c)"},
    {0x00f68288, Kind::W_u32, Group::G1, "G1.W (instr 0x00f68287)"},
    {0x00f6828e, Kind::H_u32, Group::G1, "G1.H (instr 0x00f6828c)"},
    {0x00f682b4, Kind::W_u32, Group::G1, "G1.W (instr 0x00f682b3)"},
    {0x00f682ba, Kind::H_u32, Group::G1, "G1.H (instr 0x00f682b8)"},
    {0x00f68493, Kind::W_u32, Group::G1, "G1.W (instr 0x00f68492)"},
    {0x00f68499, Kind::H_u32, Group::G1, "G1.H (instr 0x00f68497)"},
    {0x00f684b5, Kind::W_u32, Group::G1, "G1.W (instr 0x00f684b4)"},
    {0x00f684bb, Kind::H_u32, Group::G1, "G1.H (instr 0x00f684b9)"},
    {0x00f684d7, Kind::W_u32, Group::G1, "G1.W (instr 0x00f684d6)"},
    {0x00f684dd, Kind::H_u32, Group::G1, "G1.H (instr 0x00f684db)"},
    {0x00f684f9, Kind::W_u32, Group::G1, "G1.W (instr 0x00f684f8)"},
    {0x00f684ff, Kind::H_u32, Group::G1, "G1.H (instr 0x00f684fd)"},
    {0x00f6851b, Kind::W_u32, Group::G1, "G1.W (instr 0x00f6851a)"},
    {0x00f68521, Kind::H_u32, Group::G1, "G1.H (instr 0x00f6851f)"},
    {0x00f6853d, Kind::W_u32, Group::G1, "G1.W (instr 0x00f6853c)"},
    {0x00f68543, Kind::H_u32, Group::G1, "G1.H (instr 0x00f68541)"},
    {0x00f6855f, Kind::W_u32, Group::G1, "G1.W (instr 0x00f6855e)"},
    {0x00f68565, Kind::H_u32, Group::G1, "G1.H (instr 0x00f68563)"},
    {0x00f68581, Kind::W_u32, Group::G1, "G1.W (instr 0x00f68580)"},
    {0x00f685aa, Kind::W_u32, Group::G1, "G1.W (instr 0x00f685a9)"},
    {0x00f685b0, Kind::H_u32, Group::G1, "G1.H (instr 0x00f685ae)"},
    {0x00f68798, Kind::W_u32, Group::G1, "G1.W (instr 0x00f68797)"},
    {0x00f6879e, Kind::H_u32, Group::G1, "G1.H (instr 0x00f6879c)"},
    {0x00f687bd, Kind::W_u32, Group::G1, "G1.W (instr 0x00f687bc)"},
    {0x00f6eecd, Kind::W_u32, Group::G1, "G1.W (instr 0x00f6eecc)"},
    {0x00f6eed3, Kind::H_u32, Group::G1, "G1.H (instr 0x00f6eed1)"},
    {0x00f6eef2, Kind::W_u32, Group::G1, "G1.W (instr 0x00f6eef1)"},
    {0x00f71a38, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71a37)"},
    {0x00f71a3e, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71a3c)"},
    {0x00f71a64, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71a63)"},
    {0x00f71a6a, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71a68)"},
    {0x00f71c43, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71c42)"},
    {0x00f71c49, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71c47)"},
    {0x00f71c65, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71c64)"},
    {0x00f71c6b, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71c69)"},
    {0x00f71c87, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71c86)"},
    {0x00f71c8d, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71c8b)"},
    {0x00f71ca9, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71ca8)"},
    {0x00f71cd2, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71cd1)"},
    {0x00f71cd8, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71cd6)"},
    {0x00f71ec0, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71ebf)"},
    {0x00f71ec6, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71ec4)"},
    {0x00f71ee5, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71ee4)"},
    {0x00f88c0d, Kind::W_u32, Group::G1, "G1.W (instr 0x00f88c0c)"},
    {0x00f88c13, Kind::H_u32, Group::G1, "G1.H (instr 0x00f88c11)"},
    {0x00f89825, Kind::W_u32, Group::G1, "G1.W (instr 0x00f89824)"},
    {0x00f8982b, Kind::H_u32, Group::G1, "G1.H (instr 0x00f89829)"},
    {0x01324d9a, Kind::W_u32, Group::G1, "G1.W (instr 0x01324d99)"},
    {0x01324da0, Kind::H_u32, Group::G1, "G1.H (instr 0x01324d9e)"},
    {0x01326a74, Kind::W_u32, Group::G1, "G1.W (instr 0x01326a73)"},
    {0x01326a7a, Kind::H_u32, Group::G1, "G1.H (instr 0x01326a78)"},
    {0x0132ab44, Kind::W_u32, Group::G1, "G1.W (instr 0x0132ab43)"},
    {0x0132ab4a, Kind::H_u32, Group::G1, "G1.H (instr 0x0132ab48)"},
    {0x0132de5b, Kind::W_u32, Group::G1, "G1.W (instr 0x0132de5a)"},
    {0x0132de61, Kind::H_u32, Group::G1, "G1.H (instr 0x0132de5f)"},
    {0x0132f52c, Kind::W_u32, Group::G1, "G1.W (instr 0x0132f52b)"},
    {0x0132f532, Kind::H_u32, Group::G1, "G1.H (instr 0x0132f530)"},
    {0x01330f19, Kind::W_u32, Group::G1, "G1.W (instr 0x01330f18)"},
    {0x01330f1f, Kind::H_u32, Group::G1, "G1.H (instr 0x01330f1d)"},
    {0x0133c865, Kind::W_u32, Group::G1, "G1.W (instr 0x0133c864)"},
    {0x0133c86b, Kind::H_u32, Group::G1, "G1.H (instr 0x0133c869)"},
    {0x01345e70, Kind::W_u32, Group::G1, "G1.W (instr 0x01345e6f)"},
    {0x01345e76, Kind::H_u32, Group::G1, "G1.H (instr 0x01345e74)"},
    {0x0134e96d, Kind::W_u32, Group::G1, "G1.W (instr 0x0134e96c)"},
    {0x0134e973, Kind::H_u32, Group::G1, "G1.H (instr 0x0134e971)"},
    {0x0136d380, Kind::W_u32, Group::G1, "G1.W (instr 0x0136d37f)"},
    {0x0136d386, Kind::H_u32, Group::G1, "G1.H (instr 0x0136d384)"},
    {0x01372477, Kind::W_u32, Group::G1, "G1.W (instr 0x01372476)"},
    {0x0137247d, Kind::H_u32, Group::G1, "G1.H (instr 0x0137247b)"},
    {0x0139ee17, Kind::W_u32, Group::G1, "G1.W (instr 0x0139ee16)"},
    {0x0139ee1d, Kind::H_u32, Group::G1, "G1.H (instr 0x0139ee1b)"},
    {0x013a8c39, Kind::W_u32, Group::G1, "G1.W (instr 0x013a8c38)"},
    {0x013a8c3f, Kind::H_u32, Group::G1, "G1.H (instr 0x013a8c3d)"},

    // ── Group G: per-entity screen-space transform — fn 0xfdf9d0 callers ──
    //  109 sites, each setting `mov ecx, 0x780` (W=1920) and `mov r8d, 0x438`
    //  (H=1080) immediately before `call 0xfdf9d0`. The helper takes (W,H)
    //  plus a 4×4 viewproj matrix and computes a per-object screen-space
    //  unprojection / pick / frustum transform, written to [r15+0xC0..0xF0].
    //  Without these patches, every entity's screen-space math believes the
    //  output target is 1920×1080 even when we render at lower res, which
    //  desynchronizes LOD selection, culling, billboarding, and any
    //  screen-aligned effects (decals, lens flares, marker icons).
    //  Generated empirically: search-and-confirm of all CALL/JMP rel32
    //  instructions targeting 0xfdf9d0 (146 found) intersected with the
    //  immediately-preceding ecx=W, r8d=H mov-imm32 prep (109 unique sites).

    // ── Group H1: companion main RT (W,H) pair in fn 0x102a5b0 ─────────
    //  Inside the same function as B1, immediately before a second call
    //  to fn 0x1218370 (the multi-purpose RT/subsystem registrar). Loads
    //  edx=W, ecx=H from imm32 — same call shape as B1's struct-store
    //  pair at +0x92C0/+0x92C4 ~12 bytes later. Patches the dim values
    //  passed as call arguments to the registrar.
    {0x0102aa0b, Kind::W_u32, Group::H1, "H1.W (mov edx, 0x780)         — fn 0x102a5b0 MAIN, pre-call 0x1218370"},
    {0x0102aa10, Kind::H_u32, Group::H1, "H1.H (mov ecx, 0x438)         — fn 0x102a5b0 MAIN, pre-call 0x1218370"},

    // ── Group H2: quarter-resolution sub-RT registrations (8 pairs) ────
    //  Eight (W=480, H=270) dim pairs passed to fn 0x1218370, four with
    //  subsystem-ID esi=3 (clustered around fn 0xd9fxxx — shadow / SSR
    //  half-mip ladder candidate), four with esi=6 (fn 0x102exxx — bloom
    //  / post-process quarter-res buffer candidate). At 1080p these are
    //  480x270; at any other resolution they MUST scale (final/4) or
    //  the engine's quarter-res RTs stay at 1080p-quarter while the main
    //  RT is at the chosen res — exactly the "environment desync"
    //  signature the user observed even with v5's full 313-patch mask.
    //
    //  Confidence: HIGH. Same call target as B1 (0x1218370). Different
    //  subsystem IDs from B1's sid=4, so they don't overlap with B1's
    //  main-RT path; they cover distinct downsampled-RT subsystems.
    //  Empirical baseline verified: all 16 bytes match 0x1E0 / 0x10E.
    {0x00d9fa4f, Kind::QuarterW_u32, Group::H2, "H2.W (mov edx, 0x1e0)  — fn 0xd9fa00 sid=3 #1 quarter-res RT"},
    {0x00d9fa54, Kind::QuarterH_u32, Group::H2, "H2.H (mov ecx, 0x10e)  — fn 0xd9fa00 sid=3 #1 quarter-res RT"},
    {0x00d9faa7, Kind::QuarterW_u32, Group::H2, "H2.W (mov edx, 0x1e0)  — fn 0xd9fa00 sid=3 #2 quarter-res RT"},
    {0x00d9faac, Kind::QuarterH_u32, Group::H2, "H2.H (mov ecx, 0x10e)  — fn 0xd9fa00 sid=3 #2 quarter-res RT"},
    {0x00d9faff, Kind::QuarterW_u32, Group::H2, "H2.W (mov edx, 0x1e0)  — fn 0xd9fa00 sid=3 #3 quarter-res RT"},
    {0x00d9fb04, Kind::QuarterH_u32, Group::H2, "H2.H (mov ecx, 0x10e)  — fn 0xd9fa00 sid=3 #3 quarter-res RT"},
    {0x00d9fb57, Kind::QuarterW_u32, Group::H2, "H2.W (mov edx, 0x1e0)  — fn 0xd9fa00 sid=3 #4 quarter-res RT"},
    {0x00d9fb5c, Kind::QuarterH_u32, Group::H2, "H2.H (mov ecx, 0x10e)  — fn 0xd9fa00 sid=3 #4 quarter-res RT"},
    {0x0102dff8, Kind::QuarterW_u32, Group::H2, "H2.W (mov edx, 0x1e0)  — fn 0x102dfxxx sid=6 #1 quarter-res RT"},
    {0x0102dffd, Kind::QuarterH_u32, Group::H2, "H2.H (mov ecx, 0x10e)  — fn 0x102dfxxx sid=6 #1 quarter-res RT"},
    {0x0102e023, Kind::QuarterW_u32, Group::H2, "H2.W (mov edx, 0x1e0)  — fn 0x102dfxxx sid=6 #2 quarter-res RT"},
    {0x0102e028, Kind::QuarterH_u32, Group::H2, "H2.H (mov ecx, 0x10e)  — fn 0x102dfxxx sid=6 #2 quarter-res RT"},
    {0x0102e070, Kind::QuarterW_u32, Group::H2, "H2.W (mov edx, 0x1e0)  — fn 0x102dfxxx sid=6 #3 quarter-res RT"},
    {0x0102e075, Kind::QuarterH_u32, Group::H2, "H2.H (mov ecx, 0x10e)  — fn 0x102dfxxx sid=6 #3 quarter-res RT"},
    {0x0102e0d5, Kind::QuarterW_u32, Group::H2, "H2.W (mov edx, 0x1e0)  — fn 0x102dfxxx sid=6 #4 quarter-res RT"},
    {0x0102e0da, Kind::QuarterH_u32, Group::H2, "H2.H (mov ecx, 0x10e)  — fn 0x102dfxxx sid=6 #4 quarter-res RT"},

    // ── Group H3: 4K-selector 1080p-default-branch (W,H) pair ──────────
    //  At the 4K-toggle entry at 0xd9fa00, two values feed a cmovne:
    //  (1920,1080) on the default branch and (3840,2160) on the 4K
    //  branch. The handoff classifies this whole site as INERT — the
    //  downstream consumer is the TLS-protected registrar inside fn
    //  0x1218370, not a Gnm RT allocator. Force_4k logs show the bytes
    //  rewrite cleanly but no framebuffer changes. Provided for
    //  empirical re-test only; OFF by default in all curated presets.
    //  Only the 1080p branch dimensions are patched here (the 4K-branch
    //  values 3840/2160 sit on an unreachable code path at any non-4K
    //  target resolution and would never be selected by cmovne).
    {0x00d9fa07, Kind::W_u32, Group::H3, "H3.W (mov r14d, 0x780)        — 4K-selector default-branch (OFF default)"},
    {0x00d9fa11, Kind::H_u32, Group::H3, "H3.H (mov ebx, 0x438)         — 4K-selector default-branch (OFF default)"},

    // ── Group H4: IMUL-imm32 dim pairs at 4K-toggle entry fn ───────────
    //  Two `imul reg, eax, 0x780 / 0x438` instructions inside fn
    //  0x45d8ea / 0x45d9e3 — 6-byte IMUL encoding (69 ModRM imm32),
    //  imm32 at instr_va+2. The `eax` factor here is a 1-or-2 mode flag
    //  (the same 4K toggle that drives H3). LOW confidence — pursued
    //  only to document, OFF by default. Same caveat as H3: downstream
    //  of fn 0x1218370 is the TLS registrar, not the RT allocator.
    {0x0045d8e0, Kind::W_u32, Group::H4, "H4.W (imul edx, eax, 0x780)   — 4K-toggle entry #1 (OFF default)"},
    {0x0045d8e6, Kind::H_u32, Group::H4, "H4.H (imul ecx, eax, 0x438)   — 4K-toggle entry #1 (OFF default)"},
    {0x0045d9d9, Kind::W_u32, Group::H4, "H4.W (imul edx, eax, 0x780)   — 4K-toggle entry #2 (OFF default)"},
    {0x0045d9df, Kind::H_u32, Group::H4, "H4.H (imul ecx, eax, 0x438)   — 4K-toggle entry #2 (OFF default)"},
    // ── v7.4: I2 — UI ortho projection scale ───────────────────────────
    {0x014c565c, Kind::InvHalfW_f, Group::I2, "I2.X (2/1920 = 1/960) — ortho X scale, vmulss in fn 0x10d3b60"},
    {0x014c5664, Kind::InvHalfH_f, Group::I2, "I2.Y (2/1080 = 1/540) — ortho Y scale, vmulss in fn 0x10d3b60"},
    // ── v7.5: I3 — UI viewport SCALE row of 4-vec4 descriptor ──────────
    {0x01499ec0, Kind::HalfW_f, Group::I3, "I3.W (960.0f) — viewport scale.x, vmovaps at 0x00EABC8D → fn 0x110C110"},
    {0x01499ec4, Kind::HalfH_f, Group::I3, "I3.H (540.0f) — viewport scale.y, vmovaps at 0x00EABC8D → fn 0x110C110"},
    // ── v8: I4 — Candidate-A (960,540,0,0) vec4 at 0x0149A190 ──────────
    //  Identified by v7.5 scan as the SECOND heavily-consumed (960,540)
    //  pair in rodata after I3. Loaded by ≥8 vmovaps in fn boundary
    //  0xEBC8xx..0xEBD9xx. Struct context (0x0149A180..0x0149A1C0) looks
    //  like a 5-row 4-vec viewport-or-math descriptor. Single-function
    //  blast radius. OFF in `recommended` — opt in explicitly to test.
    {0x0149a190, Kind::HalfW_f, Group::I4, "I4.W (960.0f) — Candidate-A vec4 at 0x0149A190 (vmovaps in fn 0xEBC8xx)"},
    {0x0149a194, Kind::HalfH_f, Group::I4, "I4.H (540.0f) — Candidate-A vec4 at 0x0149A194 (vmovaps in fn 0xEBC8xx)"},

    // ── v9: J1 — flip-Y mirror vecs paired with C1/C2/C3 ───────────────
    //  Three (1920, -1080, -999.99, -1080) vecs sit at exactly C_VA - 0x20
    //  in rodata. C2's consumer fn 0x0110c110 loads the J-sibling at
    //  0x0110c474 (vmovaps) then mul's by the C-vec at 0x0110c4b9
    //  (vmulps xmm0, xmm0, [C-vec]) — 0x45 bytes apart, same function.
    //  Same pattern at C3's consumer fn 0x0110d140 (J load at 0x0110d854,
    //  C mul at 0x0110d899). The product is component-wise; patching the
    //  C-vec alone leaves the mul mismatched (one operand at target res,
    //  other at 1080p) — which is consistent with C2/C3's documented UI-
    //  text-corruption-when-solo behavior.
    //
    //  J1 MUST be tested together with the matching C-group. Solo J1
    //  leaves the same mismatch in the other direction. v16 note: the
    //  user-facing group-toggle config was removed, so J1 testing now
    //  requires temporarily editing kGroupMaskRecommended (in the .h)
    //  to include J1+C2+C3 — see the v11 finding that those C-groups
    //  break UI text positioning; pair-testing would require reverting
    //  that exclusion.
    //
    //  Each J-vec has 3 patch sites: W at +0, NegH at +4, NegH dup at +12.
    //  Offset +8 is -999.99f (depth-far mirror of C's +1000.01) — NOT
    //  patched (not a dim).
    {0x014985b0, Kind::W_f,    Group::J1, "J1.C1.W  (+1920.0f) — flip-Y sibling of C1 at -0x20 (fn 0x136d74d)"},
    {0x014985b4, Kind::NegH_f, Group::J1, "J1.C1.-H (-1080.0f) — flip-Y sibling of C1"},
    {0x014985bc, Kind::NegH_f, Group::J1, "J1.C1.-H_dup (-1080.0f) — flip-Y sibling of C1 (+12)"},
    {0x014a0f30, Kind::W_f,    Group::J1, "J1.C2.W  (+1920.0f) — flip-Y sibling of C2 at -0x20 (fn 0x110c110, vmulps pair)"},
    {0x014a0f34, Kind::NegH_f, Group::J1, "J1.C2.-H (-1080.0f) — flip-Y sibling of C2"},
    {0x014a0f3c, Kind::NegH_f, Group::J1, "J1.C2.-H_dup (-1080.0f) — flip-Y sibling of C2 (+12)"},
    {0x014a0fe0, Kind::W_f,    Group::J1, "J1.C3.W  (+1920.0f) — flip-Y sibling of C3 at -0x20 (fn 0x110d140, vmulps pair)"},
    {0x014a0fe4, Kind::NegH_f, Group::J1, "J1.C3.-H (-1080.0f) — flip-Y sibling of C3"},
    {0x014a0fec, Kind::NegH_f, Group::J1, "J1.C3.-H_dup (-1080.0f) — flip-Y sibling of C3 (+12)"},

    // ── v14: K1 — SceneParameter exposureIntensity = 2.0f sites (was v12) ───
    //  Three instructions of form `mov dword ptr [rax+0x50], 0x40000000`
    //  (= 2.0f) writing scene_param[+0x50] = exposureIntensity default.
    //  Field offset +0x50 identified via Lua-binding registrar at
    //  0x01056d9e:
    //      lea  rbx, [rip+...exposureIntensity_str_at_0x14fc812]
    //      mov  edx, 0x50    ; struct field offset
    //      call register_param
    //  The four exposure fields live at +0x50/0x54/0x58/0x5c. 2.0f is
    //  the strongest non-zero exposureIntensity default across all
    //  high-confidence cluster scans (every other +0x50 imm32 write in
    //  the scene-param-shaped clusters is 0). Three sites with this
    //  exact 2.0f baseline were found via brute-force scan; all share
    //  the encoding `C7 40 50 00 00 00 40` (7 bytes; imm32 at +3).
    //
    //  v14 formula: 2.0f * min(1.0, (target_H / 1080.0f)²)
    //  — pixel-density ratio (since pixel count scales as H² at fixed
    //  16:9 aspect, this is the framebuffer-pixel-count ratio relative
    //  to 1080p). Lower-res targets get QUADRATICALLY less exposure
    //  (v12 was linear H/1080; Jun confirmed linear didn't fully
    //  compensate low-res over-exposure):
    //    540p   → 2.0 × 0.25  = 0.5    (quarter exposure — per spec)
    //    720p   → 2.0 × 0.444 = 0.889
    //    900p   → 2.0 × 0.694 = 1.389
    //    1080p+ → 2.0                  (idempotent at native and above)
    //
    //  EXCLUDED from `recommended` — opt in via "recommended,K1".
    //  Unknown which of the three sites drives runtime exposure (could
    //  be one, two, or all three); patching all three is the safest
    //  bet. If exposure still mis-corrects at low res, candidate sites
    //  with other baselines (1.5f at 0x011bf149, 1.0f at multiple sites)
    //  can be added in a future K2/K3 group.
    {0x00b3f625, Kind::Exposure2_f, Group::K1, "K1.exposureIntensity (2.0f → (H/1080)²-scaled) at fn 0xb3f622 — scene-param ctor"},
    {0x004c7d16, Kind::Exposure2_f, Group::K1, "K1.exposureIntensity (2.0f → (H/1080)²-scaled) at fn 0x4c7d13"},
    {0x008abd31, Kind::Exposure2_f, Group::K1, "K1.exposureIntensity (2.0f → (H/1080)²-scaled) at fn 0x8abd2e"},
}};

// ─── v7.3: Variable-length instruction-replacement patches (Group I1) ───
//
// Background. The user empirically confirmed that "B1 alone causes the
// UI scaling issue" — i.e., enabling B1 (the main RT struct slot write at
// [r15+0x92C0]/[+0x92C4]) makes UI elements scale with the framebuffer
// dimension instead of staying at fixed pixel coords. Root cause: the
// UI vertex projection reads W/H from B1's struct and computes the
// pixel-to-NDC normalization (1/W, -1/H, 1/W, 1/H) used to map fixed-
// pixel UI element coords to NDC. When B1 changes the slot to 540p, the
// normalization changes too, and 1080p-design UI elements end up at
// wrong relative sizes.
//
// Fix. Exhaustive scan of the full text segment found that ALL readers
// of [reg+0x92c0]/[+0x92c4] fall into two classes:
//   * 52 readers inside fn 0x102A5B0 range that pass W/H as integer to
//     fn 0x1218370 (RT/subsystem registration), or write to GPU command
//     buffers (mov to memory). These do NOT need patching.
//   * 2 readers (outside fn 0x102A range) that do CVTSI2SS + VDIVSS,
//     i.e., the projection-math fingerprint:
//
//       0x01011306   mov ecx, [rax+0x92C0]   } compute (1/W, -1/H, 1/W, 1/H)
//       0x0101130C   mov eax, [rax+0x92C4]   } and store at [rbx+0xf0]
//       0x01018D90   mov edx, [rcx+0x92C0]   } compute same quad and
//       0x01018D96   mov ecx, [rcx+0x92C4]   } store at [rbx+0x1b0]
//
// These four reads are patched to load literal 1920/1080 instead of
// reading from B1's struct, locking the UI projection at the original
// design resolution. UI elements at fixed 1080p pixel coords then render
// at correct relative size regardless of framebuffer dim.
//
// Encoding. Each original is a 6-byte `mov r32, [rXX + disp32]`. The
// replacement is `mov r32, imm32` (5 bytes) + `nop` (1 byte) = 6 bytes,
// matching length so no instruction-boundary shift occurs.
//
// Resolution-independence. These patches always write 0x780/0x438; they
// do NOT scale with the target resolution. The reasoning: we WANT the
// UI projection to think the framebuffer is 1920×1080 even when it
// isn't, so UI element coords (baked at 1080p design res) render at
// correct relative size. A user who wants UI to scale WITH the
// resolution (e.g., for a true 540p experience that includes
// downscaled UI) should leave I1 disabled.
//
// CAVEAT (untested at time of writing). If either of the two readers
// is NOT actually a UI projection but a postprocess pass that needs
// the true framebuffer dim for texelFetch math, locking it to 1080p
// could corrupt 3D postprocess at non-1080p. Ship as opt-in; observe
// behavior; if 3D corrupts, drop I1 (or split into I1a/I1b for finer
// bisection).

struct InstrReplaceSite {
    std::uint32_t va_offset;
    Group         group;
    std::uint8_t  size;             // bytes of the original instruction (== replacement size)
    std::uint8_t  expected[8];      // up to 8 bytes; first `size` are meaningful
    std::uint8_t  replacement[8];
    const char*   label;
};

constexpr std::array<InstrReplaceSite, 6> kInstrReplaceSites = {{
    // Reader 1: in function reading [r14+0x10b0]->struct, stores proj vec at [rbx+0xf0]
    {0x01011306, Group::I1, 6,
        // Original: mov ecx, dword ptr [rax+0x92C0]
        {0x8B, 0x88, 0xC0, 0x92, 0x00, 0x00, 0x00, 0x00},
        // Replacement: mov ecx, 0x780 (=1920); nop
        {0xB9, 0x80, 0x07, 0x00, 0x00, 0x90, 0x00, 0x00},
        "I1.W#1 (mov ecx, [rax+0x92C0]) → lit 1920 — UI proj reader 1 (rbx+0xf0)"},
    {0x0101130C, Group::I1, 6,
        // Original: mov eax, dword ptr [rax+0x92C4]
        {0x8B, 0x80, 0xC4, 0x92, 0x00, 0x00, 0x00, 0x00},
        // Replacement: mov eax, 0x438 (=1080); nop
        {0xB8, 0x38, 0x04, 0x00, 0x00, 0x90, 0x00, 0x00},
        "I1.H#1 (mov eax, [rax+0x92C4]) → lit 1080 — UI proj reader 1 (rbx+0xf0)"},
    // Reader 2: in function reading [r15]->[+0x10b0]->struct, stores proj vec at [rbx+0x1b0]
    {0x01018D90, Group::I1, 6,
        // Original: mov edx, dword ptr [rcx+0x92C0]
        {0x8B, 0x91, 0xC0, 0x92, 0x00, 0x00, 0x00, 0x00},
        // Replacement: mov edx, 0x780; nop
        {0xBA, 0x80, 0x07, 0x00, 0x00, 0x90, 0x00, 0x00},
        "I1.W#2 (mov edx, [rcx+0x92C0]) → lit 1920 — UI proj reader 2 (rbx+0x1b0)"},
    {0x01018D96, Group::I1, 6,
        // Original: mov ecx, dword ptr [rcx+0x92C4]
        {0x8B, 0x89, 0xC4, 0x92, 0x00, 0x00, 0x00, 0x00},
        // Replacement: mov ecx, 0x438; nop
        {0xB9, 0x38, 0x04, 0x00, 0x00, 0x90, 0x00, 0x00},
        "I1.H#2 (mov ecx, [rcx+0x92C4]) → lit 1080 — UI proj reader 2 (rbx+0x1b0)"},
}};

// ── Excluded / dead-end sites (DO NOT add to kResSites without re-RE) ──
//
// (v5 exclusions preserved; v6 hunt additions below.)
//   * 0x00e6faaa  width-only inside a 1920/0/960 mode-switch — not an RT.
//   * 0x00273eda, 0x0027a898  stack-local 1080 mov-imm32 — orphan.
//   * 0x0036ffeb  misaligned mov disp32 — not a real instruction.
//   * 0x014c36f0, 0x014c7e68/78, 0x01488a54  uncertain data-table 1080s.
//   * ~73 false matches inside `mov eax,1; lock cmpxchg` boundaries.
//
// v6 hunt additionally checked and dismissed:
//   * V# image-descriptor dwords in rodata (Strategy E): ZERO pre-built
//     image descriptors with width=1919 + height=1079 encoded into the
//     dw2 bit-packed (height-1)<<14 | (width-1) layout (or the swapped
//     ordering). Tested across the full eboot; all V# descriptors for
//     the main RT are built at runtime from B1's struct words.
//   * PSSL shader instruction-literal patches (the GXP-analog the user
//     suggested): re-extracted GCN bytecode boundaries using each
//     shader's prologue token (0xBEEB03FF) + BinaryInfo.length:24, then
//     scanned the actual bytecode region (before the OrbShdr tail
//     signature). Only TWO dword-aligned 1920/1080/960/540 hits across
//     all 392 shaders — both at value 960; one is tail-pad adjacent to
//     the OrbShdr signature, one is a denormal-float bit pattern. PS4
//     PSSL does not embed framebuffer resolution in instruction streams
//     the way Vita GXP does — dimensions reach shaders through V#
//     descriptors and constant buffers populated by the CPU patches in
//     this module.
//   * Memory-store orphans `mov [reg+disp], 0x780/0x438/0x1e0/0x10e`
//     where the imm32 is a dim value with no paired counterpart: the
//     only NEW such site at the imm32 (not displacement) level is the
//     cluster of 60+ `mov [rbx+0x2c], 0x87` writes in 0xb4a000–0xba0000.
//     0x87 = 135 but the offset and tight repetition look like a game-
//     subsystem constant ID, not a height field; deliberately not
//     patched.

// ── Encoding helpers ────────────────────────────────────────────────────

constexpr Resolution BaseSizeFor(TargetResolution t) {
    switch (t) {
    case TargetResolution::R540p:  return { 960,  540};
    case TargetResolution::R720p:  return {1280,  720};
    case TargetResolution::R900p:  return {1600,  900};
    case TargetResolution::R1080p: return {1920, 1080};
    case TargetResolution::R1440p: return {2560, 1440};
    case TargetResolution::R2160p: return {3840, 2160};
    case TargetResolution::R2880p: return {5120, 2880};
    case TargetResolution::R3456p: return {6144, 3456};
    case TargetResolution::R4032p: return {7168, 4032};
    case TargetResolution::R4320p: return {7680, 4320};
    default:                       return {1920, 1080};
    }
}

Resolution ComposeFinalImpl(TargetResolution t, float aspect_ratio) {
    const Resolution base = BaseSizeFor(t);
    constexpr float kBaseAR = 16.0f / 9.0f;
    constexpr float kEpsilon = 1e-4f;
    if (!(aspect_ratio > 0.0f) || std::fabs(aspect_ratio - kBaseAR) < kEpsilon) {
        return base;
    }
    if (aspect_ratio < kBaseAR) {
        return {base.width, static_cast<int>(std::lround(base.width / aspect_ratio))};
    }
    return {static_cast<int>(std::lround(base.height * aspect_ratio)), base.height};
}

constexpr std::array<std::uint8_t, 4> ExpectedForKind(Kind k) {
    switch (k) {
    case Kind::W_f:     return {0x00, 0x00, 0xF0, 0x44}; // 1920.0f
    case Kind::H_f:     return {0x00, 0x00, 0x87, 0x44}; // 1080.0f
    case Kind::HalfW_f: return {0x00, 0x00, 0x70, 0x44}; //  960.0f
    case Kind::HalfH_f: return {0x00, 0x00, 0x07, 0x44}; //  540.0f
    case Kind::InvW_f:  return {0x89, 0x88, 0x08, 0x3A}; // 1/1920
    case Kind::InvH_f:  return {0xD6, 0xB9, 0x72, 0x3A}; // 1/1080
    case Kind::W_u32:   return {0x80, 0x07, 0x00, 0x00}; // 1920
    case Kind::H_u32:   return {0x38, 0x04, 0x00, 0x00}; // 1080
    // ── v6 additions ────────────────────────────────────────────────
    case Kind::QuarterW_u32: return {0xE0, 0x01, 0x00, 0x00}; //  480 = 1920/4
    case Kind::QuarterH_u32: return {0x0E, 0x01, 0x00, 0x00}; //  270 = 1080/4
    // ── v7.4 additions ──────────────────────────────────────────────
    case Kind::InvHalfW_f:   return {0x89, 0x88, 0x88, 0x3A}; // 2/1920 = 1/960
    case Kind::InvHalfH_f:   return {0xD6, 0xB9, 0xF2, 0x3A}; // 2/1080 = 1/540
    // ── v9 additions ────────────────────────────────────────────────
    case Kind::NegH_f:       return {0x00, 0x00, 0x87, 0xC4}; // -1080.0f
    // ── v12 additions ───────────────────────────────────────────────
    case Kind::Exposure2_f:  return {0x00, 0x00, 0x00, 0x40}; //  2.0f
    }
    return {0, 0, 0, 0};
}

inline std::array<std::uint8_t, 4> FloatBytes(float f) {
    std::array<std::uint8_t, 4> out{};
    std::memcpy(out.data(), &f, 4);
    return out;
}
inline std::array<std::uint8_t, 4> U32Bytes(std::uint32_t u) {
    return {static_cast<std::uint8_t>(u),       static_cast<std::uint8_t>(u >> 8),
            static_cast<std::uint8_t>(u >> 16), static_cast<std::uint8_t>(u >> 24)};
}

std::array<std::uint8_t, 4> EncodeForKind(const Resolution& sz, Kind k) {
    const float fw = static_cast<float>(sz.width);
    const float fh = static_cast<float>(sz.height);
    switch (k) {
    case Kind::W_f:     return FloatBytes(fw);
    case Kind::H_f:     return FloatBytes(fh);
    case Kind::HalfW_f: return FloatBytes(fw * 0.5f);
    case Kind::HalfH_f: return FloatBytes(fh * 0.5f);
    case Kind::InvW_f:  return FloatBytes(1.0f / fw);
    case Kind::InvH_f:  return FloatBytes(1.0f / fh);
    case Kind::W_u32:   return U32Bytes(static_cast<std::uint32_t>(sz.width));
    case Kind::H_u32:   return U32Bytes(static_cast<std::uint32_t>(sz.height));
    // ── v6 additions ────────────────────────────────────────────────
    // Integer division — quarter-res RTs are integer-aligned at every
    // supported target resolution (960/4=240, 1280/4=320, 1920/4=480,
    // 2560/4=640, 3840/4=960, 7680/4=1920; heights divide identically).
    case Kind::QuarterW_u32:
        return U32Bytes(static_cast<std::uint32_t>(sz.width / 4));
    case Kind::QuarterH_u32:
        return U32Bytes(static_cast<std::uint32_t>(sz.height / 4));
    // ── v7.4 additions ──────────────────────────────────────────────
    case Kind::InvHalfW_f:   return FloatBytes(2.0f / fw);
    case Kind::InvHalfH_f:   return FloatBytes(2.0f / fh);
    // ── v9 additions ────────────────────────────────────────────────
    case Kind::NegH_f:       return FloatBytes(-fh);
    // ── v14 (was v12) ───────────────────────────────────────────────
    // Exposure compensation: scale 2.0f baseline by pixel density
    // ratio min(1.0, (H/1080)²). At fixed 16:9 aspect, pixel count
    // scales as H², so the multiplier is the framebuffer-pixel-count
    // ratio relative to 1080p. Compensates the empirically observed
    // OVER-exposure at low res (Jun: "540p is the worst").
    //   540p   → 2.0 × 0.25  = 0.5    (quarter exposure — per spec)
    //   720p   → 2.0 × 0.444 = 0.889
    //   900p   → 2.0 × 0.694 = 1.389
    //   1080p+ → 2.0                  (idempotent at native and above)
    // The clamp at 1.0 avoids amplifying exposure above native.
    case Kind::Exposure2_f: {
        const float ratio  = fh / 1080.0f;
        const float pd_clamped = std::min(1.0f, ratio * ratio);
        return FloatBytes(2.0f * pd_clamped);
    }
    }
    return {0, 0, 0, 0};
}

const char* TargetName(TargetResolution t) {
    switch (t) {
    case TargetResolution::R540p:  return "540p";
    case TargetResolution::R720p:  return "720p";
    case TargetResolution::R900p:  return "900p";
    case TargetResolution::R1080p: return "1080p";
    case TargetResolution::R1440p: return "1440p";
    case TargetResolution::R2160p: return "2160p (4K)";
    case TargetResolution::R2880p: return "2880p (5K)";
    case TargetResolution::R3456p: return "3456p (6K)";
    case TargetResolution::R4032p: return "4032p (7K)";
    case TargetResolution::R4320p: return "4320p (8K)";
    default:                       return "Off";
    }
}

const char* GroupName(Group g) {
    switch (g) {
    case Group::A1: return "A1"; case Group::A2: return "A2";
    case Group::A3: return "A3"; case Group::A4: return "A4";
    case Group::B1: return "B1"; case Group::C1: return "C1";
    case Group::C2: return "C2"; case Group::C3: return "C3";
    case Group::D1: return "D1"; case Group::E1: return "E1";
    case Group::F1: return "F1"; case Group::F2: return "F2";
    case Group::F3: return "F3"; case Group::G1: return "G1";
    case Group::H1: return "H1"; case Group::H2: return "H2";
    case Group::H3: return "H3"; case Group::H4: return "H4";
    case Group::I1: return "I1";
    case Group::I2: return "I2";
    case Group::I3: return "I3";
    case Group::I4: return "I4";
    case Group::J1: return "J1";
    case Group::K1: return "K1";
    default:        return "?";
    }
}

// ── Group-mask config parsing ───────────────────────────────────────────

std::string LowerTrim(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == ' ' || c == '\t') continue;
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

bool MatchGroupName(std::string_view tok, Group& out) {
    // tok already lowercased
    if (tok == "a1") { out = Group::A1; return true; }
    if (tok == "a2") { out = Group::A2; return true; }
    if (tok == "a3") { out = Group::A3; return true; }
    if (tok == "a4") { out = Group::A4; return true; }
    if (tok == "b1") { out = Group::B1; return true; }
    if (tok == "c1") { out = Group::C1; return true; }
    if (tok == "c2") { out = Group::C2; return true; }
    if (tok == "c3") { out = Group::C3; return true; }
    if (tok == "d1") { out = Group::D1; return true; }
    if (tok == "e1") { out = Group::E1; return true; }
    if (tok == "f1") { out = Group::F1; return true; }
    if (tok == "f2") { out = Group::F2; return true; }
    if (tok == "f3") { out = Group::F3; return true; }
    if (tok == "g1") { out = Group::G1; return true; }
    if (tok == "h1") { out = Group::H1; return true; }
    if (tok == "h2") { out = Group::H2; return true; }
    if (tok == "h3") { out = Group::H3; return true; }
    if (tok == "h4") { out = Group::H4; return true; }
    if (tok == "i1") { out = Group::I1; return true; }
    if (tok == "i2") { out = Group::I2; return true; }
    if (tok == "i3") { out = Group::I3; return true; }
    if (tok == "i4") { out = Group::I4; return true; }
    if (tok == "j1") { out = Group::J1; return true; }
    if (tok == "k1") { out = Group::K1; return true; }
    return false;
}

} // namespace

std::uint32_t ParseGroupMaskFromConfig(std::string_view raw) {
    const std::string s = LowerTrim(raw);
    if (s.empty() || s == "all" || s == "default") return kGroupMaskAll;
    if (s == "none" || s == "off")                  return 0u;

    std::uint32_t mask = 0;
    bool seen_any = false;

    // Tokenize on ',' or '|'
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = i;
        while (j < s.size() && s[j] != ',' && s[j] != '|') j++;
        std::string_view tok(s.data() + i, j - i);
        i = j + 1;
        if (tok.empty()) continue;

        // negation prefix: ~ or !
        bool negate = false;
        if (tok.front() == '~' || tok.front() == '!') { negate = true; tok.remove_prefix(1); }
        if (tok.empty()) continue;

        // preset names
        std::uint32_t add = 0;
        if      (tok == "all" || tok == "default") add = kGroupMaskAll;
        else if (tok == "recommended" || tok == "prod") add = kGroupMaskRecommended;
        else if (tok == "safe" || tok == "min")    add = kGroupMaskSafe;
        else if (tok == "geom")                    add = kGroupMaskGeom;
        else if (tok == "ext"  || tok == "extended") add = kGroupMaskExt;
        else if (tok == "none" || tok == "off")    add = 0;
        // ── v7 bisection presets ─────────────────────────────────
        // Each one adds exactly ONE new-since-v5 group on top of the
        // v3 known-desync baseline (B1+D1) so testing isolates a
        // single variable.
        else if (tok == "baseline" || tok == "v3")              add = kGroupMaskBaseline;
        else if (tok == "add_h1"   || tok == "addh1")           add = kGroupMaskAddH1;
        else if (tok == "add_h2"   || tok == "addh2")           add = kGroupMaskAddH2;
        else if (tok == "add_g1"   || tok == "addg1")           add = kGroupMaskAddG1;
        else if (tok == "add_all_h"|| tok == "addallh")         add = kGroupMaskAddAllH;
        else if (tok == "without_g1" || tok == "withoutg1")     add = kGroupMaskWithoutG1;
        else {
            Group g{};
            if (MatchGroupName(tok, g)) {
                add = GroupBit(g);
            } else {
                LOG_WARNING(Core, "[GR2 Resolution] unknown patch-group token: '{}'",
                            std::string(tok));
                continue;
            }
        }

        if (negate) mask &= ~add;
        else        mask |= add;
        seen_any = true;
    }

    return seen_any ? mask : kGroupMaskAll;
}

int ApplyGr2ResolutionPatches(uintptr_t eboot_base, TargetResolution resolution,
                              float aspect_ratio) {
    if (eboot_base == 0) {
        LOG_ERROR(Core, "[GR2 Resolution] eboot_base is 0 — cannot patch");
        return 0;
    }

    // Build the internal group mask. The patch-groups config knob was
    // removed in v16: the resolution pipeline now ALWAYS applies
    // kGroupMaskRecommended (the empirically-verified working set, see
    // resolution_patches.h).
    std::uint32_t group_mask = 0;
    if (resolution != TargetResolution::Off) {
        group_mask |= kGroupMaskRecommended;
    }
    if (group_mask == 0) {
        LOG_INFO(Core, "[GR2 Resolution] target=Off; nothing to patch");
        return 0;
    }

    const Resolution final_sz = ComposeFinalImpl(resolution, aspect_ratio);

    // Per-group counters for visibility.
    int per_group_patched[static_cast<int>(Group::Count)] = {0};
    int per_group_already[static_cast<int>(Group::Count)] = {0};
    int per_group_mismatch[static_cast<int>(Group::Count)] = {0};
    int per_group_skipped[static_cast<int>(Group::Count)] = {0};

    for (const auto& s : kResSites) {
        const std::uint32_t bit = GroupBit(s.group);
        if ((group_mask & bit) == 0) {
            per_group_skipped[static_cast<int>(s.group)]++;
            continue;
        }

        std::uint8_t* site = reinterpret_cast<std::uint8_t*>(eboot_base + s.va_offset);
        const auto expected    = ExpectedForKind(s.kind);
        const auto replacement = EncodeForKind(final_sz, s.kind);

        if (std::memcmp(site, replacement.data(), 4) == 0) {
            per_group_already[static_cast<int>(s.group)]++;
            continue;
        }
        if (std::memcmp(site, expected.data(), 4) != 0) {
            LOG_WARNING(Core,
                "[GR2 Resolution] {} @ vaddr 0x{:x}: unexpected bytes "
                "(got {:02x} {:02x} {:02x} {:02x}, expected {:02x} {:02x} {:02x} {:02x}) — NOT patched",
                s.label, s.va_offset,
                site[0], site[1], site[2], site[3],
                expected[0], expected[1], expected[2], expected[3]);
            per_group_mismatch[static_cast<int>(s.group)]++;
            continue;
        }
        std::memcpy(site, replacement.data(), 4);
        per_group_patched[static_cast<int>(s.group)]++;
    }

    // ── v7.3: I1 variable-length instruction-replacement patches ──────
    // These do NOT scale with target resolution — replacement bytes are
    // hard-coded to encode literal 0x780/0x438 (1920/1080), locking the
    // UI projection at design res.
    for (const auto& s : kInstrReplaceSites) {
        const std::uint32_t bit = GroupBit(s.group);
        if ((group_mask & bit) == 0) {
            per_group_skipped[static_cast<int>(s.group)]++;
            continue;
        }
        std::uint8_t* site = reinterpret_cast<std::uint8_t*>(eboot_base + s.va_offset);
        if (std::memcmp(site, s.replacement, s.size) == 0) {
            per_group_already[static_cast<int>(s.group)]++;
            continue;
        }
        if (std::memcmp(site, s.expected, s.size) != 0) {
            LOG_WARNING(Core,
                "[GR2 Resolution] {} @ vaddr 0x{:x}: unexpected bytes — NOT patched",
                s.label, s.va_offset);
            per_group_mismatch[static_cast<int>(s.group)]++;
            continue;
        }
        std::memcpy(site, s.replacement, s.size);
        per_group_patched[static_cast<int>(s.group)]++;
    }

    int total_patched = 0, total_already = 0, total_mismatch = 0, total_skipped = 0;
    for (int i = 0; i < static_cast<int>(Group::Count); i++) {
        total_patched  += per_group_patched[i];
        total_already  += per_group_already[i];
        total_mismatch += per_group_mismatch[i];
        total_skipped  += per_group_skipped[i];
    }

    LOG_INFO(Core,
        "[GR2 Resolution] target={} aspect={:.4f} -> "
        "final={}x{} mask=0x{:04x} — "
        "{} patched, {} already, {} mismatched, {} skipped (of {} total)",
        TargetName(resolution), aspect_ratio,
        final_sz.width, final_sz.height,
        group_mask, total_patched, total_already, total_mismatch, total_skipped,
        static_cast<int>(kResSites.size() + kInstrReplaceSites.size()));

    // Per-group breakdown so user can see what happened.
    for (int gi = 0; gi < static_cast<int>(Group::Count); gi++) {
        const auto g = static_cast<Group>(gi);
        const int p = per_group_patched[gi], a = per_group_already[gi];
        const int m = per_group_mismatch[gi], k = per_group_skipped[gi];
        if (p + a + m + k == 0) continue;
        const bool enabled = (group_mask & GroupBit(g)) != 0;
        LOG_INFO(Core,
            "[GR2 Resolution]   {} ({}): patched={} already={} mismatch={} skipped={}",
            GroupName(g), enabled ? "ENABLED" : "disabled", p, a, m, k);
    }
    return total_patched;
}

TargetResolution ParseResolutionFromConfig(std::string_view s) {
    if (s == "540p"  || s == "960x540")                  return TargetResolution::R540p;
    if (s == "720p"  || s == "1280x720")                 return TargetResolution::R720p;
    if (s == "900p"  || s == "1600x900")                 return TargetResolution::R900p;
    if (s == "1080p" || s == "1920x1080")                return TargetResolution::R1080p;
    if (s == "1440p" || s == "2560x1440")                return TargetResolution::R1440p;
    if (s == "2160p" || s == "3840x2160" || s == "4K")   return TargetResolution::R2160p;
    if (s == "2880p" || s == "5120x2880" || s == "5K")   return TargetResolution::R2880p;
    if (s == "3456p" || s == "6144x3456" || s == "6K")   return TargetResolution::R3456p;
    if (s == "4032p" || s == "7168x4032" || s == "7K")   return TargetResolution::R4032p;
    if (s == "4320p" || s == "7680x4320" || s == "8K")   return TargetResolution::R4320p;
    return TargetResolution::Off;
}

Resolution TargetResolutionToBaseSize(TargetResolution t) {
    return BaseSizeFor(t);
}

Resolution ComputeFinalResolution(TargetResolution t, float aspect_ratio) {
    return ComposeFinalImpl(t, aspect_ratio);
}

} // namespace Libraries::ResolutionPatches
