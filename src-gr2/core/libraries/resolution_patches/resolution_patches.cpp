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

#include "common/config.h"
#include "common/logging/log.h"
#include "common/types.h"
#include "core/libraries/resolution_patches/resolution_patches.h"

namespace Libraries::ResolutionPatches {

namespace {

enum class Kind {
    W_f, H_f, HalfW_f, HalfH_f, InvW_f, InvH_f, W_u32, H_u32,
    // Quarter-resolution integer dimensions. Encoded as (final_W / 4)
    // and (final_H / 4) at apply-time; baseline bytes are 0x1E0 (480)
    // and 0x10E (270) at 1080p.
    QuarterW_u32,
    QuarterH_u32,
    // 2/W, 2/H - orthographic projection X/Y scale for the 1920x1080 design canvas, encoded as
    // 2.0f/W_actual and 2.0f/H_actual (idempotent at native). Scales the 1080p-design UI ortho
    // projection with the framebuffer dimension.
    InvHalfW_f,
    InvHalfH_f,
    // Negative-Y height, encoded as -H_actual (baseline -1080.0f = 0xC4870000). Used in flip-Y
    // mirror projection vecs (J-group) that are component-multiplied with adjacent C-group vecs.
    NegH_f,
    // SceneParameter.exposureIntensity default 2.0f scaled by the pixel-density ratio
    // min(1.0, (H/1080)^2) to compensate low-res over-exposure: 540p -> 0.5, 720p -> 0.889,
    // 900p -> 1.389, 1080p+ -> 2.0 (idempotent). Baseline bytes: 0x40000000 (2.0f LE).
    Exposure2_f,
    // Q1: negative viewport-center half-dims, sign-complement of HalfW_f/HalfH_f. Additive
    // centering offsets in screen = (ndc*scale) + (-halfDim) transforms, unreachable by P1's
    // positive-only encoders. Baseline: -960.0f = 0xC4700000, -540.0f = 0xC4070000.
    NegHalfW_f,
    NegHalfH_f,
};

// Parallel to PatchGroupBit; small integer index for table compactness.
enum class Group : std::uint8_t {
    A1 = 0, A2, A3, A4, B1, C1, C2, C3, D1, E1, F1, F2, F3, G1,
    H1, H2, H3, H4,
    I1,  // UI projection lock. The only two readers of [reg+0x92c0]/[+0x92c4] doing float math
         // (cvtsi2ss+vdivss at 0x01011306/0x0101130C, 0x01018D90/0x01018D96) build the UI
         // (1/W,-1/H,1/W,1/H) quad; patching them to literal 1920/1080 keeps UI element size.
    I2,  // UI ortho projection scale - the (2/1920, 2/1080) pair at rodata 0x14C565C/0x14C5664.
         // No visible effect at any resolution (consumer fn 0x10D3B60 is dead/niche code);
         // excluded from `recommended`, retained for completeness.
    I3,  // UI viewport scale - the (960, 540, 0, 0) vec4 at rodata 0x01499EC0, single vmovaps
         // consumer at VA 0x00EABC8D feeding fn 0x110C110. No visible effect on UI scaling;
         // excluded from `recommended`, opt-in for re-verification only.
    I4,  // Second heavy-use (960, 540) pair at rodata 0x0149A190/0x0149A194, 8+ vmovaps
         // consumers in fns 0xEBC8xx-0xEBD9xx (row 1 of a 5-row vec4 descriptor at 0x0149A180).
         // Alone it does not fix UI canvas scaling; excluded from `recommended`, opt-in.
    J1,  // Flip-Y mirror vecs (1920, -1080, -999.99, -1080) at C_VA - 0x20 (0x014985b0,
         // 0x014a0f30, 0x014a0fe0), component-multiplied with the C-vecs in fns 0x0110c110 and
         // 0x0110d140; the product drives text positioning, not canvas sizing. Not recommended.
    K1,  // SceneParameter.exposureIntensity (struct[+0x50]) 2.0f defaults - `mov dword ptr
         // [rax+0x50], 0x40000000` at 0x00b3f622, 0x004c7d13, 0x008abd2e - rewritten to the
         // Exposure2_f pixel-density-scaled value. Excluded from `recommended`; opt in to test.
    // Reserved slot: GroupBit(g) == (1u << ordinal(g)) and must stay equal to the explicit
    // shift in PatchGroupBit, which keeps bit 24 vacant (K1 = 1<<23, M1 = 1<<25) so persisted
    // bitmask values stay stable. Keeps ordinal(M1) == 25; no patch sites, so it never applies.
    L1_reserved,
    M1,  // PS4-Pro 4K-mode master enable: forces the BSS flag at VA 0x01b586c0 (gates the
         // swap-chain buffer attribute) to the 4K path via two writer patches, two shl NOPs,
         // and an apply-time BSS write. In `recommended`, gated to >= R2160p (see apply block).
    N1,  // Real sceVideoOutSetBufferAttribute caller patch: the two BSS-load instructions at
         // eboot VA 0x00446933/0x00446939 become `mov reg, imm32` encoding target_W/target_H.
         // See the N1 application block in ApplyGr2ResolutionPatches.
    O1,  // Swap-chain buffer allocation size fix, paired with N1: four imm16 patches at
         // 0x00446825/29/2d/31 force the allocator wrapper to target_W/target_H regardless of
         // the PS4-Pro cmova branch. See the O1 application block in ApplyGr2ResolutionPatches.
    P1,  // Viewport-center / half-dim rodata floats - 150 sites across 68 clusters. Scales
         // hardcoded (960.0f, 540.0f) NDC-to-screen center constants to (target_W/2,
         // target_H/2); fixes offset comic-scene images.
    P2,  // BSS_FLAG-indexed (1080p, 4K) mode table at 0x014c7e60. 8
         // floats patched to target dims, making M1's FLAG indexing
         // irrelevant for this path.
    P3,  // Screen-edge diamond vec4 array at 0x01488a00. 6 floats
         // patched (W/H/HalfW/HalfH mix) to keep the diamond sample
         // pattern inscribed in the framebuffer at any target.
    // -------- Q1 addition (opt-in; OFF in `recommended` AND in `all`) --------
    Q1,  // Negative viewport-center half-dims, sign-complement of P1: 7 rodata floats
         // (-960/-540) in additive screen-center vaddps/vaddss transforms. Ordinal must be 31
         // to match PatchGroupBit::Q1. Hardware-unverified: only an explicit `q1` token, no `all`.
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

// Sites are tagged with a consumer-function group. Confirmed-working set: A1-A4, B1, C1, D1,
// F1-F3, H1-H4 (B1/D1 in fn 0x102a5b0 are the main RT init and screen-vec uniform - highest
// confidence). C2/C3 corrupt UI text solo; E1 and G1 corrupt UI canvas sizing (not recommended).
constexpr std::array<ResSite, 540> kResSites = {{
    // -------- Group A: float mov-imm32 paired struct-init writes (8 sites) --------
    {0x0044cad5, Kind::W_f, Group::A1, "A1.W (mov [rsi+rax*4],1920.0f) — fn 0x44c093"},
    {0x0044cadc, Kind::H_f, Group::A1, "A1.H (mov [rdx+rax*4],1080.0f) — fn 0x44c093"},
    {0x00fde098, Kind::W_f, Group::A2, "A2.W (mov [rsi+0x20],1920.0f)  — fn 0xfdcb86"},
    {0x00fde0a0, Kind::H_f, Group::A2, "A2.H (mov [rsi+0x24],1080.0f)  — fn 0xfdcb86"},
    {0x00fde146, Kind::W_f, Group::A3, "A3.W (mov [rsi+rdi+0x18],…)    — fn 0xfdcb86"},
    {0x00fde14f, Kind::H_f, Group::A3, "A3.H (mov [rsi+rdi+0x1c],…)    — fn 0xfdcb86"},
    {0x0110b4a7, Kind::W_f, Group::A4, "A4.W (mov [rsi+rax*4],1920.0f) — fn 0x110adfd — UI TEXT POS (do not enable at non-1080p)"},
    {0x0110b4ae, Kind::H_f, Group::A4, "A4.H (mov [rdx+rax*4],1080.0f) — fn 0x110adfd — UI TEXT POS (do not enable at non-1080p)"},

    // -------- Group B: integer mov-imm32 paired struct-init (2 sites) --------
    {0x0102aa21, Kind::W_u32, Group::B1, "B1.W (mov [rdi+0x92c0],0x780) — fn 0x102a5b0 MAIN"},
    {0x0102aa2c, Kind::H_u32, Group::B1, "B1.H (mov [r15+0x92c4],0x438) — fn 0x102a5b0 MAIN"},

    // -------- Group C: (W,H,1000.01,H) vec multipliers (9 sites) --------
    {0x014985d0, Kind::W_f, Group::C1, "C1.W — vmulps in fn 0x136d74d"},
    {0x014985d4, Kind::H_f, Group::C1, "C1.H — vmulps in fn 0x136d74d"},
    {0x014985dc, Kind::H_f, Group::C1, "C1.H_again(+12) — vmulps in fn 0x136d74d"},
    {0x014a0f50, Kind::W_f, Group::C2, "C2.W — vmulps in fn 0x110c110 — UI TEXT POS (do not enable at non-1080p)"},
    {0x014a0f54, Kind::H_f, Group::C2, "C2.H — vmulps in fn 0x110c110 — UI TEXT POS"},
    {0x014a0f5c, Kind::H_f, Group::C2, "C2.H_again(+12) — vmulps in fn 0x110c110 — UI TEXT POS"},
    {0x014a1000, Kind::W_f, Group::C3, "C3.W — vmulps in fn 0x110d140 — UI TEXT POS (do not enable at non-1080p)"},
    {0x014a1004, Kind::H_f, Group::C3, "C3.H — vmulps in fn 0x110d140 — UI TEXT POS"},
    {0x014a100c, Kind::H_f, Group::C3, "C3.H_again(+12) — vmulps in fn 0x110d140 — UI TEXT POS"},

    // -------- Group D: (W,H,1/W,1/H) projection vec (4 sites) --------
    {0x0149e630, Kind::W_f,    Group::D1, "D1.W   — vmovaps in fn 0x102a5b0 MAIN"},
    {0x0149e634, Kind::H_f,    Group::D1, "D1.H   — vmovaps in fn 0x102a5b0 MAIN"},
    {0x0149e638, Kind::InvW_f, Group::D1, "D1.invW— vmovaps in fn 0x102a5b0 MAIN"},
    {0x0149e63c, Kind::InvH_f, Group::D1, "D1.invH— vmovaps in fn 0x102a5b0 MAIN"},

    // -------- Group E: (1/W,1/H) reciprocal pair (2 sites) --------
    {0x014c4d60, Kind::InvW_f, Group::E1, "E1.invW — vmulss in fn 0x0fe6090"},
    {0x014c4d64, Kind::InvH_f, Group::E1, "E1.invH — vmulss in fn 0x0fe6090"},

    // -------- Group F: (halfW,halfH,W,H) sub-camera projection tuples (12) --------
    //  Each F-group is consumed by one function calling 0x10cbc00 + 0x10cf0e0 (sub-camera
    //  projection build); strongest suspect for UI/sub-viewport corruption when patched.
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

    // -------- Group G: per-entity screen-space transform - fn 0xfdf9d0 callers --------
    //  145 call sites of the per-entity unproject/pick/frustum builder, each prepping
    //  `mov ecx, 0x780` / `mov r8d, 0x438`; a 256-byte search window catches interleaved math.
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
    {0x00e5d610, Kind::H_u32, Group::G1, "G1.H (instr 0x00e5d60e) — v18 paired completion"},
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
    {0x00e932c6, Kind::H_u32, Group::G1, "G1.H (instr 0x00e932c4) — v18 paired completion"},
    {0x00e932e9, Kind::W_u32, Group::G1, "G1.W (instr 0x00e932e8)"},
    {0x00e932ef, Kind::H_u32, Group::G1, "G1.H (instr 0x00e932ed)"},
    {0x00e934c8, Kind::W_u32, Group::G1, "G1.W (instr 0x00e934c7)"},
    {0x00e934ce, Kind::H_u32, Group::G1, "G1.H (instr 0x00e934cc)"},
    {0x00e934ea, Kind::W_u32, Group::G1, "G1.W (instr 0x00e934e9)"},
    {0x00e934f0, Kind::H_u32, Group::G1, "G1.H (instr 0x00e934ee)"},
    {0x00e9350c, Kind::W_u32, Group::G1, "G1.W (instr 0x00e9350b)"},
    {0x00e93512, Kind::H_u32, Group::G1, "G1.H (instr 0x00e93510) — v18 paired completion"},
    {0x00e97ac0, Kind::W_u32, Group::G1, "G1.W (instr 0x00e97abf)"},
    {0x00e97ac6, Kind::H_u32, Group::G1, "G1.H (instr 0x00e97ac4)"},
    {0x00e99d6d, Kind::W_u32, Group::G1, "G1.W (instr 0x00e99d6c)"},
    {0x00e99d73, Kind::H_u32, Group::G1, "G1.H (instr 0x00e99d71)"},
    {0x00e99d92, Kind::W_u32, Group::G1, "G1.W (instr 0x00e99d91)"},
    {0x00e99d98, Kind::H_u32, Group::G1, "G1.H (instr 0x00e99d96) — v18 paired completion"},
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
    {0x00ef0ae0, Kind::H_u32, Group::G1, "G1.H (instr 0x00ef0ade) — v18 paired completion"},
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
    {0x00f587c5, Kind::H_u32, Group::G1, "G1.H (instr 0x00f587c3) — v18 paired completion"},
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
    {0x00f58946, Kind::H_u32, Group::G1, "G1.H (instr 0x00f58944) — v18 paired completion"},
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
    {0x00f5c609, Kind::H_u32, Group::G1, "G1.H (instr 0x00f5c607) — v18 paired completion"},
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
    {0x00f65bec, Kind::H_u32, Group::G1, "G1.H (instr 0x00f65bea) — v18 paired completion"},
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
    {0x00f68587, Kind::H_u32, Group::G1, "G1.H (instr 0x00f68585) — v18 paired completion"},
    {0x00f685aa, Kind::W_u32, Group::G1, "G1.W (instr 0x00f685a9)"},
    {0x00f685b0, Kind::H_u32, Group::G1, "G1.H (instr 0x00f685ae)"},
    {0x00f68798, Kind::W_u32, Group::G1, "G1.W (instr 0x00f68797)"},
    {0x00f6879e, Kind::H_u32, Group::G1, "G1.H (instr 0x00f6879c)"},
    {0x00f687bd, Kind::W_u32, Group::G1, "G1.W (instr 0x00f687bc)"},
    {0x00f687c3, Kind::H_u32, Group::G1, "G1.H (instr 0x00f687c1) — v18 paired completion"},
    {0x00f6eecd, Kind::W_u32, Group::G1, "G1.W (instr 0x00f6eecc)"},
    {0x00f6eed3, Kind::H_u32, Group::G1, "G1.H (instr 0x00f6eed1)"},
    {0x00f6eef2, Kind::W_u32, Group::G1, "G1.W (instr 0x00f6eef1)"},
    {0x00f6eef8, Kind::H_u32, Group::G1, "G1.H (instr 0x00f6eef6) — v18 paired completion"},
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
    {0x00f71caf, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71cad) — v18 paired completion"},
    {0x00f71cd2, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71cd1)"},
    {0x00f71cd8, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71cd6)"},
    {0x00f71ec0, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71ebf)"},
    {0x00f71ec6, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71ec4)"},
    {0x00f71ee5, Kind::W_u32, Group::G1, "G1.W (instr 0x00f71ee4)"},
    {0x00f71eeb, Kind::H_u32, Group::G1, "G1.H (instr 0x00f71ee9) — v18 paired completion"},
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

    // -------- Group G: per-entity screen-space transform - fn 0xfdf9d0 callers --------
    //  109 sites prep `mov ecx, 0x780` / `mov r8d, 0x438` before `call 0xfdf9d0` (per-object
    //  unprojection to [r15+0xC0..0xF0]); unpatched, LOD/culling/screen effects assume 1080p.

    // -------- Group H1: companion main RT (W,H) pair in fn 0x102a5b0 --------
    //  Same function as B1, immediately before a second call to the RT/subsystem registrar
    //  fn 0x1218370; patches the imm32 edx=W, ecx=H dim arguments passed to it.
    {0x0102aa0b, Kind::W_u32, Group::H1, "H1.W (mov edx, 0x780)         — fn 0x102a5b0 MAIN, pre-call 0x1218370"},
    {0x0102aa10, Kind::H_u32, Group::H1, "H1.H (mov ecx, 0x438)         — fn 0x102a5b0 MAIN, pre-call 0x1218370"},

    // -------- Group H2: quarter-resolution sub-RT registrations (8 pairs) --------
    //  Eight (480, 270) dim pairs passed to fn 0x1218370, four with subsystem-ID esi=3 (shadow/
    //  SSR) and four with esi=6 (bloom/post); they must scale to final/4 or desync the main RT.
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

    // -------- Group H3: 4K-selector 1080p-default-branch (W,H) pair --------
    //  (1920,1080)/(3840,2160) cmovne inputs at the 4K-toggle entry 0xd9fa00; inert, since the
    //  consumer is the TLS registrar in fn 0x1218370, not an RT allocator. OFF by default.
    {0x00d9fa07, Kind::W_u32, Group::H3, "H3.W (mov r14d, 0x780)        — 4K-selector default-branch (OFF default)"},
    {0x00d9fa11, Kind::H_u32, Group::H3, "H3.H (mov ebx, 0x438)         — 4K-selector default-branch (OFF default)"},

    // -------- Group H4: IMUL-imm32 dim pairs at 4K-toggle entry fn --------
    //  `imul reg, eax, 0x780/0x438` at 0x45d8ea/0x45d9e3 (imm32 at instr_va+2); eax is the same
    //  1-or-2 4K mode flag as H3. Low confidence, same TLS-registrar caveat; OFF by default.
    {0x0045d8e0, Kind::W_u32, Group::H4, "H4.W (imul edx, eax, 0x780)   — 4K-toggle entry #1 (OFF default)"},
    {0x0045d8e6, Kind::H_u32, Group::H4, "H4.H (imul ecx, eax, 0x438)   — 4K-toggle entry #1 (OFF default)"},
    {0x0045d9d9, Kind::W_u32, Group::H4, "H4.W (imul edx, eax, 0x780)   — 4K-toggle entry #2 (OFF default)"},
    {0x0045d9df, Kind::H_u32, Group::H4, "H4.H (imul ecx, eax, 0x438)   — 4K-toggle entry #2 (OFF default)"},
    // -------- I2: UI ortho projection scale --------
    {0x014c565c, Kind::InvHalfW_f, Group::I2, "I2.X (2/1920 = 1/960) — ortho X scale, vmulss in fn 0x10d3b60"},
    {0x014c5664, Kind::InvHalfH_f, Group::I2, "I2.Y (2/1080 = 1/540) — ortho Y scale, vmulss in fn 0x10d3b60"},
    // -------- I3: UI viewport SCALE row of 4-vec4 descriptor --------
    {0x01499ec0, Kind::HalfW_f, Group::I3, "I3.W (960.0f) — viewport scale.x, vmovaps at 0x00EABC8D → fn 0x110C110"},
    {0x01499ec4, Kind::HalfH_f, Group::I3, "I3.H (540.0f) — viewport scale.y, vmovaps at 0x00EABC8D → fn 0x110C110"},
    // -------- I4: Candidate-A (960,540,0,0) vec4 at 0x0149A190 --------
    //  Second heavily-consumed (960,540) rodata pair after I3; 8+ vmovaps consumers in fns
    //  0xEBC8xx-0xEBD9xx, single-function blast radius. OFF in `recommended` - opt in to test.
    {0x0149a190, Kind::HalfW_f, Group::I4, "I4.W (960.0f) — Candidate-A vec4 at 0x0149A190 (vmovaps in fn 0xEBC8xx)"},
    {0x0149a194, Kind::HalfH_f, Group::I4, "I4.H (540.0f) — Candidate-A vec4 at 0x0149A194 (vmovaps in fn 0xEBC8xx)"},

    // -------- J1: flip-Y mirror vecs paired with C1/C2/C3 --------
    //  Three (1920, -1080, -999.99, -1080) vecs at C_VA - 0x20, component-multiplied with the
    //  C-vecs - meaningful only with C2/C3. Per vec: W at +0, NegH at +4/+12; +8 is not a dim.
    {0x014985b0, Kind::W_f,    Group::J1, "J1.C1.W  (+1920.0f) — flip-Y sibling of C1 at -0x20 (fn 0x136d74d)"},
    {0x014985b4, Kind::NegH_f, Group::J1, "J1.C1.-H (-1080.0f) — flip-Y sibling of C1"},
    {0x014985bc, Kind::NegH_f, Group::J1, "J1.C1.-H_dup (-1080.0f) — flip-Y sibling of C1 (+12)"},
    {0x014a0f30, Kind::W_f,    Group::J1, "J1.C2.W  (+1920.0f) — flip-Y sibling of C2 at -0x20 (fn 0x110c110, vmulps pair)"},
    {0x014a0f34, Kind::NegH_f, Group::J1, "J1.C2.-H (-1080.0f) — flip-Y sibling of C2"},
    {0x014a0f3c, Kind::NegH_f, Group::J1, "J1.C2.-H_dup (-1080.0f) — flip-Y sibling of C2 (+12)"},
    {0x014a0fe0, Kind::W_f,    Group::J1, "J1.C3.W  (+1920.0f) — flip-Y sibling of C3 at -0x20 (fn 0x110d140, vmulps pair)"},
    {0x014a0fe4, Kind::NegH_f, Group::J1, "J1.C3.-H (-1080.0f) — flip-Y sibling of C3"},
    {0x014a0fec, Kind::NegH_f, Group::J1, "J1.C3.-H_dup (-1080.0f) — flip-Y sibling of C3 (+12)"},

    // -------- K1: SceneParameter exposureIntensity = 2.0f sites --------
    //  Three `mov dword ptr [rax+0x50], 0x40000000` writes (C7 40 50 00 00 00 40, imm32 at +3;
    //  +0x50 from the Lua registrar at 0x01056d9e), re-encoded as 2.0f * min(1.0, (H/1080)^2).
    {0x00b3f625, Kind::Exposure2_f, Group::K1, "K1.exposureIntensity (2.0f → (H/1080)²-scaled) at fn 0xb3f622 — scene-param ctor"},
    {0x004c7d16, Kind::Exposure2_f, Group::K1, "K1.exposureIntensity (2.0f → (H/1080)²-scaled) at fn 0x4c7d13"},
    {0x008abd31, Kind::Exposure2_f, Group::K1, "K1.exposureIntensity (2.0f → (H/1080)²-scaled) at fn 0x8abd2e"},

    // -------- P1: viewport-center / half-dim rodata floats (150 sites) --------
    // 68 clusters of (960.0f, 540.0f) NDC-to-screen center constants (screen = halfDim +/-
    // ndc*scale, confirmed at cluster 0x014c2a48); off-center comic panels at non-1080p.
    // -------- P2: BSS_FLAG-indexed mode table at 0x014c7e60 (8 sites) --------
    // (W, W4K, H, H4K) x 2 table read via vmovss [base + flag*4] by 0x00feeb8e and 0x00fef0dd;
    // patching all 8 entries makes both readers return target dims regardless of FLAG.
    // cluster head 0x014c2a48  (hWhH)
    {0x014c2a48, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2a48"},
    {0x014c2a4c, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2a48"},
    // cluster head 0x014c2b1c  (hWhH)
    {0x014c2b1c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2b1c"},
    {0x014c2b20, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2b1c"},
    // cluster head 0x014c2b5c  (hWhH)
    {0x014c2b5c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2b5c"},
    {0x014c2b60, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2b5c"},
    // cluster head 0x014c2ba4  (hWhH)
    {0x014c2ba4, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2ba4"},
    {0x014c2ba8, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2ba4"},
    // cluster head 0x014c2bf0  (hWhH)
    {0x014c2bf0, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2bf0"},
    {0x014c2bf4, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2bf0"},
    // cluster head 0x014c2c34  (hWhH)
    {0x014c2c34, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2c34"},
    {0x014c2c38, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2c34"},
    // cluster head 0x014c2c78  (hWhH)
    {0x014c2c78, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2c78"},
    {0x014c2c7c, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2c78"},
    // cluster head 0x014c2ce0  (hWhH)
    {0x014c2ce0, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2ce0"},
    {0x014c2ce4, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2ce0"},
    // cluster head 0x014c2d40  (hWhH)
    {0x014c2d40, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2d40"},
    {0x014c2d44, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2d40"},
    // cluster head 0x014c2d7c  (hWhHhWhH)
    {0x014c2d7c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2d7c"},
    {0x014c2d80, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2d7c"},
    {0x014c2d90, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2d7c"},
    {0x014c2d94, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2d7c"},
    // cluster head 0x014c2dc4  (hWhH)
    {0x014c2dc4, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2dc4"},
    {0x014c2dc8, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2dc4"},
    // cluster head 0x014c2dec  (hWhH)
    {0x014c2dec, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2dec"},
    {0x014c2df0, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2dec"},
    // cluster head 0x014c2e84  (hWhH)
    {0x014c2e84, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2e84"},
    {0x014c2e88, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2e84"},
    // cluster head 0x014c2ef8  (hWhH)
    {0x014c2ef8, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2ef8"},
    {0x014c2efc, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2ef8"},
    // cluster head 0x014c2f7c  (hWhH)
    {0x014c2f7c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c2f7c"},
    {0x014c2f80, Kind::HalfH_f, Group::P1, "P1.hH@0x14c2f7c"},
    // cluster head 0x014c3014  (hWhHhWhHhWhH)
    {0x014c3014, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3014"},
    {0x014c3018, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3014"},
    {0x014c3028, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3014"},
    {0x014c302c, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3014"},
    {0x014c303c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3014"},
    {0x014c3040, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3014"},
    // cluster head 0x014c3084  (hWhH)
    {0x014c3084, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3084"},
    {0x014c3088, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3084"},
    // cluster head 0x014c30b0  (hWhH)
    {0x014c30b0, Kind::HalfW_f, Group::P1, "P1.hW@0x14c30b0"},
    {0x014c30b4, Kind::HalfH_f, Group::P1, "P1.hH@0x14c30b0"},
    // cluster head 0x014c30fc  (hWhH)
    {0x014c30fc, Kind::HalfW_f, Group::P1, "P1.hW@0x14c30fc"},
    {0x014c3100, Kind::HalfH_f, Group::P1, "P1.hH@0x14c30fc"},
    // cluster head 0x014c3154  (hWhH)
    {0x014c3154, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3154"},
    {0x014c3158, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3154"},
    // cluster head 0x014c3174  (hWhH)
    {0x014c3174, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3174"},
    {0x014c3178, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3174"},
    // cluster head 0x014c31a4  (hWhH)
    {0x014c31a4, Kind::HalfW_f, Group::P1, "P1.hW@0x14c31a4"},
    {0x014c31a8, Kind::HalfH_f, Group::P1, "P1.hH@0x14c31a4"},
    // cluster head 0x014c31f4  (hWhH)
    {0x014c31f4, Kind::HalfW_f, Group::P1, "P1.hW@0x14c31f4"},
    {0x014c31f8, Kind::HalfH_f, Group::P1, "P1.hH@0x14c31f4"},
    // cluster head 0x014c3218  (hWhH)
    {0x014c3218, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3218"},
    {0x014c321c, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3218"},
    // cluster head 0x014c324c  (hWhH)
    {0x014c324c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c324c"},
    {0x014c3250, Kind::HalfH_f, Group::P1, "P1.hH@0x14c324c"},
    // cluster head 0x014c326c  (hWhH)
    {0x014c326c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c326c"},
    {0x014c3270, Kind::HalfH_f, Group::P1, "P1.hH@0x14c326c"},
    // cluster head 0x014c33b8  (hWhH)
    {0x014c33b8, Kind::HalfW_f, Group::P1, "P1.hW@0x14c33b8"},
    {0x014c33bc, Kind::HalfH_f, Group::P1, "P1.hH@0x14c33b8"},
    // cluster head 0x014c33ec  (hWhH)
    {0x014c33ec, Kind::HalfW_f, Group::P1, "P1.hW@0x14c33ec"},
    {0x014c33f0, Kind::HalfH_f, Group::P1, "P1.hH@0x14c33ec"},
    // cluster head 0x014c343c  (hWhH)
    {0x014c343c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c343c"},
    {0x014c3440, Kind::HalfH_f, Group::P1, "P1.hH@0x14c343c"},
    // cluster head 0x014c349c  (hWhH)
    {0x014c349c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c349c"},
    {0x014c34a0, Kind::HalfH_f, Group::P1, "P1.hH@0x14c349c"},
    // cluster head 0x014c34d0  (hWhH)
    {0x014c34d0, Kind::HalfW_f, Group::P1, "P1.hW@0x14c34d0"},
    {0x014c34d4, Kind::HalfH_f, Group::P1, "P1.hH@0x14c34d0"},
    // cluster head 0x014c3520  (hWhH)
    {0x014c3520, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3520"},
    {0x014c3524, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3520"},
    // cluster head 0x014c357c  (hWhH)
    {0x014c357c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c357c"},
    {0x014c3580, Kind::HalfH_f, Group::P1, "P1.hH@0x14c357c"},
    // cluster head 0x014c35d8  (hWhH)
    {0x014c35d8, Kind::HalfW_f, Group::P1, "P1.hW@0x14c35d8"},
    {0x014c35dc, Kind::HalfH_f, Group::P1, "P1.hH@0x14c35d8"},
    // cluster head 0x014c35f4  (hWhH)
    {0x014c35f4, Kind::HalfW_f, Group::P1, "P1.hW@0x14c35f4"},
    {0x014c35f8, Kind::HalfH_f, Group::P1, "P1.hH@0x14c35f4"},
    // cluster head 0x014c3668  (hWhH)
    {0x014c3668, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3668"},
    {0x014c366c, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3668"},
    // cluster head 0x014c36c4  (hWhH)
    {0x014c36c4, Kind::HalfW_f, Group::P1, "P1.hW@0x14c36c4"},
    {0x014c36c8, Kind::HalfH_f, Group::P1, "P1.hH@0x14c36c4"},
    // cluster head 0x014c371c  (hWhH)
    {0x014c371c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c371c"},
    {0x014c3720, Kind::HalfH_f, Group::P1, "P1.hH@0x14c371c"},
    // cluster head 0x014c3764  (hWhH)
    {0x014c3764, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3764"},
    {0x014c3768, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3764"},
    // cluster head 0x014c38e0  (hWhH)
    {0x014c38e0, Kind::HalfW_f, Group::P1, "P1.hW@0x14c38e0"},
    {0x014c38e4, Kind::HalfH_f, Group::P1, "P1.hH@0x14c38e0"},
    // cluster head 0x014c3918  (hWhH)
    {0x014c3918, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3918"},
    {0x014c391c, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3918"},
    // cluster head 0x014c3990  (hWhH)
    {0x014c3990, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3990"},
    {0x014c3994, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3990"},
    // cluster head 0x014c39e0  (hWhH)
    {0x014c39e0, Kind::HalfW_f, Group::P1, "P1.hW@0x14c39e0"},
    {0x014c39e4, Kind::HalfH_f, Group::P1, "P1.hH@0x14c39e0"},
    // cluster head 0x014c3af0  (hWhH)
    {0x014c3af0, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3af0"},
    {0x014c3af4, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3af0"},
    // cluster head 0x014c3b30  (hWhH)
    {0x014c3b30, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3b30"},
    {0x014c3b34, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3b30"},
    // cluster head 0x014c3d0c  (hWhH)
    {0x014c3d0c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3d0c"},
    {0x014c3d10, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3d0c"},
    // cluster head 0x014c3e14  (hWhH)
    {0x014c3e14, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3e14"},
    {0x014c3e18, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3e14"},
    // cluster head 0x014c3e34  (hWhH)
    {0x014c3e34, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3e34"},
    {0x014c3e38, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3e34"},
    // cluster head 0x014c3e5c  (hWhH)
    {0x014c3e5c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3e5c"},
    {0x014c3e60, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3e5c"},
    // cluster head 0x014c3e84  (hWhH)
    {0x014c3e84, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3e84"},
    {0x014c3e88, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3e84"},
    // cluster head 0x014c3eb4  (hWhH)
    {0x014c3eb4, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3eb4"},
    {0x014c3eb8, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3eb4"},
    // cluster head 0x014c3f70  (hWhH)
    {0x014c3f70, Kind::HalfW_f, Group::P1, "P1.hW@0x14c3f70"},
    {0x014c3f74, Kind::HalfH_f, Group::P1, "P1.hH@0x14c3f70"},
    // cluster head 0x014c40f4  (hWhH)
    {0x014c40f4, Kind::HalfW_f, Group::P1, "P1.hW@0x14c40f4"},
    {0x014c40f8, Kind::HalfH_f, Group::P1, "P1.hH@0x14c40f4"},
    // cluster head 0x014c418c  (hWhH)
    {0x014c418c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c418c"},
    {0x014c4190, Kind::HalfH_f, Group::P1, "P1.hH@0x14c418c"},
    // cluster head 0x014c41e8  (hWhH)
    {0x014c41e8, Kind::HalfW_f, Group::P1, "P1.hW@0x14c41e8"},
    {0x014c41ec, Kind::HalfH_f, Group::P1, "P1.hH@0x14c41e8"},
    // cluster head 0x014c42cc  (hWhH)
    {0x014c42cc, Kind::HalfW_f, Group::P1, "P1.hW@0x14c42cc"},
    {0x014c42d0, Kind::HalfH_f, Group::P1, "P1.hH@0x14c42cc"},
    // cluster head 0x014c4300  (hWhHhWhHhWhH)
    {0x014c4300, Kind::HalfW_f, Group::P1, "P1.hW@0x14c4300"},
    {0x014c4304, Kind::HalfH_f, Group::P1, "P1.hH@0x14c4300"},
    {0x014c4314, Kind::HalfW_f, Group::P1, "P1.hW@0x14c4300"},
    {0x014c4318, Kind::HalfH_f, Group::P1, "P1.hH@0x14c4300"},
    {0x014c4328, Kind::HalfW_f, Group::P1, "P1.hW@0x14c4300"},
    {0x014c432c, Kind::HalfH_f, Group::P1, "P1.hH@0x14c4300"},
    // cluster head 0x014c434c  (hWhHhWhH)
    {0x014c434c, Kind::HalfW_f, Group::P1, "P1.hW@0x14c434c"},
    {0x014c4350, Kind::HalfH_f, Group::P1, "P1.hH@0x14c434c"},
    {0x014c4360, Kind::HalfW_f, Group::P1, "P1.hW@0x14c434c"},
    {0x014c4364, Kind::HalfH_f, Group::P1, "P1.hH@0x14c434c"},
    // cluster head 0x014c4408  (hWhH)
    {0x014c4408, Kind::HalfW_f, Group::P1, "P1.hW@0x14c4408"},
    {0x014c440c, Kind::HalfH_f, Group::P1, "P1.hH@0x14c4408"},
    // cluster head 0x014c4424  (hWhH)
    {0x014c4424, Kind::HalfW_f, Group::P1, "P1.hW@0x14c4424"},
    {0x014c4428, Kind::HalfH_f, Group::P1, "P1.hH@0x14c4424"},
    // cluster head 0x014c4448  (hWhH)
    {0x014c4448, Kind::HalfW_f, Group::P1, "P1.hW@0x14c4448"},
    {0x014c444c, Kind::HalfH_f, Group::P1, "P1.hH@0x14c4448"},
    // cluster head 0x014c46a4  (hWhH)
    {0x014c46a4, Kind::HalfW_f, Group::P1, "P1.hW@0x14c46a4"},
    {0x014c46a8, Kind::HalfH_f, Group::P1, "P1.hH@0x14c46a4"},
    // cluster head 0x014c46bc  (hWhH)
    {0x014c46bc, Kind::HalfW_f, Group::P1, "P1.hW@0x14c46bc"},
    {0x014c46c0, Kind::HalfH_f, Group::P1, "P1.hH@0x14c46bc"},
    // cluster head 0x014c46ec  (hWhH)
    {0x014c46ec, Kind::HalfW_f, Group::P1, "P1.hW@0x14c46ec"},
    {0x014c46f0, Kind::HalfH_f, Group::P1, "P1.hH@0x14c46ec"},
    // cluster head 0x014c49a4  (hWhH)
    {0x014c49a4, Kind::HalfW_f, Group::P1, "P1.hW@0x14c49a4"},
    {0x014c49a8, Kind::HalfH_f, Group::P1, "P1.hH@0x14c49a4"},
    // cluster head 0x014c49cc  (hWhHhWhH)
    {0x014c49cc, Kind::HalfW_f, Group::P1, "P1.hW@0x14c49cc"},
    {0x014c49d0, Kind::HalfH_f, Group::P1, "P1.hH@0x14c49cc"},
    {0x014c49e0, Kind::HalfW_f, Group::P1, "P1.hW@0x14c49cc"},
    {0x014c49e4, Kind::HalfH_f, Group::P1, "P1.hH@0x14c49cc"},
    // cluster head 0x014c4a30  (hWhH)
    {0x014c4a30, Kind::HalfW_f, Group::P1, "P1.hW@0x14c4a30"},
    {0x014c4a34, Kind::HalfH_f, Group::P1, "P1.hH@0x14c4a30"},
    // cluster head 0x014c4a68  (hWhH)
    {0x014c4a68, Kind::HalfW_f, Group::P1, "P1.hW@0x14c4a68"},
    {0x014c4a6c, Kind::HalfH_f, Group::P1, "P1.hH@0x14c4a68"},
    {0x014c7e60, Kind::W_f, Group::P2, "P2.W@0x14c7e60"},
    {0x014c7e64, Kind::W_f, Group::P2, "P2.W@0x14c7e64"},
    {0x014c7e68, Kind::H_f, Group::P2, "P2.H@0x14c7e68"},
    {0x014c7e6c, Kind::H_f, Group::P2, "P2.H@0x14c7e6c"},
    {0x014c7e70, Kind::W_f, Group::P2, "P2.W@0x14c7e70"},
    {0x014c7e74, Kind::W_f, Group::P2, "P2.W@0x14c7e74"},
    {0x014c7e78, Kind::H_f, Group::P2, "P2.H@0x14c7e78"},
    {0x014c7e7c, Kind::H_f, Group::P2, "P2.H@0x14c7e7c"},


    // -------- P3: screen-edge diamond vec4 array @ 0x01488a00 --------
    // Six vec4s loaded by fn ~0x00bc4b00; vec4[2..5] form a diamond inscribed in the 1080p
    // framebuffer (edge midpoints), vec4[0]/[1] are basis vectors and stay untouched.
    {0x01488a20, Kind::W_f,     Group::P3, "P3.W      (right-edge X @ 0x1488a20)"},
    {0x01488a24, Kind::HalfH_f, Group::P3, "P3.hH#1   (right-edge Y @ 0x1488a24)"},
    {0x01488a34, Kind::HalfH_f, Group::P3, "P3.hH#2   (left-edge  Y @ 0x1488a34)"},
    {0x01488a40, Kind::HalfW_f, Group::P3, "P3.hW#1   (top-edge   X @ 0x1488a40)"},
    {0x01488a50, Kind::HalfW_f, Group::P3, "P3.hW#2   (bottom-edge X @ 0x1488a50)"},
    {0x01488a54, Kind::H_f,     Group::P3, "P3.H      (bottom-edge Y @ 0x1488a54)"},

    // -------- Q1: negative viewport-center half-dims (P1 complement) --------
    // Screen-center additive offsets the P1 (positive) encoder can't
    // reach. Baselines verified -960.0f / -540.0f in CUSA04943 v1.11.
    {0x01497ef0, Kind::NegHalfW_f, Group::Q1, "Q1.-hW splat  — vaddps 0x135d173/0x135d46f (screen-center X)"},
    {0x01497f00, Kind::NegHalfH_f, Group::Q1, "Q1.-hH splat  — vaddps 0x135d187/0x135d483 (pair w/0x1497ef0)"},
    {0x01498560, Kind::NegHalfW_f, Group::Q1, "Q1.-hW        — vaddps 0x136e6e5"},
    {0x01498564, Kind::NegHalfH_f, Group::Q1, "Q1.-hH        — vaddps 0x136e6e5 (pair w/0x1498560)"},
    {0x01498280, Kind::NegHalfW_f, Group::Q1, "Q1.-hW single — vaddps 0x1364d1c (+4 = -500, unrelated)"},
    {0x014c36e8, Kind::NegHalfW_f, Group::Q1, "Q1.-hW        — vaddss 0x136494c (cluster -960,-540,1080,100)"},
    {0x014c36ec, Kind::NegHalfH_f, Group::Q1, "Q1.-hH        — vaddss 0x136494c (pair w/0x14c36e8)"},

}};

// Group I1: same-length instruction replacements - the two float-math readers of B1's
// [+0x92c0]/[+0x92c4] W/H slots become `mov r32, imm32` + nop with literal 1920/1080, locking
// the UI projection at design res. Opt-in: a postprocess reader would corrupt 3D at non-1080p.

struct InstrReplaceSite {
    std::uint32_t va_offset;
    Group         group;
    std::uint8_t  size;             // bytes of the original instruction (== replacement size)
    std::uint8_t  expected[8];      // up to 8 bytes; first `size` are meaningful
    std::uint8_t  replacement[8];
    const char*   label;
};

constexpr std::array<InstrReplaceSite, 8> kInstrReplaceSites = {{
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

    // M1: the 4K dispatch hangs on one BSS byte at VA 0x01b586c0; base-PS4 emulation leaves it 0
    // and the cmovne selector at 0x00d9f9c0 keeps a 1920x1080 swap chain. M1 forces FLAG=1 and
    // NOPs the shl dim doublers (B1 supplies target dims); >= R2160p only (4K branch: 3840/2160).
    {0x00446852, Group::M1, 7,
        // Original: seta byte ptr [rip + 0x1711e67]  (writer_seta)
        {0x0F, 0x97, 0x05, 0x67, 0x1E, 0x71, 0x01, 0x00},
        // Replacement: mov byte ptr [rip + 0x1711e67], 1
        {0xC6, 0x05, 0x67, 0x1E, 0x71, 0x01, 0x01, 0x00},
        "M1.WRITER_SETA (seta byte FLAG) → mov FLAG, 1 — force PS4-Pro detection to always-true"},
    {0x004466dd, Group::M1, 7,
        // Original: mov byte ptr [rip + 0x1711fdc], 0  (writer_zero)
        {0xC6, 0x05, 0xDC, 0x1F, 0x71, 0x01, 0x00, 0x00},
        // Replacement: mov byte ptr [rip + 0x1711fdc], 1
        {0xC6, 0x05, 0xDC, 0x1F, 0x71, 0x01, 0x01, 0x00},
        "M1.WRITER_ZERO (mov FLAG, 0) → mov FLAG, 1 — keep flag set across the reset-path writer"},
    {0x0102b9a5, Group::M1, 4,
        // Original: shl eax, cl ; shl edx, cl (D3 E0 D3 E2)
        {0xD3, 0xE0, 0xD3, 0xE2, 0x00, 0x00, 0x00, 0x00},
        // Replacement: 4x NOP
        {0x90, 0x90, 0x90, 0x90, 0x00, 0x00, 0x00, 0x00},
        "M1.SHIFT_X (shl eax,cl; shl edx,cl @ init_x/cb3) → 4xNOP — prevent B1 4K-dim double-amplification"},
    {0x0102ba8e, Group::M1, 4,
        // Original: shl eax, cl ; shl edx, cl (D3 E0 D3 E2)
        {0xD3, 0xE0, 0xD3, 0xE2, 0x00, 0x00, 0x00, 0x00},
        // Replacement: 4x NOP
        {0x90, 0x90, 0x90, 0x90, 0x00, 0x00, 0x00, 0x00},
        "M1.SHIFT_Y (shl eax,cl; shl edx,cl @ init_y/cb4) → 4xNOP — prevent B1 4K-dim double-amplification"},
}};

// Excluded / dead-end sites - not added to kResSites without fresh RE: assorted orphan
// 1920/1080 immediates and data-table hits are not RT dims, and no pre-built V# descriptor or
// PSSL bytecode carries the framebuffer resolution (dims reach shaders via B1-derived state).

// -------- Encoding helpers --------

constexpr Resolution BaseSizeFor(TargetResolution t) {
    switch (t) {
    case TargetResolution::R270p:  return { 480,  270};
    case TargetResolution::R360p:  return { 640,  360};
    // 16:9 at 480 rows is 853.33 px wide. The swap-chain pitch (N1) encodes in 8-pixel steps
    // and must not fall below the width, so the width rounds up to 856 (+0.31% aspect error).
    case TargetResolution::R480p:  return { 856,  480};
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
    // Wider-than-16:9 targets extend the width. Round it to the nearest multiple of 8: the
    // swap-chain pitch (N1) encodes in 8-pixel steps, and a pitch below the width shears the
    // presented image (e.g. 21:10 at 1080 rows is 2268 -> 2272).
    const long wide_w = std::lround(base.height * aspect_ratio);
    return {static_cast<int>((wide_w + 4) / 8 * 8), base.height};
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
    case Kind::QuarterW_u32: return {0xE0, 0x01, 0x00, 0x00}; //  480 = 1920/4
    case Kind::QuarterH_u32: return {0x0E, 0x01, 0x00, 0x00}; //  270 = 1080/4
    case Kind::InvHalfW_f:   return {0x89, 0x88, 0x88, 0x3A}; // 2/1920 = 1/960
    case Kind::InvHalfH_f:   return {0xD6, 0xB9, 0xF2, 0x3A}; // 2/1080 = 1/540
    case Kind::NegH_f:       return {0x00, 0x00, 0x87, 0xC4}; // -1080.0f
    case Kind::Exposure2_f:  return {0x00, 0x00, 0x00, 0x40}; //  2.0f
    case Kind::NegHalfW_f:   return {0x00, 0x00, 0x70, 0xC4}; // -960.0f
    case Kind::NegHalfH_f:   return {0x00, 0x00, 0x07, 0xC4}; // -540.0f
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
    // Integer division - widths divide exactly (every base width is a multiple of 8, and
    // composed wide widths round to one); heights that are not multiples of 4 (270p, some
    // composed 4:3 heights) truncate, undersizing the quarter-res RTs by under one row.
    case Kind::QuarterW_u32:
        return U32Bytes(static_cast<std::uint32_t>(sz.width / 4));
    case Kind::QuarterH_u32:
        return U32Bytes(static_cast<std::uint32_t>(sz.height / 4));
    case Kind::InvHalfW_f:   return FloatBytes(2.0f / fw);
    case Kind::InvHalfH_f:   return FloatBytes(2.0f / fh);
    case Kind::NegH_f:       return FloatBytes(-fh);
    // Exposure compensation: scale the 2.0f baseline by pixel-density ratio
    // min(1.0, (H/1080)^2) - the framebuffer pixel-count ratio at fixed 16:9. Compensates
    // low-res over-exposure (540p -> 0.5, 720p -> 0.889, 900p -> 1.389, 1080p+ -> 2.0).
    case Kind::Exposure2_f: {
        const float ratio  = fh / 1080.0f;
        const float pd_clamped = std::min(1.0f, ratio * ratio);
        return FloatBytes(2.0f * pd_clamped);
    }
    // -------- Q1 additions: negated half-dimension screen-center terms --------
    case Kind::NegHalfW_f:   return FloatBytes(-fw * 0.5f);
    case Kind::NegHalfH_f:   return FloatBytes(-fh * 0.5f);
    }
    return {0, 0, 0, 0};
}

const char* TargetName(TargetResolution t) {
    switch (t) {
    case TargetResolution::R270p:  return "270p";
    case TargetResolution::R360p:  return "360p";
    case TargetResolution::R480p:  return "480p";
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
    case Group::L1_reserved: return "L1(reserved)";
    case Group::M1: return "M1";
    case Group::N1: return "N1";
    case Group::O1: return "O1";
    case Group::P1: return "P1";
    case Group::P2: return "P2";
    case Group::P3: return "P3";
    case Group::Q1: return "Q1";
    default:        return "?";
    }
}

// -------- Group-mask config parsing --------

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
    if (tok == "m1") { out = Group::M1; return true; }
    if (tok == "n1") { out = Group::N1; return true; }
    if (tok == "o1") { out = Group::O1; return true; }
    if (tok == "p1") { out = Group::P1; return true; }
    if (tok == "p2") { out = Group::P2; return true; }
    if (tok == "p3") { out = Group::P3; return true; }
    if (tok == "q1") { out = Group::Q1; return true; }
    return false;
}

// Gravity Rush Remastered (GRR) resolution patches. GRR's RT family differs from GR2's: main
// RT 1920x1088, half 960x544, quarter 480x272 - the RT height is always tile-aligned up to a
// multiple of 64 (align64: 1080 -> 1088). Idempotent at native: 1080p reproduces stock bytes.

constexpr std::uint32_t Align64(std::uint32_t v) {
    return ((v + 63u) / 64u) * 64u;
}

// GRR dimension Kinds. "W" is the composed final width; "Halign" is the
// 64-aligned final height. Half/Quarter are computed in float so a non-16:9
// aspect can never produce an integer-parity surprise.
enum class GrrKind {
    W_f,             // final_W                  (stock 1920.0f)
    Halign_f,        // align64(final_H)         (stock 1088.0f)
    HalfW_f,         // final_W / 2              (stock  960.0f)
    HalfHalign_f,    // align64(final_H) / 2     (stock  544.0f)
    QuarterW_f,      // final_W / 4              (stock  480.0f)
    QuarterHalign_f, // align64(final_H) / 4     (stock  272.0f)
    Hexact_f,        // final_H (NOT aligned)    (stock 1080.0f) - C1 only
    // Integer register-immediate RT dimensions (groups S1..S4): `mov r32, imm32` operands
    // feeding the RT allocator at 0x223e0 / wrapper 0x1cac0, not rodata floats. Width and pitch
    // encode final_W; height passes the exact unaligned final_H (stock 1080 - allocator aligns).
    W_u32,           // final_W (also pitch=W)   (stock 1920 = 80 07 00 00)
    H_u32,           // final_H exact            (stock 1080 = 38 04 00 00)
    // Half-resolution integer RT dims (groups S5..S6 - post-process buffers
    // created at half the scene resolution via the same wrapper 0x1cac0).
    HalfW_u32,       // final_W / 2 (also pitch) (stock  960 = c0 03 00 00)
    HalfH_u32,       // final_H / 2 exact        (stock  540 = 1c 02 00 00)
    HalfHalign_u32,  // align64(final_H) / 2     (stock  544 = 20 02 00 00)
};

// GRR patch groups. Their own bit-space, fully independent of the GR2
// PatchGroupBit enum.
enum class GrrGroup : std::uint8_t {
    R1 = 0,  // Main RT dimension array - the GRR analog of GR2's B1: 28 sites (14 W/H pairs) of
             // `mov dword ptr [rcx+disp], imm32` in the descriptor-array init fn at 0x1ebfac.
             // Highest confidence, the working main lever; in `recommended`.
    U1,      // Half/quarter UI float cluster @ 0x105f440..0x105f4cc. 24 rodata
             // floats (960/544/480/272). Lower confidence (UI sub-RT scale);
             // OPT-IN - off in `recommended`. For RenderDoc bisection.
    C1,      // Reference-resolution table at 0x104cd10/0x104cd14 (1920.0f/1080.0f exact - 1080,
             // not 1088). Sits inside a camera/effect config table and may be a divisor, so it
             // could corrupt; opt-in, off in `recommended`.
    // Scene render-target allocation groups. R1/U1/C1 move only the UI; the 3D scene targets
    // come from the RT allocator 0x223e0 / wrapper 0x1cac0 with integer (W,H) in registers.
    // Opt-in, unverified one-at-a-time levers; in `all` but not `recommended` (UI-only set).
    S1,      // 3-RT MRT set (pixel-format 4) created together in init fn 0x1c440, object slots
             // [ctx+0x188/0x218/0x2a8]; strongest main scene colour / G-buffer candidate.
             // 6 sites: 3x W (r8d) + 3x H (r9d).
    S2,      // Single RT, pixel-format 9 (likely an HDR/float scene buffer),
             // created via wrapper 0x1cac0 @ 0x2fb3a. 3 sites: W (ecx),
             // H (r8d), pitch (r9d).
    S3,      // Single RT, pixel-format 2, via 0x1cac0 @ 0x6ed3f. 3 sites:
             // W (ecx), H (r8d), pitch (r9d).
    S4,      // Single RT, pixel-format 0xA, via 0x1cac0 @ 0x256d76. 3 sites:
             // W (ecx), H (r8d), pitch (r9d).
    // Half-resolution post-process RTs (960x540 / 960x544 - bloom/DOF/SSAO class) via the RT
    // wrapper 0x153b0; left unscaled they mismatch a scaled main RT. Opt-in, in `all`. RTs
    // sized at runtime from word[ctx+0xf984]/[+0xf9e4] have no patchable immediate.
    S5,      // Half-res RT, pixel-format 9 (likely half-res HDR/bloom), via
             // 0x153b0. 3 sites: W (ecx 960), H (r8d 540 = exact half), pitch
             // (r9d 960).
    S6,      // Half-res RT, pixel-format 11, via 0x153b0. 3 sites: W (ecx 960),
             // H (r8d 544 = align64(1080)/2), pitch (r9d 960).
    // Group S7: second-allocator scene RT (the GR2 "H group" analog). A parallel wrapper family
    // (0x161b0) feeds allocator 0x1aea0 via surface-size fn 0x4850e0; its one resolution-class
    // RT (fmt 2, 1920x1080) must scale with S1-S6 or the 3D image desyncs. Sites: W ecx, H r8d.
    S7,
    // Group T1: comic text-cull neutralizer (raw GCN patches, kGrrBytePatchSites) - rewrites
    // s_andn2_b64 -> s_and_b64 (6 sites) so scaled scene RTs stop culling ComicDemo *Scissor*
    // text, at the cost of panel-edge clipping. Obsolete (FragCoordResolutionScalePass); `t1` only.
    T1,
    // Group T2: comic FrameParam canvas pin (in-place rewrite, kGrrBlobPatchSites): the comic
    // frame-rect ring reader at 0x67150 is pinned to the baked 960.0f/544.0f half-canvas so
    // u_v4FrameParam stays 1080p under scaled RTs. Textbox-only. Obsolete; `t2` token only.
    T2,
    Count
};

constexpr std::uint32_t GrrGroupBit(GrrGroup g) {
    return 1u << static_cast<std::uint32_t>(g);
}

// GRR group masks (independent of the GR2 mask space). `recommended` = R1 + U1 + C1 + S1..S7,
// identical to `all`; `safe`/`min` selects R1 only. T1/T2 are obsolete (the FragCoord
// recompiler pass normalizes screen-space coords) and excluded from every preset.
constexpr std::uint32_t kGrrMaskRecommended =
    GrrGroupBit(GrrGroup::R1) | GrrGroupBit(GrrGroup::U1) | GrrGroupBit(GrrGroup::C1) |
    GrrGroupBit(GrrGroup::S1) | GrrGroupBit(GrrGroup::S2) | GrrGroupBit(GrrGroup::S3) |
    GrrGroupBit(GrrGroup::S4) | GrrGroupBit(GrrGroup::S5) | GrrGroupBit(GrrGroup::S6) |
    GrrGroupBit(GrrGroup::S7);
constexpr std::uint32_t kGrrMaskAll =
    GrrGroupBit(GrrGroup::R1) | GrrGroupBit(GrrGroup::U1) | GrrGroupBit(GrrGroup::C1) |
    GrrGroupBit(GrrGroup::S1) | GrrGroupBit(GrrGroup::S2) | GrrGroupBit(GrrGroup::S3) |
    GrrGroupBit(GrrGroup::S4) | GrrGroupBit(GrrGroup::S5) | GrrGroupBit(GrrGroup::S6) |
    GrrGroupBit(GrrGroup::S7);

struct GrrResSite {
    std::uint32_t va_offset;
    GrrKind       kind;
    GrrGroup      group;
    const char*   label;
};

// va_offset values are LOADED VAs (eboot_base + va_offset at runtime), not ELF file offsets:
// the GRR SELF stores ph[0] (p_vaddr=0) at payload offset 0xB710, not at ELF p_offset 0x4000.
// New sites map as va = file_offset - 0xB710; 0x4000-based VAs land 0x7710 high and never apply.
constexpr std::array<GrrResSite, 77> kGrrResSites = {{
    // -------- Group R1: main RT (W,H) descriptor-array init - fn 0x1e489c (28) --------
    //  14 paired `mov dword ptr [rcx+disp], imm32` stores. Each entry's imm
    //  is the last 4 bytes of the mov. W -> final_W, H -> align64(final_H).
    {0x1e48b0, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 0"},
    {0x1e48b7, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 0"},
    {0x1e48e8, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 1"},
    {0x1e48ef, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 1"},
    {0x1e4a4d, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 2"},
    {0x1e4a57, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 2"},
    {0x1e4a9d, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 3"},
    {0x1e4aa7, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 3"},
    {0x1e4c62, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 4"},
    {0x1e4c6c, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 4"},
    {0x1e4cb2, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 5"},
    {0x1e4cbc, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 5"},
    {0x1e4e71, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 6"},
    {0x1e4e7b, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 6"},
    {0x1e4ec1, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 7"},
    {0x1e4ecb, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 7"},
    {0x1e5075, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 8"},
    {0x1e507f, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 8"},
    {0x1e50c5, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 9"},
    {0x1e50cf, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 9"},
    {0x1e527d, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 10"},
    {0x1e5287, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 10"},
    {0x1e52cd, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 11"},
    {0x1e52d7, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 11"},
    {0x1e5481, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 12"},
    {0x1e548b, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 12"},
    {0x1e54d1, GrrKind::W_f,      GrrGroup::R1, "R1.W  (mov [rcx+disp],1920.0f) — fn 0x1e489c entry 13"},
    {0x1e54db, GrrKind::Halign_f, GrrGroup::R1, "R1.H  (mov [rcx+disp],1088.0f) — fn 0x1e489c entry 13"},

    // -------- Group U1: half/quarter UI float cluster 0x1057d30..0x1057dbc (24) --------
    //  HalfW=960 -> final_W/2, HalfH=544 -> align64(final_H)/2,
    //  QW=480 -> final_W/4,   QH=272 -> align64(final_H)/4.
    {0x1057d30, GrrKind::HalfW_f,         GrrGroup::U1, "U1.halfW (960.0f) @ 0x1057d30"},
    {0x1057d38, GrrKind::QuarterW_f,      GrrGroup::U1, "U1.quartW(480.0f) @ 0x1057d38"},
    {0x1057d3c, GrrKind::HalfHalign_f,    GrrGroup::U1, "U1.halfH (544.0f) @ 0x1057d3c"},
    {0x1057d40, GrrKind::QuarterHalign_f, GrrGroup::U1, "U1.quartH(272.0f) @ 0x1057d40"},
    {0x1057d44, GrrKind::HalfW_f,         GrrGroup::U1, "U1.halfW (960.0f) @ 0x1057d44"},
    {0x1057d4c, GrrKind::QuarterW_f,      GrrGroup::U1, "U1.quartW(480.0f) @ 0x1057d4c"},
    {0x1057d50, GrrKind::HalfHalign_f,    GrrGroup::U1, "U1.halfH (544.0f) @ 0x1057d50"},
    {0x1057d54, GrrKind::QuarterHalign_f, GrrGroup::U1, "U1.quartH(272.0f) @ 0x1057d54"},
    {0x1057d60, GrrKind::HalfW_f,         GrrGroup::U1, "U1.halfW (960.0f) @ 0x1057d60"},
    {0x1057d64, GrrKind::HalfHalign_f,    GrrGroup::U1, "U1.halfH (544.0f) @ 0x1057d64"},
    {0x1057d68, GrrKind::HalfW_f,         GrrGroup::U1, "U1.halfW (960.0f) @ 0x1057d68"},
    {0x1057d6c, GrrKind::HalfHalign_f,    GrrGroup::U1, "U1.halfH (544.0f) @ 0x1057d6c"},
    {0x1057d80, GrrKind::QuarterW_f,      GrrGroup::U1, "U1.quartW(480.0f) @ 0x1057d80"},
    {0x1057d84, GrrKind::QuarterHalign_f, GrrGroup::U1, "U1.quartH(272.0f) @ 0x1057d84"},
    {0x1057d88, GrrKind::QuarterW_f,      GrrGroup::U1, "U1.quartW(480.0f) @ 0x1057d88"},
    {0x1057d8c, GrrKind::QuarterHalign_f, GrrGroup::U1, "U1.quartH(272.0f) @ 0x1057d8c"},
    {0x1057d90, GrrKind::HalfW_f,         GrrGroup::U1, "U1.halfW (960.0f) @ 0x1057d90"},
    {0x1057d94, GrrKind::HalfHalign_f,    GrrGroup::U1, "U1.halfH (544.0f) @ 0x1057d94"},
    {0x1057d98, GrrKind::HalfW_f,         GrrGroup::U1, "U1.halfW (960.0f) @ 0x1057d98"},
    {0x1057d9c, GrrKind::HalfHalign_f,    GrrGroup::U1, "U1.halfH (544.0f) @ 0x1057d9c"},
    {0x1057db0, GrrKind::QuarterW_f,      GrrGroup::U1, "U1.quartW(480.0f) @ 0x1057db0"},
    {0x1057db4, GrrKind::QuarterHalign_f, GrrGroup::U1, "U1.quartH(272.0f) @ 0x1057db4"},
    {0x1057db8, GrrKind::QuarterW_f,      GrrGroup::U1, "U1.quartW(480.0f) @ 0x1057db8"},
    {0x1057dbc, GrrKind::QuarterHalign_f, GrrGroup::U1, "U1.quartH(272.0f) @ 0x1057dbc"},

    // -------- Group C1: reference-resolution table - RISKY, opt-in (2) --------
    //  W -> final_W, H -> final_H EXACT (NOT aligned: stock is 1080, not 1088).
    {0x1045600, GrrKind::W_f,      GrrGroup::C1, "C1.W (1920.0f) @ 0x1045600 — camera/effect config table (risky)"},
    {0x1045604, GrrKind::Hexact_f, GrrGroup::C1, "C1.H (1080.0f EXACT) @ 0x1045604 — camera/effect config table (risky)"},

    // -------- Group S1: 3-RT MRT set (fmt 4) via direct 0x1acd0 in fn 0x14d30 (6) -
    //  W -> r8d imm, H -> r9d imm. The [rsp] arg (also 1920) is NOT read by
    //  0x1acd0's body, so it is intentionally left unpatched.
    {0x14d54, GrrKind::W_u32, GrrGroup::S1, "S1.W #1 (r8d 1920) @ 0x14d54 — scene MRT, call 0x1acd0"},
    {0x14d5a, GrrKind::H_u32, GrrGroup::S1, "S1.H #1 (r9d 1080) @ 0x14d5a — scene MRT, call 0x1acd0"},
    {0x14de9, GrrKind::W_u32, GrrGroup::S1, "S1.W #2 (r8d 1920) @ 0x14de9 — scene MRT, call 0x1acd0"},
    {0x14def, GrrKind::H_u32, GrrGroup::S1, "S1.H #2 (r9d 1080) @ 0x14def — scene MRT, call 0x1acd0"},
    {0x14e89, GrrKind::W_u32, GrrGroup::S1, "S1.W #3 (r8d 1920) @ 0x14e89 — scene MRT, call 0x1acd0"},
    {0x14e8f, GrrKind::H_u32, GrrGroup::S1, "S1.H #3 (r9d 1080) @ 0x14e8f — scene MRT, call 0x1acd0"},

    // -------- Group S2: single RT (fmt 9) via wrapper 0x153b0 @ 0x2842a (3) --------
    //  ecx=W, r8d=H, r9d=pitch(=W).
    {0x28417, GrrKind::W_u32, GrrGroup::S2, "S2.W (ecx 1920) @ 0x28417 — fmt9 RT, call 0x153b0"},
    {0x2841d, GrrKind::H_u32, GrrGroup::S2, "S2.H (r8d 1080) @ 0x2841d — fmt9 RT, call 0x153b0"},
    {0x28423, GrrKind::W_u32, GrrGroup::S2, "S2.pitch (r9d 1920) @ 0x28423 — fmt9 RT, call 0x153b0"},

    // -------- Group S3: single RT (fmt 2) via wrapper 0x153b0 @ 0x6762f (3) --------
    {0x6761c, GrrKind::W_u32, GrrGroup::S3, "S3.W (ecx 1920) @ 0x6761c — fmt2 RT, call 0x153b0"},
    {0x67622, GrrKind::H_u32, GrrGroup::S3, "S3.H (r8d 1080) @ 0x67622 — fmt2 RT, call 0x153b0"},
    {0x67628, GrrKind::W_u32, GrrGroup::S3, "S3.pitch (r9d 1920) @ 0x67628 — fmt2 RT, call 0x153b0"},

    // -------- Group S4: single RT (fmt 0xA) via wrapper 0x153b0 @ 0x24f666 (3) --------
    {0x24f653, GrrKind::W_u32, GrrGroup::S4, "S4.W (ecx 1920) @ 0x24f653 — fmtA RT, call 0x153b0"},
    {0x24f659, GrrKind::H_u32, GrrGroup::S4, "S4.H (r8d 1080) @ 0x24f659 — fmtA RT, call 0x153b0"},
    {0x24f65f, GrrKind::W_u32, GrrGroup::S4, "S4.pitch (r9d 1920) @ 0x24f65f — fmtA RT, call 0x153b0"},

    // -------- Group S5: half-res RT (fmt 9) via wrapper 0x153b0 @ 0x2878c (3) --------
    //  ecx=W(960), r8d=H(540 exact half), r9d=pitch(960).
    {0x28779, GrrKind::HalfW_u32,      GrrGroup::S5, "S5.W (ecx 960) @ 0x28779 — half-res fmt9, call 0x153b0"},
    {0x2877f, GrrKind::HalfH_u32,      GrrGroup::S5, "S5.H (r8d 540) @ 0x2877f — half-res fmt9, call 0x153b0"},
    {0x28785, GrrKind::HalfW_u32,      GrrGroup::S5, "S5.pitch (r9d 960) @ 0x28785 — half-res fmt9, call 0x153b0"},

    // -------- Group S6: half-res RT (fmt 11) via wrapper 0x153b0 @ 0x1b622e (3) --------
    //  ecx=W(960), r8d=H(544 = align64(1080)/2), r9d=pitch(960).
    {0x1b621b, GrrKind::HalfW_u32,      GrrGroup::S6, "S6.W (ecx 960) @ 0x1b621b — half-res fmt11, call 0x153b0"},
    {0x1b6221, GrrKind::HalfHalign_u32, GrrGroup::S6, "S6.H (r8d 544) @ 0x1b6221 — half-res fmt11, call 0x153b0"},
    {0x1b6227, GrrKind::HalfW_u32,      GrrGroup::S6, "S6.pitch (r9d 960) @ 0x1b6227 — half-res fmt11, call 0x153b0"},

    // -------- Group S7: 2nd-allocator scene RT (fmt2 1920x1080) via 0x161b0->0x1aea0
    //  @ 0x284ad. ecx=W(1920), r8d=H(1080). The GR2 "H group" analog.
    {0x2849a, GrrKind::W_u32, GrrGroup::S7, "S7.W (ecx 1920) @ 0x2849a — fmt2 RT, 2nd allocator (call 0x161b0->0x1aea0)"},
    {0x284a0, GrrKind::H_u32, GrrGroup::S7, "S7.H (r8d 1080) @ 0x284a0 — fmt2 RT, 2nd allocator (call 0x161b0->0x1aea0)"},
}};

// Raw GCN-bytecode patch sites (group T1): fixed 4-byte instruction overwrites, resolution-
// independent, in ph[1] (file = loaded_VA + 0xEC10, unlike ph[0]'s +0xB710; runtime is still
// eboot_base + va_offset). s_andn2_b64 -> s_and_b64 keeps the entry exec mask and sets SCC=1.
struct GrrBytePatchSite {
    std::uint32_t              va_offset;
    std::array<std::uint8_t,4> expected;     // stock 4 bytes (memcmp-guarded)
    std::array<std::uint8_t,4> replacement;  // patched 4 bytes
    GrrGroup                   group;
    const char*                label;
};

// 6 sites: 2 (ComicDemoTextScissor_fso, s24) + 4 (ComicDemoTextMaskedScissor_fso, s20).
constexpr std::array<GrrBytePatchSite, 6> kGrrBytePatchSites = {{
    // ComicDemoTextScissor_fso - 2-bound scissor; survivor mask s24.
    {0x16b1bd4, {0x18, 0x6a, 0x98, 0x8a}, {0x18, 0x18, 0x98, 0x87}, GrrGroup::T1,
     "T1 TextScissor s_andn2_b64 s24,s24,vcc -> s_and_b64 s24,s24,s24 (bound 0)"},
    {0x16b1be8, {0x18, 0x6a, 0x98, 0x8a}, {0x18, 0x18, 0x98, 0x87}, GrrGroup::T1,
     "T1 TextScissor s_andn2_b64 s24,s24,vcc -> s_and_b64 s24,s24,s24 (bound 1)"},
    // ComicDemoTextMaskedScissor_fso - 4-bound rect scissor; survivor mask s20.
    {0x16b305c, {0x14, 0x6a, 0x94, 0x8a}, {0x14, 0x14, 0x94, 0x87}, GrrGroup::T1,
     "T1 MaskedScissor s_andn2_b64 s20,s20,vcc -> s_and_b64 s20,s20,s20 (bound 0)"},
    {0x16b3078, {0x14, 0x6a, 0x94, 0x8a}, {0x14, 0x14, 0x94, 0x87}, GrrGroup::T1,
     "T1 MaskedScissor s_andn2_b64 s20,s20,vcc -> s_and_b64 s20,s20,s20 (bound 1)"},
    {0x16b308c, {0x14, 0x6a, 0x94, 0x8a}, {0x14, 0x14, 0x94, 0x87}, GrrGroup::T1,
     "T1 MaskedScissor s_andn2_b64 s20,s20,vcc -> s_and_b64 s20,s20,s20 (bound 2)"},
    {0x16b30a8, {0x14, 0x6a, 0x94, 0x8a}, {0x14, 0x14, 0x94, 0x87}, GrrGroup::T1,
     "T1 MaskedScissor s_andn2_b64 s20,s20,vcc -> s_and_b64 s20,s20,s20 (bound 3)"},
}};

// Variable-length in-place patch sites (group T2): fixed rewrites of len <= kGrrBlobMax bytes,
// resolution-independent, ph0-addressed (file = VA + 0xB710). The ring reader 0x67150's two
// 17-byte size blocks become `mov dword ptr [*out], <960.0f|544.0f>` + NOPs; origins untouched.
constexpr std::size_t kGrrBlobMax = 24;
struct GrrBlobPatchSite {
    std::uint32_t                       va_offset;
    std::uint32_t                       len;       // active bytes (<= kGrrBlobMax)
    std::array<std::uint8_t, kGrrBlobMax> expected; // first `len` are stock
    std::array<std::uint8_t, kGrrBlobMax> replacement;
    GrrGroup                            group;
    const char*                         label;
};

// 2 sites: the width and height components of the 0x67150 ring reader.
// 960.0f = 0x44700000 (LE 00 00 70 44), 544.0f = 0x44080000 (LE 00 00 08 44).
constexpr std::array<GrrBlobPatchSite, 2> kGrrBlobPatchSites = {{
    // comp2 (canvas half-WIDTH -> *rdx): 17-byte stock lea/vmovss/vmovss block rewritten to
    // c7 02 00 00 70 44 mov dword[rdx],0x44700000 (960.0f) + 11x NOP.
    {0x67194, 17,
     {0x48,0x8d,0x05,0xfd,0xc9,0xe6,0x01, 0xc4,0xa1,0x7a,0x10,0x04,0x80, 0xc5,0xfa,0x11,0x02, 0,0,0,0,0,0,0},
     {0xc7,0x02,0x00,0x00,0x70,0x44, 0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90, 0,0,0,0,0,0,0},
     GrrGroup::T2, "T2 comic canvas WIDTH @0x67194 -> 960.0f (frame-rect ring reader)"},
    // comp3 (canvas half-HEIGHT -> *rcx): 17-byte stock lea/vmovss/vmovss block rewritten to
    // c7 01 00 00 08 44 mov dword[rcx],0x44080000 (544.0f) + 11x NOP.
    {0x671a5, 17,
     {0x48,0x8d,0x05,0xf8,0xc9,0xe6,0x01, 0xc4,0xa1,0x7a,0x10,0x04,0x80, 0xc5,0xfa,0x11,0x01, 0,0,0,0,0,0,0},
     {0xc7,0x01,0x00,0x00,0x08,0x44, 0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90,0x90, 0,0,0,0,0,0,0},
     GrrGroup::T2, "T2 comic canvas HEIGHT @0x671a5 -> 544.0f (frame-rect ring reader)"},
}};

constexpr std::array<std::uint8_t, 4> GrrExpectedForKind(GrrKind k) {
    switch (k) {
    case GrrKind::W_f:             return {0x00, 0x00, 0xF0, 0x44}; // 1920.0f
    case GrrKind::Halign_f:        return {0x00, 0x00, 0x88, 0x44}; // 1088.0f
    case GrrKind::HalfW_f:         return {0x00, 0x00, 0x70, 0x44}; //  960.0f
    case GrrKind::HalfHalign_f:    return {0x00, 0x00, 0x08, 0x44}; //  544.0f
    case GrrKind::QuarterW_f:      return {0x00, 0x00, 0xF0, 0x43}; //  480.0f
    case GrrKind::QuarterHalign_f: return {0x00, 0x00, 0x88, 0x43}; //  272.0f
    case GrrKind::Hexact_f:        return {0x00, 0x00, 0x87, 0x44}; // 1080.0f
    case GrrKind::W_u32:           return {0x80, 0x07, 0x00, 0x00}; // 1920
    case GrrKind::H_u32:           return {0x38, 0x04, 0x00, 0x00}; // 1080
    case GrrKind::HalfW_u32:       return {0xC0, 0x03, 0x00, 0x00}; //  960
    case GrrKind::HalfH_u32:       return {0x1C, 0x02, 0x00, 0x00}; //  540
    case GrrKind::HalfHalign_u32:  return {0x20, 0x02, 0x00, 0x00}; //  544
    }
    return {0, 0, 0, 0};
}

std::array<std::uint8_t, 4> GrrEncodeForKind(const Resolution& sz, GrrKind k) {
    const float fw          = static_cast<float>(sz.width);
    const float fh_aligned  = static_cast<float>(Align64(static_cast<std::uint32_t>(sz.height)));
    const float fh_exact    = static_cast<float>(sz.height);
    switch (k) {
    case GrrKind::W_f:             return FloatBytes(fw);
    case GrrKind::Halign_f:        return FloatBytes(fh_aligned);
    case GrrKind::HalfW_f:         return FloatBytes(fw * 0.5f);
    case GrrKind::HalfHalign_f:    return FloatBytes(fh_aligned * 0.5f);
    case GrrKind::QuarterW_f:      return FloatBytes(fw * 0.25f);
    case GrrKind::QuarterHalign_f: return FloatBytes(fh_aligned * 0.25f);
    case GrrKind::Hexact_f:        return FloatBytes(fh_exact);
    case GrrKind::W_u32:           return U32Bytes(static_cast<std::uint32_t>(sz.width));
    case GrrKind::H_u32:           return U32Bytes(static_cast<std::uint32_t>(sz.height));
    case GrrKind::HalfW_u32:
        return U32Bytes(static_cast<std::uint32_t>(sz.width) / 2u);
    case GrrKind::HalfH_u32:
        return U32Bytes(static_cast<std::uint32_t>(sz.height) / 2u);
    case GrrKind::HalfHalign_u32:
        return U32Bytes(Align64(static_cast<std::uint32_t>(sz.height)) / 2u);
    }
    return {0, 0, 0, 0};
}

const char* GrrGroupName(GrrGroup g) {
    switch (g) {
    case GrrGroup::R1: return "R1";
    case GrrGroup::U1: return "U1";
    case GrrGroup::C1: return "C1";
    case GrrGroup::S1: return "S1";
    case GrrGroup::S2: return "S2";
    case GrrGroup::S3: return "S3";
    case GrrGroup::S4: return "S4";
    case GrrGroup::S5: return "S5";
    case GrrGroup::S6: return "S6";
    case GrrGroup::S7: return "S7";
    case GrrGroup::T1: return "T1";
    case GrrGroup::T2: return "T2";
    default:           return "?";
    }
}

// GRR group-mask parser; shares the GR2 config string with non-colliding tokens (grr_-prefixed
// or bare r1/u1/s1..s7/t1/t2; "grr_c1" is prefixed-only since bare "c1" is GR2's C1). Unknown
// tokens, including GR2's a1..q1, are no-ops on this path.
std::uint32_t ParseGrrGroupMaskFromConfig(std::string_view raw) {
    const std::string s = LowerTrim(raw);
    if (s.empty() || s == "recommended" || s == "prod") return kGrrMaskRecommended;
    if (s == "all" || s == "default")                   return kGrrMaskAll;
    if (s == "none" || s == "off")                      return 0u;

    std::uint32_t mask = 0;
    bool seen_any = false;

    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = i;
        while (j < s.size() && s[j] != ',' && s[j] != '|') j++;
        std::string_view tok(s.data() + i, j - i);
        i = j + 1;
        if (tok.empty()) continue;

        bool negate = false;
        if (tok.front() == '~' || tok.front() == '!') { negate = true; tok.remove_prefix(1); }
        if (tok.empty()) continue;

        std::uint32_t add = 0;
        bool known = true;
        if      (tok == "all" || tok == "default")      add = kGrrMaskAll;
        else if (tok == "recommended" || tok == "prod") add = kGrrMaskRecommended;
        else if (tok == "safe" || tok == "min")         add = GrrGroupBit(GrrGroup::R1); // pure R1, no T1
        else if (tok == "none" || tok == "off")         add = 0;
        else if (tok == "grr_r1" || tok == "r1")        add = GrrGroupBit(GrrGroup::R1);
        else if (tok == "grr_u1" || tok == "u1")        add = GrrGroupBit(GrrGroup::U1);
        else if (tok == "grr_c1")                       add = GrrGroupBit(GrrGroup::C1);
        else if (tok == "grr_s1" || tok == "s1")        add = GrrGroupBit(GrrGroup::S1);
        else if (tok == "grr_s2" || tok == "s2")        add = GrrGroupBit(GrrGroup::S2);
        else if (tok == "grr_s3" || tok == "s3")        add = GrrGroupBit(GrrGroup::S3);
        else if (tok == "grr_s4" || tok == "s4")        add = GrrGroupBit(GrrGroup::S4);
        else if (tok == "grr_s5" || tok == "s5")        add = GrrGroupBit(GrrGroup::S5);
        else if (tok == "grr_s6" || tok == "s6")        add = GrrGroupBit(GrrGroup::S6);
        else if (tok == "grr_s7" || tok == "s7")        add = GrrGroupBit(GrrGroup::S7);
        else if (tok == "grr_t1" || tok == "t1")        add = GrrGroupBit(GrrGroup::T1);
        else if (tok == "grr_t2" || tok == "t2")        add = GrrGroupBit(GrrGroup::T2);
        else                                            known = false; // GR2-only / unknown - no-op here

        if (known) {
            if (negate) mask &= ~add;
            else        mask |= add;
            seen_any = true;
        }
    }

    return seen_any ? mask : kGrrMaskRecommended;
}

} // namespace

std::uint32_t ParseGroupMaskFromConfig(std::string_view raw,
                                       std::uint32_t recommended_override = kGroupMaskRecommended) {
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
        else if (tok == "recommended" || tok == "prod") add = recommended_override;
        else if (tok == "safe" || tok == "min")    add = kGroupMaskSafe;
        else if (tok == "geom")                    add = kGroupMaskGeom;
        else if (tok == "ext"  || tok == "extended") add = kGroupMaskExt;
        else if (tok == "none" || tok == "off")    add = 0;
        // Bisection presets: each adds exactly one group on top of the known-desync baseline
        // (B1+D1) so a test isolates a single variable.
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

// Resolution-aware recommended mask: above 1080p auto-add E1 to keep text pixel-perfect (the
// glyph atlas otherwise stretches and text overflows speech bubbles); at or below 1080p add
// nothing - text should shrink with the framebuffer. User `~tok` negations apply on top.
constexpr std::uint32_t kTextScalingExtras_AboveDesign =
    // C2/C3 and I1-I4 stay out of recommended at every height: at non-1080p the I-groups'
    // projection-lock constants offset screen-space effects (e.g. the gravity-field overlay)
    // from their world anchor. Only E1 is auto-added above design res.
      static_cast<std::uint32_t>(PatchGroupBit::E1);

constexpr std::uint32_t kTextScalingExtras_BelowDesign =
    // C2/C3 + I1/I2/I3/I4 excluded from recommended entirely.
    // Nothing is auto-added below design res.
    0u;

std::uint32_t ResolutionAwareRecommended(int target_height) {
    std::uint32_t mask = kGroupMaskRecommended;
    if (target_height > 1080)      mask |= kTextScalingExtras_AboveDesign;
    else if (target_height < 1080) mask |= kTextScalingExtras_BelowDesign;
    return mask;
}

int ApplyGr2ResolutionPatches(uintptr_t eboot_base, TargetResolution resolution,
                              float aspect_ratio,
                              std::string_view groups_config,
                              ResPatchStats* stats) {
    if (stats) {
        *stats = ResPatchStats{};
    }
    if (eboot_base == 0) {
        LOG_ERROR(Core, "[GR2 Resolution] eboot_base is 0 — cannot patch");
        return 0;
    }

    // Build the group mask from config.toml [GPU] resolutionPatchGroups (default "recommended"):
    // presets, group tokens (a1..m1), ~/! negations ("recommended,~m1", "all,~e1,~g1,~k1"); the
    // parsed mask is logged. Final dims are computed early for the mask logic and the encoders.
    const Resolution final_sz = ComposeFinalImpl(resolution, aspect_ratio);

    // kGroupMaskRecommended is augmented by target height: above 1080p adds E1 (pixel-perfect
    // glyph atlas); below, text shrinks proportionally instead. Explicit `~e1` still overrides.
    const std::uint32_t effective_recommended =
        (resolution != TargetResolution::Off)
            ? ResolutionAwareRecommended(final_sz.height)
            : kGroupMaskRecommended;

    // The chosen mask is logged with the parsed-from string so post-
    // mortem bisection can match the log line against the config value.
    std::uint32_t group_mask = 0;
    if (resolution != TargetResolution::Off) {
        const std::string cfg(groups_config);
        const std::uint32_t parsed = ParseGroupMaskFromConfig(cfg, effective_recommended);
        if (cfg.empty() || cfg == "recommended" || cfg == "prod") {
            // Hot path: don't spam the log when running on defaults.
            group_mask |= (cfg.empty() ? effective_recommended : parsed);
        } else {
            LOG_INFO(Core,
                "[GR2 Resolution] resolutionPatchGroups='{}' parsed to "
                "mask=0x{:08x} (effective_recommended=0x{:08x}, base=0x{:08x})",
                cfg, parsed, effective_recommended, kGroupMaskRecommended);
            group_mask |= parsed;
        }
        // One-shot log of the resolution-aware augmentation so it's
        // discoverable in the log without surprise.
        const std::uint32_t augmentation = effective_recommended & ~kGroupMaskRecommended;
        if (augmentation != 0) {
            LOG_INFO(Core,
                "[GR2 Resolution] target H={} → auto-added groups to recommended: "
                "0x{:08x} (text-scaling fixes for non-1080p targets)",
                final_sz.height, augmentation);
        }
    }
    if (group_mask == 0) {
        LOG_INFO(Core, "[GR2 Resolution] target=Off; nothing to patch");
        return 0;
    }

    // M1 gate + apply-time BSS write. M1 only makes sense at >= R2160p (the cmovne 4K branch
    // hard-codes 3840/2160), so lower targets strip it from the mask. The BSS write at
    // 0x01b586c0 runs before any game code, covering readers that run before the patched writers.
    if ((group_mask & static_cast<std::uint32_t>(PatchGroupBit::M1)) != 0) {
        if (static_cast<int>(resolution) < static_cast<int>(TargetResolution::R2160p)) {
            LOG_INFO(Core,
                "[GR2 Resolution] M1 requested but target={} (<2160p) — stripped "
                "from mask (M1's hard-coded 4K-branch dims would mismatch sub-4K RTs)",
                TargetName(resolution));
            group_mask &= ~static_cast<std::uint32_t>(PatchGroupBit::M1);
        } else {
            constexpr std::uint32_t kBssFlagOffset = 0x01b586c0;
            // Defense-in-depth for the one fixed-offset write with no per-site byte check (the
            // GRR eboot keeps real .data at 0x01b586c0): require the B1.W site (0x0102aa21) to
            // hold stock 1920 or the patched target_W. M1 needs B1 anyway, so skipping is correct.
            constexpr std::uint32_t kB1WOffset = 0x0102aa21;
            const std::uint8_t* b1w =
                reinterpret_cast<const std::uint8_t*>(eboot_base + kB1WOffset);
            const std::uint8_t stock_w[4] = {0x80, 0x07, 0x00, 0x00}; // 1920
            const auto target_w_bytes =
                U32Bytes(static_cast<std::uint32_t>(final_sz.width));
            const bool b1_is_gr2 =
                (std::memcmp(b1w, stock_w, 4) == 0) ||
                (std::memcmp(b1w, target_w_bytes.data(), 4) == 0);
            if (!b1_is_gr2) {
                LOG_INFO(Core,
                    "[GR2 Resolution] M1: BSS_FLAG write SKIPPED — B1.W @ 0x{:x} "
                    "holds {:02x} {:02x} {:02x} {:02x} (neither stock 1920 nor "
                    "target_W={}); does not look like the GR2 eboot",
                    kB1WOffset, b1w[0], b1w[1], b1w[2], b1w[3], final_sz.width);
            } else {
                auto* flag_ptr =
                    reinterpret_cast<std::uint8_t*>(eboot_base + kBssFlagOffset);
                const std::uint8_t prev = *flag_ptr;
                *flag_ptr = 0x01;
                LOG_INFO(Core,
                    "[GR2 Resolution] M1: wrote BSS_FLAG[0x{:x}]=1 (was 0x{:02x}) — PS4-Pro 4K path enabled",
                    kBssFlagOffset, prev);
            }
        }
    }

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

    // I1 instruction replacements do not scale with the target resolution - the replacement
    // bytes hard-code literal 0x780/0x438 (1920/1080), locking the UI projection at design res.
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

    // N1: the sceVideoOutSetBufferAttribute caller at VA 0x00446975 reads packed W|H<<16 and a
    // pitch code from BSS slots written via untraceable indirect pointers; the loads at
    // 0x00446933/39 become same-size immediates (pitch_enc = (W-8)/8, packed_wh = W | H<<16).
    if ((group_mask & static_cast<std::uint32_t>(PatchGroupBit::N1)) != 0) {
        const std::uint32_t target_W = static_cast<std::uint32_t>(final_sz.width);
        const std::uint32_t target_H = static_cast<std::uint32_t>(final_sz.height);
        const std::uint32_t pitch_enc = (target_W >= 8) ? ((target_W - 8) / 8) : 0;
        const std::uint32_t packed_wh = (target_W & 0xFFFFu) | ((target_H & 0xFFFFu) << 16);

        struct N1Site {
            std::uint32_t va_offset;
            std::uint8_t  size;
            std::uint8_t  expected[8];
            std::uint8_t  rep_prefix[2];  // bytes before the imm32
            std::uint8_t  rep_prefix_len;
            std::uint8_t  rep_suffix;     // single byte after the imm32 (nop)
            std::uint32_t imm32;
            const char*   label;
        };
        const N1Site n1_sites[2] = {
            // mov eax, [rip+0x16ea9df]  ->  mov eax, pitch_enc ; nop
            {0x00446933, 6,
             {0x8B, 0x05, 0xDF, 0xA9, 0x6E, 0x01, 0, 0},
             {0xB8, 0x00}, 1, 0x90, pitch_enc,
             "N1.PITCH_ENC (mov eax, [rip+0x16ea9df]) → mov eax, imm32 ; nop"},
            // mov r9d, [rip+0x16eaa10]  ->  mov r9d, packed_wh ; nop
            {0x00446939, 7,
             {0x44, 0x8B, 0x0D, 0x10, 0xAA, 0x6E, 0x01, 0},
             {0x41, 0xB9}, 2, 0x90, packed_wh,
             "N1.PACKED_WH (mov r9d, [rip+0x16eaa10]) → mov r9d, imm32 ; nop"},
        };

        for (const auto& n : n1_sites) {
            std::uint8_t* p = reinterpret_cast<std::uint8_t*>(eboot_base + n.va_offset);

            // Build the replacement bytes inline.
            std::uint8_t rep[8] = {0};
            std::memcpy(rep, n.rep_prefix, n.rep_prefix_len);
            std::memcpy(rep + n.rep_prefix_len, &n.imm32, 4);
            rep[n.rep_prefix_len + 4] = n.rep_suffix;
            // size = prefix + 4 + 1 (suffix nop) = 6 or 7

            if (std::memcmp(p, rep, n.size) == 0) {
                per_group_already[static_cast<int>(Group::N1)]++;
                continue;
            }
            if (std::memcmp(p, n.expected, n.size) != 0) {
                LOG_WARNING(Core,
                    "[GR2 Resolution] {} @ vaddr 0x{:x}: unexpected bytes — NOT patched",
                    n.label, n.va_offset);
                per_group_mismatch[static_cast<int>(Group::N1)]++;
                continue;
            }
            std::memcpy(p, rep, n.size);
            LOG_INFO(Core,
                "[GR2 Resolution] {} (imm32=0x{:08x}) @ vaddr 0x{:x} patched",
                n.label, n.imm32, n.va_offset);
            per_group_patched[static_cast<int>(Group::N1)]++;
        }
        LOG_INFO(Core,
            "[GR2 Resolution] N1: pitch_enc=0x{:x} (= ({}-8)/8), packed_wh=0x{:08x} (W={} H={})",
            pitch_enc, target_W, packed_wh, target_W, target_H);
    }

    // O1: with N1 alone the game still allocates 8 MB 1080p swap-chain buffers and TextureCache
    // walks off them. The allocator wrapper 0x1215570 gets (W, H) from four imm16 movs at
    // 0x00446823/27/2b/2f (cmova 4K alternates); patching all four makes both paths target dims.
    if ((group_mask & static_cast<std::uint32_t>(PatchGroupBit::O1)) != 0) {
        const std::uint16_t target_W = static_cast<std::uint16_t>(final_sz.width);
        const std::uint16_t target_H = static_cast<std::uint16_t>(final_sz.height);

        struct O1Site {
            std::uint32_t instr_va;
            std::uint8_t  instr_bytes[4];  // full 4-byte expected instruction
            bool          is_height;       // true -> patch with target_H, false -> target_W
            const char*   label;
        };
        const O1Site o1_sites[4] = {
            {0x00446823, {0x66, 0xB8, 0x70, 0x08}, true,
             "O1.H_ALT     (mov ax, 0x870 — cmova-alt H, PS4-Pro 4K branch)"},
            {0x00446827, {0x66, 0xB9, 0x38, 0x04}, true,
             "O1.H_DEFAULT (mov cx, 0x438 — default H, base-PS4 fall-through)"},
            {0x0044682B, {0x66, 0xBE, 0x00, 0x0F}, false,
             "O1.W_ALT     (mov si, 0xf00 — cmova-alt W, PS4-Pro 4K branch)"},
            {0x0044682F, {0x66, 0xBA, 0x80, 0x07}, false,
             "O1.W_DEFAULT (mov dx, 0x780 — default W, base-PS4 fall-through)"},
        };

        for (const auto& o : o1_sites) {
            std::uint8_t* p = reinterpret_cast<std::uint8_t*>(eboot_base + o.instr_va);
            const std::uint16_t target_val = o.is_height ? target_H : target_W;

            // Build expected replacement: same opcode prefix, new imm16.
            std::uint8_t expected_rep[4] = {o.instr_bytes[0], o.instr_bytes[1], 0, 0};
            std::memcpy(expected_rep + 2, &target_val, 2);

            if (std::memcmp(p, expected_rep, 4) == 0) {
                per_group_already[static_cast<int>(Group::O1)]++;
                continue;
            }
            // Check the original bytes (opcode+default imm16).
            if (std::memcmp(p, o.instr_bytes, 4) != 0) {
                LOG_WARNING(Core,
                    "[GR2 Resolution] {} @ vaddr 0x{:x}: unexpected bytes — NOT patched",
                    o.label, o.instr_va);
                per_group_mismatch[static_cast<int>(Group::O1)]++;
                continue;
            }
            // Patch ONLY the imm16 (last 2 bytes).
            std::memcpy(p + 2, &target_val, 2);
            LOG_INFO(Core,
                "[GR2 Resolution] {} (imm16=0x{:04x}) @ vaddr 0x{:x} patched",
                o.label, target_val, o.instr_va + 2);
            per_group_patched[static_cast<int>(Group::O1)]++;
        }
        LOG_INFO(Core,
            "[GR2 Resolution] O1: target_W=0x{:x} ({}) target_H=0x{:x} ({}) — "
            "allocator wrapper @ 0x1215570 will receive target dims",
            target_W, target_W, target_H, target_H);
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

    if (stats) {
        stats->patched = total_patched;
        stats->already = total_already;
        stats->mismatched = total_mismatch;
        stats->skipped = total_skipped;
    }
    return total_patched;
}

int ApplyGrrResolutionPatches(uintptr_t eboot_base, TargetResolution resolution,
                              float aspect_ratio,
                              std::string_view groups_config,
                              ResPatchStats* stats) {
    if (stats) {
        *stats = ResPatchStats{};
    }
    if (eboot_base == 0) {
        LOG_ERROR(Core, "[GRR Resolution] eboot_base is 0 — cannot patch");
        return 0;
    }
    if (resolution == TargetResolution::Off) {
        LOG_INFO(Core, "[GRR Resolution] target=Off; nothing to patch");
        return 0;
    }

    // Final framebuffer dims compose exactly as on the GR2 path (pixel
    // density x aspect); GRR then aligns the RT HEIGHT up to a multiple of 64.
    const Resolution final_sz = ComposeFinalImpl(resolution, aspect_ratio);
    const std::uint32_t aligned_h =
        Align64(static_cast<std::uint32_t>(final_sz.height));

    const std::uint32_t group_mask = ParseGrrGroupMaskFromConfig(groups_config);
    if (group_mask == 0) {
        LOG_INFO(Core,
            "[GRR Resolution] group mask is empty (groups='{}') — nothing to patch",
            std::string(groups_config));
        return 0;
    }

    s32 per_group_patched[static_cast<int>(GrrGroup::Count)] = {0};
    s32 per_group_already[static_cast<int>(GrrGroup::Count)] = {0};
    s32 per_group_mismatch[static_cast<int>(GrrGroup::Count)] = {0};
    s32 per_group_skipped[static_cast<int>(GrrGroup::Count)] = {0};

    for (const auto& site_info : kGrrResSites) {
        const std::uint32_t bit = GrrGroupBit(site_info.group);
        if ((group_mask & bit) == 0) {
            per_group_skipped[static_cast<int>(site_info.group)]++;
            continue;
        }

        std::uint8_t* site = reinterpret_cast<std::uint8_t*>(eboot_base + site_info.va_offset);
        const auto expected    = GrrExpectedForKind(site_info.kind);
        const auto replacement = GrrEncodeForKind(final_sz, site_info.kind);

        if (std::memcmp(site, replacement.data(), 4) == 0) {
            per_group_already[static_cast<int>(site_info.group)]++;
            continue;
        }
        if (std::memcmp(site, expected.data(), 4) != 0) {
            LOG_WARNING(Core,
                "[GRR Resolution] {} @ vaddr 0x{:x}: unexpected bytes "
                "(got {:02x} {:02x} {:02x} {:02x}, expected {:02x} {:02x} {:02x} {:02x}) — NOT patched",
                site_info.label, site_info.va_offset,
                site[0], site[1], site[2], site[3],
                expected[0], expected[1], expected[2], expected[3]);
            per_group_mismatch[static_cast<int>(site_info.group)]++;
            continue;
        }
        std::memcpy(site, replacement.data(), 4);
        per_group_patched[static_cast<int>(site_info.group)]++;
    }

    // Raw GCN-bytecode patches (group T1): fixed expected[]/replacement[] bytes with the same
    // three-way memcmp guard (replacement = already applied, expected = write, else warn+skip),
    // counted into the same per-group arrays. ph[1] sites; eboot_base + va_offset still applies.
    for (const auto& bp : kGrrBytePatchSites) {
        const std::uint32_t bit = GrrGroupBit(bp.group);
        if ((group_mask & bit) == 0) {
            per_group_skipped[static_cast<int>(bp.group)]++;
            continue;
        }

        std::uint8_t* site = reinterpret_cast<std::uint8_t*>(eboot_base + bp.va_offset);

        if (std::memcmp(site, bp.replacement.data(), 4) == 0) {
            per_group_already[static_cast<int>(bp.group)]++;
            continue;
        }
        if (std::memcmp(site, bp.expected.data(), 4) != 0) {
            LOG_WARNING(Core,
                "[GRR Resolution] {} @ vaddr 0x{:x}: unexpected bytes "
                "(got {:02x} {:02x} {:02x} {:02x}, expected {:02x} {:02x} {:02x} {:02x}) — NOT patched",
                bp.label, bp.va_offset,
                site[0], site[1], site[2], site[3],
                bp.expected[0], bp.expected[1], bp.expected[2], bp.expected[3]);
            per_group_mismatch[static_cast<int>(bp.group)]++;
            continue;
        }
        std::memcpy(site, bp.replacement.data(), 4);
        per_group_patched[static_cast<int>(bp.group)]++;
    }

    // Variable-length in-place rewrites (group T2): same three-way guard over bp.len bytes,
    // counted into the same per-group arrays. ph0 sites (file = VA + 0xB710).
    for (const auto& bp : kGrrBlobPatchSites) {
        const std::uint32_t bit = GrrGroupBit(bp.group);
        if ((group_mask & bit) == 0) {
            per_group_skipped[static_cast<int>(bp.group)]++;
            continue;
        }
        if (bp.len == 0 || bp.len > kGrrBlobMax) {
            LOG_WARNING(Core, "[GRR Resolution] {} @ vaddr 0x{:x}: bad blob len {} — skipped",
                        bp.label, bp.va_offset, bp.len);
            per_group_skipped[static_cast<int>(bp.group)]++;
            continue;
        }

        std::uint8_t* site = reinterpret_cast<std::uint8_t*>(eboot_base + bp.va_offset);

        if (std::memcmp(site, bp.replacement.data(), bp.len) == 0) {
            per_group_already[static_cast<int>(bp.group)]++;
            continue;
        }
        if (std::memcmp(site, bp.expected.data(), bp.len) != 0) {
            LOG_WARNING(Core,
                "[GRR Resolution] {} @ vaddr 0x{:x}: unexpected bytes "
                "(first byte got {:02x}, expected {:02x}) over {} bytes — NOT patched",
                bp.label, bp.va_offset, site[0], bp.expected[0], bp.len);
            per_group_mismatch[static_cast<int>(bp.group)]++;
            continue;
        }
        std::memcpy(site, bp.replacement.data(), bp.len);
        per_group_patched[static_cast<int>(bp.group)]++;
    }

    s32 total_patched = 0, total_already = 0, total_mismatch = 0, total_skipped = 0;
    for (int i = 0; i < static_cast<int>(GrrGroup::Count); i++) {
        total_patched  += per_group_patched[i];
        total_already  += per_group_already[i];
        total_mismatch += per_group_mismatch[i];
        total_skipped  += per_group_skipped[i];
    }

    LOG_INFO(Core,
        "[GRR Resolution] target={} aspect={:.4f} -> final={}x{} "
        "(RT height aligned to {} = align64({})) mask=0x{:04x} — "
        "{} patched, {} already, {} mismatched, {} skipped (of {} total)",
        TargetName(resolution), aspect_ratio,
        final_sz.width, final_sz.height, aligned_h, final_sz.height,
        group_mask, total_patched, total_already, total_mismatch, total_skipped,
        static_cast<int>(kGrrResSites.size() + kGrrBytePatchSites.size()
                         + kGrrBlobPatchSites.size()));

    for (int gi = 0; gi < static_cast<int>(GrrGroup::Count); gi++) {
        const auto g = static_cast<GrrGroup>(gi);
        const int p = per_group_patched[gi], a = per_group_already[gi];
        const int m = per_group_mismatch[gi], k = per_group_skipped[gi];
        if (p + a + m + k == 0) continue;
        const bool enabled = (group_mask & GrrGroupBit(g)) != 0;
        LOG_INFO(Core,
            "[GRR Resolution]   {} ({}): patched={} already={} mismatch={} skipped={}",
            GrrGroupName(g), enabled ? "ENABLED" : "disabled", p, a, m, k);
    }

    if (stats) {
        stats->patched = total_patched;
        stats->already = total_already;
        stats->mismatched = total_mismatch;
        stats->skipped = total_skipped;
    }
    return total_patched;
}

int ApplyResolutionPatches(uintptr_t eboot_base, TargetResolution resolution,
                           float aspect_ratio,
                           std::string_view gr2_groups_config,
                           std::string_view grr_groups_config) {
    if (eboot_base == 0) {
        LOG_ERROR(Core, "[Resolution] eboot_base is 0 — cannot patch");
        return 0;
    }
    if (resolution == TargetResolution::Off) {
        LOG_INFO(Core, "[Resolution] resolutionOverride=Off — resolution patching disabled");
        return 0;
    }

    const bool is_grr = Config::isGravityRushRemastered();

    // GR2 first unless the GRR title flag (set in emulator.cpp before the eboot loads) is set:
    // GR2's 540-site loop on a GRR eboot spams "unexpected bytes" warnings and its M1 BSS write
    // lands in GRR's real .data. Without the flag, GR2 runs first and GRR is the zero-hit fallback.
    if (!is_grr) {
        ResPatchStats gr2{};
        const int gr2_patched = ApplyGr2ResolutionPatches(
            eboot_base, resolution, aspect_ratio, gr2_groups_config, &gr2);
        const bool gr2_matched = (gr2.patched > 0 || gr2.already > 0);
        if (gr2_matched) {
            return gr2_patched;
        }
        LOG_INFO(Core,
            "[Resolution] GR2 resolution patches matched no sites "
            "(patched={} already={} mismatched={}) — falling back to GRR",
            gr2.patched, gr2.already, gr2.mismatched);
    } else {
        LOG_INFO(Core,
            "[Resolution] Gravity Rush Remastered detected — using GRR "
            "resolution patches (GR2 path skipped)");
    }

    // GRR pass - reached directly when the title flag is set, or as the GR2 zero-hit fallback.
    ResPatchStats grr{};
    const int grr_patched = ApplyGrrResolutionPatches(
        eboot_base, resolution, aspect_ratio, grr_groups_config, &grr);
    const bool grr_matched = (grr.patched > 0 || grr.already > 0);
    if (grr_matched) {
        return grr_patched;
    }

    LOG_WARNING(Core,
        "[Resolution] neither GR2 nor GRR resolution constants matched this "
        "eboot (GRR: patched={} already={} mismatched={}) — no resolution "
        "patches applied. The eboot may be an unsupported title or version.",
        grr.patched, grr.already, grr.mismatched);
    return 0;
}

TargetResolution ParseResolutionFromConfig(std::string_view s) {
    if (s == "270p"  || s == "480x270")                  return TargetResolution::R270p;
    if (s == "360p"  || s == "640x360")                  return TargetResolution::R360p;
    if (s == "480p"  || s == "856x480")                  return TargetResolution::R480p;
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
