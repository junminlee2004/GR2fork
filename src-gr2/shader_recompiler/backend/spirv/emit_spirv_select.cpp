// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <string_view>

#include "common/elf_info.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

// [GR2FORK] Gravity Rush Remastered composite (fs 0x9a4b146c) highlight ceiling.
//
// The composite's tonemap (verified from the dumped IR of this exact shader) is:
//   L      = 0.299*R + 0.587*G + 0.114*B            (Rec.601 luminance of exposed scene+bloom)
//   tone   = (1 + L*CB15) / (1 + L)                 (extended Reinhard; CB15 = "emission")
//   tone   = Select(mask > 0.5, CB14, tone)         (flat replacement in the masked region)
//   out_ch = min(1, exposed_ch * tone)              (PER-CHANNEL clamp -- the white-out happens HERE)
//
// `tone` is a single luminance-derived scalar multiplied into all three channels, after which each
// channel is independently clamped at 1.0. For a bright coloured pixel (fountain, 3D UI markers,
// enemy orbs) `exposed_ch * tone` exceeds 1 in every channel, so all three clamp to (1,1,1) and the
// hue is destroyed -- blue/yellow/green read as flat white. Raising CB15 ("emission") only makes
// `tone` larger for high-L pixels (it asymptotes to CB15), so it pushes those elements to white
// HARDER while leaving Kat / shadows (low L, tone ~= 1) untouched. The downstream rolloff (CB12)
// acts on the already-clamped value, so it can only uniformly dim a blown pixel, never recolour it.
//
// The only place to keep those elements from saturating is `tone` itself, BEFORE the per-channel
// clamp. `tone` is the single SelectF32 in this shader (%145 in the IR dump), so we clamp the result
// of that select to a ceiling. Effect:
//   * Dark / mid pixels (tone <= cap): untouched -- lift those with GR2_EXPOSURE (CB0), the real
//     linear brightness lever; emission stays available for mid-bright glow under the cap.
//   * Bright pixels (tone would exceed cap): held at the cap, so the dominant channel saturates far
//     less and the subordinate channels stay < 1 -- the marker stays coloured instead of going white.
//   Lower cap = more hue retained on the hottest elements (and a harder highlight knee).
//
// GR2_TONE_CAP=<f> -> ceiling on the composite tone-scale multiplier. Defaults to 1.6 when unset.
// Read once at shader compile, same as the other GR2_* knobs, so the composite must be re-emitted
// (clear/disable the shader+pipeline cache) for a change to take effect.
static constexpr u64 kGrrCompositePgmHash = 0x9a4b146cull;
static const f32 kGrrToneCap = []() -> f32 {
    if (const char* e = std::getenv("GR2_TONE_CAP")) {
        return static_cast<f32>(std::strtod(e, nullptr));
    }
    return 1.6f;
}();

Id EmitSelectU1(EmitContext& ctx, Id cond, Id true_value, Id false_value) {
    return ctx.OpSelect(ctx.U1[1], cond, true_value, false_value);
}

Id EmitSelectU32(EmitContext& ctx, Id cond, Id true_value, Id false_value) {
    return ctx.OpSelect(ctx.U32[1], cond, true_value, false_value);
}

Id EmitSelectF32(EmitContext& ctx, Id cond, Id true_value, Id false_value) {
    const Id result = ctx.OpSelect(ctx.F32[1], cond, true_value, false_value);
    // [GR2FORK] In the GRR composite this is the tonemap scale (%145). Cap it so the hottest pixels
    // can't run every channel past the downstream per-channel clamp and wash to white. This is the
    // ONLY SelectF32 in fs 0x9a4b146c, so the hash gate uniquely identifies it.
    // Function-local static: ElfInfo queried on first shader compile, not at program startup.
    static const bool kIsGravityRushRemastered = [] {
        static constexpr std::array<std::string_view, 12> kGrrSerials = {
            "CUSA01112", "CUSA01113", "CUSA01113P", "CUSA01130",
            "CUSA02318", "CUSA00546", "CUSA02443",  "CUSA04246",
            "PCJS50004", "PCJS50008", "PCJS66015",  "PCJS66029",
        };
        const auto serial = Common::ElfInfo::Instance().GameSerial();
        return std::find(kGrrSerials.begin(), kGrrSerials.end(), serial) != kGrrSerials.end();
    }();
    if (kIsGravityRushRemastered && ctx.info.pgm_hash == kGrrCompositePgmHash) {
        return ctx.OpFMin(ctx.F32[1], result, ctx.ConstF32(kGrrToneCap));
    }
    return result;
}

} // namespace Shader::Backend::SPIRV
