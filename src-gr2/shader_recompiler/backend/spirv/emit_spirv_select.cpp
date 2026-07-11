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

// GR2FORK: GRR composite (fs 0x9a4b146c) highlight ceiling. The tonemap scale (the shader's
// single SelectF32) multiplies all channels before a per-channel clamp, so bright coloured pixels
// wash to white. GR2_TONE_CAP=<f> caps it (default 1.6; read once at shader compile).
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
    // GR2FORK: in the GRR composite this is the tonemap scale (%145); the hash gate uniquely
    // identifies it as the only SelectF32 in fs 0x9a4b146c. Function-local static: ElfInfo is
    // queried on first shader compile, not at program startup.
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
