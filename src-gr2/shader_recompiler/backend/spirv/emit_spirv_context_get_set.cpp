// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/config.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "shader_recompiler/backend/spirv/emit_spirv_bounds.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/attribute.h"
#include "shader_recompiler/ir/patch.h"
#include "shader_recompiler/runtime_info.h"

#include <magic_enum/magic_enum.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Shader::Backend::SPIRV {

using PointerType = EmitContext::PointerType;
using PointerSize = EmitContext::PointerSize;

// GR2FORK: GRR's final composite (fs 0x9a4b146c) multiplies the frame by an exposure it reads
// from constant-buffer #0 dword 0, which arrives at ~0.40 and darkens the frame versus PS4. A
// multiplier (see EmitReadConstBuffer) preserves the game's auto-exposure; 0.40 * 2.5 == 1.0.
static constexpr f32 kGrrExposureBoost = 1.0f;
static constexpr u64 kGrrCompositePgmHash = 0x9a4b146cull;
static constexpr u64 kGrrSkyboxPgmHash = 0xd65913b0ull; // skybox (sky+sun appearance); GR2_SKY_WARMTH tints it
static constexpr u64 kGrrCloudPgmHash = 0xc7508c8dull;  // clouds; GR2_CLOUD_BRIGHTNESS scales it

// GR2FORK: lit-material shaders with sun constants in CB slot 2: direction at CB2[20..22];
// strength is scalar CB2[27] on character/NPC shaders, colour triplet CB2[16/17/18] on world
// surfaces. GR2_SUN_* pins to these hashes - fs 0x222149f4/0x8d3aa7a0 read CB2[17] as non-sun data.
static constexpr std::array<u64, 7> kGrrSunCharShaders = {
    0xb30c6b28ull, 0xc5f496c0ull, 0x4d450ae2ull, 0xb4adb7b9ull,
    0x91a764f1ull, 0xe9ba8491ull, 0x504b03e0ull}; // last three: Kat gravity-shift (shift-state body)
static constexpr std::array<u64, 5> kGrrSunWorldShaders = {
    0x18c39cfcull, 0x59bfa3bdull, 0x65da829eull, 0xb062b18cull, 0xd46b4b82ull};

// GR2FORK: bloom intensity. The bloom texture (fs_img16, the 480x270 blur) is multiplied
// per-channel by CB#0 dwords 8/9/10 and added to the scene (dword 11 is a depth-gated boost);
// scaling 8/9/10 together scales overall bloom while preserving its tint. 1.0 == stock.
static constexpr f32 kGrrBloomScale = 0.6f;

// GR2FORK: highlight shaping in the composite's extended-Reinhard tonemap. kGrrHighlightLift
// scales dword 15 (white-point term - scales with luma, so mid-tones barely move) and
// kGrrHighlightRolloff scales dword 12 (compresses the brightest pixels). Both 1.0 == stock.
static constexpr f32 kGrrHighlightLift = 1.0f;    // dword 15
static constexpr f32 kGrrHighlightRolloff = 1.0f; // dword 12

// Permanent (dword -> multiplier) table baked into every build. Entries at 1.0 are no-ops.
static constexpr std::array<std::pair<u32, f32>, 6> kGrrPermanentScales{{
    {0u, kGrrExposureBoost},
    {8u, kGrrBloomScale},
    {9u, kGrrBloomScale},
    {10u, kGrrBloomScale},
    {12u, kGrrHighlightRolloff},
    {15u, kGrrHighlightLift},
}};

// GR2FORK: live tuning: GR2_EXPOSURE (dword 0, linear pre-tonemap gain, lifts darks), GR2_BLOOM
// (dwords 8/9/10), GR2_EMISSION (dword 15, lifts only brights), GR2_ROLLOFF (dword 12); each
// multiplies its baked constant. Read at first shader compile - clear the shader cache to apply.
static const std::unordered_map<u32, f32> kGrrEnvScales = [] {
    std::unordered_map<u32, f32> m;
    const auto read_f32 = [](const char* name) -> std::optional<f32> {
        if (const char* e = std::getenv(name)) {
            return static_cast<f32>(std::strtod(e, nullptr));
        }
        return std::nullopt;
    };
    if (const auto v = read_f32("GR2_EXPOSURE")) {
        m[0u] = *v; // exposure -> dword 0
    }
    if (const auto v = read_f32("GR2_BLOOM")) {
        m[8u] = *v; // bloom -> dwords 8/9/10 (per channel, scaled equally)
        m[9u] = *v;
        m[10u] = *v;
    }
    if (const auto v = read_f32("GR2_EMISSION")) {
        m[15u] = *v; // highlight/white-point lift -> dword 15
    }
    if (const auto v = read_f32("GR2_ROLLOFF")) {
        m[12u] = *v; // highlight rolloff -> dword 12 (compress hottest pixels: fountain, orbs, markers)
    }
    return m;
}();

// GR2FORK: per-shader brightness harness + full-screen warmth, GRR serials only (the composite
// knobs above cannot lift one object class). Class knobs: GR2_TV_BRIGHTNESS, GR2_NPC_BRIGHTNESS
// (Kat + NPCs), GR2_FOUNTAIN_BRIGHTNESS, GR2_3DUI_BRIGHTNESS, GR2_WARMTH ([-1,1]). Probes:
// GR2_FS_KILL, GR2_FS_PAINT / GR2_FS_PAINT_RGB, GR2_FS_GAIN, GR2_FS_RT, GR2_SUN=<hash>:<r>:<g>:<b>.
// Precedence: KILL > PAINT > SUN > warmth > GAIN > class; re-emit shaders (cache clear) to apply.

// Member shader hashes per object class. To re-assign a shader, move its hash between these lists.
static constexpr std::array<u64, 1> kGrrGrpTv = {0x78225d34ull};
static constexpr std::array<u64, 7> kGrrGrpNpc = {
    0xbf0ddf75ull, 0xc5f496c0ull, 0x4d450ae2ull, 0xb30c6b28ull, // body x3 + Kat's mouth
    0x91a764f1ull, 0xe9ba8491ull, 0x504b03e0ull,                // Kat gravity-shift (shift-state body)
};
static constexpr std::array<u64, 3> kGrrGrpFountain = {0xbd127633ull, 0xccff88ecull, 0xdcbe504eull};
static constexpr std::array<u64, 2> kGrrGrp3dUi = {0xf321ac6bull, 0x6f9fd325ull};

// Object-class brightness groups: name (for the log), env var, baked default multiplier, members.
// Baked defaults are the final tuned values: TV/NPC lift the dark objects; fountain water and the
// 3D UI markers are dimmed to 0.3 to rein in their blowout. Env vars still override per class.
struct Gr2BrightGroup {
    std::string_view name;
    const char* env;
    f32 baked;
    std::span<const u64> hashes;
};
static const std::array<Gr2BrightGroup, 4> kGrrBrightGroups = {{
    {"TV", "GR2_TV_BRIGHTNESS", 3.3f, kGrrGrpTv},
    {"NPC", "GR2_NPC_BRIGHTNESS", 1.65f, kGrrGrpNpc},
    {"FOUNTAIN", "GR2_FOUNTAIN_BRIGHTNESS", 1.0f, kGrrGrpFountain},
    {"3DUI", "GR2_3DUI_BRIGHTNESS", 1.0f, kGrrGrp3dUi},
}};

namespace {

std::vector<std::string> Gr2SplitTokens(std::string_view s) {
    static constexpr std::string_view kDelims = " \t\r\n,;";
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && kDelims.find(s[i]) != std::string_view::npos) {
            ++i;
        }
        size_t j = i;
        while (j < s.size() && kDelims.find(s[j]) == std::string_view::npos) {
            ++j;
        }
        if (j > i) {
            out.emplace_back(s.substr(i, j - i));
        }
        i = j;
    }
    return out;
}

// Hashes are always hex (the log prints them as 0x........); parse base-16 with or without "0x".
u64 Gr2ParseHashHex(std::string_view t) {
    if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) {
        t.remove_prefix(2);
    }
    return std::strtoull(std::string(t).c_str(), nullptr, 16);
}

struct Gr2FragHarness {
    bool active = false;                            // GRR serial matched
    u32 rt = 0;                                     // target render-target index
    f32 warmth = 0.00f;                             // full-screen tint on the composite, [-1, 1]
    f32 contrast = 1.015f;                            // GR2_CONTRAST: full-screen contrast about 0.5 on the composite
    f32 sky_warmth = 0.00f;                         // GR2_SKY_WARMTH: warm/cool tint on the skybox, [-1, 1]
    f32 sky_bright = 0.70f;                          // GR2_SKY_BRIGHTNESS: uniform brightness on the skybox
    f32 cloud_bright = 1.50f;                        // GR2_CLOUD_BRIGHTNESS: uniform brightness on the clouds
    f32 sun_ceil = 1.0f;                             // GR2_SUN_CEIL: soft highlight ceiling on sun-lit materials (0=off)
    std::array<f32, 3> npc_rgb{1.0f, 1.0f, 1.0f};   // GR2_NPC_RGB: per-channel tint on the Kat/NPC group
    f32 npc_warmth = 0.10f;                           // GR2_NPC_WARMTH: easy warm/cool knob on the Kat/NPC group, [-1,1]
    std::array<f32, 3> paint_rgb{1.0f, 0.0f, 1.0f}; // sentinel colour for PAINT (default magenta)
    std::vector<u64> paint;                         // hashes forced to the sentinel colour
    std::vector<u64> kill;                          // GR2_FS_KILL: hashes whose RGBA output is zeroed
    std::unordered_map<u64, f32> gain;              // resolved hash -> RGB multiplier
    std::unordered_map<u64, std::array<f32, 3>> rgb_gain; // GR2_SUN: hash -> per-channel RGB multiply
};

const Gr2FragHarness& Gr2Harness() {
    static const Gr2FragHarness cfg = [] {
        Gr2FragHarness h;
        // GRR gate -- query ElfInfo lazily (the serial is known by the first shader compile, not at
        // static-init time), exactly like the composite hooks.
        static constexpr std::array<std::string_view, 12> kGrrSerials = {
            "CUSA01112", "CUSA01113", "CUSA01113P", "CUSA01130",
            "CUSA02318", "CUSA00546", "CUSA02443",  "CUSA04246",
            "PCJS50004", "PCJS50008", "PCJS66015",  "PCJS66029",
        };
        const auto serial = Common::ElfInfo::Instance().GameSerial();
        h.active = std::find(kGrrSerials.begin(), kGrrSerials.end(), serial) != kGrrSerials.end();
        if (!h.active) {
            return h;
        }
        if (const char* e = std::getenv("GR2_FS_RT")) {
            h.rt = static_cast<u32>(std::strtoul(e, nullptr, 0));
        }
        if (const char* e = std::getenv("GR2_WARMTH")) {
            h.warmth = std::clamp(static_cast<f32>(std::strtod(e, nullptr)), -1.0f, 1.0f);
        }
        if (const char* e = std::getenv("GR2_CONTRAST")) {
            h.contrast = std::max(0.0f, static_cast<f32>(std::strtod(e, nullptr)));
        }
        if (const char* e = std::getenv("GR2_SKY_WARMTH")) {
            h.sky_warmth = std::clamp(static_cast<f32>(std::strtod(e, nullptr)), -1.0f, 1.0f);
        }
        if (const char* e = std::getenv("GR2_SKY_BRIGHTNESS")) {
            h.sky_bright = std::max(0.0f, static_cast<f32>(std::strtod(e, nullptr)));
        }
        if (const char* e = std::getenv("GR2_CLOUD_BRIGHTNESS")) {
            h.cloud_bright = std::max(0.0f, static_cast<f32>(std::strtod(e, nullptr)));
        }
        if (const char* e = std::getenv("GR2_SUN_CEIL")) {
            h.sun_ceil = std::max(0.0f, static_cast<f32>(std::strtod(e, nullptr)));
        }
        if (const char* e = std::getenv("GR2_FS_PAINT_RGB")) {
            const auto parts = Gr2SplitTokens(e);
            for (size_t i = 0; i < parts.size() && i < 3; ++i) {
                h.paint_rgb[i] = static_cast<f32>(std::strtod(parts[i].c_str(), nullptr));
            }
        }
        if (const char* e = std::getenv("GR2_FS_PAINT")) {
            for (const auto& t : Gr2SplitTokens(e)) {
                h.paint.push_back(Gr2ParseHashHex(t));
            }
        }
        // KILL / NOP probe: zero the whole output of each listed shader so its draw disappears
        // (the preferred identification probe -- see header). Additive/alpha-blended overlays vanish.
        if (const char* e = std::getenv("GR2_FS_KILL")) {
            for (const auto& t : Gr2SplitTokens(e)) {
                h.kill.push_back(Gr2ParseHashHex(t));
            }
        }
        // Object-class brightness: each member hash gets its class's gain (env overrides baked).
        LOG_INFO(Render_Recompiler, "[GR2FORK] FS colour harness active (GRR {}): rt={} warmth {:+.2f}",
                 serial, h.rt, h.warmth);
        for (const auto& g : kGrrBrightGroups) {
            f32 mul = g.baked;
            bool from_env = false;
            if (const char* e = std::getenv(g.env)) {
                mul = static_cast<f32>(std::strtod(e, nullptr));
                from_env = true;
            }
            for (const u64 hsh : g.hashes) {
                h.gain[hsh] = mul;
            }
            LOG_INFO(Render_Recompiler, "[GR2FORK]   {} brightness x{} ({}, {} shaders)", g.name,
                     mul, from_env ? "env" : "baked", g.hashes.size());
        }
        // Per-hash override (live testing / a hash not yet in a class). Wins over the class value.
        if (const char* e = std::getenv("GR2_FS_GAIN")) {
            for (const auto& t : Gr2SplitTokens(e)) {
                const auto colon = t.find(':');
                if (colon == std::string::npos) {
                    continue;
                }
                const u64 hsh = Gr2ParseHashHex(std::string_view{t}.substr(0, colon));
                h.gain[hsh] = static_cast<f32>(std::strtod(t.c_str() + colon + 1, nullptr));
                LOG_INFO(Render_Recompiler, "[GR2FORK]   per-hash override fs {:#x} -> x{}", hsh,
                         h.gain[hsh]);
            }
        }
        // GR2FORK: sun/sky per-channel enhancement. Entries are colon-separated
        // (<hash>:<r>:<g>:<b>, or <hash>:<s> scalar shorthand) so they share the comma/space
        // splitter with the other lists; a per-channel multiply shifts hue as well as level.
        if (const char* e = std::getenv("GR2_SUN")) {
            for (const auto& t : Gr2SplitTokens(e)) {
                const auto c0 = t.find(':');
                if (c0 == std::string::npos) {
                    continue;
                }
                const u64 hsh = Gr2ParseHashHex(std::string_view{t}.substr(0, c0));
                std::string_view rest{t};
                rest.remove_prefix(c0 + 1);
                std::array<f32, 3> parsed{};
                size_t n = 0;
                while (n < 3 && !rest.empty()) {
                    const auto c = rest.find(':');
                    parsed[n++] =
                        static_cast<f32>(std::strtod(std::string(rest.substr(0, c)).c_str(), nullptr));
                    if (c == std::string_view::npos) {
                        break;
                    }
                    rest.remove_prefix(c + 1);
                }
                std::array<f32, 3> rgb;
                if (n == 1) {
                    rgb = {parsed[0], parsed[0], parsed[0]};
                } else if (n >= 3) {
                    rgb = {parsed[0], parsed[1], parsed[2]};
                } else {
                    LOG_WARNING(Render_Recompiler,
                                "[GR2FORK]   GR2_SUN fs {:#x}: need 1 or 3 channels, got {} -- ignored",
                                hsh, n);
                    continue;
                }
                h.rgb_gain[hsh] = rgb;
                LOG_INFO(Render_Recompiler, "[GR2FORK]   SUN fs {:#x} -> rgb x({},{},{})", hsh, rgb[0],
                         rgb[1], rgb[2]);
            }
        }
        // GR2FORK: GR2_NPC_RGB="r,g,b": per-channel tint on the Kat/NPC group, applied in its own
        // Gr2ApplyFragOutput block so it composes with GR2_NPC_BRIGHTNESS and the highlight knee;
        // characters carry no sun-colour constant, so this output tint is how they are recoloured.
        if (const char* e = std::getenv("GR2_NPC_RGB")) {
            const auto parts = Gr2SplitTokens(e);
            for (size_t i = 0; i < parts.size() && i < 3; ++i) {
                h.npc_rgb[i] = static_cast<f32>(std::strtod(parts[i].c_str(), nullptr));
            }
            LOG_INFO(Render_Recompiler, "[GR2FORK]   NPC_RGB -> rgb x({},{},{})", h.npc_rgb[0],
                     h.npc_rgb[1], h.npc_rgb[2]);
        }
        // GR2FORK: GR2_NPC_WARMTH: the easy single knob for Kat/NPC colour temperature, [-1,1].
        // + = warmer (red up, blue down), - = cooler (red down, blue up), green pivots. Same maths as
        // the sky/composite warmth, applied in the same NPC tint block (multiplies with GR2_NPC_RGB).
        if (const char* e = std::getenv("GR2_NPC_WARMTH")) {
            h.npc_warmth = std::clamp(static_cast<f32>(std::strtod(e, nullptr)), -1.0f, 1.0f);
            LOG_INFO(Render_Recompiler, "[GR2FORK]   NPC_WARMTH -> {}", h.npc_warmth);
        }
        // Drop pure no-ops (1:1:1) so nothing is emitted for them.
        std::erase_if(h.rgb_gain, [](const auto& kv) {
            return kv.second[0] == 1.0f && kv.second[1] == 1.0f && kv.second[2] == 1.0f;
        });
        // A resolved gain of exactly 1.0 is a no-op; drop it so no multiply is emitted.
        std::erase_if(h.gain, [](const auto& kv) { return kv.second == 1.0f; });
        for (const u64 hsh : h.paint) {
            LOG_INFO(Render_Recompiler, "[GR2FORK]   PAINT fs {:#x} -> ({},{},{})", hsh,
                     h.paint_rgb[0], h.paint_rgb[1], h.paint_rgb[2]);
        }
        return h;
    }();
    return cfg;
}

// Applies the KILL / PAINT / warmth / brightness harness to one channel of one MRT write. Returns
// `value` unchanged unless this exact fragment shader + render target is targeted. PAINT/GAIN/warmth
// touch RGB float channels only; KILL zeroes RGBA so a targeted draw disappears entirely.
Id Gr2ApplyFragOutput(EmitContext& ctx, u32 rt_index, bool is_integer, Id value, u32 element) {
    const Gr2FragHarness& h = Gr2Harness();
    if (!h.active || is_integer || rt_index != h.rt) {
        return value;
    }
    const u64 hash = ctx.info.pgm_hash;
    // KILL probe: zero the entire RGBA output so the shader's draw contributes nothing (additive
    // passes add 0, alpha-blended get src.a = 0, opaque render black). Applies to alpha, so it
    // sits ahead of the RGB-only guard below and wins over PAINT/GAIN/warmth on the same shader.
    for (const u64 k : h.kill) {
        if (k == hash) {
            return ctx.ConstF32(0.0f);
        }
    }
    if (element > 2) {
        return value; // PAINT / GR2_SUN / warmth / gain touch RGB only; alpha passes through.
    }
    // PAINT is the identification override and wins over everything on the same shader.
    for (const u64 p : h.paint) {
        if (p == hash) {
            return ctx.ConstF32(h.paint_rgb[element]);
        }
    }
    // GR2FORK: Sun/sky per-channel enhancement (GR2_SUN). Authoritative for the targeted shader so
    // an identified sun hash is unaffected by any object-class membership; per channel, so it lifts
    // brightness/intensity and shifts colour together. element is already guaranteed 0..2 above.
    if (const auto it = h.rgb_gain.find(hash); it != h.rgb_gain.end()) {
        const f32 g = it->second[element];
        return g == 1.0f ? value : ctx.OpFMul(ctx.F32[1], value, ctx.ConstF32(g));
    }
    // Full-screen grade on the composite's post-tonemap output: contrast about mid-grey 0.5 first
    // (GR2_CONTRAST, default 1.0), then the warm/cool tint (GR2_WARMTH). The contrast result is
    // clamped to [0,1] since this is display-referred output; both are no-ops at their defaults.
    if (hash == kGrrCompositePgmHash && (h.contrast != 1.0f || h.warmth != 0.0f)) {
        Id v = value;
        if (h.contrast != 1.0f) {
            // v = clamp((v - 0.5) * contrast + 0.5, 0, 1)
            v = ctx.OpFAdd(ctx.F32[1],
                           ctx.OpFMul(ctx.F32[1], ctx.OpFSub(ctx.F32[1], v, ctx.ConstF32(0.5f)),
                                      ctx.ConstF32(h.contrast)),
                           ctx.ConstF32(0.5f));
            v = ctx.OpFClamp(ctx.F32[1], v, ctx.ConstF32(0.0f), ctx.ConstF32(1.0f));
        }
        if (h.warmth != 0.0f) {
            if (element == 0) {
                v = ctx.OpFMul(ctx.F32[1], v, ctx.ConstF32(1.0f + h.warmth)); // red up = warmer
            } else if (element == 2) {
                v = ctx.OpFMul(ctx.F32[1], v, ctx.ConstF32(1.0f - h.warmth)); // blue down = warmer
            }
        }
        return v;
    }
    // Skybox brightness + warmth on the sky/sun shader only (GR2_SKY_BRIGHTNESS default 1.0,
    // GR2_SKY_WARMTH default 0; they compose). Both act pre-tonemap, so a brighten/warm push on
    // an already-bright sky can clamp in the composite; GR2_SUN=<skybox>:r:g:b wins over both.
    if (hash == kGrrSkyboxPgmHash && (h.sky_bright != 1.0f || h.sky_warmth != 0.0f)) {
        f32 m = h.sky_bright;
        if (element == 0) {
            m *= 1.0f + h.sky_warmth; // red up = warmer
        } else if (element == 2) {
            m *= 1.0f - h.sky_warmth; // blue down = warmer
        }
        return m == 1.0f ? value : ctx.OpFMul(ctx.F32[1], value, ctx.ConstF32(m));
    }
    // Cloud brightness (GR2_CLOUD_BRIGHTNESS, default 1.0): uniform multiply on the cloud
    // shader's RGB; alpha is untouched (RGB-only guard above) so the blend shape is preserved.
    // Same pre-tonemap caveat as the sky.
    if (h.cloud_bright != 1.0f && hash == kGrrCloudPgmHash) {
        return ctx.OpFMul(ctx.F32[1], value, ctx.ConstF32(h.cloud_bright));
    }
    // GR2FORK: Kat/NPC tint, its own step composing with class brightness below and the knee
    // after. GR2_NPC_WARMTH is the single knob ([-1,1]; + warmer, - cooler, green pivot);
    // GR2_NPC_RGB is explicit per-channel. They multiply together; element is 0..2 here.
    if ((h.npc_rgb[0] != 1.0f || h.npc_rgb[1] != 1.0f || h.npc_rgb[2] != 1.0f || h.npc_warmth != 0.0f) &&
        std::find(kGrrGrpNpc.begin(), kGrrGrpNpc.end(), hash) != kGrrGrpNpc.end()) {
        f32 m = h.npc_rgb[element];
        if (element == 0) {
            m *= 1.0f + h.npc_warmth; // warmer = red up
        } else if (element == 2) {
            m *= 1.0f - h.npc_warmth; // warmer = blue down
        }
        if (m != 1.0f) {
            value = ctx.OpFMul(ctx.F32[1], value, ctx.ConstF32(m));
        }
    }
    // Object-class brightness (or per-hash override) -- single resolved multiplier per shader.
    if (const auto it = h.gain.find(hash); it != h.gain.end()) {
        value = ctx.OpFMul(ctx.F32[1], value, ctx.ConstF32(it->second));
    }
    // GR2FORK: highlight knee on the sun-lit shaders - soft-ceils final RGB so brightness lifts
    // asymptote instead of washing out to white and feeding bloom bleed; identity below the knee
    // keeps shadows/mid-tones untouched. GR2_SUN_CEIL=<c> sets the ceiling (0/unset = off);
    // e = max(x-k,0); x = min(x,k) + e/(1 + e/R), with k=0.7c, R=0.3c (asymptote c).
    if (h.sun_ceil > 0.0f &&
        (std::find(kGrrSunCharShaders.begin(), kGrrSunCharShaders.end(), hash) !=
             kGrrSunCharShaders.end() ||
         std::find(kGrrSunWorldShaders.begin(), kGrrSunWorldShaders.end(), hash) !=
             kGrrSunWorldShaders.end())) {
        const f32 k = 0.7f * h.sun_ceil;
        const f32 R = 0.3f * h.sun_ceil; // asymptote = k + R = sun_ceil
        const Id kc = ctx.ConstF32(k);
        const Id excess =
            ctx.OpFMax(ctx.F32[1], ctx.OpFSub(ctx.F32[1], value, kc), ctx.ConstF32(0.0f));
        const Id denom = ctx.OpFAdd(ctx.F32[1], ctx.ConstF32(1.0f),
                                    ctx.OpFMul(ctx.F32[1], excess, ctx.ConstF32(1.0f / R)));
        value = ctx.OpFAdd(ctx.F32[1], ctx.OpFMin(ctx.F32[1], value, kc),
                           ctx.OpFDiv(ctx.F32[1], excess, denom));
    }
    return value;
}

} // namespace

static std::pair<Id, bool> OutputAttrComponentType(EmitContext& ctx, IR::Attribute attr) {
    if (IR::IsParam(attr)) {
        const u32 index{u32(attr) - u32(IR::Attribute::Param0)};
        const auto& info{ctx.output_params.at(index)};
        return {info.component_type, info.is_integer};
    }
    if (IR::IsMrt(attr)) {
        const u32 index{u32(attr) - u32(IR::Attribute::RenderTarget0)};
        const auto& info{ctx.frag_outputs.at(index)};
        return {info.component_type, info.is_integer};
    }
    switch (attr) {
    case IR::Attribute::Position0:
    case IR::Attribute::ClipDistance:
    case IR::Attribute::CullDistance:
    case IR::Attribute::Depth:
    case IR::Attribute::PointSize:
        return {ctx.F32[1], false};
    case IR::Attribute::RenderTargetIndex:
    case IR::Attribute::ViewportIndex:
    case IR::Attribute::SampleMask:
    case IR::Attribute::StencilRef:
        return {ctx.U32[1], true};
    default:
        UNREACHABLE_MSG("Write attribute {}", attr);
    }
}

Id EmitGetUserData(EmitContext& ctx, IR::ScalarReg reg) {
    const u32 index = ctx.binding.user_data + ctx.info.ud_mask.Index(reg);
    const u32 half = PushData::UdRegsIndex + (index >> 2);
    const Id ud_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[1]),
                                      ctx.push_data_block, ctx.ConstU32(half),
                                      ctx.ConstU32(index & 3))};
    const Id ud_reg{ctx.OpLoad(ctx.U32[1], ud_ptr)};
    ctx.Name(ud_reg, fmt::format("ud_{}", u32(reg)));
    return ud_reg;
}

Id EmitReadConst(EmitContext& ctx, IR::Inst* inst, Id addr, Id offset) {
    const u32 flatbuf_off_dw = inst->Flags<u32>();
    if (!Config::directMemoryAccess()) {
        return ctx.EmitFlatbufferLoad(ctx.ConstU32(flatbuf_off_dw));
    }
    // We can only provide a fallback for immediate offsets.
    if (flatbuf_off_dw == 0) {
        return ctx.OpFunctionCall(ctx.U32[1], ctx.read_const_dynamic, addr, offset);
    } else {
        return ctx.OpFunctionCall(ctx.U32[1], ctx.read_const, addr, offset,
                                  ctx.ConstU32(flatbuf_off_dw));
    }
}

Id EmitReadConstBuffer(EmitContext& ctx, u32 handle, Id index) {
    // GR2FORK: GRR composite (fs 0x9a4b146c) CB#0 float loads: S_BUFFER_LOAD_DWORD constant-folds
    // the index to ConstU32(k), so exact-constant index + program hash identifies a parameter load
    // (no match = untouched). GRR-only; the local static queries ElfInfo at first shader compile.
    static const bool kIsGravityRushRemastered = [] {
        static constexpr std::array<std::string_view, 12> kGrrSerials = {
            "CUSA01112", "CUSA01113", "CUSA01113P", "CUSA01130",
            "CUSA02318", "CUSA00546", "CUSA02443",  "CUSA04246",
            "PCJS50004", "PCJS50008", "PCJS66015",  "PCJS66029",
        };
        const auto serial = Common::ElfInfo::Instance().GameSerial();
        return std::find(kGrrSerials.begin(), kGrrSerials.end(), serial) != kGrrSerials.end();
    }();
    const bool is_composite_cb0 =
        kIsGravityRushRemastered && ctx.info.pgm_hash == kGrrCompositePgmHash && handle == 0;

    // GR2FORK: GR2_SUN_INTENSITY=<f> scales the sun strength constants on both rigs (scalar
    // CB2[27] characters, triplet CB2[16/17/18] world); GR2_SUN_RGB="r,g,b" additionally tints
    // the world sun. Only the clamped directional term is touched - ambient/shadow unaffected.
    static const f32 kGrrSunIntensity = [] {
        if (const char* e = std::getenv("GR2_SUN_INTENSITY")) {
            return static_cast<f32>(std::strtod(e, nullptr));
        }
        return 1.0f;
    }();
    static const std::array<f32, 3> kGrrSunRgb = [] {
        std::array<f32, 3> rgb{1.0f, 1.0f, 1.0f};
        if (const char* e = std::getenv("GR2_SUN_RGB")) {
            const auto parts = Gr2SplitTokens(e);
            for (size_t i = 0; i < parts.size() && i < 3; ++i) {
                rgb[i] = static_cast<f32>(std::strtod(parts[i].c_str(), nullptr));
            }
        }
        return rgb;
    }();

    // Decide whether this dword load should be scaled, and by how much. The permanent table
    // (exposure / bloom / highlight knobs) applies first; an env entry for the same dword overrides
    // it for live tuning. Entries of 1.0 are skipped so they emit nothing.
    f32 scale = 1.0f;
    bool do_scale = false;
    if (is_composite_cb0) {
        for (const auto& [d, f] : kGrrPermanentScales) {
            if (f != 1.0f && index.value == ctx.ConstU32(d).value) {
                scale = f;
                do_scale = true;
                break;
            }
        }
        for (const auto& [d, f] : kGrrEnvScales) {
            if (index.value == ctx.ConstU32(d).value) {
                scale = f; // env override wins (live tuning)
                do_scale = true;
                break;
            }
        }
    }
    // Sun key-light, pinned per shader by hash so we only touch real sun constants. Character shaders
    // -> CB2[27] scalar (intensity); world shaders -> CB2[16/17/18] colour (intensity x per-channel
    // tint). Sun direction CB2[20..22] is deliberately never scaled.
    if (!do_scale && kIsGravityRushRemastered && handle == 2) {
        const u64 sun_hash = ctx.info.pgm_hash;
        const bool is_char = std::find(kGrrSunCharShaders.begin(), kGrrSunCharShaders.end(),
                                       sun_hash) != kGrrSunCharShaders.end();
        const bool is_world = std::find(kGrrSunWorldShaders.begin(), kGrrSunWorldShaders.end(),
                                        sun_hash) != kGrrSunWorldShaders.end();
        f32 sun = 1.0f;
        if (is_char && index.value == ctx.ConstU32(27u).value) {
            sun = kGrrSunIntensity;
        } else if (is_world && index.value == ctx.ConstU32(16u).value) {
            sun = kGrrSunIntensity * kGrrSunRgb[0];
        } else if (is_world && index.value == ctx.ConstU32(17u).value) {
            sun = kGrrSunIntensity * kGrrSunRgb[1];
        } else if (is_world && index.value == ctx.ConstU32(18u).value) {
            sun = kGrrSunIntensity * kGrrSunRgb[2];
        }
        if (sun != 1.0f) {
            scale = sun;
            do_scale = true;
        }
    }

    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        index = ctx.OpIAdd(ctx.U32[1], index, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, index)};
    Id result{ctx.OpLoad(ctx.U32[1], ptr)};

    if (do_scale) {
        // The constant is a raw float stored as a dword; bitcast, scale, bitcast back.
        const Id as_f32 = ctx.OpBitcast(ctx.F32[1], result);
        const Id scaled = ctx.OpFMul(ctx.F32[1], as_f32, ctx.ConstF32(scale));
        result = ctx.OpBitcast(ctx.U32[1], scaled);
    }

    if (const Id size = buffer.Size(PointerSize::B32); Sirit::ValidId(size)) {
        const Id in_bounds = ctx.OpULessThan(ctx.U1[1], index, size);
        return ctx.OpSelect(ctx.U32[1], in_bounds, result, ctx.u32_zero_value);
    }
    return result;
}

Id EmitGetAttribute(EmitContext& ctx, IR::Attribute attr, u32 comp, u32 index) {
    if (IR::IsParam(attr)) {
        const u32 param_index{u32(attr) - u32(IR::Attribute::Param0)};
        const auto& param{ctx.input_params.at(param_index)};
        const Id value = [&] {
            if (param.is_array) {
                ASSERT(param.num_components > 1);
                if (param.is_loaded) {
                    return ctx.OpCompositeExtract(param.component_type, param.id_array[index],
                                                  comp);
                } else {
                    return ctx.OpLoad(param.component_type,
                                      ctx.OpAccessChain(param.pointer_type, param.id,
                                                        ctx.ConstU32(index), ctx.ConstU32(comp)));
                }
            } else {
                ASSERT(!param.is_loaded);
                if (param.num_components > 1) {
                    return ctx.OpLoad(
                        param.component_type,
                        ctx.OpAccessChain(param.pointer_type, param.id, ctx.ConstU32(comp)));
                } else {
                    return ctx.OpLoad(param.component_type, param.id);
                }
            }
        }();
        return param.is_integer ? ctx.OpBitcast(ctx.F32[1], value) : value;
    }
    if (IR::IsBarycentricCoord(attr) && ctx.profile.supports_fragment_shader_barycentric) {
        ++comp;
    }
    switch (attr) {
    case IR::Attribute::Position0:
        ASSERT(ctx.l_stage == LogicalStage::Geometry);
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.gl_in, ctx.ConstU32(index),
                                            ctx.ConstU32(0U), ctx.ConstU32(comp)));
    case IR::Attribute::FragCoord:
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.frag_coord, ctx.ConstU32(comp)));
    case IR::Attribute::TessellationEvaluationPointU:
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.tess_coord, ctx.u32_zero_value));
    case IR::Attribute::TessellationEvaluationPointV:
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.tess_coord, ctx.ConstU32(1U)));
    case IR::Attribute::BaryCoordSmooth:
        return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_smooth,
                                                        ctx.ConstU32(comp)));
    case IR::Attribute::BaryCoordSmoothCentroid:
        return ctx.OpLoad(
            ctx.F32[1],
            ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_smooth_centroid, ctx.ConstU32(comp)));
    case IR::Attribute::BaryCoordSmoothSample:
        return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_smooth_sample,
                                                        ctx.ConstU32(comp)));
    case IR::Attribute::BaryCoordNoPersp:
        return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_nopersp,
                                                        ctx.ConstU32(comp)));
    default:
        UNREACHABLE_MSG("Read attribute {}", attr);
    }
}

Id EmitGetAttributeU32(EmitContext& ctx, IR::Attribute attr, u32 comp) {
    switch (attr) {
    case IR::Attribute::VertexId:
        return ctx.OpLoad(ctx.U32[1], ctx.vertex_index);
    case IR::Attribute::InstanceId:
        return ctx.OpLoad(ctx.U32[1], ctx.instance_id);
    case IR::Attribute::WorkgroupIndex:
        return ctx.workgroup_index_id;
    case IR::Attribute::WorkgroupId:
        return ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.workgroup_id), comp);
    case IR::Attribute::LocalInvocationId:
        return ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.local_invocation_id),
                                      comp);
    case IR::Attribute::IsFrontFace:
        return ctx.OpSelect(ctx.U32[1], ctx.OpLoad(ctx.U1[1], ctx.front_facing), ctx.u32_one_value,
                            ctx.u32_zero_value);
    case IR::Attribute::SampleIndex:
        return ctx.OpLoad(ctx.U32[1], ctx.sample_index);
    case IR::Attribute::RenderTargetIndex:
        return ctx.OpLoad(ctx.U32[1], ctx.output_layer);
    case IR::Attribute::PrimitiveId:
        return ctx.OpLoad(ctx.U32[1], ctx.primitive_id);
    case IR::Attribute::InvocationId:
        ASSERT(ctx.info.l_stage == LogicalStage::Geometry ||
               ctx.info.l_stage == LogicalStage::TessellationControl);
        return ctx.OpLoad(ctx.U32[1], ctx.invocation_id);
    case IR::Attribute::PatchVertices:
        ASSERT(ctx.info.l_stage == LogicalStage::TessellationControl);
        return ctx.OpLoad(ctx.U32[1], ctx.patch_vertices);
    case IR::Attribute::PackedHullInvocationInfo: {
        ASSERT(ctx.info.l_stage == LogicalStage::TessellationControl);
        // [0:8]: patch id within VGT
        // [8:12]: output control point id
        // But 0:8 should be treated as 0 for attribute addressing purposes
        if (ctx.runtime_info.hs_info.IsPassthrough()) {
            // Gcn shader would run with 1 thread, but we need to run a thread for
            // each output control point.
            // If Gcn shader uses this value, we should make sure all threads in the
            // Vulkan shader use 0
            return ctx.ConstU32(0u);
        } else {
            const Id invocation_id = ctx.OpLoad(ctx.U32[1], ctx.invocation_id);
            return ctx.OpShiftLeftLogical(ctx.U32[1], invocation_id, ctx.ConstU32(8u));
        }
    }
    default:
        UNREACHABLE_MSG("Read U32 attribute {}", attr);
    }
}

void EmitSetAttribute(EmitContext& ctx, IR::Attribute attr, Id value, u32 element) {
    const auto op_store = [&](Id pointer) {
        const auto [component_type, is_integer] = OutputAttrComponentType(ctx, attr);
        if (is_integer) {
            ctx.OpStore(pointer, ctx.OpBitcast(component_type, value));
        } else {
            ctx.OpStore(pointer, value);
        }
    };
    if (IR::IsParam(attr)) {
        const u32 attr_index{u32(attr) - u32(IR::Attribute::Param0)};
        if (ctx.stage == Stage::Local) {
            const auto component_ptr = ctx.TypePointer(spv::StorageClass::Output, ctx.F32[1]);
            return op_store(ctx.OpAccessChain(component_ptr, ctx.output_attr_array,
                                              ctx.ConstU32(attr_index), ctx.ConstU32(element)));
        } else {
            const auto& info{ctx.output_params.at(attr_index)};
            ASSERT(info.num_components > 0);
            if (info.num_components == 1) {
                return op_store(info.id);
            } else {
                return op_store(
                    ctx.OpAccessChain(info.pointer_type, info.id, ctx.ConstU32(element)));
            }
        }
    }
    if (IR::IsMrt(attr)) {
        const u32 index{u32(attr) - u32(IR::Attribute::RenderTarget0)};
        const auto& info{ctx.frag_outputs.at(index)};
        // GR2FORK: NPC/TV identification + brightness harness. No-op unless GR2_FS_KILL / GR2_FS_PAINT
        // / GR2_FS_GAIN is set for this shader (GRR only). KILL zeroes RGBA; PAINT/GAIN touch RGB.
        value = Gr2ApplyFragOutput(ctx, index, info.is_integer, value, element);
        if (info.num_components == 1) {
            return op_store(info.id);
        } else {
            return op_store(ctx.OpAccessChain(info.pointer_type, info.id, ctx.ConstU32(element)));
        }
    }
    switch (attr) {
    case IR::Attribute::Position0:
        return op_store(
            ctx.OpAccessChain(ctx.output_f32, ctx.output_position, ctx.ConstU32(element)));
    case IR::Attribute::ClipDistance:
        return op_store(
            ctx.OpAccessChain(ctx.output_f32, ctx.clip_distances, ctx.ConstU32(element)));
    case IR::Attribute::CullDistance:
        return op_store(
            ctx.OpAccessChain(ctx.output_f32, ctx.cull_distances, ctx.ConstU32(element)));
    case IR::Attribute::PointSize:
        return op_store(ctx.output_point_size);
    case IR::Attribute::RenderTargetIndex:
        return op_store(ctx.output_layer);
    case IR::Attribute::ViewportIndex:
        return op_store(ctx.output_viewport_index);
    case IR::Attribute::Depth:
        return op_store(ctx.frag_depth);
    case IR::Attribute::SampleMask:
        return op_store(ctx.OpAccessChain(ctx.output_u32, ctx.sample_mask, ctx.u32_zero_value));
    default:
        UNREACHABLE_MSG("Write attribute {}", attr);
    }
}

Id EmitGetTessGenericAttribute(EmitContext& ctx, Id vertex_index, Id attr_index, Id comp_index) {
    const auto attr_comp_ptr = ctx.TypePointer(spv::StorageClass::Input, ctx.F32[1]);
    return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(attr_comp_ptr, ctx.input_attr_array,
                                                    vertex_index, attr_index, comp_index));
}

Id EmitReadTcsGenericOuputAttribute(EmitContext& ctx, Id vertex_index, Id attr_index,
                                    Id comp_index) {
    const auto attr_comp_ptr = ctx.TypePointer(spv::StorageClass::Output, ctx.F32[1]);
    return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(attr_comp_ptr, ctx.output_attr_array,
                                                    vertex_index, attr_index, comp_index));
}

void EmitSetTcsGenericAttribute(EmitContext& ctx, Id value, Id attr_index, Id comp_index) {
    // Implied vertex index is invocation_id
    const auto component_ptr = ctx.TypePointer(spv::StorageClass::Output, ctx.F32[1]);
    Id pointer =
        ctx.OpAccessChain(component_ptr, ctx.output_attr_array,
                          ctx.OpLoad(ctx.U32[1], ctx.invocation_id), attr_index, comp_index);
    ctx.OpStore(pointer, value);
}

Id EmitGetPatch(EmitContext& ctx, IR::Patch patch) {
    const u32 index{IR::GenericPatchIndex(patch)};
    const Id element{ctx.ConstU32(IR::GenericPatchElement(patch))};
    const Id type{ctx.l_stage == LogicalStage::TessellationControl ? ctx.output_f32
                                                                   : ctx.input_f32};
    const Id pointer{ctx.OpAccessChain(type, ctx.patches.at(index), element)};
    return ctx.OpLoad(ctx.F32[1], pointer);
}

void EmitSetPatch(EmitContext& ctx, IR::Patch patch, Id value) {
    const Id pointer{[&] {
        if (IR::IsGeneric(patch)) {
            const u32 index{IR::GenericPatchIndex(patch)};
            const Id element{ctx.ConstU32(IR::GenericPatchElement(patch))};
            return ctx.OpAccessChain(ctx.output_f32, ctx.patches.at(index), element);
        }
        switch (patch) {
        case IR::Patch::TessellationLodLeft:
        case IR::Patch::TessellationLodRight:
        case IR::Patch::TessellationLodTop:
        case IR::Patch::TessellationLodBottom: {
            const u32 index{static_cast<u32>(patch) - u32(IR::Patch::TessellationLodLeft)};
            const Id index_id{ctx.ConstU32(index)};
            return ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_outer, index_id);
        }
        case IR::Patch::TessellationLodInteriorU:
            return ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_inner,
                                     ctx.u32_zero_value);
        case IR::Patch::TessellationLodInteriorV:
            return ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_inner, ctx.ConstU32(1u));
        default:
            UNREACHABLE_MSG("Patch {}", u32(patch));
        }
    }()};
    ctx.OpStore(pointer, value);
}

template <u32 N, PointerType alias>
static Id EmitLoadBufferB32xN(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    constexpr bool is_float = alias == PointerType::F32;
    const auto flags = inst->Flags<IR::BufferInstInfo>();
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto& data_types = alias == PointerType::U32 ? ctx.U32 : ctx.F32;
    const auto [id, pointer_type] = spv_buffer.Alias(alias);

    boost::container::static_vector<Id, N> ids;
    for (u32 i = 0; i < N; i++) {
        const Id index_i = i == 0 ? address : ctx.OpIAdd(ctx.U32[1], address, ctx.ConstU32(i));
        const Id ptr_i = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, index_i);
        const Id result_i = ctx.OpLoad(data_types[1], ptr_i);
        if (!flags.typed) {
            // Untyped loads have bounds checking per-component.
            ids.push_back(LoadAccessBoundsCheck<32, 1, is_float>(
                ctx, index_i, spv_buffer.Size(PointerSize::B32), result_i));
        } else {
            ids.push_back(result_i);
        }
    }

    const Id result = N == 1 ? ids[0] : ctx.OpCompositeConstruct(data_types[N], ids);
    if (flags.typed) {
        // Typed loads have single bounds check for the whole load.
        return LoadAccessBoundsCheck<32, N, is_float>(ctx, address,
                                                      spv_buffer.Size(PointerSize::B32), result);
    }
    return result;
}

Id EmitLoadBufferU8(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B8); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U8);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address)};
    const Id result{ctx.OpLoad(ctx.U8, ptr)};
    return LoadAccessBoundsCheck<8>(ctx, address, spv_buffer.Size(PointerSize::B8), result);
}

Id EmitLoadBufferU16(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B16); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U16);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address)};
    const Id result{ctx.OpLoad(ctx.U16, ptr)};
    return LoadAccessBoundsCheck<16>(ctx, address, spv_buffer.Size(PointerSize::B16), result);
}

Id EmitLoadBufferU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<1, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<2, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<3, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<4, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B64); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U64);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u64_zero_value, address)};
    const Id result{ctx.OpLoad(ctx.U64, ptr)};
    return LoadAccessBoundsCheck<64>(ctx, address, spv_buffer.Size(PointerSize::B64), result);
}

Id EmitLoadBufferF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<1, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferF32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<2, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferF32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<3, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferF32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<4, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferFormatF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    UNREACHABLE_MSG("SPIR-V instruction");
}

template <u32 N, PointerType alias>
static void EmitStoreBufferB32xN(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address,
                                 Id value) {
    constexpr bool is_float = alias == PointerType::F32;
    const auto flags = inst->Flags<IR::BufferInstInfo>();
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto& data_types = alias == PointerType::U32 ? ctx.U32 : ctx.F32;
    const auto [id, pointer_type] = spv_buffer.Alias(alias);

    auto store = [&] {
        for (u32 i = 0; i < N; i++) {
            const Id index_i = i == 0 ? address : ctx.OpIAdd(ctx.U32[1], address, ctx.ConstU32(i));
            const Id ptr_i = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, index_i);
            const Id value_i = N == 1 ? value : ctx.OpCompositeExtract(data_types[1], value, i);
            auto store_i = [&] {
                ctx.OpStore(ptr_i, value_i);
                return Id{};
            };
            if (!flags.typed) {
                // Untyped stores have bounds checking per-component.
                AccessBoundsCheck<32, 1, is_float>(ctx, index_i, spv_buffer.Size(PointerSize::B32),
                                                   store_i);
            } else {
                store_i();
            }
        }
        return Id{};
    };

    if (flags.typed) {
        // Typed stores have single bounds check for the whole store.
        AccessBoundsCheck<32, N, is_float>(ctx, address, spv_buffer.Size(PointerSize::B32), store);
    } else {
        store();
    }
}

void EmitStoreBufferU8(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B8); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U8);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address)};
    AccessBoundsCheck<8>(ctx, address, spv_buffer.Size(PointerSize::B8), [&] {
        ctx.OpStore(ptr, value);
        return Id{};
    });
}

void EmitStoreBufferU16(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B16); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U16);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address)};
    AccessBoundsCheck<16>(ctx, address, spv_buffer.Size(PointerSize::B16), [&] {
        ctx.OpStore(ptr, value);
        return Id{};
    });
}

void EmitStoreBufferU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<1, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<2, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<3, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<4, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU64(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    const auto& spv_buffer = ctx.buffers[handle];
    if (const Id offset = spv_buffer.Offset(PointerSize::B64); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    const auto [id, pointer_type] = spv_buffer.Alias(PointerType::U64);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u64_zero_value, address)};
    AccessBoundsCheck<64>(ctx, address, spv_buffer.Size(PointerSize::B64), [&] {
        ctx.OpStore(ptr, value);
        return Id{};
    });
}

void EmitStoreBufferF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<1, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferF32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<2, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferF32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<3, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferF32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<4, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferFormatF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    UNREACHABLE_MSG("SPIR-V instruction");
}

void EmitGetThreadBitScalarReg(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetThreadBitScalarReg(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetScalarRegister(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetScalarRegister(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetVectorRegister(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetVectorRegister(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetGotoVariable(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetGotoVariable(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetMaskLaneVariable(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetMaskLaneVariable(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

} // namespace Shader::Backend::SPIRV
