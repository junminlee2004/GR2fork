// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <span>

#include "common/config.h"
#include "common/logging/log.h"
#include "core/libraries/aspect_patches/aspect_patches.h"

namespace Libraries::AspectPatches {

namespace {

struct AspectSite {
    std::uint32_t va_offset;  // VA offset from eboot base
    const char*   group;      // for logging
};

// GR2: 45 eboot sites of the float constant 1.7777778f (0x3FE38E39), verified byte-exact against
// the cross-region 1.11 layout. Group A (5): MOV-imm32 struct-init writes; B (28): vec4 splats for
// SIMD aspect math; C (12): rip-relative singletons feeding the projection builder at 0x010cf0e0.
constexpr std::array<AspectSite, 45> kGr2AspectSites = {{
    // Group A - struct-init MOV-imm32 (5)
    {0x0045b42b, "A.singleton"},
    {0x00465cc3, "A.struct[+0x12c]"},
    {0x010d9dbb, "A.cam1[+0x94ac]"},
    {0x010d9fc9, "A.cam2[+0xa5ac]"},
    {0x010da1eb, "A.cam3[+0xb6ac]"},

    // Group B - vec4 splats (7 groups of 4 = 28)
    {0x01496d30, "B.splat0+0"}, {0x01496d34, "B.splat0+4"},
    {0x01496d38, "B.splat0+8"}, {0x01496d3c, "B.splat0+c"},
    {0x01497960, "B.splat1+0"}, {0x01497964, "B.splat1+4"},
    {0x01497968, "B.splat1+8"}, {0x0149796c, "B.splat1+c"},
    {0x0149a2c0, "B.splat2+0"}, {0x0149a2c4, "B.splat2+4"},
    {0x0149a2c8, "B.splat2+8"}, {0x0149a2cc, "B.splat2+c"},
    {0x0149b080, "B.splat3+0"}, {0x0149b084, "B.splat3+4"},
    {0x0149b088, "B.splat3+8"}, {0x0149b08c, "B.splat3+c"},
    {0x0149b990, "B.splat4+0"}, {0x0149b994, "B.splat4+4"},
    {0x0149b998, "B.splat4+8"}, {0x0149b99c, "B.splat4+c"},
    {0x0149bd90, "B.splat5+0"}, {0x0149bd94, "B.splat5+4"},
    {0x0149bd98, "B.splat5+8"}, {0x0149bd9c, "B.splat5+c"},
    {0x014a1130, "B.splat6+0"}, {0x014a1134, "B.splat6+4"},
    {0x014a1138, "B.splat6+8"}, {0x014a113c, "B.splat6+c"},

    // Group C - projection-builder argument constants (12)
    {0x014c2ba0, "C.proj0"},  {0x014c2be8, "C.proj1"},
    {0x014c3098, "C.proj2"},  {0x014c3168, "C.proj3"},
    {0x014c3198, "C.proj4"},  {0x014c3260, "C.proj5"},
    {0x014c35ec, "C.proj6"},  {0x014c365c, "C.proj7"},
    {0x014c36b8, "C.proj8"},  {0x014c3714, "C.proj9"},
    {0x014c375c, "C.proj10"}, {0x014c4ea4, "C.proj11(vmulss)"},
}};

// GRR (CUSA01130/01112/01113/02318): the remaster hard-codes 1.7777778f in only 4 places (vs GR2's
// 45), so most projection aspect is derived from the render resolution, not a baked constant. Of
// the four, three are patched here; the writable-.data occurrence at VA 0x1693978 (adjacent to a
// -3.5555556f) is deliberately excluded - it may be a live GPU/matrix structure that runtime writes
// would clobber, and a stray aspect edit there risks corrupting rendering.
// Offsets are LOADED VAs (eboot_base + offset), matching kGrrResSites: the GRR SELF stores ph[0]
// (p_vaddr=0) at payload offset 0xB710, not at its declared ELF p_offset 0x4000, so sites map as
// va = file_offset - 0xB710. A 0x4000-based value lands 0x7710 high and never applies.
constexpr std::array<AspectSite, 3> kGrrAspectSites = {{
    // Camera FOV+aspect: MOVABS RAX, imm64 {0.87266f (FOV), 1.7777778f (aspect)}; MOV [R13+0x138],
    // RAX. The aspect is the high dword of the immediate; rewriting it keeps the FOV. High
    // confidence - a struct-init write like GR2 Group A.
    {0x0026a337, "G1.cam_movabs[+0x138]"},
    // rodata float constants (read-only segment). 1.7777778f is aspect-specific, so these are game
    // constants the linker placed near C-runtime rodata, not stdlib values. Byte-guarded and
    // read-only-safe, but their consumer is unconfirmed - verify visually per aspect.
    {0x01025b88, "G2.rodata"},
    {0x0104c524, "G3.rodata"},
}};

// Expected pre-patch bytes: 1.7777778f LE = 39 8E E3 3F
constexpr std::array<std::uint8_t, 4> kExpected = {0x39, 0x8E, 0xE3, 0x3F};

constexpr std::array<std::uint8_t, 4> EncodeReplacement(TargetAspect t) {
    switch (t) {
    case TargetAspect::R16_10: return {0xCD, 0xCC, 0xCC, 0x3F}; // 1.6f
    case TargetAspect::R21_9:  return {0x55, 0x55, 0x15, 0x40}; // 2.3333333f
    case TargetAspect::R32_9:  return {0x39, 0x8E, 0x63, 0x40}; // 3.5555556f
    case TargetAspect::R21_10: return {0x66, 0x66, 0x06, 0x40}; // 2.1f
    case TargetAspect::R32_10: return {0xCD, 0xCC, 0x4C, 0x40}; // 3.2f
    case TargetAspect::R4_3:   return {0xAB, 0xAA, 0xAA, 0x3F}; // 1.3333334f
    default:                   return {0x39, 0x8E, 0xE3, 0x3F}; // 16:9 (no-op)
    }
}

const char* TargetName(TargetAspect t) {
    switch (t) {
    case TargetAspect::R16_10: return "16:10 (1.6)";
    case TargetAspect::R21_9:  return "21:9 (2.333)";
    case TargetAspect::R32_9:  return "32:9 (3.556)";
    case TargetAspect::R21_10: return "21:10 (2.1)";
    case TargetAspect::R32_10: return "32:10 (3.2)";
    case TargetAspect::R4_3:   return "4:3 (1.333)";
    default:                   return "Off";
    }
}

// Shared byte-guarded patch loop. Only an exact 1.7777778f is rewritten; anything else is left
// untouched and logged. Fills *stats when non-null.
int ApplyAspectSites(uintptr_t eboot_base, TargetAspect target,
                     std::span<const AspectSite> sites, const char* tag, AspectStats* stats) {
    const auto replacement = EncodeReplacement(target);
    int patched = 0, already = 0, mismatched = 0;

    for (const auto& s : sites) {
        std::uint8_t* site = reinterpret_cast<std::uint8_t*>(eboot_base + s.va_offset);

        if (std::memcmp(site, replacement.data(), 4) == 0) {
            already++;
            continue;
        }
        if (std::memcmp(site, kExpected.data(), 4) != 0) {
            LOG_WARNING(Core,
                "[{} Aspect] {} @ vaddr 0x{:x}: unexpected bytes "
                "(got {:02x} {:02x} {:02x} {:02x}, expected {:02x} {:02x} {:02x} {:02x}) - NOT patched",
                tag, s.group, s.va_offset,
                site[0], site[1], site[2], site[3],
                kExpected[0], kExpected[1], kExpected[2], kExpected[3]);
            mismatched++;
            continue;
        }
        std::memcpy(site, replacement.data(), 4);
        patched++;
    }

    LOG_INFO(Core,
        "[{} Aspect] target={} - {} patched, {} already, {} mismatched (of {} total)",
        tag, TargetName(target), patched, already, mismatched, static_cast<int>(sites.size()));

    if (stats) {
        stats->patched = patched;
        stats->already = already;
        stats->mismatched = mismatched;
    }
    return patched;
}

} // namespace

int ApplyGr2AspectPatches(uintptr_t eboot_base, TargetAspect target, AspectStats* stats) {
    if (target == TargetAspect::Off) {
        LOG_INFO(Core, "[GR2 Aspect] target=Off, skipping");
        return 0;
    }
    if (eboot_base == 0) {
        LOG_ERROR(Core, "[GR2 Aspect] eboot_base is 0 - cannot patch");
        return 0;
    }
    return ApplyAspectSites(eboot_base, target, kGr2AspectSites, "GR2", stats);
}

int ApplyGrrAspectPatches(uintptr_t eboot_base, TargetAspect target, AspectStats* stats) {
    if (target == TargetAspect::Off) {
        LOG_INFO(Core, "[GRR Aspect] target=Off, skipping");
        return 0;
    }
    if (eboot_base == 0) {
        LOG_ERROR(Core, "[GRR Aspect] eboot_base is 0 - cannot patch");
        return 0;
    }
    return ApplyAspectSites(eboot_base, target, kGrrAspectSites, "GRR", stats);
}

int ApplyAspectPatches(uintptr_t eboot_base, TargetAspect target) {
    if (target == TargetAspect::Off) {
        LOG_INFO(Core, "[Aspect] target=Off - aspect patching disabled");
        return 0;
    }
    if (eboot_base == 0) {
        LOG_ERROR(Core, "[Aspect] eboot_base is 0 - cannot patch");
        return 0;
    }

    const bool is_grr = Config::isGravityRushRemastered();

    // GR2 first unless the GRR title flag (set in emulator.cpp before the eboot loads) is set:
    // running GR2's 45-site loop on a GRR eboot only spams "unexpected bytes" warnings. Without the
    // flag, GR2 runs first and GRR is the zero-hit fallback.
    if (!is_grr) {
        AspectStats gr2{};
        const int gr2_patched = ApplyGr2AspectPatches(eboot_base, target, &gr2);
        if (gr2.patched > 0 || gr2.already > 0) {
            return gr2_patched;
        }
        LOG_INFO(Core,
            "[Aspect] GR2 aspect patches matched no sites "
            "(patched={} already={} mismatched={}) - falling back to GRR",
            gr2.patched, gr2.already, gr2.mismatched);
    } else {
        LOG_INFO(Core,
            "[Aspect] Gravity Rush Remastered detected - using GRR aspect patches "
            "(GR2 path skipped)");
    }

    // GRR pass - reached directly when the title flag is set, or as the GR2 zero-hit fallback.
    AspectStats grr{};
    const int grr_patched = ApplyGrrAspectPatches(eboot_base, target, &grr);
    if (grr.patched > 0 || grr.already > 0) {
        return grr_patched;
    }

    LOG_WARNING(Core,
        "[Aspect] neither GR2 nor GRR aspect constants matched this eboot "
        "(GRR: patched={} already={} mismatched={}) - no aspect patches applied. "
        "The eboot may be an unsupported title or version.",
        grr.patched, grr.already, grr.mismatched);
    return 0;
}

TargetAspect ParseAspectFromConfig(std::string_view s) {
    if (s == "16:10") return TargetAspect::R16_10;
    if (s == "21:9")  return TargetAspect::R21_9;
    if (s == "32:9")  return TargetAspect::R32_9;
    if (s == "21:10") return TargetAspect::R21_10;
    if (s == "32:10") return TargetAspect::R32_10;
    if (s == "4:3")   return TargetAspect::R4_3;
    // "16:9", "Off", empty, or anything unrecognized -> no override.
    return TargetAspect::Off;
}

float TargetAspectToRatio(TargetAspect t) {
    switch (t) {
    case TargetAspect::R16_10: return 1.6f;
    case TargetAspect::R21_9:  return 21.0f / 9.0f;       // 2.3333333f
    case TargetAspect::R32_9:  return 32.0f / 9.0f;       // 3.5555556f
    case TargetAspect::R21_10: return 21.0f / 10.0f;      // 2.1f
    case TargetAspect::R32_10: return 32.0f / 10.0f;      // 3.2f
    case TargetAspect::R4_3:   return 4.0f / 3.0f;        // 1.3333334f
    default:                   return 16.0f / 9.0f;       // 1.7777778f (Off)
    }
}

} // namespace Libraries::AspectPatches
