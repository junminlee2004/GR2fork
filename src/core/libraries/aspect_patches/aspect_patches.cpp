// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>

#include "common/logging/log.h"
#include "core/libraries/aspect_patches/aspect_patches.h"

namespace Libraries::AspectPatches {

namespace {

// 45 sites where the float constant 1.7777778f (0x3FE38E39) appears in the
// eboot. Verified to match cross-region 1.11 layout (bytes 39 8E E3 3F at
// each VA offset below).
//
// Group A (5):  MOV-imm32 struct-init writes (camera/aspect struct fields).
// Group B (28): vec4 (1.7778, 1.7778, 1.7778, 1.7778) splats loaded by
//               vmovaps for SIMD aspect math.
// Group C (12): rip-relative singletons consumed as the `aspect` argument
//               to the projection-matrix-building function at vaddr
//               0x010cf0e0 (all 12 callers feed the same builder).
struct AspectSite {
    std::uint32_t va_offset;  // VA offset from eboot base
    const char*   group;      // for logging
};

constexpr std::array<AspectSite, 45> kAspectSites = {{
    // Group A — struct-init MOV-imm32 (5)
    {0x0045b42b, "A.singleton"},
    {0x00465cc3, "A.struct[+0x12c]"},
    {0x010d9dbb, "A.cam1[+0x94ac]"},
    {0x010d9fc9, "A.cam2[+0xa5ac]"},
    {0x010da1eb, "A.cam3[+0xb6ac]"},

    // Group B — vec4 splats (7 groups of 4 = 28)
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

    // Group C — projection-builder argument constants (12)
    {0x014c2ba0, "C.proj0"},  {0x014c2be8, "C.proj1"},
    {0x014c3098, "C.proj2"},  {0x014c3168, "C.proj3"},
    {0x014c3198, "C.proj4"},  {0x014c3260, "C.proj5"},
    {0x014c35ec, "C.proj6"},  {0x014c365c, "C.proj7"},
    {0x014c36b8, "C.proj8"},  {0x014c3714, "C.proj9"},
    {0x014c375c, "C.proj10"}, {0x014c4ea4, "C.proj11(vmulss)"},
}};

// Expected pre-patch bytes: 1.7777778f LE = 39 8E E3 3F
constexpr std::array<std::uint8_t, 4> kExpected = {0x39, 0x8E, 0xE3, 0x3F};

constexpr std::array<std::uint8_t, 4> EncodeReplacement(TargetAspect t) {
    switch (t) {
    case TargetAspect::R16_10: return {0xCD, 0xCC, 0xCC, 0x3F}; // 1.6f
    case TargetAspect::R21_9:  return {0x55, 0x55, 0x15, 0x40}; // 2.3333333f
    case TargetAspect::R32_9:  return {0x39, 0x8E, 0x63, 0x40}; // 3.5555556f
    default:                   return {0x39, 0x8E, 0xE3, 0x3F}; // 16:9 (no-op)
    }
}

const char* TargetName(TargetAspect t) {
    switch (t) {
    case TargetAspect::R16_10: return "16:10 (1.6)";
    case TargetAspect::R21_9:  return "21:9 (2.333)";
    case TargetAspect::R32_9:  return "32:9 (3.556)";
    default:                   return "Off";
    }
}

} // namespace

int ApplyGr2AspectPatches(uintptr_t eboot_base, TargetAspect target) {
    if (target == TargetAspect::Off) {
        LOG_INFO(Core, "[GR2 Aspect] target=Off, skipping");
        return 0;
    }
    if (eboot_base == 0) {
        LOG_ERROR(Core, "[GR2 Aspect] eboot_base is 0 — cannot patch");
        return 0;
    }

    const auto replacement = EncodeReplacement(target);
    int patched = 0, already = 0, mismatched = 0;

    for (const auto& s : kAspectSites) {
        std::uint8_t* site = reinterpret_cast<std::uint8_t*>(eboot_base + s.va_offset);

        if (std::memcmp(site, replacement.data(), 4) == 0) {
            already++;
            continue;
        }
        if (std::memcmp(site, kExpected.data(), 4) != 0) {
            LOG_WARNING(Core,
                "[GR2 Aspect] {} @ vaddr 0x{:x}: unexpected bytes "
                "(got {:02x} {:02x} {:02x} {:02x}, expected {:02x} {:02x} {:02x} {:02x}) — NOT patched",
                s.group, s.va_offset,
                site[0], site[1], site[2], site[3],
                kExpected[0], kExpected[1], kExpected[2], kExpected[3]);
            mismatched++;
            continue;
        }
        std::memcpy(site, replacement.data(), 4);
        patched++;
    }

    LOG_INFO(Core,
        "[GR2 Aspect] target={} — {} patched, {} already, {} mismatched (of {} total)",
        TargetName(target), patched, already, mismatched,
        static_cast<int>(kAspectSites.size()));
    return patched;
}

TargetAspect ParseAspectFromConfig(std::string_view s) {
    if (s == "16:10") return TargetAspect::R16_10;
    if (s == "21:9")  return TargetAspect::R21_9;
    if (s == "32:9")  return TargetAspect::R32_9;
    // "16:9", "Off", empty, or anything unrecognized → no override.
    return TargetAspect::Off;
}

float TargetAspectToRatio(TargetAspect t) {
    switch (t) {
    case TargetAspect::R16_10: return 1.6f;
    case TargetAspect::R21_9:  return 21.0f / 9.0f;       // 2.3333333f
    case TargetAspect::R32_9:  return 32.0f / 9.0f;       // 3.5555556f
    default:                   return 16.0f / 9.0f;       // 1.7777778f (Off)
    }
}

} // namespace Libraries::AspectPatches
