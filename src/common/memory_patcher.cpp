// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <codecvt>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>
#include "common/config.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/file_format/psf.h"
#include "memory_patcher.h"

namespace MemoryPatcher {

EXPORT uintptr_t g_eboot_address;
uint64_t g_eboot_image_size;
std::string g_game_serial;
std::string patch_file;
bool patches_applied = false;
std::vector<patchInfo> pending_patches;

std::string toHex(u64 value, size_t byteSize) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(byteSize * 2) << value;
    return ss.str();
}

std::string convertValueToHex(const std::string type, const std::string valueStr) {
    std::string result;

    if (type == "byte") {
        const u32 value = std::stoul(valueStr, nullptr, 16);
        result = toHex(value, 1);
    } else if (type == "bytes16") {
        const u32 value = std::stoul(valueStr, nullptr, 16);
        result = toHex(value, 2);
    } else if (type == "bytes32") {
        const u32 value = std::stoul(valueStr, nullptr, 16);
        result = toHex(value, 4);
    } else if (type == "bytes64") {
        const u64 value = std::stoull(valueStr, nullptr, 16);
        result = toHex(value, 8);
    } else if (type == "float32") {
        union {
            float f;
            uint32_t i;
        } floatUnion;
        floatUnion.f = std::stof(valueStr);
        result = toHex(floatUnion.i, sizeof(floatUnion.i));
    } else if (type == "float64") {
        union {
            double d;
            uint64_t i;
        } doubleUnion;
        doubleUnion.d = std::stod(valueStr);
        result = toHex(doubleUnion.i, sizeof(doubleUnion.i));
    } else if (type == "utf8") {
        std::vector<unsigned char> byteArray =
            std::vector<unsigned char>(valueStr.begin(), valueStr.end());
        byteArray.push_back('\0');
        std::stringstream ss;
        for (unsigned char c : byteArray) {
            ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(c);
        }
        result = ss.str();
    } else if (type == "utf16") {
        std::wstring wide_str(valueStr.size(), L'\0');
        std::mbstowcs(&wide_str[0], valueStr.c_str(), valueStr.size());
        wide_str.resize(std::wcslen(wide_str.c_str()));

        std::u16string valueStringU16;

        for (wchar_t wc : wide_str) {
            if (wc <= 0xFFFF) {
                valueStringU16.push_back(static_cast<char16_t>(wc));
            } else {
                wc -= 0x10000;
                valueStringU16.push_back(static_cast<char16_t>(0xD800 | (wc >> 10)));
                valueStringU16.push_back(static_cast<char16_t>(0xDC00 | (wc & 0x3FF)));
            }
        }

        std::vector<unsigned char> byteArray;
        // convert to little endian
        for (char16_t ch : valueStringU16) {
            unsigned char low_byte = static_cast<unsigned char>(ch & 0x00FF);
            unsigned char high_byte = static_cast<unsigned char>((ch >> 8) & 0x00FF);

            byteArray.push_back(low_byte);
            byteArray.push_back(high_byte);
        }
        byteArray.push_back('\0');
        byteArray.push_back('\0');
        std::stringstream ss;

        for (unsigned char ch : byteArray) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(ch);
        }
        result = ss.str();
    } else if (type == "bytes") {
        result = valueStr;
    } else if (type == "mask" || type == "mask_jump32") {
        result = valueStr;
    } else {
        LOG_INFO(Loader, "Error applying Patch, unknown type: {}", type);
    }
    return result;
}

void ApplyPendingPatches();

// Defined later in this file. Applies any config-toggled GR2-specific
// render-pass disable patches; safe to call on non-GR2 binaries (each site
// signature-checks before writing).
static void ApplyGR2GraphicsPatches();

void ApplyPatchesFromXML(std::filesystem::path path) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(path.c_str());

    auto* param_sfo = Common::Singleton<PSF>::Instance();
    auto app_version = param_sfo->GetString("APP_VER").value_or("Unknown version");

    if (result) {
        auto patchXML = doc.child("Patch");
        for (pugi::xml_node_iterator it = patchXML.children().begin();
             it != patchXML.children().end(); ++it) {

            if (std::string(it->name()) == "Metadata") {
                if (std::string(it->attribute("isEnabled").value()) == "true") {
                    std::string currentPatchName = it->attribute("Name").value();
                    std::string metadataAppVer = it->attribute("AppVer").value();
                    bool versionMatches = metadataAppVer == app_version;

                    auto patchList = it->first_child();
                    for (pugi::xml_node_iterator patchLineIt = patchList.children().begin();
                         patchLineIt != patchList.children().end(); ++patchLineIt) {

                        std::string type = patchLineIt->attribute("Type").value();
                        if (!versionMatches && type != "mask" && type != "mask_jump32")
                            continue;

                        std::string address = patchLineIt->attribute("Address").value();
                        std::string patchValue = patchLineIt->attribute("Value").value();
                        std::string maskOffsetStr = patchLineIt->attribute("Offset").value();
                        std::string targetStr = "";
                        std::string sizeStr = "";
                        if (type == "mask_jump32") {
                            targetStr = patchLineIt->attribute("Target").value();
                            sizeStr = patchLineIt->attribute("Size").value();
                        } else {
                            patchValue = convertValueToHex(type, patchValue);
                        }

                        bool littleEndian = false;
                        if (type == "bytes16" || type == "bytes32" || type == "bytes64") {
                            littleEndian = true;
                        }

                        MemoryPatcher::PatchMask patchMask = MemoryPatcher::PatchMask::None;
                        int maskOffsetValue = 0;

                        if (type == "mask")
                            patchMask = MemoryPatcher::PatchMask::Mask;

                        if (type == "mask_jump32")
                            patchMask = MemoryPatcher::PatchMask::Mask_Jump32;

                        if ((type == "mask" || type == "mask_jump32") && !maskOffsetStr.empty()) {
                            maskOffsetValue = std::stoi(maskOffsetStr, 0, 10);
                        }

                        MemoryPatcher::PatchMemory(currentPatchName, address, patchValue, targetStr,
                                                   sizeStr, false, littleEndian, patchMask,
                                                   maskOffsetValue);
                    }
                }
            }
        }
    } else {
        LOG_ERROR(Loader, "Could not parse patch XML: {}", result.description());
    }
}

void OnGameLoaded() {
    std::filesystem::path patch_dir = Common::FS::GetUserPath(Common::FS::PathType::PatchesDir);
    if (!patch_file.empty()) {

        auto file_path = (patch_dir / patch_file).native();
        if (std::filesystem::exists(patch_file)) {
            ApplyPatchesFromXML(patch_file);
        } else {
            ApplyPatchesFromXML(file_path);
        }
    } else if (Config::getLoadAutoPatches()) {
        for (auto const& repo : std::filesystem::directory_iterator(patch_dir)) {
            if (!repo.is_directory()) {
                continue;
            }
            std::ifstream json_file{repo.path() / "files.json"};
            nlohmann::json available_patches = nlohmann::json::parse(json_file);
            std::filesystem::path game_patch_file;
            for (auto const& [filename, serials] : available_patches.items()) {
                if (std::find(serials.begin(), serials.end(), g_game_serial) != serials.end()) {
                    game_patch_file = repo.path() / filename;
                    break;
                }
            }
            if (std::filesystem::exists(game_patch_file)) {
                ApplyPatchesFromXML(game_patch_file);
            }
        }
    }
    ApplyPendingPatches();
    // Apply GR2-specific render-pass disable patches (config-toggled). Safe to
    // call on non-GR2 binaries — each patch site signature-checks before
    // writing and bails out cleanly on a mismatch.
    ApplyGR2GraphicsPatches();
}

void AddPatchToQueue(patchInfo patchToAdd) {
    if (patches_applied) {
        PatchMemory(patchToAdd.modNameStr, patchToAdd.offsetStr, patchToAdd.valueStr,
                    patchToAdd.targetStr, patchToAdd.sizeStr, patchToAdd.isOffset,
                    patchToAdd.littleEndian, patchToAdd.patchMask, patchToAdd.maskOffset);
        return;
    }
    pending_patches.push_back(patchToAdd);
}

void ApplyPendingPatches() {
    patches_applied = true;
    for (size_t i = 0; i < pending_patches.size(); ++i) {
        const patchInfo& currentPatch = pending_patches[i];

        if (currentPatch.gameSerial != "*" && currentPatch.gameSerial != g_game_serial)
            continue;

        PatchMemory(currentPatch.modNameStr, currentPatch.offsetStr, currentPatch.valueStr,
                    currentPatch.targetStr, currentPatch.sizeStr, currentPatch.isOffset,
                    currentPatch.littleEndian, currentPatch.patchMask, currentPatch.maskOffset);
    }

    pending_patches.clear();
}

void PatchMemory(std::string modNameStr, std::string offsetStr, std::string valueStr,
                 std::string targetStr, std::string sizeStr, bool isOffset, bool littleEndian,
                 PatchMask patchMask, int maskOffset) {
    // Send a request to modify the process memory.
    void* cheatAddress = nullptr;

    if (patchMask == PatchMask::None) {
        if (isOffset) {
            cheatAddress = reinterpret_cast<void*>(g_eboot_address + std::stoi(offsetStr, 0, 16));
        } else {
            cheatAddress =
                reinterpret_cast<void*>(g_eboot_address + (std::stoi(offsetStr, 0, 16) - 0x400000));
        }
    }

    if (patchMask == PatchMask::Mask) {
        cheatAddress = reinterpret_cast<void*>(PatternScan(offsetStr) + maskOffset);
    }

    if (patchMask == PatchMask::Mask_Jump32) {
        int jumpSize = std::stoi(sizeStr);

        constexpr int MAX_PATTERN_LENGTH = 256;
        if (jumpSize < 5) {
            LOG_ERROR(Loader, "Jump size must be at least 5 bytes");
            return;
        }
        if (jumpSize > MAX_PATTERN_LENGTH) {
            LOG_ERROR(Loader, "Jump size must be no more than {} bytes.", MAX_PATTERN_LENGTH);
            return;
        }

        // Find the base address using "Address"
        uintptr_t baseAddress = PatternScan(offsetStr);
        if (baseAddress == 0) {
            LOG_ERROR(Loader, "PatternScan failed for mask_jump32 with pattern: {}", offsetStr);
            return;
        }
        uintptr_t patchAddress = baseAddress + maskOffset;

        // Fills the original region (jumpSize bytes) with NOPs
        std::vector<u8> nopBytes(jumpSize, 0x90);
        std::memcpy(reinterpret_cast<void*>(patchAddress), nopBytes.data(), nopBytes.size());

        // Use "Target" to locate the start of the code cave
        uintptr_t jump_target = PatternScan(targetStr);
        if (jump_target == 0) {
            LOG_ERROR(Loader, "PatternScan failed to Target with pattern: {}", targetStr);
            return;
        }

        // Converts the Value attribute to a byte array (payload)
        std::vector<u8> payload;
        for (size_t i = 0; i < valueStr.length(); i += 2) {

            std::string tempStr = valueStr.substr(i, 2);
            const char* byteStr = tempStr.c_str();
            char* endPtr;
            unsigned int byteVal = std::strtoul(byteStr, &endPtr, 16);

            if (endPtr != byteStr + 2) {
                LOG_ERROR(Loader, "Invalid byte in Value: {}", valueStr.substr(i, 2));
                return;
            }
            payload.push_back(static_cast<u8>(byteVal));
        }

        // Calculates the end of the code cave (where the return jump will be inserted)
        uintptr_t code_cave_end = jump_target + payload.size();

        // Write the payload to the code cave, from jump_target
        std::memcpy(reinterpret_cast<void*>(jump_target), payload.data(), payload.size());

        // Inserts the initial jump in the original region to divert to the code cave
        u8 jumpInstruction[5];
        jumpInstruction[0] = 0xE9;
        s32 relJump = static_cast<s32>(jump_target - patchAddress - 5);
        std::memcpy(&jumpInstruction[1], &relJump, sizeof(relJump));
        std::memcpy(reinterpret_cast<void*>(patchAddress), jumpInstruction,
                    sizeof(jumpInstruction));

        // Inserts jump back at the end of the code cave to resume execution after patching
        u8 jumpBack[5];
        jumpBack[0] = 0xE9;
        // Calculates the relative offset to return to the instruction immediately following the
        // overwritten region
        s32 target_return = static_cast<s32>((patchAddress + jumpSize) - (code_cave_end + 5));
        std::memcpy(&jumpBack[1], &target_return, sizeof(target_return));
        std::memcpy(reinterpret_cast<void*>(code_cave_end), jumpBack, sizeof(jumpBack));

        LOG_INFO(Loader,
                 "Applied Patch mask_jump32: {}, PatchAddress: {:#x}, JumpTarget: {:#x}, "
                 "CodeCaveEnd: {:#x}, JumpSize: {}",
                 modNameStr, patchAddress, jump_target, code_cave_end, jumpSize);
        return;
    }

    if (cheatAddress == nullptr) {
        LOG_ERROR(Loader, "Failed to get address for patch {}", modNameStr);
        return;
    }

    std::vector<unsigned char> bytePatch;

    for (size_t i = 0; i < valueStr.length(); i += 2) {
        unsigned char byte =
            static_cast<unsigned char>(std::strtol(valueStr.substr(i, 2).c_str(), nullptr, 16));

        bytePatch.push_back(byte);
    }

    if (littleEndian) {
        std::reverse(bytePatch.begin(), bytePatch.end());
    }

    std::memcpy(cheatAddress, bytePatch.data(), bytePatch.size());

    LOG_INFO(Loader, "Applied patch: {}, Offset: {}, Value: {}", modNameStr,
             (uintptr_t)cheatAddress, valueStr);
}

static std::vector<int32_t> PatternToByte(const std::string& pattern) {
    std::vector<int32_t> bytes;
    const char* start = pattern.data();
    const char* end = start + pattern.size();

    for (const char* current = start; current < end; ++current) {
        if (*current == '?') {
            ++current;
            if (*current == '?')
                ++current;
            bytes.push_back(-1);
        } else {
            bytes.push_back(strtoul(current, const_cast<char**>(&current), 16));
        }
    }

    return bytes;
}

uintptr_t PatternScan(const std::string& signature) {
    std::vector<int32_t> patternBytes = PatternToByte(signature);
    const auto scanBytes = static_cast<uint8_t*>((void*)g_eboot_address);

    const int32_t* sigPtr = patternBytes.data();
    const size_t sigSize = patternBytes.size();

    uint32_t foundResults = 0;
    for (uint32_t i = 0; i < g_eboot_image_size - sigSize; ++i) {
        bool found = true;
        for (uint32_t j = 0; j < sigSize; ++j) {
            if (scanBytes[i + j] != sigPtr[j] && sigPtr[j] != -1) {
                found = false;
                break;
            }
        }

        if (found) {
            foundResults++;
            return reinterpret_cast<uintptr_t>(&scanBytes[i]);
        }
    }

    return 0;
}

// =============================================================================
// GR2 (Gravity Rush 2) granular render-pass disable patches.
//
// GR2's renderer dispatches every named pass through a uniform per-frame
// template:
//
//     mov   al, byte ptr [rip + SKIP_BYTE_DISP]   ; 6 bytes: 8A 05 dd dd dd dd
//     test  al, al                                 ; 2 bytes: 84 C0
//     jne   SKIP                                   ; 2 bytes: 75 xx
//     lea   rdi, [rip + arg1]
//     call  RenderState::IsActive                  ; secondary gate
//     test  eax, eax
//     je    SKIP
//     lea   rdi, [rip + arg1]
//     lea   rsi, [rip + "PassName"]
//     lea   rdx, [rip + arg3]
//     call  RenderPass::Execute
//     ...
//   SKIP:
//     ; next pass
//
// SKIP_BYTE points into BSS (zero-initialised) so by default the pass runs.
// Setting that byte to a non-zero value at runtime forces every dispatch into
// the SKIP branch, eliminating the pass body's draw calls AND the surrounding
// `IsActive` query + `Execute` invocation. This is the engine's own debug
// toggle mechanism; we just flip the byte.
//
// We patch in two redundant ways for every site:
//   1. Rewrite the 6-byte `mov al, byte ptr [rip+...]` to `mov al, 1; nop x4`
//      (B0 01 90 90 90 90). The subsequent test+jne now always branches.
//      This is the primary guarantee.
//   2. Also write 1 to the BSS skip byte directly, in case any other code path
//      reads the same flag without going through the patched mov.
//
// Each site is signature-verified before being patched: if the opcode bytes
// don't match `8A 05 ?? ?? ?? ??` or the embedded RIP-relative disp doesn't
// resolve to the expected BSS address, the patch is skipped with a warning.
// This makes the patches safe to leave wired in for non-GR2 binaries — they
// will simply no-op rather than corrupting an unrelated game.
//
// VAs below were extracted from a CUSA02xxx-style GR2 eboot of size
// 0x1a7da38 (ph0 vaddr=0, ph2 vaddr=0x1708000). They are eboot offsets, not
// PS4 logical addresses, and so are used directly against g_eboot_address.
//
// Layout of this section:
//   - `GR2PassPatchSite`: one row per dispatch site in the binary.
//   - `kSites*[]`: per-toggle arrays grouping sites that share a config flag
//     (most toggles map to one site; MotionBlur and ParticleDistortion each
//     have two dispatch sites and need both patched).
//   - `GR2Toggle`: ties a config getter to its set of sites.
//   - `kAllToggles[]`: the master list. Adding a new toggle = add a row here
//     plus the corresponding sites array; no other code changes required.
// =============================================================================

namespace {

struct GR2PassPatchSite {
    const char* name;       // pass identifier (for logging)
    u64 mov_va;             // VA of the 6-byte `mov al, byte ptr [rip+disp32]`
    u64 skip_byte_va;       // VA of the skip flag in BSS (computed from disp32)
};

// 6 bytes: `mov al, 1; nop; nop; nop; nop` — overwrites the original mov so AL
// is always loaded with 1, forcing the subsequent `test al, al; jne SKIP` to
// always branch past the pass body.
constexpr u8 kForceSkipPatch[6] = {0xB0, 0x01, 0x90, 0x90, 0x90, 0x90};

// --- Per-toggle dispatch-site tables ---
// Single-site toggles (one dispatch per pass).
constexpr GR2PassPatchSite kSitesShadowCast0[] =
    {{"ShadowCast0", 0x10318aa, 0x1b15d88}};
constexpr GR2PassPatchSite kSitesShadowCast1[] =
    {{"ShadowCast1", 0x10318ea, 0x1b15da8}};
constexpr GR2PassPatchSite kSitesShadowCast2[] =
    {{"ShadowCast2", 0x103192a, 0x1b15dc8}};
constexpr GR2PassPatchSite kSitesShadowTrace[] =
    {{"ShadowTrace", 0x1036434, 0x1b15fa8}};
constexpr GR2PassPatchSite kSitesSSAO[] =
    {{"SSAO", 0x1036719, 0x1b15fc8}};
constexpr GR2PassPatchSite kSitesIBL[] =
    {{"IBL", 0x1037f0f, 0x1b16008}};
constexpr GR2PassPatchSite kSitesContour[] =
    {{"Contour", 0x10369db, 0x1b15fe8}};
constexpr GR2PassPatchSite kSitesSelfTranslucent[] =
    {{"SelfTranslucent", 0x1034042, 0x1b15ee8}};
constexpr GR2PassPatchSite kSitesBloom[] =
    {{"Bloom", 0x103de9c, 0x1b16248}};
constexpr GR2PassPatchSite kSitesAntialias[] =
    {{"Antialias", 0x103f9c4, 0x1b162e8}};
constexpr GR2PassPatchSite kSitesEffector[] =
    {{"Effector", 0x102f239, 0x1b16340}};
constexpr GR2PassPatchSite kSitesFogRender[] =
    {{"FogRender", 0x103c012, 0x1b16148}};
constexpr GR2PassPatchSite kSitesFogCloud[] =
    {{"FogCloud", 0x102fdc0, 0x1b16378}};
constexpr GR2PassPatchSite kSitesFogDist[] =
    {{"FogDist", 0x10306ce, 0x1b163b0}};
constexpr GR2PassPatchSite kSitesFogCast[] =
    {{"FogCast", 0x1031849, 0x1b163e8}};
constexpr GR2PassPatchSite kSitesParticleCompute[] =
    {{"Particle_Compute", 0x102f415, 0x1b15d68}};

// Multi-site toggles.
// ParticleDistortion: heat-haze particles dispatch from both a low-watermark
// path (`Particle_Distortion`) and a high-watermark path
// (`Particle_Distortion_FogEnable_HW`). Patch both.
constexpr GR2PassPatchSite kSitesParticleDistortion[] = {
    {"Particle_Distortion",                0x103b895, 0x1b16108},
    {"Particle_Distortion_FogEnable_HW",   0x103c503, 0x1b161c8},
};
// MotionBlur: two dispatch sites in this binary (high-watermark and
// low-watermark paths). Patching only one (as the existing CUSA04943-targeted
// disableMotionBlur apparently does) leaves the other live, which is why
// motion blur appears un-disable-able on this build. Patch both.
constexpr GR2PassPatchSite kSitesMotionBlur[] = {
    {"MotionBlur#1", 0x103e30b, 0x1b165f8},
    {"MotionBlur#2", 0x103ea47, 0x1b16268},
};

// --- Toggle table ---
struct GR2Toggle {
    const char* config_key;             // human-readable name for log lines
    bool (*enabled)();                  // pointer to the Config::getDisableX accessor
    const GR2PassPatchSite* sites;
    size_t num_sites;
};

constexpr GR2Toggle kAllToggles[] = {
    // Shadow group
    {"disableShadowCast0",      &Config::getDisableShadowCast0,
        kSitesShadowCast0,      std::size(kSitesShadowCast0)},
    {"disableShadowCast1",      &Config::getDisableShadowCast1,
        kSitesShadowCast1,      std::size(kSitesShadowCast1)},
    {"disableShadowCast2",      &Config::getDisableShadowCast2,
        kSitesShadowCast2,      std::size(kSitesShadowCast2)},
    {"disableShadowTrace",      &Config::getDisableShadowTrace,
        kSitesShadowTrace,      std::size(kSitesShadowTrace)},
    // Lighting / shading group
    {"disableSSAO",             &Config::getDisableSSAO,
        kSitesSSAO,             std::size(kSitesSSAO)},
    {"disableIBL",              &Config::getDisableIBL,
        kSitesIBL,              std::size(kSitesIBL)},
    {"disableContour",          &Config::getDisableContour,
        kSitesContour,          std::size(kSitesContour)},
    {"disableSelfTranslucent",  &Config::getDisableSelfTranslucent,
        kSitesSelfTranslucent,  std::size(kSitesSelfTranslucent)},
    // Post-process group
    {"disableBloom",            &Config::getDisableBloom,
        kSitesBloom,            std::size(kSitesBloom)},
    {"disableAntialias",        &Config::getDisableAntialias,
        kSitesAntialias,        std::size(kSitesAntialias)},
    {"disableMotionBlur",       &Config::getDisableMotionBlur,
        kSitesMotionBlur,       std::size(kSitesMotionBlur)},
    {"disableEffector",         &Config::getDisableEffector,
        kSitesEffector,         std::size(kSitesEffector)},
    // Fog group
    {"disableFogRender",        &Config::getDisableFogRender,
        kSitesFogRender,        std::size(kSitesFogRender)},
    {"disableFogCloud",         &Config::getDisableFogCloud,
        kSitesFogCloud,         std::size(kSitesFogCloud)},
    {"disableFogDist",          &Config::getDisableFogDist,
        kSitesFogDist,          std::size(kSitesFogDist)},
    {"disableFogCast",          &Config::getDisableFogCast,
        kSitesFogCast,          std::size(kSitesFogCast)},
    // Particle group
    {"disableParticleCompute",      &Config::getDisableParticleCompute,
        kSitesParticleCompute,      std::size(kSitesParticleCompute)},
    {"disableParticleDistortion",   &Config::getDisableParticleDistortion,
        kSitesParticleDistortion,   std::size(kSitesParticleDistortion)},
};

} // namespace

// Read the 6 bytes at mov_va and check they encode `mov al, byte ptr [rip+disp32]`
// with disp32 resolving to skip_byte_va. Returns true only if the binary at
// this offset looks like the GR2 dispatch template we expect.
static bool VerifyGR2PassSignature(const GR2PassPatchSite& site) {
    if (g_eboot_address == 0) {
        return false;
    }
    const u8* p = reinterpret_cast<const u8*>(g_eboot_address + site.mov_va);
    if (p[0] != 0x8A || p[1] != 0x05) {
        return false;
    }
    s32 disp;
    std::memcpy(&disp, p + 2, sizeof(disp));
    const u64 computed_target = site.mov_va + 6 + static_cast<s64>(disp);
    return computed_target == site.skip_byte_va;
}

// Apply one pass-disable patch: signature-check, then code-patch the mov +
// belt-and-suspenders BSS write. Returns true on success, false on signature
// mismatch (leaves the binary untouched). Safe to call on any binary.
static bool PatchGR2DisablePass(const GR2PassPatchSite& site) {
    if (!VerifyGR2PassSignature(site)) {
        LOG_WARNING(Loader,
                    "GR2 graphics patch: signature mismatch for pass '{}' at "
                    "VA 0x{:x} (expected 8A 05 ?? ?? ?? ?? -> 0x{:x}); "
                    "skipping. This is not the expected GR2 eboot version.",
                    site.name, site.mov_va, site.skip_byte_va);
        return false;
    }

    // Primary: rewrite the mov so AL is unconditionally 1.
    void* code_addr = reinterpret_cast<void*>(g_eboot_address + site.mov_va);
    std::memcpy(code_addr, kForceSkipPatch, sizeof(kForceSkipPatch));

    // Secondary: also set the BSS skip flag, in case any other dispatch path
    // reads it directly. Cheap insurance against missed dispatch sites.
    void* skip_addr = reinterpret_cast<void*>(g_eboot_address + site.skip_byte_va);
    *static_cast<volatile u8*>(skip_addr) = 1;

    LOG_INFO(Loader,
             "GR2 graphics patch: disabled pass '{}' (mov@0x{:x}, skip@0x{:x})",
             site.name, site.mov_va, site.skip_byte_va);
    return true;
}

// Top-level: walk the toggle table, apply each enabled toggle's sites.
// Called once from OnGameLoaded() after the XML patch queue has been drained.
// Safe to call unconditionally — patches no-op on non-GR2 binaries via the
// per-site signature check.
static void ApplyGR2GraphicsPatches() {
    if (g_eboot_address == 0) {
        LOG_WARNING(Loader,
                    "GR2 graphics patch: g_eboot_address is 0, skipping all "
                    "GR2-specific render-pass patches");
        return;
    }

    // Are any toggles enabled? Avoid spamming logs on every non-GR2 game.
    bool any_enabled = false;
    for (const auto& t : kAllToggles) {
        if (t.enabled()) {
            any_enabled = true;
            break;
        }
    }
    if (!any_enabled) {
        return;
    }

    // Fast bailout: probe the first known site. If the bytes don't match the
    // expected `mov al, byte ptr [rip+disp32]` opcode, this isn't a GR2 build
    // we know — bail without spamming per-site mismatch warnings.
    const u8* probe = reinterpret_cast<const u8*>(
        g_eboot_address + kSitesShadowCast0[0].mov_va);
    if (probe[0] != 0x8A || probe[1] != 0x05) {
        LOG_INFO(Loader,
                 "GR2 graphics patch: eboot probe at VA 0x{:x} = {:02x} {:02x} "
                 "(expected 8A 05); this is not a recognised GR2 binary. "
                 "Skipping all GR2-specific patches.",
                 kSitesShadowCast0[0].mov_va, probe[0], probe[1]);
        return;
    }

    LOG_INFO(Loader,
             "GR2 graphics patch: GR2 binary detected at g_eboot_address=0x{:x}",
             g_eboot_address);

    // Apply every enabled toggle.
    int total_sites_applied = 0;
    int total_sites_attempted = 0;
    int toggles_applied = 0;
    for (const auto& t : kAllToggles) {
        if (!t.enabled()) {
            continue;
        }
        ++toggles_applied;
        LOG_INFO(Loader, "GR2 graphics patch: applying toggle '{}' ({} site{})",
                 t.config_key, static_cast<int>(t.num_sites),
                 t.num_sites == 1 ? "" : "s");
        for (size_t i = 0; i < t.num_sites; ++i) {
            ++total_sites_attempted;
            if (PatchGR2DisablePass(t.sites[i])) {
                ++total_sites_applied;
            }
        }
    }
    LOG_INFO(Loader,
             "GR2 graphics patch: {} toggle(s) active, {}/{} dispatch sites patched",
             toggles_applied, total_sites_applied, total_sites_attempted);
}

} // namespace MemoryPatcher
