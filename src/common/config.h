// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <vector>
#include "types.h"

namespace Config {

enum class ConfigMode {
    Default,
    Global,
    Clean,
};
void setConfigMode(ConfigMode mode);

struct GameInstallDir {
    std::filesystem::path path;
    bool enabled;
};

enum HideCursorState : int { Never, Idle, Always };

void load(const std::filesystem::path& path, bool is_game_specific = false);
void save(const std::filesystem::path& path, bool is_game_specific = false);
void resetGameSpecificValue(std::string entry);

bool getGameRunning();
void setGameRunning(bool running);
int getVolumeSlider();
void setVolumeSlider(int volumeValue, bool is_game_specific = false);
std::string getTrophyKey();
void setTrophyKey(std::string key);
bool getIsFullscreen();
void setIsFullscreen(bool enable, bool is_game_specific = false);
std::string getFullscreenMode();
void setFullscreenMode(std::string mode, bool is_game_specific = false);
std::string getPresentMode();
void setPresentMode(std::string mode, bool is_game_specific = false);
u32 getWindowWidth();
u32 getWindowHeight();
void setWindowWidth(u32 width, bool is_game_specific = false);
void setWindowHeight(u32 height, bool is_game_specific = false);
u32 getInternalScreenWidth();
u32 getInternalScreenHeight();
void setInternalScreenWidth(u32 width);
void setInternalScreenHeight(u32 height);
bool debugDump();
void setDebugDump(bool enable, bool is_game_specific = false);
s32 getGpuId();
void setGpuId(s32 selectedGpuId, bool is_game_specific = false);
bool allowHDR();
void setAllowHDR(bool enable, bool is_game_specific = false);
bool collectShadersForDebug();
void setCollectShaderForDebug(bool enable, bool is_game_specific = false);
bool showSplash();
void setShowSplash(bool enable, bool is_game_specific = false);
std::string sideTrophy();
void setSideTrophy(std::string side, bool is_game_specific = false);
bool nullGpu();
void setNullGpu(bool enable, bool is_game_specific = false);
bool copyGPUCmdBuffers();
void setCopyGPUCmdBuffers(bool enable, bool is_game_specific = false);
bool readbacks();
void setReadbacks(bool enable, bool is_game_specific = false);
bool readbackLinearImages();
void setReadbackLinearImages(bool enable, bool is_game_specific = false);
bool directMemoryAccess();
void setDirectMemoryAccess(bool enable, bool is_game_specific = false);
bool dumpShaders();
void setDumpShaders(bool enable, bool is_game_specific = false);
u32 vblankFreq();
void setVblankFreq(u32 value, bool is_game_specific = false);
bool getisTrophyPopupDisabled();
void setisTrophyPopupDisabled(bool disable, bool is_game_specific = false);
s16 getCursorState();
void setCursorState(s16 cursorState, bool is_game_specific = false);
bool vkValidationEnabled();
void setVkValidation(bool enable, bool is_game_specific = false);
bool vkValidationSyncEnabled();
void setVkSyncValidation(bool enable, bool is_game_specific = false);
bool vkValidationGpuEnabled();
void setVkGpuValidation(bool enable, bool is_game_specific = false);
bool vkValidationCoreEnabled();
void setVkCoreValidation(bool enable, bool is_game_specific = false);
bool getVkCrashDiagnosticEnabled();
void setVkCrashDiagnosticEnabled(bool enable, bool is_game_specific = false);
bool getVkHostMarkersEnabled();
void setVkHostMarkersEnabled(bool enable, bool is_game_specific = false);
bool getVkGuestMarkersEnabled();
void setVkGuestMarkersEnabled(bool enable, bool is_game_specific = false);
bool vkForcePushDescriptorsEnabled();
void setVkForcePushDescriptors(bool enable, bool is_game_specific = false);
bool vkDisablePushDescriptorsEnabled();
void setVkDisablePushDescriptors(bool enable, bool is_game_specific = false);
bool isBeginRenderingCacheEnabled();
void setBeginRenderingCacheEnabled(bool enable, bool is_game_specific = false);
bool isPipelineUdHashLruEnabled();
void setPipelineUdHashLruEnabled(bool enable, bool is_game_specific = false);
bool accurateRenderTargetCacheEnabled();
void setAccurateRenderTargetCacheEnabled(bool enable, bool is_game_specific = false);
bool accurateVertexBufferCacheEnabled();
void setAccurateVertexBufferCacheEnabled(bool enable, bool is_game_specific = false);
// GR2FORK: enforce the two GR-Remastered-only cache shims as a single policy:
// ON for Remastered, hard-forced OFF for every other title. Pass the
// authoritative Remastered-SKU result from emulator.cpp; call once at startup
// after per-game config load.
void setGameSpecificCacheToggles(bool is_gr_remastered);
bool getEnableDiscordRPC();
void setEnableDiscordRPC(bool enable);
bool isRdocEnabled();
bool isPipelineCacheEnabled();
bool isPipelineCacheArchived();
void setRdocEnabled(bool enable, bool is_game_specific = false);
void setPipelineCacheEnabled(bool enable, bool is_game_specific = false);
void setPipelineCacheArchived(bool enable, bool is_game_specific = false);
std::string getLogType();
void setLogType(const std::string& type, bool is_game_specific = false);
std::string getLogFilter();
void setLogFilter(const std::string& type, bool is_game_specific = false);
double getTrophyNotificationDuration();
void setTrophyNotificationDuration(double newTrophyNotificationDuration,
                                   bool is_game_specific = false);
int getCursorHideTimeout();
std::string getMainOutputDevice();
void setMainOutputDevice(std::string device, bool is_game_specific = false);
std::string getPadSpkOutputDevice();
void setPadSpkOutputDevice(std::string device, bool is_game_specific = false);
std::string getMicDevice();
void setCursorHideTimeout(int newcursorHideTimeout, bool is_game_specific = false);
void setMicDevice(std::string device, bool is_game_specific = false);
void setSeparateLogFilesEnabled(bool enabled, bool is_game_specific = false);
bool getSeparateLogFilesEnabled();
u32 GetLanguage();
void setLanguage(u32 language, bool is_game_specific = false);
void setUseSpecialPad(bool use);
bool getUseSpecialPad();
void setSpecialPadClass(int type);
int getSpecialPadClass();
bool getPSNSignedIn();
void setPSNSignedIn(bool sign, bool is_game_specific = false);
bool patchShaders(); // no set
bool fpsColor();     // no set
bool getShowFpsCounter();
void setShowFpsCounter(bool enable, bool is_game_specific = false);
bool isNeoModeConsole();
void setNeoMode(bool enable, bool is_game_specific = false);
bool isDevKitConsole();
void setDevKitConsole(bool enable, bool is_game_specific = false);

int getExtraDmemInMbytes();
void setExtraDmemInMbytes(int value, bool is_game_specific = false);
bool getIsMotionControlsEnabled();
void setIsMotionControlsEnabled(bool use, bool is_game_specific = false);
std::string getDefaultControllerID();
void setDefaultControllerID(std::string id);
bool getBackgroundControllerInput();
void setBackgroundControllerInput(bool enable, bool is_game_specific = false);
bool getLoggingEnabled();
void setLoggingEnabled(bool enable, bool is_game_specific = false);
bool getFsrEnabled();
void setFsrEnabled(bool enable, bool is_game_specific = false);
bool getRcasEnabled();
void setRcasEnabled(bool enable, bool is_game_specific = false);
int getRcasAttenuation();
void setRcasAttenuation(int value, bool is_game_specific = false);
std::string getAspectRatioOverride();
void setAspectRatioOverride(const std::string& value, bool is_game_specific = false);

std::string getResolutionOverride();
void setResolutionOverride(const std::string& value, bool is_game_specific = false);

// GR2 (CUSA04943) resolution-patch group selector. Bisection knob for
// the resolution_patches.cpp pipeline. Default: "recommended" (the
// empirically-verified working set, currently A1+A2+A3+A4+B1+C1+D1+F1+
// F2+F3+H1+H2+H3+H4+M1 — M1 is target-resolution-gated to ≥2160p).
//
// Grammar: comma- or pipe-separated tokens with optional ~ / ! negation.
// Recognized tokens:
//   preset names      "recommended" | "prod" | "safe" | "min" | "geom" |
//                     "ext" | "extended" | "all" | "default" | "none" |
//                     "off" | "baseline" | "v3" | "add_h1" | "add_h2" |
//                     "add_g1" | "add_all_h" | "without_g1"
//   individual groups "a1" .. "m1" (case-insensitive)
//
// Examples:
//   "recommended"            (default — M1 included at 2160p)
//   "recommended,~m1"        (everything except M1 — baseline reproducer
//                            for the 4K Display2DThin swap-chain bug)
//   "m1"                     (M1 alone, no other resolution patches)
//   "b1,d1,h1,h2,m1"         (minimal 4K-target set)
//   "all,~e1,~g1,~k1"        (everything minus the known UI-corruptors
//                            and the opt-in exposure scaler)
//
// Empty string also defaults to "recommended". See resolution_patches.cpp
// ParseGroupMaskFromConfig for the full grammar.
std::string getResolutionPatchGroups();
void setResolutionPatchGroups(const std::string& value, bool is_game_specific = false);

bool getIsConnectedToNetwork();
void setConnectedToNetwork(bool enable, bool is_game_specific = false);
void setUserName(const std::string& name, bool is_game_specific = false);
std::filesystem::path getSysModulesPath();
void setSysModulesPath(const std::filesystem::path& path);
bool getLoadAutoPatches();
void setLoadAutoPatches(bool enable);

enum UsbBackendType : int { Real, SkylandersPortal, InfinityBase, DimensionsToypad };
int getUsbDeviceBackend();
void setUsbDeviceBackend(int value, bool is_game_specific = false);

// TODO
std::filesystem::path GetSaveDataPath();
std::string getUserName();
bool GetUseUnifiedInputConfig();
void SetUseUnifiedInputConfig(bool use);
bool GetOverrideControllerColor();
void SetOverrideControllerColor(bool enable);
int* GetControllerCustomColor();
void SetControllerCustomColor(int r, int b, int g);
void setGameInstallDirs(const std::vector<std::filesystem::path>& dirs_config);
void setAllGameInstallDirs(const std::vector<GameInstallDir>& dirs_config);
void setSaveDataPath(const std::filesystem::path& path);
// Gui
bool addGameInstallDir(const std::filesystem::path& dir, bool enabled = true);
void removeGameInstallDir(const std::filesystem::path& dir);
void setGameInstallDirEnabled(const std::filesystem::path& dir, bool enabled);
void setAddonInstallDir(const std::filesystem::path& dir);

const std::vector<std::filesystem::path> getGameInstallDirs();
const std::vector<bool> getGameInstallDirsEnabled();
std::filesystem::path getAddonInstallDir();

void setDefaultValues(bool is_game_specific = false);

constexpr std::string_view GetDefaultGlobalConfig();
std::filesystem::path GetFoolproofInputConfigFile(const std::string& game_id = "");

}; // namespace Config
