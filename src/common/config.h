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

// GR2 (CUSA04943) motion-blur disable toggle. When true, the motion-blur
// post-process pass is skipped at game load by NOPing the two pass-
// registration calls in the render-graph builder. Default: false.
bool getDisableMotionBlur();
void setDisableMotionBlur(bool value, bool is_game_specific = false);

// =============================================================================
// GR2 granular render-pass disable toggles.
//
// GR2's renderer dispatches every named pass through a uniform per-frame
// template; each toggle below independently neutralises one pass (or two, for
// passes with both low-watermark and high-watermark dispatch sites) by
// rewriting its skip-flag load to `mov al, 1` so the immediately following
// `test al, al; jne SKIP` always branches past the pass body.
//
// All default false. Each toggle is independent — combine freely. Critical
// passes (Tonemap, Composite, Resolve, LightPass, opaque/Lambert, skypass,
// G2_BG_*, etc.) are intentionally not exposed because disabling them would
// black-screen the game.
// =============================================================================

// --- Shadow group ---
// ShadowCast0/1/2 are the three cascade shadow-map renders (near/mid/far).
// The far cascade (Cast2) typically carries the most shadow-caster draws
// because its frustum covers the most world space, so disableShadowCast2
// alone is the highest-payoff/lowest-visible-impact starting toggle.
// ShadowTrace is the screen-space pass that samples the cascades into the
// shadow mask consumed by opaque shading; disable it together with all
// three casts for a clean shadow-free look.
bool getDisableShadowCast0();
void setDisableShadowCast0(bool value, bool is_game_specific = false);
bool getDisableShadowCast1();
void setDisableShadowCast1(bool value, bool is_game_specific = false);
bool getDisableShadowCast2();
void setDisableShadowCast2(bool value, bool is_game_specific = false);
bool getDisableShadowTrace();
void setDisableShadowTrace(bool value, bool is_game_specific = false);

// --- Lighting / shading group ---
// SSAO: screen-space ambient occlusion. Soft contact darkening; cheap to lose
// visually, modest per-pixel cost saved.
// IBL: image-based lighting. Diffuse/specular environment lighting; the world
// gets a flatter look without it.
// Contour: cel-shading outlines. GR2's signature comic-book line work runs
// here — disabling makes characters look "modern PBR" rather than cel-shaded.
// Visually drastic; only enable if you don't care about the art style.
// SelfTranslucent: character self-translucency (subsurface scattering on
// skin/cloth). Modest fragment cost; characters look slightly more opaque.
bool getDisableSSAO();
void setDisableSSAO(bool value, bool is_game_specific = false);
bool getDisableIBL();
void setDisableIBL(bool value, bool is_game_specific = false);
bool getDisableContour();
void setDisableContour(bool value, bool is_game_specific = false);
bool getDisableSelfTranslucent();
void setDisableSelfTranslucent(bool value, bool is_game_specific = false);

// --- Post-process group ---
// Bloom: bright-pixel glow/halation post-process pyramid.
// Antialias: post-process AA (FXAA-ish); edges get crawlier without it.
// Effector: impact / screen-distortion effects (hit flashes, gravity-slip
// rings, etc.). Tied to gameplay feedback — disable with care.
bool getDisableBloom();
void setDisableBloom(bool value, bool is_game_specific = false);
bool getDisableAntialias();
void setDisableAntialias(bool value, bool is_game_specific = false);
bool getDisableEffector();
void setDisableEffector(bool value, bool is_game_specific = false);

// --- Fog group ---
// FogRender: main per-pixel fog application pass.
// FogCloud: cloud-layer fog contribution.
// FogDist: distance / global-distance fog.
// FogCast: volumetric fog "casting" (god-ray-ish) pass.
// On the floating cities, fog stacks add up; disabling individual fog
// passes can make the world look "clean" but lose atmosphere.
bool getDisableFogRender();
void setDisableFogRender(bool value, bool is_game_specific = false);
bool getDisableFogCloud();
void setDisableFogCloud(bool value, bool is_game_specific = false);
bool getDisableFogDist();
void setDisableFogDist(bool value, bool is_game_specific = false);
bool getDisableFogCast();
void setDisableFogCast(bool value, bool is_game_specific = false);

// --- Particle group ---
// ParticleCompute: particle simulation kernel. Disabling stops particle
// motion / spawning entirely — effectively "no particles" mode. Huge
// savings if a scene is particle-heavy, but VFX cues vanish.
// ParticleDistortion: heat-haze / refractive distortion particles (patches
// both the LW and HW dispatch sites). Particularly expensive because each
// dispatch forces a backup-color-buffer copy; cheapest particles to lose.
bool getDisableParticleCompute();
void setDisableParticleCompute(bool value, bool is_game_specific = false);
bool getDisableParticleDistortion();
void setDisableParticleDistortion(bool value, bool is_game_specific = false);
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
