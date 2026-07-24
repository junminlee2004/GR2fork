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

// GR2FORK: 3-state GPU readback mode, binary-compatible with the prerelease's GPU.readbacks_mode
// json key (u32) so the value round-trips losslessly. Consume sites still use the readbacks()
// bool getter (any nonzero mode means readbacks on).
// TODO: branch Precise vs Relaxed at the consume sites.
enum GpuReadbacksMode : u32 { Disabled, Relaxed, Precise };

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
bool deviceRecovery();
void setNullGpu(bool enable, bool is_game_specific = false);
bool copyGPUCmdBuffers();
void setCopyGPUCmdBuffers(bool enable, bool is_game_specific = false);
bool readbacks();
void setReadbacks(bool enable, bool is_game_specific = false);
// GR2FORK: 3-state readback accessor matching the prerelease GPU.readbacks_mode
// (0 Disabled / 1 Relaxed / 2 Precise). readbacks() above returns (mode != 0).
u32 readbacksMode();
void setReadbacksMode(u32 mode, bool is_game_specific = false);
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
// GpuAsse Phase D - D3': suppress dynamic_dirty_ on user_data-only SH writes.
bool isShDynamicDirtySkipEnabled();
void setShDynamicDirtySkipEnabled(bool enable, bool is_game_specific = false);
bool isPipelineUdHashLruEnabled();
void setPipelineUdHashLruEnabled(bool enable, bool is_game_specific = false);
// GR2FORK GpuAsse: spec-fingerprint -> permutation LRU (see vk_pipeline_cache).
bool isPipelineSpecFpLruEnabled();
void setPipelineSpecFpLruEnabled(bool enable, bool is_game_specific = false);
// GR2FORK GpuAsse: skip the RefreshGraphicsKey context rebuild on user_data-only
// draws (see vk_pipeline_cache GetGraphicsPipeline level 2.5).
bool isPipelineGfxKeyCtxSkipEnabled();
void setPipelineGfxKeyCtxSkipEnabled(bool enable, bool is_game_specific = false);
// GpuAsse: extend the per-draw binding-skip fast path to the descriptor-set path (the
// push-descriptor path is dead on RADV): a bit-identical consecutive same-pipeline draw keeps
// the bound set and skips the set_writes rebuild. Default ON; ~0.03% hit on GR2/CUSA04943.
bool isDescSetBindingSkipCacheEnabled();
void setDescSetBindingSkipCacheEnabled(bool enable, bool is_game_specific = false);
// GR2FORK: the sole config-controlled GR2 toggle. The skybox cubemap fix causes GCN-era
// (Polaris) graphical regressions, so it keeps a persisted [GR2Fork] key (default true) plus a
// GR2_NOCUBEVIEW=1 env kill; every other GR2 fix is forced on with only per-site GR2_NO* kills.
bool gr2FixNativeCubeViews(); // skybox seam fix: native cubemap image views
void setGr2FixNativeCubeViews(bool enable, bool is_game_specific = false);
// GR2FORK: replace the title-screen theme with a user audio file from the "mods" folder
// (see ajm/title_theme_mod.h). Default false.
bool getGR2TitleThemeMod();
void setGR2TitleThemeMod(bool enable, bool is_game_specific = false);
// GR2FORK: user toggle to remove GR2's fullscreen motion-blur pass. When true the
// rasterizer drops the motion-blur fragment shader (hash 0xf696fe23) draw. Default
// false (motion blur on). Persisted [GR2Fork] disableMotionBlur key.
bool disableMotionBlur();
void setDisableMotionBlur(bool enable, bool is_game_specific = false);
// GR2FORK: mute the pad/controller speaker output (PadSpk). Default true. [GR2Fork] padSpkOutputDisabled.
bool isPadSpkOutputDisabled();
void setPadSpkOutputDisabled(bool disabled, bool is_game_specific = false);
bool accurateRenderTargetCacheEnabled();
void setAccurateRenderTargetCacheEnabled(bool enable, bool is_game_specific = false);
bool accurateVertexBufferCacheEnabled();
void setAccurateVertexBufferCacheEnabled(bool enable, bool is_game_specific = false);
// GR2FORK: enforce the two GR-Remastered-only cache shims as one policy - ON for Remastered,
// hard-forced OFF for every other title. Takes the authoritative Remastered-SKU result from
// emulator.cpp; called once at startup after per-game config load.
void setGameSpecificCacheToggles(bool force_rt_cache, bool force_vb_cache);
// GR2FORK: per-session flag for a Gravity Rush Remastered title. Set once at startup from
// emulator.cpp's SKU detection; read by GPU-side code that sizes structures differently for
// GRR. Not persisted to TOML.
bool isGravityRushRemastered();
void setIsGravityRushRemastered(bool is_gr_remastered);
// GR2FORK: per-session flag for an inFAMOUS Second Son title, set once at startup from
// emulator.cpp's SKU detection. Read by BundleAssembler::ActiveQueueSize to select the deeper
// 4096-slot intent queue ISS needs (measured peak ~1822). Not persisted to TOML.
bool isInfamousSecondSon();
void setIsInfamousSecondSon(bool is_infamous_second_son);
// GR2FORK: per-session flag forcing the monolithic GpuComm path, where the PM4 parser records
// each draw inline instead of handing a DrawIntent to a GpuAssembler jthread. Auto-forced ON
// (overriding all config) on a 4-physical-core GRR host - see emulator.cpp - because the
// two-thread split would reserve 2 of the 4 cores and starve the guest. Read by BundleAssembler
// and Common::DecideReservedCores. Not persisted to TOML.
bool isLegacyMonolithicGpuComm();
void setLegacyMonolithicGpuComm(bool enable);
// GR2FORK: non-exclusive GpuComm/GpuAssembler pinning - the two hot threads keep their core
// pins but the cores are not isolated, so the scheduler may place other threads on them when
// idle. Auto-forced on a 4-physical-core GRR session (see emulator.cpp); read by
// Common::ExcludeReservedCoresFromAllOtherThreads and Common::GetHotCoreFenceMask. Mutually
// exclusive with the monolithic path above. Not persisted to TOML.
bool isGpuCoresNonExclusive();
void setGpuCoresNonExclusive(bool enable);
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
// gr2fork: handheld gyro orientation fix. A device held upright (Steam Deck, ROG Ally) instead
// of flat like a DualShock has its physical yaw and roll gyro axes effectively swapped; this
// swaps them back in the motion data. Auto-enabled by the launcher's Steam Deck preset.
bool getGyroSwapYawRoll();
void setGyroSwapYawRoll(bool enable, bool is_game_specific = false);
// gr2fork: optional gyro yaw inversion - negates the resolved yaw channel so motion aiming
// turns the opposite way; independent of (and composes with) the yaw/roll swap above.
// Auto-enabled by the launcher's Steam Deck preset.
bool getGyroInvertYaw();
void setGyroInvertYaw(bool enable, bool is_game_specific = false);
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
// GR2FORK: pipeline-compile synchronous-wait budget in milliseconds - on a new graphics
// pipeline the GpuComm thread blocks up to this long for the async compile before skipping the
// draw. Default 50, [GPU] section, Qt slider 0..50000 ms; see vk_pipeline_cache for the wait
// site and why it must not be set near zero.
int getGameplaySyncBudgetMs();
void setGameplaySyncBudgetMs(int value, bool is_game_specific = false);
std::string getAspectRatioOverride();
void setAspectRatioOverride(const std::string& value, bool is_game_specific = false);

std::string getResolutionOverride();
void setResolutionOverride(const std::string& value, bool is_game_specific = false);

// GR2 (CUSA04943) resolution-patch group selector, a bisection knob for resolution_patches.cpp.
// Comma- or pipe-separated preset/group tokens with optional ~/! negation, e.g. "recommended"
// (the default, also used for an empty string), "b1,d1,h1,h2,m1", "all,~e1,~g1,~k1"; see
// ParseGroupMaskFromConfig in resolution_patches.cpp for the full grammar.
std::string getResolutionPatchGroups();
void setResolutionPatchGroups(const std::string& value, bool is_game_specific = false);

// GRR (CUSA01130 family) resolution-patch group selector, independent of the GR2 selector
// above; only the selector for the path that actually runs has any effect. Default
// "recommended" = R1 (UI RT) only; scene-RT groups S1..S4 are opt-in.
std::string getResolutionPatchGroupsGrr();
void setResolutionPatchGroupsGrr(const std::string& value, bool is_game_specific = false);

// GR2FORK shader-hash GPU-work skip: comma-separated selectors "0x<hash>" (skip all
// draws/dispatches whose pgm_hash matches) or "0x<hash>@<W>x<H>" (draws only, at that render
// extent). Hashes match the shader-dump filenames (fs_0x<hash> / cs_0x<hash>); empty = disabled.
std::string getGr2SkippedShaders();
void setGr2SkippedShaders(const std::string& value, bool is_game_specific = false);

// Fast gate + lookup used by the rasterizer per draw/dispatch: gr2ShaderSkipActive is a cheap
// atomic (false when the list is empty, so zero per-draw cost); when true, gr2IsShaderSkipped
// does the hash/extent lookup. Pass the render extent for draws, or 0/0 for compute dispatches.
bool gr2ShaderSkipActive();
bool gr2IsShaderSkipped(u64 hash, u32 width, u32 height);

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
// GR2FORK: PS4 "home" root, binary-compatible with the prerelease General.home_dir json key.
// Savedata lives at GetHomeDir()/<uid>/savedata/<serial>; GetSaveDataPath()/setSaveDataPath()
// remain as thin compatibility shims over the same home_dir value for the launcher/GUI.
std::filesystem::path GetHomeDir();
void setHomeDir(const std::filesystem::path& path);
std::string getUserName();
// GR2FORK (online restoration): host that all guest sceHttp URLs are rewritten to
// (host swapped + scheme forced to http) so GR2's discontinued online traffic is
// redirected to a local/self-hosted server. Default "localhost". See network/http.cpp.
std::string GetHttpHostOverride();
int GetHttpHostOverridePort();
bool GetHttpForceHttp();
// GR2FORK (online restoration): shadnet login credentials (npid = Online ID). Forwarded to the
// restoration server as the verified identity; empty npid disables auth.
std::string GetShadnetNpid();
std::string GetShadnetPassword();
std::string GetShadnetServer();
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
