// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <fmt/core.h>
#include <fmt/xchar.h> // for wstring support
#include <nlohmann/json.hpp>
#include <toml.hpp>

#include "common/assert.h"
#include "common/config.h"
#include "common/logging/formatter.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/scm_rev.h"

using std::nullopt;
using std::optional;
using std::string;

namespace toml {
template <typename TC, typename K>
std::filesystem::path find_fs_path_or(const basic_value<TC>& v, const K& ky,
                                      std::filesystem::path opt) {
    try {
        auto str = find<string>(v, ky);
        if (str.empty()) {
            return opt;
        }
        std::u8string u8str{(char8_t*)&str.front(), (char8_t*)&str.back() + 1};
        return std::filesystem::path{u8str};
    } catch (...) {
        return opt;
    }
}

// why is it so hard to avoid exceptions with this library
template <typename T>
std::optional<T> get_optional(const toml::value& v, const std::string& key) {
    if (!v.is_table())
        return std::nullopt;
    const auto& tbl = v.as_table();
    auto it = tbl.find(key);
    if (it == tbl.end())
        return std::nullopt;

    if constexpr (std::is_same_v<T, int>) {
        if (it->second.is_integer()) {
            return static_cast<int>(toml::get<int>(it->second));
        }
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        if (it->second.is_integer()) {
            return static_cast<u32>(toml::get<unsigned int>(it->second));
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (it->second.is_floating()) {
            return toml::get<double>(it->second);
        }
    } else if constexpr (std::is_same_v<T, std::string>) {
        if (it->second.is_string()) {
            return toml::get<std::string>(it->second);
        }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (it->second.is_boolean()) {
            return toml::get<bool>(it->second);
        }
    } else {
        static_assert([] { return false; }(), "Unsupported type in get_optional<T>");
    }

    return std::nullopt;
}

} // namespace toml

// GR2FORK: serialize std::filesystem::path as a UTF-8 json string, byte-identical to the
// prerelease's serializer (core/emulator_settings.cpp), so paths written by either build are
// read back correctly by the other.
namespace nlohmann {
template <>
struct adl_serializer<std::filesystem::path> {
    static void to_json(json& j, const std::filesystem::path& p) {
        const auto u8 = p.u8string();
        j = std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
    }
    static void from_json(const json& j, std::filesystem::path& p) {
        const std::string s = j.get<std::string>();
        p = std::filesystem::path(
            std::u8string_view(reinterpret_cast<const char8_t*>(s.data()), s.size()));
    }
};
} // namespace nlohmann

namespace Config {

using json = nlohmann::json;

ConfigMode config_mode = ConfigMode::Default;

void setConfigMode(ConfigMode mode) {
    config_mode = mode;
}

template <typename T>
class ConfigEntry {
public:
    const T default_value;
    T base_value;
    optional<T> game_specific_value;
    ConfigEntry(const T& t = T()) : default_value(t), base_value(t), game_specific_value(nullopt) {}
    ConfigEntry operator=(const T& t) {
        base_value = t;
        return *this;
    }
    // The resolution mode defaults to the global config_mode. Game-facing reads pass
    // ConfigMode::Default explicitly (see getUserName) so a per-title override always wins at
    // runtime, independent of whatever mode the settings UI last set.
    const T get(ConfigMode mode = config_mode) const {
        switch (mode) {
        case ConfigMode::Default:
            return game_specific_value.value_or(base_value);
        case ConfigMode::Global:
            return base_value;
        case ConfigMode::Clean:
            return default_value;
        default:
            UNREACHABLE();
        }
    }
    void setFromToml(const toml::value& v, const std::string& key, bool is_game_specific = false) {
        if (is_game_specific) {
            game_specific_value = toml::get_optional<T>(v, key);
        } else {
            base_value = toml::get_optional<T>(v, key).value_or(base_value);
        }
    }
    void set(const T value, bool is_game_specific = false) {
        is_game_specific ? game_specific_value = value : base_value = value;
    }
    void setTomlValue(toml::ordered_value& data, const std::string& header, const std::string& key,
                      bool is_game_specific = false) {
        if (is_game_specific) {
            data[header][key] = game_specific_value.value_or(base_value);
            game_specific_value = std::nullopt;
        } else {
            data[header][key] = base_value;
        }
    }
    // GR2FORK: json mirrors of setFromToml/setTomlValue. A missing key or a type mismatch
    // leaves the current value untouched, matching the toml::get_optional semantics above.
    void setFromJson(const nlohmann::json& section, const std::string& key,
                     bool is_game_specific = false) {
        if (!section.is_object() || !section.contains(key)) {
            return;
        }
        try {
            T value = section.at(key).get<T>();
            if (is_game_specific) {
                game_specific_value = value;
            } else {
                base_value = value;
            }
        } catch (const std::exception&) {
            // Wrong json type for this key - keep the existing value.
        }
    }
    void setJsonValue(nlohmann::json& data, const std::string& header, const std::string& key,
                      bool is_game_specific = false) {
        if (is_game_specific) {
            data[header][key] = game_specific_value.value_or(base_value);
            game_specific_value = std::nullopt;
        } else {
            data[header][key] = base_value;
        }
    }
    // operator T() {
    //     return get();
    // }
};

// General
static ConfigEntry<int> volumeSlider(100);
static ConfigEntry<bool> isNeo(false);
static ConfigEntry<bool> isDevKit(false);
// GR2FORK: Windows static guest red-zone protection (upstream shadPS4 #4802). Windows exception
// dispatch writes into the 128-byte SysV red zone below the guest's RSP, corrupting leaf-function
// stacks on every tracked page fault. Static patching relocates the affected guest memory accesses
// so the dispatch cannot land in a live red zone. Windows-only at runtime; the setting is stored
// on every platform so configs stay portable.
static ConfigEntry<bool> windowsGuestRedZoneProtection(false);
static ConfigEntry<int> extraDmemInMbytes(0);
static ConfigEntry<bool> isPSNSignedIn(false);
static ConfigEntry<bool> isTrophyPopupDisabled(false);
static ConfigEntry<double> trophyNotificationDuration(6.0);
static ConfigEntry<string> logFilter("");
static ConfigEntry<string> logType("sync");
static ConfigEntry<string> userName("shadPS4");
// GR2FORK (online restoration): redirect target for all guest sceHttp requests.
static ConfigEntry<string> httpHostOverride("localhost");
// httpHostOverridePort: 8443 default so the bundled local server needs no elevation.
// httpForceHttp: downgrade scheme to http (true = local no-cert server; set false to keep
// https when pointing the override at a public TLS server).
static ConfigEntry<int> httpHostOverridePort(8443);
static ConfigEntry<bool> httpForceHttp(true);
// GR2FORK (online restoration): shadnet login credentials. npid is the Online ID registered at the
// shadnet account server; the verified identity and bearer token are forwarded to the restoration
// server. Empty npid disables auth.
static ConfigEntry<string> shadnetNpid("");
static ConfigEntry<string> shadnetPassword("");
static ConfigEntry<string> shadnetServer("srv.shadps4.net:31313");
static ConfigEntry<bool> isShowSplash(false);
static ConfigEntry<string> isSideTrophy("right");
static ConfigEntry<bool> isConnectedToNetwork(false);
static bool enableDiscordRPC = false;
static std::filesystem::path sys_modules_path = {};

// Input
static ConfigEntry<int> cursorState(HideCursorState::Idle);
static ConfigEntry<int> cursorHideTimeout(5); // 5 seconds (default)
static ConfigEntry<bool> useSpecialPad(false);
static ConfigEntry<int> specialPadClass(1);
static ConfigEntry<bool> isMotionControlsEnabled(true);
static ConfigEntry<bool> gyroSwapYawRoll(false);
static ConfigEntry<bool> gyroInvertYaw(false);
static ConfigEntry<bool> gyroInvertX(false);
static ConfigEntry<bool> gyroInvertRoll(false);
static ConfigEntry<bool> useUnifiedInputConfig(true);
static ConfigEntry<string> defaultControllerID("");
static ConfigEntry<bool> backgroundControllerInput(false);

// Audio
static ConfigEntry<string> micDevice("Default Device");
static ConfigEntry<string> mainOutputDevice("Default Device");
static ConfigEntry<string> padSpkOutputDevice("Default Device");

// GPU
static ConfigEntry<u32> windowWidth(1280);
static ConfigEntry<u32> windowHeight(720);
static ConfigEntry<u32> internalScreenWidth(1280);
static ConfigEntry<u32> internalScreenHeight(720);
static ConfigEntry<bool> isNullGpu(false);
static ConfigEntry<bool> isDeviceRecovery(false);
static ConfigEntry<bool> shouldCopyGPUBuffers(false);
// GR2FORK: 3-state readback mode (was bool readbacksEnabled). 0 Disabled /
// 1 Relaxed / 2 Precise - binary-compatible with the prerelease GPU.readbacks_mode.
static ConfigEntry<u32> readbacksModeEntry(GpuReadbacksMode::Disabled);
static ConfigEntry<bool> readbackLinearImagesEnabled(false);
static ConfigEntry<bool> directMemoryAccessEnabled(false);
static ConfigEntry<bool> shouldDumpShaders(false);
static ConfigEntry<bool> shouldPatchShaders(false);
static ConfigEntry<u32> vblankFrequency(60);
static ConfigEntry<bool> isFullscreen(false);
static ConfigEntry<string> fullscreenMode("Windowed");
static ConfigEntry<string> presentMode("Mailbox");
static ConfigEntry<bool> isHDRAllowed(false);
static ConfigEntry<bool> fsrEnabled(false);
static ConfigEntry<bool> rcasEnabled(true);
static ConfigEntry<int> rcasAttenuation(250);
// GR2FORK: pipeline-compile sync-wait budget in ms (see config.h). Default
// 100; user-tunable from the Qt launcher.
static ConfigEntry<int> gameplaySyncBudgetMs(100);
static ConfigEntry<string> aspectRatioOverride("16:9");
static ConfigEntry<string> resolutionOverride("Off");
static ConfigEntry<string> resolutionPatchGroups("recommended");
static ConfigEntry<string> resolutionPatchGroupsGrr("recommended");
static ConfigEntry<string> gr2SkippedShaders("");

// Vulkan
static ConfigEntry<s32> gpuId(-1);
static ConfigEntry<bool> vkValidation(false);
static ConfigEntry<bool> vkValidationCore(true);
static ConfigEntry<bool> vkValidationSync(false);
static ConfigEntry<bool> vkValidationGpu(false);
static ConfigEntry<bool> vkCrashDiagnostic(false);
static ConfigEntry<bool> vkHostMarkers(false);
static ConfigEntry<bool> vkGuestMarkers(false);
static ConfigEntry<bool> rdocEnable(false);
static ConfigEntry<bool> pipelineCacheEnable(false);
static ConfigEntry<bool> pipelineCacheArchive(false);
static ConfigEntry<bool> vkForcePushDescriptors(false);
static ConfigEntry<bool> vkDisablePushDescriptors(false);
// GR2FORK: stamp-skip cache in Rasterizer::BeginRendering, caching the eLoad-forced RenderState
// keyed on {pipeline, gfx_pipeline_stamp, tick}. Default ON; false if shadow flicker or other
// rendering artifacts appear (e.g. register-driven depth_clear_enable edge cases).
static ConfigEntry<bool> beginRenderingCacheEnable(true);
// GR2FORK: suppress dynamic_dirty_ on user_data-only SH writes in Liverpool::SetShReg;
// UpdateDynamicState's subs read only context regs, which user_data cannot affect (GR2: ~88% of
// key-dirty SH writes are user_data-only, dirty-rate 100% -> ~32%). Default ON.
static ConfigEntry<bool> shDynamicDirtySkip(true);
// GR2FORK: 32-slot LRU on Program (vk_pipeline_cache) keyed on ud_hash, skipping the
// StageSpecialization constructor when a permutation was already resolved for the hash.
// Default ON; false if warmup flicker or stale-permutation artifacts appear.
static ConfigEntry<bool> pipelineUdHashLruEnable(true);
// GR2FORK: spec-fingerprint -> permutation LRU. Resolves a permutation without constructing
// StageSpecialization when only resource addresses changed between draws (per-draw UBO-pointer
// SGPR churn ud_hash cannot catch). Address-independent key; ~97% reclaim on GR2. Default ON.
static ConfigEntry<bool> pipelineSpecFpLruEnable(true);
// GR2FORK: skip RefreshGraphicsKey's context-register rebuild on draws whose only change was a
// user_data-only SH re-emit. Modules are still re-resolved and the full key memcmp guards
// reuse; ~99.7% skip on GR2. Default ON.
static ConfigEntry<bool> pipelineGfxKeyCtxSkipEnable(true);
// GR2FORK: extend the BindResources binding-skip fast path to the descriptor-set path (the
// push-descriptor-only gate is statically false on RADV); an empty set_writes keeps the prior
// identical draw's set bound. Default ON; GR2 (CUSA04943) hit rate ~0.03% over 9.4M calls.
static ConfigEntry<bool> descSetBindingSkipCache(true);
// GR2FORK: gates rt_cache_ in vk_rasterizer.cpp PrepareRenderState. The (address+extent) hash
// misses TextureCache image-recreate at the same VAddr -> stale image_id -> shadow flicker.
// Default false (cache active); hard-forced true on Gravity Rush Remastered SKUs at startup.
static ConfigEntry<bool> accurateRenderTargetCache(false);
// GR2FORK: gates the cmdbuf-rotation-aware vertex bind path in
// BufferCache::BindVertexBuffers. Default off (upstream behavior); force-enabled
// for Gravity Rush Remastered SKUs in emulator.cpp.
static ConfigEntry<bool> accurateVertexBufferCache(false);

// GR2FORK graphics-fix toggle ([GR2Fork] section; launcher "Debug" tab). The fps optimizations
// are forced on with env-only GR2_NO* kills; only the skybox cubemap fix keeps a persisted key,
// because it alone regresses GCN-era parts (Polaris RX 400/500). GR2_NOCUBEVIEW=1 still wins.
static ConfigEntry<bool> gr2FixNativeCubeViewsEnable(true);  // skybox seams GR2_NOCUBEVIEW
static ConfigEntry<bool> gr2TitleThemeMod(false);           // GR2FORK: title-theme replacement

// GR2FORK: user toggle to remove GR2's fullscreen motion-blur pass. When true, the
// rasterizer drops the motion-blur fragment shader (hash 0xf696fe23) draw.
// Persisted [GR2Fork] key, default false (motion blur on, matching the game).
static ConfigEntry<bool> disableMotionBlurEnable(false);
// GR2FORK: mute the DualShock/pad speaker output port (PadSpk). Default true (disabled): PC playback
// duplicates the controller-speaker audio through the main mix, so it stays off unless re-enabled.
static ConfigEntry<bool> padSpkOutputDisabled(true);

// GR2FORK: per-session title flags, not persisted - recorded once at startup from emulator.cpp's
// SKU detection and read by GPU-side code that sizes structures per title (snapshot pool /
// intent queue). Set before any GPU object is constructed, so a plain bool read is safe.
static bool g_is_gr_remastered = false;
static bool g_is_infamous_second_son = false;

// GR2FORK: per-session monolithic-GpuComm flag, not persisted - computed once at startup in
// emulator.cpp (4 physical cores && GRR). When true, BundleAssembler dispatches inline with no
// GpuAssembler jthread and DecideReservedCores skips the assembler core reservation.
static bool g_legacy_monolithic_gpucomm = false;

// GR2FORK: when true, GpuComm/GpuAssembler keep their pins but the cores are not isolated from
// other threads. Auto-forced on 4-physical-core hosts running GRR (see emulator.cpp) to avoid
// compile-burst core starvation; mutually exclusive with g_legacy_monolithic_gpucomm.
static bool g_gpu_cores_non_exclusive = false;

// Debug
static ConfigEntry<bool> isDebugDump(false);
static ConfigEntry<bool> isShaderDebug(false);
static ConfigEntry<bool> isSeparateLogFilesEnabled(false);
static ConfigEntry<bool> isFpsColor(true);
static ConfigEntry<bool> showFpsCounter(false);
static ConfigEntry<bool> logEnabled(true);

// GUI
static std::vector<GameInstallDir> settings_install_dirs = {};
std::vector<bool> install_dirs_enabled = {};
std::filesystem::path settings_addon_install_dir = {};
// GR2FORK: renamed save_data_path -> home_dir (prerelease General.home_dir). Savedata
// lives under home_dir/<uid>/savedata/<serial>. GetSaveDataPath()/setSaveDataPath()
// remain as compat shims over this value.
std::filesystem::path home_dir = {};

// Settings
ConfigEntry<u32> m_language(1); // english

// USB Device
static ConfigEntry<int> usbDeviceBackend(UsbBackendType::Real);

// Keys
static string trophyKey = "";

// Config version, used to determine if a user's config file is outdated.
static string config_version = Common::g_scm_rev;

// These entries aren't stored in the config
static bool overrideControllerColor = false;
static int controllerCustomColorRGB[3] = {0, 0, 255};
static bool isGameRunning = false;
static bool load_auto_patches = true;

bool getGameRunning() {
    return isGameRunning;
}

void setGameRunning(bool running) {
    isGameRunning = running;
}

std::filesystem::path getSysModulesPath() {
    if (sys_modules_path.empty()) {
        return Common::FS::GetUserPath(Common::FS::PathType::SysModuleDir);
    }
    return sys_modules_path;
}

void setSysModulesPath(const std::filesystem::path& path) {
    sys_modules_path = path;
}

int getVolumeSlider() {
    return volumeSlider.get();
}
bool allowHDR() {
    return isHDRAllowed.get();
}

bool GetUseUnifiedInputConfig() {
    return useUnifiedInputConfig.get();
}

void SetUseUnifiedInputConfig(bool use) {
    useUnifiedInputConfig.base_value = use;
}

bool GetOverrideControllerColor() {
    return overrideControllerColor;
}

void SetOverrideControllerColor(bool enable) {
    overrideControllerColor = enable;
}

int* GetControllerCustomColor() {
    return controllerCustomColorRGB;
}

bool getLoggingEnabled() {
    return logEnabled.get();
}

void SetControllerCustomColor(int r, int b, int g) {
    controllerCustomColorRGB[0] = r;
    controllerCustomColorRGB[1] = b;
    controllerCustomColorRGB[2] = g;
}

string getTrophyKey() {
    return trophyKey;
}

void setTrophyKey(string key) {
    trophyKey = key;
}

std::filesystem::path GetHomeDir() {
    if (home_dir.empty()) {
        return Common::FS::GetUserPath(Common::FS::PathType::HomeDir);
    }
    return home_dir;
}

void setHomeDir(const std::filesystem::path& path) {
    home_dir = path;
}

std::filesystem::path GetSaveDataPath() {
    // GR2FORK: compat shim. Savedata lives under the PS4 home dir, matching the prerelease
    // layout (home_dir/<uid>/savedata/<serial>); the SaveInstance path builders append
    // "<uid>/savedata". Callers that want the savedata root should prefer GetHomeDir().
    return GetHomeDir();
}

void setVolumeSlider(int volumeValue, bool is_game_specific) {
    volumeSlider.set(volumeValue, is_game_specific);
}

bool isNeoModeConsole() {
    return isNeo.get();
}

bool isDevKitConsole() {
    return isDevKit.get();
}

bool getWindowsGuestRedZoneProtection() {
    return windowsGuestRedZoneProtection.get();
}

int getExtraDmemInMbytes() {
    return extraDmemInMbytes.get();
}

void setExtraDmemInMbytes(int value, bool is_game_specific) {
    // Disable setting in global config
    is_game_specific ? extraDmemInMbytes.game_specific_value = value
                     : extraDmemInMbytes.base_value = 0;
}

bool getIsFullscreen() {
    return isFullscreen.get();
}

string getFullscreenMode() {
    return fullscreenMode.get();
}

std::string getPresentMode() {
    return presentMode.get();
}

bool getisTrophyPopupDisabled() {
    return isTrophyPopupDisabled.get();
}

bool getEnableDiscordRPC() {
    return enableDiscordRPC;
}

s16 getCursorState() {
    return cursorState.get();
}

int getCursorHideTimeout() {
    return cursorHideTimeout.get();
}

string getMicDevice() {
    return micDevice.get();
}

std::string getMainOutputDevice() {
    return mainOutputDevice.get();
}

std::string getPadSpkOutputDevice() {
    return padSpkOutputDevice.get();
}

double getTrophyNotificationDuration() {
    return trophyNotificationDuration.get();
}

u32 getWindowWidth() {
    return windowWidth.get();
}

u32 getWindowHeight() {
    return windowHeight.get();
}

u32 getInternalScreenWidth() {
    // Deliberately returns the height (square display): returning
    // internalScreenWidth.get() regresses the area-transition /
    // AvPlayer-loop loading hang. Do not change without re-testing that hang.
    return internalScreenHeight.get();
}

u32 getInternalScreenHeight() {
    return internalScreenHeight.get();
}

s32 getGpuId() {
    return gpuId.get();
}

string getLogFilter() {
    return logFilter.get();
}

string getLogType() {
    return logType.get();
}

string getUserName() {
    // Read by the running game (sceNpGetOnlineId / sceNpGetNpId), so a per-title username
    // override must win even when the settings UI left config_mode in Global. Force Default;
    // the global value is still returned when no per-game override exists.
    return userName.get(ConfigMode::Default);
}

string GetHttpHostOverride() {
    return httpHostOverride.get();
}

int GetHttpHostOverridePort() {
    return httpHostOverridePort.get();
}

string GetShadnetNpid() {
    return shadnetNpid.get();
}

string GetShadnetPassword() {
    return shadnetPassword.get();
}

string GetShadnetServer() {
    return shadnetServer.get();
}

bool GetHttpForceHttp() {
    return httpForceHttp.get();
}

bool getUseSpecialPad() {
    return useSpecialPad.get();
}

int getSpecialPadClass() {
    return specialPadClass.get();
}

bool getIsMotionControlsEnabled() {
    return isMotionControlsEnabled.get();
}

bool getGyroSwapYawRoll() {
    return gyroSwapYawRoll.get();
}

bool getGyroInvertYaw() {
    return gyroInvertYaw.get();
}

bool getGyroInvertX() {
    return gyroInvertX.get();
}

bool getGyroInvertRoll() {
    return gyroInvertRoll.get();
}

bool debugDump() {
    return isDebugDump.get();
}

bool collectShadersForDebug() {
    return isShaderDebug.get();
}

bool showSplash() {
    return isShowSplash.get();
}

string sideTrophy() {
    return isSideTrophy.get();
}

bool nullGpu() {
    return isNullGpu.get();
}

// GR2FORK: opt-in VK_ERROR_DEVICE_LOST recovery (rebuild the renderer + device on a loss instead
// of the fatal assert). Default OFF - see vk_device_recovery.h. [GPU] deviceRecovery.
bool deviceRecovery() {
    return isDeviceRecovery.get();
}

bool copyGPUCmdBuffers() {
    return shouldCopyGPUBuffers.get();
}

bool readbacks() {
    return readbacksModeEntry.get() != GpuReadbacksMode::Disabled;
}

u32 readbacksMode() {
    return readbacksModeEntry.get();
}

bool readbackLinearImages() {
    return readbackLinearImagesEnabled.get();
}

bool directMemoryAccess() {
    return directMemoryAccessEnabled.get();
}

bool dumpShaders() {
    return shouldDumpShaders.get();
}

bool patchShaders() {
    return shouldPatchShaders.get();
}

bool isRdocEnabled() {
    return rdocEnable.get();
}

bool isPipelineCacheEnabled() {
    return pipelineCacheEnable.get();
}

bool isPipelineCacheArchived() {
    return pipelineCacheArchive.get();
}

bool fpsColor() {
    return isFpsColor.get();
}

bool getShowFpsCounter() {
    return showFpsCounter.get();
}

void setShowFpsCounter(bool enable, bool is_game_specific) {
    showFpsCounter.set(enable, is_game_specific);
}

bool isLoggingEnabled() {
    return logEnabled.get();
}

u32 vblankFreq() {
    if (vblankFrequency.get() < 60) {
        vblankFrequency = 60;
    }
    return vblankFrequency.get();
}

bool vkValidationEnabled() {
    return vkValidation.get();
}

bool vkValidationCoreEnabled() {
    return vkValidationCore.get();
}

bool vkValidationSyncEnabled() {
    return vkValidationSync.get();
}

bool vkValidationGpuEnabled() {
    return vkValidationGpu.get();
}

bool getVkCrashDiagnosticEnabled() {
    return vkCrashDiagnostic.get();
}

bool getVkHostMarkersEnabled() {
    return vkHostMarkers.get();
}

bool getVkGuestMarkersEnabled() {
    return vkGuestMarkers.get();
}

bool vkForcePushDescriptorsEnabled() {
    return vkForcePushDescriptors.get();
}

bool vkDisablePushDescriptorsEnabled() {
    return vkDisablePushDescriptors.get();
}

// GR2FORK: forced-on perf gates, immune to TOML contents - a persisted key that ever carried
// false would silently disable a shipped optimization (shDynamicDirtySkip off alone drops
// ctx_stable to 0 across 11.5M GetProgram calls). Env kills only, echoed at boot in emulator.cpp.
static bool ForcedOnUnlessEnv(const char* kill_env) noexcept {
    const char* e = std::getenv(kill_env);
    return !(e && e[0] == '1');
}

bool isBeginRenderingCacheEnabled() {
    static const bool enabled = ForcedOnUnlessEnv("GR2_NOBRCACHE");
    return enabled;
}

bool isShDynamicDirtySkipEnabled() {
    static const bool enabled = ForcedOnUnlessEnv("GR2_NOSHDYNSKIP");
    return enabled;
}

bool isPipelineUdHashLruEnabled() {
    static const bool enabled = ForcedOnUnlessEnv("GR2_NOUDHASHLRU");
    return enabled;
}

bool isPipelineSpecFpLruEnabled() {
    static const bool enabled = ForcedOnUnlessEnv("GR2_NOSPECFPLRU");
    return enabled;
}

bool isPipelineGfxKeyCtxSkipEnabled() {
    static const bool enabled = ForcedOnUnlessEnv("GR2_NOKEYCTXSKIP");
    return enabled;
}

bool isDescSetBindingSkipCacheEnabled() {
    static const bool enabled = ForcedOnUnlessEnv("GR2_NODESCSKIP");
    return enabled;
}

// GR2FORK: the sole config-controlled GR2 toggle - the skybox cubemap fix is the one GR2 change
// confirmed to regress GCN-era parts, so it keeps a persisted key ([GR2Fork]
// gr2FixNativeCubeViews, default true). Its GR2_NOCUBEVIEW=1 env kill latches in resource.h.
bool gr2FixNativeCubeViews() {
    return gr2FixNativeCubeViewsEnable.get();
}

// GR2FORK: setter for the sole config-controlled GR2 toggle (cubemap fix). Used
// by in-process settings surfaces; the external launcher edits the toml
// directly via its own Config.
void setGr2FixNativeCubeViews(bool enable, bool is_game_specific) {
    gr2FixNativeCubeViewsEnable.set(enable, is_game_specific);
}

// GR2FORK: gate for the title-screen-theme replacement (ajm/title_theme_mod.h). Default false.
bool getGR2TitleThemeMod() {
    return gr2TitleThemeMod.get();
}

void setGR2TitleThemeMod(bool enable, bool is_game_specific) {
    gr2TitleThemeMod.set(enable, is_game_specific);
}

// GR2FORK: motion-blur removal toggle. Read by the rasterizer per draw to drop
// GR2's fullscreen motion-blur pass (fragment hash 0xf696fe23).
bool disableMotionBlur() {
    return disableMotionBlurEnable.get();
}

void setDisableMotionBlur(bool enable, bool is_game_specific) {
    disableMotionBlurEnable.set(enable, is_game_specific);
}

bool isPadSpkOutputDisabled() {
    return padSpkOutputDisabled.get();
}

void setPadSpkOutputDisabled(bool disabled, bool is_game_specific) {
    padSpkOutputDisabled.set(disabled, is_game_specific);
}

bool accurateRenderTargetCacheEnabled() {
    return accurateRenderTargetCache.get();
}

bool accurateVertexBufferCacheEnabled() {
    return accurateVertexBufferCache.get();
}

void setVkCrashDiagnosticEnabled(bool enable, bool is_game_specific) {
    vkCrashDiagnostic.set(enable, is_game_specific);
}

void setVkHostMarkersEnabled(bool enable, bool is_game_specific) {
    vkHostMarkers.set(enable, is_game_specific);
}

void setVkGuestMarkersEnabled(bool enable, bool is_game_specific) {
    vkGuestMarkers.set(enable, is_game_specific);
}

void setVkForcePushDescriptors(bool enable, bool is_game_specific) {
    vkForcePushDescriptors.set(enable, is_game_specific);
}

void setVkDisablePushDescriptors(bool enable, bool is_game_specific) {
    vkDisablePushDescriptors.set(enable, is_game_specific);
}

void setBeginRenderingCacheEnabled(bool enable, bool is_game_specific) {
    beginRenderingCacheEnable.set(enable, is_game_specific);
}

void setShDynamicDirtySkipEnabled(bool enable, bool is_game_specific) {
    shDynamicDirtySkip.set(enable, is_game_specific);
}

void setPipelineUdHashLruEnabled(bool enable, bool is_game_specific) {
    pipelineUdHashLruEnable.set(enable, is_game_specific);
}

void setPipelineSpecFpLruEnabled(bool enable, bool is_game_specific) {
    pipelineSpecFpLruEnable.set(enable, is_game_specific);
}

void setPipelineGfxKeyCtxSkipEnabled(bool enable, bool is_game_specific) {
    pipelineGfxKeyCtxSkipEnable.set(enable, is_game_specific);
}

void setDescSetBindingSkipCacheEnabled(bool enable, bool is_game_specific) {
    descSetBindingSkipCache.set(enable, is_game_specific);
}

void setAccurateRenderTargetCacheEnabled(bool enable, bool is_game_specific) {
    accurateRenderTargetCache.set(enable, is_game_specific);
}

void setAccurateVertexBufferCacheEnabled(bool enable, bool is_game_specific) {
    accurateVertexBufferCache.set(enable, is_game_specific);
}

// GR2FORK: hard-force the per-title correctness shims from a single call site so a stray saved
// config value can never enable one under the wrong game. emulator.cpp owns the SKU detection
// and passes the per-title decisions in; called once at startup after the per-game config loads.
void setGameSpecificCacheToggles(bool force_rt_cache, bool force_vb_cache) {
    accurateRenderTargetCache.set(force_rt_cache, /*is_game_specific=*/true);
    accurateVertexBufferCache.set(force_vb_cache, /*is_game_specific=*/true);
    LOG_INFO(Config,
             "per-title cache shims forced: accurateRenderTargetCache={}, "
             "accurateVertexBufferCache={}",
             force_rt_cache ? "ON" : "OFF", force_vb_cache ? "ON" : "OFF");
}

bool isGravityRushRemastered() {
    return g_is_gr_remastered;
}

void setIsGravityRushRemastered(bool is_gr_remastered) {
    g_is_gr_remastered = is_gr_remastered;
}

bool isInfamousSecondSon() {
    return g_is_infamous_second_son;
}

void setIsInfamousSecondSon(bool is_infamous_second_son) {
    g_is_infamous_second_son = is_infamous_second_son;
}

bool isLegacyMonolithicGpuComm() {
    return g_legacy_monolithic_gpucomm;
}

void setLegacyMonolithicGpuComm(bool enable) {
    g_legacy_monolithic_gpucomm = enable;
}

bool isGpuCoresNonExclusive() {
    return g_gpu_cores_non_exclusive;
}

void setGpuCoresNonExclusive(bool enable) {
    g_gpu_cores_non_exclusive = enable;
}

bool getIsConnectedToNetwork() {
    return isConnectedToNetwork.get();
}

void setConnectedToNetwork(bool enable, bool is_game_specific) {
    isConnectedToNetwork.set(enable, is_game_specific);
}

void setGpuId(s32 selectedGpuId, bool is_game_specific) {
    gpuId.set(selectedGpuId, is_game_specific);
}

void setWindowWidth(u32 width, bool is_game_specific) {
    windowWidth.set(width, is_game_specific);
}

void setWindowHeight(u32 height, bool is_game_specific) {
    windowHeight.set(height, is_game_specific);
}

void setInternalScreenWidth(u32 width) {
    internalScreenWidth.base_value = width;
}

void setInternalScreenHeight(u32 height) {
    internalScreenHeight.base_value = height;
}

void setDebugDump(bool enable, bool is_game_specific) {
    isDebugDump.set(enable, is_game_specific);
}

void setLoggingEnabled(bool enable, bool is_game_specific) {
    logEnabled.set(enable, is_game_specific);
}

void setCollectShaderForDebug(bool enable, bool is_game_specific) {
    isShaderDebug.set(enable, is_game_specific);
}

void setShowSplash(bool enable, bool is_game_specific) {
    isShowSplash.set(enable, is_game_specific);
}

void setSideTrophy(string side, bool is_game_specific) {
    isSideTrophy.set(side, is_game_specific);
}

void setNullGpu(bool enable, bool is_game_specific) {
    isNullGpu.set(enable, is_game_specific);
}

void setAllowHDR(bool enable, bool is_game_specific) {
    isHDRAllowed.set(enable, is_game_specific);
}

void setCopyGPUCmdBuffers(bool enable, bool is_game_specific) {
    shouldCopyGPUBuffers.set(enable, is_game_specific);
}

void setReadbacks(bool enable, bool is_game_specific) {
    readbacksModeEntry.set(enable ? GpuReadbacksMode::Relaxed : GpuReadbacksMode::Disabled,
                           is_game_specific);
}

void setReadbacksMode(u32 mode, bool is_game_specific) {
    readbacksModeEntry.set(mode, is_game_specific);
}

void setReadbackLinearImages(bool enable, bool is_game_specific) {
    readbackLinearImagesEnabled.set(enable, is_game_specific);
}

void setDirectMemoryAccess(bool enable, bool is_game_specific) {
    directMemoryAccessEnabled.set(enable, is_game_specific);
}

void setDumpShaders(bool enable, bool is_game_specific) {
    shouldDumpShaders.set(enable, is_game_specific);
}

void setVkValidation(bool enable, bool is_game_specific) {
    vkValidation.set(enable, is_game_specific);
}

void setVkSyncValidation(bool enable, bool is_game_specific) {
    vkValidationSync.set(enable, is_game_specific);
}

void setVkCoreValidation(bool enable, bool is_game_specific) {
    vkValidationCore.set(enable, is_game_specific);
}

void setVkGpuValidation(bool enable, bool is_game_specific) {
    vkValidationGpu.set(enable, is_game_specific);
}

void setRdocEnabled(bool enable, bool is_game_specific) {
    rdocEnable.set(enable, is_game_specific);
}

void setPipelineCacheEnabled(bool enable, bool is_game_specific) {
    pipelineCacheEnable.set(enable, is_game_specific);
}

void setPipelineCacheArchived(bool enable, bool is_game_specific) {
    pipelineCacheArchive.set(enable, is_game_specific);
}

void setVblankFreq(u32 value, bool is_game_specific) {
    vblankFrequency.set(value, is_game_specific);
}

void setIsFullscreen(bool enable, bool is_game_specific) {
    isFullscreen.set(enable, is_game_specific);
}

void setFullscreenMode(string mode, bool is_game_specific) {
    fullscreenMode.set(mode, is_game_specific);
}

void setPresentMode(std::string mode, bool is_game_specific) {
    presentMode.set(mode, is_game_specific);
}

void setisTrophyPopupDisabled(bool disable, bool is_game_specific) {
    isTrophyPopupDisabled.set(disable, is_game_specific);
}

void setEnableDiscordRPC(bool enable) {
    enableDiscordRPC = enable;
}

void setCursorState(s16 newCursorState, bool is_game_specific) {
    cursorState.set(newCursorState, is_game_specific);
}

void setCursorHideTimeout(int newcursorHideTimeout, bool is_game_specific) {
    cursorHideTimeout.set(newcursorHideTimeout, is_game_specific);
}

void setMicDevice(std::string device, bool is_game_specific) {
    micDevice.set(device, is_game_specific);
}

void setMainOutputDevice(std::string device, bool is_game_specific) {
    mainOutputDevice.set(device, is_game_specific);
}

void setPadSpkOutputDevice(std::string device, bool is_game_specific) {
    padSpkOutputDevice.set(device, is_game_specific);
}

void setTrophyNotificationDuration(double newTrophyNotificationDuration, bool is_game_specific) {
    trophyNotificationDuration.set(newTrophyNotificationDuration, is_game_specific);
}

void setLanguage(u32 language, bool is_game_specific) {
    m_language.set(language, is_game_specific);
}

void setNeoMode(bool enable, bool is_game_specific) {
    isNeo.set(enable, is_game_specific);
}

void setDevKitConsole(bool enable, bool is_game_specific) {
    isDevKit.set(enable, is_game_specific);
}

void setWindowsGuestRedZoneProtection(bool enable, bool is_game_specific) {
    windowsGuestRedZoneProtection.set(enable, is_game_specific);
}

void setLogType(const string& type, bool is_game_specific) {
    logType.set(type, is_game_specific);
}

void setLogFilter(const string& type, bool is_game_specific) {
    logFilter.set(type, is_game_specific);
}

void setSeparateLogFilesEnabled(bool enabled, bool is_game_specific) {
    isSeparateLogFilesEnabled.set(enabled, is_game_specific);
}

void setUserName(const string& name, bool is_game_specific) {
    userName.set(name, is_game_specific);
}

void setUseSpecialPad(bool use) {
    useSpecialPad.base_value = use;
}

void setSpecialPadClass(int type) {
    specialPadClass.base_value = type;
}

void setIsMotionControlsEnabled(bool use, bool is_game_specific) {
    isMotionControlsEnabled.set(use, is_game_specific);
}

void setGyroSwapYawRoll(bool enable, bool is_game_specific) {
    gyroSwapYawRoll.set(enable, is_game_specific);
}

void setGyroInvertYaw(bool enable, bool is_game_specific) {
    gyroInvertYaw.set(enable, is_game_specific);
}

void setGyroInvertX(bool enable, bool is_game_specific) {
    gyroInvertX.set(enable, is_game_specific);
}

void setGyroInvertRoll(bool enable, bool is_game_specific) {
    gyroInvertRoll.set(enable, is_game_specific);
}

bool addGameInstallDir(const std::filesystem::path& dir, bool enabled) {
    for (const auto& install_dir : settings_install_dirs) {
        if (install_dir.path == dir) {
            return false;
        }
    }
    settings_install_dirs.push_back({dir, enabled});
    return true;
}

void removeGameInstallDir(const std::filesystem::path& dir) {
    auto iterator =
        std::find_if(settings_install_dirs.begin(), settings_install_dirs.end(),
                     [&dir](const GameInstallDir& install_dir) { return install_dir.path == dir; });
    if (iterator != settings_install_dirs.end()) {
        settings_install_dirs.erase(iterator);
    }
}

void setGameInstallDirEnabled(const std::filesystem::path& dir, bool enabled) {
    auto iterator =
        std::find_if(settings_install_dirs.begin(), settings_install_dirs.end(),
                     [&dir](const GameInstallDir& install_dir) { return install_dir.path == dir; });
    if (iterator != settings_install_dirs.end()) {
        iterator->enabled = enabled;
    }
}

void setAddonInstallDir(const std::filesystem::path& dir) {
    settings_addon_install_dir = dir;
}

void setGameInstallDirs(const std::vector<std::filesystem::path>& dirs_config) {
    settings_install_dirs.clear();
    for (const auto& dir : dirs_config) {
        settings_install_dirs.push_back({dir, true});
    }
}

void setAllGameInstallDirs(const std::vector<GameInstallDir>& dirs_config) {
    settings_install_dirs = dirs_config;
}

void setSaveDataPath(const std::filesystem::path& path) {
    // GR2FORK: compat shim - the GUI/launcher "save data path" maps to the PS4 home dir.
    home_dir = path;
}

const std::vector<std::filesystem::path> getGameInstallDirs() {
    std::vector<std::filesystem::path> enabled_dirs;
    for (const auto& dir : settings_install_dirs) {
        if (dir.enabled) {
            enabled_dirs.push_back(dir.path);
        }
    }
    return enabled_dirs;
}

const std::vector<bool> getGameInstallDirsEnabled() {
    std::vector<bool> enabled_dirs;
    for (const auto& dir : settings_install_dirs) {
        enabled_dirs.push_back(dir.enabled);
    }
    return enabled_dirs;
}

std::filesystem::path getAddonInstallDir() {
    if (settings_addon_install_dir.empty()) {
        // Default for users without a config file or a config file from before this option existed
        return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "addcont";
    }
    return settings_addon_install_dir;
}

u32 GetLanguage() {
    return m_language.get();
}

bool getSeparateLogFilesEnabled() {
    return isSeparateLogFilesEnabled.get();
}

bool getPSNSignedIn() {
    return isPSNSignedIn.get();
}

void setPSNSignedIn(bool sign, bool is_game_specific) {
    isPSNSignedIn.set(sign, is_game_specific);
}

string getDefaultControllerID() {
    return defaultControllerID.get();
}

void setDefaultControllerID(string id) {
    defaultControllerID.base_value = id;
}

bool getBackgroundControllerInput() {
    return backgroundControllerInput.get();
}

void setBackgroundControllerInput(bool enable, bool is_game_specific) {
    backgroundControllerInput.set(enable, is_game_specific);
}

bool getFsrEnabled() {
    return fsrEnabled.get();
}

void setFsrEnabled(bool enable, bool is_game_specific) {
    fsrEnabled.set(enable, is_game_specific);
}

bool getRcasEnabled() {
    return rcasEnabled.get();
}

void setRcasEnabled(bool enable, bool is_game_specific) {
    rcasEnabled.set(enable, is_game_specific);
}

int getRcasAttenuation() {
    return rcasAttenuation.get();
}

void setRcasAttenuation(int value, bool is_game_specific) {
    rcasAttenuation.set(value, is_game_specific);
}

int getGameplaySyncBudgetMs() {
    // Defensive clamp: the wait site feeds this straight into future::wait_for. The launcher
    // slider is bounded to [0, 50000], but a hand-edited config.toml could be anything, and a
    // typo must not wedge the GpuComm thread for minutes.
    int v = gameplaySyncBudgetMs.get();
    if (v < 0) {
        v = 0;
    }
    if (v > 50000) {
        v = 50000;
    }
    return v;
}

void setGameplaySyncBudgetMs(int value, bool is_game_specific) {
    gameplaySyncBudgetMs.set(value, is_game_specific);
}

std::string getAspectRatioOverride() {
    return aspectRatioOverride.get();
}

void setAspectRatioOverride(const std::string& value, bool is_game_specific) {
    aspectRatioOverride.set(value, is_game_specific);
}

std::string getResolutionOverride() {
    return resolutionOverride.get();
}

void setResolutionOverride(const std::string& value, bool is_game_specific) {
    resolutionOverride.set(value, is_game_specific);
}

std::string getResolutionPatchGroups() {
    return resolutionPatchGroups.get();
}

void setResolutionPatchGroups(const std::string& value, bool is_game_specific) {
    resolutionPatchGroups.set(value, is_game_specific);
}

std::string getResolutionPatchGroupsGrr() {
    return resolutionPatchGroupsGrr.get();
}

void setResolutionPatchGroupsGrr(const std::string& value, bool is_game_specific) {
    resolutionPatchGroupsGrr.set(value, is_game_specific);
}

// GR2FORK shader-hash GPU-work skip. gr2SkippedShaders is a comma-separated selector list
// matched against each draw's fragment / dispatch's compute pgm_hash (the fs_0x/cs_0x dump
// names); a match drops the work. "0x<hash>" always skips; "0x<hash>@<W>x<H>" draws at WxH only.
static std::mutex gr2_skip_mtx;
static std::unordered_set<u64> gr2_skip_any;   // hash-only selectors
static std::unordered_set<u64> gr2_skip_ext;   // hash+extent selectors (packed)
static std::atomic<bool> gr2_skip_active{false};

static u64 gr2SkipExtKey(u64 hash, u32 w, u32 h) noexcept {
    return hash ^ (static_cast<u64>(w) * 0x9e3779b97f4a7c15ULL) ^ (static_cast<u64>(h) << 33);
}

static void rebuildGr2ShaderSkipCache() {
    const std::string spec = gr2SkippedShaders.get();
    std::unordered_set<u64> any, ext;
    size_t pos = 0;
    while (pos <= spec.size()) {
        const size_t comma = spec.find(',', pos);
        std::string tok =
            spec.substr(pos, comma == std::string::npos ? spec.size() - pos : comma - pos);
        pos = (comma == std::string::npos) ? spec.size() + 1 : comma + 1;
        const size_t b = tok.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) {
            continue;
        }
        const size_t e = tok.find_last_not_of(" \t\r\n");
        tok = tok.substr(b, e - b + 1);
        const size_t at = tok.find('@');
        const std::string hpart = (at == std::string::npos) ? tok : tok.substr(0, at);
        u64 hash = 0;
        try {
            hash = std::stoull(hpart, nullptr, 16);
        } catch (...) {
            LOG_WARNING(Config, "gr2SkippedShaders: ignoring malformed entry '{}'", tok);
            continue;
        }
        if (at == std::string::npos) {
            any.insert(hash);
            continue;
        }
        const std::string rpart = tok.substr(at + 1);
        const size_t xpos = rpart.find('x');
        if (xpos == std::string::npos) {
            LOG_WARNING(Config, "gr2SkippedShaders: ignoring bad extent in '{}'", tok);
            continue;
        }
        try {
            const u32 w = static_cast<u32>(std::stoul(rpart.substr(0, xpos)));
            const u32 h = static_cast<u32>(std::stoul(rpart.substr(xpos + 1)));
            ext.insert(gr2SkipExtKey(hash, w, h));
        } catch (...) {
            LOG_WARNING(Config, "gr2SkippedShaders: ignoring bad extent in '{}'", tok);
            continue;
        }
    }
    const size_t n_any = any.size();
    const size_t n_ext = ext.size();
    {
        std::lock_guard<std::mutex> lk(gr2_skip_mtx);
        gr2_skip_any = std::move(any);
        gr2_skip_ext = std::move(ext);
        gr2_skip_active.store(!(gr2_skip_any.empty() && gr2_skip_ext.empty()),
                              std::memory_order_relaxed);
    }
    LOG_INFO(Config, "gr2SkippedShaders: {} hash-only + {} hash@extent selector(s) (active={})",
             n_any, n_ext, (n_any + n_ext) != 0);
}

std::string getGr2SkippedShaders() {
    return gr2SkippedShaders.get();
}

void setGr2SkippedShaders(const std::string& value, bool is_game_specific) {
    gr2SkippedShaders.set(value, is_game_specific);
    rebuildGr2ShaderSkipCache();
}

bool gr2ShaderSkipActive() {
    return gr2_skip_active.load(std::memory_order_relaxed);
}

bool gr2IsShaderSkipped(u64 hash, u32 width, u32 height) {
    std::lock_guard<std::mutex> lk(gr2_skip_mtx);
    if (gr2_skip_any.find(hash) != gr2_skip_any.end()) {
        return true;
    }
    if (width != 0 && height != 0 &&
        gr2_skip_ext.find(gr2SkipExtKey(hash, width, height)) != gr2_skip_ext.end()) {
        return true;
    }
    return false;
}

int getUsbDeviceBackend() {
    return usbDeviceBackend.get();
}

void setUsbDeviceBackend(int value, bool is_game_specific) {
    usbDeviceBackend.set(value, is_game_specific);
}

bool getLoadAutoPatches() {
    return load_auto_patches;
}
void setLoadAutoPatches(bool enable) {
    load_auto_patches = enable;
}

// GR2FORK: import a legacy pre-port TOML config into the in-memory settings. Used
// only by the one-time migration path in load(); does not re-save.
static void loadFromToml(const std::filesystem::path& path, bool is_game_specific) {
    toml::value data;
    try {
        std::ifstream ifs;
        ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        ifs.open(path, std::ios_base::binary);
        data = toml::parse(ifs, string{fmt::UTF(path.filename().u8string()).data});
    } catch (std::exception& ex) {
        fmt::print("Got exception trying to load toml config file. Exception: {}\n", ex.what());
        return;
    }

    if (data.contains("General")) {
        const toml::value& general = data.at("General");

        volumeSlider.setFromToml(general, "volumeSlider", is_game_specific);
        isNeo.setFromToml(general, "isPS4Pro", is_game_specific);
        isDevKit.setFromToml(general, "isDevKit", is_game_specific);
        windowsGuestRedZoneProtection.setFromToml(general, "windowsGuestRedZoneProtection",
                                                  is_game_specific);
        if (is_game_specific) { // do not get this value from the base config
            extraDmemInMbytes.setFromToml(general, "extraDmemInMbytes", is_game_specific);
        }
        isPSNSignedIn.setFromToml(general, "isPSNSignedIn", is_game_specific);
        isTrophyPopupDisabled.setFromToml(general, "isTrophyPopupDisabled", is_game_specific);
        trophyNotificationDuration.setFromToml(general, "trophyNotificationDuration",
                                               is_game_specific);
        enableDiscordRPC = toml::find_or<bool>(general, "enableDiscordRPC", enableDiscordRPC);
        logFilter.setFromToml(general, "logFilter", is_game_specific);
        logType.setFromToml(general, "logType", is_game_specific);
        userName.setFromToml(general, "userName", is_game_specific);
        httpHostOverride.setFromToml(general, "httpHostOverride", is_game_specific);
        httpHostOverridePort.setFromToml(general, "httpHostOverridePort", is_game_specific);
        httpForceHttp.setFromToml(general, "httpForceHttp", is_game_specific);
        shadnetNpid.setFromToml(general, "shadnetNpid", is_game_specific);
        shadnetPassword.setFromToml(general, "shadnetPassword", is_game_specific);
        shadnetServer.setFromToml(general, "shadnetServer", is_game_specific);
        isShowSplash.setFromToml(general, "showSplash", is_game_specific);
        isSideTrophy.setFromToml(general, "sideTrophy", is_game_specific);

        isConnectedToNetwork.setFromToml(general, "isConnectedToNetwork", is_game_specific);
        defaultControllerID.setFromToml(general, "defaultControllerID", is_game_specific);
        sys_modules_path = toml::find_fs_path_or(general, "sysModulesPath", sys_modules_path);
    }

    if (data.contains("Input")) {
        const toml::value& input = data.at("Input");

        cursorState.setFromToml(input, "cursorState", is_game_specific);
        cursorHideTimeout.setFromToml(input, "cursorHideTimeout", is_game_specific);
        useSpecialPad.setFromToml(input, "useSpecialPad", is_game_specific);
        specialPadClass.setFromToml(input, "specialPadClass", is_game_specific);
        isMotionControlsEnabled.setFromToml(input, "isMotionControlsEnabled", is_game_specific);
        gyroSwapYawRoll.setFromToml(input, "gyroSwapYawRoll", is_game_specific);
        gyroInvertYaw.setFromToml(input, "gyroInvertYaw", is_game_specific);
        gyroInvertX.setFromToml(input, "gyroInvertX", is_game_specific);
        gyroInvertRoll.setFromToml(input, "gyroInvertRoll", is_game_specific);
        useUnifiedInputConfig.setFromToml(input, "useUnifiedInputConfig", is_game_specific);
        backgroundControllerInput.setFromToml(input, "backgroundControllerInput", is_game_specific);
        usbDeviceBackend.setFromToml(input, "usbDeviceBackend", is_game_specific);
    }

    if (data.contains("Audio")) {
        const toml::value& audio = data.at("Audio");

        micDevice.setFromToml(audio, "micDevice", is_game_specific);
        mainOutputDevice.setFromToml(audio, "mainOutputDevice", is_game_specific);
        padSpkOutputDevice.setFromToml(audio, "padSpkOutputDevice", is_game_specific);
    }

    if (data.contains("GPU")) {
        const toml::value& gpu = data.at("GPU");

        windowWidth.setFromToml(gpu, "screenWidth", is_game_specific);
        windowHeight.setFromToml(gpu, "screenHeight", is_game_specific);
        internalScreenWidth.setFromToml(gpu, "internalScreenWidth", is_game_specific);
        internalScreenHeight.setFromToml(gpu, "internalScreenHeight", is_game_specific);
        isNullGpu.setFromToml(gpu, "nullGpu", is_game_specific);
        isDeviceRecovery.setFromToml(gpu, "deviceRecovery", is_game_specific);
        shouldCopyGPUBuffers.setFromToml(gpu, "copyGPUBuffers", is_game_specific);
        // GR2FORK: legacy toml stored readbacks as a bool; map it to the 3-state mode.
        if (auto rb = toml::get_optional<bool>(gpu, "readbacks")) {
            readbacksModeEntry.set(*rb ? GpuReadbacksMode::Relaxed : GpuReadbacksMode::Disabled,
                                   is_game_specific);
        }
        readbackLinearImagesEnabled.setFromToml(gpu, "readbackLinearImages", is_game_specific);
        directMemoryAccessEnabled.setFromToml(gpu, "directMemoryAccess", is_game_specific);
        shouldDumpShaders.setFromToml(gpu, "dumpShaders", is_game_specific);
        shouldPatchShaders.setFromToml(gpu, "patchShaders", is_game_specific);
        vblankFrequency.setFromToml(gpu, "vblankFrequency", is_game_specific);
        isFullscreen.setFromToml(gpu, "Fullscreen", is_game_specific);
        fullscreenMode.setFromToml(gpu, "FullscreenMode", is_game_specific);
        presentMode.setFromToml(gpu, "presentMode", is_game_specific);
        isHDRAllowed.setFromToml(gpu, "allowHDR", is_game_specific);
        fsrEnabled.setFromToml(gpu, "fsrEnabled", is_game_specific);
        rcasEnabled.setFromToml(gpu, "rcasEnabled", is_game_specific);
        rcasAttenuation.setFromToml(gpu, "rcasAttenuation", is_game_specific);
        gameplaySyncBudgetMs.setFromToml(gpu, "gameplaySyncBudgetMs", is_game_specific);
        aspectRatioOverride.setFromToml(gpu, "aspectRatioOverride", is_game_specific);
        resolutionOverride.setFromToml(gpu, "resolutionOverride", is_game_specific);
        resolutionPatchGroups.setFromToml(gpu, "resolutionPatchGroups", is_game_specific);
        resolutionPatchGroupsGrr.setFromToml(gpu, "resolutionPatchGroupsGrr", is_game_specific);
        gr2FixNativeCubeViewsEnable.setFromToml(gpu, "gr2FixNativeCubeViews", is_game_specific);
        disableMotionBlurEnable.setFromToml(gpu, "disableMotionBlur", is_game_specific);
    }

    if (data.contains("Vulkan")) {
        const toml::value& vk = data.at("Vulkan");

        gpuId.setFromToml(vk, "gpuId", is_game_specific);
        vkValidation.setFromToml(vk, "validation", is_game_specific);
        vkValidationCore.setFromToml(vk, "validation_core", is_game_specific);
        vkValidationSync.setFromToml(vk, "validation_sync", is_game_specific);
        vkValidationGpu.setFromToml(vk, "validation_gpu", is_game_specific);
        vkCrashDiagnostic.setFromToml(vk, "crashDiagnostic", is_game_specific);
        vkHostMarkers.setFromToml(vk, "hostMarkers", is_game_specific);
        vkGuestMarkers.setFromToml(vk, "guestMarkers", is_game_specific);
        rdocEnable.setFromToml(vk, "rdocEnable", is_game_specific);
        pipelineCacheEnable.setFromToml(vk, "pipelineCacheEnable", is_game_specific);
        pipelineCacheArchive.setFromToml(vk, "pipelineCacheArchive", is_game_specific);
        vkForcePushDescriptors.setFromToml(vk, "forcePushDescriptors", is_game_specific);
        vkDisablePushDescriptors.setFromToml(vk, "disablePushDescriptors", is_game_specific);
        beginRenderingCacheEnable.setFromToml(vk, "beginRenderingCacheEnable", is_game_specific);
        shDynamicDirtySkip.setFromToml(vk, "shDynamicDirtySkip", is_game_specific);
        pipelineUdHashLruEnable.setFromToml(vk, "pipelineUdHashLruEnable", is_game_specific);
        pipelineSpecFpLruEnable.setFromToml(vk, "pipelineSpecFpLruEnable", is_game_specific);
        pipelineGfxKeyCtxSkipEnable.setFromToml(vk, "pipelineGfxKeyCtxSkipEnable", is_game_specific);
        descSetBindingSkipCache.setFromToml(vk, "descSetBindingSkipCache", is_game_specific);
        accurateRenderTargetCache.setFromToml(vk, "accurateRenderTargetCache", is_game_specific);
        accurateVertexBufferCache.setFromToml(vk, "accurateVertexBufferCache", is_game_specific);
    }

    if (data.contains("Debug")) {
        const toml::value& debug = data.at("Debug");

        isDebugDump.setFromToml(debug, "DebugDump", is_game_specific);
        isSeparateLogFilesEnabled.setFromToml(debug, "isSeparateLogFilesEnabled", is_game_specific);
        isShaderDebug.setFromToml(debug, "CollectShader", is_game_specific);
        isFpsColor.setFromToml(debug, "FPSColor", is_game_specific);
        showFpsCounter.setFromToml(debug, "showFpsCounter", is_game_specific);
        logEnabled.setFromToml(debug, "logEnabled", is_game_specific);
    }

    if (data.contains("GUI")) {
        const toml::value& gui = data.at("GUI");

        const auto install_dir_array =
            toml::find_or<std::vector<std::u8string>>(gui, "installDirs", {});

        try {
            install_dirs_enabled = toml::find<std::vector<bool>>(gui, "installDirsEnabled");
        } catch (...) {
            // If it does not exist, assume that all are enabled.
            install_dirs_enabled.resize(install_dir_array.size(), true);
        }

        if (install_dirs_enabled.size() < install_dir_array.size()) {
            install_dirs_enabled.resize(install_dir_array.size(), true);
        }

        settings_install_dirs.clear();
        for (size_t i = 0; i < install_dir_array.size(); i++) {
            settings_install_dirs.push_back(
                {std::filesystem::path{install_dir_array[i]}, install_dirs_enabled[i]});
        }

        home_dir = toml::find_fs_path_or(gui, "saveDataPath", home_dir);

        settings_addon_install_dir =
            toml::find_fs_path_or(gui, "addonInstallDir", settings_addon_install_dir);
    }

    if (data.contains("Settings")) {
        const toml::value& settings = data.at("Settings");
        m_language.setFromToml(settings, "consoleLanguage", is_game_specific);
    }

    if (data.contains("Keys")) {
        const toml::value& keys = data.at("Keys");
        trophyKey = toml::find_or<string>(keys, "TrophyKey", trophyKey);
    }
}

// GR2FORK: canonical config loader. Reads the shared config.json schema (binary-
// compatible with the prerelease build) plus gr2fork's private "GR2Fork" section.
static void loadFromJson(const std::filesystem::path& path, bool is_game_specific) {
    json data;
    try {
        std::ifstream ifs(path);
        ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        ifs >> data;
    } catch (const std::exception& ex) {
        fmt::print("Got exception trying to load json config file. Exception: {}\n", ex.what());
        return;
    }
    if (!data.is_object()) {
        return;
    }

    string current_version = {};

    if (data.contains("General") && data.at("General").is_object()) {
        const json& general = data.at("General");

        volumeSlider.setFromJson(general, "volume_slider", is_game_specific);
        isNeo.setFromJson(general, "neo_mode", is_game_specific);
        isDevKit.setFromJson(general, "dev_kit_mode", is_game_specific);
        windowsGuestRedZoneProtection.setFromJson(general, "windows_guest_red_zone_protection",
                                                  is_game_specific);
        if (is_game_specific) {
            extraDmemInMbytes.setFromJson(general, "extra_dmem_in_mbytes", is_game_specific);
        }
        isTrophyPopupDisabled.setFromJson(general, "trophy_popup_disabled", is_game_specific);
        trophyNotificationDuration.setFromJson(general, "trophy_notification_duration",
                                               is_game_specific);
        isShowSplash.setFromJson(general, "show_splash", is_game_specific);
        isSideTrophy.setFromJson(general, "trophy_notification_side", is_game_specific);
        isConnectedToNetwork.setFromJson(general, "connected_to_network", is_game_specific);
        showFpsCounter.setFromJson(general, "show_fps_counter", is_game_specific);
        m_language.setFromJson(general, "console_language", is_game_specific);

        if (general.contains("discord_rpc_enabled")) {
            try {
                enableDiscordRPC = general.at("discord_rpc_enabled").get<bool>();
            } catch (const std::exception&) {
            }
        }
        if (general.contains("sys_modules_dir")) {
            try {
                sys_modules_path = general.at("sys_modules_dir").get<std::filesystem::path>();
            } catch (const std::exception&) {
            }
        }
        if (general.contains("addon_install_dir")) {
            try {
                settings_addon_install_dir =
                    general.at("addon_install_dir").get<std::filesystem::path>();
            } catch (const std::exception&) {
            }
        }
        if (general.contains("home_dir")) {
            try {
                home_dir = general.at("home_dir").get<std::filesystem::path>();
            } catch (const std::exception&) {
            }
        }
        if (general.contains("install_dirs") && general.at("install_dirs").is_array()) {
            settings_install_dirs.clear();
            install_dirs_enabled.clear();
            for (const auto& entry : general.at("install_dirs")) {
                if (!entry.is_object()) {
                    continue;
                }
                std::filesystem::path dir_path;
                bool enabled = true;
                try {
                    if (entry.contains("path")) {
                        dir_path = entry.at("path").get<std::filesystem::path>();
                    }
                    if (entry.contains("enabled")) {
                        enabled = entry.at("enabled").get<bool>();
                    }
                } catch (const std::exception&) {
                    continue;
                }
                settings_install_dirs.push_back({dir_path, enabled});
                install_dirs_enabled.push_back(enabled);
            }
        }
    }

    if (data.contains("Log") && data.at("Log").is_object()) {
        const json& log = data.at("Log");

        logEnabled.setFromJson(log, "enable", is_game_specific);
        isSeparateLogFilesEnabled.setFromJson(log, "separate", is_game_specific);
        logFilter.setFromJson(log, "filter", is_game_specific);
        // Derive gr2fork's string logType from the shared Log.sync flag. The exact
        // string (if gr2fork wrote one) is restored from the GR2Fork section below.
        if (log.contains("sync")) {
            try {
                const bool sync = log.at("sync").get<bool>();
                logType.set(sync ? string("sync") : string("async"), is_game_specific);
            } catch (const std::exception&) {
            }
        }
    }

    if (data.contains("Debug") && data.at("Debug").is_object()) {
        const json& debug = data.at("Debug");

        isDebugDump.setFromJson(debug, "debug_dump", is_game_specific);
        isShaderDebug.setFromJson(debug, "shader_collect", is_game_specific);
        if (debug.contains("config_version")) {
            try {
                current_version = debug.at("config_version").get<string>();
            } catch (const std::exception&) {
            }
        }
    }

    if (data.contains("Input") && data.at("Input").is_object()) {
        const json& input = data.at("Input");

        cursorState.setFromJson(input, "cursor_state", is_game_specific);
        cursorHideTimeout.setFromJson(input, "cursor_hide_timeout", is_game_specific);
        useSpecialPad.setFromJson(input, "use_special_pad", is_game_specific);
        specialPadClass.setFromJson(input, "special_pad_class", is_game_specific);
        isMotionControlsEnabled.setFromJson(input, "motion_controls_enabled", is_game_specific);
        useUnifiedInputConfig.setFromJson(input, "use_unified_input_config", is_game_specific);
        defaultControllerID.setFromJson(input, "default_controller_id", is_game_specific);
        backgroundControllerInput.setFromJson(input, "background_controller_input",
                                              is_game_specific);
        usbDeviceBackend.setFromJson(input, "usb_device_backend", is_game_specific);
    }

    if (data.contains("Audio") && data.at("Audio").is_object()) {
        const json& audio = data.at("Audio");

        micDevice.setFromJson(audio, "sdl_mic_device", is_game_specific);
        mainOutputDevice.setFromJson(audio, "sdl_main_output_device", is_game_specific);
        padSpkOutputDevice.setFromJson(audio, "sdl_padSpk_output_device", is_game_specific);
    }

    if (data.contains("GPU") && data.at("GPU").is_object()) {
        const json& gpu = data.at("GPU");

        windowWidth.setFromJson(gpu, "window_width", is_game_specific);
        windowHeight.setFromJson(gpu, "window_height", is_game_specific);
        internalScreenWidth.setFromJson(gpu, "internal_screen_width", is_game_specific);
        internalScreenHeight.setFromJson(gpu, "internal_screen_height", is_game_specific);
        isNullGpu.setFromJson(gpu, "null_gpu", is_game_specific);
        shouldCopyGPUBuffers.setFromJson(gpu, "copy_gpu_buffers", is_game_specific);
        readbacksModeEntry.setFromJson(gpu, "readbacks_mode", is_game_specific);
        readbackLinearImagesEnabled.setFromJson(gpu, "readback_linear_images_enabled",
                                                is_game_specific);
        directMemoryAccessEnabled.setFromJson(gpu, "direct_memory_access_enabled",
                                              is_game_specific);
        shouldDumpShaders.setFromJson(gpu, "dump_shaders", is_game_specific);
        shouldPatchShaders.setFromJson(gpu, "patch_shaders", is_game_specific);
        vblankFrequency.setFromJson(gpu, "vblank_frequency", is_game_specific);
        isFullscreen.setFromJson(gpu, "full_screen", is_game_specific);
        fullscreenMode.setFromJson(gpu, "full_screen_mode", is_game_specific);
        presentMode.setFromJson(gpu, "present_mode", is_game_specific);
        isHDRAllowed.setFromJson(gpu, "hdr_allowed", is_game_specific);
        fsrEnabled.setFromJson(gpu, "fsr_enabled", is_game_specific);
        rcasEnabled.setFromJson(gpu, "rcas_enabled", is_game_specific);
        rcasAttenuation.setFromJson(gpu, "rcas_attenuation", is_game_specific);
    }

    if (data.contains("Vulkan") && data.at("Vulkan").is_object()) {
        const json& vk = data.at("Vulkan");

        gpuId.setFromJson(vk, "gpu_id", is_game_specific);
        vkValidation.setFromJson(vk, "vkvalidation_enabled", is_game_specific);
        vkValidationCore.setFromJson(vk, "vkvalidation_core_enabled", is_game_specific);
        vkValidationSync.setFromJson(vk, "vkvalidation_sync_enabled", is_game_specific);
        vkValidationGpu.setFromJson(vk, "vkvalidation_gpu_enabled", is_game_specific);
        vkCrashDiagnostic.setFromJson(vk, "vkcrash_diagnostic_enabled", is_game_specific);
        vkHostMarkers.setFromJson(vk, "vkhost_markers", is_game_specific);
        vkGuestMarkers.setFromJson(vk, "vkguest_markers", is_game_specific);
        rdocEnable.setFromJson(vk, "renderdoc_enabled", is_game_specific);
        pipelineCacheEnable.setFromJson(vk, "pipeline_cache_enabled", is_game_specific);
        pipelineCacheArchive.setFromJson(vk, "pipeline_cache_archived", is_game_specific);
    }

    // gr2fork-only settings (prerelease preserves this section verbatim on save).
    if (data.contains("GR2Fork") && data.at("GR2Fork").is_object()) {
        const json& g = data.at("GR2Fork");

        isPSNSignedIn.setFromJson(g, "isPSNSignedIn", is_game_specific);
        userName.setFromJson(g, "userName", is_game_specific);
        httpHostOverride.setFromJson(g, "httpHostOverride", is_game_specific);
        httpHostOverridePort.setFromJson(g, "httpHostOverridePort", is_game_specific);
        httpForceHttp.setFromJson(g, "httpForceHttp", is_game_specific);
        shadnetNpid.setFromJson(g, "shadnetNpid", is_game_specific);
        shadnetPassword.setFromJson(g, "shadnetPassword", is_game_specific);
        shadnetServer.setFromJson(g, "shadnetServer", is_game_specific);
        // Prefer gr2fork's exact logType string over the Log.sync-derived value above.
        logType.setFromJson(g, "logType", is_game_specific);
        isFpsColor.setFromJson(g, "fpsColor", is_game_specific);
        gyroSwapYawRoll.setFromJson(g, "gyroSwapYawRoll", is_game_specific);
        gyroInvertYaw.setFromJson(g, "gyroInvertYaw", is_game_specific);
        gyroInvertX.setFromJson(g, "gyroInvertX", is_game_specific);
        gyroInvertRoll.setFromJson(g, "gyroInvertRoll", is_game_specific);
        gameplaySyncBudgetMs.setFromJson(g, "gameplaySyncBudgetMs", is_game_specific);
        aspectRatioOverride.setFromJson(g, "aspectRatioOverride", is_game_specific);
        resolutionOverride.setFromJson(g, "resolutionOverride", is_game_specific);
        resolutionPatchGroups.setFromJson(g, "resolutionPatchGroups", is_game_specific);
        resolutionPatchGroupsGrr.setFromJson(g, "resolutionPatchGroupsGrr", is_game_specific);
        gr2SkippedShaders.setFromJson(g, "gr2SkippedShaders", is_game_specific);
        rebuildGr2ShaderSkipCache();
        gr2FixNativeCubeViewsEnable.setFromJson(g, "gr2FixNativeCubeViews", is_game_specific);
        gr2TitleThemeMod.setFromJson(g, "GR2TitleThemeMod", is_game_specific);
        disableMotionBlurEnable.setFromJson(g, "disableMotionBlur", is_game_specific);
        padSpkOutputDisabled.setFromJson(g, "padSpkOutputDisabled", is_game_specific);
        vkForcePushDescriptors.setFromJson(g, "vkForcePushDescriptors", is_game_specific);
        vkDisablePushDescriptors.setFromJson(g, "vkDisablePushDescriptors", is_game_specific);
        beginRenderingCacheEnable.setFromJson(g, "beginRenderingCacheEnable", is_game_specific);
        shDynamicDirtySkip.setFromJson(g, "shDynamicDirtySkip", is_game_specific);
        pipelineUdHashLruEnable.setFromJson(g, "pipelineUdHashLruEnable", is_game_specific);
        pipelineSpecFpLruEnable.setFromJson(g, "pipelineSpecFpLruEnable", is_game_specific);
        pipelineGfxKeyCtxSkipEnable.setFromJson(g, "pipelineGfxKeyCtxSkipEnable", is_game_specific);
        descSetBindingSkipCache.setFromJson(g, "descSetBindingSkipCache", is_game_specific);
        accurateRenderTargetCache.setFromJson(g, "accurateRenderTargetCache", is_game_specific);
        accurateVertexBufferCache.setFromJson(g, "accurateVertexBufferCache", is_game_specific);
        if (g.contains("trophyKey")) {
            try {
                trophyKey = g.at("trophyKey").get<string>();
            } catch (const std::exception&) {
            }
        }
    }

    // Re-save once if this config came from a build with a different revision, so any
    // newly-added keys get written with defaults (mirrors the prerelease behavior).
    if (config_version != current_version && !is_game_specific) {
        save(path);
    }
}

void load(const std::filesystem::path& path, bool is_game_specific) {
    std::error_code error;

    // Preferred path: the json config already exists.
    if (std::filesystem::exists(path, error)) {
        loadFromJson(path, is_game_specific);
        return;
    }

    // GR2FORK: one-time migration. If the json config is absent but the sibling
    // pre-port .toml exists, import it and write the json so that both this build
    // and the prerelease pick up the user's existing settings going forward.
    std::filesystem::path toml_path = path;
    toml_path.replace_extension(".toml");
    if (std::filesystem::exists(toml_path, error)) {
        LOG_INFO(Config, "Migrating legacy TOML config {} -> {}",
                 string{fmt::UTF(toml_path.filename().u8string()).data},
                 string{fmt::UTF(path.filename().u8string()).data});
        loadFromToml(toml_path, is_game_specific);
        if (is_game_specific) {
            // Persist the imported overrides as json. save() clears the in-memory
            // game_specific_value side as it writes, so re-load to repopulate them.
            save(path, true);
            loadFromJson(path, true);
        } else {
            save(path);
        }
        return;
    }

    // No config of either format. Create a fresh json with defaults (global only).
    if (!is_game_specific) {
        save(path);
    }
}

void save(const std::filesystem::path& path, bool is_game_specific) {
    // Build the json this build knows about, then merge it into any existing file so
    // sections/keys written by the prerelease (or a newer gr2fork) are preserved.
    json data = json::object();

    // ---- General (shared schema) ----
    volumeSlider.setJsonValue(data, "General", "volume_slider", is_game_specific);
    isTrophyPopupDisabled.setJsonValue(data, "General", "trophy_popup_disabled", is_game_specific);
    trophyNotificationDuration.setJsonValue(data, "General", "trophy_notification_duration",
                                            is_game_specific);
    isShowSplash.setJsonValue(data, "General", "show_splash", is_game_specific);
    isSideTrophy.setJsonValue(data, "General", "trophy_notification_side", is_game_specific);
    isNeo.setJsonValue(data, "General", "neo_mode", is_game_specific);
    isDevKit.setJsonValue(data, "General", "dev_kit_mode", is_game_specific);
    windowsGuestRedZoneProtection.setJsonValue(data, "General",
                                               "windows_guest_red_zone_protection",
                                               is_game_specific);
    if (is_game_specific) {
        extraDmemInMbytes.setJsonValue(data, "General", "extra_dmem_in_mbytes", is_game_specific);
    }
    isConnectedToNetwork.setJsonValue(data, "General", "connected_to_network", is_game_specific);
    m_language.setJsonValue(data, "General", "console_language", is_game_specific);

    // ---- Input (shared schema) ----
    cursorState.setJsonValue(data, "Input", "cursor_state", is_game_specific);
    cursorHideTimeout.setJsonValue(data, "Input", "cursor_hide_timeout", is_game_specific);
    isMotionControlsEnabled.setJsonValue(data, "Input", "motion_controls_enabled",
                                         is_game_specific);
    backgroundControllerInput.setJsonValue(data, "Input", "background_controller_input",
                                           is_game_specific);
    usbDeviceBackend.setJsonValue(data, "Input", "usb_device_backend", is_game_specific);

    // ---- Audio (shared schema) ----
    micDevice.setJsonValue(data, "Audio", "sdl_mic_device", is_game_specific);
    mainOutputDevice.setJsonValue(data, "Audio", "sdl_main_output_device", is_game_specific);
    padSpkOutputDevice.setJsonValue(data, "Audio", "sdl_padSpk_output_device", is_game_specific);

    // ---- GPU (shared schema) ----
    windowWidth.setJsonValue(data, "GPU", "window_width", is_game_specific);
    windowHeight.setJsonValue(data, "GPU", "window_height", is_game_specific);
    isNullGpu.setJsonValue(data, "GPU", "null_gpu", is_game_specific);
    shouldCopyGPUBuffers.setJsonValue(data, "GPU", "copy_gpu_buffers", is_game_specific);
    readbacksModeEntry.setJsonValue(data, "GPU", "readbacks_mode", is_game_specific);
    readbackLinearImagesEnabled.setJsonValue(data, "GPU", "readback_linear_images_enabled",
                                             is_game_specific);
    directMemoryAccessEnabled.setJsonValue(data, "GPU", "direct_memory_access_enabled",
                                           is_game_specific);
    shouldDumpShaders.setJsonValue(data, "GPU", "dump_shaders", is_game_specific);
    vblankFrequency.setJsonValue(data, "GPU", "vblank_frequency", is_game_specific);
    isFullscreen.setJsonValue(data, "GPU", "full_screen", is_game_specific);
    fullscreenMode.setJsonValue(data, "GPU", "full_screen_mode", is_game_specific);
    presentMode.setJsonValue(data, "GPU", "present_mode", is_game_specific);
    isHDRAllowed.setJsonValue(data, "GPU", "hdr_allowed", is_game_specific);
    fsrEnabled.setJsonValue(data, "GPU", "fsr_enabled", is_game_specific);
    rcasEnabled.setJsonValue(data, "GPU", "rcas_enabled", is_game_specific);
    rcasAttenuation.setJsonValue(data, "GPU", "rcas_attenuation", is_game_specific);

    // ---- Vulkan (shared schema) ----
    gpuId.setJsonValue(data, "Vulkan", "gpu_id", is_game_specific);
    vkValidation.setJsonValue(data, "Vulkan", "vkvalidation_enabled", is_game_specific);
    vkValidationSync.setJsonValue(data, "Vulkan", "vkvalidation_sync_enabled", is_game_specific);
    vkValidationCore.setJsonValue(data, "Vulkan", "vkvalidation_core_enabled", is_game_specific);
    vkValidationGpu.setJsonValue(data, "Vulkan", "vkvalidation_gpu_enabled", is_game_specific);
    vkCrashDiagnostic.setJsonValue(data, "Vulkan", "vkcrash_diagnostic_enabled", is_game_specific);
    vkHostMarkers.setJsonValue(data, "Vulkan", "vkhost_markers", is_game_specific);
    vkGuestMarkers.setJsonValue(data, "Vulkan", "vkguest_markers", is_game_specific);
    rdocEnable.setJsonValue(data, "Vulkan", "renderdoc_enabled", is_game_specific);
    pipelineCacheEnable.setJsonValue(data, "Vulkan", "pipeline_cache_enabled", is_game_specific);
    pipelineCacheArchive.setJsonValue(data, "Vulkan", "pipeline_cache_archived", is_game_specific);

    // ---- Debug (shared schema) ----
    isDebugDump.setJsonValue(data, "Debug", "debug_dump", is_game_specific);
    isShaderDebug.setJsonValue(data, "Debug", "shader_collect", is_game_specific);

    // ---- Log (shared schema) ----
    logEnabled.setJsonValue(data, "Log", "enable", is_game_specific);
    isSeparateLogFilesEnabled.setJsonValue(data, "Log", "separate", is_game_specific);
    logFilter.setJsonValue(data, "Log", "filter", is_game_specific);
    // Log.sync mirrors gr2fork's logType ("sync" vs anything else). Compute the value
    // BEFORE the GR2Fork logType.setJsonValue below clears any game-specific override.
    {
        const string effective_log_type =
            is_game_specific ? logType.game_specific_value.value_or(logType.base_value)
                             : logType.base_value;
        data["Log"]["sync"] = (effective_log_type == "sync");
    }

    // ---- GR2Fork (gr2fork-only; preserved verbatim by the prerelease) ----
    isPSNSignedIn.setJsonValue(data, "GR2Fork", "isPSNSignedIn", is_game_specific);
    userName.setJsonValue(data, "GR2Fork", "userName", is_game_specific);
    httpHostOverride.setJsonValue(data, "GR2Fork", "httpHostOverride", is_game_specific);
    httpHostOverridePort.setJsonValue(data, "GR2Fork", "httpHostOverridePort", is_game_specific);
    httpForceHttp.setJsonValue(data, "GR2Fork", "httpForceHttp", is_game_specific);
    shadnetNpid.setJsonValue(data, "GR2Fork", "shadnetNpid", is_game_specific);
    shadnetPassword.setJsonValue(data, "GR2Fork", "shadnetPassword", is_game_specific);
    shadnetServer.setJsonValue(data, "GR2Fork", "shadnetServer", is_game_specific);
    logType.setJsonValue(data, "GR2Fork", "logType", is_game_specific);
    gyroSwapYawRoll.setJsonValue(data, "GR2Fork", "gyroSwapYawRoll", is_game_specific);
    gyroInvertYaw.setJsonValue(data, "GR2Fork", "gyroInvertYaw", is_game_specific);
    gyroInvertX.setJsonValue(data, "GR2Fork", "gyroInvertX", is_game_specific);
    gyroInvertRoll.setJsonValue(data, "GR2Fork", "gyroInvertRoll", is_game_specific);
    gameplaySyncBudgetMs.setJsonValue(data, "GR2Fork", "gameplaySyncBudgetMs", is_game_specific);
    aspectRatioOverride.setJsonValue(data, "GR2Fork", "aspectRatioOverride", is_game_specific);
    resolutionOverride.setJsonValue(data, "GR2Fork", "resolutionOverride", is_game_specific);
    resolutionPatchGroups.setJsonValue(data, "GR2Fork", "resolutionPatchGroups", is_game_specific);
    gr2SkippedShaders.setJsonValue(data, "GR2Fork", "gr2SkippedShaders", is_game_specific);
    resolutionPatchGroupsGrr.setJsonValue(data, "GR2Fork", "resolutionPatchGroupsGrr",
                                          is_game_specific);
    gr2FixNativeCubeViewsEnable.setJsonValue(data, "GR2Fork", "gr2FixNativeCubeViews",
                                             is_game_specific);
    gr2TitleThemeMod.setJsonValue(data, "GR2Fork", "GR2TitleThemeMod", is_game_specific);
    disableMotionBlurEnable.setJsonValue(data, "GR2Fork", "disableMotionBlur", is_game_specific);
    padSpkOutputDisabled.setJsonValue(data, "GR2Fork", "padSpkOutputDisabled", is_game_specific);
    vkForcePushDescriptors.setJsonValue(data, "GR2Fork", "vkForcePushDescriptors", is_game_specific);
    vkDisablePushDescriptors.setJsonValue(data, "GR2Fork", "vkDisablePushDescriptors",
                                          is_game_specific);
    beginRenderingCacheEnable.setJsonValue(data, "GR2Fork", "beginRenderingCacheEnable",
                                           is_game_specific);
    shDynamicDirtySkip.setJsonValue(data, "GR2Fork", "shDynamicDirtySkip", is_game_specific);
    pipelineUdHashLruEnable.setJsonValue(data, "GR2Fork", "pipelineUdHashLruEnable",
                                         is_game_specific);
    pipelineSpecFpLruEnable.setJsonValue(data, "GR2Fork", "pipelineSpecFpLruEnable",
                                         is_game_specific);
    pipelineGfxKeyCtxSkipEnable.setJsonValue(data, "GR2Fork", "pipelineGfxKeyCtxSkipEnable",
                                             is_game_specific);
    descSetBindingSkipCache.setJsonValue(data, "GR2Fork", "descSetBindingSkipCache",
                                         is_game_specific);
    accurateRenderTargetCache.setJsonValue(data, "GR2Fork", "accurateRenderTargetCache",
                                           is_game_specific);
    accurateVertexBufferCache.setJsonValue(data, "GR2Fork", "accurateVertexBufferCache",
                                           is_game_specific);

    if (!is_game_specific) {
        // Install dirs as an array of {path, enabled}, alphabetically ordered (matches
        // the prerelease GameInstallDir json and the prior toml ordering behavior).
        std::vector<GameInstallDir> sorted_dirs = settings_install_dirs;
        std::sort(sorted_dirs.begin(), sorted_dirs.end(),
                  [](const GameInstallDir& a, const GameInstallDir& b) {
                      const auto sa = string{fmt::UTF(a.path.u8string()).data};
                      const auto sb = string{fmt::UTF(b.path.u8string()).data};
                      return std::lexicographical_compare(
                          sa.begin(), sa.end(), sb.begin(), sb.end(),
                          [](char x, char y) { return std::tolower(x) < std::tolower(y); });
                  });
        json install_dirs_array = json::array();
        for (const auto& dirInfo : sorted_dirs) {
            json entry = json::object();
            entry["path"] = dirInfo.path;
            entry["enabled"] = dirInfo.enabled;
            install_dirs_array.push_back(entry);
        }
        data["General"]["install_dirs"] = install_dirs_array;

        data["General"]["discord_rpc_enabled"] = enableDiscordRPC;
        data["General"]["sys_modules_dir"] = sys_modules_path;
        data["General"]["addon_install_dir"] = settings_addon_install_dir;
        data["General"]["home_dir"] = home_dir;
        data["General"]["show_fps_counter"] = showFpsCounter.base_value;
        data["Debug"]["config_version"] = config_version;

        // Not present in the game-specific dialog.
        data["Input"]["default_controller_id"] = defaultControllerID.base_value;
        data["Input"]["use_special_pad"] = useSpecialPad.base_value;
        data["Input"]["special_pad_class"] = specialPadClass.base_value;
        data["Input"]["use_unified_input_config"] = useUnifiedInputConfig.base_value;
        data["GPU"]["internal_screen_width"] = internalScreenWidth.base_value;
        data["GPU"]["internal_screen_height"] = internalScreenHeight.base_value;
        data["GPU"]["patch_shaders"] = shouldPatchShaders.base_value;
        data["GR2Fork"]["fpsColor"] = isFpsColor.base_value;
        data["GR2Fork"]["trophyKey"] = trophyKey;
    }

    // Merge into the existing file, preserving unknown sections/keys. This is how the
    // prerelease keeps gr2fork's GR2Fork section, and how gr2fork keeps prerelease-only
    // keys (General.shad_net_enabled, Audio.openal_*, Log.append, etc.).
    json existing = json::object();
    {
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            try {
                std::ifstream in(path);
                if (in.good()) {
                    in >> existing;
                }
            } catch (const std::exception&) {
                existing = json::object();
            }
        }
        if (!existing.is_object()) {
            existing = json::object();
        }
    }

    for (auto& [section, val] : data.items()) {
        if (existing.contains(section) && existing[section].is_object() && val.is_object()) {
            existing[section].update(val);
        } else {
            existing[section] = val;
        }
    }

    try {
        std::ofstream file(path);
        file << existing.dump(2);
        file.close();
    } catch (const std::exception& ex) {
        fmt::print("Exception trying to write json config file. Exception: {}\n", ex.what());
    }
}

void setDefaultValues(bool is_game_specific) {

    // Entries with game-specific settings that are in the game-specific setings GUI but not in
    // the global settings GUI
    if (is_game_specific) {
        readbacksModeEntry.set(GpuReadbacksMode::Disabled, is_game_specific);
        readbackLinearImagesEnabled.set(false, is_game_specific);
        isNeo.set(false, is_game_specific);
        isDevKit.set(false, is_game_specific);
        windowsGuestRedZoneProtection.set(false, is_game_specific);
        isPSNSignedIn.set(false, is_game_specific);
        isConnectedToNetwork.set(false, is_game_specific);
        directMemoryAccessEnabled.set(false, is_game_specific);
        extraDmemInMbytes.set(0, is_game_specific);
    }

    // Entries with game-specific settings that are in both the game-specific and global GUI
    // GS - General
    volumeSlider.set(100, is_game_specific);
    isTrophyPopupDisabled.set(false, is_game_specific);
    trophyNotificationDuration.set(6.0, is_game_specific);
    logFilter.set("", is_game_specific);
    logType.set("sync", is_game_specific);
    userName.set("shadPS4", is_game_specific);
    httpHostOverride.set("localhost", is_game_specific);
    httpHostOverridePort.set(8443, is_game_specific);
    httpForceHttp.set(true, is_game_specific);
    shadnetNpid.set("", is_game_specific);
    shadnetPassword.set("", is_game_specific);
    shadnetServer.set("srv.shadps4.net:31313", is_game_specific);
    isShowSplash.set(false, is_game_specific);
    isSideTrophy.set("right", is_game_specific);

    // GS - Input
    cursorState.set(HideCursorState::Idle, is_game_specific);
    cursorHideTimeout.set(5, is_game_specific);
    isMotionControlsEnabled.set(true, is_game_specific);
    gyroSwapYawRoll.set(false, is_game_specific);
    gyroInvertYaw.set(false, is_game_specific);
    gyroInvertX.set(false, is_game_specific);
    gyroInvertRoll.set(false, is_game_specific);
    backgroundControllerInput.set(false, is_game_specific);
    usbDeviceBackend.set(UsbBackendType::Real, is_game_specific);

    // GS - Audio
    micDevice.set("Default Device", is_game_specific);

    // GS - GPU
    windowWidth.set(1280, is_game_specific);
    windowHeight.set(720, is_game_specific);
    isNullGpu.set(false, is_game_specific);
    shouldCopyGPUBuffers.set(false, is_game_specific);
    shouldDumpShaders.set(false, is_game_specific);
    vblankFrequency.set(60, is_game_specific);
    isFullscreen.set(false, is_game_specific);
    fullscreenMode.set("Windowed", is_game_specific);
    presentMode.set("Mailbox", is_game_specific);
    isHDRAllowed.set(false, is_game_specific);
    fsrEnabled.set(true, is_game_specific);
    rcasEnabled.set(true, is_game_specific);
    rcasAttenuation.set(250, is_game_specific);
    gameplaySyncBudgetMs.set(100, is_game_specific);
    aspectRatioOverride.set("16:9", is_game_specific);
    resolutionOverride.set("Off", is_game_specific);
    resolutionPatchGroups.set("recommended", is_game_specific);
    resolutionPatchGroupsGrr.set("recommended", is_game_specific);
    gr2SkippedShaders.set("", is_game_specific);
    rebuildGr2ShaderSkipCache();

    // GR2FORK: reset-to-default re-enables every optimization
    // and fix and turns the GCN compat master off. Polaris users
    // disable individual toggles from here, never the other way around.
    gr2FixNativeCubeViewsEnable.set(true, is_game_specific);
    gr2TitleThemeMod.set(false, is_game_specific); // GR2FORK: title-theme replacement off
    disableMotionBlurEnable.set(false, is_game_specific);
    padSpkOutputDisabled.set(true, is_game_specific);

    // GS - Vulkan
    gpuId.set(-1, is_game_specific);
    vkValidation.set(false, is_game_specific);
    vkValidationCore.set(true, is_game_specific);
    vkValidationSync.set(false, is_game_specific);
    vkValidationGpu.set(false, is_game_specific);
    vkCrashDiagnostic.set(false, is_game_specific);
    vkHostMarkers.set(false, is_game_specific);
    vkGuestMarkers.set(false, is_game_specific);
    rdocEnable.set(false, is_game_specific);
    pipelineCacheEnable.set(false, is_game_specific);
    pipelineCacheArchive.set(false, is_game_specific);
    vkForcePushDescriptors.set(false, is_game_specific);
    vkDisablePushDescriptors.set(false, is_game_specific);
    beginRenderingCacheEnable.set(true, is_game_specific);
    shDynamicDirtySkip.set(true, is_game_specific);
    pipelineUdHashLruEnable.set(true, is_game_specific);
    pipelineSpecFpLruEnable.set(true, is_game_specific);
    pipelineGfxKeyCtxSkipEnable.set(true, is_game_specific);
    descSetBindingSkipCache.set(true, is_game_specific);
    accurateRenderTargetCache.set(false, is_game_specific);
    accurateVertexBufferCache.set(false, is_game_specific);

    // GS - Debug
    isDebugDump.set(false, is_game_specific);
    isShaderDebug.set(false, is_game_specific);
    isSeparateLogFilesEnabled.set(false, is_game_specific);
    logEnabled.set(true, is_game_specific);

    // GS - Settings
    m_language.set(1, is_game_specific);

    // All other entries
    if (!is_game_specific) {

        // General
        enableDiscordRPC = false;

        // Input
        useSpecialPad.base_value = false;
        specialPadClass.base_value = 1;
        useUnifiedInputConfig.base_value = true;
        controllerCustomColorRGB[0] = 0;
        controllerCustomColorRGB[1] = 0;
        controllerCustomColorRGB[2] = 255;

        // TODO: Change to be game specific
        mainOutputDevice = "Default Device";
        padSpkOutputDevice = "Default Device";

        // GPU
        shouldPatchShaders.base_value = false;
        internalScreenWidth.base_value = 1280;
        internalScreenHeight.base_value = 720;

        // Debug
        isFpsColor.base_value = true;
        showFpsCounter.base_value = false;
    }
}

constexpr std::string_view GetDefaultGlobalConfig() {
    return R"(# Anything put here will be loaded for all games,
# alongside the game's config or default.ini depending on your preference.

hotkey_renderdoc_capture = f12
hotkey_fullscreen = f11
hotkey_show_fps = f10
hotkey_pause = f9
hotkey_reload_inputs = f8
hotkey_toggle_mouse_to_joystick = f7
hotkey_toggle_mouse_to_gyro = f6
hotkey_toggle_mouse_to_touchpad = delete
hotkey_quit = lctrl, lshift, end
)";
}

constexpr std::string_view GetDefaultInputConfig() {
    return R"(#Feeling lost? Check out the Help section!

# Keyboard bindings

triangle = kp8
circle = kp6
cross = kp2
square = kp4
# Alternatives for users without a keypad
triangle = c
circle = b
cross = n
square = v

l1 = q
r1 = u
l2 = e
r2 = o
l3 = x
r3 = m

options = enter
touchpad_center = space

pad_up = up
pad_down = down
pad_left = left
pad_right = right

axis_left_x_minus = a
axis_left_x_plus = d
axis_left_y_minus = w
axis_left_y_plus = s

axis_right_x_minus = j
axis_right_x_plus = l
axis_right_y_minus = i
axis_right_y_plus = k

# Controller bindings

triangle = triangle
cross = cross
square = square
circle = circle

l1 = l1
l2 = l2
l3 = l3
r1 = r1
r2 = r2
r3 = r3

options = options
touchpad_center = back

pad_up = pad_up
pad_down = pad_down
pad_left = pad_left
pad_right = pad_right

axis_left_x = axis_left_x
axis_left_y = axis_left_y
axis_right_x = axis_right_x
axis_right_y = axis_right_y

# Range of deadzones: 1 (almost none) to 127 (max)
analog_deadzone = leftjoystick, 2, 127
analog_deadzone = rightjoystick, 2, 127

override_controller_color = false, 0, 0, 255

# Hold-combo touchpad swipes: while the hold input is held, the four combo inputs play back touchpad swipes instead of their normal actions
touchpad_swipe_combo_enabled = false
touchpad_swipe_combo_hold = l3
touchpad_swipe_combo_up = triangle
touchpad_swipe_combo_down = cross
touchpad_swipe_combo_left = square
touchpad_swipe_combo_right = circle
touchpad_swipe_combo_hold_passthrough = false
)";
}
std::filesystem::path GetFoolproofInputConfigFile(const string& game_id) {
    // Read configuration file of the game, and if it doesn't exist, generate it from default
    // If that doesn't exist either, generate that from getDefaultConfig() and try again
    // If even the folder is missing, we start with that.

    const auto config_dir = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "input_config";
    const auto config_file = config_dir / (game_id + ".ini");
    const auto default_config_file = config_dir / "default.ini";

    // Ensure the config directory exists
    if (!std::filesystem::exists(config_dir)) {
        std::filesystem::create_directories(config_dir);
    }

    // Check if the default config exists
    if (!std::filesystem::exists(default_config_file)) {
        // If the default config is also missing, create it from getDefaultConfig()
        const auto default_config = GetDefaultInputConfig();
        std::ofstream default_config_stream(default_config_file);
        if (default_config_stream) {
            default_config_stream << default_config;
        }
    }

    // if empty, we only need to execute the function up until this point
    if (game_id.empty()) {
        return default_config_file;
    }

    // Create global config if it doesn't exist yet
    if (game_id == "global" && !std::filesystem::exists(config_file)) {
        if (!std::filesystem::exists(config_file)) {
            const auto global_config = GetDefaultGlobalConfig();
            std::ofstream global_config_stream(config_file);
            if (global_config_stream) {
                global_config_stream << global_config;
            }
        }
    }

    // If game-specific config doesn't exist, create it from the default config
    if (!std::filesystem::exists(config_file)) {
        std::filesystem::copy(default_config_file, config_file);
    }
    return config_file;
}

void resetGameSpecificValue(std::string entry) {
    if (entry == "volumeSlider") {
        volumeSlider.game_specific_value = std::nullopt;
    }
}

} // namespace Config
