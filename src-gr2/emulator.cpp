// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <fmt/core.h>
#include <fmt/xchar.h>
#include <hwinfo/hwinfo.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string_view>
#include "common/config.h"
#include "common/crash_handler.h"
#include "common/debug.h"
#include "common/hang_watchdog.h"
#include "common/logging/backend.h"
#include "common/logging/log.h"
#include "core/ipc/ipc.h"
#ifdef ENABLE_DISCORD_RPC
#include "common/discord_rpc_handler.h"
#endif
#include "common/elf_info.h"
#include "common/memory_patcher.h"
#include "common/ntapi.h"
#include "common/path_util.h"
#include "common/polyfill_thread.h"
#include "common/scm_rev.h"
#include "common/singleton.h"
#include "common/thread.h"
#include "core/debugger.h"
#include "core/devtools/widget/module_list.h"
#include "core/file_format/npbind.h"
#include "core/file_format/psf.h"
#include "core/file_format/trp.h"
#include "core/file_sys/fs.h"
#include "core/libraries/disc_map/disc_map.h"
#include "core/libraries/font/font.h"
#include "core/libraries/font/fontft.h"
#include "core/libraries/jpeg/jpegenc.h"
#include "core/libraries/libc_internal/libc_internal.h"
#include "core/libraries/libs.h"
#include "core/libraries/ngs2/ngs2.h"
#include "core/libraries/np/np_trophy.h"
#include "core/user_settings.h"
#include "core/libraries/rtc/rtc.h"
#include "core/libraries/save_data/save_backup.h"
// FIX(GR2FORK v3.1): pulled in for the window-size-follows-resolutionOverride
// path below. The window dimensions need to be computed from the resolution
// patch target BEFORE WindowSDL is constructed, so we need to peek at the
// same enums/helpers that module.cpp later passes to ApplyGr2ResolutionPatches.
#include "core/libraries/aspect_patches/aspect_patches.h"
#include "core/libraries/resolution_patches/resolution_patches.h"
#include "core/linker.h"
#include "core/memory.h"
#include "emulator.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/cache_storage.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

// FIX(GR2FORK): globals owned by gnmdriver.cpp (presenter) and videoout (liverpool).
// Declared extern here so the hang watchdog can sample them through its
// null-safe callbacks. These unique_ptrs are lazily populated when the game
// calls sceGnmDriverInitializer; callbacks below tolerate null.
namespace Vulkan { class Presenter; }
namespace AmdGpu  { struct Liverpool; }
extern std::unique_ptr<Vulkan::Presenter>  presenter;
extern std::unique_ptr<AmdGpu::Liverpool>  liverpool;

#ifdef _WIN32
#include <WinSock2.h>
#endif

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

Frontend::WindowSDL* g_window = nullptr;

namespace Core {

Emulator::Emulator() {
    // Initialize NT API functions, set high priority and disable WER.
    //
    // FIX(GR2FORK v3): CrashHandler::Install() was previously called HERE,
    // before SEM_NOGPFAULTERRORBOX, on the theory that early WER suppression
    // would otherwise lose silent deaths. In practice this ordering meant
    // every LOG_INFO emitted by Install() (including the "[CrashHandler]
    // installed: ..." banner that reports which inline hooks took) ran
    // before Common::Log::Initialize() and got dropped by
    // initialization_in_progress_suppress_logging — leaving zero evidence
    // in the log of whether handlers/hooks were live for the run. Install()
    // now runs in Run() right after Common::Log::Start(), so the banner
    // actually reaches shad_log.txt. The window between SetErrorMode here
    // and Install() in Run() is only the deterministic startup path
    // (NtApi/WSA init, param.sfo parse, Config::load) — no game code runs
    // there, so the loss of WER coverage in that window is inert.
#ifdef _WIN32
    Common::NtApi::Initialize();
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
    SetErrorMode(SetErrorMode(0) | SEM_NOGPFAULTERRORBOX);
    // need to init this in order for winsock2 to work
    WORD versionWanted = MAKEWORD(2, 2);
    WSADATA wsaData;
    WSAStartup(versionWanted, &wsaData);
#endif
}

Emulator::~Emulator() {}

// GR2FORK: extract a game's trophies into the shared user/trophy/<npcommid> tree
// (definition + icons) and seed each user's home/<uid>/trophy/<npcommid>.xml from
// TROPCONF.XML, matching the prerelease layout. Returns a trophy-index -> npcommid
// map (stored on the ElfInfo) so np_trophy can resolve the npcommid from the
// service label at sceNpTrophyCreateContext. The trophy key comes from KeyManager
// (keys.json) via TRP::Extract.
static std::map<s32, std::string> ExtractTrophies(const std::filesystem::path& npbind_path,
                                                  const std::filesystem::path& trophy_dir) {
    std::map<s32, std::string> trophy_index_map{};

    NPBindFile npbind;
    if (!npbind.Load(npbind_path.string())) {
        LOG_WARNING(Common_Filesystem, "Failed to load npbind.dat file");
        return trophy_index_map;
    }

    auto np_comm_ids = npbind.GetNpCommIds();
    if (np_comm_ids.empty()) {
        LOG_WARNING(Common_Filesystem, "No NPCommIDs in npbind.dat");
        return trophy_index_map;
    }

    if (!std::filesystem::exists(trophy_dir)) {
        LOG_WARNING(Common_Filesystem, "Game does not contain a trophy directory");
        return trophy_index_map;
    }

    const std::string pattern = "trophy";
    for (const auto& entry : std::filesystem::directory_iterator(trophy_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".trp") {
            continue;
        }
        std::string filename = entry.path().stem().string(); // "trophy00", "trophy01", etc.
        if (filename.find(pattern) != 0) {
            continue;
        }

        const std::string num_str = filename.substr(pattern.length());
        s32 trophy_index = std::stoi(num_str);

        // This currently assumes the order of NPCommIDs matches the order of trophies.
        if (np_comm_ids.size() <= static_cast<size_t>(trophy_index)) {
            LOG_WARNING(Common_Filesystem,
                        "Trophy index {} does not have a corresponding NPCommId", trophy_index);
            continue;
        }

        const std::string np_comm_id = np_comm_ids[trophy_index];
        trophy_index_map[trophy_index] = np_comm_id;
        LOG_DEBUG(Loader, "Mapped trophy index {} to NPCommID: {}", trophy_index, np_comm_id);

        // Extract the trophy assets + definition once into the shared tree.
        const auto trophy_output_dir =
            Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "trophy" / np_comm_id;
        if (!std::filesystem::exists(trophy_output_dir)) {
            TRP trp;
            if (!trp.Extract(entry.path(), np_comm_id, trophy_output_dir)) {
                LOG_ERROR(Loader, "Couldn't extract trophy file {}", filename);
                continue;
            }
        }

        // Seed each user's per-user progress file from the trophy config.
        for (const User& user : UserSettings.GetUserManager().GetValidUsers()) {
            const auto user_trophy_file = Config::GetHomeDir() / std::to_string(user.user_id) /
                                          "trophy" / (np_comm_id + ".xml");
            if (!std::filesystem::exists(user_trophy_file)) {
                std::error_code ec;
                std::filesystem::create_directories(user_trophy_file.parent_path(), ec);
                std::filesystem::copy_file(trophy_output_dir / "Xml" / "TROPCONF.XML",
                                           user_trophy_file, ec);
            }
        }
    }
    return trophy_index_map;
}

void Emulator::Run(std::filesystem::path file, std::vector<std::string> args,
                   std::optional<std::filesystem::path> p_game_folder) {
    if (waitForDebuggerBeforeRun) {
        Debugger::WaitForDebuggerAttach();
    }

    if (std::filesystem::is_directory(file)) {
        file /= "eboot.bin";
    }

    std::filesystem::path game_folder;
    if (p_game_folder.has_value()) {
        game_folder = p_game_folder.value();
    } else {
        game_folder = file.parent_path();
        if (const auto game_folder_name = game_folder.filename().string();
            game_folder_name.ends_with("-UPDATE") || game_folder_name.ends_with("-patch")) {
            // If an executable was launched from a separate update directory,
            // use the base game directory as the game folder.
            const std::string base_name = game_folder_name.substr(0, game_folder_name.rfind('-'));
            const auto base_path = game_folder.parent_path() / base_name;
            if (std::filesystem::is_directory(base_path)) {
                game_folder = base_path;
            }
        }
    }

    std::filesystem::path eboot_name = std::filesystem::relative(file, game_folder);

    // Applications expect to be run from /app0 so mount the file's parent path as app0.
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    mnt->Mount(game_folder, "/app0", true);
    // Certain games may use /hostapp as well such as CUSA001100
    mnt->Mount(game_folder, "/hostapp", true);

    const auto param_sfo_path = mnt->GetHostPath("/app0/sce_sys/param.sfo");
    const auto param_sfo_exists = std::filesystem::exists(param_sfo_path);

    // Load param.sfo details if it exists
    std::string id;
    std::string title;
    std::string app_version;
    u32 sdk_version;
    u32 fw_version;
    Common::PSFAttributes psf_attributes{};
    if (param_sfo_exists) {
        auto* param_sfo = Common::Singleton<PSF>::Instance();
        ASSERT_MSG(param_sfo->Open(param_sfo_path), "Failed to open param.sfo");

        const auto content_id = param_sfo->GetString("CONTENT_ID");
        const auto title_id = param_sfo->GetString("TITLE_ID");
        if (content_id.has_value() && !content_id->empty()) {
            id = std::string(*content_id, 7, 9);
        } else if (title_id.has_value()) {
            id = *title_id;
        }
        title = param_sfo->GetString("TITLE").value_or("Unknown title");
        fw_version = param_sfo->GetInteger("SYSTEM_VER").value_or(0x4700000);
        app_version = param_sfo->GetString("APP_VER").value_or("Unknown version");
        if (const auto raw_attributes = param_sfo->GetInteger("ATTRIBUTE")) {
            psf_attributes.raw = *raw_attributes;
        }

        // Extract sdk version from pubtool info.
        std::string_view pubtool_info =
            param_sfo->GetString("PUBTOOLINFO").value_or("Unknown value");
        u64 sdk_ver_offset = pubtool_info.find("sdk_ver");

        if (sdk_ver_offset == pubtool_info.npos) {
            // Default to using firmware version if SDK version is not found.
            sdk_version = fw_version;
        } else {
            // Increment offset to account for sdk_ver= part of string.
            sdk_ver_offset += 8;
            u64 sdk_ver_len = pubtool_info.find(",", sdk_ver_offset);
            if (sdk_ver_len == pubtool_info.npos) {
                // If there's no more commas, this is likely the last entry of pubtool info.
                // Use string length instead.
                sdk_ver_len = pubtool_info.size();
            }
            sdk_ver_len -= sdk_ver_offset;
            std::string sdk_ver_string = pubtool_info.substr(sdk_ver_offset, sdk_ver_len).data();
            // Number is stored in base 16.
            sdk_version = std::stoi(sdk_ver_string, nullptr, 16);
        }
    }

    auto& game_info = Common::ElfInfo::Instance();
    game_info.initialized = true;
    game_info.game_serial = id;
    game_info.title = title;
    game_info.app_ver = app_version;
    game_info.firmware_ver = fw_version & 0xFFF00000;
    game_info.raw_firmware_ver = fw_version;
    game_info.sdk_ver = sdk_version;
    game_info.psf_attributes = psf_attributes;

    const auto pic1_path = mnt->GetHostPath("/app0/sce_sys/pic1.png");
    if (std::filesystem::exists(pic1_path)) {
        game_info.splash_path = pic1_path;
    }

    game_info.game_folder = game_folder;

    Config::load(Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs) / (id + ".json"),
                 true);

    // Initialize logging as soon as possible
    if (!id.empty() && Config::getSeparateLogFilesEnabled()) {
        Common::Log::Initialize(id + ".log");
    } else {
        Common::Log::Initialize();
    }
    Common::Log::Start();
    // FIX(GR2FORK v3): install crash handler immediately after the log
    // backend is up. Earlier (constructor-time) installs ran before
    // Log::Initialize and the install banner was silently dropped by
    // initialization_in_progress_suppress_logging — there was no way to
    // verify post-mortem whether terminate/SEH/VEH/atexit/at_quick_exit/
    // inv_param/purecall handlers and the four inline hooks (ExitProcess,
    // TerminateProcess, RtlExitUserProcess, NtTerminateProcess) actually
    // installed for that run. With this placement the
    // "[CrashHandler] installed: ... hooks: ..." line now lands in
    // shad_log.txt as the first non-initialization log entry, and any
    // subsequent quick_exit/exit/SEH below this point is captured.
    //
    // The eboot.bin existence check just below uses std::quick_exit(0)
    // on failure — that path now correctly trips OnAtQuickExit. As of
    // this patch, no SignalCleanShutdown call precedes it, so it will
    // be logged as a stack-walked crash entry. That matches the prior
    // behavior of the constructor-time install (the handler was armed
    // either way) and is fine: a missing eboot is an error condition
    // worth a stack walk anyway.
    Common::CrashHandler::Install();
    if (!std::filesystem::exists(file)) {
        LOG_CRITICAL(Loader, "eboot.bin does not exist: {}",
                     std::filesystem::absolute(file).string());
        std::quick_exit(0);
    }

    LOG_INFO(Loader, "Starting gr2fork (shadps4 v{} base)", Common::g_version);
    LOG_INFO(Loader, "Fork: Gravity Rush 2 focus");
    LOG_INFO(Loader, "Build: v4.8.002");

    const bool has_game_config = std::filesystem::exists(
        Common::FS::GetUserPath(Common::FS::PathType::CustomConfigs) / (id + ".json"));
    LOG_INFO(Config, "Game-specific config exists: {}", has_game_config);

    LOG_INFO(Config, "General LogType: {}", Config::getLogType());
    LOG_INFO(Config, "General isNeo: {}", Config::isNeoModeConsole());
    LOG_INFO(Config, "General isDevKit: {}", Config::isDevKitConsole());
    LOG_INFO(Config, "General isConnectedToNetwork: {}", Config::getIsConnectedToNetwork());
    LOG_INFO(Config, "General isPsnSignedIn: {}", Config::getPSNSignedIn());
    LOG_INFO(Config, "GPU isNullGpu: {}", Config::nullGpu());
    LOG_INFO(Config, "GPU readbacks: {}", Config::readbacks());
    LOG_INFO(Config, "GPU readbackLinearImages: {}", Config::readbackLinearImages());
    LOG_INFO(Config, "GPU directMemoryAccess: {}", Config::directMemoryAccess());
    LOG_INFO(Config, "GPU shouldDumpShaders: {}", Config::dumpShaders());
    LOG_INFO(Config, "GPU vblankFrequency: {}", Config::vblankFreq());
    LOG_INFO(Config, "GPU shouldCopyGPUBuffers: {}", Config::copyGPUCmdBuffers());
    LOG_INFO(Config, "Vulkan gpuId: {}", Config::getGpuId());
    LOG_INFO(Config, "Vulkan vkValidation: {}", Config::vkValidationEnabled());
    LOG_INFO(Config, "Vulkan vkValidationCore: {}", Config::vkValidationCoreEnabled());
    LOG_INFO(Config, "Vulkan vkValidationSync: {}", Config::vkValidationSyncEnabled());
    LOG_INFO(Config, "Vulkan vkValidationGpu: {}", Config::vkValidationGpuEnabled());
    LOG_INFO(Config, "Vulkan crashDiagnostics: {}", Config::getVkCrashDiagnosticEnabled());
    LOG_INFO(Config, "Vulkan hostMarkers: {}", Config::getVkHostMarkersEnabled());
    LOG_INFO(Config, "Vulkan guestMarkers: {}", Config::getVkGuestMarkersEnabled());
    LOG_INFO(Config, "Vulkan rdocEnable: {}", Config::isRdocEnabled());
    // GR2FORK 2026-06-15: the v4.5 assembler-perf gates are FORCED ON (env-only
    // kills now; no config keys). The legacy [Vulkan] keys stay ignored (see the
    // block in config.cpp). br_cache and descSetBindingSkipCache were already
    // env-only — config-disabling br_cache is banned per project rule, and the
    // BindingSkipCache machinery was removed by BSC-STRIP so its getter is
    // echo-only. All are echoed here so the active state — including any GR2_NO*
    // env kill switch — is unambiguous in every log. The shad_log incident
    // (ctx_stable=0, STAGE-MEMO dead) was only caught because D3' was echoed.
    LOG_INFO(Config, "Vulkan shDynamicDirtySkip (D3'): {} (forced on; GR2_NOSHDYNSKIP)",
             Config::isShDynamicDirtySkipEnabled());
    LOG_INFO(Config, "Vulkan descSetBindingSkipCache: {} (forced on; GR2_NODESCSKIP)",
             Config::isDescSetBindingSkipCacheEnabled());
    LOG_INFO(Config, "Vulkan beginRenderingCacheEnable: {} (forced on; GR2_NOBRCACHE)",
             Config::isBeginRenderingCacheEnabled());
    LOG_INFO(Config, "Vulkan pipelineUdHashLruEnable: {} (forced on; GR2_NOUDHASHLRU)",
             Config::isPipelineUdHashLruEnabled());
    LOG_INFO(Config, "Vulkan pipelineSpecFpLruEnable: {} (forced on; GR2_NOSPECFPLRU)",
             Config::isPipelineSpecFpLruEnabled());
    LOG_INFO(Config, "Vulkan pipelineGfxKeyCtxSkipEnable: {} (forced on; GR2_NOKEYCTXSKIP)",
             Config::isPipelineGfxKeyCtxSkipEnabled());
    // GR2FORK: the cubemap fix is the sole config-controlled GR2 toggle (it is
    // the one v4.5 change confirmed to cause GCN-era graphical regressions).
    // Every other v4.5 fps optimization and GCN-accuracy fix is forced on and
    // echoes individually from its own consuming-site ctor with its GR2_NO*
    // kill; this central note records that those levers are no longer
    // config-switchable.
    LOG_INFO(Config, "GR2 cubemap fix nativeCubeViews: {} (config [GR2Fork]; GR2_NOCUBEVIEW)",
             Config::gr2FixNativeCubeViews());
    LOG_INFO(Config, "GR2 v4.5 opts/fixes: forced on (env-only kills, echoed per consuming site)");

    hwinfo::Memory ram;
    hwinfo::OS os;
    const auto cpus = hwinfo::getAllCPUs();
    for (const auto& cpu : cpus) {
        LOG_INFO(Config, "CPU Model: {}", cpu.modelName());
        LOG_INFO(Config, "CPU Physical Cores: {}, Logical Cores: {}", cpu.numPhysicalCores(),
                 cpu.numLogicalCores());
    }
    LOG_INFO(Config, "Total RAM: {} GB", std::round(ram.total_Bytes() / pow(1024, 3)));
    LOG_INFO(Config, "Operating System: {}", os.name());

    if (param_sfo_exists) {
        LOG_INFO(Loader, "Game id: {} Title: {}", id, title);

        // HARD OVERRIDE for Gravity Rush Remastered (all known SKUs).
        // rt_cache_ in PrepareRenderState hashes render-target content but not
        // ImageId identity. When TextureCache invalidates / recreates an image
        // at the same VAddr, rt_hash still matches the prior call and the
        // cache returns a stale image_id -> draw goes into the wrong / freed
        // image (visible as shadow flicker on this title). Force the accurate
        // path on regardless of user config.
        static constexpr std::array<std::string_view, 12> gr_remastered_ids = {
            "CUSA01112", "CUSA01113", "CUSA01113P", "CUSA01130",
            "CUSA02318", "CUSA00546", "CUSA02443",  "CUSA04246",
            "PCJS50004", "PCJS50008", "PCJS66015",  "PCJS66029",
        };
        const bool is_gr_remastered =
            std::find(gr_remastered_ids.begin(), gr_remastered_ids.end(), id) !=
            gr_remastered_ids.end();
        if (is_gr_remastered) {
            LOG_INFO(Loader,
                     "Gravity Rush Remastered ({}) detected -> forcing "
                     "accurateRenderTargetCache + accurateVertexBufferCache on for "
                     "this session.",
                     id);
        }

        // GR2FORK: inFAMOUS Second Son SKU detection (disjoint from the GRR list
        // above). ISS gets the same per-title shims as GRR, applied below:
        //   * accurateRenderTargetCache + accurateVertexBufferCache — ISS
        //     exhibits the SAME stale-image_id shadow flicker GRR does (rt_cache_
        //     in PrepareRenderState keys on render-target content, not ImageId
        //     identity, so a recreated image at the same VAddr returns a stale
        //     id), so force both accurate-cache shims ON for ISS too.
        //   * deeper BundleAssembler intent queue — ISS bursts far more draw
        //     intents per frame than the 256-slot default can hold (the SPSC
        //     ring asserts "intent queue full"); the per-session flag below
        //     selects kQueueSizeIss (4096, sized from the measured peak of 1822).
        static constexpr std::array<std::string_view, 5> infamous_second_son_ids = {
            "CUSA00004", "CUSA00223", "CUSA00263", "CUSA00305", "CUSA00309",
        };
        const bool is_infamous_second_son =
            std::find(infamous_second_son_ids.begin(), infamous_second_son_ids.end(), id) !=
            infamous_second_son_ids.end();
        if (is_infamous_second_son) {
            LOG_INFO(Loader,
                     "inFAMOUS Second Son ({}) detected -> forcing "
                     "accurateRenderTargetCache + accurateVertexBufferCache on "
                     "(shadow-flicker fix) + deepening BundleAssembler intent queue "
                     "(256 -> 4096, sized to ~2.2x the measured peak of 1822).",
                     id);
        }

        // Symmetric per-title cache policy from a single source of truth, forced
        // over global config / per-game TOML / the GUI. Both accurate-cache shims
        // (accurateRenderTargetCache — the stale-image_id shadow-flicker fix —
        // and accurateVertexBufferCache) are forced ON for GRR and ISS; every
        // other title gets both hard-forced OFF (a stray saved 'true' would
        // otherwise enable a render-target path it never wanted).
        Config::setGameSpecificCacheToggles(
            /*force_rt_cache=*/is_gr_remastered || is_infamous_second_son,
            /*force_vb_cache=*/is_gr_remastered || is_infamous_second_son);
        // Record the per-session SKU flags for GPU-side consumers that size
        // structures differently (e.g. the GpuComm snapshot pool / the
        // bundle-assembler queue depth).
        Config::setIsGravityRushRemastered(is_gr_remastered);
        Config::setIsInfamousSecondSon(is_infamous_second_son);
        // GR2FORK GpuComm scheduling fix for Gravity Rush Remastered.
        // ------------------------------------------------------------
        // On a 4-physical-core host, reserving two cores for GpuComm + the
        // GpuAssembler leaves the guest and the async pipeline-compile workers
        // sharing only two cores. During a GRR compile burst (level load /
        // shader pop-in) those workers are fenced off BOTH hot cores while
        // GpuComm spins in the snapshot-pool backpressure loop and the assembler
        // is parked waiting on a compile — so the burst has nowhere to run and
        // the frame drops to 0 FPS for seconds. This does NOT happen on >= 8-core
        // hosts (the workers have spare cores) or in GR2 (lighter shader churn),
        // and it reproduces on an 8-core part with 4 cores disabled in BIOS.
        //
        // Default fix on a 4-physical-core GRR session: NON-EXCLUSIVE pinning.
        // GpuComm and the assembler keep their pins (they still run on their own
        // cores) but those cores are no longer isolated from everything else, so
        // the scheduler can reclaim them for the compile burst the moment the hot
        // threads go idle. The async pipeline is otherwise unchanged. SMT thread
        // count is irrelevant: 4c/8t and 4c/4t both report 4 physical cores.
        //
        // The mode is derived purely from host topology + the detected title, so
        // a stale TOML/GUI value can neither enable it on the wrong host nor
        // disable it here. For A/B testing on a reproduction rig, the dev-only
        // env var GR2_GPUCOMM_FIX overrides the choice when GRR is detected:
        //   "nonexclusive" / "shared" — pins stay, cores shared (default on 4c)
        //   "monolithic"              — collapse to a single synchronous GpuComm
        //                               thread, no assembler jthread (see
        //                               BundleAssembler / DecideReservedCores)
        //   "isolated" / "off"        — force the original isolated async split
        //                               (reproduces the stall; baseline for A/B)
        const unsigned phys_cores = Common::GetPhysicalCoreCount();
        enum class GpuCommFix { Isolated, NonExclusive, Monolithic };
        GpuCommFix gpucomm_fix = (is_gr_remastered && phys_cores == 4)
                                     ? GpuCommFix::NonExclusive
                                     : GpuCommFix::Isolated;
        const char* fix_source = "auto (host topology)";
        if (is_gr_remastered) {
            if (const char* env = std::getenv("GR2_GPUCOMM_FIX")) {
                const std::string_view sv{env};
                if (sv == "isolated" || sv == "off") {
                    gpucomm_fix = GpuCommFix::Isolated;
                    fix_source = "GR2_GPUCOMM_FIX env override";
                } else if (sv == "monolithic") {
                    gpucomm_fix = GpuCommFix::Monolithic;
                    fix_source = "GR2_GPUCOMM_FIX env override";
                } else if (sv == "nonexclusive" || sv == "shared") {
                    gpucomm_fix = GpuCommFix::NonExclusive;
                    fix_source = "GR2_GPUCOMM_FIX env override";
                } else {
                    LOG_WARNING(Loader,
                                "GR2_GPUCOMM_FIX='{}' not recognized; using auto "
                                "selection (expected isolated|monolithic|"
                                "nonexclusive).",
                                sv);
                }
            }
        }
        // Exactly one of the two flags is set (or neither, for isolated).
        Config::setLegacyMonolithicGpuComm(gpucomm_fix == GpuCommFix::Monolithic);
        Config::setGpuCoresNonExclusive(gpucomm_fix == GpuCommFix::NonExclusive);
        if (is_gr_remastered) {
            switch (gpucomm_fix) {
            case GpuCommFix::NonExclusive:
                LOG_INFO(Loader,
                         "Gravity Rush Remastered on a {}-physical-core host [{}] "
                         "-> NON-EXCLUSIVE GpuComm pinning: GpuComm and the "
                         "GpuAssembler keep their core pins but those cores are no "
                         "longer isolated, so the scheduler may place compile "
                         "workers and guest threads on them when idle.",
                         phys_cores, fix_source);
                break;
            case GpuCommFix::Monolithic:
                LOG_INFO(Loader,
                         "Gravity Rush Remastered on a {}-physical-core host [{}] "
                         "-> LEGACY MONOLITHIC GpuComm path: synchronous inline "
                         "dispatch, no GpuAssembler jthread, assembler core "
                         "returned to the guest.",
                         phys_cores, fix_source);
                break;
            case GpuCommFix::Isolated:
                LOG_INFO(Loader,
                         "Gravity Rush Remastered on a {}-physical-core host [{}] "
                         "-> isolated async GpuComm + GpuAssembler split (no "
                         "scheduling fix applied for this session).",
                         phys_cores, fix_source);
                break;
            }
        }
        LOG_INFO(Loader, "Fw: {:#x} App Version: {}", fw_version, app_version);
        LOG_INFO(Loader, "Compiled SDK version: {:#x}", sdk_version);
        LOG_INFO(Loader, "PSVR Supported: {}", (bool)psf_attributes.support_ps_vr.Value());
        LOG_INFO(Loader, "PSVR Required: {}", (bool)psf_attributes.require_ps_vr.Value());
    }
    if (!args.empty()) {
        const auto argc = std::min<size_t>(args.size(), 32);
        for (auto i = 0; i < argc; i++) {
            LOG_INFO(Loader, "Game argument {}: {}", i, args[i]);
        }
        if (args.size() > 32) {
            LOG_ERROR(Loader, "Too many game arguments, only passing the first 32");
        }
    }

    // Create stdin/stdout/stderr
    Common::Singleton<FileSys::HandleTable>::Instance()->CreateStdHandles();

    // Initialize components
    memory = Core::Memory::Instance();
    controller = Common::Singleton<Input::GameController>::Instance();
    linker = Common::Singleton<Core::Linker>::Instance();

    // Load renderdoc module
    VideoCore::LoadRenderDoc();

    // Initialize patcher and trophies
    if (!id.empty()) {
        MemoryPatcher::g_game_serial = id;

        // GR2FORK: extract trophies into the shared user/trophy tree + each user's
        // home/<uid>/trophy, and record the trophy-index -> npcommid map on the
        // ElfInfo so np_trophy can resolve the npcommid at context creation.
        const auto npbind_path = mnt->GetHostPath("/app0/sce_sys/npbind.dat");
        const auto trophy_dir = mnt->GetHostPath("/app0/sce_sys/trophy");
        game_info.trophy_index_map = ExtractTrophies(npbind_path, trophy_dir);
    }

    std::string game_title = fmt::format("{} - {} <{}>", id, title, app_version);
    std::string window_title = "";
    std::string remote_url(Common::g_scm_remote_url);
    std::string remote_host = Common::GetRemoteNameFromLink();
    // TITLE(GR2FORK v1.0): Personal branding — Common::g_version carries the
    // upstream base (0.13.0) that the fork tracks; the "v1.0" is the fork's
    // own release tag. Format: "Junmin Lee GR2FORK v1.0 (v0.13.0) | <game>".
    if (Common::g_is_release) {
        if (remote_host == "shadps4-emu" || remote_url.length() == 0) {
            window_title = fmt::format("Junmin Lee GR2FORK v4.8.002 (v{}) | {}", Common::g_version,
                                       game_title);
        } else {
            window_title = fmt::format("Junmin Lee GR2FORK v4.8.002 {}/(v{}) | {}", remote_host,
                                       Common::g_version, game_title);
        }
    } else {
        if (remote_host == "shadps4-emu" || remote_url.length() == 0) {
            window_title = fmt::format("Junmin Lee GR2FORK v4.8.002 (v{}) {} {} | {}",
                                       Common::g_version, Common::g_scm_branch,
                                       Common::g_scm_desc, game_title);
        } else {
            window_title = fmt::format("Junmin Lee GR2FORK v4.8.002 (v{}) {}/{} {} | {}",
                                       Common::g_version, remote_host, Common::g_scm_branch,
                                       Common::g_scm_desc, game_title);
        }
    }
    // FIX(GR2FORK v3.1): output resolution follows [GPU] resolutionOverride.
    //
    // Previously the SDL window was created at the user's configured
    // (screenWidth, screenHeight) — i.e. 1280x720 in stock shadPS4, 1920x1080
    // in this fork's typical config.toml — regardless of whether a resolution
    // override was active. The chain was:
    //
    //   game RT (patched, e.g. 4K) → blit-target Frame (window-size, 1080p)
    //     → swapchain (window-size, 1080p) → SDL surface (1080p) → monitor
    //
    // So enabling [GPU] resolutionOverride = "2160p" got you 4K internal
    // rendering but the very first hop downsampled the game image to the
    // window's 1080p before it ever reached the swapchain. The "output" the
    // user saw was always the window size, not the patched render size.
    //
    // Fix: when resolutionOverride is anything other than "Off", compute the
    // composed (W, H) the eboot will be patched with (same helpers
    // module.cpp uses one game-module-load later) and feed THAT to WindowSDL
    // instead of the configured screenWidth/Height. The aspect override
    // composes with the resolution target the same way it does inside
    // ApplyGr2ResolutionPatches — so e.g. 2160p × 21:9 yields 5040x2160.
    //
    // When resolutionOverride is "Off" (default for non-GR2 titles, since
    // this option is loaded from the game-specific {id}.toml), behaviour
    // is unchanged from stock — the configured screenWidth/Height wins.
    //
    // SDL constraints: the window may end up larger than the user's display
    // (e.g. 4K window on a 1920x1200 Legion Go panel). SDL still creates the
    // window; the window manager will clip or scale per the user's setup.
    // When [General] isFullScreen=true, sdl_window.cpp's SetWindowFullscreen
    // path will switch to display-mode size anyway and the swapchain will
    // recreate to the surface extent at first present, so this override is
    // inert in fullscreen — exactly what we want.
    s32 window_w = static_cast<s32>(Config::getWindowWidth());
    s32 window_h = static_cast<s32>(Config::getWindowHeight());
    {
        const auto _res_target = Libraries::ResolutionPatches::ParseResolutionFromConfig(
            Config::getResolutionOverride());
        if (_res_target != Libraries::ResolutionPatches::TargetResolution::Off) {
            const auto _aspect_target = Libraries::AspectPatches::ParseAspectFromConfig(
                Config::getAspectRatioOverride());
            const float _ar =
                Libraries::AspectPatches::TargetAspectToRatio(_aspect_target);
            const auto _final =
                Libraries::ResolutionPatches::ComputeFinalResolution(_res_target, _ar);
            LOG_INFO(Loader,
                     "[GR2FORK] window size overridden by [GPU] resolutionOverride='{}' "
                     "× aspectRatioOverride='{}': {}x{} → {}x{}",
                     Config::getResolutionOverride(), Config::getAspectRatioOverride(),
                     window_w, window_h, _final.width, _final.height);
            window_w = _final.width;
            window_h = _final.height;
        }
    }

    window = std::make_unique<Frontend::WindowSDL>(
        window_w, window_h, controller, window_title);

    g_window = window.get();

    // [GR2FORK] Point the window/taskbar icon at the game's own icon0.png so GR2
    // (and every other title) shows its cover art instead of the generic shadPS4
    // icon. /app0 is already mounted above, so this resolves into the game's
    // sce_sys folder. Ported from upstream shadPS4 PR #4586. Silently skipped if
    // the game ships no icon0.png; SetIcon itself is best-effort past that.
    const auto icon_path = mnt->GetHostPath("/app0/sce_sys/icon0.png");
    if (std::filesystem::exists(icon_path)) {
        window->SetIcon(icon_path);
    }

    const auto& mount_data_dir = Common::FS::GetUserPath(Common::FS::PathType::GameDataDir) / id;
    if (!std::filesystem::exists(mount_data_dir)) {
        std::filesystem::create_directory(mount_data_dir);
    }
    mnt->Mount(mount_data_dir, "/data"); // should just exist, manually create with game serial

    // Mounting temp folders
    const auto& mount_temp_dir = Common::FS::GetUserPath(Common::FS::PathType::TempDataDir) / id;
    if (std::filesystem::exists(mount_temp_dir)) {
        // Temp folder should be cleared on each boot.
        std::filesystem::remove_all(mount_temp_dir);
    }
    std::filesystem::create_directory(mount_temp_dir);
    mnt->Mount(mount_temp_dir, "/temp0");
    mnt->Mount(mount_temp_dir, "/temp");

    const auto& mount_download_dir =
        Common::FS::GetUserPath(Common::FS::PathType::DownloadDir) / id;
    if (!std::filesystem::exists(mount_download_dir)) {
        std::filesystem::create_directory(mount_download_dir);
    }
    mnt->Mount(mount_download_dir, "/download0");

    const auto& mount_captures_dir = Common::FS::GetUserPath(Common::FS::PathType::CapturesDir);
    if (!std::filesystem::exists(mount_captures_dir)) {
        std::filesystem::create_directory(mount_captures_dir);
    }
    VideoCore::SetOutputDir(mount_captures_dir, id);

    // Initialize kernel and library facilities.
    Libraries::InitHLELibs(&linker->GetHLESymbols());

    // Load the module with the linker
    auto guest_eboot_path = "/app0/" + eboot_name.generic_string();
    const auto eboot_path = mnt->GetHostPath(guest_eboot_path);
    if (linker->LoadModule(eboot_path) == -1) {
        LOG_CRITICAL(Loader, "Failed to load game's eboot.bin: {}",
                     Common::FS::PathToUTF8String(std::filesystem::absolute(eboot_path)));
        std::quick_exit(0);
    }

    // check if we have system modules to load
    LoadSystemModules(game_info.game_serial);

    // Load all prx from game's sce_module folder
    mnt->IterateDirectory("/app0/sce_module", [this](const auto& path, const auto is_file) {
        if (is_file) {
            LOG_INFO(Loader, "Loading {}", fmt::UTF(path.u8string()));
            linker->LoadModule(path);
        }
    });

#ifdef ENABLE_DISCORD_RPC
    // Discord RPC
    if (Config::getEnableDiscordRPC()) {
        auto* rpc = Common::Singleton<DiscordRPCHandler::RPC>::Instance();
        if (rpc->getRPCEnabled() == false) {
            rpc->init();
        }
        rpc->setStatusPlaying(game_info.title, id);
    }
#endif

    if (!id.empty()) {
        start_time = std::chrono::steady_clock::now();

        std::thread([this, id]() {
            Common::SetCurrentThreadName("shadPS4:PlayTimeUpdater");
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(60));
                UpdatePlayTime(id);
                start_time = std::chrono::steady_clock::now();
            }
        }).detach();
    }

    args.insert(args.begin(), eboot_name.generic_string());

    // FIX(GR2FORK): install hang watchdog before guest execution starts.
    // Callbacks tolerate null presenter/liverpool — they return 0 until the
    // game initializes the renderer via sceGnmDriverInitializer.
    {
        Common::HangWatchdogCallbacks cb;
        cb.scheduler_tick = []() -> u64 {
            if (!presenter) return 0;
            return presenter->GetRasterizer().GetScheduler().CurrentTick();
        };
        cb.num_submits = []() -> u32 {
            if (!liverpool) return 0;
            return liverpool->GetNumSubmits();
        };
        cb.vram_used = []() -> u64 {
            if (!presenter) return 0;
            const auto& inst = presenter->GetRasterizer().GetInstance();
            return inst.CanReportMemoryUsage() ? inst.GetDeviceMemoryUsage() : 0;
        };
        cb.vram_budget = []() -> u64 {
            if (!presenter) return 0;
            return presenter->GetRasterizer().GetInstance().GetTotalMemoryBudget();
        };
        cb.texture_mem = []() -> u64 {
            if (!presenter) return 0;
            return presenter->GetRasterizer().GetTextureCache().GetTotalUsedMemory();
        };
        cb.num_images  = []() -> size_t { return 0; }; // not wired
        cb.num_buffers = []() -> size_t { return 0; }; // not wired
        Common::HangWatchdog::Start(std::move(cb));
    }

    linker->Execute(args);

    window->InitTimers();
    while (window->IsOpen()) {
        window->WaitEvent();
    }

    UpdatePlayTime(id);
    Storage::DataBase::Instance().Close();

    // FIX(GR2FORK v2): tell the crash handler this quick_exit is a
    // routine shutdown (SDL loop exited because the window was closed,
    // not a crash). Without this, every clean quit produces a spurious
    // "*** quick_exit() / _Exit() (at_quick_exit) ***" crash entry in
    // crash_dump.txt with a stack walk of the shutdown path.
    Common::CrashHandler::SignalCleanShutdown();
    std::quick_exit(0);
}

void Emulator::Restart(std::filesystem::path eboot_path,
                       const std::vector<std::string>& guest_args) {
    std::vector<std::string> args;

    auto mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    auto game_path = mnt->GetHostPath("/app0");

    args.push_back("--log-append");
    args.push_back("--game");
    args.push_back(Common::FS::PathToUTF8String(eboot_path));

    args.push_back("--override-root");
    args.push_back(Common::FS::PathToUTF8String(game_path));

    if (FileSys::MntPoints::ignore_game_patches) {
        args.push_back("--ignore-game-patch");
    }

    if (!MemoryPatcher::patch_file.empty()) {
        args.push_back("--patch");
        args.push_back(MemoryPatcher::patch_file);
    }

    args.push_back("--wait-for-pid");
    args.push_back(std::to_string(Debugger::GetCurrentPid()));

    if (waitForDebuggerBeforeRun) {
        args.push_back("--wait-for-debugger");
    }

    if (guest_args.size() > 0) {
        args.push_back("--");
        for (const auto& arg : guest_args) {
            args.push_back(arg);
        }
    }

    LOG_INFO(Common, "Restarting the emulator with args: {}", fmt::join(args, " "));
    Libraries::SaveData::Backup::StopThread();
    Common::Log::Denitializer();

    auto& ipc = IPC::Instance();

    if (ipc.IsEnabled()) {
        ipc.SendRestart(args);
        while (true) {
            std::this_thread::sleep_for(std::chrono::minutes(1));
        }
    }
#if defined(_WIN32)
    std::string cmdline;
    // Emulator executable
    cmdline += "\"";
    cmdline += executableName;
    cmdline += "\"";
    for (const auto& arg : args) {
        cmdline += " \"";
        cmdline += arg;
        cmdline += "\"";
    }
    cmdline += "\0";

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    bool success = CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, TRUE, 0, nullptr,
                                  nullptr, &si, &pi);

    if (!success) {
        std::cerr << "Failed to restart game: {}" << GetLastError() << std::endl;
        std::quick_exit(1);
    }

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
#elif defined(__APPLE__) || defined(__linux__)
    std::vector<char*> argv;

    // Emulator executable
    argv.push_back(const_cast<char*>(executableName));

    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid == 0) {
        // Child process - execute the new instance
        execvp(executableName, argv.data());
        std::cerr << "Failed to restart game: execvp failed" << std::endl;
        std::quick_exit(1);
    } else if (pid < 0) {
        std::cerr << "Failed to restart game: fork failed" << std::endl;
        std::quick_exit(1);
    }
#else
#error "Unsupported platform"
#endif

    std::quick_exit(0);
}

void Emulator::LoadSystemModules(const std::string& game_serial) {
    constexpr auto ModulesToLoad = std::to_array<SysModules>(
        {{"libSceNgs2.sprx", &Libraries::Ngs2::RegisterLib},
        {"libSceUlt.sprx", nullptr},
        {"libSceJpegDec.sprx", nullptr},
        {"libSceJpegEnc.sprx", &Libraries::JpegEnc::RegisterLib},
        {"libScePngEnc.sprx", nullptr},
         {"libSceJson.sprx", nullptr},
         {"libSceJson2.sprx", nullptr},
         {"libSceLibcInternal.sprx", &Libraries::LibcInternal::RegisterLib},
         {"libSceCesCs.sprx", nullptr},
         {"libSceFont.sprx", &Libraries::Font::RegisterlibSceFont},
         {"libSceFontFt.sprx", &Libraries::FontFt::RegisterlibSceFontFt},
         {"libSceFreeTypeOt.sprx", nullptr},
         // GR2 gallery: load ScreenShot browse API as LLE.
         // The fiber at 0x10914c0 calls browse functions from libSceScreenShot
         // which are NOT covered by our HLE (capture-only). The LLE provides
         // the browse functions. Dependency chain:
         //   libSceScreenShot → libSceIpmi, libSceSysUtil
         //   libSceIpmi → libkernel (HLE)
         // HLE capture functions (RegisterLib in libs.cpp) still win for
         // matching NIDs since the linker checks HLE first.
         {"libSceIpmi.sprx", nullptr},
         {"libSceSysUtil.sprx", nullptr},
         {"libSceScreenShot.sprx", nullptr}});

    std::vector<std::filesystem::path> found_modules;
    const auto& sys_module_path = Config::getSysModulesPath();
    for (const auto& entry : std::filesystem::directory_iterator(sys_module_path)) {
        found_modules.push_back(entry.path());
    }
    for (const auto& [module_name, init_func] : ModulesToLoad) {
        const auto it = std::ranges::find_if(
            found_modules, [&](const auto& path) { return path.filename() == module_name; });
        if (it != found_modules.end()) {
            LOG_INFO(Loader, "Loading {}", it->string());
            if (linker->LoadModule(*it) != -1) {
                continue;
            }
        }
        if (init_func) {
            LOG_INFO(Loader, "Can't Load {} switching to HLE", module_name);
            init_func(&linker->GetHLESymbols());
        } else {
            LOG_INFO(Loader, "No HLE available for {} module", module_name);
        }
    }
    if (!game_serial.empty() && std::filesystem::exists(sys_module_path / game_serial)) {
        for (const auto& entry :
             std::filesystem::directory_iterator(sys_module_path / game_serial)) {
            LOG_INFO(Loader, "Loading {} from game serial file {}", entry.path().string(),
                     game_serial);
            linker->LoadModule(entry.path());
        }
    }
}

void Emulator::UpdatePlayTime(const std::string& serial) {
    const auto user_dir = Common::FS::GetUserPath(Common::FS::PathType::UserDir);
    const auto filePath = (user_dir / "play_time.txt").string();

    std::ifstream in(filePath);
    if (!in && !std::ofstream(filePath)) {
        LOG_INFO(Loader, "Error opening play_time.txt");
        return;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    int total_seconds = static_cast<int>(duration.count());

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    in.close();

    int accumulated_seconds = 0;
    bool found = false;

    for (const auto& l : lines) {
        std::istringstream iss(l);
        std::string s, time_str;
        if (iss >> s >> time_str && s == serial) {
            int h, m, s_;
            char c1, c2;
            std::istringstream ts(time_str);
            if (ts >> h >> c1 >> m >> c2 >> s_ && c1 == ':' && c2 == ':') {
                accumulated_seconds = h * 3600 + m * 60 + s_;
                found = true;
                break;
            }
        }
    }

    accumulated_seconds += total_seconds;
    int hours = accumulated_seconds / 3600;
    int minutes = (accumulated_seconds % 3600) / 60;
    int seconds = accumulated_seconds % 60;

    std::string playTimeSaved = fmt::format("{:d}:{:02d}:{:02d}", hours, minutes, seconds);

    std::ofstream outfile(filePath, std::ios::trunc);
    bool lineUpdated = false;
    for (const auto& l : lines) {
        std::istringstream iss(l);
        std::string s;
        if (iss >> s && s == serial) {
            outfile << fmt::format("{} {}\n", serial, playTimeSaved);
            lineUpdated = true;
        } else {
            outfile << l << "\n";
        }
    }

    if (!lineUpdated) {
        outfile << fmt::format("{} {}\n", serial, playTimeSaved);
    }

    LOG_INFO(Loader, "Playing time for {}: {}", serial, playTimeSaved);
}

} // namespace Core
