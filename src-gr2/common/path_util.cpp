// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <unordered_map>
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/scope_exit.h"

#ifdef __APPLE__
#include <CoreFoundation/CFBundle.h>
#include <dlfcn.h>
#include <sys/param.h>
#endif

#ifndef MAX_PATH
#ifdef _WIN32
// This is the maximum number of UTF-16 code units permissible in Windows file paths
#define MAX_PATH 260
#include <Shlobj.h>
#include <windows.h>
#else
// This is the maximum number of UTF-8 code units permissible in all other OSes' file paths
#define MAX_PATH 1024
#endif
#endif

namespace Common::FS {

namespace fs = std::filesystem;

#ifdef __APPLE__
using IsTranslocatedURLFunc = Boolean (*)(CFURLRef path, bool* isTranslocated,
                                          CFErrorRef* __nullable error);
using CreateOriginalPathForURLFunc = CFURLRef __nullable (*)(CFURLRef translocatedPath,
                                                             CFErrorRef* __nullable error);

static CFURLRef UntranslocateBundlePath(const CFURLRef bundle_path) {
    if (void* security_handle =
            dlopen("/System/Library/Frameworks/Security.framework/Security", RTLD_LAZY)) {
        SCOPE_EXIT {
            dlclose(security_handle);
        };

        const auto IsTranslocatedURL = reinterpret_cast<IsTranslocatedURLFunc>(
            dlsym(security_handle, "SecTranslocateIsTranslocatedURL"));
        const auto CreateOriginalPathForURL = reinterpret_cast<CreateOriginalPathForURLFunc>(
            dlsym(security_handle, "SecTranslocateCreateOriginalPathForURL"));

        bool is_translocated = false;
        if (IsTranslocatedURL && CreateOriginalPathForURL &&
            IsTranslocatedURL(bundle_path, &is_translocated, nullptr) && is_translocated) {
            return CreateOriginalPathForURL(bundle_path, nullptr);
        }
    }
    return nullptr;
}

static std::optional<std::filesystem::path> GetBundleParentDirectory() {
    if (CFBundleRef bundle_ref = CFBundleGetMainBundle()) {
        if (CFURLRef bundle_url_ref = CFBundleCopyBundleURL(bundle_ref)) {
            SCOPE_EXIT {
                CFRelease(bundle_url_ref);
            };

            CFURLRef untranslocated_url_ref = UntranslocateBundlePath(bundle_url_ref);
            SCOPE_EXIT {
                if (untranslocated_url_ref) {
                    CFRelease(untranslocated_url_ref);
                }
            };

            char app_bundle_path[MAXPATHLEN];
            if (CFURLGetFileSystemRepresentation(
                    untranslocated_url_ref ? untranslocated_url_ref : bundle_url_ref, true,
                    reinterpret_cast<u8*>(app_bundle_path), sizeof(app_bundle_path))) {
                std::filesystem::path bundle_path{app_bundle_path};
                return bundle_path.parent_path();
            }
        }
    }
    return std::nullopt;
}
#endif

static auto UserPaths = [] {
    // Try the portable user directory first.
    auto user_dir = std::filesystem::current_path() / PORTABLE_DIR;
    if (!std::filesystem::exists(user_dir)) {
        // If it doesn't exist, use the standard path for the platform instead.
        // NOTE: On Windows we currently just create the portable directory instead.
#ifdef __APPLE__
        user_dir =
            std::filesystem::path(getenv("HOME")) / "Library" / "Application Support" / "shadPS4";
#elif defined(__linux__)
        const char* xdg_data_home = getenv("XDG_DATA_HOME");
        if (xdg_data_home != nullptr && strlen(xdg_data_home) > 0) {
            user_dir = std::filesystem::path(xdg_data_home) / "shadPS4";
        } else {
            user_dir = std::filesystem::path(getenv("HOME")) / ".local" / "share" / "shadPS4";
        }
#elif _WIN32
        TCHAR appdata[MAX_PATH] = {0};
        SHGetFolderPath(NULL, CSIDL_APPDATA, NULL, 0, appdata);
        user_dir = std::filesystem::path(appdata) / "shadPS4";
#endif
    }

    std::unordered_map<PathType, fs::path> paths;

    const auto create_path = [&](PathType shad_path, const fs::path& new_path) {
        std::filesystem::create_directory(new_path);
        paths.insert_or_assign(shad_path, new_path);
    };

    create_path(PathType::UserDir, user_dir);
    create_path(PathType::LogDir, user_dir / LOG_DIR);

    // MIGRATION(GR2FORK v1.0): if the legacy "screenshots" directory exists
    // and the new "Capture Gallery" doesn't, rename in place so existing
    // photos and GR2 gallery entries carry over without user intervention.
    {
        const auto legacy = user_dir / "screenshots";
        const auto current = user_dir / SCREENSHOTS_DIR;
        std::error_code ec;
        if (std::filesystem::exists(legacy, ec) && !std::filesystem::exists(current, ec)) {
            std::filesystem::rename(legacy, current, ec);
            // If rename fails (e.g. cross-device), the create_path below
            // just creates the new dir empty. Old dir stays around.
        }
        // Sub-rename: "GR2_PhotoApp_HLE" -> "Gravity Rush 2" inside the gallery.
        const auto legacy_gr2 = current / "GR2_PhotoApp_HLE";
        const auto new_gr2 = current / "Gravity Rush 2";
        if (std::filesystem::exists(legacy_gr2, ec) && !std::filesystem::exists(new_gr2, ec)) {
            std::filesystem::rename(legacy_gr2, new_gr2, ec);
        }
    }

    create_path(PathType::ScreenshotsDir, user_dir / SCREENSHOTS_DIR);
    create_path(PathType::ShaderDir, user_dir / SHADER_DIR);
    create_path(PathType::GameDataDir, user_dir / GAMEDATA_DIR);
    create_path(PathType::TempDataDir, user_dir / TEMPDATA_DIR);
    create_path(PathType::SysModuleDir, user_dir / SYSMODULES_DIR);
    create_path(PathType::DownloadDir, user_dir / DOWNLOAD_DIR);
    create_path(PathType::CapturesDir, user_dir / CAPTURES_DIR);
    create_path(PathType::CheatsDir, user_dir / CHEATS_DIR);
    create_path(PathType::PatchesDir, user_dir / PATCHES_DIR);
    create_path(PathType::MetaDataDir, user_dir / METADATA_DIR);
    create_path(PathType::CustomTrophy, user_dir / CUSTOM_TROPHY);
    create_path(PathType::CustomConfigs, user_dir / CUSTOM_CONFIGS);
    create_path(PathType::CacheDir, user_dir / CACHE_DIR);
    create_path(PathType::FontsDir, user_dir / FONTS_DIR);
    create_path(PathType::HomeDir, user_dir / HOME_DIR);
    create_path(PathType::CustomModulesDir, user_dir / CUSTOM_MODULES_DIR);
    create_path(PathType::ModsDir, user_dir / MODS_DIR);

    // GR2FORK: provision the prerelease PS4 home layout (UserManager::CreateDefaultUsers): four
    // accounts 1000..1003, each with savedata/trophy/inputs, so both builds share one home tree.
    // Only account 1000's savedata is used; creates are guarded and never clear existing folders.
    {
        const auto home_root = user_dir / HOME_DIR;
        std::error_code dec;
        for (const char* account : {"1000", "1001", "1002", "1003"}) {
            const auto account_dir = home_root / account;
            std::filesystem::create_directories(account_dir / "savedata", dec);
            std::filesystem::create_directories(account_dir / "trophy", dec);
            std::filesystem::create_directories(account_dir / "inputs", dec);
        }
    }

    // GR2FORK: migrate saves from the legacy user/savedata/<uid>/<serial> layout into the PS4
    // home layout user/home/<uid>/savedata/<serial>, folding the old default account "1" into
    // "1000". Only the default save location is handled, and no destination is ever overwritten.
    {
        const auto legacy_root = user_dir / "savedata";
        const auto home_root = user_dir / HOME_DIR;
        std::error_code mec;

        // Map the legacy single-user account "1" onto the new default "1000";
        // every other uid keeps its own folder name.
        const auto map_uid = [](const std::filesystem::path& uid_dir) -> std::filesystem::path {
            return uid_dir.filename().string() == "1" ? std::filesystem::path("1000")
                                                      : uid_dir.filename();
        };

        // Move one savedata tree's per-serial entries into dest_savedata, never
        // overwriting an entry that already exists at the destination.
        const auto move_serials = [](const std::filesystem::path& src_uid_dir,
                                     const std::filesystem::path& dest_savedata) {
            std::error_code lec;
            std::filesystem::create_directories(dest_savedata, lec);
            for (const auto& serial_entry :
                 std::filesystem::directory_iterator(src_uid_dir, lec)) {
                const auto dest = dest_savedata / serial_entry.path().filename();
                if (!std::filesystem::exists(dest, lec)) {
                    std::filesystem::rename(serial_entry.path(), dest, lec);
                }
            }
        };

        // Pass 1: legacy user/savedata/<uid> -> user/home/<mapped uid>/savedata.
        if (std::filesystem::is_directory(legacy_root, mec)) {
            for (const auto& uid_entry : std::filesystem::directory_iterator(legacy_root, mec)) {
                if (!uid_entry.is_directory()) {
                    continue;
                }
                move_serials(uid_entry.path(), home_root / map_uid(uid_entry.path()) / "savedata");
            }
        }

        // Pass 2: consolidate an older gr2fork user/home/1/savedata into user/home/1000/savedata,
        // then drop the "1" account only if fully empty (remove() never deletes a non-empty dir,
        // so any unmerged save keeps the old account around for manual reconcile).
        const auto old_account = home_root / "1";
        const auto old_savedata = old_account / "savedata";
        if (std::filesystem::is_directory(old_savedata, mec)) {
            move_serials(old_savedata, home_root / "1000" / "savedata");
            std::filesystem::remove(old_savedata, mec);
            std::filesystem::remove(old_account / "trophy", mec);
            std::filesystem::remove(old_account / "inputs", mec);
            std::filesystem::remove(old_account, mec);
        }
    }

    std::ofstream notice_file(user_dir / CUSTOM_TROPHY / "Notice.txt");
    if (notice_file.is_open()) {
        notice_file
            << "++++++++++++++++++++++++++++++++\n+ Custom Trophy Images / Sound "
               "+\n++++++++++++++++++++++++++++++++\n\nYou can add custom images to the "
               "trophies.\n*We recommend a square resolution image, for example 200x200, 500x500, "
               "the same size as the height and width.\nIn this folder ('user\\custom_trophy'), "
               "add the files with the following "
               "names:\n\nbronze.png\nsilver.png\ngold.png\nplatinum.png\n\nYou can add a custom "
               "sound for trophy notifications.\n*By default, no audio is played unless it is in "
               "this folder and you are using the QT version.\nIn this folder "
               "('user\\custom_trophy'), add the files with the following names:\n\ntrophy.mp3";
        notice_file.close();
    }

    return paths;
}();

bool ValidatePath(const fs::path& path) {
    if (path.empty()) {
        LOG_ERROR(Common_Filesystem, "Input path is empty, path={}", PathToUTF8String(path));
        return false;
    }

#ifdef _WIN32
    if (path.u16string().size() >= MAX_PATH) {
        LOG_ERROR(Common_Filesystem, "Input path is too long, path={}", PathToUTF8String(path));
        return false;
    }
#else
    if (path.u8string().size() >= MAX_PATH) {
        LOG_ERROR(Common_Filesystem, "Input path is too long, path={}", PathToUTF8String(path));
        return false;
    }
#endif

    return true;
}

std::string PathToUTF8String(const std::filesystem::path& path) {
    const auto u8_string = path.u8string();
    return std::string{u8_string.begin(), u8_string.end()};
}

const fs::path& GetUserPath(PathType shad_path) {
    return UserPaths.at(shad_path);
}

std::string GetUserPathString(PathType shad_path) {
    return PathToUTF8String(GetUserPath(shad_path));
}

void SetUserPath(PathType shad_path, const fs::path& new_path) {
    if (!std::filesystem::is_directory(new_path)) {
        LOG_ERROR(Common_Filesystem, "Filesystem object at new_path={} is not a directory",
                  PathToUTF8String(new_path));
        return;
    }

    UserPaths.insert_or_assign(shad_path, new_path);
}

std::optional<fs::path> FindGameByID(const fs::path& dir, const std::string& game_id,
                                     int max_depth) {
    if (max_depth < 0) {
        return std::nullopt;
    }

    // Check if this is the game we're looking for
    if (dir.filename() == game_id && fs::exists(dir / "sce_sys" / "param.sfo")) {
        auto eboot_path = dir / "eboot.bin";
        if (fs::exists(eboot_path)) {
            return eboot_path;
        }
    }

    // Recursively search subdirectories
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_directory()) {
            continue;
        }
        if (auto found = FindGameByID(entry.path(), game_id, max_depth - 1)) {
            return found;
        }
    }

    return std::nullopt;
}

} // namespace Common::FS
