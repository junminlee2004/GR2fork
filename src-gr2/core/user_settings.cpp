// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <map>
#include <common/config.h>
#include <common/path_util.h>
#include <common/scm_rev.h>
#include "common/logging/log.h"
#include "user_settings.h"

using json = nlohmann::json;

// Singleton storage
std::shared_ptr<UserSettingsImpl> UserSettingsImpl::s_instance = nullptr;
std::mutex UserSettingsImpl::s_mutex;

// Singleton
UserSettingsImpl::UserSettingsImpl() = default;

UserSettingsImpl::~UserSettingsImpl() {
    if (m_loaded)
        Save();
}

std::shared_ptr<UserSettingsImpl> UserSettingsImpl::GetInstance() {
    std::lock_guard lock(s_mutex);
    if (!s_instance)
        s_instance = std::make_shared<UserSettingsImpl>();
    return s_instance;
}

void UserSettingsImpl::SetInstance(std::shared_ptr<UserSettingsImpl> instance) {
    std::lock_guard lock(s_mutex);
    s_instance = std::move(instance);
}

namespace {
// Mirror the Settings "Username" (Config::userName, which already feeds sceNpGetOnlineId) onto
// the primary user (player_index 1): sceUserServiceGetUserName reads users.json, seeded from
// CreateDefaultUsers() ("shadPS4"), so without the mirror the on-screen name and the online ID
// disagree. Returns true when the in-memory name changed so the caller persists via Save();
// a users.json with no player_index-1 user is left untouched.
bool MirrorConfigUserName(UserManager& user_manager) {
    const std::string cfg_name = Config::getUserName();
    if (cfg_name.empty())
        return false;
    User* primary = user_manager.GetUserByPlayerIndex(1);
    if (primary == nullptr || primary->user_name == cfg_name)
        return false;
    primary->user_name = cfg_name;
    return true;
}
} // namespace

bool UserSettingsImpl::Save() const {
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "users.json";
    try {
        json j;
        j["Users"] = m_userManager.GetUsers();
        j["Users"]["commit_hash"] = std::string(Common::g_scm_rev);

        // Persist the live Config::userName onto the primary user (player_index 1) at serialize
        // time: the teardown Save runs after per-game config is active, so writing the resolved
        // getUserName() here makes users.json agree with the sceNpGetOnlineId the game actually
        // used instead of freezing the pre-per-game name. An empty name leaves the stored value.
        if (const std::string cfg_name = Config::getUserName();
            !cfg_name.empty() && j["Users"].contains("user") && j["Users"]["user"].is_array()) {
            for (auto& ju : j["Users"]["user"]) {
                if (ju.value("player_index", 0) == 1) {
                    ju["user_name"] = cfg_name;
                    break;
                }
            }
        }

        json existing = json::object();
        if (std::ifstream existingIn{path}; existingIn.good()) {
            try {
                existingIn >> existing;
            } catch (...) {
                existing = json::object();
            }
        }

        if (existing.contains("Users") && existing["Users"].is_object())
            existing["Users"].update(j["Users"]);
        else
            existing["Users"] = j["Users"];

        std::ofstream out(path);
        if (!out) {
            LOG_DEBUG(Config, "Failed to open user settings for writing: {}", path.string());
            return false;
        }
        out << std::setw(2) << existing;
        return !out.fail();
    } catch (const std::exception& e) {
        LOG_DEBUG(Config, "Error saving user settings: {}", e.what());
        return false;
    }
}

bool UserSettingsImpl::Load() {
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "users.json";
    try {
        if (!std::filesystem::exists(path)) {
            if (m_userManager.GetUsers().user.empty())
                m_userManager.GetUsers() = m_userManager.CreateDefaultUsers();
            // First run: adopt the configured username before the initial write so the freshly
            // created users.json already carries it instead of the "shadPS4" default.
            MirrorConfigUserName(m_userManager);
            m_loaded = true;
            Save();
            return false;
        }

        std::ifstream in(path);
        if (!in) {
            LOG_DEBUG(Config, "Failed to open user settings: {}", path.string());
            return false;
        }

        json j;
        in >> j;

        auto default_users = m_userManager.CreateDefaultUsers();
        json default_json;
        default_json["Users"] = default_users;

        if (j.contains("Users")) {
            json current = default_json["Users"];
            current.update(j["Users"]);
            m_userManager.GetUsers() = current.get<Users>();
        } else {
            m_userManager.GetUsers() = default_users;
        }

        m_loaded = true;
        // Mirror the configured username onto the primary user, then persist if either that
        // changed the stored name or the build's commit hash advanced.
        const bool name_changed = MirrorConfigUserName(m_userManager);
        if (name_changed || m_userManager.GetUsers().commit_hash != Common::g_scm_rev)
            Save();

        return true;
    } catch (const std::exception& e) {
        LOG_DEBUG(Config, "Error loading user settings: {}", e.what());
        if (m_userManager.GetUsers().user.empty())
            m_userManager.GetUsers() = m_userManager.CreateDefaultUsers();
        return false;
    }
}
