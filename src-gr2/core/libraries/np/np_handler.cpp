// SPDX-FileCopyrightText: Copyright 2019-2026 rpcs3 Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <string_view>
#include <chrono>
#include <cstring>
#include "common/config.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "core/libraries/np/np_error.h"
#include "core/libraries/np/np_handler.h"
#include "core/libraries/np/np_score.h"
#include "core/user_settings.h"
#include "shadnet.pb.h"
#include "shadnet/server_probe.h"

namespace Libraries::Np {

NpHandler& NpHandler::GetInstance() {
    static NpHandler s_instance;
    return s_instance;
}

bool NpHandler::ShadNetEnabled() const {
    static const bool is_grr = [] {
        static constexpr std::array<std::string_view, 12> kGrrSerials = {
            "CUSA01112", "CUSA01113", "CUSA01113P", "CUSA01130",
            "CUSA02318", "CUSA00546", "CUSA02443",  "CUSA04246",
            "PCJS50004", "PCJS50008", "PCJS66015",  "PCJS66029",
        };
        const auto serial = Common::ElfInfo::Instance().GameSerial();
        return std::find(kGrrSerials.begin(), kGrrSerials.end(), serial) != kGrrSerials.end();
    }();
    return is_grr && !m_session_disabled.load() && !Config::GetShadnetNpid().empty() &&
           !Config::GetShadnetPassword().empty() && Config::getIsConnectedToNetwork() &&
           Config::getPSNSignedIn();
}

std::pair<std::string, u16> NpHandler::ParseServerAddress() const {
    const std::string server_str = Config::GetShadnetServer();
    u16 port = 31313; // default port
    if (server_str.empty()) {
        LOG_ERROR(NpHandler,
                  "shadNet server address is empty,set the shadNet server field in settings. "
                  "Connection will fail.");
        return {std::string{}, port};
    }
    std::string host = server_str;
    const auto colon = server_str.rfind(':');
    if (colon != std::string::npos) {
        host = server_str.substr(0, colon);
        const std::string port_str = server_str.substr(colon + 1);
        try {
            port = static_cast<u16>(std::stoi(port_str));
        } catch (const std::exception&) {
            LOG_WARNING(NpHandler,
                        "shadNet server port '{}' is not a valid number; using default {}",
                        port_str, port);
        }
    }
    return {host, port};
}

bool NpHandler::ConnectUserById(s32 user_id) {
    if (!ShadNetEnabled())
        return false;

    {
        std::lock_guard lock(m_mutex_clients);
        if (m_clients.find(user_id) != m_clients.end())
            return false; // already connected
    }

    const auto [host, port] = ParseServerAddress();
    return ConnectUser(user_id, host, port, Config::GetShadnetNpid(),
                       Config::GetShadnetPassword(), std::string{});
}

void NpHandler::StartWorker() {
    bool expected = false;
    if (m_worker_running.compare_exchange_strong(expected, true)) {
        m_worker_thread = std::thread(&NpHandler::WorkerThread, this);
    }
}

void NpHandler::Initialize() {
    if (m_initialized.exchange(true)) {
        LOG_WARNING(NpHandler, "Initialize called more than once");
        return;
    }

    if (!ShadNetEnabled()) {
        LOG_INFO(NpHandler, "shadNet disabled for this title/config; offline mode");
        return;
    }

    {
        const auto [host, port] = ParseServerAddress();
        const ShadNet::ProbeInfo probe = ShadNet::ProbeServer(host, port);
        if (probe.result != ShadNet::ProbeResult::Ok) {
            m_session_disabled = true;
            switch (probe.result) {
            case ShadNet::ProbeResult::VersionMismatch:
                LOG_WARNING(NpHandler,
                            "shadNet server {}:{} protocol version mismatch (server v{}, emulator "
                            "v{}); disabling shadNet for this run (saved setting unchanged)",
                            host, port, probe.server_version, ShadNet::SHAD_PROTOCOL_VERSION);
                break;
            case ShadNet::ProbeResult::ProtocolError:
                LOG_WARNING(NpHandler,
                            "shadNet server {}:{} sent an invalid ServerInfo handshake; disabling "
                            "shadNet for this run (saved setting unchanged)",
                            host, port);
                break;
            default: // Unreachable
                LOG_WARNING(NpHandler,
                            "shadNet server {}:{} is offline/unreachable; disabling shadNet for "
                            "this run (saved setting unchanged)",
                            host, port);
                break;
            }
            return;
        }
    }

    if (!ConnectUserById(UserManagement.GetDefaultUser().user_id)) {
        LOG_WARNING(NpHandler, "no users connected to shadNet");
        return;
    }

    StartWorker();
}

void NpHandler::Shutdown() {
    if (!m_initialized.exchange(false))
        return;

    m_worker_running = false;

    // Stop any pending reconnect retries.
    {
        std::lock_guard lock(m_mutex_clients);
        m_reconnect.clear();
    }

    // Collect user IDs to disconnect (avoid holding m_mutex_clients during Stop)
    std::vector<s32> ids;
    {
        std::lock_guard lock(m_mutex_clients);
        for (auto& [uid, _] : m_clients)
            ids.push_back(uid);
    }
    for (s32 uid : ids)
        DisconnectUser(uid);

    if (m_worker_thread.joinable())
        m_worker_thread.join();

    LOG_INFO(NpHandler, "Shutdown complete");
}

bool NpHandler::ConnectUser(s32 user_id, const std::string& host, u16 port, const std::string& npid,
                            const std::string& password, const std::string& token) {
    LOG_INFO(NpHandler, "Connecting user_id={} npid='{}' to {}:{} (timeout {}s)", user_id, npid,
             host, port, ShadNet::SHAD_CONNECT_TIMEOUT_MS / 1000);

    auto client = std::make_shared<ShadNet::ShadNetClient>();

    client->onAsyncReply = [this, user_id](ShadNet::CommandType cmd, u64 pkt_id,
                                           ShadNet::ErrorType err, const std::vector<u8>& body) {
        OnAsyncReply(user_id, cmd, pkt_id, err, body);
    };
    client->onLoginResult = [this, user_id](const ShadNet::LoginResult& res) {
        OnLoginResult(user_id, res);
    };

    // Seed the current Appear-Offline preference so the login packet carries it (the send
    // is suppressed pre-auth; it just caches on the client).
    client->SetAppearOffline(m_appear_offline.load());
    client->Start(host, port, npid, password, token);

    // Shared handling for an incompatible-protocol failure (version mismatch or
    // corrupt/unparseable stream): disable shadNet for this run, since reconnect attempts
    // against an incompatible server are pointless.
    const auto handle_protocol_mismatch = [this, &client](ShadNet::ShadNetState st) {
        if (st != ShadNet::ShadNetState::FailureProtocol)
            return;
        const u32 server_ver = client->GetServerProtocolVersion();
        m_session_disabled = true;
        if (server_ver != 0) {
            LOG_ERROR(NpHandler,
                      "shadNet protocol version mismatch (server v{}, emulator v{}); disabling "
                      "shadNet for this run (saved setting unchanged)",
                      server_ver, ShadNet::SHAD_PROTOCOL_VERSION);
        } else {
            LOG_ERROR(NpHandler, "shadNet protocol error during handshake; disabling shadNet for "
                                 "this run (saved setting unchanged)");
        }
    };

    const ShadNet::ShadNetState conn_state = client->WaitForConnection();
    if (conn_state != ShadNet::ShadNetState::Ok) {
        LOG_ERROR(NpHandler, "user_id={} connection failed (state={})", user_id,
                  static_cast<int>(conn_state));
        handle_protocol_mismatch(conn_state);
        client->Stop();
        return false;
    }

    const ShadNet::ShadNetState auth_state = client->WaitForAuthenticated();
    if (auth_state != ShadNet::ShadNetState::Ok) {
        LOG_ERROR(NpHandler, "user_id={} authentication failed (state={})", user_id,
                  static_cast<int>(auth_state));
        handle_protocol_mismatch(auth_state);
        client->Stop();
        return false;
    }

    LOG_INFO(NpHandler, "user_id={} signed in npid='{}' accountId={}", user_id, npid,
             client->GetUserId());

    // Build OrbisNpId
    {
        OrbisNpId np_id{};
        SetNpId(np_id, npid);
        std::lock_guard lock(m_mutex_clients);
        m_np_ids[user_id] = np_id;
        m_clients[user_id] = std::move(client);
    }

    return true;
}

void NpHandler::FailPendingRequests(s32 user_id, s32 error_code) {
    size_t failed = 0;
    {
        std::lock_guard lock(m_mutex_pending_score);
        for (auto it = m_pending_score.begin(); it != m_pending_score.end();) {
            if (it->second.user_id == user_id) {
                if (it->second.req) {
                    it->second.req->SetResult(error_code);
                }
                it = m_pending_score.erase(it);
                ++failed;
            } else {
                ++it;
            }
        }
    }
    if (failed > 0) {
        LOG_WARNING(NpHandler, "user_id={} failed {} pending request(s) with {:#x}", user_id,
                    failed, static_cast<u32>(error_code));
    }
}

void NpHandler::DisconnectUser(s32 user_id) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end())
            return;
        client = std::move(it->second);
        m_clients.erase(it);
    }
    client->Stop();
    // The reader thread is joined, so no reply can race us: complete every
    // in-flight request now, or PollAsync spins and WaitAsync blocks forever.
    FailPendingRequests(user_id, ORBIS_NP_ERROR_SIGNED_OUT);
    LOG_INFO(NpHandler, "user_id={} disconnected", user_id);
}

void NpHandler::WorkerThread() {
    constexpr auto INTERVAL = std::chrono::milliseconds(500);

    while (m_worker_running) {
        std::this_thread::sleep_for(INTERVAL);

        // Collect ids of dropped clients
        std::vector<s32> dropped;
        {
            std::lock_guard lock(m_mutex_clients);
            for (auto& [uid, client] : m_clients) {
                if (!client->IsConnected())
                    dropped.push_back(uid);
            }
        }

        for (s32 uid : dropped) {
            LOG_WARNING(NpHandler, "user_id={} connection dropped (network); will retry", uid);
            DisconnectUser(uid);   // stops/removes the dead client, fails pending requests
            MarkForReconnect(uid); // schedule transparent reconnect (network drop, not logout)
        }

        TryReconnect();
    }
}

void NpHandler::MarkForReconnect(s32 user_id) {
    if (!ShadNetEnabled())
        return; // offline mode: nothing to reconnect to
    constexpr auto kInitialBackoff = std::chrono::milliseconds(2000);
    std::lock_guard lock(m_mutex_clients);
    auto& st = m_reconnect[user_id];
    st.backoff = kInitialBackoff;
    st.next_attempt = std::chrono::steady_clock::now() + st.backoff;
}

void NpHandler::TryReconnect() {
    constexpr auto kMaxBackoff = std::chrono::milliseconds(30000);
    const auto now = std::chrono::steady_clock::now();

    std::vector<s32> due;
    {
        std::lock_guard lock(m_mutex_clients);
        for (auto& [uid, st] : m_reconnect) {
            if (m_clients.count(uid) || now >= st.next_attempt)
                due.push_back(uid);
        }
    }

    for (s32 uid : due) {
        if (!m_worker_running)
            return;
        if (!ShadNetEnabled()) {
            std::lock_guard lock(m_mutex_clients);
            m_reconnect.erase(uid);
            continue;
        }
        // ConnectUserById locks m_mutex_clients internally; call it unlocked.
        const bool ok = ConnectUserById(uid);
        std::lock_guard lock(m_mutex_clients);
        if (ok || m_clients.count(uid)) {
            m_reconnect.erase(uid);
            LOG_INFO(NpHandler, "user_id={} reconnected to shadNet", uid);
        } else if (auto it = m_reconnect.find(uid); it != m_reconnect.end()) {
            it->second.backoff = std::min(it->second.backoff * 2, kMaxBackoff);
            it->second.next_attempt = std::chrono::steady_clock::now() + it->second.backoff;
            LOG_DEBUG(NpHandler, "user_id={} reconnect failed; next attempt in {}ms", uid,
                      it->second.backoff.count());
        }
    }
}

void NpHandler::OnLoginResult(s32 user_id, const ShadNet::LoginResult& res) {
    if (res.error != ShadNet::ErrorType::NoError) {
        return;
    }
    LOG_INFO(NpHandler, "user_id={} login accepted ({} friends on account)", user_id,
             res.friends.size());
}

void NpHandler::OnAsyncReply(s32 user_id, ShadNet::CommandType cmd, u64 pkt_id,
                             ShadNet::ErrorType error, const std::vector<u8>& body) {
    switch (cmd) {
    case ShadNet::CommandType::GetBoardInfos:
    case ShadNet::CommandType::RecordScore:
    case ShadNet::CommandType::RecordScoreData:
    case ShadNet::CommandType::GetScoreData:
    case ShadNet::CommandType::GetScoreRange:
    case ShadNet::CommandType::GetScoreFriends:
    case ShadNet::CommandType::GetScoreNpid:
    case ShadNet::CommandType::GetScoreAccountId:
    case ShadNet::CommandType::GetScoreGameDataByAccId:
        OnScoreReply(user_id, cmd, pkt_id, error, body);
        break;
    case ShadNet::CommandType::UnlockTrophy:
    case ShadNet::CommandType::SyncTrophies:
        OnTrophyReply(user_id, cmd, pkt_id, error, body);
        break;
    default:
        LOG_DEBUG(NpHandler, "OnAsyncReply: unhandled cmd={} pkt_id={}", static_cast<int>(cmd),
                  pkt_id);
        break;
    }
}

bool NpHandler::IsPsnSignedIn(s32 user_id) const {
    std::lock_guard lock(m_mutex_clients);
    auto it = m_clients.find(user_id);
    return it != m_clients.end() && it->second->IsAuthenticated();
}

bool NpHandler::IsAnySignedIn() const {
    std::lock_guard lock(m_mutex_clients);
    for (auto& [_, client] : m_clients)
        if (client->IsAuthenticated())
            return true;
    return false;
}

OrbisNpId NpHandler::GetNpId(s32 user_id) const {
    std::lock_guard lock(m_mutex_clients);
    auto it = m_np_ids.find(user_id);
    return it != m_np_ids.end() ? it->second : OrbisNpId{};
}

OrbisNpOnlineId NpHandler::GetOnlineId(s32 user_id) const {
    return GetNpId(user_id).handle;
}

std::string NpHandler::GetAvatarUrl(s32 user_id) const {
    std::lock_guard lock(m_mutex_clients);
    auto it = m_clients.find(user_id);
    return it != m_clients.end() ? it->second->GetAvatarUrl() : std::string{};
}

OrbisNpAccountId NpHandler::GetAccountId(s32 user_id) const {
    std::lock_guard lock(m_mutex_clients);
    auto it = m_clients.find(user_id);
    return it != m_clients.end() ? static_cast<OrbisNpAccountId>(it->second->GetUserId()) : 0;
}

std::string NpHandler::GetBearerToken(s32 user_id) const {
    std::lock_guard lock(m_mutex_clients);
    auto it = m_clients.find(user_id);
    return it != m_clients.end() ? it->second->GetBearerToken() : std::string{};
}


s32 NpHandler::GetUserIdByAccountId(u64 account_id) const {
    std::lock_guard lock(m_mutex_clients);
    for (auto& [uid, client] : m_clients) {
        if (static_cast<u64>(client->GetUserId()) == account_id)
            return uid;
    }
    return -1;
}

s32 NpHandler::GetUserIdByOnlineId(const OrbisNpOnlineId& online_id) const {
    std::lock_guard lock(m_mutex_clients);
    for (auto& [uid, np_id] : m_np_ids) {
        if (strncmp(np_id.handle.data, online_id.data, ORBIS_NP_ONLINEID_MAX_LENGTH) == 0)
            return uid;
    }
    return -1;
}


// all-NUL buffer means npbind.dat was missing or unparsed,treat it as invalid
// so callers reject the request instead of sending a blank comm id on the wire.
bool IsValidNpCommId(const std::string& id) {
    return !id.empty() && std::any_of(id.begin(), id.end(), [](char c) { return c != '\0'; });
}

std::string NpHandler::GetNpCommId(s32 service_label) const {
    // TODO complete guess of how commid is mapping to service_label.
    constexpr size_t COM_ID_LEN = 12;
    const auto& ids = Common::ElfInfo::Instance().GetNpCommIds();
    std::string com_id;
    if (!ids.empty()) {
        const size_t idx = (service_label >= 0 && static_cast<size_t>(service_label) < ids.size())
                               ? static_cast<size_t>(service_label)
                               : 0;
        com_id = ids[idx];
        if (static_cast<size_t>(service_label) >= ids.size()) {
            LOG_WARNING(NpHandler,
                        "GetNpCommId: service_label={} >= npbind entry count {} — falling back "
                        "to index 0",
                        service_label, ids.size());
        }
    }
    if (com_id.size() < COM_ID_LEN) {
        com_id.resize(COM_ID_LEN, '\0');
    } else if (com_id.size() > COM_ID_LEN) {
        com_id.resize(COM_ID_LEN);
    }
    return com_id;
}

s32 NpHandler::RecordScore(s32 user_id, s32 service_label, u32 boardId, s32 pcId, s64 score,
                           const char* comment, size_t commentLen, const u8* gameInfoData,
                           size_t gameInfoSize, std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    // Look up the user's shadNet session.
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "RecordScore: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "RecordScore: user_id={} not authenticated", user_id);
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    // Build RecordScoreRequest proto.
    shadnet::RecordScoreRequest proto;
    proto.set_boardid(boardId);
    proto.set_pcid(pcId);
    proto.set_score(score);
    if (comment && commentLen > 0) {
        proto.set_comment(std::string(comment, commentLen));
    }
    if (gameInfoData && gameInfoSize > 0) {
        proto.set_data(std::string(reinterpret_cast<const char*>(gameInfoData), gameInfoSize));
    }

    const std::string proto_bytes = proto.SerializeAsString();
    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    std::vector<u8> payload;
    payload.reserve(12 + 4 + proto_bytes.size());
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    const u32 sz = static_cast<u32>(proto_bytes.size());
    payload.push_back(static_cast<u8>(sz));
    payload.push_back(static_cast<u8>(sz >> 8));
    payload.push_back(static_cast<u8>(sz >> 16));
    payload.push_back(static_cast<u8>(sz >> 24));
    payload.insert(payload.end(), proto_bytes.begin(), proto_bytes.end());

    const u64 pkt_id = client->SubmitRequest(ShadNet::CommandType::RecordScore, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::RecordScore;
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler,
             "RecordScore: user_id={} service_label={} board={} pcId={} score={} commentLen={} "
             "gameInfoSize={} pkt_id={} com_id='{}'",
             user_id, service_label, boardId, pcId, score, commentLen, gameInfoSize, pkt_id,
             com_id);
    return ORBIS_OK;
}

s32 NpHandler::RecordGameData(s32 user_id, s32 service_label, u32 boardId, s32 pcId, s64 score,
                              const u8* data, size_t size,
                              std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "RecordGameData: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "RecordGameData: user_id={} not authenticated", user_id);
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    shadnet::RecordScoreGameDataRequest proto;
    proto.set_boardid(boardId);
    proto.set_pcid(pcId);
    proto.set_score(score);

    const std::string proto_bytes = proto.SerializeAsString();
    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    std::vector<u8> payload;
    payload.reserve(12 + 4 + proto_bytes.size() + size);
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    const u32 sz = static_cast<u32>(proto_bytes.size());
    payload.push_back(static_cast<u8>(sz));
    payload.push_back(static_cast<u8>(sz >> 8));
    payload.push_back(static_cast<u8>(sz >> 16));
    payload.push_back(static_cast<u8>(sz >> 24));
    payload.insert(payload.end(), proto_bytes.begin(), proto_bytes.end());
    if (data != nullptr && size > 0) {
        payload.insert(payload.end(), data, data + size);
    }

    const u64 pkt_id = client->SubmitRequest(ShadNet::CommandType::RecordScoreData, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::RecordScoreData;
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler,
             "RecordGameData: user_id={} service_label={} board={} pcId={} score={} dataSize={} "
             "pkt_id={} com_id='{}'",
             user_id, service_label, boardId, pcId, score, size, pkt_id, com_id);
    return ORBIS_OK;
}

s32 NpHandler::GetGameData(s32 user_id, s32 service_label, u32 boardId, const std::string& npId,
                           s32 pcId, void* dataOut, u64 recvSize, u64* totalSizeOut,
                           std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "GetGameData: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "GetGameData: user_id={} not authenticated", user_id);
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    shadnet::GetScoreGameDataRequest proto;
    proto.set_boardid(boardId);
    proto.set_npid(npId);
    proto.set_pcid(pcId);

    const std::string proto_bytes = proto.SerializeAsString();
    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    std::vector<u8> payload;
    payload.reserve(12 + 4 + proto_bytes.size());
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    const u32 sz = static_cast<u32>(proto_bytes.size());
    payload.push_back(static_cast<u8>(sz));
    payload.push_back(static_cast<u8>(sz >> 8));
    payload.push_back(static_cast<u8>(sz >> 16));
    payload.push_back(static_cast<u8>(sz >> 24));
    payload.insert(payload.end(), proto_bytes.begin(), proto_bytes.end());

    const u64 pkt_id = client->SubmitRequest(ShadNet::CommandType::GetScoreData, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::GetScoreData;
        pending.dataOut = dataOut;
        pending.recvSize = recvSize;
        pending.totalSizeOut = totalSizeOut;
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler,
             "GetGameData: user_id={} service_label={} board={} npId='{}' pcId={} recvSize={} "
             "pkt_id={} com_id='{}'",
             user_id, service_label, boardId, npId, pcId, recvSize, pkt_id, com_id);
    return ORBIS_OK;
}

s32 NpHandler::GetGameDataByAccountId(s32 user_id, s32 service_label, u32 boardId, u64 accountId,
                                      s32 pcId, void* dataOut, u64 recvSize, u64* totalSizeOut,
                                      std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "GetGameDataByAccountId: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "GetGameDataByAccountId: user_id={} not authenticated", user_id);
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    shadnet::GetScoreGameDataByAccountIdRequest proto;
    proto.set_boardid(boardId);
    proto.set_accountid(accountId);
    proto.set_pcid(pcId);

    const std::string proto_bytes = proto.SerializeAsString();
    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    std::vector<u8> payload;
    payload.reserve(12 + 4 + proto_bytes.size());
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    const u32 sz = static_cast<u32>(proto_bytes.size());
    payload.push_back(static_cast<u8>(sz));
    payload.push_back(static_cast<u8>(sz >> 8));
    payload.push_back(static_cast<u8>(sz >> 16));
    payload.push_back(static_cast<u8>(sz >> 24));
    payload.insert(payload.end(), proto_bytes.begin(), proto_bytes.end());

    const u64 pkt_id =
        client->SubmitRequest(ShadNet::CommandType::GetScoreGameDataByAccId, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::GetScoreGameDataByAccId;
        pending.dataOut = dataOut;
        pending.recvSize = recvSize;
        pending.totalSizeOut = totalSizeOut;
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler,
             "GetGameDataByAccountId: user_id={} service_label={} board={} accountId={} pcId={} "
             "recvSize={} pkt_id={} com_id='{}'",
             user_id, service_label, boardId, accountId, pcId, recvSize, pkt_id, com_id);
    return ORBIS_OK;
}

s32 NpHandler::GetBoardInfo(s32 user_id, s32 service_label, u32 boardId,
                            NpScore::OrbisNpScoreBoardInfo* boardInfo,
                            std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "GetBoardInfo: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "GetBoardInfo: user_id={} not authenticated", user_id);
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }
    std::vector<u8> payload;
    payload.reserve(12 + 4);
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    payload.push_back(static_cast<u8>(boardId));
    payload.push_back(static_cast<u8>(boardId >> 8));
    payload.push_back(static_cast<u8>(boardId >> 16));
    payload.push_back(static_cast<u8>(boardId >> 24));

    const u64 pkt_id = client->SubmitRequest(ShadNet::CommandType::GetBoardInfos, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::GetBoardInfos;
        pending.boardInfo = boardInfo;
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler, "GetBoardInfo: user_id={} service_label={} board={} pkt_id={} com_id='{}'",
             user_id, service_label, boardId, pkt_id, com_id);
    return ORBIS_OK;
}

s32 NpHandler::GetRankingByNpId(s32 user_id, s32 service_label, u32 boardId,
                                const std::vector<std::string>& npIds,
                                const std::vector<s32>& pcIds,
                                NpScore::OrbisNpScorePlayerRankData* rankArray,
                                NpScore::OrbisNpScoreComment* commentArray,
                                NpScore::OrbisNpScoreGameInfo* infoArray,
                                Libraries::Rtc::OrbisRtcTick* lastSortDate, u32* totalRecord,
                                std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    // pcIds must either be empty (use 0 for everything) or match npIds in size.
    if (!pcIds.empty() && pcIds.size() != npIds.size()) {
        LOG_ERROR(NpHandler, "GetRankingByNpId: pcIds size {} != npIds size {}", pcIds.size(),
                  npIds.size());
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    // Look up the user's session.
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "GetRankingByNpId: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "GetRankingByNpId: user_id={} not authenticated", user_id);
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    shadnet::GetScoreNpIdRequest proto;
    proto.set_boardid(boardId);
    proto.set_withcomment(commentArray != nullptr);
    proto.set_withgameinfo(infoArray != nullptr);
    for (size_t i = 0; i < npIds.size(); ++i) {
        auto* entry = proto.add_npids();
        entry->set_npid(npIds[i]);
        entry->set_pcid(pcIds.empty() ? 0 : pcIds[i]);
    }

    const std::string proto_bytes = proto.SerializeAsString();
    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    std::vector<u8> payload;
    payload.reserve(12 + 4 + proto_bytes.size());
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    const u32 sz = static_cast<u32>(proto_bytes.size());
    payload.push_back(static_cast<u8>(sz));
    payload.push_back(static_cast<u8>(sz >> 8));
    payload.push_back(static_cast<u8>(sz >> 16));
    payload.push_back(static_cast<u8>(sz >> 24));
    payload.insert(payload.end(), proto_bytes.begin(), proto_bytes.end());

    const u64 pkt_id = client->SubmitRequest(ShadNet::CommandType::GetScoreNpid, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::GetScoreNpid;
        pending.requestedNpIds = npIds;
        pending.rankArray = rankArray;
        pending.commentArray = commentArray;
        pending.infoArray = infoArray;
        pending.lastSortDate = lastSortDate;
        pending.totalRecord = totalRecord;
        pending.arrayNum = npIds.size();
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler,
             "GetRankingByNpId: user_id={} service_label={} board={} npIdCount={} "
             "withPcId={} withComment={} withGameInfo={} pkt_id={} com_id='{}'",
             user_id, service_label, boardId, npIds.size(), !pcIds.empty(), commentArray != nullptr,
             infoArray != nullptr, pkt_id, com_id);
    return ORBIS_OK;
}

s32 NpHandler::GetRankingByRange(s32 user_id, s32 service_label, u32 boardId, u32 startSerialRank,
                                 u32 arrayNum, NpScore::OrbisNpScoreRankData* rankArray,
                                 NpScore::OrbisNpScoreComment* commentArray,
                                 NpScore::OrbisNpScoreGameInfo* infoArray,
                                 Libraries::Rtc::OrbisRtcTick* lastSortDate, u32* totalRecord,
                                 std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "GetRankingByRange: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "GetRankingByRange: user_id={} not authenticated", user_id);
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    shadnet::GetScoreRangeRequest proto;
    proto.set_boardid(boardId);
    proto.set_startrank(startSerialRank);
    proto.set_numranks(arrayNum);
    proto.set_withcomment(commentArray != nullptr);
    proto.set_withgameinfo(infoArray != nullptr);

    const std::string proto_bytes = proto.SerializeAsString();
    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    std::vector<u8> payload;
    payload.reserve(12 + 4 + proto_bytes.size());
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    const u32 sz = static_cast<u32>(proto_bytes.size());
    payload.push_back(static_cast<u8>(sz));
    payload.push_back(static_cast<u8>(sz >> 8));
    payload.push_back(static_cast<u8>(sz >> 16));
    payload.push_back(static_cast<u8>(sz >> 24));
    payload.insert(payload.end(), proto_bytes.begin(), proto_bytes.end());

    const u64 pkt_id = client->SubmitRequest(ShadNet::CommandType::GetScoreRange, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::GetScoreRange;
        pending.plainRankArray = rankArray;
        pending.commentArray = commentArray;
        pending.infoArray = infoArray;
        pending.lastSortDate = lastSortDate;
        pending.totalRecord = totalRecord;
        pending.arrayNum = arrayNum;
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler,
             "GetRankingByRange: user_id={} service_label={} board={} startRank={} numRanks={} "
             "withComment={} withGameInfo={} pkt_id={} com_id='{}'",
             user_id, service_label, boardId, startSerialRank, arrayNum, commentArray != nullptr,
             infoArray != nullptr, pkt_id, com_id);
    return ORBIS_OK;
}

s32 NpHandler::GetRankingByRangeA(s32 user_id, s32 service_label, u32 boardId, u32 startSerialRank,
                                  u32 arrayNum, NpScore::OrbisNpScoreRankDataA* rankArray,
                                  NpScore::OrbisNpScoreComment* commentArray,
                                  NpScore::OrbisNpScoreGameInfo* infoArray,
                                  Libraries::Rtc::OrbisRtcTick* lastSortDate, u32* totalRecord,
                                  std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "GetRankingByRangeA: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "GetRankingByRangeA: user_id={} not authenticated", user_id);
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    shadnet::GetScoreRangeRequest proto;
    proto.set_boardid(boardId);
    proto.set_startrank(startSerialRank);
    proto.set_numranks(arrayNum);
    proto.set_withcomment(commentArray != nullptr);
    proto.set_withgameinfo(infoArray != nullptr);

    const std::string proto_bytes = proto.SerializeAsString();
    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    std::vector<u8> payload;
    payload.reserve(12 + 4 + proto_bytes.size());
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    const u32 sz = static_cast<u32>(proto_bytes.size());
    payload.push_back(static_cast<u8>(sz));
    payload.push_back(static_cast<u8>(sz >> 8));
    payload.push_back(static_cast<u8>(sz >> 16));
    payload.push_back(static_cast<u8>(sz >> 24));
    payload.insert(payload.end(), proto_bytes.begin(), proto_bytes.end());

    const u64 pkt_id = client->SubmitRequest(ShadNet::CommandType::GetScoreRange, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::GetScoreRange;
        // Note: aRankArray is set, plainRankArray remains null. The shared
        // GetScoreRange/GetScoreFriends branch in OnScoreReply dispatches
        // on which one is non-null to pick the right fill helper.
        pending.aRankArray = rankArray;
        pending.commentArray = commentArray;
        pending.infoArray = infoArray;
        pending.lastSortDate = lastSortDate;
        pending.totalRecord = totalRecord;
        pending.arrayNum = arrayNum;
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler,
             "GetRankingByRangeA: user_id={} service_label={} board={} startRank={} numRanks={} "
             "withComment={} withGameInfo={} pkt_id={} com_id='{}'",
             user_id, service_label, boardId, startSerialRank, arrayNum, commentArray != nullptr,
             infoArray != nullptr, pkt_id, com_id);
    return ORBIS_OK;
}

s32 NpHandler::GetRankingByAccountId(s32 user_id, s32 service_label, u32 boardId,
                                     const std::vector<u64>& accountIds,
                                     const std::vector<s32>& pcIds,
                                     NpScore::OrbisNpScorePlayerRankDataA* rankArray,
                                     NpScore::OrbisNpScoreComment* commentArray,
                                     NpScore::OrbisNpScoreGameInfo* infoArray,
                                     Libraries::Rtc::OrbisRtcTick* lastSortDate, u32* totalRecord,
                                     std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "GetRankingByAccountId: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    shadnet::GetScoreAccountIdRequest proto;
    proto.set_boardid(boardId);
    for (size_t i = 0; i < accountIds.size(); ++i) {
        auto* entry = proto.add_ids();
        entry->set_accountid(static_cast<int64_t>(accountIds[i]));
        // If the caller provided per-entry pcIds, use them; otherwise all zero.
        entry->set_pcid(i < pcIds.size() ? pcIds[i] : 0);
    }
    proto.set_withcomment(commentArray != nullptr);
    proto.set_withgameinfo(infoArray != nullptr);

    const std::string proto_bytes = proto.SerializeAsString();
    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    std::vector<u8> payload;
    payload.reserve(12 + 4 + proto_bytes.size());
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    const u32 sz = static_cast<u32>(proto_bytes.size());
    payload.push_back(static_cast<u8>(sz));
    payload.push_back(static_cast<u8>(sz >> 8));
    payload.push_back(static_cast<u8>(sz >> 16));
    payload.push_back(static_cast<u8>(sz >> 24));
    payload.insert(payload.end(), proto_bytes.begin(), proto_bytes.end());

    const u64 pkt_id = client->SubmitRequest(ShadNet::CommandType::GetScoreAccountId, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::GetScoreAccountId;
        pending.aPlayerRankArray = rankArray;
        pending.commentArray = commentArray;
        pending.infoArray = infoArray;
        pending.lastSortDate = lastSortDate;
        pending.totalRecord = totalRecord;
        pending.arrayNum = accountIds.size();
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler,
             "GetRankingByAccountId: user_id={} service_label={} board={} accountIdCount={} "
             "withComment={} withGameInfo={} pkt_id={} com_id='{}'",
             user_id, service_label, boardId, accountIds.size(), commentArray != nullptr,
             infoArray != nullptr, pkt_id, com_id);
    return ORBIS_OK;
}

s32 NpHandler::GetFriendsRanking(s32 user_id, s32 service_label, u32 boardId, bool includeSelf,
                                 u32 arrayNum, NpScore::OrbisNpScoreRankData* rankArray,
                                 NpScore::OrbisNpScoreComment* commentArray,
                                 NpScore::OrbisNpScoreGameInfo* infoArray,
                                 Libraries::Rtc::OrbisRtcTick* lastSortDate, u32* totalRecord,
                                 std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "GetFriendsRanking: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "GetFriendsRanking: user_id={} not authenticated", user_id);
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    shadnet::GetScoreFriendsRequest proto;
    proto.set_boardid(boardId);
    proto.set_includeself(includeSelf);
    proto.set_max(arrayNum);
    proto.set_withcomment(commentArray != nullptr);
    proto.set_withgameinfo(infoArray != nullptr);

    const std::string proto_bytes = proto.SerializeAsString();
    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    std::vector<u8> payload;
    payload.reserve(12 + 4 + proto_bytes.size());
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    const u32 sz = static_cast<u32>(proto_bytes.size());
    payload.push_back(static_cast<u8>(sz));
    payload.push_back(static_cast<u8>(sz >> 8));
    payload.push_back(static_cast<u8>(sz >> 16));
    payload.push_back(static_cast<u8>(sz >> 24));
    payload.insert(payload.end(), proto_bytes.begin(), proto_bytes.end());

    const u64 pkt_id = client->SubmitRequest(ShadNet::CommandType::GetScoreFriends, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::GetScoreFriends;
        pending.plainRankArray = rankArray;
        pending.commentArray = commentArray;
        pending.infoArray = infoArray;
        pending.lastSortDate = lastSortDate;
        pending.totalRecord = totalRecord;
        pending.arrayNum = arrayNum;
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler,
             "GetFriendsRanking: user_id={} service_label={} board={} includeSelf={} "
             "arrayNum={} withComment={} withGameInfo={} pkt_id={} com_id='{}'",
             user_id, service_label, boardId, includeSelf, arrayNum, commentArray != nullptr,
             infoArray != nullptr, pkt_id, com_id);
    return ORBIS_OK;
}

s32 NpHandler::GetFriendsRankingA(s32 user_id, s32 service_label, u32 boardId, bool includeSelf,
                                  u32 arrayNum, NpScore::OrbisNpScoreRankDataA* rankArray,
                                  NpScore::OrbisNpScoreComment* commentArray,
                                  NpScore::OrbisNpScoreGameInfo* infoArray,
                                  Libraries::Rtc::OrbisRtcTick* lastSortDate, u32* totalRecord,
                                  std::shared_ptr<NpScore::ScoreRequestCtx> req) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_WARNING(NpHandler, "GetFriendsRankingA: user_id={} not connected", user_id);
            return ORBIS_NP_ERROR_SIGNED_OUT;
        }
        client = it->second;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "GetFriendsRankingA: user_id={} not authenticated", user_id);
        return ORBIS_NP_COMMUNITY_ERROR_NO_LOGIN;
    }

    shadnet::GetScoreFriendsRequest proto;
    proto.set_boardid(boardId);
    proto.set_includeself(includeSelf);
    proto.set_max(arrayNum);
    proto.set_withcomment(commentArray != nullptr);
    proto.set_withgameinfo(infoArray != nullptr);

    const std::string proto_bytes = proto.SerializeAsString();
    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_ERROR(NpHandler, "no valid NP Communication ID (npbind.dat missing or blank); "
                             "rejecting request");
        return ORBIS_NP_COMMUNITY_ERROR_INVALID_ARGUMENT;
    }

    std::vector<u8> payload;
    payload.reserve(12 + 4 + proto_bytes.size());
    payload.insert(payload.end(), com_id.begin(), com_id.end());
    const u32 sz = static_cast<u32>(proto_bytes.size());
    payload.push_back(static_cast<u8>(sz));
    payload.push_back(static_cast<u8>(sz >> 8));
    payload.push_back(static_cast<u8>(sz >> 16));
    payload.push_back(static_cast<u8>(sz >> 24));
    payload.insert(payload.end(), proto_bytes.begin(), proto_bytes.end());

    const u64 pkt_id = client->SubmitRequest(ShadNet::CommandType::GetScoreFriends, payload);
    {
        std::lock_guard lock(m_mutex_pending_score);
        PendingScoreRequest pending;
        pending.req = std::move(req);
        pending.cmd = ShadNet::CommandType::GetScoreFriends;
        // Note: aRankArray is set, plainRankArray remains null. OnScoreReply
        // dispatches on which one is non-null.
        pending.aRankArray = rankArray;
        pending.commentArray = commentArray;
        pending.infoArray = infoArray;
        pending.lastSortDate = lastSortDate;
        pending.totalRecord = totalRecord;
        pending.arrayNum = arrayNum;
        pending.user_id = user_id;
        m_pending_score.emplace(pkt_id, std::move(pending));
    }
    LOG_INFO(NpHandler,
             "GetFriendsRankingA: user_id={} service_label={} board={} includeSelf={} "
             "arrayNum={} withComment={} withGameInfo={} pkt_id={} com_id='{}'",
             user_id, service_label, boardId, includeSelf, arrayNum, commentArray != nullptr,
             infoArray != nullptr, pkt_id, com_id);
    return ORBIS_OK;
}


static u32 FillPlainRankArrayFromProto(const shadnet::GetScoreResponse& resp, u64 maxSlots,
                                       NpScore::OrbisNpScoreRankData* rankArray,
                                       NpScore::OrbisNpScoreComment* commentArray,
                                       NpScore::OrbisNpScoreGameInfo* infoArray) {
    const int n_resp = resp.rankarray_size();
    u32 out_i = 0;
    for (int i = 0; i < n_resp && out_i < maxSlots; ++i) {
        const auto& r = resp.rankarray(i);
        if (r.npid().empty()) {
            continue;
        }
        auto& out = rankArray[out_i];
        // Copy OnlineId (the npId string) into the 16-byte data field.
        const std::string& npid = r.npid();
        const size_t cp = std::min<size_t>(npid.size(), sizeof(out.npId.handle.data));
        std::memcpy(out.npId.handle.data, npid.data(), cp);
        out.npId.handle.term = 0;
        LOG_INFO(NpHandler,
                 "FillPlainRankArrayFromProto: out[{}] (resp[{}]) npid='{}' (len={}) rank={} "
                 "score={}",
                 out_i, i, npid, npid.size(), r.rank(), r.score());
        out.pcId = r.pcid();
        out.serialRank = r.rank();
        out.rank = r.rank();
        out.highestRank = r.rank();
        out.hasGameData = r.hasgamedata() ? 1 : 0;
        out.scoreValue = r.score();
        out.recordDate.tick = r.recorddate();

        if (commentArray != nullptr && i < resp.commentarray_size()) {
            const std::string& cmt = resp.commentarray(i);
            const size_t ccp =
                std::min<size_t>(cmt.size(), sizeof(commentArray[out_i].utf8Comment) - 1);
            std::memcpy(commentArray[out_i].utf8Comment, cmt.data(), ccp);
        }
        if (infoArray != nullptr && i < resp.infoarray_size()) {
            const std::string& gi = resp.infoarray(i).data();
            const size_t gcp = std::min<size_t>(gi.size(), sizeof(infoArray[out_i].data));
            infoArray[out_i].infoSize = static_cast<u32>(gcp);
            std::memcpy(infoArray[out_i].data, gi.data(), gcp);
        }
        ++out_i;
    }
    return out_i;
}

static u32 FillPlainRankArrayAFromProto(const shadnet::GetScoreResponse& resp, u64 maxSlots,
                                        NpScore::OrbisNpScoreRankDataA* rankArray,
                                        NpScore::OrbisNpScoreComment* commentArray,
                                        NpScore::OrbisNpScoreGameInfo* infoArray) {
    const int n_resp = resp.rankarray_size();
    u32 out_i = 0;
    for (int i = 0; i < n_resp && out_i < maxSlots; ++i) {
        const auto& r = resp.rankarray(i);
        if (r.npid().empty()) {
            continue;
        }
        auto& out = rankArray[out_i];
        // RankDataA stores OnlineId directly (not wrapped in NpId).
        const std::string& npid = r.npid();
        const size_t cp = std::min<size_t>(npid.size(), sizeof(out.onlineId.data));
        std::memcpy(out.onlineId.data, npid.data(), cp);
        out.onlineId.term = 0;
        LOG_INFO(NpHandler,
                 "FillPlainRankArrayAFromProto: out[{}] (resp[{}]) npid='{}' (len={}) rank={} "
                 "score={} accountId={}",
                 out_i, i, npid, npid.size(), r.rank(), r.score(), r.accountid());
        out.pcId = r.pcid();
        out.serialRank = r.rank();
        out.rank = r.rank();
        out.highestRank = r.rank();
        out.hasGameData = r.hasgamedata() ? 1 : 0;
        out.scoreValue = r.score();
        out.recordDate.tick = r.recorddate();
        out.accountId = static_cast<Libraries::Np::OrbisNpAccountId>(r.accountid());

        if (commentArray != nullptr && i < resp.commentarray_size()) {
            const std::string& cmt = resp.commentarray(i);
            const size_t ccp =
                std::min<size_t>(cmt.size(), sizeof(commentArray[out_i].utf8Comment) - 1);
            std::memcpy(commentArray[out_i].utf8Comment, cmt.data(), ccp);
        }
        if (infoArray != nullptr && i < resp.infoarray_size()) {
            const std::string& gi = resp.infoarray(i).data();
            const size_t gcp = std::min<size_t>(gi.size(), sizeof(infoArray[out_i].data));
            infoArray[out_i].infoSize = static_cast<u32>(gcp);
            std::memcpy(infoArray[out_i].data, gi.data(), gcp);
        }
        ++out_i;
    }
    return out_i;
}

static u32 FillPlayerRankArrayAFromProto(const shadnet::GetScoreResponse& resp, u64 requestedCount,
                                         NpScore::OrbisNpScorePlayerRankDataA* rankArray,
                                         NpScore::OrbisNpScoreComment* commentArray,
                                         NpScore::OrbisNpScoreGameInfo* infoArray) {
    const int n_resp = resp.rankarray_size();
    const int n_req = static_cast<int>(requestedCount);
    if (n_resp != n_req) {
        LOG_WARNING(NpHandler,
                    "FillPlayerRankArrayAFromProto: response count {} != request count {} — "
                    "will fill up to min() and leave extras at hasData=0",
                    n_resp, n_req);
    }
    const int n = std::min(n_resp, n_req);
    u32 found = 0;
    for (int i = 0; i < n; ++i) {
        const auto& r = resp.rankarray(i);
        if (r.npid().empty()) {
            continue; // no score on this board for this input accountId
        }
        auto& out = rankArray[i];
        out.hasData = 1;
        const std::string& npid = r.npid();
        const size_t cp = std::min<size_t>(npid.size(), sizeof(out.rankData.onlineId.data));
        std::memcpy(out.rankData.onlineId.data, npid.data(), cp);
        out.rankData.onlineId.term = 0;
        out.rankData.pcId = r.pcid();
        out.rankData.serialRank = r.rank();
        out.rankData.rank = r.rank();
        out.rankData.highestRank = r.rank();
        out.rankData.hasGameData = r.hasgamedata() ? 1 : 0;
        out.rankData.scoreValue = r.score();
        out.rankData.recordDate.tick = r.recorddate();
        out.rankData.accountId = static_cast<Libraries::Np::OrbisNpAccountId>(r.accountid());

        if (commentArray != nullptr && i < resp.commentarray_size()) {
            const std::string& cmt = resp.commentarray(i);
            const size_t ccp =
                std::min<size_t>(cmt.size(), sizeof(commentArray[i].utf8Comment) - 1);
            std::memcpy(commentArray[i].utf8Comment, cmt.data(), ccp);
        }
        if (infoArray != nullptr && i < resp.infoarray_size()) {
            const std::string& gi = resp.infoarray(i).data();
            const size_t gcp = std::min<size_t>(gi.size(), sizeof(infoArray[i].data));
            infoArray[i].infoSize = static_cast<u32>(gcp);
            std::memcpy(infoArray[i].data, gi.data(), gcp);
        }
        ++found;
    }
    return found;
}

static u32 FillRankArrayFromProto(const shadnet::GetScoreResponse& resp,
                                  const std::vector<std::string>& requestedNpIds,
                                  NpScore::OrbisNpScorePlayerRankData* rankArray,
                                  NpScore::OrbisNpScoreComment* commentArray,
                                  NpScore::OrbisNpScoreGameInfo* infoArray) {
    const int n_resp = resp.rankarray_size();
    const int n_req = static_cast<int>(requestedNpIds.size());
    if (n_resp != n_req) {
        LOG_WARNING(NpHandler,
                    "FillRankArrayFromProto: response count {} != request count {} — "
                    "will fill up to min() and leave extras at hasData=0",
                    n_resp, n_req);
    }
    const int n = std::min(n_resp, n_req);
    u32 found = 0;
    for (int i = 0; i < n; ++i) {
        const auto& r = resp.rankarray(i);
        if (r.npid().empty()) {
            continue; // server signalled "no data for this slot"
        }

        auto& out = rankArray[i];
        out.hasData = 1;
        const std::string& npid = r.npid();
        const size_t cp = std::min<size_t>(npid.size(), sizeof(out.rankData.npId.handle.data));
        std::memcpy(out.rankData.npId.handle.data, npid.data(), cp);
        out.rankData.npId.handle.term = 0;
        out.rankData.pcId = r.pcid();
        out.rankData.serialRank = r.rank();
        out.rankData.rank = r.rank();
        out.rankData.highestRank = r.rank();
        out.rankData.hasGameData = r.hasgamedata() ? 1 : 0;
        out.rankData.scoreValue = r.score();
        out.rankData.recordDate.tick = r.recorddate();

        if (commentArray != nullptr && i < resp.commentarray_size()) {
            const std::string& cmt = resp.commentarray(i);
            const size_t ccp =
                std::min<size_t>(cmt.size(), sizeof(commentArray[i].utf8Comment) - 1);
            std::memcpy(commentArray[i].utf8Comment, cmt.data(), ccp);
        }
        if (infoArray != nullptr && i < resp.infoarray_size()) {
            const std::string& gi = resp.infoarray(i).data();
            const size_t gcp = std::min<size_t>(gi.size(), sizeof(infoArray[i].data));
            infoArray[i].infoSize = static_cast<u32>(gcp);
            std::memcpy(infoArray[i].data, gi.data(), gcp);
        }
        ++found;
    }
    return found;
}

void NpHandler::OnScoreReply(s32 user_id, ShadNet::CommandType cmd, u64 pkt_id,
                             ShadNet::ErrorType error, const std::vector<u8>& body) {
    PendingScoreRequest pending;
    {
        std::lock_guard lock(m_mutex_pending_score);
        auto it = m_pending_score.find(pkt_id);
        if (it == m_pending_score.end()) {
            LOG_WARNING(NpHandler, "OnScoreReply: no pending request for pkt_id={} cmd={}", pkt_id,
                        static_cast<int>(cmd));
            return;
        }
        pending = std::move(it->second);
        m_pending_score.erase(it);
    }
    auto& req = pending.req;

    // Server-side errors take precedence over success-path parsing.
    if (error != ShadNet::ErrorType::NoError) {
        s32 orbis_err = ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE;
        switch (error) {
        case ShadNet::ErrorType::ScoreNotBest:
            orbis_err = ORBIS_NP_COMMUNITY_SERVER_ERROR_NOT_BEST_SCORE;
            break;
        case ShadNet::ErrorType::ScoreInvalid:
            orbis_err = ORBIS_NP_COMMUNITY_SERVER_ERROR_INVALID_SCORE;
            break;
        case ShadNet::ErrorType::NotFound:
            orbis_err = ORBIS_NP_COMMUNITY_SERVER_ERROR_RANKING_BOARD_MASTER_NOT_FOUND;
            break;
        case ShadNet::ErrorType::Unauthorized:
            orbis_err = ORBIS_NP_COMMUNITY_SERVER_ERROR_FORBIDDEN;
            break;
        case ShadNet::ErrorType::DbFail:
            orbis_err = ORBIS_NP_COMMUNITY_SERVER_ERROR_INTERNAL_SERVER_ERROR;
            break;
        case ShadNet::ErrorType::Malformed:
        case ShadNet::ErrorType::Invalid:
        case ShadNet::ErrorType::InvalidInput:
            orbis_err = ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE;
            break;
        default:
            break;
        }
        LOG_WARNING(NpHandler,
                    "OnScoreReply: user_id={} pkt_id={} cmd={} server_error={} → orbis_err={:#x}",
                    user_id, pkt_id, static_cast<int>(cmd), static_cast<int>(error), orbis_err);
        req->SetResult(orbis_err);
        return;
    }

    // Per-command success-path parsing.
    switch (cmd) {
    case ShadNet::CommandType::RecordScore: {
        // Reply body = u32 LE rank.
        if (req->tmpRankOut != nullptr && body.size() >= 4) {
            const u32 rank = static_cast<u32>(body[0]) | (static_cast<u32>(body[1]) << 8) |
                             (static_cast<u32>(body[2]) << 16) | (static_cast<u32>(body[3]) << 24);
            *req->tmpRankOut = rank;
            LOG_INFO(NpHandler, "OnScoreReply: RecordScore user_id={} pkt_id={} rank={}", user_id,
                     pkt_id, rank);
        }
        req->SetResult(ORBIS_OK);
        break;
    }

    case ShadNet::CommandType::RecordScoreData: {
        // Reply body is empty on success — error byte already consumed by
        // the caller. Server-side errors are mapped to ErrorType values
        // (NotFound / ScoreInvalid / ScoreHasData) which arrive as the
        // packet's error byte; if we got here, error == NoError.
        LOG_INFO(NpHandler, "OnScoreReply: RecordScoreData user_id={} pkt_id={} ok", user_id,
                 pkt_id);
        req->SetResult(ORBIS_OK);
        break;
    }

    case ShadNet::CommandType::GetScoreData:
    case ShadNet::CommandType::GetScoreGameDataByAccId: {
        const char* cmd_name = (cmd == ShadNet::CommandType::GetScoreData)
                                   ? "GetScoreData"
                                   : "GetScoreGameDataByAccId";
        if (body.size() < 4) {
            LOG_ERROR(NpHandler, "OnScoreReply: {} body too small ({})", cmd_name, body.size());
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        const u32 stored_size = static_cast<u32>(body[0]) | (static_cast<u32>(body[1]) << 8) |
                                (static_cast<u32>(body[2]) << 16) |
                                (static_cast<u32>(body[3]) << 24);
        if (static_cast<size_t>(4) + stored_size > body.size()) {
            LOG_ERROR(NpHandler, "OnScoreReply: {} blob size {} exceeds body size {}", cmd_name,
                      stored_size, body.size());
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }

        // Always populate *totalSizeOut with the actual stored size on
        // the server, even when truncating into a smaller dataOut buffer.
        // Lets the caller know if they undersized their buffer and need
        // to retry with a larger one.
        if (pending.totalSizeOut != nullptr) {
            *pending.totalSizeOut = stored_size;
        }
        if (pending.dataOut != nullptr && pending.recvSize > 0) {
            const u64 copy_n = std::min<u64>(static_cast<u64>(stored_size), pending.recvSize);
            std::memcpy(pending.dataOut, body.data() + 4, static_cast<size_t>(copy_n));
            if (copy_n < stored_size) {
                LOG_WARNING(NpHandler,
                            "OnScoreReply: {} truncated user_id={} pkt_id={} stored={} recv={}",
                            cmd_name, user_id, pkt_id, stored_size, pending.recvSize);
            }
        }
        LOG_INFO(NpHandler, "OnScoreReply: {} user_id={} pkt_id={} storedSize={} recvSize={}",
                 cmd_name, user_id, pkt_id, stored_size, pending.recvSize);
        req->SetResult(ORBIS_OK);
        break;
    }

    case ShadNet::CommandType::GetBoardInfos: {
        if (body.size() < 4) {
            LOG_ERROR(NpHandler, "OnScoreReply: GetBoardInfos body too small ({})", body.size());
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        const u32 proto_size = static_cast<u32>(body[0]) | (static_cast<u32>(body[1]) << 8) |
                               (static_cast<u32>(body[2]) << 16) |
                               (static_cast<u32>(body[3]) << 24);
        if (static_cast<size_t>(4) + proto_size > body.size()) {
            LOG_ERROR(NpHandler, "OnScoreReply: GetBoardInfos proto size {} exceeds body size {}",
                      proto_size, body.size());
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        shadnet::BoardInfo bi;
        if (!bi.ParseFromArray(body.data() + 4, static_cast<int>(proto_size))) {
            LOG_ERROR(NpHandler, "OnScoreReply: GetBoardInfos proto parse failed");
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        if (pending.boardInfo != nullptr) {
            pending.boardInfo->rankLimit = bi.ranklimit();
            pending.boardInfo->updateMode = bi.updatemode();
            pending.boardInfo->sortMode = bi.sortmode();
            pending.boardInfo->uploadNumLimit = bi.uploadnumlimit();
            pending.boardInfo->uploadSizeLimit = bi.uploadsizelimit();
        }
        LOG_INFO(NpHandler,
                 "OnScoreReply: GetBoardInfos user_id={} pkt_id={} rankLimit={} updateMode={} "
                 "sortMode={} uploadNumLimit={} uploadSizeLimit={}",
                 user_id, pkt_id, bi.ranklimit(), bi.updatemode(), bi.sortmode(),
                 bi.uploadnumlimit(), bi.uploadsizelimit());
        req->SetResult(ORBIS_OK);
        break;
    }

    case ShadNet::CommandType::GetScoreNpid: {
        // Reply body = u32 LE proto size + GetScoreResponse proto bytes.
        if (body.size() < 4) {
            LOG_ERROR(NpHandler, "OnScoreReply: GetScoreNpid body too small ({})", body.size());
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        const u32 proto_size = static_cast<u32>(body[0]) | (static_cast<u32>(body[1]) << 8) |
                               (static_cast<u32>(body[2]) << 16) |
                               (static_cast<u32>(body[3]) << 24);
        if (static_cast<size_t>(4) + proto_size > body.size()) {
            LOG_ERROR(NpHandler, "OnScoreReply: GetScoreNpid proto size {} exceeds body size {}",
                      proto_size, body.size());
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        shadnet::GetScoreResponse resp;
        if (!resp.ParseFromArray(body.data() + 4, static_cast<int>(proto_size))) {
            LOG_ERROR(NpHandler, "OnScoreReply: GetScoreNpid proto parse failed");
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        const u32 found = FillRankArrayFromProto(resp, pending.requestedNpIds, pending.rankArray,
                                                 pending.commentArray, pending.infoArray);
        if (pending.lastSortDate != nullptr) {
            pending.lastSortDate->tick = resp.lastsortdate();
        }
        if (pending.totalRecord != nullptr) {
            *pending.totalRecord = resp.totalrecord();
        }
        LOG_INFO(NpHandler,
                 "OnScoreReply: GetScoreNpid user_id={} pkt_id={} found={}/{} totalRecord={}",
                 user_id, pkt_id, found, pending.arrayNum, resp.totalrecord());
        req->SetResult(static_cast<s32>(found));
        break;
    }
    case ShadNet::CommandType::GetScoreAccountId: {
        // Reply body = u32 LE proto size + GetScoreResponse proto bytes.
        // Index-aligned with the request's accountIds[] (empty-npid sentinel
        // marks "no score on this board"), same contract as GetScoreNpid.
        if (body.size() < 4) {
            LOG_ERROR(NpHandler, "OnScoreReply: GetScoreAccountId body too small ({})",
                      body.size());
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        const u32 proto_size = static_cast<u32>(body[0]) | (static_cast<u32>(body[1]) << 8) |
                               (static_cast<u32>(body[2]) << 16) |
                               (static_cast<u32>(body[3]) << 24);
        if (static_cast<size_t>(4) + proto_size > body.size()) {
            LOG_ERROR(NpHandler,
                      "OnScoreReply: GetScoreAccountId proto size {} exceeds body size {}",
                      proto_size, body.size());
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        shadnet::GetScoreResponse resp;
        if (!resp.ParseFromArray(body.data() + 4, static_cast<int>(proto_size))) {
            LOG_ERROR(NpHandler, "OnScoreReply: GetScoreAccountId proto parse failed");
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        const u32 found =
            FillPlayerRankArrayAFromProto(resp, pending.arrayNum, pending.aPlayerRankArray,
                                          pending.commentArray, pending.infoArray);
        if (pending.lastSortDate != nullptr) {
            pending.lastSortDate->tick = resp.lastsortdate();
        }
        if (pending.totalRecord != nullptr) {
            *pending.totalRecord = resp.totalrecord();
        }
        LOG_INFO(NpHandler,
                 "OnScoreReply: GetScoreAccountId user_id={} pkt_id={} found={}/{} totalRecord={}",
                 user_id, pkt_id, found, pending.arrayNum, resp.totalrecord());
        req->SetResult(static_cast<s32>(found));
        break;
    }

    case ShadNet::CommandType::GetScoreRange:
    case ShadNet::CommandType::GetScoreFriends: {
        const char* cmd_name =
            (cmd == ShadNet::CommandType::GetScoreRange) ? "GetScoreRange" : "GetScoreFriends";
        if (body.size() < 4) {
            LOG_ERROR(NpHandler, "OnScoreReply: {} body too small ({})", cmd_name, body.size());
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        const u32 proto_size = static_cast<u32>(body[0]) | (static_cast<u32>(body[1]) << 8) |
                               (static_cast<u32>(body[2]) << 16) |
                               (static_cast<u32>(body[3]) << 24);
        if (static_cast<size_t>(4) + proto_size > body.size()) {
            LOG_ERROR(NpHandler, "OnScoreReply: {} proto size {} exceeds body size {}", cmd_name,
                      proto_size, body.size());
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        shadnet::GetScoreResponse resp;
        if (!resp.ParseFromArray(body.data() + 4, static_cast<int>(proto_size))) {
            LOG_ERROR(NpHandler, "OnScoreReply: {} proto parse failed", cmd_name);
            req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
            break;
        }
        u32 found = 0;
        if (pending.aRankArray != nullptr) {
            found = FillPlainRankArrayAFromProto(resp, pending.arrayNum, pending.aRankArray,
                                                 pending.commentArray, pending.infoArray);
        } else {
            found = FillPlainRankArrayFromProto(resp, pending.arrayNum, pending.plainRankArray,
                                                pending.commentArray, pending.infoArray);
        }
        if (pending.lastSortDate != nullptr) {
            pending.lastSortDate->tick = resp.lastsortdate();
        }
        if (pending.totalRecord != nullptr) {
            *pending.totalRecord = resp.totalrecord();
        }
        LOG_INFO(NpHandler, "OnScoreReply: {} user_id={} pkt_id={} found={}/{} totalRecord={}{}",
                 cmd_name, user_id, pkt_id, found, pending.arrayNum, resp.totalrecord(),
                 pending.aRankArray != nullptr ? " (A-variant)" : "");
        req->SetResult(static_cast<s32>(found));
        break;
    }

    default:
        LOG_WARNING(NpHandler, "OnScoreReply: unexpected cmd={} pkt_id={}", static_cast<int>(cmd),
                    pkt_id);
        req->SetResult(ORBIS_NP_COMMUNITY_ERROR_BAD_RESPONSE);
        break;
    }
}

void NpHandler::ReportTrophyUnlock(s32 user_id, s32 service_label, s32 trophy_id, u64 timestamp) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_DEBUG(NpHandler, "ReportTrophyUnlock: no shadNet session for user_id={}; skipping",
                      user_id);
            return;
        }
        client = it->second;
    }
    if (!client) {
        LOG_DEBUG(NpHandler, "ReportTrophyUnlock: null client for user_id={}; skipping", user_id);
        return;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "ReportTrophyUnlock: not authenticated; trophy {} not mirrored",
                    trophy_id);
        return;
    }
    if (!client->IsTrophiesEnabled()) {
        LOG_WARNING(NpHandler,
                    "ReportTrophyUnlock: server has not advertised trophy support; trophy {} "
                    "not mirrored",
                    trophy_id);
        return;
    }

    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_WARNING(NpHandler,
                    "ReportTrophyUnlock: no valid NP Communication ID for service_label={}; "
                    "skipping trophy {}",
                    service_label, trophy_id);
        return;
    }

    const u64 pkt_id = client->UnlockTrophy(com_id, trophy_id, timestamp);
    LOG_INFO(NpHandler, "ReportTrophyUnlock: user_id={} trophy={} com_id='{}' pkt_id={}", user_id,
             trophy_id, com_id, pkt_id);
}

void NpHandler::SyncTrophies(
    s32 user_id, s32 service_label, const std::vector<std::pair<s32, u64>>& local_trophies,
    std::function<void(const std::vector<std::pair<s32, u64>>&)> on_merged) {
    std::shared_ptr<ShadNet::ShadNetClient> client;
    {
        std::lock_guard lock(m_mutex_clients);
        auto it = m_clients.find(user_id);
        if (it == m_clients.end()) {
            LOG_DEBUG(NpHandler, "SyncTrophies: no shadNet session for user_id={}; skipping",
                      user_id);
            return;
        }
        client = it->second;
    }
    if (!client) {
        LOG_DEBUG(NpHandler, "SyncTrophies: null client for user_id={}; skipping", user_id);
        return;
    }
    if (!client->IsAuthenticated()) {
        LOG_WARNING(NpHandler, "SyncTrophies: not authenticated; skipping sync");
        return;
    }
    if (!client->IsTrophiesEnabled()) {
        LOG_WARNING(NpHandler, "SyncTrophies: server has not advertised trophy support; "
                               "skipping sync");
        return;
    }

    const std::string com_id = GetNpCommId(service_label);
    if (!IsValidNpCommId(com_id)) {
        LOG_WARNING(NpHandler,
                    "SyncTrophies: no valid NP Communication ID for service_label={}; skipping",
                    service_label);
        return;
    }

    const u64 pkt_id = client->SyncTrophies(com_id, local_trophies);
    if (on_merged) {
        std::lock_guard lock(m_mutex_pending_trophy);
        m_pending_trophy.emplace(pkt_id, std::move(on_merged));
    }
    LOG_INFO(NpHandler, "SyncTrophies: user_id={} uploading {} com_id='{}' pkt_id={}", user_id,
             local_trophies.size(), com_id, pkt_id);
}

void NpHandler::OnTrophyReply(s32 user_id, ShadNet::CommandType cmd, u64 pkt_id,
                              ShadNet::ErrorType error, const std::vector<u8>& body) {
    if (cmd == ShadNet::CommandType::UnlockTrophy) {
        if (error != ShadNet::ErrorType::NoError) {
            LOG_WARNING(NpHandler, "UnlockTrophy rejected by server: error={}",
                        static_cast<int>(error));
        }
        return;
    }
    if (cmd != ShadNet::CommandType::SyncTrophies)
        return;

    std::function<void(const std::vector<std::pair<s32, u64>>&)> on_merged;
    {
        std::lock_guard lock(m_mutex_pending_trophy);
        auto it = m_pending_trophy.find(pkt_id);
        if (it == m_pending_trophy.end())
            return;
        on_merged = std::move(it->second);
        m_pending_trophy.erase(it);
    }

    if (error != ShadNet::ErrorType::NoError) {
        LOG_WARNING(NpHandler, "SyncTrophies failed: error={}", static_cast<int>(error));
        return; // leave local trophies exactly as they were
    }

    // Reply body is u32 LE blob size followed by the proto.
    if (body.size() < 4) {
        LOG_WARNING(NpHandler, "SyncTrophies: truncated reply");
        return;
    }
    const u32 blob_sz = static_cast<u32>(body[0]) | (static_cast<u32>(body[1]) << 8) |
                        (static_cast<u32>(body[2]) << 16) | (static_cast<u32>(body[3]) << 24);
    if (body.size() < 4 + blob_sz) {
        LOG_WARNING(NpHandler, "SyncTrophies: reply shorter than declared blob");
        return;
    }

    shadnet::SyncTrophiesReply pb;
    if (!pb.ParseFromArray(body.data() + 4, static_cast<int>(blob_sz))) {
        LOG_WARNING(NpHandler, "SyncTrophies: could not parse reply");
        return;
    }

    std::vector<std::pair<s32, u64>> merged;
    merged.reserve(pb.trophies_size());
    for (const auto& t : pb.trophies())
        merged.emplace_back(t.trophyid(), t.timestamp());

    LOG_INFO(NpHandler, "SyncTrophies: server holds {} trophies for user_id={}", merged.size(),
             user_id);
    on_merged(merged);
}

} // namespace Libraries::Np
