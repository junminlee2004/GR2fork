// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>
#include <thread>
#include <vector>
#include "common/config.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/network/gr2_launch_gate.h"
#include "core/libraries/network/http.h"
#include "http_error.h"

namespace Libraries::Http {

static bool g_isHttpInitialized = true; // TODO temp always inited
static bool g_isConnectedToNetwork = false;

static std::map<s32, RequestTemplate> g_templates;
static std::map<s32, RequestObj> g_requests;
static std::mutex g_templates_map_mutex;
static std::mutex g_requests_map_mutex;

std::string host_override = "localhost";

// std::string construction from a null char* throws std::logic_error and aborts the emulator.
// Guest HLE entry points can legitimately pass null string pointers, so every LOG_* arg or
// std::string built from a guest char* goes through this guard (sceHttpUriParse null-URL case).
static inline std::string SafeCStr(const char* p) {
    return p ? std::string(p) : std::string("(null)");
}

std::string ReplaceHost(std::string url, const std::string& new_host, bool force_http = true) {

    std::string separator = "://";
    u64 protocol_pos = url.find(separator);

    u64 host_start = 0;

    if (protocol_pos != std::string::npos) {
        host_start = protocol_pos + separator.length();
    }

    // Consume the ENTIRE authority -- host AND any original :port -- so the override's port
    // wins instead of the source URL's. Stop only at the path/query/fragment, NOT at the
    // original ':' (otherwise the source ":443" survives and the request misses the server).
    u64 host_end = url.find_first_of("/?#", host_start);
    if (host_end == std::string::npos) {
        host_end = url.length();
    }

    // Force the local GR2 server's port: honor a port already present in new_host, otherwise
    // append the configured override port (httpHostOverridePort, default 8443). This is what
    // makes "ssps4...:443" become "localhost:8443" rather than "localhost:443".
    std::string authority = new_host;
    if (authority.find(':') == std::string::npos) {
        authority += ":" + std::to_string(Config::GetHttpHostOverridePort());
    }

    url.replace(host_start, host_end - host_start, authority);

    if (force_http) {
        if (protocol_pos != std::string::npos) {

            url.replace(0, protocol_pos, "http");
        } else {

            url.insert(0, "http://");
        }
    }

    LOG_INFO(Lib_Http, "Replaced URL host, new URL: {}", url);

    return url;
}

void NormalizeAndAppendPath(char* dest, char* src) {
    char* lastSlash;
    u64 length;

    lastSlash = strrchr(dest, '/');
    if (lastSlash == NULL) {
        length = strlen(dest);
        dest[length] = '/';
        dest[length + 1] = '\0';
    } else {
        lastSlash[1] = '\0';
    }
    if (*src == '/') {
        dest[0] = '\0';
    }
    length = strnlen(dest, 0x3fff);
    strncat(dest, src, 0x3fff - length);
    return;
}

int HttpRequestInternal_Acquire(HttpRequestInternal** outRequest, u32 requestId) {
    return 0; // TODO dummy
}
int HttpRequestInternal_Release(HttpRequestInternal* request) {
    return 0; // TODO dummy
}

int PS4_SYSV_ABI sceHttpAbortRequest() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAbortRequestForce() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAbortWaitRequest() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAddCookie() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAddQuery() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAddRequestHeader(int id, const char* name, const char* value, s32 mode) {

    LOG_INFO(Lib_Http, "called request id = '{}', name = '{}', value = '{}', mode = '{}'", id,
             SafeCStr(name), SafeCStr(value), mode);

    if (!g_isHttpInitialized) {

        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    std::lock_guard<std::mutex> lock(g_templates_map_mutex);

    auto it = g_templates.find(id);
    if (it == g_templates.end()) {

        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    it->second.AddHeader(name, value);

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAddRequestHeaderRaw() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAuthCacheExport() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAuthCacheFlush() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpAuthCacheImport() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCacheRedirectedConnectionEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCookieExport() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCookieFlush() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCookieImport() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCreateConnection() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCreateConnectionWithURL(int tmplId, const char* url, bool enableKeepalive) {
    LOG_ERROR(Lib_Http, "(STUBBED) called tmpid = {} url = {} enableKeepalive = {}", tmplId,
              SafeCStr(url), enableKeepalive ? 1 : 0);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpCreateEpoll() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

// GR2FORK: the "2" request-create variants pass the HTTP method as a string ("GET", "POST", ...).
// Map it to the OrbisHttpRequestMethod the RequestObj path expects; an unknown verb maps to
// INVALID so the method-range guard rejects it.
static s32 HttpMethodFromString(const char* method) {
    if (method == nullptr) {
        return ORBIS_INTERNAL_HTTP_REQUEST_METHOD_INVALID;
    }
    const std::string_view m{method};
    if (m == "GET") {
        return ORBIS_INTERNAL_HTTP_REQUEST_METHOD_GET;
    }
    if (m == "POST") {
        return ORBIS_INTERNAL_HTTP_REQUEST_METHOD_POST;
    }
    if (m == "HEAD") {
        return ORBIS_INTERNAL_HTTP_REQUEST_METHOD_HEAD;
    }
    if (m == "OPTIONS") {
        return ORBIS_INTERNAL_HTTP_REQUEST_METHOD_OPTIONS;
    }
    if (m == "PUT") {
        return ORBIS_INTERNAL_HTTP_REQUEST_METHOD_PUT;
    }
    if (m == "DELETE") {
        return ORBIS_INTERNAL_HTTP_REQUEST_METHOD_DELETE;
    }
    if (m == "TRACE") {
        return ORBIS_INTERNAL_HTTP_REQUEST_METHOD_TRACE;
    }
    if (m == "CONNECT") {
        return ORBIS_INTERNAL_HTTP_REQUEST_METHOD_CONNECT;
    }
    return ORBIS_INTERNAL_HTTP_REQUEST_METHOD_INVALID;
}

// GR2FORK: the fork stores no connection objects, so a connection-based request has no known
// host. ReplaceHost forces every request to the restoration server anyway, so binding the path
// to the override host reproduces the real route; only the path identifies the endpoint.
static std::string BuildOverrideUrlFromPath(const char* path) {
    std::string url = "http://" + host_override;
    if (path != nullptr && path[0] != '\0') {
        if (path[0] != '/') {
            url.push_back('/');
        }
        url.append(path);
    }
    return url;
}

// GR2FORK: shared body for the request-create entry points. Builds a host-overridden RequestObj
// bound to template `tmpl_id` (connection ids alias template ids -- the fork stores no connections)
// and returns its request id. Only sceHttpCreateRequestWithURL runs the one-shot avatar upload.
static s32 CreateRequestWithURLInternal(s32 tmpl_id, s32 method, const char* url,
                                        u64 content_length) {
    if (!g_isHttpInitialized) {
        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    if (method >= ORBIS_INTERNAL_HTTP_REQUEST_METHOD_INVALID) {
        return ORBIS_HTTP_ERROR_UNKNOWN_METHOD;
    }

    if (url == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    static s32 request_id_counter = 0;
    s32 request_id = request_id_counter++;

    std::string url_str = ReplaceHost(std::string(url), host_override);

    std::lock_guard<std::mutex> lock_t(g_templates_map_mutex);

    auto it = g_templates.find(tmpl_id);
    if (it == g_templates.end()) {
        return ORBIS_HTTP_ERROR_INVALID_ID;
    }

    auto new_request = RequestObj(request_id, &it->second, method, url_str, content_length);

    std::lock_guard<std::mutex> lock_r(g_requests_map_mutex);
    g_requests.emplace(request_id, std::move(new_request));

    return request_id;
}

int PS4_SYSV_ABI sceHttpCreateRequest(s32 conn_id, s32 method, const char* path,
                                      u64 content_length) {
    LOG_INFO(Lib_Http, "called conn id = '{}' method = '{}' path = '{}', content length = '{}'",
             conn_id, method, SafeCStr(path), content_length);

    if (!g_isHttpInitialized) {
        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    if (path == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    std::string url = BuildOverrideUrlFromPath(path);

    return CreateRequestWithURLInternal(conn_id, method, url.c_str(), content_length);
}

int PS4_SYSV_ABI sceHttpCreateRequest2(s32 conn_id, const char* method, const char* path,
                                       u64 content_length) {
    LOG_INFO(Lib_Http, "called conn id = '{}' method = '{}' path = '{}', content length = '{}'",
             conn_id, SafeCStr(method), SafeCStr(path), content_length);

    if (!g_isHttpInitialized) {
        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    if (method == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    if (path == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    const s32 int_method = HttpMethodFromString(method);

    std::string url = BuildOverrideUrlFromPath(path);

    return CreateRequestWithURLInternal(conn_id, int_method, url.c_str(), content_length);
}

// GR2FORK: fork-side profile avatar upload. The restoration server stores each player's avatar
// but, once remote, cannot read the client's filesystem, so the fork POSTs the local avatar file
// (user/home/1000/gr2_avatar.png) to /uploadavatar. The server keys the image by the
// X-GR2-Player header, then center-crops and resizes it, so the raw bytes are sent as-is. Runs on
// a detached thread and only for GR2 titles.
static void Gr2UploadAvatarOnce() {
    // GR2-only: the avatar file and endpoint belong to the GR2 restoration flow. Other titles
    // (GRR: CUSA01130) have no gr2_avatar.png and must not post. Mirrors np_auth's serial gate.
    static constexpr const char* kGr2Serials[] = {
        "CUSA03694", "CUSA04943", "CUSA04934", "CUSA00547",
        "PCJS50010", "PCAS00079", "CUSA04935",
    };
    const std::string_view serial = Common::ElfInfo::Instance().GameSerial();
    bool is_gr2 = false;
    for (const char* s : kGr2Serials) {
        if (serial == s) {
            is_gr2 = true;
            break;
        }
    }
    if (!is_gr2) {
        return;
    }

    const std::filesystem::path avatar_path =
        Common::FS::GetUserPath(Common::FS::PathType::HomeDir) / "1000" / "gr2_avatar.png";
    std::error_code ec;
    if (!std::filesystem::exists(avatar_path, ec)) {
        return;
    }

    const std::string host = Config::GetHttpHostOverride();
    const int port = Config::GetHttpHostOverridePort();
    GR2Fork::Auth::EnsureAuthed();
    std::string player = GR2Fork::Auth::VerifiedNpid();
    if (player.empty()) {
        player = UserManagement.GetDefaultUser().user_name;
    }
    const std::string auth_token = GR2Fork::Auth::BearerToken();

    // Detached so the game thread never blocks on file I/O or the network round-trip. The thread
    // owns copies of every input and touches no shared state after launch.
    std::thread([avatar_path, host, port, player, auth_token] {
        std::ifstream f(avatar_path, std::ios::binary);
        if (!f) {
            return;
        }
        std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (body.empty()) {
            return;
        }

        // Sanitize the display name into a valid header value, matching the X-GR2-Player rule the
        // leaderboard path uses (drop control bytes and DEL). The server rejects an empty name.
        std::string safe;
        safe.reserve(player.size());
        for (unsigned char c : player) {
            if (c >= 0x20 && c != 0x7f) {
                safe.push_back(static_cast<char>(c));
            }
        }
        if (safe.empty()) {
            return;
        }

        httplib::Client cli(host, port);
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(5, 0);
        cli.set_write_timeout(5, 0);

        httplib::Headers headers;
        headers.emplace("X-GR2-Player", safe);
        if (!auth_token.empty()) {
            headers.emplace("X-GR2-Auth", auth_token);
        }
        headers.emplace("X-GR2-Version", std::to_string(GR2Fork::OnlineVersion));
        cli.Post("/uploadavatar", headers, body.data(), body.size(), "application/octet-stream");
    }).detach();
}

// GR2FORK: server-authoritative Dusty Token enforcement. The restoration server's ledger is the
// sole source of truth -- every Dusty Token is earned through an online reward it grants -- so a
// tampered ggdL save that inflates the count would let a player unlock cosmetics never earned. The
// in-game count is monotonic (the sync mail can only RAISE it), so a hacked-high value can only be
// corrected by a direct save write. On boot the fork fetches the authoritative total and rewrites
// the DustyToken field of the ggdL save slots to match -- up or down. This runs before the game
// loads the save (first game HTTP request precedes the Continue/load), so the corrected value is
// what the game reads; each boot re-enforces, so a hacked save cannot persist.
static void Gr2EnforceDustyTokenOnce() {
    static constexpr const char* kGr2Serials[] = {
        "CUSA03694", "CUSA04943", "CUSA04934", "CUSA00547",
        "PCJS50010", "PCAS00079", "CUSA04935",
    };
    const std::string_view serial = Common::ElfInfo::Instance().GameSerial();
    bool is_gr2 = false;
    for (const char* s : kGr2Serials) {
        if (serial == s) {
            is_gr2 = true;
            break;
        }
    }
    if (!is_gr2) {
        return;
    }

    const std::string host = Config::GetHttpHostOverride();
    if (host.empty()) {
        return;  // no restoration server configured -- nothing to enforce against
    }
    const int port = Config::GetHttpHostOverridePort();
    GR2Fork::Auth::EnsureAuthed();
    std::string player = GR2Fork::Auth::VerifiedNpid();
    if (player.empty()) {
        player = UserManagement.GetDefaultUser().user_name;
    }
    const std::string auth_token = GR2Fork::Auth::BearerToken();
    const std::filesystem::path save_root =
        Common::FS::GetUserPath(Common::FS::PathType::HomeDir) / "1000" / "savedata";

    std::thread([host, port, player, save_root, auth_token] {
        std::string safe;
        safe.reserve(player.size());
        for (unsigned char c : player) {
            if (c >= 0x20 && c != 0x7f) {
                safe.push_back(static_cast<char>(c));
            }
        }
        if (safe.empty()) {
            return;
        }

        httplib::Client cli(host, port);
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(5, 0);
        cli.set_write_timeout(5, 0);
        httplib::Headers headers;
        headers.emplace("X-GR2-Player", safe);
        if (!auth_token.empty()) {
            headers.emplace("X-GR2-Auth", auth_token);
        }
        headers.emplace("X-GR2-Version", std::to_string(GR2Fork::OnlineVersion));
        auto res = cli.Get("/gr2token", headers);
        if (!res || res->status != 200) {
            return;  // server unreachable -- leave the save untouched rather than guess
        }
        long long authoritative = -1;
        try {
            authoritative = std::stoll(res->body);
        } catch (...) {
            return;
        }
        if (authoritative < 0) {
            return;
        }
        const float want = static_cast<float>(authoritative);

        // ggdL save layout: 'ggdL' magic @0, entry count u32 @12, entries @16 each 16 bytes with the
        // value f32 at +8 and the FNV-1a key u32 at +12. GR2's slots are data0000.bin / data0001.bin.
        static constexpr std::uint32_t kDustyTokenFnv = 0x8492dbd1u;
        std::error_code ec;
        if (!std::filesystem::exists(save_root, ec)) {
            return;
        }
        for (auto& de : std::filesystem::recursive_directory_iterator(save_root, ec)) {
            if (ec) {
                break;
            }
            if (!de.is_regular_file(ec)) {
                continue;
            }
            const std::string fn = de.path().filename().string();
            if (fn != "data0000.bin" && fn != "data0001.bin") {
                continue;
            }
            std::vector<char> data;
            {
                std::ifstream in(de.path(), std::ios::binary);
                if (!in) {
                    continue;
                }
                data.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            }
            if (data.size() < 16 || std::memcmp(data.data(), "ggdL", 4) != 0) {
                continue;
            }
            std::uint32_t count = 0;
            std::memcpy(&count, data.data() + 12, 4);
            for (std::uint32_t i = 0; i < count; ++i) {
                const std::size_t off = 16 + static_cast<std::size_t>(i) * 16;
                if (off + 16 > data.size()) {
                    break;
                }
                std::uint32_t fnv = 0;
                std::memcpy(&fnv, data.data() + off + 12, 4);
                if (fnv != kDustyTokenFnv) {
                    continue;
                }
                float cur = 0.0f;
                std::memcpy(&cur, data.data() + off + 8, 4);
                if (cur != want) {
                    std::memcpy(data.data() + off + 8, &want, 4);
                    std::ofstream out(de.path(), std::ios::binary | std::ios::trunc);
                    if (out) {
                        out.write(data.data(), static_cast<std::streamsize>(data.size()));
                        LOG_INFO(Lib_Http,
                                 "[GR2FORK] DustyToken enforced in {}: {} -> {} (server authoritative)",
                                 fn, cur, want);
                    }
                }
                break;
            }
        }
    }).detach();
}

int PS4_SYSV_ABI sceHttpCreateRequestWithURL(s32 tmpl_id, s32 method, const char* url,
                                             u64 content_length) {
    LOG_INFO(Lib_Http, "called template id = '{}' method = '{}' url = '{}', content length = '{}'",
             tmpl_id, method, url, content_length);

    if (!g_isHttpInitialized) {

        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    if (method >= ORBIS_INTERNAL_HTTP_REQUEST_METHOD_INVALID) {

        return ORBIS_HTTP_ERROR_UNKNOWN_METHOD;
    }

    if (url == nullptr) {

        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    // GR2FORK: the first game HTTP request is a clean one-shot point where both the configured
    // host and the local player's name are known. Push the profile avatar to the restoration
    // server exactly once per boot; the guard latches before the attempt so it never repeats.
    static std::atomic<bool> avatar_upload_started{false};
    if (!avatar_upload_started.exchange(true)) {
        Gr2UploadAvatarOnce();
    }

    // GR2FORK: enforce the server-authoritative Dusty Token total onto the local ggdL save once per
    // boot, at this same pre-load point, so a tampered count is corrected before the game reads it.
    static std::atomic<bool> dusty_enforce_started{false};
    if (!dusty_enforce_started.exchange(true)) {
        Gr2EnforceDustyTokenOnce();
    }

    return CreateRequestWithURLInternal(tmpl_id, method, url, content_length);
}

int PS4_SYSV_ABI sceHttpCreateRequestWithURL2(s32 conn_id, const char* method, const char* url,
                                              u64 content_length) {
    LOG_INFO(Lib_Http, "called conn id = '{}' method = '{}' url = '{}', content length = '{}'",
             conn_id, SafeCStr(method), SafeCStr(url), content_length);

    if (!g_isHttpInitialized) {
        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    if (method == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    if (url == nullptr) {
        return ORBIS_HTTP_ERROR_INVALID_URL;
    }

    const s32 int_method = HttpMethodFromString(method);

    return CreateRequestWithURLInternal(conn_id, int_method, url, content_length);
}

int PS4_SYSV_ABI sceHttpCreateTemplate(s32 conn_id, const char* user_agent, s32 http_v, s32 flags) {
    LOG_INFO(Lib_Http, "called, conn id: '{}', user agent: '{}', http version: '{}', flags: '{}'",
             conn_id, user_agent, http_v, flags);

    if (!g_isHttpInitialized) {

        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    if (user_agent == nullptr) {

        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    static s32 template_counter = 0;

    s32 template_id = template_counter++;

    std::lock_guard<std::mutex> lock(g_templates_map_mutex);

    auto new_template = RequestTemplate(template_id, SafeCStr(user_agent));

    g_templates.emplace(template_id, new_template);

    return template_id;
}

int PS4_SYSV_ABI sceHttpDbgEnableProfile() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgGetConnectionStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgGetRequestStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgSetPrintf() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgShowConnectionStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgShowMemoryPoolStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgShowRequestStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDbgShowStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDeleteConnection() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDeleteRequest(s32 req_id) {
    LOG_INFO(Lib_Http, "called, request id: '{}'", req_id);

    if (!g_isHttpInitialized) {

        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    std::lock_guard<std::mutex> lock(g_requests_map_mutex);
    auto it = g_requests.find(req_id);
    if (it == g_requests.end()) {

        return ORBIS_HTTP_ERROR_INVALID_ID;
    }

    g_requests.erase(it);

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDeleteTemplate() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpDestroyEpoll() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetAcceptEncodingGZIPEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetAllResponseHeaders(int reqId, char** header, u64* headerSize) {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_FAIL;
}

int PS4_SYSV_ABI sceHttpGetAuthEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetAutoRedirect() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetConnectionStat() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetCookie() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetCookieEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetCookieStats() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetEpoll() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetEpollId() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetLastErrno() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetMemoryPoolStats() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetNonblock() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetRegisteredCtxIds() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetResponseContentLength(u32 req_id, s32* result, u64* content_length) {

    LOG_INFO(Lib_Http, "called request id: '{}'", req_id);

    if (!g_isHttpInitialized) {

        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    // ABI: sceHttpGetResponseContentLength(int reqId, int* result, u64* contentLength). Arg #2 is
    // the "content-length available" flag (0 = ORBIS_HTTP_CONTENTLEN_EXIST), arg #3 the length;
    // swapped outputs make the game conclude the body is empty and skip sceHttpReadData.
    if (result == nullptr || content_length == nullptr) {

        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_requests_map_mutex);
    auto it = g_requests.find(req_id);
    if (it == g_requests.end()) {

        return ORBIS_HTTP_ERROR_INVALID_ID;
    }

    if (!it->second.IsSent() && !it->second.req_template->is_async) {

        return ORBIS_HTTP_ERROR_BEFORE_SEND;
    }

    if (!it->second.IsCompleted()) {

        return ORBIS_HTTP_ERROR_EAGAIN;
    }

    *result = 0; // ORBIS_HTTP_CONTENTLEN_EXIST -- a concrete Content-Length is available
    *content_length = it->second.GetContentLength();

    LOG_INFO(Lib_Http, "result={} content_length={}", *result, *content_length);

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpGetStatusCode(s32 req_id, s32* status_code) {
    LOG_INFO(Lib_Http, "called request id = {}", req_id);
#if 0
    if (!g_isHttpInitialized)
        return ORBIS_HTTP_ERROR_BEFORE_INIT;

    if (statusCode == nullptr)
        return ORBIS_HTTP_ERROR_INVALID_VALUE;

    int ret = 0;
    // Lookup HttpRequestInternal by reqId
    HttpRequestInternal* request = nullptr;
    ret = HttpRequestInternal_Acquire(&request, reqId);
    if (ret < 0)
        return ret;
    request->m_mutex.lock();
    if (request->state > 0x11) {
        if (request->state == 0x16) {
            ret = request->errorCode;
        } else {
            *statusCode = request->httpStatusCode;
            ret = 0;
        }
    } else {
        ret = ORBIS_HTTP_ERROR_BEFORE_SEND;
    }
    request->m_mutex.unlock();
    HttpRequestInternal_Release(request);

    return ret;
#else

    if (!g_isHttpInitialized) {

        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    std::lock_guard<std::mutex> lock(g_requests_map_mutex);
    auto it = g_requests.find(req_id);
    if (it == g_requests.end()) {

        return ORBIS_HTTP_ERROR_INVALID_ID;
    }

    if (!it->second.IsSent() && !it->second.req_template->is_async) {

        return ORBIS_HTTP_ERROR_BEFORE_SEND;
    }

    if (!it->second.IsCompleted()) {

        return ORBIS_HTTP_ERROR_EAGAIN;
    }

    *status_code = it->second.GetStatusCode();

    return ORBIS_OK;
#endif
}

int PS4_SYSV_ABI sceHttpInit(int libnetMemId, int libsslCtxId, u64 poolSize) {
    LOG_ERROR(Lib_Http, "(DUMMY) called libnetMemId = {} libsslCtxId = {} poolSize = {}",
              libnetMemId, libsslCtxId, poolSize);

    // No ORBIS_HTTP_ERROR_ALREADY_INITED check since it could be
    // initialized multiple times

    // return a value >1
    static int id = 0;
    return ++id;
}

int PS4_SYSV_ABI sceHttpParseResponseHeader(const char* header, u64 headerLen, const char* fieldStr,
                                            const char** fieldValue, u64* valueLen) {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpParseStatusLine(const char* statusLine, u64 lineLen, int32_t* httpMajorVer,
                                        int32_t* httpMinorVer, int32_t* responseCode,
                                        const char** reasonPhrase, u64* phraseLen) {
    if (!statusLine) {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    if (!httpMajorVer || !httpMinorVer || !responseCode || !reasonPhrase || !phraseLen) {
        LOG_ERROR(Lib_Http, "Invalid value");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_VALUE;
    }
    *httpMajorVer = 0;
    *httpMinorVer = 0;
    if (lineLen < 8) {
        LOG_ERROR(Lib_Http, "Linelen is smaller than 8");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    if (strncmp(statusLine, "HTTP/", 5) != 0) {
        LOG_ERROR(Lib_Http, "statusLine doesn't start with HTTP/");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }

    u64 index = 5;

    if (!isdigit(statusLine[index])) {
        LOG_ERROR(Lib_Http, "Invalid response");

        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }

    while (isdigit(statusLine[index])) {
        *httpMajorVer = *httpMajorVer * 10 + (statusLine[index] - '0');
        index++;
    }

    if (statusLine[index] != '.') {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    index++;

    if (!isdigit(statusLine[index])) {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }

    while (isdigit(statusLine[index])) {
        *httpMinorVer = *httpMinorVer * 10 + (statusLine[index] - '0');
        index++;
    }

    if (statusLine[index] != ' ') {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    index++;

    // Validate and parse the 3-digit HTTP response code
    if (lineLen - index < 3 || !isdigit(statusLine[index]) || !isdigit(statusLine[index + 1]) ||
        !isdigit(statusLine[index + 2])) {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }

    *responseCode = (statusLine[index] - '0') * 100 + (statusLine[index + 1] - '0') * 10 +
                    (statusLine[index + 2] - '0');
    index += 3;

    if (statusLine[index] != ' ') {
        LOG_ERROR(Lib_Http, "Invalid response");
        return ORBIS_HTTP_ERROR_PARSE_HTTP_INVALID_RESPONSE;
    }
    index++;

    // Set the reason phrase start position
    *reasonPhrase = &statusLine[index];
    u64 phraseStart = index;

    while (index < lineLen && statusLine[index] != '\n') {
        index++;
    }

    // Determine the length of the reason phrase, excluding trailing \r if present
    if (index == phraseStart) {
        *phraseLen = 0;
    } else {
        *phraseLen =
            (statusLine[index - 1] == '\r') ? (index - phraseStart - 1) : (index - phraseStart);
    }

    // Return the number of bytes processed
    return index + 1;
}

int PS4_SYSV_ABI sceHttpReadData(u32 req_id, char* dest, u32 size) {

    if (!g_isHttpInitialized) {

        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    if (dest == nullptr || size == 0) {

        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    std::lock_guard<std::mutex> lock(g_requests_map_mutex);
    auto it = g_requests.find(req_id);

    if (it == g_requests.end()) {

        return ORBIS_HTTP_ERROR_INVALID_ID;
    }

    if (!it->second.IsSent() && !it->second.req_template->is_async) {

        return ORBIS_HTTP_ERROR_BEFORE_SEND;
    }

    if (!it->second.IsCompleted()) {

        return ORBIS_HTTP_ERROR_EAGAIN;
    }

    auto read_len = it->second.ReadData(dest, size);

    return read_len;
}

int PS4_SYSV_ABI sceHttpRedirectCacheFlush() {
    LOG_ERROR(Lib_Http, "(STUBBED) called, returning zero to {}", __builtin_return_address(0));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpRemoveRequestHeader() {
    LOG_ERROR(Lib_Http, "(STUBBED) called, returning zero to {}", __builtin_return_address(0));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpRequestGetAllHeaders() {
    LOG_ERROR(Lib_Http, "(STUBBED) called, returning zero to {}", __builtin_return_address(0));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsDisableOption() {
    LOG_ERROR(Lib_Http, "(STUBBED) called, returning zero to {}", __builtin_return_address(0));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsDisableOptionPrivate() {
    LOG_ERROR(Lib_Http, "(STUBBED) called, returning zero to {}", __builtin_return_address(0));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsEnableOption(u32 options) {
    LOG_ERROR(Lib_Http, "(STUBBED) called, returning zero to {}", __builtin_return_address(0));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsEnableOptionPrivate() {
    LOG_ERROR(Lib_Http, "(STUBBED) called, returning zero to {}", __builtin_return_address(0));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSendRequest(int req_id, const void* post_data, u64 size) {

    LOG_INFO(Lib_Http, "called, request id = '{}', size = '{}'", req_id, size);

    if (!g_isHttpInitialized) {

        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    std::lock_guard<std::mutex> lock(g_requests_map_mutex);
    auto it = g_requests.find(req_id);

    if (it == g_requests.end()) {

        return ORBIS_HTTP_ERROR_INVALID_ID;
    }

    // Hard offline gate: with the emulator set offline (isConnectedToNetwork), no guest HTTP
    // leaves the box - return the canonical error a real PS4 yields offline so the game's online
    // layer sees a failed request instead of hanging. Read live so mid-session toggles apply.
    g_isConnectedToNetwork = Config::getIsConnectedToNetwork();
    if (!g_isConnectedToNetwork) {
        LOG_INFO(Lib_Http,
                 "offline (isConnectedToNetwork=false) -> not sending; "
                 "ORBIS_HTTP_ERROR_NETWORK for request id '{}'",
                 req_id);
        return ORBIS_HTTP_ERROR_NETWORK;
    }

    // SendRequest() blocks for sync requests and returns immediately for async ones, so no
    // pre-send gate belongs here. The body is dispatched only once the declared content_length
    // arrives - the ARC login streams its multipart body over three calls (78 + 152 + 38 = 268).
    it->second.AccumulateAndSend(post_data, size);

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetAcceptEncodingGZIPEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetAuthEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetAuthInfoCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetAutoRedirect() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetChunkedTransferEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetConnectTimeOut() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieMaxNum() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieMaxNumPerDomain() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieMaxSize() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieRecvCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieSendCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetCookieTotalMaxSize() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetDefaultAcceptEncodingGZIPEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetDelayBuildRequestEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetEpoll() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetEpollId() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetHttp09Enabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetInflateGZIPEnabled() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetNonblock(s32 tmpl_id, bool enable) {
    LOG_ERROR(Lib_Http, "(STUBBED) called");

    if (!g_isHttpInitialized) {

        return ORBIS_HTTP_ERROR_BEFORE_INIT;
    }

    std::lock_guard<std::mutex> lock_t(g_templates_map_mutex);

    auto it = g_templates.find(tmpl_id);
    if (it == g_templates.end()) {

        return ORBIS_HTTP_ERROR_INVALID_ID;
    }

    it->second.is_async = enable;

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetPolicyOption() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetPriorityOption() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetProxy() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRecvBlockSize() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRecvTimeOut() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRedirectCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRequestContentLength() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetRequestStatusCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetResolveRetry() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetResolveTimeOut() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetResponseHeaderMaxSize() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetSendTimeOut() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpSetSocketCreationCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsFreeCaList() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsGetCaList(int httpCtxId, OrbisHttpsCaList* list) {
    LOG_ERROR(Lib_Http, "(DUMMY) called, httpCtxId = {}", httpCtxId);
    list->certsNum = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsGetSslError() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsLoadCert() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsSetMinSslVersion() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsSetSslCallback() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsSetSslVersion() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpsUnloadCert() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpTerm() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpTryGetNonblock() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpTrySetNonblock() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUnsetEpoll() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriBuild(char* out, u64* require, u64 prepare,
                                 const OrbisHttpUriElement* srcElement, u32 option) {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriCopy() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriEscape(char* out, u64* require, u64 prepare, const char* in) {
    LOG_TRACE(Lib_Http, "called");

    if (!in) {
        LOG_ERROR(Lib_Http, "Invalid input string");
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    auto IsUnreserved = [](unsigned char c) -> bool {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
               c == '-' || c == '_' || c == '.' || c == '~';
    };

    u64 needed = 0;
    const char* src = in;
    while (*src) {
        unsigned char c = static_cast<unsigned char>(*src);
        if (IsUnreserved(c)) {
            needed++;
        } else {
            needed += 3; // %XX format
        }
        src++;
    }
    needed++; // null terminator

    if (require) {
        *require = needed;
    }

    if (!out) {
        return ORBIS_OK;
    }

    if (prepare < needed) {
        LOG_ERROR(Lib_Http, "Buffer too small: need {} but only {} available", needed, prepare);
        return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
    }

    static const char hex_chars[] = "0123456789ABCDEF";
    src = in;
    char* dst = out;
    while (*src) {
        unsigned char c = static_cast<unsigned char>(*src);
        if (IsUnreserved(c)) {
            *dst++ = *src;
        } else {
            *dst++ = '%';
            *dst++ = hex_chars[(c >> 4) & 0x0F];
            *dst++ = hex_chars[c & 0x0F];
        }
        src++;
    }
    *dst = '\0';

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriMerge(char* mergedUrl, char* url, char* relativeUri, u64* require,
                                 u64 prepare, u32 option) {
    u64 requiredLength;
    int returnValue;
    u64 baseUrlLength;
    u64 relativeUriLength;
    u64 totalLength;
    u64 combinedLength;
    int parseResult;
    u64 localSizeRelativeUri;
    u64 localSizeBaseUrl;
    OrbisHttpUriElement parsedUriElement;

    if (option != 0 || url == NULL || relativeUri == NULL) {
        LOG_ERROR(Lib_Http, "Invalid value");
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    returnValue = sceHttpUriParse(NULL, url, NULL, &localSizeBaseUrl, 0);
    if (returnValue < 0) {
        LOG_ERROR(Lib_Http, "returning {:#x}", returnValue);
        return returnValue;
    }

    returnValue = sceHttpUriParse(NULL, relativeUri, NULL, &localSizeRelativeUri, 0);
    if (returnValue < 0) {
        LOG_ERROR(Lib_Http, "returning {:#x}", returnValue);
        return returnValue;
    }

    baseUrlLength = strnlen(url, 0x3fff);
    relativeUriLength = strnlen(relativeUri, 0x3fff);
    requiredLength = localSizeBaseUrl + 2 + (relativeUriLength + baseUrlLength) * 2;

    if (require) {
        *require = requiredLength;
    }

    if (mergedUrl == NULL) {
        return ORBIS_OK;
    }

    if (prepare < requiredLength) {
        LOG_ERROR(Lib_Http, "Error Out of memory");
        return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
    }

    totalLength = strnlen(url, 0x3fff);
    baseUrlLength = strnlen(relativeUri, 0x3fff);
    combinedLength = totalLength + 1 + baseUrlLength;
    relativeUriLength = prepare - combinedLength;

    returnValue =
        sceHttpUriParse(&parsedUriElement, relativeUri, mergedUrl + totalLength + baseUrlLength + 1,
                        &localSizeRelativeUri, relativeUriLength);
    if (returnValue < 0) {
        LOG_ERROR(Lib_Http, "returning {:#x}", returnValue);
        return returnValue;
    }
    if (parsedUriElement.scheme == NULL) {
        strncpy(mergedUrl, relativeUri, requiredLength);
        if (require) {
            *require = strnlen(relativeUri, 0x3fff) + 1;
        }
        return ORBIS_OK;
    }

    returnValue =
        sceHttpUriParse(&parsedUriElement, url, mergedUrl + totalLength + baseUrlLength + 1,
                        &localSizeBaseUrl, relativeUriLength);
    if (returnValue < 0) {
        LOG_ERROR(Lib_Http, "returning {:#x}", returnValue);
        return returnValue;
    }

    combinedLength += localSizeBaseUrl;
    strncpy(mergedUrl + combinedLength, parsedUriElement.path, prepare - combinedLength);
    NormalizeAndAppendPath(mergedUrl + combinedLength, relativeUri);

    returnValue = sceHttpUriBuild(mergedUrl, 0, ~(baseUrlLength + totalLength) + prepare,
                                  &parsedUriElement, 0x3f);
    if (returnValue >= 0) {
        return ORBIS_OK;
    } else {
        LOG_ERROR(Lib_Http, "returning {:#x}", returnValue);
        return returnValue;
    }
}

int PS4_SYSV_ABI sceHttpUriParse(OrbisHttpUriElement* out, const char* srcUri, void* pool,
                                 u64* require, u64 prepare) {
    if (!srcUri) {
        LOG_ERROR(Lib_Http, "invalid url (null srcUri)");
        return ORBIS_HTTP_ERROR_INVALID_URL;
    }
    LOG_INFO(Lib_Http, "srcUri = {}", std::string(srcUri));
    if (!out && !pool && !require) {
        LOG_ERROR(Lib_Http, "invalid values");
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    if (out && pool) {
        memset(out, 0, sizeof(OrbisHttpUriElement));
        out->scheme = (char*)pool;
    }

    // Track the total required buffer size
    u64 requiredSize = 0;

    // Parse the scheme (e.g., "http:", "https:", "file:")
    u64 schemeLength = 0;
    while (srcUri[schemeLength] && srcUri[schemeLength] != ':') {
        if (!isalnum(srcUri[schemeLength])) {
            LOG_ERROR(Lib_Http, "invalid url");
            return ORBIS_HTTP_ERROR_INVALID_URL;
        }
        schemeLength++;
    }

    if (pool && prepare < schemeLength + 1) {
        LOG_ERROR(Lib_Http, "out of memory");
        return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
    }

    if (out && pool) {
        memcpy(out->scheme, srcUri, schemeLength);
        out->scheme[schemeLength] = '\0';
    }

    requiredSize += schemeLength + 1;

    // Move past the scheme and ':' character
    u64 offset = schemeLength + 1;

    // Check if "//" appears after the scheme
    if (strncmp(srcUri + offset, "//", 2) == 0) {
        // "//" is present
        if (out) {
            out->opaque = false;
        }
        offset += 2; // Move past "//"
    } else {
        // "//" is not present
        if (out) {
            out->opaque = true;
        }
    }

    // Handle "file" scheme
    if (strncmp(srcUri, "file", 4) == 0) {
        // File URIs typically start with "file://"
        if (out && !out->opaque) {
            // Skip additional slashes (e.g., "////")
            while (srcUri[offset] == '/') {
                offset++;
            }

            // Parse the path (everything after the slashes)
            char* pathStart = (char*)srcUri + offset;
            u64 pathLength = 0;
            while (pathStart[pathLength] && pathStart[pathLength] != '?' &&
                   pathStart[pathLength] != '#') {
                pathLength++;
            }

            // Ensure the path starts with '/'
            if (pathLength > 0 && pathStart[0] != '/') {
                // Prepend '/' to the path
                requiredSize += pathLength + 2; // Include '/' and null terminator

                if (pool && prepare < requiredSize) {
                    LOG_ERROR(Lib_Http, "out of memory");
                    return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
                }

                if (out && pool) {
                    out->path = (char*)pool + (requiredSize - pathLength - 2);
                    out->path[0] = '/'; // Add leading '/'
                    memcpy(out->path + 1, pathStart, pathLength);
                    out->path[pathLength + 1] = '\0';
                }
            } else {
                // Path already starts with '/'
                requiredSize += pathLength + 1;

                if (pool && prepare < requiredSize) {
                    LOG_ERROR(Lib_Http, "out of memory");
                    return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
                }

                if (out && pool) {
                    memcpy((char*)pool + (requiredSize - pathLength - 1), pathStart, pathLength);
                    out->path = (char*)pool + (requiredSize - pathLength - 1);
                    out->path[pathLength] = '\0';
                }
            }

            // Move past the path
            offset += pathLength;
        }
    }

    // Handle non-file schemes (e.g., "http", "https")
    else {
        // Parse the host and port
        char* hostStart = (char*)srcUri + offset;
        while (*hostStart == '/') {
            hostStart++;
        }

        u64 hostLength = 0;
        while (hostStart[hostLength] && hostStart[hostLength] != '/' &&
               hostStart[hostLength] != '?' && hostStart[hostLength] != ':') {
            hostLength++;
        }

        requiredSize += hostLength + 1;

        if (pool && prepare < requiredSize) {
            LOG_ERROR(Lib_Http, "out of memory");
            return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
        }

        if (out && pool) {
            memcpy((char*)pool + (requiredSize - hostLength - 1), hostStart, hostLength);
            out->hostname = (char*)pool + (requiredSize - hostLength - 1);
            out->hostname[hostLength] = '\0';
        }

        // Move past the host
        offset += hostLength;

        // Parse the port (if present)
        if (hostStart[hostLength] == ':') {
            char* portStart = hostStart + hostLength + 1;
            u64 portLength = 0;
            while (portStart[portLength] && isdigit(portStart[portLength])) {
                portLength++;
            }

            requiredSize += portLength + 1;

            if (pool && prepare < requiredSize) {
                LOG_ERROR(Lib_Http, "out of memory");
                return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
            }

            // Convert the port string to a uint16_t
            char portStr[6]; // Max length for a port number (65535)
            if (portLength > 5) {
                LOG_ERROR(Lib_Http, "invalid url");
                return ORBIS_HTTP_ERROR_INVALID_URL;
            }
            memcpy(portStr, portStart, portLength);
            portStr[portLength] = '\0';

            uint16_t port = (uint16_t)atoi(portStr);
            if (port == 0 && portStr[0] != '0') {
                LOG_ERROR(Lib_Http, "invalid url");
                return ORBIS_HTTP_ERROR_INVALID_URL;
            }

            // Set the port in the output structure
            if (out) {
                out->port = port;
            }

            // Move past the port
            offset += portLength + 1;
        }
    }

    // Parse the path (if present)
    if (srcUri[offset] == '/') {
        char* pathStart = (char*)srcUri + offset;
        u64 pathLength = 0;
        while (pathStart[pathLength] && pathStart[pathLength] != '?' &&
               pathStart[pathLength] != '#') {
            pathLength++;
        }

        requiredSize += pathLength + 1;

        if (pool && prepare < requiredSize) {
            LOG_ERROR(Lib_Http, "out of memory");
            return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
        }

        if (out && pool) {
            memcpy((char*)pool + (requiredSize - pathLength - 1), pathStart, pathLength);
            out->path = (char*)pool + (requiredSize - pathLength - 1);
            out->path[pathLength] = '\0';
        }

        // Move past the path
        offset += pathLength;
    }

    // Parse the query (if present)
    if (srcUri[offset] == '?') {
        char* queryStart = (char*)srcUri + offset + 1;
        u64 queryLength = 0;
        while (queryStart[queryLength] && queryStart[queryLength] != '#') {
            queryLength++;
        }

        requiredSize += queryLength + 1;

        if (pool && prepare < requiredSize) {
            LOG_ERROR(Lib_Http, "out of memory");
            return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
        }

        if (out && pool) {
            memcpy((char*)pool + (requiredSize - queryLength - 1), queryStart, queryLength);
            out->query = (char*)pool + (requiredSize - queryLength - 1);
            out->query[queryLength] = '\0';
        }

        // Move past the query
        offset += queryLength + 1;
    }

    // Parse the fragment (if present)
    if (srcUri[offset] == '#') {
        char* fragmentStart = (char*)srcUri + offset + 1;
        u64 fragmentLength = 0;
        while (fragmentStart[fragmentLength]) {
            fragmentLength++;
        }

        requiredSize += fragmentLength + 1;

        if (pool && prepare < requiredSize) {
            LOG_ERROR(Lib_Http, "out of memory");
            return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
        }

        if (out && pool) {
            memcpy((char*)pool + (requiredSize - fragmentLength - 1), fragmentStart,
                   fragmentLength);
            out->fragment = (char*)pool + (requiredSize - fragmentLength - 1);
            out->fragment[fragmentLength] = '\0';
        }
    }

    // Calculate the total required buffer size
    if (require) {
        *require = requiredSize; // Update with actual required size
    }

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriSweepPath(char* dst, const char* src, u64 srcSize) {
    LOG_TRACE(Lib_Http, "called");

    if (!dst || !src) {
        LOG_ERROR(Lib_Http, "Invalid parameters");
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    if (srcSize == 0) {
        dst[0] = '\0';
        return ORBIS_OK;
    }

    u64 len = 0;
    while (len < srcSize && src[len] != '\0') {
        len++;
    }

    for (u64 i = 0; i < len; i++) {
        dst[i] = src[i];
    }
    dst[len] = '\0';

    char* read = dst;
    char* write = dst;

    while (*read) {
        if (read[0] == '.' && read[1] == '.' && read[2] == '/') {
            read += 3;
            continue;
        }

        if (read[0] == '.' && read[1] == '/') {
            read += 2;
            continue;
        }

        if (read[0] == '/' && read[1] == '.' && read[2] == '/') {
            read += 2;
            continue;
        }

        if (read[0] == '/' && read[1] == '.' && read[2] == '\0') {
            if (write == dst) {
                *write++ = '/';
            }
            break;
        }

        bool is_dotdot_mid = (read[0] == '/' && read[1] == '.' && read[2] == '.' && read[3] == '/');
        bool is_dotdot_end =
            (read[0] == '/' && read[1] == '.' && read[2] == '.' && read[3] == '\0');

        if (is_dotdot_mid || is_dotdot_end) {
            if (write > dst) {
                if (*(write - 1) == '/') {
                    write--;
                }
                while (write > dst && *(write - 1) != '/') {
                    write--;
                }

                if (is_dotdot_mid && write > dst) {
                    write--;
                }
            }

            if (is_dotdot_mid) {
                read += 3;
            } else {
                break;
            }
            continue;
        }

        if ((read[0] == '.' && read[1] == '\0') ||
            (read[0] == '.' && read[1] == '.' && read[2] == '\0')) {
            break;
        }

        if (read[0] == '/') {
            *write++ = *read++;
        }
        while (*read && *read != '/') {
            *write++ = *read++;
        }
    }

    *write = '\0';
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpUriUnescape(char* out, u64* require, u64 prepare, const char* in) {
    LOG_TRACE(Lib_Http, "called");

    if (!in) {
        LOG_ERROR(Lib_Http, "Invalid input string");
        return ORBIS_HTTP_ERROR_INVALID_VALUE;
    }

    // Locale-independent hex digit check
    auto IsHex = [](char c) -> bool {
        return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
    };

    // Convert hex char to int value
    auto HexToInt = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        return 0;
    };

    // Check for valid percent-encoded sequence (%XX)
    auto IsValidPercentSequence = [&](const char* s) -> bool {
        return s[0] == '%' && s[1] != '\0' && s[2] != '\0' && IsHex(s[1]) && IsHex(s[2]);
    };

    u64 needed = 0;
    const char* src = in;
    while (*src) {
        if (IsValidPercentSequence(src)) {
            src += 3;
        } else {
            src++;
        }
        needed++;
    }
    needed++; // null terminator

    if (require) {
        *require = needed;
    }

    if (!out) {
        return ORBIS_OK;
    }

    if (prepare < needed) {
        LOG_ERROR(Lib_Http, "Buffer too small: need {} but only {} available", needed, prepare);
        return ORBIS_HTTP_ERROR_OUT_OF_MEMORY;
    }

    src = in;
    char* dst = out;
    while (*src) {
        if (IsValidPercentSequence(src)) {
            *dst++ = static_cast<char>((HexToInt(src[1]) << 4) | HexToInt(src[2]));
            src += 3;
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceHttpWaitRequest() {
    LOG_ERROR(Lib_Http, "(STUBBED) called");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    g_isConnectedToNetwork = Config::getIsConnectedToNetwork();
    host_override = Config::GetHttpHostOverride();

    LIB_FUNCTION("hvG6GfBMXg8", "libSceHttp", 1, "libSceHttp", sceHttpAbortRequest);
    LIB_FUNCTION("JKl06ZIAl6A", "libSceHttp", 1, "libSceHttp", sceHttpAbortRequestForce);
    LIB_FUNCTION("sWQiqKvYTVA", "libSceHttp", 1, "libSceHttp", sceHttpAbortWaitRequest);
    LIB_FUNCTION("mNan6QSnpeY", "libSceHttp", 1, "libSceHttp", sceHttpAddCookie);
    LIB_FUNCTION("JM58a21mtrQ", "libSceHttp", 1, "libSceHttp", sceHttpAddQuery);
    LIB_FUNCTION("EY28T2bkN7k", "libSceHttp", 1, "libSceHttp", sceHttpAddRequestHeader);
    LIB_FUNCTION("lGAjftanhFs", "libSceHttp", 1, "libSceHttp", sceHttpAddRequestHeaderRaw);
    LIB_FUNCTION("Y1DCjN-s2BA", "libSceHttp", 1, "libSceHttp", sceHttpAuthCacheExport);
    LIB_FUNCTION("zzB0StvRab4", "libSceHttp", 1, "libSceHttp", sceHttpAuthCacheFlush);
    LIB_FUNCTION("wF0KcxK20BE", "libSceHttp", 1, "libSceHttp", sceHttpAuthCacheImport);
    LIB_FUNCTION("A7n9nNg7NBg", "libSceHttp", 1, "libSceHttp",
                 sceHttpCacheRedirectedConnectionEnabled);
    LIB_FUNCTION("nOkViL17ZOo", "libSceHttp", 1, "libSceHttp", sceHttpCookieExport);
    LIB_FUNCTION("seCvUt91WHY", "libSceHttp", 1, "libSceHttp", sceHttpCookieFlush);
    LIB_FUNCTION("pFnXDxo3aog", "libSceHttp", 1, "libSceHttp", sceHttpCookieImport);
    LIB_FUNCTION("Kiwv9r4IZCc", "libSceHttp", 1, "libSceHttp", sceHttpCreateConnection);
    LIB_FUNCTION("qgxDBjorUxs", "libSceHttp", 1, "libSceHttp", sceHttpCreateConnectionWithURL);
    LIB_FUNCTION("6381dWF+xsQ", "libSceHttp", 1, "libSceHttp", sceHttpCreateEpoll);
    LIB_FUNCTION("tsGVru3hCe8", "libSceHttp", 1, "libSceHttp", sceHttpCreateRequest);
    LIB_FUNCTION("rGNm+FjIXKk", "libSceHttp", 1, "libSceHttp", sceHttpCreateRequest2);
    LIB_FUNCTION("Aeu5wVKkF9w", "libSceHttp", 1, "libSceHttp", sceHttpCreateRequestWithURL);
    LIB_FUNCTION("Cnp77podkCU", "libSceHttp", 1, "libSceHttp", sceHttpCreateRequestWithURL2);
    LIB_FUNCTION("0gYjPTR-6cY", "libSceHttp", 1, "libSceHttp", sceHttpCreateTemplate);
    LIB_FUNCTION("Lffcxao-QMM", "libSceHttp", 1, "libSceHttp", sceHttpDbgEnableProfile);
    LIB_FUNCTION("6gyx-I0Oob4", "libSceHttp", 1, "libSceHttp", sceHttpDbgGetConnectionStat);
    LIB_FUNCTION("fzzBpJjm9Kw", "libSceHttp", 1, "libSceHttp", sceHttpDbgGetRequestStat);
    LIB_FUNCTION("VmqSnjZ5mE4", "libSceHttp", 1, "libSceHttp", sceHttpDbgSetPrintf);
    LIB_FUNCTION("KJtUHtp6y0U", "libSceHttp", 1, "libSceHttp", sceHttpDbgShowConnectionStat);
    LIB_FUNCTION("oEuPssSYskA", "libSceHttp", 1, "libSceHttp", sceHttpDbgShowMemoryPoolStat);
    LIB_FUNCTION("L2gM3qptqHs", "libSceHttp", 1, "libSceHttp", sceHttpDbgShowRequestStat);
    LIB_FUNCTION("pxBsD-X9eH0", "libSceHttp", 1, "libSceHttp", sceHttpDbgShowStat);
    LIB_FUNCTION("P6A3ytpsiYc", "libSceHttp", 1, "libSceHttp", sceHttpDeleteConnection);
    LIB_FUNCTION("qe7oZ+v4PWA", "libSceHttp", 1, "libSceHttp", sceHttpDeleteRequest);
    LIB_FUNCTION("4I8vEpuEhZ8", "libSceHttp", 1, "libSceHttp", sceHttpDeleteTemplate);
    LIB_FUNCTION("wYhXVfS2Et4", "libSceHttp", 1, "libSceHttp", sceHttpDestroyEpoll);
    LIB_FUNCTION("1rpZqxdMRwQ", "libSceHttp", 1, "libSceHttp", sceHttpGetAcceptEncodingGZIPEnabled);
    LIB_FUNCTION("aCYPMSUIaP8", "libSceHttp", 1, "libSceHttp", sceHttpGetAllResponseHeaders);
    LIB_FUNCTION("9m8EcOGzcIQ", "libSceHttp", 1, "libSceHttp", sceHttpGetAuthEnabled);
    LIB_FUNCTION("mmLexUbtnfY", "libSceHttp", 1, "libSceHttp", sceHttpGetAutoRedirect);
    LIB_FUNCTION("L-DwVoHXLtU", "libSceHttp", 1, "libSceHttp", sceHttpGetConnectionStat);
    LIB_FUNCTION("+G+UsJpeXPc", "libSceHttp", 1, "libSceHttp", sceHttpGetCookie);
    LIB_FUNCTION("iSZjWw1TGiA", "libSceHttp", 1, "libSceHttp", sceHttpGetCookieEnabled);
    LIB_FUNCTION("xkymWiGdMiI", "libSceHttp", 1, "libSceHttp", sceHttpGetCookieStats);
    LIB_FUNCTION("7j9VcwnrZo4", "libSceHttp", 1, "libSceHttp", sceHttpGetEpoll);
    LIB_FUNCTION("IQOP6McWJcY", "libSceHttp", 1, "libSceHttp", sceHttpGetEpollId);
    LIB_FUNCTION("0onIrKx9NIE", "libSceHttp", 1, "libSceHttp", sceHttpGetLastErrno);
    LIB_FUNCTION("16sMmVuOvgU", "libSceHttp", 1, "libSceHttp", sceHttpGetMemoryPoolStats);
    LIB_FUNCTION("Wq4RNB3snSQ", "libSceHttp", 1, "libSceHttp", sceHttpGetNonblock);
    LIB_FUNCTION("hkcfqAl+82w", "libSceHttp", 1, "libSceHttp", sceHttpGetRegisteredCtxIds);
    LIB_FUNCTION("yuO2H2Uvnos", "libSceHttp", 1, "libSceHttp", sceHttpGetResponseContentLength);
    LIB_FUNCTION("0a2TBNfE3BU", "libSceHttp", 1, "libSceHttp", sceHttpGetStatusCode);
    LIB_FUNCTION("A9cVMUtEp4Y", "libSceHttp", 1, "libSceHttp", sceHttpInit);
    LIB_FUNCTION("hPTXo3bICzI", "libSceHttp", 1, "libSceHttp", sceHttpParseResponseHeader);
    LIB_FUNCTION("Qq8SfuJJJqE", "libSceHttp", 1, "libSceHttp", sceHttpParseStatusLine);
    LIB_FUNCTION("P5pdoykPYTk", "libSceHttp", 1, "libSceHttp", sceHttpReadData);
    LIB_FUNCTION("u05NnI+P+KY", "libSceHttp", 1, "libSceHttp", sceHttpRedirectCacheFlush);
    LIB_FUNCTION("zNGh-zoQTD0", "libSceHttp", 1, "libSceHttp", sceHttpRemoveRequestHeader);
    LIB_FUNCTION("4fgkfVeVsGU", "libSceHttp", 1, "libSceHttp", sceHttpRequestGetAllHeaders);
    LIB_FUNCTION("mSQCxzWTwVI", "libSceHttp", 1, "libSceHttp", sceHttpsDisableOption);
    LIB_FUNCTION("zJYi5br6ZiQ", "libSceHttp", 1, "libSceHttp", sceHttpsDisableOptionPrivate);
    LIB_FUNCTION("f42K37mm5RM", "libSceHttp", 1, "libSceHttp", sceHttpsEnableOption);
    LIB_FUNCTION("I4+4hKttt1w", "libSceHttp", 1, "libSceHttp", sceHttpsEnableOptionPrivate);
    LIB_FUNCTION("1e2BNwI-XzE", "libSceHttp", 1, "libSceHttp", sceHttpSendRequest);
    LIB_FUNCTION("HRX1iyDoKR8", "libSceHttp", 1, "libSceHttp", sceHttpSetAcceptEncodingGZIPEnabled);
    LIB_FUNCTION("qFg2SuyTJJY", "libSceHttp", 1, "libSceHttp", sceHttpSetAuthEnabled);
    LIB_FUNCTION("jf4TB2nUO40", "libSceHttp", 1, "libSceHttp", sceHttpSetAuthInfoCallback);
    LIB_FUNCTION("T-mGo9f3Pu4", "libSceHttp", 1, "libSceHttp", sceHttpSetAutoRedirect);
    LIB_FUNCTION("PDxS48xGQLs", "libSceHttp", 1, "libSceHttp", sceHttpSetChunkedTransferEnabled);
    LIB_FUNCTION("0S9tTH0uqTU", "libSceHttp", 1, "libSceHttp", sceHttpSetConnectTimeOut);
    LIB_FUNCTION("XNUoD2B9a6A", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieEnabled);
    LIB_FUNCTION("pM--+kIeW-8", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieMaxNum);
    LIB_FUNCTION("Kp6juCJUJGQ", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieMaxNumPerDomain);
    LIB_FUNCTION("7Y4364GBras", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieMaxSize);
    LIB_FUNCTION("Kh6bS2HQKbo", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieRecvCallback);
    LIB_FUNCTION("GnVDzYfy-KI", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieSendCallback);
    LIB_FUNCTION("pHc3bxUzivU", "libSceHttp", 1, "libSceHttp", sceHttpSetCookieTotalMaxSize);
    LIB_FUNCTION("8kzIXsRy1bY", "libSceHttp", 1, "libSceHttp",
                 sceHttpSetDefaultAcceptEncodingGZIPEnabled);
    LIB_FUNCTION("22buO-UufJY", "libSceHttp", 1, "libSceHttp", sceHttpSetDelayBuildRequestEnabled);
    LIB_FUNCTION("-xm7kZQNpHI", "libSceHttp", 1, "libSceHttp", sceHttpSetEpoll);
    LIB_FUNCTION("LG1YW1Uhkgo", "libSceHttp", 1, "libSceHttp", sceHttpSetEpollId);
    LIB_FUNCTION("pk0AuomQM1o", "libSceHttp", 1, "libSceHttp", sceHttpSetHttp09Enabled);
    LIB_FUNCTION("i9mhafzkEi8", "libSceHttp", 1, "libSceHttp", sceHttpSetInflateGZIPEnabled);
    LIB_FUNCTION("s2-NPIvz+iA", "libSceHttp", 1, "libSceHttp", sceHttpSetNonblock);
    LIB_FUNCTION("gZ9TpeFQ7Gk", "libSceHttp", 1, "libSceHttp", sceHttpSetPolicyOption);
    LIB_FUNCTION("2NeZnMEP3-0", "libSceHttp", 1, "libSceHttp", sceHttpSetPriorityOption);
    LIB_FUNCTION("i+quCZCL+D8", "libSceHttp", 1, "libSceHttp", sceHttpSetProxy);
    LIB_FUNCTION("mMcB2XIDoV4", "libSceHttp", 1, "libSceHttp", sceHttpSetRecvBlockSize);
    LIB_FUNCTION("yigr4V0-HTM", "libSceHttp", 1, "libSceHttp", sceHttpSetRecvTimeOut);
    LIB_FUNCTION("h9wmFZX4i-4", "libSceHttp", 1, "libSceHttp", sceHttpSetRedirectCallback);
    LIB_FUNCTION("PTiFIUxCpJc", "libSceHttp", 1, "libSceHttp", sceHttpSetRequestContentLength);
    LIB_FUNCTION("vO4B-42ef-k", "libSceHttp", 1, "libSceHttp", sceHttpSetRequestStatusCallback);
    LIB_FUNCTION("K1d1LqZRQHQ", "libSceHttp", 1, "libSceHttp", sceHttpSetResolveRetry);
    LIB_FUNCTION("Tc-hAYDKtQc", "libSceHttp", 1, "libSceHttp", sceHttpSetResolveTimeOut);
    LIB_FUNCTION("a4VsZ4oqn68", "libSceHttp", 1, "libSceHttp", sceHttpSetResponseHeaderMaxSize);
    LIB_FUNCTION("xegFfZKBVlw", "libSceHttp", 1, "libSceHttp", sceHttpSetSendTimeOut);
    LIB_FUNCTION("POJ0azHZX3w", "libSceHttp", 1, "libSceHttp", sceHttpSetSocketCreationCallback);
    LIB_FUNCTION("7WcNoAI9Zcw", "libSceHttp", 1, "libSceHttp", sceHttpsFreeCaList);
    LIB_FUNCTION("gcUjwU3fa0M", "libSceHttp", 1, "libSceHttp", sceHttpsGetCaList);
    LIB_FUNCTION("JBN6N-EY+3M", "libSceHttp", 1, "libSceHttp", sceHttpsGetSslError);
    LIB_FUNCTION("DK+GoXCNT04", "libSceHttp", 1, "libSceHttp", sceHttpsLoadCert);
    LIB_FUNCTION("jUjp+yqMNdQ", "libSceHttp", 1, "libSceHttp", sceHttpsSetMinSslVersion);
    LIB_FUNCTION("htyBOoWeS58", "libSceHttp", 1, "libSceHttp", sceHttpsSetSslCallback);
    LIB_FUNCTION("U5ExQGyyx9s", "libSceHttp", 1, "libSceHttp", sceHttpsSetSslVersion);
    LIB_FUNCTION("zXqcE0fizz0", "libSceHttp", 1, "libSceHttp", sceHttpsUnloadCert);
    LIB_FUNCTION("Ik-KpLTlf7Q", "libSceHttp", 1, "libSceHttp", sceHttpTerm);
    LIB_FUNCTION("V-noPEjSB8c", "libSceHttp", 1, "libSceHttp", sceHttpTryGetNonblock);
    LIB_FUNCTION("fmOs6MzCRqk", "libSceHttp", 1, "libSceHttp", sceHttpTrySetNonblock);
    LIB_FUNCTION("59tL1AQBb8U", "libSceHttp", 1, "libSceHttp", sceHttpUnsetEpoll);
    LIB_FUNCTION("5LZA+KPISVA", "libSceHttp", 1, "libSceHttp", sceHttpUriBuild);
    LIB_FUNCTION("CR-l-yI-o7o", "libSceHttp", 1, "libSceHttp", sceHttpUriCopy);
    LIB_FUNCTION("YuOW3dDAKYc", "libSceHttp", 1, "libSceHttp", sceHttpUriEscape);
    LIB_FUNCTION("3lgQ5Qk42ok", "libSceHttp", 1, "libSceHttp", sceHttpUriMerge);
    LIB_FUNCTION("IWalAn-guFs", "libSceHttp", 1, "libSceHttp", sceHttpUriParse);
    LIB_FUNCTION("mUU363n4yc0", "libSceHttp", 1, "libSceHttp", sceHttpUriSweepPath);
    LIB_FUNCTION("thTS+57zoLM", "libSceHttp", 1, "libSceHttp", sceHttpUriUnescape);
    LIB_FUNCTION("qISjDHrxONc", "libSceHttp", 1, "libSceHttp", sceHttpWaitRequest);
};

} // namespace Libraries::Http

// GR2FORK: launch gate against the restoration server (declared in gr2_launch_gate.h). One
// /clientgate GET carrying the fork's identity headers lets the server answer "update required"
// (426), "banned" (403 + X-GR2-Banned: days|perm), or "debug lockdown" (403 + X-GR2-Gate: debug)
// before the guest runs, instead of degrading into silent per-request failures in-game.
namespace GR2Fork {

LaunchGate LaunchGateCheck() {
    LaunchGate gate{};
    // GR2 SKUs only: core_gr2 also boots GRR and other titles, which have no restoration server.
    static constexpr const char* kGr2Serials[] = {
        "CUSA03694", "CUSA04943", "CUSA04934", "CUSA00547",
        "PCJS50010", "PCAS00079", "CUSA04935",
    };
    const std::string_view serial = Common::ElfInfo::Instance().GameSerial();
    bool is_gr2 = false;
    for (const char* s : kGr2Serials) {
        if (serial == s) {
            is_gr2 = true;
            break;
        }
    }
    if (!is_gr2 || !Auth::IsOnlineEnabled()) {
        return gate;
    }
    const std::string host = Config::GetHttpHostOverride();
    const int port = Config::GetHttpHostOverridePort();
    if (host.empty()) {
        return gate;
    }
    Auth::EnsureAuthed();
    const std::string player = Auth::EffectiveOnlineId();
    // Same header-value sanitation as the other X-GR2-Player producers (drop control bytes + DEL).
    std::string safe;
    safe.reserve(player.size());
    for (unsigned char c : player) {
        if (c >= 0x20 && c != 0x7f) {
            safe.push_back(static_cast<char>(c));
        }
    }
    httplib::Client cli(host, port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(5, 0);
    cli.set_write_timeout(5, 0);
    httplib::Headers headers;
    if (!safe.empty()) {
        headers.emplace("X-GR2-Player", safe);
    }
    const std::string token = Auth::BearerToken();
    if (!token.empty()) {
        headers.emplace("X-GR2-Auth", token);
    }
    headers.emplace("X-GR2-Version", std::to_string(OnlineVersion));
    auto res = cli.Get("/clientgate?v=" + std::to_string(OnlineVersion), headers);
    if (!res) {
        return gate; // unreachable server: boot normally, offline play stays available
    }
    if (res->status == 426) {
        gate.status = LaunchGate::Status::UpdateRequired;
        return gate;
    }
    if (res->status == 403) {
        const std::string banned = res->get_header_value("X-GR2-Banned");
        if (!banned.empty()) {
            gate.status = LaunchGate::Status::Banned;
            int days = 0;
            for (char c : banned) {
                if (c < '0' || c > '9') {
                    days = 0; // "perm" or malformed reports as a permanent ban
                    break;
                }
                days = days * 10 + (c - '0');
                if (days > 99999) {
                    break;
                }
            }
            gate.ban_days = days;
            return gate;
        }
        if (res->get_header_value("X-GR2-Gate") == "debug") {
            gate.status = LaunchGate::Status::DebugDenied;
            return gate;
        }
    }
    return gate;
}

} // namespace GR2Fork
