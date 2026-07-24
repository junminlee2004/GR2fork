// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <future>
#include <map>
#include <mutex>
#include <string>

#include <httplib.h>

#include "common/gr2_online_version.h"
#include "common/logging/log.h"
#include "common/types.h"
#include "core/libraries/network/ssl.h"
#include "core/libraries/np/gr2_online_auth.h" // shadnet-verified identity + bearer token
#include "core/user_settings.h" // UserManagement -> the local player's display name (leaderboard)

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::Http {

struct OrbisHttpUriElement {
    bool opaque;
    char* scheme;
    char* username;
    char* password;
    char* hostname;
    char* path;
    char* query;
    char* fragment;
    u16 port;
    u8 reserved[10];
};

enum OrbisHttpRequestMethod : s32 {
    ORBIS_INTERNAL_HTTP_REQUEST_METHOD_GET = 0,
    ORBIS_INTERNAL_HTTP_REQUEST_METHOD_POST = 1,
    ORBIS_INTERNAL_HTTP_REQUEST_METHOD_HEAD = 2,
    ORBIS_INTERNAL_HTTP_REQUEST_METHOD_OPTIONS = 3,
    ORBIS_INTERNAL_HTTP_REQUEST_METHOD_PUT = 4,
    ORBIS_INTERNAL_HTTP_REQUEST_METHOD_DELETE = 5,
    ORBIS_INTERNAL_HTTP_REQUEST_METHOD_TRACE = 6,
    ORBIS_INTERNAL_HTTP_REQUEST_METHOD_CONNECT = 7,
    ORBIS_INTERNAL_HTTP_REQUEST_METHOD_INVALID = 8,
};

class RequestTemplate {
public:
    int id;
    std::map<std::string, std::string> headers;
    std::string user_agent = {};
    bool is_async = false;

    void AddHeader(const char* name, const char* value) {

        headers[std::string(name)] = std::string(value);
    }

    RequestTemplate() : id(0) {}
    explicit RequestTemplate(int tmpl_id, std::string user_agent = "")
        : id(tmpl_id), user_agent(user_agent) {}
};

class RequestObj {
public:
    int id;
    RequestTemplate* req_template = nullptr;

    void SendRequest() {
        request_future = std::async(std::launch::async, [this] { _SendRequest(); });

        if (!req_template->is_async) {
            WaitForRequest();
        }
    }

    bool IsRequestComplete() {
        if (!request_future.valid())
            return true;
        return request_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    void WaitForRequest() {
        if (request_future.valid()) {
            request_future.get();
        }
    }

    u32 ReadData(char* dest, u32 size) {

        if (result_body == nullptr || dest == nullptr || size == 0 || result_body_size == 0) {

            return 0;
        }

        // Byte offset into result_body, advanced by the bytes actually copied each call. A
        // chunk-index-times-size scheme corrupts bodies read in variable-sized chunks (e.g. the
        // ~22 KB ss.info container); a byte offset is correct for any read-size pattern.
        u64 start_index = current_result_read_offset;

        if (start_index >= result_body_size) {

            return 0;
        }

        u64 remaining_bytes = result_body_size - start_index;

        u64 bytes_to_copy = (remaining_bytes < size) ? remaining_bytes : size;

        std::memcpy(dest, result_body + start_index, bytes_to_copy);

        current_result_read_offset += bytes_to_copy;

        LOG_INFO(Lib_Http, "ReadData: offset={} req_size={} copied={} body_total={}", start_index,
                 size, bytes_to_copy, result_body_size);

        return static_cast<u32>(bytes_to_copy);
    }

    std::string GetUrl() const {
        return url;
    }

    s32 GetMethod() const {
        return static_cast<s32>(method);
    }

    void SetUrl(const std::string& new_url) {

        if (new_url.empty()) {
            return;
        }

        // TODO checks
        url = new_url;

        u64 scheme_end = url.find("://");
        u64 path_start = url.find('/', scheme_end + 3);

        if (path_start == std::string::npos) {
            host = url;
            path = "/";
        } else {
            host = url.substr(0, path_start);
            path = url.substr(path_start);
        }
    }

    void SetPostData(const void* data, u64 size) {

        if (data == nullptr || size == 0) {

            post_data = nullptr;
            post_data_size = 0;
            return;
        }

        post_data_size = size;

        post_data = new u8[post_data_size];
        std::memcpy(post_data, data, post_data_size);
    }

    // Streaming-body accumulation: the game can deliver one request body across multiple
    // sceHttpSendRequest calls (the ARC login multipart streams as 78 + 152 + 38 = 268 bytes);
    // append each chunk to post_data so the whole body goes out as one request, not fragments.
    void AppendPostData(const void* data, u64 size) {

        if (data == nullptr || size == 0) {

            return;
        }

        u8* combined = new u8[post_data_size + size];
        if (post_data != nullptr && post_data_size > 0) {

            std::memcpy(combined, post_data, post_data_size);
            delete[] static_cast<u8*>(post_data);
        }
        std::memcpy(combined + post_data_size, data, size);

        post_data = combined;
        post_data_size += size;
    }

    // True once the accumulated body has reached the content_length declared at create time.
    // content_length == 0 (a GET, or a request with no declared body) means "send immediately".
    bool IsBodyComplete() const {

        return content_length == 0 || post_data_size >= content_length;
    }

    // Append one body chunk and dispatch the request only once the full declared body has
    // arrived. Returns true if the request was actually sent on this call.
    bool AccumulateAndSend(const void* data, u64 size) {

        AppendPostData(data, size);

        if (IsBodyComplete()) {

            LOG_INFO(Lib_Http, "sceHttpSendRequest: body complete ({} bytes) -- dispatching request",
                     post_data_size);
            SendRequest();
            return true;
        }

        LOG_INFO(Lib_Http,
                 "sceHttpSendRequest: buffered streamed chunk (+{} bytes); accumulated {} of {} "
                 "declared -- holding until body is complete",
                 size, post_data_size, content_length);
        return false;
    }

    void AddHeader(const char* name, const char* value) {

        req_headers[std::string(name)] = std::string(value);
    }

    bool IsSuccessful() const {
        return status_code / 100 == 2;
    }

    u32 GetStatusCode() const {
        return status_code;
    }

    u64 GetContentLength() const {
        return static_cast<u64>(result_body_size);
    }

    bool IsSent() const {
        return is_sent;
    }

    bool IsCompleted() {
        return status_code != -1;
    }

    void DebugPrint() const {
        LOG_DEBUG(Lib_Http, "===== HTTP Request Debug Info =====");

        LOG_DEBUG(Lib_Http, "Is Sent:        {}", (is_sent ? "true" : "false"));
        LOG_DEBUG(Lib_Http, "Future Valid:   {}", (request_future.valid() ? "true" : "false"));
        LOG_DEBUG(Lib_Http, "Status Code:    {}", status_code);
        LOG_DEBUG(Lib_Http, "Method (Enum):  {}", static_cast<int>(method));

        LOG_DEBUG(Lib_Http, "URL:            {}", (url.empty() ? "[Empty]" : url));
        LOG_DEBUG(Lib_Http, "Host:           {}", (host.empty() ? "[Empty]" : host));
        LOG_DEBUG(Lib_Http, "Path:           {}", (path.empty() ? "[Empty]" : path));

        LOG_DEBUG(Lib_Http, "Post Data Size: {} bytes", post_data_size);
        LOG_DEBUG(Lib_Http, "Post Data Ptr:  {}", post_data);

        if (post_data && post_data_size > 0) {
            std::string preview_str;
            const char* preview = static_cast<const char*>(post_data);
            for (u64 i = 0; i < std::min<u64>(post_data_size, 20); ++i) {
                char c = preview[i];
                preview_str += (std::isprint(static_cast<unsigned char>(c)) ? c : '.');
            }
            LOG_DEBUG(Lib_Http, "  -> Preview: {}...", preview_str);
        }

        LOG_DEBUG(Lib_Http, "Content Length: {}", content_length);
        LOG_DEBUG(Lib_Http, "Body Size:      {} bytes", result_body_size);
        LOG_DEBUG(Lib_Http, "Read Offset: {}", current_result_read_offset);
        LOG_DEBUG(Lib_Http, "Body Pointer:   {}", (void*)result_body);

        if (result_body) {
            std::string body_preview;
            for (u64 i = 0; i < std::min<u64>(result_body_size, 50); ++i) {
                char c = result_body[i];
                body_preview += (std::isprint(static_cast<unsigned char>(c)) ? c : '.');
            }
            LOG_DEBUG(Lib_Http, "  -> Body Preview: {}...", body_preview);
        }

        LOG_DEBUG(Lib_Http, "Request Headers ({})", req_headers.size());
        if (req_headers.empty()) {
            LOG_DEBUG(Lib_Http, "  [None]");
        } else {
            for (const auto& pair : req_headers) {
                LOG_DEBUG(Lib_Http, "  [{}] : {}", pair.first, pair.second);
            }
        }

        LOG_DEBUG(Lib_Http, "===================================");
    }

    RequestObj()
        : id(0), req_template(nullptr), method(ORBIS_INTERNAL_HTTP_REQUEST_METHOD_INVALID), url(""),
          content_length(0), status_code(-1), result_body(nullptr), result_body_size(-1),
          current_result_read_offset(0), post_data(nullptr), is_sent(false) {}
    explicit RequestObj(s32 req_id, RequestTemplate* req_template, s32 method, std::string url_str,
                        u64 cntLen)
        : id(req_id), req_template(req_template),
          method(static_cast<OrbisHttpRequestMethod>(method)), content_length(cntLen),
          status_code(-1), result_body(nullptr), result_body_size(-1),
          current_result_read_offset(0), post_data(nullptr), is_sent(false) {

        SetUrl(url_str);
    }

private:
    std::future<void> request_future = {};

    char* result_body = nullptr;
    u64 current_result_read_offset = 0;
    u32 result_body_size = 0;

    OrbisHttpRequestMethod method = ORBIS_INTERNAL_HTTP_REQUEST_METHOD_INVALID;
    std::string host = {};
    std::string path = {};
    u64 content_length = 0;

    u32 status_code = -1; // check
    bool is_sent = false;

    std::map<std::string, std::string> req_headers;
    std::string url = {};
    void* post_data = nullptr;
    u64 post_data_size = 0;

    void _SendRequest() {

        httplib::Client cli(host);

        httplib::Result response = {};

        auto templ_headers = req_template->headers;

        httplib::Headers headers;
        for (const auto& pair : templ_headers) {
            headers.emplace(pair.first, pair.second);
        }

        for (const auto& pair : req_headers) {
            headers.emplace(pair.first, pair.second);
        }

        // GR2 identity: authenticate to shadnet once, then forward the verified Online ID plus the
        // bearer token the restoration server validates. Falls back to the local UserManagement
        // username when shadnet is unconfigured, so an un-set-up player still reaches the server
        // (which may reject in enforce mode). Sanitized to a safe header value.
        {
            GR2Fork::Auth::EnsureAuthed();
            std::string player = GR2Fork::Auth::VerifiedNpid();
            if (player.empty()) {
                player = UserManagement.GetDefaultUser().user_name;
            }
            std::string safe;
            safe.reserve(player.size());
            for (unsigned char c : player) {
                if (c >= 0x20 && c != 0x7f) {
                    safe.push_back(static_cast<char>(c));
                }
            }
            if (!safe.empty()) {
                headers.emplace("X-GR2-Player", safe);
            }
            const std::string token = GR2Fork::Auth::BearerToken();
            if (!token.empty()) {
                headers.emplace("X-GR2-Auth", token);
            }
            headers.emplace("X-GR2-Version", std::to_string(GR2Fork::OnlineVersion));
        }

        std::string content_type = "application/json";

        auto it = headers.find("Content-Type");
        if (it != headers.end()) {
            content_type = it->second;
        }

        is_sent = true;

        LOG_INFO(Lib_Http, "Sending HTTP request: method={} host='{}' path='{}' post_size={}",
                 static_cast<int>(method), host, path, post_data_size);

        // GR2 upload probe: log a printable preview of the outgoing body so the multipart form
        // fields the game assembled are visible - the only record of the S3 POST fields if an
        // upload targets a foreign server. Non-printables shown as '.'; capped; body-only.
        if (post_data != nullptr && post_data_size > 0) {
            const u64 n = std::min<u64>(post_data_size, 512);
            std::string preview;
            preview.reserve(n);
            const u8* pv = static_cast<const u8*>(post_data);
            for (u64 i = 0; i < n; ++i) {
                const u8 c = pv[i];
                preview.push_back((c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.');
            }
            LOG_INFO(Lib_Http, "  body[{} of {}]: {}", n, post_data_size, preview);
        }

        switch (method) {
        case ORBIS_INTERNAL_HTTP_REQUEST_METHOD_GET:

            response = cli.Get(path, headers);
            break;
        case ORBIS_INTERNAL_HTTP_REQUEST_METHOD_POST:

            response = cli.Post(path, headers, static_cast<char*>(post_data),
                                static_cast<u64>(post_data_size), content_type);
            break;
        case ORBIS_INTERNAL_HTTP_REQUEST_METHOD_HEAD:
            response = cli.Head(path, headers);
            break;
        case ORBIS_INTERNAL_HTTP_REQUEST_METHOD_OPTIONS:
            response = cli.Options(path, headers);
            break;
        case ORBIS_INTERNAL_HTTP_REQUEST_METHOD_PUT:
            response = cli.Put(path, headers, static_cast<char*>(post_data),
                               static_cast<u64>(post_data_size), content_type);
            break;
        case ORBIS_INTERNAL_HTTP_REQUEST_METHOD_DELETE:
            response = cli.Delete(path, headers);
            break;
        case ORBIS_INTERNAL_HTTP_REQUEST_METHOD_TRACE:
            LOG_ERROR(Lib_Http, "TRACE HTTP method not implemented");
            return;
        case ORBIS_INTERNAL_HTTP_REQUEST_METHOD_CONNECT:
            LOG_ERROR(Lib_Http, "CONNECT HTTP method not implemented");
            return;

        default:
        case ORBIS_INTERNAL_HTTP_REQUEST_METHOD_INVALID:
            LOG_ERROR(Lib_Http, "Invalid HTTP method");
            return;
        }

        if (!response) {

            LOG_ERROR(Lib_Http,
                      "HTTP request to host='{}' path='{}' got NO response (httplib error {}) -- "
                      "connection refused / wrong port / server down?",
                      host, path, static_cast<int>(response.error()));
            return;
        }

        status_code = response->status;

        LOG_INFO(Lib_Http, "HTTP response: host='{}' path='{}' status={} body_size={}", host, path,
                 status_code, response->body.size());

        if (response && response->status / 100 == 2) {

            result_body_size = static_cast<u32>(response->body.size());
            result_body = new char[result_body_size];
            std::memcpy(result_body, response->body.data(), result_body_size);
        }
    }
};

struct HttpRequestInternal {
    int state;          // +0x20
    int errorCode;      // +0x28
    int httpStatusCode; // +0x20C
    std::mutex m_mutex;
};
using OrbisHttpsCaList = Libraries::Ssl::OrbisSslCaList;

int PS4_SYSV_ABI sceHttpAbortRequest();
int PS4_SYSV_ABI sceHttpAbortRequestForce();
int PS4_SYSV_ABI sceHttpAbortWaitRequest();
int PS4_SYSV_ABI sceHttpAddCookie();
int PS4_SYSV_ABI sceHttpAddQuery();
int PS4_SYSV_ABI sceHttpAddRequestHeader(int id, const char* name, const char* value, s32 mode);
int PS4_SYSV_ABI sceHttpAddRequestHeaderRaw();
int PS4_SYSV_ABI sceHttpAuthCacheExport();
int PS4_SYSV_ABI sceHttpAuthCacheFlush();
int PS4_SYSV_ABI sceHttpAuthCacheImport();
int PS4_SYSV_ABI sceHttpCacheRedirectedConnectionEnabled();
int PS4_SYSV_ABI sceHttpCookieExport();
int PS4_SYSV_ABI sceHttpCookieFlush();
int PS4_SYSV_ABI sceHttpCookieImport();
int PS4_SYSV_ABI sceHttpCreateConnection();
int PS4_SYSV_ABI sceHttpCreateConnectionWithURL(int tmplId, const char* url, bool enableKeepalive);
int PS4_SYSV_ABI sceHttpCreateEpoll();
int PS4_SYSV_ABI sceHttpCreateRequest(s32 conn_id, s32 method, const char* path,
                                      u64 content_length);
int PS4_SYSV_ABI sceHttpCreateRequest2(s32 conn_id, const char* method, const char* path,
                                       u64 content_length);
int PS4_SYSV_ABI sceHttpCreateRequestWithURL(s32 conn_id, s32 method, const char* url,
                                             u64 content_length);
int PS4_SYSV_ABI sceHttpCreateRequestWithURL2(s32 conn_id, const char* method, const char* url,
                                              u64 content_length);
int PS4_SYSV_ABI sceHttpCreateTemplate(s32 conn_id, const char* user_agent, s32 http_v, s32 flags);
int PS4_SYSV_ABI sceHttpDbgEnableProfile();
int PS4_SYSV_ABI sceHttpDbgGetConnectionStat();
int PS4_SYSV_ABI sceHttpDbgGetRequestStat();
int PS4_SYSV_ABI sceHttpDbgSetPrintf();
int PS4_SYSV_ABI sceHttpDbgShowConnectionStat();
int PS4_SYSV_ABI sceHttpDbgShowMemoryPoolStat();
int PS4_SYSV_ABI sceHttpDbgShowRequestStat();
int PS4_SYSV_ABI sceHttpDbgShowStat();
int PS4_SYSV_ABI sceHttpDeleteConnection();
int PS4_SYSV_ABI sceHttpDeleteRequest(s32 req_id);
int PS4_SYSV_ABI sceHttpDeleteTemplate();
int PS4_SYSV_ABI sceHttpDestroyEpoll();
int PS4_SYSV_ABI sceHttpGetAcceptEncodingGZIPEnabled();
int PS4_SYSV_ABI sceHttpGetAllResponseHeaders(int reqId, char** header, u64* headerSize);
int PS4_SYSV_ABI sceHttpGetAuthEnabled();
int PS4_SYSV_ABI sceHttpGetAutoRedirect();
int PS4_SYSV_ABI sceHttpGetConnectionStat();
int PS4_SYSV_ABI sceHttpGetCookie();
int PS4_SYSV_ABI sceHttpGetCookieEnabled();
int PS4_SYSV_ABI sceHttpGetCookieStats();
int PS4_SYSV_ABI sceHttpGetEpoll();
int PS4_SYSV_ABI sceHttpGetEpollId();
int PS4_SYSV_ABI sceHttpGetLastErrno();
int PS4_SYSV_ABI sceHttpGetMemoryPoolStats();
int PS4_SYSV_ABI sceHttpGetNonblock();
int PS4_SYSV_ABI sceHttpGetRegisteredCtxIds();
int PS4_SYSV_ABI sceHttpGetResponseContentLength(u32 req_id, s32* result, u64* content_length);
int PS4_SYSV_ABI sceHttpGetStatusCode(s32 req_id, s32* status_code);
int PS4_SYSV_ABI sceHttpInit(int libnetMemId, int libsslCtxId, u64 poolSize);
int PS4_SYSV_ABI sceHttpParseResponseHeader(const char* header, u64 headerLen, const char* fieldStr,
                                            const char** fieldValue, u64* valueLen);
int PS4_SYSV_ABI sceHttpParseStatusLine(const char* statusLine, u64 lineLen, int32_t* httpMajorVer,
                                        int32_t* httpMinorVer, int32_t* responseCode,
                                        const char** reasonPhrase, u64* phraseLen);
int PS4_SYSV_ABI sceHttpReadData(u32 req_id, char* dest, u32 size);
int PS4_SYSV_ABI sceHttpRedirectCacheFlush();
int PS4_SYSV_ABI sceHttpRemoveRequestHeader();
int PS4_SYSV_ABI sceHttpRequestGetAllHeaders();
int PS4_SYSV_ABI sceHttpsDisableOption();
int PS4_SYSV_ABI sceHttpsDisableOptionPrivate();
int PS4_SYSV_ABI sceHttpsEnableOption(u32 options);
int PS4_SYSV_ABI sceHttpsEnableOptionPrivate();
int PS4_SYSV_ABI sceHttpSendRequest(int reqId, const void* postData, u64 size);
int PS4_SYSV_ABI sceHttpSetAcceptEncodingGZIPEnabled();
int PS4_SYSV_ABI sceHttpSetAuthEnabled();
int PS4_SYSV_ABI sceHttpSetAuthInfoCallback();
int PS4_SYSV_ABI sceHttpSetAutoRedirect();
int PS4_SYSV_ABI sceHttpSetChunkedTransferEnabled();
int PS4_SYSV_ABI sceHttpSetConnectTimeOut();
int PS4_SYSV_ABI sceHttpSetCookieEnabled();
int PS4_SYSV_ABI sceHttpSetCookieMaxNum();
int PS4_SYSV_ABI sceHttpSetCookieMaxNumPerDomain();
int PS4_SYSV_ABI sceHttpSetCookieMaxSize();
int PS4_SYSV_ABI sceHttpSetCookieRecvCallback();
int PS4_SYSV_ABI sceHttpSetCookieSendCallback();
int PS4_SYSV_ABI sceHttpSetCookieTotalMaxSize();
int PS4_SYSV_ABI sceHttpSetDefaultAcceptEncodingGZIPEnabled();
int PS4_SYSV_ABI sceHttpSetDelayBuildRequestEnabled();
int PS4_SYSV_ABI sceHttpSetEpoll();
int PS4_SYSV_ABI sceHttpSetEpollId();
int PS4_SYSV_ABI sceHttpSetHttp09Enabled();
int PS4_SYSV_ABI sceHttpSetInflateGZIPEnabled();
int PS4_SYSV_ABI sceHttpSetNonblock(s32 tmpl_id, bool enable);
int PS4_SYSV_ABI sceHttpSetPolicyOption();
int PS4_SYSV_ABI sceHttpSetPriorityOption();
int PS4_SYSV_ABI sceHttpSetProxy();
int PS4_SYSV_ABI sceHttpSetRecvBlockSize();
int PS4_SYSV_ABI sceHttpSetRecvTimeOut();
int PS4_SYSV_ABI sceHttpSetRedirectCallback();
int PS4_SYSV_ABI sceHttpSetRequestContentLength();
int PS4_SYSV_ABI sceHttpSetRequestStatusCallback();
int PS4_SYSV_ABI sceHttpSetResolveRetry();
int PS4_SYSV_ABI sceHttpSetResolveTimeOut();
int PS4_SYSV_ABI sceHttpSetResponseHeaderMaxSize();
int PS4_SYSV_ABI sceHttpSetSendTimeOut();
int PS4_SYSV_ABI sceHttpSetSocketCreationCallback();
int PS4_SYSV_ABI sceHttpsFreeCaList();
int PS4_SYSV_ABI sceHttpsGetCaList(int httpCtxId, OrbisHttpsCaList* list);
int PS4_SYSV_ABI sceHttpsGetSslError();
int PS4_SYSV_ABI sceHttpsLoadCert();
int PS4_SYSV_ABI sceHttpsSetMinSslVersion();
int PS4_SYSV_ABI sceHttpsSetSslCallback();
int PS4_SYSV_ABI sceHttpsSetSslVersion();
int PS4_SYSV_ABI sceHttpsUnloadCert();
int PS4_SYSV_ABI sceHttpTerm();
int PS4_SYSV_ABI sceHttpTryGetNonblock();
int PS4_SYSV_ABI sceHttpTrySetNonblock();
int PS4_SYSV_ABI sceHttpUnsetEpoll();
int PS4_SYSV_ABI sceHttpUriBuild(char* out, u64* require, u64 prepare,
                                 const OrbisHttpUriElement* srcElement, u32 option);
int PS4_SYSV_ABI sceHttpUriCopy();
int PS4_SYSV_ABI sceHttpUriEscape(char* out, u64* require, u64 prepare, const char* in);
int PS4_SYSV_ABI sceHttpUriMerge(char* mergedUrl, char* url, char* relativeUri, u64* require,
                                 u64 prepare, u32 option);
int PS4_SYSV_ABI sceHttpUriParse(OrbisHttpUriElement* out, const char* srcUri, void* pool,
                                 u64* require, u64 prepare);
int PS4_SYSV_ABI sceHttpUriSweepPath(char* dst, const char* src, u64 srcSize);
int PS4_SYSV_ABI sceHttpUriUnescape(char* out, u64* require, u64 prepare, const char* in);
int PS4_SYSV_ABI sceHttpWaitRequest();

void RegisterLib(Core::Loader::SymbolsResolver* sym);
} // namespace Libraries::Http
