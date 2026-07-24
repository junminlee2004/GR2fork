// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "common/config.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "core/libraries/np/gr2_online_auth.h"

// Server-verified login for the GR2 online restoration. Enough of the shadnet framed-protobuf TCP
// protocol to authenticate the configured Online ID + password and capture the bearer token; the
// restoration server re-validates the token, so only the token and the login outcome matter here. The
// upstream shadnet client lives in core_main and its protobuf codegen is not generated for core_gr2, so
// the minimal protocol is inlined instead of taking that cross-core dependency. Packet layout: a 15-byte
// header [type u8][cmd u16-LE][total u32-LE][id u64-LE] then, for requests, a u32-LE-prefixed protobuf
// blob; replies prefix the blob with a one-byte error code.

namespace GR2Fork::Auth {
namespace {

#ifdef _WIN32
using SockT = SOCKET;
constexpr SockT kBadSock = INVALID_SOCKET;
inline void CloseSock(SockT s) {
    ::closesocket(s);
}
#else
using SockT = int;
constexpr SockT kBadSock = -1;
inline void CloseSock(SockT s) {
    ::close(s);
}
#endif

std::once_flag g_once;
std::atomic<bool> g_authed{false};
std::mutex g_state_mutex;
std::string g_token;
std::string g_npid;
// GR2FORK: login outcome for the launch-time preflight. Seeded FailConnect at LoginThread entry (the
// default for any connection-level failure) and overwritten as the handshake progresses.
enum class LoginResult { Pending, Ok, FailConnect, FailCreds, FailOther };
std::atomic<LoginResult> g_login_result{LoginResult::Pending};
std::atomic<int> g_login_err{0};

// ---- proto3 wire helpers (length-delimited fields + varints, which is all Login/GetToken use) ----

void PutVarint(std::vector<uint8_t>& out, uint64_t v) {
    while (v >= 0x80) {
        out.push_back(static_cast<uint8_t>(v) | 0x80);
        v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v));
}

void PutStrField(std::vector<uint8_t>& out, uint32_t field, const std::string& s) {
    PutVarint(out, (static_cast<uint64_t>(field) << 3) | 2);
    PutVarint(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

uint64_t GetVarint(const std::vector<uint8_t>& b, size_t& i) {
    uint64_t r = 0;
    int shift = 0;
    while (i < b.size()) {
        const uint8_t c = b[i++];
        r |= static_cast<uint64_t>(c & 0x7F) << shift;
        if (!(c & 0x80)) {
            break;
        }
        shift += 7;
    }
    return r;
}

// Extract the string value of proto field `field` from a blob, or empty if absent.
std::string GetStrField(const std::vector<uint8_t>& b, uint32_t field) {
    size_t i = 0;
    while (i < b.size()) {
        const uint64_t tag = GetVarint(b, i);
        const uint32_t f = static_cast<uint32_t>(tag >> 3);
        const uint32_t wt = static_cast<uint32_t>(tag & 7);
        if (wt == 2) {
            const uint64_t len = GetVarint(b, i);
            if (i + len > b.size()) {
                break;
            }
            if (f == field) {
                return std::string(reinterpret_cast<const char*>(&b[i]), len);
            }
            i += len;
        } else if (wt == 0) {
            GetVarint(b, i);
        } else if (wt == 5) {
            i += 4;
        } else if (wt == 1) {
            i += 8;
        } else {
            break;
        }
    }
    return {};
}

// ---- framing ----

void PutLE(std::vector<uint8_t>& o, uint64_t v, int bytes) {
    for (int i = 0; i < bytes; ++i) {
        o.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
    }
}

uint16_t GetLE16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t GetLE32(const uint8_t* p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

bool SendAll(SockT s, const std::vector<uint8_t>& d) {
    size_t sent = 0;
    while (sent < d.size()) {
        const auto n = ::send(s, reinterpret_cast<const char*>(d.data() + sent),
                              static_cast<int>(d.size() - sent), 0);
        if (n <= 0) {
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool RecvN(SockT s, uint8_t* buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        const auto r = ::recv(s, reinterpret_cast<char*>(buf + got), static_cast<int>(n - got), 0);
        if (r <= 0) {
            return false;
        }
        got += static_cast<size_t>(r);
    }
    return true;
}

// Read one packet; fills type/cmd/payload. Payload is everything after the 15-byte header.
bool ReadPacket(SockT s, uint8_t& type, uint16_t& cmd, std::vector<uint8_t>& payload) {
    uint8_t hdr[15];
    if (!RecvN(s, hdr, sizeof(hdr))) {
        return false;
    }
    type = hdr[0];
    cmd = GetLE16(hdr + 1);
    const uint32_t total = GetLE32(hdr + 3);
    if (total < sizeof(hdr) || total > 0x800000) {
        return false;
    }
    payload.resize(total - sizeof(hdr));
    return payload.empty() || RecvN(s, payload.data(), payload.size());
}

std::vector<uint8_t> BuildPacket(uint16_t cmd, uint64_t id, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> out;
    out.push_back(0); // PacketType::Request
    PutLE(out, cmd, 2);
    PutLE(out, 15 + payload.size(), 4);
    PutLE(out, id, 8);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

SockT ConnectTo(const std::string& host, uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
        return kBadSock;
    }
    SockT s = kBadSock;
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
        s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s == kBadSock) {
            continue;
        }
        if (::connect(s, ai->ai_addr, static_cast<int>(ai->ai_addrlen)) == 0) {
            break;
        }
        CloseSock(s);
        s = kBadSock;
    }
    ::freeaddrinfo(res);
    return s;
}

std::pair<std::string, uint16_t> SplitHostPort(const std::string& hp) {
    const auto colon = hp.rfind(':');
    if (colon == std::string::npos || colon + 1 >= hp.size()) {
        return {hp, 31313};
    }
    const int p = std::atoi(hp.c_str() + colon + 1);
    return {hp.substr(0, colon), (p > 0 && p <= 65535) ? static_cast<uint16_t>(p) : uint16_t{31313}};
}

void LoginThread() {
    const std::string npid = Config::GetShadnetNpid();
    const auto [host, port] = SplitHostPort(Config::GetShadnetServer());
    const SockT s = ConnectTo(host, port);
    if (s == kBadSock) {
        LOG_ERROR(Lib_Http, "[GR2FORK auth] shadnet connect to {}:{} failed", host, port);
        g_login_result.store(LoginResult::FailConnect);
        return;
    }

    uint8_t type = 0;
    uint16_t cmd = 0;
    std::vector<uint8_t> pl;
    // The server speaks first with a ServerInfo packet (type 3).
    if (!ReadPacket(s, type, cmd, pl) || type != 3) {
        g_login_result.store(LoginResult::FailConnect);
        CloseSock(s);
        return;
    }

    std::vector<uint8_t> req;
    PutStrField(req, 1, npid);
    PutStrField(req, 2, Config::GetShadnetPassword());
    PutStrField(req, 4, std::string(Common::ElfInfo::Instance().GameSerial()));
    PutStrField(req, 5, std::string(Common::ElfInfo::Instance().Title()));
    std::vector<uint8_t> framed;
    PutLE(framed, req.size(), 4);
    framed.insert(framed.end(), req.begin(), req.end());
    if (!SendAll(s, BuildPacket(0, 1, framed))) { // cmd 0 = Login
        g_login_result.store(LoginResult::FailConnect);
        CloseSock(s);
        return;
    }

    for (int i = 0; i < 12; ++i) {
        if (!ReadPacket(s, type, cmd, pl)) {
            g_login_result.store(LoginResult::FailConnect);
            CloseSock(s);
            return;
        }
        if (type == 1 && cmd == 0) { // Reply / Login
            if (pl.empty() || pl[0] != 0) {
                const int e = pl.empty() ? 255 : pl[0];
                LOG_ERROR(Lib_Http, "[GR2FORK auth] shadnet login rejected (err={})", e);
                g_login_err.store(e);
                // shadnet ErrorType: 7 = invalid username, 8 = invalid password.
                g_login_result.store((e == 7 || e == 8) ? LoginResult::FailCreds
                                                        : LoginResult::FailOther);
                CloseSock(s);
                return;
            }
            SendAll(s, BuildPacket(39, 2, {})); // cmd 39 = GetToken
        }
        if (type == 1 && cmd == 39) { // Reply / GetToken
            if (pl.size() > 5 && pl[0] == 0) {
                const uint32_t len = GetLE32(pl.data() + 1);
                if (5 + len <= pl.size()) {
                    const std::vector<uint8_t> blob(pl.begin() + 5, pl.begin() + 5 + len);
                    const std::string tok = GetStrField(blob, 1);
                    if (!tok.empty()) {
                        {
                            std::lock_guard lk(g_state_mutex);
                            g_token = tok;
                            g_npid = npid;
                        }
                        g_authed.store(true);
                        g_login_result.store(LoginResult::Ok);
                        LOG_INFO(Lib_Http, "[GR2FORK auth] shadnet authenticated npid='{}'", npid);
                    }
                }
            }
            break;
        }
    }

    if (!g_authed.load()) {
        g_login_result.store(LoginResult::FailConnect);
        CloseSock(s);
        return;
    }
    // Hold the session open (draining any notifications) so the token stays valid for the run. The
    // captured token is retained even after the link drops; the restoration server is the authority
    // on whether it is still valid.
    std::vector<uint8_t> discard;
    while (ReadPacket(s, type, cmd, discard)) {
    }
    CloseSock(s);
}

} // namespace

bool IsOnlineEnabled() {
    // Online (and thus the shadnet login) is intended only with an Online ID configured AND both the
    // "connected to network" and "signed in to PSN" toggles on - the same gates the game's own online
    // path respects (guest HTTP checks isConnectedToNetwork; NP WebApi checks both).
    return !Config::GetShadnetNpid().empty() && Config::getIsConnectedToNetwork() &&
           Config::getPSNSignedIn();
}

std::string GetLoginErrorMessage() {
    switch (g_login_result.load()) {
    case LoginResult::FailConnect:
        return "Could not connect to shadNet.\n\nCheck your internet connection and the shadNet server "
               "address in the launcher, then relaunch.";
    case LoginResult::FailCreds:
        return "shadNet login failed: wrong Online ID or password.\n\nUpdate your credentials on the "
               "launcher's User tab (shadNet Account), then relaunch.";
    case LoginResult::FailOther:
        return "shadNet login failed (server error " + std::to_string(g_login_err.load()) +
               ").\n\nTry again in a moment.";
    default:
        return {};
    }
}

bool PreflightBlockingLogin() {
    if (!IsOnlineEnabled()) {
        return true; // online off; nothing to preflight
    }
    std::call_once(g_once, [] { std::thread(LoginThread).detach(); });
    // Wait for a definitive outcome (bounded ~12s to cover connect + login round-trips).
    for (int i = 0; i < 240 && g_login_result.load() == LoginResult::Pending; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (g_login_result.load() == LoginResult::Pending) {
        g_login_result.store(LoginResult::FailConnect); // timed out
    }
    return g_login_result.load() == LoginResult::Ok;
}

bool EnsureAuthed() {
    if (!IsOnlineEnabled()) {
        return false; // online off (no Online ID, or the network/PSN toggles are disabled)
    }
    std::call_once(g_once, [] { std::thread(LoginThread).detach(); });
    // Bounded wait (~2.5s) so the first online request already carries the token; a slow or absent
    // server just proceeds unauthenticated and the request retries later.
    for (int i = 0; i < 50 && !g_authed.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return g_authed.load();
}

std::string VerifiedNpid() {
    if (!g_authed.load()) {
        return {};
    }
    std::lock_guard lk(g_state_mutex);
    return g_npid;
}

std::string BearerToken() {
    if (!g_authed.load()) {
        return {};
    }
    std::lock_guard lk(g_state_mutex);
    return g_token;
}

std::string EffectiveOnlineId() {
    if (std::string npid = VerifiedNpid(); !npid.empty()) {
        return npid;
    }
    return Config::getUserName();
}

} // namespace GR2Fork::Auth
