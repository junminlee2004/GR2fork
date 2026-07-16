// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/np/np_error.h"
#include "core/libraries/np/np_web_api.h"

// GR2-NPWEB userProfile responder: the World-Rankings screen resolves each row's player via
// userProfile requests, and a 0-byte stub stalls the board. Requests forward to the local GR2
// server (httpHostOverride) as GET /npwebapi/<apiGroup><path>; no response/non-2xx yields 0 bytes.
#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>

#include <httplib.h>

#include "common/config.h"

#include <cstdint>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace Libraries::Np::NpWebApi {

namespace {

// Per-request response state keyed by the requestId from CreateRequest. SendRequest2 performs
// the host fetch and fills `body`; ReadData streams it out and returns 0 at EOF. The map
// tolerates overlapping requests.
struct NpWebResponse {
    std::string api_group;
    std::string path;
    std::string body;
    size_t offset = 0;
    bool fetched = false;
};

std::mutex g_np_web_mtx;
std::map<s64, NpWebResponse> g_np_web_requests;
s64 g_np_web_next_id = 0x4e500001; // keep the recognizable id prefix from the old fixed value

// GR2 own-name fix: the game-bundled NpToolkit2 PRX renders the World-Rankings OWN line from a
// singular /v1/users/<onlineId>/profile fetch, then hands sceNpWebApiUtilityParseNpId an EMPTY npId
// (it does not extract the flat-string npId the server sends in that form). We cannot patch the
// closed PRX, so remember the onlineId of the profile it just fetched and return it as the handle
// when ParseNpId is passed an empty npId -> the live player's own name renders instead of a garbage
// glyph. The challenger path (plural /profiles) passes a real base64 npId and never hits the fallback.
std::mutex g_last_profile_mtx;
std::string g_last_profile_online_id;

void CaptureProfileOnlineId(std::string_view path) {
    // match ".../users/<onlineId>/profile" (the singular userProfile form only)
    const auto u = path.find("/users/");
    if (u == std::string_view::npos) {
        return;
    }
    const auto start = u + 7; // strlen("/users/")
    const auto slash = path.find('/', start);
    if (slash == std::string_view::npos || path.compare(slash, 8, "/profile") != 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_last_profile_mtx);
    g_last_profile_online_id = std::string(path.substr(start, slash - start));
}

// GR2 challenge probe: the "Incoming Challenge" entry comes from the challenge subsystem (cat-0
// slot, item+0xa0=challenger name, item+0xb4=FNV("cm%02d")) on a PSN push event that a discarding
// stub loses. These handlers log the six argument registers raw (no deref) and return unique ids.
// TODO: capture + deliver a synthesized challenge push (same pump model np_manager uses).
std::atomic<s32> g_push_next_id{0x5001};

// Observed ABI: CreatePushEventFilter(userCtxId=RDI, pDataType=RSI, num=RDX);
// RegisterPushEventCallback(userCtxId=RDI, filterId=RSI, cbFunc=RDX, userArg=RCX). GR2 subscribes
// one filter (num=2) at boot; the libSceNpToolkit2 callback is captured for a synthesized push.
std::mutex g_push_cb_mtx;
u64 g_push_cb_func = 0;    // guest callback fn ptr (RDX at RegisterPushEventCallback)
u64 g_push_cb_userarg = 0; // its user-arg          (RCX)
s32 g_push_cb_filter = 0;  // filterId              (RSI)

// Percent-encode any byte outside printable ASCII (0x21..0x7e); structural URL chars pass
// through. The per-row userProfile path is built from an unset getscorelist entry field, so the
// {npid} segment is raw garbage bytes; unencoded control bytes make httplib fail outright.
std::string UrlEncodePath(const std::string& in) {
    static const char hexd[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(in.size());
    for (unsigned char c : in) {
        if (c >= 0x21 && c <= 0x7e) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(hexd[c >> 4]);
            out.push_back(hexd[c & 0xf]);
        }
    }
    return out;
}

bool HasNonPrintable(const std::string& s) {
    for (unsigned char c : s) {
        if (c < 0x20 || c > 0x7e) {
            return true;
        }
    }
    return false;
}

std::string ToHex(const std::string& s) {
    static const char hexd[] = "0123456789abcdef";
    std::string out;
    out.reserve(s.size() * 3);
    for (unsigned char c : s) {
        out.push_back(hexd[c >> 4]);
        out.push_back(hexd[c & 0xf]);
        out.push_back(' ');
    }
    return out;
}

// Hard gate for all NpWebApi network activity: the host-side forward runs only when both
// PSN-signed-in and network-connected, otherwise the 0-byte stub behavior applies. The gate must
// live here: nothing in the guest HTTP path blocks on them (g_isConnectedToNetwork is dead).
bool NpOnlineEnabled() {
    return Config::getPSNSignedIn() && Config::getIsConnectedToNetwork();
}


// TEMPORARY DIAGNOSTIC (remove once rankings name resolution is settled): the non-printable
// {npid} bytes (b0 0a ac 0b 08) reconstruct to 0x80bac0ab0, a guest heap pointer (guest VA ==
// host VA at base 0x800000000). Guard-read and dump it; the gate still returns 0 bytes.

// Pull the {npid} bytes out of "/v1/users/<npid>/profile?...". Returns the raw bytes
// between the "/v1/users/" prefix and the following '/'. Empty if the shape does not match.
std::string ExtractNpidFromPath(const std::string& path) {
    static const std::string prefix = "/v1/users/";
    const size_t a = path.find(prefix);
    if (a == std::string::npos) {
        return {};
    }
    const size_t start = a + prefix.size();
    size_t end = path.find('/', start);
    if (end == std::string::npos) {
        end = path.size();
    }
    return path.substr(start, end - start);
}

// Read up to maxlen bytes from host address `addr`, stopping at the first unmapped page so a
// bad/short pointer never faults. Returns the number of bytes copied into dst.
size_t SafeReadHostMax(uintptr_t addr, u8* dst, size_t maxlen) {
#ifdef _WIN32
    size_t total = 0;
    while (total < maxlen) {
        const uintptr_t cur = addr + total;
        const size_t to_page = 0x1000 - (cur & 0xfff);
        const size_t want = (maxlen - total < to_page) ? (maxlen - total) : to_page;
        SIZE_T got = 0;
        if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(cur), dst + total,
                               want, &got) ||
            got == 0) {
            break;
        }
        total += got;
        if (got < want) {
            break;
        }
    }
    return total;
#else
    const long pl = sysconf(_SC_PAGESIZE);
    const size_t ps = (pl > 0) ? static_cast<size_t>(pl) : 0x1000;
    size_t total = 0;
    while (total < maxlen) {
        const uintptr_t cur = addr + total;
        const uintptr_t page = cur & ~(static_cast<uintptr_t>(ps) - 1);
        // msync returns -1/ENOMEM when the page is not mapped -- the "is it readable" probe.
        if (msync(reinterpret_cast<void*>(page), ps, MS_ASYNC) != 0) {
            break;
        }
        const size_t to_page = ps - (cur & (ps - 1));
        const size_t want = (maxlen - total < to_page) ? (maxlen - total) : to_page;
        std::memcpy(dst + total, reinterpret_cast<const void*>(cur), want);
        total += want;
    }
    return total;
#endif
}

void ProbeNpidPointer(const std::string& path) {
    const std::string npid = ExtractNpidFromPath(path);
    if (npid.empty()) {
        LOG_INFO(Lib_NpWebApi, "[GR2-NPWEB-PTR] no /v1/users/<npid>/ in path -- skip probe");
        return;
    }
    // Reconstruct the little-endian integer the toolkit truncated at the first NUL.
    uintptr_t ptr = 0;
    for (size_t i = 0; i < npid.size() && i < 8; ++i) {
        ptr |= static_cast<uintptr_t>(static_cast<unsigned char>(npid[i])) << (8 * i);
    }
    const u64 base_off = (ptr >= 0x800000000ull) ? static_cast<u64>(ptr - 0x800000000ull) : 0ull;
    LOG_INFO(Lib_NpWebApi,
             "[GR2-NPWEB-PTR] npid raw={} ({} bytes) -> LE u64 = {:#018x} (guest base + {:#x})",
             ToHex(npid), npid.size(), static_cast<u64>(ptr), base_off);

    u8 buf[256];
    const size_t got = SafeReadHostMax(ptr, buf, sizeof(buf));
    if (got == 0) {
        LOG_INFO(Lib_NpWebApi,
                 "[GR2-NPWEB-PTR] target {:#018x} is UNMAPPED (page-probe failed) -- the npid is "
                 "NOT a readable pointer; treat as no-such-player (pursue a tolerated not-found, "
                 "not an id forward)",
                 static_cast<u64>(ptr));
        return;
    }

    LOG_INFO(Lib_NpWebApi, "[GR2-NPWEB-PTR] target {:#018x} mapped, dumping {} bytes:",
             static_cast<u64>(ptr), got);
    // 32 bytes/line so the logger never truncates the dump.
    for (size_t off = 0; off < got; off += 32) {
        std::string lh, la;
        for (size_t i = off; i < off + 32 && i < got; ++i) {
            lh += fmt::format("{:02x} ", buf[i]);
            la += (buf[i] >= 0x20 && buf[i] < 0x7f) ? static_cast<char>(buf[i]) : '.';
        }
        LOG_INFO(Lib_NpWebApi, "[GR2-NPWEB-PTR]   +{:04x}: {:<96} |{}|", static_cast<unsigned>(off),
                 lh, la);
    }

    // Heuristic readout: is there a printable, NUL-terminated id at the front?
    size_t run = 0;
    while (run < got && buf[run] >= 0x20 && buf[run] < 0x7f) {
        ++run;
    }
    const bool nul_terminated = (run < got && buf[run] == 0x00);
    if (run >= 3 && nul_terminated) {
        LOG_INFO(Lib_NpWebApi,
                 "[GR2-NPWEB-PTR] >>> RECOVERED candidate id at +0: '{}' (len {}) -- next step: "
                 "dereference the npid pointer and forward THIS id",
                 std::string(reinterpret_cast<const char*>(buf), run), run);
        return;
    }
    // Otherwise look for the longest printable run anywhere in the dump (id may sit at an
    // offset inside a struct the pointer references).
    size_t best_off = 0, best_len = 0, cur_run = 0, cur_off = 0;
    for (size_t i = 0; i < got; ++i) {
        if (buf[i] >= 0x20 && buf[i] < 0x7f) {
            if (cur_run == 0) {
                cur_off = i;
            }
            ++cur_run;
            if (cur_run > best_len) {
                best_len = cur_run;
                best_off = cur_off;
            }
        } else {
            cur_run = 0;
        }
    }
    if (best_len >= 4) {
        LOG_INFO(Lib_NpWebApi,
                 "[GR2-NPWEB-PTR] longest printable run: '{}' at +{:#x} (len {}) -- candidate id "
                 "field inside the referenced struct",
                 std::string(reinterpret_cast<const char*>(buf + best_off), best_len),
                 static_cast<unsigned>(best_off), best_len);
    } else {
        LOG_INFO(Lib_NpWebApi,
                 "[GR2-NPWEB-PTR] no printable id-like run found at the top level -- the target is "
                 "likely a struct of pointers; following them one level below");
    }

    // The target is not itself an id string: its leading qwords are guest-heap pointers, so a
    // real onlineId likely lives one level down. Follow the first non-null guest pointers one
    // level and dump their targets, bounded to a few slots. Still pure diagnostic.
    auto qword_at = [&](size_t o) -> u64 {
        u64 v = 0;
        for (size_t k = 0; k < 8 && o + k < got; ++k) {
            v |= static_cast<u64>(buf[o + k]) << (8 * k);
        }
        return v;
    };
    auto looks_guest = [](u64 v) -> bool { return v >= 0x800000000ull && v < 0x900000000ull; };

    size_t guest_ptrs = 0;
    for (size_t o = 0; o + 8 <= got && o < 64; o += 8) {
        if (looks_guest(qword_at(o))) {
            ++guest_ptrs;
        }
    }
    if (guest_ptrs < 4) {
        return; // not a pointer table -- the top-level dump above is the evidence
    }

    LOG_INFO(Lib_NpWebApi,
             "[GR2-NPWEB-PTR] target is an array of guest pointers ({} of the first 8 slots are "
             "0x8________); following the first non-null entries one level:",
             guest_ptrs);
    size_t followed = 0;
    for (size_t o = 0; o + 8 <= got && followed < 8; o += 8) {
        const u64 sub = qword_at(o);
        if (!looks_guest(sub)) {
            continue;
        }
        ++followed;
        u8 sb[96];
        const size_t sgot = SafeReadHostMax(static_cast<uintptr_t>(sub), sb, sizeof(sb));
        if (sgot == 0) {
            LOG_INFO(Lib_NpWebApi, "[GR2-NPWEB-PTR]   slot +{:#x} -> {:#018x} UNMAPPED",
                     static_cast<unsigned>(o), sub);
            continue;
        }
        std::string lh, la;
        for (size_t i = 0; i < sgot && i < 48; ++i) {
            lh += fmt::format("{:02x} ", sb[i]);
            la += (sb[i] >= 0x20 && sb[i] < 0x7f) ? static_cast<char>(sb[i]) : '.';
        }
        LOG_INFO(Lib_NpWebApi, "[GR2-NPWEB-PTR]   slot +{:#x} -> {:#018x}: {} |{}|",
                 static_cast<unsigned>(o), sub, lh, la);
        size_t b_off = 0, b_len = 0, c_run = 0, c_off = 0;
        for (size_t i = 0; i < sgot; ++i) {
            if (sb[i] >= 0x20 && sb[i] < 0x7f) {
                if (c_run == 0) {
                    c_off = i;
                }
                ++c_run;
                if (c_run > b_len) {
                    b_len = c_run;
                    b_off = c_off;
                }
            } else {
                c_run = 0;
            }
        }
        if (b_len >= 3) {
            LOG_INFO(Lib_NpWebApi,
                     "[GR2-NPWEB-PTR]     >>> printable run '{}' at +{:#x} (len {}) -- candidate id",
                     std::string(reinterpret_cast<const char*>(sb + b_off), b_len),
                     static_cast<unsigned>(b_off), b_len);
        }
    }
}

} // namespace

s32 PS4_SYSV_ABI sceNpWebApiCreateContext() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiCreatePushEventFilter(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    const s32 id = g_push_next_id++;
    LOG_ERROR(Lib_NpWebApi,
              "[GR2-CHALLENGE] CreatePushEventFilter raw a={:#x} b={:#x} c={:#x} d={:#x} e={:#x} "
              "f={:#x} -> filter={}",
              a, b, c, d, e, f, id);
    // b=pDataType array ptr, c=num. Dump the array's printable bytes (don't assume the
    // per-entry stride -- the '.'-gaps reveal it). GUARDED so a poison/garbage b or c can never
    // fault: only deref a plausibly-mapped pointer with a tiny, bounded span.
    if (b >= 0x10000ull && b < 0x10000000000ull && c > 0 && c <= 16) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(b);
        const u64 n = (c <= 8 ? c : 8) * 48; // bounded: <=384 bytes
        std::string ascii;
        ascii.reserve(n);
        for (u64 i = 0; i < n; ++i) {
            const unsigned char ch = p[i];
            ascii += (ch >= 0x20 && ch <= 0x7e) ? static_cast<char>(ch) : '.';
        }
        LOG_ERROR(Lib_NpWebApi, "[GR2-CHALLENGE]   dataType blob num={} ({} bytes) ascii='{}'", c, n,
                  ascii);
    }
    return id;
}

s32 PS4_SYSV_ABI sceNpWebApiCreateServicePushEventFilter(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    const s32 id = g_push_next_id++;
    LOG_ERROR(Lib_NpWebApi,
              "[GR2-CHALLENGE] CreateServicePushEventFilter raw a={:#x} b={:#x} c={:#x} d={:#x} "
              "e={:#x} f={:#x} -> filter={}",
              a, b, c, d, e, f, id);
    return id;
}

s32 PS4_SYSV_ABI sceNpWebApiDeletePushEventFilter() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiDeleteServicePushEventFilter() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiRegisterExtdPushEventCallback() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiRegisterNotificationCallback(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    const s32 id = g_push_next_id++;
    LOG_ERROR(Lib_NpWebApi,
              "[GR2-CHALLENGE] RegisterNotificationCallback raw a={:#x} b={:#x} c={:#x} d={:#x} "
              "e={:#x} f={:#x} -> cbId={}",
              a, b, c, d, e, f, id);
    return id;
}

s32 PS4_SYSV_ABI sceNpWebApiRegisterPushEventCallback(u64 a, u64 b, u64 c, u64 d, u64 e, u64 f) {
    const s32 id = g_push_next_id++;
    {
        // Confirmed ABI: b=filterId, c=cbFunc, d=userArg. Capture for the delivery delta.
        std::lock_guard<std::mutex> lk(g_push_cb_mtx);
        g_push_cb_filter = static_cast<s32>(b);
        g_push_cb_func = c;
        g_push_cb_userarg = d;
    }
    LOG_ERROR(Lib_NpWebApi,
              "[GR2-CHALLENGE] RegisterPushEventCallback raw a={:#x} b={:#x} c={:#x} d={:#x} e={:#x} "
              "f={:#x} -> CAPTURED cb={:#x} userArg={:#x} cbId={}",
              a, b, c, d, e, f, c, d, id);
    return id;
}

s32 PS4_SYSV_ABI sceNpWebApiRegisterServicePushEventCallback(u64 a, u64 b, u64 c, u64 d, u64 e,
                                                             u64 f) {
    const s32 id = g_push_next_id++;
    LOG_ERROR(Lib_NpWebApi,
              "[GR2-CHALLENGE] RegisterServicePushEventCallback raw a={:#x} b={:#x} c={:#x} d={:#x} "
              "e={:#x} f={:#x} -> cbId={}",
              a, b, c, d, e, f, id);
    return id;
}

s32 PS4_SYSV_ABI sceNpWebApiUnregisterNotificationCallback() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiUnregisterPushEventCallback() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiUnregisterServicePushEventCallback() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiAbortHandle() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiAbortRequest() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiAddHttpRequestHeader() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiAddMultipartPart() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiCheckTimeout() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiClearAllUnusedConnection() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiClearUnusedConnection() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiCreateContextA() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiCreateExtdPushEventFilter() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiCreateHandle() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiCreateMultipartRequest() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiCreateRequest(s32 handle, const char* apiGroup, const char* path,
                                          s32 method, const void* contentParameter,
                                          s64* requestId) {
    const std::string group = apiGroup ? apiGroup : "";
    const std::string req_path = path ? path : "";
    s64 id;
    {
        std::lock_guard<std::mutex> lk(g_np_web_mtx);
        id = g_np_web_next_id++;
        auto& slot = g_np_web_requests[id];
        slot.api_group = group;
        slot.path = req_path;
        slot.body.clear();
        slot.offset = 0;
        slot.fetched = false;
    }
    CaptureProfileOnlineId(req_path); // remember own-line onlineId for the ParseNpId empty-npId fallback
    LOG_ERROR(Lib_NpWebApi,
              "[GR2-NPWEB] CreateRequest handle={} apiGroup='{}' path='{}' method={} "
              "contentParam={} -> req={:#x}",
              handle, group.empty() ? "(null)" : group.c_str(),
              req_path.empty() ? "(null)" : req_path.c_str(), method, fmt::ptr(contentParameter),
              id);
    if (HasNonPrintable(req_path)) {
        // The path's {npid} segment is non-printable -- log the raw bytes so the exact npid
        // the game extracted from the getscorelist entry is visible (the printable log line
        // above truncates at the first control byte).
        LOG_ERROR(Lib_NpWebApi, "[GR2-NPWEB]   path RAW BYTES: {}", ToHex(req_path));
    }
    if (requestId) {
        *requestId = id;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiDeleteContext() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiDeleteExtdPushEventFilter() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiDeleteHandle() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiDeleteRequest(s64 requestId) {
    {
        std::lock_guard<std::mutex> lk(g_np_web_mtx);
        g_np_web_requests.erase(requestId);
    }
    LOG_ERROR(Lib_NpWebApi, "[GR2-NPWEB] DeleteRequest req={:#x} (freed)", requestId);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiGetConnectionStats() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiGetErrorCode() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiGetHttpResponseHeaderValue() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiGetHttpResponseHeaderValueLength() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiGetHttpStatusCode() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiGetMemoryPoolStats() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiInitialize() {
    LOG_ERROR(Lib_NpWebApi, "(DUMMY) called");
    static s32 id = 0;
    return ++id;
}

s32 PS4_SYSV_ABI sceNpWebApiInitializeForPresence() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiIntCreateCtxIndExtdPushEventFilter() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiIntCreateRequest() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiIntCreateServicePushEventFilter() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiIntInitialize() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiIntRegisterServicePushEventCallback() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiIntRegisterServicePushEventCallbackA() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiReadData(s64 requestId, void* data, u64 size) {
    std::lock_guard<std::mutex> lk(g_np_web_mtx);
    auto it = g_np_web_requests.find(requestId);
    if (it == g_np_web_requests.end() || it->second.body.empty()) {
        LOG_ERROR(Lib_NpWebApi, "[GR2-NPWEB] ReadData req={:#x} bufSize={} -> 0 bytes (no body)",
                  requestId, size);
        return 0; // 0 bytes => EOF; matches the old stub when there is nothing to return
    }
    auto& slot = it->second;
    const size_t remaining = slot.body.size() - slot.offset;
    const size_t n = remaining < size ? remaining : static_cast<size_t>(size);
    if (n > 0 && data) {
        std::memcpy(data, slot.body.data() + slot.offset, n);
        slot.offset += n;
    }
    LOG_ERROR(Lib_NpWebApi,
              "[GR2-NPWEB] ReadData req={:#x} bufSize={} -> {} bytes ({}/{} delivered)", requestId,
              size, n, slot.offset, slot.body.size());
    return static_cast<s32>(n); // bytes copied this call (0 once fully drained)
}

s32 PS4_SYSV_ABI sceNpWebApiRegisterExtdPushEventCallbackA() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiSendMultipartRequest() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiSendMultipartRequest2() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiSendRequest() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiSendRequest2(s64 requestId, const void* data, u64 dataSize,
                                         void* respInfoOption) {
    LOG_ERROR(Lib_NpWebApi, "[GR2-NPWEB] SendRequest2 req={:#x} dataSize={} respInfo={}", requestId,
              dataSize, fmt::ptr(respInfoOption));

    if (!NpOnlineEnabled()) {
        // PSN-signed-in and/or network-connected toggle is off -> behave as the inert stub:
        // no forward, leave the response body empty, ReadData returns 0. Offline play unaffected.
        LOG_ERROR(Lib_NpWebApi,
                  "[GR2-NPWEB]   gated off (isPSNSignedIn={} isConnectedToNetwork={}) -- not "
                  "forwarding; returning 0 bytes (stub)",
                  Config::getPSNSignedIn(), Config::getIsConnectedToNetwork());
        return ORBIS_OK;
    }
    if (data && dataSize > 0) {
        const u8* p = static_cast<const u8*>(data);
        const u64 n = dataSize < 512 ? dataSize : 512;
        std::string hex, asc;
        for (u64 i = 0; i < n; ++i) {
            hex += fmt::format("{:02x} ", p[i]);
            asc += (p[i] >= 0x20 && p[i] < 0x7f) ? static_cast<char>(p[i]) : '.';
        }
        LOG_ERROR(Lib_NpWebApi, "[GR2-NPWEB]   body({} bytes){}: {}", dataSize,
                  dataSize > n ? " [first 512]" : "", hex);
        LOG_ERROR(Lib_NpWebApi, "[GR2-NPWEB]   ascii: {}", asc);
    }

    // Look up the request captured by CreateRequest and forward it to the local server.
    std::string group, path;
    {
        std::lock_guard<std::mutex> lk(g_np_web_mtx);
        auto it = g_np_web_requests.find(requestId);
        if (it == g_np_web_requests.end()) {
            LOG_ERROR(Lib_NpWebApi,
                      "[GR2-NPWEB]   req={:#x} unknown (no matching CreateRequest) -- skip forward",
                      requestId);
            return ORBIS_OK;
        }
        group = it->second.api_group;
        path = it->second.path;
    }

    // Do not forward a non-printable {npid}: row names are board-sourced (getscorelist tag 0x20),
    // so this is the player's own VS panel whose {npid} (0x80bac0ab0) is a guest vtable table; any
    // profile body is written back through that pointer, corrupting the heap; 0 bytes = not-found.
    if (HasNonPrintable(path)) {
        LOG_ERROR(Lib_NpWebApi,
                  "[GR2-NPWEB]   non-printable npid in path -- NOT forwarding (garbage/placeholder "
                  "entry, would crash the guest heap); returning 0 bytes. raw npid: {}",
                  ToHex(path));
        ProbeNpidPointer(path);
        return ORBIS_OK;
    }

    // Forwarded URL: /npwebapi/<apiGroup><path>. The local server matches the /npwebapi/
    // prefix and answers userProfile .../profile (and .../friendList) with JSON. Reuse the
    // SAME host/port capone uses (httpHostOverride), so no extra setup is needed.
    const std::string raw_path = "/npwebapi/" + group + path;
    const std::string fwd_path = UrlEncodePath(raw_path); // identity for printable paths
    const std::string host = Config::GetHttpHostOverride();
    const int port = Config::GetHttpHostOverridePort();

    httplib::Client cli(host, port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(5, 0);
    cli.set_write_timeout(5, 0);

    httplib::Result res;
    if (data && dataSize > 0) {
        res = cli.Post(fwd_path, static_cast<const char*>(data), static_cast<size_t>(dataSize),
                       "application/json");
    } else {
        res = cli.Get(fwd_path);
    }

    std::string body;
    if (!res) {
        LOG_ERROR(Lib_NpWebApi,
                  "[GR2-NPWEB]   forward host='{}' port={} path='{}' NO response (httplib error "
                  "{}) -- server down / endpoint missing? returning 0 bytes",
                  host, port, fwd_path, static_cast<int>(res.error()));
    } else if (res->status / 100 != 2) {
        LOG_ERROR(Lib_NpWebApi,
                  "[GR2-NPWEB]   forward path='{}' status={} -- returning 0 bytes (empty profile)",
                  fwd_path, res->status);
    } else {
        body = res->body;
        LOG_ERROR(Lib_NpWebApi, "[GR2-NPWEB]   forward path='{}' status={} body_size={}", fwd_path,
                  res->status, body.size());
    }

    {
        std::lock_guard<std::mutex> lk(g_np_web_mtx);
        auto it = g_np_web_requests.find(requestId);
        if (it != g_np_web_requests.end()) {
            it->second.body = std::move(body);
            it->second.offset = 0;
            it->second.fetched = true;
        }
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiSetHandleTimeout() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiSetMaxConnection() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiSetMultipartContentType() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiSetRequestTimeout() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiTerminate() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiUnregisterExtdPushEventCallback() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

static std::string DecodeBase64(std::string_view in) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z')
            return c - 'A';
        if (c >= 'a' && c <= 'z')
            return c - 'a' + 26;
        if (c >= '0' && c <= '9')
            return c - '0' + 52;
        if (c == '+')
            return 62;
        if (c == '/')
            return 63;
        return -1;
    };
    std::string out;
    int buf = 0, bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\0')
            break;
        const int v = val(c);
        if (v < 0)
            continue; // skip whitespace / invalid chars
        buf = (buf << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((buf >> bits) & 0xFF));
        }
    }
    return out;
}

s32 PS4_SYSV_ABI sceNpWebApiUtilityParseNpId(const char* pJsonNpId,
                                             Libraries::Np::OrbisNpId* pNpId) {
    // Temporary input probe: capture the exact bytes the game passes as pJsonNpId. The input can
    // be base64 or binary and non-printable, so log bounded hex alongside the quoted string.
    {
        std::string in_str, in_hex;
        size_t in_len = 0;
        if (pJsonNpId == nullptr) {
            in_str = "(null)";
        } else {
            in_len = strnlen(pJsonNpId, 256);
            in_str = std::string(pJsonNpId, in_len);
            const size_t hex_n = in_len < 48 ? in_len : 48;
            in_hex.reserve(hex_n * 3);
            for (size_t i = 0; i < hex_n; ++i) {
                in_hex += fmt::format("{:02x} ", static_cast<unsigned char>(pJsonNpId[i]));
            }
        }
        LOG_INFO(Lib_NpWebApi, "ParseNpId input pJsonNpId='{}' strnlen={} hex: {}", in_str, in_len,
                 in_hex);
    }
    // Zero the output up front so an early return or empty parse never leaves the
    // caller with an uninitialized OrbisNpId.
    if (pNpId != nullptr) {
        std::memset(pNpId, 0, sizeof(Libraries::Np::OrbisNpId));
    }
    if (pJsonNpId == nullptr) {
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    // The serialized npId decodes to "<handle>@<seg>[/<seg>][.<seg>]": the part
    // before '@' is the online-id handle (<=16 bytes), and the segments after it,
    // with '/' and '.' separators removed, concatenate into the 8-byte opt.
    const std::string decoded = DecodeBase64(pJsonNpId);
    std::string handle;
    std::string opt;
    const size_t at = decoded.find('@');
    if (at == std::string::npos) {
        handle = decoded;
    } else {
        handle = decoded.substr(0, at);
        for (size_t i = at + 1; i < decoded.size(); ++i) {
            const char c = decoded[i];
            if (c != '/' && c != '.') {
                opt.push_back(c);
            }
        }
    }
    if (handle.empty()) {
        // GR2 own-name fix: NpToolkit2 renders the World-Rankings OWN line from the singular
        // /users/<id>/profile fetch, then calls this with an EMPTY npId (it does not extract the
        // flat-string npId the server sends in that form). Fall back to the onlineId of the profile
        // it just fetched (see CaptureProfileOnlineId) so the live player's own name renders instead
        // of a garbage glyph. A non-empty npId (the challenger's plural /profiles path) decodes above
        // and never reaches this fallback.
        std::lock_guard<std::mutex> lk(g_last_profile_mtx);
        if (!g_last_profile_online_id.empty()) {
            handle = g_last_profile_online_id;
            opt.clear();
            LOG_INFO(Lib_NpWebApi, "ParseNpId empty npId -> fallback to profile onlineId '{}'",
                     handle);
        }
    }
    if (handle.empty()) {
        // An empty decoded handle is not a usable identity; return an error with the output
        // left zeroed (valid-handle flag unset) so the caller does not treat it as valid and
        // write through a null identity pointer.
        return ORBIS_NP_ERROR_INVALID_ARGUMENT;
    }
    if (pNpId != nullptr) {
        std::memcpy(pNpId->handle.data, handle.data(),
                    std::min<size_t>(handle.size(), ORBIS_NP_ONLINEID_MAX_LENGTH));
        std::memcpy(pNpId->opt, opt.data(), std::min<size_t>(opt.size(), sizeof(pNpId->opt)));
        pNpId->reserved[0] = 0x01; // valid-handle flag the PRX sets
    }
    LOG_INFO(Lib_NpWebApi, "parsed npId -> handle='{}' opt='{}'", handle, opt);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpWebApiVshInitialize() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_064C4ED1EDBEB9E8() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_0783955D4E9563DA() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_1A6D77F3FD8323A8() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_1E0693A26FE0F954() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_24A9B5F1D77000CF() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_24AAA6F50E4C2361() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_24D8853D6B47FC79() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_279B3E9C7C4A9DC5() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_28461E29E9F8D697() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_3C29624704FAB9E0() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_3F027804ED2EC11E() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_4066C94E782997CD() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_47C85356815DBE90() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_4FCE8065437E3B87() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_536280BE3DABB521() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_57A0E1BC724219F3() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_5819749C040B6637() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_6198D0C825E86319() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_61F2B9E8AB093743() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_6BC388E6113F0D44() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_7500F0C4F8DC2D16() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_75A03814C7E9039F() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_789D6026C521416E() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_7DED63D06399EFFF() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_7E55A2DCC03D395A() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_7E6C8F9FB86967F4() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_7F04B7D4A7D41E80() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_8E167252DFA5C957() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_95D0046E504E3B09() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_97284BFDA4F18FDF() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_99E32C1F4737EAB4() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_9CFF661EA0BCBF83() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_9EB0E1F467AC3B29() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_A2318FE6FBABFAA3() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_BA07A2E1BF7B3971() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_BD0803EEE0CC29A0() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_BE6F4E5524BB135F() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_C0D490EB481EA4D0() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_C175D392CA6D084A() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_CD0136AF165D2F2F() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_D1C0ADB7B52FEAB5() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_E324765D18EE4D12() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_E789F980D907B653() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI Func_F9A32E8685627436() {
    LOG_ERROR(Lib_NpWebApi, "(STUBBED) called");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("x1Y7yiYSk7c", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiCreateContext);
    LIB_FUNCTION("y5Ta5JCzQHY", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiCreatePushEventFilter);
    LIB_FUNCTION("sIFx734+xys", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiCreateServicePushEventFilter);
    LIB_FUNCTION("zE+R6Rcx3W0", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiDeletePushEventFilter);
    LIB_FUNCTION("PfQ+f6ws764", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiDeleteServicePushEventFilter);
    LIB_FUNCTION("vrM02A5Gy1M", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiRegisterExtdPushEventCallback);
    LIB_FUNCTION("HVgWmGIOKdk", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiRegisterNotificationCallback);
    LIB_FUNCTION("PfSTDCgNMgc", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiRegisterPushEventCallback);
    LIB_FUNCTION("kJQJE0uKm5w", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiRegisterServicePushEventCallback);
    LIB_FUNCTION("wjYEvo4xbcA", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiUnregisterNotificationCallback);
    LIB_FUNCTION("qK4o2656W4w", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiUnregisterPushEventCallback);
    LIB_FUNCTION("2edrkr0c-wg", "libSceNpWebApiCompat", 1, "libSceNpWebApi",
                 sceNpWebApiUnregisterServicePushEventCallback);
    LIB_FUNCTION("WKcm4PeyJww", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiAbortHandle);
    LIB_FUNCTION("JzhYTP2fG18", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiAbortRequest);
    LIB_FUNCTION("joRjtRXTFoc", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiAddHttpRequestHeader);
    LIB_FUNCTION("19KgfJXgM+U", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiAddMultipartPart);
    LIB_FUNCTION("gVNNyxf-1Sg", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiCheckTimeout);
    LIB_FUNCTION("KQIkDGf80PQ", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiClearAllUnusedConnection);
    LIB_FUNCTION("f-pgaNSd1zc", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiClearUnusedConnection);
    LIB_FUNCTION("x1Y7yiYSk7c", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiCreateContext);
    LIB_FUNCTION("zk6c65xoyO0", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiCreateContextA);
    LIB_FUNCTION("M2BUB+DNEGE", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiCreateExtdPushEventFilter);
    LIB_FUNCTION("79M-JqvvGo0", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiCreateHandle);
    LIB_FUNCTION("KBxgeNpoRIQ", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiCreateMultipartRequest);
    LIB_FUNCTION("y5Ta5JCzQHY", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiCreatePushEventFilter);
    LIB_FUNCTION("rdgs5Z1MyFw", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiCreateRequest);
    LIB_FUNCTION("sIFx734+xys", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiCreateServicePushEventFilter);
    LIB_FUNCTION("XUjdsSTTZ3U", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiDeleteContext);
    LIB_FUNCTION("pfaJtb7SQ80", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiDeleteExtdPushEventFilter);
    LIB_FUNCTION("5Mn7TYwpl30", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiDeleteHandle);
    LIB_FUNCTION("zE+R6Rcx3W0", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiDeletePushEventFilter);
    LIB_FUNCTION("noQgleu+KLE", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiDeleteRequest);
    LIB_FUNCTION("PfQ+f6ws764", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiDeleteServicePushEventFilter);
    LIB_FUNCTION("UJ8H+7kVQUE", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiGetConnectionStats);
    LIB_FUNCTION("2qSZ0DgwTsc", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiGetErrorCode);
    LIB_FUNCTION("VwJ5L0Higg0", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiGetHttpResponseHeaderValue);
    LIB_FUNCTION("743ZzEBzlV8", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiGetHttpResponseHeaderValueLength);
    LIB_FUNCTION("k210oKgP80Y", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiGetHttpStatusCode);
    LIB_FUNCTION("3OnubUs02UM", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiGetMemoryPoolStats);
    LIB_FUNCTION("G3AnLNdRBjE", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiInitialize);
    LIB_FUNCTION("FkuwsD64zoQ", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiInitializeForPresence);
    LIB_FUNCTION("c1pKoztonB8", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiIntCreateCtxIndExtdPushEventFilter);
    LIB_FUNCTION("N2Jbx4tIaQ4", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiIntCreateRequest);
    LIB_FUNCTION("TZSep4xB4EY", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiIntCreateServicePushEventFilter);
    LIB_FUNCTION("8Vjplhyyc44", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiIntInitialize);
    LIB_FUNCTION("VjVukb2EWPc", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiIntRegisterServicePushEventCallback);
    LIB_FUNCTION("sfq23ZVHVEw", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiIntRegisterServicePushEventCallbackA);
    LIB_FUNCTION("CQtPRSF6Ds8", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiReadData);
    LIB_FUNCTION("vrM02A5Gy1M", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiRegisterExtdPushEventCallback);
    LIB_FUNCTION("jhXKGQJ4egI", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiRegisterExtdPushEventCallbackA);
    LIB_FUNCTION("HVgWmGIOKdk", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiRegisterNotificationCallback);
    LIB_FUNCTION("PfSTDCgNMgc", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiRegisterPushEventCallback);
    LIB_FUNCTION("kJQJE0uKm5w", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiRegisterServicePushEventCallback);
    LIB_FUNCTION("KCItz6QkeGs", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiSendMultipartRequest);
    LIB_FUNCTION("DsPOTEvSe7M", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiSendMultipartRequest2);
    LIB_FUNCTION("kVbL4hL3K7w", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiSendRequest);
    LIB_FUNCTION("KjNeZ-29ysQ", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiSendRequest2);
    LIB_FUNCTION("6g6q-g1i4XU", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiSetHandleTimeout);
    LIB_FUNCTION("gRiilVCvfAI", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiSetMaxConnection);
    LIB_FUNCTION("i0dr6grIZyc", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiSetMultipartContentType);
    LIB_FUNCTION("qWcbJkBj1Lg", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiSetRequestTimeout);
    LIB_FUNCTION("asz3TtIqGF8", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiTerminate);
    LIB_FUNCTION("PqCY25FMzPs", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiUnregisterExtdPushEventCallback);
    LIB_FUNCTION("wjYEvo4xbcA", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiUnregisterNotificationCallback);
    LIB_FUNCTION("qK4o2656W4w", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiUnregisterPushEventCallback);
    LIB_FUNCTION("2edrkr0c-wg", "libSceNpWebApi", 1, "libSceNpWebApi",
                 sceNpWebApiUnregisterServicePushEventCallback);
    LIB_FUNCTION("or0e885BlXo", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiUtilityParseNpId);
    LIB_FUNCTION("uRsskUhAfnM", "libSceNpWebApi", 1, "libSceNpWebApi", sceNpWebApiVshInitialize);
    LIB_FUNCTION("BkxO0e2+ueg", "libSceNpWebApi", 1, "libSceNpWebApi", Func_064C4ED1EDBEB9E8);
    LIB_FUNCTION("B4OVXU6VY9o", "libSceNpWebApi", 1, "libSceNpWebApi", Func_0783955D4E9563DA);
    LIB_FUNCTION("Gm138-2DI6g", "libSceNpWebApi", 1, "libSceNpWebApi", Func_1A6D77F3FD8323A8);
    LIB_FUNCTION("HgaTom-g+VQ", "libSceNpWebApi", 1, "libSceNpWebApi", Func_1E0693A26FE0F954);
    LIB_FUNCTION("JKm18ddwAM8", "libSceNpWebApi", 1, "libSceNpWebApi", Func_24A9B5F1D77000CF);
    LIB_FUNCTION("JKqm9Q5MI2E", "libSceNpWebApi", 1, "libSceNpWebApi", Func_24AAA6F50E4C2361);
    LIB_FUNCTION("JNiFPWtH-Hk", "libSceNpWebApi", 1, "libSceNpWebApi", Func_24D8853D6B47FC79);
    LIB_FUNCTION("J5s+nHxKncU", "libSceNpWebApi", 1, "libSceNpWebApi", Func_279B3E9C7C4A9DC5);
    LIB_FUNCTION("KEYeKen41pc", "libSceNpWebApi", 1, "libSceNpWebApi", Func_28461E29E9F8D697);
    LIB_FUNCTION("PCliRwT6ueA", "libSceNpWebApi", 1, "libSceNpWebApi", Func_3C29624704FAB9E0);
    LIB_FUNCTION("PwJ4BO0uwR4", "libSceNpWebApi", 1, "libSceNpWebApi", Func_3F027804ED2EC11E);
    LIB_FUNCTION("QGbJTngpl80", "libSceNpWebApi", 1, "libSceNpWebApi", Func_4066C94E782997CD);
    LIB_FUNCTION("R8hTVoFdvpA", "libSceNpWebApi", 1, "libSceNpWebApi", Func_47C85356815DBE90);
    LIB_FUNCTION("T86AZUN+O4c", "libSceNpWebApi", 1, "libSceNpWebApi", Func_4FCE8065437E3B87);
    LIB_FUNCTION("U2KAvj2rtSE", "libSceNpWebApi", 1, "libSceNpWebApi", Func_536280BE3DABB521);
    LIB_FUNCTION("V6DhvHJCGfM", "libSceNpWebApi", 1, "libSceNpWebApi", Func_57A0E1BC724219F3);
    LIB_FUNCTION("WBl0nAQLZjc", "libSceNpWebApi", 1, "libSceNpWebApi", Func_5819749C040B6637);
    LIB_FUNCTION("YZjQyCXoYxk", "libSceNpWebApi", 1, "libSceNpWebApi", Func_6198D0C825E86319);
    LIB_FUNCTION("YfK56KsJN0M", "libSceNpWebApi", 1, "libSceNpWebApi", Func_61F2B9E8AB093743);
    LIB_FUNCTION("a8OI5hE-DUQ", "libSceNpWebApi", 1, "libSceNpWebApi", Func_6BC388E6113F0D44);
    LIB_FUNCTION("dQDwxPjcLRY", "libSceNpWebApi", 1, "libSceNpWebApi", Func_7500F0C4F8DC2D16);
    LIB_FUNCTION("daA4FMfpA58", "libSceNpWebApi", 1, "libSceNpWebApi", Func_75A03814C7E9039F);
    LIB_FUNCTION("eJ1gJsUhQW4", "libSceNpWebApi", 1, "libSceNpWebApi", Func_789D6026C521416E);
    LIB_FUNCTION("fe1j0GOZ7-8", "libSceNpWebApi", 1, "libSceNpWebApi", Func_7DED63D06399EFFF);
    LIB_FUNCTION("flWi3MA9OVo", "libSceNpWebApi", 1, "libSceNpWebApi", Func_7E55A2DCC03D395A);
    LIB_FUNCTION("fmyPn7hpZ-Q", "libSceNpWebApi", 1, "libSceNpWebApi", Func_7E6C8F9FB86967F4);
    LIB_FUNCTION("fwS31KfUHoA", "libSceNpWebApi", 1, "libSceNpWebApi", Func_7F04B7D4A7D41E80);
    LIB_FUNCTION("jhZyUt+lyVc", "libSceNpWebApi", 1, "libSceNpWebApi", Func_8E167252DFA5C957);
    LIB_FUNCTION("ldAEblBOOwk", "libSceNpWebApi", 1, "libSceNpWebApi", Func_95D0046E504E3B09);
    LIB_FUNCTION("lyhL-aTxj98", "libSceNpWebApi", 1, "libSceNpWebApi", Func_97284BFDA4F18FDF);
    LIB_FUNCTION("meMsH0c36rQ", "libSceNpWebApi", 1, "libSceNpWebApi", Func_99E32C1F4737EAB4);
    LIB_FUNCTION("nP9mHqC8v4M", "libSceNpWebApi", 1, "libSceNpWebApi", Func_9CFF661EA0BCBF83);
    LIB_FUNCTION("nrDh9GesOyk", "libSceNpWebApi", 1, "libSceNpWebApi", Func_9EB0E1F467AC3B29);
    LIB_FUNCTION("ojGP5vur+qM", "libSceNpWebApi", 1, "libSceNpWebApi", Func_A2318FE6FBABFAA3);
    LIB_FUNCTION("ugei4b97OXE", "libSceNpWebApi", 1, "libSceNpWebApi", Func_BA07A2E1BF7B3971);
    LIB_FUNCTION("vQgD7uDMKaA", "libSceNpWebApi", 1, "libSceNpWebApi", Func_BD0803EEE0CC29A0);
    LIB_FUNCTION("vm9OVSS7E18", "libSceNpWebApi", 1, "libSceNpWebApi", Func_BE6F4E5524BB135F);
    LIB_FUNCTION("wNSQ60gepNA", "libSceNpWebApi", 1, "libSceNpWebApi", Func_C0D490EB481EA4D0);
    LIB_FUNCTION("wXXTksptCEo", "libSceNpWebApi", 1, "libSceNpWebApi", Func_C175D392CA6D084A);
    LIB_FUNCTION("zQE2rxZdLy8", "libSceNpWebApi", 1, "libSceNpWebApi", Func_CD0136AF165D2F2F);
    LIB_FUNCTION("0cCtt7Uv6rU", "libSceNpWebApi", 1, "libSceNpWebApi", Func_D1C0ADB7B52FEAB5);
    LIB_FUNCTION("4yR2XRjuTRI", "libSceNpWebApi", 1, "libSceNpWebApi", Func_E324765D18EE4D12);
    LIB_FUNCTION("54n5gNkHtlM", "libSceNpWebApi", 1, "libSceNpWebApi", Func_E789F980D907B653);
    LIB_FUNCTION("+aMuhoVidDY", "libSceNpWebApi", 1, "libSceNpWebApi", Func_F9A32E8685627436);
};

} // namespace Libraries::Np::NpWebApi