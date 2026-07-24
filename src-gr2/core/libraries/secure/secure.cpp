// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/secure/secure.h"

namespace Libraries::LibSecure {

// A plausible guest pointer sits in the canonical low half at or above 4 GiB: guest heap, stack,
// and module mappings all live above that line, while unframed calls pass scalar garbage below it
// (sizes and flags - observed 0x1, 0x50, 0x9c) and poison fills above it (0xDEADBEEF........).
// Rejecting both cheaply, without touching the memory manager, keeps every deref and store in
// this file off unmapped pages - a store to a low scalar lands outside the null window on hosts
// that reserve guest space with placeholder pages, where the fault absorber cannot claim it.
static inline bool PlausiblePtr(const void* p) {
    const auto v = reinterpret_cast<std::uintptr_t>(p);
    return v >= 0x100000000ull && (v >> 47) == 0;
}

s32 PS4_SYSV_ABI sceLibSecureCryptographyDecrypt(void* ctx, void* dst, u64 dst_size,
                                                 const void* src, u64 src_size, u64* processed) {
    // Plaintext passthrough: copy min(dst_size, src_size) so the server's plaintext TLV survives
    // the game's "decrypt" step. The ARC decode entry FUN_011972A0 also calls this speculatively
    // before a response body is framed - e.g. a poison src (0xDEADBEEF........) with a GB-range
    // dst_size, where the memcpy faults and the null-page absorber spins (boot hang). Legitimate
    // rkg* TLV bodies are KB-scale, so implausible pointers or lengths become a no-op that
    // returns OK and touches no memory (the empty-body case the game already tolerates).
    constexpr u64 kMaxPayload = 64ull * 1024 * 1024; // 64 MiB; any real rkg TLV is far smaller
    const u64 n = std::min(dst_size, src_size);
    const bool ok = PlausiblePtr(dst) && PlausiblePtr(src) && n > 0 && n <= kMaxPayload;

    LOG_DEBUG(Lib_Ssl,
             "sceLibSecureCryptographyDecrypt: ctx={:#x} dst={:#x} dst_size={} src={:#x} "
             "src_size={} processed={:#x} copy={} {}",
             reinterpret_cast<std::uintptr_t>(ctx), reinterpret_cast<std::uintptr_t>(dst),
             dst_size, reinterpret_cast<std::uintptr_t>(src), src_size,
             reinterpret_cast<std::uintptr_t>(processed), n,
             ok ? "(passthrough)" : "(SKIPPED: implausible/unframed call)");

    // A skipped call did no work and must write NOTHING. The 3-arg in-place ARC decrypt form
    // (FUN_014ff120) leaves this 6-arg signature's "processed" slot as a leftover register that
    // can hold a live guest stack address (observed local-0x60, i.e. inside the caller's own
    // frame). A PlausiblePtr floor rejects scalar garbage (0x1, 0x50) but not a stack pointer, so
    // writing 0 through it corrupts the caller's frame - nulling a saved register that becomes the
    // getuploadinfo TLV-walk cursor, crashing the treasure-photo upload. Only the real streaming
    // form (ok == true, an actual copy) is trusted enough to report a count.
    if (ok) {
        std::memcpy(dst, src, n);
        if (PlausiblePtr(processed)) {
            *processed = n;
        }
    }
    return ORBIS_OK;
}

// The eboot's ARC crypto uses the 3-arg in-place form fn(ctxA, ArcCryptoBuf* {data,size}, ctxB)
// for both decrypt (FUN_014ff120) and encrypt (FUN_014ff0d0). The decrypt HLE above keeps its
// 6-arg streaming signature (its misread args never touch the in-place data buffer, which
// already holds plaintext); encrypt uses the precise 3-arg form to no-op cleanly and read the
// plaintext for the capture log.
struct ArcCryptoBuf {
    void* data;
    u64 size;
};

s32 PS4_SYSV_ABI sceLibSecureCryptographyEncrypt([[maybe_unused]] void* ctx_a, ArcCryptoBuf* buf,
                                                 [[maybe_unused]] void* ctx_b) {
    // No-op encrypt: the request buffer stays plaintext so GR2's ARC request 'p' blob reaches the
    // local restoration server readable. Padding lands after the TLV's 0x8010 terminator and the
    // x-k signature becomes base64(plaintext) - the server ignores both. All 30 callers of this
    // NID are capone op encoders; no save-data or trophy path uses libSceSecure encrypt. Also
    // logs the bounded plaintext (board id tag 0x4140, score 0x4500, category encoding).
    constexpr u64 kCap = 64ull * 1024 * 1024; // any real ARC request is KB-scale
    if (buf != nullptr && PlausiblePtr(buf) && PlausiblePtr(buf->data) && buf->size > 0 &&
        buf->size <= kCap) {
        const auto* p = static_cast<const u8*>(buf->data);
        const u64 n = std::min<u64>(buf->size, 256);
        static const char hd[] = "0123456789abcdef";
        std::string hex, asc;
        hex.reserve(n * 3);
        asc.reserve(n);
        for (u64 i = 0; i < n; ++i) {
            hex.push_back(hd[p[i] >> 4]);
            hex.push_back(hd[p[i] & 0xf]);
            hex.push_back(' ');
            asc.push_back((p[i] >= 0x20 && p[i] < 0x7f) ? static_cast<char>(p[i]) : '.');
        }
        LOG_DEBUG(Lib_Ssl,
                 "sceLibSecureCryptographyEncrypt: NO-OP plaintext-passthrough, size={} "
                 "first {}B hex: {} | ascii: {}",
                 buf->size, n, hex, asc);
    } else {
        LOG_DEBUG(Lib_Ssl, "sceLibSecureCryptographyEncrypt: NO-OP (no/implausible buffer {:#x})",
                 reinterpret_cast<std::uintptr_t>(buf));
    }
    return ORBIS_OK; // leave buf->data untouched -> plaintext survives
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    // sceLibSecureCryptographyDecrypt, NID hMYgMP-Vuno. The game bundles libSceSecure.prx while
    // the firmware module is libSceLibSecure; symbol resolution matches the full
    // nid#library#version#module#type string, so registering under both names is harmless - only
    // the name the eboot imports is looked up. A "Linker: Stub resolved ..." boot line means
    // neither name matched and the registration must use the lib/mod pair it prints.
    LIB_FUNCTION("hMYgMP-Vuno", "libSceLibSecure", 1, "libSceLibSecure",
                 sceLibSecureCryptographyDecrypt);
    LIB_FUNCTION("hMYgMP-Vuno", "libSceSecure", 1, "libSceSecure",
                 sceLibSecureCryptographyDecrypt);

    // sceLibSecureCryptographyEncrypt, NID aEoi0u2FOiQ. NO-OP passthrough: makes the ARC request
    // 'p' blob plaintext so the server can read board id / score / category (and logs it).
    // Registered under both candidate library names, same as decrypt.
    LIB_FUNCTION("aEoi0u2FOiQ", "libSceLibSecure", 1, "libSceLibSecure",
                 sceLibSecureCryptographyEncrypt);
    LIB_FUNCTION("aEoi0u2FOiQ", "libSceSecure", 1, "libSceSecure",
                 sceLibSecureCryptographyEncrypt);
}

} // namespace Libraries::LibSecure
