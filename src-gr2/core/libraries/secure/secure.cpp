// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/secure/secure.h"

namespace Libraries::LibSecure {

s32 PS4_SYSV_ABI sceLibSecureCryptographyDecrypt(void* ctx, void* dst, u64 dst_size,
                                                 const void* src, u64 src_size, u64* processed) {
    // Plaintext passthrough: copy the smaller of the two buffer sizes from src to dst so the
    // server's plaintext TLV survives the game's "decrypt" step untouched. Defensive on every
    // pointer/size so a malformed call can never corrupt guest memory.
    const u64 n = std::min(dst_size, src_size);
    LOG_INFO(Lib_Ssl, "libSecure decrypt passthrough: dst_size = {}, src_size = {}, copied = {}",
             dst_size, src_size, n);
    if (dst != nullptr && src != nullptr && n > 0) {
        std::memcpy(dst, src, n);
    }
    if (processed != nullptr) {
        *processed = n;
    }
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    // sceLibSecureCryptographyDecrypt, NID hMYgMP-Vuno.
    //
    // The eboot imports this under one library/module name; the journal cites the bundled
    // libSceSecure.prx, while the firmware module is libSceLibSecure. Resolution is an exact
    // match on "{nid}#{library}#{library_version}#{module}#{type}", so registering the same
    // function under BOTH candidate names is harmless: only the name the eboot actually
    // imports is ever looked up, the other entry is dead weight. If the boot log still prints
    //   Linker: Stub resolved sceLibSecureCryptographyDecrypt as ... (lib: X, mod: Y)
    // then neither matched -> set X/Y here and rebuild.
    LIB_FUNCTION("hMYgMP-Vuno", "libSceLibSecure", 1, "libSceLibSecure",
                 sceLibSecureCryptographyDecrypt);
    LIB_FUNCTION("hMYgMP-Vuno", "libSceSecure", 1, "libSceSecure",
                 sceLibSecureCryptographyDecrypt);
}

} // namespace Libraries::LibSecure
