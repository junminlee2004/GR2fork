// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::LibSecure {

// GR2 online restoration: GR2's ARC data ops (ghost/treasure/photo) return an encrypted
// binary TLV stream that the game decrypts via libSceSecure before parsing. On real HW the
// decrypt key is app-baked; under emulation we own the crypto boundary, so this is HLE'd as
// a plaintext passthrough (memcpy ciphertext->plaintext, return OK). That lets the
// restoration server emit plaintext TLV and the game's decrypt becomes a no-op.
//
// Canonical SCE streaming-cipher signature: (context, dst, dstSize, src, srcSize, *processed).
// If a first-boot trace shows a different arg layout, adjust here (see scripts/
// dump_libsecure_decrypt.py for a Ghidra confirmation pass).
s32 PS4_SYSV_ABI sceLibSecureCryptographyDecrypt(void* ctx, void* dst, u64 dst_size,
                                                 const void* src, u64 src_size, u64* processed);

void RegisterLib(Core::Loader::SymbolsResolver* sym);

} // namespace Libraries::LibSecure
