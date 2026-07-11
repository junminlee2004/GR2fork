// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::LibSecure {

// GR2 online restoration: ARC data ops (ghost/treasure/photo) decrypt a binary TLV stream via
// libSceSecure. The emulator owns the crypto boundary, so this is HLE'd as a plaintext passthrough
// and the restoration server emits plaintext TLV. The arg layout is the SCE streaming-cipher form.
s32 PS4_SYSV_ABI sceLibSecureCryptographyDecrypt(void* ctx, void* dst, u64 dst_size,
                                                 const void* src, u64 src_size, u64* processed);

// ARC request crypto is the in-place 3-arg form fn(ctxA, ArcCryptoBuf*{data,size}, ctxB),
// confirmed for both decrypt and encrypt in the eboot. Encrypt is HLE'd as a no-op so the ARC
// request 'p' blob reaches the local server as plaintext TLV - the mirror of the decrypt no-op.
struct ArcCryptoBuf;
s32 PS4_SYSV_ABI sceLibSecureCryptographyEncrypt(void* ctx_a, ArcCryptoBuf* buf, void* ctx_b);

void RegisterLib(Core::Loader::SymbolsResolver* sym);

} // namespace Libraries::LibSecure
