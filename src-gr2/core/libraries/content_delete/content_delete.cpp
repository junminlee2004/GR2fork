// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// sceContentDelete HLE — deletes the photo the gallery selected.
//
// Three NIDs are imported from libSceContentDelete:
//   pXJh3aVk8Ks  (FuncA)
//   5XLSih32qHA  (FuncB)
//   zoxb0wEChEM  (FuncC)
//
// They are Init / DeleteById / Terminate in some order we don't need to pin
// down. GR2 deletes by looping its selected-photo list and calling
// sceContentDeleteById(idHandle) once per photo, where idHandle is the opaque
// 8-byte value our search wrote to SceContentSearchContentInfo +0x00 (tracked
// in ContentSearch's id<->handle table). We map that handle back to a content
// id and delete the file + drop it from the search.
//
// Only the DeleteById call carries a real handle; Init (workspace) and
// Terminate (ctx) miss the table and no-op. We do a pure table lookup and never
// dereference the arg as a pointer, so an unknown value is harmless. Repeated
// calls for the same photo are idempotent (DeleteContentById no-ops once the id
// is gone).

#include <string>

#include "common/logging/log.h"
#include "core/libraries/libs.h"
#include "core/libraries/content_delete/content_delete.h"
#include "core/libraries/content_search/content_search.h"

namespace Libraries::ContentDelete {

// Map a delete handle to its content id and delete. The handle is the first
// integer arg, but we check the first two for ABI slack — GetContentIdByHandle
// is a pure table lookup (no dereference) and DeleteContentById is self-guarding
// (no-ops on unknown ids), so checking both is safe and can never act on the
// wrong photo.
static int HandleDeleteCall(const char* func_name, u64 a1, u64 a2, u64 a3,
                             u64 a4, u64 a5, u64 a6) {
    LOG_INFO(Core, "[ContentDelete] {} called: "
             "a1={:#x} a2={:#x} a3={:#x} a4={:#x} a5={:#x} a6={:#x}",
             func_name, a1, a2, a3, a4, a5, a6);

    for (u64 arg : {a1, a2}) {
        const std::string id = Libraries::ContentSearch::GetContentIdByHandle(arg);
        if (id.empty()) continue;

        const bool deleted = Libraries::ContentSearch::DeleteContentById(id);
        LOG_INFO(Core, "[ContentDelete] {} — handle {:#x} -> id '{}' -> {}",
                 func_name, arg, id, deleted ? "deleted" : "not found");
        return 0; // handled this photo; don't double-act on the other arg
    }

    return 0; // ORBIS_OK — Init/Terminate, or a handle we don't track: nothing to do
}

// NID pXJh3aVk8Ks
int PS4_SYSV_ABI sceContentDeleteFuncA(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6) {
    return HandleDeleteCall("FuncA(pXJh3aVk8Ks)", a1, a2, a3, a4, a5, a6);
}

// NID 5XLSih32qHA
int PS4_SYSV_ABI sceContentDeleteFuncB(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6) {
    return HandleDeleteCall("FuncB(5XLSih32qHA)", a1, a2, a3, a4, a5, a6);
}

// NID zoxb0wEChEM
int PS4_SYSV_ABI sceContentDeleteFuncC(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6) {
    return HandleDeleteCall("FuncC(zoxb0wEChEM)", a1, a2, a3, a4, a5, a6);
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("pXJh3aVk8Ks", "libSceContentDelete", 1, "libSceContentDelete",
                 sceContentDeleteFuncA);
    LIB_FUNCTION("5XLSih32qHA", "libSceContentDelete", 1, "libSceContentDelete",
                 sceContentDeleteFuncB);
    LIB_FUNCTION("zoxb0wEChEM", "libSceContentDelete", 1, "libSceContentDelete",
                 sceContentDeleteFuncC);
    LOG_INFO(Core, "[ContentDelete] HLE registered — sceContentDeleteById maps "
             "the content-ID handle back to a file via ContentSearch");
}

} // namespace Libraries::ContentDelete
