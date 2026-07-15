// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/libraries/kernel/file_system.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/libs.h"
#include "core/signals.h"

namespace Libraries::Kernel {

void PS4_SYSV_ABI sceKernelDebugOutText(void* unk, char* text) {
    sceKernelWrite(1, text, strlen(text));
    return;
}

// GR2FORK: the guest SceLibc heap guard calls sceKernelPrintBacktraceWithModuleInfo right before a
// ud2 abort (double free). HLE runs on the guest thread stack - HOST_CALL adds no stack switch - so
// the aborting chain's eboot return addresses sit just above this frame. Log each with its ghidra
// offset and return ORBIS_OK; control flow is unchanged and the ud2 abort still follows.
s32 PS4_SYSV_ABI sceKernelPrintBacktraceWithModuleInfo() {
    volatile u8 stack_anchor = 0;
    // Align down to an 8-byte slot so the scan lines up with pushed (8-byte-aligned) return
    // addresses; a lone u8 local may otherwise sit at an odd offset.
    const u64 sp = reinterpret_cast<u64>(&stack_anchor) & ~u64(7);
    LOG_ERROR(Kernel, "GR2-BT: sceKernelPrintBacktraceWithModuleInfo called - guest backtrace "
                      "follows (eboot base 0x800000000, ghidra = va - 0x800000000 + 0x107BF0):");
    // rbp is 0: this frame's rbp is a host frame, not the guest chain, so only the stack scan runs;
    // it recovers the eboot callers on its own.
    Core::DumpGuestEbootBacktrace(sp, 0, "GR2-BT", 128);
    return 0;
}

void RegisterDebug(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("9JYNqN6jAKI", "libkernel", 1, "libkernel", sceKernelDebugOutText);
    LIB_FUNCTION("Wl2o5hOVZdw", "libkernel", 1, "libkernel", sceKernelPrintBacktraceWithModuleInfo);
}

} // namespace Libraries::Kernel