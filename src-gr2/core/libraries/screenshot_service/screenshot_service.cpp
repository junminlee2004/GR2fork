// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "core/libraries/screenshot_service/screenshot_service.h"

namespace Libraries::ScreenshotService {

// No View/Mark eboot patches are needed: search results carry real records and paths, so the
// game's null-cell gates pass on their own.

// Gallery visibility detection: the game keeps a photo-mode state object reachable from BSS
// PHOTO_MODE_ROOT (0x1AA3E78); field +0x178 is 2 while the film-album gallery UI is visible,
// and something else (or a null link in the chain) otherwise.

constexpr u64 kPhotoModeRoot = 0x1AA3E78;

// Pointer-sanity check - filter out obviously bogus values (low system
// addresses and addresses above the ~47-bit user-space ceiling).
static inline bool PtrLooksValid(u64 v) {
    return v > 0x100000ULL && v < 0x800000000000ULL;
}

bool IsGalleryVisible(uintptr_t eboot_base) {
    if (eboot_base == 0) return false;
    const u64 root = *reinterpret_cast<u64*>(eboot_base + kPhotoModeRoot);
    if (!PtrLooksValid(root)) return false;
    const u64 obj = *reinterpret_cast<u64*>(root + 8);
    if (!PtrLooksValid(obj)) return false;
    const u32 state = *reinterpret_cast<u32*>(obj + 0x178);
    return (state == 2);
}

bool PollGalleryStateAndLogEdges(uintptr_t eboot_base) {
    static bool was_visible = false;
    const bool now_visible = IsGalleryVisible(eboot_base);
    if (now_visible != was_visible) {
        LOG_INFO(Core, "[Gallery] {}",
                 now_visible ? "ENTERED film-album" : "EXITED film-album");
        was_visible = now_visible;
    }
    return now_visible;
}

} // namespace Libraries::ScreenshotService
