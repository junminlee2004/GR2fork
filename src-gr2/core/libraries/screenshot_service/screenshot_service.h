// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::ScreenshotService {

// The GR2 View/Mark eboot binary patches were removed: once HLE search results
// carry real per-photo records and paths, the game's own View (Cross) / Mark
// (Square) gates pass without NOPing them.
//
// The gallery-visibility helpers (IsGalleryVisible / PollGalleryStateAndLogEdges)
// remain defined in screenshot_service.cpp.

} // namespace Libraries::ScreenshotService
