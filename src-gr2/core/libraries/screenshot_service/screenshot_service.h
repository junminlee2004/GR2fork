// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Libraries::ScreenshotService {

// GR2's own View (Cross) / Mark (Square) gates pass when HLE search results carry real
// per-photo records and paths, so no eboot patch is needed. The gallery-visibility helpers
// (IsGalleryVisible / PollGalleryStateAndLogEdges) are defined in screenshot_service.cpp.

} // namespace Libraries::ScreenshotService
