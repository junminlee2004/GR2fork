// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/config.h"

namespace Vulkan {

// GR2FORK: opt-in VK_ERROR_DEVICE_LOST recovery. Default OFF - when off, a device loss keeps the
// existing fatal ASSERT path so normal/shipped builds are byte-identical and pay nothing. Enable
// with [GPU] deviceRecovery=true to make a loss set g_device_lost and defuse the fatal GPU waits
// (Stage 1) ahead of the off-thread device rebuild (Stage 2). Cached on first use.
inline bool DeviceRecoveryEnabled() {
    // Cached: the toggle is not changed at runtime, and the GPU waits that call this only run
    // after the config is loaded. [GPU] deviceRecovery in the config.
    static const bool enabled = Config::deviceRecovery();
    return enabled;
}

// Set at the eErrorDeviceLost detection sites (scheduler submit, presenter fence wait) when
// recovery is enabled; observed by the defused waits so they stop spinning/asserting, and by the
// main thread that performs the rebuild.
inline std::atomic<bool> g_device_lost{false};

} // namespace Vulkan
