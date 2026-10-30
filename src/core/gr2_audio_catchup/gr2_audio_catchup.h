// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <stop_token>

namespace Libraries::AudioOut {
struct PortOut;
}

// Audio output pacing that keeps the guest's audio clock in real time across host stalls. Each
// port's output thread takes one guest block per buffer period. When the thread wakes late, the
// periods it missed are played as fast as the guest hands over blocks, instead of being spent on
// ticks that find no block. Without this the guest's audio falls behind real time by the length
// of every stall, and games that sync video to their audio position drift.
namespace Gr2AudioCatchup {

void RunOutputThread(const std::shared_ptr<Libraries::AudioOut::PortOut>& port,
                     const std::stop_token& stop);

// The output thread paces the backends, so they skip their own pacing sleep.
constexpr bool OutputThreadPaces() {
    return true;
}

} // namespace Gr2AudioCatchup
