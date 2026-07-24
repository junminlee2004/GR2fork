// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace GR2Fork {

// Result of the launch preflight against the restoration server's /clientgate endpoint.
// UpdateRequired means the server refuses this OnlineVersion (the caller quits); Banned and
// DebugDenied are informational (the caller shows a dialog and the game continues offline).
struct LaunchGate {
    enum class Status { Ok, UpdateRequired, Banned, DebugDenied };
    Status status = Status::Ok;
    int ban_days = 0; // days remaining on a timed ban; 0 = permanent
};

// One bounded /clientgate GET carrying the fork's identity headers. Skips non-GR2 titles,
// online-off, and an unconfigured server host; an unreachable server reports Ok so offline play
// never blocks on the network.
LaunchGate LaunchGateCheck();

} // namespace GR2Fork
