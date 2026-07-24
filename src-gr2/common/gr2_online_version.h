// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace GR2Fork {

// Monotonic gr2fork online-protocol version, sent to the restoration server as X-GR2-Version on every
// request. Bump on each release that changes the online wire or that clients must update to. The
// server alone enforces it (rejects any client below its configured minimum with HTTP 426), so this is
// a compatibility gate, not a security one (a patched client can report any value).
inline constexpr int OnlineVersion = 1;

} // namespace GR2Fork
