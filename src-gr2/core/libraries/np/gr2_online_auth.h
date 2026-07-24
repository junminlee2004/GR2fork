// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

namespace GR2Fork::Auth {

// Starts the shadnet login once from the GR2Fork config credentials and reports readiness. A no-op when
// no Online ID is configured (auth off). Safe to call on every online request; only the first call
// connects, and it briefly waits so that first request already carries the bearer token.
bool EnsureAuthed();

// The shadnet-verified Online ID once authenticated, else empty. The restoration server independently
// re-checks the bearer token, so this is the claimed identity the token must back.
std::string VerifiedNpid();

// The shadnet bearer token for the restoration server to validate, else empty.
std::string BearerToken();

// The online identity to present in-game: the shadnet-verified Online ID once authenticated, else the
// configured Username. The signed-in Online ID replaces the Username so the on-screen name, ghost
// nameplates, and NP identity all match the account the server authenticated.
std::string EffectiveOnlineId();

// True when online is intended: an Online ID is configured AND the "connected to network" + "signed in
// to PSN" toggles are both on. The login and every online request are gated on this.
bool IsOnlineEnabled();

// Blocks (bounded) on the shadnet login at launch. Returns true if online is off (nothing to do) or the
// login succeeds; false if it fails - the caller shows GetLoginErrorMessage() and quits the game.
bool PreflightBlockingLogin();

// Human-readable reason for the last login failure (connect / wrong credentials / server error), or
// empty when the login has not failed.
std::string GetLoginErrorMessage();

} // namespace GR2Fork::Auth
