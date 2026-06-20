// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// [SNDMOD] GR2 title-screen-theme replacement via a SEPARATE, direct audio player.
//
// Earlier we injected the user's PCM back into the decoder so it rode the game's movie-audio
// pipeline. That pipeline mangles the replacement into harsh distortion on loud/dense passages,
// independent of level (verified down to -18 dBFS), so injection can never sound clean.
//
// Instead this plays the user's file straight to the host through its OWN SDL3 mixer device (the
// same library trophy_ui uses), fully bypassing the guest pipeline. AjmAt9Decoder silences the
// original title theme in-place and calls NotifyFrame() once per decoded frame while it plays; a
// small watchdog stops playback shortly after those frames stop (i.e. the title screen is left).
//
// Gated entirely by Config::getGR2TitleThemeMod() (default false). The first file in the user
// "mods" folder is used. SDL3_mixer decodes it, so WAV / MP3 / OGG / FLAC all work, and looping +
// resampling are handled by the mixer (no custom DSP here, which also drops the old resampler and
// headroom logic).

#include <chrono>
#include <mutex>

namespace Libraries::Ajm::TitleThemeMod {

class Player {
public:
    // True when the feature is on AND a usable file is loaded (lazy, thread-safe init on first
    // call). AjmAt9Decoder uses this to decide whether to engage (silence original + drive frames).
    bool Active();

    // Called once per decoded title-theme frame, given the peak (0..1) of that just-decoded
    // ORIGINAL frame. Starts looped playback on the first frame, continuously scales the
    // replacement so its peak matches the original's level, and refreshes the watchdog heartbeat
    // so playback stops shortly after the title theme ends.
    void NotifyFrame(float ref_peak);

private:
    void EnsureInit();   // one-time MIX setup; caller must hold m_mtx
    void StartLocked();  // begin looped playback; caller must hold m_mtx
    void StopLocked();   // halt playback; caller must hold m_mtx

    std::mutex m_mtx;
    bool m_init_done = false;
    bool m_loaded = false;
    bool m_playing = false;
    bool m_watchdog_started = false;
    std::chrono::steady_clock::time_point m_last_frame{};
    float m_ref_peak = 0.0f;  // running max |sample| of the original; drives the loudness-match gain
    void* m_mixer = nullptr; // MIX_Mixer*
    void* m_audio = nullptr; // MIX_Audio*
    void* m_track = nullptr; // MIX_Track*
};

// Session-lifetime singleton (intentionally leaked so the SDL/MIX resources outlive static
// teardown and we avoid destruction-order hazards on exit).
Player& Get();

} // namespace Libraries::Ajm::TitleThemeMod
