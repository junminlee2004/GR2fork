// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// [SNDMOD] GR2 title-screen-theme replacement (Config::getGR2TitleThemeMod()): plays the first
// file in the user "mods" folder through its own SDL3 mixer device - the guest movie-audio
// pipeline distorts injected PCM even at -18 dBFS. A watchdog stops playback when frames stop.

#include <chrono>
#include <mutex>

namespace Libraries::Ajm::TitleThemeMod {

class Player {
public:
    // True when the feature is on AND a usable file is loaded (lazy, thread-safe init on first
    // call). AjmAt9Decoder uses this to decide whether to engage (silence original + drive frames).
    bool Active();

    // Called once per decoded title-theme frame with the original frame's peak (0..1). Starts
    // looped playback on the first frame, scales the replacement to match the original's level,
    // and refreshes the watchdog heartbeat so playback stops shortly after the theme ends.
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

// [SNDMOD] Guest BGM bus volume set via sceAudioOutSetVolume (normalized, 1.0 = 0 dB). AudioOut
// writes it per output; the player folds it into its gain so the replacement tracks the game's
// level even though the original is silenced upstream of the bus. Lock-free.
void SetBusGain(float gain01);
float GetBusGain();

} // namespace Libraries::Ajm::TitleThemeMod
