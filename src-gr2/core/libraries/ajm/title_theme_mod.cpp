// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/ajm/title_theme_mod.h"

#include "common/config.h"
#include "common/logging/log.h"
#include "common/path_util.h"

#include <SDL3_mixer/SDL_mixer.h> // also pulls in <SDL3/SDL.h> (properties, audio-device defaults)

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace Libraries::Ajm::TitleThemeMod {

using namespace std::chrono_literals;

// Stop playback this long after the last title-theme frame was decoded. Long enough to ride across
// the short gaps the movie demuxer can leave between buffer refills / loops; short enough that the
// music does not bleed far into the game once the title screen is left.
static constexpr auto kStopAfterIdle = 600ms;

// [SNDMOD] Bus-volume handoff (see header). Written by the AudioOut output thread, read by the
// player thread; relaxed atomics suffice since it's a single scalar with no ordering dependencies.
static std::atomic<float> g_bus_gain{1.0f};
void SetBusGain(float gain01) {
    g_bus_gain.store(std::clamp(gain01, 0.0f, 1.0f), std::memory_order_relaxed);
}
float GetBusGain() {
    return g_bus_gain.load(std::memory_order_relaxed);
}

void Player::EnsureInit() {
    if (m_init_done) {
        return;
    }
    m_init_done = true; // only ever attempt setup once

    if (!Config::getGR2TitleThemeMod()) {
        return; // feature off (default)
    }

    const auto& dir = Common::FS::GetUserPath(Common::FS::PathType::ModsDir);
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        return;
    }

    // "Any file in the folder" -> the first regular file in a deterministic order.
    std::vector<std::filesystem::path> entries;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (e.is_regular_file(ec)) {
            entries.push_back(e.path());
        }
    }
    if (entries.empty()) {
        LOG_INFO(Lib_Ajm, "[SNDMOD] GR2TitleThemeMod is on but the mods folder is empty: {}",
                 Common::FS::PathToUTF8String(dir));
        return;
    }
    std::sort(entries.begin(), entries.end());
    const std::string path = Common::FS::PathToUTF8String(entries.front());

    if (!MIX_Init()) {
        LOG_ERROR(Lib_Ajm, "[SNDMOD] MIX_Init failed: {}", SDL_GetError());
        return;
    }
    auto* mixer = MIX_CreateMixerDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, nullptr);
    if (mixer == nullptr) {
        LOG_ERROR(Lib_Ajm, "[SNDMOD] MIX_CreateMixerDevice failed: {}", SDL_GetError());
        return;
    }
    // Start fully attenuated; NotifyFrame() sets the real, loudness-matched gain (original peak x
    // the volume slider) before playback actually begins, so the replacement never blasts at the
    // file's native full-scale level even for a frame.
    MIX_SetMixerGain(mixer, 0.0f);

    auto* audio = MIX_LoadAudio(mixer, path.c_str(), false);
    if (audio == nullptr) {
        LOG_ERROR(Lib_Ajm, "[SNDMOD] MIX_LoadAudio failed for {}: {}", path, SDL_GetError());
        MIX_DestroyMixer(mixer);
        return;
    }
    auto* track = MIX_CreateTrack(mixer);
    if (track == nullptr) {
        LOG_ERROR(Lib_Ajm, "[SNDMOD] MIX_CreateTrack failed: {}", SDL_GetError());
        MIX_DestroyAudio(audio);
        MIX_DestroyMixer(mixer);
        return;
    }
    MIX_SetTrackAudio(track, audio);

    m_mixer = mixer;
    m_audio = audio;
    m_track = track;
    m_loaded = true;
    LOG_INFO(Lib_Ajm, "[SNDMOD] title-theme direct player loaded: {}", path);
}

void Player::StartLocked() {
    if (!m_loaded || m_playing) {
        return;
    }
    SDL_PropertiesID opts = SDL_CreateProperties();
    SDL_SetNumberProperty(opts, MIX_PROP_PLAY_LOOPS_NUMBER, -1); // -1 = loop forever
    const bool ok = MIX_PlayTrack(static_cast<MIX_Track*>(m_track), opts);
    SDL_DestroyProperties(opts);
    m_playing = ok;
    LOG_INFO(Lib_Ajm, "[SNDMOD] title-theme direct player: {}",
             ok ? "playing replacement (looped)" : "MIX_PlayTrack failed");
}

void Player::StopLocked() {
    if (!m_playing) {
        return;
    }
    MIX_StopTrack(static_cast<MIX_Track*>(m_track), 0); // 0 = no fade, stop immediately
    m_playing = false;
    LOG_INFO(Lib_Ajm, "[SNDMOD] title-theme direct player: stopped (title theme ended)");
}

bool Player::Active() {
    std::lock_guard lk(m_mtx);
    EnsureInit();
    return m_loaded;
}

void Player::NotifyFrame(float ref_peak) {
    std::lock_guard lk(m_mtx);
    EnsureInit();
    if (!m_loaded) {
        return;
    }
    m_last_frame = std::chrono::steady_clock::now();

    // Loudness-match: track the loudest sample of the original title theme and scale the
    // replacement so its peak lands at that level x the game's BGM bus volume (from AudioOut) x
    // the user's volume slider; re-applied every frame, and set before the first StartLocked().
    const float bus = GetBusGain();
    if (ref_peak > m_ref_peak) {
        m_ref_peak = ref_peak;
        // Sparse: fires only on a new maximum (a handful of times as it converges). Shows where the
        // level comes from -- source headroom (peak < 1) vs the bus volume the game set.
        LOG_INFO(Lib_Ajm,
                 "[SNDMOD] loudness-match: original peak ~{:.3f}, bus gain ~{:.3f} -> player gain "
                 "{:.3f}",
                 m_ref_peak, bus,
                 std::min(m_ref_peak, 1.0f) * bus *
                     (static_cast<float>(Config::getVolumeSlider()) / 100.0f));
    }
    if (m_mixer != nullptr) {
        const float gain = std::min(m_ref_peak, 1.0f) * bus *
                           (static_cast<float>(Config::getVolumeSlider()) / 100.0f);
        MIX_SetMixerGain(static_cast<MIX_Mixer*>(m_mixer), gain);
    }

    if (!m_playing) {
        StartLocked();
    }
    if (!m_watchdog_started) {
        m_watchdog_started = true;
        // Detached, session-lifetime: polls the heartbeat and stops playback once title-theme
        // frames stop arriving. Capturing `this` is safe because Get()'s Player is never destroyed.
        std::thread([this] {
            for (;;) {
                std::this_thread::sleep_for(100ms);
                std::lock_guard lk(m_mtx);
                if (m_playing &&
                    std::chrono::steady_clock::now() - m_last_frame > kStopAfterIdle) {
                    StopLocked();
                }
            }
        }).detach();
    }
}

Player& Get() {
    static Player* p = new Player(); // intentionally leaked; outlives static teardown
    return *p;
}

} // namespace Libraries::Ajm::TitleThemeMod
