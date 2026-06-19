// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>
#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_hints.h>

#include "common/config.h"
#include "common/logging/log.h"
#include "core/libraries/audio/audioout.h"
#include "core/libraries/audio/audioout_backend.h"

#define SDL_INVALID_AUDIODEVICEID 0 // Defined in SDL_audio.h but not made a macro
namespace Libraries::AudioOut {

// One-time mute window for first Audio3D activity (ms).
// Helps dodge the initial Audio3D SFX burst some games trigger on first use.
constexpr u32 AUDIO3D_WARMUP_MUTE_MS = 325;

class SDLPortBackend : public PortBackend {
public:
    explicit SDLPortBackend(const PortOut& port)
        : frame_size(port.format_info.FrameSize()),
          guest_buffer_size(port.BufferSize()),
          sample_rate(port.sample_rate),
          is_audio3d(port.type == OrbisAudioOutPort::Audio3d),
          is_float(port.format_info.is_float) {
        const SDL_AudioSpec fmt = {
            .format = port.format_info.is_float ? SDL_AUDIO_F32LE : SDL_AUDIO_S16LE,
            .channels = port.format_info.num_channels,
            .freq = static_cast<int>(port.sample_rate),
        };

        // Determine port type
        std::string port_name = port.type == OrbisAudioOutPort::PadSpk
                                    ? Config::getPadSpkOutputDevice()
                                    : Config::getMainOutputDevice();
        SDL_AudioDeviceID dev_id = SDL_INVALID_AUDIODEVICEID;
        if (port_name == "None") {
            stream = nullptr;
            return;
        } else if (port_name == "Default Device") {
            dev_id = SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
        } else {
            try {
                SDL_AudioDeviceID* dev_array = SDL_GetAudioPlaybackDevices(nullptr);
                for (; dev_array != 0;) {
                    std::string dev_name(SDL_GetAudioDeviceName(*dev_array));
                    if (dev_name == port_name) {
                        dev_id = *dev_array;
                        break;
                    } else {
                        dev_array++;
                    }
                }
                if (dev_id == SDL_INVALID_AUDIODEVICEID) {
                    LOG_WARNING(Lib_AudioOut, "Audio device not found: {}", port_name);
                    dev_id = SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
                }
            } catch (const std::exception& e) {
                LOG_ERROR(Lib_AudioOut, "Invalid audio output device: {}", port_name);
                stream = nullptr;
                return;
            }
        }

        // Open the audio stream
        stream = SDL_OpenAudioDeviceStream(dev_id, &fmt, nullptr, nullptr);
        if (stream == nullptr) {
            LOG_ERROR(Lib_AudioOut, "Failed to create SDL audio stream: {}", SDL_GetError());
            return;
        }
        CalculateQueueThreshold();
        if (!SDL_SetAudioStreamInputChannelMap(stream, port.format_info.channel_layout.data(),
                                               port.format_info.num_channels)) {
            LOG_ERROR(Lib_AudioOut, "Failed to configure SDL audio stream channel map: {}",
                      SDL_GetError());
            SDL_DestroyAudioStream(stream);
            stream = nullptr;
            return;
        }
        if (!SDL_ResumeAudioStreamDevice(stream)) {
            LOG_ERROR(Lib_AudioOut, "Failed to resume SDL audio stream: {}", SDL_GetError());
            SDL_DestroyAudioStream(stream);
            stream = nullptr;
            return;
        }
        SDL_SetAudioStreamGain(stream, Config::getVolumeSlider() / 100.0f);
    }

    ~SDLPortBackend() override {
        if (!stream) {
            return;
        }
        SDL_DestroyAudioStream(stream);
        stream = nullptr;
    }

    void Output(void* ptr) override {
        if (!stream) {
            return;
        }

        // AudioOut library manages timing, but we still need to guard against the SDL
        // audio queue stalling, which may happen during device changes, for example.
        // Otherwise, latency may grow over time unbounded.
        if (const auto queued = SDL_GetAudioStreamQueued(stream); queued >= queue_threshold) {
            LOG_INFO(Lib_AudioOut, "SDL audio queue backed up ({} queued, {} threshold), clearing.",
                     queued, queue_threshold);
            SDL_ClearAudioStream(stream);
            // Recalculate the threshold in case this happened because of a device change.
            CalculateQueueThreshold();
        }

        // Default output pointer is the guest buffer; if it's null, output silence.
        const void* out_ptr = ptr;

        if (out_ptr == nullptr) [[unlikely]] {
            if (zero_buf.size() != guest_buffer_size) {
                zero_buf.assign(guest_buffer_size, 0);
            } else {
                std::memset(zero_buf.data(), 0, guest_buffer_size);
            }
            out_ptr = zero_buf.data();
        }

        // Per-buffer safety guard (runs for EVERY port) plus a one-time Audio3D warmup mute.
        //
        // The whole point of this guard is to kill the "gravity-kick earrape": certain SFX (notably
        // Kat's cross+square ground-slam melee combo) overflow the guest's master mix and arrive here
        // as a glitched buffer. There are two distinct failure modes, one per sample format, and they
        // need different handling:
        //
        //   * Float busses (Main/Bgm = Float_8CH, PadSpk = FloatMono). The bad buffer carries NaN/Inf
        //     or finite-but-out-of-range values (peak >> 1.0). These are RECOVERABLE: we fold every
        //     sample back into [-1, 1] in place. Clamping is *identity* on correct audio (already
        //     <= 0 dBFS), so everything else summed into the same master buffer -- music, ambience --
        //     is left bit-exact; only the runaway SFX excursion is tamed. We deliberately do NOT mute
        //     these: muting the shared master buffer would gate every other sound off too, and a
        //     spammed combo would chop the whole bus on/off (audible as its own crackle).
        //   * The int16 Audio3D bus (S16Stereo). By the time we see it, the overflow has already been
        //     truncated/wrapped into the int16 range, so the garbage is baked in and UNRECOVERABLE --
        //     clamping is a no-op. A short mute is the only recourse, but it must fire ONLY on the real
        //     wrap/square signature, never on merely-loud-but-clean audio.
        //
        // Old behaviour this replaces (the holes that let the crackle through): the float path passed
        // finite buffers with peak in (1.0, 1.5] to SDL verbatim, so the over-unity shoulder of a hot
        // transient (the slam-kick tail) still clipped at the device and crackled; it also muted the
        // entire buffer on a SINGLE stray NaN. The int16 path muted on peak > 32000, i.e. anything past
        // -0.18 dBFS, chopping legitimately loud sounds.
        if (ptr != nullptr) {
            if (is_float) {
                const float* in = static_cast<const float*>(ptr);
                const size_t n = static_cast<size_t>(guest_buffer_size) / sizeof(float);

                float peak = 0.0f;
                size_t non_finite = 0;
                for (size_t i = 0; i < n; ++i) {
                    const float v = in[i];
                    if (!std::isfinite(v)) {
                        ++non_finite;
                        continue;
                    }
                    peak = std::max(peak, std::fabs(v));
                }

                // Audio3D only: first time the port becomes non-silent, mute a short window to
                // dodge the one-time init SFX burst some games trigger on first use.
                if (is_audio3d && !audio3d_seen_non_silent && peak > 1e-4f) {
                    audio3d_seen_non_silent = true;
                    ArmWarmupMute();
                }

                // The actual earrape fix for the float busses. Sanitize in place whenever the buffer
                // is hot (any sample past 0 dBFS) OR carries any non-finite sample: NaN/Inf -> 0 and
                // finite values hard-clamped to [-1, 1]. Nothing past full scale can ever reach SDL,
                // and because clamping does nothing to in-range samples the rest of the mix is
                // untouched -- no dropouts, no gating, just the runaway excursion capped.
                if (non_finite > 0 || peak > 1.0f) {
                    if (f32_buf.size() != n) {
                        f32_buf.resize(n);
                    }
                    for (size_t i = 0; i < n; ++i) {
                        const float v = in[i];
                        f32_buf[i] = std::isfinite(v) ? std::clamp(v, -1.0f, 1.0f) : 0.0f;
                    }
                    out_ptr = f32_buf.data();
                    LogLimiterEdge(peak, non_finite);
                } else {
                    limiter_engaged = false; // clean buffer: re-arm the edge-triggered log
                }

                // A buffer that is *mostly* non-finite is too broken to repair cleanly (NaN->0
                // scattered across most of it would click), so silence it for a short tail. A lone
                // stray NaN does NOT trigger this -- it was already neutralised to 0 above, with no
                // dropout. This is the only path on which a float port mutes.
                if (n > 0 && non_finite * 4 >= n) {
                    ArmGlitchMute(peak, true);
                }
            } else {
                const auto* in = static_cast<const int16_t*>(ptr);
                const size_t n = static_cast<size_t>(guest_buffer_size) / sizeof(int16_t);

                int peak = 0;
                size_t harsh = 0; // adjacent samples a near-full-scale jump apart: the wrap/square tell
                int prev = (n > 0) ? static_cast<int>(in[0]) : 0;
                for (size_t i = 0; i < n; ++i) {
                    const int s = static_cast<int>(in[i]);
                    const int a = (s < 0) ? -s : s;
                    peak = std::max(peak, a);
                    const int d = s - prev;
                    if (d > INT16_HARSH_DELTA || d < -INT16_HARSH_DELTA) {
                        ++harsh;
                    }
                    prev = s;
                }

                // Audio3D only: first time it becomes non-silent, mute a short window.
                if (is_audio3d && !audio3d_seen_non_silent && peak > 4) {
                    audio3d_seen_non_silent = true;
                    ArmWarmupMute();
                }

                // int16 earrape == wrap-around / full-scale square content, already baked in and
                // unrecoverable, so mute briefly. Discriminate it from clean-but-loud audio by the
                // SIGNATURE, not the level: it pins near full scale AND has a high density of
                // near-full-scale sample-to-sample jumps. Smooth loud audio (even at 0 dBFS) has
                // essentially none of those jumps, so it now passes through untouched instead of
                // being chopped by the old peak-only test.
                if (n > 0 && peak >= INT16_EARRAPE_PEAK && harsh * INT16_HARSH_DENOM >= n) {
                    ArmGlitchMute(static_cast<float>(peak) / 32768.0f, false);
                }
            }

            if (mute_buffers > 0) {
                if (zero_buf.size() != guest_buffer_size) {
                    zero_buf.assign(guest_buffer_size, 0);
                } else {
                    std::memset(zero_buf.data(), 0, guest_buffer_size);
                }
                out_ptr = zero_buf.data();
                --mute_buffers;
            }
        }

        if (!SDL_PutAudioStreamData(stream, out_ptr, static_cast<int>(guest_buffer_size))) {
            LOG_ERROR(Lib_AudioOut, "Failed to output to SDL audio stream: {}", SDL_GetError());
        }
    }

    void SetVolume(const std::array<int, 8>& ch_volumes) override {
        if (!stream) {
            return;
        }
        // SDL does not have per-channel volumes, for now just take the maximum of the channels.
        const auto vol = *std::ranges::max_element(ch_volumes);
        if (!SDL_SetAudioStreamGain(stream, static_cast<float>(vol) / SCE_AUDIO_OUT_VOLUME_0DB *
                                                Config::getVolumeSlider() / 100.0f)) {
            LOG_WARNING(Lib_AudioOut, "Failed to change SDL audio stream volume: {}",
                        SDL_GetError());
        }
    }

private:
    void CalculateQueueThreshold() {
        SDL_AudioSpec discard;
        int sdl_buffer_frames;
        if (!SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(stream), &discard,
                                      &sdl_buffer_frames)) {
            LOG_WARNING(Lib_AudioOut, "Failed to get SDL audio stream buffer size: {}",
                        SDL_GetError());
            sdl_buffer_frames = 0;
        }
        const auto sdl_buffer_size = sdl_buffer_frames * frame_size;
        const auto new_threshold = std::max(guest_buffer_size, sdl_buffer_size) * 4;
        if (host_buffer_size != sdl_buffer_size || queue_threshold != new_threshold) {
            host_buffer_size = sdl_buffer_size;
            queue_threshold = new_threshold;
            LOG_INFO(Lib_AudioOut,
                     "SDL audio buffers: guest = {} bytes, host = {} bytes, threshold = {} bytes",
                     guest_buffer_size, host_buffer_size, queue_threshold);
        }
    }

    // Arm the one-time Audio3D warmup mute window (~AUDIO3D_WARMUP_MUTE_MS), sized in whole
    // guest buffers. Only ever called for the Audio3D port.
    void ArmWarmupMute() {
        const u32 buffer_frames = frame_size ? (guest_buffer_size / frame_size) : 0;
        const u32 buf_ms = (buffer_frames && sample_rate)
                               ? static_cast<u32>((static_cast<u64>(buffer_frames) * 1000ull +
                                                   static_cast<u64>(sample_rate) - 1) /
                                                  static_cast<u64>(sample_rate))
                               : 1;
        const u32 warmup = std::max<u32>(1, (AUDIO3D_WARMUP_MUTE_MS + buf_ms - 1) / buf_ms);
        mute_buffers = std::max(mute_buffers, warmup);
    }

    // Force a short silence window to ride out a glitched (NaN/Inf or absurdly hot) buffer.
    // Re-armed every glitched buffer, so a multi-buffer glitch stays fully suppressed and a
    // GLITCH_MUTE_BUFFERS-long silent tail plays out once clean buffers resume. The log is
    // edge-triggered (only when a fresh mute window opens) to keep it out of the hot path.
    void ArmGlitchMute(float normalized_peak, bool non_finite) {
        if (mute_buffers == 0) {
            LOG_WARNING(Lib_AudioOut,
                        "Glitched audio buffer suppressed (audio3d={}, non_finite={}, peak={:.3f}); "
                        "muting {} buffers",
                        is_audio3d, non_finite, normalized_peak, GLITCH_MUTE_BUFFERS);
        }
        mute_buffers = std::max(mute_buffers, GLITCH_MUTE_BUFFERS);
    }

    // Edge-triggered log for the float limiter so the hot path never logs per-buffer: it fires once
    // on a clean->limiting transition and stays quiet until a clean buffer re-arms it (see the
    // `limiter_engaged = false` reset in Output()).
    void LogLimiterEdge(float peak, size_t non_finite) {
        if (!limiter_engaged) {
            limiter_engaged = true;
            LOG_WARNING(Lib_AudioOut,
                        "Audio limiter engaged (is_float={}, audio3d={}, non_finite={}, peak={:.3f}); "
                        "folding buffer to <= 0 dBFS",
                        is_float, is_audio3d, non_finite, peak);
        }
    }

    static constexpr u32 GLITCH_MUTE_BUFFERS = 8u;

    // int16 wrap/square earrape discriminator (Audio3D S16 bus). BOTH must hold before we mute:
    //   * peak within (32767 - INT16_EARRAPE_PEAK) of full scale -- essentially pinned at the rails;
    //   * at least 1/INT16_HARSH_DENOM of adjacent sample pairs jump by more than INT16_HARSH_DELTA.
    // Clean loud audio satisfies the first but never the second (a smooth waveform has no near-full-
    // scale adjacent jumps), so it is no longer chopped the way the old peak-only test chopped it.
    static constexpr int INT16_EARRAPE_PEAK = 32760;    // <= 8 LSB below full scale (32767)
    static constexpr int INT16_HARSH_DELTA = 48000;     // adjacent-sample jump (max possible 65535)
    static constexpr size_t INT16_HARSH_DENOM = 16u;    // density gate: >= 6.25% of pairs are harsh

    u32 frame_size;
    u32 guest_buffer_size;
    u32 sample_rate;
    bool is_audio3d;
    bool is_float;
    bool audio3d_seen_non_silent{false};
    bool limiter_engaged{false};
    u32 mute_buffers{0};
    std::vector<std::uint8_t> zero_buf{};
    std::vector<float> f32_buf{};
    u32 host_buffer_size{};
    u32 queue_threshold{};
    SDL_AudioStream* stream{};
};

std::unique_ptr<PortBackend> SDLAudioOut::Open(PortOut& port) {
    return std::make_unique<SDLPortBackend>(port);
}

} // namespace Libraries::AudioOut
