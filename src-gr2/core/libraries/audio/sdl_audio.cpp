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
        // GR2FORK: mute the pad/controller speaker output when disabled in config (default). Skips the
        // device so the DualShock-speaker mix is not duplicated through PC playback.
        if (port.type == OrbisAudioOutPort::PadSpk && Config::isPadSpkOutputDisabled()) {
            stream = nullptr;
            return;
        }
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

        // Per-buffer safety guard (every port) plus a one-time Audio3D warmup mute. Glitched float
        // buffers (NaN/Inf, peak > 1.0) fold into [-1, 1], an identity on clean audio; the int16
        // Audio3D bus arrives already wrapped, so it gets a short wrap/square-signature-keyed mute.
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

                // Sanitize in place when the buffer is hot (any sample past 0 dBFS) or carries a
                // non-finite sample: NaN/Inf -> 0, finite values clamped to [-1, 1]. Clamping does
                // nothing to in-range samples, so the rest of the mix is untouched.
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

                // A mostly-non-finite buffer is too broken to repair cleanly (scattered NaN -> 0
                // would click), so silence it for a short tail. A lone stray NaN was already
                // neutralized above with no dropout; this is the only float-port mute path.
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

                // Wrap-around/full-scale square content is baked in and unrecoverable - mute
                // briefly. Discriminated by signature (pinned near full scale AND dense
                // near-full-scale jumps), so clean-but-loud audio passes through untouched.
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

    // Force a short silence window over a glitched buffer; re-armed per glitched buffer, so a
    // multi-buffer glitch stays suppressed with a GLITCH_MUTE_BUFFERS-long tail once clean
    // buffers resume. The log is edge-triggered to keep it out of the hot path.
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

    // int16 wrap/square discriminator (Audio3D S16 bus). Mute only when the peak is within
    // (32767 - INT16_EARRAPE_PEAK) of full scale AND at least 1/INT16_HARSH_DENOM of adjacent
    // pairs jump more than INT16_HARSH_DELTA; smooth loud audio never satisfies the second.
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
