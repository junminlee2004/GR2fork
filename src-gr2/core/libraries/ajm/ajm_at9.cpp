// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "ajm_result.h"
#include "common/assert.h"
#include "core/libraries/ajm/ajm_at9.h"
#include "core/libraries/ajm/title_theme_mod.h"
#include "error_codes.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {
#include <decoder.h>
#include <libatrac9.h>
}

namespace Libraries::Ajm {

struct ChunkHeader {
    u32 tag;
    u32 length;
};
static_assert(sizeof(ChunkHeader) == 8);

// [SNDMOD] Peak magnitude (max |sample|, normalized to [0,1]) of one just-decoded frame, read per
// the decoder's output format. Lets the direct player scale its replacement to the same level the
// game plays the original title theme at, instead of blasting a full-scale file over the SFX.
static float Sndmod_PeakOf(const std::vector<u8>& buf, AjmFormatEncoding fmt) {
    float peak = 0.0f;
    switch (fmt) {
    case AjmFormatEncoding::S16: {
        const auto* p = reinterpret_cast<const s16*>(buf.data());
        const size_t n = buf.size() / sizeof(s16);
        for (size_t i = 0; i < n; ++i) {
            const float v = std::abs(static_cast<float>(p[i])) / 32768.0f;
            if (v > peak) {
                peak = v;
            }
        }
        break;
    }
    case AjmFormatEncoding::S32: {
        const auto* p = reinterpret_cast<const s32*>(buf.data());
        const size_t n = buf.size() / sizeof(s32);
        for (size_t i = 0; i < n; ++i) {
            const float v = std::abs(static_cast<float>(p[i])) / 2147483648.0f;
            if (v > peak) {
                peak = v;
            }
        }
        break;
    }
    case AjmFormatEncoding::Float: {
        const auto* p = reinterpret_cast<const float*>(buf.data());
        const size_t n = buf.size() / sizeof(float);
        for (size_t i = 0; i < n; ++i) {
            const float v = std::abs(p[i]);
            if (v > peak) {
                peak = v;
            }
        }
        break;
    }
    default:
        break;
    }
    return peak;
}

struct AudioFormat {
    u16 fmt_type;
    u16 num_channels;
    u32 avg_sample_rate;
    u32 avg_byte_rate;
    u16 block_align;
    u16 bits_per_sample;
    u16 ext_size;
    union {
        u16 valid_bits_per_sample;
        u16 samples_per_block;
        u16 reserved;
    };
    u32 channel_mask;
    u8 guid[16];
    u32 version;
    u8 config_data[4];
    u32 reserved2;
};
static_assert(sizeof(AudioFormat) == 52);

struct SampleData {
    u32 sample_length;
    u32 encoder_delay;
    u32 encoder_delay2;
};
static_assert(sizeof(SampleData) == 12);

struct RIFFHeader {
    u32 riff;
    u32 size;
    u32 wave;
};
static_assert(sizeof(RIFFHeader) == 12);

AjmAt9Decoder::AjmAt9Decoder(AjmFormatEncoding format, AjmAt9CodecFlags flags)
    : m_format(format), m_flags(flags), m_handle(Atrac9GetHandle()) {}

AjmAt9Decoder::~AjmAt9Decoder() {
    Atrac9ReleaseHandle(m_handle);
}

void AjmAt9Decoder::Reset() {
    Atrac9ReleaseHandle(m_handle);
    m_handle = Atrac9GetHandle();
    Atrac9InitDecoder(m_handle, m_config_data);
    Atrac9GetCodecInfo(m_handle, &m_codec_info);

    m_num_frames = 0;
    m_superframe_bytes_remain = m_codec_info.superframeSize;

    // [SNDMOD] re-evaluate the title-theme fingerprint for each new stream on a recycled decoder.
    m_swap_checked = false;
    m_swap_active = false;
}

void AjmAt9Decoder::Initialize(const void* buffer, u32 buffer_size) {
    ASSERT_MSG(buffer_size == sizeof(AjmDecAt9InitializeParameters),
               "Incorrect At9 initialization buffer size {}", buffer_size);
    const auto params = reinterpret_cast<const AjmDecAt9InitializeParameters*>(buffer);
    std::memcpy(m_config_data, params->config_data, ORBIS_AT9_CONFIG_DATA_SIZE);
    AjmAt9Decoder::Reset();
    m_pcm_buffer.resize(m_codec_info.frameSamples * m_codec_info.channels * GetPCMSize(m_format),
                        0);
    m_is_initialized = true;
}

void AjmAt9Decoder::GetInfo(void* out_info) const {
    auto* info = reinterpret_cast<AjmSidebandDecAt9CodecInfo*>(out_info);
    info->super_frame_size = m_codec_info.superframeSize;
    info->frames_in_super_frame = m_codec_info.framesInSuperframe;
    info->next_frame_size = m_superframe_bytes_remain;
    info->frame_samples = m_codec_info.frameSamples;
}

u8 g_at9_guid[] = {0xD2, 0x42, 0xE1, 0x47, 0xBA, 0x36, 0x8D, 0x4D,
                   0x88, 0xFC, 0x61, 0x65, 0x4F, 0x8C, 0x83, 0x6C};

void AjmAt9Decoder::ParseRIFFHeader(std::span<u8>& in_buf, AjmInstanceGapless& gapless) {
    auto* header = reinterpret_cast<RIFFHeader*>(in_buf.data());
    in_buf = in_buf.subspan(sizeof(RIFFHeader));

    ASSERT(header->riff == 'FFIR');
    ASSERT(header->wave == 'EVAW');

    auto* chunk = reinterpret_cast<ChunkHeader*>(in_buf.data());
    in_buf = in_buf.subspan(sizeof(ChunkHeader));
    while (chunk->tag != 'atad') {
        switch (chunk->tag) {
        case ' tmf': {
            ASSERT(chunk->length == sizeof(AudioFormat));
            auto* fmt = reinterpret_cast<AudioFormat*>(in_buf.data());

            ASSERT(fmt->fmt_type == 0xFFFE);
            ASSERT(memcmp(fmt->guid, g_at9_guid, 16) == 0);
            AjmDecAt9InitializeParameters init_params = {};
            std::memcpy(init_params.config_data, fmt->config_data, ORBIS_AT9_CONFIG_DATA_SIZE);
            Initialize(&init_params, sizeof(init_params));
            break;
        }
        case 'tcaf': {
            ASSERT(chunk->length == sizeof(SampleData));
            auto* samples = reinterpret_cast<SampleData*>(in_buf.data());

            gapless.init.total_samples = samples->sample_length;
            gapless.init.skip_samples = samples->encoder_delay;
            gapless.Reset();
            break;
        }
        default:
            break;
        }
        in_buf = in_buf.subspan(chunk->length);

        chunk = reinterpret_cast<ChunkHeader*>(in_buf.data());
        in_buf = in_buf.subspan(sizeof(ChunkHeader));
    }
}

u32 AjmAt9Decoder::GetMinimumInputSize() const {
    return m_superframe_bytes_remain;
}

DecoderResult AjmAt9Decoder::ProcessData(std::span<u8>& in_buf, SparseOutputBuffer& output,
                                         AjmInstanceGapless& gapless) {
    DecoderResult result{};
    if (True(m_flags & AjmAt9CodecFlags::ParseRiffHeader) &&
        *reinterpret_cast<u32*>(in_buf.data()) == 'FFIR') {
        ParseRIFFHeader(in_buf, gapless);
        result.is_reset = true;
    }

    if (!m_is_initialized) {
        result.result = ORBIS_AJM_RESULT_NOT_INITIALIZED;
        return result;
    }

    // [SNDMOD] Detect the title-screen theme by its 32-byte first-frame fingerprint on a stream's
    // first frame. Active() (toggle ON + usable file loaded) is tested first, so the compare never
    // runs with the mod off; only the title theme is targeted.
    if (!m_swap_checked) {
        m_swap_checked = true;
        if (TitleThemeMod::Get().Active()) {
            static constexpr u8 kTitleTheme[32] = {
                0x32, 0x93, 0x05, 0x20, 0xaa, 0x12, 0x9f, 0x9d, 0x23, 0x12, 0xc0,
                0x0b, 0x32, 0x49, 0x55, 0xfb, 0xff, 0xef, 0x89, 0x13, 0x2e, 0xd4,
                0x1c, 0x7e, 0xde, 0x34, 0x07, 0x51, 0x3c, 0x1d, 0x54, 0x42};
            if (in_buf.size() >= sizeof(kTitleTheme) &&
                std::memcmp(in_buf.data(), kTitleTheme, sizeof(kTitleTheme)) == 0) {
                m_swap_active = true;
                LOG_INFO(Lib_Ajm,
                         "[SNDMOD] title-screen theme matched -> muting original, playing via "
                         "direct player");
            }
        }
    }

    int ret = 0;
    int bytes_used = 0;
    switch (m_format) {
    case AjmFormatEncoding::S16:
        ret = Atrac9Decode(m_handle, in_buf.data(), reinterpret_cast<s16*>(m_pcm_buffer.data()),
                           &bytes_used, True(m_flags & AjmAt9CodecFlags::NonInterleavedOutput));
        break;
    case AjmFormatEncoding::S32:
        ret = Atrac9DecodeS32(m_handle, in_buf.data(), reinterpret_cast<s32*>(m_pcm_buffer.data()),
                              &bytes_used, True(m_flags & AjmAt9CodecFlags::NonInterleavedOutput));
        break;
    case AjmFormatEncoding::Float:
        ret =
            Atrac9DecodeF32(m_handle, in_buf.data(), reinterpret_cast<float*>(m_pcm_buffer.data()),
                            &bytes_used, True(m_flags & AjmAt9CodecFlags::NonInterleavedOutput));
        break;
    default:
        UNREACHABLE();
    }
    if (ret != At9Status::ERR_SUCCESS) {
        LOG_ERROR(Lib_Ajm, "Atrac9Decode failed ret = {:#x}", ret);
        result.result = ORBIS_AJM_RESULT_CODEC_ERROR | ORBIS_AJM_RESULT_FATAL;
        result.internal_result = ret;
        return result;
    }

    // [SNDMOD] Mute the original title theme in-place; the separate direct player (its own SDL
    // mixer device) supplies the replacement, bypassing the game's movie-audio pipeline that
    // mangles injected PCM. NotifyFrame() starts it and feeds the watchdog until the theme ends.
    if (m_swap_active) {
        // Measure the original's level BEFORE muting it, and hand it to the player so it can scale
        // the replacement to match (rather than playing a full-scale file far louder than the game
        // ever plays its BGM). Then mute the original in-place.
        const float ref_peak = Sndmod_PeakOf(m_pcm_buffer, m_format);
        std::memset(m_pcm_buffer.data(), 0, m_pcm_buffer.size());
        TitleThemeMod::Get().NotifyFrame(ref_peak);
    }

    result.frames_decoded += 1;
    in_buf = in_buf.subspan(bytes_used);

    m_superframe_bytes_remain -= bytes_used;

    u32 skip_samples = 0;
    if (gapless.current.skip_samples > 0) {
        skip_samples = std::min(u16(m_codec_info.frameSamples), gapless.current.skip_samples);
        gapless.current.skip_samples -= skip_samples;
    }

    const auto max_pcm = gapless.init.total_samples != 0
                             ? gapless.current.total_samples * m_codec_info.channels
                             : std::numeric_limits<u32>::max();

    size_t pcm_written = 0;
    switch (m_format) {
    case AjmFormatEncoding::S16:
        pcm_written = WriteOutputSamples<s16>(output, skip_samples, max_pcm);
        break;
    case AjmFormatEncoding::S32:
        pcm_written = WriteOutputSamples<s32>(output, skip_samples, max_pcm);
        break;
    case AjmFormatEncoding::Float:
        pcm_written = WriteOutputSamples<float>(output, skip_samples, max_pcm);
        break;
    default:
        UNREACHABLE();
    }

    result.samples_written = pcm_written / m_codec_info.channels;
    gapless.current.skipped_samples += m_codec_info.frameSamples - result.samples_written;
    if (gapless.init.total_samples != 0) {
        gapless.current.total_samples -= result.samples_written;
    }

    m_num_frames += 1;
    if ((m_num_frames % m_codec_info.framesInSuperframe) == 0) {
        if (m_superframe_bytes_remain) {
            in_buf = in_buf.subspan(m_superframe_bytes_remain);
        }
        m_superframe_bytes_remain = m_codec_info.superframeSize;
        m_num_frames = 0;
    } else if (gapless.IsEnd()) {
        // GR2FORK PERF: drain the remaining superframe into thread_local scratch instead of a
        // fresh zero-filled vector - decode overwrites the frame and the drain discards output,
        // so neither allocation nor zero-fill is needed. Worker-thread-only; shared sequentially.
        thread_local std::vector<s16> buf;
        buf.resize(static_cast<size_t>(m_codec_info.frameSamples) * m_codec_info.channels);
        while ((m_num_frames % m_codec_info.framesInSuperframe) != 0) {
            ret = Atrac9Decode(m_handle, in_buf.data(), buf.data(), &bytes_used,
                               True(m_flags & AjmAt9CodecFlags::NonInterleavedOutput));
            in_buf = in_buf.subspan(bytes_used);
            m_superframe_bytes_remain -= bytes_used;
            result.frames_decoded += 1;
            m_num_frames += 1;
        }
        in_buf = in_buf.subspan(m_superframe_bytes_remain);
        m_superframe_bytes_remain = m_codec_info.superframeSize;
        m_num_frames = 0;
    }

    return result;
}

AjmSidebandFormat AjmAt9Decoder::GetFormat() const {
    return AjmSidebandFormat{
        .num_channels = u32(m_codec_info.channels),
        .channel_mask = GetChannelMask(u32(m_codec_info.channels)),
        .sampl_freq = u32(m_codec_info.samplingRate),
        .sample_encoding = m_format,
        .bitrate = u32((m_codec_info.samplingRate * m_codec_info.superframeSize * 8) /
                       (m_codec_info.framesInSuperframe * m_codec_info.frameSamples)),
        .reserved = 0,
    };
}

u32 AjmAt9Decoder::GetNextFrameSize(const AjmInstanceGapless& gapless) const {
    const auto skip_samples =
        std::min<u32>(gapless.current.skip_samples, m_codec_info.frameSamples);
    const auto samples =
        gapless.init.total_samples != 0
            ? std::min<u32>(gapless.current.total_samples, m_codec_info.frameSamples - skip_samples)
            : m_codec_info.frameSamples - skip_samples;
    return samples * m_codec_info.channels * GetPCMSize(m_format);
}

} // namespace Libraries::Ajm
