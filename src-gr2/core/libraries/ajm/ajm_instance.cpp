// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "ajm_at9.h"
#include "ajm_instance.h"
#include "ajm_mp3.h"
#include "ajm_result.h"
#include "common/config.h"

#include <algorithm>
#include <string>
#include <magic_enum/magic_enum.hpp>

namespace Libraries::Ajm {

u8 GetPCMSize(AjmFormatEncoding format) {
    switch (format) {
    case AjmFormatEncoding::S16:
        return sizeof(s16);
    case AjmFormatEncoding::S32:
        return sizeof(s32);
    case AjmFormatEncoding::Float:
        return sizeof(float);
    default:
        UNREACHABLE();
    }
}

AjmInstance::AjmInstance(AjmCodecType codec_type, AjmInstanceFlags flags)
    : m_flags(flags), m_codec_type(codec_type) {
    switch (codec_type) {
    case AjmCodecType::At9Dec: {
        m_codec = std::make_unique<AjmAt9Decoder>(AjmFormatEncoding(flags.format),
                                                  AjmAt9CodecFlags(flags.codec));
        break;
    }
    case AjmCodecType::Mp3Dec: {
        m_codec = std::make_unique<AjmMp3Decoder>(AjmFormatEncoding(flags.format),
                                                  AjmMp3CodecFlags(flags.codec));
        break;
    }
    default:
        UNREACHABLE_MSG("Unimplemented codec type {}", magic_enum::enum_name(codec_type));
    }
}

void AjmInstance::Reset() {
    m_total_samples = 0;
    m_gapless.Reset();
    m_codec->Reset();
    m_probe_logged = false; // [SNDMOD-PROBE] re-log on the next stream for a recycled instance
}

void AjmInstance::ExecuteJob(AjmJob& job) {
    const auto control_flags = job.flags.control_flags;
    job.output.p_result->result = 0;
    if (True(control_flags & AjmJobControlFlags::Reset)) {
        LOG_TRACE(Lib_Ajm, "Resetting instance {}", job.instance_id);
        Reset();
    }
    if (job.input.init_params.has_value()) {
        LOG_TRACE(Lib_Ajm, "Initializing instance {}", job.instance_id);
        auto& params = job.input.init_params.value();
        m_codec->Initialize(&params, sizeof(params));
    }
    if (job.input.resample_parameters.has_value()) {
        LOG_ERROR(Lib_Ajm, "Unimplemented: resample parameters");
        m_resample_parameters = job.input.resample_parameters.value();
    }
    if (job.input.format.has_value()) {
        LOG_ERROR(Lib_Ajm, "Unimplemented: format parameters");
        m_format = job.input.format.value();
    }
    if (job.input.gapless_decode.has_value()) {
        auto& params = job.input.gapless_decode.value();

        const auto samples_processed =
            m_gapless.init.total_samples - m_gapless.current.total_samples;
        if (params.total_samples != 0 || params.skip_samples == 0) {
            if (params.total_samples >= samples_processed) {
                const auto sample_difference =
                    s64(m_gapless.init.total_samples) - params.total_samples;

                m_gapless.init.total_samples = params.total_samples;
                m_gapless.current.total_samples -= sample_difference;
            } else {
                LOG_WARNING(Lib_Ajm, "ORBIS_AJM_RESULT_INVALID_PARAMETER");
                job.output.p_result->result |= ORBIS_AJM_RESULT_INVALID_PARAMETER;
            }
        }

        const auto samples_skipped = m_gapless.init.skip_samples - m_gapless.current.skip_samples;
        if (params.skip_samples != 0 || params.total_samples == 0) {
            if (params.skip_samples >= samples_skipped) {
                const auto sample_difference =
                    s32(m_gapless.init.skip_samples) - params.skip_samples;

                m_gapless.init.skip_samples = params.skip_samples;
                m_gapless.current.skip_samples -= sample_difference;
            } else {
                LOG_WARNING(Lib_Ajm, "ORBIS_AJM_RESULT_INVALID_PARAMETER");
                job.output.p_result->result |= ORBIS_AJM_RESULT_INVALID_PARAMETER;
            }
        }
    }

    std::span<u8> in_buf = job.input.Data(); // T5.E2
    SparseOutputBuffer out_buf(job.output.buffers);
    auto in_size = in_buf.size();
    auto out_size = out_buf.Size();
    u32 frames_decoded = 0;

    // [SNDMOD-PROBE] Investigation probe gated behind the mod toggle: one line per decode stream
    // dumping the leading input bytes (a stable fingerprint) and the decoded format, so the
    // name->hash table can be built offline from the unpacked PSARC.
    if (!m_probe_logged && in_size != 0 && Config::getGR2TitleThemeMod()) {
        m_probe_logged = true;
        static constexpr char kHex[] = "0123456789abcdef";
        const size_t n = std::min<size_t>(in_size, size_t{32});
        std::string head;
        head.reserve(n * 3);
        for (size_t i = 0; i < n; ++i) {
            const u8 b = in_buf[i];
            head.push_back(kHex[b >> 4]);
            head.push_back(kHex[b & 0xF]);
            head.push_back(' ');
        }
        const AjmSidebandFormat sf = m_codec->GetFormat();
        LOG_INFO(Lib_Ajm,
                 "[SNDMOD-PROBE] decode stream: instance={} codec={} in_size={} channels={} "
                 "sampl_freq={} head=[ {}]",
                 job.instance_id, magic_enum::enum_name(m_codec_type), in_size, sf.num_channels,
                 sf.sampl_freq, head);
    }

    if (in_size != 0) { // T5.E2: via Data()
        for (;;) {
            if (m_flags.gapless_loop && m_gapless.IsEnd()) {
                m_gapless.Reset();
                m_total_samples = 0;
            }
            if (!HasEnoughSpace(out_buf)) {
                LOG_TRACE(Lib_Ajm, "ORBIS_AJM_RESULT_NOT_ENOUGH_ROOM ({} < {})", out_buf.Size(),
                          m_codec->GetNextFrameSize(m_gapless));
                job.output.p_result->result |= ORBIS_AJM_RESULT_NOT_ENOUGH_ROOM;
            }
            if (in_buf.size() < m_codec->GetMinimumInputSize()) {
                job.output.p_result->result |= ORBIS_AJM_RESULT_PARTIAL_INPUT;
            }
            if (job.output.p_result->result != 0) {
                break;
            }
            const auto result = m_codec->ProcessData(in_buf, out_buf, m_gapless);
            if (result.is_reset) {
                m_total_samples = 0;
            } else {
                m_total_samples += result.samples_written;
            }
            frames_decoded += result.frames_decoded;
            if (result.result != 0) {
                job.output.p_result->result |= result.result;
                job.output.p_result->internal_result = result.internal_result;
                break;
            }
            if (False(job.flags.run_flags & AjmJobRunFlags::MultipleFrames)) {
                break;
            }
        }
    }

    if (job.output.p_mframe) {
        job.output.p_mframe->num_frames = frames_decoded;
    }
    if (job.output.p_stream) {
        job.output.p_stream->input_consumed = in_size - in_buf.size();
        job.output.p_stream->output_written = out_size - out_buf.Size();
        job.output.p_stream->total_decoded_samples = m_total_samples;
    }

    if (job.output.p_format != nullptr) {
        *job.output.p_format = m_codec->GetFormat();
    }
    if (job.output.p_gapless_decode != nullptr) {
        *job.output.p_gapless_decode = m_gapless.current;
    }
    if (job.output.p_codec_info != nullptr) {
        m_codec->GetInfo(job.output.p_codec_info);
    }
}

bool AjmInstance::HasEnoughSpace(const SparseOutputBuffer& output) const {
    if (m_gapless.IsEnd()) {
        return true;
    }
    return output.Size() >= m_codec->GetNextFrameSize(m_gapless);
}

} // namespace Libraries::Ajm
