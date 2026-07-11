// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include "common/assert.h"
#include "video_core/amdgpu/pixel_format.h"

namespace AmdGpu {

std::string_view NameOf(DataFormat fmt) {
    switch (fmt) {
    case DataFormat::FormatInvalid:
        return "FormatInvalid";
    case DataFormat::Format8:
        return "Format8";
    case DataFormat::Format16:
        return "Format16";
    case DataFormat::Format8_8:
        return "Format8_8";
    case DataFormat::Format32:
        return "Format32";
    case DataFormat::Format16_16:
        return "Format16_16";
    case DataFormat::Format10_11_11:
        return "Format10_11_11";
    case DataFormat::Format11_11_10:
        return "Format11_11_10";
    case DataFormat::Format10_10_10_2:
        return "Format10_10_10_2";
    case DataFormat::Format2_10_10_10:
        return "Format2_10_10_10";
    case DataFormat::Format8_8_8_8:
        return "Format8_8_8_8";
    case DataFormat::Format32_32:
        return "Format32_32";
    case DataFormat::Format16_16_16_16:
        return "Format16_16_16_16";
    case DataFormat::Format32_32_32:
        return "Format32_32_32";
    case DataFormat::Format32_32_32_32:
        return "Format32_32_32_32";
    case DataFormat::Format5_6_5:
        return "Format5_6_5";
    case DataFormat::Format1_5_5_5:
        return "Format1_5_5_5";
    case DataFormat::Format5_5_5_1:
        return "Format5_5_5_1";
    case DataFormat::Format4_4_4_4:
        return "Format4_4_4_4";
    case DataFormat::Format8_24:
        return "Format8_24";
    case DataFormat::Format24_8:
        return "Format24_8";
    case DataFormat::FormatX24_8_32:
        return "FormatX24_8_32";
    case DataFormat::FormatGB_GR:
        return "FormatGB_GR";
    case DataFormat::FormatBG_RG:
        return "FormatBG_RG";
    case DataFormat::Format5_9_9_9:
        return "Format5_9_9_9";
    case DataFormat::FormatBc1:
        return "FormatBc1";
    case DataFormat::FormatBc2:
        return "FormatBc2";
    case DataFormat::FormatBc3:
        return "FormatBc3";
    case DataFormat::FormatBc4:
        return "FormatBc4";
    case DataFormat::FormatBc5:
        return "FormatBc5";
    case DataFormat::FormatBc6:
        return "FormatBc6";
    case DataFormat::FormatBc7:
        return "FormatBc7";
    default:
        UNREACHABLE();
    }
}

std::string_view NameOf(NumberFormat fmt) {
    switch (fmt) {
    case NumberFormat::Unorm:
        return "Unorm";
    case NumberFormat::Snorm:
        return "Snorm";
    case NumberFormat::Uscaled:
        return "Uscaled";
    case NumberFormat::Sscaled:
        return "Sscaled";
    case NumberFormat::Uint:
        return "Uint";
    case NumberFormat::Sint:
        return "Sint";
    case NumberFormat::SnormNz:
        return "SnormNz";
    case NumberFormat::Float:
        return "Float";
    case NumberFormat::Srgb:
        return "Srgb";
    case NumberFormat::Ubnorm:
        return "Ubnorm";
    case NumberFormat::UbnormNz:
        return "UbnormNz";
    case NumberFormat::Ubint:
        return "Ubint";
    case NumberFormat::Ubscaled:
        return "Unscaled";
    default:
        UNREACHABLE();
    }
}

} // namespace AmdGpu
