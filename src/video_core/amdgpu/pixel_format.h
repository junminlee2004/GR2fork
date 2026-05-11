// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <string_view>
#include <fmt/format.h>
#include "common/assert.h"
#include "common/types.h"

namespace AmdGpu {

// Table 8.13 Data and Image Formats [Sea Islands Series Instruction Set Architecture]
enum class DataFormat : u32 {
    FormatInvalid = 0,
    Format8 = 1,
    Format16 = 2,
    Format8_8 = 3,
    Format32 = 4,
    Format16_16 = 5,
    Format10_11_11 = 6,
    Format11_11_10 = 7,
    Format10_10_10_2 = 8,
    Format2_10_10_10 = 9,
    Format8_8_8_8 = 10,
    Format32_32 = 11,
    Format16_16_16_16 = 12,
    Format32_32_32 = 13,
    Format32_32_32_32 = 14,
    Format5_6_5 = 16,
    Format1_5_5_5 = 17,
    Format5_5_5_1 = 18,
    Format4_4_4_4 = 19,
    Format8_24 = 20,
    Format24_8 = 21,
    FormatX24_8_32 = 22,
    FormatGB_GR = 32,
    FormatBG_RG = 33,
    Format5_9_9_9 = 34,
    FormatBc1 = 35,
    FormatBc2 = 36,
    FormatBc3 = 37,
    FormatBc4 = 38,
    FormatBc5 = 39,
    FormatBc6 = 40,
    FormatBc7 = 41,
    FormatFmask8_1 = 47,
    FormatFmask8_2 = 48,
    FormatFmask8_4 = 49,
    FormatFmask16_1 = 50,
    FormatFmask16_2 = 51,
    FormatFmask32_2 = 52,
    FormatFmask32_4 = 53,
    FormatFmask32_8 = 54,
    FormatFmask64_4 = 55,
    FormatFmask64_8 = 56,
    Format4_4 = 57,
    Format6_5_5 = 58,
    Format1 = 59,
    Format1_Reversed = 60,
    Format32_As_8 = 61,
    Format32_As_8_8 = 62,
    Format32_As_32_32_32_32 = 63,
};

enum class NumberFormat : u32 {
    Unorm = 0,
    Snorm = 1,
    Uscaled = 2,
    Sscaled = 3,
    Uint = 4,
    Sint = 5,
    SnormNz = 6,
    Float = 7,
    Srgb = 9,
    Ubnorm = 10,
    UbnormNz = 11,
    Ubint = 12,
    Ubscaled = 13,
};

enum class NumberClass : u8 {
    Float = 0,
    Sint = 1,
    Uint = 2,
};

enum class CompSwizzle : u8 {
    Zero = 0,
    One = 1,
    Red = 4,
    Green = 5,
    Blue = 6,
    Alpha = 7,
};

enum class NumberConversion : u32 {
    None = 0,
    UintToUscaled = 1,
    SintToSscaled = 2,
    UnormToUbnorm = 3,
    Sint8ToSnormNz = 4,
    Sint16ToSnormNz = 5,
    Uint32ToUnorm = 6,
    SrgbToNorm = 7,
};

union CompMapping {
    struct {
        CompSwizzle r;
        CompSwizzle g;
        CompSwizzle b;
        CompSwizzle a;
    };
    std::array<CompSwizzle, 4> array;

    bool operator==(const CompMapping& other) const {
        return array == other.array;
    }

    template <typename T>
    [[nodiscard]] std::array<T, 4> Apply(const std::array<T, 4>& data) const {
        return {
            ApplySingle(data, r),
            ApplySingle(data, g),
            ApplySingle(data, b),
            ApplySingle(data, a),
        };
    }

    [[nodiscard]] u32 ApplyMask(u32 mask) const {
        u32 swizzled_mask{};
        for (u32 i = 0; i < 4; ++i) {
            swizzled_mask |= ((mask >> i) & 1) << Map(i);
        }
        return swizzled_mask;
    }

    [[nodiscard]] CompMapping Inverse() const {
        CompMapping result{};
        InverseSingle(result.r, CompSwizzle::Red);
        InverseSingle(result.g, CompSwizzle::Green);
        InverseSingle(result.b, CompSwizzle::Blue);
        InverseSingle(result.a, CompSwizzle::Alpha);
        return result;
    }

    [[nodiscard]] u32 Map(u32 comp) const {
        const u32 swizzled_comp = u32(array[comp]);
        constexpr u32 min_comp = u32(AmdGpu::CompSwizzle::Red);
        return swizzled_comp >= min_comp ? swizzled_comp - min_comp : comp;
    }

private:
    template <typename T>
    T ApplySingle(const std::array<T, 4>& data, const CompSwizzle swizzle) const {
        switch (swizzle) {
        case CompSwizzle::Zero:
            return T(0);
        case CompSwizzle::One:
            return T(1);
        case CompSwizzle::Red:
            return data[0];
        case CompSwizzle::Green:
            return data[1];
        case CompSwizzle::Blue:
            return data[2];
        case CompSwizzle::Alpha:
            return data[3];
        default:
            UNREACHABLE();
        }
    }

    void InverseSingle(CompSwizzle& dst, const CompSwizzle target) const {
        if (r == target) {
            dst = CompSwizzle::Red;
        } else if (g == target) {
            dst = CompSwizzle::Green;
        } else if (b == target) {
            dst = CompSwizzle::Blue;
        } else if (a == target) {
            dst = CompSwizzle::Alpha;
        } else {
            dst = CompSwizzle::Zero;
        }
    }
};

static constexpr CompMapping IdentityMapping = {
    .r = CompSwizzle::Red,
    .g = CompSwizzle::Green,
    .b = CompSwizzle::Blue,
    .a = CompSwizzle::Alpha,
};

constexpr DataFormat RemapDataFormat(const DataFormat format) {
    switch (format) {
    case DataFormat::Format11_11_10:
        return DataFormat::Format10_11_11;
    case DataFormat::Format10_10_10_2:
        return DataFormat::Format2_10_10_10;
    case DataFormat::Format5_5_5_1:
        return DataFormat::Format1_5_5_5;
    default:
        return format;
    }
}

constexpr NumberFormat RemapNumberFormat(const NumberFormat format, const DataFormat data_format) {
    switch (format) {
    case NumberFormat::Unorm: {
        switch (data_format) {
        case DataFormat::Format32:
        case DataFormat::Format32_32:
        case DataFormat::Format32_32_32:
        case DataFormat::Format32_32_32_32:
            return NumberFormat::Uint;
        default:
            return format;
        }
    }
    case NumberFormat::Srgb:
        return data_format == DataFormat::FormatBc6 ? NumberFormat::Unorm : format;
    case NumberFormat::Uscaled:
        return NumberFormat::Uint;
    case NumberFormat::Sscaled:
    case NumberFormat::SnormNz:
        return NumberFormat::Sint;
    case NumberFormat::Ubnorm:
        return NumberFormat::Unorm;
    case NumberFormat::Float:
        if (data_format == DataFormat::Format8) {
            // Games may ask for 8-bit float when they want to access the stencil component
            // of a depth-stencil image. Change to unsigned int to match the stencil format.
            // This is also the closest approximation to pass the bits through unconverted.
            return NumberFormat::Uint;
        }
        [[fallthrough]];
    default:
        return format;
    }
}

constexpr CompMapping RemapSwizzle(const DataFormat format, const CompMapping swizzle) {
    switch (format) {
    case DataFormat::Format1_5_5_5:
    case DataFormat::Format11_11_10: {
        CompMapping result;
        result.r = swizzle.b;
        result.g = swizzle.g;
        result.b = swizzle.r;
        result.a = swizzle.a;
        return result;
    }
    case DataFormat::Format10_10_10_2: {
        CompMapping result;
        result.r = swizzle.a;
        result.g = swizzle.b;
        result.b = swizzle.g;
        result.a = swizzle.r;
        return result;
    }
    case DataFormat::Format4_4_4_4: {
        // Remap to a more supported component order.
        CompMapping result;
        result.r = swizzle.g;
        result.g = swizzle.b;
        result.b = swizzle.a;
        result.a = swizzle.r;
        return result;
    }
    case DataFormat::Format5_6_5: {
        // Remap to a more supported component order.
        CompMapping result;
        result.r = swizzle.b;
        result.g = swizzle.g;
        result.b = swizzle.r;
        result.a = swizzle.a;
        return result;
    }
    default:
        return swizzle;
    }
}

constexpr NumberConversion MapNumberConversion(const NumberFormat num_fmt,
                                               const DataFormat data_fmt) {
    switch (num_fmt) {
    case NumberFormat::Unorm: {
        switch (data_fmt) {
        case DataFormat::Format32:
        case DataFormat::Format32_32:
        case DataFormat::Format32_32_32:
        case DataFormat::Format32_32_32_32:
            return NumberConversion::Uint32ToUnorm;
        default:
            return NumberConversion::None;
        }
    }
    case NumberFormat::Srgb:
        return data_fmt == DataFormat::FormatBc6 ? NumberConversion::SrgbToNorm
                                                 : NumberConversion::None;
    case NumberFormat::Uscaled:
        return NumberConversion::UintToUscaled;
    case NumberFormat::Sscaled:
        return NumberConversion::SintToSscaled;
    case NumberFormat::Ubnorm:
        return NumberConversion::UnormToUbnorm;
    case NumberFormat::SnormNz: {
        switch (data_fmt) {
        case DataFormat::Format8:
        case DataFormat::Format8_8:
        case DataFormat::Format8_8_8_8:
            return NumberConversion::Sint8ToSnormNz;
        case DataFormat::Format16:
        case DataFormat::Format16_16:
        case DataFormat::Format16_16_16_16:
            return NumberConversion::Sint16ToSnormNz;
        default:
            UNREACHABLE_MSG("data_fmt = {}", u32(data_fmt));
        }
    }
    default:
        return NumberConversion::None;
    }
}

constexpr NumberClass GetNumberClass(const NumberFormat nfmt) {
    switch (nfmt) {
    case NumberFormat::Sint:
        return NumberClass::Sint;
    case NumberFormat::Uint:
        return NumberClass::Uint;
    default:
        return NumberClass::Float;
    }
}

constexpr bool IsRgb(CompSwizzle swizzle) {
    return swizzle == CompSwizzle::Red || swizzle == CompSwizzle::Green ||
           swizzle == CompSwizzle::Blue;
}

constexpr bool IsInteger(const NumberFormat nfmt) {
    return nfmt == NumberFormat::Sint || nfmt == NumberFormat::Uint;
}

constexpr bool IsBlockCoded(DataFormat format) {
    return format >= DataFormat::FormatBc1 && format <= DataFormat::FormatBc7;
}

constexpr bool IsFmask(DataFormat format) {
    return format >= DataFormat::FormatFmask8_1 && format <= DataFormat::FormatFmask64_8;
}

std::string_view NameOf(DataFormat fmt);
std::string_view NameOf(NumberFormat fmt);

// PERF(GR2FORK v1.14): Headerized LUT accessors so call sites can inline the
// array load and constant-fold when format is known at compile time. The .cpp
// previously hid the LUT behind a function call, defeating inlining and
// constant propagation. ImageInfo size accounting and per-binding format
// queries hit these on the BindTextures / FindImage paths.
namespace detail {

inline constexpr std::array<u32, 42> kNumComponentsLut = {
    0, //  0 FormatInvalid
    1, //  1 Format8
    1, //  2 Format16
    2, //  3 Format8_8
    1, //  4 Format32
    2, //  5 Format16_16
    3, //  6 Format10_11_11
    3, //  7 Format11_11_10
    4, //  8 Format10_10_10_2
    4, //  9 Format2_10_10_10
    4, // 10 Format8_8_8_8
    2, // 11 Format32_32
    4, // 12 Format16_16_16_16
    3, // 13 Format32_32_32
    4, // 14 Format32_32_32_32
    0, // 15
    3, // 16 Format5_6_5
    4, // 17 Format1_5_5_5
    4, // 18 Format5_5_5_1
    4, // 19 Format4_4_4_4
    2, // 20 Format8_24
    2, // 21 Format24_8
    2, // 22 FormatX24_8_32
    0, // 23
    0, // 24
    0, // 25
    0, // 26
    0, // 27
    0, // 28
    0, // 29
    0, // 30
    0, // 31
    3, // 32 FormatGB_GR
    3, // 33 FormatBG_RG
    4, // 34 Format5_9_9_9
    4, // 35 FormatBc1
    4, // 36 FormatBc2
    4, // 37 FormatBc3
    1, // 38 FormatBc4
    2, // 39 FormatBc5
    3, // 40 FormatBc6
    4, // 41 FormatBc7
};

inline constexpr std::array<s32, 42> kBitsPerBlockLut = {
    0,   //  0 FormatInvalid
    8,   //  1 Format8
    16,  //  2 Format16
    16,  //  3 Format8_8
    32,  //  4 Format32
    32,  //  5 Format16_16
    32,  //  6 Format10_11_11
    32,  //  7 Format11_11_10
    32,  //  8 Format10_10_10_2
    32,  //  9 Format2_10_10_10
    32,  // 10 Format8_8_8_8
    64,  // 11 Format32_32
    64,  // 12 Format16_16_16_16
    96,  // 13 Format32_32_32
    128, // 14 Format32_32_32_32
    -1,  // 15
    16,  // 16 Format5_6_5
    16,  // 17 Format1_5_5_5
    16,  // 18 Format5_5_5_1
    16,  // 19 Format4_4_4_4
    32,  // 20 Format8_24
    32,  // 21 Format24_8
    64,  // 22 FormatX24_8_32
    -1,  // 23
    -1,  // 24
    -1,  // 25
    -1,  // 26
    -1,  // 27
    -1,  // 28
    -1,  // 29
    -1,  // 30
    -1,  // 31
    16,  // 32 FormatGB_GR
    16,  // 33 FormatBG_RG
    32,  // 34 Format5_9_9_9
    64,  // 35 FormatBc1
    128, // 36 FormatBc2
    128, // 37 FormatBc3
    64,  // 38 FormatBc4
    128, // 39 FormatBc5
    128, // 40 FormatBc6
    128, // 41 FormatBc7
};

inline constexpr std::array<s32, 42> kBitsPerElementLut = {
    0,   //  0 FormatInvalid
    8,   //  1 Format8
    16,  //  2 Format16
    16,  //  3 Format8_8
    32,  //  4 Format32
    32,  //  5 Format16_16
    32,  //  6 Format10_11_11
    32,  //  7 Format11_11_10
    32,  //  8 Format10_10_10_2
    32,  //  9 Format2_10_10_10
    32,  // 10 Format8_8_8_8
    64,  // 11 Format32_32
    64,  // 12 Format16_16_16_16
    96,  // 13 Format32_32_32
    128, // 14 Format32_32_32_32
    -1,  // 15
    16,  // 16 Format5_6_5
    16,  // 17 Format1_5_5_5
    16,  // 18 Format5_5_5_1
    16,  // 19 Format4_4_4_4
    32,  // 20 Format8_24
    32,  // 21 Format24_8
    64,  // 22 FormatX24_8_32
    -1,  // 23
    -1,  // 24
    -1,  // 25
    -1,  // 26
    -1,  // 27
    -1,  // 28
    -1,  // 29
    -1,  // 30
    -1,  // 31
    16,  // 32 FormatGB_GR
    16,  // 33 FormatBG_RG
    32,  // 34 Format5_9_9_9
    4,   // 35 FormatBc1
    8,   // 36 FormatBc2
    8,   // 37 FormatBc3
    4,   // 38 FormatBc4
    8,   // 39 FormatBc5
    8,   // 40 FormatBc6
    8,   // 41 FormatBc7
};

} // namespace detail

[[nodiscard]] inline u32 NumComponents(DataFormat format) noexcept {
    const u32 index = static_cast<u32>(format);
    ASSERT_MSG(index < detail::kNumComponentsLut.size(), "Invalid data format = {}", format);
    return detail::kNumComponentsLut[index];
}

[[nodiscard]] inline u32 NumBitsPerBlock(DataFormat format) noexcept {
    const u32 index = static_cast<u32>(format);
    ASSERT_MSG(index < detail::kBitsPerBlockLut.size(), "Invalid data format = {}", format);
    return static_cast<u32>(detail::kBitsPerBlockLut[index]);
}

[[nodiscard]] inline u32 NumBitsPerElement(DataFormat format) noexcept {
    const u32 index = static_cast<u32>(format);
    ASSERT_MSG(index < detail::kBitsPerElementLut.size(), "Invalid data format = {}", format);
    return static_cast<u32>(detail::kBitsPerElementLut[index]);
}

} // namespace AmdGpu

template <>
struct fmt::formatter<AmdGpu::DataFormat> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }
    auto format(AmdGpu::DataFormat fmt, format_context& ctx) const {
        return fmt::format_to(ctx.out(), "{}", AmdGpu::NameOf(fmt));
    }
};

template <>
struct fmt::formatter<AmdGpu::NumberFormat> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }
    auto format(AmdGpu::NumberFormat fmt, format_context& ctx) const {
        return fmt::format_to(ctx.out(), "{}", AmdGpu::NameOf(fmt));
    }
};
