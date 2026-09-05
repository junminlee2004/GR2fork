// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "shader_recompiler/ir/type.h"
#include "video_core/amdgpu/resource.h"

#include <boost/container/static_vector.hpp>

namespace Shader {

static constexpr u32 NUM_USER_DATA_REGS = 16;
static constexpr u32 NUM_IMAGES = 64;
static constexpr u32 NUM_BUFFERS = 40;
static constexpr u32 NUM_SAMPLERS = 16;
static constexpr u32 NUM_FMASKS = 8;

using SharpLocation = u16;

constexpr SharpLocation UNKNOWN_LOCATION = std::numeric_limits<u16>::max();

// Readers take the sharp through Info::flat_ud, never flattened_ud_buf: the fork's
// RefreshFlatBuf leaves that vector empty for walker-less shaders and aliases flat_ud
// to the user-data registers instead of copying them per bind.
template <typename T>
struct SharpFetch {
    static constexpr std::size_t N = sizeof(T) / sizeof(u32);
    static_assert(N <= 8);

    std::array<u32, N> immediates;
    std::array<SharpLocation, N> offsets;
    u8 load_mask;
    // Derived read verdict, never trusted from disk: 1 when the sharp is one contiguous
    // full-mask load with no post-op, so readers take it in place from flat_ud. The owner's
    // ResolveDirectRead sets it after the passes and after a preload; the defaulted
    // operator== compares it, which only the patching-pass dedupe of fresh Infos uses.
    u8 direct{};

    bool operator==(const SharpFetch&) const = default;

    template <u32 num_dwords = N>
        requires(num_dwords <= N)
    constexpr bool Fetch(const u32* flatbuf, T* out) const {
        u8 mask = load_mask;
        for (u32 i = 0; i < num_dwords; i++) {
            if (offsets[i] == UNKNOWN_LOCATION) {
                return false;
            }
        }
        std::array<u32, num_dwords> out_dw;
        for (u32 i = 0; i < num_dwords; i++) {
            out_dw[i] = (mask & 1) ? flatbuf[offsets[i]] : immediates[i];
            mask >>= 1;
        }
        std::memcpy(out, out_dw.data(), sizeof(out_dw));
        return true;
    }

    // True when every dword comes from consecutive flat-buffer slots, so the
    // sharp can be read in place rather than assembled by Fetch.
    constexpr bool IsContiguousLoad() const noexcept {
        if (load_mask != (1u << N) - 1 || offsets[0] == UNKNOWN_LOCATION) {
            return false;
        }
        for (u32 i = 1; i < N; i++) {
            if (offsets[i] != offsets[0] + i) {
                return false;
            }
        }
        return true;
    }
};

enum class SharpFetchPostOp : u8 {
    None,
    // For buffers
    BitwiseOrDw1WithImm,
    OffsetByProgramBase,
    // For images,
    ConvertCubeTo2DArray,
    // For samplers
    DisableAnisoIfSingleLod,
    ForceRepeatXyzClamp,
    ForceLastTexelXyClamp,
    ClearAnisoRatioAndThreshold,
};

enum class BufferType : u8 {
    Guest,
    Flatbuf,
    BdaPagetable,
    FaultBuffer,
    GdsBuffer,
    SharedMemory,
    ClipPlanes,
};

struct BufferResource {
    SharpFetch<AmdGpu::Buffer> sharp_fetch{};
    IR::Type used_types{};
    BufferType buffer_type{};
    bool is_written{};
    bool is_formatted{};
    SharpFetchPostOp post_op{};
    u32 post_op_dw1_mask{};

    bool IsSpecial() const noexcept {
        return buffer_type != BufferType::Guest;
    }

    // Derives sharp_fetch.direct; the offset bound keeps the in-place read inside the flat
    // buffer where Fetch's UNKNOWN_LOCATION check would have rejected the last dwords.
    void ResolveDirectRead() noexcept {
        sharp_fetch.direct = post_op == SharpFetchPostOp::None && sharp_fetch.IsContiguousLoad() &&
                             sharp_fetch.offsets[0] < UNKNOWN_LOCATION - decltype(sharp_fetch)::N;
    }

    constexpr AmdGpu::Buffer GetSharp(const auto& info) const noexcept {
        AmdGpu::Buffer buffer;
        if (!sharp_fetch.Fetch(info.flat_ud, &buffer)) {
            return AmdGpu::Buffer::Null();
        }
        if (post_op == SharpFetchPostOp::BitwiseOrDw1WithImm) {
            reinterpret_cast<u32*>(&buffer)[1] |= post_op_dw1_mask;
        } else if (post_op == SharpFetchPostOp::OffsetByProgramBase) {
            buffer.base_address += info.pgm_base;
        }
        // No logging here: the fmt machinery a log line drags in makes this
        // function too big to inline, and it runs once per descriptor per
        // draw across every bind site.
        if (!buffer.Valid()) [[unlikely]] {
            return AmdGpu::Buffer::Null();
        }
        return buffer;
    }
};
using BufferResourceList = boost::container::static_vector<BufferResource, NUM_BUFFERS>;

enum class MipStorageFallbackMode : u16 {
    None,
    DynamicIndex,
    ConstantIndex,
};

struct ImageResource {
    SharpFetch<AmdGpu::Image> sharp_fetch{};
    bool is_depth{};
    bool is_atomic{};
    bool is_array{};
    bool is_written{};
    bool is_r128{};
    u8 constant_mip_index{};
    MipStorageFallbackMode mip_fallback_mode{};
    SharpFetchPostOp post_op{};

    // Derives sharp_fetch.direct; an r128 T# is half a sharp and always assembles.
    void ResolveDirectRead() noexcept {
        sharp_fetch.direct = !is_r128 && post_op == SharpFetchPostOp::None &&
                             sharp_fetch.IsContiguousLoad() &&
                             sharp_fetch.offsets[0] < UNKNOWN_LOCATION - decltype(sharp_fetch)::N;
    }

    constexpr AmdGpu::Image GetSharp(const auto& info) const noexcept {
        AmdGpu::Image image{};
        if (sharp_fetch.direct) [[likely]] {
            std::memcpy(&image, info.flat_ud + sharp_fetch.offsets[0], sizeof(image));
        } else {
            if (!Fetch(info.flat_ud, &image)) {
                return AmdGpu::Image::Null(is_depth);
            }
            if (post_op == SharpFetchPostOp::ConvertCubeTo2DArray) {
                image.type = u64(AmdGpu::ImageType::Color2DArray);
                image.depth = (image.depth + 1) * 6 - 1;
            }
        }
        // Unlogged for the same inlining reason as the buffer form above.
        if (!image.Valid()) [[unlikely]] {
            image = AmdGpu::Image::Null(is_depth);
        } else if (is_depth) {
            const auto data_fmt = image.GetDataFmt();
            if (data_fmt != AmdGpu::DataFormat::Format16 &&
                data_fmt != AmdGpu::DataFormat::Format32) [[unlikely]] {
                image = AmdGpu::Image::Null(true);
            }
        }
        return image;
    }

    constexpr bool Fetch(const u32* flatbuf, AmdGpu::Image* out) const {
        if (!is_r128) {
            // Fetch full 8 byte T#
            return sharp_fetch.Fetch(flatbuf, out);
        }
        // Fetch r128 T# and fix pitch
        if (!sharp_fetch.Fetch<4>(flatbuf, out)) {
            return false;
        }
        out->pitch = out->width;
        return true;
    }

    // Bind sites that only read the T# take this form: the by-value read
    // forces a 32-byte stack copy whose 8-byte reloads cannot forward. The
    // reference is only valid when the fetch is one contiguous load with no
    // post-op; scratch carries the value on every other arm and for the null
    // fixups, which build the same Null objects the by-value form builds.
    const AmdGpu::Image& GetSharpRef(const auto& info, AmdGpu::Image& scratch) const noexcept {
        if (!sharp_fetch.direct) [[unlikely]] {
            scratch = GetSharp(info);
            return scratch;
        }
        const auto& raw =
            *reinterpret_cast<const AmdGpu::Image*>(info.flat_ud + sharp_fetch.offsets[0]);
        if (!raw.Valid()) [[unlikely]] {
            scratch = AmdGpu::Image::Null(is_depth);
            return scratch;
        }
        if (is_depth) [[unlikely]] {
            const auto data_fmt = raw.GetDataFmt();
            if (data_fmt != AmdGpu::DataFormat::Format16 &&
                data_fmt != AmdGpu::DataFormat::Format32) [[unlikely]] {
                scratch = AmdGpu::Image::Null(true);
                return scratch;
            }
        }
        return raw;
    }

    // Bind sites already holding the decoded T# take this form: the other one
    // re-reads 32 bytes of user data per image per draw to reach the same count.
    u32 NumBindings(const AmdGpu::Image& tsharp) const {
        // A malformed or rejected T# carries unordered 4-bit level fields; the promoted
        // subtraction is signed, so an inverted pair would wrap this u32 to ~4e9 descriptors.
        return (mip_fallback_mode == MipStorageFallbackMode::DynamicIndex &&
                tsharp.last_level >= tsharp.base_level)
                   ? (tsharp.last_level - tsharp.base_level + 1)
                   : 1;
    }

    u32 NumBindings(const auto& info) const {
        return NumBindings(GetSharp(info));
    }
};
using ImageResourceList = boost::container::static_vector<ImageResource, NUM_IMAGES>;

struct SamplerResource {
    SharpFetch<AmdGpu::Sampler> sharp_fetch{};
    SharpFetchPostOp post_op{};
    SharpLocation post_op_tsharp_dw3_off{};

    // Derives sharp_fetch.direct; see BufferResource::ResolveDirectRead.
    void ResolveDirectRead() noexcept {
        sharp_fetch.direct = post_op == SharpFetchPostOp::None && sharp_fetch.IsContiguousLoad() &&
                             sharp_fetch.offsets[0] < UNKNOWN_LOCATION - decltype(sharp_fetch)::N;
    }

    constexpr AmdGpu::Sampler GetSharp(const auto& info) const noexcept {
        AmdGpu::Sampler sampler{};
        sharp_fetch.Fetch(info.flat_ud, &sampler);
        if (post_op == SharpFetchPostOp::DisableAnisoIfSingleLod) {
            const u32 tsharp_dw3 = info.flat_ud[post_op_tsharp_dw3_off];
            if (((tsharp_dw3 >> 12) & 0xff) == 0) {
                sampler.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        } else if (post_op == SharpFetchPostOp::ForceRepeatXyzClamp) {
            sampler.clamp_x.Assign(AmdGpu::ClampMode::Wrap);
            sampler.clamp_y.Assign(AmdGpu::ClampMode::Wrap);
            sampler.clamp_z.Assign(AmdGpu::ClampMode::Wrap);
        } else if (post_op == SharpFetchPostOp::ForceLastTexelXyClamp) {
            sampler.clamp_x.Assign(AmdGpu::ClampMode::ClampLastTexel);
            sampler.clamp_y.Assign(AmdGpu::ClampMode::ClampLastTexel);
            sampler.clamp_z.Assign(AmdGpu::ClampMode::Wrap);
        } else if (post_op == SharpFetchPostOp::ClearAnisoRatioAndThreshold) {
            sampler.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            sampler.aniso_threshold.Assign(0);
        }
        return sampler;
    }
};
using SamplerResourceList = boost::container::static_vector<SamplerResource, NUM_SAMPLERS>;

struct FMaskResource {
    SharpLocation sharp_idx;

    constexpr AmdGpu::Image GetSharp(const auto& info) const noexcept {
        return info.template ReadUdSharp<AmdGpu::Image>(sharp_idx);
    }
};
using FMaskResourceList = boost::container::static_vector<FMaskResource, NUM_FMASKS>;

struct PushData {
    static constexpr u32 XOffsetIndex = 0;
    static constexpr u32 YOffsetIndex = 1;
    static constexpr u32 XScaleIndex = 2;
    static constexpr u32 YScaleIndex = 3;
    static constexpr u32 UdRegsIndex = 4;
    static constexpr u32 BufOffsetIndex = UdRegsIndex + NUM_USER_DATA_REGS / 4;

    float xoffset;
    float yoffset;
    float xscale;
    float yscale;
    std::array<u32, NUM_USER_DATA_REGS> ud_regs;
    std::array<u8, NUM_BUFFERS> buf_offsets;

    void AddOffset(u32 binding, u32 offset) {
        ASSERT(offset < 256 && binding < buf_offsets.size());
        buf_offsets[binding] = offset;
    }
};
static_assert(sizeof(PushData) <= 128,
              "PushData size is greater than minimum size guaranteed by Vulkan spec");

} // namespace Shader
