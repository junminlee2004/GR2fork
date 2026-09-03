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

enum class BufferType : u32 {
    Guest,
    Flatbuf,
    BdaPagetable,
    FaultBuffer,
    GdsBuffer,
    SharedMemory,
    ClipPlanes,
};

struct Info;

struct BufferResource {
    u32 sharp_idx;
    IR::Type used_types;
    AmdGpu::Buffer inline_cbuf;
    BufferType buffer_type;
    u8 instance_attrib{};
    bool is_written{};
    bool is_formatted{};

    bool IsSpecial() const noexcept {
        return buffer_type != BufferType::Guest;
    }

    constexpr AmdGpu::Buffer GetSharp(const auto& info) const noexcept {
        AmdGpu::Buffer buffer{};
        if (inline_cbuf) {
            buffer = inline_cbuf;
            if (inline_cbuf.base_address != 1) {
                buffer.base_address += info.pgm_base; // address fixup
            }
        } else {
            buffer = info.template ReadUdSharp<AmdGpu::Buffer>(sharp_idx);
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

enum class MipStorageFallbackMode : u32 { None, DynamicIndex, ConstantIndex };

struct ImageResource {
    u32 sharp_idx;
    bool is_depth{};
    bool is_atomic{};
    bool is_array{};
    bool is_written{};
    bool is_r128{};
    MipStorageFallbackMode mip_fallback_mode{};
    u32 constant_mip_index{};

    constexpr AmdGpu::Image GetSharp(const auto& info) const noexcept {
        AmdGpu::Image image{};
        if (!is_r128) {
            image = info.template ReadUdSharp<AmdGpu::Image>(sharp_idx);
        } else {
            const auto raw = info.template ReadUdSharp<u128>(sharp_idx);
            std::memcpy(&image, &raw, sizeof(raw));
            image.pitch = image.width;
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

    // Bind sites that only read the T# take this form: the by-value read
    // forces a 32-byte stack copy whose 8-byte reloads cannot forward. scratch
    // carries the value only on the r128 arm and for the null fixups, which
    // build the same Null objects the by-value form builds.
    const AmdGpu::Image& GetSharpRef(const auto& info, AmdGpu::Image& scratch) const noexcept {
        if (is_r128) [[unlikely]] {
            scratch = GetSharp(info);
            return scratch;
        }
        const auto& raw = info.template ReadUdSharpRef<AmdGpu::Image>(sharp_idx);
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
    u32 sharp_idx;
    AmdGpu::Sampler inline_sampler;
    u32 is_inline_sampler : 1;
    u32 associated_image : 4;
    u32 disable_aniso : 1;

    constexpr AmdGpu::Sampler GetSharp(const auto& info) const noexcept {
        return is_inline_sampler ? inline_sampler
                                 : info.template ReadUdSharp<AmdGpu::Sampler>(sharp_idx);
    }
};
using SamplerResourceList = boost::container::static_vector<SamplerResource, NUM_SAMPLERS>;

struct FMaskResource {
    u32 sharp_idx;

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
