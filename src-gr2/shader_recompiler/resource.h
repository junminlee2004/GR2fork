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

    bool IsStorage([[maybe_unused]] const AmdGpu::Buffer buffer) const noexcept {
        // When using uniform buffers, a size is required at compilation time, so we need to
        // either compile a lot of shader specializations to handle each size or just force it to
        // the maximum possible size always. However, for some vendors the shader-supplied size is
        // used for bounds checking uniform buffer accesses, so the latter would effectively turn
        // off buffer robustness behavior. Instead, force storage buffers which are bounds checked
        // using the actual buffer size. We are assuming the performance hit from this is
        // acceptable.
        return true; // buffer.GetSize() > profile.max_ubo_size || is_written;
    }

    constexpr AmdGpu::Buffer GetSharp(const auto& info) const noexcept {
        AmdGpu::Buffer buffer{};
        // GR2FORK PERF: inline cbuf is set only for shaders that
        // embed constants directly (rare); the dominant path reads the
        // V# from the user_data sharp slot.
        if (inline_cbuf) [[unlikely]] {
            buffer = inline_cbuf;
            if (inline_cbuf.base_address > 1) {
                buffer.base_address += info.pgm_base; // address fixup
            }
        } else {
            buffer = info.template ReadUdSharp<AmdGpu::Buffer>(sharp_idx);
        }
        // GR2FORK PERF: invalid sharps are an error path -
        // valid V#s dominate steady-state draws.
        if (!buffer.Valid()) [[unlikely]] {
            LOG_DEBUG(Render, "Encountered invalid buffer sharp");
            return AmdGpu::Buffer::Null();
        }
        return buffer;
    }
};
using BufferResourceList = boost::container::static_vector<BufferResource, NUM_BUFFERS>;

// PORT(upstream #4075): IMAGE_STORE_MIP fallback - without VK_AMD_shader_image_load_store_lod
// (or with it disabled for correctness), a shader writing an explicit mip level binds each mip
// as a separate descriptor and indexes into them.
enum class MipStorageFallbackMode : u32 { None, DynamicIndex, ConstantIndex };

struct ImageResource {
    u32 sharp_idx;
    bool is_depth{};
    bool is_atomic{};
    bool is_array{};
    bool is_written{};
    // True when any instruction accesses this binding with raw integer texel coordinates
    // (IMAGE_LOAD / IMAGE_STORE / atomics), merged in Descriptors::Add. A raw-accessed cube T#
    // stays 2D-array (GetViewType): OpImageFetch is invalid for Dim=Cube; Metal cannot write cubes.
    bool is_raw_access{};
    bool is_r128{};
    MipStorageFallbackMode mip_fallback_mode{};
    u32 constant_mip_index{};

    constexpr AmdGpu::Image GetSharp(const auto& info) const noexcept {
        AmdGpu::Image image{};
        // GR2FORK PERF: r128 (read-128-bit-as-buffer-shaped image)
        // is a rarely used shader-image format on shadPS4 - the dominant
        // path reads a regular T# (sharp) from user_data.
        if (!is_r128) [[likely]] {
            image = info.template ReadUdSharp<AmdGpu::Image>(sharp_idx);
        } else {
            const auto raw = info.template ReadUdSharp<u128>(sharp_idx);
            std::memcpy(&image, &raw, sizeof(raw));
        }
        // GR2FORK PERF: invalid T#s are an error path; valid sharps dominate. The is_depth branch
        // fires only for an image-depth instruction on a non-depth format - also an error case.
        if (!image.Valid()) [[unlikely]] {
            LOG_DEBUG(Render_Vulkan, "Encountered invalid image sharp");
            image = AmdGpu::Image::Null(is_depth);
        } else if (is_depth) [[unlikely]] {
            const auto data_fmt = image.GetDataFmt();
            if (data_fmt != AmdGpu::DataFormat::Format16 &&
                data_fmt != AmdGpu::DataFormat::Format32) [[unlikely]] {
                LOG_DEBUG(Render_Vulkan,
                          "Encountered non-depth image used with depth instruction!");
                image = AmdGpu::Image::Null(true);
            }
        }
        return image;
    }

    // PORT(upstream #4075): how many consecutive descriptor slots this image
    // consumes. DynamicIndex mode binds one slot per mip level so the shader
    // can index with a runtime lod value; everything else uses a single slot.
    constexpr u32 NumBindings(const AmdGpu::Image& tsharp) const noexcept {
        return (mip_fallback_mode == MipStorageFallbackMode::DynamicIndex)
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

struct ResolvedStageResources {
    boost::container::static_vector<AmdGpu::Buffer, NUM_BUFFERS> buffers;
    boost::container::static_vector<AmdGpu::Image, NUM_IMAGES> images;
    boost::container::static_vector<AmdGpu::Image, NUM_FMASKS> fmasks;
    boost::container::static_vector<AmdGpu::Sampler, NUM_SAMPLERS> samplers;
    u64 pgm_hash{};
    VAddr pgm_base{};
    u32 stage{};
    u32 l_stage{};
    bool valid{};

    bool Matches(const auto& info) const noexcept {
        return valid && pgm_hash == info.pgm_hash && pgm_base == info.pgm_base &&
               stage == static_cast<u32>(info.stage) &&
               l_stage == static_cast<u32>(info.l_stage) && buffers.size() == info.buffers.size() &&
               images.size() == info.images.size() && fmasks.size() == info.fmasks.size() &&
               samplers.size() == info.samplers.size();
    }
};

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
