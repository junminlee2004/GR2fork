// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/static_vector.hpp>
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/microinstruction.h"

namespace Shader::Backend::SPIRV {

template <bool is_float>
static Id FixImageCoords(EmitContext& ctx, Id coords, AmdGpu::ImageType image_type) {
    const auto coord_type = is_float ? ctx.F32 : ctx.U32;
    const auto zero = is_float ? ctx.f32_zero_value : ctx.u32_zero_value;

    switch (image_type) {
    case AmdGpu::ImageType::Color1D: {
        // Lowered to 2D with height 1
        const auto x = coords;
        return ctx.OpCompositeConstruct(coord_type[2], x, zero);
    }
    case AmdGpu::ImageType::Color1DArray: {
        // Lowered to 2D array with height 1
        const auto x = ctx.OpCompositeExtract(coord_type[1], coords, 0U);
        const auto slice = ctx.OpCompositeExtract(coord_type[1], coords, 1U);
        return ctx.OpCompositeConstruct(coord_type[3], x, zero, slice);
    }
    case AmdGpu::ImageType::Cube: {
        // Lowered to 2D array
        if (is_float) {
            const auto x = ctx.OpCompositeExtract(coord_type[1], coords, 0U);
            const auto y = ctx.OpCompositeExtract(coord_type[1], coords, 1U);
            const auto face = ctx.OpCompositeExtract(coord_type[1], coords, 2U);

            // AMD cube math results in coordinates in the range [1.0, 2.0]. We need
            // to convert this to the range [0.0, 1.0] to get correct results.
            const auto one = ctx.ConstF32(1.f);
            const auto fixed_x = ctx.OpFSub(coord_type[1], x, one);
            const auto fixed_y = ctx.OpFSub(coord_type[1], y, one);
            const auto fixed_face = ctx.OpFma(
                coord_type[1],
                ctx.OpFloor(coord_type[1], ctx.OpFDiv(coord_type[1], face, ctx.ConstF32(8.f))),
                ctx.ConstF32(-2.f), face);
            return ctx.OpCompositeConstruct(coord_type[3], fixed_x, fixed_y, fixed_face);
        }
        return coords;
    }
    default:
        return coords;
    }
}

template <bool is_float>
static Id PadImageOperand(EmitContext& ctx, Id operand, AmdGpu::ImageType image_type) {
    if (image_type == AmdGpu::ImageType::Color1D || image_type == AmdGpu::ImageType::Color1DArray) {
        const auto coord_type = is_float ? ctx.F32 : ctx.U32;
        const auto zero = is_float ? ctx.f32_zero_value : ctx.u32_zero_value;
        return ctx.OpCompositeConstruct(coord_type[2], operand, zero);
    }
    return operand;
}

struct ImageOperands {
    void Add(spv::ImageOperandsMask new_mask, Id value) {
        if (!Sirit::ValidId(value)) {
            return;
        }
        mask = static_cast<spv::ImageOperandsMask>(static_cast<u32>(mask) |
                                                   static_cast<u32>(new_mask));
        operands.push_back(value);
    }
    void Add(spv::ImageOperandsMask new_mask, Id value1, Id value2) {
        mask = static_cast<spv::ImageOperandsMask>(static_cast<u32>(mask) |
                                                   static_cast<u32>(new_mask));
        operands.push_back(value1);
        operands.push_back(value2);
    }

    void AddOffset(EmitContext& ctx, AmdGpu::ImageType image_type, const IR::Value& offset,
                   bool can_use_runtime_offsets = false) {
        if (offset.IsEmpty()) {
            return;
        }
        spv::ImageOperandsMask op_mask{};
        Id value{};
        if (offset.IsImmediate()) {
            op_mask = spv::ImageOperandsMask::ConstOffset;
            value = ctx.ConstS32(static_cast<s32>(offset.U32()));
        } else {
            IR::Inst* const inst{offset.Inst()};
            if (inst->AreAllArgsImmediates()) {
                switch (inst->GetOpcode()) {
                case IR::Opcode::CompositeConstructU32x2:
                    op_mask = spv::ImageOperandsMask::ConstOffset;
                    value = ctx.ConstS32(static_cast<s32>(inst->Arg(0).U32()),
                                         static_cast<s32>(inst->Arg(1).U32()));
                    break;
                case IR::Opcode::CompositeConstructU32x3:
                    op_mask = spv::ImageOperandsMask::ConstOffset;
                    value = ctx.ConstS32(static_cast<s32>(inst->Arg(0).U32()),
                                         static_cast<s32>(inst->Arg(1).U32()),
                                         static_cast<s32>(inst->Arg(2).U32()));
                    break;
                default:
                    break;
                }
            }
        }

        if (op_mask != spv::ImageOperandsMask::ConstOffset) {
            if (!can_use_runtime_offsets) {
                LOG_WARNING(Render_Vulkan,
                            "Runtime offset provided to unsupported image sample instruction");
                return;
            }
            op_mask = spv::ImageOperandsMask::Offset;
            value = ctx.Def(offset);
        }

        value = PadImageOperand<false>(ctx, value, image_type);
        Add(op_mask, value);
    }

    void AddDerivatives(EmitContext& ctx, AmdGpu::ImageType image_type, Id derivatives_dx,
                        Id derivatives_dy) {
        if (!Sirit::ValidId(derivatives_dx) || !Sirit::ValidId(derivatives_dy)) {
            return;
        }
        derivatives_dx = PadImageOperand<true>(ctx, derivatives_dx, image_type);
        derivatives_dy = PadImageOperand<true>(ctx, derivatives_dy, image_type);
        Add(spv::ImageOperandsMask::Grad, derivatives_dx, derivatives_dy);
    }

    spv::ImageOperandsMask mask{};
    boost::container::static_vector<Id, 4> operands;
};

/// Arguments of a cube lookup besides its coordinates. An invalid id means the lookup has none.
struct CubeLookup {
    Id dref{};
    Id lod{};
    Id bias{};
    Id derivatives_dx{};
    Id derivatives_dy{};
};

static bool IsCubeLookup(const EmitContext& ctx, u32 handle) {
    return ctx.info.samplers[handle >> 16].cube == CubeSampling::Face;
}

// A cube face is named by its outward axis n, a signed unit vector. Its coordinates grow along:
//          +X          -X          +Y          -Y          +Z          -Z
//   s   (0, 0,-1)   (0, 0, 1)   (1, 0, 0)   (1, 0, 0)   (1, 0, 0)   (-1, 0, 0)
//   t   (0,-1, 0)   (0,-1, 0)   (0, 0, 1)   (0, 0,-1)   (0,-1, 0)   (0,-1, 0)
static Id CubeFaceS(EmitContext& ctx, Id n) {
    const Id x{ctx.OpCompositeExtract(ctx.F32[1], n, 0U)};
    const Id y{ctx.OpCompositeExtract(ctx.F32[1], n, 1U)};
    const Id z{ctx.OpCompositeExtract(ctx.F32[1], n, 2U)};
    return ctx.OpCompositeConstruct(ctx.F32[3],
                                    ctx.OpFAdd(ctx.F32[1], ctx.OpFAbs(ctx.F32[1], y), z),
                                    ctx.f32_zero_value, ctx.OpFNegate(ctx.F32[1], x));
}

static Id CubeFaceT(EmitContext& ctx, Id n) {
    const Id x{ctx.OpCompositeExtract(ctx.F32[1], n, 0U)};
    const Id y{ctx.OpCompositeExtract(ctx.F32[1], n, 1U)};
    const Id z{ctx.OpCompositeExtract(ctx.F32[1], n, 2U)};
    const Id along_y{ctx.OpFAdd(ctx.F32[1], ctx.OpFAbs(ctx.F32[1], x), ctx.OpFAbs(ctx.F32[1], z))};
    return ctx.OpCompositeConstruct(ctx.F32[3], ctx.f32_zero_value,
                                    ctx.OpFNegate(ctx.F32[1], along_y), y);
}

/// Samples a cube that is bound as a 2D array the way the hardware samples a cube.
///
/// The level of detail comes from the point on the cube surface, which unlike the per-face
/// coordinates is continuous from one face to the next. Where the filter can reach a face edge
/// the lookup becomes a sum of taps on the faces around that edge, all taken at the same point
/// with each face's plane unfolded into the others'. A tap finds nothing outside its own face,
/// so together the taps make one filter kernel that carries on across the edge.
static Id EmitCubeSample(EmitContext& ctx, u32 handle, Id coords, const CubeLookup& args) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const u32 sampler_index = handle >> 16;
    const auto& sampler_res = ctx.info.samplers[sampler_index];
    const CubeFilter filter = sampler_res.GetCubeFilter(sampler_res.GetSharp(ctx.info));
    const bool is_dref = Sirit::ValidId(args.dref);
    const Id f32 = ctx.F32[1];
    const Id vec2 = ctx.F32[2];
    const Id vec3 = ctx.F32[3];
    const Id result_type = texture.data_types->Get(is_dref ? 1 : 4);
    const Id zero = ctx.f32_zero_value;
    const Id half = ctx.ConstF32(0.5f);
    const Id one = ctx.ConstF32(1.f);
    const Id two = ctx.ConstF32(2.f);
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id face_sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[sampler_index]);

    // GCN passes (s + 1, t + 1, slice * 8 + face). The faces are +X, -X, +Y, -Y, +Z, -Z.
    const Id x = ctx.OpCompositeExtract(f32, coords, 0U);
    const Id y = ctx.OpCompositeExtract(f32, coords, 1U);
    const Id packed_face = ctx.OpCompositeExtract(f32, coords, 2U);
    const Id slice = ctx.OpFloor(f32, ctx.OpFDiv(f32, packed_face, ctx.ConstF32(8.f)));
    const Id face = ctx.OpFma(f32, slice, ctx.ConstF32(-8.f), packed_face);
    const Id first_layer = ctx.OpFMul(f32, slice, ctx.ConstF32(6.f));
    const Id face_coords = ctx.OpCompositeConstruct(
        vec3, ctx.OpFSub(f32, x, one), ctx.OpFSub(f32, y, one), ctx.OpFAdd(f32, first_layer, face));

    // The point on the cube surface, with the face coordinates rescaled to [-1, 1].
    const Id sc = ctx.OpFma(f32, x, two, ctx.ConstF32(-3.f));
    const Id tc = ctx.OpFma(f32, y, two, ctx.ConstF32(-3.f));
    const Id axis = ctx.OpFloor(f32, ctx.OpFMul(f32, face, half));
    const Id is_negative = ctx.OpFma(f32, axis, ctx.ConstF32(-2.f), face);
    const Id sign = ctx.OpFma(f32, is_negative, ctx.ConstF32(-2.f), one);
    const Id n_x = ctx.OpSelect(f32, ctx.OpFOrdLessThan(ctx.U1[1], axis, half), sign, zero);
    const Id n_z =
        ctx.OpSelect(f32, ctx.OpFOrdGreaterThan(ctx.U1[1], axis, ctx.ConstF32(1.5f)), sign, zero);
    const Id n_y = ctx.OpFSub(f32, ctx.OpFSub(f32, sign, n_x), n_z);
    const Id n = ctx.OpCompositeConstruct(vec3, n_x, n_y, n_z);
    const Id es = CubeFaceS(ctx, n);
    const Id et = CubeFaceT(ctx, n);
    const Id q = ctx.OpFAdd(vec3, n,
                            ctx.OpFAdd(vec3, ctx.OpVectorTimesScalar(vec3, es, sc),
                                       ctx.OpVectorTimesScalar(vec3, et, tc)));

    Id lod = args.lod;
    Id dx = args.derivatives_dx;
    Id dy = args.derivatives_dy;
    if (!Sirit::ValidId(lod) && !Sirit::ValidId(dx)) {
        if (ctx.hw_stage == HwStage::Fragment) {
            // Project the change of the surface point onto the plane of this pixel's face.
            const auto project = [&](Id d) {
                const Id d_n = ctx.OpDot(f32, d, n);
                const Id d_s = ctx.OpFSub(f32, ctx.OpDot(f32, d, es), ctx.OpFMul(f32, sc, d_n));
                const Id d_t = ctx.OpFSub(f32, ctx.OpDot(f32, d, et), ctx.OpFMul(f32, tc, d_n));
                return ctx.OpCompositeConstruct(vec2, ctx.OpFMul(f32, d_s, half),
                                                ctx.OpFMul(f32, d_t, half));
            };
            dx = project(ctx.OpDPdx(vec3, q));
            dy = project(ctx.OpDPdy(vec3, q));
            if (Sirit::ValidId(args.bias)) {
                const Id scale = ctx.OpExp2(f32, args.bias);
                dx = ctx.OpVectorTimesScalar(vec2, dx, scale);
                dy = ctx.OpVectorTimesScalar(vec2, dy, scale);
            }
        } else {
            lod = Sirit::ValidId(args.bias) ? args.bias : zero;
        }
    }
    const bool has_lod = Sirit::ValidId(lod);

    const auto sample = [&](Id sampler, Id coord, Id grad_x, Id grad_y) {
        const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
        ImageOperands operands;
        if (has_lod) {
            operands.Add(spv::ImageOperandsMask::Lod, lod);
        } else {
            operands.Add(spv::ImageOperandsMask::Grad, grad_x, grad_y);
        }
        if (is_dref) {
            return ctx.OpImageSampleDrefExplicitLod(result_type, sampled_image, coord, args.dref,
                                                    operands.mask, operands.operands);
        }
        return ctx.OpImageSampleExplicitLod(result_type, sampled_image, coord, operands.mask,
                                            operands.operands);
    };
    const auto finish = [&](Id value) {
        if (is_dref) {
            return ctx.OpCompositeConstruct(ctx.F32[4], value, zero, zero, zero);
        }
        return value;
    };
    if (!filter.seamless) {
        return finish(sample(face_sampler, face_coords, dx, dy));
    }

    // How far the filter reaches from the sample point, in the units of sc and tc: half a texel
    // of the coarsest level the sampler may use, plus half the footprint an anisotropic filter
    // spreads its taps along.
    const Id size = ctx.OpConvertUToF(
        f32, ctx.OpCompositeExtract(
                 ctx.U32[1], ctx.OpImageQuerySizeLod(ctx.U32[3], image, ctx.u32_zero_value), 0U));
    const Id last_level = ctx.OpConvertUToF(
        f32, ctx.OpISub(ctx.U32[1], ctx.OpImageQueryLevels(ctx.U32[1], image), ctx.u32_one_value));
    Id footprint = zero;
    Id level = lod;
    if (!has_lod) {
        footprint = ctx.OpFMax(f32, ctx.OpLength(f32, dx), ctx.OpLength(f32, dy));
        level =
            ctx.OpLog2(f32, ctx.OpFMax(f32, ctx.OpFMul(f32, footprint, size), ctx.ConstF32(1e-6f)));
    }
    const Id blended_level = ctx.OpFAdd(
        f32, ctx.OpFloor(f32, ctx.OpFAdd(f32, level, ctx.ConstF32(filter.lod_bias + 0.5f))), one);
    const Id coarsest_level = ctx.OpFClamp(
        f32, ctx.OpFMax(f32, blended_level, ctx.ConstF32(filter.min_lod + 1.f)), zero, last_level);
    const Id texels = ctx.OpFMax(
        f32,
        ctx.OpFloor(f32,
                    ctx.OpFMul(f32, size, ctx.OpExp2(f32, ctx.OpFNegate(f32, coarsest_level)))),
        one);
    const Id reach = ctx.OpFAdd(f32, ctx.OpFDiv(f32, one, texels), footprint);
    const Id gap_u = ctx.OpFSub(f32, one, ctx.OpFAbs(f32, sc));
    const Id gap_v = ctx.OpFSub(f32, one, ctx.OpFAbs(f32, tc));
    const Id near_edge = ctx.OpLogicalOr(ctx.U1[1], ctx.OpFOrdLessThan(ctx.U1[1], gap_u, reach),
                                         ctx.OpFOrdLessThan(ctx.U1[1], gap_v, reach));

    const Id interior_label = ctx.OpLabel();
    const Id edge_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(near_edge, edge_label, interior_label);

    ctx.AddLabel(interior_label);
    const Id interior = sample(face_sampler, face_coords, dx, dy);
    ctx.OpBranch(merge_label);

    ctx.AddLabel(edge_label);
    const Id edge_sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[sampler_index + 1]);
    // The taps only add up if they all blend the same levels with the same kernel. The edge
    // sampler is never anisotropic, so from this pixel's gradients every tap gets the levels the
    // host picks for a single tap.
    Id edge_dx = dx;
    Id edge_dy = dy;
    if (!has_lod && filter.anisotropic) {
        // An anisotropic filter takes its level from the short axis of the footprint and makes
        // up for it with taps along the long one. One tap can stand in for two of those, so
        // shrink the gradients by the ratio of the axes, to half at most.
        const Id xx = ctx.OpDot(f32, dx, dx);
        const Id yy = ctx.OpDot(f32, dy, dy);
        const Id xy = ctx.OpDot(f32, dx, dy);
        const Id diff = ctx.OpFSub(f32, xx, yy);
        const Id root = ctx.OpSqrt(
            f32, ctx.OpFAdd(f32, ctx.OpFMul(f32, diff, diff),
                            ctx.OpFMul(f32, ctx.ConstF32(4.f), ctx.OpFMul(f32, xy, xy))));
        const Id sum = ctx.OpFAdd(f32, xx, yy);
        const Id major_sq = ctx.OpFMax(f32, ctx.OpFAdd(f32, sum, root), ctx.ConstF32(1e-30f));
        const Id minor_sq = ctx.OpFMax(f32, ctx.OpFSub(f32, sum, root), zero);
        const Id scale =
            ctx.OpFClamp(f32, ctx.OpSqrt(f32, ctx.OpFDiv(f32, minor_sq, major_sq)), half, one);
        edge_dx = ctx.OpVectorTimesScalar(vec2, dx, scale);
        edge_dy = ctx.OpVectorTimesScalar(vec2, dy, scale);
    }
    const auto tap = [&](Id n_f, Id p) {
        const Id s = ctx.OpFma(f32, ctx.OpDot(f32, p, CubeFaceS(ctx, n_f)), half, half);
        const Id t = ctx.OpFma(f32, ctx.OpDot(f32, p, CubeFaceT(ctx, n_f)), half, half);
        const Id points_back =
            ctx.OpFOrdLessThan(ctx.U1[1], ctx.OpDot(f32, n_f, ctx.ConstF32(1.f, 1.f, 1.f)), zero);
        const Id index =
            ctx.OpFAdd(f32, ctx.OpDot(f32, ctx.OpFAbs(vec3, n_f), ctx.ConstF32(0.f, 2.f, 4.f)),
                       ctx.OpSelect(f32, points_back, one, zero));
        const Id coord = ctx.OpCompositeConstruct(vec3, s, t, ctx.OpFAdd(f32, first_layer, index));
        return sample(edge_sampler, coord, edge_dx, edge_dy);
    };
    // The faces across the two nearest edges, and the sample point in each of their planes.
    const Id minus_one = ctx.ConstF32(-1.f);
    const Id n_u = ctx.OpVectorTimesScalar(
        vec3, es, ctx.OpSelect(f32, ctx.OpFOrdLessThan(ctx.U1[1], sc, zero), minus_one, one));
    const Id n_v = ctx.OpVectorTimesScalar(
        vec3, et, ctx.OpSelect(f32, ctx.OpFOrdLessThan(ctx.U1[1], tc, zero), minus_one, one));
    const Id p_u =
        ctx.OpFAdd(vec3, q, ctx.OpVectorTimesScalar(vec3, ctx.OpFAdd(vec3, n, n_u), gap_u));
    const Id p_v =
        ctx.OpFAdd(vec3, q, ctx.OpVectorTimesScalar(vec3, ctx.OpFAdd(vec3, n, n_v), gap_v));
    const Id tap_face = sample(edge_sampler, face_coords, edge_dx, edge_dy);
    const Id tap_u = tap(n_u, p_u);
    const Id tap_v = tap(n_v, p_v);
    // No face has the fourth texel at a cube corner; it is the average of the other three.
    // Mirrored through the corner, each face returns its corner texel times that texel's weight.
    const Id past_u = ctx.OpVectorTimesScalar(vec3, n_u, ctx.OpFMul(f32, gap_u, two));
    const Id past_v = ctx.OpVectorTimesScalar(vec3, n_v, ctx.OpFMul(f32, gap_v, two));
    const Id corner_face = tap(n, ctx.OpFAdd(vec3, q, ctx.OpFAdd(vec3, past_u, past_v)));
    const Id corner_u = tap(n_u, ctx.OpFAdd(vec3, p_u, past_v));
    const Id corner_v = tap(n_v, ctx.OpFAdd(vec3, p_v, past_u));
    const Id corners =
        ctx.OpFAdd(result_type, ctx.OpFAdd(result_type, corner_face, corner_u), corner_v);
    const Id third = ctx.ConstF32(1.f / 3.f);
    Id edge = ctx.OpFAdd(result_type, ctx.OpFAdd(result_type, tap_face, tap_u), tap_v);
    edge = ctx.OpFAdd(result_type, edge,
                      is_dref ? ctx.OpFMul(result_type, corners, third)
                              : ctx.OpVectorTimesScalar(result_type, corners, third));
    if (is_dref) {
        edge = ctx.OpFClamp(result_type, edge, zero, one);
    } else {
        // The view swizzles border texels too, so a component that reads as one counted per tap.
        const auto swizzle = ctx.info.images[handle & 0xFFFF].GetSharp(ctx.info).DstSelect();
        for (u32 i = 0; i < 4; i++) {
            if (swizzle.array[i] == AmdGpu::CompSwizzle::One) {
                edge = ctx.OpCompositeInsert(result_type, one, edge, i);
            }
        }
    }
    ctx.OpBranch(merge_label);

    ctx.AddLabel(merge_label);
    return finish(ctx.OpPhi(result_type, interior, interior_label, edge, edge_label));
}

Id EmitImageHandle(EmitContext& ctx, Id, Id) {
    UNREACHABLE_MSG("Unreachable instruction");
}

Id EmitImageSampleRaw(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address1, Id address2,
                      Id address3, Id address4) {
    UNREACHABLE_MSG("Unreachable instruction");
}

Id EmitImageSampleImplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id bias,
                              const IR::Value& offset) {
    if (IsCubeLookup(ctx, handle)) {
        return EmitCubeSample(ctx, handle, coords, {.bias = bias});
    }
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const Id fixed_coords = FixImageCoords<true>(ctx, coords, texture.view_type);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Bias, bias);
    operands.AddOffset(ctx, texture.view_type, offset);
    const Id sample = ctx.OpImageSampleImplicitLod(result_type, sampled_image, fixed_coords,
                                                   operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageSampleExplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod,
                              const IR::Value& offset) {
    if (IsCubeLookup(ctx, handle)) {
        return EmitCubeSample(ctx, handle, coords, {.lod = lod});
    }
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const Id fixed_coords = FixImageCoords<true>(ctx, coords, texture.view_type);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, texture.view_type, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(result_type, sampled_image, fixed_coords,
                                                   operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageSampleDrefImplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id dref,
                                  Id bias, const IR::Value& offset) {
    if (IsCubeLookup(ctx, handle)) {
        return EmitCubeSample(ctx, handle, coords, {.dref = dref, .bias = bias});
    }
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(1);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const Id fixed_coords = FixImageCoords<true>(ctx, coords, texture.view_type);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Bias, bias);
    operands.AddOffset(ctx, texture.view_type, offset);
    const Id sample = ctx.OpImageSampleDrefImplicitLod(result_type, sampled_image, fixed_coords,
                                                       dref, operands.mask, operands.operands);
    const Id sample_typed = texture.is_integer ? ctx.OpBitcast(ctx.F32[1], sample) : sample;
    return ctx.OpCompositeConstruct(ctx.F32[4], sample_typed, ctx.f32_zero_value,
                                    ctx.f32_zero_value, ctx.f32_zero_value);
}

Id EmitImageSampleDrefExplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id dref,
                                  Id lod, const IR::Value& offset) {
    if (IsCubeLookup(ctx, handle)) {
        return EmitCubeSample(ctx, handle, coords, {.dref = dref, .lod = lod});
    }
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(1);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const Id fixed_coords = FixImageCoords<true>(ctx, coords, texture.view_type);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, texture.view_type, offset);
    const Id sample = ctx.OpImageSampleDrefExplicitLod(result_type, sampled_image, fixed_coords,
                                                       dref, operands.mask, operands.operands);
    const Id sample_typed = texture.is_integer ? ctx.OpBitcast(ctx.F32[1], sample) : sample;
    return ctx.OpCompositeConstruct(ctx.F32[4], sample_typed, ctx.f32_zero_value,
                                    ctx.f32_zero_value, ctx.f32_zero_value);
}

Id EmitImageGather(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords,
                   const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const Id fixed_coords = FixImageCoords<true>(ctx, coords, texture.view_type);
    const u32 comp = inst->Flags<IR::TextureInstInfo>().gather_comp.Value();
    ImageOperands operands;
    operands.AddOffset(ctx, texture.view_type, offset, true);
    const Id texels = ctx.OpImageGather(result_type, sampled_image, fixed_coords,
                                        ctx.ConstU32(comp), operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], texels) : texels;
}

Id EmitImageGatherDref(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords,
                       const IR::Value& offset, Id dref) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const Id fixed_coords = FixImageCoords<true>(ctx, coords, texture.view_type);
    ImageOperands operands;
    operands.AddOffset(ctx, texture.view_type, offset, true);
    const Id texels = ctx.OpImageDrefGather(result_type, sampled_image, fixed_coords, dref,
                                            operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], texels) : texels;
}

Id EmitImageQueryDimensions(EmitContext& ctx, IR::Inst* inst, u32 handle, Id lod, bool has_mips) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id zero = ctx.u32_zero_value;
    const auto mips{[&] { return has_mips ? ctx.OpImageQueryLevels(ctx.U32[1], image) : zero; }};
    const bool uses_lod{texture.view_type != AmdGpu::ImageType::Color2DMsaa && !texture.is_storage};
    const auto query{[&](Id type) {
        return uses_lod ? ctx.OpImageQuerySizeLod(type, image, lod)
                        : ctx.OpImageQuerySize(type, image);
    }};
    switch (texture.view_type) {
    case AmdGpu::ImageType::Color1D: {
        const auto width = ctx.OpCompositeExtract(ctx.U32[1], query(ctx.U32[2]), 0U);
        return ctx.OpCompositeConstruct(ctx.U32[4], width, zero, zero, mips());
    }
    case AmdGpu::ImageType::Color1DArray: {
        const auto base = query(ctx.U32[3]);
        const auto width = ctx.OpCompositeExtract(ctx.U32[1], base, 0U);
        const auto slices = ctx.OpCompositeExtract(ctx.U32[1], base, 2U);
        return ctx.OpCompositeConstruct(ctx.U32[4], width, slices, zero, mips());
    }
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[2]), zero, mips());
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Cube:
    case AmdGpu::ImageType::Color3D:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[3]), mips());
    default:
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
}

Id EmitImageQueryLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const Id fixed_coords = FixImageCoords<true>(ctx, coords, texture.view_type);
    return ctx.OpImageQueryLod(ctx.F32[2], sampled_image, fixed_coords);
}

Id EmitImageGradient(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id derivatives_dx,
                     Id derivatives_dy, const IR::Value& offset, const IR::Value& lod_clamp) {
    if (IsCubeLookup(ctx, handle)) {
        return EmitCubeSample(ctx, handle, coords,
                              {.derivatives_dx = derivatives_dx, .derivatives_dy = derivatives_dy});
    }
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const Id fixed_coords = FixImageCoords<true>(ctx, coords, texture.view_type);
    ImageOperands operands;
    operands.AddDerivatives(ctx, texture.view_type, derivatives_dx, derivatives_dy);
    operands.AddOffset(ctx, texture.view_type, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(result_type, sampled_image, fixed_coords,
                                                   operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageRead(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod, Id ms) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id color_type = texture.data_types->Get(4);
    const Id fixed_coords = FixImageCoords<false>(ctx, coords, texture.view_type);
    ImageOperands operands;
    Id texel;
    if (!texture.is_storage) {
        const Id image = ctx.OpLoad(texture.image_type, texture.id);
        if (texture.view_type == AmdGpu::ImageType::Color2DMsaa) {
            // GCN hardware wraps out-of-range MSAA sample indices
            if (Sirit::ValidId(ms)) {
                const Id sample_count = ctx.OpImageQuerySamples(ctx.U32[1], image);
                const Id wrapped_ms = ctx.OpUMod(ctx.U32[1], ms, sample_count);
                operands.Add(spv::ImageOperandsMask::Sample, wrapped_ms);
            }
        } else {
            if (Sirit::ValidId(ms)) {
                LOG_ERROR(Render_Recompiler, "image is not MS but ms operand is provided");
            }
            operands.Add(spv::ImageOperandsMask::Lod, lod);
            operands.Add(spv::ImageOperandsMask::Sample, ms);
        }
        texel = ctx.OpImageFetch(color_type, image, fixed_coords, operands.mask, operands.operands);
    } else {
        Id image_ptr = texture.id;
        if (ctx.profile.supports_image_load_store_lod) {
            operands.Add(spv::ImageOperandsMask::Lod, lod);
        } else if (Sirit::ValidId(lod)) {
#if 1
            // It's confusing what interactions will cause this code path so leave it as
            // unreachable until a case is found.
            // Normally IMAGE_LOAD_MIP should translate -> OpImageFetch
            UNREACHABLE_MSG("Unsupported ImageRead with Lod");
#else
            LOG_WARNING(Render, "Fallback for ImageRead with LOD");
            ASSERT(texture.mip_fallback_mode == MipStorageFallbackMode::DynamicIndex);
            const Id single_image_ptr_type =
                ctx.TypePointer(spv::StorageClass::UniformConstant, texture.image_type);
            image_ptr = ctx.OpAccessChain(single_image_ptr_type, image_ptr, std::array{lod});
#endif
        }
        const Id image = ctx.OpLoad(texture.image_type, image_ptr);
        texel = ctx.OpImageRead(color_type, image, fixed_coords, operands.mask, operands.operands);
    }
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], texel) : texel;
}

void EmitImageWrite(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod, Id ms,
                    Id color) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    Id image_ptr = texture.id;
    const Id color_type = texture.data_types->Get(4);
    const Id fixed_coords = FixImageCoords<false>(ctx, coords, texture.view_type);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Sample, ms);
    if (ctx.profile.supports_image_load_store_lod) {
        operands.Add(spv::ImageOperandsMask::Lod, lod);
    } else if (Sirit::ValidId(lod)) {
        LOG_WARNING(Render, "Fallback for ImageWrite with LOD");
        ASSERT(texture.mip_fallback_mode == MipStorageFallbackMode::DynamicIndex);
        const Id single_image_ptr_type =
            ctx.TypePointer(spv::StorageClass::UniformConstant, texture.image_type);
        image_ptr = ctx.OpAccessChain(single_image_ptr_type, image_ptr, std::array{lod});
    }
    const Id image = ctx.OpLoad(texture.image_type, image_ptr);
    const Id texel = texture.is_integer ? ctx.OpBitcast(color_type, color) : color;
    ctx.OpImageWrite(image, fixed_coords, texel, operands.mask, operands.operands);
}

static Id SelectCubeResult(EmitContext& ctx, Id x, Id y, Id z, Id x_res, Id y_res, Id z_res) {
    const auto abs_x = ctx.OpFAbs(ctx.F32[1], x);
    const auto abs_y = ctx.OpFAbs(ctx.F32[1], y);
    const auto abs_z = ctx.OpFAbs(ctx.F32[1], z);

    const auto z_face_cond{ctx.OpLogicalAnd(ctx.U1[1],
                                            ctx.OpFOrdGreaterThanEqual(ctx.U1[1], abs_z, abs_x),
                                            ctx.OpFOrdGreaterThanEqual(ctx.U1[1], abs_z, abs_y))};
    const auto y_face_cond{ctx.OpFOrdGreaterThanEqual(ctx.U1[1], abs_y, abs_x)};

    return ctx.OpSelect(ctx.F32[1], z_face_cond, z_res,
                        ctx.OpSelect(ctx.F32[1], y_face_cond, y_res, x_res));
}

Id EmitCubeFaceIndex(EmitContext& ctx, IR::Inst* inst, Id x, Id y, Id z) {
    if (ctx.profile.supports_native_cube_calc) {
        return ctx.OpCubeFaceIndexAMD(ctx.F32[1], ctx.OpCompositeConstruct(ctx.F32[3], x, y, z));
    }

    const auto x_neg_cond{ctx.OpFOrdLessThan(ctx.U1[1], x, ctx.f32_zero_value)};
    const auto y_neg_cond{ctx.OpFOrdLessThan(ctx.U1[1], y, ctx.f32_zero_value)};
    const auto z_neg_cond{ctx.OpFOrdLessThan(ctx.U1[1], z, ctx.f32_zero_value)};
    const auto x_face{ctx.OpSelect(ctx.F32[1], x_neg_cond, ctx.ConstF32(1.f), ctx.ConstF32(0.f))};
    const auto y_face{ctx.OpSelect(ctx.F32[1], y_neg_cond, ctx.ConstF32(3.f), ctx.ConstF32(2.f))};
    const auto z_face{ctx.OpSelect(ctx.F32[1], z_neg_cond, ctx.ConstF32(5.f), ctx.ConstF32(4.f))};

    return SelectCubeResult(ctx, x, y, z, x_face, y_face, z_face);
}

Id EmitCubeFaceCoordS(EmitContext& ctx, IR::Inst* inst, Id x, Id y, Id z) {
    const auto x_neg_cond{ctx.OpFOrdLessThan(ctx.U1[1], x, ctx.f32_zero_value)};
    const auto z_neg_cond{ctx.OpFOrdLessThan(ctx.U1[1], z, ctx.f32_zero_value)};
    const auto x_sc{ctx.OpSelect(ctx.F32[1], x_neg_cond, z, ctx.OpFNegate(ctx.F32[1], z))};
    const auto y_sc{x};
    const auto z_sc{ctx.OpSelect(ctx.F32[1], z_neg_cond, ctx.OpFNegate(ctx.F32[1], x), x)};

    return SelectCubeResult(ctx, x, y, z, x_sc, y_sc, z_sc);
}

Id EmitCubeFaceCoordT(EmitContext& ctx, IR::Inst* inst, Id x, Id y, Id z) {
    const auto y_neg_cond{ctx.OpFOrdLessThan(ctx.U1[1], y, ctx.f32_zero_value)};
    const auto x_z_tc{ctx.OpFNegate(ctx.F32[1], y)};
    const auto y_tc{ctx.OpSelect(ctx.F32[1], y_neg_cond, ctx.OpFNegate(ctx.F32[1], z), z)};

    return SelectCubeResult(ctx, x, y, z, x_z_tc, y_tc, x_z_tc);
}

Id EmitCubeFaceMajorAxis(EmitContext& ctx, IR::Inst* inst, Id x, Id y, Id z) {
    const auto two{ctx.ConstF32(2.f)};
    const auto x_major_axis{ctx.OpFMul(ctx.F32[1], x, two)};
    const auto y_major_axis{ctx.OpFMul(ctx.F32[1], y, two)};
    const auto z_major_axis{ctx.OpFMul(ctx.F32[1], z, two)};

    return SelectCubeResult(ctx, x, y, z, x_major_axis, y_major_axis, z_major_axis);
}

} // namespace Shader::Backend::SPIRV
