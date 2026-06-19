// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cmath>
#include <boost/container/small_vector.hpp>

#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader {

namespace {
// GR2FORK internal-resolution scaling: FragCoord correction for screen-space
// fragment shaders.
//
// gl_FragCoord is delivered RAW by the hardware, so when a scene render target is
// scaled it spans the SCALED extent (e.g. 0..3840 at 2x), not the native extent
// the guest shader was written against. That breaks the two FragCoord idioms in
// OPPOSITE directions, which is why a blanket fix-up cannot work (landmine #1):
//
//   * NORMALIZED screen UVs: uv = FragCoord.xy * inv_resolution, where the game
//     feeds inv_resolution = 1/NATIVE via a constant buffer. With FragCoord in
//     scaled space the uv runs 0..scale instead of 0..1 -> the effect samples
//     off-screen (the broken post-process: bloom, AO, DoF, tonemap, ...). These
//     want FragCoord in NATIVE space.
//   * INTEGER texelFetch: ivec2(FragCoord) -> ImageRead of a SCALED buffer
//     (scene colour/depth). Scaled FragCoord indexes the scaled buffer correctly
//     and must be LEFT ALONE; dividing it would address only the native-sized
//     top-left corner.
//
// The two are distinguishable at FragCoord's consumer: the texelFetch path begins
// with an integer conversion (ConvertS32F32 / ConvertU32F32), the normalized path
// uses floating-point arithmetic. So we give each CONSUMER the FragCoord it needs:
// integer-convert consumers keep the raw (scaled) value; every other consumer is
// redirected to FragCoord * (1/scale), i.e. native space. A shader that does both
// gets both correct. Only x/y are touched (z/w are not pixel positions).
//
// No-op when scaling is inactive (factor == 1.0 or invalid) -> emitted IR is then
// byte-identical to stock. The factor is baked from profile.internal_resolution_
// scale, part of the shader cache key, so each scale gets its own variant.
void FragCoordResolutionScalePass(IR::Program& program, const Profile& profile) {
    const float scale = profile.internal_resolution_scale;
    if (!std::isfinite(scale) || scale <= 0.0f || std::abs(scale - 1.0f) < 1.0e-6f) {
        return;
    }
    const float inv_scale = 1.0f / scale;

    // Collect the FragCoord x/y reads first; the rewrite below inserts
    // instructions, so structural edits stay out of the discovery walk.
    boost::container::small_vector<IR::Inst*, 4> frag_coords;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::GetAttribute) {
                continue;
            }
            if (inst.Arg(0).Attribute() != IR::Attribute::FragCoord) {
                continue;
            }
            const u32 comp = inst.Arg(1).U32();
            if (comp == 0 || comp == 1) {
                frag_coords.push_back(&inst);
            }
        }
    }

    for (IR::Inst* const fc : frag_coords) {
        // Snapshot the float-path consumers before mutating. Integer conversions
        // are the texelFetch addressing path and stay on raw (scaled) coords.
        boost::container::small_vector<IR::Use, 4> float_uses;
        for (const IR::Use& use : fc->Uses()) {
            const IR::Opcode uop = use.user->GetOpcode();
            if (uop == IR::Opcode::ConvertS32F32 || uop == IR::Opcode::ConvertU32F32) {
                continue;
            }
            float_uses.push_back(use);
        }
        if (float_uses.empty()) {
            continue;
        }

        // native = FragCoord * (1/scale), inserted right AFTER the read so it
        // dominates every use it replaces.
        IR::Block* const block = fc->GetParent();
        auto insertion_point = IR::Block::InstructionList::s_iterator_to(*fc);
        ++insertion_point;
        IR::IREmitter ir{*block, insertion_point};
        const IR::Value native = ir.FPMul(IR::F32{fc}, ir.Imm32(inv_scale));

        for (const IR::Use& use : float_uses) {
            use.user->SetArg(use.operand, native);
        }
    }
}
} // namespace


IR::BlockList GenerateBlocks(const IR::AbstractSyntaxList& syntax_list) {
    size_t num_syntax_blocks{};
    for (const auto& node : syntax_list) {
        if (node.type == IR::AbstractSyntaxNode::Type::Block) {
            ++num_syntax_blocks;
        }
    }
    IR::BlockList blocks;
    blocks.reserve(num_syntax_blocks);
    u32 order_index{};
    for (const auto& node : syntax_list) {
        if (node.type == IR::AbstractSyntaxNode::Type::Block) {
            blocks.push_back(node.data.block);
        }
    }
    return blocks;
}

IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools, Info& info,
                             RuntimeInfo& runtime_info, const Profile& profile) {
    // Ensure first instruction is expected.
    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    if (code[0] != token_mov_vcchi) {
        LOG_WARNING(Render_Recompiler, "First instruction is not s_mov_b32 vcc_hi, #imm");
    }

    Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Gcn::GcnDecodeContext decoder;

    // Decode and save instructions
    IR::Program program{info};
    program.ins_list.reserve(code.size());
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
    }

    // Clear any previous pooled data.
    pools.ReleaseContents();

    // Create control flow graph
    Common::ObjectPool<Gcn::Block> gcn_block_pool{64};
    Gcn::CFG cfg{gcn_block_pool, program.ins_list};

    // Structurize control flow graph and create program.
    program.syntax_list =
        Shader::Gcn::BuildASL(pools.inst_pool, pools.block_pool, cfg, info, runtime_info, profile);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    // Run optimization passes
    if (!profile.support_float64) {
        Shader::Optimization::LowerFp64ToFp32(program);
    }
    Shader::Optimization::SsaRewritePass(program.post_order_blocks);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    if (info.l_stage == LogicalStage::TessellationControl) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::HullShaderTransform(program, runtime_info);
    } else if (info.l_stage == LogicalStage::TessellationEval) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::DomainShaderTransform(program, runtime_info);
    }
    Shader::Optimization::RingAccessElimination(program, runtime_info);
    Shader::Optimization::ReadLaneEliminationPass(program);
    Shader::Optimization::FlattenExtendedUserdataPass(program);
    Shader::Optimization::ResourceTrackingPass(program, profile);
    Shader::Optimization::LowerBufferFormatToRaw(program);
    Shader::Optimization::SharedMemorySimplifyPass(program, profile);
    Shader::Optimization::SharedMemoryToStoragePass(program, runtime_info, profile);
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);
    // GR2FORK internal-resolution scaling: correct FragCoord per-consumer (native
    // for normalized screen UVs, scaled left intact for integer texelFetch).
    // No-op at factor 1.0. Placed before the final cleanup so the inserted
    // multiply and redirected uses are folded/DCE'd.
    FragCoordResolutionScalePass(program, profile);
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::CollectShaderInfoPass(program, profile);

    Shader::IR::DumpProgram(program, info);

    return program;
}

} // namespace Shader
