// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

// Rewrites wave-op lane numbering from the 32-wide host subgroup space into the
// 64-wide GCN wave space. Runs after ReadLaneEliminationPass so immediate-lane
// spill chains are matched unmasked, and before SharedMemoryBarrierPass so the
// inserted LocalInvocationId chains feed its divergence detection.
void Wave64EmulationPass(IR::Program& program, const RuntimeInfo& runtime_info,
                         const Profile& profile) {
    if (program.info.stage != Stage::Compute || profile.subgroup_size != 32 ||
        !profile.supports_required_subgroup_size_32) {
        return;
    }
    const auto& ws = runtime_info.cs_info.workgroup_size;
    const u32 tg_size = ws[0] * ws[1] * ws[2];
    if (tg_size <= 32) {
        // One pinned subgroup covers the whole group: exactly one partial GCN wave.
        return;
    }
    const auto flat_tid = [&](IR::IREmitter& ir) {
        IR::U32 flat = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
        if (ws[1] > 1) {
            flat = IR::U32{
                ir.IAdd(flat, ir.IMul(ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 1),
                                      ir.Imm32(ws[0])))};
        }
        if (ws[2] > 1) {
            flat = IR::U32{
                ir.IAdd(flat, ir.IMul(ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 2),
                                      ir.Imm32(ws[0] * ws[1])))};
        }
        return flat;
    };
    bool logged_dynamic_lane = false;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            switch (inst.GetOpcode()) {
            case IR::Opcode::LaneId: {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                IR::U32 lane = flat_tid(ir);
                if (tg_size > 64) {
                    lane = IR::U32{ir.BitwiseAnd(lane, ir.Imm32(63))};
                }
                inst.ReplaceUsesWith(lane);
                break;
            }
            case IR::Opcode::BallotFindLsb: {
                // First branch-active bit of the caller's own half, in 64-space.
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                const IR::U32 lsb = ir.BallotFindLsb(inst.Arg(0));
                const IR::U32 base = IR::U32{ir.BitwiseAnd(flat_tid(ir), ir.Imm32(32))};
                inst.ReplaceUsesWith(IR::U32{ir.IAdd(lsb, base)});
                break;
            }
            case IR::Opcode::ReadLane: {
                // 64-space lane indices are localized into the caller's subgroup;
                // reachable producers carry the caller's own bit 5, so the masked
                // lane is the intended one.
                const IR::Value lane = inst.Arg(1);
                if (lane.IsImmediate()) {
                    if (lane.U32() >= 32) {
                        LOG_DEBUG(Render_Recompiler,
                                  "shader {:#x}: readlane from lane {} masked into the subgroup",
                                  program.info.pgm_hash, lane.U32());
                        inst.SetArg(1, IR::Value{lane.U32() & 31});
                    }
                } else {
                    IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                    inst.SetArg(1, ir.BitwiseAnd(IR::U32{lane}, ir.Imm32(31)));
                    if (!logged_dynamic_lane) {
                        LOG_DEBUG(Render_Recompiler, "shader {:#x}: dynamic readlane lane masked",
                                  program.info.pgm_hash);
                        logged_dynamic_lane = true;
                    }
                }
                break;
            }
            default:
                break;
            }
        }
    }
}

} // namespace Shader::Optimization
