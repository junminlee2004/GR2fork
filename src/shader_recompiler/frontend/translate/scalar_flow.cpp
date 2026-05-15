// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/opcodes.h"
#include "shader_recompiler/frontend/translate/translate.h"

namespace Shader::Gcn {

void Translator::EmitFlowControl(const GcnInst& inst) {
    switch (inst.opcode) {
    case Opcode::S_BARRIER:
        return S_BARRIER();
    case Opcode::S_TTRACEDATA:
        LOG_WARNING(Render_Vulkan, "S_TTRACEDATA instruction!");
        return;
    case Opcode::S_SETPRIO:
        LOG_WARNING(Render_Vulkan, "S_SETPRIO instruction!");
        return;
    case Opcode::S_TRAP:
        LOG_WARNING(Render_Vulkan, "S_TRAP instruction!");
        return;
    case Opcode::S_GETPC_B64:
        return S_GETPC_B64(inst);
    case Opcode::S_SETPC_B64:
    case Opcode::S_WAITCNT:
    case Opcode::S_NOP:
    case Opcode::S_ENDPGM:
    case Opcode::S_CBRANCH_EXECZ:
    case Opcode::S_CBRANCH_SCC0:
    case Opcode::S_CBRANCH_SCC1:
    case Opcode::S_CBRANCH_VCCNZ:
    case Opcode::S_CBRANCH_VCCZ:
    case Opcode::S_CBRANCH_EXECNZ:
    case Opcode::S_BRANCH:
        return;
    case Opcode::S_SENDMSG:
        S_SENDMSG(inst);
        return;
    default:
        UNREACHABLE();
    }
}

void Translator::S_BARRIER() {
    ir.Barrier();
}

void Translator::S_GETPC_B64(const GcnInst& inst) {
    const IR::ScalarReg dst{inst.dst[0].code};
    ir.SetScalarReg(dst, ir.Imm32(pc));
    ir.SetScalarReg(dst + 1, ir.Imm32(0));
}

// GR2FORK v3.1 (grass fix): a single helper applies the GsOp carried by both
// the Gs and GsDone S_SENDMSG variants. EmitCut (op=3) was the missing case
// that broke grass-blade GS shaders — those emit a discrete 4-vertex strip
// per blade and terminate each one with EmitCut, so dropping the op into
// the prior default → UNREACHABLE meant zero primitives reached the
// rasterizer for the blade carpet (no log, pipeline compiled fine).
// GsDone can carry the same GsOp values at GS thread exit; ignoring the op
// dropped the terminating Cut on the last in-flight primitive.
static inline void ApplyGsOp(IR::IREmitter& ir, SendMsgSimm::GsOp op) {
    switch (op) {
    case SendMsgSimm::GsOp::Nop:
        break;
    case SendMsgSimm::GsOp::Cut:
        ir.EmitPrimitive();
        break;
    case SendMsgSimm::GsOp::Emit:
        ir.EmitVertex();
        break;
    case SendMsgSimm::GsOp::EmitCut:
        ir.EmitVertex();
        ir.EmitPrimitive();
        break;
    }
}

void Translator::S_SENDMSG(const GcnInst& inst) {
    const auto& simm = reinterpret_cast<const SendMsgSimm&>(inst.control.sopp.simm);
    switch (simm.msg) {
    case SendMsgSimm::Message::Gs:
        ApplyGsOp(ir, simm.op);
        break;
    case SendMsgSimm::Message::GsDone:
        ApplyGsOp(ir, simm.op);
        break;
    case SendMsgSimm::Message::Interrupt:
    case SendMsgSimm::Message::System:
        // Host-side messages with no rasterizer effect. Drop quietly rather
        // than UNREACHABLE — guest may emit them in unrelated control flow.
        break;
    default:
        UNREACHABLE();
    }
}

} // namespace Shader::Gcn
