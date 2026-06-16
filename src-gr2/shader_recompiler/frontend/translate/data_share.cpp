// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>

#include "common/config.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Gcn {

// GR2FORK v4.5 grass-fix latch: GCN exec-mask accuracy chain. Covers the
// mask-spill lane shadows here (V_READLANE/V_WRITELANE) AND the exec-masked
// V_CMP dst writes in vector_alu.cpp under ONE key — the two halves are only
// correct together (partial combinations are known-broken). Forced on; env
// kill GR2_NOGCNMASK=1 restores the exact v4.4 translation: no lane shadows,
// raw SetExec + raw dst writes. Latched on first shader translation. The IR
// opcodes / SSA MaskLaneVariable machinery stays compiled in either way — it
// is simply never emitted when killed.
static bool Gr2GcnMaskAccuracyEnabled() noexcept {
    static const bool enabled = []() noexcept {
        const char* e = std::getenv("GR2_NOGCNMASK");
        return !(e && e[0] == '1');
    }();
    return enabled;
}

void Translator::EmitDataShare(const GcnInst& inst) {
    switch (inst.opcode) {
        // DS
    case Opcode::DS_ADD_U32:
        return DS_OP(inst, AtomicOp::Add, false);
    case Opcode::DS_ADD_U64:
        return DS_OP<IR::U64>(inst, AtomicOp::Add, false);
    case Opcode::DS_SUB_U32:
        return DS_OP(inst, AtomicOp::Sub, false);
    case Opcode::DS_INC_U32:
        return DS_OP(inst, AtomicOp::Inc, false);
    case Opcode::DS_DEC_U32:
        return DS_OP(inst, AtomicOp::Dec, false);
    case Opcode::DS_MIN_I32:
        return DS_OP(inst, AtomicOp::Smin, false);
    case Opcode::DS_MAX_I32:
        return DS_OP(inst, AtomicOp::Smax, false);
    case Opcode::DS_MIN_U32:
        return DS_OP(inst, AtomicOp::Umin, false);
    case Opcode::DS_MAX_U32:
        return DS_OP(inst, AtomicOp::Umax, false);
    case Opcode::DS_AND_B32:
        return DS_OP(inst, AtomicOp::And, false);
    case Opcode::DS_OR_B32:
        return DS_OP(inst, AtomicOp::Or, false);
    case Opcode::DS_XOR_B32:
        return DS_OP(inst, AtomicOp::Xor, false);
    case Opcode::DS_WRITE_B32:
        return DS_WRITE(32, false, false, false, inst);
    case Opcode::DS_WRITE2_B32:
        return DS_WRITE(32, false, true, false, inst);
    case Opcode::DS_WRITE2ST64_B32:
        return DS_WRITE(32, false, true, true, inst);
    case Opcode::DS_ADD_RTN_U32:
        return DS_OP(inst, AtomicOp::Add, true);
    case Opcode::DS_SUB_RTN_U32:
        return DS_OP(inst, AtomicOp::Sub, true);
    case Opcode::DS_MIN_RTN_U32:
        return DS_OP(inst, AtomicOp::Umin, true);
    case Opcode::DS_MAX_RTN_U32:
        return DS_OP(inst, AtomicOp::Umax, true);
    case Opcode::DS_AND_RTN_B32:
        return DS_OP(inst, AtomicOp::And, true);
    case Opcode::DS_OR_RTN_B32:
        return DS_OP(inst, AtomicOp::Or, true);
    case Opcode::DS_XOR_RTN_B32:
        return DS_OP(inst, AtomicOp::Xor, true);
    case Opcode::DS_SWIZZLE_B32:
        return DS_SWIZZLE_B32(inst);
    case Opcode::DS_READ_B32:
        return DS_READ(32, false, false, false, inst);
    case Opcode::DS_READ2_B32:
        return DS_READ(32, false, true, false, inst);
    case Opcode::DS_READ2ST64_B32:
        return DS_READ(32, false, true, true, inst);
    case Opcode::DS_READ_U16:
        return DS_READ(16, false, false, false, inst);
    case Opcode::DS_CONSUME:
        return DS_CONSUME(inst);
    case Opcode::DS_APPEND:
        return DS_APPEND(inst);
    case Opcode::DS_WRITE_B16:
        return DS_WRITE(16, false, false, false, inst);
    case Opcode::DS_WRITE_B64:
        return DS_WRITE(64, false, false, false, inst);
    case Opcode::DS_WRITE2_B64:
        return DS_WRITE(64, false, true, false, inst);
    case Opcode::DS_WRITE2ST64_B64:
        return DS_WRITE(64, false, true, true, inst);
    case Opcode::DS_READ_B64:
        return DS_READ(64, false, false, false, inst);
    case Opcode::DS_READ2_B64:
        return DS_READ(64, false, true, false, inst);
    case Opcode::DS_READ2ST64_B64:
        return DS_READ(64, false, true, true, inst);
    default:
        LogMissingOpcode(inst);
    }
}

// VOP2

void Translator::V_READFIRSTLANE_B32(const GcnInst& inst) {
    const IR::U32 value{GetSrc(inst.src[0])};

    if (info.l_stage == LogicalStage::Compute ||
        info.l_stage == LogicalStage::TessellationControl) {
        SetDst(inst.dst[0], ir.ReadFirstLane(value));
    } else {
        SetDst(inst.dst[0], value);
    }
}

void Translator::V_READLANE_B32(const GcnInst& inst) {
    const IR::U32 value{GetSrc(inst.src[0])};
    const IR::U32 lane{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.ReadLane(value, lane));

    // FIX(GR2FORK mask-spill): GCN compilers spill EXEC/VCC/SGPR-pair thread
    // masks into VGPR lanes (v_writelane) and restore them later (v_readlane).
    // SetDst above only defines the destination SGPR's U32 view, while mask
    // consumers (S_MOV_B64 exec, sN / S_NOT_B64 / S_AND_SAVEEXEC_B64) read the
    // separate thread-bit U1 view. Without this, SSA finds no U1 def and
    // materializes UndefU1, folding exec restores and store guards to constant
    // false. (GR2: the grass seed CS cs_0x3f0a3c48's final vertexCount store
    // was guarded by Undef && !Undef, so the indirect args stayed {0,1,0,0}
    // and grass never rendered.) Restore the U1 view from the mask-lane shadow
    // saved by V_WRITELANE_B32. Architecturally a write to an SGPR clobbers
    // both views, so unconditionally redefining the U1 view here is strictly
    // more faithful than leaving it stale.
    // Kill switch GR2_NOGCNMASK=1 skips the
    // shadow restore — exact v4.4 (stale/Undef U1 view).
    if (Gr2GcnMaskAccuracyEnabled() && lane.IsImmediate()) {
        const u32 key = (inst.src[0].code << 6) | (lane.U32() & 63u);
        const IR::U1 shadow = ir.GetMaskLaneVariable(key);
        switch (inst.dst[0].field) {
        case OperandField::ScalarGPR:
            ir.SetThreadBitScalarReg(IR::ScalarReg(inst.dst[0].code), shadow);
            break;
        case OperandField::VccLo:
            ir.SetVcc(shadow);
            break;
        default:
            break;
        }
    }
}

void Translator::V_WRITELANE_B32(const GcnInst& inst) {
    const IR::VectorReg dst{inst.dst[0].code};
    const IR::U32 value{GetSrc(inst.src[0])};
    const IR::U32 lane{GetSrc(inst.src[1])};
    const IR::U32 old_value{GetSrc(inst.dst[0])};
    ir.SetVectorReg(dst, ir.WriteLane(old_value, value, lane));

    // FIX(GR2FORK mask-spill): also shadow the per-lane (U1) view of spilled
    // thread masks so V_READLANE_B32 can restore it (see comment there). For
    // non-mask (plain data) sources the shadowed U1 simply resolves to Undef,
    // which matches the pre-fix behavior for any later mask-typed read of the
    // restored SGPR.
    // Kill switch GR2_NOGCNMASK=1 skips the
    // shadow save — exact v4.4.
    if (Gr2GcnMaskAccuracyEnabled() && lane.IsImmediate()) {
        const u32 key = (inst.dst[0].code << 6) | (lane.U32() & 63u);
        switch (inst.src[0].field) {
        case OperandField::ScalarGPR:
            ir.SetMaskLaneVariable(
                key, ir.GetThreadBitScalarReg(IR::ScalarReg(inst.src[0].code)));
            break;
        case OperandField::VccLo:
            ir.SetMaskLaneVariable(key, ir.GetVcc());
            break;
        case OperandField::ExecLo:
            ir.SetMaskLaneVariable(key, ir.GetExec());
            break;
        default:
            break;
        }
    }
}

// DS

template <typename T>
void Translator::DS_OP(const GcnInst& inst, AtomicOp op, bool rtn) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{GetSrc(inst.src[0])};
    const T data = [&] {
        if (op == AtomicOp::Inc || op == AtomicOp::Dec) {
            return T{};
        }
        if constexpr (std::is_same_v<T, IR::U32>) {
            return GetSrc(inst.src[1]);
        } else {
            return GetSrc64(inst.src[1]);
        }
    }();
    const IR::U32 offset =
        ir.Imm32((u32(inst.control.ds.offset1) << 8u) + u32(inst.control.ds.offset0));
    const IR::U32 addr_offset = ir.IAdd(addr, offset);
    const T original_val = [&] -> T {
        switch (op) {
        case AtomicOp::Add:
            return ir.SharedAtomicIAdd(addr_offset, data, is_gds);
        case AtomicOp::Umin:
            return ir.SharedAtomicIMin(addr_offset, data, false, is_gds);
        case AtomicOp::Smin:
            return ir.SharedAtomicIMin(addr_offset, data, true, is_gds);
        case AtomicOp::Umax:
            return ir.SharedAtomicIMax(addr_offset, data, false, is_gds);
        case AtomicOp::Smax:
            return ir.SharedAtomicIMax(addr_offset, data, true, is_gds);
        case AtomicOp::And:
            return ir.SharedAtomicAnd(addr_offset, data, is_gds);
        case AtomicOp::Or:
            return ir.SharedAtomicOr(addr_offset, data, is_gds);
        case AtomicOp::Xor:
            return ir.SharedAtomicXor(addr_offset, data, is_gds);
        case AtomicOp::Sub:
            return ir.SharedAtomicISub(addr_offset, data, is_gds);
        case AtomicOp::Inc:
            return ir.SharedAtomicInc<T>(addr_offset, is_gds);
        case AtomicOp::Dec:
            return ir.SharedAtomicDec<T>(addr_offset, is_gds);
        default:
            UNREACHABLE();
        }
    }();
    if (rtn) {
        if constexpr (std::is_same_v<T, IR::U32>) {
            SetDst(inst.dst[0], original_val);
        } else {
            SetDst64(inst.dst[0], original_val);
        }
    }
}

void Translator::DS_WRITE(int bit_size, bool is_signed, bool is_pair, bool stride64,
                          const GcnInst& inst) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{ir.GetVectorReg(IR::VectorReg(inst.src[0].code))};
    const IR::VectorReg data0{inst.src[1].code};
    const IR::VectorReg data1{inst.src[2].code};
    const u32 offset = (inst.control.ds.offset1 << 8u) + inst.control.ds.offset0;
    if (info.stage == Stage::Fragment) {
        ASSERT_MSG(!is_pair && bit_size == 32 && offset % 256 == 0,
                   "Unexpected shared memory offset alignment: {}", offset);
        ir.SetVectorReg(GetScratchVgpr(offset), ir.GetVectorReg(data0));
        return;
    }
    if (is_pair) {
        const u32 adj = (bit_size == 32 ? 4 : 8) * (stride64 ? 64 : 1);
        const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset0 * adj)));
        if (bit_size == 64) {
            ir.WriteShared(64,
                           ir.PackUint2x32(ir.CompositeConstruct(ir.GetVectorReg(data0),
                                                                 ir.GetVectorReg(data0 + 1))),
                           addr0, is_gds);
        } else if (bit_size == 32) {
            ir.WriteShared(32, ir.GetVectorReg(data0), addr0, is_gds);
        } else if (bit_size == 16) {
            ir.WriteShared(16, ir.UConvert(16, ir.GetVectorReg(data0)), addr0, is_gds);
        }
        const IR::U32 addr1 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset1 * adj)));
        if (bit_size == 64) {
            ir.WriteShared(64,
                           ir.PackUint2x32(ir.CompositeConstruct(ir.GetVectorReg(data1),
                                                                 ir.GetVectorReg(data1 + 1))),
                           addr1, is_gds);
        } else if (bit_size == 32) {
            ir.WriteShared(32, ir.GetVectorReg(data1), addr1, is_gds);
        } else if (bit_size == 16) {
            ir.WriteShared(16, ir.UConvert(16, ir.GetVectorReg(data1)), addr1, is_gds);
        }
    } else {
        const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(offset));
        if (bit_size == 64) {
            const IR::Value data =
                ir.CompositeConstruct(ir.GetVectorReg(data0), ir.GetVectorReg(data0 + 1));
            ir.WriteShared(bit_size, ir.PackUint2x32(data), addr0, is_gds);
        } else if (bit_size == 32) {
            ir.WriteShared(bit_size, ir.GetVectorReg(data0), addr0, is_gds);
        } else if (bit_size == 16) {
            ir.WriteShared(bit_size, ir.UConvert(16, ir.GetVectorReg(data0)), addr0, is_gds);
        }
    }
}

void Translator::DS_READ(int bit_size, bool is_signed, bool is_pair, bool stride64,
                         const GcnInst& inst) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{ir.GetVectorReg(IR::VectorReg(inst.src[0].code))};
    IR::VectorReg dst_reg{inst.dst[0].code};
    const u32 offset = (inst.control.ds.offset1 << 8u) + inst.control.ds.offset0;
    if (info.stage == Stage::Fragment) {
        ASSERT_MSG(!is_pair && bit_size == 32 && offset % 256 == 0,
                   "Unexpected shared memory offset alignment: {}", offset);
        ir.SetVectorReg(dst_reg, ir.GetVectorReg(GetScratchVgpr(offset)));
        return;
    }
    if (is_pair) {
        // Pair loads are either 32 or 64-bit
        const u32 adj = (bit_size == 32 ? 4 : 8) * (stride64 ? 64 : 1);
        const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset0 * adj)));
        const IR::Value data0 = ir.LoadShared(bit_size, is_signed, addr0, is_gds);
        if (bit_size == 64) {
            const auto vector = ir.UnpackUint2x32(IR::U64{data0});
            ir.SetVectorReg(dst_reg++, IR::U32{ir.CompositeExtract(vector, 0)});
            ir.SetVectorReg(dst_reg++, IR::U32{ir.CompositeExtract(vector, 1)});
        } else if (bit_size == 32) {
            ir.SetVectorReg(dst_reg++, IR::U32{data0});
        } else if (bit_size == 16) {
            ir.SetVectorReg(dst_reg++, IR::U32{ir.UConvert(32, IR::U16{data0})});
        }
        const IR::U32 addr1 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset1 * adj)));
        const IR::Value data1 = ir.LoadShared(bit_size, is_signed, addr1, is_gds);
        if (bit_size == 64) {
            const auto vector = ir.UnpackUint2x32(IR::U64{data1});
            ir.SetVectorReg(dst_reg++, IR::U32{ir.CompositeExtract(vector, 0)});
            ir.SetVectorReg(dst_reg++, IR::U32{ir.CompositeExtract(vector, 1)});
        } else if (bit_size == 32) {
            ir.SetVectorReg(dst_reg++, IR::U32{data1});
        } else if (bit_size == 16) {
            ir.SetVectorReg(dst_reg++, IR::U32{ir.UConvert(32, IR::U16{data1})});
        }
    } else {
        const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(offset));
        const IR::Value data = ir.LoadShared(bit_size, is_signed, addr0, is_gds);
        if (bit_size == 64) {
            const auto vector = ir.UnpackUint2x32(IR::U64{data});
            ir.SetVectorReg(dst_reg, IR::U32{ir.CompositeExtract(vector, 0)});
            ir.SetVectorReg(dst_reg + 1, IR::U32{ir.CompositeExtract(vector, 1)});
        } else if (bit_size == 32) {
            ir.SetVectorReg(dst_reg, IR::U32{data});
        } else if (bit_size == 16) {
            ir.SetVectorReg(dst_reg++, IR::U32{ir.UConvert(32, IR::U16{data})});
        }
    }
}

void Translator::DS_SWIZZLE_B32(const GcnInst& inst) {
    const u8 offset0 = inst.control.ds.offset0;
    const u8 offset1 = inst.control.ds.offset1;
    const IR::U32 src{GetSrc(inst.src[0])};
    // ASSERT(offset1 & 0x80);
    const IR::U32 lane_id = ir.LaneId();
    const IR::U32 id_in_group = ir.BitwiseAnd(lane_id, ir.Imm32(0b11));
    const IR::U32 base = ir.ShiftLeftLogical(id_in_group, ir.Imm32(1));
    const IR::U32 index = ir.BitFieldExtract(ir.Imm32(offset0), base, ir.Imm32(2));
    SetDst(inst.dst[0], ir.QuadShuffle(src, index));
}

void Translator::DS_APPEND(const GcnInst& inst) {
    const u32 inst_offset = (u32(inst.control.ds.offset1) << 8u) + inst.control.ds.offset0;
    const IR::U32 gds_offset = ir.IAdd(ir.GetM0(), ir.Imm32(inst_offset));
    const IR::U32 prev = ir.DataAppend(gds_offset);
    SetDst(inst.dst[0], prev);
}

void Translator::DS_CONSUME(const GcnInst& inst) {
    const u32 inst_offset = (u32(inst.control.ds.offset1) << 8u) + inst.control.ds.offset0;
    const IR::U32 gds_offset = ir.IAdd(ir.GetM0(), ir.Imm32(inst_offset));
    const IR::U32 prev = ir.DataConsume(gds_offset);
    SetDst(inst.dst[0], prev);
}

} // namespace Shader::Gcn
