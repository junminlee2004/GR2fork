// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <memory>
#include "common/assert.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/fetch_shader.h"

namespace Shader::Gcn {

/**
 * s_load_dwordx4 s[8:11], s[2:3], 0x00
 * s_load_dwordx4 s[12:15], s[2:3], 0x04
 * s_load_dwordx4 s[16:19], s[2:3], 0x08
 * s_waitcnt     lgkmcnt(0)
 * buffer_load_format_xyzw v[4:7], v0, s[8:11], 0 idxen
 * buffer_load_format_xyz v[8:10], v0, s[12:15], 0 idxen
 * buffer_load_format_xy v[12:13], v0, s[16:19], 0 idxen
 * s_waitcnt     0
 * s_setpc_b64   s[0:1]

 * s_load_dwordx4  s[4:7], s[2:3], 0x0
 * s_waitcnt       lgkmcnt(0)
 * buffer_load_format_xyzw v[4:7], v0, s[4:7], 0 idxen
 * s_load_dwordx4  s[4:7], s[2:3], 0x8
 * s_waitcnt       lgkmcnt(0)
 * buffer_load_format_xyzw v[8:11], v0, s[4:7], 0 idxen
 * s_waitcnt       vmcnt(0) & expcnt(0) & lgkmcnt(0)
 * s_setpc_b64     s[0:1]

 * A normal fetch shader looks like the above, the instructions are generated
 * using input semantics on cpu side. Load instructions can either be separate or interleaved
 * We take the reverse way, extract the original input semantics from these instructions.
 **/

static bool IsTypedBufferLoad(const Gcn::GcnInst& inst) {
    return inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_X ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XY ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XYZ ||
           inst.opcode == Opcode::TBUFFER_LOAD_FORMAT_XYZW;
}

const u32* GetFetchShaderCode(const Info& info, u32 sgpr_base) {
    const u32* code;
    std::memcpy(&code, &info.user_data[sgpr_base], sizeof(code));
    return code;
}

namespace {

/**
 * Decoding a fetch shader is a pure function of the instruction bytes, but it
 * runs once per pipeline lookup - that is, once per draw - because the code
 * pointer is resolved from per-draw user data. The same handful of fetch
 * shaders repeat all frame, so the decode is re-derived thousands of times a
 * frame for an answer that never changed.
 *
 * A hit is accepted only after comparing every byte the previous decode
 * consumed, so a guest that rewrites shader code in place still gets a fresh
 * parse. That makes the memo exactly equivalent to decoding, not an
 * approximation of it: bytes past the decoded region cannot affect the result
 * because the decode never read them.
 */
struct FetchShaderMemo {
    static constexpr size_t NumEntries = 64; // direct mapped, power of two
    static constexpr u32 MaxCachedBytes = 4096;

    struct Entry {
        const u32* code{};
        std::vector<u32> words; // exact copy of the decoded region
        FetchShaderData data;
    };

    std::array<Entry, NumEntries> entries{};

    static size_t Index(const u32* code) {
        u64 key = reinterpret_cast<uintptr_t>(code);
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdULL;
        key ^= key >> 33;
        return static_cast<size_t>(key & (NumEntries - 1));
    }
};

// Heap backed: only the pointer lives in thread-local storage, since fetch
// shaders are also parsed from shader compilation workers.
FetchShaderMemo& GetFetchShaderMemo() {
    static thread_local std::unique_ptr<FetchShaderMemo> memo;
    if (!memo) {
        memo = std::make_unique<FetchShaderMemo>();
    }
    return *memo;
}

} // Anonymous namespace

std::optional<FetchShaderData> ParseFetchShader(const Shader::Info& info) {
    if (!info.has_fetch_shader) {
        return std::nullopt;
    }

    const auto* code = GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
    auto& memo = GetFetchShaderMemo();
    auto& entry = memo.entries[FetchShaderMemo::Index(code)];
    if (entry.code == code && !entry.words.empty() &&
        std::memcmp(code, entry.words.data(), entry.words.size() * sizeof(u32)) == 0) {
        return entry.data;
    }

    FetchShaderData data{};
    GcnCodeSlice code_slice(code, code + std::numeric_limits<u32>::max());
    GcnDecodeContext decoder;

    struct VsharpLoad {
        u32 dword_offset{};
        u32 base_sgpr{};
    };
    std::array<VsharpLoad, 104> loads{};

    u32 semantic_index = 0;
    while (!code_slice.atEnd()) {
        const auto inst = decoder.decodeInstruction(code_slice);
        data.size += inst.length;

        if (inst.opcode == Opcode::S_SETPC_B64) {
            break;
        }

        if (inst.inst_class == InstClass::ScalarMemRd) {
            loads[inst.dst[0].code] = VsharpLoad{inst.control.smrd.offset, inst.src[0].code * 2};
            continue;
        }

        if (inst.opcode == Opcode::V_ADD_I32) {
            const auto vgpr = inst.dst[0].code;
            const auto sgpr = s8(inst.src[0].code);
            switch (vgpr) {
            case 0: // V0 is always the vertex offset
                data.vertex_offset_sgpr = sgpr;
                break;
            case 3: // V3 is always the instance offset
                data.instance_offset_sgpr = sgpr;
                break;
            default:
                UNREACHABLE();
            }
        }

        if (inst.inst_class == InstClass::VectorMemBufFmt) {
            // SRSRC is in units of 4 SPGRs while SBASE is in pairs of SGPRs
            const u32 base_sgpr = inst.src[2].code * 4;
            auto& attrib = data.attributes.emplace_back();
            attrib.semantic = semantic_index++;
            attrib.dest_vgpr = inst.src[1].code;
            attrib.num_elements = inst.control.mubuf.count;
            attrib.sgpr_base = loads[base_sgpr].base_sgpr;
            attrib.dword_offset = loads[base_sgpr].dword_offset;
            attrib.inst_offset = inst.control.mtbuf.offset;
            attrib.instance_data = inst.src[0].code;
            if (IsTypedBufferLoad(inst)) {
                attrib.data_format = inst.control.mtbuf.dfmt;
                attrib.num_format = inst.control.mtbuf.nfmt;
            }
        }
    }

    if (data.size > 0 && data.size <= FetchShaderMemo::MaxCachedBytes &&
        data.size % sizeof(u32) == 0) {
        const size_t num_words = data.size / sizeof(u32);
        entry.words.assign(code, code + num_words);
        entry.data = data;
        entry.code = code;
    } else {
        entry.code = nullptr;
        entry.words.clear();
    }
    return data;
}

} // namespace Shader::Gcn
