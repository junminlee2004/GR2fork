// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {
void MemoryBarrier(EmitContext& ctx, spv::Scope scope) {
    const auto semantics{
        spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::UniformMemory |
        spv::MemorySemanticsMask::WorkgroupMemory | spv::MemorySemanticsMask::AtomicCounterMemory |
        spv::MemorySemanticsMask::ImageMemory};
    ctx.OpMemoryBarrier(ctx.ConstU32(static_cast<u32>(scope)),
                        ctx.ConstU32(static_cast<u32>(semantics)));
}
} // Anonymous namespace

void EmitBarrier(EmitContext& ctx) {
    const auto execution{spv::Scope::Workgroup};
    spv::Scope memory;
    spv::MemorySemanticsMask memory_semantics;
    if (ctx.l_stage == Shader::LogicalStage::TessellationControl) {
        memory = spv::Scope::Invocation;
        memory_semantics = spv::MemorySemanticsMask::MaskNone;
    } else {
        memory = spv::Scope::Workgroup;
        // A guest s_barrier is paired with s_waitcnt, and vmcnt covers vector memory, so a
        // barrier orders buffer and image traffic as well as LDS. Naming only WorkgroupMemory
        // leaves a shader that hands results between threads through a storage buffer relying
        // on caches that happen to be shared, which is true of a compute unit but not required.
        memory_semantics = spv::MemorySemanticsMask::AcquireRelease |
                           spv::MemorySemanticsMask::WorkgroupMemory |
                           spv::MemorySemanticsMask::UniformMemory |
                           spv::MemorySemanticsMask::ImageMemory;
    }
    ctx.OpControlBarrier(ctx.ConstU32(static_cast<u32>(execution)),
                         ctx.ConstU32(static_cast<u32>(memory)),
                         ctx.ConstU32(static_cast<u32>(memory_semantics)));
}

void EmitWorkgroupMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Workgroup);
}

void EmitDeviceMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Device);
}

} // namespace Shader::Backend::SPIRV
