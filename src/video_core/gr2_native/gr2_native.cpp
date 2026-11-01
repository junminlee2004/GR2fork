// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <limits>
#include <span>

#include <xbyak/xbyak.h>

#include "common/gravity_rush.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "core/libraries/error_codes.h"
#include "core/memory.h"
#include "video_core/gr2_native/frame_sync.h"
#include "video_core/gr2_native/gr2_native.h"
#include "video_core/gr2_native/marker.h"

namespace Gr2Native {

namespace {

// Addresses are GoldHEN patch addresses: the executable's first segment at 0x400000.
constexpr u64 CodeBase = 0x400000;
constexpr u64 StubAreaSize = 16_KB;
// The stubs are mapped this far past the executable, clear of the graphics trace's stubs and
// within reach of a rel32 jump from the executable and back.
constexpr u64 StubAreaOffset = 1_MB;
constexpr u64 RedZone = 128;

// sce::Gnm::DrawCommandBuffer::writeAtEndOfPipe.
constexpr u64 WriteAtEndOfPipeAddress = 0x4164d0;
// DrawThread FUN_0084b700: `call writeAtEndOfPipe` for the frame label.
constexpr u64 LabelWriteCall = 0x84dd3f;
constexpr std::array<u8, 5> LabelWriteBytes{0xe8, 0x8c, 0x87, 0xbc, 0xff};
// DrawThread: the spin until the label of the frame two before holds its number, with the
// label address in rsi and the number in rdx. A 15-byte nop, then the loop
// `mov rdi, [rsi]; cmp rdi, rdx; jne` back to its mov. Only that jne branches into the range.
constexpr u64 SpinStart = 0x84e091;
constexpr u64 SpinEnd = 0x84e0a8;
constexpr std::array<u8, SpinEnd - SpinStart> SpinBytes{
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x48, 0x8b, 0x3e, 0x48, 0x39, 0xd7, 0x75, 0xf8};

u8* GameCode(VAddr module_base, u64 address) {
    return reinterpret_cast<u8*>(module_base + address - CodeBase);
}

bool Matches(const u8* code, std::span<const u8> bytes) {
    return std::memcmp(code, bytes.data(), bytes.size()) == 0;
}

void WriteRel32(u8* at, u8 opcode, const void* target) {
    const s32 displacement = static_cast<s32>(static_cast<const u8*>(target) - (at + 5));
    at[0] = opcode;
    std::memcpy(at + 1, &displacement, sizeof(displacement));
}

// The game's call lands here and goes on to WriteFrameLabel, which returns to the game.
void EmitLabelWriteStub(Xbyak::CodeGenerator& c) {
    using namespace Xbyak::util;
    c.mov(r11, reinterpret_cast<u64>(&WriteFrameLabel));
    c.jmp(r11);
}

// Calls WaitFrameLabel(rsi, rdx) and resumes after the spin. The game's caller-saved general
// registers are kept. From the spin to the game's next call no vector register is read and the
// flags are written before they are read, so those are free. The stub stays below the red zone
// and aligns the stack for the call.
void EmitSpinStub(Xbyak::CodeGenerator& c, const u8* resume) {
    using namespace Xbyak::util;
    const std::array<Xbyak::Reg64, 9> saved{rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11};
    c.lea(rsp, ptr[rsp - RedZone]);
    for (const auto& reg : saved) {
        c.push(reg);
    }
    c.push(rbp);
    c.mov(rbp, rsp);
    c.and_(rsp, ~u32{15});
    c.mov(rdi, rsi);
    c.mov(rsi, rdx);
    c.mov(rax, reinterpret_cast<u64>(&WaitFrameLabel));
    c.call(rax);
    c.mov(rsp, rbp);
    c.pop(rbp);
    for (auto reg = saved.rbegin(); reg != saved.rend(); ++reg) {
        c.pop(*reg);
    }
    c.lea(rsp, ptr[rsp + RedZone]);
    c.jmp(resume);
}

} // namespace

void Install(VAddr module_base) {
    if (module_base != MemoryPatcher::g_eboot_address || !Common::IsGravityRush2()) {
        return;
    }
    u8* label_write = GameCode(module_base, LabelWriteCall);
    u8* spin = GameCode(module_base, SpinStart);
    if (SpinEnd - CodeBase > MemoryPatcher::g_eboot_image_size ||
        !Matches(label_write, LabelWriteBytes) || !Matches(spin, SpinBytes)) {
        LOG_WARNING(Render_Vulkan, "GR2 native: the draw thread's frame code is not the 01.11 "
                                   "executable's, so the frame synchronization stays emulated");
        return;
    }
    void* area =
        reinterpret_cast<void*>(module_base + MemoryPatcher::g_eboot_image_size + StubAreaOffset);
    if (Core::Memory::Instance()->MapMemory(
            &area, reinterpret_cast<VAddr>(area), StubAreaSize,
            Core::MemoryProt::CpuReadWrite | Core::MemoryProt::CpuExec,
            Core::MemoryMapFlags::NoFlags, Core::VMAType::Code, "GR2 native") != ORBIS_OK ||
        reinterpret_cast<VAddr>(area) + StubAreaSize - module_base >
            std::numeric_limits<s32>::max()) {
        LOG_ERROR(Render_Vulkan, "GR2 native: no memory for the stubs within reach of the "
                                 "executable, so the frame synchronization stays emulated");
        return;
    }
    Xbyak::CodeGenerator c(StubAreaSize, area);
    const u8* label_stub = c.getCurr();
    EmitLabelWriteStub(c);
    const u8* spin_stub = c.getCurr();
    EmitSpinStub(c, GameCode(module_base, SpinEnd));
    SetWriteAtEndOfPipe(module_base + WriteAtEndOfPipeAddress - CodeBase);

    // Both sites change together: the wait is woken by the label marker's write.
    WriteRel32(label_write, 0xe8, label_stub);
    WriteRel32(spin, 0xe9, spin_stub);
    std::memset(spin + 5, 0xcc, SpinEnd - SpinStart - 5);
    LOG_INFO(Render_Vulkan, "GR2 native: frame labels are written when the GPU finishes a frame, "
                            "and the draw thread waits for them without spinning");
}

void OnNop(const u32* payload, u32 dwords, Vulkan::Rasterizer* rasterizer) {
    if (dwords < 2 || payload[0] != MarkerMagic) {
        return;
    }
    switch (static_cast<MarkerKind>(payload[1])) {
    case MarkerKind::FrameLabel:
        if (dwords >= FrameLabelPayloadDwords) {
            RunFrameLabel(payload, rasterizer);
        }
        break;
    default:
        LOG_ERROR(Render_Vulkan, "GR2 native: unknown marker kind {}", payload[1]);
        break;
    }
}

} // namespace Gr2Native
