// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <xbyak/xbyak.h>

#include "common/decoder.h"
#include "common/gravity_rush.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "core/gr2_gfx_trace/gr2_gfx_trace.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/memory.h"

namespace Gr2GfxTrace {

namespace {

// Hook addresses are GoldHEN patch addresses: the executable's first segment at 0x400000.
constexpr u64 CodeBase = 0x400000;
constexpr u64 JmpSize = 5;
constexpr u64 StubAreaSize = 64_KB;
// sce::Gnmx::LightweightGfxContext::submitAndFlip, called once a frame by the draw thread.
constexpr u64 FrameFunction = 0x430590;
constexpr u64 DumpInterval = 900;
// Arguments are kept from a function's calls number 1, 16, 256 and 4096.
constexpr u32 SamplesPerFunction = 4;
constexpr u32 MaxThreads = 64;
constexpr u32 SiteBits = 14;
constexpr u32 SiteSlots = 1u << SiteBits;
constexpr u32 XmmArguments = 8;

struct Hook {
    u64 address;
    u8 size;
    std::array<u8, 15> bytes;
};

constexpr Hook Hooks[] = {
#include "core/gr2_gfx_trace/hook_table.inc"
};

constexpr u32 FrameHook =
    static_cast<u32>(std::ranges::find(Hooks, FrameFunction, &Hook::address) - std::begin(Hooks));
static_assert(FrameHook < std::size(Hooks));

// The registers a stub saves before it calls Probe, from the lowest address up.
struct Frame {
    std::array<std::array<u64, 2>, XmmArguments> xmm;
    u64 r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax;
    u64 return_address;
};
static_assert(sizeof(Frame) == XmmArguments * 16 + 10 * 8);

struct Sample {
    std::atomic<bool> ready;
    u64 returns_to;
    std::array<u64, 6> args;
    std::array<u32, 4> floats;
};

struct Stats {
    std::atomic<u64> calls;
    std::atomic<u64> threads;
    std::array<Sample, SamplesPerFunction> samples;
    // Written by the draw thread only.
    u64 calls_at_last_frame;
    u64 max_per_frame;
    u64 frames_called;
};

// A call site: the function called and the return address of the call, and how often it ran.
struct Site {
    std::atomic<u64> key;
    std::atomic<u64> calls;
};

std::array<Stats, std::size(Hooks)> g_stats;
std::array<Site, SiteSlots> g_sites;
std::atomic<u64> g_lost_sites;
u64 g_frames;

std::mutex g_thread_mutex;
std::array<std::string, MaxThreads> g_thread_names;
u32 g_thread_count;
thread_local u64 t_thread_bit;

// Threads past the last slot share it.
u64 ThreadBit() {
    if (t_thread_bit == 0) {
        std::scoped_lock lock{g_thread_mutex};
        const auto* thread = Libraries::Kernel::g_curthread;
        const u32 index = std::min(g_thread_count, MaxThreads - 1);
        if (g_thread_count < MaxThreads) {
            g_thread_names[index] = thread != nullptr ? thread->name : "host";
            ++g_thread_count;
        }
        t_thread_bit = u64{1} << index;
    }
    return t_thread_bit;
}

void CountSite(u32 id, u64 return_address) {
    const u64 key = (u64{id} + 1) << 48 | return_address;
    u64 slot = (key * 0x9e3779b97f4a7c15ull) >> (64 - SiteBits);
    for (u32 probe = 0; probe < SiteSlots; ++probe, slot = (slot + 1) % SiteSlots) {
        Site& site = g_sites[slot];
        u64 current = site.key.load(std::memory_order_relaxed);
        if (current == 0 && site.key.compare_exchange_strong(current, key)) {
            current = key;
        }
        if (current == key) {
            site.calls.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    g_lost_sites.fetch_add(1, std::memory_order_relaxed);
}

std::string GameAddress(u64 address) {
    const u64 offset = address - MemoryPatcher::g_eboot_address;
    if (offset < MemoryPatcher::g_eboot_image_size) {
        return fmt::format("{:#x}", offset + CodeBase);
    }
    return fmt::format("guest:{:#x}", address);
}

void Dump() {
    std::string out = fmt::format("# frames {}, lost call sites {}\n", g_frames,
                                  g_lost_sites.load(std::memory_order_relaxed));
    out += "# func\taddress\tcalls\tframes_called\tmax_per_frame\tthreads\n";
    for (u32 id = 0; id < std::size(Hooks); ++id) {
        const Stats& stats = g_stats[id];
        std::string threads;
        for (u64 mask = stats.threads.load(std::memory_order_relaxed); mask != 0;
             mask &= mask - 1) {
            threads += fmt::format("{}{}", threads.empty() ? "" : ",", std::countr_zero(mask));
        }
        out += fmt::format("func\t{:#x}\t{}\t{}\t{}\t{}\n", Hooks[id].address,
                           stats.calls.load(std::memory_order_relaxed), stats.frames_called,
                           stats.max_per_frame, threads);
    }
    out += "# site\tfunction\treturns_to\tcalls\n";
    for (const Site& site : g_sites) {
        const u64 key = site.key.load(std::memory_order_relaxed);
        if (key != 0) {
            out += fmt::format("site\t{:#x}\t{}\t{}\n", Hooks[(key >> 48) - 1].address,
                               GameAddress(key & ((u64{1} << 48) - 1)),
                               site.calls.load(std::memory_order_relaxed));
        }
    }
    out += "# sample\tfunction\treturns_to\trdi\trsi\trdx\trcx\tr8\tr9\txmm0\txmm1\txmm2\txmm3\n";
    for (u32 id = 0; id < std::size(Hooks); ++id) {
        for (const Sample& sample : g_stats[id].samples) {
            if (sample.ready.load(std::memory_order_acquire)) {
                out += fmt::format("sample\t{:#x}\t{}\t{:#x}\t{:#x}\n", Hooks[id].address,
                                   GameAddress(sample.returns_to), fmt::join(sample.args, "\t"),
                                   fmt::join(sample.floats, "\t"));
            }
        }
    }
    out += "# thread\tindex\tname\n";
    {
        std::scoped_lock lock{g_thread_mutex};
        for (u32 index = 0; index < g_thread_count; ++index) {
            out += fmt::format("thread\t{}\t{}\n", index, g_thread_names[index]);
        }
    }
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "gr2_gfx_trace.tsv";
    std::ofstream file{path};
    file << out;
    LOG_INFO(Lib_GnmDriver, "Graphics trace: {} frames written to {}", g_frames, path.string());
}

void OnFrame() {
    ++g_frames;
    for (Stats& stats : g_stats) {
        const u64 calls = stats.calls.load(std::memory_order_relaxed);
        const u64 frame_calls = calls - stats.calls_at_last_frame;
        stats.calls_at_last_frame = calls;
        if (frame_calls != 0) {
            ++stats.frames_called;
            stats.max_per_frame = std::max(stats.max_per_frame, frame_calls);
        }
    }
    if (g_frames % DumpInterval == 0) {
        Dump();
    }
}

void PS4_SYSV_ABI Probe(u32 id, const Frame* frame) {
    Stats& stats = g_stats[id];
    const u64 call = stats.calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 thread_bit = ThreadBit();
    if ((stats.threads.load(std::memory_order_relaxed) & thread_bit) == 0) {
        stats.threads.fetch_or(thread_bit, std::memory_order_relaxed);
    }
    CountSite(id, frame->return_address);
    const u32 sample_index = std::countr_zero(call) / 4;
    if (std::has_single_bit(call) && std::countr_zero(call) % 4 == 0 &&
        sample_index < SamplesPerFunction) {
        Sample& sample = stats.samples[sample_index];
        sample.returns_to = frame->return_address;
        sample.args = {frame->rdi, frame->rsi, frame->rdx, frame->rcx, frame->r8, frame->r9};
        for (u32 i = 0; i < sample.floats.size(); ++i) {
            sample.floats[i] = static_cast<u32>(frame->xmm[i][0]);
        }
        sample.ready.store(true, std::memory_order_release);
    }
    if (id == FrameHook) {
        OnFrame();
    }
}

// Saves the argument registers, calls Probe(id, saved registers) and restores them. The stack is
// 16-byte aligned at the call because nine registers sit above the return address.
void EmitProbeCall(Xbyak::CodeGenerator& c, u32 id) {
    using namespace Xbyak::util;
    const std::array<Xbyak::Reg64, 9> saved{rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11};
    for (const auto& reg : saved) {
        c.push(reg);
    }
    c.sub(rsp, XmmArguments * 16);
    for (u32 i = 0; i < XmmArguments; ++i) {
        c.vmovdqu(ptr[rsp + i * 16], Xbyak::Xmm(i));
    }
    c.mov(edi, id);
    c.mov(rsi, rsp);
    c.mov(rax, reinterpret_cast<u64>(&Probe));
    c.call(rax);
    for (u32 i = 0; i < XmmArguments; ++i) {
        c.vmovdqu(Xbyak::Xmm(i), ptr[rsp + i * 16]);
    }
    c.add(rsp, XmmArguments * 16);
    for (auto reg = saved.rbegin(); reg != saved.rend(); ++reg) {
        c.pop(*reg);
    }
}

// Copies the instructions a hook replaces into the stub. The displacement of a RIP-relative operand
// or a relative branch moves with the instruction, so it still reaches the same address.
bool Relocate(const u8* code, u32 size, Xbyak::CodeGenerator& c) {
    for (u32 offset = 0; offset < size;) {
        std::array<u8, ZYDIS_MAX_INSTRUCTION_LENGTH> bytes{};
        std::memcpy(bytes.data(), code + offset, size - offset);
        ZydisDecodedInstruction insn;
        std::array<ZydisDecodedOperand, ZYDIS_MAX_OPERAND_COUNT> operands;
        if (!ZYAN_SUCCESS(Common::Decoder::Instance()->decodeInstruction(
                insn, operands.data(), bytes.data(), size - offset))) {
            return false;
        }
        if ((insn.attributes & ZYDIS_ATTRIB_IS_RELATIVE) != 0) {
            const auto& imm = insn.raw.imm[0];
            const u8 field = imm.is_relative ? imm.offset : insn.raw.disp.offset;
            if ((imm.is_relative ? imm.size : insn.raw.disp.size) != 32) {
                return false;
            }
            s32 displacement;
            std::memcpy(&displacement, bytes.data() + field, sizeof(displacement));
            const s64 moved = displacement + (reinterpret_cast<s64>(code + offset) -
                                              reinterpret_cast<s64>(c.getCurr()));
            if (moved != static_cast<s32>(moved)) {
                return false;
            }
            displacement = static_cast<s32>(moved);
            std::memcpy(bytes.data() + field, &displacement, sizeof(displacement));
        }
        c.db(bytes.data(), insn.length);
        offset += insn.length;
    }
    return true;
}

} // namespace

void Install(VAddr module_base) {
    const char* enabled = std::getenv("GR2_GFX_TRACE");
    if (enabled == nullptr || std::string_view{enabled} != "1" ||
        module_base != MemoryPatcher::g_eboot_address || !Common::IsGravityRush2()) {
        return;
    }
    // Stubs sit past the executable, close enough for a jmp rel32 from it and back.
    void* area = reinterpret_cast<void*>(module_base + MemoryPatcher::g_eboot_image_size);
    if (Core::Memory::Instance()->MapMemory(
            &area, reinterpret_cast<VAddr>(area), StubAreaSize,
            Core::MemoryProt::CpuReadWrite | Core::MemoryProt::CpuExec,
            Core::MemoryMapFlags::NoFlags, Core::VMAType::Code, "GR2 graphics trace") != ORBIS_OK ||
        reinterpret_cast<VAddr>(area) + StubAreaSize - module_base >
            std::numeric_limits<s32>::max()) {
        LOG_ERROR(Loader, "Graphics trace: no memory for the hooks within reach of the executable");
        return;
    }
    Xbyak::CodeGenerator c(StubAreaSize, area);
    u32 hooked = 0;
    bool frame_hooked = false;
    for (u32 id = 0; id < std::size(Hooks); ++id) {
        const Hook& hook = Hooks[id];
        u8* entry = reinterpret_cast<u8*>(module_base + hook.address - CodeBase);
        if (hook.address - CodeBase + hook.size > MemoryPatcher::g_eboot_image_size ||
            std::memcmp(entry, hook.bytes.data(), hook.size) != 0) {
            LOG_WARNING(Loader, "Graphics trace: the code at {:#x} is not the 01.11 executable's",
                        hook.address);
            continue;
        }
        const auto start = c.getSize();
        const u8* stub = c.getCurr();
        EmitProbeCall(c, id);
        if (!Relocate(entry, hook.size, c)) {
            c.setSize(start);
            LOG_WARNING(Loader, "Graphics trace: the code at {:#x} cannot be moved", hook.address);
            continue;
        }
        c.jmp(entry + hook.size);
        const s32 jump = static_cast<s32>(stub - (entry + JmpSize));
        entry[0] = 0xe9;
        std::memcpy(entry + 1, &jump, sizeof(jump));
        std::memset(entry + JmpSize, 0xcc, hook.size - JmpSize);
        ++hooked;
        frame_hooked |= id == FrameHook;
    }
    LOG_INFO(Loader, "Graphics trace: hooked {} of {} library functions, {} bytes of stubs", hooked,
             std::size(Hooks), c.getSize());
    if (frame_hooked) {
        LOG_INFO(Loader,
                 "Graphics trace: written every {} frames to gr2_gfx_trace.tsv in the log "
                 "folder",
                 DumpInterval);
    } else if (hooked != 0) {
        LOG_WARNING(Loader, "Graphics trace: submitAndFlip is not hooked, so no frames are counted "
                            "and nothing is written");
    }
}

} // namespace Gr2GfxTrace
