// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>

#include <realtimeapiset.h>
#include <tlhelp32.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <xbyak/xbyak.h>

#include "common/decoder.h"
#include "common/gravity_rush.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/path_util.h"
#include "common/rdtsc.h"
#include "common/string_util.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/gr2_gfx_trace/gr2_gfx_trace.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/memory.h"

namespace Gr2GfxTrace {

namespace {

using Clock = std::chrono::steady_clock;

// Hook addresses are GoldHEN patch addresses: the executable's first segment at 0x400000.
constexpr u64 CodeBase = 0x400000;
constexpr u64 JmpSize = 5;
constexpr u64 StubAreaSize = 64_KB;
// sce::Gnmx::LightweightGfxContext::submitAndFlip, called once a frame by the draw thread.
constexpr u64 FrameFunction = 0x430590;
constexpr auto SampleInterval = std::chrono::seconds{1};
// The call and call site tables are rewritten every this many samples.
constexpr u32 DumpSamples = 10;
// Arguments are kept from a function's calls number 1, 16, 256 and 4096.
constexpr u32 SamplesPerFunction = 4;
constexpr u32 MaxThreads = 256;
// Library callers past the mask's last bit share it.
constexpr u32 ThreadMaskBits = 64;
constexpr u32 NoThreadSlot = std::numeric_limits<u32>::max();
constexpr u32 SiteBits = 14;
constexpr u32 SiteSlots = 1u << SiteBits;
constexpr u32 XmmArguments = 8;
// A probe that seems to take longer than this moved between cores whose time stamp counters
// disagree, and its time is dropped.
constexpr u64 MaxProbeCycles = 100'000'000;
// Words a timed hook copies from its caller's stack: a callee with up to 10 arguments finds them
// where the caller put them.
constexpr u32 TimedStackArgs = 4;
// A timed call on a timeline thread shorter than this is only counted, unless its hook says so.
constexpr u64 TimelineMinNs = 50'000;
constexpr u32 EventBits = 16;
constexpr auto PollInterval = std::chrono::microseconds{500};
constexpr auto TimelineFlushInterval = std::chrono::milliseconds{250};
// The main loop's frame counter, and the DrawThread's three frame labels: {u64* label, u64 value
// the label holds once the GPU front end has reached the end of that frame}.
constexpr u64 FrameCounterAddress = 0x1efb08c;
// The DrawThread's frame number, the value its next frame label gets. It moves on once the frame
// is submitted and the frame two before it has reached its label.
constexpr u64 DrawFrameAddress = 0x1f585f8;
constexpr u64 FrameLabelsAddress = 0x1f7b680;
constexpr u32 FrameLabels = 3;

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

struct TimedHook {
    u64 address;
    u8 size;
    std::array<u8, 15> bytes;
    const char* name;
    // The call goes on the timeline whatever its duration.
    bool always;
};

constexpr TimedHook TimedHooks[] = {
#include "core/gr2_gfx_trace/timed_hook_table.inc"
};

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
    std::atomic<u64> max_per_frame;
    std::atomic<u64> frames_called;
};

// A call site: the function called and the return address of the call, and how often it ran.
struct Site {
    std::atomic<u64> key;
    std::atomic<u64> calls;
};

// A thread that calls the library. Each slot has one writer, except the last one, which threads
// past it share and whose probe time is approximate.
struct ProbeThread {
    std::atomic<u64> os_id;
    // Time stamp counter cycles spent in Probe.
    std::atomic<u64> probe_cycles;
    // DrawThread and the main thread, whose timed calls go on the timeline.
    std::atomic<bool> timeline;
};

struct TimedStats {
    std::atomic<u64> calls;
    std::atomic<u64> nanoseconds;
    std::atomic<u64> max_nanoseconds;
};

// A timed call of a timeline thread. seq is the event's index + 1 once the fields are written.
struct CallEvent {
    std::atomic<u64> seq;
    u64 start_ns;
    u64 duration_ns;
    u64 argument;
    u32 hook;
    u32 thread_slot;
};

std::array<Stats, std::size(Hooks)> g_stats;
std::array<Site, SiteSlots> g_sites;
std::atomic<u64> g_lost_sites;
std::atomic<u64> g_frames;
Clock::time_point g_start;

std::mutex g_thread_mutex;
std::array<std::string, MaxThreads> g_thread_names;
std::array<ProbeThread, MaxThreads> g_probe_threads;
u32 g_thread_count;
thread_local u32 t_thread_slot = NoThreadSlot;

std::array<std::array<TimedStats, std::size(TimedHooks)>, MaxThreads> g_timed;
std::array<CallEvent, 1u << EventBits> g_events;
std::atomic<u64> g_event_head;

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - g_start).count();
}

u64 OsThreadId() {
#ifdef _WIN32
    return GetCurrentThreadId();
#elif defined(__linux__)
    return static_cast<u64>(syscall(SYS_gettid));
#else
    return 0;
#endif
}

u32 ThreadSlot() {
    if (t_thread_slot == NoThreadSlot) {
        std::scoped_lock lock{g_thread_mutex};
        const auto* thread = Libraries::Kernel::g_curthread;
        const u32 index = std::min(g_thread_count, MaxThreads - 1);
        if (g_thread_count < MaxThreads) {
            g_thread_names[index] = thread != nullptr ? thread->name : "host";
            g_probe_threads[index].os_id.store(OsThreadId(), std::memory_order_relaxed);
            g_probe_threads[index].timeline.store(g_thread_names[index] == "DrawThread" ||
                                                      g_thread_names[index] == "Game:Main",
                                                  std::memory_order_relaxed);
            ++g_thread_count;
        }
        t_thread_slot = index;
    }
    return t_thread_slot;
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

double Millis(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

void Dump() {
    std::string out =
        fmt::format("# frames {}, lost call sites {}\n", g_frames.load(std::memory_order_relaxed),
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
                           stats.calls.load(std::memory_order_relaxed),
                           stats.frames_called.load(std::memory_order_relaxed),
                           stats.max_per_frame.load(std::memory_order_relaxed), threads);
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
    out += "# thread\tindex\tname\tos_thread_id\n";
    {
        std::scoped_lock lock{g_thread_mutex};
        for (u32 index = 0; index < g_thread_count; ++index) {
            out += fmt::format("thread\t{}\t{}\t{}\n", index, g_thread_names[index],
                               g_probe_threads[index].os_id.load(std::memory_order_relaxed));
        }
    }
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "gr2_gfx_trace.tsv";
    std::ofstream file{path};
    file << out;
}

void OnFrame() {
    g_frames.store(g_frames.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    for (Stats& stats : g_stats) {
        const u64 calls = stats.calls.load(std::memory_order_relaxed);
        const u64 frame_calls = calls - stats.calls_at_last_frame;
        stats.calls_at_last_frame = calls;
        if (frame_calls != 0) {
            stats.frames_called.store(stats.frames_called.load(std::memory_order_relaxed) + 1,
                                      std::memory_order_relaxed);
            if (frame_calls > stats.max_per_frame.load(std::memory_order_relaxed)) {
                stats.max_per_frame.store(frame_calls, std::memory_order_relaxed);
            }
        }
    }
}

void PS4_SYSV_ABI Probe(u32 id, const Frame* frame) {
    const u64 start = __rdtsc();
    Stats& stats = g_stats[id];
    const u64 call = stats.calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const u32 slot = ThreadSlot();
    const u64 thread_bit = u64{1} << std::min(slot, ThreadMaskBits - 1);
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
    const u64 cycles = __rdtsc() - start;
    if (cycles < MaxProbeCycles) {
        auto& total = g_probe_threads[slot].probe_cycles;
        total.store(total.load(std::memory_order_relaxed) + cycles, std::memory_order_relaxed);
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

u64 PS4_SYSV_ABI EnterTimed() {
    return NowNs();
}

void PS4_SYSV_ABI ExitTimed(u32 hook, u64 start, u64 argument) {
    const u64 duration = NowNs() - start;
    const u32 slot = ThreadSlot();
    TimedStats& stats = g_timed[slot][hook];
    stats.calls.store(stats.calls.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    stats.nanoseconds.store(stats.nanoseconds.load(std::memory_order_relaxed) + duration,
                            std::memory_order_relaxed);
    if (duration > stats.max_nanoseconds.load(std::memory_order_relaxed)) {
        stats.max_nanoseconds.store(duration, std::memory_order_relaxed);
    }
    if (!g_probe_threads[slot].timeline.load(std::memory_order_relaxed) ||
        (!TimedHooks[hook].always && duration < TimelineMinNs)) {
        return;
    }
    const u64 index = g_event_head.fetch_add(1, std::memory_order_relaxed);
    CallEvent& event = g_events[index & (g_events.size() - 1)];
    event.start_ns = start;
    event.duration_ns = duration;
    event.argument = argument;
    event.hook = hook;
    event.thread_slot = slot;
    event.seq.store(index + 1, std::memory_order_release);
}

// Calls the original function through the trampoline between EnterTimed and ExitTimed. The
// stub's own frame sits between the caller's and the callee's, so the caller's first
// TimedStackArgs stack words are copied below it, where the callee looks for them. Frame layout
// from rsp: the copied words, the start time, the first argument, padding to keep rsp 16-byte
// aligned at both calls.
void EmitTimedHook(Xbyak::CodeGenerator& c, u32 id, const u8* trampoline) {
    using namespace Xbyak::util;
    constexpr u32 StartSlot = TimedStackArgs * 8;
    constexpr u32 ArgumentSlot = StartSlot + 8;
    constexpr u32 FrameSize = ArgumentSlot + 16;
    static_assert(FrameSize % 16 == 8);
    constexpr u32 SavedSize = 9 * 8 + XmmArguments * 16 + 8;
    c.sub(rsp, FrameSize);
    for (u32 i = 0; i < TimedStackArgs; ++i) {
        c.mov(r11, ptr[rsp + FrameSize + 8 + i * 8]);
        c.mov(ptr[rsp + i * 8], r11);
    }
    c.mov(ptr[rsp + ArgumentSlot], rdi);
    const std::array<Xbyak::Reg64, 9> saved{rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11};
    for (const auto& reg : saved) {
        c.push(reg);
    }
    c.sub(rsp, XmmArguments * 16 + 8);
    for (u32 i = 0; i < XmmArguments; ++i) {
        c.vmovdqu(ptr[rsp + i * 16], Xbyak::Xmm(i));
    }
    c.mov(rax, reinterpret_cast<u64>(&EnterTimed));
    c.call(rax);
    c.mov(ptr[rsp + SavedSize + StartSlot], rax);
    for (u32 i = 0; i < XmmArguments; ++i) {
        c.vmovdqu(Xbyak::Xmm(i), ptr[rsp + i * 16]);
    }
    c.add(rsp, XmmArguments * 16 + 8);
    for (auto reg = saved.rbegin(); reg != saved.rend(); ++reg) {
        c.pop(*reg);
    }
    c.mov(r11, reinterpret_cast<u64>(trampoline));
    c.call(r11);
    // Keep the return values: rax, rdx, xmm0 and xmm1.
    constexpr u32 ResultSize = 16 + 32;
    c.push(rax);
    c.push(rdx);
    c.sub(rsp, 32);
    c.vmovdqu(ptr[rsp], xmm0);
    c.vmovdqu(ptr[rsp + 16], xmm1);
    c.mov(edi, id);
    c.mov(rsi, ptr[rsp + ResultSize + StartSlot]);
    c.mov(rdx, ptr[rsp + ResultSize + ArgumentSlot]);
    c.mov(rax, reinterpret_cast<u64>(&ExitTimed));
    c.call(rax);
    c.vmovdqu(xmm0, ptr[rsp]);
    c.vmovdqu(xmm1, ptr[rsp + 16]);
    c.add(rsp, 32);
    c.pop(rdx);
    c.pop(rax);
    c.add(rsp, FrameSize);
    c.ret();
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

// The CPU time a thread of the process has run, emulator and game threads alike.
struct CpuTime {
    u64 os_id;
    std::string name;
    u64 nanoseconds;
};

#ifdef _WIN32
// Handles stay open, so a listed thread id is not given to a new thread.
std::unordered_map<DWORD, HANDLE> g_thread_handles;

// Thread cycle times count time stamp counter cycles.
std::vector<CpuTime> ReadCpuTimes(u64 tsc_hz) {
    std::vector<CpuTime> out;
    const HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return out;
    }
    const DWORD process = GetCurrentProcessId();
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != process) {
            continue;
        }
        auto [it, inserted] = g_thread_handles.try_emplace(entry.th32ThreadID, nullptr);
        if (inserted) {
            it->second = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
        }
        ULONG64 cycles;
        if (it->second == nullptr || !QueryThreadCycleTime(it->second, &cycles)) {
            continue;
        }
        std::string name;
        PWSTR description = nullptr;
        if (SUCCEEDED(GetThreadDescription(it->second, &description))) {
            name = Common::UTF16ToUTF8(description);
            LocalFree(description);
        }
        out.push_back({entry.th32ThreadID, std::move(name),
                       static_cast<u64>(static_cast<double>(cycles) * 1e9 / tsc_hz)});
    }
    CloseHandle(snapshot);
    return out;
}
#elif defined(__linux__)
// The first field of schedstat is the time the thread has run on a CPU, in nanoseconds. Names are
// cut to 15 characters.
std::vector<CpuTime> ReadCpuTimes(u64) {
    std::vector<CpuTime> out;
    try {
        std::error_code error;
        for (const auto& task : std::filesystem::directory_iterator{"/proc/self/task", error}) {
            std::ifstream schedstat{task.path() / "schedstat"};
            u64 runtime;
            if (!(schedstat >> runtime)) {
                continue;
            }
            std::ifstream comm{task.path() / "comm"};
            std::string name;
            std::getline(comm, name);
            out.push_back({std::stoull(task.path().filename().string()), std::move(name), runtime});
        }
    } catch (const std::exception&) {
    }
    return out;
}
#else
std::vector<CpuTime> ReadCpuTimes(u64) {
    return {};
}
#endif

// NVIDIA's management library, loaded at run time. Without it, as on other vendors' GPUs, no GPU
// rows are written.
class Nvml {
public:
    Nvml() {
#ifdef _WIN32
        library = LoadLibraryW(L"nvml.dll");
        if (library == nullptr) {
            library = LoadLibraryW(L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        }
#elif defined(__linux__)
        library = dlopen("libnvidia-ml.so.1", RTLD_NOW);
#endif
        if (library == nullptr || !Load(init, "nvmlInit_v2") ||
            !Load(get_count, "nvmlDeviceGetCount_v2") ||
            !Load(get_handle, "nvmlDeviceGetHandleByIndex_v2") ||
            !Load(get_name, "nvmlDeviceGetName") ||
            !Load(get_utilization, "nvmlDeviceGetUtilizationRates") ||
            !Load(get_clock, "nvmlDeviceGetClockInfo") || init() != Success) {
            return;
        }
        u32 count = 0;
        if (get_count(&count) != Success) {
            return;
        }
        for (u32 index = 0; index < count; ++index) {
            Device device{};
            std::array<char, 96> name{};
            if (get_handle(index, &device.handle) == Success &&
                get_name(device.handle, name.data(), static_cast<u32>(name.size())) == Success) {
                device.name = name.data();
                devices.push_back(std::move(device));
            }
        }
    }

    // One row per GPU: utilization (the share of the driver's last sample period in which a
    // kernel ran), memory controller utilization and the graphics clock in MHz.
    std::string Sample(u32 window) const {
        std::string out;
        for (u32 index = 0; index < devices.size(); ++index) {
            const Device& device = devices[index];
            Utilization utilization{};
            u32 clock = 0;
            if (get_utilization(device.handle, &utilization) != Success) {
                continue;
            }
            get_clock(device.handle, ClockGraphics, &clock);
            out += fmt::format("gpu\t{}\t{}\t{}\t{}\t{}\t{}\n", window, index, device.name,
                               utilization.gpu, utilization.memory, clock);
        }
        return out;
    }

    bool Loaded() const {
        return !devices.empty();
    }

private:
    struct Utilization {
        u32 gpu;
        u32 memory;
    };
    struct Device {
        void* handle;
        std::string name;
    };
    static constexpr int Success = 0;
    static constexpr int ClockGraphics = 0;

    template <typename Function>
    bool Load(Function*& function, const char* name) {
#ifdef _WIN32
        function = reinterpret_cast<Function*>(GetProcAddress(library, name));
#elif defined(__linux__)
        function = reinterpret_cast<Function*>(dlsym(library, name));
#endif
        return function != nullptr;
    }

#ifdef _WIN32
    HMODULE library = nullptr;
#else
    void* library = nullptr;
#endif
    int (*init)() = nullptr;
    int (*get_count)(u32*) = nullptr;
    int (*get_handle)(u32, void**) = nullptr;
    int (*get_name)(void*, char*, u32) = nullptr;
    int (*get_utilization)(void*, Utilization*) = nullptr;
    int (*get_clock)(void*, int, u32*) = nullptr;
    std::vector<Device> devices;
};

// Once a second writes the window's game frames, presented frames, library calls, the CPU time of
// every thread and the GPU utilization, and every DumpSamples seconds rewrites the call tables.
void SamplerThread() {
    Common::SetCurrentThreadName("shadPS4:Gr2GfxTrace");
    const u64 tsc_hz = Common::EstimateRDTSCFrequency();
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "gr2_gfx_cpu.tsv";
    std::ofstream file{path, std::ios::trunc};
    file << fmt::format("# tsc_hz {}\n", tsc_hz)
         << "# window\tindex\tend_ms\tlength_ms\tgame_frames\tpresented_frames\t"
            "draws_per_presented_frame\tdispatches_per_presented_frame\n"
            "# calls\tindex\tfunction\tcalls\n"
            "# cpu\tindex\tos_thread_id\tname\tcpu_ms\tprobe_ms\n"
            "# gpu\tindex\tdevice\tname\tutilization_percent\tmemory_utilization_percent\t"
            "graphics_clock_mhz\n"
            "# wait\tindex\tos_thread_id\tname\tfunction\tcalls\ttotal_ms\tlongest_ms\n";
    const Nvml nvml;
    if (!nvml.Loaded()) {
        file << "# no NVIDIA management library, so no gpu rows\n";
    }
    file.flush();

    std::unordered_map<u64, u64> last_cpu;
    for (const CpuTime& thread : ReadCpuTimes(tsc_hz)) {
        last_cpu[thread.os_id] = thread.nanoseconds;
    }
    std::array<u64, std::size(Hooks)> last_calls{};
    for (u32 id = 0; id < std::size(Hooks); ++id) {
        last_calls[id] = g_stats[id].calls.load(std::memory_order_relaxed);
    }
    std::array<u64, MaxThreads> last_probe{};
    for (u32 slot = 0; slot < MaxThreads; ++slot) {
        last_probe[slot] = g_probe_threads[slot].probe_cycles.load(std::memory_order_relaxed);
    }
    struct TimedTotals {
        u64 calls;
        u64 nanoseconds;
    };
    std::vector<std::array<TimedTotals, std::size(TimedHooks)>> last_timed(MaxThreads);
    std::vector<std::string> names(MaxThreads);
    u64 last_frames = g_frames.load(std::memory_order_relaxed);
    u32 last_presents = DebugState.GetFrameNum();
    auto last_time = Clock::now();
    auto next_time = last_time;

    for (u32 window = 1;; ++window) {
        next_time += SampleInterval;
        std::this_thread::sleep_until(next_time);
        const auto now = Clock::now();
        const auto cpu = ReadCpuTimes(tsc_hz);
        const u64 frames = g_frames.load(std::memory_order_relaxed);
        const u32 presents = DebugState.GetFrameNum();

        std::string out = fmt::format("window\t{}\t{:.1f}\t{:.1f}\t{}\t{}\t{:.1f}\t{:.1f}\n",
                                      window, Millis(now - g_start), Millis(now - last_time),
                                      frames - last_frames, presents - last_presents,
                                      DebugState.GetDrawCallsAvg(), DebugState.GetDispatchesAvg());
        for (u32 id = 0; id < std::size(Hooks); ++id) {
            const u64 calls = g_stats[id].calls.load(std::memory_order_relaxed);
            if (calls != last_calls[id]) {
                out += fmt::format("calls\t{}\t{:#x}\t{}\n", window, Hooks[id].address,
                                   calls - last_calls[id]);
                last_calls[id] = calls;
            }
        }
        std::unordered_map<u64, u64> probe_ns;
        for (u32 slot = 0; slot < MaxThreads; ++slot) {
            const u64 cycles = g_probe_threads[slot].probe_cycles.load(std::memory_order_relaxed);
            probe_ns[g_probe_threads[slot].os_id.load(std::memory_order_relaxed)] +=
                static_cast<u64>(static_cast<double>(cycles - last_probe[slot]) * 1e9 / tsc_hz);
            last_probe[slot] = cycles;
        }
        for (const CpuTime& thread : cpu) {
            u64& last = last_cpu[thread.os_id];
            const u64 ran = thread.nanoseconds - std::min(last, thread.nanoseconds);
            last = thread.nanoseconds;
            const auto probe = probe_ns.find(thread.os_id);
            const u64 probed = probe != probe_ns.end() ? probe->second : 0;
            if (ran != 0 || probed != 0) {
                out += fmt::format("cpu\t{}\t{}\t{}\t{:.3f}\t{:.3f}\n", window, thread.os_id,
                                   thread.name, ran / 1e6, probed / 1e6);
            }
        }
        u32 thread_count;
        {
            std::scoped_lock lock{g_thread_mutex};
            thread_count = g_thread_count;
            for (u32 slot = 0; slot < thread_count; ++slot) {
                names[slot] = g_thread_names[slot];
            }
        }
        for (u32 slot = 0; slot < thread_count; ++slot) {
            const u64 os_id = g_probe_threads[slot].os_id.load(std::memory_order_relaxed);
            for (u32 hook = 0; hook < std::size(TimedHooks); ++hook) {
                TimedStats& stats = g_timed[slot][hook];
                TimedTotals& last = last_timed[slot][hook];
                const u64 calls = stats.calls.load(std::memory_order_relaxed);
                if (calls == last.calls) {
                    continue;
                }
                const u64 nanoseconds = stats.nanoseconds.load(std::memory_order_relaxed);
                const u64 longest = stats.max_nanoseconds.exchange(0, std::memory_order_relaxed);
                out += fmt::format("wait\t{}\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\n", window, os_id,
                                   names[slot], TimedHooks[hook].name, calls - last.calls,
                                   (nanoseconds - last.nanoseconds) / 1e6, longest / 1e6);
                last = {calls, nanoseconds};
            }
        }
        out += nvml.Sample(window);
        file << out;
        file.flush();

        if (window % DumpSamples == 0) {
            Dump();
        }
        last_frames = frames;
        last_presents = presents;
        last_time = now;
    }
}

template <typename T>
T ReadGuest(u64 address) {
    T value;
    std::memcpy(&value,
                reinterpret_cast<const void*>(MemoryPatcher::g_eboot_address + address - CodeBase),
                sizeof(value));
    return value;
}

// Polls the frame counter, the presented frame count and the frame labels every PollInterval
// and writes them with the timed calls of the timeline threads to gr2_gfx_timeline.tsv.
void TimelineThread() {
    Common::SetCurrentThreadName("shadPS4:Gr2GfxTimeline");
    const auto path =
        Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "gr2_gfx_timeline.tsv";
    std::ofstream file{path, std::ios::trunc};
    file << "# times in microseconds since the trace started\n"
            "# call\tstart\tduration\tthread\tfunction\tfirst_argument: timed calls of DrawThread "
            "and Game:Main\n"
            "# mark\ttime\tframe\tcount: the main loop's frame counter moved\n"
            "# mark\ttime\tdraw\tnumber: DrawThread started the frame whose label gets this "
            "value\n"
            "# mark\ttime\tpresent\tcount: a frame was presented\n"
            "# mark\ttime\tlabel\tindex\tvalue: a frame label got its value, so the GPU front end "
            "has processed that frame's commands\n";
    file.flush();
    if (FrameLabelsAddress + FrameLabels * 16 - CodeBase > MemoryPatcher::g_eboot_image_size) {
        return;
    }
    u32 last_frame = ReadGuest<u32>(FrameCounterAddress);
    u32 last_draw = ReadGuest<u32>(DrawFrameAddress);
    u32 last_present = DebugState.GetFrameNum();
    std::array<u64, FrameLabels> last_label{};
    std::vector<std::string> names(MaxThreads);
    u64 tail = 0;
    u64 lost = 0;
    auto next_flush = Clock::now() + TimelineFlushInterval;
    std::string out;
    for (;;) {
        Common::AccurateSleep(PollInterval, nullptr, false);
        const double now_us = NowNs() / 1e3;
        if (const u32 frame = ReadGuest<u32>(FrameCounterAddress); frame != last_frame) {
            last_frame = frame;
            out += fmt::format("mark\t{:.1f}\tframe\t{}\n", now_us, frame);
        }
        if (const u32 draw = ReadGuest<u32>(DrawFrameAddress); draw != last_draw) {
            last_draw = draw;
            out += fmt::format("mark\t{:.1f}\tdraw\t{}\n", now_us, draw);
        }
        if (const u32 present = DebugState.GetFrameNum(); present != last_present) {
            last_present = present;
            out += fmt::format("mark\t{:.1f}\tpresent\t{}\n", now_us, present);
        }
        // The labels are read once DrawThread has waited on them, so they are mapped.
        if (g_frames.load(std::memory_order_relaxed) >= 2) {
            for (u32 i = 0; i < FrameLabels; ++i) {
                const u64 label = ReadGuest<u64>(FrameLabelsAddress + i * 16);
                if (label == 0 || label % 8 != 0) {
                    continue;
                }
                u64 value;
                std::memcpy(&value, reinterpret_cast<const void*>(label), sizeof(value));
                if (value != last_label[i]) {
                    last_label[i] = value;
                    out += fmt::format("mark\t{:.1f}\tlabel\t{}\t{}\n", now_us, i, value);
                }
            }
        }
        if (Clock::now() < next_flush) {
            continue;
        }
        next_flush = Clock::now() + TimelineFlushInterval;
        {
            std::scoped_lock lock{g_thread_mutex};
            for (u32 slot = 0; slot < g_thread_count; ++slot) {
                names[slot] = g_thread_names[slot];
            }
        }
        const u64 head = g_event_head.load(std::memory_order_acquire);
        if (head - tail > g_events.size()) {
            lost += head - tail - g_events.size();
            tail = head - g_events.size();
        }
        for (; tail < head; ++tail) {
            const CallEvent& event = g_events[tail & (g_events.size() - 1)];
            const u64 seq = event.seq.load(std::memory_order_acquire);
            if (seq < tail + 1) {
                break;
            }
            // A writer a whole ring ahead can overwrite the event while it is copied, which seq
            // shows once it is read again.
            const u64 start = event.start_ns;
            const u64 duration = event.duration_ns;
            const u64 argument = event.argument;
            const u32 hook = event.hook;
            const u32 slot = event.thread_slot;
            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq != tail + 1 || event.seq.load(std::memory_order_relaxed) != seq ||
                hook >= std::size(TimedHooks) || slot >= MaxThreads) {
                ++lost;
                continue;
            }
            out += fmt::format("call\t{:.1f}\t{:.1f}\t{}\t{}\t{:#x}\n", start / 1e3, duration / 1e3,
                               names[slot], TimedHooks[hook].name, argument);
        }
        if (lost != 0) {
            out += fmt::format("# lost {} timed calls\n", lost);
            lost = 0;
        }
        file << out;
        file.flush();
        out.clear();
    }
}

} // namespace

void Install(VAddr module_base) {
    if (module_base != MemoryPatcher::g_eboot_address || !Common::IsGravityRush2()) {
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
    g_start = Clock::now();
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
    // A timed hook's trampoline runs the covered instructions and jumps back into the function;
    // its stub calls the trampoline between the two timing calls.
    u32 timed = 0;
    for (u32 id = 0; id < std::size(TimedHooks); ++id) {
        const TimedHook& hook = TimedHooks[id];
        u8* entry = reinterpret_cast<u8*>(module_base + hook.address - CodeBase);
        if (hook.address - CodeBase + hook.size > MemoryPatcher::g_eboot_image_size ||
            std::memcmp(entry, hook.bytes.data(), hook.size) != 0) {
            LOG_WARNING(Loader,
                        "Graphics trace: the code of {} at {:#x} is not the 01.11 "
                        "executable's",
                        hook.name, hook.address);
            continue;
        }
        const auto start = c.getSize();
        const u8* trampoline = c.getCurr();
        if (!Relocate(entry, hook.size, c)) {
            c.setSize(start);
            LOG_WARNING(Loader, "Graphics trace: the code of {} at {:#x} cannot be moved",
                        hook.name, hook.address);
            continue;
        }
        c.jmp(entry + hook.size);
        const u8* stub = c.getCurr();
        EmitTimedHook(c, id, trampoline);
        const s32 jump = static_cast<s32>(stub - (entry + JmpSize));
        entry[0] = 0xe9;
        std::memcpy(entry + 1, &jump, sizeof(jump));
        std::memset(entry + JmpSize, 0xcc, hook.size - JmpSize);
        ++timed;
    }
    LOG_INFO(Loader,
             "Graphics trace: hooked {} of {} library functions and timed {} of {} calls, {} "
             "bytes of stubs",
             hooked, std::size(Hooks), timed, std::size(TimedHooks), c.getSize());
    if (hooked == 0 && timed == 0) {
        return;
    }
    if (!frame_hooked) {
        LOG_WARNING(Loader, "Graphics trace: submitAndFlip is not hooked, so no game frames are "
                            "counted");
    }
    std::thread{SamplerThread}.detach();
    std::thread{TimelineThread}.detach();
    LOG_INFO(Loader,
             "Graphics trace: gr2_gfx_cpu.tsv in the log folder gets a row group every second, "
             "gr2_gfx_timeline.tsv gets every frame's timed calls and marks, gr2_gfx_trace.tsv is "
             "rewritten every {} seconds",
             DumpSamples);
}

} // namespace Gr2GfxTrace
