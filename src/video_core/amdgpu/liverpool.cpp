// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/preprocessor/stringize.hpp>
#include <chrono>
#include <cstdint>
#include <thread>

#include "common/assert.h"
#include "common/config.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}
inline void CopyRegWordsFast(u32* dst, const u32* src, u32 word_count) {
    switch (word_count) {
        case 0:
            return;
        case 1:
            dst[0] = src[0];
            return;
        case 2:
            dst[0] = src[0];
            dst[1] = src[1];
            return;
        case 3:
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            return;
        case 4:
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            return;
        case 5:
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            dst[4] = src[4];
            return;
        case 6:
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            dst[4] = src[4];
            dst[5] = src[5];
            return;
        case 7:
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            dst[4] = src[4];
            dst[5] = src[5];
            dst[6] = src[6];
            return;
        case 8:
            dst[0] = src[0];
            dst[1] = src[1];
            dst[2] = src[2];
            dst[3] = src[3];
            dst[4] = src[4];
            dst[5] = src[5];
            dst[6] = src[6];
            dst[7] = src[7];
            return;
        default:
            std::memcpy(dst, src, static_cast<size_t>(word_count) * sizeof(u32));
            return;
    }
}

inline void CopyBytesFast(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    const auto dst_addr = reinterpret_cast<std::uintptr_t>(dst);
    const auto src_addr = reinterpret_cast<std::uintptr_t>(src);
    if (((dst_addr | src_addr | bytes) & 0x3u) == 0 && bytes <= 128) {
        CopyRegWordsFast(reinterpret_cast<u32*>(dst), reinterpret_cast<const u32*>(src),
                         static_cast<u32>(bytes >> 2));
        return;
    }
    std::memcpy(dst, src, bytes);
}

[[gnu::always_inline]] inline bool CopyRegWordsFastIfChangedSmall(
    u32* dst, const u32* src, u32 word_count) {
    // PERF(v8): Use 64-bit comparisons (memcmp-like) to halve the number of branches
    // for the common 2-8 word cases. The compiler can vectorize 64-bit loads.
    const auto* s64 = reinterpret_cast<const u64*>(src);
    const auto* d64 = reinterpret_cast<const u64*>(dst);
    switch (word_count) {
        case 0:
            return false;
        case 1:
            if (dst[0] == src[0]) return false;
            dst[0] = src[0];
            return true;
        case 2:
            if (d64[0] == s64[0]) return false;
            dst[0] = src[0]; dst[1] = src[1];
            return true;
        case 3:
            if (d64[0] == s64[0] && dst[2] == src[2]) return false;
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            return true;
        case 4:
            if (d64[0] == s64[0] && d64[1] == s64[1]) return false;
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
            return true;
        case 5:
            if (d64[0] == s64[0] && d64[1] == s64[1] && dst[4] == src[4]) return false;
            CopyRegWordsFast(dst, src, word_count);
            return true;
        case 6:
            if (d64[0] == s64[0] && d64[1] == s64[1] && d64[2] == s64[2]) return false;
            CopyRegWordsFast(dst, src, word_count);
            return true;
        case 7:
            if (d64[0] == s64[0] && d64[1] == s64[1] && d64[2] == s64[2] && dst[6] == src[6]) return false;
            CopyRegWordsFast(dst, src, word_count);
            return true;
        case 8:
            if (d64[0] == s64[0] && d64[1] == s64[1] && d64[2] == s64[2] && d64[3] == s64[3]) return false;
            CopyRegWordsFast(dst, src, word_count);
            return true;
        default: {
            const size_t bytes = static_cast<size_t>(word_count) * sizeof(u32);
            if (bytes <= 128 && std::memcmp(dst, src, bytes) == 0) {
                return false;
            }
            CopyRegWordsFast(dst, src, word_count);
            return true;
        }
    }
}

inline bool CopyRegWordsFastIfChangedSmall(u32* dst, const PM4Header* src, u32 word_count) {
    return CopyRegWordsFastIfChangedSmall(dst, reinterpret_cast<const u32*>(src), word_count);
}

inline void CopyRegWordsFast(u32* dst, const PM4Header* src, u32 word_count) {
    CopyRegWordsFast(dst, reinterpret_cast<const u32*>(src), word_count);
}

Liverpool::Liverpool() {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::ProcessCommands() {
    Common::UniqueFunction<void> callback{};
    while (num_commands.load(std::memory_order_acquire)) {
        if (!command_queue_.TryPop(callback)) {
            std::this_thread::yield();
            continue;
        }
        num_commands.fetch_sub(1, std::memory_order_relaxed);
        callback();
        // S-2: a host-side command callback (rasterizer ops, resource
        // updates pushed by other threads via SendCommand) executing
        // counts as forward progress for the stall-detection in
        // Process(). If commands are flowing through, we are not stalled.
        ++pm4_progress_counter_;
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();

    // PERF(GR2FORK): pin GpuComm to a host-side physical core that's not
    // contended by guest work.
    //
    // GR2 partitions every guest thread onto host CPUs 0..6 via the now-
    // honored Pthread::SetAffinity. With no host affinity policy in place
    // GpuComm floats wherever the OS scheduler wedges it — typically a SMT
    // sibling of an overloaded guest core, where it timeshares execution
    // units with JoBwOrKeR / MTTW / Havok and bleeds both throughputs.
    //
    // Mask is computed dynamically from /sys (Linux) or the Win32 topology
    // API. Behavior by host topology:
    //
    //   * 8c/16t Z1 Extreme (Zen 4 Phoenix), 8c/16t Ryzen 7, 12c/24t+ Ryzen
    //     9 / Threadripper: tight pin to {CPU 7, CPU 7's SMT sibling} — a
    //     genuinely-free physical core no guest thread can run on.
    //
    //   * 6c/12t Ryzen 5 (3600/5600/7600), i7-8700, 4c/8t Steam Deck Aerith:
    //     no truly-free physical core (CPU 7 is the SMT alternate of a
    //     guest-claimed core). Pin to the broader "host CPUs not in 0..6"
    //     pool — bits 7..11 on 6c/12t, just bit 7 on 4c/8t. GpuComm still
    //     SMT-fights guest threads on the parent physical core, but is
    //     guaranteed never to land on the same logical CPU as a guest
    //     thread, which is strictly better than no pin at all.
    //
    //   * <8 logical CPUs (6c/6t, 4c/4t, etc.): GetReservedCoreMask returns
    //     0 — the if() falls through, no pin happens, OS scheduler decides.
    //
    // PERF(GR2FORK): pin GpuComm to a host-side physical core that's not
    // contended by guest work, and (on hosts that don't naturally provide
    // one) STEAL one from the guest's claimed range to give it exclusively.
    //
    // Without affinity GpuComm is the documented single-threaded bottleneck:
    // on Steam Deck it drops fps into the teens because GpuComm SMT-fights
    // a guest thread continuously. Isolating GpuComm on a private physical
    // core recovers most of that loss; the guest can absorb the loss of
    // one core (3-of-4 cores on Steam Deck still hits ~30 fps in GR2,
    // 5-of-6 on Ryzen 5).
    //
    // Mask is computed dynamically from /sys (Linux) or the Win32 topology
    // API. Behavior by host topology:
    //
    //   * 8c/16t Z1 Extreme, 8c/16t Ryzen 7, 12c/24t+ Ryzen 9, Threadripper:
    //     CPU 7's physical core is naturally outside the guest's claimed
    //     range. Tight pin to {CPU 7, sibling}, exclusion walk runs, no
    //     theft from guest.
    //
    //   * 8c/8t SMT-off: same story with single-thread cores. Pin to {7}.
    //
    //   * 6c/12t Ryzen 5 (3600/5600/7600), i7-8700: enumerate physical cores,
    //     steal the highest (#5, CPUs {5, 11}). Pin GpuComm to {5, 11},
    //     strip bit 5 from every guest thread mask via Pthread::SetAffinity,
    //     run the exclusion walk. Guest goes 6 -> 5 physical cores; GpuComm
    //     gets one private.
    //
    //   * 6c/6t (no SMT, e.g. i5-9400): steal physical core 5 (CPU 5).
    //     Strip bit 5 from guest. 6 -> 5 cores.
    //
    //   * 4c/8t Steam Deck Aerith: steal physical core 3 (CPUs {3, 7}).
    //     Strip bit 3 from guest. 4 -> 3 cores. This is the host where
    //     isolation matters most because of the GpuComm bottleneck.
    //
    //   * 4c/4t and below: GetReservedCoreMask returns 0 — no pin, no walk.
    //     Stealing here would leave the guest with too few cores.
    //
    // ExcludeReservedCoresFromAllOtherThreads walks /proc/self/task (Linux)
    // or Toolhelp32Snapshot (Windows) once and AND-strips the reserved bits
    // from every other thread's mask. Forward coverage for not-yet-spawned
    // threads is automatic via pthread_create / CreateThread inheritance.
    //
    // The guest-side strip and SMT-widen (bit-stripping AND OR-expansion
    // inside Pthread::SetAffinity) is applied at every guest scePthread-
    // Setaffinity call. The strip narrows guest masks to exclude any
    // stolen GpuComm core; the widen ORs in the SMT siblings of guest's
    // physical cores so the OS scheduler can place guest threads on
    // previously-idle SMT halves (CPUs 8..14 on Z1 Extreme, 12..18 on
    // 12c/24t Ryzen 9, 7..10 on 6c/12t, etc.). Even guest threads that
    // hadn't existed at exclusion-walk time get the right mask.
    //
    // Best-effort: any individual syscall failure is a logged warning, not
    // fatal — the emulator still works, it just falls back to the OS
    // scheduler's choice for that thread.
    if (const u64 mask = Common::GetReservedCoreMask()) {
        Common::SetCurrentThreadAffinityMask(mask);
        Common::ExcludeReservedCoresFromAllOtherThreads();
        // PERF(GR2FORK): start the periodic re-walk thread to catch
        // post-startup thread spawns. Critical on Windows (CreateThread
        // doesn't inherit thread-level affinity — every Havok / AvDemuxer
        // / audio worker spawned by the game after this point comes back
        // with the full host mask and can land on GpuComm's reserved
        // core). Mostly a redundancy net on Linux where pthread_create
        // inheritance already covers most cases. Idempotent — safe to
        // call again on every Liverpool::Process invocation but only the
        // first call actually spawns the worker.
        Common::StartPeriodicAffinityRewalk();
    }

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{wake_mutex_};
            Common::CondvarWait(wake_cv_, lk, stoken, [this] {
                return num_commands.load(std::memory_order_acquire) ||
                       num_submits.load(std::memory_order_acquire) ||
                       submit_done.load(std::memory_order_acquire);
            });
        }
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();

        curr_qid = -1;

        // === GR2FORK S-2: all-coroutines-stalled detection ===
        //
        // Each iteration of the inner loop visits one queue (graphics or
        // a compute pipe), invoking ProcessCommands and then resuming
        // that queue's PM4 coroutine. If neither the resumed coroutine
        // nor ProcessCommands made any forward progress (no new packet
        // entered, no host-command callback fired) for many consecutive
        // iterations, every queue is stuck spinning on a wait — and the
        // signal that resolves any wait we host-side honor (assembler
        // fence writes after Vulkan completion, presenter VO labels)
        // comes from ANOTHER thread, so this thread can sleep without
        // delaying its own progress.
        //
        // pm4_progress_counter_ is bumped at the top of every PM4
        // dispatch loop (per packet entered, including the entry-into-
        // wait packet) and per ProcessCommands callback. The bump
        // discipline makes a stalled wait-spin observable: the wait
        // bumps once on entry, then YIELD_GFX/YIELD_ASC re-enters the
        // case body without re-entering the dispatch loop, so no
        // additional bumps fire until the wait's Test() returns true
        // (or the queue's task ends).
        //
        // Threshold: 256 iterations. With both queues stuck spinning,
        // each iteration of this loop is ~200-300 cycles (peek + resume
        // + Test + YIELD + advance), so 256 × ~250 ≈ 25 µs of
        // confirmed-stall before sleeping. The 50 µs sleep then frees
        // the CPU until either a guest write resolves a fence (most
        // common) or another submit arrives.
        //
        // Hysteresis: after a sleep we reset stalled_iters_local to 224,
        // not zero — if we wake to find we're still stalled, the next
        // sleep arrives in ~32 iterations (~8 µs) rather than another
        // full 25 µs. Keeps the steady-state CPU usage low during long
        // waits without thrash if we wake into briefly-resumed work.
        //
        // Why this is structurally different from S-1 (which regressed):
        // S-1 modified the snapshot pool's wait primitive — a producer/
        // consumer wire whose surrounding kernel-mediated coherence
        // turned out to be load-bearing. This change touches the OUTER
        // dispatcher loop's behavior between coroutine resumes; the
        // signals being waited on cross thread boundaries (assembler
        // and presenter), not inter-core within GpuComm's own pipeline.
        u64 progress_at_check = pm4_progress_counter_;
        u32 stalled_iters_local = 0;
        constexpr u32 kStalledIterThreshold = 256;
        constexpr u32 kStalledIterHysteresis = kStalledIterThreshold - 32;

        while (num_submits.load(std::memory_order_acquire) ||
               num_commands.load(std::memory_order_acquire)) {
            // S-2 stall check at iteration top: observes whether the
            // PRIOR iteration produced any forward progress. Placed
            // here (rather than at iteration bottom) so that early-
            // continue paths — empty queues, the locked peek that
            // finds no submits — naturally count as stalled iterations
            // without needing the check duplicated.
            const u64 progress_now = pm4_progress_counter_;
            if (progress_now != progress_at_check) {
                stalled_iters_local = 0;
                progress_at_check = progress_now;
            } else if (++stalled_iters_local >= kStalledIterThreshold) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                stalled_iters_local = kStalledIterHysteresis;
                // Re-sync progress_at_check so the iter immediately
                // after wake-up starts a fresh window. Without this,
                // a stale progress_at_check from before the sleep
                // could erroneously indicate progress on the next
                // iteration if the counter advanced during sleep
                // (e.g. another thread enqueued a host command).
                progress_at_check = pm4_progress_counter_;
            }

            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues.load(std::memory_order_acquire);


            auto& queue = mapped_queues[curr_qid];

            // Fast path: avoid locking empty queues every round.
            if (queue.submit_count.load(std::memory_order_relaxed) == 0) {
                continue;
            }

            // PERF(GR2FORK v1.65): use cached front handle when available.
            // The peek lock only fires when transitioning between tasks
            // (current_task null). v1.64 [LP_MUTEX_METRICS] showed the
            // un-cached version fired this lock on every coroutine yield
            // (~99.88% of all mutex acquisitions, ~3.5% of GpuComm CPU).
            // FIFO invariant: producer pushes don't change front;
            // GpuComm is the only popper. While current_task is set,
            // submit_count >= 1 (producer increments before observable
            // push, we decrement after pop), so the fast-path check
            // above won't evict a valid cached task.
            if (!queue.current_task) {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                queue.current_task = queue.submits.front();
            }

            queue.current_task.resume();


            if (queue.current_task.done()) {
                queue.current_task.destroy();
                queue.current_task = {};

                {
                    std::scoped_lock lock{queue.m_access};
                    queue.submits.pop();
                }
                queue.submit_count.fetch_sub(1, std::memory_order_relaxed);

                num_submits.fetch_sub(1, std::memory_order_release);
                NotifyIdle();
            }
        }

        if (submit_done.load(std::memory_order_acquire)) {
            VideoCore::EndCapture();
            if (rasterizer) [[likely]] {
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done.store(false, std::memory_order_release);
        }

        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        // S-2: each packet entered is one unit of forward progress for
        // stall detection. Bump unconditionally at loop top so every
        // packet type counts (including type-2 padding); the inner
        // spin of wait-style opcodes re-enters the case body without
        // re-entering this while loop, which is exactly the "stalled"
        // signal Process() consumes.
        ++pm4_progress_counter_;

        if (num_commands.load(std::memory_order_relaxed)) {
            ProcessCommands();
        }

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);

    cblock.Reset();
    pipeline_dirty_ = false;
    gfx_key_dirty_ = false;
    dynamic_dirty_ = true;  // First draw needs full dynamic state setup
    u32 packet_count = 0;
    const bool guest_markers = rasterizer && Config::getVkGuestMarkersEnabled();
    // PERF(GR2FORK): mirror guest_markers — Config::getVkHostMarkersEnabled()
    // is a non-inlined accessor (Setting<bool>::get()) and was previously
    // called once per Draw* opcode case below (9 sites in this function),
    // adding a function call + atomic-relaxed load + cmpb on the per-draw
    // fast path. Hoist once per ProcessGraphics resume; the 9 sites use
    // host_markers below. Aligns with the pattern pre-release upstream
    // already follows for both flags.
    const bool host_markers = rasterizer && Config::getVkHostMarkersEnabled();
    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }
    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    while (!dcb.empty()) {
        // S-2: bump per-packet (see member-variable comment in liverpool.h
        // for full rationale). Wait-style opcodes (WaitRegMem,
        // WaitOnCeCounter, the IB completion poll a few cases below)
        // bump exactly once on case entry, then spin internally without
        // re-entering this loop — that single bump is the entry-into-
        // wait signal; subsequent absence of bumps is the in-wait
        // signal.
        ++pm4_progress_counter_;

        // PERF(GR2FORK v1.20): the bitmask gate is true on 1 in 32 packets
        // by construction; num_commands is also typically zero in steady-
        // state DCB processing. The body fires very rarely — keep it out
        // of the straight-line fast path so the per-packet dispatch loop
        // stays compact.
        if ((packet_count++ & 31) == 0 &&
            num_commands.load(std::memory_order_relaxed)) [[unlikely]] {
            ProcessCommands();
        }

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (!guest_markers) {
                        break;
                    }
                    const auto marker_sz = nop->header.count.Value() * 2;
                    const std::string_view label{reinterpret_cast<const char*>(&nop->data_block[1]),
                        marker_sz};
                    if (rasterizer) [[likely]] {
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (!guest_markers) {
                        break;
                    }
                    const auto marker_sz = nop->header.count.Value() * 2;
                    const std::string_view label{reinterpret_cast<const char*>(&nop->data_block[1]),
                        marker_sz};
                    const u32 color = *reinterpret_cast<const u32*>(
                        reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                    if (rasterizer) [[likely]] {
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (!guest_markers) {
                        break;
                    }
                    if (rasterizer) [[likely]] {
                        rasterizer->ScopeMarkerEnd(true);
                    }

                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                MarkGfxKeyDirty();
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                const bool cfg_regs_changed =
                CopyRegWordsFastIfChangedSmall(&regs.reg_array[reg_addr], payload, count - 1);
                if (cfg_regs_changed) {
                    MarkGfxKeyDirty();
                }
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                const bool ctx_regs_changed =
                CopyRegWordsFastIfChangedSmall(&regs.reg_array[reg_addr], payload, count - 1);
                if (!ctx_regs_changed) {
                    break;
                }

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                // PERF(GR2FORK v1.23): the `if (ctx_regs_changed)` check
                // that was here is redundant — line 497 early-broke out of
                // this case statement if `ctx_regs_changed == false`, so by
                // construction reaching this point implies `ctx_regs_changed
                // == true`. The compiler may DCE the redundant compare
                // through the inner switch's complex control flow, but
                // eliminating it explicitly guarantees a cmp+jcc pair is
                // gone from the SetContextReg hot path (per-PM4-packet,
                // many times per frame).
                if (IsDynamicStateOnlyContextReg(reg_addr)) {
                    MarkGfxPipelineDirty();
                } else {
                    MarkGfxKeyDirty();
                }
                break;
            }
                case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    CopyRegWordsFast(addr, header + 2, static_cast<u32>(set_size / sizeof(u32)));

                } else {
                    if (CopyRegWordsFastIfChangedSmall(
                        &regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                        header + 2, count - 1)) {
                        MarkGfxKeyDirty();
                        }
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                if (CopyRegWordsFastIfChangedSmall(
                    &regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                    header + 2, count - 1)) {
                    MarkGfxKeyDirty();
                    }
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }

                if (rasterizer) [[likely]] {
                    FlushGfxPipelineDirty();
                    if (host_markers) [[unlikely]] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndex2", cmd_address));
                        rasterizer->Draw(true);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) [[likely]] {
                    FlushGfxPipelineDirty();
                    if (host_markers) [[unlikely]] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexOffset2", cmd_address));
                        rasterizer->Draw(true, draw_index_off->index_offset);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true, draw_index_off->index_offset);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) [[likely]] {
                    FlushGfxPipelineDirty();
                    if (host_markers) [[unlikely]] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexAuto", cmd_address));
                        rasterizer->Draw(false);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(false);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) [[likely]] {
                    FlushGfxPipelineDirty();
                    if (host_markers) [[unlikely]] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirect", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) [[likely]] {
                    FlushGfxPipelineDirty();
                    if (host_markers) [[unlikely]] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirect", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) [[likely]] {
                    FlushGfxPipelineDirty();
                    if (host_markers) [[unlikely]] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) [[likely]] {
                    FlushGfxPipelineDirty();
                    if (host_markers) [[unlikely]] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectCountMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                 ? draw_index_indirect->count_addr
                                                 : 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                 ? draw_index_indirect->count_addr
                                                 : 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    FlushGfxPipelineDirty();
                    if (host_markers) [[unlikely]] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchDirect", cmd_address));
                        rasterizer->DispatchDirect();
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchDirect();
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    FlushGfxPipelineDirty();
                    if (host_markers) [[unlikely]] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchIndirect", cmd_address));
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                    }
                }
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* results = event->Address<u64*>();
                        for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                            *results = pixel_counter | OcclusionCounterValidMask;
                        }
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                event_eos->SignalFence([](void* address, u64 data, u32 num_bytes) {
                    auto* memory = Core::Memory::Instance();
                    if (!memory->TryWriteBacking(address, &data, num_bytes)) {
                        memcpy(address, &data, num_bytes);
                    }
                });
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) [[likely]] {
                        rasterizer->Finish();
                        const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                        *event_eos->Address() = value;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                event_eop->SignalFence(
                    [](void* address, u64 data, u32 num_bytes) {
                        auto* memory = Core::Memory::Instance();
                        if (!memory->TryWriteBacking(address, &data, num_bytes)) {
                            memcpy(address, &data, num_bytes);
                        }
                    },
                    [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                           dma_data->data, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                           dma_data->NumBytes(), true, false);
                } else if (dma_data->src_sel == DmaDataSrc::Data &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                           dma_data->data, false);
                } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                           dma_data->NumBytes(), false, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                           dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                           false, false);
                } else {
                    UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                    u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                if (!write_data->wr_one_addr.Value()) {
                    CopyBytesFast(address, write_data->data, data_size);
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (mem_semaphore->IsSignaling()) {
                    mem_semaphore->Signal();
                } else {
                    while (!mem_semaphore->Signaled()) {
                        YIELD_GFX();
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    break;
                }
                while (!wait_reg_mem->Test(regs.reg_array)) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                if (rasterizer) [[likely]] {
                    rasterizer->CpSync();
                }
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        // S-2: bump per-packet (see ProcessGraphics comment + the member-
        // variable comment in liverpool.h).
        ++pm4_progress_counter_;

        if (num_commands.load(std::memory_order_relaxed)) {
            ProcessCommands();
        }

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), false, false);
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                CopyRegWordsFast(addr, header + 2, static_cast<u32>(set_size / sizeof(u32)));

            } else {
                if (CopyRegWordsFastIfChangedSmall(
                    &regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                    header + 2, static_cast<u32>(set_size / sizeof(u32)))) {
                    BumpGfxPipelineStamp();
                    }
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarkerBegin(
                    fmt::format("asc[{}]:{}:DispatchDirect", vqid, cmd_address));
                rasterizer->DispatchDirect();
                rasterizer->ScopeMarkerEnd();
            }
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) [[unlikely]] {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                if (Config::getVkHostMarkersEnabled()) [[unlikely]] {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchIndirect", vqid, cmd_address));
                    rasterizer->DispatchIndirect(ib_address, 0, size);
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchIndirect(ib_address, 0, size);
                }
            }
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (!write_data->wr_one_addr.Value()) {
                CopyBytesFast(write_data->Address<void*>(), write_data->data, data_size);
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
            } else {
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            while (!wait_reg_mem->Test(regs.reg_array)) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            release_mem->SignalFence([pipe_id = queue.pipe_id] {
                Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
            });
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    if (Config::copyGPUCmdBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }


    auto task = ProcessGraphics(dcb, ccb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }
    queue.submit_count.fetch_add(1, std::memory_order_relaxed);

    num_submits.fetch_add(1, std::memory_order_release);
    NotifyGpu();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }
    queue.submit_count.fetch_add(1, std::memory_order_relaxed);

    u32 prev = num_mapped_queues.load(std::memory_order_relaxed);
    while (prev < gnm_vqid + 1) {
        if (num_mapped_queues.compare_exchange_weak(prev, gnm_vqid + 1,
                                                     std::memory_order_release,
                                                     std::memory_order_relaxed)) {
            break;
        }
    }

    num_submits.fetch_add(1, std::memory_order_release);
    NotifyGpu();
}

} // namespace AmdGpu
