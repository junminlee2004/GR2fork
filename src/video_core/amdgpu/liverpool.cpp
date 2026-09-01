// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/preprocessor/stringize.hpp>

#include "common/assert.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

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

static SHAD_NO_INLINE std::span<const u32> NextPacketOverflow(size_t offset, size_t remaining) {
    LOG_ERROR(Lib_GnmDriver,
              ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
              "submission dwords={}",
              offset, remaining);
    // Return empty subspan so check for next packet bails out
    return {};
}

// The log machinery in the overflow branch kept this out of line at packet
// rate; split so the common path inlines.
static inline std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) [[unlikely]] {
        return NextPacketOverflow(offset, span.size());
    }
    return span.subspan(offset);
}

// The messages are formatted here rather than in the parser loop: fmt binds its
// arguments by reference, and an address-exposed local in a coroutine is pinned
// to the frame and written through a spilled pointer on every packet.
[[noreturn]] static SHAD_NO_INLINE void UnhandledPm4Type(const PM4Header* header) {
    if (header->type == 0) {
        UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                        header->type0.base.Value(), header->type0.NumWords());
    }
    UNREACHABLE_MSG("Wrong PM4 type {}", header->type.Value());
}

[[noreturn]] static SHAD_NO_INLINE void UnknownPm4Opcode(u32 opcode, u32 count) {
    UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}", opcode, count);
}

// Must be the first statement of every draw arm, above the rasterizer guard, so
// the counts are identical with and without a rasterizer. There is no central
// opcode list any more: a new draw opcode needs its own call here.
static void CountDraw(Liverpool::PacketStats& stats, const PM4Header* header) {
    ++stats.draws;
    if (header->type3.predicate.Value() == PM4Predicate::PredEnable) {
        ++stats.predicated_draws;
    }
}

// The per-stage user_data words are excluded from the graphics state stamp by
// contract: engines rewrite them every draw, and marking them would advance
// the stamp per draw and starve every stamp-keyed cache for zero correctness
// benefit (no cache consumes user_data through the stamp; the binding probe
// compares the words directly). Writes that only partially overlap a block
// fall through to the stamped path, which is the conservative direction.
static bool IsGfxUserDataWrite(const Regs& regs, u32 word_addr, u32 num_words) {
    const u32* base = regs.reg_array.data();
    const auto in_ud = [&](const ShaderProgram& prog) {
        const u32 ud = static_cast<u32>(reinterpret_cast<const u32*>(prog.user_data.data()) - base);
        return word_addr >= ud && word_addr + num_words <= ud + NUM_USER_DATA;
    };
    return in_ud(regs.ps_program) || in_ud(regs.vs_program) || in_ud(regs.gs_program) ||
           in_ud(regs.es_program) || in_ud(regs.hs_program) || in_ud(regs.ls_program);
}

// Only for blocks IsGfxUserDataWrite has already bounded to NUM_USER_DATA
// words, so every real call lands on a fixed-size arm; the >64 arm carries the
// malformed packet whose NumWords() masked to 0 and underflowed the size, which
// faults the same way either way. Every copy must stay a constant size - a
// length-driven loop is turned back into a memcpy call.
static void StoreRegBlock(void* dst, const void* src, size_t bytes) {
    auto* d = static_cast<u8*>(dst);
    const auto* s = static_cast<const u8*>(src);
    if (bytes > 64) [[unlikely]] {
        std::memcpy(d, s, bytes);
    } else if (bytes > 32) {
        std::memcpy(d, s, 32);
        std::memcpy(d + bytes - 32, s + bytes - 32, 32);
    } else if (bytes > 16) {
        std::memcpy(d, s, 16);
        std::memcpy(d + bytes - 16, s + bytes - 16, 16);
    } else if (bytes > 8) {
        std::memcpy(d, s, 8);
        std::memcpy(d + bytes - 8, s + bytes - 8, 8);
    } else if (bytes >= 4) {
        std::memcpy(d, s, 4);
        std::memcpy(d + bytes - 4, s + bytes - 4, 4);
    }
}

Liverpool::Liverpool() {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    // Stamp dormancy is decided before the process thread can observe work;
    // when the adaptive skip caches are disabled the register funnel is a
    // plain memcpy with zero compares.
    gfx_stamp.active =
        EmulatorSettings.GetAdaptiveSkipCachesMode() != AdaptiveSkipCachesMode::SkipCachesDisabled;
    occlude_all_ = EmulatorSettings.IsOccludeAll();
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::DrainCommands() {
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken,
                                [this] { return num_commands || num_submits || submit_done; });
        }
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();

        curr_qid = -1;

        while (num_submits || num_commands) {
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            task.resume();

            if (task.done()) {
                task.destroy();

                std::scoped_lock lock{queue.m_access};
                queue.submits.pop();

                // The only submit_cv waiters on this edge block until the
                // queue drains (num_submits == 0); waking them per popped
                // submit burned a futex broadcast each. The lock is still
                // held for the notify so the wakeup cannot be lost.
                if (--num_submits == 0) {
                    std::scoped_lock lock2{submit_mutex};
                    submit_cv.notify_all();
                }
            }
        }

        if (submit_done) {
            VideoCore::EndCapture();
            if (rasterizer) {
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done = false;
        }

        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

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

void Liverpool::SetContextRegExtentTail(u32 reg_addr, const PM4Header* header, const u32* payload) {
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
            if (last_cb_extent[col_buf_id].raw != payload[nop_offset + 1]) {
                last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                gfx_stamp.MarkDirty();
            }
        } else if (last_cb_extent[col_buf_id].raw != 0) {
            last_cb_extent[col_buf_id].raw = 0;
            gfx_stamp.MarkDirty();
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
        const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Cmask) /
                                (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
        ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

        const auto nop_offset = header->type3.count;
        if (nop_offset == 0x04) {
            ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                       "NOP hint is missing in CB setup sequence");
            if (last_cb_extent[col_buf_id].raw != payload[nop_offset + 1]) {
                last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                gfx_stamp.MarkDirty();
            }
        }
        break;
    }
    case ContextRegs::DbZInfo: {
        if (header->type3.count == 8) {
            ASSERT_MSG(payload[20] == 0xc0001000, "NOP hint is missing in DB setup sequence");
            if (last_db_extent.raw != payload[21]) {
                last_db_extent.raw = payload[21];
                gfx_stamp.MarkDirty();
            }
        } else if (last_db_extent.raw != 0) {
            last_db_extent.raw = 0;
            gfx_stamp.MarkDirty();
        }
        break;
    }
    default:
        break;
    }
}

SHAD_FORCE_INLINE void Liverpool::SetContextRegHot(const PM4Header* header, u32 count) {
    const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
    const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
    const auto* payload = reinterpret_cast<const u32*>(header + 2);

    gfx_stamp.WriteRegs(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));

    // In the case of HW, render target memory has alignment as color block operates on
    // tiles. There is no information of actual resource extents stored in CB context
    // regs, so any deduction of it from slices/pitch will lead to a larger surface
    // created. The same applies to the depth targets. Fortunately, the guest always
    // sends a trailing NOP packet right after the context regs setup, so we can use the
    // heuristic below and extract the hint to determine actual resource dims. The
    // per-register bodies carry asserts, so they live in the outlined tail; the switch
    // itself stays here and costs nothing on the default path.
    switch (reg_addr) {
    case ContextRegs::CbColor0Base:
    case ContextRegs::CbColor1Base:
    case ContextRegs::CbColor2Base:
    case ContextRegs::CbColor3Base:
    case ContextRegs::CbColor4Base:
    case ContextRegs::CbColor5Base:
    case ContextRegs::CbColor6Base:
    case ContextRegs::CbColor7Base:
    case ContextRegs::CbColor0Cmask:
    case ContextRegs::CbColor1Cmask:
    case ContextRegs::CbColor2Cmask:
    case ContextRegs::CbColor3Cmask:
    case ContextRegs::CbColor4Cmask:
    case ContextRegs::CbColor5Cmask:
    case ContextRegs::CbColor6Cmask:
    case ContextRegs::CbColor7Cmask:
    case ContextRegs::DbZInfo:
        SetContextRegExtentTail(reg_addr, header, payload);
        break;
    default:
        break;
    }
}

SHAD_NO_INLINE void Liverpool::SetShRegCompute(const PM4Header* header, u32 count) {
    const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
    const auto set_size = (count - 1) * sizeof(u32);
    ASSERT(set_size <= sizeof(ComputeProgram));
    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                 (set_data->reg_offset - 0x200);
    std::memcpy(addr, header + 2, set_size);
}

SHAD_FORCE_INLINE void Liverpool::SetShRegHot(const PM4Header* header, u32 count) {
    const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
    const auto set_size = (count - 1) * sizeof(u32);

    if (set_data->reg_offset >= 0x200 &&
        set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
        SetShRegCompute(header, count);
        return;
    }
    const u32 word = Regs::ShRegWordOffset + set_data->reg_offset;
    if (IsGfxUserDataWrite(regs, word, set_size / sizeof(u32))) {
        StoreRegBlock(&regs.reg_array[word], header + 2, set_size);
    } else {
        gfx_stamp.WriteRegs(&regs.reg_array[word], header + 2, set_size);
    }
}

std::span<const u32> Liverpool::RunGraphicsPackets(std::span<const u32> dcb, Task& ce_task,
                                                   uintptr_t base_addr) {
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();
    const bool guest_markers_enabled = rasterizer && EmulatorSettings.IsVkGuestMarkersEnabled();
    // Cached only for the length of one run segment: every dump entry a submit
    // can match is registered before that submit is queued, so a segment that
    // starts with none pending cannot be handed one it could have matched.
    const bool dumping_regs = DebugState.DumpingCurrentReg();

    while (!dcb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        if (header->type != 3) [[unlikely]] {
            // Type-2 packets are used for padding purposes.
            if (header->type != 2) {
                UnhandledPm4Type(header);
            }
            dcb = NextPacket(dcb, 1);
            continue;
        }

        const u32 count = header->type3.NumWords();
        const PM4ItOpcode opcode = header->type3.opcode;
        // Register-write runs dominate the packet mix; one masked compare on
        // type+opcode routes them past the 142-target indirect jump. The mask
        // drops predicate and shader_type, which both handlers ignore, so the
        // routing is bit-identical to the switch arms below.
        const u32 hdr_masked = header->raw & 0xC000FF00u;
        if (hdr_masked == 0xC0007600u) { // SetShReg
            SetShRegHot(header, count);
            dcb = NextPacket(dcb, count + 1);
            continue;
        }
        if (hdr_masked == 0xC0006900u) { // SetContextReg
            SetContextRegHot(header, count);
            dcb = NextPacket(dcb, count + 1);
            continue;
        }
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
                if (guest_markers_enabled) {
                    const auto marker_sz = nop->header.count.Value() * 2;
                    const std::string_view label{reinterpret_cast<const char*>(&nop->data_block[1]),
                                                 marker_sz};
                    rasterizer->ScopeMarkerBegin(label, true);
                }
                break;
            }
            case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                if (guest_markers_enabled) {
                    const auto marker_sz = nop->header.count.Value() * 2;
                    const std::string_view label{reinterpret_cast<const char*>(&nop->data_block[1]),
                                                 marker_sz};
                    const u32 color = *reinterpret_cast<const u32*>(
                        reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                    rasterizer->ScopedMarkerInsertColor(label, color, true);
                }
                break;
            }
            case PM4CmdNop::PayloadType::DebugMarkerPop: {
                if (guest_markers_enabled) {
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
            gfx_stamp.MarkDirty();
            break;
        }
        case PM4ItOpcode::SetConfigReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
            const auto* payload = reinterpret_cast<const u32*>(header + 2);
            gfx_stamp.WriteRegs(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
            break;
        }
        case PM4ItOpcode::SetContextReg: {
            SetContextRegHot(header, count);
            break;
        }
        case PM4ItOpcode::SetShReg: {
            SetShRegHot(header, count);
            break;
        }
        case PM4ItOpcode::SetUconfigReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            gfx_stamp.WriteRegs(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset],
                                header + 2, (count - 1) * sizeof(u32));
            break;
        }
        case PM4ItOpcode::SetPredication: {
            ++packet_stats.set_predication;
            LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
            break;
        }
        case PM4ItOpcode::IndexType: {
            const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
            if (regs.index_buffer_type.raw != index_type->raw) {
                regs.index_buffer_type.raw = index_type->raw;
                gfx_stamp.MarkDirty();
            }
            break;
        }
        case PM4ItOpcode::DrawIndex2: {
            CountDraw(packet_stats, header);
            gfx_stamp.FlushAtDraw();
            const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
            regs.max_index_size = draw_index->max_size;
            regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
            regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
            regs.num_indices = draw_index->index_count;
            regs.draw_initiator = draw_index->draw_initiator;
            if (dumping_regs) {
                DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
            }
            if (rasterizer) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:DrawIndex2", cmd_address));
                    rasterizer->Draw(true);
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->Draw(true);
                }
            }
            break;
        }
        case PM4ItOpcode::DrawIndexOffset2: {
            CountDraw(packet_stats, header);
            gfx_stamp.FlushAtDraw();
            const auto* draw_index_off = reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
            regs.max_index_size = draw_index_off->max_size;
            regs.num_indices = draw_index_off->index_count;
            regs.draw_initiator = draw_index_off->draw_initiator;
            if (dumping_regs) {
                DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
            }
            if (rasterizer) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
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
            CountDraw(packet_stats, header);
            gfx_stamp.FlushAtDraw();
            const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
            regs.num_indices = draw_index->index_count;
            regs.draw_initiator = draw_index->draw_initiator;
            if (dumping_regs) {
                DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
            }
            if (rasterizer) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:DrawIndexAuto", cmd_address));
                    rasterizer->Draw(false);
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->Draw(false);
                }
            }
            break;
        }
        case PM4ItOpcode::DrawIndirect: {
            CountDraw(packet_stats, header);
            gfx_stamp.FlushAtDraw();
            const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
            const auto offset = draw_indirect->data_offset;
            const auto stride = sizeof(DrawIndirectArgs);
            if (dumping_regs) {
                DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
            }
            if (rasterizer) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:DrawIndirect", cmd_address));
                    rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                }
            }
            break;
        }
        case PM4ItOpcode::DrawIndirectMulti: {
            CountDraw(packet_stats, header);
            gfx_stamp.FlushAtDraw();
            const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
            const auto offset = draw_indirect->data_offset;
            if (dumping_regs) {
                DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
            }
            if (rasterizer) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("gfx:{}:DrawIndirectMulti", cmd_address));
                    rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                             draw_indirect->stride, draw_indirect->count, 0);
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                             draw_indirect->stride, draw_indirect->count, 0);
                }
            }
            break;
        }
        case PM4ItOpcode::DrawIndexIndirect: {
            CountDraw(packet_stats, header);
            gfx_stamp.FlushAtDraw();
            const auto* draw_index_indirect =
                reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
            const auto offset = draw_index_indirect->data_offset;
            const auto stride = sizeof(DrawIndexedIndirectArgs);
            if (dumping_regs) {
                DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
            }
            if (rasterizer) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
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
            CountDraw(packet_stats, header);
            gfx_stamp.FlushAtDraw();
            const auto* draw_index_indirect =
                reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
            const auto offset = draw_index_indirect->data_offset;
            if (dumping_regs) {
                DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
            }
            if (rasterizer) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
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
            CountDraw(packet_stats, header);
            gfx_stamp.FlushAtDraw();
            const auto* draw_index_indirect =
                reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
            const auto offset = draw_index_indirect->data_offset;
            if (dumping_regs) {
                DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
            }
            if (rasterizer) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
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
            ++packet_stats.dispatches;
            gfx_stamp.FlushAtDraw();
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (dumping_regs) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:DispatchDirect", cmd_address));
                    rasterizer->DispatchDirect();
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchDirect();
                }
            }
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            gfx_stamp.FlushAtDraw();
            const auto* dispatch_indirect = reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
            auto& cs_program = GetCsRegs();
            const auto offset = dispatch_indirect->data_offset;
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (dumping_regs) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
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
                    ++packet_stats.occlusion_events;
                    static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                    static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                    u64* results = event->Address<u64*>();
                    for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                        *results = pixel_counter | OcclusionCounterValidMask;
                    }
                    if (!occlude_all_) {
                        pixel_counter += OcclusionCounterStep;
                    }
                }
            }
            break;
        }
        case PM4ItOpcode::EventWriteEos: {
            const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
            if (rasterizer) {
                rasterizer->ProcessDownloadImages();
            }
            event_eos->SignalFence([](void* address, u64 data, u32 num_bytes) {
                auto* memory = Core::Memory::Instance();
                if (!memory->TryWriteBacking(address, &data, num_bytes)) {
                    memcpy(address, &data, num_bytes);
                }
            });
            if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                ASSERT(event_eos->size == 1);
                if (rasterizer) {
                    rasterizer->Finish();
                    const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                    *event_eos->Address() = value;
                }
            }
            break;
        }
        case PM4ItOpcode::EventWriteEop: {
            const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
            if (rasterizer) {
                rasterizer->ProcessDownloadImages();
            }
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
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            u64* address = write_data->Address<u64*>();
            if (!write_data->wr_one_addr.Value()) {
                std::memcpy(address, write_data->data, data_size);
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
        // The only graphics opcodes whose handling can suspend the parser. They
        // are returned unconsumed; the coroutine owns them.
        case PM4ItOpcode::MemSemaphore:
        case PM4ItOpcode::Rewind:
        case PM4ItOpcode::WaitRegMem:
        case PM4ItOpcode::IndirectBuffer:
            return dcb;
        case PM4ItOpcode::AcquireMem: {
            // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
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
            if (rasterizer) {
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
                dcb = NextPacket(dcb, count + 1 + cond_exec->exec_count.Value());
                continue;
            }
            break;
        }
        default:
            UnknownPm4Opcode(static_cast<u32>(opcode), count);
        }
        dcb = NextPacket(dcb, count + 1);
    }
    return dcb;
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    while (true) {
        dcb = RunGraphicsPackets(dcb, ce_task, base_addr);
        if (dcb.empty()) {
            break;
        }

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        switch (header->type3.opcode) {
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
        default:
            UNREACHABLE_MSG("Unstoppable PM4 opcode {:#x}", u32(header->type3.opcode.Value()));
        }
        dcb = NextPacket(dcb, header->type3.NumWords() + 1);
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
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();

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
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
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
                std::memcpy(addr, header + 2, set_size);
            } else {
                const u32 word = Regs::ShRegWordOffset + set_data->reg_offset;
                if (IsGfxUserDataWrite(regs, word, set_size / sizeof(u32))) {
                    StoreRegBlock(&regs.reg_array[word], header + 2, set_size);
                } else {
                    gfx_stamp.WriteRegsImmediate(&regs.reg_array[word], header + 2, set_size);
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
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchDirect", vqid, cmd_address));
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
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
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
                std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
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
            if (rasterizer) {
                rasterizer->ProcessDownloadImages();
            }
            release_mem->SignalFence(
                [pipe_id = queue.pipe_id] {
                    Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
                },
                [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                    rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
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

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    auto task = ProcessGraphics(dcb, ccb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    submit_cv.notify_one();
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

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    submit_cv.notify_one();
}

} // namespace AmdGpu
