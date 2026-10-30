// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <mutex>
#include <fmt/format.h>

#include "common/thread.h"
#include "core/gr2_audio_catchup/gr2_audio_catchup.h"
#include "core/gr2_av_trace/gr2_av_trace.h"
#include "core/libraries/audio/audioout.h"
#include "core/libraries/audio/audioout_backend.h"
#include "core/libraries/kernel/time.h"

namespace Gr2AudioCatchup {

namespace {

using Clock = std::chrono::steady_clock;

// Missed periods beyond this are dropped: after a longer stall (loading, a debugger break) the
// guest is not expected to make up the time.
constexpr auto MaxCatchUp = std::chrono::milliseconds{500};

} // namespace

void RunOutputThread(const std::shared_ptr<Libraries::AudioOut::PortOut>& port,
                     const std::stop_token& stop) {
    using namespace Libraries;
    {
        const auto thread_name = fmt::format("shadPS4:AudioOutputThread:{}", fmt::ptr(port.get()));
        Common::SetCurrentThreadName(thread_name.c_str());
    }

    const std::chrono::nanoseconds period{1000000000ULL * port->buffer_frames / port->sample_rate};
    auto next_tick = Clock::now();
    // An idle port, or one whose guest missed a block, has no missed periods to play.
    bool took_block = false;

    while (true) {
        const auto now = Clock::now();
        if (next_tick > now) {
            Common::AccurateSleep(next_tick - now, nullptr, false);
        } else if (!took_block || now - next_tick > MaxCatchUp) {
            next_tick = now;
        }

        Gr2AvTrace::OnTickBegin(port->impl.get());
        {
            std::unique_lock lock{port->mutex};
            if (!port->impl || stop.stop_requested()) {
                break;
            }
            // While behind schedule the guest is rendering its next block right after the last
            // one was taken; waiting up to one period for it plays the missed period.
            if (!port->output_ready && Clock::now() - next_tick >= period) {
                port->output_cv.wait_for(lock, period, [&] {
                    return port->output_ready || port->closing || stop.stop_requested();
                });
            }
            Gr2AvTrace::OnAudioTick(port->impl.get(), static_cast<s32>(port->type),
                                    port->sample_rate, port->buffer_frames, port->output_ready);
            took_block = port->output_ready;
            if (port->output_ready) {
                port->impl->Output(port->output_buffer);
                port->output_ready = false;
                port->last_output_time = Kernel::sceKernelGetProcessTime();
            }
            Gr2AvTrace::OnTickEnd(port->impl.get());
        }

        port->output_cv.notify_all();

        if (stop.stop_requested()) {
            break;
        }
        next_tick += period;
    }

    {
        std::unique_lock lock{port->mutex};
        port->closing = true;
        port->output_ready = false;
    }
    port->output_cv.notify_all();
}

} // namespace Gr2AudioCatchup
