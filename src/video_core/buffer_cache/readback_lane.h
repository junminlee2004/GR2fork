// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

namespace VideoCore {

class BufferCache;
struct Buffer;

/// Copy lanes serve the readback windows a direct read cannot: a lane thread copies the retired
/// GPU writes of a window from the arena into its staging buffer on its own queue and writes them
/// back to guest memory, so the faulting thread records no Vulkan commands and the GPU thread
/// never waits for the copy. The copies run on a transfer-only queue family when the device has
/// one, otherwise on the graphics queue under the scheduler's submit lock.
class ReadbackLanes {
public:
    /// One lane per copy queue, or two on the graphics queue.
    explicit ReadbackLanes(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                           BufferCache& buffer_cache, vk::Semaphore bind_semaphore);
    ~ReadbackLanes();

    struct Request {
        const Buffer* arena;
        VAddr window;
        u64 wait_tick; ///< Scheduler tick the GPU has reached for the window's writes.
        u64 copy_tick; ///< Writes stamped up to this tick are copied.
        u64 bind_tick; ///< Bind semaphore value of the last sparse bind submission.
    };

    /// Copies the writes of the request's window into guest memory and returns the byte count,
    /// or nothing while the lanes shut down. The caller owns the window.
    std::optional<u64> Copy(const Request& request);

private:
    struct Job {
        Request request;
        u64 copied;
        bool done;
    };
    struct Lane;

    void Run(Lane& lane, std::stop_token stop);
    void Process(Lane& lane, Job& job);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    BufferCache& buffer_cache;
    vk::Semaphore bind_semaphore;
    std::mutex mutex;
    std::condition_variable_any job_cv;
    std::condition_variable done_cv;
    boost::container::small_vector<Job*, 8> pending;
    bool stopping{};
    std::vector<std::unique_ptr<Lane>> lanes;
};

} // namespace VideoCore
