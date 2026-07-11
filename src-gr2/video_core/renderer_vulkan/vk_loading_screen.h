// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Renders a "LOADING SHADERS" overlay and progress bar while PipelineCache::WarmUp deserializes
// the on-disk cache (~30s at 41,896 pipelines on CUSA04943). The PresentThread does not exist
// yet, so the Presenter pumps frames itself; ImGui's foreground draw list keeps this on top.

#include <atomic>

#include "common/types.h"
#include "imgui/imgui_layer.h"

namespace Vulkan {

class LoadingScreenLayer final : public ImGui::Layer {
public:
    LoadingScreenLayer();
    ~LoadingScreenLayer() override;

    // ImGui::Layer
    void Draw() override;

    // Toggle visibility. Cheap; the layer's Draw() is a no-op when not visible.
    void SetVisible(bool visible) {
        visible_.store(visible, std::memory_order_release);
    }
    bool IsVisible() const {
        return visible_.load(std::memory_order_acquire);
    }

    // Called from the warmup thread once per processed blob. `total` is set once (right after
    // CountBlobs) and stays constant for the warmup; `loaded` advances every blob.
    void SetProgress(u32 loaded, u32 total) {
        loaded_.store(loaded, std::memory_order_relaxed);
        total_.store(total, std::memory_order_relaxed);
    }

    // Called once when warmup transitions from "loading" to "done" so the
    // overlay can flash a final "Ready" frame before the layer is removed.
    void MarkFinished() {
        finished_.store(true, std::memory_order_release);
    }

    // Sets the swapchain extent Draw() overrides io.DisplaySize with. SDL reports the window's
    // logical size (often 720p) while the warmup swapchain uses the game's output resolution, so
    // ImGui would paint into the top-left corner; Sdl::NewFrame() re-syncs DisplaySize afterward.
    void SetSwapchainExtent(u32 width, u32 height) {
        sw_width_.store(width, std::memory_order_relaxed);
        sw_height_.store(height, std::memory_order_relaxed);
    }

private:
    std::atomic<bool> visible_{false};
    std::atomic<bool> finished_{false};
    std::atomic<u32> loaded_{0};
    std::atomic<u32> total_{0};
    std::atomic<u32> sw_width_{0};
    std::atomic<u32> sw_height_{0};
};

// Installed as a ::Vulkan-namespace singleton so the gnmdriver thread (which
// drives WarmUp) and the Presenter (which owns the layer) can both reach the
// same instance without plumbing the pointer through PipelineCache.
LoadingScreenLayer& GetLoadingScreenLayer();

} // namespace Vulkan
