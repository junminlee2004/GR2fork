// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// LoadingScreenLayer
// =====================================================================
// Renders a "LOADING SHADERS" overlay on top of the swapchain while
// PipelineCache::WarmUp is busy deserializing the on-disk shader cache.
//
// Why this exists:
//
// On startup, gnmdriver::RegisterLib constructs the Presenter, which in
// turn constructs the Rasterizer, which constructs the PipelineCache.
// Historically the PipelineCache constructor itself called WarmUp(),
// blocking the registration thread for as long as it took to recompile
// every cached pipeline (~30s with a 41,896-pipeline cache observed on
// CUSA04943 / Gravity Rush 2). videoout::RegisterLib runs *after*
// gnmdriver, so the PresentThread that drives swapchain frames does not
// yet exist while WarmUp is blocked — the user sees a black window for
// the full duration and reasonably mistakes it for a hang or crash.
//
// This layer is part of the fix. WarmUp is moved out of the
// PipelineCache constructor and is now invoked explicitly from
// Presenter::WarmUpPipelineCache() *after* the Presenter is fully
// constructed (so `swapchain` and the schedulers are valid). While
// WarmUp runs, the Presenter manually pumps frames via
// PrepareBlankFrame() + Present(); during each pump, this layer paints
// over the swapchain via ImGui's foreground draw list with a centered
// "LOADING SHADERS" message rendered in the embedded HelveticaLTStd
// Black Condensed font, plus a determinate progress bar showing the
// fraction of pipelines deserialized so far. The total is precomputed
// via DataBase::CountBlobs (a metadata-only directory walk) right
// before the slow ForEachBlob deserialization pass starts.
//
// The layer uses ImGui::GetForegroundDrawList() rather than its own
// window, because the existing presenter code unconditionally opens the
// "Display##game_display" window inside Present() (drawn *after* layer
// Draw()s during NewFrame); a regular layer window would render behind
// it. The foreground draw list is rendered after every window, so we
// always sit on top.
//
// Centering uses ImGui::GetIO().DisplaySize rather than
// ImGui::GetMainViewport()->Size — the viewport size lives in logical
// (DPI-scaled) pixels while the swapchain renders at physical pixels;
// on any HiDPI display the two diverge and the overlay collapses
// toward the origin. DisplaySize is whatever ImGui itself paints into,
// so painting in that coordinate system aligns 1:1 with the swapchain.
//
// State is held atomically so the warmup thread can update progress
// from inside ForEachBlob without taking any lock.

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

    // Called from the warmup thread once per processed blob. `total` is set
    // exactly once before the first real tick (right after CountBlobs) and
    // remains constant for the duration of the warmup; `loaded` advances
    // every blob.
    void SetProgress(u32 loaded, u32 total) {
        loaded_.store(loaded, std::memory_order_relaxed);
        total_.store(total, std::memory_order_relaxed);
    }

    // Called once when warmup transitions from "loading" to "done" so the
    // overlay can flash a final "Ready" frame before the layer is removed.
    void MarkFinished() {
        finished_.store(true, std::memory_order_release);
    }

    // Sets the swapchain extent the overlay should treat as its rendering
    // surface. ImGui's io.DisplaySize comes from SDL_GetWindowSize, which
    // reports the SDL window's logical size (often 720p) — but during
    // warmup the swapchain is created at the game's output resolution
    // (often 1080p), so without this override ImGui content paints into
    // the top-left rectangle of an oversized swapchain. The Layer's
    // Draw() reads this value, overrides io.DisplaySize at the start of
    // the frame so the Vulkan ImGui renderer sets its viewport to match
    // the swapchain, and uses it for centering math. The override is
    // not restored — Sdl::NewFrame() at the start of the next frame
    // re-syncs DisplaySize back to SDL state, so the override only
    // applies to frames driven through the warmup pump.
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
