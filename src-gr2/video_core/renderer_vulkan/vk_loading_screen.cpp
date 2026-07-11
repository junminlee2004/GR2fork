// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_loading_screen.h"

#include <algorithm>
#include <cstdio>

#include <imgui.h>
#include <imgui_internal.h>

#include "imgui/renderer/imgui_core.h"

namespace Vulkan {

namespace {

// Title pixel height as a fraction of viewport height, clamped so it stays readable on both
// 720p and 4K without dominating the counter/bar/note layout.
float ComputeTitleSize(float viewport_h) {
    const float candidate = viewport_h * 0.0425f;
    return std::clamp(candidate, 24.0f, 78.0f);
}

void FormatGrouped(char* dst, size_t dst_size, u32 value) {
    char raw[16];
    const int n = std::snprintf(raw, sizeof(raw), "%u", value);
    if (n <= 0) {
        if (dst_size > 0) dst[0] = '\0';
        return;
    }
    int out = 0;
    for (int i = 0; i < n; ++i) {
        if (i > 0 && (n - i) % 3 == 0) {
            if (out + 1 < static_cast<int>(dst_size)) dst[out++] = ',';
        }
        if (out + 1 < static_cast<int>(dst_size)) dst[out++] = raw[i];
    }
    if (out < static_cast<int>(dst_size)) dst[out] = '\0';
    else if (dst_size > 0) dst[dst_size - 1] = '\0';
}

} // namespace

LoadingScreenLayer& GetLoadingScreenLayer() {
    static LoadingScreenLayer instance;
    return instance;
}

LoadingScreenLayer::LoadingScreenLayer() = default;
LoadingScreenLayer::~LoadingScreenLayer() = default;

void LoadingScreenLayer::Draw() {
    if (!visible_.load(std::memory_order_acquire)) {
        return;
    }

    // Override io.DisplaySize (Sdl::NewFrame() sets it from the SDL window size) to the
    // swapchain extent, which tracks the game's output resolution - otherwise ImGui paints a
    // window-sized top-left rectangle with black bars. Sdl::NewFrame() re-syncs it next frame.
    ImGuiIO& io = ImGui::GetIO();
    const u32 sw_w = sw_width_.load(std::memory_order_relaxed);
    const u32 sw_h = sw_height_.load(std::memory_order_relaxed);
    if (sw_w > 0 && sw_h > 0) {
        io.DisplaySize = ImVec2(static_cast<float>(sw_w),
                                static_cast<float>(sw_h));
        io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);
    }

    // Layout against the (possibly overridden) DisplaySize; without the override this is the
    // SDL-window value, which is still the right coordinate space for ImGui's renderer.
    const ImVec2 disp = io.DisplaySize;
    if (disp.x <= 0.0f || disp.y <= 0.0f) {
        return;
    }
    const ImVec2 origin{0.0f, 0.0f};
    const ImVec2 disp_max{disp.x, disp.y};
    const ImVec2 center{disp.x * 0.5f, disp.y * 0.5f};

    // Draw through a real ImGui window pinned to the override rect: foreground draw lists clip
    // against the MainViewport bounds (SDL window size, not swapchain extent) and would clip
    // anything outside the SDL window region; window-local lists clip against their own rect.
    ImGui::SetNextWindowPos(origin);
    ImGui::SetNextWindowSize(disp_max);
    ImGui::SetNextWindowBgAlpha(0.0f); // we paint our own opaque black

    constexpr ImGuiWindowFlags kFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs |
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBringToFrontOnFocus;

    if (!ImGui::Begin("##LoadingShadersOverlay", nullptr, kFlags)) {
        ImGui::End();
        return;
    }

    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Opaque black backdrop covering the full swapchain.
    dl->AddRectFilled(origin, disp_max, IM_COL32(0, 0, 0, 255));

    ImFont* helvetica = ImGui::Core::GetHelveticaFont();
    ImFont* default_font = io.FontDefault;

    // ---- Title -----------------------------------------------------------
    const bool is_finished = finished_.load(std::memory_order_acquire);
    const char* title = is_finished ? "READY" : "LOADING SHADERS";
    const float title_size = ComputeTitleSize(disp.y);
    if (helvetica != nullptr) {
        const ImVec2 sz = helvetica->CalcTextSizeA(title_size, FLT_MAX, 0.0f, title);
        const ImVec2 pos{center.x - sz.x * 0.5f, center.y - sz.y - 8.0f};
        dl->AddText(helvetica, title_size,
                    ImVec2{pos.x + 1.0f, pos.y + 1.0f},
                    IM_COL32(0, 0, 0, 200), title);
        dl->AddText(helvetica, title_size, pos,
                    IM_COL32(235, 235, 235, 255), title);
    } else if (default_font != nullptr) {
        const float fallback_size = title_size * 0.6f;
        const ImVec2 sz =
            default_font->CalcTextSizeA(fallback_size, FLT_MAX, 0.0f, title);
        const ImVec2 pos{center.x - sz.x * 0.5f, center.y - sz.y - 8.0f};
        dl->AddText(default_font, fallback_size, pos,
                    IM_COL32(235, 235, 235, 255), title);
    }

    // ---- Counter ---------------------------------------------------------
    const u32 loaded = loaded_.load(std::memory_order_relaxed);
    const u32 total = total_.load(std::memory_order_relaxed);
    if (default_font != nullptr) {
        char loaded_buf[24];
        char total_buf[24];
        char counter_full[64];
        FormatGrouped(loaded_buf, sizeof(loaded_buf), loaded);
        FormatGrouped(total_buf, sizeof(total_buf), total);
        if (total > 0) {
            std::snprintf(counter_full, sizeof(counter_full),
                          "%s / %s pipelines", loaded_buf, total_buf);
        } else {
            std::snprintf(counter_full, sizeof(counter_full), "%s pipelines",
                          loaded_buf);
        }

        const float counter_size = std::clamp(disp.y * 0.028f, 16.0f, 44.0f);
        const ImVec2 csz = default_font->CalcTextSizeA(counter_size, FLT_MAX,
                                                       0.0f, counter_full);
        const ImVec2 cpos{center.x - csz.x * 0.5f, center.y + 12.0f};
        dl->AddText(default_font, counter_size, cpos,
                    IM_COL32(180, 180, 180, 255), counter_full);

        // ---- Progress bar ------------------------------------------------
        const float bar_w = std::clamp(disp.x * 0.30f, 240.0f, 640.0f);
        const float bar_h = std::clamp(disp.y * 0.010f, 6.0f, 14.0f);
        const float bar_x0 = center.x - bar_w * 0.5f;
        const float bar_y0 = cpos.y + counter_size + 18.0f;

        dl->AddRectFilled(ImVec2{bar_x0, bar_y0},
                          ImVec2{bar_x0 + bar_w, bar_y0 + bar_h},
                          IM_COL32(40, 40, 40, 255));

        float fraction = 0.0f;
        if (is_finished) {
            fraction = 1.0f;
        } else if (total > 0) {
            fraction = static_cast<float>(loaded) / static_cast<float>(total);
            fraction = std::clamp(fraction, 0.0f, 1.0f);
        }
        if (fraction > 0.0f) {
            const float fill_w = bar_w * fraction;
            dl->AddRectFilled(ImVec2{bar_x0, bar_y0},
                              ImVec2{bar_x0 + fill_w, bar_y0 + bar_h},
                              IM_COL32(220, 220, 220, 255));
        }

        if (total > 0) {
            char pct_text[16];
            std::snprintf(pct_text, sizeof(pct_text), "%d%%",
                          static_cast<int>(fraction * 100.0f + 0.5f));
            const float pct_size =
                std::clamp(disp.y * 0.020f, 12.0f, 28.0f);
            const ImVec2 psz =
                default_font->CalcTextSizeA(pct_size, FLT_MAX, 0.0f, pct_text);
            const ImVec2 ppos{center.x - psz.x * 0.5f,
                              bar_y0 + bar_h + 6.0f};
            dl->AddText(default_font, pct_size, ppos,
                        IM_COL32(180, 180, 180, 255), pct_text);
        }

        // ---- Note --------------------------------------------------------
        const char* note = is_finished
                               ? "Starting game..."
                               : "First-time shader cache load. This may "
                                 "take up to a minute.";
        const float note_size = std::clamp(disp.y * 0.020f, 12.0f, 28.0f);
        const ImVec2 nsz =
            default_font->CalcTextSizeA(note_size, FLT_MAX, 0.0f, note);
        const ImVec2 npos{center.x - nsz.x * 0.5f,
                          bar_y0 + bar_h + note_size * 2.0f + 12.0f};
        dl->AddText(default_font, note_size, npos,
                    IM_COL32(140, 140, 140, 255), note);
    }

    ImGui::End();

    // Lift our overlay above any windows opened later in the same frame
    // (notably the Presenter's "Display##game_display" docked window).
    if (ImGuiWindow* w = ImGui::FindWindowByName("##LoadingShadersOverlay")) {
        ImGui::BringWindowToDisplayFront(w);
    }
}

} // namespace Vulkan
