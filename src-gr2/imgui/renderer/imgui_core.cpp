// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <SDL3/SDL_events.h>
#include <imgui.h>

#include "common/config.h"
#include "common/path_util.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include <vulkan/vulkan.hpp>
#include "video_core/renderer_vulkan/vk_instance.h"
#include "core/debug_state.h"
#include "core/devtools/layer.h"
#include "imgui/imgui_layer.h"
#include "imgui_core.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "sdl_window.h"
#include "texture_manager.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

#include "imgui_fonts/notosansjp_regular.ttf.g.cpp"
#include "imgui_fonts/proggyvector_regular.ttf.g.cpp"

// Embedded HelveticaLTStd-BlkCond.otf used by Vulkan::LoadingScreenLayer for
// the "LOADING SHADERS" overlay. Lives in video_core/ so the binary blob is
// adjacent to the only consumer (LoadingScreenLayer); imgui_core just owns
// the atlas registration and the accessor below.
#include "video_core/renderer_vulkan/helvetica_blkcond_font.h"

static void CheckVkResult(const vk::Result err) {
    if (err == vk::Result::eSuccess) {
        return;
    }
    LOG_ERROR(ImGui, "Vulkan error {}", vk::to_string(err));
}

static std::vector<ImGui::Layer*> layers;

// Update layers before rendering to allow layer changes to be applied during rendering.
// Using deque to keep the order of changes in case a Layer is removed then added again between
// frames.
static std::deque<std::pair<bool, ImGui::Layer*>> change_layers{};
static std::mutex change_layers_mutex{};

static ImGuiID dock_id;

// Cached pointer to the embedded HelveticaLTStd Black Condensed font. Set
// once during Initialize() (after AddFontFromMemoryTTF, before Vulkan::Init
// builds the atlas texture). nullptr if registration fails — accessor
// surfaces that to consumers so they can fall back gracefully.
static ImFont* g_helvetica_blkcond_font = nullptr;

namespace ImGui {

namespace Core {

void Initialize(const ::Vulkan::Instance& instance, const Frontend::WindowSDL& window,
                const u32 image_count, vk::Format surface_format,
                const vk::AllocationCallbacks* allocator) {

    const auto config_path = GetUserPath(Common::FS::PathType::UserDir) / "imgui.ini";
    const auto log_path = GetUserPath(Common::FS::PathType::LogDir) / "imgui_log.txt";

    CreateContext();
    ImGuiIO& io = GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.DisplaySize = ImVec2((float)window.GetWidth(), (float)window.GetHeight());
    PushStyleVar(ImGuiStyleVar_WindowRounding, 6.0f); // Makes the window edges rounded

    auto path = config_path.u8string();
    char* config_file_buf = new char[path.size() + 1]();
    std::memcpy(config_file_buf, path.c_str(), path.size());
    io.IniFilename = config_file_buf;

    path = log_path.u8string();
    char* log_file_buf = new char[path.size() + 1]();
    std::memcpy(log_file_buf, path.c_str(), path.size());
    io.LogFilename = log_file_buf;

    ImFontGlyphRangesBuilder rb{};
    rb.AddRanges(io.Fonts->GetGlyphRangesDefault());
    rb.AddRanges(io.Fonts->GetGlyphRangesGreek());
    rb.AddRanges(io.Fonts->GetGlyphRangesKorean());
    rb.AddRanges(io.Fonts->GetGlyphRangesJapanese());
    rb.AddRanges(io.Fonts->GetGlyphRangesCyrillic());
    ImVector<ImWchar> ranges{};
    rb.BuildRanges(&ranges);
    ImFontConfig font_cfg{};
    font_cfg.OversampleH = 2;
    font_cfg.OversampleV = 1;
    io.FontDefault = io.Fonts->AddFontFromMemoryCompressedTTF(
        imgui_font_notosansjp_regular_compressed_data,
        imgui_font_notosansjp_regular_compressed_size, 32.0f, &font_cfg, ranges.Data);
    io.Fonts->AddFontFromMemoryCompressedTTF(imgui_font_proggyvector_regular_compressed_data,
                                             imgui_font_proggyvector_regular_compressed_size,
                                             32.0f);
    io.Fonts->AddFontFromMemoryCompressedTTF(imgui_font_notosansjp_regular_compressed_data,
                                             imgui_font_notosansjp_regular_compressed_size, 128.0f,
                                             &font_cfg, ranges.Data);

    // HelveticaLTStd-BlkCond.otf — used for the "LOADING SHADERS" overlay
    // painted by Vulkan::LoadingScreenLayer during the PipelineCache::WarmUp
    // pause at gnmdriver registration. Registered here, before Vulkan::Init,
    // so it lands in the same atlas texture upload as the other fonts and
    // we don't need to trigger a CreateFontsTexture rebuild after init.
    //
    // Important: FontDataOwnedByAtlas MUST be false. The font bytes live in
    // a constexpr `unsigned char[]` blob in helvetica_blkcond_font.h and
    // would crash IM_FREE on shutdown if the atlas tried to own them.
    // We rasterize at 128px for good headroom; LoadingScreenLayer renders at
    // ~48-156px depending on viewport size, so we're never aggressively
    // upscaling.
    {
        ImFontConfig helv_cfg{};
        helv_cfg.OversampleH = 2;
        helv_cfg.OversampleV = 1;
        helv_cfg.FontDataOwnedByAtlas = false;
        // AddFontFromMemoryTTF takes a void*, but the const_cast is safe
        // because FontDataOwnedByAtlas=false guarantees ImGui never
        // mutates or frees the buffer.
        g_helvetica_blkcond_font = io.Fonts->AddFontFromMemoryTTF(
            const_cast<unsigned char*>(Fonts::kHelveticaBlkCondFontData),
            static_cast<int>(Fonts::kHelveticaBlkCondFontSize), 128.0f,
            &helv_cfg);
    }

    io.FontGlobalScale = 0.5f;

    StyleColorsDark();

    ::Core::Devtools::Layer::SetupSettings();
    Sdl::Init(window.GetSDLWindow());

    vk::PipelineRenderingCreateInfo _prci{};
    _prci.colorAttachmentCount = 1;
    _prci.pColorAttachmentFormats = &surface_format;
    const Vulkan::InitInfo vk_info{
        .instance = instance.GetInstance(),
        .physical_device = instance.GetPhysicalDevice(),
        .device = instance.GetDevice(),
        .queue_family = instance.GetPresentQueueFamilyIndex(),
        .queue = instance.GetPresentQueue(),
        .image_count = image_count,
        .min_allocation_size = 1024 * 1024,
        .pipeline_rendering_create_info = _prci,
        .allocator = allocator,
        .check_vk_result_fn = &CheckVkResult,
    };
    Vulkan::Init(vk_info);

    TextureManager::StartWorker();

    char label[32];
    ImFormatString(label, IM_ARRAYSIZE(label), "WindowOverViewport_%08X", GetMainViewport()->ID);
    dock_id = ImHashStr(label);

    if (const auto dpi = SDL_GetWindowDisplayScale(window.GetSDLWindow()); dpi > 0.0f) {
        GetIO().FontGlobalScale *= dpi;
    }

    // TEMP PROBE (GR2FORK trophy-font DPI investigation): always-on, fires once
    // at init. Captures every quantity that feeds trophy/menu font sizing so we
    // can compare Windows vs Linux in a single run. Remove once resolved.
    {
        SDL_Window* probe_w = window.GetSDLWindow();
        s32 log_w = 0, log_h = 0, px_w = 0, px_h = 0;
        SDL_GetWindowSize(probe_w, &log_w, &log_h);
        SDL_GetWindowSizeInPixels(probe_w, &px_w, &px_h);
        const float probe_disp_scale = SDL_GetWindowDisplayScale(probe_w);
        const ImGuiIO& probe_io = GetIO();
        LOG_INFO(ImGui,
                 "[GR2FORK][DPIPROBE] init: WindowGet={}x{} SDL_logical={}x{} "
                 "SDL_pixels={}x{} DisplayScale={} -> io.DisplaySize={}x{} "
                 "io.FontGlobalScale={} io.FBScale={}x{}",
                 window.GetWidth(), window.GetHeight(), log_w, log_h, px_w, px_h,
                 probe_disp_scale, probe_io.DisplaySize.x, probe_io.DisplaySize.y,
                 probe_io.FontGlobalScale, probe_io.DisplayFramebufferScale.x,
                 probe_io.DisplayFramebufferScale.y);
    }

    std::at_quick_exit([] { SaveIniSettingsToDisk(GetIO().IniFilename); });
}

void OnResize() {
    Sdl::OnResize();
}

void OnSurfaceFormatChange(vk::Format surface_format) {
    Vulkan::OnSurfaceFormatChange(surface_format);
}

void Shutdown(const vk::Device& device) {
    CheckVkResult(device.waitIdle());

    TextureManager::StopWorker();

    const ImGuiIO& io = GetIO();
    const auto ini_filename = (void*)io.IniFilename;
    const auto log_filename = (void*)io.LogFilename;

    Vulkan::Shutdown();
    Sdl::Shutdown();
    DestroyContext();

    delete[] (char*)ini_filename;
    delete[] (char*)log_filename;
}

bool ProcessEvent(SDL_Event* event) {
    Sdl::ProcessEvent(event);
    switch (event->type) {
    // Don't block release/up events
    case SDL_EVENT_MOUSE_MOTION:
    case SDL_EVENT_MOUSE_WHEEL:
    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
        const auto& io = GetIO();
        return io.WantCaptureMouse && io.Ctx->NavWindow != nullptr &&
               (io.Ctx->NavWindow->Flags & ImGuiWindowFlags_NoNav) == 0;
    }
    case SDL_EVENT_TEXT_INPUT:
    case SDL_EVENT_KEY_DOWN: {
        const auto& io = GetIO();
        return io.WantCaptureKeyboard && io.Ctx->NavWindow != nullptr &&
               io.Ctx->NavWindow->ID != dock_id;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION: {
        const auto& io = GetIO();
        return io.NavActive && io.Ctx->NavWindow != nullptr && io.Ctx->NavWindow->ID != dock_id;
    }
    default:
        return false;
    }
}

ImGuiID NewFrame(bool is_reusing_frame) {
    {
        std::scoped_lock lock{change_layers_mutex};
        while (!change_layers.empty()) {
            const auto [to_be_added, layer] = change_layers.front();
            if (to_be_added) {
                layers.push_back(layer);
            } else {
                const auto [begin, end] = std::ranges::remove(layers, layer);
                layers.erase(begin, end);
            }
            change_layers.pop_front();
        }
    }

    Sdl::NewFrame(is_reusing_frame);
    ImGui::NewFrame();

    ImGuiWindowFlags flags =
        ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar;
    if (!DebugState.IsShowingDebugMenuBar()) {
        flags |= ImGuiDockNodeFlags_NoTabBar;
    }
    ImGuiID dockId = DockSpaceOverViewport(0, GetMainViewport(), flags);

    for (auto* layer : layers) {
        layer->Draw();
    }

    return dockId;
}

void Render(const vk::CommandBuffer& cmdbuf, const vk::ImageView& image_view,
            const vk::Extent2D& extent) {
    ImGui::Render();
    ImDrawData* draw_data = GetDrawData();
    if (draw_data->CmdListsCount == 0) {
        return;
    }

    if (Config::getVkHostMarkersEnabled()) {
        vk::DebugUtilsLabelEXT _di_tmp1{};
        _di_tmp1.pLabelName = "ImGui Render";
        cmdbuf.beginDebugUtilsLabelEXT(_di_tmp1);
    }

    vk::RenderingAttachmentInfo color_attachments[1]{};
    color_attachments[0].imageView = image_view;
    color_attachments[0].imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    color_attachments[0].loadOp = vk::AttachmentLoadOp::eClear;
    color_attachments[0].storeOp = vk::AttachmentStoreOp::eStore;
    vk::RenderingInfo render_info{};
    vk::Rect2D _di_tmp2{};
    _di_tmp2.offset = vk::Offset2D{0, 0};
    _di_tmp2.extent = extent;
    render_info.renderArea = _di_tmp2;
    render_info.layerCount = 1;
    render_info.colorAttachmentCount = 1;
    render_info.pColorAttachments = color_attachments;
    cmdbuf.beginRendering(render_info);
    Vulkan::RenderDrawData(*draw_data, cmdbuf);
    cmdbuf.endRendering();
    if (Config::getVkHostMarkersEnabled()) {
        cmdbuf.endDebugUtilsLabelEXT();
    }
}

bool MustKeepDrawing() {
    return layers.size() > 1 || change_layers.size() > 1 || DebugState.IsShowingDebugMenuBar();
}

ImFont* GetHelveticaFont() {
    return g_helvetica_blkcond_font;
}

} // namespace Core

void Layer::AddLayer(Layer* layer) {
    std::scoped_lock lock{change_layers_mutex};
    change_layers.emplace_back(true, layer);
}

void Layer::RemoveLayer(Layer* layer) {
    std::scoped_lock lock{change_layers_mutex};
    change_layers.emplace_back(false, layer);
}

} // namespace ImGui
