// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "SDL3/SDL_events.h"
#include "SDL3/SDL_hints.h"
#include "SDL3/SDL_init.h"
#include "SDL3/SDL_properties.h"
#include "SDL3/SDL_timer.h"
#include "SDL3/SDL_video.h"
#include <mutex>
#include "common/assert.h"
#include "common/config.h"
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/logging/formatter.h"
#include "common/scope_exit.h"
#include "common/stb.h"
#include "core/debug_state.h"
#include "core/devtools/layer.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/pad/pad.h"
#include "imgui/renderer/imgui_core.h"
#include "input/controller.h"
#include "input/input_handler.h"
#include "input/input_mouse.h"
#include "sdl_window.h"
#include "video_core/renderdoc.h"

#ifdef __APPLE__
#include "SDL3/SDL_metal.h"
#endif

namespace Input {

using Libraries::Pad::OrbisPadButtonDataOffset;

static OrbisPadButtonDataOffset SDLGamepadToOrbisButton(u8 button) {
    using OPBDO = OrbisPadButtonDataOffset;

    switch (button) {
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN:
        return OPBDO::Down;
    case SDL_GAMEPAD_BUTTON_DPAD_UP:
        return OPBDO::Up;
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT:
        return OPBDO::Left;
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT:
        return OPBDO::Right;
    case SDL_GAMEPAD_BUTTON_SOUTH:
        return OPBDO::Cross;
    case SDL_GAMEPAD_BUTTON_NORTH:
        return OPBDO::Triangle;
    case SDL_GAMEPAD_BUTTON_WEST:
        return OPBDO::Square;
    case SDL_GAMEPAD_BUTTON_EAST:
        return OPBDO::Circle;
    case SDL_GAMEPAD_BUTTON_START:
        return OPBDO::Options;
    case SDL_GAMEPAD_BUTTON_TOUCHPAD:
        return OPBDO::TouchPad;
    case SDL_GAMEPAD_BUTTON_BACK:
        return OPBDO::TouchPad;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER:
        return OPBDO::L1;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER:
        return OPBDO::R1;
    case SDL_GAMEPAD_BUTTON_LEFT_STICK:
        return OPBDO::L3;
    case SDL_GAMEPAD_BUTTON_RIGHT_STICK:
        return OPBDO::R3;
    default:
        return OPBDO::None;
    }
}

static SDL_GamepadAxis InputAxisToSDL(Axis axis) {
    switch (axis) {
    case Axis::LeftX:
        return SDL_GAMEPAD_AXIS_LEFTX;
    case Axis::LeftY:
        return SDL_GAMEPAD_AXIS_LEFTY;
    case Axis::RightX:
        return SDL_GAMEPAD_AXIS_RIGHTX;
    case Axis::RightY:
        return SDL_GAMEPAD_AXIS_RIGHTY;
    case Axis::TriggerLeft:
        return SDL_GAMEPAD_AXIS_LEFT_TRIGGER;
    case Axis::TriggerRight:
        return SDL_GAMEPAD_AXIS_RIGHT_TRIGGER;
    default:
        UNREACHABLE();
    }
}

SDLInputEngine::~SDLInputEngine() {
    for (SDL_Gamepad* pad : m_gamepads) {
        SDL_CloseGamepad(pad);
    }
    m_gamepads.clear();
    m_gamepad = nullptr;
}

void SDLInputEngine::Init() {
    // Init is re-run on hotplug (add/remove). Close anything previously opened.
    for (SDL_Gamepad* pad : m_gamepads) {
        SDL_CloseGamepad(pad);
    }
    m_gamepads.clear();
    m_gamepad = nullptr;
    m_gyro_poll_rate = 0.0f;
    m_accel_poll_rate = 0.0f;

    int gamepad_count;
    SDL_JoystickID* gamepads = SDL_GetGamepads(&gamepad_count);
    if (!gamepads) {
        LOG_ERROR(Input, "Cannot get gamepad list: {}", SDL_GetError());
        return;
    }
    if (gamepad_count == 0) {
        LOG_INFO(Input, "No gamepad found!");
        SDL_free(gamepads);
        return;
    }

    // Primary pad (drives imgui nav + LED/rumble focus): GUI-selected, else the
    // configured default, else the first enumerated pad. ALL pads are opened
    // regardless so any connected controller can drive the game (multi-controller
    // input is enabled by default).
    int selectedIndex = GamepadSelect::GetIndexfromGUID(gamepads, gamepad_count,
                                                        GamepadSelect::GetSelectedGamepad());
    int defaultIndex =
        GamepadSelect::GetIndexfromGUID(gamepads, gamepad_count, Config::getDefaultControllerID());
    int primaryIndex = (selectedIndex != -1) ? selectedIndex
                       : (defaultIndex != -1) ? defaultIndex
                                              : 0;

    const bool motion = Config::getIsMotionControlsEnabled();
    int* rgb = Config::GetControllerCustomColor();

    LOG_INFO(Input, "Got {} gamepad(s). Opening all for multi-controller input.", gamepad_count);

    for (int i = 0; i < gamepad_count; ++i) {
        SDL_Gamepad* pad = SDL_OpenGamepad(gamepads[i]);
        if (!pad) {
            LOG_ERROR(Input, "Failed to open gamepad index {}: {}", i, SDL_GetError());
            continue;
        }
        m_gamepads.push_back(pad);
        if (i == primaryIndex) {
            m_gamepad = pad;
        }

        SDL_Joystick* joystick = SDL_GetGamepadJoystick(pad);
        Uint16 vendor = SDL_GetJoystickVendor(joystick);
        Uint16 product = SDL_GetJoystickProduct(joystick);
        bool isDualSense = (vendor == 0x054C && product == 0x0CE6);

        LOG_INFO(Input, "Gamepad {} Vendor: {:04X}, Product: {:04X}", i, vendor, product);
        if (isDualSense) {
            LOG_INFO(Input, "Detected DualSense Controller (gamepad {})", i);
        }

        if (motion) {
            if (SDL_SetGamepadSensorEnabled(pad, SDL_SENSOR_GYRO, true)) {
                float rate = SDL_GetGamepadSensorDataRate(pad, SDL_SENSOR_GYRO);
                if (m_gyro_poll_rate == 0.0f) {
                    m_gyro_poll_rate = rate;
                }
                LOG_INFO(Input, "Gyro initialized on gamepad {}, poll rate: {}", i, rate);
            } else {
                LOG_ERROR(Input, "Failed to initialize gyro for gamepad {}, error: {}", i,
                          SDL_GetError());
                SDL_SetGamepadSensorEnabled(pad, SDL_SENSOR_GYRO, false);
            }
            if (SDL_SetGamepadSensorEnabled(pad, SDL_SENSOR_ACCEL, true)) {
                float rate = SDL_GetGamepadSensorDataRate(pad, SDL_SENSOR_ACCEL);
                if (m_accel_poll_rate == 0.0f) {
                    m_accel_poll_rate = rate;
                }
                LOG_INFO(Input, "Accel initialized on gamepad {}, poll rate: {}", i, rate);
            } else {
                LOG_ERROR(Input, "Failed to initialize accel for gamepad {}, error: {}", i,
                          SDL_GetError());
                SDL_SetGamepadSensorEnabled(pad, SDL_SENSOR_ACCEL, false);
            }
        }

        if (isDualSense) {
            if (SDL_SetJoystickLED(joystick, rgb[0], rgb[1], rgb[2]) == 0) {
                LOG_INFO(Input, "Set DualSense LED to R:{} G:{} B:{} (gamepad {})", rgb[0], rgb[1],
                         rgb[2], i);
            } else {
                LOG_ERROR(Input, "Failed to set DualSense LED (gamepad {}): {}", i, SDL_GetError());
            }
        } else {
            SDL_SetGamepadLED(pad, rgb[0], rgb[1], rgb[2]);
        }
    }

    SDL_free(gamepads);

    if (m_gamepads.empty()) {
        LOG_ERROR(Input, "Failed to open any gamepad: {}", SDL_GetError());
        return;
    }
    // If the chosen primary failed to open, fall back to the first pad that did.
    if (!m_gamepad) {
        m_gamepad = m_gamepads.front();
    }
}

void SDLInputEngine::SetLightBarRGB(u8 r, u8 g, u8 b) {
    for (SDL_Gamepad* pad : m_gamepads) {
        SDL_SetGamepadLED(pad, r, g, b);
    }
}

void SDLInputEngine::SetVibration(u8 smallMotor, u8 largeMotor) {
    const auto low_freq = (smallMotor / 255.0f) * 0xFFFF;
    const auto high_freq = (largeMotor / 255.0f) * 0xFFFF;
    for (SDL_Gamepad* pad : m_gamepads) {
        SDL_RumbleGamepad(pad, low_freq, high_freq, -1);
    }
}

State SDLInputEngine::ReadState() {
    State state{};
    state.time = Libraries::Kernel::sceKernelGetProcessTime();

    if (m_gamepads.empty()) {
        return state;
    }

    // Deflection of an axis value from its rest position (sticks rest at 128,
    // triggers at 0). Used to pick the most-active pad per axis. Local helper so
    // no extra <cstdlib> dependency for std::abs.
    const auto deflection = [](int v, int rest) { return v > rest ? v - rest : rest - v; };

    // Buttons: pressed if ANY active pad has it pressed. OnButton sets-or-clears
    // per call, so the OR must be computed before the single OnButton call --
    // otherwise a released pad would wipe another pad's press.
    for (u8 i = 0; i < SDL_GAMEPAD_BUTTON_COUNT; ++i) {
        auto orbisButton = SDLGamepadToOrbisButton(i);
        if (orbisButton == OrbisPadButtonDataOffset::None) {
            continue;
        }
        bool pressed = false;
        for (SDL_Gamepad* pad : m_gamepads) {
            pressed |= SDL_GetGamepadButton(pad, (SDL_GamepadButton)i);
        }
        state.OnButton(orbisButton, pressed);
    }

    // Axes: take the value with the largest deflection from rest across all pads.
    for (int i = 0; i < static_cast<int>(Axis::AxisMax); ++i) {
        const auto axis = static_cast<Axis>(i);
        const bool is_trigger = (axis == Axis::TriggerLeft || axis == Axis::TriggerRight);
        const int rest = is_trigger ? 0 : 128;
        int best = rest;
        for (SDL_Gamepad* pad : m_gamepads) {
            const auto raw = SDL_GetGamepadAxis(pad, InputAxisToSDL(axis));
            const int value =
                is_trigger ? GetAxis(0, 0x8000, raw) : GetAxis(-0x8000, 0x8000, raw);
            if (deflection(value, rest) > deflection(best, rest)) {
                best = value;
            }
        }
        state.OnAxis(axis, best);
    }

    // Touchpad / Gyro / Accel: from the first active pad that provides each.
    for (SDL_Gamepad* pad : m_gamepads) {
        if (SDL_GetNumGamepadTouchpads(pad) > 0) {
            for (int finger = 0; finger < 2; ++finger) {
                bool down;
                float x, y;
                if (SDL_GetGamepadTouchpadFinger(pad, 0, finger, &down, &x, &y, NULL)) {
                    state.OnTouchpad(finger, down, x, y);
                }
            }
            break;
        }
    }

    for (SDL_Gamepad* pad : m_gamepads) {
        if (SDL_GamepadHasSensor(pad, SDL_SENSOR_GYRO)) {
            float gyro[3];
            if (SDL_GetGamepadSensorData(pad, SDL_SENSOR_GYRO, gyro, 3)) {
                state.OnGyro(gyro);
            }
            break;
        }
    }

    for (SDL_Gamepad* pad : m_gamepads) {
        if (SDL_GamepadHasSensor(pad, SDL_SENSOR_ACCEL)) {
            float accel[3];
            if (SDL_GetGamepadSensorData(pad, SDL_SENSOR_ACCEL, accel, 3)) {
                state.OnAccel(accel);
            }
            break;
        }
    }

    return state;
}

float SDLInputEngine::GetGyroPollRate() const {
    return m_gyro_poll_rate;
}

float SDLInputEngine::GetAccelPollRate() const {
    return m_accel_poll_rate;
}

} // namespace Input

namespace Frontend {

using namespace Libraries::Pad;

std::mutex motion_control_mutex;
float gyro_buf[3] = {0.0f, 0.0f, 0.0f}, accel_buf[3] = {0.0f, 9.81f, 0.0f};
static Uint32 SDLCALL PollGyroAndAccel(void* userdata, SDL_TimerID timer_id, Uint32 interval) {
    auto* controller = reinterpret_cast<Input::GameController*>(userdata);
    std::scoped_lock l{motion_control_mutex};
    controller->Gyro(0, gyro_buf);
    controller->Acceleration(0, accel_buf);
    return 4;
}

WindowSDL::WindowSDL(s32 width_, s32 height_, Input::GameController* controller_,
                     std::string_view window_title)
    : width{width_}, height{height_}, controller{controller_} {
    if (!SDL_SetHint(SDL_HINT_APP_NAME, "shadPS4")) {
        UNREACHABLE_MSG("Failed to set SDL window hint: {}", SDL_GetError());
    }
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        UNREACHABLE_MSG("Failed to initialize SDL video subsystem: {}", SDL_GetError());
    }
    SDL_InitSubSystem(SDL_INIT_AUDIO);

    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_TITLE_STRING,
                          std::string(window_title).c_str());
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, width);
    SDL_SetNumberProperty(props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, height);
    SDL_SetNumberProperty(props, "flags", SDL_WINDOW_VULKAN);
    SDL_SetBooleanProperty(props, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true);
    window = SDL_CreateWindowWithProperties(props);
    SDL_DestroyProperties(props);
    if (window == nullptr) {
        UNREACHABLE_MSG("Failed to create window handle: {}", SDL_GetError());
    }

    SDL_SetWindowMinimumSize(window, 640, 360);

    bool error = false;
    const SDL_DisplayID displayIndex = SDL_GetDisplayForWindow(window);
    if (displayIndex < 0) {
        LOG_ERROR(Frontend, "Error getting display index: {}", SDL_GetError());
        error = true;
    }
    const SDL_DisplayMode* displayMode;
    if ((displayMode = SDL_GetCurrentDisplayMode(displayIndex)) == 0) {
        LOG_ERROR(Frontend, "Error getting display mode: {}", SDL_GetError());
        error = true;
    }
    if (!error) {
        SDL_SetWindowFullscreenMode(
            window, Config::getFullscreenMode() == "Fullscreen" ? displayMode : NULL);
    }
    SDL_SetWindowFullscreen(window, Config::getIsFullscreen());

    // FIX(GR2FORK v3.1): "fullscreen stays at 1080p" bug.
    //
    // Before this block, the ctor returned with `width`/`height` still
    // holding the values passed in by the caller (e.g. 1920x1080 from
    // config screenWidth/Height) regardless of whether SDL had just
    // transitioned the window to a 4K fullscreen surface. OnResize is the
    // only thing that ever read back the real size — and OnResize only
    // runs when SDL_EVENT_WINDOW_RESIZED actually fires on the main
    // event loop. The event loop doesn't run during ctor, and on some
    // platforms (gamescope, certain Wayland compositors, exclusive
    // fullscreen with no logical-size delta) the RESIZED event for the
    // construct-time fullscreen transition either fires late or not at
    // all.
    //
    // Meanwhile Presenter -> Swapchain -> ImGui::Core::Initialize ran
    // soon after, and Initialize sets `io.DisplaySize` from
    // `window.GetWidth()/GetHeight()`. Stale 1080p propagated to
    // io.DisplaySize -> DockSpaceOverViewport WorkSize -> contentArea ->
    // SetExpectedGameSize -> the blit-target Frame image. The swapchain
    // itself was the real surface size (Vulkan WSI gives us
    // capabilities.currentExtent = 4K), so the chain became:
    //
    //   game RT (4K, patched) -> Frame (1080p) -> swapchain (4K) -> monitor
    //
    // The 4K -> 1080p downscale at the Frame blit is what users saw as
    // "stays at 1080p" even with fullscreen enabled on a 4K display.
    //
    // SDL_SyncWindow() blocks (up to ~100ms) until the window manager
    // has finished applying the pending state changes — exactly the
    // settling step we need before reading sizes back. We then re-read
    // the actual surface pixel size and write it into the member
    // variables, so the values io.DisplaySize sees in Initialize match
    // ground truth.
    //
    // Done unconditionally: even in windowed mode it's safe (no-op WM
    // sync, read back returns the same requested size), and it catches
    // the windowed-but-clamped case where a compositor refuses the
    // requested size (e.g. a 4K window request on a 1080p display).
    if (!SDL_SyncWindow(window)) {
        LOG_WARNING(Frontend, "[GR2FORK] SDL_SyncWindow returned false (timeout?): {}",
                    SDL_GetError());
    }
    {
        int actual_w = width;
        int actual_h = height;
        SDL_GetWindowSizeInPixels(window, &actual_w, &actual_h);
        if (actual_w != width || actual_h != height) {
            LOG_INFO(Frontend,
                     "[GR2FORK] window size reconciled after SDL setup: "
                     "requested {}x{} → actual {}x{} (fullscreen={}, mode='{}')",
                     width, height, actual_w, actual_h,
                     Config::getIsFullscreen(), Config::getFullscreenMode());
        } else {
            LOG_INFO(Frontend,
                     "[GR2FORK] window size confirmed: {}x{} (fullscreen={}, mode='{}')",
                     actual_w, actual_h,
                     Config::getIsFullscreen(), Config::getFullscreenMode());
        }
        width = actual_w;
        height = actual_h;
    }

    // FIX(GR2FORK): Steam Deck / handheld native gyro & accelerometer support.
    //
    // When shadPS4 is launched through Steam (notably on Steam Deck, but also Legion Go,
    // ROG Ally, and other handhelds running the Steam client), Steam's runtime injects
    // SDL_GAMECONTROLLER_IGNORE_DEVICES / SDL_JOYSTICK_IGNORE_DEVICES into the process
    // environment so SDL hides the underlying controller hardware (e.g. the Deck's
    // 0x28de/0x1205 device) and the game only sees the Steam Virtual Gamepad. The virtual
    // gamepad emulates an Xbox-style controller and does NOT expose gyro / accelerometer
    // even on devices that physically have motion sensors.
    //
    // Forcing the ignore list to empty re-exposes the native controller. SDL then
    // enumerates the underlying device including its motion sensors, and the existing
    // gyro / accel handling (GameController::Gyro / GameController::Acceleration via the
    // SDL_EVENT_GAMEPAD_SENSOR_UPDATE handler) just works -- no input mapping changes
    // required by the user.
    //
    // Equivalent to adding "SDL_GAMECONTROLLER_IGNORE_DEVICES=0 %command%" to Steam launch
    // options, but done internally so it always applies regardless of how the emulator
    // was launched.
    //
    // CRITICAL DETAIL: plain SDL_SetHint (NORMAL priority) does NOT override a hint set
    // via the environment by the parent process. SDL_HINT_OVERRIDE priority is required
    // to clobber Steam's injected env var. Both the SDL2 (GAMECONTROLLER) and SDL3
    // (JOYSTICK) hint names are set so the fix is robust across SDL versions and any
    // future renames -- SDL reads whichever one matches its internal table.
    //
    // Must run BEFORE SDL_InitSubSystem(SDL_INIT_GAMEPAD) -- that's where SDL parses the
    // ignore list and builds its device filter for gamepad enumeration.
    SDL_SetHintWithPriority("SDL_GAMECONTROLLER_IGNORE_DEVICES", "0", SDL_HINT_OVERRIDE);
    SDL_SetHintWithPriority("SDL_JOYSTICK_IGNORE_DEVICES", "0", SDL_HINT_OVERRIDE);

    SDL_InitSubSystem(SDL_INIT_GAMEPAD);
    controller->SetEngine(std::make_unique<Input::SDLInputEngine>());

#if defined(SDL_PLATFORM_WIN32)
    window_info.type = WindowSystemType::Windows;
    window_info.render_surface = SDL_GetPointerProperty(SDL_GetWindowProperties(window),
                                                        SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
#elif defined(SDL_PLATFORM_LINUX)
    if (SDL_strcmp(SDL_GetCurrentVideoDriver(), "x11") == 0) {
        window_info.type = WindowSystemType::X11;
        window_info.display_connection = SDL_GetPointerProperty(
            SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
        window_info.render_surface = (void*)SDL_GetNumberProperty(
            SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
    } else if (SDL_strcmp(SDL_GetCurrentVideoDriver(), "wayland") == 0) {
        window_info.type = WindowSystemType::Wayland;
        window_info.display_connection = SDL_GetPointerProperty(
            SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, NULL);
        window_info.render_surface = SDL_GetPointerProperty(
            SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, NULL);
    }
#elif defined(SDL_PLATFORM_MACOS)
    window_info.type = WindowSystemType::Metal;
    window_info.render_surface = SDL_Metal_GetLayer(SDL_Metal_CreateView(window));
#endif
    // input handler init-s
    Input::ControllerOutput::SetControllerOutputController(controller);
    Input::ControllerOutput::LinkJoystickAxes();
    Input::ParseInputConfig(std::string(Common::ElfInfo::Instance().GameSerial()));

    // Apply default mouse mode if one was set in the config
    auto default_mouse_mode = Input::GetMouseMode();
    if (default_mouse_mode != Input::MouseMode::Off &&
        default_mouse_mode != Input::MouseMode::Touchpad) {
        SDL_SetWindowRelativeMouseMode(window, true);
    }

    if (Config::getBackgroundControllerInput()) {
        SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    }
}

WindowSDL::~WindowSDL() = default;

// [GR2FORK] Load an RGBA image off disk and apply it as this window's icon (taskbar /
// title-bar / Alt-Tab). Used to show the game's own sce_sys/icon0.png instead of the
// generic shadPS4 icon. Ported from upstream shadPS4 PR #4586, adapted to the fork's
// IOFile/stb wrappers and the Frontend log category. Best-effort: every failure path
// logs and bails without disturbing window creation.
void WindowSDL::SetIcon(const std::filesystem::path& path) {
    Common::FS::IOFile file{path, Common::FS::FileAccessMode::Read,
                            Common::FS::FileType::BinaryFile,
                            Common::FS::FileShareFlag::ShareReadWrite};
    if (!file.IsOpen()) {
        LOG_ERROR(Frontend, "Failed to open window icon file '{}'.", fmt::UTF(path.u8string()));
        return;
    }

    const u64 file_size = file.GetSize();
    std::vector<u8> buf(file_size);
    const size_t bytes_read = file.ReadRaw<u8>(buf.data(), file_size);
    file.Close();
    if (bytes_read < file_size) {
        LOG_ERROR(Frontend, "Failed to read window icon file '{}'.", fmt::UTF(path.u8string()));
        return;
    }

    s32 image_width = 0;
    s32 image_height = 0;
    constexpr s32 num_channels = 4;
    unsigned char* image_data =
        stbi_load_from_memory(buf.data(), static_cast<s32>(buf.size()), &image_width, &image_height,
                              nullptr, num_channels);
    if (image_data == nullptr) {
        LOG_ERROR(Frontend, "Failed to load window icon image '{}': {}", fmt::UTF(path.u8string()),
                  stbi_failure_reason());
        return;
    }
    SCOPE_EXIT {
        stbi_image_free(image_data);
    };

    SDL_Surface* surface = SDL_CreateSurfaceFrom(image_width, image_height, SDL_PIXELFORMAT_RGBA32,
                                                 image_data, image_width * num_channels);
    if (surface == nullptr) {
        LOG_ERROR(Frontend, "Failed to create SDL surface for window icon: {}", SDL_GetError());
        return;
    }
    SDL_SetWindowIcon(window, surface);
    SDL_DestroySurface(surface);
}

void WindowSDL::WaitEvent() {
    // Called on main thread
    SDL_Event event;

    if (!SDL_WaitEvent(&event)) {
        return;
    }

    if (ImGui::Core::ProcessEvent(&event)) {
        return;
    }

    switch (event.type) {
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_MAXIMIZED:
    case SDL_EVENT_WINDOW_RESTORED:
        OnResize();
        break;
    case SDL_EVENT_WINDOW_MINIMIZED:
    case SDL_EVENT_WINDOW_EXPOSED:
        is_shown = event.type == SDL_EVENT_WINDOW_EXPOSED;
        OnResize();
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
    case SDL_EVENT_MOUSE_WHEEL:
    case SDL_EVENT_MOUSE_WHEEL_OFF:
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        OnKeyboardMouseInput(&event);
        break;
    case SDL_EVENT_GAMEPAD_ADDED:
    case SDL_EVENT_GAMEPAD_REMOVED:
        controller->SetEngine(std::make_unique<Input::SDLInputEngine>());
        break;
    case SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_UP:
    case SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION:
        controller->SetTouchpadState(event.gtouchpad.finger,
                                     event.type != SDL_EVENT_GAMEPAD_TOUCHPAD_UP, event.gtouchpad.x,
                                     event.gtouchpad.y);
        break;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
    case SDL_EVENT_GAMEPAD_AXIS_MOTION:
        OnGamepadEvent(&event);
        break;
    // i really would have appreciated ANY KIND OF DOCUMENTATION ON THIS
    // AND IT DOESN'T EVEN USE PROPER ENUMS
    case SDL_EVENT_GAMEPAD_SENSOR_UPDATE:
        switch ((SDL_SensorType)event.gsensor.sensor) {
        case SDL_SENSOR_GYRO: {
            std::scoped_lock l{motion_control_mutex};
            memcpy(gyro_buf, event.gsensor.data, sizeof(gyro_buf));
            break;
        }
        case SDL_SENSOR_ACCEL: {
            std::scoped_lock l{motion_control_mutex};
            memcpy(accel_buf, event.gsensor.data, sizeof(accel_buf));
            break;
        }
        default:
            break;
        }
        break;
    case SDL_EVENT_QUIT:
        is_open = false;
        break;
    case SDL_EVENT_QUIT_DIALOG:
        Overlay::ToggleQuitWindow();
        break;
    case SDL_EVENT_TOGGLE_FULLSCREEN: {
        if (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) {
            SDL_SetWindowFullscreen(window, 0);
        } else {
            SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
        }
        break;
    }
    case SDL_EVENT_TOGGLE_PAUSE:
        if (DebugState.IsGuestThreadsPaused()) {
            LOG_INFO(Frontend, "Game Resumed");
            DebugState.ResumeGuestThreads();
        } else {
            LOG_INFO(Frontend, "Game Paused");
            DebugState.PauseGuestThreads();
        }
        break;
    case SDL_EVENT_CHANGE_CONTROLLER:
        controller->GetEngine()->Init();
        break;
    case SDL_EVENT_TOGGLE_SIMPLE_FPS:
        Overlay::ToggleSimpleFps();
        break;
    case SDL_EVENT_RELOAD_INPUTS:
        Input::ParseInputConfig(std::string(Common::ElfInfo::Instance().GameSerial()));
        break;
    case SDL_EVENT_MOUSE_TO_JOYSTICK:
        SDL_SetWindowRelativeMouseMode(this->GetSDLWindow(),
                                       Input::ToggleMouseModeTo(Input::MouseMode::Joystick));
        break;
    case SDL_EVENT_MOUSE_TO_GYRO:
        SDL_SetWindowRelativeMouseMode(this->GetSDLWindow(),
                                       Input::ToggleMouseModeTo(Input::MouseMode::Gyro));
        break;
    case SDL_EVENT_MOUSE_TO_TOUCHPAD:
        SDL_SetWindowRelativeMouseMode(this->GetSDLWindow(),
                                       Input::ToggleMouseModeTo(Input::MouseMode::Touchpad));
        SDL_SetWindowRelativeMouseMode(this->GetSDLWindow(), false);
        break;
    case SDL_EVENT_MOUSE_TO_TOUCHPAD_SWIPE:
        Input::EnableTouchpadSwipe(!Input::IsTouchpadSwipeEnabled());
        break;
    case SDL_EVENT_RDOC_CAPTURE:
        VideoCore::TriggerCapture();
        break;
    default:
        break;
    }
}

void WindowSDL::InitTimers() {
    SDL_AddTimer(4, &PollGyroAndAccel, controller);
    SDL_AddTimer(33, Input::MousePolling, (void*)controller);
}

void WindowSDL::RequestKeyboard() {
    if (keyboard_grab == 0) {
        SDL_RunOnMainThread(
            [](void* userdata) { SDL_StartTextInput(static_cast<SDL_Window*>(userdata)); }, window,
            true);
    }
    keyboard_grab++;
}

void WindowSDL::ReleaseKeyboard() {
    ASSERT(keyboard_grab > 0);
    keyboard_grab--;
    if (keyboard_grab == 0) {
        SDL_RunOnMainThread(
            [](void* userdata) { SDL_StopTextInput(static_cast<SDL_Window*>(userdata)); }, window,
            true);
    }
}

void WindowSDL::OnResize() {
    SDL_GetWindowSizeInPixels(window, &width, &height);
    ImGui::Core::OnResize();
}

Uint32 wheelOffCallback(void* og_event, Uint32 timer_id, Uint32 interval) {
    SDL_Event off_event = *(SDL_Event*)og_event;
    off_event.type = SDL_EVENT_MOUSE_WHEEL_OFF;
    SDL_PushEvent(&off_event);
    delete (SDL_Event*)og_event;
    return 0;
}

void WindowSDL::OnKeyboardMouseInput(const SDL_Event* event) {
    using Libraries::Pad::OrbisPadButtonDataOffset;

    // Touchpad swipe interception — completely independent of mouse mode.
    // Only intercepts left mouse button down/up. All other events pass through.
    if (Input::IsTouchpadSwipeEnabled()) {
        if (event->type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
            event->button.button == SDL_BUTTON_LEFT) {
            LOG_INFO(Input, "Swipe intercept: BUTTON_DOWN at ({}, {})", event->button.x,
                     event->button.y);
            Input::TouchpadSwipeOnFingerDown(controller, event->button.x, event->button.y);
            return;
        }
        if (event->type == SDL_EVENT_MOUSE_BUTTON_UP &&
            event->button.button == SDL_BUTTON_LEFT) {
            LOG_INFO(Input, "Swipe intercept: BUTTON_UP at ({}, {})", event->button.x,
                     event->button.y);
            Input::TouchpadSwipeOnFingerUp(controller, event->button.x, event->button.y);
            return;
        }
    }

    // get the event's id, if it's keyup or keydown
    const bool input_down = event->type == SDL_EVENT_KEY_DOWN ||
                            event->type == SDL_EVENT_MOUSE_BUTTON_DOWN ||
                            event->type == SDL_EVENT_MOUSE_WHEEL;
    Input::InputEvent input_event = Input::InputBinding::GetInputEventFromSDLEvent(*event);

    // if it's a wheel event, make a timer that turns it off after a set time
    if (event->type == SDL_EVENT_MOUSE_WHEEL) {
        const SDL_Event* copy = new SDL_Event(*event);
        SDL_AddTimer(33, wheelOffCallback, (void*)copy);
    }

    // add/remove it from the list
    bool inputs_changed = Input::UpdatePressedKeys(input_event);

    // update bindings
    if (inputs_changed) {
        Input::ActivateOutputsFromInputs();
    }
}

void WindowSDL::OnGamepadEvent(const SDL_Event* event) {
    bool input_down = event->type == SDL_EVENT_GAMEPAD_AXIS_MOTION ||
                      event->type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
    Input::InputEvent input_event = Input::InputBinding::GetInputEventFromSDLEvent(*event);

    // the touchpad button shouldn't be rebound to anything else,
    // as it would break the entire touchpad handling
    // You can still bind other things to it though
    if (event->gbutton.button == SDL_GAMEPAD_BUTTON_TOUCHPAD) {
        controller->CheckButton(0, OrbisPadButtonDataOffset::TouchPad, input_down);
        return;
    }

    // add/remove it from the list
    bool inputs_changed = Input::UpdatePressedKeys(input_event);

    if (inputs_changed) {
        // update bindings
        Input::ActivateOutputsFromInputs();
    }
}

} // namespace Frontend
