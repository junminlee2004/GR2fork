// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/types.h"
#include "input/controller.h"
#include "input/input_handler.h"
#include "input_mouse.h"

#include <common/singleton.h>
#include <emulator.h>
#include "SDL3/SDL.h"

extern Frontend::WindowSDL* g_window;

namespace Input {

extern std::list<std::pair<InputEvent, bool>> pressed_keys;

int mouse_joystick_binding = 0;
float mouse_deadzone_offset = 0.5, mouse_speed = 1, mouse_speed_offset = 0.1250;
bool mouse_gyro_roll_mode = false;
Uint32 mouse_polling_id = 0;
MouseMode mouse_mode = MouseMode::Off;
// Mouse-to-joystick sensitivity: global multiplies both axes, the pair scales each axis; applied
// to the raw deltas before the magnitude/angle split.
float mouse_sensitivity = 1.0f, mouse_sensitivity_x = 1.0f, mouse_sensitivity_y = 1.0f;

// Touchpad swipe emulation state, independent of mouse_mode.
float touchpad_swipe_speed = 0.005f;
float touchpad_swipe_threshold = 15.0f;
float swipe_start_x = 0.0f, swipe_start_y = 0.0f;
bool swipe_active = false;
bool touchpad_swipe_enabled = false;

// Staged swipe playback: the game needs several frames to register a swipe.
enum class SwipePlayback { Idle, TouchDown, SwipeMove, Done };
SwipePlayback swipe_playback = SwipePlayback::Idle;
float swipe_end_x = 0.5f, swipe_end_y = 0.5f;
int swipe_frame_counter = 0;

// Button-triggered swipe, driven by SDL_AddTimer so its timing does not depend on MousePolling:
// touch down at the centre, touch down at the endpoint after the delay, then release all.
namespace {

constexpr int kButtonSwipeDefaultDelayMs = 200;
constexpr int kButtonSwipeHoldMs = 100;

// Endpoints in normalized (x, y), matching the touchpad_* region outputs.
constexpr float kButtonSwipeEndpoints[4][2] = {
    {0.5f, 0.25f}, // BUTTON_SWIPE_UP
    {0.5f, 0.75f}, // BUTTON_SWIPE_DOWN
    {0.25f, 0.5f}, // BUTTON_SWIPE_LEFT
    {0.75f, 0.5f}, // BUTTON_SWIPE_RIGHT
};

std::atomic<int> g_button_swipe_delay_ms{kButtonSwipeDefaultDelayMs};

struct ButtonSwipeState {
    int direction = 0;
    std::atomic<bool> active{false};
};
ButtonSwipeState g_button_swipe;

Uint32 ButtonSwipeReleaseCallback(void* param, SDL_TimerID /*id*/, Uint32 /*interval*/) {
    auto* controller = static_cast<GameController*>(param);
    const int dir = g_button_swipe.direction;
    controller->SetTouchpadState(0, false, kButtonSwipeEndpoints[dir][0],
                                 kButtonSwipeEndpoints[dir][1]);
    controller->Button(Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, false);
    g_button_swipe.active.store(false, std::memory_order_release);
    return 0; // one-shot
}

Uint32 ButtonSwipeMoveCallback(void* param, SDL_TimerID /*id*/, Uint32 /*interval*/) {
    auto* controller = static_cast<GameController*>(param);
    const int dir = g_button_swipe.direction;
    // The touch stays down; only the contact point moves.
    controller->SetTouchpadState(0, true, kButtonSwipeEndpoints[dir][0],
                                 kButtonSwipeEndpoints[dir][1]);
    SDL_AddTimer(kButtonSwipeHoldMs, ButtonSwipeReleaseCallback, param);
    return 0; // one-shot
}

} // namespace

void SetTouchpadSwipeButtonDelay(int delay_ms) {
    g_button_swipe_delay_ms.store(std::max(delay_ms, 1), std::memory_order_release);
}

void TriggerButtonSwipe(GameController* controller, int direction) {
    if (!controller || direction < 0 || direction > 3) {
        return;
    }
    // Claim the swipe slot; a swipe already in flight wins.
    bool expected = false;
    if (!g_button_swipe.active.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }
    g_button_swipe.direction = direction;
    // Phase 0: touch down at the centre.
    controller->SetTouchpadState(0, true, 0.5f, 0.5f);
    controller->Button(Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, true);
    const int delay_ms = g_button_swipe_delay_ms.load(std::memory_order_acquire);
    if (SDL_AddTimer(static_cast<Uint32>(delay_ms), ButtonSwipeMoveCallback,
                     static_cast<void*>(controller)) == 0) {
        LOG_ERROR(Input, "TriggerButtonSwipe: SDL_AddTimer failed, releasing immediately");
        controller->SetTouchpadState(0, false, 0.5f, 0.5f);
        controller->Button(Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, false);
        g_button_swipe.active.store(false, std::memory_order_release);
    }
}

// Switches mouse to a set mode or turns mouse emulation off if it was already in that mode.
// Returns whether the mode is turned on.
bool ToggleMouseModeTo(MouseMode m) {
    if (mouse_mode == m) {
        mouse_mode = MouseMode::Off;
    } else {
        mouse_mode = m;
    }
    return mouse_mode == m;
}

void SetMouseMode(MouseMode m) {
    mouse_mode = m;
}

MouseMode GetMouseMode() {
    return mouse_mode;
}

void SetMouseToJoystick(int joystick) {
    mouse_joystick_binding = joystick;
}

void SetMouseParams(float mdo, float ms, float mso) {
    mouse_deadzone_offset = mdo;
    mouse_speed = ms;
    mouse_speed_offset = mso;
}

void SetMouseGyroRollMode(bool mode) {
    mouse_gyro_roll_mode = mode;
}

void SetMouseSensitivity(float global, float horizontal, float vertical) {
    mouse_sensitivity = global;
    mouse_sensitivity_x = horizontal;
    mouse_sensitivity_y = vertical;
}

void SetTouchpadSwipeSpeed(float speed) {
    touchpad_swipe_speed = speed;
}

void SetTouchpadSwipeThreshold(float threshold) {
    touchpad_swipe_threshold = threshold;
}

void EnableTouchpadSwipe(bool enable) {
    touchpad_swipe_enabled = enable;
    LOG_INFO(Input, "Touchpad swipe emulation {}", enable ? "enabled" : "disabled");
    if (!enable) {
        swipe_active = false;
    }
}

bool IsTouchpadSwipeEnabled() {
    return touchpad_swipe_enabled;
}

void EmulateJoystick(GameController* controller, u32 interval) {

    Axis axis_x, axis_y;
    switch (mouse_joystick_binding) {
    case 1:
        axis_x = Axis::LeftX;
        axis_y = Axis::LeftY;
        break;
    case 2:
        axis_x = Axis::RightX;
        axis_y = Axis::RightY;
        break;
    default:
        return; // no update needed
    }

    float d_x = 0, d_y = 0;
    SDL_GetRelativeMouseState(&d_x, &d_y);
    d_x *= mouse_sensitivity * mouse_sensitivity_x;
    d_y *= mouse_sensitivity * mouse_sensitivity_y;

    float output_speed =
        SDL_clamp(sqrt(d_x * d_x + d_y * d_y) * mouse_speed + mouse_speed_offset * 128,
                  mouse_deadzone_offset * 128, 128.0);

    float angle = atan2(d_y, d_x);
    float a_x = cos(angle) * output_speed, a_y = sin(angle) * output_speed;

    if (d_x != 0 || d_y != 0) {
        controller->Axis(axis_x, GetAxis(-0x80, 0x7f, a_x), false);
        controller->Axis(axis_y, GetAxis(-0x80, 0x7f, a_y), false);
    } else {
        controller->Axis(axis_x, GetAxis(-0x80, 0x7f, 0), false);
        controller->Axis(axis_y, GetAxis(-0x80, 0x7f, 0), false);
    }
}

constexpr float constant_down_accel[3] = {0.0f, 9.81f, 0.0f};
void EmulateGyro(GameController* controller, u32 interval) {
    float d_x = 0, d_y = 0;
    SDL_GetRelativeMouseState(&d_x, &d_y);
    controller->UpdateAcceleration(constant_down_accel);
    float gyro_from_mouse[3] = {-d_y / 100, -d_x / 100, 0.0f};
    if (mouse_gyro_roll_mode) {
        gyro_from_mouse[1] = 0.0f;
        gyro_from_mouse[2] = -d_x / 100;
    }
    controller->UpdateGyro(gyro_from_mouse);
}

void EmulateTouchpad(GameController* controller, u32 interval) {
    float x, y;
    SDL_MouseButtonFlags mouse_buttons = SDL_GetMouseState(&x, &y);
    controller->SetTouchpadState(0, (mouse_buttons & SDL_BUTTON_LMASK) != 0,
                                 std::clamp(x / g_window->GetWidth(), 0.0f, 1.0f),
                                 std::clamp(y / g_window->GetHeight(), 0.0f, 1.0f));
    controller->Button(Libraries::Pad::OrbisPadButtonDataOffset::TouchPad,
                       (mouse_buttons & SDL_BUTTON_RMASK) != 0);
}

// Finger down only records the start position and cancels any playback still in progress.
void TouchpadSwipeOnFingerDown(GameController* controller, float abs_x, float abs_y) {
    if (swipe_playback != SwipePlayback::Idle) {
        controller->SetTouchpadState(0, false, swipe_end_x, swipe_end_y);
        controller->Button(Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, false);
        swipe_playback = SwipePlayback::Idle;
    }
    swipe_active = true;
    swipe_start_x = abs_x;
    swipe_start_y = abs_y;
}

// Finger up snaps the motion to one of eight directions (or a tap) and starts staged playback.
void TouchpadSwipeOnFingerUp(GameController* controller, float abs_x, float abs_y) {
    if (!swipe_active) {
        return;
    }
    swipe_active = false;
    const float d_x = abs_x - swipe_start_x;
    const float d_y = abs_y - swipe_start_y;
    swipe_end_x = 0.5f;
    swipe_end_y = 0.5f;
    if (std::sqrt(d_x * d_x + d_y * d_y) >= touchpad_swipe_threshold) {
        // 45-degree sectors: 0 = right, 1 = down-right, 2 = down, ...
        const float angle = std::atan2(d_y, d_x);
        const int sector =
            static_cast<int>(std::round(angle / (std::numbers::pi_v<float> / 4.0f))) & 7;
        constexpr float endpoints[8][2] = {
            {0.9f, 0.5f}, {0.9f, 0.9f}, {0.5f, 0.9f}, {0.1f, 0.9f},
            {0.1f, 0.5f}, {0.1f, 0.1f}, {0.5f, 0.1f}, {0.9f, 0.1f},
        };
        swipe_end_x = endpoints[sector][0];
        swipe_end_y = endpoints[sector][1];
        LOG_DEBUG(Input, "Touchpad swipe: dx={} dy={} sector={}", d_x, d_y, sector);
    } else {
        LOG_DEBUG(Input, "Touchpad tap: dx={} dy={}", d_x, d_y);
    }
    swipe_playback = SwipePlayback::TouchDown;
    swipe_frame_counter = 0;
}

// Advances the staged playback from MousePolling: two polls pressed at the centre, two at the
// endpoint, then release.
static void AdvanceTouchpadSwipe(GameController* controller) {
    switch (swipe_playback) {
    case SwipePlayback::Idle:
        return;
    case SwipePlayback::TouchDown:
        controller->SetTouchpadState(0, true, 0.5f, 0.5f);
        controller->Button(Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, true);
        if (++swipe_frame_counter >= 2) {
            swipe_playback = SwipePlayback::SwipeMove;
            swipe_frame_counter = 0;
        }
        break;
    case SwipePlayback::SwipeMove:
        controller->SetTouchpadState(0, true, swipe_end_x, swipe_end_y);
        if (++swipe_frame_counter >= 2) {
            swipe_playback = SwipePlayback::Done;
            swipe_frame_counter = 0;
        }
        break;
    case SwipePlayback::Done:
        controller->SetTouchpadState(0, false, swipe_end_x, swipe_end_y);
        controller->Button(Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, false);
        swipe_playback = SwipePlayback::Idle;
        break;
    }
}

void ApplyMouseInputBlockers() {
    switch (mouse_mode) {
    case MouseMode::Touchpad:
        for (auto& k : pressed_keys) {
            if (k.first.input.sdl_id == SDL_BUTTON_LEFT ||
                k.first.input.sdl_id == SDL_BUTTON_RIGHT) {
                k.second = true;
            }
        }
        break;
    default:
        break;
    }
}

Uint32 MousePolling(void* param, Uint32 id, Uint32 interval) {
    auto* controller = (GameController*)param;
    switch (mouse_mode) {
    case MouseMode::Joystick:
        EmulateJoystick(controller, interval);
        break;
    case MouseMode::Gyro:
        EmulateGyro(controller, interval);
        break;
    case MouseMode::Touchpad:
        EmulateTouchpad(controller, interval);
        break;

    default:
        break;
    }
    // Swipe playback runs independently of the mouse mode.
    if (touchpad_swipe_enabled) {
        AdvanceTouchpadSwipe(controller);
    }
    return interval;
}

} // namespace Input
