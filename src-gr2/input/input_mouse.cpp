// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#define _USE_MATH_DEFINES
#include <atomic>
#include <cmath>

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

// Touchpad swipe emulation state - completely independent of mouse_mode
float touchpad_swipe_speed = 0.005f;
float touchpad_swipe_threshold = 15.0f;
float swipe_start_x = 0.0f, swipe_start_y = 0.0f;
bool swipe_active = false;
bool touchpad_swipe_enabled = false;

// Staged swipe playback - needs multiple frames for game to register
enum class SwipePlayback { Idle, TouchDown, SwipeMove, Done };
SwipePlayback swipe_playback = SwipePlayback::Idle;
float swipe_end_x = 0.5f, swipe_end_y = 0.5f;
int swipe_frame_counter = 0;

// Button-triggered touchpad swipe, independent of the mouse-driven swipe above and driven by
// SDL_AddTimer so its timing does not depend on MousePolling. TriggerButtonSwipe runs three
// phases: touch down at center, touch down at endpoint after the delay, then release all.

namespace {

constexpr int kButtonSwipeDefaultDelayMs = 200;
constexpr int kButtonSwipeHoldMs = 100;

// Endpoints in (x, y) normalized 0..1, matching the existing touchpad_* region buttons.
constexpr float kButtonSwipeEndpoints[4][2] = {
    {0.5f, 0.25f}, // BUTTON_SWIPE_UP
    {0.5f, 0.75f}, // BUTTON_SWIPE_DOWN
    {0.25f, 0.5f}, // BUTTON_SWIPE_LEFT
    {0.75f, 0.5f}, // BUTTON_SWIPE_RIGHT
};

std::atomic<int> g_button_swipe_delay_ms{kButtonSwipeDefaultDelayMs};

struct ButtonSwipeState {
    GameController* controller = nullptr;
    int direction = 0;
    std::atomic<bool> active{false};
};
ButtonSwipeState g_button_swipe;

Uint32 ButtonSwipeReleaseCallback(void* param, SDL_TimerID /*id*/, Uint32 /*interval*/) {
    auto* controller = static_cast<GameController*>(param);
    int dir = g_button_swipe.direction;
    if (controller) {
        controller->SetTouchpadState(0, false, kButtonSwipeEndpoints[dir][0],
                                     kButtonSwipeEndpoints[dir][1]);
        controller->CheckButton(0, Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, false);
    }
    g_button_swipe.active.store(false, std::memory_order_release);
    LOG_INFO(Input, "Button swipe complete dir={}", dir);
    return 0; // one-shot
}

Uint32 ButtonSwipeMoveCallback(void* param, SDL_TimerID /*id*/, Uint32 /*interval*/) {
    auto* controller = static_cast<GameController*>(param);
    int dir = g_button_swipe.direction;
    if (controller) {
        // Touch stays DOWN; we just move the contact point.
        controller->SetTouchpadState(0, true, kButtonSwipeEndpoints[dir][0],
                                     kButtonSwipeEndpoints[dir][1]);
    }
    SDL_AddTimer(kButtonSwipeHoldMs, ButtonSwipeReleaseCallback, param);
    return 0; // one-shot
}

} // namespace

void SetTouchpadSwipeButtonDelay(int delay_ms) {
    if (delay_ms < 1) {
        delay_ms = 1;
    }
    g_button_swipe_delay_ms.store(delay_ms, std::memory_order_release);
    LOG_INFO(Input, "Touchpad swipe button delay set to {}ms", delay_ms);
}

void TriggerButtonSwipe(GameController* controller, int direction) {
    if (!controller) {
        return;
    }
    if (direction < 0 || direction > 3) {
        LOG_WARNING(Input, "TriggerButtonSwipe: invalid direction {}", direction);
        return;
    }
    // Atomically claim the swipe slot. If a swipe is already in flight, bail.
    bool expected = false;
    if (!g_button_swipe.active.compare_exchange_strong(expected, true,
                                                       std::memory_order_acq_rel)) {
        LOG_DEBUG(Input, "TriggerButtonSwipe: swipe already in flight, ignoring");
        return;
    }

    g_button_swipe.controller = controller;
    g_button_swipe.direction = direction;

    // Phase 0: immediate touch DOWN at center.
    controller->SetTouchpadState(0, true, 0.5f, 0.5f);
    controller->CheckButton(0, Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, true);

    int delay_ms = g_button_swipe_delay_ms.load(std::memory_order_acquire);
    SDL_TimerID move_timer = SDL_AddTimer(static_cast<Uint32>(delay_ms), ButtonSwipeMoveCallback,
                                          static_cast<void*>(controller));
    if (move_timer == 0) {
        LOG_ERROR(Input, "TriggerButtonSwipe: SDL_AddTimer failed, releasing immediately");
        controller->SetTouchpadState(0, false, 0.5f, 0.5f);
        controller->CheckButton(0, Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, false);
        g_button_swipe.active.store(false, std::memory_order_release);
        return;
    }
    LOG_INFO(Input, "Button swipe triggered dir={} delay={}ms hold={}ms", direction, delay_ms,
             kButtonSwipeHoldMs);
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

void SetTouchpadSwipeSpeed(float speed) {
    touchpad_swipe_speed = speed;
}

void SetTouchpadSwipeThreshold(float threshold) {
    touchpad_swipe_threshold = threshold;
}

void EnableTouchpadSwipe(bool enable) {
    touchpad_swipe_enabled = enable;
    LOG_INFO(Input, "TouchpadSwipe enabled: {}", enable);
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

    float output_speed =
        SDL_clamp((sqrt(d_x * d_x + d_y * d_y) + mouse_speed_offset * 128) * mouse_speed,
                  mouse_deadzone_offset * 128, 128.0);

    float angle = atan2(d_y, d_x);
    float a_x = cos(angle) * output_speed, a_y = sin(angle) * output_speed;

    if (d_x != 0 || d_y != 0) {
        controller->Axis(0, axis_x, GetAxis(-0x80, 0x7f, a_x));
        controller->Axis(0, axis_y, GetAxis(-0x80, 0x7f, a_y));
    } else {
        controller->Axis(0, axis_x, GetAxis(-0x80, 0x7f, 0));
        controller->Axis(0, axis_y, GetAxis(-0x80, 0x7f, 0));
    }
}

constexpr float constant_down_accel[3] = {0.0f, 10.0f, 0.0f};
void EmulateGyro(GameController* controller, u32 interval) {
    float d_x = 0, d_y = 0;
    SDL_GetRelativeMouseState(&d_x, &d_y);
    controller->Acceleration(1, constant_down_accel);
    float gyro_from_mouse[3] = {-d_y / 100, -d_x / 100, 0.0f};
    if (mouse_gyro_roll_mode) {
        gyro_from_mouse[1] = 0.0f;
        gyro_from_mouse[2] = -d_x / 100;
    }
    controller->Gyro(1, gyro_from_mouse);
}

void EmulateTouchpad(GameController* controller, u32 interval) {
    float x, y;
    SDL_MouseButtonFlags mouse_buttons = SDL_GetMouseState(&x, &y);
    controller->SetTouchpadState(0, (mouse_buttons & SDL_BUTTON_LMASK) != 0,
                                 std::clamp(x / g_window->GetWidth(), 0.0f, 1.0f),
                                 std::clamp(y / g_window->GetHeight(), 0.0f, 1.0f));
    controller->CheckButton(0, Libraries::Pad::OrbisPadButtonDataOffset::TouchPad,
                            (mouse_buttons & SDL_BUTTON_RMASK) != 0);
}

// Touchpad swipe: on finger down, just record start position.
// Cancels any in-progress playback from a previous swipe.
void TouchpadSwipeOnFingerDown(GameController* controller, float abs_x, float abs_y) {
    // If a previous swipe is still playing back, cancel it immediately
    if (swipe_playback != SwipePlayback::Idle) {
        controller->SetTouchpadState(0, false, swipe_end_x, swipe_end_y);
        controller->CheckButton(0, Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, false);
        swipe_playback = SwipePlayback::Idle;
    }
    swipe_active = true;
    swipe_start_x = abs_x;
    swipe_start_y = abs_y;
    LOG_INFO(Input, "TouchpadSwipe FINGER DOWN at ({}, {})", abs_x, abs_y);
}

// Touchpad swipe: on finger up, compute direction and start staged playback.
void TouchpadSwipeOnFingerUp(GameController* controller, float abs_x, float abs_y) {
    if (!swipe_active) {
        return;
    }
    swipe_active = false;

    float d_x = abs_x - swipe_start_x;
    float d_y = abs_y - swipe_start_y;

    swipe_end_x = 0.5f;
    swipe_end_y = 0.5f;

    // Only register a swipe if movement exceeds threshold
    float dist = std::sqrt(d_x * d_x + d_y * d_y);
    if (dist >= touchpad_swipe_threshold) {

        // 8-way snap: compute angle, divide into 45-degree sectors
        float angle = std::atan2(d_y, d_x); // radians, -PI to PI
        // Normalize to 0-7 sector: 0=right, 1=down-right, 2=down, etc.
        int sector = static_cast<int>(std::round(angle / (M_PI / 4.0f))) & 7;

        constexpr float endpoints[8][2] = {
            {0.9f, 0.5f}, // 0: right
            {0.9f, 0.9f}, // 1: down-right
            {0.5f, 0.9f}, // 2: down
            {0.1f, 0.9f}, // 3: down-left
            {0.1f, 0.5f}, // 4: left
            {0.1f, 0.1f}, // 5: up-left
            {0.5f, 0.1f}, // 6: up
            {0.9f, 0.1f}, // 7: up-right
        };

        swipe_end_x = endpoints[sector][0];
        swipe_end_y = endpoints[sector][1];

        LOG_INFO(Input, "TouchpadSwipe DIRECTION: dx={}, dy={}, sector={}, endpoint=({}, {})",
                 d_x, d_y, sector, swipe_end_x, swipe_end_y);
    } else {
        LOG_INFO(Input, "TouchpadSwipe TAP: dx={}, dy={}", d_x, d_y);
    }

    // Start staged playback - the polling timer will advance it frame by frame
    swipe_playback = SwipePlayback::TouchDown;
    swipe_frame_counter = 0;
}

// Called from MousePolling every 33ms - advances swipe playback across frames
void AdvanceTouchpadSwipe(GameController* controller) {
    switch (swipe_playback) {
    case SwipePlayback::Idle:
        return;

    case SwipePlayback::TouchDown:
        // Frames 0-1: finger pressed at center
        controller->SetTouchpadState(0, true, 0.5f, 0.5f);
        controller->CheckButton(0, Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, true);
        swipe_frame_counter++;
        if (swipe_frame_counter >= 2) {
            swipe_playback = SwipePlayback::SwipeMove;
            swipe_frame_counter = 0;
        }
        break;

    case SwipePlayback::SwipeMove:
        // Frames 2-3: finger at swipe endpoint
        controller->SetTouchpadState(0, true, swipe_end_x, swipe_end_y);
        swipe_frame_counter++;
        if (swipe_frame_counter >= 2) {
            swipe_playback = SwipePlayback::Done;
            swipe_frame_counter = 0;
        }
        break;

    case SwipePlayback::Done:
        // Frame 6: release
        controller->SetTouchpadState(0, false, swipe_end_x, swipe_end_y);
        controller->CheckButton(0, Libraries::Pad::OrbisPadButtonDataOffset::TouchPad, false);
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
    // Touchpad swipe playback runs independently of mouse_mode
    if (touchpad_swipe_enabled) {
        AdvanceTouchpadSwipe(controller);
    }
    return interval;
}

} // namespace Input
