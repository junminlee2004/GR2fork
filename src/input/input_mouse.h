// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "SDL3/SDL.h"
#include "common/types.h"

namespace Input {

enum MouseMode {
    Off = 0,
    Joystick,
    Gyro,
    Touchpad,
};

// Direction codes used by the button-triggered touchpad swipe.
enum ButtonSwipeDirection {
    BUTTON_SWIPE_UP = 0,
    BUTTON_SWIPE_DOWN = 1,
    BUTTON_SWIPE_LEFT = 2,
    BUTTON_SWIPE_RIGHT = 3,
};

bool ToggleMouseModeTo(MouseMode m);
void SetMouseMode(MouseMode m);
MouseMode GetMouseMode();
void SetMouseToJoystick(int joystick);
void SetMouseParams(float mouse_deadzone_offset, float mouse_speed, float mouse_speed_offset);
void SetMouseGyroRollMode(bool mode);
// Global and per-axis (horizontal, vertical) mouse-to-joystick sensitivity multipliers.
void SetMouseSensitivity(float global, float horizontal, float vertical);
// Touchscreen/mouse swipes played back as PS4 touchpad swipes; independent of the mouse mode.
void SetTouchpadSwipeSpeed(float speed);
void SetTouchpadSwipeThreshold(float threshold);
void EnableTouchpadSwipe(bool enable);
bool IsTouchpadSwipeEnabled();
void TouchpadSwipeOnFingerDown(GameController* controller, float abs_x, float abs_y);
void TouchpadSwipeOnFingerUp(GameController* controller, float abs_x, float abs_y);
// Delay between the centre press and the direction press of a button-triggered swipe, in
// milliseconds (default 200). The hold at the endpoint before release is fixed at 100 ms.
void SetTouchpadSwipeButtonDelay(int delay_ms);
// One-shot timed swipe: touch down at (0.5, 0.5) with the TouchPad button, move to the
// direction's edge after the delay, release after the hold. Ignored while a swipe is in flight.
void TriggerButtonSwipe(GameController* controller, int direction);

void EmulateJoystick(GameController* controller, u32 interval);
void EmulateGyro(GameController* controller, u32 interval);

void ApplyMouseInputBlockers();

// Polls the mouse for changes
Uint32 MousePolling(void* param, Uint32 id, Uint32 interval);

} // namespace Input
