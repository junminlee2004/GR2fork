// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "SDL3/SDL.h"
#include "common/types.h"

namespace Input {

class GameController;

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
void SetTouchpadSwipeSpeed(float speed);
void SetTouchpadSwipeThreshold(float threshold);

// Touchpad swipe — completely independent of the mouse mode system
void EnableTouchpadSwipe(bool enable);
bool IsTouchpadSwipeEnabled();
void TouchpadSwipeOnFingerDown(GameController* controller, float abs_x, float abs_y);
void TouchpadSwipeOnFingerUp(GameController* controller, float abs_x, float abs_y);

// Button-triggered touchpad swipe.
// Configures the delay between center-press and direction-press for the
// button-triggered swipe, in milliseconds. Default is 200ms. Hold time at the
// endpoint before release is hardcoded at 100ms.
void SetTouchpadSwipeButtonDelay(int delay_ms);

// Kicks off a one-shot timed swipe from the touchpad center to the given
// direction's edge. Sequence:
//     t=0           touch DOWN at (0.5, 0.5) + TouchPad button DOWN
//     t=delay       touch moves to direction endpoint (still DOWN)
//     t=delay+hold  release everything
// `direction` must be one of ButtonSwipeDirection. Repeat triggers while a
// swipe is already in flight are ignored.
void TriggerButtonSwipe(GameController* controller, int direction);

void ApplyMouseInputBlockers();

// Polls the mouse for changes
Uint32 MousePolling(void* param, Uint32 id, Uint32 interval);

} // namespace Input
