// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <SDL3/SDL.h>
#include "common/config.h"
#include "common/logging/log.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/pad/pad.h"
#include "input/controller.h"

static std::string SelectedGamepad = "";

namespace Input {

using Libraries::Pad::OrbisPadButtonDataOffset;

void State::OnButton(OrbisPadButtonDataOffset button, bool isPressed) {
    if (isPressed) {
        buttonsState |= button;
    } else {
        buttonsState &= ~button;
    }
}

void State::OnAxis(Axis axis, int value) {
    const auto toggle = [&](const auto button) {
        if (value > 0) {
            buttonsState |= button;
        } else {
            buttonsState &= ~button;
        }
    };
    switch (axis) {
    case Axis::TriggerLeft:
        toggle(OrbisPadButtonDataOffset::L2);
        break;
    case Axis::TriggerRight:
        toggle(OrbisPadButtonDataOffset::R2);
        break;
    default:
        break;
    }
    axes[static_cast<int>(axis)] = value;
}

void State::OnTouchpad(int touchIndex, bool isDown, float x, float y) {
    touchpad[touchIndex].state = isDown;
    touchpad[touchIndex].x = static_cast<u16>(x * 1920);
    touchpad[touchIndex].y = static_cast<u16>(y * 941);
}

void State::OnGyro(const float gyro[3]) {
    // GR2FORK: handheld gyro orientation fix. A handheld is held upright, ~90 degrees about X
    // from a flat DualShock, so the game's yaw is physically measured on the roll channel and
    // vice versa; swapping the two fixes tilt/motion aiming. Pitch (X) is shared and untouched.
    if (Config::getGyroSwapYawRoll()) {
        angularVelocity.x = gyro[0]; // pitch (unchanged)
        angularVelocity.y = gyro[2]; // yaw  <- roll
        angularVelocity.z = gyro[1]; // roll <- yaw
    } else {
        angularVelocity.x = gyro[0];
        angularVelocity.y = gyro[1];
        angularVelocity.z = gyro[2];
    }

    // GR2FORK: optional yaw inversion. The yaw rate is always angularVelocity.y here, so one
    // negation works with or without the swap above. OnAccel is untouched - the accelerometer
    // cannot sense rotation about the gravity axis. Auto-enabled by the Steam Deck preset.
    if (Config::getGyroInvertYaw()) {
        angularVelocity.y = -angularVelocity.y;
    }
    // GR2FORK: optional pitch (X axis) and roll inversions, each negating its resolved channel.
    if (Config::getGyroInvertX()) {
        angularVelocity.x = -angularVelocity.x;
    }
    if (Config::getGyroInvertRoll()) {
        angularVelocity.z = -angularVelocity.z;
    }
}

void State::OnAccel(const float accel[3]) {
    // GR2FORK: mirror the OnGyro yaw/roll swap on the accelerometer so the motion frame stays
    // internally consistent; harmless for titles reading only the fused orientation quaternion,
    // correct for any that read raw acceleration directly.
    if (Config::getGyroSwapYawRoll()) {
        acceleration.x = accel[0]; // unchanged
        acceleration.y = accel[2]; // <- Z
        acceleration.z = accel[1]; // <- Y
    } else {
        acceleration.x = accel[0];
        acceleration.y = accel[1];
        acceleration.z = accel[2];
    }
}

GameController::GameController() {
    m_states_num = 0;
    m_last_state = State();
}

void GameController::ReadState(State* state, bool* isConnected, int* connectedCount) {
    std::scoped_lock lock{m_mutex};

    *isConnected = m_connected;
    *connectedCount = m_connected_count;
    *state = GetLastState();
}

int GameController::ReadStates(State* states, int states_num, bool* isConnected,
                               int* connectedCount) {
    std::scoped_lock lock{m_mutex};

    *isConnected = m_connected;
    *connectedCount = m_connected_count;

    int ret_num = 0;

    if (m_connected) {
        if (m_states_num == 0) {
            ret_num = 1;
            states[0] = m_last_state;
        } else {
            for (uint32_t i = 0; i < m_states_num; i++) {
                if (ret_num >= states_num) {
                    break;
                }
                auto index = (m_first_state + i) % MAX_STATES;
                if (!m_private[index].obtained) {
                    m_private[index].obtained = true;

                    states[ret_num++] = m_states[index];
                }
            }
        }
    }

    return ret_num;
}

State GameController::GetLastState() const {
    if (m_states_num == 0) {
        return m_last_state;
    }
    const u32 last = (m_first_state + m_states_num - 1) % MAX_STATES;
    return m_states[last];
}

void GameController::AddState(const State& state) {
    if (m_states_num >= MAX_STATES) {
        m_states_num = MAX_STATES - 1;
        m_first_state = (m_first_state + 1) % MAX_STATES;
    }

    const u32 index = (m_first_state + m_states_num) % MAX_STATES;
    m_states[index] = state;
    m_last_state = state;
    m_private[index].obtained = false;
    m_states_num++;
}

void GameController::CheckButton(int id, OrbisPadButtonDataOffset button, bool is_pressed) {
    std::scoped_lock lock{m_mutex};
    auto state = GetLastState();

    state.time = Libraries::Kernel::sceKernelGetProcessTime();
    state.OnButton(button, is_pressed);

    AddState(state);
}

void GameController::Axis(int id, Input::Axis axis, int value) {
    std::scoped_lock lock{m_mutex};
    auto state = GetLastState();

    state.time = Libraries::Kernel::sceKernelGetProcessTime();
    state.OnAxis(axis, value);

    AddState(state);
}

void GameController::Gyro(int id, const float gyro[3]) {
    std::scoped_lock lock{m_mutex};
    auto state = GetLastState();
    state.time = Libraries::Kernel::sceKernelGetProcessTime();

    // Update the angular velocity (gyro data)
    state.OnGyro(gyro);

    AddState(state);
}
void GameController::Acceleration(int id, const float acceleration[3]) {
    std::scoped_lock lock{m_mutex};
    auto state = GetLastState();
    state.time = Libraries::Kernel::sceKernelGetProcessTime();

    // Update the acceleration values
    state.OnAccel(acceleration);

    AddState(state);
}

void GameController::CalculateOrientation(Libraries::Pad::OrbisFVector3& acceleration,
                                          Libraries::Pad::OrbisFVector3& angularVelocity,
                                          float deltaTime,
                                          Libraries::Pad::OrbisFQuaternion& lastOrientation,
                                          Libraries::Pad::OrbisFQuaternion& orientation) {
    // GR2FORK: clamp the integration step. The first poll and frame stalls produce a huge dt; in
    // a first-order Euler step qDot*dt then dwarfs q and the attitude snaps to the angular-velocity
    // direction (the ISS "camera points above Delsin" softlock). NaN/non-positive dt also guarded.
    constexpr float kMaxDeltaTime = 0.05f; // 50 ms (~3 frames @ 60 fps)
    if (!(deltaTime > 0.0f)) {
        deltaTime = 0.0f; // NaN or non-positive: integrate no rotation this step
    } else if (deltaTime > kMaxDeltaTime) {
        if (deltaTime > 0.25f) {
            LOG_WARNING(Lib_Pad,
                        "orientation dt abnormally large ({:.3f}s) -> clamped to {:.3f}s; "
                        "stale timestamp or frame stall — attitude would have flipped",
                        deltaTime, kMaxDeltaTime);
        }
        deltaTime = kMaxDeltaTime;
    }

    Libraries::Pad::OrbisFQuaternion q = lastOrientation;
    Libraries::Pad::OrbisFQuaternion ω = {angularVelocity.x, angularVelocity.y, angularVelocity.z,
                                          0.0f};

    Libraries::Pad::OrbisFQuaternion qω = {q.w * ω.x + q.x * ω.w + q.y * ω.z - q.z * ω.y,
                                           q.w * ω.y + q.y * ω.w + q.z * ω.x - q.x * ω.z,
                                           q.w * ω.z + q.z * ω.w + q.x * ω.y - q.y * ω.x,
                                           q.w * ω.w - q.x * ω.x - q.y * ω.y - q.z * ω.z};

    Libraries::Pad::OrbisFQuaternion qDot = {0.5f * qω.x, 0.5f * qω.y, 0.5f * qω.z, 0.5f * qω.w};

    q.x += qDot.x * deltaTime;
    q.y += qDot.y * deltaTime;
    q.z += qDot.z * deltaTime;
    q.w += qDot.w * deltaTime;

    float norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    q.x /= norm;
    q.y /= norm;
    q.z /= norm;
    q.w /= norm;

    orientation.x = q.x;
    orientation.y = q.y;
    orientation.z = q.z;
    orientation.w = q.w;
    LOG_DEBUG(Lib_Pad, "Calculated orientation: {:.2f} {:.2f} {:.2f} {:.2f}", orientation.x,
              orientation.y, orientation.z, orientation.w);
}

void GameController::SetLightBarRGB(u8 r, u8 g, u8 b) {
    if (!m_engine) {
        return;
    }
    std::scoped_lock _{m_mutex};
    m_engine->SetLightBarRGB(r, g, b);
}

void GameController::SetVibration(u8 smallMotor, u8 largeMotor) {
    if (!m_engine) {
        return;
    }
    std::scoped_lock _{m_mutex};
    m_engine->SetVibration(smallMotor, largeMotor);
}

void GameController::SetTouchpadState(int touchIndex, bool touchDown, float x, float y) {
    if (touchIndex < 2) {
        std::scoped_lock lock{m_mutex};
        auto state = GetLastState();

        state.time = Libraries::Kernel::sceKernelGetProcessTime();
        state.OnTouchpad(touchIndex, touchDown, x, y);

        AddState(state);
    }
}

u8 GameController::GetTouchCount() {
    std::scoped_lock lock{m_mutex};
    return m_touch_count;
}

void GameController::SetTouchCount(u8 touchCount) {
    std::scoped_lock lock{m_mutex};
    m_touch_count = touchCount;
}

u8 GameController::GetSecondaryTouchCount() {
    std::scoped_lock lock{m_mutex};
    return m_secondary_touch_count;
}

void GameController::SetSecondaryTouchCount(u8 touchCount) {
    std::scoped_lock lock{m_mutex};
    m_secondary_touch_count = touchCount;
    if (touchCount == 0) {
        m_was_secondary_reset = true;
    }
}

u8 GameController::GetPreviousTouchNum() {
    std::scoped_lock lock{m_mutex};
    return m_previous_touchnum;
}

void GameController::SetPreviousTouchNum(u8 touchNum) {
    std::scoped_lock lock{m_mutex};
    m_previous_touchnum = touchNum;
}

bool GameController::WasSecondaryTouchReset() {
    std::scoped_lock lock{m_mutex};
    return m_was_secondary_reset;
}

void GameController::UnsetSecondaryTouchResetBool() {
    std::scoped_lock lock{m_mutex};
    m_was_secondary_reset = false;
}

void GameController::SetLastOrientation(Libraries::Pad::OrbisFQuaternion& orientation) {
    std::scoped_lock lock{m_mutex};
    m_orientation = orientation;
}

Libraries::Pad::OrbisFQuaternion GameController::GetLastOrientation() {
    std::scoped_lock lock{m_mutex};
    return m_orientation;
}

std::chrono::steady_clock::time_point GameController::GetLastUpdate() {
    std::scoped_lock lock{m_mutex};
    return m_last_update;
}

void GameController::SetLastUpdate(std::chrono::steady_clock::time_point lastUpdate) {
    std::scoped_lock lock{m_mutex};
    m_last_update = lastUpdate;
}

void GameController::SetEngine(std::unique_ptr<Engine> engine) {
    std::scoped_lock _{m_mutex};
    m_engine = std::move(engine);
    if (m_engine) {
        m_engine->Init();
    }
}

Engine* GameController::GetEngine() {
    return m_engine.get();
}

u32 GameController::Poll() {
    if (m_connected) {
        std::scoped_lock lock{m_mutex};
        auto time = Libraries::Kernel::sceKernelGetProcessTime();
        if (m_states_num == 0) {
            auto diff = (time - m_last_state.time) / 1000;
            if (diff >= 100) {
                AddState(GetLastState());
            }
        } else {
            auto index = (m_first_state - 1 + m_states_num) % MAX_STATES;
            auto diff = (time - m_states[index].time) / 1000;
            if (m_private[index].obtained && diff >= 100) {
                AddState(GetLastState());
            }
        }
    }
    return 100;
}

} // namespace Input

namespace GamepadSelect {

int GetDefaultGamepad(SDL_JoystickID* gamepadIDs, int gamepadCount) {
    char GUIDbuf[33];
    if (Config::getDefaultControllerID() != "") {
        for (int i = 0; i < gamepadCount; i++) {
            SDL_GUIDToString(SDL_GetGamepadGUIDForID(gamepadIDs[i]), GUIDbuf, 33);
            std::string currentGUID = std::string(GUIDbuf);
            if (currentGUID == Config::getDefaultControllerID()) {
                return i;
            }
        }
    }
    return -1;
}

int GetIndexfromGUID(SDL_JoystickID* gamepadIDs, int gamepadCount, std::string GUID) {
    char GUIDbuf[33];
    for (int i = 0; i < gamepadCount; i++) {
        SDL_GUIDToString(SDL_GetGamepadGUIDForID(gamepadIDs[i]), GUIDbuf, 33);
        std::string currentGUID = std::string(GUIDbuf);
        if (currentGUID == GUID) {
            return i;
        }
    }
    return -1;
}

std::string GetGUIDString(SDL_JoystickID* gamepadIDs, int index) {
    char GUIDbuf[33];
    SDL_GUIDToString(SDL_GetGamepadGUIDForID(gamepadIDs[index]), GUIDbuf, 33);
    std::string GUID = std::string(GUIDbuf);
    return GUID;
}

std::string GetSelectedGamepad() {
    return SelectedGamepad;
}

void SetSelectedGamepad(std::string GUID) {
    SelectedGamepad = GUID;
}

} // namespace GamepadSelect
