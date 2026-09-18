// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <utility>

#include "common/types.h"
#include "core/emulator_settings.h"
#include "core/libraries/pad/pad.h"
#include "input/controller.h"

namespace Input {

using Libraries::Pad::OrbisPadButtonDataOffset;

void State::OnButton(OrbisPadButtonDataOffset button, bool is_pressed) {
    if (is_pressed) {
        buttonsState |= button;
    } else {
        buttonsState &= ~button;
    }
}

void State::OnAxis(Axis axis, int value, u64 timestamp, bool smooth) {
    const auto index = std::to_underlying(axis);
    axes[index] = axis_smoothing_end_values[index];

    axis_smoothing_start_times[index] = timestamp;
    axis_smoothing_start_values[index] = axes[index];
    axis_smoothing_end_values[index] = value;
    axis_smoothing_flags[index] = smooth;
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
}

void State::OnTouchpad(int touch_index, bool is_down, float x, float y) {
    touchpad[touch_index].state = is_down;
    touchpad[touch_index].x = static_cast<u16>(x * 1920);
    touchpad[touch_index].y = static_cast<u16>(y * 941);
}

void State::OnGyro(const float gyro[3]) {
    // A handheld held upright (Steam Deck, ROG Ally) instead of flat like a DualShock measures
    // the game's yaw on its roll channel and vice versa; the swap puts them back. Each inversion
    // negates its resolved channel, so they compose with the swap.
    const bool swap = EmulatorSettings.IsGyroSwapYawRoll();
    const float yaw = swap ? gyro[2] : gyro[1];
    const float roll = swap ? gyro[1] : gyro[2];
    angularVelocity.x = EmulatorSettings.IsGyroInvertX() ? -gyro[0] : gyro[0];
    angularVelocity.y = EmulatorSettings.IsGyroInvertYaw() ? -yaw : yaw;
    angularVelocity.z = EmulatorSettings.IsGyroInvertRoll() ? -roll : roll;
}

void State::OnAccel(const float accel[3]) {
    // Mirror the yaw/roll swap so the motion frame stays consistent for titles that read the
    // raw acceleration; the accelerometer cannot sense rotation about gravity, so no inversions.
    const bool swap = EmulatorSettings.IsGyroSwapYawRoll();
    acceleration.x = accel[0];
    acceleration.y = swap ? accel[2] : accel[1];
    acceleration.z = swap ? accel[1] : accel[2];
}

void State::UpdateAxisSmoothing(u64 timestamp) {
    for (int i = 0; i < std::to_underlying(Axis::AxisMax); ++i) {
        if (!axis_smoothing_flags[i] || std::abs(axes[i] - axis_smoothing_end_values[i]) < 16) {
            axes[i] = axis_smoothing_end_values[i];
            continue;
        }
        const f32 t = std::clamp(
            (timestamp - axis_smoothing_start_times[i]) / f32{axis_smoothing_time}, 0.f, 1.f);
        axes[i] = s32(axis_smoothing_start_values[i] * (1 - t) + axis_smoothing_end_values[i] * t);
    }
}

} // namespace Input
