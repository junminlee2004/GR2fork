// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

namespace Common {

// GR2FORK: photo-capture scope for the pipeline-compile sync budget.
//
// GR2 renders the photo scene into a 1024x1024 target moments before sceJpegEncEncode reads it
// back. A draw whose pipeline is still compiling gets skipped, so that target stays empty and
// the saved photo is a black square. A low gameplaySyncBudgetMs is good for gameplay (short
// waits mean fewer chances for the device to time out) but makes that skip likely, so the wait
// is raised to PhotoCaptureSyncBudgetMs for the capture only, then drops back to the configured
// value.
//
// Opened at sceJpegEncCreate, closed when the photo encode returns (also on sceJpegEncDelete,
// so a create that never encodes cannot pin the budget high).
//
// This scope alone is NOT enough: the game renders the photo before it creates the encoder, so
// on the first capture of a session the photo draw hits a cold pipeline outside this window and
// the photo saves black (later captures survive only because the pipeline is cached by then).
// The primary trigger is therefore IsPhotoRenderTarget() at the pipeline lookup in
// vk_pipeline_cache; this scope just covers whatever still compiles during the encode itself.
inline constexpr int PhotoCaptureSyncBudgetMs = 700;

inline std::atomic<bool>& PhotoCaptureActiveFlag() {
    static std::atomic<bool> active{false};
    return active;
}

inline void BeginPhotoCapture() {
    PhotoCaptureActiveFlag().store(true, std::memory_order_release);
}

inline void EndPhotoCapture() {
    PhotoCaptureActiveFlag().store(false, std::memory_order_release);
}

inline bool IsPhotoCaptureActive() {
    return PhotoCaptureActiveFlag().load(std::memory_order_acquire);
}

} // namespace Common
