// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "core/libraries/avplayer/avplayer.h"
#include "core/libraries/avplayer/avplayer_error.h"
#include "core/libraries/avplayer/avplayer_impl.h"
#include "core/libraries/libs.h"
#include <mutex>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace Libraries::AvPlayer {

// FIX(GR2FORK v6 / v7): tutorial-wins serialization — SIZE-identified, with
// frozen-stream flush on release.
//
// Tutorial overlays collide with a concurrent background video in the
// GpuComm/assembler split (shared bind-state in one cmdbuf), which can't be
// fixed without collapsing the threaded architecture. So while a tutorial is
// live we serialize: it becomes the only stream that receives frames, others
// are suppressed (frozen). The v6 stall-watchdog (avplayer_source.cpp) keeps a
// suppressed stream's decoder from wedging.
//
// v7 reverts identification from source FILENAME back to the FRAME SIZE
// signature. In GR2 the tutorial info videos are uniquely 768x416 (e.g.
// info/ep03_info_06_01.mp4); that distinctive dimension IS the tutorial
// signature. Keying on size rather than a hardcoded twelve-filename list keeps
// the policy working when filenames differ — across game versions / regions
// and other titles in the fork (e.g. GRR) — without maintaining a name table.
// Trade-off vs the v6.3 filename gate: any non-tutorial stream that happens to
// be exactly 768x416 would now be treated as the tutorial. No such look-alike
// is known in GR2's video set; this is the deliberately chosen behavior.
//
// The rest is the corrected v6 form, unchanged:
//   - decide suppression BEFORE consuming (never discard a consumed frame);
//   - drop only the first racing frame of a freshly-claimed tutorial
//     (FIX v6.1/6.2: the first tutorial frame presents from a stale UBO/texture
//     cache at the shared VAddr; by frame 2 the tick key has turned over);
//   - when the active tutorial ends and no tutorial remains, flush every other
//     live stream (all suppressed/frozen by it) at once, so a frozen video
//     can't sit dead and stall GR2's streaming queue into an infinite load.
namespace {

// The GR2 tutorial info videos are uniquely 768x416. That dimension is the
// tutorial signature; a frame measuring exactly this claims tutorial status.
constexpr u32 kTutorialWidth = 768;
constexpr u32 kTutorialHeight = 416;

bool IsTutorialDims(u32 w, u32 h) noexcept {
    return w == kTutorialWidth && h == kTutorialHeight;
}

std::mutex g_tutorial_mutex;
AvPlayerHandle g_tutorial_handle = nullptr; // currently-active (suppressing) tutorial
bool g_tutorial_primed = false;             // first-frame drop done for the active tutorial

// Every live handle (Init..Close), so suppression release can flush the others.
std::unordered_set<AvPlayerHandle> g_live_handles;

void RegisterLiveHandle(AvPlayerHandle h) {
    std::scoped_lock lock{g_tutorial_mutex};
    g_live_handles.insert(h);
}

// Suppress every handle except the one currently designated the tutorial.
bool ShouldSuppress(AvPlayerHandle h) noexcept {
    std::scoped_lock lock{g_tutorial_mutex};
    return g_tutorial_handle != nullptr && g_tutorial_handle != h;
}

// Claim tutorial status when a handle delivers a frame of the tutorial SIZE
// (768x416), and decide whether THIS frame must be dropped.
//
// FIX(GR2FORK v6.1/6.2): the very first tutorial frame races its own
// invalidation — Path D still holds a stale UBO copy and the texture cache a
// stale image at the shared VAddr from the prior stream, so frame 1 presents
// corrupt; by the next frame the tick key has turned over and the cache
// re-copies clean. So drop the first frame after claiming and present from the
// second. FIX(GR2FORK v7): the claim is gated on the just-retrieved frame's
// dimensions matching the 768x416 tutorial signature, so identification is the
// frame size again (no filename table). A non-768x416 frame never claims.
// Returns true if this (first) frame is withheld.
bool ClaimAndShouldDropFirstFrame(AvPlayerHandle h, u32 w, u32 h_px) noexcept {
    std::scoped_lock lock{g_tutorial_mutex};
    if (!IsTutorialDims(w, h_px)) {
        return false;
    }
    if (g_tutorial_handle == nullptr) {
        g_tutorial_handle = h;
        g_tutorial_primed = false;
    }
    if (g_tutorial_handle == h && !g_tutorial_primed) {
        g_tutorial_primed = true; // every subsequent frame is delivered
        return true;              // withhold this first (racing) frame
    }
    return false;
}

// Release the active-tutorial designation if this handle holds it, and return
// the OTHER live handles to flush. When the active tutorial ends, every other
// stream was suppressed (frozen) by it; flushing them lets them resume at once
// rather than sitting wedged and stalling the streaming queue (the infinite-
// load failure). The caller flushes the returned handles AFTER releasing the
// lock, since FlushStreams reaches into the source layer.
std::vector<AvPlayerHandle> ClearTutorialAndCollectFlush(AvPlayerHandle h) {
    std::vector<AvPlayerHandle> to_flush;
    std::scoped_lock lock{g_tutorial_mutex};
    if (g_tutorial_handle == h) {
        g_tutorial_handle = nullptr;
        g_tutorial_primed = false;
        for (auto live : g_live_handles) {
            if (live != h) {
                to_flush.push_back(live);
            }
        }
    }
    return to_flush;
}

// Drop a handle from the live registry (on Close).
void ForgetHandle(AvPlayerHandle h) {
    std::scoped_lock lock{g_tutorial_mutex};
    g_live_handles.erase(h);
}

// Release tutorial designation held by `h` (if any) and flush the streams that
// were suppressed by it. Safe to call on any lifecycle exit; a no-op when `h`
// wasn't the active tutorial.
void ReleaseTutorialAndFlush(AvPlayerHandle h) {
    for (auto fh : ClearTutorialAndCollectFlush(h)) {
        fh->FlushStreams();
    }
}

}  // anonymous namespace


s32 PS4_SYSV_ABI sceAvPlayerAddSource(AvPlayerHandle handle, const char* filename) {
    LOG_TRACE(Lib_AvPlayer, "filename = {}", filename);
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    // FIX(GR2FORK v6): handle reuse releases any prior designation (and flushes
    // the streams it was suppressing). Tutorial identity is now decided at
    // frame time by size (v7), so there is no per-source marking to set here.
    ReleaseTutorialAndFlush(handle);
    return handle->AddSource(filename);
}

s32 PS4_SYSV_ABI sceAvPlayerAddSourceEx(AvPlayerHandle handle, AvPlayerUriType uri_type,
                                        AvPlayerSourceDetails* source_details) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr || uri_type != AvPlayerUriType::Source) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    ReleaseTutorialAndFlush(handle);  // FIX(GR2FORK v6): handle reuse
    const auto path = std::string_view(source_details->uri.name, source_details->uri.length);
    return handle->AddSourceEx(path, source_details->source_type);
}

int PS4_SYSV_ABI sceAvPlayerChangeStream() {
    LOG_ERROR(Lib_AvPlayer, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAvPlayerClose(AvPlayerHandle handle) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    ReleaseTutorialAndFlush(handle);  // FIX(GR2FORK v6): release + flush frozen others
    ForgetHandle(handle);             // drop from the live-handle registry
    delete handle;
    return ORBIS_OK;
}

u64 PS4_SYSV_ABI sceAvPlayerCurrentTime(AvPlayerHandle handle) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return handle->CurrentTime();
}

s32 PS4_SYSV_ABI sceAvPlayerDisableStream(AvPlayerHandle handle, u32 stream_id) {
    LOG_ERROR(Lib_AvPlayer, "(STUBBED) called");
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAvPlayerEnableStream(AvPlayerHandle handle, u32 stream_id) {
    LOG_TRACE(Lib_AvPlayer, "stream_id = {}", stream_id);
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return handle->EnableStream(stream_id);
}

bool PS4_SYSV_ABI sceAvPlayerGetAudioData(AvPlayerHandle handle, AvPlayerFrameInfo* p_info) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr || p_info == nullptr) {
        return false;
    }
    return handle->GetAudioData(*p_info);
}

s32 PS4_SYSV_ABI sceAvPlayerGetStreamInfo(AvPlayerHandle handle, u32 stream_id,
                                          AvPlayerStreamInfo* p_info) {
    LOG_TRACE(Lib_AvPlayer, "stream_id = {}", stream_id);
    if (handle == nullptr || p_info == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return handle->GetStreamInfo(stream_id, *p_info);
}

bool PS4_SYSV_ABI sceAvPlayerGetVideoData(AvPlayerHandle handle, AvPlayerFrameInfo* video_info) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr || video_info == nullptr) {
        return false;
    }
    // FIX(GR2FORK v6): tutorial-wins — decide suppression BEFORE consuming, so
    // we never pop a frame we then throw away. A suppressed handle returns
    // without touching its queue; its decoder keeps running and the v6
    // watchdog prevents it from wedging.
    if (ShouldSuppress(handle)) {
        return false;
    }
    const bool got = handle->GetVideoData(*video_info);
    if (got) {
        // FIX(GR2FORK v7): claim/first-frame-drop keyed on the just-retrieved
        // frame's dimensions matching the 768x416 tutorial signature. A
        // non-tutorial-sized frame is a no-op here.
        if (ClaimAndShouldDropFirstFrame(handle, video_info->details.video.width,
                                         video_info->details.video.height)) {
            return false; // withhold the first (racing) frame; deliver from #2
        }
    }
    // Always deliver a consumed frame (never discard) — that was the v3.3 race.
    // The single first-frame drop above is a deliberate, bounded exception.
    return got;
}

bool PS4_SYSV_ABI sceAvPlayerGetVideoDataEx(AvPlayerHandle handle,
                                            AvPlayerFrameInfoEx* video_info) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr || video_info == nullptr) {
        return false;
    }
    // FIX(GR2FORK v6): see sceAvPlayerGetVideoData — suppress before consume,
    // drop the first (racing) tutorial frame, never discard a consumed frame.
    if (ShouldSuppress(handle)) {
        return false;
    }
    const bool got = handle->GetVideoData(*video_info);
    if (got) {
        // FIX(GR2FORK v7): see sceAvPlayerGetVideoData — claim by frame size
        // (768x416), drop only the first racing frame.
        if (ClaimAndShouldDropFirstFrame(handle, video_info->details.video.width,
                                         video_info->details.video.height)) {
            return false;
        }
    }
    return got;
}

AvPlayerHandle PS4_SYSV_ABI sceAvPlayerInit(AvPlayerInitData* data) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (data == nullptr) {
        return nullptr;
    }

    if (data->memory_replacement.allocate == nullptr ||
        data->memory_replacement.allocate_texture == nullptr ||
        data->memory_replacement.deallocate == nullptr ||
        data->memory_replacement.deallocate_texture == nullptr) {
        LOG_ERROR(Lib_AvPlayer, "All allocators are required for AvPlayer Initialisation.");
        return nullptr;
    }

    auto* handle = new AvPlayer(*data);
    RegisterLiveHandle(handle);  // FIX(GR2FORK v6.3): track for suppression-release flush
    return handle;
}

s32 PS4_SYSV_ABI sceAvPlayerInitEx(const AvPlayerInitDataEx* p_data, AvPlayerHandle* p_player) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (p_data == nullptr || p_player == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }

    if (p_data->memory_replacement.allocate == nullptr ||
        p_data->memory_replacement.allocate_texture == nullptr ||
        p_data->memory_replacement.deallocate == nullptr ||
        p_data->memory_replacement.deallocate_texture == nullptr) {
        LOG_ERROR(Lib_AvPlayer, "All allocators are required for AvPlayer Initialisation.");
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }

    AvPlayerInitData data = {};
    data.memory_replacement = p_data->memory_replacement;
    data.file_replacement = p_data->file_replacement;
    data.event_replacement = p_data->event_replacement;
    data.default_language = p_data->default_language;
    data.num_output_video_framebuffers = p_data->num_output_video_framebuffers;
    data.auto_start = p_data->auto_start;

    auto* handle = new AvPlayer(data);
    RegisterLiveHandle(handle);  // FIX(GR2FORK v6.3): track for suppression-release flush
    *p_player = handle;
    return ORBIS_OK;
}

bool PS4_SYSV_ABI sceAvPlayerIsActive(AvPlayerHandle handle) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr) {
        return false;
    }
    return handle->IsActive();
}

s32 PS4_SYSV_ABI sceAvPlayerJumpToTime(AvPlayerHandle handle, uint64_t time) {
    LOG_ERROR(Lib_AvPlayer, "(STUBBED) called, time (msec) = {}", time);
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAvPlayerPause(AvPlayerHandle handle) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    ReleaseTutorialAndFlush(handle);  // FIX(GR2FORK v6/v6.3): paused tutorial unfreezes + flushes others
    return handle->Pause();
}

s32 PS4_SYSV_ABI sceAvPlayerPostInit(AvPlayerHandle handle, AvPlayerPostInitData* data) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr || data == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return handle->PostInit(*data);
}

s32 PS4_SYSV_ABI sceAvPlayerPrintf(const char* format, ...) {
    LOG_ERROR(Lib_AvPlayer, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAvPlayerResume(AvPlayerHandle handle) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return handle->Resume();
}

s32 PS4_SYSV_ABI sceAvPlayerSetAvSyncMode(AvPlayerHandle handle, AvPlayerAvSyncMode sync_mode) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return handle->SetAvSyncMode(sync_mode);
}

s32 PS4_SYSV_ABI sceAvPlayerSetLogCallback(AvPlayerLogCallback log_cb, void* user_data) {
    LOG_ERROR(Lib_AvPlayer, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAvPlayerSetLooping(AvPlayerHandle handle, bool loop_flag) {
    LOG_TRACE(Lib_AvPlayer, "called, looping = {}", loop_flag);
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    if (!handle->SetLooping(loop_flag)) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAvPlayerSetTrickSpeed(AvPlayerHandle handle, s32 trick_speed) {
    LOG_ERROR(Lib_AvPlayer, "(STUBBED) called speed = {}", trick_speed);
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceAvPlayerStart(AvPlayerHandle handle) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return handle->Start();
}

s32 PS4_SYSV_ABI sceAvPlayerStop(AvPlayerHandle handle) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    ReleaseTutorialAndFlush(handle);  // FIX(GR2FORK v6/v6.3): tutorial stop flushes frozen others
    return handle->Stop();
}

s32 PS4_SYSV_ABI sceAvPlayerStreamCount(AvPlayerHandle handle) {
    LOG_TRACE(Lib_AvPlayer, "called");
    if (handle == nullptr) {
        return ORBIS_AVPLAYER_ERROR_INVALID_PARAMS;
    }
    return handle->GetStreamCount();
}

s32 PS4_SYSV_ABI sceAvPlayerVprintf(const char* format, va_list args) {
    LOG_ERROR(Lib_AvPlayer, "(STUBBED) called");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("KMcEa+rHsIo", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerAddSource);
    LIB_FUNCTION("x8uvuFOPZhU", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerAddSourceEx);
    LIB_FUNCTION("buMCiJftcfw", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerChangeStream);
    LIB_FUNCTION("NkJwDzKmIlw", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerClose);
    LIB_FUNCTION("wwM99gjFf1Y", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerCurrentTime);
    LIB_FUNCTION("BOVKAzRmuTQ", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerDisableStream);
    LIB_FUNCTION("ODJK2sn9w4A", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerEnableStream);
    LIB_FUNCTION("Wnp1OVcrZgk", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerGetAudioData);
    LIB_FUNCTION("d8FcbzfAdQw", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerGetStreamInfo);
    LIB_FUNCTION("o3+RWnHViSg", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerGetVideoData);
    LIB_FUNCTION("JdksQu8pNdQ", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerGetVideoDataEx);
    LIB_FUNCTION("aS66RI0gGgo", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerInit);
    LIB_FUNCTION("o9eWRkSL+M4", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerInitEx);
    LIB_FUNCTION("UbQoYawOsfY", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerIsActive);
    LIB_FUNCTION("XC9wM+xULz8", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerJumpToTime);
    LIB_FUNCTION("9y5v+fGN4Wk", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerPause);
    LIB_FUNCTION("HD1YKVU26-M", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerPostInit);
    // LIB_FUNCTION("agig-iDRrTE", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerPrintf);
    LIB_FUNCTION("w5moABNwnRY", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerResume);
    LIB_FUNCTION("k-q+xOxdc3E", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerSetAvSyncMode);
    LIB_FUNCTION("eBTreZ84JFY", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerSetLogCallback);
    LIB_FUNCTION("OVths0xGfho", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerSetLooping);
    LIB_FUNCTION("av8Z++94rs0", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerSetTrickSpeed);
    LIB_FUNCTION("ET4Gr-Uu07s", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerStart);
    LIB_FUNCTION("ZC17w3vB5Lo", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerStop);
    LIB_FUNCTION("hdTyRzCXQeQ", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerStreamCount);
    LIB_FUNCTION("yN7Jhuv8g24", "libSceAvPlayer", 1, "libSceAvPlayer", sceAvPlayerVprintf);
};

} // namespace Libraries::AvPlayer
