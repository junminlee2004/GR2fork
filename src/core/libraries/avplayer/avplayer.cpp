// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "core/libraries/avplayer/avplayer.h"
#include "core/libraries/avplayer/avplayer_error.h"
#include "core/libraries/avplayer/avplayer_impl.h"
#include "core/libraries/libs.h"
#include <array>
#include <mutex>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace Libraries::AvPlayer {

// FIX(GR2FORK v6 / v6.3): tutorial-wins serialization — filename-identified,
// with frozen-stream flush on release.
//
// Tutorial overlays collide with a concurrent background video in the
// GpuComm/assembler split (shared bind-state in one cmdbuf), which can't be
// fixed without collapsing the threaded architecture. So while a tutorial is
// live we serialize: it becomes the only stream that receives frames, others
// are suppressed (frozen). The v6 stall-watchdog (avplayer_source.cpp) keeps a
// suppressed stream's decoder from wedging.
//
// v6.3 changes two things:
//   1. Identification is by source FILENAME, not the 768x416 frame size. The
//      size matched any like-sized cutscene/preview and would wrongly suppress
//      legitimate streams; the twelve info/ tutorial files are matched exactly.
//   2. When the active tutorial ends and no tutorial remains, every other live
//      stream (all of which were suppressed/frozen) is flushed immediately, so
//      a frozen video can't sit dead and stall GR2's streaming queue into an
//      infinite load.
//
// The gate is otherwise the corrected v6 form: decide suppression BEFORE
// consuming (never discard a consumed frame), and drop only the first racing
// frame of a freshly-claimed tutorial.
namespace {

// GR2FORK: master switch for the tutorial-wins serialization described above.
// DISABLED. With this false, no AvPlayer stream is ever suppressed/frozen and
// no first frame is ever dropped — every concurrent stream receives frames
// normally (stock shadPS4 behavior). The filename/handle bookkeeping below is
// still compiled and called but is inert at runtime: because the claim path is
// gated off, g_tutorial_handle is never set, so ShouldSuppress() is always
// false and ClearTutorialAndCollectFlush() never collects anyone to flush.
// Flip to true to restore the full v6.3 tutorial-wins behavior.
constexpr bool kTutorialWinsEnabled = false;

// The GR2 tutorial info videos, by exact source filename. These twelve files
// (under the game's info/ folder) ARE the tutorial overlays; matching the
// source path against them is definitive.
constexpr std::array<std::string_view, 12> kTutorialVideoFiles{{
    "ep02_info_02.mp4", "ep02_info_04.mp4", "ep02_info_05.mp4", "ep02_info_06.mp4",
    "ep02_info_07.mp4", "ep02_info_08.mp4", "ep02_info_09.mp4", "ep02_info_13.mp4",
    "ep03_info_02_01.mp4", "ep03_info_03_01.mp4", "ep03_info_04_01.mp4", "ep03_info_06_01.mp4",
}};

bool IsTutorialFilename(std::string_view path) noexcept {
    // Substring match handles full paths, e.g. /app0/.../info/ep03_info_06_01.mp4.
    for (const auto name : kTutorialVideoFiles) {
        if (path.find(name) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

std::mutex g_tutorial_mutex;
AvPlayerHandle g_tutorial_handle = nullptr; // currently-active (suppressing) tutorial
bool g_tutorial_primed = false;             // first-frame drop done for the active tutorial

// Handles whose current source is one of the tutorial files above.
std::unordered_set<AvPlayerHandle> g_tutorial_sources;
// Every live handle (Init..Close), so suppression release can flush the others.
std::unordered_set<AvPlayerHandle> g_live_handles;

void RegisterLiveHandle(AvPlayerHandle h) {
    std::scoped_lock lock{g_tutorial_mutex};
    g_live_handles.insert(h);
}

// Mark/unmark a handle's source as a tutorial from its filename. A handle being
// repurposed for a non-tutorial source loses tutorial status.
void SetTutorialSource(AvPlayerHandle h, bool is_tutorial) {
    std::scoped_lock lock{g_tutorial_mutex};
    if (is_tutorial) {
        g_tutorial_sources.insert(h);
    } else {
        g_tutorial_sources.erase(h);
    }
}

// Suppress every handle except the one currently designated the tutorial.
bool ShouldSuppress(AvPlayerHandle h) noexcept {
    if constexpr (kTutorialWinsEnabled) {
        std::scoped_lock lock{g_tutorial_mutex};
        return g_tutorial_handle != nullptr && g_tutorial_handle != h;
    } else {
        (void)h;
        return false; // tutorial-wins disabled: never suppress any stream
    }
}

// Claim tutorial status when a confirmed tutorial-source handle delivers a
// frame, and decide whether THIS frame must be dropped.
//
// FIX(GR2FORK v6.1/6.2): the very first tutorial frame races its own
// invalidation — Path D still holds a stale UBO copy and the texture cache a
// stale image at the shared VAddr from the prior stream, so frame 1 presents
// corrupt; by the next frame the tick key has turned over and the cache
// re-copies clean. So drop the first frame after claiming and present from the
// second. FIX(GR2FORK v6.3): the claim requires the handle's source to be a
// known tutorial file (set at AddSource), so a like-sized clip can never be
// mistaken for the tutorial. Returns true if this (first) frame is withheld.
bool ClaimAndShouldDropFirstFrame(AvPlayerHandle h) noexcept {
    if constexpr (kTutorialWinsEnabled) {
        std::scoped_lock lock{g_tutorial_mutex};
        if (!g_tutorial_sources.contains(h)) {
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
    } else {
        (void)h;
        return false; // tutorial-wins disabled: never claim, never drop a frame
    }
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

// Drop a handle from both registries (on Close).
void ForgetHandle(AvPlayerHandle h) {
    std::scoped_lock lock{g_tutorial_mutex};
    g_tutorial_sources.erase(h);
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
    // FIX(GR2FORK v6/v6.3): handle reuse releases any prior designation (and
    // flushes the streams it was suppressing), then re-evaluate this handle's
    // tutorial status from the new source filename.
    ReleaseTutorialAndFlush(handle);
    SetTutorialSource(handle, filename != nullptr && IsTutorialFilename(filename));
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
    SetTutorialSource(handle, IsTutorialFilename(path));  // FIX(GR2FORK v6.3)
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
    ReleaseTutorialAndFlush(handle);  // FIX(GR2FORK v6/v6.3): release + flush frozen others
    ForgetHandle(handle);             // drop from live/tutorial-source registries
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
        // FIX(GR2FORK v6.3): claim/first-frame-drop keyed on the handle's
        // tutorial-source identity (set from the filename at AddSource), not
        // the frame size. ClaimAndShouldDropFirstFrame is a no-op for any
        // non-tutorial handle.
        if (ClaimAndShouldDropFirstFrame(handle)) {
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
        // FIX(GR2FORK v6.3): see sceAvPlayerGetVideoData — claim by source
        // identity, drop only the first racing frame.
        if (ClaimAndShouldDropFirstFrame(handle)) {
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
