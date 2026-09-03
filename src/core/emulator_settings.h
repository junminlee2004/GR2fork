// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream> // Windows static guest red-zone protection
#include <sstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "common/types.h"
#include "core/cpu_patches.h" // Windows static guest red-zone protection

#define EmulatorSettings (*EmulatorSettingsImpl::GetInstance())

enum HideCursorState : int {
    Never,
    Idle,
    Always,
};

enum UsbBackendType : int {
    Real,
    SkylandersPortal,
    InfinityBase,
    DimensionsToypad,
};

enum AdaptiveSkipCachesMode : int {
    SkipCachesDisabled = 0,
    SkipCachesAdaptive = 1,
    // All consumer caches pinned on at boot; the controller, verify tripwire,
    // telemetry and timer sampling never run.
    SkipCachesForced = 2,
    SkipCachesValidateOnly = 3,
};

enum GpuReadbacksMode : int {
    Disabled,
    Relaxed,
    Precise,
};

// Guest read faults on GPU-written memory. OffloadFull moves the fence wait
// onto the faulting guest thread. OffloadBounded caps that wait and hands the
// write-back to the priority waiter on timeout, so a title whose GPU work
// waits on writes the faulting thread makes after the read cannot deadlock.
enum GpuReadbackOffloadMode : int {
    OffloadDisabled,
    OffloadBounded,
    OffloadFull,
};

// Windows static guest red-zone protection
NLOHMANN_JSON_SERIALIZE_ENUM(WindowsGuestRedZoneProtectionMode,
                             {{WindowsGuestRedZoneProtectionMode::Disabled, "Disabled"},
                              {WindowsGuestRedZoneProtectionMode::StaticPatching,
                               "StaticPatching"}})

inline std::ostream& operator<<(std::ostream& output, WindowsGuestRedZoneProtectionMode mode) {
    return output << nlohmann::json(mode).get<std::string>();
}

enum class ConfigMode {
    Default,
    Global,
    Clean,
};

enum AudioBackend : int {
    SDL,
    OpenAL,
    // Add more backends as needed
};

enum OpenALHrtfMode : int {
    HrtfAuto, // Let OpenAL Soft decide (on for headphone-like stereo outputs)
    HrtfOn,   // Force HRTF binaural rendering
    HrtfOff,  // Never use HRTF
};

enum OpenALOutputMode : int {
    OutputAuto,       // Let OpenAL Soft negotiate with the device
    OutputStereo,     // Force stereo output
    OutputQuad,       // Force quadraphonic output
    OutputSurround51, // Force 5.1 surround output
    OutputSurround71, // Force 7.1 surround output
};

template <typename T>
struct Setting {
    T default_value{};
    T value{};
    std::optional<T> game_specific_value{};

    Setting() = default;
    // Single-argument ctor: initialises both default_value and value so
    // that CleanMode can always recover the intended factory default.
    /*implicit*/ Setting(T init) : default_value(std::move(init)), value(default_value) {}

    /// Return the active value under the given mode.
    T get(ConfigMode mode = ConfigMode::Default) const {
        switch (mode) {
        case ConfigMode::Default:
            return game_specific_value.value_or(value);
        case ConfigMode::Global:
            return value;
        case ConfigMode::Clean:
            return default_value;
        }
        return value;
    }

    /// Write v to the base layer.
    /// Set proper value as base or game_specific
    void set(const T& v, bool game_specific = false) {
        if (game_specific) {
            game_specific_value = v;
        } else {
            value = v;
        }
    }

    /// Discard the game-specific override; subsequent get(Default) will
    /// fall back to the base value.
    void reset_game_specific() {
        game_specific_value = std::nullopt;
    }
};

template <typename T>
void to_json(nlohmann::json& j, const Setting<T>& s) {
    j = s.value;
}

template <typename T>
void from_json(const nlohmann::json& j, Setting<T>& s) {
    s.value = j.get<T>();
}

struct OverrideItem {
    const char* key;
    std::function<void(void* group_ptr, const nlohmann::json& entry,
                       std::vector<std::string>& changed)>
        apply;
    /// Return the value that should be written to the per-game config file.
    /// Falls back to base value if no game-specific override is set.
    std::function<nlohmann::json(const void* group_ptr)> get_for_save;

    /// Clear game_specific_value for this field.
    std::function<void(void* group_ptr)> reset_game_specific;
};

template <typename Struct, typename T>
inline OverrideItem make_override(const char* key, Setting<T> Struct::* member) {
    return OverrideItem{
        key,
        [member, key](void* base, const nlohmann::json& entry, std::vector<std::string>& changed) {
            Struct* obj = reinterpret_cast<Struct*>(base);
            Setting<T>& dst = obj->*member;
            try {
                T newValue = entry.get<T>();
                if (dst.value != newValue) {
                    std::ostringstream oss;
                    oss << key << " ( " << dst.value << " -> " << newValue << " )";
                    changed.push_back(oss.str());
                }
                dst.game_specific_value = newValue;
            } catch (const std::exception& e) {
                LOG_ERROR(Config, "[make_override] error parsing {}: {}", key, e.what());
                LOG_ERROR(Config, "[make_override] Entry was: {}", entry.dump());
                LOG_ERROR(Config, "[make_override] Type name: {}", entry.type_name());
            }
        },

        // --- get_for_save -------------------------------------------
        // Returns game_specific_value when present, otherwise base value.
        // This means a freshly-opened game-specific dialog still shows
        // useful (current-global) values rather than empty entries.
        [member](const void* base) -> nlohmann::json {
            const Struct* obj = reinterpret_cast<const Struct*>(base);
            const Setting<T>& src = obj->*member;
            return nlohmann::json(src.game_specific_value.value_or(src.value));
        },

        // --- reset_game_specific ------------------------------------
        [member](void* base) {
            Struct* obj = reinterpret_cast<Struct*>(base);
            (obj->*member).reset_game_specific();
        }};
}

// -------------------------------
// Support types
// -------------------------------
struct GameInstallDir {
    std::filesystem::path path;
    bool enabled;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GameInstallDir, path, enabled)

// -------------------------------
// General settings
// -------------------------------
struct GeneralSettings {
    Setting<std::vector<GameInstallDir>> install_dirs;
    Setting<std::filesystem::path> addon_install_dir;
    Setting<std::filesystem::path> home_dir;
    Setting<std::filesystem::path> sys_modules_dir;
    Setting<std::filesystem::path> font_dir;

    Setting<int> volume_slider{100};
    Setting<bool> neo_mode{false};
    Setting<bool> dev_kit_mode{false};
    Setting<int> extra_dmem_in_mbytes{0};
    Setting<int> extra_fmem_in_mbytes{0};
    Setting<bool> shad_net_enabled{false};
    Setting<bool> trophy_popup_disabled{false};
    Setting<double> trophy_notification_duration{6.0};
    Setting<std::string> trophy_notification_side{"right"};
    Setting<bool> show_splash{false};
    Setting<bool> connected_to_network{false};
    Setting<bool> discord_rpc_enabled{false};
    Setting<bool> show_fps_counter{false};
    Setting<int> console_language{1};
    Setting<int> big_picture_scale{1000};
    Setting<std::string> shadnet_server{"srv.shadps4.net:31313"};
    Setting<std::string> shadnet_webapi_server{"http://srv.shadps4.net:31315"};
    Setting<std::string> signaling_info{};
    Setting<bool> enable_upnp{true};

    // return a vector of override descriptors (runtime, but tiny)
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<GeneralSettings>("volume_slider", &GeneralSettings::volume_slider),
            make_override<GeneralSettings>("neo_mode", &GeneralSettings::neo_mode),
            make_override<GeneralSettings>("dev_kit_mode", &GeneralSettings::dev_kit_mode),
            make_override<GeneralSettings>("extra_dmem_in_mbytes",
                                           &GeneralSettings::extra_dmem_in_mbytes),
            make_override<GeneralSettings>("extra_fmem_in_mbytes",
                                           &GeneralSettings::extra_fmem_in_mbytes),
            make_override<GeneralSettings>("shad_net_enabled", &GeneralSettings::shad_net_enabled),
            make_override<GeneralSettings>("trophy_popup_disabled",
                                           &GeneralSettings::trophy_popup_disabled),
            make_override<GeneralSettings>("trophy_notification_duration",
                                           &GeneralSettings::trophy_notification_duration),
            make_override<GeneralSettings>("show_splash", &GeneralSettings::show_splash),
            make_override<GeneralSettings>("trophy_notification_side",
                                           &GeneralSettings::trophy_notification_side),
            make_override<GeneralSettings>("connected_to_network",
                                           &GeneralSettings::connected_to_network),
            make_override<GeneralSettings>("console_language", &GeneralSettings::console_language),
            make_override<GeneralSettings>("shadnet_server", &GeneralSettings::shadnet_server),
            make_override<GeneralSettings>("shadnet_webapi_server",
                                           &GeneralSettings::shadnet_webapi_server),
            make_override<GeneralSettings>("signaling_info", &GeneralSettings::signaling_info),
            make_override<GeneralSettings>("enable_upnp", &GeneralSettings::enable_upnp)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GeneralSettings, install_dirs, addon_install_dir, home_dir,
                                   sys_modules_dir, font_dir, volume_slider, neo_mode, dev_kit_mode,
                                   extra_dmem_in_mbytes, extra_fmem_in_mbytes, shad_net_enabled,
                                   trophy_popup_disabled, trophy_notification_duration, show_splash,
                                   trophy_notification_side, connected_to_network,
                                   discord_rpc_enabled, show_fps_counter, console_language,
                                   big_picture_scale, shadnet_server, shadnet_webapi_server,
                                   signaling_info, enable_upnp)

// -------------------------------
// Log settings
// -------------------------------
struct LogSettings {
    Setting<bool> append{false}; // specific
    Setting<bool> enable{true};  // specific
    Setting<std::string> filter{""};
    Setting<std::string> flush_level{""};
    Setting<u32> max_skip_duration{5'000};
    Setting<bool> separate{false}; // specific
    Setting<unsigned long long> size_limit{100_MB};
    Setting<bool> skip_duplicate{true};
    Setting<bool> sync{true};
#ifdef _WIN32
    Setting<std::string> type{"wincolor"};
#endif

    // return a vector of override descriptors (runtime, but tiny)
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<LogSettings>("append", &LogSettings::append),
            make_override<LogSettings>("enable", &LogSettings::enable),
            make_override<LogSettings>("filter", &LogSettings::filter),
            make_override<LogSettings>("flush_level", &LogSettings::flush_level),
            make_override<LogSettings>("max_skip_duration", &LogSettings::max_skip_duration),
            make_override<LogSettings>("separate", &LogSettings::separate),
            make_override<LogSettings>("size_limit", &LogSettings::size_limit),
            make_override<LogSettings>("skip_duplicate", &LogSettings::skip_duplicate),
            make_override<LogSettings>("sync", &LogSettings::sync),
#ifdef _WIN32
            make_override<LogSettings>("type", &LogSettings::type),
#endif
        };
    }
};
#ifdef _WIN32
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LogSettings, append, enable, filter, flush_level,
                                   max_skip_duration, separate, size_limit, skip_duplicate, sync,
                                   type)
#else
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LogSettings, append, enable, filter, flush_level,
                                   max_skip_duration, separate, size_limit, skip_duplicate, sync)
#endif

// -------------------------------
// Debug settings
// -------------------------------
struct DebugSettings {
    Setting<bool> debug_dump{false};         // specific
    Setting<bool> shader_collect{false};     // specific
    Setting<std::string> config_version{""}; // specific

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<DebugSettings>("debug_dump", &DebugSettings::debug_dump),
            make_override<DebugSettings>("shader_collect", &DebugSettings::shader_collect)};
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DebugSettings, debug_dump, shader_collect, config_version)

// -------------------------------
// Input settings
// -------------------------------

struct InputSettings {
    Setting<int> cursor_state{HideCursorState::Idle};      // specific
    Setting<int> cursor_hide_timeout{5};                   // specific
    Setting<int> usb_device_backend{UsbBackendType::Real}; // specific
    Setting<bool> use_special_pad{false};
    Setting<int> special_pad_class{1};
    Setting<bool> motion_controls_enabled{true}; // specific
    Setting<bool> use_unified_input_config{true};
    Setting<std::string> default_controller_id{""};
    Setting<bool> background_controller_input{false}; // specific
    Setting<bool> ime_accessibility_enabled{false};   // specific
    Setting<bool> ime_url_mail_short_panel{false};    // specific
    Setting<bool> is_circle_enter{false};             // specific
    Setting<s32> camera_id{-1};
    Setting<bool> use_mice_as_mice{false};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<InputSettings>("cursor_state", &InputSettings::cursor_state),
            make_override<InputSettings>("cursor_hide_timeout",
                                         &InputSettings::cursor_hide_timeout),
            make_override<InputSettings>("usb_device_backend", &InputSettings::usb_device_backend),
            make_override<InputSettings>("motion_controls_enabled",
                                         &InputSettings::motion_controls_enabled),
            make_override<InputSettings>("background_controller_input",
                                         &InputSettings::background_controller_input),
            make_override<InputSettings>("ime_accessibility_enabled",
                                         &InputSettings::ime_accessibility_enabled),
            make_override<InputSettings>("ime_url_mail_short_panel",
                                         &InputSettings::ime_url_mail_short_panel),
            make_override<InputSettings>("is_circle_enter", &InputSettings::is_circle_enter),
            make_override<InputSettings>("camera_id", &InputSettings::camera_id),
            make_override<InputSettings>("use_mice_as_mice", &InputSettings::use_mice_as_mice)};
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InputSettings, cursor_state, cursor_hide_timeout,
                                   usb_device_backend, use_special_pad, special_pad_class,
                                   motion_controls_enabled, use_unified_input_config,
                                   default_controller_id, background_controller_input,
                                   ime_accessibility_enabled, ime_url_mail_short_panel, camera_id,
                                   is_circle_enter, use_mice_as_mice)
// -------------------------------
// Audio settings
// -------------------------------
struct AudioSettings {
    Setting<u32> audio_backend{AudioBackend::SDL};
    Setting<std::string> sdl_mic_device{"Default Device"};
    Setting<std::string> sdl_main_output_device{"Default Device"};
    Setting<std::string> sdl_padSpk_output_device{"Default Device"};
    Setting<std::string> openal_mic_device{"Default Device"};
    Setting<std::string> openal_main_output_device{"Default Device"};
    Setting<std::string> openal_padSpk_output_device{"Default Device"};
    Setting<u32> openal_hrtf{OpenALHrtfMode::HrtfAuto};
    Setting<u32> openal_output_mode{OpenALOutputMode::OutputAuto};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<AudioSettings>("audio_backend", &AudioSettings::audio_backend),
            make_override<AudioSettings>("sdl_mic_device", &AudioSettings::sdl_mic_device),
            make_override<AudioSettings>("sdl_main_output_device",
                                         &AudioSettings::sdl_main_output_device),
            make_override<AudioSettings>("sdl_padSpk_output_device",
                                         &AudioSettings::sdl_padSpk_output_device),
            make_override<AudioSettings>("openal_mic_device", &AudioSettings::openal_mic_device),
            make_override<AudioSettings>("openal_main_output_device",
                                         &AudioSettings::openal_main_output_device),
            make_override<AudioSettings>("openal_padSpk_output_device",
                                         &AudioSettings::openal_padSpk_output_device),
            make_override<AudioSettings>("openal_hrtf", &AudioSettings::openal_hrtf),
            make_override<AudioSettings>("openal_output_mode", &AudioSettings::openal_output_mode)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AudioSettings, audio_backend, sdl_mic_device,
                                   sdl_main_output_device, sdl_padSpk_output_device,
                                   openal_mic_device, openal_main_output_device,
                                   openal_padSpk_output_device, openal_hrtf, openal_output_mode)

// Windows static guest red-zone protection
struct WindowsGuestRedZoneProtectionSettings {
    Setting<WindowsGuestRedZoneProtectionMode> windows_guest_red_zone_protection_mode{
        WindowsGuestRedZoneProtectionMode::Disabled};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{make_override<WindowsGuestRedZoneProtectionSettings>(
            "windows_guest_red_zone_protection_mode",
            &WindowsGuestRedZoneProtectionSettings::windows_guest_red_zone_protection_mode)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WindowsGuestRedZoneProtectionSettings,
                                   windows_guest_red_zone_protection_mode)

// -------------------------------
// GPU settings
// -------------------------------
struct GPUSettings {
    Setting<u32> window_width{1280};
    Setting<u32> window_height{720};
    Setting<u32> internal_screen_width{1280};
    Setting<u32> internal_screen_height{720};
    Setting<bool> null_gpu{false};
    Setting<bool> copy_gpu_buffers{false};
    Setting<u32> readbacks_mode{GpuReadbacksMode::Disabled};
    Setting<bool> readback_linear_images_enabled{false};
    Setting<u32> adaptive_skipcaches_mode{AdaptiveSkipCachesMode::SkipCachesDisabled};
    // Size of the uniform stream ring in MiB. The ring blocks the GPU command
    // thread whenever it wraps, until the GPU drains the previous lap, so a
    // larger ring trades host memory for fewer stalls. 0 keeps the default.
    Setting<u32> stream_buffer_size_mb{64};
    // Widen each guest readback to cover every GPU-modified range in the
    // affected buffer instead of a fixed window. The same bytes are copied,
    // but the whole buffer is serviced by one GPU drain rather than one per
    // window, which is where the cost of readbacks actually is.
    Setting<bool> readback_batching_enabled{false};
    Setting<u32> readback_offload_mode{GpuReadbackOffloadMode::OffloadDisabled};
    Setting<bool> stream_buffer_prefer_host{false};
    // Phase-1 instrumentation mode; 0 keeps the hot paths byte-identical.
    Setting<u32> stream_upload_mirror_mode{0};
    // Widens a guest write fault's dirty marking to this power-of-two block
    // size when the block holds no GPU-modified pages (page-exact semantics
    // otherwise). Streaming writers then fault once per block instead of once
    // per 4KB page, cutting fault and protection-call volume; the extra pages
    // only ever re-upload bytes the guest already owns. 0 keeps page-exact
    // faults.
    Setting<u32> fault_widen_bytes{0};
    // While a deferred operation waits out its GPU tick, attempt the pending
    // pop (a lock plus a fence-query ioctl) once per this many draw-rate
    // polls instead of every draw. 0 polls every call.
    Setting<u32> pending_pop_throttle{0};
    // Stream copy lane mode. 0 disables it (copies stay inline on the GPU
    // command thread). 1 runs the unsafe fast path: no foreign-producer
    // refusal and no unmap push windows - only for titles that never unmap
    // mid-play. 2 runs the hardened path, safe everywhere. Both modes drain
    // through two worker threads, fenced before every submit.
    Setting<u32> stream_copy_workers{0};
    // Skip the eager FindBuffer in binding pass 1 for read-only descriptors
    // small enough for the stream path, which never dereferences the id.
    // DMA-using stages keep the eager call (BDA page-table registration).
    Setting<bool> stream_findbuffer_elide{false};
    // Resolves shader permutations through an address-masked specialization
    // fingerprint: a hit skips the StageSpecialization rebuild and the deep
    // permutation compares entirely.
    Setting<bool> spec_fp_cache{false};
    // Skips the five dynamic-state updaters and their commit when the stamped
    // graphics registers, the pipeline and the dirty-bit re-arm generation all
    // match the previous draw's, which can set no bit the commit has not
    // already emitted.
    Setting<bool> dyn_state_memo{false};
    // Skip BuildRuntimeInfo and its fingerprint hash for the vertex and
    // fragment stages while the graphics register stamp is unchanged; those
    // two arms read only stamp-covered registers and boot constants.
    Setting<bool> runtime_info_stamp_gate{false};
    // Answer every occlusion query as fully occluded instead of fully
    // visible. Titles that gate effects on visibility (inFAMOUS lens flares)
    // then cull those draws themselves before submission.
    Setting<bool> occlude_all{false};
    // Drain read-only staging upload copies through the stream copy lane
    // workers instead of copying inline on the GPU command thread. Written
    // binds always copy inline under their region locks.
    Setting<bool> stream_copy_upload_drain{false};
    // Flush the graphics command buffer every this many draws (0 = only at
    // submit-done and faults). A guest readback then waits on a command
    // buffer holding at most this many draws instead of the whole recorded
    // body. Values below 64 are raised to 64 (each flush costs a submit).
    Setting<u32> flush_draw_interval{0};
    // Reuse the previous graphics pipeline key while the register stamp
    // repeats: only the stage resolve reruns. Needs runtime_info_stamp_gate
    // and dynamic vertex input; otherwise the lookup runs unchanged.
    Setting<bool> pipeline_key_stamp_reuse{false};
    // Reuse the binary-info search result for a stage while its code pointer
    // and the hash stored inside the binary repeat.
    Setting<bool> shader_params_memo{false};
    // Specialization fingerprint over the sharp bits the specialization
    // reads: 1 keys the tier on it and carries the resolved module in the
    // MRU, 2 adds a per-stage slot answered by a memcmp. Needs spec_fp_cache.
    Setting<u32> spec_fp_canonical{0};
    // Hands a texture binding the view handle its FINDIMG memo hit recorded,
    // keyed on the image backing; the view record scan runs only on a miss.
    Setting<bool> texture_view_memo{false};
    // Skip the sampler map mutex: GetSampler and the sampler GC both run on the
    // GPU thread only, so the lock pair per sampler bind is dead synchronization.
    Setting<bool> sampler_memo_lockfree{false};
    // Compare and store descriptor writes into the delta slot in one walk
    // instead of serializing to a scratch buffer and comparing afterwards.
    Setting<bool> desc_delta_inplace{false};
    // Prefetch, during the first texture binding pass, the three image lines
    // the second pass reads first (props, backing pointer, backing state).
    // Read once at boot.
    Setting<bool> bind_line_prefetch{false};
    // Hold the guest-copy shared lock once per graphics packet run instead of
    // once per draw; the hold drops before every flush, GPU wait, command
    // drain and pipeline compile.
    Setting<bool> guest_copy_hold_segment{false};
    // A consumed image memo hit stamps its access tick without the texture
    // mutex; the LRU touch stays under it and runs once per image per GC tick.
    Setting<bool> findimg_touch_lockfree{false};
    // Stream-copy and index-bind memo entries remember the tracker region that
    // covered their range, so a hit re-certifies the word-epoch sum with that
    // region's own loads instead of the tracker walk.
    Setting<bool> stream_copy_resolved_epoch{false};
    // Written binds: 1 skips the range-set containment probe when the mark set
    // a GPU-clean page; 2 adds a memo of ranges proven contained, cleared by
    // every download subtract; 3 defers the adds to a per-region pending log
    // that the next reader of an overlapping range folds in. Higher values act
    // as 3.
    Setting<u32> written_range_fast{0};
    // Compare the gathered specialization key against its per-stage slot and
    // store it in one pass. Needs spec_fp_canonical 2.
    Setting<bool> spec_fp_slot_inplace{false};
    // A 16-entry associative front over each program's fingerprint table, for
    // programs that cycle through more specializations per frame than the MRU
    // pair holds. Needs spec_fp_canonical.
    Setting<bool> spec_fp_front{false};
    // Image memo geometry: 0 keeps the 1024-slot direct-mapped probe; 1, 2 or 4
    // index 2048 entries by every T# word into sets of that many ways with LRU
    // replacement. 3 acts as 2, higher values as 4.
    Setting<u32> findimg_memo_ways{0};
    // Per texture memo entry, the backing epoch at which the shader-read
    // transit was a no-op and the layout it held; a repeat under that epoch skips
    // the transit probe and the backing's lines. Needs texture_view_memo.
    Setting<bool> bind_noop_memo{false};
    // Canonical specialization key layout: 1 starts every key word on an 8-byte
    // boundary so the in-place fold's loads forward from the gather's stores; 2
    // also warms the slot lines ahead of the gather. Higher values act as 2.
    Setting<u32> spec_key_fast{0};
    // The GPU-modified range set gets its own node pool whose mutex is skipped:
    // every operation on that set runs on the GPU command thread.
    Setting<bool> gpu_range_set_lockfree{false};
    // Hold the guest-copy shared lock once per readback write-back loop instead
    // of once per island.
    Setting<bool> readback_writeback_hold{false};
    // Serve guest-visible backing writes from a per-thread memo of the last
    // resolved physical chunks, revalidated by the memory map generation; the
    // map descent runs only on a miss.
    Setting<bool> backing_write_memo{false};
    // Run the per-image fast-state check directly for sampled bindings instead
    // of the per-binding dedup probe. Needs image_fast_state. In Adaptive mode
    // the dedup cache cycles Learning/Off with no eligible calls; in ValidateOnly
    // its premise is no longer checked.
    Setting<bool> image_update_direct{false};
    // One descriptor set layout and pipeline layout per distinct binding list,
    // shared by every pipeline of that shape. Read once at boot.
    Setting<bool> desc_layout_share{false};
    // Derive the vertex-input signatures from the fetch shader's V# words and
    // build the Vulkan descriptions only for a layout change.
    Setting<bool> vertex_input_lazy_desc{false};
    // Per-stage two-entry memo of the register words the Vertex, Fragment and
    // Compute runtime-info builds read; an equal snapshot restores the struct
    // and its fingerprint hash wherever the rebuild runs.
    Setting<bool> runtime_info_input_memo{false};
    // Runs an offloaded readback's write-back on the thread that waited out its
    // fence; the GPU command thread keeps the per-island verdict and the unmark.
    // Islands another job still owns are left out of a new download.
    Setting<bool> readback_writeback_offload{false};
    // Decides the stamp-keyed key reuse from a running XOR of the stage hashes
    // the resolve rewrites instead of re-reading the hash array it just stored.
    // Needs pipeline_key_stamp_reuse.
    Setting<bool> key_reuse_hash_diff{false};
    // Pushes only the descriptors whose bytes differ from the last push on the
    // same command buffer and layout; the rest stay as the driver holds them.
    // Needs desc_delta_inplace.
    Setting<bool> desc_delta_partial{false};
    // Direct-mapped table of that many entries behind the per-stage binary-info
    // memo, indexed by the code address, so the search's two lines are read
    // independently and the Vertex lines are prefetched ahead of the Fragment
    // resolve. Needs shader_params_memo; 0 keeps the single entry.
    Setting<u32> shader_params_memo_entries{0};
    // Keys the dynamic-state memo on a stamp lane bumped only by the context and
    // uconfig registers its updaters read, and on the pipeline's write masks
    // instead of its identity. Needs dyn_state_memo.
    Setting<bool> dyn_state_stamp{false};
    // Keeps the image LRU as an append-only touch log with tombstones instead of
    // a linked list relinked on every first touch per submit; the GC walk skips
    // tombstones and compacts.
    Setting<bool> texture_lru_log{false};
    // Lets read-only formatted binds record a no-upload walk in the buffer's sync
    // memo; spans past the epoch-sum limit key on the host-memory generation.
    Setting<bool> texel_sync_noop{false};
    // Arms the read watcher of GPU-written pages at the next guest-visible
    // completion point (fence, wait, submit, packet-run end, idle) instead of at
    // each written bind, so consecutive marks share one protection call.
    Setting<bool> deferred_read_arm{false};
    // Bakes the color write mask into the pipeline's blend state instead of
    // declaring it dynamic. The mask is already a pipeline key field, so the
    // pipeline count is unchanged.
    Setting<bool> static_color_write_mask{false};
    // Gathers the canonical specialization key straight into its compare slot,
    // folding the compare into the gather's stores. Needs spec_fp_canonical 2,
    // spec_fp_slot_inplace and spec_key_fast.
    Setting<bool> spec_key_fused{false};
    // Collapses the clean steady state of per-binding texture updates to one
    // atomic load instead of a texture-cache mutex acquisition; every
    // dirtying path stamps the per-image word back to dirty.
    Setting<bool> image_fast_state{false};
    // Holds the memory map's shared lock across a whole buffer-bind batch so
    // each guest copy inside stops paying its own pair of contended atomic
    // lock operations.
    Setting<bool> guest_copy_lock_batch{false};
    // Probe the most recently matched shader permutation before the linear search
    // in the pipeline cache. May select a different compare-equal permutation when
    // several stored specializations satisfy the probe.
    Setting<bool> spec_mru_perm_probe{false};
    Setting<bool> direct_memory_access_enabled{false};
    Setting<bool> dump_shaders{false};
    Setting<bool> patch_shaders{false};
    Setting<u32> vblank_frequency{60};
    Setting<bool> full_screen{false};
    Setting<std::string> full_screen_mode{"Windowed"};
    Setting<std::string> present_mode{"Mailbox"};
    Setting<bool> hdr_allowed{false};
    Setting<bool> fsr_enabled{false};
    Setting<bool> rcas_enabled{true};
    Setting<int> rcas_attenuation{250};
    // TODO add overrides
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<GPUSettings>("null_gpu", &GPUSettings::null_gpu),
            make_override<GPUSettings>("copy_gpu_buffers", &GPUSettings::copy_gpu_buffers),
            make_override<GPUSettings>("full_screen", &GPUSettings::full_screen),
            make_override<GPUSettings>("full_screen_mode", &GPUSettings::full_screen_mode),
            make_override<GPUSettings>("present_mode", &GPUSettings::present_mode),
            make_override<GPUSettings>("window_height", &GPUSettings::window_height),
            make_override<GPUSettings>("window_width", &GPUSettings::window_width),
            make_override<GPUSettings>("hdr_allowed", &GPUSettings::hdr_allowed),
            make_override<GPUSettings>("fsr_enabled", &GPUSettings::fsr_enabled),
            make_override<GPUSettings>("rcas_enabled", &GPUSettings::rcas_enabled),
            make_override<GPUSettings>("rcas_attenuation", &GPUSettings::rcas_attenuation),
            make_override<GPUSettings>("dump_shaders", &GPUSettings::dump_shaders),
            make_override<GPUSettings>("patch_shaders", &GPUSettings::patch_shaders),
            make_override<GPUSettings>("readbacks_mode", &GPUSettings::readbacks_mode),
            make_override<GPUSettings>("readback_linear_images_enabled",
                                       &GPUSettings::readback_linear_images_enabled),
            make_override<GPUSettings>("adaptive_skipcaches_mode",
                                       &GPUSettings::adaptive_skipcaches_mode),
            make_override<GPUSettings>("stream_buffer_size_mb",
                                       &GPUSettings::stream_buffer_size_mb),
            make_override<GPUSettings>("readback_batching_enabled",
                                       &GPUSettings::readback_batching_enabled),
            make_override<GPUSettings>("readback_offload_mode",
                                       &GPUSettings::readback_offload_mode),
            make_override<GPUSettings>("stream_buffer_prefer_host",
                                       &GPUSettings::stream_buffer_prefer_host),
            make_override<GPUSettings>("stream_upload_mirror_mode",
                                       &GPUSettings::stream_upload_mirror_mode),
            make_override<GPUSettings>("fault_widen_bytes", &GPUSettings::fault_widen_bytes),
            make_override<GPUSettings>("pending_pop_throttle", &GPUSettings::pending_pop_throttle),
            make_override<GPUSettings>("stream_copy_workers", &GPUSettings::stream_copy_workers),
            make_override<GPUSettings>("stream_findbuffer_elide",
                                       &GPUSettings::stream_findbuffer_elide),
            make_override<GPUSettings>("spec_fp_cache", &GPUSettings::spec_fp_cache),
            make_override<GPUSettings>("dyn_state_memo", &GPUSettings::dyn_state_memo),
            make_override<GPUSettings>("runtime_info_stamp_gate",
                                       &GPUSettings::runtime_info_stamp_gate),
            make_override<GPUSettings>("occlude_all", &GPUSettings::occlude_all),
            make_override<GPUSettings>("stream_copy_upload_drain",
                                       &GPUSettings::stream_copy_upload_drain),
            make_override<GPUSettings>("flush_draw_interval", &GPUSettings::flush_draw_interval),
            make_override<GPUSettings>("pipeline_key_stamp_reuse",
                                       &GPUSettings::pipeline_key_stamp_reuse),
            make_override<GPUSettings>("shader_params_memo", &GPUSettings::shader_params_memo),
            make_override<GPUSettings>("spec_fp_canonical", &GPUSettings::spec_fp_canonical),
            make_override<GPUSettings>("texture_view_memo", &GPUSettings::texture_view_memo),
            make_override<GPUSettings>("sampler_memo_lockfree",
                                       &GPUSettings::sampler_memo_lockfree),
            make_override<GPUSettings>("desc_delta_inplace", &GPUSettings::desc_delta_inplace),
            make_override<GPUSettings>("bind_line_prefetch", &GPUSettings::bind_line_prefetch),
            make_override<GPUSettings>("guest_copy_hold_segment",
                                       &GPUSettings::guest_copy_hold_segment),
            make_override<GPUSettings>("findimg_touch_lockfree",
                                       &GPUSettings::findimg_touch_lockfree),
            make_override<GPUSettings>("stream_copy_resolved_epoch",
                                       &GPUSettings::stream_copy_resolved_epoch),
            make_override<GPUSettings>("written_range_fast", &GPUSettings::written_range_fast),
            make_override<GPUSettings>("spec_fp_slot_inplace", &GPUSettings::spec_fp_slot_inplace),
            make_override<GPUSettings>("spec_fp_front", &GPUSettings::spec_fp_front),
            make_override<GPUSettings>("findimg_memo_ways", &GPUSettings::findimg_memo_ways),
            make_override<GPUSettings>("bind_noop_memo", &GPUSettings::bind_noop_memo),
            make_override<GPUSettings>("spec_key_fast", &GPUSettings::spec_key_fast),
            make_override<GPUSettings>("gpu_range_set_lockfree",
                                       &GPUSettings::gpu_range_set_lockfree),
            make_override<GPUSettings>("readback_writeback_hold",
                                       &GPUSettings::readback_writeback_hold),
            make_override<GPUSettings>("backing_write_memo", &GPUSettings::backing_write_memo),
            make_override<GPUSettings>("image_update_direct", &GPUSettings::image_update_direct),
            make_override<GPUSettings>("desc_layout_share", &GPUSettings::desc_layout_share),
            make_override<GPUSettings>("vertex_input_lazy_desc",
                                       &GPUSettings::vertex_input_lazy_desc),
            make_override<GPUSettings>("runtime_info_input_memo",
                                       &GPUSettings::runtime_info_input_memo),
            make_override<GPUSettings>("readback_writeback_offload",
                                       &GPUSettings::readback_writeback_offload),
            make_override<GPUSettings>("key_reuse_hash_diff", &GPUSettings::key_reuse_hash_diff),
            make_override<GPUSettings>("desc_delta_partial", &GPUSettings::desc_delta_partial),
            make_override<GPUSettings>("shader_params_memo_entries",
                                       &GPUSettings::shader_params_memo_entries),
            make_override<GPUSettings>("dyn_state_stamp", &GPUSettings::dyn_state_stamp),
            make_override<GPUSettings>("texture_lru_log", &GPUSettings::texture_lru_log),
            make_override<GPUSettings>("texel_sync_noop", &GPUSettings::texel_sync_noop),
            make_override<GPUSettings>("deferred_read_arm", &GPUSettings::deferred_read_arm),
            make_override<GPUSettings>("static_color_write_mask",
                                       &GPUSettings::static_color_write_mask),
            make_override<GPUSettings>("spec_key_fused", &GPUSettings::spec_key_fused),
            make_override<GPUSettings>("image_fast_state", &GPUSettings::image_fast_state),
            make_override<GPUSettings>("guest_copy_lock_batch",
                                       &GPUSettings::guest_copy_lock_batch),
            make_override<GPUSettings>("spec_mru_perm_probe", &GPUSettings::spec_mru_perm_probe),
            make_override<GPUSettings>("direct_memory_access_enabled",
                                       &GPUSettings::direct_memory_access_enabled),
            make_override<GPUSettings>("vblank_frequency", &GPUSettings::vblank_frequency),
        };
    }
};
// nlohmann's field macros take at most 63 names, so the GPU settings are
// serialized in two groups; new settings go at the end of the second.
// clang-format off
#define GPU_SETTINGS_JSON_FIELDS_A \
    window_width, window_height, internal_screen_width, internal_screen_height, null_gpu, \
    copy_gpu_buffers, readbacks_mode, readback_linear_images_enabled, adaptive_skipcaches_mode, \
    stream_buffer_size_mb, readback_batching_enabled, readback_offload_mode, \
    stream_buffer_prefer_host, direct_memory_access_enabled, dump_shaders, patch_shaders, \
    vblank_frequency, full_screen, full_screen_mode, present_mode, hdr_allowed, fsr_enabled, \
    rcas_enabled, rcas_attenuation, spec_mru_perm_probe, stream_upload_mirror_mode, \
    image_fast_state, guest_copy_lock_batch, spec_fp_cache, pending_pop_throttle, \
    fault_widen_bytes, stream_copy_workers, stream_findbuffer_elide, dyn_state_memo, \
    runtime_info_stamp_gate, occlude_all, stream_copy_upload_drain, flush_draw_interval, \
    pipeline_key_stamp_reuse, shader_params_memo
#define GPU_SETTINGS_JSON_FIELDS_B \
    spec_fp_canonical, texture_view_memo, sampler_memo_lockfree, desc_delta_inplace, \
    bind_line_prefetch, guest_copy_hold_segment, findimg_touch_lockfree, \
    stream_copy_resolved_epoch, written_range_fast, spec_fp_slot_inplace, spec_fp_front, \
    findimg_memo_ways, bind_noop_memo, spec_key_fast, gpu_range_set_lockfree, \
    readback_writeback_hold, backing_write_memo, image_update_direct, desc_layout_share, \
    vertex_input_lazy_desc, runtime_info_input_memo, readback_writeback_offload, \
    key_reuse_hash_diff, desc_delta_partial, shader_params_memo_entries, dyn_state_stamp, texture_lru_log, texel_sync_noop, deferred_read_arm, static_color_write_mask, spec_key_fused
// clang-format on
template <
    typename BasicJsonType,
    nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0>
void to_json(BasicJsonType& nlohmann_json_j, const GPUSettings& nlohmann_json_t) {
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, GPU_SETTINGS_JSON_FIELDS_A))
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_TO, GPU_SETTINGS_JSON_FIELDS_B))
}
template <
    typename BasicJsonType,
    nlohmann::detail::enable_if_t<nlohmann::detail::is_basic_json<BasicJsonType>::value, int> = 0>
void from_json(const BasicJsonType& nlohmann_json_j, GPUSettings& nlohmann_json_t) {
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, GPU_SETTINGS_JSON_FIELDS_A))
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(NLOHMANN_JSON_FROM, GPU_SETTINGS_JSON_FIELDS_B))
}
// -------------------------------
// Vulkan settings
// -------------------------------
struct VulkanSettings {
    Setting<s32> gpu_id{-1};
    Setting<bool> renderdoc_enabled{false};
    Setting<bool> vkvalidation_enabled{false};
    Setting<bool> vkvalidation_core_enabled{true};
    Setting<bool> vkvalidation_sync_enabled{false};
    Setting<bool> vkvalidation_gpu_enabled{false};
    Setting<bool> vkcrash_diagnostic_enabled{false};
    Setting<bool> vkhost_markers{false};
    Setting<bool> vkguest_markers{false};
    Setting<bool> pipeline_cache_enabled{false};
    Setting<bool> pipeline_cache_archived{false};
    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<VulkanSettings>("gpu_id", &VulkanSettings::gpu_id),
            make_override<VulkanSettings>("renderdoc_enabled", &VulkanSettings::renderdoc_enabled),
            make_override<VulkanSettings>("vkvalidation_enabled",
                                          &VulkanSettings::vkvalidation_enabled),
            make_override<VulkanSettings>("vkvalidation_core_enabled",
                                          &VulkanSettings::vkvalidation_core_enabled),
            make_override<VulkanSettings>("vkvalidation_sync_enabled",
                                          &VulkanSettings::vkvalidation_sync_enabled),
            make_override<VulkanSettings>("vkvalidation_gpu_enabled",
                                          &VulkanSettings::vkvalidation_gpu_enabled),
            make_override<VulkanSettings>("vkcrash_diagnostic_enabled",
                                          &VulkanSettings::vkcrash_diagnostic_enabled),
            make_override<VulkanSettings>("vkhost_markers", &VulkanSettings::vkhost_markers),
            make_override<VulkanSettings>("vkguest_markers", &VulkanSettings::vkguest_markers),
            make_override<VulkanSettings>("pipeline_cache_enabled",
                                          &VulkanSettings::pipeline_cache_enabled),
            make_override<VulkanSettings>("pipeline_cache_archived",
                                          &VulkanSettings::pipeline_cache_archived),
        };
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VulkanSettings, gpu_id, renderdoc_enabled, vkvalidation_enabled,
                                   vkvalidation_core_enabled, vkvalidation_sync_enabled,
                                   vkvalidation_gpu_enabled, vkcrash_diagnostic_enabled,
                                   vkhost_markers, vkguest_markers, pipeline_cache_enabled,
                                   pipeline_cache_archived)

// -------------------------------
// Main manager
// -------------------------------
class EmulatorSettingsImpl {
public:
    EmulatorSettingsImpl();
    ~EmulatorSettingsImpl();

    static std::shared_ptr<EmulatorSettingsImpl> GetInstance();
    static void SetInstance(std::shared_ptr<EmulatorSettingsImpl> instance);

    bool Save(const std::string& serial = "");
    bool Load(const std::string& serial = "");
    void SetDefaultValues();
    bool TransferSettings();

    // Config mode
    ConfigMode GetConfigMode() const {
        return m_configMode;
    }
    void SetConfigMode(ConfigMode mode) {
        m_configMode = mode;
    }

    //
    // Game-specific override management
    /// Clears all per-game overrides.  Call this when a game exits so
    /// the emulator reverts to global settings.
    void ClearGameSpecificOverrides();

    /// Reset a single field's game-specific override by its JSON ke
    void ResetGameSpecificValue(const std::string& key);

    // general accessors
    bool AddGameInstallDir(const std::filesystem::path& dir, bool enabled = true);
    std::vector<std::filesystem::path> GetGameInstallDirs() const;
    void SetAllGameInstallDirs(const std::vector<GameInstallDir>& dirs);
    void RemoveGameInstallDir(const std::filesystem::path& dir);
    void SetGameInstallDirEnabled(const std::filesystem::path& dir, bool enabled);
    void SetGameInstallDirs(const std::vector<std::filesystem::path>& dirs_config);
    const std::vector<bool> GetGameInstallDirsEnabled();
    const std::vector<GameInstallDir>& GetAllGameInstallDirs() const;

    std::filesystem::path GetHomeDir();
    void SetHomeDir(const std::filesystem::path& dir);
    std::filesystem::path GetSysModulesDir();
    void SetSysModulesDir(const std::filesystem::path& dir);
    std::filesystem::path GetFontsDir();
    void SetFontsDir(const std::filesystem::path& dir);
    std::filesystem::path GetAddonInstallDir();
    void SetAddonInstallDir(const std::filesystem::path& dir);

private:
    GeneralSettings m_general{};
    LogSettings m_log{};
    DebugSettings m_debug{};
    InputSettings m_input{};
    AudioSettings m_audio{};
    // Windows static guest red-zone protection
    WindowsGuestRedZoneProtectionSettings m_windows_guest_red_zone_protection{};
    GPUSettings m_gpu{};
    VulkanSettings m_vulkan{};
    ConfigMode m_configMode{ConfigMode::Default};

    // Runtime-only override: when true, IsShadNetEnabled() reports false for the
    // rest of this run regardless of the persisted setting
    std::atomic<bool> m_shadnet_session_disabled{false};

    bool m_loaded{false};

    static std::shared_ptr<EmulatorSettingsImpl> s_instance;
    static std::mutex s_mutex;

    /// Apply overrideable fields from groupJson into group.game_specific_value.
    template <typename Group>
    void ApplyGroupOverrides(Group& group, const nlohmann::json& groupJson,
                             std::vector<std::string>& changed) {
        for (auto& item : group.GetOverrideableFields()) {
            if (!groupJson.contains(item.key))
                continue;
            item.apply(&group, groupJson.at(item.key), changed);
        }
    }

    // Write all overrideable fields from group into out (for game-specific save).
    template <typename Group>
    static void SaveGroupGameSpecific(const Group& group, nlohmann::json& out) {
        for (auto& item : group.GetOverrideableFields())
            out[item.key] = item.get_for_save(&group);
    }

    // Discard every game-specific override in group.
    template <typename Group>
    static void ClearGroupOverrides(Group& group) {
        for (auto& item : group.GetOverrideableFields())
            item.reset_game_specific(&group);
    }

    static void PrintChangedSummary(const std::vector<std::string>& changed);

public:
    // Add these getters to access overrideable fields
    std::vector<OverrideItem> GetGeneralOverrideableFields() const {
        return m_general.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetDebugOverrideableFields() const {
        return m_debug.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetInputOverrideableFields() const {
        return m_input.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetAudioOverrideableFields() const {
        return m_audio.GetOverrideableFields();
    }
    // Windows static guest red-zone protection
    std::vector<OverrideItem> GetWindowsGuestRedZoneProtectionOverrideableFields() const {
        return m_windows_guest_red_zone_protection.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetGPUOverrideableFields() const {
        return m_gpu.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetVulkanOverrideableFields() const {
        return m_vulkan.GetOverrideableFields();
    }
    std::vector<std::string> GetAllOverrideableKeys() const;

#define SETTING_FORWARD(group, Name, field)                                                        \
    auto Get##Name() const {                                                                       \
        return (group).field.get(m_configMode);                                                    \
    }                                                                                              \
    void Set##Name(const decltype((group).field.value)& v, bool specific = false) {                \
        (group).field.set(v, specific);                                                            \
    }
#define SETTING_FORWARD_BOOL(group, Name, field)                                                   \
    bool Is##Name() const {                                                                        \
        return (group).field.get(m_configMode);                                                    \
    }                                                                                              \
    void Set##Name(bool v, bool specific = false) {                                                \
        (group).field.set(v, specific);                                                            \
    }
#define SETTING_FORWARD_BOOL_READONLY(group, Name, field)                                          \
    bool Is##Name() const {                                                                        \
        return (group).field.get(m_configMode);                                                    \
    }

    // General settings
    SETTING_FORWARD(m_general, VolumeSlider, volume_slider)
    SETTING_FORWARD_BOOL(m_general, Neo, neo_mode)
    SETTING_FORWARD_BOOL(m_general, DevKit, dev_kit_mode)
    SETTING_FORWARD(m_general, ExtraDmemInMBytes, extra_dmem_in_mbytes)
    SETTING_FORWARD(m_general, ExtraFmemInMBytes, extra_fmem_in_mbytes)
    bool IsShadNetEnabled() const {
        return m_general.shad_net_enabled.get(m_configMode) &&
               !m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    void SetShadNetEnabled(bool v, bool specific = false) {
        m_general.shad_net_enabled.set(v, specific);
    }
    bool IsShadNetEnabledSetting() const {
        return m_general.shad_net_enabled.get(m_configMode);
    }
    void SetShadNetSessionDisabled(bool v) {
        m_shadnet_session_disabled.store(v, std::memory_order_relaxed);
    }
    bool IsShadNetSessionDisabled() const {
        return m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    SETTING_FORWARD_BOOL(m_general, TrophyPopupDisabled, trophy_popup_disabled)
    SETTING_FORWARD(m_general, TrophyNotificationDuration, trophy_notification_duration)
    SETTING_FORWARD(m_general, TrophyNotificationSide, trophy_notification_side)
    SETTING_FORWARD_BOOL(m_general, ShowSplash, show_splash)
    SETTING_FORWARD_BOOL(m_general, ConnectedToNetwork, connected_to_network)
    SETTING_FORWARD_BOOL(m_general, DiscordRPCEnabled, discord_rpc_enabled)
    SETTING_FORWARD_BOOL(m_general, ShowFpsCounter, show_fps_counter)
    SETTING_FORWARD(m_general, ConsoleLanguage, console_language)
    SETTING_FORWARD(m_general, BigPictureScale, big_picture_scale)
    SETTING_FORWARD(m_general, ShadNetServer, shadnet_server)
    SETTING_FORWARD(m_general, ShadNetWebApiServer, shadnet_webapi_server)
    SETTING_FORWARD(m_general, SignalingInfo, signaling_info)
    SETTING_FORWARD_BOOL(m_general, UPnPEnabled, enable_upnp)

    // Log settings
    SETTING_FORWARD_BOOL(m_log, LogAppend, append)
    SETTING_FORWARD_BOOL(m_log, LogEnable, enable)
    SETTING_FORWARD(m_log, LogFilter, filter)
    SETTING_FORWARD(m_log, LogFlushLevel, flush_level)
    SETTING_FORWARD(m_log, LogMaxSkipDuration, max_skip_duration)
    SETTING_FORWARD_BOOL(m_log, LogSeparate, separate)
    SETTING_FORWARD(m_log, LogSizeLimit, size_limit)
    SETTING_FORWARD_BOOL(m_log, LogSkipDuplicate, skip_duplicate)
    SETTING_FORWARD_BOOL(m_log, LogSync, sync)
#ifdef _WIN32
    SETTING_FORWARD(m_log, LogType, type)
#endif

    // Audio settings
    SETTING_FORWARD(m_audio, AudioBackend, audio_backend)
    SETTING_FORWARD(m_audio, SDLMicDevice, sdl_mic_device)
    SETTING_FORWARD(m_audio, SDLMainOutputDevice, sdl_main_output_device)
    SETTING_FORWARD(m_audio, SDLPadSpkOutputDevice, sdl_padSpk_output_device)
    SETTING_FORWARD(m_audio, OpenALMicDevice, openal_mic_device)
    SETTING_FORWARD(m_audio, OpenALMainOutputDevice, openal_main_output_device)
    SETTING_FORWARD(m_audio, OpenALPadSpkOutputDevice, openal_padSpk_output_device)
    SETTING_FORWARD(m_audio, OpenALHrtf, openal_hrtf)
    SETTING_FORWARD(m_audio, OpenALOutputMode, openal_output_mode)

    // Windows static guest red-zone protection
    SETTING_FORWARD(m_windows_guest_red_zone_protection, WindowsGuestRedZoneProtectionMode,
                    windows_guest_red_zone_protection_mode)

    // Debug settings
    SETTING_FORWARD_BOOL(m_debug, DebugDump, debug_dump)
    SETTING_FORWARD_BOOL(m_debug, ShaderCollect, shader_collect)
    SETTING_FORWARD(m_debug, ConfigVersion, config_version)

    // GPU Settings
    SETTING_FORWARD(m_gpu, AdaptiveSkipCachesMode, adaptive_skipcaches_mode)
    SETTING_FORWARD(m_gpu, StreamBufferSizeMb, stream_buffer_size_mb)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackBatchingEnabled, readback_batching_enabled)
    SETTING_FORWARD(m_gpu, ReadbackOffloadMode, readback_offload_mode)
    SETTING_FORWARD_BOOL(m_gpu, StreamBufferPreferHost, stream_buffer_prefer_host)
    SETTING_FORWARD(m_gpu, StreamUploadMirrorMode, stream_upload_mirror_mode)
    SETTING_FORWARD(m_gpu, FaultWidenBytes, fault_widen_bytes)
    SETTING_FORWARD(m_gpu, PendingPopThrottle, pending_pop_throttle)
    SETTING_FORWARD(m_gpu, StreamCopyWorkers, stream_copy_workers)
    SETTING_FORWARD_BOOL(m_gpu, StreamFindBufferElide, stream_findbuffer_elide)
    SETTING_FORWARD_BOOL(m_gpu, SpecFpCache, spec_fp_cache)
    SETTING_FORWARD_BOOL(m_gpu, DynStateMemo, dyn_state_memo)
    SETTING_FORWARD_BOOL(m_gpu, RuntimeInfoStampGate, runtime_info_stamp_gate)
    SETTING_FORWARD_BOOL(m_gpu, OccludeAll, occlude_all)
    SETTING_FORWARD_BOOL(m_gpu, StreamCopyUploadDrain, stream_copy_upload_drain)
    SETTING_FORWARD(m_gpu, FlushDrawInterval, flush_draw_interval)
    SETTING_FORWARD_BOOL(m_gpu, PipelineKeyStampReuse, pipeline_key_stamp_reuse)
    SETTING_FORWARD_BOOL(m_gpu, ShaderParamsMemo, shader_params_memo)
    SETTING_FORWARD(m_gpu, SpecFpCanonical, spec_fp_canonical)
    SETTING_FORWARD_BOOL(m_gpu, TextureViewMemo, texture_view_memo)
    SETTING_FORWARD_BOOL(m_gpu, SamplerMemoLockfree, sampler_memo_lockfree)
    SETTING_FORWARD_BOOL(m_gpu, DescDeltaInplace, desc_delta_inplace)
    SETTING_FORWARD_BOOL(m_gpu, BindLinePrefetch, bind_line_prefetch)
    SETTING_FORWARD_BOOL(m_gpu, GuestCopyHoldSegment, guest_copy_hold_segment)
    SETTING_FORWARD_BOOL(m_gpu, FindimgTouchLockfree, findimg_touch_lockfree)
    SETTING_FORWARD_BOOL(m_gpu, StreamCopyResolvedEpoch, stream_copy_resolved_epoch)
    SETTING_FORWARD(m_gpu, WrittenRangeFast, written_range_fast)
    SETTING_FORWARD_BOOL(m_gpu, SpecFpSlotInplace, spec_fp_slot_inplace)
    SETTING_FORWARD_BOOL(m_gpu, SpecFpFront, spec_fp_front)
    SETTING_FORWARD(m_gpu, FindimgMemoWays, findimg_memo_ways)
    SETTING_FORWARD_BOOL(m_gpu, BindNoopMemo, bind_noop_memo)
    SETTING_FORWARD(m_gpu, SpecKeyFast, spec_key_fast)
    SETTING_FORWARD_BOOL(m_gpu, GpuRangeSetLockfree, gpu_range_set_lockfree)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackWritebackHold, readback_writeback_hold)
    SETTING_FORWARD_BOOL(m_gpu, BackingWriteMemo, backing_write_memo)
    SETTING_FORWARD_BOOL(m_gpu, ImageUpdateDirect, image_update_direct)
    SETTING_FORWARD_BOOL(m_gpu, DescLayoutShare, desc_layout_share)
    SETTING_FORWARD_BOOL(m_gpu, VertexInputLazyDesc, vertex_input_lazy_desc)
    SETTING_FORWARD_BOOL(m_gpu, RuntimeInfoInputMemo, runtime_info_input_memo)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackWritebackOffload, readback_writeback_offload)
    SETTING_FORWARD_BOOL(m_gpu, KeyReuseHashDiff, key_reuse_hash_diff)
    SETTING_FORWARD_BOOL(m_gpu, DescDeltaPartial, desc_delta_partial)
    SETTING_FORWARD(m_gpu, ShaderParamsMemoEntries, shader_params_memo_entries)
    SETTING_FORWARD_BOOL(m_gpu, DynStateStamp, dyn_state_stamp)
    SETTING_FORWARD_BOOL(m_gpu, TextureLruLog, texture_lru_log)
    SETTING_FORWARD_BOOL(m_gpu, TexelSyncNoop, texel_sync_noop)
    SETTING_FORWARD_BOOL(m_gpu, DeferredReadArm, deferred_read_arm)
    SETTING_FORWARD_BOOL(m_gpu, StaticColorWriteMask, static_color_write_mask)
    SETTING_FORWARD_BOOL(m_gpu, SpecKeyFused, spec_key_fused)
    SETTING_FORWARD_BOOL(m_gpu, ImageFastState, image_fast_state)
    SETTING_FORWARD_BOOL(m_gpu, GuestCopyLockBatch, guest_copy_lock_batch)
    SETTING_FORWARD_BOOL(m_gpu, SpecMruPermProbe, spec_mru_perm_probe)
    SETTING_FORWARD_BOOL(m_gpu, NullGPU, null_gpu)
    SETTING_FORWARD_BOOL(m_gpu, DumpShaders, dump_shaders)
    SETTING_FORWARD_BOOL(m_gpu, CopyGpuBuffers, copy_gpu_buffers)
    SETTING_FORWARD_BOOL(m_gpu, FullScreen, full_screen)
    SETTING_FORWARD(m_gpu, FullScreenMode, full_screen_mode)
    SETTING_FORWARD(m_gpu, PresentMode, present_mode)
    SETTING_FORWARD(m_gpu, WindowHeight, window_height)
    SETTING_FORWARD(m_gpu, WindowWidth, window_width)
    SETTING_FORWARD(m_gpu, InternalScreenHeight, internal_screen_height)
    SETTING_FORWARD(m_gpu, InternalScreenWidth, internal_screen_width)
    SETTING_FORWARD_BOOL(m_gpu, HdrAllowed, hdr_allowed)
    SETTING_FORWARD_BOOL(m_gpu, FsrEnabled, fsr_enabled)
    SETTING_FORWARD_BOOL(m_gpu, RcasEnabled, rcas_enabled)
    SETTING_FORWARD(m_gpu, RcasAttenuation, rcas_attenuation)
    SETTING_FORWARD(m_gpu, ReadbacksMode, readbacks_mode)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackLinearImagesEnabled, readback_linear_images_enabled)
    SETTING_FORWARD_BOOL(m_gpu, DirectMemoryAccessEnabled, direct_memory_access_enabled)
    SETTING_FORWARD_BOOL_READONLY(m_gpu, PatchShaders, patch_shaders)

    u32 GetVblankFrequency() {
        if (m_gpu.vblank_frequency.value < 30) {
            return 30;
        }
        return m_gpu.vblank_frequency.get();
    }
    void SetVblankFrequency(const u32& v, bool is_specific = false) {
        u32 val = v < 30 ? 30 : v;
        if (is_specific) {
            m_gpu.vblank_frequency.game_specific_value = val;
        } else {
            m_gpu.vblank_frequency.value = val;
        }
    }

    // Input Settings
    SETTING_FORWARD(m_input, CursorState, cursor_state)
    SETTING_FORWARD(m_input, CursorHideTimeout, cursor_hide_timeout)
    SETTING_FORWARD(m_input, UsbDeviceBackend, usb_device_backend)
    SETTING_FORWARD_BOOL(m_input, MotionControlsEnabled, motion_controls_enabled)
    SETTING_FORWARD_BOOL(m_input, BackgroundControllerInput, background_controller_input)
    SETTING_FORWARD_BOOL(m_input, ImeAccessibilityEnabled, ime_accessibility_enabled)
    SETTING_FORWARD_BOOL(m_input, ImeUrlMailShortPanel, ime_url_mail_short_panel)
    SETTING_FORWARD(m_input, DefaultControllerId, default_controller_id)
    SETTING_FORWARD_BOOL(m_input, UsingSpecialPad, use_special_pad)
    SETTING_FORWARD(m_input, SpecialPadClass, special_pad_class)
    SETTING_FORWARD_BOOL(m_input, UseUnifiedInputConfig, use_unified_input_config)
    SETTING_FORWARD(m_input, CameraId, camera_id)
    SETTING_FORWARD_BOOL(m_input, CircleEnter, is_circle_enter)
    SETTING_FORWARD_BOOL(m_input, MiceUsedAsMice, use_mice_as_mice)

    // Vulkan settings
    SETTING_FORWARD(m_vulkan, GpuId, gpu_id)
    SETTING_FORWARD_BOOL(m_vulkan, RenderdocEnabled, renderdoc_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationEnabled, vkvalidation_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationCoreEnabled, vkvalidation_core_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationSyncEnabled, vkvalidation_sync_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkValidationGpuEnabled, vkvalidation_gpu_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkCrashDiagnosticEnabled, vkcrash_diagnostic_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, VkHostMarkersEnabled, vkhost_markers)
    SETTING_FORWARD_BOOL(m_vulkan, VkGuestMarkersEnabled, vkguest_markers)
    SETTING_FORWARD_BOOL(m_vulkan, PipelineCacheEnabled, pipeline_cache_enabled)
    SETTING_FORWARD_BOOL(m_vulkan, PipelineCacheArchived, pipeline_cache_archived)

#undef SETTING_FORWARD
#undef SETTING_FORWARD_BOOL
#undef SETTING_FORWARD_BOOL_READONLY
};
