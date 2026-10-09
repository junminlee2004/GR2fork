// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "common/types.h"

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
    Setting<bool> trophy_popup_disabled{false};
    Setting<double> trophy_notification_duration{6.0};
    Setting<std::string> trophy_notification_side{"right"};
    Setting<bool> show_splash{false};
    Setting<bool> discord_rpc_enabled{false};
    Setting<bool> show_fps_counter{false};
    Setting<int> console_language{1};
    Setting<int> big_picture_scale{1000};
    Setting<int> big_picture_folder_depth{2};
    Setting<bool> redzone_patches{false};
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
            make_override<GeneralSettings>("trophy_popup_disabled",
                                           &GeneralSettings::trophy_popup_disabled),
            make_override<GeneralSettings>("trophy_notification_duration",
                                           &GeneralSettings::trophy_notification_duration),
            make_override<GeneralSettings>("show_splash", &GeneralSettings::show_splash),
            make_override<GeneralSettings>("trophy_notification_side",
                                           &GeneralSettings::trophy_notification_side),
            make_override<GeneralSettings>("console_language", &GeneralSettings::console_language),
            make_override<GeneralSettings>("redzone_patches", &GeneralSettings::redzone_patches)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(GeneralSettings, install_dirs, addon_install_dir, home_dir,
                                   sys_modules_dir, font_dir, volume_slider, neo_mode, dev_kit_mode,
                                   extra_dmem_in_mbytes, extra_fmem_in_mbytes,
                                   trophy_popup_disabled, trophy_notification_duration, show_splash,
                                   trophy_notification_side, discord_rpc_enabled, show_fps_counter,
                                   console_language, big_picture_scale, big_picture_folder_depth,
                                   redzone_patches)

// -------------------------------
// Network settings
// -------------------------------
struct NetworkSettings {
    Setting<bool> connected_to_network{false};
    Setting<bool> shad_net_enabled{false};
    Setting<std::string> shadnet_server{"srv.shadps4.net:31313"};
    Setting<std::string> shadnet_webapi_server{"http://srv.shadps4.net:31315"};
    Setting<std::string> signaling_info{};
    Setting<bool> enable_upnp{true};
    Setting<int> p2p_port{3658}; // 0 is any free port, 3658 is the default PS4 port. With UPnP
                                 // enabled it is forwarded on the router.
    Setting<bool> disable_https{false};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<NetworkSettings>("connected_to_network",
                                           &NetworkSettings::connected_to_network),
            make_override<NetworkSettings>("shad_net_enabled", &NetworkSettings::shad_net_enabled),
            make_override<NetworkSettings>("shadnet_server", &NetworkSettings::shadnet_server),
            make_override<NetworkSettings>("shadnet_webapi_server",
                                           &NetworkSettings::shadnet_webapi_server),
            make_override<NetworkSettings>("signaling_info", &NetworkSettings::signaling_info),
            make_override<NetworkSettings>("enable_upnp", &NetworkSettings::enable_upnp),
            make_override<NetworkSettings>("p2p_port", &NetworkSettings::p2p_port),
            make_override<NetworkSettings>("disable_https", &NetworkSettings::disable_https)};
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NetworkSettings, connected_to_network, shad_net_enabled,
                                   shadnet_server, shadnet_webapi_server, signaling_info,
                                   enable_upnp, p2p_port, disable_https)

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
    // Handheld gyro fixes: a device held upright (Steam Deck, ROG Ally) measures the game's yaw
    // on its roll channel and vice versa; the inversions compose with the swap.
    Setting<bool> gyro_swap_yaw_roll{false}; // specific
    Setting<bool> gyro_invert_yaw{false};    // specific
    Setting<bool> gyro_invert_x{false};      // specific
    Setting<bool> gyro_invert_roll{false};   // specific
    Setting<bool> use_unified_input_config{true};
    Setting<std::string> default_controller_id{""};
    Setting<bool> background_controller_input{false}; // specific
    Setting<bool> ime_accessibility_enabled{false};   // specific
    Setting<bool> ime_url_mail_short_panel{false};    // specific
    Setting<bool> is_circle_enter{false};             // specific
    Setting<s32> camera_id{-1};
    Setting<bool> use_mice_as_mice{false};
    Setting<bool> use_keyboard_as_keyboard{false};

    std::vector<OverrideItem> GetOverrideableFields() const {
        return std::vector<OverrideItem>{
            make_override<InputSettings>("cursor_state", &InputSettings::cursor_state),
            make_override<InputSettings>("cursor_hide_timeout",
                                         &InputSettings::cursor_hide_timeout),
            make_override<InputSettings>("usb_device_backend", &InputSettings::usb_device_backend),
            make_override<InputSettings>("motion_controls_enabled",
                                         &InputSettings::motion_controls_enabled),
            make_override<InputSettings>("gyro_swap_yaw_roll", &InputSettings::gyro_swap_yaw_roll),
            make_override<InputSettings>("gyro_invert_yaw", &InputSettings::gyro_invert_yaw),
            make_override<InputSettings>("gyro_invert_x", &InputSettings::gyro_invert_x),
            make_override<InputSettings>("gyro_invert_roll", &InputSettings::gyro_invert_roll),
            make_override<InputSettings>("background_controller_input",
                                         &InputSettings::background_controller_input),
            make_override<InputSettings>("ime_accessibility_enabled",
                                         &InputSettings::ime_accessibility_enabled),
            make_override<InputSettings>("ime_url_mail_short_panel",
                                         &InputSettings::ime_url_mail_short_panel),
            make_override<InputSettings>("is_circle_enter", &InputSettings::is_circle_enter),
            make_override<InputSettings>("camera_id", &InputSettings::camera_id),
            make_override<InputSettings>("use_mice_as_mice", &InputSettings::use_mice_as_mice),
            make_override<InputSettings>("use_keyboard_as_keyboard",
                                         &InputSettings::use_keyboard_as_keyboard)};
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InputSettings, cursor_state, cursor_hide_timeout,
                                   usb_device_backend, use_special_pad, special_pad_class,
                                   motion_controls_enabled, gyro_swap_yaw_roll, gyro_invert_yaw,
                                   gyro_invert_x, gyro_invert_roll, use_unified_input_config,
                                   default_controller_id, background_controller_input,
                                   ime_accessibility_enabled, ime_url_mail_short_panel, camera_id,
                                   is_circle_enter, use_mice_as_mice, use_keyboard_as_keyboard)
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

// -------------------------------
// GPU settings
// -------------------------------
struct GPUSettings {
    Setting<u32> window_width{1280};  // 720p for 4GB VRAM
    Setting<u32> window_height{720};   // 720p for 4GB VRAM
    Setting<u32> internal_screen_width{1280};
    Setting<u32> internal_screen_height{720};
    Setting<bool> null_gpu{false};
    Setting<bool> copy_gpu_buffers{false};
    Setting<u32> readbacks_mode{GpuReadbacksMode::Relaxed};  // Optimized for 4GB VRAM (Shadlix)
    Setting<bool> readback_linear_images_enabled{false};
    // readback_linear_images_async: with readback_linear_images, a fence no longer waits for the
    // GPU. Each queued image is copied into its own staging buffer, and a background thread writes
    // it to guest memory once the GPU finishes, so the guest sees the pixels up to a frame late.
    // Nothing is protected or tracked and any readbacks_mode works. Suits values a game reads
    // every frame, like exposure and lighting, not one-off reads.
    Setting<bool> readback_linear_images_async{true};
    Setting<u32> adaptive_skipcaches_mode{AdaptiveSkipCachesMode::SkipCachesForced};
    // Image touches only stamp the per-image gc tick; the LRU list is relinked when the garbage
    // collector walk meets an entry touched since its list tick, so a hot image is relinked once
    // per ticks_to_destroy instead of once per tick.
    Setting<bool> texture_lru_lazy_touch{true};
    // GetProgram keeps the spec-key gather inputs (flat user data, pgm_base, RI hash, start
    // bindings) in the per-stage slot; a byte-identical repeat for the same program is a slot
    // hit without the gather. Needs spec_key_fused; off while spec_fp_validate is on.
    Setting<bool> gather_input_memo{true};
    // On the GPU command thread a contended tracker region lock is spun on (a
    // try_lock every 16 PAUSE) for up to this many rounds before blocking.
    // 0 keeps the plain blocking lock on every thread.
    Setting<u32> tracker_lock_spin_rounds{32};
    // Pin the GPU command thread to a physical core of its own (both hyperthreads) and strip
    // that core from every other thread of the process, guest threads included, re-walked every
    // 5 s. Without it the OS can park a busy thread on the command thread's sibling hyperthread,
    // which costs it a quarter to a third of its speed. Hosts with fewer than 6 logical CPUs or
    // 4 physical cores are left alone.
    Setting<bool> gpu_thread_core_reserve{true};
    // Windows only. Restrict the whole process to the first logical CPU of every physical core,
    // the same thing as ticking only the even CPUs in Task Manager: no two emulator or guest
    // threads can then share a core. Windows otherwise packs busy threads onto both hyperthreads
    // of a few cores while others idle. Works together with gpu_thread_core_reserve.
    Setting<bool> one_thread_per_core{false};
    // Rebuild the vertex input layout only when the pipeline, the instance step rates or an
    // attribute's format or stride changed, instead of on every draw.
    Setting<bool> vertex_layout_memo{true};
    // Skip adding a buffer range to the barrier lists when one recorded range already covers it.
    // The lists come out identical, without the vector shifts of the insert.
    Setting<bool> covered_range_skip{true};
    // Answer "already resident" from one bit per sparse block (2 MB, or 8 MB with 16 KB blocks)
    // instead of searching the resident range list on every buffer bind.
    Setting<bool> residency_bitmap{true};
    // Leave the stream buffer out of the barrier lists. Every access it reports to them is a
    // read, so none of its binds can need a barrier, yet each one was looked up in the lists and
    // recorded in them.
    Setting<bool> stream_barrier_skip{true};
    // Look at the CPU modified bits of a read-only bind before taking the tracker lock, and skip
    // the locked walk over them when none is set.
    Setting<bool> clean_sync_peek{true};
    // The GPU command thread submits a readback download without waiting for it; the faulting
    // thread waits and writes the bytes back. A GPU write marked in the window meanwhile keeps
    // the window GPU modified for the next fault.
    Setting<bool> readback_offload{false};
    // Flush the open graphics batch early when it already holds this many draws and every batch
    // submitted so far has retired (the ring runs dry while the rest of the batch is recorded).
    // Rounded up to a multiple of 32, and ignored unless flush_draw_interval is set larger than
    // the rounded value. 0 = off.
    Setting<u32> ring_drain_flush_draws{64};
    // While a deferred operation waits out its GPU tick, attempt the pending
    // pop (a lock plus a fence-query ioctl) once per this many draw-rate
    // polls instead of every draw. 0 polls every call.
    Setting<u32> pending_pop_throttle{64};
    // Stream copy lane mode. 0 disables it (copies stay inline on the GPU
    // command thread). 1 runs the unsafe fast path: no foreign-producer
    // refusal and no unmap push windows - only for titles that never unmap
    // mid-play. 2 runs the hardened path, safe everywhere. Both modes drain
    // through worker threads, two unless stream_copy_lane_threads says
    // otherwise, fenced before every submit.
    Setting<u32> stream_copy_workers{1};
    // Resolves shader permutations through an address-masked specialization
    // fingerprint: a hit skips the StageSpecialization rebuild and the deep
    // permutation compares entirely.
    Setting<bool> spec_fp_cache{true};
    // Skips the five dynamic-state updaters and their commit when the stamped
    // graphics registers, the pipeline and the dirty-bit re-arm generation all
    // match the previous draw's, which can set no bit the commit has not
    // already emitted.
    Setting<bool> dyn_state_memo{true};
    // Skip BuildRuntimeInfo and its fingerprint hash for the vertex and
    // fragment stages while the graphics register stamp is unchanged; those
    // two arms read only stamp-covered registers and boot constants.
    Setting<bool> runtime_info_stamp_gate{true};
    // Answer every occlusion query as fully occluded instead of fully
    // visible. Titles that gate effects on visibility (inFAMOUS lens flares)
    // then cull those draws themselves before submission.
    Setting<bool> occlude_all{false};
    // Flush the graphics command buffer every this many draws (0 = only at
    // submit-done and faults). A guest readback then waits on a command
    // buffer holding at most this many draws instead of the whole recorded
    // body. Values below 64 are raised to 64 (each flush costs a submit).
    Setting<u32> flush_draw_interval{384};
    // Reuse the previous graphics pipeline key while the register stamp
    // repeats: only the stage resolve reruns. Needs runtime_info_stamp_gate
    // and dynamic vertex input; otherwise the lookup runs unchanged.
    Setting<bool> pipeline_key_stamp_reuse{true};
    // Reuse the binary-info search result for a stage while its code pointer
    // and the hash stored inside the binary repeat.
    Setting<bool> shader_params_memo{true};
    // Specialization fingerprint over the sharp bits the specialization
    // reads: 1 keys the tier on it and carries the resolved module in the
    // MRU, 2 adds a per-stage slot answered by a memcmp. Needs spec_fp_cache.
    Setting<u32> spec_fp_canonical{2};
    // Hands a texture binding the view handle its FINDIMG memo hit recorded,
    // keyed on the image backing; the view record scan runs only on a miss.
    Setting<bool> texture_view_memo{true};
    // Compare and store descriptor writes into the delta slot in one walk
    // instead of serializing to a scratch buffer and comparing afterwards.
    Setting<bool> desc_delta_inplace{true};
    // Prefetch, during the first texture binding pass, the three image lines
    // the second pass reads first (props, backing pointer, backing state).
    // Read once at boot.
    Setting<bool> bind_line_prefetch{true};
    // Hold the guest-copy shared lock once per graphics packet run instead of
    // once per draw; the hold drops before every flush, GPU wait, command
    // drain and pipeline compile.
    Setting<bool> guest_copy_hold_segment{true};
    // A consumed image memo hit with an equal texture generation trusts the entry:
    // every register, unregister and slot delete bumps the generation, so the hit
    // skips the image record's uid check and re-touches the image once per GC tick
    // per entry.
    Setting<bool> findimg_trust_gen{true};
    // Each populated image memo entry records the T# range it answers for, and
    // RegisterImage/UnregisterImage clear only the entries their image intersects
    // instead of the whole memo riding a global texture generation. The guest-thread
    // unmap route and the two rebind arms keep a global invalidation.
    // Needs findimg_trust_gen.
    Setting<bool> findimg_range_invalidate{true};
    // Compare the gathered specialization key against its per-stage slot and
    // store it in one pass. Needs spec_fp_canonical 2.
    Setting<bool> spec_fp_slot_inplace{true};
    // A 16-entry associative front over each program's fingerprint table, for
    // programs that cycle through more specializations per frame than the MRU
    // pair holds. Needs spec_fp_canonical.
    Setting<bool> spec_fp_front{true};
    // Image memo geometry: 0 keeps the 1024-slot direct-mapped probe; 1, 2 or 4
    // index 2048 entries by every T# word into sets of that many ways with LRU
    // replacement. 3 acts as 2, higher values as 4.
    Setting<u32> findimg_memo_ways{4};
    // Entry count of the image memo; sets = entries / findimg_memo_ways. Clamped to
    // [1024, 32768] then rounded down to a power of two; 0 keeps 2048. Inert and
    // unreported at findimg_memo_ways 0.
    Setting<u32> findimg_memo_entries{4096};
    // Per texture memo entry, the backing epoch at which the shader-read
    // transit was a no-op and the layout it held; a repeat under that epoch skips
    // the transit probe and the backing's lines. Needs texture_view_memo.
    Setting<bool> bind_noop_memo{true};
    // Canonical specialization key layout: 1 starts every key word on an 8-byte
    // boundary so the in-place fold's loads forward from the gather's stores; 2
    // also warms the slot lines ahead of the gather. Higher values act as 2.
    Setting<u32> spec_key_fast{2};
    // Serve guest-visible backing writes from a per-thread memo of the last
    // resolved physical chunks, revalidated by the memory map generation; the
    // map descent runs only on a miss.
    Setting<bool> backing_write_memo{true};
    // Run the per-image fast-state check directly for sampled bindings instead
    // of the per-binding dedup probe. Needs image_fast_state. In Adaptive mode
    // the dedup cache cycles Learning/Off with no eligible calls; in ValidateOnly
    // its premise is no longer checked.
    Setting<bool> image_update_direct{true};
    // One descriptor set layout and pipeline layout per distinct binding list,
    // shared by every pipeline of that shape. Read once at boot.
    Setting<bool> desc_layout_share{true};
    // Per-stage two-entry memo of the register words the Vertex, Fragment and
    // Compute runtime-info builds read; an equal snapshot restores the struct
    // and its fingerprint hash wherever the rebuild runs.
    Setting<bool> runtime_info_input_memo{true};
    // Decides the stamp-keyed key reuse from a running XOR of the stage hashes
    // the resolve rewrites instead of re-reading the hash array it just stored.
    // Needs pipeline_key_stamp_reuse.
    Setting<bool> key_reuse_hash_diff{true};
    // Pushes only the descriptors whose bytes differ from the last push on the
    // same command buffer and layout; the rest stay as the driver holds them.
    // Needs desc_delta_inplace.
    Setting<bool> desc_delta_partial{true};
    // Direct-mapped table of that many entries behind the per-stage binary-info
    // memo, indexed by the code address, so the search's two lines are read
    // independently and the Vertex lines are prefetched ahead of the Fragment
    // resolve. Needs shader_params_memo; 0 keeps the single entry.
    Setting<u32> shader_params_memo_entries{1024};
    // Keys the dynamic-state memo on a stamp lane bumped only by the context and
    // uconfig registers its updaters read, and on the pipeline's write masks
    // instead of its identity. Needs dyn_state_memo.
    Setting<bool> dyn_state_stamp{true};
    // Bakes the color write mask into the pipeline's blend state instead of
    // declaring it dynamic. The mask is already a pipeline key field, so the
    // pipeline count is unchanged.
    Setting<bool> static_color_write_mask{true};
    // Gathers the canonical specialization key straight into its compare slot,
    // folding the compare into the gather's stores. Needs spec_fp_canonical 2,
    // spec_fp_slot_inplace and spec_key_fast.
    Setting<bool> spec_key_fused{true};
    // Runs consecutive register writes, padding and empty NOPs in a tight loop
    // inside the graphics packet parser, so a run of them takes one branch
    // pair instead of a trip through the far packet dispatch each.
    Setting<bool> parser_reg_run{true};
    // Skips a push constant update when the previous push on this command
    // buffer carried the same bytes with the same layout, so a run of draws
    // sharing one push block records one vkCmdPushConstants.
    Setting<bool> push_const_dedup{true};
    // Idle wait of a stream copy worker between drain attempts, in
    // microseconds. 0 keeps the pause spin; a positive value parks the worker
    // in a timed monitor wait on the publish word, which frees its core
    // sibling until a push or the timeout wakes it. Ignored without MWAITX.
    Setting<u32> stream_copy_idle_us{0};
    // Worker threads of the stream copy lane. 0 keeps the measured default of
    // two; 1 to 4 set the count directly.
    Setting<u32> stream_copy_lane_threads{4};
    // Answers a guest write fault against a lock-free coverage bitmap of the
    // registered images before the page table walk, so a fault in memory no
    // image covers skips the walk.
    Setting<bool> texture_invalidate_filter{true};
    // Keys the render-target memo and the render-scope cache on a stamp lane bumped only by
    // the CB/DB registers their bodies read, and on mrt_mask/color_samples instead of the
    // pipeline identity. Needs adaptive_skipcaches_mode != 0.
    Setting<bool> rt_state_stamp{true};
    // Rebuilds the four viewport push constants only when the register stamp lane moved and
    // clears only the push-constant prefixes the previous draw wrote. Needs
    // adaptive_skipcaches_mode != 0 (a dormant funnel would freeze the stamp).
    Setting<bool> push_vp_memo{true};
    // Folds the runtime-info snapshot's compare against the last memo entry into the snapshot
    // itself, so a hit needs no library memcmp. Needs runtime_info_input_memo.
    Setting<bool> ri_memo_fused_cmp{true};
    // Re-certifies the render-scope cache on a moved memory generation from each bound
    // attachment's image fast-state word (the word UpdateImage's no-op tier reads) instead
    // of rebuilding. Needs image_fast_state and adaptive_skipcaches_mode != 0.
    Setting<bool> br_mem_fast_state{true};
    // Hands a heap descriptor set out again once the tick that recorded it retired,
    // instead of allocating a fresh one per push and resetting whole pools.
    // Pools are never reset in this mode.
    Setting<bool> desc_heap_recycle{true};
    // Lets a pipeline whose set-0 descriptor total equals maxPushDescriptors use push
    // descriptors; the limit is inclusive.
    Setting<bool> push_desc_full_limit{true};
    // 0 off, 1 the per-pipeline descriptor write plan in place of the per-bind rebuild,
    // 2 or more the plan built and compared with the rebuilt list (shadow).
    Setting<u32> bind_write_plan{1};
    // Probes the image memo before validating the T#; the validation runs only on the
    // routes that reach the full lookup, where every memo entry was populated from.
    Setting<bool> findimg_memo_first{true};
    // Per pipeline, the image memo slot each image binding last matched; the probe
    // compares that entry before the hashed way scan. Never a certificate: the
    // entry must pass the full key compare and the generation checks as before.
    Setting<bool> findimg_slot_hint{true};
    // Deferred image bindings prime the fields the memo probe reads in place
    // instead of running the full ImageDesc constructor; the probe writes every
    // field pass two reads on all of its exits. Needs bind_noop_memo.
    Setting<bool> bind_image_lean{true};
    // Under a bind_write_plan hit the descriptor delta compares the two info arrays the
    // plan tiles, in 24-byte descriptors, and compacts from a change mask; needs
    // desc_delta_inplace and desc_layout_share.
    Setting<bool> desc_delta_flat{true};
    // One certificate for the three per-draw memos on the all-hits path:
    // 0 off, 1 fold the probes, 2 fold + scope serial (behaves as 1 until that
    // leg lands), 3 shadow; values above 3 clamp to 3. Needs
    // adaptive_skipcaches_mode 2; the boot latch turns it off otherwise.
    Setting<u32> draw_glue_memo{1};
    // Collapses the clean steady state of per-binding texture updates to one
    // atomic load instead of the touch, track and refresh pass; every
    // dirtying path stamps the per-image word back to dirty.
    Setting<bool> image_fast_state{true};
    // Holds the memory map's shared lock across a whole buffer-bind batch so
    // each guest copy inside stops paying its own pair of contended atomic
    // lock operations.
    Setting<bool> guest_copy_lock_batch{true};
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
    Setting<bool> fsr_enabled{true};  // Optimized for 4GB VRAM (FSR upscaling)
    Setting<bool> rcas_enabled{true};
    Setting<int> rcas_attenuation{250};
    Setting<bool> userfaultfd{false};
    Setting<bool> inline_fetch_shader{false};

#define GPU_OVERRIDE(field) make_override<GPUSettings>(#field, &GPUSettings::field)
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
            make_override<GPUSettings>("readback_linear_images_async",
                                       &GPUSettings::readback_linear_images_async),
            GPU_OVERRIDE(adaptive_skipcaches_mode),
            GPU_OVERRIDE(texture_lru_lazy_touch),
            GPU_OVERRIDE(gather_input_memo),
            GPU_OVERRIDE(tracker_lock_spin_rounds),
            GPU_OVERRIDE(gpu_thread_core_reserve),
            GPU_OVERRIDE(one_thread_per_core),
            GPU_OVERRIDE(vertex_layout_memo),
            GPU_OVERRIDE(covered_range_skip),
            GPU_OVERRIDE(residency_bitmap),
            GPU_OVERRIDE(stream_barrier_skip),
            GPU_OVERRIDE(clean_sync_peek),
            GPU_OVERRIDE(readback_offload),
            GPU_OVERRIDE(ring_drain_flush_draws),
            GPU_OVERRIDE(pending_pop_throttle),
            GPU_OVERRIDE(stream_copy_workers),
            GPU_OVERRIDE(spec_fp_cache),
            GPU_OVERRIDE(dyn_state_memo),
            GPU_OVERRIDE(runtime_info_stamp_gate),
            GPU_OVERRIDE(occlude_all),
            GPU_OVERRIDE(flush_draw_interval),
            GPU_OVERRIDE(pipeline_key_stamp_reuse),
            GPU_OVERRIDE(shader_params_memo),
            GPU_OVERRIDE(spec_fp_canonical),
            GPU_OVERRIDE(texture_view_memo),
            GPU_OVERRIDE(desc_delta_inplace),
            GPU_OVERRIDE(bind_line_prefetch),
            GPU_OVERRIDE(guest_copy_hold_segment),
            GPU_OVERRIDE(findimg_trust_gen),
            GPU_OVERRIDE(findimg_range_invalidate),
            GPU_OVERRIDE(spec_fp_slot_inplace),
            GPU_OVERRIDE(spec_fp_front),
            GPU_OVERRIDE(findimg_memo_ways),
            GPU_OVERRIDE(findimg_memo_entries),
            GPU_OVERRIDE(bind_noop_memo),
            GPU_OVERRIDE(spec_key_fast),
            GPU_OVERRIDE(backing_write_memo),
            GPU_OVERRIDE(image_update_direct),
            GPU_OVERRIDE(desc_layout_share),
            GPU_OVERRIDE(runtime_info_input_memo),
            GPU_OVERRIDE(key_reuse_hash_diff),
            GPU_OVERRIDE(desc_delta_partial),
            GPU_OVERRIDE(shader_params_memo_entries),
            GPU_OVERRIDE(dyn_state_stamp),
            GPU_OVERRIDE(static_color_write_mask),
            GPU_OVERRIDE(spec_key_fused),
            GPU_OVERRIDE(parser_reg_run),
            GPU_OVERRIDE(push_const_dedup),
            GPU_OVERRIDE(stream_copy_idle_us),
            GPU_OVERRIDE(stream_copy_lane_threads),
            GPU_OVERRIDE(texture_invalidate_filter),
            GPU_OVERRIDE(rt_state_stamp),
            GPU_OVERRIDE(push_vp_memo),
            GPU_OVERRIDE(ri_memo_fused_cmp),
            GPU_OVERRIDE(br_mem_fast_state),
            GPU_OVERRIDE(desc_heap_recycle),
            GPU_OVERRIDE(push_desc_full_limit),
            GPU_OVERRIDE(bind_write_plan),
            GPU_OVERRIDE(findimg_memo_first),
            GPU_OVERRIDE(findimg_slot_hint),
            GPU_OVERRIDE(bind_image_lean),
            GPU_OVERRIDE(desc_delta_flat),
            GPU_OVERRIDE(draw_glue_memo),
            GPU_OVERRIDE(image_fast_state),
            GPU_OVERRIDE(guest_copy_lock_batch),
            GPU_OVERRIDE(spec_mru_perm_probe),
            make_override<GPUSettings>("direct_memory_access_enabled",
                                       &GPUSettings::direct_memory_access_enabled),
            make_override<GPUSettings>("vblank_frequency", &GPUSettings::vblank_frequency),
            make_override<GPUSettings>("userfaultfd", &GPUSettings::userfaultfd),
            make_override<GPUSettings>("inline_fetch_shader", &GPUSettings::inline_fetch_shader),
        };
    }
#undef GPU_OVERRIDE
};
// nlohmann's field macros take at most 63 names, so the GPU settings are
// serialized in two groups; new settings go at the end of the second.
// clang-format off
#define GPU_SETTINGS_JSON_FIELDS_A \
    window_width, window_height, internal_screen_width, internal_screen_height, \
    null_gpu, copy_gpu_buffers, readbacks_mode, readback_linear_images_enabled, \
    adaptive_skipcaches_mode, texture_lru_lazy_touch, gather_input_memo, \
    tracker_lock_spin_rounds, ring_drain_flush_draws, direct_memory_access_enabled, \
    dump_shaders, patch_shaders, vblank_frequency, full_screen, full_screen_mode, \
    present_mode, hdr_allowed, fsr_enabled, rcas_enabled, rcas_attenuation, \
    spec_mru_perm_probe, spec_fp_canonical, texture_view_memo, \
    desc_delta_inplace, bind_line_prefetch, guest_copy_hold_segment, \
    findimg_trust_gen, \
    findimg_range_invalidate, spec_fp_slot_inplace, spec_fp_front, findimg_memo_ways, \
    findimg_memo_entries, bind_noop_memo, spec_key_fast, backing_write_memo, \
    image_update_direct, desc_layout_share, runtime_info_input_memo
#define GPU_SETTINGS_JSON_FIELDS_B \
    key_reuse_hash_diff, desc_delta_partial, shader_params_memo_entries, \
    dyn_state_stamp, static_color_write_mask, spec_key_fused, \
    parser_reg_run, push_const_dedup, stream_copy_idle_us, stream_copy_lane_threads, \
    texture_invalidate_filter, rt_state_stamp, push_vp_memo, ri_memo_fused_cmp, \
    br_mem_fast_state, desc_heap_recycle, push_desc_full_limit, occlude_all, \
    flush_draw_interval, pipeline_key_stamp_reuse, shader_params_memo, \
    pending_pop_throttle, stream_copy_workers, dyn_state_memo, bind_write_plan, \
    findimg_memo_first, findimg_slot_hint, bind_image_lean, desc_delta_flat, \
    draw_glue_memo, image_fast_state, guest_copy_lock_batch, spec_fp_cache, \
    runtime_info_stamp_gate, userfaultfd, gpu_thread_core_reserve, one_thread_per_core, \
    vertex_layout_memo, covered_range_skip, residency_bitmap, \
    readback_linear_images_async, inline_fetch_shader, stream_barrier_skip, \
    clean_sync_peek, readback_offload
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
    Setting<bool> vkvalidation_core_enabled{false};  // Disabled for performance (4GB VRAM)
    Setting<bool> vkvalidation_sync_enabled{false};
    Setting<bool> vkvalidation_gpu_enabled{false};
    Setting<bool> vkcrash_diagnostic_enabled{false};
    Setting<bool> vkhost_markers{false};
    Setting<bool> vkguest_markers{false};
    Setting<bool> pipeline_cache_enabled{true};
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
    NetworkSettings m_network{};
    LogSettings m_log{};
    DebugSettings m_debug{};
    InputSettings m_input{};
    AudioSettings m_audio{};
    GPUSettings m_gpu{};
    VulkanSettings m_vulkan{};
    ConfigMode m_configMode{ConfigMode::Default};

    // Runtime-only override: when true, IsShadNetEnabled() reports false for the
    // rest of this run regardless of the persisted setting
    std::atomic<bool> m_shadnet_session_disabled{false};

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

    // Write all overrideable fields from group into out (for game-specific save). A field that
    // equals the group's base value is written as null, which removes its key in the merge.
    template <typename Group>
    static void SaveGroupGameSpecific(const Group& group, nlohmann::json& out) {
        const nlohmann::json global = group;
        for (auto& item : group.GetOverrideableFields()) {
            out[item.key] = item.get_for_save(&group);
            if (out[item.key] == global.at(item.key)) {
                out[item.key] = nullptr;
            }
        }
    }

    // Discard every game-specific override in group.
    template <typename Group>
    static void ClearGroupOverrides(Group& group) {
        for (auto& item : group.GetOverrideableFields())
            item.reset_game_specific(&group);
    }

    // Call fn on the group stored under the given config section and return its result, or a
    // value-initialized result for an unknown section.
    template <typename Self, typename Fn>
    static auto VisitGroup(Self& self, std::string_view section, Fn&& fn) {
        if (section == "General")
            return fn(self.m_general);
        if (section == "Network")
            return fn(self.m_network);
        if (section == "Log")
            return fn(self.m_log);
        if (section == "Debug")
            return fn(self.m_debug);
        if (section == "Input")
            return fn(self.m_input);
        if (section == "Audio")
            return fn(self.m_audio);
        if (section == "GPU")
            return fn(self.m_gpu);
        if (section == "Vulkan")
            return fn(self.m_vulkan);
        return decltype(fn(self.m_general)){};
    }

    static void PrintChangedSummary(const std::vector<std::string>& changed);
    void MigrateNetworkKeys(const nlohmann::json& general, const nlohmann::json& network);
    static void SyncLegacyNetworkKeys(const nlohmann::json& old_general, nlohmann::json& general,
                                      const nlohmann::json& network);

public:
    // Add these getters to access overrideable fields
    std::vector<OverrideItem> GetGeneralOverrideableFields() const {
        return m_general.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetNetworkOverrideableFields() const {
        return m_network.GetOverrideableFields();
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
    std::vector<OverrideItem> GetGPUOverrideableFields() const {
        return m_gpu.GetOverrideableFields();
    }
    std::vector<OverrideItem> GetVulkanOverrideableFields() const {
        return m_vulkan.GetOverrideableFields();
    }
    std::vector<std::string> GetAllOverrideableKeys() const;

    /// Return the fields of a config section ("GPU", "Input", ...) by JSON key, each with its
    /// value in the current config mode. Null for an unknown section.
    nlohmann::json GetGroupValues(std::string_view section) const;
    /// Write the given keys of a config section as global values, or as per-game overrides when
    /// specific is set. Keys that cannot be written this way (unknown, or not overrideable for a
    /// per-game write) are skipped, and a value of the wrong JSON type rejects the whole call.
    /// Returns whether every given key was applied.
    bool SetGroupValues(std::string_view section, const nlohmann::json& values, bool specific);

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
        return m_network.shad_net_enabled.get(m_configMode) &&
               !m_shadnet_session_disabled.load(std::memory_order_relaxed);
    }
    void SetShadNetEnabled(bool v, bool specific = false) {
        m_network.shad_net_enabled.set(v, specific);
    }
    bool IsShadNetEnabledSetting() const {
        return m_network.shad_net_enabled.get(m_configMode);
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
    SETTING_FORWARD_BOOL(m_general, DiscordRPCEnabled, discord_rpc_enabled)
    SETTING_FORWARD_BOOL(m_general, ShowFpsCounter, show_fps_counter)
    SETTING_FORWARD(m_general, ConsoleLanguage, console_language)
    SETTING_FORWARD(m_general, BigPictureScale, big_picture_scale)
    SETTING_FORWARD(m_general, BigPictureFolderDepth, big_picture_folder_depth)
    SETTING_FORWARD_BOOL(m_general, RedZonePatchingEnabled, redzone_patches)

    // Network settings
    SETTING_FORWARD_BOOL(m_network, ConnectedToNetwork, connected_to_network)
    SETTING_FORWARD(m_network, ShadNetServer, shadnet_server)
    SETTING_FORWARD(m_network, ShadNetWebApiServer, shadnet_webapi_server)
    SETTING_FORWARD(m_network, SignalingInfo, signaling_info)
    SETTING_FORWARD_BOOL(m_network, UPnPEnabled, enable_upnp)
    SETTING_FORWARD(m_network, P2PPort, p2p_port)
    SETTING_FORWARD_BOOL(m_network, ForcedHttpsDisabled, disable_https)

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

    // Debug settings
    SETTING_FORWARD_BOOL(m_debug, DebugDump, debug_dump)
    SETTING_FORWARD_BOOL(m_debug, ShaderCollect, shader_collect)
    SETTING_FORWARD(m_debug, ConfigVersion, config_version)

    // GPU Settings
    SETTING_FORWARD(m_gpu, AdaptiveSkipCachesMode, adaptive_skipcaches_mode)
    SETTING_FORWARD_BOOL(m_gpu, TextureLruLazyTouch, texture_lru_lazy_touch)
    SETTING_FORWARD_BOOL(m_gpu, GatherInputMemo, gather_input_memo)
    SETTING_FORWARD(m_gpu, TrackerLockSpinRounds, tracker_lock_spin_rounds)
    SETTING_FORWARD_BOOL(m_gpu, GpuThreadCoreReserve, gpu_thread_core_reserve)
    SETTING_FORWARD_BOOL(m_gpu, OneThreadPerCore, one_thread_per_core)
    SETTING_FORWARD_BOOL(m_gpu, VertexLayoutMemo, vertex_layout_memo)
    SETTING_FORWARD_BOOL(m_gpu, CoveredRangeSkip, covered_range_skip)
    SETTING_FORWARD_BOOL(m_gpu, ResidencyBitmap, residency_bitmap)
    SETTING_FORWARD_BOOL(m_gpu, StreamBarrierSkip, stream_barrier_skip)
    SETTING_FORWARD_BOOL(m_gpu, CleanSyncPeek, clean_sync_peek)
    SETTING_FORWARD_BOOL(m_gpu, ReadbackOffload, readback_offload)
    SETTING_FORWARD(m_gpu, RingDrainFlushDraws, ring_drain_flush_draws)
    SETTING_FORWARD(m_gpu, PendingPopThrottle, pending_pop_throttle)
    SETTING_FORWARD(m_gpu, StreamCopyWorkers, stream_copy_workers)
    SETTING_FORWARD_BOOL(m_gpu, SpecFpCache, spec_fp_cache)
    SETTING_FORWARD_BOOL(m_gpu, DynStateMemo, dyn_state_memo)
    SETTING_FORWARD_BOOL(m_gpu, RuntimeInfoStampGate, runtime_info_stamp_gate)
    SETTING_FORWARD_BOOL(m_gpu, OccludeAll, occlude_all)
    SETTING_FORWARD(m_gpu, FlushDrawInterval, flush_draw_interval)
    SETTING_FORWARD_BOOL(m_gpu, PipelineKeyStampReuse, pipeline_key_stamp_reuse)
    SETTING_FORWARD_BOOL(m_gpu, ShaderParamsMemo, shader_params_memo)
    SETTING_FORWARD(m_gpu, SpecFpCanonical, spec_fp_canonical)
    SETTING_FORWARD_BOOL(m_gpu, TextureViewMemo, texture_view_memo)
    SETTING_FORWARD_BOOL(m_gpu, DescDeltaInplace, desc_delta_inplace)
    SETTING_FORWARD_BOOL(m_gpu, BindLinePrefetch, bind_line_prefetch)
    SETTING_FORWARD_BOOL(m_gpu, GuestCopyHoldSegment, guest_copy_hold_segment)
    SETTING_FORWARD_BOOL(m_gpu, FindimgTrustGen, findimg_trust_gen)
    SETTING_FORWARD_BOOL(m_gpu, FindimgRangeInvalidate, findimg_range_invalidate)
    SETTING_FORWARD_BOOL(m_gpu, SpecFpSlotInplace, spec_fp_slot_inplace)
    SETTING_FORWARD_BOOL(m_gpu, SpecFpFront, spec_fp_front)
    SETTING_FORWARD(m_gpu, FindimgMemoWays, findimg_memo_ways)
    SETTING_FORWARD(m_gpu, FindimgMemoEntries, findimg_memo_entries)
    SETTING_FORWARD_BOOL(m_gpu, BindNoopMemo, bind_noop_memo)
    SETTING_FORWARD(m_gpu, SpecKeyFast, spec_key_fast)
    SETTING_FORWARD_BOOL(m_gpu, BackingWriteMemo, backing_write_memo)
    SETTING_FORWARD_BOOL(m_gpu, ImageUpdateDirect, image_update_direct)
    SETTING_FORWARD_BOOL(m_gpu, DescLayoutShare, desc_layout_share)
    SETTING_FORWARD_BOOL(m_gpu, RuntimeInfoInputMemo, runtime_info_input_memo)
    SETTING_FORWARD_BOOL(m_gpu, KeyReuseHashDiff, key_reuse_hash_diff)
    SETTING_FORWARD_BOOL(m_gpu, DescDeltaPartial, desc_delta_partial)
    SETTING_FORWARD(m_gpu, ShaderParamsMemoEntries, shader_params_memo_entries)
    SETTING_FORWARD_BOOL(m_gpu, DynStateStamp, dyn_state_stamp)
    SETTING_FORWARD_BOOL(m_gpu, StaticColorWriteMask, static_color_write_mask)
    SETTING_FORWARD_BOOL(m_gpu, SpecKeyFused, spec_key_fused)
    SETTING_FORWARD_BOOL(m_gpu, ParserRegRun, parser_reg_run)
    SETTING_FORWARD_BOOL(m_gpu, PushConstDedup, push_const_dedup)
    SETTING_FORWARD(m_gpu, StreamCopyIdleUs, stream_copy_idle_us)
    SETTING_FORWARD(m_gpu, StreamCopyLaneThreads, stream_copy_lane_threads)
    SETTING_FORWARD_BOOL(m_gpu, TextureInvalidateFilter, texture_invalidate_filter)
    SETTING_FORWARD_BOOL(m_gpu, RtStateStamp, rt_state_stamp)
    SETTING_FORWARD_BOOL(m_gpu, PushVpMemo, push_vp_memo)
    SETTING_FORWARD_BOOL(m_gpu, RiMemoFusedCmp, ri_memo_fused_cmp)
    SETTING_FORWARD_BOOL(m_gpu, BrMemFastState, br_mem_fast_state)
    SETTING_FORWARD_BOOL(m_gpu, DescHeapRecycle, desc_heap_recycle)
    SETTING_FORWARD_BOOL(m_gpu, PushDescFullLimit, push_desc_full_limit)
    SETTING_FORWARD(m_gpu, BindWritePlan, bind_write_plan)
    SETTING_FORWARD_BOOL(m_gpu, FindimgMemoFirst, findimg_memo_first)
    SETTING_FORWARD_BOOL(m_gpu, FindimgSlotHint, findimg_slot_hint)
    SETTING_FORWARD_BOOL(m_gpu, BindImageLean, bind_image_lean)
    SETTING_FORWARD_BOOL(m_gpu, DescDeltaFlat, desc_delta_flat)
    SETTING_FORWARD(m_gpu, DrawGlueMemo, draw_glue_memo)
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
    SETTING_FORWARD_BOOL(m_gpu, ReadbackLinearImagesAsync, readback_linear_images_async)
    SETTING_FORWARD_BOOL(m_gpu, DirectMemoryAccessEnabled, direct_memory_access_enabled)
    SETTING_FORWARD_BOOL_READONLY(m_gpu, PatchShaders, patch_shaders)
    SETTING_FORWARD_BOOL(m_gpu, UserfaultfdTracking, userfaultfd)
    SETTING_FORWARD_BOOL_READONLY(m_gpu, InlineFetchShader, inline_fetch_shader)

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
    SETTING_FORWARD_BOOL(m_input, GyroSwapYawRoll, gyro_swap_yaw_roll)
    SETTING_FORWARD_BOOL(m_input, GyroInvertYaw, gyro_invert_yaw)
    SETTING_FORWARD_BOOL(m_input, GyroInvertX, gyro_invert_x)
    SETTING_FORWARD_BOOL(m_input, GyroInvertRoll, gyro_invert_roll)
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
    SETTING_FORWARD_BOOL(m_input, KeyboardUsedAsKeyboard, use_keyboard_as_keyboard)

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
