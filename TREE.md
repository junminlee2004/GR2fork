# Combined shadPS4 — final tree (`combined/`)

Legend: `← …` = role of that entry · `(N .cpp, M .h)` = ordinary source leaves collapsed to counts · `◄ build-relevant` / `… (source leaves)` = elided ordinary files. `src/` and `src-gr2/` are the **standard shadPS4 source trees** (pure source, no root CMakeLists); only the build-relevant spots (the 2 nested CMakeLists, `scm_rev.cpp.in`, the 16 host-shaders, the fonts, the cmrc images) are expanded in full.

```
combined_final/
├── cmake/   ← SHARED: tree modules + the 3 umbrella files
│   ├── CMakeRC.cmake
│   ├── Combine.cmake
│   ├── FindFFmpeg.cmake
│   ├── FindLibreSSL.cmake
│   ├── FindRenderDoc.cmake
│   ├── Findhalf.cmake
│   ├── Findlibusb.cmake
│   ├── Findminiupnpc.cmake
│   ├── Findstb.cmake
│   ├── FindxxHash.cmake
│   ├── embed_cores.S.in
│   └── export.map
├── dispatcher/   ← the selector/loader exe
│   ├── core_entry.cpp
│   ├── core_loader.cpp
│   ├── core_loader.h
│   └── main.cpp
├── externals/   ← SHARED vendored deps (44 dep dirs + externals CMakeLists), built once
│   ├── CLI11/
│   ├── CMakeLists.txt
│   ├── ImGuiFileDialog/
│   ├── LibAtrac9/
│   ├── aacdec/
│   ├── cmake-modules/
│   ├── cpp-httplib/
│   ├── date/
│   ├── dear_imgui/
│   ├── discord-rpc/
│   ├── epoll-shim/
│   ├── ext-boost/
│   ├── ext-wepoll/
│   ├── ffmpeg-core/
│   ├── fmt/
│   ├── freetype/
│   ├── gcn/
│   ├── glslang/
│   ├── half/
│   ├── hwinfo/
│   ├── json/
│   ├── libpng/
│   ├── libressl/
│   ├── libusb/
│   ├── magic_enum/
│   ├── mesa-kosmickrisp/
│   ├── minimp3/
│   ├── miniupnp/
│   ├── miniz/
│   ├── openal-soft/
│   ├── pugixml/
│   ├── renderdoc/
│   ├── robin-map/
│   ├── sdl3/
│   ├── sirit/
│   ├── spdlog/
│   ├── stb/
│   ├── toml11/
│   ├── tracy/
│   ├── vma/
│   ├── vulkan-headers/
│   ├── xbyak/
│   ├── xxhash/
│   ├── zlib-ng/
│   └── zydis/
├── src/   ← prerelease SOURCE → core_main
│   ├── common/   (19 .cpp, 59 .h)
│   │   └── scm_rev.cpp.in   ◄ build-relevant
│   ├── core/   (225 .cpp, 268 .h, 2 other)
│   │   ├── libraries/kernel/threads/stack.S   ◄ build-relevant
│   │   └── devtools/help.txt   ◄ build-relevant
│   ├── images/
│   │   ├── big_picture/
│   │   │   ├── controller.png
│   │   │   ├── experimental.png
│   │   │   ├── folder.png
│   │   │   ├── graphics.png
│   │   │   ├── log.png
│   │   │   ├── profiles.png
│   │   │   ├── settings.png
│   │   │   └── trophy.png
│   │   ├── KBM.png
│   │   ├── about_icon.png
│   │   ├── bronze.png
│   │   ├── controller_icon.png
│   │   ├── discord.png
│   │   ├── discord.svg
│   │   ├── dump_icon.png
│   │   ├── exit_icon.png
│   │   ├── favorite_icon.png
│   │   ├── flag_china.png
│   │   ├── flag_eu.png
│   │   ├── flag_jp.png
│   │   ├── flag_unk.png
│   │   ├── flag_us.png
│   │   ├── flag_world.png
│   │   ├── folder_icon.png
│   │   ├── fullscreen_icon.png
│   │   ├── game_settings.png
│   │   ├── github.png
│   │   ├── github.svg
│   │   ├── gold.png
│   │   ├── grid_icon.png
│   │   ├── hotkey.png
│   │   ├── iconsize_icon.png
│   │   ├── keyboard_icon.png
│   │   ├── ko-fi.png
│   │   ├── ko-fi.svg
│   │   ├── list_icon.png
│   │   ├── list_mode_icon.png
│   │   ├── net.shadps4.shadPS4.svg
│   │   ├── pause_icon.png
│   │   ├── platinum.png
│   │   ├── play_icon.png
│   │   ├── ps4_controller.png
│   │   ├── refreshlist_icon.png
│   │   ├── restart_game_icon.png
│   │   ├── settings_icon.png
│   │   ├── shadPS4.icns
│   │   ├── shadps4.ico
│   │   ├── shadps4.png
│   │   ├── shadps4.svg
│   │   ├── silver.png
│   │   ├── stop_icon.png
│   │   ├── themes_icon.png
│   │   ├── trophy.wav
│   │   ├── trophy_icon.png
│   │   ├── update_icon.png
│   │   ├── utils_icon.png
│   │   ├── website.png
│   │   ├── website.svg
│   │   ├── youtube.png
│   │   └── youtube.svg
│   ├── imgui/
│   │   ├── big_picture/   (5 .cpp, 5 .h)
│   │   ├── renderer/
│   │   │   ├── fonts/
│   │   │   │   ├── NotoSans-Regular.ttf
│   │   │   │   ├── NotoSansArabic-Regular.ttf
│   │   │   │   ├── NotoSansCJK-Regular.ttc
│   │   │   │   ├── NotoSansSymbols2-Regular.ttf
│   │   │   │   ├── NotoSansThai-Regular.ttf
│   │   │   │   └── ProggyVector-Regular.ttf
│   │   │   ├── CMakeLists.txt
│   │   │   ├── font_data.cpp
│   │   │   ├── font_data.h
│   │   │   ├── font_stack.cpp
│   │   │   ├── font_stack.h
│   │   │   ├── imgui_core.cpp
│   │   │   ├── imgui_core.h
│   │   │   ├── imgui_impl_sdl3.cpp
│   │   │   ├── imgui_impl_sdl3.h
│   │   │   ├── imgui_impl_vulkan.cpp
│   │   │   ├── imgui_impl_vulkan.h
│   │   │   ├── texture_manager.cpp
│   │   │   └── texture_manager.h
│   │   └── … 2 .cpp, 6 .h (source leaves)
│   ├── input/   (3 .cpp, 3 .h)
│   ├── shader_recompiler/   (62 .cpp, 43 .h, 1 other)
│   ├── video_core/
│   │   ├── amdgpu/   (4 .cpp, 14 .h)
│   │   ├── buffer_cache/   (3 .cpp, 7 .h)
│   │   ├── host_shaders/
│   │   │   ├── detilers/
│   │   │   │   ├── display_micro_64bpp.comp
│   │   │   │   ├── macro_32bpp.comp
│   │   │   │   ├── macro_64bpp.comp
│   │   │   │   ├── macro_8bpp.comp
│   │   │   │   ├── micro_128bpp.comp
│   │   │   │   ├── micro_16bpp.comp
│   │   │   │   ├── micro_32bpp.comp
│   │   │   │   ├── micro_64bpp.comp
│   │   │   │   └── micro_8bpp.comp
│   │   │   ├── fsr/
│   │   │   │   ├── ffx_a.h
│   │   │   │   └── ffx_fsr1.h
│   │   │   ├── CMakeLists.txt
│   │   │   ├── StringShaderHeader.cmake
│   │   │   ├── color_to_ms_depth.frag
│   │   │   ├── fault_buffer_process.comp
│   │   │   ├── fs_tri.vert
│   │   │   ├── fsr.comp
│   │   │   ├── ms_image_blit.frag
│   │   │   ├── post_process.frag
│   │   │   ├── source_shader.h.in
│   │   │   └── tiling.comp
│   │   ├── renderer_vulkan/   (19 .cpp, 19 .h)
│   │   ├── texture_cache/   (8 .cpp, 10 .h)
│   │   └── … 3 .cpp, 4 .h (source leaves)
│   ├── shadps4.rc
│   └── … 3 .cpp, 2 .h, 1 other (source leaves)
├── src-gr2/   ← gr2fork SOURCE → core_gr2
│   ├── common/   (25 .cpp, 65 .h)
│   │   └── scm_rev.cpp.in   ◄ build-relevant
│   ├── core/   (208 .cpp, 246 .h, 2 other)
│   │   └── devtools/help.txt   ◄ build-relevant
│   ├── images/
│   │   ├── KBM.png
│   │   ├── about_icon.png
│   │   ├── bronze.png
│   │   ├── controller_icon.png
│   │   ├── discord.png
│   │   ├── discord.svg
│   │   ├── dump_icon.png
│   │   ├── exit_icon.png
│   │   ├── favorite_icon.png
│   │   ├── flag_china.png
│   │   ├── flag_eu.png
│   │   ├── flag_jp.png
│   │   ├── flag_unk.png
│   │   ├── flag_us.png
│   │   ├── flag_world.png
│   │   ├── folder_icon.png
│   │   ├── fullscreen_icon.png
│   │   ├── game_settings.png
│   │   ├── github.png
│   │   ├── github.svg
│   │   ├── gold.png
│   │   ├── grid_icon.png
│   │   ├── hotkey.png
│   │   ├── iconsize_icon.png
│   │   ├── keyboard_icon.png
│   │   ├── ko-fi.png
│   │   ├── ko-fi.svg
│   │   ├── list_icon.png
│   │   ├── list_mode_icon.png
│   │   ├── net.shadps4.shadPS4.svg
│   │   ├── pause_icon.png
│   │   ├── platinum.png
│   │   ├── play_icon.png
│   │   ├── ps4_controller.png
│   │   ├── refreshlist_icon.png
│   │   ├── restart_game_icon.png
│   │   ├── settings_icon.png
│   │   ├── shadPS4.icns
│   │   ├── shadps4.ico
│   │   ├── shadps4.png
│   │   ├── shadps4.svg
│   │   ├── silver.png
│   │   ├── stop_icon.png
│   │   ├── themes_icon.png
│   │   ├── trophy.wav
│   │   ├── trophy_icon.png
│   │   ├── update_icon.png
│   │   ├── utils_icon.png
│   │   ├── website.png
│   │   ├── website.svg
│   │   ├── youtube.png
│   │   └── youtube.svg
│   ├── imgui/
│   │   ├── renderer/
│   │   │   ├── fonts/
│   │   │   │   ├── NotoSansJP-Regular.ttf
│   │   │   │   └── ProggyVector-Regular.ttf
│   │   │   ├── CMakeLists.txt
│   │   │   ├── imgui_core.cpp
│   │   │   ├── imgui_core.h
│   │   │   ├── imgui_impl_sdl3.cpp
│   │   │   ├── imgui_impl_sdl3.h
│   │   │   ├── imgui_impl_vulkan.cpp
│   │   │   ├── imgui_impl_vulkan.h
│   │   │   ├── texture_manager.cpp
│   │   │   └── texture_manager.h
│   │   └── … 4 .h (source leaves)
│   ├── input/   (3 .cpp, 3 .h)
│   ├── shader_recompiler/   (62 .cpp, 44 .h, 1 other)
│   ├── video_core/
│   │   ├── amdgpu/   (5 .cpp, 17 .h)
│   │   ├── buffer_cache/   (3 .cpp, 7 .h)
│   │   ├── host_shaders/
│   │   │   ├── detilers/
│   │   │   │   ├── display_micro_64bpp.comp
│   │   │   │   ├── display_micro_64bpp_comp.h
│   │   │   │   ├── macro_32bpp.comp
│   │   │   │   ├── macro_32bpp_comp.h
│   │   │   │   ├── macro_64bpp.comp
│   │   │   │   ├── macro_64bpp_comp.h
│   │   │   │   ├── macro_8bpp.comp
│   │   │   │   ├── macro_8bpp_comp.h
│   │   │   │   ├── micro_128bpp.comp
│   │   │   │   ├── micro_128bpp_comp.h
│   │   │   │   ├── micro_16bpp.comp
│   │   │   │   ├── micro_16bpp_comp.h
│   │   │   │   ├── micro_32bpp.comp
│   │   │   │   ├── micro_32bpp_comp.h
│   │   │   │   ├── micro_64bpp.comp
│   │   │   │   ├── micro_64bpp_comp.h
│   │   │   │   ├── micro_8bpp.comp
│   │   │   │   └── micro_8bpp_comp.h
│   │   │   ├── fsr/
│   │   │   │   ├── ffx_a.h
│   │   │   │   └── ffx_fsr1.h
│   │   │   ├── CMakeLists.txt
│   │   │   ├── StringShaderHeader.cmake
│   │   │   ├── color_to_ms_depth.frag
│   │   │   ├── color_to_ms_depth_frag.h
│   │   │   ├── fault_buffer_process.comp
│   │   │   ├── fault_buffer_process_comp.h
│   │   │   ├── fs_tri.vert
│   │   │   ├── fs_tri_vert.h
│   │   │   ├── fsr.comp
│   │   │   ├── fsr_comp.h
│   │   │   ├── ms_image_blit.frag
│   │   │   ├── ms_image_blit_frag.h
│   │   │   ├── post_process.frag
│   │   │   ├── post_process_frag.h
│   │   │   ├── source_shader.h.in
│   │   │   ├── tiling.comp
│   │   │   └── tiling_comp.h
│   │   ├── renderer_vulkan/   (22 .cpp, 27 .h)
│   │   ├── texture_cache/   (8 .cpp, 10 .h)
│   │   └── … 3 .cpp, 6 .h (source leaves)
│   ├── shadps4.rc
│   └── … 3 .cpp, 3 .h, 1 other (source leaves)
├── CMakeLists.txt
├── README_COMBINED.md
└── verify_mechanism.sh
