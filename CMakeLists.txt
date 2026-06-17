# SPDX-License-Identifier: GPL-2.0-or-later
#
# Combined shadPS4 — two cores, one binary.
#
#   core_gr2  : the editable gr2fork (all features + ported accuracy work)
#   core_main : the byte-untouched prerelease (6a20eba), a clean A/B reference
#
# This combined repo is shaped like a normal shadPS4 repo, except this umbrella
# IS the root CMakeLists and there are TWO source folders side by side:
#
#   <root>/CMakeLists.txt   this file (the only top-level CMakeLists)
#   <root>/cmake/           SHARED cmake modules (CMakeRC, Find*, …) + Combine.cmake
#   <root>/externals/       SHARED vendored deps (identical between trees) — built ONCE
#   <root>/src/             prerelease SOURCE (pure source, no CMakeLists)  -> core_main
#   <root>/src-gr2/         gr2fork SOURCE    (pure source, no CMakeLists)  -> core_gr2
#   <root>/dispatcher/      the selector/loader exe
#
# Each tree becomes its own -fvisibility=hidden shared object exporting one
# symbol (core_entry). A tiny dispatcher loads exactly one per run with
# RTLD_LOCAL, so the two identical symbol sets never collide.
#
# Dev   (default)                    : cores are loose libcore_*.so next to the exe.
# Release -DCOMBINED_EMBED_CORES=ON  : cores are .incbin-embedded -> one file.

cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Windows: the .incbin embed stub AND the cores' GAS-syntax fiber/stack .s/.S must
# be assembled by clang's integrated assembler, NOT MASM -- ml64 has no .incbin and
# cannot parse AT&T syntax. CMake defaults the plain ASM language to MASM on Windows,
# so pin the ASM compiler to clang-cl BEFORE project() enables ASM. Guard with
# CMAKE_HOST_WIN32 (the only platform var available this early) and honor an explicit
# -DCMAKE_ASM_COMPILER override (the CI passes one).
if(CMAKE_HOST_WIN32 AND NOT DEFINED CMAKE_ASM_COMPILER)
    set(CMAKE_ASM_COMPILER clang-cl)
endif()

project(shadPS4_combined CXX C ASM)

option(ENABLE_DISCORD_RPC     "Enable the Discord RPC integration" ON)
option(ENABLE_UPDATER         "Enable updater options"             OFF)
option(ENABLE_USERFAULTFD     "Use userfaultfd on Linux"           OFF)
option(COMBINED_EMBED_CORES   "Embed both cores into one self-contained binary" OFF)

# The two SOURCE folders (pure source: common/, core/, main.cpp, … — no CMakeLists),
# plus the shared externals/ and cmake/ that both cores use.
set(SRC_GR2  ${CMAKE_CURRENT_SOURCE_DIR}/src-gr2)        # gr2fork source     -> core_gr2
set(SRC_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/src)            # prerelease source  -> core_main
set(COMBINED_EXTERNALS ${CMAKE_CURRENT_SOURCE_DIR}/externals)
set(COMBINED_CMAKE     ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
foreach(s ${SRC_GR2} ${SRC_MAIN})
    if(NOT EXISTS ${s}/main.cpp)
        message(FATAL_ERROR
            "Expected a shadPS4 SOURCE tree at ${s} (with ${s}/main.cpp, common/, core/, …). "
            "This combined repo holds two source folders side by side: gr2fork -> src-gr2 , "
            "prerelease -> src ; plus the shared externals/ and cmake/.")
    endif()
endforeach()
if(NOT EXISTS ${COMBINED_EXTERNALS}/CMakeLists.txt)
    message(FATAL_ERROR "Missing shared externals/ at ${COMBINED_EXTERNALS} "
                        "(copy one tree's externals/ here; the two are identical).")
endif()

# --- PIE: keep the base address high so it doesn't clash with PS4 guest memory.
if(UNIX AND NOT APPLE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# --- Windows single-file self-containment. The chosen core DLL is extracted to
#     %TEMP% with NO sidecar beside it, so it must carry all its deps -- exactly
#     like the Linux .so statically links externals. Both knobs are set here,
#     before any target (and before add_subdirectory(externals)), so they seed the
#     externals and the cores:
#       * static CRT (/MT) -> no vcruntime/msvcp DLLs needed beside the temp core.
#         Seeds each target's MSVC_RUNTIME_LIBRARY (CMP0091 is NEW; we require >= 3.24).
#       * BUILD_SHARED_LIBS OFF -> externals build as static libs baked into the core,
#         not separate DLLs. (This only affects externals that don't force their own
#         type; the cores stay SHARED via their explicit add_library(... SHARED ...),
#         and matches upstream's lone-shadPS4.exe Windows artifact.)
#     Two static CRTs across the core_entry boundary is safe: only int/char** cross
#     it (raw pointers the core never frees with the dispatcher's CRT), the same
#     isolation the Linux RTLD_LOCAL dlopen boundary already relies on. Verify with
#     `dumpbin /dependents libcore_gr2.dll` -> only system DLLs (kernel32, user32, …).
if(WIN32)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(BUILD_SHARED_LIBS OFF)
endif()

# --- Architecture variable that externals (ffmpeg-core, …) read. The trees set
#     this near the top BEFORE add_subdirectory(externals); the umbrella must too,
#     or ffmpeg-core fails with "ARCHITECTURE variable is not set up".
if(CMAKE_SYSTEM_PROCESSOR)
    set(BASE_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}")
else()
    set(BASE_ARCHITECTURE "${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()
if(BASE_ARCHITECTURE MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(ARCHITECTURE "x86_64")
elseif(BASE_ARCHITECTURE MATCHES "(aarch64)|(AARCH64)|(arm64)|(ARM64)")
    set(ARCHITECTURE "arm64")
else()
    message(FATAL_ERROR "Unsupported CPU architecture: ${BASE_ARCHITECTURE}")
endif()

# --- Architecture baseline. Applied ONLY if the caller passed no -march via
#     CMAKE_C/CXX_FLAGS (clang takes the LAST -march, so an unconditional one
#     here would silently override a caller's -march=znver4). Floor = x86-64-v3.
if(NOT (CMAKE_CXX_FLAGS MATCHES "-march=" OR CMAKE_C_FLAGS MATCHES "-march="))
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        add_compile_options(-march=x86-64-v3)
    endif()
endif()

# --- libc++ needs this for std::jthread / std::stop_token (skipped on libstdc++).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(_LIBCPP_VERSION version LIBCPP)
    if(LIBCPP)
        add_compile_options(-fexperimental-library)
    endif()
endif()

# ============================ PREAMBLE SLOT ====================================
# Reduced from the trees' top-level CMakeLists to the Linux/x86_64 essentials. If
# a build needs extra top-level flags (sanitizer/LTO toggles, etc.), paste them
# HERE — this runs before externals and before any core is added.
#   (intentionally empty by default)
# ===============================================================================

# --- Dependency discovery (mirrors the trees' find_package set). Find-modules
#     come from the SHARED cmake/ dir; externals/ supplies most via OVERRIDE_FIND_PACKAGE.
list(APPEND CMAKE_MODULE_PATH ${COMBINED_CMAKE})
find_package(Boost 1.84.0 CONFIG)
find_package(CLI11 2.6.1 CONFIG)
find_package(FFmpeg 5.1.2 MODULE)
find_package(fmt 10.2.0 CONFIG)
find_package(Freetype 2.14.1 MODULE)
find_package(glslang 15 CONFIG)
find_package(half 1.12.0 MODULE)
find_package(LibreSSL 4.3.1 MODULE)
find_package(magic_enum 0.9.7 CONFIG)
find_package(miniupnpc 2.3.3 MODULE)
find_package(miniz 3.1 CONFIG)
find_package(nlohmann_json 3.12 CONFIG)
find_package(OpenAL CONFIG)
find_package(PNG 1.6 MODULE)
find_package(RenderDoc 1.6.0 MODULE)
find_package(SDL3 3.1.2 CONFIG)
find_package(SDL3_mixer 2.8.1 CONFIG)
find_package(stb MODULE)
find_package(toml11 4.2.0 CONFIG)
find_package(tsl-robin-map 1.3.0 CONFIG)
find_package(VulkanHeaders 1.4.329 CONFIG)
find_package(VulkanMemoryAllocator 3.1.0 CONFIG)
find_package(xbyak 7.07 CONFIG)
find_package(xxHash 0.8.2 MODULE)
find_package(ZLIB 1.3 MODULE)
find_package(Zydis 5.0.0 CONFIG)
find_package(pugixml 1.14 CONFIG)
# (Most of these are also provided as targets by externals/ via OVERRIDE_FIND_PACKAGE;
#  the find_package calls mirror the trees' top-level set so OVERRIDE resolution fires.
#  date::date-tz + epoll-shim are Apple/FreeBSD-only in the trees — not needed on Linux,
#  which uses libstdc++'s std::chrono::time_zone — so they're intentionally absent.)

# ============================ DEPENDENCY STACK (ONCE) ==========================
# Built from the PRERELEASE's externals (the combined repo is based on the prerelease
# checkout). That set covers everything core_main needs and everything shared between
# the cores; core_gr2's one extra Linux dep (sdl3_mixer) is added just below.
add_subdirectory(${COMBINED_EXTERNALS} ${CMAKE_BINARY_DIR}/externals-build)

# Dear_ImGui (built once, shared) must compile against shadPS4's imgui_config.h, or
# the imgui library TUs get a different ImGui struct layout than the cores that
# include imgui headers -> ODR/ABI mismatch. The two trees' imgui_config.h are
# byte-identical, so one shared Dear_ImGui with one config is correct.
if(TARGET Dear_ImGui)
    target_compile_definitions(Dear_ImGui PRIVATE
        IMGUI_USER_CONFIG="${SRC_MAIN}/imgui/imgui_config.h")
endif()

# cmrc lives in the shared cmake/ dir; include it once.
include(${COMBINED_CMAKE}/CMakeRC.cmake)

# core_gr2 links SDL3_mixer, but the prerelease's externals/ omit sdl3_mixer (it's a
# gr2-only dependency). Build it here from the added submodule — minimal: core mixer
# only, every format decoder OFF, so no extra decoder submodules are needed. SDL_mixer
# consumes the SDL3 target built by externals above. If the submodule isn't present,
# stop with the exact command to add it (core_gr2 can't link without it).
if(NOT TARGET SDL3_mixer::SDL3_mixer)
    if(EXISTS ${COMBINED_EXTERNALS}/sdl3_mixer/CMakeLists.txt)
        set(SDLMIXER_INSTALL OFF CACHE BOOL "" FORCE)
        set(SDLMIXER_SAMPLES OFF CACHE BOOL "" FORCE)
        set(SDLMIXER_FLAC    OFF CACHE BOOL "" FORCE)
        set(SDLMIXER_MOD     OFF CACHE BOOL "" FORCE)
        set(SDLMIXER_MP3     OFF CACHE BOOL "" FORCE)
        set(SDLMIXER_MIDI    OFF CACHE BOOL "" FORCE)
        set(SDLMIXER_OPUS    OFF CACHE BOOL "" FORCE)
        set(SDLMIXER_VORBIS  OFF CACHE BOOL "" FORCE)
        set(SDLMIXER_WAVPACK OFF CACHE BOOL "" FORCE)
        set(SDLMIXER_GME     OFF CACHE BOOL "" FORCE)
        add_subdirectory(${COMBINED_EXTERNALS}/sdl3_mixer
            ${CMAKE_BINARY_DIR}/sdl3_mixer-build EXCLUDE_FROM_ALL SYSTEM)
    else()
        message(FATAL_ERROR
            "core_gr2 needs externals/sdl3_mixer, which the prerelease base omits. Add it:\n"
            "  git submodule add https://github.com/libsdl-org/SDL_mixer.git externals/sdl3_mixer\n"
            "  git -C externals/sdl3_mixer checkout 4182794ea45fe28568728670c6f1583855d0e85c\n"
            "  git submodule update --init --recursive externals/sdl3_mixer")
    endif()
endif()

# ============================ THE TWO CORES ===================================
include(${COMBINED_CMAKE}/Combine.cmake)

# Land the loose .so next to the dispatcher exe (dev mode dlopens by path).
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# --- core_gr2: the editable fork. ----------------------------------------------
add_core(
    NAME     gr2
    SRC      ${SRC_GR2}
    EXCLUDES core/libraries/screenshot_service/screenshot_service2.cpp
             shader_recompiler/frontend/translate/emit_spirv.cpp
    LIBS     SDL3_mixer::SDL3_mixer miniz)

# --- core_main: the byte-untouched prerelease. ---------------------------------
add_core(
    NAME     main
    SRC      ${SRC_MAIN}
    EXCLUDES core/file_sys/file.cpp
             core/libraries/network/net_obj.cpp
    LIBS     CLI11::CLI11 OpenAL::OpenAL Cpp_Httplib miniupnpc::miniupnpc
             fdk-aac miniz::miniz Freetype::Freetype ImGuiFileDialog
             minimp3 spdlog::spdlog)

# ============================ THE ONE DISPATCHER ==============================
add_executable(shadps4 dispatcher/main.cpp dispatcher/core_loader.cpp)
target_link_libraries(shadps4 PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(shadps4 PROPERTIES OUTPUT_NAME shadps4)
if(WIN32)
    # std::getenv in dispatcher/main.cpp trips MSVC's "deprecated" CRT warning; it
    # is the standard, correct call -- silence it. ${CMAKE_DL_LIBS} is empty on
    # Windows (LoadLibrary lives in kernel32, always linked), so the line above is a
    # harmless no-op there. A target with main() defaults to the console subsystem,
    # which is what the dispatcher's stderr diagnostics want.
    target_compile_definitions(shadps4 PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

if(COMBINED_EMBED_CORES)
    # Release: bake both .so into the dispatcher. file(GENERATE) resolves the
    # $<TARGET_FILE:...> genexes in the .S template to the real built paths.
    set(_embed_s ${CMAKE_BINARY_DIR}/embed_cores.generated.S)
    # COFF (Windows) and ELF (Linux) need different read-only sections and the ELF
    # build has a .note.GNU-stack line COFF rejects, so pick the matching template.
    # Both resolve the same $<TARGET_FILE:core_*> genexes (to .dll vs .so paths).
    if(WIN32)
        set(_embed_in ${COMBINED_CMAKE}/embed_cores_win.S.in)
    else()
        set(_embed_in ${COMBINED_CMAKE}/embed_cores.S.in)
    endif()
    file(GENERATE
        OUTPUT ${_embed_s}
        INPUT  ${_embed_in})
    set_source_files_properties(${_embed_s} PROPERTIES GENERATED ON LANGUAGE ASM)
    target_sources(shadps4 PRIVATE ${_embed_s})
    target_compile_definitions(shadps4 PRIVATE COMBINED_EMBED_CORES)
    add_dependencies(shadps4 core_gr2 core_main)
    message(STATUS "Combined build: RELEASE (single self-contained binary)")
else()
    add_dependencies(shadps4 core_gr2 core_main)
    message(STATUS "Combined build: DEV (loose libcore_*.so next to shadps4)")
endif()

install(TARGETS shadps4 RUNTIME DESTINATION .)
