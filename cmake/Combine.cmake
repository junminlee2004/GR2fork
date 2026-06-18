# SPDX-License-Identifier: GPL-2.0-or-later
#
# Combine.cmake — turns one shadPS4 SOURCE tree into an isolated core shared
# library (libcore_<name>.so) that exports exactly one symbol: core_entry.
#
# Layout (combined repo is shaped like a shadPS4 repo):
#   <root>/CMakeLists.txt   the umbrella (this is the only top-level CMakeLists)
#   <root>/cmake/           SHARED cmake modules (CMakeRC, Find*, …) + this file
#   <root>/externals/       SHARED vendored deps, built ONCE (identical between trees)
#   <root>/src/             prerelease SOURCE  -> core_main   (pure source, no CMakeLists)
#   <root>/src-gr2/         gr2fork SOURCE     -> core_gr2    (pure source, no CMakeLists)
#
# add_core() is given a SOURCE dir directly (e.g. <root>/src). The two cores share
# the one externals/ and the one cmake/. The dispatcher loads exactly one core per
# run with RTLD_LOCAL, so the two identical symbol sets never collide.
#
# Sub-targets the trees normally define with FIXED names (host_shaders,
# ImGui_Resources, embedded-resources) would collide if built for both source
# trees, so they're regenerated here under unique per-core names. The font-embed
# host tool (binary_to_compressed_c) is identical and built once.

include_guard(GLOBAL)

# 16 host-shader sources, identical in both trees. Explicit (not globbed) so the
# list matches what StringShaderHeader.cmake expects.
set(_COMBINE_SHADER_FILES
    detilers/display_micro_64bpp.comp
    detilers/macro_32bpp.comp
    detilers/macro_64bpp.comp
    detilers/macro_8bpp.comp
    detilers/micro_128bpp.comp
    detilers/micro_16bpp.comp
    detilers/micro_32bpp.comp
    detilers/micro_64bpp.comp
    detilers/micro_8bpp.comp
    color_to_ms_depth.frag
    ms_image_blit.frag
    fault_buffer_process.comp
    fs_tri.vert
    fsr.comp
    post_process.frag
    tiling.comp
)

# Build the font-embed host tool ONCE from the shared externals.
function(combine_build_fontembed)
    if(TARGET Dear_ImGui_FontEmbed)
        return()
    endif()
    add_executable(Dear_ImGui_FontEmbed
        ${COMBINED_EXTERNALS}/dear_imgui/misc/fonts/binary_to_compressed_c.cpp)
endfunction()

# host_shaders_<name>: regenerate shader headers into a per-core include dir.
# Mirrors <src>/video_core/host_shaders/CMakeLists.txt, target renamed.
function(combine_host_shaders name src out_var)
    set(shader_src ${src}/video_core/host_shaders)
    set(shader_inc ${CMAKE_BINARY_DIR}/core_${name}_gen/host_shaders/include)
    set(shader_dir ${shader_inc}/video_core/host_shaders)
    set(input_file ${shader_src}/source_shader.h.in)
    set(generator  ${shader_src}/StringShaderHeader.cmake)

    set(headers "")
    foreach(filename IN LISTS _COMBINE_SHADER_FILES)
        string(REPLACE "." "_" shader_name ${filename})
        add_custom_command(
            OUTPUT  ${shader_dir}/${shader_name}.h
            COMMAND ${CMAKE_COMMAND} -P ${generator}
                    ${shader_src}/${filename} ${shader_dir}/${shader_name}.h ${input_file}
            MAIN_DEPENDENCY ${shader_src}/${filename}
            DEPENDS ${input_file})
        list(APPEND headers ${shader_dir}/${shader_name}.h)
    endforeach()

    add_custom_target(host_shaders_${name} DEPENDS ${headers})
    set(${out_var} ${shader_inc} PARENT_SCOPE)
endfunction()

# imgui_fonts_<name>: embed THIS tree's font set (globbed from its fonts/ dir, so
# gr2's JP-only set and the prerelease's full set are each picked up). Mirrors the
# tree's imgui/renderer/CMakeLists.txt naming so the renderer's symbols resolve.
# (Each tree's fonts/ holds exactly its FONT_LIST, so glob == the explicit list.)
function(combine_imgui_fonts name src out_var)
    set(fonts_src ${src}/imgui/renderer/fonts)
    set(gen       ${CMAKE_BINARY_DIR}/core_${name}_gen/imgui_fonts)

    file(GLOB font_files CONFIGURE_DEPENDS
         ${fonts_src}/*.ttf ${fonts_src}/*.ttc ${fonts_src}/*.otf)

    set(outputs "")
    foreach(font_path IN LISTS font_files)
        get_filename_component(font_file ${font_path} NAME)
        string(REGEX REPLACE "-" "_" fontname ${font_file})
        string(TOLOWER ${fontname} fontname)
        string(REGEX REPLACE "\\.(ttf|otf|ttc)$" "" fontname_cpp ${fontname})
        set(fontname_cpp "imgui_font_${fontname_cpp}")

        set(out "${gen}/imgui_fonts/${fontname}.g.cpp")
        add_custom_command(
            OUTPUT  "${out}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${gen}/imgui_fonts"
            COMMAND $<TARGET_FILE:Dear_ImGui_FontEmbed> -nostatic
                    "${font_path}" ${fontname_cpp} > "${out}"
            DEPENDS Dear_ImGui_FontEmbed "${font_path}"
            USES_TERMINAL)
        list(APPEND outputs "${out}")
    endforeach()

    add_library(imgui_fonts_${name} STATIC ${outputs})
    set(${out_var} ${gen} PARENT_SCOPE)
endfunction()

# embedded resources (cmrc): match the tree's EXPLICIT per-tree resource set, without
# pulling in the ~40 Qt-launcher icons that also live alongside them.
#   - The 4 trophy-tier pngs + trophy.wav are opened via a CONSTRUCTED alias
#     ("src/<dir>/" + tier + ".png"), so they are NOT greppable -> include the known
#     set when present.
#   - Everything else the core embeds (shadps4.png, big_picture/*) is opened as a
#     STRING LITERAL, so derive those from the source's "src/<dir>/…" literals.
#
# RESOURCE-DIR SPLIT: upstream #4590 ("Clean up resource files") renamed src/images ->
# src/resources. core_main, once synced past that commit, uses resources/ and opens
# "src/resources/…"; older trees (core_gr2, v0.13.0) still use images/ and open
# "src/images/…". Detect which THIS tree uses so BOTH the WHENCE dir and the cmrc alias
# PREFIX match what the source actually opens. The source hardcodes "src/<dir>/…" (the
# literal "src/", not the folder name), so the alias reads "src/resources/…" or
# "src/images/…" for either core regardless of the on-disk folder (src vs src-gr2).
# When core_gr2 is eventually synced past #4590 too, the resources/ dir appears and this
# switches over automatically -- no edit needed.
function(combine_resources name src)
    if(EXISTS ${src}/resources)
        set(resdir    ${src}/resources)
        set(resprefix src/resources)
    else()
        set(resdir    ${src}/images)
        set(resprefix src/images)
    endif()
    set(cmrc_rel "")

    # (1) dynamically-opened, not greppable: include the known trophy set if present
    foreach(f trophy.wav bronze.png gold.png platinum.png silver.png)
        if(EXISTS ${resdir}/${f})
            list(APPEND cmrc_rel ${f})
        endif()
    endforeach()

    # (2) literal-opened: derive from the source's "src/(images|resources)/…" literals
    execute_process(
        COMMAND grep -rhoE "\"src/(images|resources)/[^\"]+\"" ${src}
        OUTPUT_VARIABLE _grep_out ERROR_QUIET)
    string(REGEX MATCHALL "src/(images|resources)/[^\"]+" _aliases "${_grep_out}")
    foreach(a IN LISTS _aliases)
        string(REGEX REPLACE "^src/(images|resources)/" "" rel "${a}")
        if(EXISTS ${resdir}/${rel})
            list(APPEND cmrc_rel ${rel})
        endif()
    endforeach()

    list(REMOVE_DUPLICATES cmrc_rel)
    set(res_files "")
    foreach(rel IN LISTS cmrc_rel)
        list(APPEND res_files ${resdir}/${rel})
    endforeach()

    list(LENGTH res_files _n)
    message(STATUS "core_${name}: embedding ${_n} cmrc resource(s) from ${resdir} (alias ${resprefix}/…)")

    cmrc_add_resource_library(embedded_resources_${name}
        ALIAS res_${name}::embedded
        NAMESPACE res)
    cmrc_add_resources(embedded_resources_${name}
        WHENCE ${resdir}
        PREFIX ${resprefix}
        ${res_files})
endfunction()

# ===========================================================================
# add_core(NAME <id> SRC <source-dir> [EXCLUDES ...] [LIBS ...] [DEFINES ...])
#   NAME     short id -> target core_<id>, file libcore_<id>.so
#   SRC      the pure SOURCE dir (e.g. <root>/src or <root>/src-gr2); the files
#            common/, core/, video_core/, main.cpp, … live directly inside it
#   EXCLUDES extra .cpp present-on-disk-but-not-compiled, relative to SRC
#   LIBS     extra link targets unique to this core
#   DEFINES  extra compile definitions unique to this core
# ===========================================================================
function(add_core)
    cmake_parse_arguments(AC "" "NAME;SRC" "EXCLUDES;LIBS;DEFINES" ${ARGN})
    set(name ${AC_NAME})
    set(src  ${AC_SRC})
    set(gen  ${CMAKE_BINARY_DIR}/core_${name}_gen)
    file(MAKE_DIRECTORY ${gen})

    # ---- C++ sources: glob, minus main.cpp and the per-tree non-compiled strays ----
    file(GLOB_RECURSE core_srcs CONFIGURE_DEPENDS ${src}/*.cpp)
    list(REMOVE_ITEM core_srcs ${src}/main.cpp)
    foreach(ex IN LISTS AC_EXCLUDES)
        list(REMOVE_ITEM core_srcs ${src}/${ex})
    endforeach()

    # ---- assembly sources (.s/.S): fiber + thread-stack context switching ----
    # glob is case-sensitive on Linux, so match both: fiber_context.s + stack.S.
    file(GLOB_RECURSE asm_srcs CONFIGURE_DEPENDS ${src}/*.s ${src}/*.S)

    # ---- scm_rev (8 @VARS@, computed per-tree; safe no-.git fallback) ----
    set(APP_VERSION "${name}")
    set(APP_IS_RELEASE true)
    string(TIMESTAMP BUILD_DATE "%Y-%m-%d %H:%M:%S")
    set(GIT_REV "unknown")
    set(GIT_BRANCH "unknown")
    set(GIT_DESC "unknown")
    set(GIT_REMOTE_NAME "origin")
    set(GIT_REMOTE_URL "")
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS ${src}/../.git)
        execute_process(COMMAND ${GIT_EXECUTABLE} -C ${src}/.. rev-parse HEAD
            OUTPUT_VARIABLE _r OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        execute_process(COMMAND ${GIT_EXECUTABLE} -C ${src}/.. rev-parse --abbrev-ref HEAD
            OUTPUT_VARIABLE _b OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        execute_process(COMMAND ${GIT_EXECUTABLE} -C ${src}/.. describe --always --long --dirty
            OUTPUT_VARIABLE _d OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_r)
            set(GIT_REV "${_r}")
        endif()
        if(_b)
            set(GIT_BRANCH "${_b}")
        endif()
        if(_d)
            set(GIT_DESC "${_d}")
        endif()
    endif()
    set(scm_out ${gen}/scm_rev_${name}.cpp)
    configure_file(${src}/common/scm_rev.cpp.in ${scm_out} @ONLY)

    # ---- per-core entry shim (copy so -DCORE_REAL_MAIN is per-core) ----
    set(entry_out ${gen}/core_${name}_entry.cpp)
    configure_file(${CMAKE_SOURCE_DIR}/dispatcher/core_entry.cpp ${entry_out} COPYONLY)

    # ---- regenerate the colliding sub-targets under unique names ----
    combine_build_fontembed()
    combine_host_shaders(${name} ${src} hs_inc)
    combine_imgui_fonts(${name} ${src} fonts_inc)
    combine_resources(${name} ${src})

    # ---- the core: one hidden-visibility shared object ----
    add_library(core_${name} SHARED
        ${core_srcs}
        ${asm_srcs}
        ${src}/main.cpp
        ${scm_out}
        ${entry_out})

    # assembly TUs: suppress the "unused -x c++ flag" warning the C-driver assembler emits
    if(asm_srcs)
        set_source_files_properties(${asm_srcs}
            TARGET_DIRECTORY core_${name}
            PROPERTIES COMPILE_OPTIONS -Wno-unused-command-line-argument)
    endif()

    add_dependencies(core_${name} host_shaders_${name})
    target_link_libraries(core_${name} PRIVATE imgui_fonts_${name} res_${name}::embedded)

    set_target_properties(core_${name} PROPERTIES
        OUTPUT_NAME core_${name}
        PREFIX "lib"
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON)

    # Export ONLY core_entry; localize everything else (incl. statically linked
    # externals' symbols) so two cores never interpose. ELF-only: link.exe/lld-link
    # ignore --version-script. On Windows a DLL exports nothing unless marked, so
    # the __declspec(dllexport) on core_entry (dispatcher/core_entry.cpp) already
    # makes it the SOLE export -- there is no flat global namespace to interpose in.
    if(NOT WIN32)
        target_link_options(core_${name} PRIVATE
            -Wl,--version-script=${CMAKE_SOURCE_DIR}/cmake/export.map)
    endif()

    # Retarget main() for THIS TU only (source stays byte-identical).
    set_source_files_properties(${src}/main.cpp
        TARGET_DIRECTORY core_${name}
        PROPERTIES COMPILE_DEFINITIONS "main=${name}_main")
    set_source_files_properties(${entry_out}
        TARGET_DIRECTORY core_${name}
        PROPERTIES COMPILE_DEFINITIONS "CORE_REAL_MAIN=${name}_main")

    # Include dirs. shadPS4 sources mix two styles: most are source-dir-relative
    # ("common/…", "core/…" → resolved via ${src} or the including file's own dir),
    # but a few are REPO-ROOT-relative: "src/common/{arch,decoder,types}.h",
    # "src/core/libraries/usbd/usbd.h", and "externals/aacdec/fdk-aac/…". The repo
    # root can't be added directly — gr2's source lives in src-gr2/, so "src/…" would
    # wrongly resolve to core_main's src/. Build a per-core include root whose `src`
    # symlink maps to THIS core's source and `externals` to the shared externals, so
    # "src/…" / "externals/…" resolve correctly and per-core.
    set(inc_root ${gen}/incroot)
    file(MAKE_DIRECTORY ${inc_root})
    if(NOT EXISTS ${inc_root}/src)
        file(CREATE_LINK ${src} ${inc_root}/src SYMBOLIC)
    endif()
    if(NOT EXISTS ${inc_root}/externals)
        file(CREATE_LINK ${COMBINED_EXTERNALS} ${inc_root}/externals SYMBOLIC)
    endif()

    target_include_directories(core_${name} PRIVATE
        ${inc_root}
        ${src}
        ${hs_inc}
        ${fonts_inc}
        ${src}/video_core/host_shaders)

    # Shared base link set (Linux/x86_64), present in both trees.
    target_link_libraries(core_${name} PRIVATE
        magic_enum::magic_enum fmt::fmt tsl::robin_map xbyak::xbyak
        Tracy::TracyClient RenderDoc::API FFmpeg::ffmpeg Dear_ImGui gcn
        half::half ZLIB::ZLIB PNG::PNG Boost::headers
        GPUOpen::VulkanMemoryAllocator LibAtrac9 sirit Vulkan::Headers
        xxHash::xxhash Zydis::Zydis glslang::glslang SDL3::SDL3 pugixml::pugixml
        stb::headers lfreist-hwinfo::hwinfo nlohmann_json::nlohmann_json
        libusb::usb toml11::toml11)

    target_compile_definitions(core_${name} PRIVATE
        IMGUI_USER_CONFIG="imgui/imgui_config.h"
        BOOST_ASIO_STANDALONE)

    if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        target_link_libraries(core_${name} PRIVATE uuid)
        if(ENABLE_USERFAULTFD)
            target_compile_definitions(core_${name} PRIVATE ENABLE_USERFAULTFD)
        endif()
    endif()
    if(WIN32)
        # Win32 system import libs the emulator's Windows code paths reference that
        # are NOT already auto-linked via #pragma comment(lib, ...) -- dbghelp/psapi
        # are pulled in that way by common/{crash_handler,hang_watchdog}.cpp. An
        # unreferenced import lib costs nothing (no DLL is added to the import table
        # unless a symbol is actually used), so over-listing here is harmless; tune
        # the set against the first real link if a symbol is still undefined. SDL3
        # and FFmpeg (built static, Workstream D) carry most of their own deps via
        # INTERFACE link libraries.
        target_link_libraries(core_${name} PRIVATE
            ws2_32 iphlpapi winmm bcrypt userenv version)
        # wepoll = the epoll()-for-Windows shim (externals/ext-wepoll, a WIN32-only
        # add_subdirectory). The network layer's net_epoll.{h,cpp} and net.cpp do
        # #include <wepoll.h>, so the core MUST link the wepoll target -- that is what
        # puts wepoll's header on the include path (via its INTERFACE include dir) AND
        # its symbols on the link line. mincore + wbemuuid round out upstream's Windows
        # link set. Upstream: target_link_libraries(shadps4 PRIVATE mincore wepoll wbemuuid).
        target_link_libraries(core_${name} PRIVATE mincore wepoll wbemuuid)
        if(MSVC)
            # ThinLTO via clang-cl can emit calls to compiler-rt builtins (e.g. 128-bit
            # integer division) that aren't otherwise pulled in; upstream links this
            # explicitly. clang-cl drives the link, so its clang-rt lib dir is on the
            # search path and the bare lib name resolves.
            target_link_libraries(core_${name} PRIVATE clang_rt.builtins-x86_64.lib)
        endif()
    endif()
    if(ENABLE_DISCORD_RPC)
        target_compile_definitions(core_${name} PRIVATE ENABLE_DISCORD_RPC)
        target_link_libraries(core_${name} PRIVATE discord-rpc)
    endif()

    # Per-core extras (gr2: SDL3_mixer + miniz; prerelease: CLI11/OpenAL/httplib/…).
    if(AC_LIBS)
        target_link_libraries(core_${name} PRIVATE ${AC_LIBS})
    endif()
    if(AC_DEFINES)
        target_compile_definitions(core_${name} PRIVATE ${AC_DEFINES})
    endif()

    # libstdc++ MUST stay shared/dynamic across the dlopen boundary.
endfunction()
