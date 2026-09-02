# GENERATED FILE - do not edit; regenerate with ./upstream-merge.py
#
# core_main's upstream build surface, extracted from shadPS4's root
# CMakeLists.txt under the combined build's profile (Linux/x86_64, SDL-only,
# no tests). The umbrella includes this file; Combine.cmake consumes the
# CORE_MAIN_* lists via add_core(NAME main ...).
#
#   upstream commit : ae1539d3 2026-09-02
#   emulator version: 0.18.1

set(CORE_MAIN_UPSTREAM_SHA "ae1539d32ae041135564c4c61a7b46d026f268ce")

# --- find_package mirror. Gated on ENABLE_SYSTEM_LIBRARIES exactly like upstream
#     (default OFF): with it off, the vendored externals/ provide every target and
#     the build cannot break when host/container dev packages come and go.
#     Apple/FreeBSD-only entries are intentionally absent. ---
if(ENABLE_SYSTEM_LIBRARIES)
    find_package(Boost 1.84.0 CONFIG)
    find_package(CLI11 2.6.1 CONFIG)
    find_package(FFmpeg 5.1.2 MODULE)
    find_package(fmt 12.0.0 CONFIG)
    find_package(Freetype 2.14.1 MODULE)
    find_package(glslang 15 CONFIG)
    find_package(half 1.12.0 MODULE)
    find_package(magic_enum 0.9.7 CONFIG)
    find_package(miniupnpc 2.3.3 MODULE)
    find_package(miniz 3.1 CONFIG)
    find_package(nlohmann_json 3.12 CONFIG)
    find_package(PNG 1.6 MODULE)
    find_package(OpenAL CONFIG)
    find_package(LibreSSL 4.3.1 MODULE)
    find_package(RenderDoc 1.6.0 MODULE)
    find_package(SDL3 3.1.2 CONFIG)
    find_package(stb MODULE)
    find_package(toml11 4.2.0 CONFIG)
    find_package(tsl-robin-map 1.3.0 CONFIG)
    find_package(VulkanHeaders 1.4.330 CONFIG)
    find_package(VulkanMemoryAllocator 3.1.0 CONFIG)
    find_package(xbyak 7.07 CONFIG)
    find_package(xxHash 0.8.2 MODULE)
    find_package(ZArchive 0.1.2 MODULE)
    find_package(ZLIB 1.3 MODULE)
    find_package(Zydis 5.0.0 MODULE)
    find_package(pugixml 1.14 CONFIG)
    find_package(absl 20250512.1 CONFIG)
endif()

# --- on-disk .cpp files upstream does NOT compile (add_core globs, then removes these) ---
set(CORE_MAIN_EXCLUDES
    core/file_sys/file.cpp
    core/libraries/network/net_obj.cpp)

# --- link libraries beyond Combine.cmake's shared base set (upstream order) ---
set(CORE_MAIN_LIBS
    spdlog::spdlog
    ImGuiFileDialog
    minimp3
    miniz::miniz
    fdk-aac
    CLI11::CLI11
    OpenAL::OpenAL
    Cpp_Httplib
    miniupnpc::miniupnpc
    Freetype::Freetype
    libprotobuf
    ZArchive::zarchive)

# --- compile definitions beyond the shared base ---
set(CORE_MAIN_DEFINES "")

# --- compiled sources the *.cpp/*.s/*.S glob would miss ---
set(CORE_MAIN_EXTRA_SOURCES "")

# --- embedded (cmrc) resources, relative to the tree's resources/ dir ---
set(CORE_MAIN_CMRC_FILES
    big_picture/controller.png
    big_picture/experimental.png
    big_picture/folder.png
    big_picture/graphics.png
    big_picture/log.png
    big_picture/profiles.png
    big_picture/settings.png
    big_picture/trophy.png
    bronze.png
    gold.png
    platinum.png
    shadps4.png
    silver.png
    trophy.wav)
