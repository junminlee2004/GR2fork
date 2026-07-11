// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/logging/log.h"
#include "common/elf_info.h"
#include "common/path_util.h"
#include "common/singleton.h"
#include "common/thread.h"
#include "core/file_sys/fs.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/content_search/content_search.h"

namespace Libraries::ContentSearch {

namespace {

constexpr u32 ENTRY_SIZE = 0x960;
constexpr u32 MAX_GALLERY_PHOTOS = 999;

static std::mutex s_ids_mutex;
static std::vector<std::string> s_exported_ids;
static bool s_ids_loaded = false;
static bool s_screenshot_mounted = false;

// Metadata API state
static u32 s_metadata_handle_counter = 0x4D440001;
static std::mutex s_metadata_mutex;

struct MetadataSession {
    u64 record_id;
    bool valid;
};
static MetadataSession s_metadata_sessions[16] = {};

static u32 AllocMetadataHandle(u64 record_id) {
    std::lock_guard lock(s_metadata_mutex);
    u32 handle = s_metadata_handle_counter++;
    u32 slot = handle & 0xF;
    s_metadata_sessions[slot].record_id = record_id;
    s_metadata_sessions[slot].valid = true;
    return handle;
}

static u64 LookupMetadataRecordId(u32 handle) {
    std::lock_guard lock(s_metadata_mutex);
    u32 slot = handle & 0xF;
    if (s_metadata_sessions[slot].valid)
        return s_metadata_sessions[slot].record_id;
    return 0;
}

static void FreeMetadataHandle(u32 handle) {
    std::lock_guard lock(s_metadata_mutex);
    u32 slot = handle & 0xF;
    s_metadata_sessions[slot].valid = false;
}

// Stable content-ID handle table. GR2 stores the 8-byte handle at SceContentSearchContentInfo
// +0x00 and hands it back to OpenMetadataByContentId() and batched sceContentDeleteById(), so
// the handle must be a stable per-id identity - a list position renumbers on every delete.
static std::mutex s_handle_mutex;
static std::unordered_map<u64, std::string> s_handle_to_id;
static std::unordered_map<std::string, u64> s_id_to_handle;
static u64 s_next_handle = 0x4752324400000001ULL; // 'G''R''2''D' | ordinal 1

static u64 GetOrAssignHandle(const std::string& cid) {
    std::lock_guard lock(s_handle_mutex);
    auto it = s_id_to_handle.find(cid);
    if (it != s_id_to_handle.end()) return it->second;
    const u64 h = s_next_handle++;
    s_id_to_handle.emplace(cid, h);
    s_handle_to_id.emplace(h, cid);
    return h;
}

static void ForgetHandleForId(const std::string& cid) {
    std::lock_guard lock(s_handle_mutex);
    auto it = s_id_to_handle.find(cid);
    if (it == s_id_to_handle.end()) return;
    s_handle_to_id.erase(it->second);
    s_id_to_handle.erase(it);
}

// Screenshot directory + mount
std::filesystem::path GetScreenshotDir() {
    std::error_code ec;
    // RENAME(GR2FORK v1.0): "GR2_PhotoApp_HLE" -> "Gravity Rush 2". Must stay
    // in sync with jpegenc.cpp::GetGR2PhotoDir().
    const auto dir =
        Common::FS::GetUserPath(Common::FS::PathType::ScreenshotsDir) / "Gravity Rush 2";
    std::filesystem::create_directories(dir, ec);
    return dir;
}

void MountScreenshotDir() {
    if (s_screenshot_mounted) return;
    s_screenshot_mounted = true;

    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    const auto dir = GetScreenshotDir();

    mnt->Mount(dir, "/screenshot", true);
    mnt->Mount(dir, "/photo", true);
    mnt->Mount(dir, "/app0/photo", true);
    LOG_INFO(Core, "[GR2PhotoHLE] Mounted /screenshot, /photo, /app0/photo -> {}", dir.string());
}

// The on-disk .jpg files are the source of truth: s_exported_ids is rebuilt from a directory
// scan (LoadIdsFromDisk, lazy and idempotent; RescanIdsFromDisk forces a rebuild) and kept in
// memory for O(1) indexed access by the gallery fiber.

static void ScanJpegsInto(const std::filesystem::path& dir,
                          std::vector<std::string>& out) {
    out.clear();
    std::set<std::string> seen;
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        const auto ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG") continue;
        auto stem = entry.path().stem().string();
        // Accept well-formed PS4 content ids and user-dropped photo stems alike: the HLE
        // gallery looks up by index, not name, so format flexibility costs nothing.
        if (stem.empty()) continue;
        if (seen.insert(stem).second) {
            out.push_back(std::move(stem));
        }
    }
    // Stable ordering so gallery index N always maps to the same photo across
    // runs. directory_iterator order is fs-dependent; sort lexicographically.
    std::sort(out.begin(), out.end());
}

void LoadIdsFromDisk() {
    if (s_ids_loaded) return;
    s_ids_loaded = true;
    const auto dir = GetScreenshotDir();
    ScanJpegsInto(dir, s_exported_ids);
    LOG_INFO(Core, "[GR2PhotoHLE] Scanned {} -> {} photo(s) (runtime index, no .txt)",
             dir.string(), s_exported_ids.size());
}

void RescanIdsFromDisk() {
    s_ids_loaded = false;
    LoadIdsFromDisk();
}

// Kept only to satisfy the header export; content ids are timestamp-based (see
// GenerateTimestampContentId), so this returns the total photo count, which is what callers
// actually want.
u32 GetNextCounterValue() {
    return static_cast<u32>(s_exported_ids.size());
}

// Search output helpers
void WriteSearchOutputs(u64 a7, u64 hit) {
    if (a7 == 0) return;
    u64* pair0 = reinterpret_cast<u64*>(a7 - 0x10);
    u64* pair1 = reinterpret_cast<u64*>(a7 - 0x08);
    u64* adv   = reinterpret_cast<u64*>(a7);
    *pair0 = hit;
    *pair1 = hit;
    *adv   = hit;
}

// Populate a single ContentSearch entry (0x960 bytes). GR2's FilmAlbum copies the handle, path
// fields, and content-ID string into its own record and opens the paths via the eboot's FIOS
// wrapper (0x5745c0), which prepends "/arc0/" to relative paths - both paths must be absolute.
void PopulateEntry(u8* entry, const std::string& cid) {
    // +0x000 (8B): opaque content-ID handle, stored verbatim by GR2 and handed back to
    // OpenMetadataByContentId() and sceContentDeleteById(). Must be a stable per-id handle so
    // deleting one photo never shifts another photo's handle mid-batch.
    u64 record_id = GetOrAssignHandle(cid);
    std::memcpy(entry + 0x000, &record_id, 8);

    // +0x00C (4B): content type/format. Type 1 already yields correct grid-slot
    // population (the photo count is honored), so it is left unchanged.
    u32 ctype = 1;
    std::memcpy(entry + 0x00C, &ctype, 4);

    // Build the absolute guest path once and use it for BOTH path fields.
    char filepath[512] = {};
    std::snprintf(filepath, sizeof(filepath), "/screenshot/%s.jpg", cid.c_str());

    // +0x018 (<=0x401): path #1 -> full JPEG. MUST be absolute (leading '/') so
    // the FIOS wrapper opens it as-is instead of prepending "/arc0/".
    std::memset(entry + 0x018, 0, 0x401);
    std::strncpy(reinterpret_cast<char*>(entry + 0x018), filepath, 0x400);

    // +0x424 (<=0x101): content-ID as a STRING. The metadata worker can re-find
    // the photo by this id, so it must be the real id (not a display label).
    std::memset(entry + 0x424, 0, 0x101);
    std::strncpy(reinterpret_cast<char*>(entry + 0x424), cid.c_str(), 0x100);

    // +0x52B (<=0x401): path #2 -> thumbnail. No separate thumbnail is written,
    // so point it at the same JPEG; the UI decoder downscales for the grid.
    std::memset(entry + 0x52B, 0, 0x401);
    std::strncpy(reinterpret_cast<char*>(entry + 0x52B), filepath, 0x400);
}

} // namespace

// Public API

// Generate a content id "YYYYMMDD_HHMMSS_NNN" (NNN = per-second disambiguation counter, checked
// against the in-memory id list so the filename is unique on disk). Pure ASCII + underscore.
// Namespace scope because jpegenc.cpp and content_export.cpp call it across TUs.
std::string GenerateTimestampContentId() {
    std::lock_guard lock(s_ids_mutex);
    LoadIdsFromDisk();

    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char ts[20] = {};
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);

    // Most seconds will have exactly 1 capture, so n=1 usually hits.
    for (u32 n = 1; n < 1000; n++) {
        char candidate[40] = {};
        std::snprintf(candidate, sizeof(candidate), "%s_%03u", ts, n);
        std::string id(candidate);
        if (std::find(s_exported_ids.begin(), s_exported_ids.end(), id) ==
            s_exported_ids.end()) {
            return id;
        }
    }
    // > 999 captures within a single second: spill to epoch-ms fallback.
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch())
                        .count();
    char fallback[56] = {};
    std::snprintf(fallback, sizeof(fallback), "%s_FB%06lld", ts,
                  static_cast<long long>(ms & 0xFFFFFF));
    return std::string(fallback);
}

u32 GetExportedCount() {
    std::lock_guard lock(s_ids_mutex);
    return static_cast<u32>(s_exported_ids.size());
}

std::string GetContentIdByIndex(u32 index) {
    std::lock_guard lock(s_ids_mutex);
    if (index >= s_exported_ids.size()) return {};
    return s_exported_ids[index];
}

std::string GetContentIdByHandle(u64 handle) {
    std::lock_guard lock(s_handle_mutex);
    auto it = s_handle_to_id.find(handle);
    return (it == s_handle_to_id.end()) ? std::string{} : it->second;
}

bool DeleteContentById(const std::string& content_id) {
    {
        std::lock_guard lock(s_ids_mutex);
        auto it = std::find(s_exported_ids.begin(), s_exported_ids.end(), content_id);
        if (it == s_exported_ids.end()) return false;
        s_exported_ids.erase(it);
    }
    // Drop the id<->handle mapping so a stale handle can never resolve to a
    // since-deleted (or future-reused) id. Done AFTER releasing s_ids_mutex so
    // the lock order stays one-directional (s_ids_mutex then s_handle_mutex).
    ForgetHandleForId(content_id);

    auto path = GetScreenshotDir() / (content_id + ".jpg");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    // PURGED(GR2FORK v1.0): no SaveIdsToDisk - filesystem is source of truth.
    return true;
}

void NotifyExportedContentId(const std::string& content_id) {
    std::lock_guard lock(s_ids_mutex);
    LoadIdsFromDisk();
    for (const auto& existing : s_exported_ids) {
        if (existing == content_id) return;
    }
    // PURGED(GR2FORK v1.0): in-memory append only, no .txt write. The JPEG
    // file itself (written by jpegenc.cpp before this call) is the canonical
    // record. Keep sorted so indices remain stable for the gallery fiber.
    s_exported_ids.push_back(content_id);
    std::sort(s_exported_ids.begin(), s_exported_ids.end());
    LOG_INFO(Core, "[GR2PhotoHLE] ContentSearch: registered id '{}' (total={})",
             content_id, s_exported_ids.size());
}

std::filesystem::path GetScreenshotHostDir() {
    return GetScreenshotDir();
}

u32 GetNextPhotoCounter() {
    std::lock_guard lock(s_ids_mutex);
    LoadIdsFromDisk();
    return GetNextCounterValue();
}

static std::string s_last_saved_content_id;
static std::mutex s_last_saved_mutex;

void SetLastSavedContentId(const std::string& content_id) {
    std::lock_guard lock(s_last_saved_mutex);
    s_last_saved_content_id = content_id;
}

std::string GetLastSavedContentId() {
    std::lock_guard lock(s_last_saved_mutex);
    return s_last_saved_content_id;
}

// HLE API

int PS4_SYSV_ABI sceContentSearchInit(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7,
                                      u64 a8) {
    MountScreenshotDir();
    {
        std::lock_guard lock(s_ids_mutex);
        LoadIdsFromDisk();
    }

    if (a1 != 0) {
        *reinterpret_cast<u32*>(a1) = 0x43530001;
    }

    LOG_INFO(Core, "[GR2PhotoHLE] ContentSearchInit: mode={}", a2);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentSearchSearchContent(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6,
                                               u64 a7, u64 a8) {
    std::lock_guard lock(s_ids_mutex);

    MountScreenshotDir();
    LoadIdsFromDisk();

    u32 total = static_cast<u32>(s_exported_ids.size());
    if (total > MAX_GALLERY_PHOTOS) total = MAX_GALLERY_PHOTOS;

    u32 start = static_cast<u32>(a5);
    u32 limit = static_cast<u32>(a6);
    u32 available = (start < total) ? (total - start) : 0;
    u32 hit = std::min(available, limit);

    if (a8 != 0 && hit > 0) {
        u64 buf_size = static_cast<u64>(hit) * ENTRY_SIZE;
        std::memset(reinterpret_cast<void*>(a8), 0, buf_size);

        for (u32 i = 0; i < hit; i++) {
            u8* entry = reinterpret_cast<u8*>(a8) + static_cast<u64>(i) * ENTRY_SIZE;
            u32 global_idx = start + i;
            PopulateEntry(entry, s_exported_ids[global_idx]);
        }
    }

    WriteSearchOutputs(a7, hit);

    LOG_INFO(Core, "[GR2PhotoHLE] SearchContent: start={} limit={} hit={} total={}",
             start, limit, hit, total);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentSearchTerm(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7,
                                      u64 a8) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentSearchSearchApplication(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5,
                                                   u64 a6, u64 a7, u64 a8) {
    if (a3 != 0) *reinterpret_cast<u32*>(a3) = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentSearchGetMyApplicationIndex(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5,
                                                       u64 a6, u64 a7, u64 a8) {
    if (a1 != 0) *reinterpret_cast<u32*>(a1) = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentSearchGetNumOfContent(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6,
                                                 u64 a7, u64 a8) {
    std::lock_guard lock(s_ids_mutex);
    LoadIdsFromDisk();
    u32 count = static_cast<u32>(s_exported_ids.size());
    if (count > MAX_GALLERY_PHOTOS) count = MAX_GALLERY_PHOTOS;

    if (a3 != 0) *reinterpret_cast<u32*>(a3) = count;
    if (a4 != 0) *reinterpret_cast<u32*>(a4) = count;
    return ORBIS_OK;
}

// Metadata API

int PS4_SYSV_ABI sceContentSearchOpenMetadata(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6,
                                              u64 a7, u64 a8) {
    u32 handle = AllocMetadataHandle(a1);
    if (a2 != 0) *reinterpret_cast<u32*>(a2) = handle;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentSearchOpenMetadataByContentId(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5,
                                                         u64 a6, u64 a7, u64 a8) {
    u32 handle = AllocMetadataHandle(0);
    if (a2 != 0) *reinterpret_cast<u32*>(a2) = handle;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentSearchGetMetadataValue(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6,
                                                  u64 a7, u64 a8) {
    if (a3 == 0) return -1;

    u8* params = reinterpret_cast<u8*>(a3);
    u32 buf_size = *reinterpret_cast<u32*>(params + 0);
    u64 buf_ptr = *reinterpret_cast<u64*>(params + 8);

    if (buf_ptr == 0 || buf_size == 0) return -1;

    char* dest = reinterpret_cast<char*>(buf_ptr);
    std::memset(dest, 0, std::min(buf_size, 0x101u));
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentSearchGetMetadataFieldInfo(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5,
                                                      u64 a6, u64 a7, u64 a8) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentSearchCloseMetadata(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6,
                                               u64 a7, u64 a8) {
    FreeMetadataHandle(static_cast<u32>(a1));
    return ORBIS_OK;
}

// RegisterLib

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("dPj4ZtRcIWk", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchInit);
    LIB_FUNCTION("TEW3IKxYfXc", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchSearchContent);
    LIB_FUNCTION("V4A4AGzxA0g", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchSearchApplication);
    LIB_FUNCTION("FRT4EYtZU1Y", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchGetMyApplicationIndex);
    LIB_FUNCTION("o-RBPV0qr8c", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchGetNumOfContent);
    LIB_FUNCTION("5opI5D0LQjg", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchOpenMetadata);
    LIB_FUNCTION("bjAlYWwRTJA", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchOpenMetadataByContentId);
    LIB_FUNCTION("ruNe-FgCzO8", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchGetMetadataValue);
    LIB_FUNCTION("0DqfN9r3jmA", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchGetMetadataFieldInfo);
    LIB_FUNCTION("-YbpaF0XS-I", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchCloseMetadata);
    LIB_FUNCTION("1xSZodB2geA", "libSceContentSearch", 1, "libSceContentSearch",
                 sceContentSearchTerm);
}

} // namespace Libraries::ContentSearch
