// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/libs.h"
#include "core/libraries/content_export/content_export.h"
#include "core/libraries/content_search/content_search.h"

namespace Libraries::ContentExport {

namespace {

// GR2FORK: extract the longest printable run (the encoded placement "comment") from an export
// argument region, offset-agnostically. Returns "" if no run of at least minlen is present.
std::string LongestRun(u64 ptr, size_t n, size_t minlen) {
    if (ptr < 0x10000 || ptr >= 0x10000000000ull) {
        return {};
    }
    const u8* p = reinterpret_cast<const u8*>(ptr);
    size_t best_start = 0, best_len = 0, cur_start = 0, cur_len = 0;
    for (size_t i = 0; i < n; ++i) {
        if (p[i] >= 0x20 && p[i] < 0x7f) {
            if (cur_len == 0) {
                cur_start = i;
            }
            if (++cur_len > best_len) {
                best_len = cur_len;
                best_start = cur_start;
            }
        } else {
            cur_len = 0;
        }
    }
    if (best_len < minlen) {
        return {};
    }
    return std::string(reinterpret_cast<const char*>(p + best_start), best_len);
}

// GR2FORK: capture the placement comment from the export payload and key it by content id.
// The data buffer (a2) carries a header before the JPEG: title C-str at +0x00 and the 256-char
// encoded placement comment at +0x101; the a4/a5 param regions only alias a truncated view of
// it, so scan the data header first and keep the longest printable run.
void CaptureComment(const std::string& content_id, u64 a2, u64 a3, u64 a4, u64 a5) {
    std::string c = LongestRun(a2, std::min<u64>(a3, 0x400), 48);
    if (c.empty()) {
        c = LongestRun(a5, 0x2b0, 48);
    }
    if (c.empty()) {
        c = LongestRun(a4, 0x2b0, 48);
    }
    if (!c.empty()) {
        Libraries::ContentSearch::SetStoredComment(content_id, c);
    } else {
        LOG_ERROR(Core, "[GR2PhotoHLE] CaptureComment: no printable run >=48 for '{}'",
                  content_id);
    }
}

static std::string s_last_content_id;

static std::string GenerateContentId() {
    // NAMING(GR2FORK v1.0): delegates to ContentSearch's timestamp generator.
    // Format: "YYYYMMDD_HHMMSS_NNN". Was "UP9000-CUSA04943_00-SCREENSHOT%05u".
    return Libraries::ContentSearch::GenerateTimestampContentId();
}

// Save exported payload as <content_id>.jpg in the screenshot directory.
// The LLE encoder output may be wrapped in a PS4 container format,
// so we search for the JPEG SOI (FF D8) and EOI (FF D9) markers.
static void SaveExportedPhoto(const std::string& content_id, const u8* data, size_t size) {
    if (data == nullptr || size == 0) return;

    const u8* jpeg_start = data;
    size_t jpeg_size = size;

    // Find JPEG SOI marker (FF D8)
    for (size_t i = 0; i + 1 < size; ++i) {
        if (data[i] == 0xFF && data[i + 1] == 0xD8) {
            jpeg_start = data + i;
            jpeg_size = size - i;
            break;
        }
    }

    // Find JPEG EOI marker (FF D9) scanning backwards
    for (size_t i = size - 1; i > static_cast<size_t>(jpeg_start - data); --i) {
        if (data[i - 1] == 0xFF && data[i] == 0xD9) {
            jpeg_size = (i + 1) - static_cast<size_t>(jpeg_start - data);
            break;
        }
    }

    const bool is_jpeg = (jpeg_size >= 2 && jpeg_start[0] == 0xFF && jpeg_start[1] == 0xD8);

    const auto dir = Libraries::ContentSearch::GetScreenshotHostDir();
    const auto path = dir / (content_id + ".jpg");

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (out) {
        out.write(reinterpret_cast<const char*>(jpeg_start),
                  static_cast<std::streamsize>(jpeg_size));
        LOG_INFO(Core, "[GR2PhotoHLE] Saved photo {} bytes (raw={}, jpeg={}) -> {}",
                 jpeg_size, size, is_jpeg ? "yes" : "no", path.string());
    } else {
        LOG_ERROR(Core, "[GR2PhotoHLE] Failed to save photo to {}", path.string());
    }
}

} // namespace

int PS4_SYSV_ABI sceContentExportInit(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7,
                                      u64 a8) {
    if (a1 != 0) {
        *reinterpret_cast<u32*>(a1) = 0x45580001;
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentExportStart(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7,
                                       u64 a8) {
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentExportFromData(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7,
                                          u64 a8) {

    // Use the content ID that was already saved by sceJpegEncEncode
    s_last_content_id = Libraries::ContentSearch::GetLastSavedContentId();
    if (s_last_content_id.empty()) {
        s_last_content_id = GenerateContentId();
        LOG_WARNING(Core, "[GR2PhotoHLE] ContentExportFromData: no encoder content ID, generated '{}'",
                    s_last_content_id);
    }

    // The game's export buffer may be in a PS4-specific format not viewable on host; the photo
    // was already saved as a JPEG by sceJpegEncEncode using the same content-ID counter, so only
    // the content ID is registered here.

    // Notify ContentSearch so gallery can list this photo
    Libraries::ContentSearch::NotifyExportedContentId(s_last_content_id);
    CaptureComment(s_last_content_id, a2, a3, a4, a5);

    // Write content_id to output buffer
    if (a6 != 0 && a7 >= 48) {
        char* out = reinterpret_cast<char*>(a6);
        std::memset(out, 0, static_cast<size_t>(a7));
        std::strncpy(out, s_last_content_id.c_str(), static_cast<size_t>(a7) - 1);
        LOG_INFO(Core, "[GR2PhotoHLE] ContentExportFromData: content_id='{}' -> {:#x}",
                 s_last_content_id, a6);
    }

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentExportFromDataWithThumbnail(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5,
                                                       u64 a6, u64 a7, u64 a8) {

    s_last_content_id = Libraries::ContentSearch::GetLastSavedContentId();
    if (s_last_content_id.empty()) {
        s_last_content_id = GenerateContentId();
    }

    // Don't save - file already saved by encoder
    Libraries::ContentSearch::NotifyExportedContentId(s_last_content_id);
    CaptureComment(s_last_content_id, a2, a3, a4, a5);

    if (a6 != 0 && a7 >= 48) {
        char* out = reinterpret_cast<char*>(a6);
        std::memset(out, 0, static_cast<size_t>(a7));
        std::strncpy(out, s_last_content_id.c_str(), static_cast<size_t>(a7) - 1);
    }

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentExportFinish(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7,
                                        u64 a8) {

    // Write content_id to Finish output buffer too
    if (a2 != 0 && a3 >= 48 && !s_last_content_id.empty()) {
        char* out = reinterpret_cast<char*>(a2);
        std::memset(out, 0, static_cast<size_t>(a3));
        std::strncpy(out, s_last_content_id.c_str(), static_cast<size_t>(a3) - 1);
    }

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceContentExportTerm(u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 a6, u64 a7,
                                      u64 a8) {
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("FzEWeYnAFlI", "libSceContentExport", 1, "libSceContentExport",
                 sceContentExportInit);
    LIB_FUNCTION("FCygF4Ec4so", "libSceContentExport", 1, "libSceContentExport",
                 sceContentExportStart);
    LIB_FUNCTION("AOWqIYsgVHs", "libSceContentExport", 1, "libSceContentExport",
                 sceContentExportFromData);
    LIB_FUNCTION("uZTQHI50WpY", "libSceContentExport", 1, "libSceContentExport",
                 sceContentExportFromDataWithThumbnail);
    LIB_FUNCTION("tb3cZTCl8Ps", "libSceContentExport", 1, "libSceContentExport",
                 sceContentExportFinish);
    LIB_FUNCTION("+KDWny9Y-6k", "libSceContentExport", 1, "libSceContentExport",
                 sceContentExportTerm);
}

} // namespace Libraries::ContentExport
