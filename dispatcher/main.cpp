// SPDX-License-Identifier: GPL-2.0-or-later
//
// The only exe on disk; selects a core and hands the process to it. Precedence: --core=gr2|main
// (consumed here) > $SHADPS4_CORE > TITLE_ID auto-detect (Gravity Rush titles -> core_gr2) >
// default core_main. The title id comes from sce_sys/param.sfo (or a bare -g CUSA id) before any
// core loads; remaining args are forwarded verbatim with argv[0] preserved.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "core_loader.h"

namespace {

constexpr const char* kCoreFlag = "--core=";

bool ValidCore(const std::string& c) {
    return c == "gr2" || c == "main";
}

// Gravity Rush 2 + Gravity Rush Remastered, all known SKUs. The GR2 set is the
// six ids supplied for this build; the GRR set mirrors gr_remastered_ids in
// emulator.cpp. Any of these routes to core_gr2 (the fork supports both titles).
bool IsGravityRushTitle(std::string_view id) {
    static constexpr std::string_view kIds[] = {
        // Gravity Rush 2
        "CUSA03694", "CUSA04943", "CUSA04934", "CUSA00547", "PCJS50010", "PCAS00079", "CUSA04935",
        // Gravity Rush Remastered
        "CUSA01112", "CUSA01113", "CUSA01113P", "CUSA01130",
        "CUSA02318", "CUSA00546", "CUSA02443", "CUSA04246",
        "PCJS50004", "PCJS50008", "PCJS66015", "PCJS66029",
    };
    for (std::string_view s : kIds) {
        if (id == s) {
            return true;
        }
    }
    return false;
}

// Minimal PSF reader: returns the TITLE_ID string, or empty on any problem. Layout
// matches core/file_format/psf.h (header 0x14, entries 0x10, magic "\0PSF").
std::string ReadTitleId(const std::filesystem::path& sfo_path) {
    std::ifstream f(sfo_path, std::ios::binary);
    if (!f) {
        return {};
    }
    const std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
    if (buf.size() < 0x14) {
        return {};
    }
    auto rd16le = [&](size_t o) -> uint32_t {
        return uint32_t(buf[o]) | (uint32_t(buf[o + 1]) << 8);
    };
    auto rd32le = [&](size_t o) -> uint32_t {
        return uint32_t(buf[o]) | (uint32_t(buf[o + 1]) << 8) |
               (uint32_t(buf[o + 2]) << 16) | (uint32_t(buf[o + 3]) << 24);
    };
    // magic "\0PSF" (stored big-endian -> these exact bytes on disk)
    if (!(buf[0] == 0x00 && buf[1] == 0x50 && buf[2] == 0x53 && buf[3] == 0x46)) {
        return {};
    }
    const uint32_t key_table  = rd32le(0x08);
    const uint32_t data_table = rd32le(0x0C);
    const uint32_t entries    = rd32le(0x10);
    for (uint32_t i = 0; i < entries; ++i) {
        const size_t e = 0x14 + size_t(i) * 0x10;
        if (e + 0x10 > buf.size()) {
            break;
        }
        const uint32_t key_off  = rd16le(e + 0x00);
        const uint32_t val_len  = rd32le(e + 0x04);
        const uint32_t data_off = rd32le(e + 0x0C);

        size_t ks = size_t(key_table) + key_off;
        std::string key;
        while (ks < buf.size() && buf[ks] != 0) {
            key.push_back(char(buf[ks++]));
        }
        if (key != "TITLE_ID") {
            continue;
        }
        size_t ds = size_t(data_table) + data_off;
        std::string val;
        for (uint32_t j = 0; j < val_len && ds + j < buf.size(); ++j) {
            const char c = char(buf[ds + j]);
            if (c == '\0') {
                break;
            }
            val.push_back(c);
        }
        return val;
    }
    return {};
}

// Resolve the game's param.sfo from argv: a candidate is a directory or a file like eboot.bin
// (both resolve to <dir>/sce_sys/param.sfo). The explicit -g/--game value is preferred;
// otherwise the first argument yielding an existing param.sfo wins (robust to option ordering).
std::filesystem::path FindParamSfo(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto from_candidate = [&](const char* c) -> fs::path {
        if (c == nullptr || c[0] == '\0' || c[0] == '-') {
            return {};
        }
        const fs::path p(c);
        if (fs::is_directory(p, ec)) {
            fs::path s = p / "sce_sys" / "param.sfo";
            if (fs::exists(s, ec)) {
                return s;
            }
        }
        fs::path s2 = p.parent_path() / "sce_sys" / "param.sfo";
        if (fs::exists(s2, ec)) {
            return s2;
        }
        return {};
    };
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "-g") == 0 || std::strcmp(argv[i], "--game") == 0) {
            if (fs::path s = from_candidate(argv[i + 1]); !s.empty()) {
                return s;
            }
        }
    }
    for (int i = 1; i < argc; ++i) {
        if (fs::path s = from_candidate(argv[i]); !s.empty()) {
            return s;
        }
    }
    return {};
}

// Auto-select a core from the launch arguments. Returns "gr2" for a Gravity Rush
// title, "main" otherwise. Logs what it decided and why (to stderr -- the
// dispatcher has no access to the emulator's logging).
std::string AutoSelectCore(int argc, char* argv[]) {
    std::string id;
    // (a) a bare title id passed as an argument (game launched by id)
    for (int i = 1; i < argc && id.empty(); ++i) {
        if (IsGravityRushTitle(argv[i])) {
            id = argv[i];
        }
    }
    // (b) otherwise read TITLE_ID from the game's param.sfo
    if (id.empty()) {
        if (std::filesystem::path sfo = FindParamSfo(argc, argv); !sfo.empty()) {
            id = ReadTitleId(sfo);
        }
    }
    if (!id.empty() && IsGravityRushTitle(id)) {
        std::fprintf(stderr, "[dispatcher] %s (Gravity Rush) -> core_gr2\n", id.c_str());
        return "gr2";
    }
    if (!id.empty()) {
        std::fprintf(stderr, "[dispatcher] %s -> core_main\n", id.c_str());
    } else {
        std::fprintf(stderr, "[dispatcher] no Gravity Rush title detected -> core_main\n");
    }
    return "main";
}

} // namespace

int main(int argc, char* argv[]) {
    std::string core;
    bool core_from_flag = false;

    // Rebuild argv, stripping the first --core=… we see.
    std::vector<char*> fwd;
    fwd.reserve(static_cast<size_t>(argc) + 1);
    fwd.push_back(argv[0]); // keep program name

    for (int i = 1; i < argc; ++i) {
        if (!core_from_flag && std::strncmp(argv[i], kCoreFlag, std::strlen(kCoreFlag)) == 0) {
            core = argv[i] + std::strlen(kCoreFlag);
            core_from_flag = true;
            continue; // consumed: do not forward
        }
        fwd.push_back(argv[i]);
    }
    fwd.push_back(nullptr); // argv is null-terminated

    if (!core_from_flag) {
        if (const char* env = std::getenv("SHADPS4_CORE"); env && *env) {
            core = env;
        }
    }
    if (core.empty()) {
        // No explicit selection: pick the core from the game's title id.
        core = AutoSelectCore(argc, argv);
    }
    if (!ValidCore(core)) {
        std::fprintf(stderr,
                     "[dispatcher] invalid core '%s' (expected 'gr2' or 'main')\n"
                     "  select with --core=gr2|main or SHADPS4_CORE=gr2|main\n",
                     core.c_str());
        return 2;
    }

    const int fwd_argc = static_cast<int>(fwd.size()) - 1; // exclude trailing nullptr
    return LoadAndRunCore(core, argv[0], fwd_argc, fwd.data());
}
