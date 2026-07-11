// SPDX-License-Identifier: GPL-2.0-or-later
//
// Loads exactly one core (a shared object/DLL exporting `core_entry`) and runs it. Linux uses
// dlopen on a memfd; Windows extracts the embedded bytes to %TEMP% and loads with LoadLibraryExW.
// Cores cannot interpose: ELF localizes all but core_entry (RTLD_LOCAL); COFF exports only it.
#include "core_loader.h"

#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <cwchar>
#else
#  include <dlfcn.h>
#endif

#if defined(COMBINED_EMBED_CORES)
#  if !defined(_WIN32)
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>
#  endif
// Provided by the .incbin stub (cmake/embed_cores*.S.in): each core is embedded as a byte blob
// bracketed by these symbols. x86_64 C symbols have no leading underscore on ELF or COFF, so the
// names match the assembler labels verbatim.
extern "C" const unsigned char core_gr2_so_start[];
extern "C" const unsigned char core_gr2_so_end[];
extern "C" const unsigned char core_main_so_start[];
extern "C" const unsigned char core_main_so_end[];
#endif

using core_entry_fn = int (*)(int, char**);

namespace {

// ---- thin library-handle abstraction ---------------------------------------
#if defined(_WIN32)
using lib_t = HMODULE;
lib_t lib_open(const wchar_t* path) {
    // LOAD_WITH_ALTERED_SEARCH_PATH: resolve the DLL's own deps relative to its
    // own dir (the cores are built fully static -- Workstream D -- so this rarely
    // matters, but it is the correct flag for an out-of-tree temp DLL).
    return ::LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}
void* lib_sym(lib_t h, const char* s) {
    return reinterpret_cast<void*>(::GetProcAddress(h, s));
}
std::string lib_err() {
    DWORD e = ::GetLastError();
    char buf[256] = {0};
    ::FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                     nullptr, e, 0, buf, sizeof(buf), nullptr);
    // FormatMessage leaves a trailing CRLF; trim it for clean single-line logs.
    std::string msg(buf);
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
        msg.pop_back();
    }
    return msg;
}
#else
using lib_t = void*;
lib_t lib_open(const char* path) {
    return ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
}
void* lib_sym(lib_t h, const char* s) {
    return ::dlsym(h, s);
}
std::string lib_err() {
    const char* e = ::dlerror();
    return e ? e : "(none)";
}
#endif

int RunHandle(lib_t handle, const std::string& core, int argc, char** argv) {
    if (!handle) {
        std::fprintf(stderr, "[dispatcher] load(core_%s) failed: %s\n",
                     core.c_str(), lib_err().c_str());
        return -2;
    }
    auto entry = reinterpret_cast<core_entry_fn>(lib_sym(handle, "core_entry"));
    if (!entry) {
        std::fprintf(stderr, "[dispatcher] core_entry missing in core_%s: %s\n",
                     core.c_str(), lib_err().c_str());
        return -3;
    }
    return entry(argc, argv);
}

#if defined(COMBINED_EMBED_CORES)
bool PickBlob(const std::string& core, const unsigned char*& b, const unsigned char*& e) {
    if (core == "gr2") {
        b = core_gr2_so_start;
        e = core_gr2_so_end;
        return true;
    }
    if (core == "main") {
        b = core_main_so_start;
        e = core_main_so_end;
        return true;
    }
    std::fprintf(stderr, "[dispatcher] unknown core '%s'\n", core.c_str());
    return false;
}

#if defined(_WIN32)
int RunEmbedded(const std::string& core, int argc, char** argv) {
    const unsigned char* b = nullptr;
    const unsigned char* e = nullptr;
    if (!PickBlob(core, b, e)) {
        return -1;
    }
    const size_t size = static_cast<size_t>(e - b);

    // Unique temp path per process+core: two instances (or both cores) must not
    // collide, and a previous run's file may still be mapped.
    wchar_t dir[MAX_PATH] = {0};
    ::GetTempPathW(MAX_PATH, dir);
    wchar_t path[MAX_PATH] = {0};
    swprintf(path, MAX_PATH, L"%sshadps4_core_%hs_%lu.dll", dir, core.c_str(),
             static_cast<unsigned long>(::GetCurrentProcessId()));

    HANDLE f = ::CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        std::fprintf(stderr, "[dispatcher] CreateFile(temp core) failed: %s\n",
                     lib_err().c_str());
        return -4;
    }
    const unsigned char* p = b;
    size_t left = size;
    while (left) {
        DWORD wrote = 0;
        DWORD chunk = (left > 0x10000000UL) ? 0x10000000UL : static_cast<DWORD>(left);
        if (!::WriteFile(f, p, chunk, &wrote, nullptr) || wrote == 0) {
            std::fprintf(stderr, "[dispatcher] WriteFile(temp core) failed: %s\n",
                         lib_err().c_str());
            ::CloseHandle(f);
            ::DeleteFileW(path);
            return -5;
        }
        p += wrote;
        left -= wrote;
    }
    ::CloseHandle(f); // must close the write handle before LoadLibrary maps it

    lib_t h = lib_open(path);
    int rc = RunHandle(h, core, argc, argv);
    // The DLL stays mapped for the core's whole lifetime; we only get here as the
    // process is exiting. Best-effort cleanup (a still-mapped DLL can't be
    // deleted -- the temp dir is reaped by the OS later either way).
    if (h) {
        ::FreeLibrary(h);
    }
    ::DeleteFileW(path);
    return rc;
}
#else
int RunEmbedded(const std::string& core, int argc, char** argv) {
    const unsigned char* b = nullptr;
    const unsigned char* e = nullptr;
    if (!PickBlob(core, b, e)) {
        return -1;
    }
    const size_t size = static_cast<size_t>(e - b);

    // Unique memfd name per core: glibc caches loaded objects by the
    // /proc/self/fd/N path string, so a reused fd number would alias cores.
    const std::string memname = "shadps4_core_" + core;
    int fd = memfd_create(memname.c_str(), MFD_CLOEXEC);
    if (fd < 0) {
        std::perror("[dispatcher] memfd_create");
        return -4;
    }
    size_t written = 0;
    while (written < size) {
        ssize_t n = write(fd, b + written, size - written);
        if (n <= 0) {
            std::perror("[dispatcher] write(memfd)");
            close(fd);
            return -5;
        }
        written += static_cast<size_t>(n);
    }

    char proc_path[64];
    std::snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
    // Intentionally leak the fd/handle: the core owns the process for its
    // lifetime, and unloading mid-run is neither needed nor safe.
    return RunHandle(lib_open(proc_path), core, argc, argv);
}
#endif
#endif // COMBINED_EMBED_CORES

std::string DirOf(const char* path) {
    std::string p = path ? path : "";
    auto slash = p.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : p.substr(0, slash);
}

int RunLoose(const std::string& core, const char* exe, int argc, char** argv) {
#if defined(_WIN32)
    // ASCII-only widening: fine for a dev build-dir path next to the exe. Use
    // MultiByteToWideChar if non-ASCII install paths ever matter here.
    std::string dll = DirOf(exe) + "\\libcore_" + core + ".dll";
    std::wstring w(dll.begin(), dll.end());
    return RunHandle(lib_open(w.c_str()), core, argc, argv);
#else
    std::string so = DirOf(exe) + "/libcore_" + core + ".so";
    return RunHandle(lib_open(so.c_str()), core, argc, argv);
#endif
}

} // namespace

int LoadAndRunCore(const std::string& core, const char* exe_path, int argc, char** argv) {
#if defined(COMBINED_EMBED_CORES)
    (void)exe_path;
    return RunEmbedded(core, argc, argv);
#else
    return RunLoose(core, exe_path, argc, argv);
#endif
}
