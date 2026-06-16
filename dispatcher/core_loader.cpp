// SPDX-License-Identifier: GPL-2.0-or-later
#include "core_loader.h"

#include <cstdio>
#include <cstring>
#include <string>

#include <dlfcn.h>

#if defined(COMBINED_EMBED_CORES)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// Provided by the .incbin stub (cmake/embed_cores.S.in). Each core's .so is
// embedded as a byte blob bracketed by these symbols.
extern "C" const unsigned char core_gr2_so_start[];
extern "C" const unsigned char core_gr2_so_end[];
extern "C" const unsigned char core_main_so_start[];
extern "C" const unsigned char core_main_so_end[];
#endif

using core_entry_fn = int (*)(int, char**);

namespace {

std::string DirOf(const char* path) {
    std::string p = path ? path : "";
    auto slash = p.find_last_of('/');
    return slash == std::string::npos ? std::string(".") : p.substr(0, slash);
}

int RunHandle(void* handle, const std::string& core, int argc, char** argv) {
    if (!handle) {
        std::fprintf(stderr, "[dispatcher] dlopen(core_%s) failed: %s\n", core.c_str(), dlerror());
        return -2;
    }
    dlerror(); // clear
    auto entry = reinterpret_cast<core_entry_fn>(dlsym(handle, "core_entry"));
    if (const char* err = dlerror()) {
        std::fprintf(stderr, "[dispatcher] dlsym(core_entry) failed in core_%s: %s\n",
                     core.c_str(), err);
        return -3;
    }
    return entry(argc, argv);
}

#if defined(COMBINED_EMBED_CORES)
int RunEmbedded(const std::string& core, int argc, char** argv) {
    const unsigned char* begin = nullptr;
    const unsigned char* end = nullptr;
    if (core == "gr2") {
        begin = core_gr2_so_start;
        end = core_gr2_so_end;
    } else if (core == "main") {
        begin = core_main_so_start;
        end = core_main_so_end;
    } else {
        std::fprintf(stderr, "[dispatcher] unknown core '%s'\n", core.c_str());
        return -1;
    }
    const size_t size = static_cast<size_t>(end - begin);

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
        ssize_t n = write(fd, begin + written, size - written);
        if (n <= 0) {
            std::perror("[dispatcher] write(memfd)");
            close(fd);
            return -5;
        }
        written += static_cast<size_t>(n);
    }

    char proc_path[64];
    std::snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
    void* handle = dlopen(proc_path, RTLD_NOW | RTLD_LOCAL);
    int rc = RunHandle(handle, core, argc, argv);
    // Intentionally leak the handle/fd: the core owns the process for its
    // lifetime, and unloading mid-run is neither needed nor safe.
    return rc;
}
#endif

int RunLoose(const std::string& core, const char* exe_path, int argc, char** argv) {
    const std::string so = DirOf(exe_path) + "/libcore_" + core + ".so";
    void* handle = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
    return RunHandle(handle, core, argc, argv);
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
