// SPDX-License-Identifier: GPL-2.0-or-later
// Per-core entry shim. Compiled once per core with -DCORE_REAL_MAIN=<core>_main.
//
// Two things are load-bearing here:
//  1. The renamed main is declared with NORMAL C++ linkage (NOT extern "C").
//     -Dmain=<core>_main turns `main` into an ordinary mangled symbol; an
//     `extern "C"` declaration would look for an unmangled name and miss it.
//  2. core_entry is the ONE symbol the dispatcher resolves out of a core, so it
//     must be the only thing the core exports.
//       - ELF (Linux): the whole core is built -fvisibility=hidden and the
//         version script localizes everything else (incl. statically-linked
//         externals), so two cores with identical internals never interpose
//         under ELF's flat global symbol namespace. core_entry has to be forced
//         back to DEFAULT visibility to survive that.
//       - COFF (Windows): a DLL exports NOTHING unless marked, so __declspec
//         (dllexport) makes core_entry the *sole* export automatically -- there
//         is no global namespace to interpose in and each DLL resolves its own
//         imports at link time, which is why the version script is unnecessary
//         (and skipped) on Windows.
#if defined(_WIN32)
#  define CORE_EXPORT __declspec(dllexport)
#else
#  define CORE_EXPORT __attribute__((visibility("default")))
#endif

extern int CORE_REAL_MAIN(int, char**);

extern "C" CORE_EXPORT int core_entry(int argc, char** argv) {
    return CORE_REAL_MAIN(argc, argv);
}
