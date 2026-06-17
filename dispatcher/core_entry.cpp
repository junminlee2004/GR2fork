// SPDX-License-Identifier: GPL-2.0-or-later
// Per-core entry shim. Compiled once per core with -DCORE_REAL_MAIN=<core>_main.
//
// Two things are load-bearing here:
//  1. The renamed main is declared with NORMAL C++ linkage (NOT extern "C")
//     AND with main's EXACT parameter spelling `int(int, char*[])`, matching
//     shadPS4's `int main(int argc, char* argv[])`. -Dmain=<core>_main turns
//     `main` into an ordinary MANGLED symbol, so this declaration's mangling
//     must equal the definition's. Two ways to miss it:
//       - an `extern "C"` decl seeks an UNMANGLED name; and
//       - the subtle one: declaring the 2nd param `char**` instead of `char*[]`
//         misses on MSVC. The Microsoft mangler encodes an array-decayed
//         parameter (Q...) differently from a plain pointer (P...), so `char**`
//         -> ?<core>_main@@YAHHPEAPEAD@Z but `char*[]` (and the real main) ->
//         ?<core>_main@@YAHHQEAPEAD@Z. The Itanium ABI mangles BOTH to ...PPc,
//         which is why Linux linked with char** yet the Windows core link could
//         not find <core>_main. Keep this `char*[]`.
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

extern int CORE_REAL_MAIN(int, char*[]);

extern "C" CORE_EXPORT int core_entry(int argc, char** argv) {
    return CORE_REAL_MAIN(argc, argv);
}
