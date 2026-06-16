// SPDX-License-Identifier: GPL-2.0-or-later
// Per-core entry shim. Compiled once per core with -DCORE_REAL_MAIN=<core>_main.
//
// Two things are load-bearing here:
//  1. The renamed main is declared with NORMAL C++ linkage (NOT extern "C").
//     -Dmain=<core>_main turns `main` into an ordinary mangled symbol; an
//     `extern "C"` declaration would look for an unmangled name and miss it.
//  2. core_entry is forced to DEFAULT visibility so it survives the
//     -fvisibility=hidden the whole core is built with. Everything else in the
//     core (including statically-linked externals) is hidden + localized by the
//     version script, so two cores with identical internals never interpose.
extern int CORE_REAL_MAIN(int, char**);

extern "C" __attribute__((visibility("default")))
int core_entry(int argc, char** argv) {
    return CORE_REAL_MAIN(argc, argv);
}
