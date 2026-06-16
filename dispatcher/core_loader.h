// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <string>

// Loads exactly one core (a -fvisibility=hidden shared object exporting a single
// `core_entry` symbol) and runs it. Returns the core's exit code, or a negative
// dispatcher error code on failure to load.
//
// `core`     : "gr2" or "main"
// `exe_path` : argv[0] of the dispatcher, used to (a) locate loose .so files in
//              dev builds and (b) hand the core a sensible argv[0].
// `argc/argv`: the argument vector to forward to the core's main (selector flag
//              already stripped by the caller).
int LoadAndRunCore(const std::string& core, const char* exe_path, int argc, char** argv);
