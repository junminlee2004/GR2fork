// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <string>

// Loads one core (a -fvisibility=hidden shared object exporting a single `core_entry` symbol)
// and runs it; returns the core's exit code or a negative dispatcher error. `core` is "gr2" or
// "main"; `exe_path` (dispatcher argv[0]) locates loose dev .so files; argv arrives pre-stripped.
int LoadAndRunCore(const std::string& core, const char* exe_path, int argc, char** argv);
