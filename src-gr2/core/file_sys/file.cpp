// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Core::FileSys::File I/O goes through its embedded Common::FS::IOFile, so this TU is
// intentionally empty (the upstream definitions are not declared on File in fs.h).
// TODO: either finish that refactor or delete this file and its CMakeLists.txt entry.

namespace Core::FileSys {} // namespace Core::FileSys
