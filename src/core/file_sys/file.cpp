// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// NOTE: File I/O for Core::FileSys::File goes through its embedded
// Common::FS::IOFile (see common/io_file.cpp). The methods that were
// stubbed here (Open/Read/Write/Pread/Pwrite/SetSize/Flush/Lseek/Unlink)
// were never declared on the File struct in fs.h, so this TU is intentionally
// empty. Delete this file and remove it from CMakeLists.txt if you don't
// plan to finish the refactor.

namespace Core::FileSys {} // namespace Core::FileSys
