// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#define STB_IMAGE_IMPLEMENTATION
// Photo Mode needs JPEG; do not restrict stb_image to PNG-only.
#define STBI_NO_STDIO
#include "common/stb.h"
