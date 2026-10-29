# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

target_sources(shadps4 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/gr2_gfx_trace.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gr2_gfx_trace.h
    ${CMAKE_CURRENT_LIST_DIR}/hook_table.inc)
