# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

target_sources(shadps4 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/frame_sync.cpp
    ${CMAKE_CURRENT_LIST_DIR}/frame_sync.h
    ${CMAKE_CURRENT_LIST_DIR}/gr2_native.cpp
    ${CMAKE_CURRENT_LIST_DIR}/gr2_native.h
    ${CMAKE_CURRENT_LIST_DIR}/marker.h)
