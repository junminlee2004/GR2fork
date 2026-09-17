// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <imgui.h>

namespace ImGui::FontStack {

ImFont* AddPrimaryUiFont(ImFontAtlas* atlas, float font_size, int console_language,
                         const ImFontConfig& base_cfg, bool include_cjk_fallback);

// Freeze the atlas after Build(): our backends upload it once, so a font must not
// rasterize a new glyph or size later (text measured outside a frame would grow the
// atlas or fail the pack). Frozen fonts fall back and scale a baked size.
void FreezeBakedFonts(ImFontAtlas* atlas);

} // namespace ImGui::FontStack
