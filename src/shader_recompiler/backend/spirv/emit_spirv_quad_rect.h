// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>
#include "common/types.h"

namespace Shader {
struct FragmentRuntimeInfo;
}

namespace Shader::Backend::SPIRV {

enum class AuxShaderType : u32 {
    RectListTCS,
    QuadListTCS,
    PassthroughTES,
};

// GR2FORK aux-TCS grass fix (v3.3):
// `vs_output_mask` is a bitmask where bit i = 1 iff the upstream VS writes to
// Param_i (i.e. info.stores.GetAny(IR::Attribute::Param0 + i) is true). The
// auxiliary TCS uses it to skip declaring Input variables for Locations the
// VS doesn't actually output — otherwise VUID-RuntimeSpirv-OpEntryPoint-08743
// fires (TCS declared Input at Location N but previous stage has no Output
// declared there) and RADV rejects the pipeline, silently dropping the draw.
// Only TCS aux types consume the mask; PassthroughTES ignores it (TES inputs
// come from the AuxTCS Outputs, which are still fully declared on all
// FS-expected Locations).
[[nodiscard]] std::vector<u32> EmitAuxilaryTessShader(AuxShaderType type,
                                                      const FragmentRuntimeInfo& fs_info,
                                                      u32 vs_output_mask = 0xFFFFFFFFu);

} // namespace Shader::Backend::SPIRV
