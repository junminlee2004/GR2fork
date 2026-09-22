// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <utility>
#include <vector>

#include "common/types.h"

// ============================================================================
// Windows static guest red-zone protection
// ============================================================================

enum class WindowsGuestRedZoneProtectionMode : u32 {
    Disabled,
    StaticPatching,
};

namespace Core::WindowsGuestRedZoneProtection {

void SetActiveMode(WindowsGuestRedZoneProtectionMode mode) noexcept;
WindowsGuestRedZoneProtectionMode GetActiveMode() noexcept;
bool IsStaticPatchingEnabled() noexcept;

} // namespace Core::WindowsGuestRedZoneProtection

// ============================================================================
// End Windows static guest red-zone protection
// ============================================================================

namespace Core {

// Windows static guest red-zone protection
struct RedZonePatchResult {
    u64 function_count{};
    u64 instruction_count{};
    u64 red_zone_function_count{};
    u64 memory_instruction_count{};
    u64 short_memory_instruction_count{};
    u64 patched_memory_instruction_count{};
    u64 stack_dependent_memory_instruction_count{};
    u64 control_flow_memory_instruction_count{};
    u64 unrelocatable_memory_instruction_count{};
    u64 indirect_red_zone_function_count{};
    u64 cpu_patch_instruction_count{};
    u64 patched_cpu_patch_instruction_count{};
    u64 unsupported_cpu_patch_instruction_count{};
    u64 inplace_cpu_patch_instruction_count{};
};

/// Registers a module for patching, providing an area to generate trampoline code.
void RegisterPatchModule(void* module_ptr, u64 module_size, void* trampoline_area_ptr,
                         u64 trampoline_area_size);

/// Applies CPU patches that need to be done before beginning executions.
void PrePatchInstructions(u64 segment_addr, u64 segment_size);

// Windows static guest red-zone protection
/// Keeps Windows exception dispatch outside live guest red zones at faultable memory accesses.
RedZonePatchResult PatchRedZoneMemoryInstructions(u64 segment_addr, u64 segment_size,
                                                  std::span<const uintptr_t> function_starts);

/// static_cpu_patching: applies the CPU patches to every function the EH frame table lists,
/// decoding each by following its control flow; a patch site too short for a jump moves its
/// neighbours only when no known branch targets them. decoded_ranges receives the proven code
/// ranges, sorted and merged.
RedZonePatchResult PatchCpuInstructionsStatically(
    u64 segment_addr, u64 segment_size, std::span<const uintptr_t> function_starts,
    std::vector<std::pair<uintptr_t, uintptr_t>>& decoded_ranges);

struct GapPatchResult {
    u64 patched{};
    u64 left_to_handler{};
};

/// static_cpu_patching: the straight-through pass over the bytes outside decoded_ranges, applying
/// only in-place patches; a matching site it cannot patch that way is left to the trap handler.
GapPatchResult PrePatchInstructionGaps(
    u64 segment_addr, u64 segment_size,
    std::span<const std::pair<uintptr_t, uintptr_t>> decoded_ranges);

} // namespace Core
