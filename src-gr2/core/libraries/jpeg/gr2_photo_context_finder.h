// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// GR2 Photo Mode: binary patcher + context finder. The context scan matches only the vtable ptr
// and self-pointer; the +0x08 magic (0x0001FFFF) is cleared at runtime, so it cannot be part of
// the fingerprint.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/types.h"

namespace Libraries::JpegEnc {

class GR2PhotoContextFinder {
public:
    static GR2PhotoContextFinder& Instance() {
        static GR2PhotoContextFinder inst;
        return inst;
    }

    void OnCreate(uintptr_t handle_addr, uintptr_t memory_addr, u32 memory_size) {
        std::lock_guard lock(mutex_);
        handle_addr_ = handle_addr;
        memory_addr_ = memory_addr;
        memory_size_ = memory_size;
        context_addr_ = 0;

        LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] OnCreate: handle={:#x} memory={:#x} size={:#x}",
                 handle_addr, memory_addr, memory_size);

        if (!patches_applied_) {
            // ApplyEbootPatches is deliberately not called: the gallery-browse patches
            // permanently modify the screenshot service's state-machine code, which breaks the
            // save menu after a photo capture (softlock). Photo capture needs no patches.
            LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] v60: eboot patches DISABLED (save-safety)");
            patches_applied_ = true;
        }
    }

    bool OnEncodeComplete(u32 jpeg_size, u32 width, u32 height) {
        std::lock_guard lock(mutex_);

        if (handle_addr_ == 0)
            return false;

        if (context_addr_ == 0)
            context_addr_ = ScanAllMappedMemory();

        if (context_addr_ == 0) {
            LOG_WARNING(Lib_Jpeg, "[GR2CtxFinder] Could not find photo context");
            return false;
        }

        const u32 state = ReadU32(context_addr_ + 0x960);
        const u64 scene_node = ReadU64(context_addr_ + 0x988);
        const u64 result_obj = ReadU64(context_addr_ + 0x9a0);

        LOG_INFO(Lib_Jpeg,
                 "[GR2CtxFinder] Context @{:#x}: state={:#x} scene={:#x} result={:#x}",
                 context_addr_, state, scene_node, result_obj);

        // Bump scene node version
        if (scene_node != 0) {
            const u32 ver = ReadU32(scene_node + 0x30);
            WriteU32(scene_node + 0x30, ver + 1);
            LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] scene_node version {} -> {}", ver, ver + 1);
        }

        // Set result ready flag
        if (result_obj != 0) {
            const u8 old = ReadU8(result_obj + 0x30);
            WriteU8(result_obj + 0x30, 1);
            const s32 count = ReadS32(result_obj + 0x18);
            LOG_INFO(Lib_Jpeg,
                     "[GR2CtxFinder] result ready: {} -> 1, item_count={}",
                     old, count);
        }

        return true;
    }

    void DumpState() const {
        std::lock_guard lock(mutex_);
        if (context_addr_ == 0) {
            LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] No context found yet");
            return;
        }

        const u32 state = ReadU32(context_addr_ + 0x960);
        const u64 scene = ReadU64(context_addr_ + 0x988);
        const u64 result = ReadU64(context_addr_ + 0x9a0);
        const u32 ver = ReadU32(context_addr_ + 0x9b8);
        const u64 render = ReadU64(context_addr_ + 0xa10);

        LOG_INFO(Lib_Jpeg,
                 "[GR2CtxFinder] DUMP @{:#x}: state={:#x} scene={:#x} result={:#x} "
                 "cached_ver={} render={:#x}",
                 context_addr_, state, scene, result, ver, render);

        if (scene != 0)
            LOG_INFO(Lib_Jpeg, "[GR2CtxFinder]   scene+0x30={}", ReadU32(scene + 0x30));
        if (result != 0)
            LOG_INFO(Lib_Jpeg, "[GR2CtxFinder]   result+0x30={} result+0x18={}",
                     ReadU8(result + 0x30), ReadS32(result + 0x18));
    }

private:
    GR2PhotoContextFinder() = default;

    void ApplyEbootPatches() {
        const uintptr_t base = MemoryPatcher::g_eboot_address;
        if (base == 0) {
            LOG_ERROR(Lib_Jpeg, "[GR2CtxFinder] g_eboot_address is 0, cannot patch");
            return;
        }

        LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Applying eboot patches (base={:#x})", base);

        // Patch A: VA 0x134080 - version callback: JNE->JMP
        {
            u8* p = reinterpret_cast<u8*>(base + 0x134080);
            if (p[0] == 0x75 && p[1] == 0x01) {
                p[0] = 0xEB;
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch A applied: VA 0x134080 JNE->JMP");
            } else if (p[0] == 0xEB) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch A already applied");
            } else {
                LOG_WARNING(Lib_Jpeg,
                            "[GR2CtxFinder] Patch A SKIPPED: {:02x} {:02x} at 0x134080",
                            p[0], p[1]);
            }
        }

        // Patch B: VA 0x18cebf - result ready check: JE->NOP*6
        {
            u8* p = reinterpret_cast<u8*>(base + 0x18cebf);
            if (p[0] == 0x0F && p[1] == 0x84) {
                std::memset(p, 0x90, 6);
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch B applied: VA 0x18cebf JE->NOP*6");
            } else if (p[0] == 0x90) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch B already applied");
            } else {
                LOG_WARNING(Lib_Jpeg,
                            "[GR2CtxFinder] Patch B SKIPPED: {:02x} {:02x} at 0x18cebf",
                            p[0], p[1]);
            }
        }

        // Patch C (VA 0x1382ea): the gallery state machine (VA 0x1380c0) dereferences the data
        // provider at widget+0x9a0, which shadPS4 never allocates. The 6-byte cmpb+je readiness
        // check becomes test rdi,rdi / je retry / nop: NULL skips safely, non-NULL falls through.
        {
            u8* p = reinterpret_cast<u8*>(base + 0x1382ea);
            if (p[0] == 0x80 && p[1] == 0x7f && p[2] == 0x30 && p[3] == 0x00 &&
                p[4] == 0x74 && p[5] == 0x62) {
                // Replace: cmpb $0x0,0x30(%rdi) / je +0x62
                // With:    test %rdi,%rdi / je +0x63 / nop
                p[0] = 0x48; p[1] = 0x85; p[2] = 0xFF;  // test %rdi, %rdi
                p[3] = 0x74; p[4] = 0x63;                 // je +0x63 (-> VA 0x138352)
                p[5] = 0x90;                               // nop
                LOG_INFO(Lib_Jpeg,
                    "[GR2CtxFinder] Patch C applied: VA 0x1382ea "
                    "cmpb+je -> test rdi+je (NULL-guard data provider)");
            } else if (p[0] == 0x48 && p[1] == 0x85) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch C already applied");
            } else {
                LOG_WARNING(Lib_Jpeg,
                    "[GR2CtxFinder] Patch C SKIPPED: {:02x} {:02x} {:02x} {:02x} "
                    "{:02x} {:02x} at 0x1382ea",
                    p[0], p[1], p[2], p[3], p[4], p[5]);
            }
        }

        // Patch D (VA 0x132b67b): contentId validation (call 0x10329c0) returns < 0 in shadPS4;
        // the result is stored to r12+0x448 and js branches to the LOCKED error path. Force eax=0
        // before the store and NOP the test+js so validation always succeeds.
        {
            u8* p = reinterpret_cast<u8*>(base + 0x132b67b);
            if (p[0] == 0x41 && p[1] == 0x89 && p[8] == 0x85 && p[10] == 0x0F) {
                // xor eax, eax
                p[0] = 0x31; p[1] = 0xC0;
                // mov [r12+0x448], eax (shifted 2 bytes forward)
                p[2] = 0x41; p[3] = 0x89; p[4] = 0x84; p[5] = 0x24;
                p[6] = 0x48; p[7] = 0x04; p[8] = 0x00; p[9] = 0x00;
                // nop*6
                p[10] = 0x90; p[11] = 0x90; p[12] = 0x90;
                p[13] = 0x90; p[14] = 0x90; p[15] = 0x90;
                LOG_INFO(Lib_Jpeg,
                    "[GR2CtxFinder] Patch D applied: VA 0x132b67b "
                    "force validation result=0 (16 bytes)");
            } else if (p[0] == 0x31 && p[1] == 0xC0) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch D already applied");
            } else {
                LOG_WARNING(Lib_Jpeg,
                    "[GR2CtxFinder] Patch D SKIPPED: {:02x} {:02x} {:02x} at 0x132b67b",
                    p[0], p[1], p[2]);
            }
        }

        // Patch E (VA 0x132b8ce): the state -1 download check (PLT 0x13b46b0) always returns
        // non-zero in shadPS4 (no download service), so entries stall forever. NOP the 6-byte jne
        // to advance entries to state 6; combined with Patch D they reach state 7.
        {
            u8* p = reinterpret_cast<u8*>(base + 0x132b8ce);
            if (p[0] == 0x0F && p[1] == 0x85) {
                std::memset(p, 0x90, 6);
                LOG_INFO(Lib_Jpeg,
                    "[GR2CtxFinder] Patch E applied: VA 0x132b8ce "
                    "JNE->NOP*6 (skip download check → state 6)");
            } else if (p[0] == 0x90) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch E already applied");
            } else {
                LOG_WARNING(Lib_Jpeg,
                    "[GR2CtxFinder] Patch E SKIPPED: {:02x} {:02x} at 0x132b8ce",
                    p[0], p[1]);
            }
        }

        // Patch F (VA 0x13421d0): isNotReady() tests the entry-ready flag at +0x159, which only
        // the PS4 screenshot download service sets; it stays 0 in shadPS4, locking every entry.
        // Replaced with xor eax,eax / ret / nop*8 so entries always report ready.
        {
            u8* p = reinterpret_cast<u8*>(base + 0x13421d0);
            if (p[0] == 0x80 && p[1] == 0xBF && p[2] == 0x59 && p[7] == 0x0F) {
                p[0] = 0x31; p[1] = 0xC0;  // xor eax, eax
                p[2] = 0xC3;                // ret
                std::memset(p + 3, 0x90, 8); // nop padding
                LOG_INFO(Lib_Jpeg,
                    "[GR2CtxFinder] Patch F applied: VA 0x13421d0 "
                    "isNotReady() → always return false (entries unlocked)");
            } else if (p[0] == 0x31 && p[1] == 0xC0 && p[2] == 0xC3) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch F already applied");
            } else {
                LOG_WARNING(Lib_Jpeg,
                    "[GR2CtxFinder] Patch F SKIPPED: {:02x} {:02x} {:02x} at 0x13421d0",
                    p[0], p[1], p[2]);
            }
        }

        // No patches for 0xc24d20 / 0xc261f0: both are shared with other widget types (reflection
        // table 0x19c9880), and patching them destroys textures on non-gallery items (green
        // artifacts).

        // Patch I: VA 0x132b4f6 - state 2 -> gstate=9 one-shot
        // Must NOT call post_action (PLT alloc with uninitialized handles blocks).
        {
            u8* p = reinterpret_cast<u8*>(base + 0x132b4f6);
            if (p[0] == 0xE8 && p[5] == 0x85 && p[6] == 0xC0) {
                // Fresh: NOP*22, then gstate=9, jmp exit
                const u8 patch[] = {
                    0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
                    0x90, 0x90, 0x90, 0x90, 0x90,
                    0x90, 0x90, 0x90, 0x90, 0x90,
                    0x90, 0x90, 0x90, 0x90, 0x90,
                    0x41, 0xC6, 0x44, 0x24, 0x1C, 0x09,
                    0xE9, 0x8B, 0x06, 0x00, 0x00, 0x90
                };
                std::memcpy(p, patch, 34);
                LOG_INFO(Lib_Jpeg,
                    "[GR2CtxFinder] Patch I applied (no post_action)");
            } else if (p[0] == 0x31 && p[1] == 0xF6) {
                // Old version with post_action call - NOP it
                std::memset(p, 0x90, 12);
                LOG_INFO(Lib_Jpeg,
                    "[GR2CtxFinder] Patch I FIXED: NOP'd post_action call");
            } else if (p[0] == 0x90) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch I already applied");
            }
        }

        // Patch J: REMOVED - Patch K prevents the worker at the call site.

        // Patch K: VA 0x132b4ec - NOP gallery-specific call to 0x5590e0
        {
            u8* p = reinterpret_cast<u8*>(base + 0x132b4ec);
            if (p[0] == 0xE8) {
                std::memset(p, 0x90, 5);
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch K applied");
            } else if (p[0] == 0x90) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch K already applied");
            }
        }

        // Patch P: VA 0x558035 - force success exit after browse update
        {
            u8* p = reinterpret_cast<u8*>(base + 0x558035);
            if (p[0] == 0x84 && p[1] == 0xC0 && p[2] == 0x0F && p[3] == 0x85) {
                p[0] = 0xE9; p[1] = 0xDB; p[2] = 0x01; p[3] = 0x00; p[4] = 0x00;
                p[5] = 0x90; p[6] = 0x90; p[7] = 0x90;
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch P applied");
            } else if (p[0] == 0xE9 && p[1] == 0xDB) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch P already applied");
            }
        }

        // Patch Q: VA 0x558268 - NOP browse_data gate in 0x558230
        {
            u8* p = reinterpret_cast<u8*>(base + 0x558268);
            if (p[0] == 0x48 && p[1] == 0x83 && p[2] == 0x3D &&
                p[8] == 0x0F && p[9] == 0x84) {
                std::memset(p, 0x90, 14);
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch Q applied");
            } else if (p[0] == 0x90) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch Q already applied");
            }
        }

        // Patch R: Force-skip fiber init blocks (jne->jmp)
        {
            u8* p1 = reinterpret_cast<u8*>(base + 0x5583ad);
            if (p1[0] == 0x0F && p1[1] == 0x85) {
                p1[0] = 0xE9; p1[1] = 0x8D; p1[2] = 0x00;
                p1[3] = 0x00; p1[4] = 0x00; p1[5] = 0x90;
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch R1 applied");
            }
            u8* p2 = reinterpret_cast<u8*>(base + 0x5580f5);
            if (p2[0] == 0x75) {
                p2[0] = 0xEB;
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch R2 applied");
            }
            u8* p3 = reinterpret_cast<u8*>(base + 0x558185);
            if (p3[0] == 0x0F && p3[1] == 0x85) {
                p3[0] = 0xE9; p3[1] = 0x81; p3[2] = 0x00;
                p3[3] = 0x00; p3[4] = 0x00; p3[5] = 0x90;
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch R3 applied");
            }
        }

        // Patch L REMOVED - shared renderer function

        // Patches N, O: NOT applied from here - applied in content_search.cpp.

        // Patch S: VA 0x5582be/0x5582c9/0x5582d4 - NOP per-frame zeroing
        // of cb+0x18/0x20/0x38 in 0x558230 registration block.
        {
            struct ZeroPatch { u64 va; const char* name; };
            ZeroPatch zp[] = {
                {0x5582be, "cb+0x18"},
                {0x5582c9, "cb+0x20"},
                {0x5582d4, "cb+0x38"},
            };
            for (auto& z : zp) {
                u8* p = reinterpret_cast<u8*>(base + z.va);
                if (p[0] == 0x48 && p[1] == 0xC7 && p[2] == 0x05) {
                    std::memset(p, 0x90, 11);
                    LOG_INFO(Lib_Jpeg,
                        "[GR2CtxFinder] Patch S applied: VA {:#x} NOP*11 ({})",
                        z.va, z.name);
                }
            }
        }

        // Patch T: VA 0x468524 - NOP fiber creation in screen registration
        {
            u8* p = reinterpret_cast<u8*>(base + 0x468524);
            if (p[0] == 0xE8 && p[1] == 0x37 && p[2] == 0xB2) {
                std::memset(p, 0x90, 5);
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch T applied");
            } else if (p[0] == 0x90) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch T already applied");
            }
        }

        // Patch U2: DISABLED - 0x46c4d0 blocks without fiber
        {
            u8* p = reinterpret_cast<u8*>(base + 0x132b7ec);
            if (p[0] == 0x49 && p[1] == 0x8B && p[2] == 0x84) {
                p[0] = 0xE9; p[1] = 0xB1; p[2] = 0x03;
                p[3] = 0x00; p[4] = 0x00;
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Reverted Patch U2");
            } else if (p[0] == 0x31 && p[1] == 0xC0) {
                p[0] = 0xE9; p[1] = 0xB1; p[2] = 0x03;
                p[3] = 0x00; p[4] = 0x00;
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Reverted Patch U");
            }
        }

        // Patch V: VA 0x132bbb3 - NOP destructive vtable[3] call
        {
            u8* p = reinterpret_cast<u8*>(base + 0x132bbb3);
            if (p[0] == 0xFF && p[1] == 0x50 && p[2] == 0x18) {
                p[0] = 0xB0; p[1] = 0x01; p[2] = 0x90;
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch V applied");
            } else if (p[0] == 0xB0 && p[1] == 0x01) {
                LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Patch V already applied");
            }
            // Revert old Patch V/W if present
            u8* g = reinterpret_cast<u8*>(base + 0x132bbaa);
            if (g[0] == 0xEB && g[1] == 0x18) { g[0] = 0x7C; }
            u8* w = reinterpret_cast<u8*>(base + 0x132b7bb);
            if (w[0] == 0xE9 && w[1] == 0x04 && w[2] == 0x04) {
                w[1] = 0xEC; w[2] = 0x03;
            }
        }

        vtable_ptr_ = base + 0x1648820;
        LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Context vtable = {:#x}", vtable_ptr_);
    }

    // Context scan checks only immutable fields: +0x00 vtable = base + 0x1648820 and +0x10
    // self-pointer, both set once in init. The +0x08 magic (0x0001FFFF) is cleared at runtime and
    // is not checked.

    struct MemRegion { uintptr_t start, end; };

    static std::vector<MemRegion> GetMappedRegions() {
        std::vector<MemRegion> regions;
        std::ifstream maps("/proc/self/maps");
        if (!maps.is_open()) return regions;

        std::string line;
        while (std::getline(maps, line)) {
            unsigned long long start = 0, end = 0;
            char perms[5] = {};
            if (std::sscanf(line.c_str(), "%llx-%llx %4s", &start, &end, perms) < 3)
                continue;
            if (perms[0] != 'r') continue;
            const auto s = static_cast<uintptr_t>(start);
            auto e = static_cast<uintptr_t>(end);
            if (!((s >= 0x200000000ULL && s < 0x300000000ULL) ||
                  (s >= 0x800000000ULL && s < 0x900000000ULL)))
                continue;
            if (e - s > 256ULL * 1024 * 1024)
                e = s + 256ULL * 1024 * 1024;
            regions.push_back({s, e});
        }
        return regions;
    }

    uintptr_t ScanAllMappedMemory() const {
        if (vtable_ptr_ == 0) return 0;

        const auto regions = GetMappedRegions();
        LOG_INFO(Lib_Jpeg, "[GR2CtxFinder] Scanning {} regions for vtable={:#x}",
                 regions.size(), vtable_ptr_);

        constexpr size_t CTX_SIZE = 0xa50;
        constexpr size_t ALIGN = 0x10;
        int vtable_hits = 0;

        for (const auto& reg : regions) {
            const uintptr_t scan_start = (reg.start + ALIGN - 1) & ~(ALIGN - 1);
            if (reg.end < scan_start + CTX_SIZE)
                continue;

            for (uintptr_t addr = scan_start;
                 addr + CTX_SIZE <= reg.end; addr += ALIGN) {

                // Check vtable pointer at +0x00
                if (ReadU64(addr) != vtable_ptr_)
                    continue;

                vtable_hits++;

                // Log everything about this candidate for diagnostics
                const u64 field08 = ReadU64(addr + 0x08);
                const u64 field10 = ReadU64(addr + 0x10);
                const u32 state = ReadU32(addr + 0x960);
                LOG_INFO(Lib_Jpeg,
                         "[GR2CtxFinder] Vtable match #{} @{:#x}: +0x08={:#x} +0x10={:#x} "
                         "state={:#x}",
                         vtable_hits, addr, field08, field10, state);

                // Only require self-pointer at +0x10
                if (field10 == static_cast<u64>(addr)) {
                    LOG_INFO(Lib_Jpeg,
                             "[GR2CtxFinder] FOUND context @{:#x} (self-ptr confirmed)",
                             addr);
                    return addr;
                }

                // Self-pointer may point to +0x10 instead of +0x00
                // (init does: lea rdi, [r13+0x10]; ... mov [r13+0x10], r13)
                // So check if +0x10 points to addr (ctx = r13, stored at r13+0x10)
                if (field10 == static_cast<u64>(addr + 0x10) ||
                    field10 == static_cast<u64>(addr - 0x10)) {
                    LOG_INFO(Lib_Jpeg,
                             "[GR2CtxFinder] FOUND context @{:#x} (offset self-ptr)",
                             addr);
                    return addr;
                }

                // If vtable matches but self-pointer doesn't, maybe this IS
                // the context but init stored something else at +0x10.
                // Accept it if state looks like a photo state.
                if (state >= 1 && state <= 0x40 && (state & (state - 1)) == 0) {
                    LOG_INFO(Lib_Jpeg,
                             "[GR2CtxFinder] FOUND context @{:#x} (vtable + power-of-2 state)",
                             addr);
                    return addr;
                }
            }
        }

        LOG_WARNING(Lib_Jpeg,
                    "[GR2CtxFinder] No context found ({} vtable matches in {} regions)",
                    vtable_hits, regions.size());
        return 0;
    }

    // -----------------------------------------------------------------------
    static u8  ReadU8 (uintptr_t a) { return *reinterpret_cast<const volatile u8*>(a);  }
    static u32 ReadU32(uintptr_t a) { return *reinterpret_cast<const volatile u32*>(a); }
    static s32 ReadS32(uintptr_t a) { return *reinterpret_cast<const volatile s32*>(a); }
    static u64 ReadU64(uintptr_t a) { return *reinterpret_cast<const volatile u64*>(a); }
    static void WriteU8 (uintptr_t a, u8  v) { *reinterpret_cast<volatile u8*>(a)  = v; }
    static void WriteU32(uintptr_t a, u32 v) { *reinterpret_cast<volatile u32*>(a) = v; }

    mutable std::mutex mutex_;
    uintptr_t handle_addr_{0};
    uintptr_t memory_addr_{0};
    u32 memory_size_{0};
    uintptr_t context_addr_{0};
    uintptr_t vtable_ptr_{0};
    bool patches_applied_{false};
};

} // namespace Libraries::JpegEnc
