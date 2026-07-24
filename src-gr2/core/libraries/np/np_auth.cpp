// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <mutex>
#include "common/config.h"
#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "core/libraries/np/gr2_online_auth.h"
#include "core/libraries/np/np_auth.h"
#include "core/libraries/np/np_auth_error.h"
#include "core/libraries/np/np_error.h"
#include "core/libraries/system/userservice.h"

// Make a code/stub page RWX for the trampolines. POSIX = mprotect, Windows = VirtualProtect.
// (windows.h included AFTER the project headers -- same order crash_handler.cpp uses safely.)
#ifdef _WIN32
#include <windows.h>
static inline int Gr2MakeRWX(void* addr, std::size_t len) {
    DWORD old = 0;
    return VirtualProtect(addr, len, PAGE_EXECUTE_READWRITE, &old) ? 0 : -1;
}
#else
#include <sys/mman.h>
static inline int Gr2MakeRWX(void* addr, std::size_t len) {
    return mprotect(addr, len, PROT_READ | PROT_WRITE | PROT_EXEC);
}
#endif

// === GR2 ONLINE PROBE (temporary diagnostic) ================================
#include "common/elf_info.h"
#include "common/singleton.h"
#include "core/linker.h"
#include "core/tls.h"  // Core::ExecuteGuest -- call guest functions (the challenge populator)
// ===========================================================================

namespace Libraries::Np::NpAuth {

static bool g_signed_in = false;
static s32 g_active_auth_requests = 0;
static std::mutex g_auth_request_mutex;

// GR2 online-flow runtime probe: dumps the 0x4b0 env block to locate where capone_url lands.
// DAT_01c02b60 -> heap online-manager, state = *(*global + 0x30); client_ver(+0xc8) parses as 110
// but login_url(+0x258) and capone_host(+0x348) stay 0. runtime_va = base + (ghidra - 0x107BF0).

// 0 == use the Linker lookup; a nonzero value overrides it with a fixed eboot base.
static constexpr u64 GR2_MANUAL_BASE = 0;

static u64 Gr2ModuleBase() {
    if constexpr (GR2_MANUAL_BASE != 0) {
        return GR2_MANUAL_BASE;
    }
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    if (linker != nullptr) {
        auto* mod = linker->GetModule(0);
        if (mod != nullptr) {
            return mod->base_virtual_addr;
        }
    }
    return 0;
}

template <typename T>
static T Gr2ReadGlobal(u64 ghidra_addr) {
    const u64 base = Gr2ModuleBase();
    if (base == 0) {
        return T{};
    }
    return *reinterpret_cast<volatile T*>(base + (ghidra_addr - 0x107BF0));
}

// Read up to 47 printable chars from a guest VA into out (NUL-terminated). Returns length.
static u32 Gr2ReadCStr(u64 va, char* out) {
    const volatile char* sp = reinterpret_cast<const volatile char*>(va);
    u32 n = 0;
    for (; n < 47; ++n) {
        const char c = sp[n];
        if (c == 0) {
            break;
        }
        out[n] = (c >= 0x20 && c < 0x7f) ? c : '.';
    }
    out[n] = 0;
    return n;
}

// One-time dump of the active env block. The slot and ASCII stages read only the block itself,
// a mapped module global.
static void Gr2DumpEnvBlock(u64 base, s32 env) {
    const u64 envblk = base + (0x1c26520 - 0x107BF0) + static_cast<u64>(env) * 0x4b0;
    LOG_INFO(Lib_NpAuth, "GR2-ENVDUMP env={} block={:#x} size=0x4b0", env, envblk);

    // (a) non-zero 8-byte slots -> tells us which fields are populated and which look like ptrs
    for (u32 off = 0; off < 0x4b0; off += 8) {
        const u64 v = *reinterpret_cast<volatile u64*>(envblk + off);
        if (v != 0) {
            LOG_INFO(Lib_NpAuth, "GR2-ENVDUMP  slot +{:#05x} = {:#018x}", off, v);
        }
    }
    // (b) ASCII view -> reveals any INLINE strings stored directly in the block
    for (u32 off = 0; off < 0x4b0; off += 48) {
        char row[49];
        for (u32 j = 0; j < 48; ++j) {
            const u8 c = *reinterpret_cast<volatile u8*>(envblk + off + j);
            row[j] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
        }
        row[48] = 0;
        LOG_INFO(Lib_NpAuth, "GR2-ENVDUMP  ascii +{:#05x}: {}", off, row);
    }
#if !defined(_WIN32)
    // (c) follow heap-range pointers -> reveals strings stored BY POINTER (std::string/char*).
    // Linux-only: the range filter also passes non-pointer slot values (flag words), and chasing
    // one reads harmless garbage on Linux's fully-readable guest arena but faults fatally on a
    // Windows placeholder page - outside the null window the access-violation absorber claims.
    for (u32 off = 0; off < 0x4b0; off += 8) {
        const u64 p = *reinterpret_cast<volatile u64*>(envblk + off);
        if (p >= 0x100000000ull && p < 0x1000000000ull) {
            char buf[64];
            const u32 n = Gr2ReadCStr(p, buf);
            if (n > 0) {
                LOG_INFO(Lib_NpAuth, "GR2-ENVDUMP  ptr  +{:#05x} -> {:#x} '{}'", off, p, buf);
            }
        }
    }
#endif
}

// GR2 challenge cat-0 catalog dump (read-only). DAT_01bea890 -> catalog; FUN_00e829a0 places
// per-mission blocks at catalog + m*0xcd0 (cm01=0 .. cm20=19), 10 slots each at +8 + i*0x148,
// cm-FNV at +0xb4 (cm12 = 0x76f0fed2). Dumps raw cm12 + cm01 slots to pin the exact layout.
static void Gr2DumpCatalogHex(u64 addr, u32 len, const char* tag) {
    for (u32 off = 0; off < len; off += 16) {
        u8 b[16];
        char asc[17];
        for (u32 j = 0; j < 16; ++j) {
            b[j] = *reinterpret_cast<volatile u8*>(addr + off + j);
            asc[j] = (b[j] >= 0x20 && b[j] < 0x7f) ? (char)b[j] : '.';
        }
        asc[16] = 0;
        LOG_INFO(Lib_NpAuth,
                 "GR2-CAT0 {} +{:#05x}: {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} "
                 "{:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x}  {}",
                 tag, off, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
                 b[12], b[13], b[14], b[15], asc);
    }
}

// Diagnostic: FUN_00e828f0 builds the announcement render list from the master item list at
// catalog+0x25e60..+0x25e68, keeping items with +0x14 state !=0 && !=3, (+0x16 & 1)==0, cat!=7,
// and a pass of gate FUN_00e827a0 (tags +0x18..+0x1f vs catalog+0x25f98). Dumps active slots.
static void Gr2DumpMasterList(u64 base) {
    (void)base;
#if defined(_WIN32)
    // Linux-only telemetry: the slot walk dereferences heap pointers behind range filters only,
    // and a partially-initialized entry turns that into a fatal fault on a Windows placeholder
    // page, outside the null window the access-violation absorber claims.
    return;
#endif
    // Scan the master pool for active slots (st != 0) every ~1200 ticks, catching what the MAIL
    // receive populates after boot: cat / state / tag / mission-FNV(+0xb4) / name bytes(+0xa0).
    // cat=0 is a received Challenge; cat=10 a mine notification.
    static u32 tick = 0;
    if ((tick++ % 1200) != 0) {
        return;
    }
    const u64 mgr = Gr2ReadGlobal<u64>(0x1bea890);
    if (mgr < 0x10000) {
        return;
    }
    const u64 begin = *reinterpret_cast<volatile u64*>(mgr + 0x25e60);
    const u64 end = *reinterpret_cast<volatile u64*>(mgr + 0x25e68);
    if (begin < 0x10000 || end <= begin || (end - begin) > 0x10000) {
        return;
    }
    const u32 n = static_cast<u32>((end - begin) >> 3);
    u32 active = 0;
    for (u32 i = 0; i < n; ++i) {
        const u64 it = *reinterpret_cast<volatile u64*>(begin + i * 8);
        if (it < 0x10000) {
            continue;
        }
        const u16 st = *reinterpret_cast<volatile u16*>(it + 0x14);
        if (st == 0) {
            continue;  // empty slot
        }
        ++active;
        char nm[17];
        for (u32 j = 0; j < 16; ++j) {
            const u8 c = *reinterpret_cast<volatile u8*>(it + 0xa0 + j);
            nm[j] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
        }
        nm[16] = 0;
        LOG_INFO(Lib_NpAuth,
                 "GR2-ACTIVE: [{}] item@{:#x} cat={} st={} tag18={:#x} cmFNV={:#x} name='{}'", i, it,
                 static_cast<u32>(*reinterpret_cast<volatile u16*>(it + 0x12)), static_cast<u32>(st),
                 static_cast<u32>(*reinterpret_cast<volatile u8*>(it + 0x18)),
                 static_cast<u32>(*reinterpret_cast<volatile u32*>(it + 0xb4)), nm);
    }
    LOG_INFO(Lib_NpAuth, "GR2-ACTIVE: scan n={} active(st!=0)={} flag25fba={:#x}", n, active,
             static_cast<u32>(*reinterpret_cast<volatile u8*>(mgr + 0x25fba)));
}

static u64 g_hook_catg = 0;  // runtime address of DAT_01bea890 (holds the catalog ptr)
static volatile u64 g_refill_count = 0;   // # times the gate trampoline (Gr2RefillSlot) ran
static volatile int g_refill_inlist = -1;  // was our slot in the master list at the last enroll instant?
static volatile long g_refill_idx = -1;    // ...at which index
static volatile long g_refill_n = -1;      // ...master-list length at that instant

// Re-asserts the synthetic cm12 slot inside FUN_00e828f0's enroll loop: the slot can be reset
// between the FUN_00fdf150-entry fill and the enroll read (FUN_00e81ac0 + a vtable refresh run in
// that gap), so the gate call FUN_00e827a0 is repointed at a trampoline that refills, returns 1.
extern "C" __attribute__((no_stack_protector)) void Gr2RefillSlot() {
    if (g_hook_catg == 0) {
        return;
    }
    const u64 cat = *reinterpret_cast<volatile u64*>(g_hook_catg);
    if (cat < 0x10000) {
        return;
    }
    const u64 S = cat + 8 + 11 * 0xcd0;  // cm12 slot 0
    if (*reinterpret_cast<volatile u16*>(S + 0x12) > 0x100) {
        return;  // doesn't look like the pool (don't scribble random heap)
    }
    *reinterpret_cast<volatile u16*>(S + 0x10) = 1;            // id (not 0xffff)
    *reinterpret_cast<volatile u16*>(S + 0x12) = 0;            // category Challenge
    *reinterpret_cast<volatile u32*>(S + 0x14) = 1;            // state active (enroll precheck)
    *reinterpret_cast<volatile u8*>(S + 0x16) &= 0xfe;         // clear hide bit
    *reinterpret_cast<volatile u32*>(S + 0xb4) = 0x76f0fed2u;  // cm12 FNV
    *reinterpret_cast<volatile u8*>(S + 0x18) = 0xff;          // tag
    // Runs inside FUN_00e828f0's enroll loop, after the vtable refresh. inlist=0 here while the
    // build-entry check saw inlist=1 means the refresh rebuilt the master list and dropped the
    // synthetic slot, so the enumerator never visits it.
    g_refill_count = g_refill_count + 1;
    const u64 lb = *reinterpret_cast<volatile u64*>(cat + 0x25e60);
    const u64 le = *reinterpret_cast<volatile u64*>(cat + 0x25e68);
    int inl = 0;
    long idx = -1;
    long nn = -1;
    if (lb >= 0x10000 && le > lb && (le - lb) <= 0x10000) {
        nn = static_cast<long>((le - lb) >> 3);
        for (long i = 0; i < nn; ++i) {
            if (*reinterpret_cast<volatile u64*>(lb + i * 8) == S) {
                inl = 1;
                idx = i;
                break;
            }
        }
    }
    g_refill_inlist = inl;
    g_refill_idx = idx;
    g_refill_n = nn;
}
alignas(16) static u8 g_gate_stub[96];

// FUN_00e81ac0 sets each enable byte (catalog+0x25f98..) to (DAT_015ca5b8 < prop[9]) with
// DAT_015ca5b8 = FLT_EPSILON; with no envtable prop=0, so every pool tag is disabled and the
// Challenges section + mine notifications are gated off. Bypass: patch FUN_00e827a0 to return 1.
static void Gr2OpenGate(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    done = true;
    g_hook_catg = base + (0x1bea890 - 0x107BF0);
    // Plain code-patch: FUN_00e827a0 -> `mov eax,1; ret`. No stub, no per-call side effects
    // (routing this gate through a refill trampoline breaks the mining notifications).
    const u64 gate = base + (0xe827a0 - 0x107BF0);
    static const u8 patch[6] = {0xb8, 0x01, 0x00, 0x00, 0x00, 0xc3};  // mov eax,1 ; ret
    u8 orig[6];
    for (u32 i = 0; i < 6; ++i) {
        orig[i] = *reinterpret_cast<volatile u8*>(gate + i);
    }
    for (u32 i = 0; i < 6; ++i) {
        *reinterpret_cast<volatile u8*>(gate + i) = patch[i];
    }
    bool stuck = true;
    for (u32 i = 0; i < 6; ++i) {
        if (*reinterpret_cast<volatile u8*>(gate + i) != patch[i]) {
            stuck = false;
        }
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-OPENGATE v32: FUN_00e827a0 @{:#x} orig {:02x}{:02x}{:02x}{:02x}{:02x}{:02x} -> "
             "mov eax,1;ret (reverted to plain, restores mines)  [{}]",
             gate, orig[0], orig[1], orig[2], orig[3], orig[4], orig[5],
             stuck ? "PATCHED" : "WRITE-DROPPED");
}

// FUN_00fa1720 is a deferred UI-teardown callback (coroutine scheduler) that frees the challenger
// ghost-user-name std::string at *(param_1+0x120) on the default heap. The challenge-delivery churn
// recycles that chunk into a 40-byte getphotoghost avatar response first, so the deferred free
// double-frees and SceLibc aborts with "A heap error is detected". NOP only the 5-byte free CALL at
// 0xfa2317; the following `MOV [R14+0x120],0` still nulls the pointer, so state stays
// consistent and the tiny name string (<=40 bytes) leaks instead of aborting.
static void Gr2NeutralizeChGhostFree(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    done = true;
    // Plain code-patch: overwrite the SceLibc free-thunk CALL with five NOPs. No stub, no per-call
    // side effects. Verify the original is a rel32 CALL (leading E8) first, so a base-mapping error
    // logs a mismatch instead of corrupting an unrelated instruction.
    const u64 site = base + (0xfa2317 - 0x107BF0);
    u8 orig[5];
    for (u32 i = 0; i < 5; ++i) {
        orig[i] = *reinterpret_cast<volatile u8*>(site + i);
    }
    if (orig[0] != 0xe8) {
        LOG_ERROR(Lib_NpAuth,
                  "GR2-CHGHOSTFIX: FUN_00fa1720 @{:#x} orig {:02x}{:02x}{:02x}{:02x}{:02x} not a "
                  "rel32 CALL (E8) -- MISMATCH, not patched",
                  site, orig[0], orig[1], orig[2], orig[3], orig[4]);
        return;
    }
    for (u32 i = 0; i < 5; ++i) {
        *reinterpret_cast<volatile u8*>(site + i) = 0x90;  // nop
    }
    bool stuck = true;
    for (u32 i = 0; i < 5; ++i) {
        if (*reinterpret_cast<volatile u8*>(site + i) != 0x90) {
            stuck = false;
        }
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-CHGHOSTFIX: FUN_00fa1720 free CALL @{:#x} orig {:02x}{:02x}{:02x}{:02x}{:02x} -> "
             "5x nop (leaks challenger-name string, avoids double-free)  [{}]",
             site, orig[0], orig[1], orig[2], orig[3], orig[4],
             stuck ? "PATCHED" : "WRITE-DROPPED");
}

// GR2FORK FIX: the Announcements/Notice view singleton (DAT_01bf5d60) caches a NON-OWNING pointer to
// a served cap_response envelope in field +0x90. The net/mail layer frees that envelope first; then
// the view teardown FUN_00feb5a0 frees the same pointer a second time via the SceLibc free thunk at
// 0xfebba5, and SceLibc's heap checker aborts on Announcements open. NOP that duplicate free CALL;
// the following instruction (MOV [R14+0x90],0) still nulls the stale pointer, so no re-free occurs.
// The crash is card-agnostic (every served card shares the envelope), so this is required regardless
// of which card populated the view. Verify a leading E8 first so a base-mapping error on a variant
// eboot logs a mismatch instead of corrupting an unrelated instruction.
static void Gr2NeutralizeNoticeEnvelopeFree(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    done = true;
    const u64 site = base + (0xfebba5 - 0x107BF0);
    u8 orig[5];
    for (u32 i = 0; i < 5; ++i) {
        orig[i] = *reinterpret_cast<volatile u8*>(site + i);
    }
    if (orig[0] != 0xe8) {
        LOG_ERROR(Lib_NpAuth,
                  "GR2-NOTICEFREE: FUN_00feb5a0 @{:#x} orig {:02x}{:02x}{:02x}{:02x}{:02x} not a "
                  "rel32 CALL (E8) -- MISMATCH, not patched",
                  site, orig[0], orig[1], orig[2], orig[3], orig[4]);
        return;
    }
    for (u32 i = 0; i < 5; ++i) {
        *reinterpret_cast<volatile u8*>(site + i) = 0x90;  // nop
    }
    bool stuck = true;
    for (u32 i = 0; i < 5; ++i) {
        if (*reinterpret_cast<volatile u8*>(site + i) != 0x90) {
            stuck = false;
        }
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-NOTICEFREE: FUN_00feb5a0 notice-envelope free CALL @{:#x} orig "
             "{:02x}{:02x}{:02x}{:02x}{:02x} -> 5x nop (skips the view's duplicate free of the shared "
             "cap_response envelope; the +0x90 pointer is still nulled next instr)  [{}]",
             site, orig[0], orig[1], orig[2], orig[3], orig[4],
             stuck ? "PATCHED" : "WRITE-DROPPED");
}

// GR2FORK FIX: the in-race challenge rank-list renders the local player's OWN row from a UserService
// participant name buffer (participant+0x23) that the game fills only while processing a UserService
// Login event. The fork delivers one Login event at boot, so the participant the race creates later is
// never filled and the row draws uninitialized bytes (a glyph). FUN_00ed2b20's own-row path (index ==
// -1) funnels through a single tail-called copy strncpy(child+0x40, name, 0x10) at 0xed2ca2, reached
// only by the own row (the rival branch has its own copy + epilogue at 0xed2bf9/0xed2c02), arriving
// with the destination already in RDI and the size in EDX. Redirect that copy's source (RSI) to the
// config user name via a stub so the own row always renders the local name regardless of participant
// lifecycle. Rival rows (named from the server online-id) are untouched.
alignas(16) static u8 g_ownname_stub[64];
static char g_ownname_buf[17] = {0};
static void Gr2FixChallengeOwnName(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    done = true;
    const u64 site = base + (0xed2ca2 - 0x107BF0);  // own-row epilogue: pop rbx/r14/rbp; jmp strncpy
    u8 orig[5];
    for (u32 i = 0; i < 5; ++i) {
        orig[i] = *reinterpret_cast<volatile u8*>(site + i);
    }
    // expect 5b 41 5e 5d e9 = pop rbx; pop r14; pop rbp; jmp rel32 (FUN_014fd320)
    if (orig[0] != 0x5b || orig[1] != 0x41 || orig[2] != 0x5e || orig[3] != 0x5d || orig[4] != 0xe9) {
        LOG_ERROR(Lib_NpAuth,
                  "GR2-OWNNAME: FUN_00ed2b20 epilogue @{:#x} orig {:02x}{:02x}{:02x}{:02x}{:02x} "
                  "unexpected -- MISMATCH, not patched",
                  site, orig[0], orig[1], orig[2], orig[3], orig[4]);
        return;
    }
    std::memset(g_ownname_buf, 0, sizeof(g_ownname_buf));
    std::strncpy(g_ownname_buf, GR2Fork::Auth::EffectiveOnlineId().c_str(),
                 sizeof(g_ownname_buf) - 1);
    const u64 strncpy_rt = base + (0x14fd320 - 0x107BF0);  // FUN_014fd320 (the tail-called copy)
    u8* s = g_ownname_stub;
    u32 p = 0;
    auto e = [&](u8 b) { s[p++] = b; };
    auto e64 = [&](u64 v) { for (int i = 0; i < 8; ++i) e(static_cast<u8>(v >> (i * 8))); };
    // replay the displaced epilogue, override the copy source (RSI = the name), tail-call the copy.
    // RDI (child+0x40) and EDX (0x10) are already set by both own-row sub-paths and are preserved.
    e(0x5b);                                                     // pop rbx
    e(0x41); e(0x5e);                                            // pop r14
    e(0x5d);                                                     // pop rbp
    e(0x48); e(0xbe); e64(reinterpret_cast<u64>(g_ownname_buf)); // movabs rsi, &name
    e(0x48); e(0xb8); e64(strncpy_rt); e(0xff); e(0xe0);         // movabs rax, strncpy; jmp rax
    const u64 pg = reinterpret_cast<u64>(g_ownname_stub) & ~static_cast<u64>(0xfff);
    const int mp = Gr2MakeRWX(reinterpret_cast<void*>(pg), 0x2000);
    // patch the epilogue: movabs rax, stub; jmp rax (12 bytes; next function starts at 0xed2cb0)
    const u64 sa = reinterpret_cast<u64>(g_ownname_stub);
    u8 patch[12] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0};
    for (int i = 0; i < 8; ++i) {
        patch[2 + i] = static_cast<u8>(sa >> (i * 8));
    }
    for (u32 i = 0; i < 12; ++i) {
        *reinterpret_cast<volatile u8*>(site + i) = patch[i];
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-OWNNAME: FUN_00ed2b20 own-row copy @{:#x} -> stub@{:#x} ({}B) mprotect={} name='{}'",
             site, sa, p, mp, g_ownname_buf);
}

// GR2FORK FIX: suppress the overworld challenge puppet. Challenges are reached from a marker access
// point and have no overworld ghost (the only overworld figures are the gold treasure and blue photo
// Kats); the ghost path otherwise spawns a replay-playing puppet at the challenge marker. The puppet
// spawns in Lua only when Ugc:findShowGhost (FUN_00e995a0) returns a valid handle: the native zeroes
// R15, then writes the real handle target with MOV R15,RAX (49 89 c7) at 0xe99759. NOP that write so
// R15 stays 0, the handle stays invalid, and no puppet spawns. The Challenges row (FUN_00e828f0,
// separate vector) and the in-mission race ghost (FUN_00edc730, separate store) do not use this
// native; the +0x14 visibility gate is untouched, so the row and race are unaffected.
static void Gr2HideChallengePuppet(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    done = true;
    const u64 site = base + (0xe99759 - 0x107BF0);  // MOV R15,RAX in Ugc:findShowGhost (FUN_00e995a0)
    u8 orig[3];
    for (u32 i = 0; i < 3; ++i) {
        orig[i] = *reinterpret_cast<volatile u8*>(site + i);
    }
    if (orig[0] != 0x49 || orig[1] != 0x89 || orig[2] != 0xc7) {
        LOG_ERROR(Lib_NpAuth,
                  "GR2-CHPUPPET: findShowGhost @{:#x} orig {:02x}{:02x}{:02x} not MOV R15,RAX "
                  "(49 89 c7) -- MISMATCH, not patched",
                  site, orig[0], orig[1], orig[2]);
        return;
    }
    for (u32 i = 0; i < 3; ++i) {
        *reinterpret_cast<volatile u8*>(site + i) = 0x90;  // nop
    }
    bool stuck = true;
    for (u32 i = 0; i < 3; ++i) {
        if (*reinterpret_cast<volatile u8*>(site + i) != 0x90) {
            stuck = false;
        }
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-CHPUPPET: findShowGhost handle write @{:#x} orig {:02x}{:02x}{:02x} -> 3x nop "
             "(overworld challenge puppet suppressed; row + race unaffected)  [{}]",
             site, orig[0], orig[1], orig[2], stuck ? "PATCHED" : "WRITE-DROPPED");
}

// GR2FORK FIX: global guest-heap double-free guard. Deferred UI-teardown callbacks fired on the
// coroutine scheduler free buffers that challenge-delivery churn already recycled and freed, so the
// same chunk is freed twice and SceLibc aborts with "A heap error is detected" (ud2 in libc.prx).
// The offending frees are a class of sites, not one, so a per-site NOP does not scale. Instead the
// single SceLibc free thunk is redirected: FUN_014fd2a0 tail-calls through the resolved import slot
// _DAT_0180c8a0 as sceLibcMspaceFree(mspace, ptr) (RDI=mspace, RSI=ptr), so overwriting that slot
// with Gr2FreeGuard routes every mspace free through the wrapper. Both game heaps (default
// DAT_019e7560, net-scratch DAT_019e7568) pass their handle as arg0, which the wrapper forwards
// unchanged.
//
// Detection is a stateless read of the SceLibc guard-heap metadata - no fork-side allocation set,
// no lock. A live chunk carries the 8-byte cookie 0x0000000258585878 at ptr-0x10 followed by its
// size at ptr-0x8; a freed chunk has that metadata and its user region overwritten with the
// 0xafafafaf poison. The wrapper skips the free only on a POSITIVE full-8-byte poison match at the
// header cookie slot (ptr-0x10) or the first user word (ptr). This is the safe direction: a wrong
// offset assumption makes the guard inert (it never matches) rather than dropping live frees, a
// live object practically never holds eight 0xaf bytes, and both reads stay within the metadata the
// real free dereferences anyway. Skipping a free of an already-free chunk leaks it (tiny, <=40-byte
// UI objects) instead of aborting - the same trade the per-site NOP makes.
using Gr2MspaceFreeFn = PS4_SYSV_ABI void (*)(void* mspace, void* ptr);
static std::atomic<Gr2MspaceFreeFn> g_gr2_orig_free{nullptr};
static std::atomic<u64> g_gr2_df_skipped{0};
static constexpr u64 kGr2FreePoison = 0xafafafafafafafafull;

static PS4_SYSV_ABI void Gr2FreeGuard(void* mspace, void* ptr) {
    const Gr2MspaceFreeFn orig = g_gr2_orig_free.load(std::memory_order_acquire);
    if (orig == nullptr) {
        return;  // never reached once installed; never re-enter the thunk that was replaced
    }
    if (ptr == nullptr) {
        orig(mspace, nullptr);  // libc free(NULL) is a no-op; delegate for identical semantics
        return;
    }
    // Already-free test: full 8-byte poison at the header cookie slot or the first user word.
    const u64 hdr = *reinterpret_cast<volatile u64*>(reinterpret_cast<u8*>(ptr) - 0x10);
    const u64 user = *reinterpret_cast<volatile u64*>(ptr);
    if (hdr == kGr2FreePoison || user == kGr2FreePoison) {
        const u64 n = g_gr2_df_skipped.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n == 1 || (n & 0xff) == 0) {
            LOG_ERROR(Lib_NpAuth,
                      "GR2-FREEGUARD: skipped double-free of {:#x} (chunk poisoned) -- {} skipped",
                      reinterpret_cast<u64>(ptr), n);
        }
        return;  // chunk is already free -> dropping this free avoids the SceLibc abort
    }
    orig(mspace, ptr);
}

// Install the guard by swapping the resolved free-thunk import slot _DAT_0180c8a0. One-shot, but
// Gr2ArcProbe runs every video flip, so an unresolved import defers and retries on the next flip
// rather than latching. GR2-serial-gated at the call site (probe_is_gr2), so other titles are
// untouched.
static void Gr2InstallFreeGuard(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    const u64 slot = base + (0x180c8a0 - 0x107BF0);
    const u64 orig = *reinterpret_cast<volatile u64*>(slot);
    if (orig < 0x10000 || orig >= 0x800000000000ull) {
        static bool deferred = false;
        if (!deferred) {
            deferred = true;
            LOG_INFO(Lib_NpAuth,
                     "GR2-FREEGUARD-INSTALL: _DAT_0180c8a0 @{:#x} = {:#x} unresolved -- DEFER",
                     slot, orig);
        }
        return;  // do not latch; retry on a later flip once the import resolves
    }
    done = true;
    g_gr2_orig_free.store(reinterpret_cast<Gr2MspaceFreeFn>(orig), std::memory_order_release);
    // The import slot lives in the eboot data segment; make its page writable before the swap.
    const u64 pg = slot & ~static_cast<u64>(0xfff);
    const int mp = Gr2MakeRWX(reinterpret_cast<void*>(pg), 0x2000);
    *reinterpret_cast<volatile u64*>(slot) = reinterpret_cast<u64>(&Gr2FreeGuard);
    const u64 readback = *reinterpret_cast<volatile u64*>(slot);
    const bool ok = (readback == reinterpret_cast<u64>(&Gr2FreeGuard));
    LOG_INFO(Lib_NpAuth,
             "GR2-FREEGUARD-INSTALL: _DAT_0180c8a0 @{:#x} orig={:#x} -> &Gr2FreeGuard={:#x} "
             "mprotect={} [{}]",
             slot, orig, reinterpret_cast<u64>(&Gr2FreeGuard), mp, ok ? "PATCHED" : "MISMATCH");
}

// Hook FUN_00fdf150 (announcement-screen list builder) at entry so the cm12 slot is re-asserted
// right before it enumerates and bins. It is vtable-dispatched, so its first 13 bytes (6 pushes +
// mov rbp,rsp, non-RIP-relative) jump to a stub that calls Gr2HookFill, replays them, jumps back.
static volatile u64 g_last_screen = 0;  // FUN_00fdf150 param_1 (announcement screen), captured by hook
// The stub preserves RDI (= FUN_00fdf150 param_1 = the screen) and passes it as arg0, so the hook
// gets the live screen object; its per-category bucket vectors (screen+0xb8+idx*0x20, {+8 begin,
// +0x10 end}) can be read post-build to check whether the slot lands in bucket 2 (Challenges).
extern "C" __attribute__((no_stack_protector)) void Gr2HookFill(u64 screen) {
    if (g_hook_catg == 0) {
        return;
    }
    const u64 cat = *reinterpret_cast<volatile u64*>(g_hook_catg);
    if (cat < 0x10000) {
        return;
    }
    const u64 S = cat + 8 + 11 * 0xcd0;  // cm12 slot 0
    if (*reinterpret_cast<volatile u16*>(S + 0x12) > 0x100) {
        return;  // doesn't look like the pool
    }
    *reinterpret_cast<volatile u16*>(S + 0x10) = 1;            // id (not 0xffff)
    *reinterpret_cast<volatile u16*>(S + 0x12) = 0;            // category Challenge
    *reinterpret_cast<volatile u32*>(S + 0x14) = 1;            // state active
    *reinterpret_cast<volatile u8*>(S + 0x16) &= 0xfe;         // clear hide bit
    *reinterpret_cast<volatile u32*>(S + 0xb4) = 0x76f0fed2u;  // cm12 FNV
    *reinterpret_cast<volatile u8*>(S + 0x18) = 0xff;          // tag (gate open anyway)
    // Always write the inline challenger online-id: if the game converts +0xa0 to a heap string,
    // reading the pointer bytes as chars yields a garbage onlineId and an empty row. The hook runs
    // right before the build, so filling the full 0x10-byte name buffer (name + null pad) is safe.
    static const char nm[] = "junminlee2004";
    for (u32 i = 0; i < 0x10; ++i) {
        *reinterpret_cast<volatile u8*>(S + 0xa0 + i) = (i < sizeof(nm)) ? static_cast<u8>(nm[i]) : 0;
    }
    // Only CAPTURE the screen ptr; its +0xb8 buckets are uninitialized at entry (poison) --
    // they get built DURING FUN_00fdf150. Gr2WatchBuckets reads them POST-build.
    if (screen >= 0x10000 && screen < 0x800000000000ull) {
        g_last_screen = screen;
    }
    // At the build instant, check the slot's state and whether its pointer is in the master list
    // catalog+0x25e60 (what FUN_00e828f0 iterates): st=1/cat=0 and present must enroll; absent
    // from the master list at build time explains why it drops out.
    static u32 ck = 0;
    if ((ck++ % 4) == 0) {
        const u64 lb = *reinterpret_cast<volatile u64*>(cat + 0x25e60);
        const u64 le = *reinterpret_cast<volatile u64*>(cat + 0x25e68);
        bool inlist = false;
        long idx = -1;
        if (lb >= 0x10000 && le > lb && (le - lb) <= 0x10000) {
            const u32 n = static_cast<u32>((le - lb) >> 3);
            for (u32 i = 0; i < n; ++i) {
                if (*reinterpret_cast<volatile u64*>(lb + i * 8) == S) {
                    inlist = true;
                    idx = static_cast<long>(i);
                    break;
                }
            }
        }
        // Spin-read +0x12/+0x14 to detect a CONCURRENT writer (the async challenge processor
        // on another thread) demoting the slot mid-build.
        u32 hidechg = 0, st0or3 = 0;
        for (u32 z = 0; z < 200000; ++z) {
            if (*reinterpret_cast<volatile u8*>(S + 0x16) & 1) {
                ++hidechg;  // +0x16 bit0 "hide" flag set (FUN_00e828f0 skips it)
            }
            const u16 st = *reinterpret_cast<volatile u16*>(S + 0x14);
            if (st == 0 || st == 3) {
                ++st0or3;  // st that would fail FUN_00e828f0's st!=0&&!=3
            }
        }
        LOG_INFO(Lib_NpAuth,
                 "GR2-HOOKCHK: build entry slot st={} cat={} b16={:#x} +0x2a4={:#x} | in list={} "
                 "idx={} n={} | spin200k: hide-bit {}x, st0/3 {}x",
                 *reinterpret_cast<volatile u16*>(S + 0x14), *reinterpret_cast<volatile u16*>(S + 0x12),
                 *reinterpret_cast<volatile u8*>(S + 0x16), *reinterpret_cast<volatile u8*>(S + 0x2a4),
                 inlist, idx, (le > lb) ? static_cast<long>((le - lb) >> 3) : -1, hidechg, st0or3);
    }
}
alignas(16) static u8 g_fdf_stub[160];
static void Gr2InstallFdf150Hook(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    done = true;
    g_hook_catg = base + (0x1bea890 - 0x107BF0);
    const u64 fn = base + (0xfdf150 - 0x107BF0);
    u8 disp[13];
    for (u32 i = 0; i < 13; ++i) {
        disp[i] = *reinterpret_cast<volatile u8*>(fn + i);
    }
    u8* s = g_fdf_stub;
    u32 p = 0;
    auto e = [&](u8 b) { s[p++] = b; };
    auto e64 = [&](u64 v) { for (int i = 0; i < 8; ++i) e(static_cast<u8>(v >> (i * 8))); };
    // push rax,rcx,rdx,rsi,rdi,r8,r9,r10,r11  (9 pushes -> 16-aligned for the SysV call)
    e(0x50); e(0x51); e(0x52); e(0x56); e(0x57);
    e(0x41); e(0x50); e(0x41); e(0x51); e(0x41); e(0x52); e(0x41); e(0x53);
    e(0x48); e(0xb8); e64(reinterpret_cast<u64>(&Gr2HookFill)); e(0xff); e(0xd0);  // mov rax,fn; call rax
    // pop r11,r10,r9,r8,rdi,rsi,rdx,rcx,rax
    e(0x41); e(0x5b); e(0x41); e(0x5a); e(0x41); e(0x59); e(0x41); e(0x58);
    e(0x5f); e(0x5e); e(0x5a); e(0x59); e(0x58);
    for (u32 i = 0; i < 13; ++i) {
        e(disp[i]);  // replay the displaced prologue
    }
    e(0x48); e(0xb8); e64(fn + 13); e(0xff); e(0xe0);  // mov rax,fn+13; jmp rax
    const u64 pg = reinterpret_cast<u64>(g_fdf_stub) & ~static_cast<u64>(0xfff);
    const int mp = Gr2MakeRWX(reinterpret_cast<void*>(pg), 0x2000);
    // patch entry: mov rax, stub ; jmp rax ; nop   (13 bytes)
    const u64 sa = reinterpret_cast<u64>(g_fdf_stub);
    u8 patch[13] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0, 0x90};
    for (int i = 0; i < 8; ++i) {
        patch[2 + i] = static_cast<u8>(sa >> (i * 8));
    }
    for (u32 i = 0; i < 13; ++i) {
        *reinterpret_cast<volatile u8*>(fn + i) = patch[i];
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-HOOK v20: FUN_00fdf150@{:#x} -> stub@{:#x} ({}B) mprotect={} catg={:#x} (re-assert "
             "slot before bin)",
             fn, sa, p, mp, g_hook_catg);
}

// Refill hook at FUN_00e828f0 entry: the enroll precheck reads st before the gate (a gate refill
// is too late) and FUN_00fdf150's vtable refresh resets st after entry-time fills, so only this
// spot sticks. Displaces the 16-byte prologue (+16 is a RIP-relative CALL), preserving rsi/rdi.
alignas(16) static u8 g_e828_stub[160];
static void Gr2InstallE828f0Hook(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    done = true;
    const u64 fn = base + (0xe828f0 - 0x107BF0);
    u8 disp[16];
    for (u32 i = 0; i < 16; ++i) {
        disp[i] = *reinterpret_cast<volatile u8*>(fn + i);
    }
    u8* s = g_e828_stub;
    u32 p = 0;
    auto e = [&](u8 b) { s[p++] = b; };
    auto e64 = [&](u64 v) { for (int i = 0; i < 8; ++i) e(static_cast<u8>(v >> (i * 8))); };
    // push rax,rcx,rdx,rsi,rdi,r8,r9,r10,r11  (9 pushes -> 16-aligned; preserves the rsi/rdi args)
    e(0x50); e(0x51); e(0x52); e(0x56); e(0x57);
    e(0x41); e(0x50); e(0x41); e(0x51); e(0x41); e(0x52); e(0x41); e(0x53);
    e(0x48); e(0xb8); e64(reinterpret_cast<u64>(&Gr2RefillSlot)); e(0xff); e(0xd0);  // mov rax,fn;call
    // pop r11,r10,r9,r8,rdi,rsi,rdx,rcx,rax  (restores rsi/rdi to the original args)
    e(0x41); e(0x5b); e(0x41); e(0x5a); e(0x41); e(0x59); e(0x41); e(0x58);
    e(0x5f); e(0x5e); e(0x5a); e(0x59); e(0x58);
    for (u32 i = 0; i < 16; ++i) {
        e(disp[i]);  // replay the displaced prologue (sets up the frame + r14=rsi/r15=rdi)
    }
    e(0x48); e(0xb8); e64(fn + 16); e(0xff); e(0xe0);  // mov rax,fn+16; jmp rax (-> the CALL at +16)
    const u64 pg = reinterpret_cast<u64>(g_e828_stub) & ~static_cast<u64>(0xfff);
    const int mp = Gr2MakeRWX(reinterpret_cast<void*>(pg), 0x2000);
    // patch entry: mov rax, stub ; jmp rax ; nop nop nop nop   (16 bytes, clean overwrite)
    const u64 sa = reinterpret_cast<u64>(g_e828_stub);
    u8 patch[16] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0, 0x90, 0x90, 0x90, 0x90};
    for (int i = 0; i < 8; ++i) {
        patch[2 + i] = static_cast<u8>(sa >> (i * 8));
    }
    for (u32 i = 0; i < 16; ++i) {
        *reinterpret_cast<volatile u8*>(fn + i) = patch[i];
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-HOOK v31: FUN_00e828f0@{:#x} -> stub@{:#x} ({}B) mprotect={} (refill at enumerator "
             "entry, after the vtable reset, before the enroll loop)",
             fn, sa, p, mp);
}

// Read-only capture hook at FUN_00e8cc30 entry (the receive/populator consuming async records at
// mgr+0x26428 / mgr+0x26448). On any state change it dumps the two async slots and any pending
// record (type@+0, FNV@+0x28, cnt@+0x30, online-id@*(rec+8)), pinning the receive trigger.
static volatile u64 g_cap_sig = 0;
static void Gr2DumpRec(const char* tag, u64 rec) {
    if (rec < 0x10000 || rec >= 0x800000000000ull) {
        return;
    }
    static const char H[] = "0123456789abcdef";
    char hex[0x40 * 3 + 1];
    u32 p = 0;
    for (u32 i = 0; i < 0x40; ++i) {
        const u8 v = *reinterpret_cast<volatile u8*>(rec + i);
        hex[p++] = H[v >> 4];
        hex[p++] = H[v & 0xf];
        hex[p++] = ' ';
    }
    hex[p] = 0;
    const u32 type = *reinterpret_cast<volatile u32*>(rec + 0);
    const u32 fnv = *reinterpret_cast<volatile u32*>(rec + 0x28);
    const u32 cnt = *reinterpret_cast<volatile u32*>(rec + 0x30);
    const u64 oidp = *reinterpret_cast<volatile u64*>(rec + 8);
    char oid[0x14] = {0};
    if (oidp >= 0x10000 && oidp < 0x800000000000ull) {
        for (u32 i = 0; i < 0x13; ++i) {
            const u8 c = *reinterpret_cast<volatile u8*>(oidp + i);
            oid[i] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
        }
    }
    LOG_INFO(Lib_NpAuth, "GR2-RECDUMP[{}] rec@{:#x} type={} FNV={:#x} cnt={} oid='{}' | {}", tag, rec,
             type, fnv, cnt, oid, hex);
}
extern "C" __attribute__((no_stack_protector)) void Gr2CaptureRecord(u64 mgr) {
    if (mgr < 0x10000 || mgr >= 0x800000000000ull) {
        return;
    }
    const u32 s1 = *reinterpret_cast<volatile u32*>(mgr + 0x26420);
    const u64 r1 = *reinterpret_cast<volatile u64*>(mgr + 0x26428);
    const u32 s2 = *reinterpret_cast<volatile u32*>(mgr + 0x26440);
    const u64 r2 = *reinterpret_cast<volatile u64*>(mgr + 0x26448);
    const u64 sig = mgr ^ r1 ^ (r2 << 1) ^ (static_cast<u64>(s1) << 2) ^ (static_cast<u64>(s2) << 3);
    if (sig == g_cap_sig) {
        return;  // no state change -> skip (dedup; only logs on transitions, not every frame)
    }
    g_cap_sig = sig;
    LOG_INFO(Lib_NpAuth, "GR2-CAP: mgr@{:#x} slot1[st={:#x} rec={:#x}] slot2[st={:#x} rec={:#x}]", mgr,
             s1, r1, s2, r2);
    // One-shot dump of the catalog challenge-name table (DAT_01bea890+0x26460..+0x26468, stride
    // 0x30: +0 FNV, +0x10 name SSO/heap, +0x28 len). A getmaillist entry with mail-id==0 is
    // FNV-hashed by NAME and resolved here, so these are the exact subjects the server must send.
    static bool g_tbl_done = false;
    if (!g_tbl_done && g_hook_catg != 0) {
        const u64 cat = *reinterpret_cast<volatile u64*>(g_hook_catg);
        if (cat >= 0x10000 && cat < 0x800000000000ull) {
            const u64 tb = *reinterpret_cast<volatile u64*>(cat + 0x26460);
            const u64 te = *reinterpret_cast<volatile u64*>(cat + 0x26468);
            // Always log the bounds once (so an empty/odd table is visible too), then iterate.
            LOG_INFO(Lib_NpAuth, "GR2-NAMETBL: catalog={:#x} +0x26460 begin={:#x} end={:#x} span={:#x}",
                     cat, tb, te, (te > tb) ? (te - tb) : 0);
            if (tb >= 0x10000 && te > tb && (te - tb) <= 0x100000) {
                g_tbl_done = true;
                const u32 cnt = static_cast<u32>((te - tb) / 0x30);
                LOG_INFO(Lib_NpAuth, "GR2-NAMETBL: catalog+0x26460 begin={:#x} end={:#x} count={}", tb,
                         te, cnt);
                for (u32 i = 0; i < cnt && i < 60; ++i) {
                    const u64 ent = tb + i * 0x30;
                    const u32 efnv = *reinterpret_cast<volatile u32*>(ent + 0);
                    const u64 elen = *reinterpret_cast<volatile u64*>(ent + 0x28);
                    u64 sp = ent + 0x10;
                    if (elen >= 0x10) {
                        sp = *reinterpret_cast<volatile u64*>(ent + 0x10);
                    }
                    char nm[0x28] = {0};
                    if (sp >= 0x10000 && sp < 0x800000000000ull) {
                        for (u32 j = 0; j < 0x27; ++j) {
                            const u8 c = *reinterpret_cast<volatile u8*>(sp + j);
                            if (c == 0) {
                                break;
                            }
                            nm[j] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                        }
                    }
                    LOG_INFO(Lib_NpAuth, "GR2-NAMETBL[{}] FNV={:#x} len={} name='{}'", i, efnv, elen, nm);
                }
            }
        }
    }
    if (r1 >= 0x10000) {
        Gr2DumpRec("1", r1);
    }
    if (r2 >= 0x10000) {
        Gr2DumpRec("2", r2);
    }
    // The category selector is *(int*)(entry+0x10) of the announcement-entry list at mgr+0x263b8
    // (consumed by FUN_00e8cc30); name at entry+0x20 (SSO, heap if entry+0x38>0xf). Selector==0
    // triggers the FNV-by-name mission lookup. Walks up to 4 entries and dumps them.
    const u64 head = *reinterpret_cast<volatile u64*>(mgr + 0x263b8);
    if (head >= 0x10000 && head < 0x800000000000ull) {
        u64 e = *reinterpret_cast<volatile u64*>(head);
        for (u32 k = 0; k < 4 && e >= 0x10000 && e < 0x800000000000ull && e != head; ++k) {
            const u32 etype = *reinterpret_cast<volatile u32*>(e + 0x10);
            const u64 elen = *reinterpret_cast<volatile u64*>(e + 0x38);
            u64 np = e + 0x20;
            if (elen > 0xf) {
                np = *reinterpret_cast<volatile u64*>(e + 0x20);
            }
            char nm[0x24] = {0};
            if (np >= 0x10000 && np < 0x800000000000ull) {
                for (u32 i = 0; i < 0x23; ++i) {
                    const u8 c = *reinterpret_cast<volatile u8*>(np + i);
                    if (c == 0) {
                        break;
                    }
                    nm[i] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                }
            }
            // RAW entry bytes 0x00..0x40 -- the real 0..3 category field is somewhere here,
            // not necessarily +0x10 (which is the id).
            static const char H2[] = "0123456789abcdef";
            char eh[0x40 * 3 + 1];
            u32 q = 0;
            for (u32 i = 0; i < 0x40; ++i) {
                const u8 v = *reinterpret_cast<volatile u8*>(e + i);
                eh[q++] = H2[v >> 4];
                eh[q++] = H2[v & 0xf];
                eh[q++] = ' ';
            }
            eh[q] = 0;
            LOG_INFO(Lib_NpAuth, "GR2-ENTRY[{}] @{:#x} +0x10={} namelen={} name='{}' | raw {}", k, e,
                     etype, elen, nm, eh);
            e = *reinterpret_cast<volatile u64*>(e);  // next (linked list)
        }
    }
    // Walk the manager's CREATED-SLOT list (mgr+0x263f8 head .. +0x26400 tail) and dump each
    // slot's category (+0x12), state (+0x14), name (+0xa0), FNV (+0xc8). This shows whether the receive
    // actually builds a CHALLENGE slot (cat 1) and what it contains -- the bridge to the Challenges tab.
    const u64 sh = *reinterpret_cast<volatile u64*>(mgr + 0x263f8);
    const u64 stl = *reinterpret_cast<volatile u64*>(mgr + 0x26400);
    if (sh >= 0x10000 && sh < 0x800000000000ull && stl > sh && (stl - sh) < 0x800) {
        const u32 ns = static_cast<u32>((stl - sh) >> 3);
        for (u32 i = 0; i < ns && i < 6; ++i) {
            const u64 sl = *reinterpret_cast<volatile u64*>(sh + i * 8);
            if (sl < 0x10000 || sl >= 0x800000000000ull) {
                continue;
            }
            const u16 scat = *reinterpret_cast<volatile u16*>(sl + 0x12);
            const u16 sst = *reinterpret_cast<volatile u16*>(sl + 0x14);
            const u32 sfnv = *reinterpret_cast<volatile u32*>(sl + 0xc8);
            char snm[0x14] = {0};
            for (u32 j = 0; j < 0x13; ++j) {
                const u8 c = *reinterpret_cast<volatile u8*>(sl + 0xa0 + j);
                snm[j] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
            }
            LOG_INFO(Lib_NpAuth, "GR2-MGRSLOT[{}] @{:#x} cat={} st={} FNV={:#x} name='{}'", i, sl, scat,
                     sst, sfnv, snm);
        }
    }
}
alignas(16) static u8 g_e8cc_stub[160];
static void Gr2InstallE8cc30Hook(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    done = true;
    const u64 fn = base + (0xe8cc30 - 0x107BF0);
    u8 disp[20];
    for (u32 i = 0; i < 20; ++i) {
        disp[i] = *reinterpret_cast<volatile u8*>(fn + i);
    }
    u8* s = g_e8cc_stub;
    u32 p = 0;
    auto e = [&](u8 b) { s[p++] = b; };
    auto e64 = [&](u64 v) { for (int i = 0; i < 8; ++i) e(static_cast<u8>(v >> (i * 8))); };
    // push rax,rcx,rdx,rsi,rdi,r8,r9,r10,r11  (9 pushes -> 16-aligned; preserves RDI=manager arg)
    e(0x50); e(0x51); e(0x52); e(0x56); e(0x57);
    e(0x41); e(0x50); e(0x41); e(0x51); e(0x41); e(0x52); e(0x41); e(0x53);
    e(0x48); e(0xb8); e64(reinterpret_cast<u64>(&Gr2CaptureRecord)); e(0xff); e(0xd0);  // mov rax;call
    // pop r11,r10,r9,r8,rdi,rsi,rdx,rcx,rax
    e(0x41); e(0x5b); e(0x41); e(0x5a); e(0x41); e(0x59); e(0x41); e(0x58);
    e(0x5f); e(0x5e); e(0x5a); e(0x59); e(0x58);
    for (u32 i = 0; i < 20; ++i) {
        e(disp[i]);  // replay displaced prologue (push block + sub rsp,0x5c8); RDI still = manager
    }
    e(0x48); e(0xb8); e64(fn + 20); e(0xff); e(0xe0);  // mov rax,fn+20; jmp rax (-> the RIP-rel MOV)
    const u64 pg = reinterpret_cast<u64>(g_e8cc_stub) & ~static_cast<u64>(0xfff);
    const int mp = Gr2MakeRWX(reinterpret_cast<void*>(pg), 0x2000);
    const u64 sa = reinterpret_cast<u64>(g_e8cc_stub);
    u8 patch[20] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0,
                    0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
    for (int i = 0; i < 8; ++i) {
        patch[2 + i] = static_cast<u8>(sa >> (i * 8));
    }
    for (u32 i = 0; i < 20; ++i) {
        *reinterpret_cast<volatile u8*>(fn + i) = patch[i];
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-HOOK v34: FUN_00e8cc30@{:#x} -> stub@{:#x} ({}B) mprotect={} (READ-ONLY receive "
             "record capture)",
             fn, sa, p, mp);
}

// Diagnostic: CALL FUN_00e828f0(catalog, tempvec) directly via ExecuteGuest and read its
// actual output to check whether the cm12 slot enrolls. Vector is {+8=begin,+0x10=end,
// +0x18=cap}; an empty {0,0,0} is safe (first push reallocs via the game allocator). Runs once.
static void Gr2CallEnumerate(u64 base) {
    static u32 tk = 0;
    static bool done = false;
    if (done) {
        return;
    }
    if (++tk != 800) {
        return;  // fire EXACTLY once at tick 800 (gate already patched by then)
    }
    done = true;
    const u64 cat = Gr2ReadGlobal<u64>(0x1bea890);
    LOG_INFO(Lib_NpAuth, "GR2-ENUM: fired @tick800 cat={:#x}", cat);
    if (cat < 0x10000) {
        LOG_INFO(Lib_NpAuth, "GR2-ENUM: catalog not ready");
        return;
    }
    // The vector OBJECT lives in fork memory (only its data buffer needs the game allocator, which
    // the push allocates). Layout for FUN_00e828f0/FUN_014e6be0: +8 begin, +0x10 end, +0x18 cap.
    static volatile u64 vecobj[4];
    vecobj[0] = 0;
    vecobj[1] = 0;
    vecobj[2] = 0;
    vecobj[3] = 0;
    const u64 vec = reinterpret_cast<u64>(const_cast<u64*>(&vecobj[0]));
    using EnumFn = PS4_SYSV_ABI void (*)(u64, u64);
    const u64 fn = base + (0xe828f0 - 0x107BF0);
    LOG_INFO(Lib_NpAuth, "GR2-ENUM: calling FUN_00e828f0 @{:#x} cat={:#x} vec={:#x} ...", fn, cat, vec);
    Core::ExecuteGuest(reinterpret_cast<EnumFn>(fn), cat, vec);
    LOG_INFO(Lib_NpAuth, "GR2-ENUM: returned from FUN_00e828f0");
    const u64 vec_dummy = vec;  // (vb/ve below read from vecobj)
    (void)vec_dummy;
    const u64 vb = *reinterpret_cast<volatile u64*>(vec + 8);
    const u64 ve = *reinterpret_cast<volatile u64*>(vec + 0x10);
    if (vb < 0x10000 || ve < vb || (ve - vb) > 0x20000) {
        LOG_INFO(Lib_NpAuth, "GR2-ENUM: FUN_00e828f0 vec begin={:#x} end={:#x} (empty/bad)", vb, ve);
        return;
    }
    const u32 n = static_cast<u32>((ve - vb) >> 3);
    const u64 myslot = cat + 8 + 11 * 0xcd0;
    const u64 dat = base + (0x15ba4d0 - 0x107BF0);  // category -> bucket table
    bool found = false;
    int myslot_bucket = -2;
    u32 c0 = 0, c8 = 0, cother = 0;
    u32 bucket[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (u32 i = 0; i < n; ++i) {
        const u64 it = *reinterpret_cast<volatile u64*>(vb + i * 8);
        if (it < 0x10000) {
            continue;
        }
        const u16 c = *reinterpret_cast<volatile u16*>(it + 0x12);
        if (it == myslot) {
            found = true;
        }
        if (c == 0) {
            ++c0;
        } else if (c == 8) {
            ++c8;
        } else {
            ++cother;
        }
        // replicate FUN_00fdf150 binning: include if c != 0xffff && (u64)(c - 0xb) >= 3
        if (c != 0xffff && (static_cast<u64>(c) - 0xb) >= 3) {
            const s32 b = *reinterpret_cast<volatile s32*>(dat + static_cast<u64>(c) * 4);
            if (b >= 0 && b < 8) {
                ++bucket[b];
            }
            if (it == myslot) {
                myslot_bucket = b;
            }
        } else if (it == myslot) {
            myslot_bucket = -1;  // excluded by the binning condition
        }
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-ENUM: output n={} | MY SLOT ENROLLED={} -> bucket {} (2=Challenges) | cat0={} "
             "cat8={} other={}",
             n, found, myslot_bucket, c0, c8, cother);
    LOG_INFO(Lib_NpAuth,
             "GR2-ENUM: bucket sizes [0=All]={} [1=News]={} [2=Challenges]={} [3=TH]={} [4=Mining]={} "
             "[5=Photo]={} [6=Tut]={}",
             bucket[0], bucket[1], bucket[2], bucket[3], bucket[4], bucket[5], bucket[6]);
}

// Read-only bucket watch (no guest call, safe while the menu is open): replicates FUN_00e828f0's
// filters (st!=0&&!=3, (b16&1)==0, cat!=7; gate patched to 1) and binning (DAT_015ba4d0[cat])
// over the live master list every ~600 ticks, showing when the cm12 slot stops enrolling.
static void Gr2WatchBuckets(u64 base) {
#if defined(_WIN32)
    // Linux-only telemetry: per-frame name-table and bucket walks chase heap pointers behind
    // range filters only; see Gr2DumpMasterList for why that is fatal on Windows.
    (void)base;
    return;
#endif
    static u32 tk = 0;
    ++tk;
    // Sample the challenge-name table (catalog+0x26460) EVERY frame (not tied to receive
    // activity or the 600-tick throttle) -- it may only populate while the Challenges/mission
    // screen is open. Log + dump whenever its span CHANGES to catch the moment it fills.
    {
        const u64 cat = Gr2ReadGlobal<u64>(0x1bea890);
        const u64 ntb = (cat >= 0x10000) ? *reinterpret_cast<volatile u64*>(cat + 0x26460) : 0;
        const u64 nte = (cat >= 0x10000) ? *reinterpret_cast<volatile u64*>(cat + 0x26468) : 0;
        const u64 nspan = (nte > ntb) ? (nte - ntb) : 0;
        static u64 g_last_nspan = 0xffffffffffffffffull;
        if (nspan != g_last_nspan) {
            g_last_nspan = nspan;
            const u32 ncnt = (ntb >= 0x10000 && nspan && nspan <= 0x100000)
                                 ? static_cast<u32>(nspan / 0x30)
                                 : 0;
            LOG_INFO(Lib_NpAuth, "GR2-NAMETBL: begin={:#x} end={:#x} span={:#x} count={}", ntb, nte,
                     nspan, ncnt);
            for (u32 i = 0; i < ncnt && i < 60; ++i) {
                const u64 ent = ntb + i * 0x30;
                const u32 efnv = *reinterpret_cast<volatile u32*>(ent + 0);
                const u64 elen = *reinterpret_cast<volatile u64*>(ent + 0x28);
                u64 sp = (elen >= 0x10) ? *reinterpret_cast<volatile u64*>(ent + 0x10) : (ent + 0x10);
                char nm[0x28] = {0};
                if (sp >= 0x10000 && sp < 0x800000000000ull) {
                    for (u32 j = 0; j < 0x27; ++j) {
                        const u8 c = *reinterpret_cast<volatile u8*>(sp + j);
                        if (c == 0) {
                            break;
                        }
                        nm[j] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                    }
                }
                LOG_INFO(Lib_NpAuth, "GR2-NAMETBL[{}] FNV={:#x} len={} name='{}'", i, efnv, elen, nm);
            }
        }
    }
    // The bucket replication below stays throttled to every 600 ticks (it's verbose).
    if ((tk % 600) != 0) {
        return;
    }
    const u64 cat = Gr2ReadGlobal<u64>(0x1bea890);
    if (cat < 0x10000) {
        return;
    }
    const u64 lb = *reinterpret_cast<volatile u64*>(cat + 0x25e60);
    const u64 le = *reinterpret_cast<volatile u64*>(cat + 0x25e68);
    if (lb < 0x10000 || le <= lb || (le - lb) > 0x10000) {
        return;
    }
    const u32 n = static_cast<u32>((le - lb) >> 3);
    const u64 dat = base + (0x15ba4d0 - 0x107BF0);
    const u64 myslot = cat + 8 + 11 * 0xcd0;
    u32 bucket[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int myb = -2;
    bool my_enroll = false;
    u16 my_st = 0xffff, my_cat = 0xffff;
    u8 my_b16 = 0xff;
    for (u32 i = 0; i < n; ++i) {
        const u64 it = *reinterpret_cast<volatile u64*>(lb + i * 8);
        if (it < 0x10000) {
            continue;
        }
        const u16 st = *reinterpret_cast<volatile u16*>(it + 0x14);
        const u8 b16 = *reinterpret_cast<volatile u8*>(it + 0x16);
        const u16 c = *reinterpret_cast<volatile u16*>(it + 0x12);
        const bool enroll = (st != 0 && st != 3) && ((b16 & 1) == 0) &&
                            (c != 7 || *reinterpret_cast<volatile u8*>(it + 0x2a4) == 0);
        if (it == myslot) {
            my_enroll = enroll;
            my_st = st;
            my_cat = c;
            my_b16 = b16;
        }
        if (!enroll) {
            continue;
        }
        if (c != 0xffff && (static_cast<u64>(c) - 0xb) >= 3) {
            const s32 bk = *reinterpret_cast<volatile s32*>(dat + static_cast<u64>(c) * 4);
            if (bk >= 0 && bk < 8) {
                ++bucket[bk];
            }
            if (it == myslot) {
                myb = bk;
            }
        }
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-WATCH: MY slot st={} cat={} b16={:#x} enroll={} bucket={} | sizes News[1]={} "
             "Chal[2]={} Mining[4]={} (n={}) | refill ran={} inlist@enroll={} idx={} n={}",
             my_st, my_cat, my_b16, my_enroll, myb, bucket[1], bucket[2], bucket[4], n,
             g_refill_count, g_refill_inlist, g_refill_idx, g_refill_n);
    // Read the LIVE screen's actual per-category buckets POST-build (captured by the hook).
    const u64 scr = g_last_screen;
    if (scr >= 0x10000 && scr < 0x800000000000ull) {
        auto sz = [&](u32 idx) -> long {
            const u64 v = scr + 0xb8 + static_cast<u64>(idx) * 0x20;
            const u64 bb = *reinterpret_cast<volatile u64*>(v + 8);
            const u64 be = *reinterpret_cast<volatile u64*>(v + 0x10);
            if (bb < 0x10000 || be < bb || (be - bb) > 0x8000) {
                return -1;
            }
            return static_cast<long>((be - bb) >> 3);
        };
        // Scan ALL 7 buckets for the slot pointer -> which bucket (if any) actually holds it.
        int foundbk = -1;
        for (u32 idx = 0; idx < 7 && foundbk < 0; ++idx) {
            const u64 vb = scr + 0xb8 + static_cast<u64>(idx) * 0x20;
            const u64 bb = *reinterpret_cast<volatile u64*>(vb + 8);
            const u64 be = *reinterpret_cast<volatile u64*>(vb + 0x10);
            if (bb < 0x10000 || be < bb || (be - bb) > 0x8000) {
                continue;
            }
            for (u64 q = bb; q < be; q += 8) {
                if (*reinterpret_cast<volatile u64*>(q) == myslot) {
                    foundbk = static_cast<int>(idx);
                    break;
                }
            }
        }
        LOG_INFO(Lib_NpAuth,
                 "GR2-LIVEBUCKET: screen@{:#x} All[0]={} News[1]={} Chal[2]={} TH[3]={} Mine[4]={} "
                 "Photo[5]={} Tut[6]={} | OUR slot found in bucket {} (-1=NONE; 2=Challenges)",
                 scr, sz(0), sz(1), sz(2), sz(3), sz(4), sz(5), sz(6), foundbk);
    }
}

// The Challenges tab renders the cat-0 POOL slots, not the master/news list. Fill the real cm12
// pool slot (catalog+8+11*0xcd0) every tick to out-race the "force empty" maintenance (+0x10 ->
// 0xffff, +0x14 -> 0); never overwrite +0xa0 once it is a heap pointer (SceLibc double-free).
static void Gr2FillPoolSlot(u64 base) {
    (void)base;
    const u64 cat = Gr2ReadGlobal<u64>(0x1bea890);
    if (cat < 0x10000) {
        return;
    }
    const u64 S = cat + 8 + 11 * 0xcd0;  // cm12 mission block, slot 0
    if (*reinterpret_cast<volatile u16*>(S + 0x12) > 0x100) {
        return;  // doesn't look like the empty pool -- bail
    }
    // Read the PRE-WRITE state -- what the game left the slot as since the last write. Repeated
    // resets (id->0xffff / st->0 / cat changes) indicate the build-time race; unchanged values
    // mean the slot is stable and any non-render is a build-source issue, not a reset.
    const u16 pre_id = *reinterpret_cast<volatile u16*>(S + 0x10);
    const u16 pre_cat = *reinterpret_cast<volatile u16*>(S + 0x12);
    const u16 pre_st = *reinterpret_cast<volatile u16*>(S + 0x14);
    const u8 pre_b16 = *reinterpret_cast<volatile u8*>(S + 0x16);
    static u32 changed_st = 0, hidebit = 0, samples = 0;
    ++samples;
    if (pre_st != 1) {
        ++changed_st;
    }
    if (pre_b16 & 1) {
        ++hidebit;  // count how often the game has set the +0x16 "hide" bit (FUN_00e828f0 filter)
    }
    const u64 a0 = *reinterpret_cast<volatile u64*>(S + 0xa0);
    if (a0 < 0x10000) {  // +0xa0 is inline (not a game-allocated heap string) -> safe to write
        static const char nm[] = "junminlee2004";
        for (u32 i = 0; i < sizeof(nm); ++i) {
            *reinterpret_cast<volatile u8*>(S + 0xa0 + i) = static_cast<u8>(nm[i]);
        }
    }
    *reinterpret_cast<volatile u16*>(S + 0x10) = 1;            // valid id (not 0xffff empty)
    *reinterpret_cast<volatile u16*>(S + 0x12) = 0;            // category Challenge
    *reinterpret_cast<volatile u32*>(S + 0x14) = 1;            // state active/unread
    *reinterpret_cast<volatile u8*>(S + 0x16) &= 0xfe;         // clear the +0x16 bit0 "hide" flag
    *reinterpret_cast<volatile u32*>(S + 0xb4) = 0x76f0fed2u;  // cm12 FNV
    *reinterpret_cast<volatile u8*>(S + 0x18) = 0xff;          // tag (gate is open regardless)
    static u32 tk = 0;
    if ((tk++ % 1200) == 0) {
        LOG_INFO(Lib_NpAuth,
                 "GR2-FILL v17: S@{:#x} a0={:#x} | PRE-WRITE id={:#x} cat={} st={} b16={:#x} | over {} "
                 "samples: st-reset {}x, HIDE-bit(+0x16&1) set {}x",
                 S, a0, pre_id, pre_cat, pre_st, pre_b16, samples, changed_st, hidebit);
    }
}

// FUN_00e828f0 builds Challenges directly from the cat-0 pool slots (catalog+8+m*0xcd0+i*0x148),
// enrolling st != 0 && != 3; the pool boots empty. Activating a real cm12 slot in place (st=1,
// name @+0xa0, FNV @+0xb4) also passes the accept pointer-identity check, so it is playable.
static void Gr2ActivatePoolSlot(u64 base) {
    (void)base;
    const u64 cat = Gr2ReadGlobal<u64>(0x1bea890);
    if (cat < 0x10000) {
        return;
    }
    const u64 S = cat + 8 + 11 * 0xcd0;  // cm12 mission block, slot 0
    static u64 tick = 0;
    ++tick;
    // Re-assert every tick (the slot must be active whenever the menu opens) but self-latch: the
    // instant the game changes st != 1 it may have allocated freeable sub-objects, and
    // re-activating a torn-down slot double-frees. After latching, dump the slot ("Spost").
    static bool ever = false, latched = false, posted = false;
    static u64 latch_tick = 0;
    const u16 curst = *reinterpret_cast<volatile u16*>(S + 0x14);
    const u16 curcat = *reinterpret_cast<volatile u16*>(S + 0x12);
    const bool ours = (curst == 1 && curcat == 0 &&
                       *reinterpret_cast<volatile u8*>(S + 0xa0) == static_cast<u8>('Z'));
    if (latched) {
        if (!posted && tick > latch_tick + 1500) {
            posted = true;
            LOG_INFO(Lib_NpAuth, "GR2-POST v10: slot S@{:#x} after game took it over:", S);
            Gr2DumpCatalogHex(S, 0xc0, "Spost");
        }
        return;
    }
    if (ever && !ours) {  // game processed/tore down our slot -> back off permanently (no double-free)
        latched = true;
        latch_tick = tick;
        LOG_INFO(Lib_NpAuth, "GR2-LATCH v10: game changed slot st={} cat={} -> stop re-asserting",
                 curst, curcat);
        return;
    }
    if (!ever && *reinterpret_cast<volatile u16*>(S + 0x12) > 0x10) {
        return;  // not the empty pool yet
    }
    // Rows sourced from the news path carry the same "junminlee2004" name, so write an
    // unmistakable marker: a row showing "ZZ_POOL_MARKER" is this cat-0 pool slot, a
    // "junminlee2004 has challenged you" row came from elsewhere.
    static const char nm[] = "ZZ_POOL_MARKER";
    for (u32 i = 0; i < sizeof(nm); ++i) {
        *reinterpret_cast<volatile u8*>(S + 0xa0 + i) = static_cast<u8>(nm[i]);
    }
    *reinterpret_cast<volatile u32*>(S + 0xb4) = 0x76f0fed2u;  // cm12 FNV
    *reinterpret_cast<volatile u16*>(S + 0x10) = 0;            // clear the 0xffff empty-id marker
    *reinterpret_cast<volatile u16*>(S + 0x12) = 0;            // category Challenge
    *reinterpret_cast<volatile u32*>(S + 0x14) = 1;            // state: active/unread (enroll)
    // The enroll gate FUN_00e827a0 checks tag S+0x18 against catalog+0x25f98, all-zero without an
    // envtable (even cat-10 mine slots fail). tag==0xff is an unconditional accept and terminates
    // the tag walk, so tag0=0xff enrolls S regardless of the table.
    *reinterpret_cast<volatile u8*>(S + 0x18) = 0xff;
    ever = true;
    static int dn = 0;
    if (dn < 2) {
        ++dn;
        LOG_INFO(Lib_NpAuth, "GR2-ACTIVATE v6: cm12 slot S@{:#x} st<-1 cat<-0 name<-junminlee2004", S);
        Gr2DumpCatalogHex(S, 0xc0, "Sact");
        // (a) the enrollment gate FUN_00e827a0 reads tag bytes S+0x18..+0x1f against the enable
        // table catalog+0x25f98. Dump the table + evaluate the gate for S so we KNOW if it enrolls.
        const u8 tag0 = *reinterpret_cast<volatile u8*>(S + 0x18);
        const u8 flag = *reinterpret_cast<volatile u8*>(cat + 0x25fba);
        const u8 ten = (tag0 != 0xff) ? *reinterpret_cast<volatile u8*>(cat + 0x25f98 + tag0) : 1;
        LOG_INFO(Lib_NpAuth,
                 "GR2-GATE: S cat={} st={} b16={:#x} tag0={:#x} flag25fba={:#x} table[tag0]={:#x} "
                 "-> gate_pass={}",
                 static_cast<u32>(*reinterpret_cast<volatile u16*>(S + 0x12)),
                 static_cast<u32>(*reinterpret_cast<volatile u32*>(S + 0x14)),
                 static_cast<u32>(*reinterpret_cast<volatile u8*>(S + 0x16)), static_cast<u32>(tag0),
                 static_cast<u32>(flag), static_cast<u32>(ten), (tag0 == 0xff || ten != 0) ? 1 : 0);
        Gr2DumpCatalogHex(cat + 0x25f98, 0x30, "tagtable");
        // (b) scan all 480 master-list slots for any naturally non-empty one (st!=0 OR vtable!=0) to
        // use as a real template (its tag, vtable, sub-records reveal what a live challenge needs).
        const u64 mbegin = *reinterpret_cast<volatile u64*>(cat + 0x25e60);
        const u64 mend = *reinterpret_cast<volatile u64*>(cat + 0x25e68);
        if (mbegin >= 0x10000 && mend > mbegin && (mend - mbegin) <= 0x10000) {
            const u32 n = static_cast<u32>((mend - mbegin) >> 3);
            int active = 0;
            for (u32 i = 0; i < n; ++i) {
                const u64 it = *reinterpret_cast<volatile u64*>(mbegin + i * 8);
                if (it < 0x10000) {
                    continue;
                }
                const u16 st = *reinterpret_cast<volatile u16*>(it + 0x14);
                const u64 vt = *reinterpret_cast<volatile u64*>(it + 0x00);
                if ((st != 0 && i != 110) || (vt != 0 && active < 4)) {
                    ++active;
                    LOG_INFO(Lib_NpAuth, "GR2-LIVE: [{}] item@{:#x} vtable={:#x} cat={} st={} tag0={:#x}",
                             i, it, vt, static_cast<u32>(*reinterpret_cast<volatile u16*>(it + 0x12)),
                             static_cast<u32>(st),
                             static_cast<u32>(*reinterpret_cast<volatile u8*>(it + 0x18)));
                }
            }
            LOG_INFO(Lib_NpAuth, "GR2-LIVE: scanned {} slots, {} with state/vtable set", n, active);
        }
    }
}

// Runtime hook on FUN_00f00de0 (accept-screen "time left" countdown formatter). param_1(RDI) =
// countdown object: +0x28 mode (1=down/2=up), +0x30 current, +0x38 max, +0x18 -> manager whose
// +0x100 -> challenge-record array (stride 0xe0). Dumps timer fields + first records.
extern "C" __attribute__((no_stack_protector)) void Gr2CountdownHook(u64 obj) {
    if (obj < 0x10000 || obj >= 0x800000000000ull) {
        return;
    }
    // Kick-start the never-started validity countdown: with no recv_time the start that sets
    // cur=validity/mode=1 never runs, so force cur=max and mode=1 on the stuck signature (a
    // transient UI object, not the save-backed pool - save-safe). Runs every frame pre-throttle.
    {
        const u32 md0 = *reinterpret_cast<volatile u32*>(obj + 0x28);
        const u64 cur0 = *reinterpret_cast<volatile u64*>(obj + 0x30);
        const u64 mx0 = *reinterpret_cast<volatile u64*>(obj + 0x38);
        if (md0 == 0 && cur0 == 0 && mx0 > 0x100000ull && mx0 < 0x80000000ull) {
            *reinterpret_cast<volatile u64*>(obj + 0x30) = mx0;
            *reinterpret_cast<volatile u32*>(obj + 0x28) = 1;
            static u32 kc = 0;
            if ((kc++ % 120) == 0) {
                LOG_INFO(Lib_NpAuth, "GR2-CDKICK obj@{:#x} started: cur=max={:#x} mode 0->1", obj, mx0);
            }
        }
    }
    static u32 ck = 0;
    if ((ck++ % 180) != 0) {  // throttle verbose dump (~every 3s at 60fps)
        return;
    }
    const auto now = std::chrono::system_clock::now();
    const u64 uus =
        static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch())
                             .count());
    const u32 mode = *reinterpret_cast<volatile u32*>(obj + 0x28);
    const u64 cur = *reinterpret_cast<volatile u64*>(obj + 0x30);
    const u64 mx = *reinterpret_cast<volatile u64*>(obj + 0x38);
    const u64 f40 = *reinterpret_cast<volatile u64*>(obj + 0x40);
    const u64 mgr = *reinterpret_cast<volatile u64*>(obj + 0x18);
    LOG_INFO(Lib_NpAuth,
             "GR2-CDHOOK obj@{:#x} mode={} cur={:#x} MAX={:#x} +0x40={:#x} mgr={:#x} | now_us={}", obj,
             mode, cur, mx, f40, mgr, uus);
    if (mgr >= 0x10000 && mgr < 0x800000000000ull) {
        const u64 arr = *reinterpret_cast<volatile u64*>(mgr + 0x100);
        if (arr >= 0x10000 && arr < 0x800000000000ull) {
            for (u32 i = 0; i < 4; ++i) {
                const u64 rec = arr + i * 0xe0;
                // dump the record's 0xe0 bytes as u64s, flagging any time-like value
                for (u32 o = 0; o < 0xe0; o += 8) {
                    const u64 v = *reinterpret_cast<volatile u64*>(rec + o);
                    const bool timey =
                        (v > uus - 63072000000000ull && v < uus + 63072000000000ull) ||
                        (v > uus / 1000000 - 63072000u && v < uus / 1000000 + 63072000u);
                    if (v != 0 && (timey || o == 0xc4 - (0xc4 % 8))) {
                        LOG_INFO(Lib_NpAuth, "  CDREC[{}] +{:#05x} = {:#018x}{}", i, o, v,
                                 timey ? "  <TIME?>" : "");
                    }
                }
            }
        }
    }
}
alignas(16) static u8 g_cd_stub[160];
static void Gr2InstallF00de0Hook(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    done = true;
    const u64 fn = base + (0xf00de0 - 0x107BF0);
    u8 disp[17];
    for (u32 i = 0; i < 17; ++i) {
        disp[i] = *reinterpret_cast<volatile u8*>(fn + i);
    }
    u8* s = g_cd_stub;
    u32 p = 0;
    auto e = [&](u8 b) { s[p++] = b; };
    auto e64 = [&](u64 v) { for (int i = 0; i < 8; ++i) e(static_cast<u8>(v >> (i * 8))); };
    e(0x50); e(0x51); e(0x52); e(0x56); e(0x57);
    e(0x41); e(0x50); e(0x41); e(0x51); e(0x41); e(0x52); e(0x41); e(0x53);
    e(0x48); e(0xb8); e64(reinterpret_cast<u64>(&Gr2CountdownHook)); e(0xff); e(0xd0);
    e(0x41); e(0x5b); e(0x41); e(0x5a); e(0x41); e(0x59); e(0x41); e(0x58);
    e(0x5f); e(0x5e); e(0x5a); e(0x59); e(0x58);
    for (u32 i = 0; i < 17; ++i) {
        e(disp[i]);
    }
    e(0x48); e(0xb8); e64(fn + 17); e(0xff); e(0xe0);
    const u64 pg = reinterpret_cast<u64>(g_cd_stub) & ~static_cast<u64>(0xfff);
    const int mp = Gr2MakeRWX(reinterpret_cast<void*>(pg), 0x2000);
    const u64 sa = reinterpret_cast<u64>(g_cd_stub);
    u8 patch[17] = {0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe0, 0x90, 0x90, 0x90, 0x90, 0x90};
    for (int i = 0; i < 8; ++i) {
        patch[2 + i] = static_cast<u8>(sa >> (i * 8));
    }
    for (u32 i = 0; i < 17; ++i) {
        *reinterpret_cast<volatile u8*>(fn + i) = patch[i];
    }
    LOG_INFO(Lib_NpAuth, "GR2-HOOK v43: FUN_00f00de0@{:#x} -> stub@{:#x} ({}B) mprotect={} (countdown "
                         "recv_time probe)",
             fn, sa, p, mp);
}

// Read-only recv_time hunt: recv_time is unset (0), so scan the catalog/manager for date-valued
// u16 (years 2016..2037 = 0x7e0..0x7f5, the getinfodetail decoder's form) and absolute-timestamp
// u32/u64 (Unix s/us, sceRtc us) to find any date/time field the game did set.
static void Gr2RecvTimeProbe(u64 base) {
    (void)base;
    static u32 tk = 0;
    if ((++tk % 600) != 0) {
        return;
    }
    const u64 cat = Gr2ReadGlobal<u64>(0x1bea890);
    if (cat < 0x10000) {
        return;
    }
    const auto now = std::chrono::system_clock::now();
    const u64 uus =
        static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch())
                             .count());
    const u64 us = uus / 1000000ull;
    const u64 rtc = uus + 62135596800000000ull;
    LOG_INFO(Lib_NpAuth,
             "GR2-RECVPROBE now: unix_s={} unix_us={} rtc_us={} | scanning catalog for date(yr "
             "2016..2037)/timestamp fields",
             us, uus, rtc);
    u32 found = 0;
    for (u32 o = 0; o + 8 <= 0x28000 && found < 80; o += 4) {
        const u32 v32 = *reinterpret_cast<volatile u32*>(cat + o);
        const u64 v64 = *reinterpret_cast<volatile u64*>(cat + o);
        const u16 lo16 = static_cast<u16>(v32 & 0xffff);
        const bool isyear = (lo16 >= 2016 && lo16 <= 2037);
        const bool ts32 = (v32 > us - 31536000u && v32 < us + 31536000u);
        const bool ts64 = (v64 > uus - 31536000000000ull && v64 < uus + 31536000000000ull) ||
                          (v64 > rtc - 31536000000000ull && v64 < rtc + 31536000000000ull);
        // Only the u64 sceRtc/unix_us matches are real timestamps; unix_s-u32 and most year-u16
        // hits are ASCII false positives. Each real ts-u64 gets 0x50 bytes of hex+ascii context
        // to identify the record and pin the challenge received-time offset.
        if (ts64) {
            static u64 g_last_cluster = 0;
            const u64 cluster = o & ~0x1ffull;  // dedup within a 0x200 window
            LOG_INFO(Lib_NpAuth, "  RECVPROBE-TS +{:#07x} u64={:#018x} (unix_us~{}? rtc~{}?)", o, v64,
                     (v64 > uus - 31536000000000ull && v64 < uus + 31536000000000ull) ? 1 : 0,
                     (v64 > rtc - 31536000000000ull && v64 < rtc + 31536000000000ull) ? 1 : 0);
            if (cluster != g_last_cluster) {
                g_last_cluster = cluster;
                const u64 base_o = (o >= 0x30) ? (o - 0x30) : 0;
                for (u32 r = 0; r < 0x50; r += 16) {
                    const u64 a = cat + base_o + r;
                    char asc[17] = {0};
                    for (u32 k = 0; k < 16; ++k) {
                        const u8 c = *reinterpret_cast<volatile u8*>(a + k);
                        asc[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                    }
                    LOG_INFO(Lib_NpAuth, "     CTX +{:#07x}: {:016x} {:016x}  {}", base_o + r,
                             *reinterpret_cast<volatile u64*>(a),
                             *reinterpret_cast<volatile u64*>(a + 8), asc);
                }
            }
            ++found;
        } else if (isyear) {
            ++found;
        }
    }
    LOG_INFO(Lib_NpAuth, "GR2-RECVPROBE done: {} date/timestamp fields found", found);
}

// Read-only field map of the game-populated cm12 slot: the accept screen's "TIME'S UP" comes
// from a wrong/unset value, not a missing field. Classifies each 8-byte field (zero/magic/
// heap-ptr/small-int/ts candidate); current time printed as Unix us and sceRtc us.
static void Gr2DumpSlotMap(u64 base) {
    (void)base;
    static u32 tk = 0;
    if ((++tk % 300) != 0) {
        return;
    }
    const u64 cat = Gr2ReadGlobal<u64>(0x1bea890);
    if (cat < 0x10000) {
        return;
    }
    const u64 S = cat + 8 + 11 * 0xcd0;  // cm12 slot 0 (fork seed target)
    const auto now = std::chrono::system_clock::now();
    const u64 unix_us =
        static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch())
                             .count());
    const u64 unix_s = unix_us / 1000000ull;
    const u64 rtc_us = unix_us + 62135596800000000ull;  // sceRtc epoch (0001-01-01) offset
    LOG_INFO(Lib_NpAuth,
             "GR2-SLOTMAP cm12 @{:#x} | now: unix_s={} unix_us={} rtc_us={} (match a field to one of "
             "these = the recv_time/deadline)",
             S, unix_s, unix_us, rtc_us);
    for (u32 o = 0; o < 0x148; o += 8) {
        const u64 v = *reinterpret_cast<volatile u64*>(S + o);
        const u32 lo = static_cast<u32>(v & 0xffffffff);
        const u32 hi = static_cast<u32>(v >> 32);
        const char* tag = "int";
        if (v == 0) {
            tag = "ZERO";
        } else if (lo == 0x54321abc || lo == 0xdeadbeef || hi == 0x54321abc || hi == 0xdeadbeef) {
            tag = "MAGIC";
        } else if (v >= 0x100000000ull && v < 0x1000000000ull) {
            tag = "HEAPPTR";
        } else if (v >= 0x7f0000000000ull && v < 0x800000000000ull) {
            tag = "HOSTPTR";
        } else if ((v > unix_s - 31536000ull && v < unix_s + 31536000ull) ||
                   (v > unix_us - 31536000000000ull && v < unix_us + 31536000000000ull) ||
                   (v > rtc_us - 31536000000000ull && v < rtc_us + 31536000000000ull)) {
            tag = "*** TIMESTAMP? ***";
        }
        LOG_INFO(Lib_NpAuth, "  SLOTMAP +{:#05x} = {:#018x}  [{}]", o, v, tag);
    }
}

// The cm FNVs are not pre-baked at the assumed slot offset (catalog+8+m*0xcd0+i*0x148, +0xb4),
// so scan the catalog object (via DAT_01bea890) and the static-data interpretation for the
// cm-mission FNV markers instead. Read-only; bounded to the ~155KB mapped catalog.
static void Gr2DumpCatalog(u64 base) {
    const u64 ptr_catalog = Gr2ReadGlobal<u64>(0x1bea890);
    const u64 static_catalog = base + (0x1bea890 - 0x107BF0);
    LOG_INFO(Lib_NpAuth, "GR2-CAT0 DAT_01bea890: ptr={:#x} static={:#x}", ptr_catalog,
             static_catalog);
    // full cm-FNV table (cm01..cm20), index = mission-1
    static const u32 kCmFnv[20] = {0x79f34222, 0x78f3408f, 0x77f33efc, 0x76f33d69, 0x75f33bd6,
                                   0x74f33a43, 0x73f338b0, 0x82f3504d, 0x81f34eba, 0x74f0fbac,
                                   0x75f0fd3f, 0x76f0fed2, 0x77f10065, 0x70f0f560, 0x71f0f6f3,
                                   0x72f0f886, 0x73f0fa19, 0x7cf10844, 0x7df109d7, 0x06f89d47};
    (void)kCmFnv;
    const u64 cat = ptr_catalog;  // DAT_01bea890 points at the catalog object
    if (cat < 0x10000) {
        LOG_INFO(Lib_NpAuth, "GR2-CAT0 catalog ptr not ready");
        return;
    }
    constexpr u32 kScan = 0x28000;  // ~160KB, covers the 155KB catalog

    // (1) The EMPTY cm12 cat-0 target slot (catalog + 8 + 11*0xcd0). Slots are constructed-empty
    // (id 0xffff) and the cm-FNV at +0xb4 is NOT baked at boot, so the slot cannot be anchored
    // by FNV -- dump it raw.
    const u64 cm12_slot = cat + 8 + 11 * 0xcd0;
    LOG_INFO(Lib_NpAuth, "GR2-CAT0 cm12 empty slot @ {:#x} (= catalog+{:#x}):", cm12_slot,
             cm12_slot - cat);
    Gr2DumpCatalogHex(cm12_slot, 0x148, "cm12empty");

    // (2) Find an ACTIVE oshirase item by locating the challenger name ("junmin...") anywhere in the
    // catalog -- inline OR via a heap pointer. An active item (even the Special-News one) reveals the
    // name offset + the state/id fields that make a row SHOW, which we replicate for the cat-0 slot.
    int inline_hits = 0, ptr_hits = 0;
    for (u32 off = 0; off + 6 <= kScan; off += 1) {
        const volatile char* p = reinterpret_cast<const volatile char*>(cat + off);
        if (p[0] == 'j' && p[1] == 'u' && p[2] == 'n' && p[3] == 'm' && p[4] == 'i' && p[5] == 'n') {
            char s[49];
            Gr2ReadCStr(cat + off, s);
            LOG_INFO(Lib_NpAuth, "GR2-CAT0 INLINE name @ catalog+{:#x}: '{}'", off, s);
            if (++inline_hits >= 12) {
                break;
            }
        }
    }
    for (u32 off = 0; off + 8 <= kScan; off += 8) {
        const u64 v = *reinterpret_cast<volatile u64*>(cat + off);
        if (v >= 0x100000000ull && v < 0x1000000000ull) {
            const volatile char* p = reinterpret_cast<const volatile char*>(v);
            // read carefully; only check the first 6 bytes for the name prefix
            char c0 = p[0], c1 = p[1], c2 = p[2], c3 = p[3], c4 = p[4], c5 = p[5];
            if (c0 == 'j' && c1 == 'u' && c2 == 'n' && c3 == 'm' && c4 == 'i' && c5 == 'n') {
                char s[49];
                Gr2ReadCStr(v, s);
                LOG_INFO(Lib_NpAuth, "GR2-CAT0 PTR name @ catalog+{:#x} -> {:#x}: '{}'", off, v, s);
                // dump 0x60 bytes around this pointer slot (back up to a 0x10-aligned base) to show
                // the item fields surrounding the name pointer.
                const u64 itembase = cat + (off & ~0x3full);
                Gr2DumpCatalogHex(itembase, 0xc0, "activeitem");
                if (++ptr_hits >= 6) {
                    break;
                }
            }
        }
    }
    LOG_INFO(Lib_NpAuth, "GR2-CAT0 name search done: inline_hits={} ptr_hits={}", inline_hits,
             ptr_hits);
}

// GR2: drive the game's own receive path: one-shot FUN_00e7f3d0(list, mgr=DAT_01bea890) via
// ExecuteGuest with {count=1,[id=0,name]}. id selects FUN_00e8cc30's category: 0 = caseD_0 builds
// the slot from getmaildetail (0x7809 mission, 0x7816 validity -> +0x140); 1 reads the save pool.
static void Gr2InjectChallenge(u64 base) {
    const u64 mgr = Gr2ReadGlobal<u64>(0x1bea890);  // catalog == populator's mgr arg
    if (mgr < 0x10000) {
        return;
    }
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    if (linker == nullptr) {
        return;
    }
    const auto heap = linker->GetHeapApi();
    if (heap == nullptr || heap->heap_malloc == nullptr) {
        LOG_INFO(Lib_NpAuth, "GR2-INJECT: guest heap_malloc not ready yet");
        return;
    }
    // guest-visible buffer for the {count,[id,name]} list + the inline name string at +0x20
    const u64 buf = reinterpret_cast<u64>(Core::ExecuteGuest(heap->heap_malloc, (size_t)0x40));
    if (buf < 0x10000) {
        LOG_INFO(Lib_NpAuth, "GR2-INJECT: guest malloc failed");
        return;
    }
    const char* nm = "junminlee2004";
    char* gname = reinterpret_cast<char*>(buf + 0x20);
    u32 i = 0;
    for (; nm[i] != 0 && i < 0x1e; ++i) {
        gname[i] = nm[i];
    }
    gname[i] = 0;
    *reinterpret_cast<volatile u32*>(buf + 0x00) = 0;
    *reinterpret_cast<volatile u32*>(buf + 0x04) = 1;          // count
    *reinterpret_cast<volatile u32*>(buf + 0x08) = 0;          // entry[0].id = 0 -> caseD_0 CHALLENGE path
    *reinterpret_cast<volatile u32*>(buf + 0x0c) = 0;
    *reinterpret_cast<volatile u64*>(buf + 0x10) = buf + 0x20; // entry[0].name ptr

    const u64 pop_va = base + (0xe7f3d0 - 0x107BF0);
    const u64 cm12 = mgr + 8 + 11 * 0xcd0;
    LOG_INFO(Lib_NpAuth, "GR2-INJECT: populator={:#x} list={:#x} mgr={:#x} cm12slot={:#x}", pop_va, buf,
             mgr, cm12);
    LOG_INFO(Lib_NpAuth, "GR2-INJECT BEFORE: cm12 +0x10={:#x} +0x14={:#x} +0xb4={:#x} pendcnt={:#x}",
             *reinterpret_cast<volatile u32*>(cm12 + 0x10),
             *reinterpret_cast<volatile u32*>(cm12 + 0x14),
             *reinterpret_cast<volatile u32*>(cm12 + 0xb4),
             *reinterpret_cast<volatile u64*>(mgr + 0x263c0));

    using PopFn = PS4_SYSV_ABI void (*)(void*, u64);
    Core::ExecuteGuest(reinterpret_cast<PopFn>(pop_va), reinterpret_cast<void*>(buf), mgr);

    LOG_INFO(Lib_NpAuth, "GR2-INJECT AFTER:  cm12 +0x10={:#x} +0x14={:#x} +0xb4={:#x} pendcnt={:#x}",
             *reinterpret_cast<volatile u32*>(cm12 + 0x10),
             *reinterpret_cast<volatile u32*>(cm12 + 0x14),
             *reinterpret_cast<volatile u32*>(cm12 + 0xb4),
             *reinterpret_cast<volatile u64*>(mgr + 0x263c0));
    Gr2DumpCatalogHex(cm12, 0xc0, "cm12after");
    // +0x263b8 holds the list sentinel; the appended entry node is head->next. Dump the node so its
    // id (+0x10) and name string are visible as FUN_00e8cc30 reads them: name data at node+0x20, size
    // at node+0x38 (heap ptr when size > 0xf).
    const u64 head = *reinterpret_cast<volatile u64*>(mgr + 0x263b8);
    const u64 node = (head >= 0x10000) ? *reinterpret_cast<volatile u64*>(head) : 0;
    LOG_INFO(Lib_NpAuth, "GR2-INJECT: pending-list head={:#x} entry-node={:#x}", head, node);
    if (node >= 0x10000 && node != head) {
        Gr2DumpCatalogHex(node, 0x40, "entry");
    }

    // The scheduler already auto-runs the receive state machine FUN_00e8cc30; calling the rebuild
    // FUN_012decf0 here would re-enter it and race the scheduler's own call (not reentrant ->
    // crash risk). Seed-only: the scheduler picks up the id=0 entry and runs caseD_0 itself.
    static constexpr bool GR2_DRIVE_REBUILD = false;  // seed-only: the scheduler completes caseD_0
    const s32 gate = Gr2ReadGlobal<s32>(0x1c37ec0);
    LOG_INFO(Lib_NpAuth, "GR2-INJECT: seeded id=0 entry; letting the scheduler process it "
             "(rebuild gate DAT_01c37ec0 = {}, drive={})", gate, GR2_DRIVE_REBUILD);
    if (GR2_DRIVE_REBUILD) {
        static bool rebuilt_once = false;  // the rebuild re-inits + takes locks -> call it AT MOST ONCE
        if (!rebuilt_once) {
            rebuilt_once = true;
            const u64 rebuild_va = base + (0x12decf0 - 0x107BF0);
            LOG_INFO(Lib_NpAuth, "GR2-INJECT: calling rebuild FUN_012decf0 @ {:#x}", rebuild_va);
            using RebuildFn = PS4_SYSV_ABI void (*)();
            Core::ExecuteGuest(reinterpret_cast<RebuildFn>(rebuild_va));
            LOG_INFO(Lib_NpAuth,
                     "GR2-INJECT AFTER REBUILD: cm12 +0x10={:#x} +0x14={:#x} +0xb4={:#x} pendcnt={:#x}",
                     *reinterpret_cast<volatile u32*>(cm12 + 0x10),
                     *reinterpret_cast<volatile u32*>(cm12 + 0x14),
                     *reinterpret_cast<volatile u32*>(cm12 + 0xb4),
                     *reinterpret_cast<volatile u64*>(mgr + 0x263c0));
            Gr2DumpCatalogHex(cm12, 0xc0, "cm12rebuilt");
        }
    }
}

// Recategorize the live announcement row: a received challenge already renders under "Special
// News", and the categorizer jump table (vaddr 0xfee5d4) files item+0x12 == 0 as "Challenge"
// (Special News = 8). Anchored on the exact sender title (+0xa0); +0xb4 = FNV("cm12") baked too.
static void Gr2ForceChallengeCategory(u64 base) {
    const u64 cat = Gr2ReadGlobal<u64>(0x1bea890);
    if (cat < 0x10000) {
        return;
    }
    // 1. locate the name STRING buffer ("junminlee2004\0" -- the sender, not "...hjk").
    u64 name_addr = 0;
    for (u32 off = 0; off + 14 <= 0x28000; ++off) {
        const volatile char* p = reinterpret_cast<const volatile char*>(cat + off);
        bool m = true;
        for (u32 i = 0; i < 13; ++i) {
            if (p[i] != "junminlee2004"[i]) {
                m = false;
                break;
            }
        }
        if (m && p[13] == 0) {
            name_addr = cat + off;
            break;
        }
    }
    if (name_addr == 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOG_INFO(Lib_NpAuth, "GR2-FORCECAT: sender name string not found in catalog");
        }
        return;
    }
    (void)name_addr;
    // The categorizer FUN_00fee150 reads a Challenge row's name from the inline string at
    // item+0xa0 but a SpecialNews row's from *(item+0x28)+4, so flipping +0x12 alone draws a
    // blank name. Copy the title into +0xa0, set +0x14=1 and the cm12 FNV, then flip +0x12=0 last.
    const u64 lo = cat, hi = cat + 0x28000;
    int cat8 = 0, fixed = 0;
    static int dumpn = 0;
    for (u32 off = 0; off + 8 <= 0x28000; off += 8) {
        const u64 D = *reinterpret_cast<volatile u64*>(cat + off);
        if (D < lo || D >= hi || (D & 7) != 0) {
            continue;
        }
        if (*reinterpret_cast<volatile u16*>(D + 0x12) != 8) {
            continue;  // only the live Special-News challenge rows
        }
        ++cat8;
        const u64 detail = *reinterpret_cast<volatile u64*>(D + 0x28);
        if (detail < lo || detail >= hi) {
            continue;  // need a valid detail block to source the title string object
        }
        // Replicate the SpecialNews title string object (at detail+4, 0x10 bytes -- the std::string-like
        // the categorizer consumes) into the Challenge name slot D+0xa0. 0xa0+0x10=0xb0 < 0xb4, so this
        // does NOT clobber the cm-FNV we set next.
        for (u32 i = 0; i < 0x10; ++i) {
            *reinterpret_cast<volatile u8*>(D + 0xa0 + i) = *reinterpret_cast<volatile u8*>(detail + 4 + i);
        }
        *reinterpret_cast<volatile u32*>(D + 0x14) = 1;            // state: unread/active -> +0x10=3
        *reinterpret_cast<volatile u32*>(D + 0xb4) = 0x76f0fed2u;  // cm12 FNV (mission, resolver hint)
        *reinterpret_cast<volatile u16*>(D + 0x12) = 0;            // category Challenge  (LAST)
        if (dumpn < 3) {
            ++dumpn;
            LOG_INFO(Lib_NpAuth, "GR2-FORCECAT v5: D@{:#x} detail@{:#x} -> cat0 name<-detail+4",
                     D, detail);
            Gr2DumpCatalogHex(D, 0xc0, "Dcat0");
        }
        ++fixed;
    }
    static u32 nf = 0;
    if ((nf++ % 120) == 0) {
        LOG_INFO(Lib_NpAuth, "GR2-FORCECAT v5: cat8 rows seen={} converted-in-place={}", cat8, fixed);
    }
}

// NOTE: external linkage (NOT static) -- this is also called from the eop-flip path in
// another TU (the "[eopflip]" probe label). Marking it static breaks that cross-file call
// with: undefined symbol _ZN9Libraries2Np6NpAuth11Gr2ArcProbeEPKc.
void Gr2ArcProbe(const char* where) {
    const u64 base = Gr2ModuleBase();
    if (base == 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOG_ERROR(Lib_NpAuth,
                      "GR2-PROBE: module base unresolved -- fix Gr2ModuleBase()/set GR2_MANUAL_BASE");
        }
        return;
    }

    // GR2-only gate: the probes resolve GR2-eboot Ghidra globals (base + ghidra - 0x107BF0) and
    // chase them as heap pointers; on other titles (GRR: CUSA01130) the offsets hold garbage, and
    // Windows has no SIGSEGV absorber. Queried lazily - the serial is known before the first use.
    static const bool probe_is_gr2 = [] {
        static constexpr const char* kGr2Serials[] = {
            "CUSA03694", "CUSA04943", "CUSA04934", "CUSA00547",
            "PCJS50010", "PCAS00079", "CUSA04935",
        };
        const std::string_view serial = Common::ElfInfo::Instance().GameSerial();
        for (const char* s : kGr2Serials) {
            if (serial == s) {
                return true;
            }
        }
        return false;
    }();
    if (!probe_is_gr2) {
        return;
    }

    // DAT_01c02b60 is a pointer to the online-manager; deref then +0x30.
    const u64 ommgr = Gr2ReadGlobal<u64>(0x1c02b60);
    u32 state = 0xFFFFFFFFu;
    if (ommgr >= 0x100000ull) {
        state = *reinterpret_cast<volatile u32*>(ommgr + 0x30);
    }

    const s32 ss = Gr2ReadGlobal<s32>(0x1c26520);
    const s32 env = Gr2ReadGlobal<s32>(0x1c28178);
    const u64 envoff = static_cast<u64>(env) * 0x4b0;
    const s32 client_ver = Gr2ReadGlobal<s32>(0x1c26520 + envoff + 0xc8);
    const u64 login_url = Gr2ReadGlobal<u64>(0x1c26520 + envoff + 0x258);
    const u64 capone_host = Gr2ReadGlobal<u64>(0x1c26868 + envoff);
    const u64 ctx = Gr2ReadGlobal<u64>(0x1c25e88);

    u32 inflight = 0, session = 0;
    if (ctx != 0) {
        inflight = *reinterpret_cast<volatile u32*>(ctx + 0x10);
        session = *reinterpret_cast<volatile u32*>(ctx + 0x78);
    }

    static u32 last_state = 0xFFFFFFFEu;
    static u64 last_url = 0xFFFFFFFFFFFFFFFFull;
    static u32 last_session = 0xFFFFFFFFu;
    if (state != last_state || login_url != last_url || session != last_session) {
        LOG_INFO(Lib_NpAuth,
                 "GR2-PROBE [{}] state={:#x} omgr={:#x} ss={} env={} client_ver={} "
                 "login_url(+0x258)={:#x} capone_host(+0x348)={:#x} ctx={:#x} inflight={} SESSION={}",
                 where, state, ommgr, ss, env, client_ver, login_url, capone_host, ctx, inflight,
                 session);
        last_state = state;
        last_url = login_url;
        last_session = session;
    }

    // Upload gate snapshot: FUN_010422b0 bails to its network-error state unless DAT_01bf6270 !=
    // 0 && *(DAT_01bf62f8+0x50) && *(DAT_01bf62f8+0x58). DAT_01bf62f8 = the 0x88-byte upload
    // config (created empty on event 0x504af044); the getgamepropertyinfo blob likely fills it.
    const u64 up_cfg = Gr2ReadGlobal<u64>(0x1bf62f8);
    const u64 up_gate2 = Gr2ReadGlobal<u64>(0x1bf6270);
    u64 cfg50 = 0;
    u32 cfg58 = 0;
    if (up_cfg >= 0x100000ull) {
        cfg50 = *reinterpret_cast<volatile u64*>(up_cfg + 0x50);
        cfg58 = *reinterpret_cast<volatile u32*>(up_cfg + 0x58);
    }
    static u64 last_cfg = 0xFFFFFFFFFFFFFFFFull;
    static u64 last_cfg50 = 0xFFFFFFFFFFFFFFFFull;
    static u32 last_cfg58 = 0xFFFFFFFFu;
    if (up_cfg != last_cfg || cfg50 != last_cfg50 || cfg58 != last_cfg58) {
        const bool gate_ok = (up_gate2 != 0) && (cfg50 != 0) && (cfg58 != 0);
        LOG_INFO(Lib_NpAuth,
                 "GR2-UPLOADGATE cfg(0x1bf62f8)={:#x} +0x50={:#x} +0x58={:#x} "
                 "gate2(0x1bf6270)={:#x} => upload {}",
                 up_cfg, cfg50, cfg58, up_gate2, gate_ok ? "ALLOWED" : "BLOCKED(gate-null)");
        last_cfg = up_cfg;
        last_cfg50 = cfg50;
        last_cfg58 = cfg58;
    }

    // getuploadinfo S3-descriptor decoder globals. The decoder FUN_011a7c00 and the upload builder
    // FUN_0117ffb0 read: DAT_01c28160 crypto ctx (FUN_014ff120 in-place decrypt uses +8/+0x28),
    // DAT_01c281e0 access key (0x7040), DAT_01c281e8 S3 policy (fed by the ss.info upload_baseurl
    // extraction), DAT_01c281d8 HMAC signature (0x7050), and the response envelopes DAT_01c28180 /
    // DAT_01c28188. A null envelope pointer during the 0x7050 walk is the treasure-photo-upload
    // crash. Reads only these module .data scalars; logged on change.
    const u64 cx_ctx = Gr2ReadGlobal<u64>(0x1c28160);
    const u64 cx_sig = Gr2ReadGlobal<u64>(0x1c281d8);
    const u64 cx_key = Gr2ReadGlobal<u64>(0x1c281e0);
    const u64 cx_pol = Gr2ReadGlobal<u64>(0x1c281e8);
    const u64 cx_env0 = Gr2ReadGlobal<u64>(0x1c28180);
    const u64 cx_env1 = Gr2ReadGlobal<u64>(0x1c28188);
    static u64 last_cx = 0xFFFFFFFFFFFFFFFFull;
    const u64 cx_sig2 = cx_ctx ^ (cx_sig << 1) ^ (cx_key << 2) ^ (cx_pol << 3) ^ cx_env0 ^ cx_env1;
    if (cx_sig2 != last_cx) {
        last_cx = cx_sig2;
        LOG_INFO(Lib_NpAuth,
                 "GR2-UPLOADCRYPTO ctx(0x1c28160)={:#x} policy(0x1c281e8)={:#x} key(0x1c281e0)={:#x} "
                 "sig(0x1c281d8)={:#x} env0(0x1c28180)={:#x} env1(0x1c28188)={:#x}",
                 cx_ctx, cx_pol, cx_key, cx_sig, cx_env0, cx_env1);
    }

    // Upload host/port snapshot: both FUN_010422b0 pipelines (0xa challenge-ghost, 0x1a photo/
    // treasure) connect via FUN_010bd3c0(obj, DAT_01bf6260 host (SSO; inline when DAT_01bf6278 <
    // 0x10), _, DAT_01bf62a0 port) - ss.info-fed ranking-service globals; empty = init never ran.
    const u64 up_host_field0 = Gr2ReadGlobal<u64>(0x1bf6260);   // inline-data start OR heap ptr
    const u64 up_host_size   = Gr2ReadGlobal<u64>(0x1bf6278);   // size; < 0x10 => SSO inline
    const u32 up_port        = Gr2ReadGlobal<u32>(0x1bf62a0);
    const u64 up_port_raw    = Gr2ReadGlobal<u64>(0x1bf62a0);
    char up_host[48];
    up_host[0] = 0;
    {
        const u64 base2 = Gr2ModuleBase();
        if (base2 != 0) {
            u64 data_va = base2 + (0x1bf6260 - 0x107BF0);  // &DAT_01bf6260 (inline / SSO case)
            if (up_host_size >= 0x10) {
                data_va = up_host_field0;                  // long string -> heap ptr
            }
            if (data_va >= 0x10000ull) {
                Gr2ReadCStr(data_va, up_host);
            }
        }
    }
    static u64 last_uhost0 = 0xFFFFFFFFFFFFFFFFull;
    static u64 last_uhsize = 0xFFFFFFFFFFFFFFFFull;
    static u32 last_uport  = 0xFFFFFFFFu;
    if (up_host_field0 != last_uhost0 || up_host_size != last_uhsize || up_port != last_uport) {
        LOG_INFO(Lib_NpAuth,
                 "GR2-UPLOADHOST host(0x1bf6260)=\"{}\" size={:#x} field0={:#x} "
                 "port(0x1bf62a0)={} (raw={:#x}) => host {}",
                 up_host, up_host_size, up_host_field0, up_port, up_port_raw,
                 (up_host[0] != 0) ? "SET" : "EMPTY");
        last_uhost0 = up_host_field0;
        last_uhsize = up_host_size;
        last_uport  = up_port;
    }

    // Hexdump the upload-config object whenever its content changes, showing which of the 0x88
    // bytes are populated and which look like heap pointers into the property blob. An FNV-1a
    // hash over the 0x88 bytes gates the re-dump so steady state stays quiet.
    static u64 last_cfg_hash = 0;
    if (up_cfg >= 0x100000ull) {
        u64 h = 0xcbf29ce484222325ull;
        for (u32 off = 0; off < 0x88; off += 8) {
            h = (h ^ *reinterpret_cast<volatile u64*>(up_cfg + off)) * 0x100000001b3ull;
        }
        if (h != last_cfg_hash) {
            last_cfg_hash = h;
            LOG_INFO(Lib_NpAuth, "GR2-UPLOADCFG object @ {:#x} (0x88 bytes, content changed):",
                     up_cfg);
            for (u32 off = 0; off < 0x88; off += 8) {
                const u64 v = *reinterpret_cast<volatile u64*>(up_cfg + off);
                if (v != 0) {
                    LOG_INFO(Lib_NpAuth, "GR2-UPLOADCFG  +{:#04x} = {:#018x}", off, v);
                }
            }
        }
    }

    // One-time env-block dump, once the env table is populated (client_ver parsed).
    static bool dumped = false;
    if (!dumped && client_ver == 110) {
        dumped = true;
        Gr2DumpEnvBlock(base, env);
    }

    static u32 cat0_calls = 0;
    ++cat0_calls;
    // Catalog scan disabled: Gr2DumpCatalog's name hunt reads 160KB byte-by-byte, and a read into
    // an unmapped page is an access violation at a non-null address - outside the null window the
    // signal-handler absorber claims, so it is fatal on both hosts.
    (void)base;
    (void)cat0_calls;
    (void)&Gr2DumpCatalog;
    // The announcement pool is save-backed: catalog slot writes are serialized into the save and
    // corrupt it (mining rows stop repopulating on reload). Only Gr2OpenGate (patches eboot code,
    // never the save; needed for mines to render) and read-only watchers run here.
    Gr2OpenGate(base);                 // code-patch only; opens the render gate (mines + all categories)
    Gr2NeutralizeChGhostFree(base);    // code-patch only; NOPs the challenger-name deferred free
    Gr2NeutralizeNoticeEnvelopeFree(base);  // code-patch only; NOPs the Announcements-open double-free
    Gr2FixChallengeOwnName(base);      // code-patch only; own-row name in the in-race challenge list
    Gr2HideChallengePuppet(base);      // code-patch only; suppresses the inaccurate overworld puppet
    Gr2InstallFreeGuard(base);         // redirects the SceLibc free thunk; skips already-free chunks
    // cm12 injection hooks disabled: with empty challenge data the game resolves the sender, then
    // reads a second uninitialized online-id field -> npwebapi profiles?onlineId=<garbage> ->
    // NpToolkit write-back -> heap double-free. Re-enable only with matching challenge data.
    (void)&Gr2InstallFdf150Hook;       // disabled: cm12 slot hook (challenge popup)
    (void)&Gr2FillPoolSlot;            // disabled: per-flip cm12 slot fill
    (void)&Gr2CallEnumerate;           // disabled: one-shot enrollment call
    Gr2WatchBuckets(base);             // read-only periodic bucket watch (no writes)
    Gr2DumpMasterList(base);           // read-only dump
    // Countdown kick-start disabled: its loose guard (mode==0 && cur==0 && max in range) can match
    // non-countdown objects and write obj+0x30 = obj+0x38, aliasing two pointer fields into a
    // SceLibc double-free (ud2 crash on menu open).
    (void)&Gr2InstallF00de0Hook;       // disabled: kick-start can double-free on menu open
    (void)&Gr2DumpSlotMap;             // disabled: noisy
    (void)&Gr2RecvTimeProbe;           // disabled: noisy
    (void)&Gr2ForceChallengeCategory;
    (void)&Gr2CaptureRecord;
    (void)&Gr2RefillSlot;
    (void)&Gr2InstallE828f0Hook;
    (void)&Gr2ActivatePoolSlot;        // superseded by Gr2FillPoolSlot
#if defined(_WIN32)
    // GR2FORK: the challenge seed and the FUN_00e8cc30 capture hook are Linux-only: the hook is a
    // SysV-ABI trampoline (invalid under the Windows x64 convention) and its guest reads assume
    // the Linux page-fault absorber. The references below silence unused-function warnings.
    (void)&Gr2InstallE8cc30Hook;
    (void)&Gr2InjectChallenge;
#else
    // The FUN_00e8cc30 receive-capture hook disturbs normal announcements (they render as
    // already-read) and is not needed here, so it stays off, referenced only below. Challenge
    // delivery is server-side only: with GR2_SEED_CHALLENGE false the id=0 seed and the cm12
    // overworld-ghost activation below do not run, so the fork writes no challenge slots.
    (void)&Gr2InstallE8cc30Hook;
    static constexpr bool GR2_SEED_CHALLENGE = false;  // off: real challenges arrive as savedata mail; the seed only faked a hardcoded cm12 slot
    if constexpr (GR2_SEED_CHALLENGE) {
        // Only this seed populates the +0x263b8 pending list, and its download_mail_detail is
        // sent only while the boot rkg pump is live (~30s from boot), so it retries every ~55
        // ticks from tick 90 (max 6) while pendcnt==0 and no slot has +0xb4 == FNV("cm12").
        static u32 seed_tk = 0;
        static u32 attempts = 0;
        static bool built = false;
        ++seed_tk;
        const u64 mgr = Gr2ReadGlobal<u64>(0x1bea890);
        if (mgr >= 0x10000) {
            const u64 cm12 = mgr + 8 + 11 * 0xcd0;  // cm12 mission block (mission index 11)
            // Retained but inactive under GR2_SEED_CHALLENGE=false. caseD_0 builds a full cm12
            // sub-slot but leaves its visibility gate (u16 at +0x14) at 0. The overworld-ghost
            // collector FUN_00e946f0/findShowGhost shows the ghost only for sub-slots whose gate
            // is in the visible set (!= 0 && != 3). When enabled, it samples the built slot on a
            // throttled cadence, logs telemetry, and forces the gate to 1 for a bounded window.
            static u32 sample_tk = 0;
            static u32 win_tk = 0;  // frames since the cm12 slot first appeared
            ++sample_tk;
            if (built) {
                ++win_tk;
            }
            if ((sample_tk % 30) == 0) {
                for (u32 i = 0; i < 10; ++i) {
                    const u64 s = cm12 + i * 0x148;
                    if (*reinterpret_cast<volatile u32*>(s + 0xb4) == 0x76f0fed2) {
                        built = true;  // slot exists; the seed retry below stops appending entries
                        const u16 gate = *reinterpret_cast<volatile u16*>(s + 0x14);
                        // Collect count: cm12 sub-slots findShowGhost would treat as visible.
                        u32 collect = 0;
                        for (u32 j = 0; j < 10; ++j) {
                            const u16 g = *reinterpret_cast<volatile u16*>(cm12 + j * 0x148 + 0x14);
                            if (g != 0 && g != 3) {
                                ++collect;
                            }
                        }
                        LOG_INFO(Lib_NpAuth,
                                 "GR2-GHOST: cm12 sub-slot {} +0x14={:#x} +0x10={:#x} +0x140={:#x} "
                                 "sm(+0x263c8)={:#x} collect={}/10 win={}",
                                 i, gate, *reinterpret_cast<volatile u32*>(s + 0x10),
                                 *reinterpret_cast<volatile u64*>(s + 0x140),
                                 *reinterpret_cast<volatile u32*>(mgr + 0x263c8), collect, win_tk);
                        // Force the gate on for a bounded window (about 4000 frames) so the puppet
                        // can be reached; re-assert only on this cadence, never per frame.
                        if (win_tk < 4000 && gate != 1) {
                            *reinterpret_cast<volatile u16*>(s + 0x14) = 1;  // force gate visible
                            LOG_INFO(Lib_NpAuth,
                                     "GR2-GHOSTACT: cm12 sub-slot {} st {:#x}->1", i, gate);
                        }
                        break;
                    }
                }
            }
            const u64 pendcnt = *reinterpret_cast<volatile u64*>(mgr + 0x263c0);
            if (!built && attempts < 6 && seed_tk >= 90 && (seed_tk % 55) == 0 && pendcnt == 0) {
                ++attempts;
                LOG_INFO(Lib_NpAuth,
                         "GR2-SEED: attempt {} at tick {} (pendcnt=0) -> seeding id=0 entry while the boot "
                         "rkg pump should be live",
                         attempts, seed_tk);
                Gr2InjectChallenge(base);  // populator: append id=0 entry -> scheduler+pump -> caseD_0
            } else if (!built && (seed_tk % 55) == 0 && seed_tk >= 90) {
                LOG_INFO(Lib_NpAuth, "GR2-SEEDWATCH: tick {} attempts={} pendcnt={:#x} cm12[0] +0x14={:#x} "
                         "+0xb4={:#x}",
                         seed_tk, attempts, pendcnt, *reinterpret_cast<volatile u32*>(cm12 + 0x14),
                         *reinterpret_cast<volatile u32*>(cm12 + 0xb4));
            }
        }
    } else {
        (void)&Gr2InjectChallenge;         // keep referenced (avoid unused-function warning)
    }
#endif
}
// =============================================================================
// END GR2 PROBE
// =============================================================================

// Internal types for storing request-related information
enum class NpAuthRequestState {
    None = 0,
    Ready = 1,
    Aborted = 2,
    Complete = 3,
};

struct NpAuthRequest {
    NpAuthRequestState state;
    bool async;
    s32 result;
};

static std::vector<NpAuthRequest> g_auth_requests;

s32 CreateNpAuthRequest(bool async) {
    if (g_active_auth_requests == ORBIS_NP_AUTH_REQUEST_LIMIT) {
        return ORBIS_NP_AUTH_ERROR_REQUEST_MAX;
    }

    std::scoped_lock lk{g_auth_request_mutex};

    s32 req_index = 0;
    while (req_index < g_auth_requests.size()) {
        // Find first nonexistant request
        if (g_auth_requests[req_index].state == NpAuthRequestState::None) {
            // There is no request at this index, set the index to ready then break.
            g_auth_requests[req_index].state = NpAuthRequestState::Ready;
            g_auth_requests[req_index].async = async;
            break;
        }
        req_index++;
    }

    if (req_index == g_auth_requests.size()) {
        // There are no requests to replace.
        NpAuthRequest new_request{NpAuthRequestState::Ready, async, 0};
        g_auth_requests.emplace_back(new_request);
    }

    // Offset by one, first returned ID is 0x10000001
    g_active_auth_requests++;
    LOG_DEBUG(Lib_NpAuth, "called, async = {}", async);
    return req_index + ORBIS_NP_AUTH_REQUEST_ID_OFFSET + 1;
}

s32 PS4_SYSV_ABI sceNpAuthCreateRequest() {
    return CreateNpAuthRequest(false);
}

s32 PS4_SYSV_ABI sceNpAuthCreateAsyncRequest(const OrbisNpAuthCreateAsyncRequestParameter* param) {
    if (param == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }
    if (param->size != sizeof(OrbisNpAuthCreateAsyncRequestParameter)) {
        return ORBIS_NP_AUTH_ERROR_INVALID_SIZE;
    }

    return CreateNpAuthRequest(true);
}

s32 GetAuthorizationCode(s32 req_id, const OrbisNpAuthGetAuthorizationCodeParameterA* param,
                         s32 flag, OrbisNpAuthorizationCode* auth_code, s32* issuer_id) {
    if (param == nullptr || auth_code == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }
    if (param->size != sizeof(OrbisNpAuthGetAuthorizationCodeParameter)) {
        return ORBIS_NP_AUTH_ERROR_INVALID_SIZE;
    }
    if (param->user_id == -1 || param->client_id == nullptr || param->scope == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }

    // GR2 PROBE: sample online state + dump the env block once. Read-only.
    Gr2ArcProbe("GetAuthCode");

    std::scoped_lock lk{g_auth_request_mutex};

    // From here the actual authorization code request is performed.
    s32 req_index = req_id - ORBIS_NP_AUTH_REQUEST_ID_OFFSET - 1;
    if (g_active_auth_requests == 0 || g_auth_requests.size() <= req_index ||
        g_auth_requests[req_index].state == NpAuthRequestState::None) {
        return ORBIS_NP_AUTH_ERROR_REQUEST_NOT_FOUND;
    }

    auto& request = g_auth_requests[req_index];
    if (request.state == NpAuthRequestState::Complete) {
        request.result = ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    } else if (request.state == NpAuthRequestState::Aborted) {
        request.result = ORBIS_NP_AUTH_ERROR_ABORTED;
        return ORBIS_NP_AUTH_ERROR_ABORTED;
    }

    request.state = NpAuthRequestState::Complete;
    if (!g_signed_in) {
        request.result = ORBIS_NP_ERROR_SIGNED_OUT;
        // If the request is processed in some form, and it's an async request, then it returns OK.
        if (request.async) {
            return ORBIS_OK;
        }
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    // GR2 online restoration: the self-hosted login endpoint distinguishes players by userName
    // and never validates this authorization code, but the photo-submit flow stalls on an empty
    // code before building a login request - so a non-empty OAuth-style dummy code is returned.
    LOG_INFO(Lib_NpAuth, "GR2: returning dummy authorization code, req_id = {:#x}, async = {}",
             req_id, request.async);

    static constexpr char kGr2DummyAuthCode[] = "v3.shadps4_gr2_dummy_auth_code_AAAAAAAAAAAAAAAA";
    std::memset(auth_code, 0, sizeof(OrbisNpAuthorizationCode));
    std::strncpy(reinterpret_cast<char*>(auth_code), kGr2DummyAuthCode,
                 sizeof(OrbisNpAuthorizationCode) - 1);
    *issuer_id = 256;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI
sceNpAuthGetAuthorizationCode(s32 req_id, const OrbisNpAuthGetAuthorizationCodeParameter* param,
                              OrbisNpAuthorizationCode* auth_code, s32* issuer_id) {
    if (param == nullptr || auth_code == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }
    if (param->size != sizeof(OrbisNpAuthGetAuthorizationCodeParameter)) {
        return ORBIS_NP_AUTH_ERROR_INVALID_SIZE;
    }
    if (param->online_id == nullptr || param->client_id == nullptr || param->scope == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }
    if (!g_signed_in) {
        // Calls sceNpManagerIntGetUserIdByOnlineId to get a user id, returning any errors.
        // This call will not succeed while signed out because games cannot retrieve an online id.
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }

    // For simplicity, call sceUserServiceGetInitialUser to get the user_id.
    s32 user_id = 0;
    ASSERT(UserService::sceUserServiceGetInitialUser(&user_id) == 0);

    // Library constructs an OrbisNpAuthGetAuthorizationCodeParameterA struct,
    // then uses that to call GetAuthorizationCode.
    OrbisNpAuthGetAuthorizationCodeParameterA internal_params;
    std::memset(&internal_params, 0, sizeof(internal_params));
    internal_params.size = sizeof(internal_params);
    internal_params.client_id = param->client_id;
    internal_params.user_id = user_id;
    internal_params.scope = param->scope;

    return GetAuthorizationCode(req_id, &internal_params, 0, auth_code, issuer_id);
}

s32 PS4_SYSV_ABI
sceNpAuthGetAuthorizationCodeA(s32 req_id, const OrbisNpAuthGetAuthorizationCodeParameterA* param,
                               OrbisNpAuthorizationCode* auth_code, s32* issuer_id) {
    return GetAuthorizationCode(req_id, param, 0, auth_code, issuer_id);
}

s32 PS4_SYSV_ABI
sceNpAuthGetAuthorizationCodeV3(s32 req_id, const OrbisNpAuthGetAuthorizationCodeParameterA* param,
                                OrbisNpAuthorizationCode* auth_code, s32* issuer_id) {
    return GetAuthorizationCode(req_id, param, 1, auth_code, issuer_id);
}

s32 GetIdToken(s32 req_id, const OrbisNpAuthGetIdTokenParameterA* param, s32 flag,
               OrbisNpIdToken* token) {
    if (param == nullptr || token == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }
    if (param->size != sizeof(OrbisNpAuthGetIdTokenParameterA)) {
        return ORBIS_NP_AUTH_ERROR_INVALID_SIZE;
    }
    if (param->user_id == -1 || param->client_id == nullptr || param->client_secret == nullptr ||
        param->scope == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_auth_request_mutex};

    // From here the actual authorization code request is performed.
    s32 req_index = req_id - ORBIS_NP_AUTH_REQUEST_ID_OFFSET - 1;
    if (g_active_auth_requests == 0 || g_auth_requests.size() <= req_index ||
        g_auth_requests[req_index].state == NpAuthRequestState::None) {
        return ORBIS_NP_AUTH_ERROR_REQUEST_NOT_FOUND;
    }

    auto& request = g_auth_requests[req_index];
    if (request.state == NpAuthRequestState::Complete) {
        request.result = ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    } else if (request.state == NpAuthRequestState::Aborted) {
        request.result = ORBIS_NP_AUTH_ERROR_ABORTED;
        return ORBIS_NP_AUTH_ERROR_ABORTED;
    }

    request.state = NpAuthRequestState::Complete;
    if (!g_signed_in) {
        request.result = ORBIS_NP_ERROR_SIGNED_OUT;
        // If the request is processed in some form, and it's an async request, then it returns OK.
        if (request.async) {
            return ORBIS_OK;
        }
        return ORBIS_NP_ERROR_SIGNED_OUT;
    }

    LOG_ERROR(Lib_NpAuth, "(STUBBED) called, req_id = {:#x}, async = {}", req_id, request.async);

    // Not sure what values are expected here, so zeroing this for now.
    // (GR2 uses the authorization-code path, not the id-token path, so this stub is unchanged.
    // Revisit if a trace shows GetIdToken being called and stalling.)
    std::memset(token, 0, sizeof(OrbisNpIdToken));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAuthGetIdToken(s32 req_id, const OrbisNpAuthGetIdTokenParameter* param,
                                     OrbisNpIdToken* token) {
    if (param == nullptr || token == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }
    if (param->size != sizeof(OrbisNpAuthGetIdTokenParameter)) {
        return ORBIS_NP_AUTH_ERROR_INVALID_SIZE;
    }
    if (param->online_id == nullptr || param->client_id == nullptr ||
        param->client_secret == nullptr || param->scope == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }
    if (!g_signed_in) {
        // Calls sceNpManagerIntGetUserIdByOnlineId to get a user id, returning any errors.
        // This call will not succeed while signed out because games cannot retrieve an online id.
        return ORBIS_NP_ERROR_USER_NOT_FOUND;
    }

    // For simplicity, call sceUserServiceGetInitialUser to get the user_id.
    s32 user_id = 0;
    ASSERT(UserService::sceUserServiceGetInitialUser(&user_id) == 0);

    // Library constructs an OrbisNpAuthGetIdTokenParameterA struct,
    // then uses that to call GetIdToken.
    OrbisNpAuthGetIdTokenParameterA internal_params;
    std::memset(&internal_params, 0, sizeof(internal_params));
    internal_params.size = sizeof(internal_params);
    internal_params.user_id = user_id;
    internal_params.client_id = param->client_id;
    internal_params.client_secret = param->client_secret;
    internal_params.scope = param->scope;

    return GetIdToken(req_id, &internal_params, 0, token);
}

s32 PS4_SYSV_ABI sceNpAuthGetIdTokenA(s32 req_id, const OrbisNpAuthGetIdTokenParameterA* param,
                                      OrbisNpIdToken* token) {
    return GetIdToken(req_id, param, 0, token);
}

s32 PS4_SYSV_ABI sceNpAuthGetIdTokenV3(s32 req_id, const OrbisNpAuthGetIdTokenParameterA* param,
                                       OrbisNpIdToken* token) {
    return GetIdToken(req_id, param, 1, token);
}

s32 PS4_SYSV_ABI sceNpAuthSetTimeout(s32 req_id, s32 resolve_retry, u32 resolve_timeout,
                                     u32 conn_timeout, u32 send_timeout, u32 recv_timeout) {
    LOG_ERROR(Lib_NpAuth, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAuthAbortRequest(s32 req_id) {
    LOG_DEBUG(Lib_NpAuth, "called req_id = {:#x}", req_id);

    std::scoped_lock lk{g_auth_request_mutex};

    s32 req_index = req_id - ORBIS_NP_AUTH_REQUEST_ID_OFFSET - 1;
    if (g_active_auth_requests == 0 || g_auth_requests.size() <= req_index ||
        g_auth_requests[req_index].state == NpAuthRequestState::None) {
        return ORBIS_NP_AUTH_ERROR_REQUEST_NOT_FOUND;
    }

    if (g_auth_requests[req_index].state == NpAuthRequestState::Complete) {
        // If the request is already complete, abort is ignored.
        return ORBIS_OK;
    }

    g_auth_requests[req_index].state = NpAuthRequestState::Aborted;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAuthWaitAsync(s32 req_id, s32* result) {
    if (result == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_auth_request_mutex};

    s32 req_index = req_id - ORBIS_NP_AUTH_REQUEST_ID_OFFSET - 1;
    if (g_active_auth_requests == 0 || g_auth_requests.size() <= req_index ||
        g_auth_requests[req_index].state == NpAuthRequestState::None) {
        return ORBIS_NP_AUTH_ERROR_REQUEST_NOT_FOUND;
    }

    if (!g_auth_requests[req_index].async ||
        g_auth_requests[req_index].state == NpAuthRequestState::Ready) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ID;
    }

    // Since we're not actually performing any sort of network request here,
    // we can just set result based on the request and return.
    *result = g_auth_requests[req_index].result;
    LOG_WARNING(Lib_NpAuth, "called req_id = {:#x}, returning result = {:#x}", req_id,
                static_cast<u32>(*result));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAuthPollAsync(s32 req_id, s32* result) {
    if (result == nullptr) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ARGUMENT;
    }

    std::scoped_lock lk{g_auth_request_mutex};

    s32 req_index = req_id - ORBIS_NP_AUTH_REQUEST_ID_OFFSET - 1;
    if (g_active_auth_requests == 0 || g_auth_requests.size() <= req_index ||
        g_auth_requests[req_index].state == NpAuthRequestState::None) {
        return ORBIS_NP_AUTH_ERROR_REQUEST_NOT_FOUND;
    }

    if (!g_auth_requests[req_index].async ||
        g_auth_requests[req_index].state == NpAuthRequestState::Ready) {
        return ORBIS_NP_AUTH_ERROR_INVALID_ID;
    }

    // Since we're not actually performing any sort of network request here,
    // we can just set result based on the request and return.
    *result = g_auth_requests[req_index].result;
    LOG_WARNING(Lib_NpAuth, "called req_id = {:#x}, returning result = {:#x}", req_id,
                static_cast<u32>(*result));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNpAuthDeleteRequest(s32 req_id) {
    LOG_DEBUG(Lib_NpAuth, "called req_id = {:#x}", req_id);

    std::scoped_lock lk{g_auth_request_mutex};

    s32 req_index = req_id - ORBIS_NP_AUTH_REQUEST_ID_OFFSET - 1;
    if (g_active_auth_requests == 0 || g_auth_requests.size() <= req_index ||
        g_auth_requests[req_index].state == NpAuthRequestState::None) {
        return ORBIS_NP_AUTH_ERROR_REQUEST_NOT_FOUND;
    }

    g_active_auth_requests--;
    g_auth_requests[req_index].state = NpAuthRequestState::None;
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    g_signed_in = Config::getPSNSignedIn();

    LIB_FUNCTION("6bwFkosYRQg", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthCreateRequest);
    LIB_FUNCTION("N+mr7GjTvr8", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthCreateAsyncRequest);
    LIB_FUNCTION("KxGkOrQJTqY", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthGetAuthorizationCode);
    LIB_FUNCTION("qAUXQ9GdWp8", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthGetAuthorizationCodeA);
    LIB_FUNCTION("KI4dHLlTNl0", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthGetAuthorizationCodeV3);
    LIB_FUNCTION("uaB-LoJqHis", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthGetIdToken);
    LIB_FUNCTION("CocbHVIKPE8", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthGetIdTokenA);
    LIB_FUNCTION("RdsFVsgSpZY", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthGetIdTokenV3);
    LIB_FUNCTION("PM3IZCw-7m0", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthSetTimeout);
    LIB_FUNCTION("cE7wIsqXdZ8", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthAbortRequest);
    LIB_FUNCTION("SK-S7daqJSE", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthWaitAsync);
    LIB_FUNCTION("gjSyfzSsDcE", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthPollAsync);
    LIB_FUNCTION("H8wG9Bk-nPc", "libSceNpAuth", 1, "libSceNpAuth", sceNpAuthDeleteRequest);

    LIB_FUNCTION("KxGkOrQJTqY", "libSceNpAuthCompat", 1, "libSceNpAuth",
                 sceNpAuthGetAuthorizationCode);
    LIB_FUNCTION("uaB-LoJqHis", "libSceNpAuthCompat", 1, "libSceNpAuth", sceNpAuthGetIdToken);
};

} // namespace Libraries::Np::NpAuth
