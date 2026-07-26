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

#include "common/elf_info.h"
#include "common/singleton.h"
#include "core/linker.h"
#include "core/tls.h"  // Core::ExecuteGuest -- call guest functions (the challenge populator)

namespace Libraries::Np::NpAuth {

static bool g_signed_in = false;
static s32 g_active_auth_requests = 0;
static std::mutex g_auth_request_mutex;

// GR2 eboot globals: runtime_va = base + (ghidra_va - 0x107BF0).

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
    // FUN_00e827a0's prologue: push rbp; mov rbp,rsp; push r15. The gate address is hardcoded
    // against one eboot build, so on any other build these bytes are unrelated mid-function code
    // and writing the stub silently corrupts it (the clobber faults later, far from here). Verify
    // before writing, as the other code patches do.
    static const u8 expect[6] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
    u8 orig[6];
    for (u32 i = 0; i < 6; ++i) {
        orig[i] = *reinterpret_cast<volatile u8*>(gate + i);
    }
    if (std::memcmp(orig, expect, sizeof(orig)) != 0) {
        LOG_ERROR(Lib_NpAuth,
                  "GR2-OPENGATE: FUN_00e827a0 @{:#x} orig {:02x}{:02x}{:02x}{:02x}{:02x}{:02x} "
                  "not the expected prologue -- MISMATCH, not patched (unsupported game build)",
                  gate, orig[0], orig[1], orig[2], orig[3], orig[4], orig[5]);
        return;
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

// GR2FORK FIX: the Treasure Send confirmation dialog (shown after photographing a found treasure,
// before uploading the hint) draws its online-id line from the global SceNpOnlineId buffer at
// 0x1bf64e8. FUN_00ee3ec0 (Lua treasureSendOpen) zeroes that buffer, then the widget update
// FUN_01092540 refills it in state 4 with strncpy(&buf, hintTable[area] + 0x20, 0x10) at 0x1092966.
// hintTable rows are parsed from the treasure_hint savedata records, whose online-id names the hint's
// POSTER - correct for the accept dialog, wrong for the send dialog, where the line is the local
// player and the field holds whatever the slot last carried (uninitialized bytes -> a glyph). The
// widget class is shared, with the mode at this+8: 0 = TreasureStart, 1 = TreasureSend,
// 2 = PhotoReview (branched away before this site), 3 = collection. Gate the substitution on mode 1
// so only the send dialog names the local player; the accept and collection screens keep the served
// name byte for byte. RAX still feeds the pending CMOVC and EDX carries the size, so the trampoline
// scratches RCX (dead here, reloaded at 0x109297a) and replays the CMOVC while the carry flag from
// CMP RCX,0x8 is still live.
alignas(16) static u8 g_tsendname_stub[96];
static char g_tsendname_buf[24] = {0};
static void Gr2FixTreasureSendOwnName(u64 base) {
    static bool done = false;
    if (done || base == 0) {
        return;
    }
    done = true;
    const u64 site = base + (0x1092966 - 0x107BF0);  // CMOVC RSI,RAX; ADD RSI,0x20; CALL strncpy
    u8 orig[13];
    for (u32 i = 0; i < sizeof(orig); ++i) {
        orig[i] = *reinterpret_cast<volatile u8*>(site + i);
    }
    // expect 48 0f 42 f0 | 48 83 c6 20 | e8 ad a9 46 00
    static constexpr u8 kOrig[13] = {0x48, 0x0f, 0x42, 0xf0, 0x48, 0x83, 0xc6,
                                     0x20, 0xe8, 0xad, 0xa9, 0x46, 0x00};
    if (std::memcmp(orig, kOrig, sizeof(kOrig)) != 0) {
        LOG_ERROR(Lib_NpAuth,
                  "GR2-TSENDNAME: FUN_01092540 fill @{:#x} orig {:02x}{:02x}{:02x}{:02x}... "
                  "unexpected -- MISMATCH, not patched",
                  site, orig[0], orig[1], orig[2], orig[3]);
        return;
    }
    std::memset(g_tsendname_buf, 0, sizeof(g_tsendname_buf));
    std::strncpy(g_tsendname_buf, GR2Fork::Auth::EffectiveOnlineId().c_str(),
                 sizeof(g_tsendname_buf) - 1);
    const u64 strncpy_rt = base + (0x14fd320 - 0x107BF0);  // the copy the site tail-calls
    const u64 resume_rt = base + (0x1092973 - 0x107BF0);   // instruction after the displaced call
    u8* s = g_tsendname_stub;
    u32 p = 0;
    auto e = [&](u8 b) { s[p++] = b; };
    auto e64 = [&](u64 v) { for (int i = 0; i < 8; ++i) e(static_cast<u8>(v >> (i * 8))); };
    // Replay the displaced source select FIRST, while the carry flag still holds.
    e(0x48); e(0x0f); e(0x42); e(0xf0);                            // cmovc rsi, rax
    e(0x48); e(0x83); e(0xc6); e(0x20);                            // add rsi, 0x20
    e(0x41); e(0x83); e(0x7d); e(0x08); e(0x01);                   // cmp dword [r13+8], 1
    e(0x75); e(0x0a);                                              // jne +10 (keep served name)
    e(0x48); e(0xbe); e64(reinterpret_cast<u64>(g_tsendname_buf)); // movabs rsi, &name
    // RDI (the global buffer) and EDX (0x10) are already set; call the copy, then resume in place.
    e(0x48); e(0xb8); e64(strncpy_rt); e(0xff); e(0xd0);           // movabs rax, strncpy; call rax
    e(0x48); e(0xb8); e64(resume_rt); e(0xff); e(0xe0);            // movabs rax, resume; jmp rax
    const u64 pg = reinterpret_cast<u64>(g_tsendname_stub) & ~static_cast<u64>(0xfff);
    const int mp = Gr2MakeRWX(reinterpret_cast<void*>(pg), 0x2000);
    const u64 sa = reinterpret_cast<u64>(g_tsendname_stub);
    // 12 bytes of patch + one NOP to cover the 13 displaced bytes.
    u8 patch[13] = {0x48, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xe1, 0x90};
    for (int i = 0; i < 8; ++i) {
        patch[2 + i] = static_cast<u8>(sa >> (i * 8));
    }
    for (u32 i = 0; i < sizeof(patch); ++i) {
        *reinterpret_cast<volatile u8*>(site + i) = patch[i];
    }
    LOG_INFO(Lib_NpAuth,
             "GR2-TSENDNAME: FUN_01092540 fill @{:#x} -> stub@{:#x} ({}B) mprotect={} name='{}'",
             site, sa, p, mp, g_tsendname_buf);
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
    // they get built DURING FUN_00fdf150.
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




    // The announcement pool is save-backed: catalog slot writes are serialized into the save and
    // corrupt it (mining rows stop repopulating on reload). Only Gr2OpenGate (patches eboot code,
    // never the save; needed for mines to render) runs here.
    Gr2OpenGate(base);                 // code-patch only; opens the render gate (mines + all categories)
    Gr2NeutralizeChGhostFree(base);    // code-patch only; NOPs the challenger-name deferred free
    Gr2NeutralizeNoticeEnvelopeFree(base);  // code-patch only; NOPs the Announcements-open double-free
    Gr2FixChallengeOwnName(base);      // code-patch only; own-row name in the in-race challenge list
    Gr2FixTreasureSendOwnName(base);   // code-patch only; own name on the treasure-hint send dialog
    Gr2HideChallengePuppet(base);      // code-patch only; suppresses the inaccurate overworld puppet
    Gr2InstallFreeGuard(base);         // redirects the SceLibc free thunk; skips already-free chunks
    // cm12 injection hooks disabled: with empty challenge data the game resolves the sender, then
    // reads a second uninitialized online-id field -> npwebapi profiles?onlineId=<garbage> ->
    // NpToolkit write-back -> heap double-free. Re-enable only with matching challenge data.
    (void)&Gr2InstallFdf150Hook;       // disabled: cm12 slot hook (challenge popup)
    (void)&Gr2FillPoolSlot;            // disabled: per-flip cm12 slot fill
    // Countdown kick-start disabled: its loose guard (mode==0 && cur==0 && max in range) can match
    // non-countdown objects and write obj+0x30 = obj+0x38, aliasing two pointer fields into a
    // SceLibc double-free (ud2 crash on menu open).
    (void)&Gr2InstallF00de0Hook;       // disabled: kick-start can double-free on menu open
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
