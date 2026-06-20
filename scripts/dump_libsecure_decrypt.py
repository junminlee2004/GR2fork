# dump_libsecure_decrypt.py  —  confirm the two libSecure unknowns for the online port
# ---------------------------------------------------------------------------------------
# Resolves the only two things the ported libSecure passthrough can't know statically:
#   (1) the exact import LIBRARY/MODULE name the eboot uses for sceLibSecureCryptographyDecrypt
#       (NID hMYgMP-Vuno) — so secure.cpp's LIB_FUNCTION name is provably right, and
#   (2) the exact CALL SIGNATURE (arg count + order) at the decrypt call sites
#       FUN_011972A0 (decode-entry) and FUN_01197150 (tag-0x60 sub-block), so the
#       passthrough memcpy maps src->dst correctly.
#
# USAGE (per GR2_RE_GUIDE §2 — PyGhidra only; close the Ghidra GUI first):
#   Paste the BODY below under your standard working dump_*.py header (the part that gives
#   you `program`), then:
#     & "C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe" `
#       "C:\Users\Administrator\PS4RE\dump_libsecure_decrypt.py"
#   Paste the printed output back into chat.
#
# FAST PATH for (1) — no Ghidra needed, you already have the table:
#   findstr /i "hMYgMP-Vuno sceLibSecureCryptographyDecrypt" C:\Users\Administrator\PS4RE\nid_map.csv
#   The 4th CSV column (lib) is the answer. If it's neither libSceLibSecure nor libSceSecure,
#   set that name in src-gr2/core/libraries/secure/secure.cpp and rebuild.
# ---------------------------------------------------------------------------------------

from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.program.model.symbol import SourceType
from ghidra.app.cmd.disassemble import DisassembleCommand
from ghidra.program.model.address import AddressSet

# Ghidra addresses (image base 0x107BF0, per the journal/guide). Both call sceLibSecureCryptographyDecrypt.
DECRYPT_CALLERS = [0x011972A0,   # FUN_011972A0 — decode-entry: frames body then calls Decrypt
                   0x01197150]   # FUN_01197150 — tag 0x60: decrypts a sub-block + RTC freshness check


def decompile_addrs(program, ghidra_addrs, out_path):
    fm = program.getFunctionManager()
    sp = program.getAddressFactory().getDefaultAddressSpace()
    di = DecompInterface()
    di.openProgram(program)
    mon = ConsoleTaskMonitor()
    with open(out_path, 'w', encoding='utf-8') as f:
        for a in ghidra_addrs:
            ad = sp.getAddress(a)
            if program.getListing().getInstructionAt(ad) is None:  # force-disassemble if needed
                DisassembleCommand(ad, AddressSet(ad), True).applyTo(program, mon)
            fn = fm.getFunctionContaining(ad) or fm.createFunction(None, ad, None,
                                                                   SourceType.USER_DEFINED)
            res = di.decompileFunction(fn, 60, mon)
            f.write("/* {} @ {} */\n".format(fn.getName(), ad))
            f.write(res.getDecompiledFunction().getC()
                    if res.decompileCompleted() else "// DECOMP FAILED\n")
            f.write("\n\n")
    print("wrote", out_path)


def report_import_library(program, needle="sceLibSecureCryptographyDecrypt"):
    # Print every symbol whose name contains the needle, with its parent namespace.
    # For an import, the parent namespace chain is typically <Library>::<Module> — that is
    # the (library, module) pair secure.cpp must register under.
    st = program.getSymbolTable()
    hits = 0
    for sym in st.getAllSymbols(True):
        nm = sym.getName()
        if needle.lower() in nm.lower():
            ns = sym.getParentNamespace()
            chain = []
            while ns is not None and ns.getName() != "Global":
                chain.append(ns.getName())
                ns = ns.getParentNamespace()
            print("  symbol {:<40} @ {:<14} ns: {}".format(
                nm, str(sym.getAddress()), " <- ".join(chain) if chain else "(global)"))
            hits += 1
    if hits == 0:
        print("  (no symbol matched '{}' — use the nid_map.csv FAST PATH above for the lib name)"
              .format(needle))


print("==== (1) import library/module name for sceLibSecureCryptographyDecrypt ====")
report_import_library(program)
print("\n==== (2) decrypt call-site decompilation (read the call's arg setup) ====")
decompile_addrs(program, DECRYPT_CALLERS, r"C:\Users\Administrator\PS4RE\libsecure_decrypt_dump.txt")
print("Open libsecure_decrypt_dump.txt and paste it back — the call to the decrypt thunk")
print("shows the exact (ctx, dst, dstSize, src, srcSize, [processed]) order to confirm.")
