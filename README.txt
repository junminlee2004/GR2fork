GR2fork online-restoration port  (combined build → src-gr2 core)
================================================================

This tree mirrors the repo layout. Each file sits where it goes; copy
CMakeLists.txt + src-gr2/ over your tree (scripts/ is optional tooling).

WHAT'S IN HERE
--------------
RE-BASED onto your latest uploads (2026-06-20) — purely additive, your
parallel work in these files is untouched:
  CMakeLists.txt                         + Cpp_Httplib on core_gr2 LIBS (1 line)
  src-gr2/common/config.cpp              + httpHostOverride, threaded like userName (7 lines)
  src-gr2/common/config.h                + GetHttpHostOverride() decl (+ comment)

THE BB TRANSPORT + GR2 ADDITION (unchanged since the first drop):
  src-gr2/core/libraries/network/http.cpp   real httplib transport (host_override + ReplaceHost);
                                            byte-identical to the metr1k-bb-server fork
  src-gr2/core/libraries/network/http.h     same, +1 line (#include common/logging/log.h)
  src-gr2/core/libraries/secure/secure.cpp  NEW — sceLibSecureCryptographyDecrypt plaintext passthrough
  src-gr2/core/libraries/secure/secure.h    NEW
  src-gr2/core/libraries/libs.cpp           +#include secure.h, +LibSecure::RegisterLib(sym)

  scripts/dump_libsecure_decrypt.py         PyGhidra: confirm libSecure import lib name + decrypt arg order

>>> ONE CAVEAT: libs.cpp here is based on the ORIGINAL repo-zip baseline, NOT a
    fresh upload. If you've locally modified libs.cpp since (e.g. new RegisterLib
    calls from other work), DON'T blind-overwrite it — just add these two lines:
        #include "core/libraries/secure/secure.h"          // with the other network includes
        Libraries::LibSecure::RegisterLib(sym);            // next to the Http/Http2 RegisterLib calls
    config.cpp/.h and CMakeLists.txt ARE fresh-from-your-uploads, so those overwrite cleanly.

secure.cpp is auto-globbed into core_gr2 (GLOB_RECURSE); no source-list edit needed.

TO RUN (game-specific config, custom_configs/CUSA04943.toml or the JSON GR2Fork section)
----------------------------------------------------------------------------------------
  isPSNSignedIn      = true
  isConnectedToNetwork = true
  userName           = "<something other than shadPS4>"
  httpHostOverride   = "<your server host>"        (General section; default "localhost")
Patches OFF: anything that blocks HTTP or forces region.
Boot log prints  Replaced URL host, new URL: http://<host>/gd2-…/ss.info?…  per request.

RUNTIME-VERIFY (self-diagnosing, only affects the data-op TLV, NOT the online flip)
-----------------------------------------------------------------------------------
libSecure's exact import lib name (journal cites libSceSecure.prx; firmware module is
libSceLibSecure) and decrypt arg order can't be pinned statically. secure.cpp registers
the passthrough under BOTH names and uses the canonical (ctx,dst,dstSize,src,srcSize,*processed)
signature. If the boot log prints:
    Linker: Stub resolved sceLibSecureCryptographyDecrypt as ... (lib: X, mod: Y)
then neither matched → put X/Y in secure.cpp. Fast lib-name check:
    findstr /i "hMYgMP-Vuno" nid_map.csv
