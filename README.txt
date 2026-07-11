GR2fork online-restoration port  (combined build → src-gr2 core)
================================================================

Adds discontinued-online support to the src-gr2 core: an httplib-based HTTP
transport with a configurable host override, plus a libSecure decrypt
passthrough so the game's online data-ops resolve.

WHAT'S IN HERE
--------------
  CMakeLists.txt                            + Cpp_Httplib on core_gr2 LIBS (1 line)
  src-gr2/common/config.cpp                 + httpHostOverride, threaded like userName
  src-gr2/common/config.h                   + GetHttpHostOverride() decl

  src-gr2/core/libraries/network/http.cpp   httplib transport (host_override + ReplaceHost);
                                            byte-identical to the metr1k-bb-server fork
  src-gr2/core/libraries/network/http.h     +1 line (#include common/logging/log.h)
  src-gr2/core/libraries/secure/secure.cpp  sceLibSecureCryptographyDecrypt plaintext passthrough
  src-gr2/core/libraries/secure/secure.h
  src-gr2/core/libraries/libs.cpp           +#include secure.h, +LibSecure::RegisterLib(sym)

  scripts/dump_libsecure_decrypt.py         PyGhidra: confirm libSecure import lib name + decrypt arg order

secure.cpp is auto-globbed into core_gr2 (GLOB_RECURSE); no source-list edit needed.
libs.cpp needs two lines, next to the other network RegisterLib calls:
    #include "core/libraries/secure/secure.h"
    Libraries::LibSecure::RegisterLib(sym);

TO RUN (game-specific config, custom_configs/CUSA04943.toml or the JSON GR2Fork section)
----------------------------------------------------------------------------------------
  isPSNSignedIn        = true
  isConnectedToNetwork = true
  userName             = "<something other than shadPS4>"
  httpHostOverride     = "<server host>"        (General section; default "localhost")
Patches OFF: anything that blocks HTTP or forces region.
Boot log prints  Replaced URL host, new URL: http://<host>/gd2-.../ss.info?...  per request.

RUNTIME-VERIFY (self-diagnosing, only affects the data-op TLV, NOT the online flip)
-----------------------------------------------------------------------------------
libSecure's exact import lib name (the bundled module is libSceSecure.prx; the firmware
module is libSceLibSecure) and decrypt arg order can't be pinned statically. secure.cpp
registers the passthrough under BOTH names and uses the canonical
(ctx,dst,dstSize,src,srcSize,*processed) signature. If the boot log prints:
    Linker: Stub resolved sceLibSecureCryptographyDecrypt as ... (lib: X, mod: Y)
then neither matched → put X/Y in secure.cpp. Fast lib-name check:
    findstr /i "hMYgMP-Vuno" nid_map.csv
