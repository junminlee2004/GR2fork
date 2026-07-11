#!/usr/bin/env bash
# One-shot: clean embed build (both cores baked into one shadps4) + package as an
# AppImage. Run from the combined repo root. This is the explicit form of
# `BUILD=1 ./build-appimage.sh` — the build step is spelled out so it's easy to edit.
#
# For a portable / Steam Deck-safe image, swap both -march values to x86-64-v3
# (or just run:  MARCH=x86-64-v3 BUILD=1 ./build-appimage.sh).
set -euo pipefail

# run from this script's own directory (the repo root), wherever it's invoked from
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 1. clean embed build ----------
# CMAKE_DISABLE_FIND_PACKAGE_SDL3=ON forces shadPS4's VENDORED externals/sdl3 to build
# (find_package would otherwise grab the Arch SYSTEM SDL3, which carries a high glibc
# floor into the AppImage). The vendored build uses this toolchain, so its glibc floor
# matches the cores' (~2.38) instead of the system package's (e.g. 2.43).
rm -rf build_opt
cmake -S . -B build_opt \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_FLAGS="-march=znver4 -mtune=znver4 -O3 -flto=thin -DNDEBUG" \
  -DCMAKE_CXX_FLAGS="-march=znver4 -mtune=znver4 -O3 -flto=thin -DNDEBUG" \
  -DCMAKE_EXE_LINKER_FLAGS="-flto=thin -fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCOMBINED_EMBED_CORES=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_SDL3=ON

# fail fast if the embed flag didn't actually take (the missing-backslash trap):
# without this you'd build a dev tree and only notice when the AppImage isn't self-contained.
grep -q '^COMBINED_EMBED_CORES:BOOL=ON' build_opt/CMakeCache.txt \
  || { echo "ERROR: COMBINED_EMBED_CORES is not ON in build_opt/CMakeCache.txt — the embed flag didn't take. Aborting before the build."; exit 1; }

cmake --build build_opt --parallel "$(nproc)"

# ---------- 2. package the AppImage ----------
[[ -x ./build-appimage.sh ]] || { echo "ERROR: ./build-appimage.sh not found next to this script"; exit 1; }
./build-appimage.sh
