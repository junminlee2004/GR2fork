#!/usr/bin/env bash
# Build a single-file AppImage from the combined (embedded-cores) shadPS4 binary.
#
# Run from the combined repo root AFTER building with -DCOMBINED_EMBED_CORES=ON
# (so build_opt/shadps4 is self-contained and build_opt/libcore_*.so exist as the
# embed inputs). To also (re)build here, pass BUILD=1.
#
#   ./build-appimage.sh                # package an already-built embed tree
#   BUILD=1 ./build-appimage.sh        # clean-build (embed) then package
#   MARCH=x86-64-v3 BUILD=1 ./build-appimage.sh   # portable/Deck-safe build
#
# WHY this isn't just "linuxdeploy --executable shadps4": the cores are
# .incbin-embedded, so the binary's own NEEDED list is tiny (libdl/libc/libstdc++).
# Their real deps (SDL3, FFmpeg, OpenAL, libusb, …) live in the .so files, which
# the bundler can't see inside the binary. So we point linuxdeploy at the loose
# cores to collect THEIR deps, then drop the redundant cores (they're in the binary).
# And a memfd-dlopened core resolves its NEEDED via the process search path, so the
# AppRun must export LD_LIBRARY_PATH=$APPDIR/usr/lib.
set -euo pipefail

# ---------- config (override via env) ----------
DO_BUILD="${BUILD:-0}"                 # BUILD=1 -> (re)build with embedded cores first
REPO="${REPO:-$PWD}"
BUILD_DIR="${BUILD_DIR:-$REPO/build_opt}"
APPDIR="${APPDIR:-$REPO/AppDir}"
OUT="${OUT:-$REPO/shadPS4-combined-x86_64.AppImage}"
ICON_SRC="${ICON_SRC:-$REPO/src/images/shadps4.png}"
MARCH="${MARCH:-znver4}"
JOBS="$(nproc)"

# linuxdeploy/appimagetool are AppImages; run them without FUSE (Bazzite-safe).
export APPIMAGE_EXTRACT_AND_RUN=1
# linuxdeploy's bundled `strip` is too old to parse RELR (.relr.dyn) sections, which
# modern distros (Arch/Bazzite) emit in their system libs — it aborts on libSDL3,
# libopenal, libpng, etc. Those libs are already stripped, so skip the redundant pass.
export NO_STRIP=1

# ---------- 0. optional (re)build with embedded cores ----------
if [[ "$DO_BUILD" == 1 ]]; then
    rm -rf "$BUILD_DIR"
    cmake -S "$REPO" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_C_FLAGS="-march=$MARCH -mtune=znver4 -O3 -flto=thin -DNDEBUG" \
        -DCMAKE_CXX_FLAGS="-march=$MARCH -mtune=znver4 -O3 -flto=thin -DNDEBUG" \
        -DCMAKE_EXE_LINKER_FLAGS="-flto=thin -fuse-ld=lld" \
        -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
        -DCOMBINED_EMBED_CORES=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_SDL3=ON
    cmake --build "$BUILD_DIR" --parallel "$JOBS"
fi

BIN="$BUILD_DIR/shadps4"
CORE_GR2="$BUILD_DIR/libcore_gr2.so"
CORE_MAIN="$BUILD_DIR/libcore_main.so"
for f in "$BIN" "$CORE_GR2" "$CORE_MAIN"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f — build with -DCOMBINED_EMBED_CORES=ON first (or pass BUILD=1)"; exit 1; }
done

# Guard: this script bundles for the EMBED build (cores baked into the binary, then the
# loose cores are DROPPED from the AppImage in step 5). A tiny binary means a DEV build
# (cores NOT embedded) -> dropping them would leave the AppImage with no cores to load.
# An embed binary is large (both emulator cores baked in); a dev dispatcher is tens of KB.
bin_kb=$(( $(stat -c%s "$BIN") / 1024 ))
if (( bin_kb < 2048 )); then
    echo "ERROR: $BIN is only ${bin_kb} KB — that's a DEV build, not an embed build."
    echo "       This script packages the embed route. Rebuild with -DCOMBINED_EMBED_CORES=ON"
    echo "       (configure must print 'Combined build: RELEASE'), then re-run."
    exit 1
fi

# ---------- 1. AppDir skeleton ----------
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/lib"
cp "$BIN" "$APPDIR/usr/bin/shadps4"

# ---------- 2. desktop entry + icon ----------
cat > "$APPDIR/shadps4.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=shadPS4 (combined)
Comment=gr2fork + prerelease, one binary, core auto-selected by title id
Exec=shadps4
Icon=shadps4
Categories=Game;Emulator;
Terminal=false
DESK
if [[ -f "$ICON_SRC" ]]; then
    cp "$ICON_SRC" "$APPDIR/shadps4.png"
else
    # minimal 1x1 PNG so appimagetool has an icon (replace with a real one anytime)
    printf '\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82' > "$APPDIR/shadps4.png"
fi

# ---------- 3. fetch tools if missing ----------
LD="${LINUXDEPLOY:-$REPO/linuxdeploy-x86_64.AppImage}"
AT="${APPIMAGETOOL:-$REPO/appimagetool-x86_64.AppImage}"
if [[ ! -x "$LD" ]]; then
    curl -fL -o "$LD" https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
    chmod +x "$LD"
fi
if [[ ! -x "$AT" ]]; then
    curl -fL -o "$AT" https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
    chmod +x "$AT"
fi

# ---------- 4. bundle deps of the binary AND the cores ----------
# --library makes linuxdeploy copy each core into usr/lib and pull in its deps
# (SDL3/FFmpeg/OpenAL/libusb/…). Its built-in excludelist keeps host-provided libs
# OUT (libvulkan, libGL, libdrm, libwayland, libX11, glibc, …) — your RADV/Mesa
# driver comes from the host, which is what you want.
"$LD" --appdir "$APPDIR" \
    --executable "$APPDIR/usr/bin/shadps4" \
    --library "$CORE_GR2" \
    --library "$CORE_MAIN" \
    --desktop-file "$APPDIR/shadps4.desktop" \
    --icon-file "$APPDIR/shadps4.png"

# ---------- 5. drop the redundant cores (they're embedded in the binary) ----------
rm -f "$APPDIR/usr/lib/libcore_gr2.so" "$APPDIR/usr/lib/libcore_main.so"

# ---------- 6. custom AppRun: memfd cores need LD_LIBRARY_PATH -> bundled libs ----------
# rm first: linuxdeploy creates AppRun as a SYMLINK to usr/bin/shadps4, so writing
# through it with `cat >` would overwrite the binary itself.
rm -f "$APPDIR/AppRun"
cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="$HERE/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/usr/bin/shadps4" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

# ---------- 7. package ----------
rm -f "$OUT"
ARCH=x86_64 "$AT" "$APPDIR" "$OUT"

# ---------- 8. glibc-floor report (portability) ----------
# The AppImage can only run on a host whose glibc is >= the highest GLIBC_x.y symbol
# any bundled lib references. Print each lib's floor and the overall max so you know
# the minimum host glibc before shipping (e.g. Bazzite 2.42, SteamOS lower).
if command -v objdump >/dev/null 2>&1; then
    echo ""
    echo "glibc floor of the packaged AppImage (highest GLIBC symbol per file):"
    for f in "$APPDIR/usr/bin/shadps4" "$APPDIR"/usr/lib/*.so*; do
        [ -f "$f" ] || continue
        v=$(objdump -T "$f" 2>/dev/null | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -V | tail -1)
        [ -n "$v" ] && printf '  %-30s %s\n' "$(basename "$f")" "$v"
    done
    floor=$(for f in "$APPDIR/usr/bin/shadps4" "$APPDIR"/usr/lib/*.so*; do
        [ -f "$f" ] && objdump -T "$f" 2>/dev/null | grep -oE 'GLIBC_[0-9]+\.[0-9]+'
    done | sort -V | tail -1)
    echo "  -> requires host glibc >= ${floor#GLIBC_}"
fi

echo ""
echo "built: $OUT  ($(du -h "$OUT" | cut -f1))"
echo "run:   $OUT /path/to/game        (core auto-selected by title id)"
echo "       $OUT --core=main /path/...  (force a core)"
