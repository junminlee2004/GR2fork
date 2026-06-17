#!/usr/bin/env bash
#
# bump-version.sh
#
# Updates the user-facing GR2FORK / GR2fork version string (e.g. v4.8 -> v4.9)
# in BOTH source trees:
#     <DIR>/src-gr2/emulator.cpp   (brand: "GR2FORK", plus the "Build: vX.Y" log)
#     <DIR>/src/emulator.cpp       (brand: "GR2fork", plus the "Starting GR2fork vX.Y" log)
#
# It ONLY rewrites the version on the real code lines (the LOG_INFO build/start
# line and the `window_title = fmt::format(...)` lines). It deliberately leaves
# every `GR2FORK vX.Y` / `GR2fork vX.Y` that appears inside a // comment
# (FIX(GR2FORK v3.1), TITLE(GR2FORK v1.0), the v0.13.0 example string, etc.)
# untouched, and never touches the `v{}` Common::g_version placeholders.
#
# Usage:
#     ./bump-version.sh [DIR]
#         DIR   parent directory holding src-gr2/ and src/   (default: . )
#
set -euo pipefail

ROOT="${1:-.}"
GR2_FILE="$ROOT/src-gr2/emulator.cpp"
SRC_FILE="$ROOT/src/emulator.cpp"

# ---------------------------------------------------------------- validate ---
missing=0
for f in "$GR2_FILE" "$SRC_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: not found: $f" >&2
        missing=1
    fi
done
if [[ $missing -ne 0 ]]; then
    echo "Aborting — expected '$ROOT/src-gr2/emulator.cpp' and '$ROOT/src/emulator.cpp'." >&2
    exit 1
fi

# ----------------------------------------- detect current version (display) ---
# Read it off a real code line (a window_title fmt::format) so we never pick up
# the v1.0 / v3.1 etc. that only live in comments.
detect_ver() {
    grep -E 'window_title = fmt::format' "$1" 2>/dev/null \
        | grep -oE '(GR2FORK |GR2fork )v[0-9]+(\.[0-9]+)*' \
        | grep -oE 'v[0-9]+(\.[0-9]+)*' \
        | head -n1 || true
}
cur_gr2="$(detect_ver "$GR2_FILE")"
cur_src="$(detect_ver "$SRC_FILE")"

echo "Current version:"
echo "  src-gr2 : ${cur_gr2:-<none found>}"
echo "  src     : ${cur_src:-<none found>}"
echo

# ------------------------------------------------------------------- prompt ---
read -rp "What version value do you desire? (e.g. 4.9) " NEWVER
NEWVER="${NEWVER#v}"                       # tolerate a typed leading 'v'
NEWVER="${NEWVER//[[:space:]]/}"           # strip stray whitespace

if [[ -z "$NEWVER" ]]; then
    echo "ERROR: no version entered. Aborting." >&2
    exit 1
fi
if [[ ! "$NEWVER" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
    echo "ERROR: '$NEWVER' doesn't look like a version (expected e.g. 4.9 or 5.0.1)." >&2
    exit 1
fi

echo
echo "Setting version -> v$NEWVER"
echo

# -------------------------------------------------------------------- apply ---
bump_file() {
    local file="$1"
    cp -p -- "$file" "$file.bak"
    sed -i -E \
        -e "s/(\"Build: )v[0-9]+(\.[0-9]+)*/\1v${NEWVER}/" \
        -e "s/(\"Starting GR2fork )v[0-9]+(\.[0-9]+)*/\1v${NEWVER}/" \
        -e "/window_title = fmt::format/ s/(GR2FORK )v[0-9]+(\.[0-9]+)*/\1v${NEWVER}/g" \
        -e "/window_title = fmt::format/ s/(GR2fork )v[0-9]+(\.[0-9]+)*/\1v${NEWVER}/g" \
        -- "$file"
}

bump_file "$GR2_FILE"
bump_file "$SRC_FILE"

# ------------------------------------------------------------------- verify ---
echo "=== src-gr2/emulator.cpp (changed lines) ==="
grep -nE '"Build: v[0-9]|window_title = fmt::format\("Junmin Lee GR2FORK ' "$GR2_FILE" || true
echo
echo "=== src/emulator.cpp (changed lines) ==="
grep -nE '"Starting GR2fork v[0-9]|window_title = fmt::format\("Junmin Lee GR2fork ' "$SRC_FILE" || true
echo
echo "Done. Original files backed up to:"
echo "  $GR2_FILE.bak"
echo "  $SRC_FILE.bak"
