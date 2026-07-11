#!/usr/bin/env bash
#
# set-version.sh  —  GR2fork deploy pipeline
#
# From a single version prompt, against the combined repo <ROOT>:
#   1. src-gr2/emulator.cpp  -> BUMP the GR2FORK version
#                               (Build log + GR2FORK window_titles, comment-safe)
#   2. src/emulator.cpp      -> BRAND if pristine upstream, else BUMP
#                               ("Junmin Lee GR2fork vX.Y - " before "shadPS4";
#                                Starting-log left untouched)
#   3. src/core/libraries/fiber/fiber.cpp
#                            -> REPLACE with the combined-binary fiber fix
#                               (companion file fiber_core_main_replacement.cpp,
#                                kept next to this script)
#   4. git add those files, commit, and push HEAD to  origin/gr2fork-main
#      (gated: must be a git repo on the gr2fork-new branch with an origin
#       remote; a final y/N confirm precedes the push)
#
# Usage:
#     ./set-version.sh [ROOT]
#         ROOT  the combined repo dir (default: /home/jlee/Downloads/safe/gr2fork-new)
#
#     FIBER_REPLACEMENT=/path/to/file.cpp   override the replacement source
#
set -euo pipefail

# --------------------------------------------------------------- configuration ---
ROOT="${1:-/home/jlee/Downloads/safe/gr2fork-new}"
BRANCH="gr2fork-main"                          # remote branch on origin to push HEAD to
LOCAL_BRANCH="${LOCAL_BRANCH:-gr2fork-new}"     # expected local working branch
REMOTE="origin"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

GR2_FILE="$ROOT/src-gr2/emulator.cpp"
SRC_FILE="$ROOT/src/emulator.cpp"
FIBER_DIR="$ROOT/src/core/libraries/fiber"
FIBER_TARGET="$FIBER_DIR/fiber.cpp"

SRC_ANCHOR='fmt::format("shadPS4 '          # pristine upstream window_title opener
SRC_BRANDED='Junmin Lee GR2fork v'          # marker that src is already branded

# Resolve the replacement source (env override, else next to this script;
# accept either the correct spelling or the original "replacment" typo).
if [[ -z "${FIBER_REPLACEMENT:-}" ]]; then
    for cand in "$SCRIPT_DIR/fiber_core_main_replacement.cpp" \
                "$SCRIPT_DIR/fiber_core_main_replacment.cpp"; do
        [[ -f "$cand" ]] && { FIBER_REPLACEMENT="$cand"; break; }
    done
fi
FIBER_REPLACEMENT="${FIBER_REPLACEMENT:-}"

# ------------------------------------------------------------------- validate ---
err=0
for f in "$GR2_FILE" "$SRC_FILE"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f" >&2; err=1; }
done
[[ -d "$FIBER_DIR" ]] || { echo "ERROR: fiber dir not found: $FIBER_DIR" >&2; err=1; }
if [[ -z "$FIBER_REPLACEMENT" || ! -f "$FIBER_REPLACEMENT" ]]; then
    echo "ERROR: replacement file not found." >&2
    echo "       Looked next to the script for fiber_core_main_replacement.cpp" >&2
    echo "       (set FIBER_REPLACEMENT=/path to override)." >&2
    err=1
fi
[[ $err -eq 0 ]] || { echo "Aborting — fix the paths above (ROOT='$ROOT')." >&2; exit 1; }

# git preconditions (checked BEFORE any file is modified)
REPO_ROOT="$(git -C "$ROOT" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
    echo "ERROR: '$ROOT' is not inside a git repository — cannot push." >&2
    exit 1
fi
CUR_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [[ "$CUR_BRANCH" != "$LOCAL_BRANCH" ]]; then
    echo "ERROR: repo is on branch '$CUR_BRANCH', not '$LOCAL_BRANCH'." >&2
    echo "       Switch first:  git -C \"$REPO_ROOT\" checkout $LOCAL_BRANCH" >&2
    echo "       (or set LOCAL_BRANCH=... to deploy from a different local branch)" >&2
    exit 1
fi
if ! git -C "$REPO_ROOT" remote | grep -qx "$REMOTE"; then
    echo "ERROR: no git remote named '$REMOTE' in $REPO_ROOT." >&2
    exit 1
fi

# ----------------------------------------- detect current state (for display) ---
detect_gr2_ver() {
    grep -E 'window_title = fmt::format' "$1" 2>/dev/null \
        | grep -oE 'GR2FORK v[0-9]+(\.[0-9]+)*' \
        | grep -oE 'v[0-9]+(\.[0-9]+)*' | head -n1 || true
}
detect_src_state() {
    if grep -qF "$SRC_BRANDED" "$1"; then
        local v; v="$(grep -oE 'Junmin Lee GR2fork v[0-9]+(\.[0-9]+)*' "$1" \
                        | grep -oE 'v[0-9]+(\.[0-9]+)*' | head -n1 || true)"
        echo "branded ${v:-?}"
    elif grep -qF "$SRC_ANCHOR" "$1"; then
        echo "pristine"
    else
        echo "unrecognized"
    fi
}

cur_gr2="$(detect_gr2_ver "$GR2_FILE")"
src_state="$(detect_src_state "$SRC_FILE")"
if [[ -f "$FIBER_TARGET" ]] && cmp -s "$FIBER_TARGET" "$FIBER_REPLACEMENT"; then
    fiber_state="already matches replacement"
else
    fiber_state="will be replaced"
fi

echo "Repo    : $REPO_ROOT  (local branch $CUR_BRANCH, push to $REMOTE/$BRANCH)"
echo "Current state:"
echo "  src-gr2 : ${cur_gr2:-<no GR2FORK version found>}  (will bump)"
case "$src_state" in
    branded*) echo "  src     : already branded ${src_state#branded }  (will bump)";;
    pristine) echo "  src     : pristine upstream  (will brand)";;
    *)        echo "  src     : UNRECOGNIZED — neither pristine nor branded";;
esac
echo "  fiber   : $fiber_state"
echo

# ------------------------------------------------------------------- prompt ---
read -rp "What version value do you desire? (e.g. 4.9) " NEWVER || NEWVER=""
NEWVER="${NEWVER#v}"
NEWVER="${NEWVER//[[:space:]]/}"
if [[ -z "$NEWVER" ]]; then
    echo "ERROR: no version entered. Aborting." >&2
    exit 1
fi
if [[ ! "$NEWVER" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
    echo "ERROR: '$NEWVER' doesn't look like a version (expected e.g. 4.9 or 5.0.1)." >&2
    exit 1
fi
COMMIT_MSG="GR2fork v${NEWVER}: version bump + fiber.cpp combined-binary resume fix"

echo
echo "Setting GR2 fork version -> v$NEWVER"
echo

# ============================ step 1/2: src-gr2 (BUMP) =========================
cp -p -- "$GR2_FILE" "$GR2_FILE.bak"
sed -i -E \
    -e "s/(\"Build: )v[0-9]+(\.[0-9]+)*/\1v${NEWVER}/" \
    -e "/window_title = fmt::format/ s/(GR2FORK )v[0-9]+(\.[0-9]+)*/\1v${NEWVER}/g" \
    -- "$GR2_FILE"

# ============================ step 2/2: src (BRAND or BUMP) ====================
cp -p -- "$SRC_FILE" "$SRC_FILE.bak"
if grep -qF "$SRC_BRANDED" "$SRC_FILE"; then
    sed -i -E "s/(Junmin Lee GR2fork )v[0-9]+(\.[0-9]+)*/\1v${NEWVER}/g" -- "$SRC_FILE"
    src_action="bumped"
elif grep -qF "$SRC_ANCHOR" "$SRC_FILE"; then
    sed -i -e "s/fmt::format(\"shadPS4 /fmt::format(\"Junmin Lee GR2fork v${NEWVER} - shadPS4 /g" -- "$SRC_FILE"
    src_action="branded"
else
    src_action="skipped (no shadPS4 / GR2fork window_title lines found)"
fi

echo "=== src-gr2/emulator.cpp (Build log + GR2FORK window_title) ==="
grep -nE '"Build: v[0-9]|fmt::format\("Junmin Lee GR2FORK ' "$GR2_FILE" || echo "  (no matching lines!)"
echo
echo "=== src/emulator.cpp ($src_action) ==="
grep -nF 'fmt::format("Junmin Lee GR2fork v' "$SRC_FILE" || echo "  (no matching lines!)"
echo

# ============================ NEW step: replace fiber.cpp ======================
echo "=== replacing fiber.cpp ==="
[[ -f "$FIBER_TARGET" ]] && cp -p -- "$FIBER_TARGET" "$FIBER_TARGET.bak"
cp -- "$FIBER_REPLACEMENT" "$FIBER_TARGET"
echo "  from: $FIBER_REPLACEMENT"
echo "  to  : $FIBER_TARGET"
echo

# ============================ NEW step: commit + push =========================
changed="$(git -C "$REPO_ROOT" status --porcelain -- "$GR2_FILE" "$SRC_FILE" "$FIBER_TARGET")"
if [[ -z "$changed" ]]; then
    echo "Deploy files already match HEAD — nothing to commit or push."
    echo "Done. Backups: ${GR2_FILE}.bak, ${SRC_FILE}.bak, ${FIBER_TARGET}.bak"
    exit 0
fi

echo "Changes to push to $REMOTE/$BRANCH (vs HEAD):"
git -C "$REPO_ROOT" diff HEAD --stat -- "$GR2_FILE" "$SRC_FILE" "$FIBER_TARGET"
echo
echo "Commit message:"
echo "  $COMMIT_MSG"
echo
read -rp "Proceed with commit + push to $REMOTE/$BRANCH? [y/N] " ans || ans=""
case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "Stopped before commit/push. Your file changes are in place (uncommitted)."; exit 0;;
esac

git -C "$REPO_ROOT" add -- "$GR2_FILE" "$SRC_FILE" "$FIBER_TARGET"
git -C "$REPO_ROOT" commit -m "$COMMIT_MSG"
git -C "$REPO_ROOT" push "$REMOTE" "HEAD:$BRANCH"

echo
echo "Pushed to $REMOTE/$BRANCH ✓"
echo "Backups: ${GR2_FILE}.bak, ${SRC_FILE}.bak, ${FIBER_TARGET}.bak"
