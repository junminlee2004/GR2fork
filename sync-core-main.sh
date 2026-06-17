#!/usr/bin/env bash
#
# sync-core-main.sh - mirror the combined repo's core_main tree (src/) to upstream.
#
# core_main is a PURE upstream mirror, so it is REPLACED, never merged. That sidesteps
# the spurious cherry-pick/merge conflicts caused by line-level drift against the umbrella
# and your real fork. This touches ONLY src/ - never src-gr2/ (your fork), the umbrella
# CMakeLists.txt, dispatcher/, cmake/, or externals/.
#
# ASSUMPTION it enforces: src/ has NO local edits. Any committed change to core_main is
# discarded by the mirror (by design - core_main is upstream verbatim). core_main
# customizations don't belong here; they go in src-gr2/ or the umbrella.
#
set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/shadps4-emu/shadPS4}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
SRC_DIR="${SRC_DIR:-src}"

DRY_RUN=0
DO_COMMIT=1
FORCE=0

usage() {
    cat <<'EOF'
Usage: sync-core-main.sh [options]
  (no options)     fetch upstream, show the delta, mirror src/, and commit
  -n, --dry-run    show exactly what would change; touch nothing
      --no-commit  mirror + stage src/, but leave the commit to you
  -f, --force      if src/ has uncommitted changes, discard them and replace
                   anyway, without the interactive prompt (for non-interactive use)
  -h, --help       this help

Config via env: UPSTREAM_REMOTE (upstream), UPSTREAM_URL, UPSTREAM_BRANCH (main), SRC_DIR (src)
EOF
}

for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        --no-commit)  DO_COMMIT=0 ;;
        -f|--force)   FORCE=1 ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

# colors only when stdout is a terminal
if [ -t 1 ]; then
    C_OK=$'\033[1;36m'; C_WARN=$'\033[1;33m'; C_ERR=$'\033[1;31m'; C_OFF=$'\033[0m'
else
    C_OK=''; C_WARN=''; C_ERR=''; C_OFF=''
fi
say()  { printf '%s::%s %s\n' "$C_OK" "$C_OFF" "$*"; }
warn() { printf '%s!!%s %s\n' "$C_WARN" "$C_OFF" "$*" >&2; }
die()  { printf '%sxx%s %s\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

# --- must be inside the combined repo ---
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not inside a git work tree."
cd "$(git rev-parse --show-toplevel)"

[ -d "$SRC_DIR" ] || die "no '$SRC_DIR/' here - run this from the combined repo root."
[ -d "src-gr2" ] || warn "no 'src-gr2/' found - this may not be the combined repo (only '$SRC_DIR/' will be touched)."

# --- src/ must be clean; we are about to replace it wholesale (tracked, staged, OR untracked) ---
SRC_DIRTY=0
if ! git diff --quiet -- "$SRC_DIR" \
   || ! git diff --cached --quiet -- "$SRC_DIR" \
   || [ -n "$(git ls-files --others --exclude-standard -- "$SRC_DIR")" ]; then
    SRC_DIRTY=1
fi

if [ "$SRC_DIRTY" -eq 1 ]; then
    warn "uncommitted changes under '$SRC_DIR/':"
    git status --short -- "$SRC_DIR" | sed 's/^/    /' >&2
    proceed=0
    if [ "$FORCE" -eq 1 ]; then
        warn "--force: discarding the above and replacing '$SRC_DIR/' anyway."
        proceed=1
    elif [ "$DRY_RUN" -eq 1 ]; then
        warn "(dry run: a real run would discard the above. Commit/stash to keep them, or pass --force.)"
    elif [ -t 0 ]; then
        printf '%s>>%s Discard ALL the above and force-replace %s/ with upstream? [y/N] ' "$C_WARN" "$C_OFF" "$SRC_DIR" >&2
        read -r reply || reply=""
        case "$reply" in
            [yY]|[yY][eE][sS]) proceed=1 ;;
            *) die "aborted - nothing changed." ;;
        esac
    else
        die "uncommitted changes under '$SRC_DIR/' and no terminal to confirm. Re-run with --force to discard them, or commit/stash first."
    fi

    if [ "$proceed" -eq 1 ]; then
        say "discarding local changes under '$SRC_DIR/' ..."
        git reset -q -- "$SRC_DIR"          # unstage
        git checkout -q -- "$SRC_DIR"       # revert tracked files to HEAD
        git clean -qfd -- "$SRC_DIR"        # remove untracked files/dirs
        SRC_DIRTY=0
    fi
fi

# --- ensure the upstream remote exists ---
if ! git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
    say "adding remote '$UPSTREAM_REMOTE' -> $UPSTREAM_URL"
    [ "$DRY_RUN" -eq 1 ] || git remote add "$UPSTREAM_REMOTE" "$UPSTREAM_URL"
fi

say "fetching $UPSTREAM_REMOTE/$UPSTREAM_BRANCH ..."
git fetch --quiet "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH" || die "fetch failed (network / remote?)."
UP_REF="$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"

# --- compute the delta (no rename detection -> clean A/M/D rows) ---
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
UP_SHA="$(git rev-parse --short "$UP_REF")"
MB="$(git merge-base HEAD "$UP_REF")"

mapfile -t CHANGED < <(git diff --no-renames --name-status HEAD "$UP_REF" -- "$SRC_DIR/")
if [ "${#CHANGED[@]}" -eq 0 ]; then
    say "core_main ('$SRC_DIR/') is already identical to $UP_REF - nothing to do."
    exit 0
fi

say "branch:      $BRANCH"
say "upstream:    $UP_REF @ $UP_SHA"
say "merge-base:  $(git rev-parse --short "$MB")"
echo
say "upstream commits touching '$SRC_DIR/' since the merge-base:"
git log --oneline --no-decorate "$MB..$UP_REF" -- "$SRC_DIR/" | sed 's/^/    /'
echo
say "files in '$SRC_DIR/' that will change (${#CHANGED[@]}):"
printf '    %s\n' "${CHANGED[@]}"

# --- glob-coverage guard: NEW source files the add_core glob (*.cpp/*.s/*.S) will NOT
#     compile into core_main (e.g. .c/.cc/.cxx/.cu/.mm/.m). Headers are fine. ---
MISSED=()
for entry in "${CHANGED[@]}"; do
    status="${entry%%$'\t'*}"
    path="${entry#*$'\t'}"
    [ "$status" = "A" ] || continue
    case "$path" in
        *.cpp|*.s|*.S) : ;;
        *.c|*.cc|*.cxx|*.cu|*.mm|*.m) MISSED+=("$path") ;;
        *) : ;;
    esac
done

if [ "${#MISSED[@]}" -gt 0 ]; then
    echo
    warn "NEW upstream source files NOT matched by the add_core glob (*.cpp/*.s/*.S):"
    printf '    %s\n' "${MISSED[@]}" >&2
    warn "If any are needed on your platform, add them to add_core's source set or EXCLUDES."
    warn "(macOS-only .mm/.m are normally safe to ignore on Linux/Windows.)"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    say "dry run - nothing changed."
    exit 0
fi

# --- the mirror: replace src/ wholesale with upstream's, INCLUDING deletions ---
echo
say "mirroring '$SRC_DIR/' -> $UP_REF (replace, not merge) ..."
rm -rf -- "$SRC_DIR"
git checkout "$UP_REF" -- "$SRC_DIR"
git add -A -- "$SRC_DIR"

# sanity: the staged tree must now match upstream exactly
if ! git diff --cached --quiet "$UP_REF" -- "$SRC_DIR"; then
    die "post-mirror mismatch under '$SRC_DIR/' - aborting before commit. Inspect: git diff --cached $UP_REF -- $SRC_DIR"
fi
say "verified: staged '$SRC_DIR/' is byte-identical to $UP_REF."

if [ "$DO_COMMIT" -eq 0 ]; then
    say "staged but not committed (--no-commit). Review with 'git diff --cached', then commit."
    exit 0
fi

MSG_BODY="$(git log --oneline --no-decorate "$MB..$UP_REF" -- "$SRC_DIR/")"
git commit --quiet \
    -m "core_main: mirror $SRC_DIR/ to $UP_REF @ $UP_SHA" \
    -m "Replace-sync of the upstream-mirror tree (no merge). Upstream commits touching $SRC_DIR/:" \
    -m "$MSG_BODY"
say "committed: $(git rev-parse --short HEAD)"
echo
say "next: rebuild -> rm -rf build_opt && <your znver4 configure> && cmake --build build_opt -j\$(nproc)"
