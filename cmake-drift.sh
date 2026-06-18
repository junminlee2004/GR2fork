#!/usr/bin/env bash
#
# cmake-drift.sh - report how upstream's shadps4 build surface has drifted from the
# combined umbrella's core_main surface. core_main is the upstream mirror, so its
# find_package set / link libraries / compile definitions are supposed to track
# upstream's shadps4 target. This does NOT touch anything - it only reports, so you
# run it after fetching upstream and BEFORE a build breaks on a missing dependency.
#
# core_gr2 is intentionally OUT OF SCOPE: it's the gr2fork (v0.13.0-era), deliberately
# divergent from upstream, and only changes when you port accuracy work by hand.
#
# Usage: ./cmake-drift.sh            # fetch upstream, diff, report
#        ./cmake-drift.sh --no-fetch # use the upstream ref already fetched
#
set -euo pipefail

UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-main}"
UMBRELLA="${UMBRELLA:-CMakeLists.txt}"
COMBINE="${COMBINE:-cmake/Combine.cmake}"
CORE_NAME="${CORE_NAME:-main}"   # the upstream-mirror core whose surface tracks upstream
DO_FETCH=1
[ "${1:-}" = "--no-fetch" ] && DO_FETCH=0

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "not in a git work tree" >&2; exit 1; }
cd "$(git rev-parse --show-toplevel)"
[ -f "$UMBRELLA" ] || { echo "no '$UMBRELLA' here - run from the combined repo root" >&2; exit 1; }
[ -f "$COMBINE" ]  || { echo "no '$COMBINE' here" >&2; exit 1; }

if [ "$DO_FETCH" -eq 1 ]; then
    git fetch --quiet "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH" || { echo "fetch failed" >&2; exit 1; }
fi
UP_REF="$UPSTREAM_REMOTE/$UPSTREAM_BRANCH"

git show "$UP_REF:CMakeLists.txt" > /tmp/cd_upstream.txt 2>/dev/null || { echo "cannot read $UP_REF:CMakeLists.txt" >&2; exit 1; }
cp "$UMBRELLA" /tmp/cd_umbrella.txt
cp "$COMBINE"  /tmp/cd_combine.txt

UP_SHA="$(git rev-parse --short "$UP_REF")"

python3 - "$UP_SHA" "$CORE_NAME" <<'PY'
import re, sys

up_sha, core_name = sys.argv[1], sys.argv[2]
UP   = open('/tmp/cd_upstream.txt').read()
UMB  = open('/tmp/cd_umbrella.txt').read()
COMB = open('/tmp/cd_combine.txt').read()

KEYWORDS = {'PRIVATE','PUBLIC','INTERFACE'}

def calls(text, cmd, first_arg=None):
    """Yield the inner-argument string of each `cmd(...)` call (paren-balanced),
       optionally only those whose first token equals first_arg."""
    out = []
    i = 0
    pat = re.compile(r'\b' + re.escape(cmd) + r'\s*\(')
    for m in pat.finditer(text):
        j = m.end(); depth = 1
        while j < len(text) and depth:
            c = text[j]
            if c == '(': depth += 1
            elif c == ')': depth -= 1
            j += 1
        inner = text[m.end():j-1]
        toks = inner.split()
        if first_arg is None or (toks and toks[0] == first_arg):
            out.append(toks)
    return out

def find_packages(text):
    """name -> (version, mode) from find_package(name [ver] [MODULE|CONFIG] ...)."""
    fp = {}
    for toks in calls(text, 'find_package'):
        if not toks: continue
        name = toks[0]
        ver = next((t for t in toks[1:] if re.match(r'^\d', t)), '')
        mode = next((t for t in toks[1:] if t in ('MODULE','CONFIG')), '')
        fp[name] = (ver, mode)
    return fp

def link_libs(text, target):
    libs = set()
    for toks in calls(text, 'target_link_libraries', first_arg=target):
        for t in toks[1:]:
            if t in KEYWORDS: continue
            if t.startswith('"') or '/' in t: continue   # skip quoted abs paths (libusb path-form)
            if t.startswith('$'): continue                # skip genexes / vars
            libs.add(t)
    return libs

def compile_defs(text, target):
    defs = set()
    for toks in calls(text, 'target_compile_definitions', first_arg=target):
        for t in toks[1:]:
            if t in KEYWORDS: continue
            defs.add(re.sub(r'=.*$', '', t.strip('"')))   # FOO="x" / FOO=1 -> FOO
    return defs

def add_core_libs(umbrella_text, name):
    """LIBS list from the add_core(NAME <name> ... LIBS ...) invocation in the umbrella."""
    for toks in calls(umbrella_text, 'add_core'):
        d = {}
        key = None
        for t in toks:
            if t in ('NAME','SRC','EXCLUDES','LIBS','DEFINES'): key = t; d.setdefault(key, []); continue
            if key: d[key].append(t)
        if d.get('NAME', [''])[:1] == [name]:
            return set(d.get('LIBS', []))
    return set()

# ---- upstream shadps4 surface ----
up_fp   = find_packages(UP)
up_libs = link_libs(UP, 'shadps4')
up_defs = compile_defs(UP, 'shadps4')

# ---- umbrella core_<name> effective surface ----
umb_fp  = find_packages(UMB)
# the per-core target in Combine.cmake is core_${name}; its TLL/TCD use the literal token
comb_libs = link_libs(COMB, 'core_${name}')
comb_defs = compile_defs(COMB, 'core_${name}')
core_libs = comb_libs | add_core_libs(UMB, core_name)
core_defs = comb_defs

def hdr(s): print('\n=== ' + s + ' ===')

print(f"CMake drift: upstream/{up_sha} shadps4  vs  umbrella core_{core_name}")

hdr("find_package: present upstream, MISSING in umbrella")
miss = sorted(set(up_fp) - set(umb_fp))
print('\n'.join(f"  + {n:24} {up_fp[n][0]:10} {up_fp[n][1]}" for n in miss) or "  (none)")

hdr("find_package: version / mode MISMATCH")
mm = []
for n in sorted(set(up_fp) & set(umb_fp)):
    uv, um = up_fp[n]; mv, mmd = umb_fp[n]
    if uv != mv: mm.append(f"  ~ {n:24} version  umbrella {mv or '-'}  vs upstream {uv or '-'}")
    if um != mmd: mm.append(f"  ~ {n:24} mode     umbrella {mmd or '-'}  vs upstream {um or '-'}")
print('\n'.join(mm) or "  (none)")

hdr(f"link libraries: in upstream shadps4, NOT in core_{core_name}'s effective set")
print("  (review each: add to core_%s LIBS or the shared base set in Combine.cmake," % core_name)
print("   UNLESS core_%s's source genuinely doesn't use it)" % core_name)
lmiss = sorted(up_libs - core_libs)
print('\n'.join(f"  + {l}" for l in lmiss) or "  (none)")

hdr(f"compile definitions: in upstream shadps4, NOT in core_{core_name}'s set")
dmiss = sorted(up_defs - core_defs)
print('\n'.join(f"  + {d}" for d in dmiss) or "  (none)")

hdr("informational: in umbrella/core, not in upstream shadps4 (likely intentional)")
extra_fp = sorted(set(umb_fp) - set(up_fp))
extra_lb = sorted(core_libs - up_libs)
if extra_fp: print("  find_package only here: " + ", ".join(extra_fp))
if extra_lb: print("  link libs only here:    " + ", ".join(extra_lb))
if not extra_fp and not extra_lb: print("  (none)")
PY