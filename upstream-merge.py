#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""
upstream-merge.py — intelligent upstream sync for the combined shadPS4 repo.

Brings the upstream-mirror half of this repo (src/ -> core_main, plus the SHARED
cmake/, externals/ and .gitmodules) up to a given shadPS4 upstream commit while
keeping the COMBINED build buildable, without ever line-merging into the
refactored umbrella CMakeLists.txt.

What one run does, in order:

  1. PREFLIGHT   fetch upstream, resolve the target commit, require the owned
                 paths (src/, cmake/, externals/, .gitmodules) to be clean.
                 src-gr2/ is never touched and may be dirty.
  2. PATCHES     bootstrap/load the carried local patches in upstream-patches/.
                 These are the deliberate deviations of the mirror from upstream
                 (today: the fiber.cpp combined-binary resume fix + the GR2fork
                 window-title branding). They are re-derived from the worktree
                 the first time (diff vs the recorded upstream base).
  3. MIRROR      replace-sync src/ to the target commit; sync upstream's cmake/
                 files (the umbrella-owned ones are preserved); sync externals/
                 (submodule adds/bumps/removals + CMakeLists.txt), preserving
                 the gr2-only submodules; merge .gitmodules (upstream verbatim +
                 local-only sections appended); git submodule sync/update.
  4. REAPPLY     re-apply each carried patch with a git 3-way merge onto the new
                 upstream code. The branding patch falls back to a regex
                 transform if the surrounding code moved. A pure-addition patch
                 that conflicts only because upstream adopted its lines verbatim
                 while the surrounding context moved is detected and dropped as
                 absorbed instead of failing; any other conflict blocks the
                 commit and prints a per-patch recovery recipe. Patches are then
                 regenerated against the NEW base so they never go stale, and
                 fiber_core_main_replacment.cpp (set-version.sh's companion) is
                 refreshed to the newly patched fiber.cpp.
  5. SURFACE     parse the TARGET commit's root CMakeLists.txt with a small
                 condition-aware CMake evaluator (Linux/x86_64/no-tests/no-Qt
                 profile) and regenerate cmake/CoreMainSurface.cmake:
                   - the find_package mirror block
                   - CORE_MAIN_EXCLUDES  (on-disk .cpp upstream does NOT compile)
                   - CORE_MAIN_LIBS      (upstream's link set minus Combine.cmake's
                                          shared base set)
                   - CORE_MAIN_DEFINES / CORE_MAIN_EXTRA_SOURCES
                   - CORE_MAIN_CMRC_FILES (upstream's explicit embedded resources)
                 The umbrella includes this file, so upstream build-surface
                 changes can never conflict with the umbrella again.
  6. CHECKS      structural safety nets: unknown build-time codegen (scm_rev and
                 shadnet/protoc are understood), imgui_config.h ODR identity
                 between the two trees, glob-missed source extensions, Windows
                 link-surface drift (report-only), shared-base drift.
  7. VERIFY      cmake configure smoke-test in a scratch dir (or --build for the
                 full production build into build_opt/).
  8. COMMIT      one commit referencing the absorbed upstream SHA range, then a
                 history-only `git merge -s ours <target>` so merge-base tracks
                 the sync (content is already identical; no conflicts possible).

Usage:
  ./upstream-merge.py                    # sync to upstream/main tip
  ./upstream-merge.py -c <sha>           # sync to a specific upstream commit
  ./upstream-merge.py -n                 # dry run: report only, touch nothing
  ./upstream-merge.py --build            # also run the full znver4 build
  ./upstream-merge.py --selftest <ref>   # parse a commit's CMakeLists and print
                                         # the computed surface (no changes)

Recovery if a run is interrupted or fails:
  git restore --source=HEAD --staged --worktree -- src cmake externals .gitmodules
  git submodule update --init --recursive
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- config

DEFAULT_ROOT = "/home/jlee/Downloads/safe/gr2fork-new"
UPSTREAM_REMOTE = "upstream"
UPSTREAM_URL = "https://github.com/shadps4-emu/shadPS4"
UPSTREAM_BRANCH = "main"

SRC_DIR = "src"                      # the upstream mirror -> core_main
PATCH_DIR = "upstream-patches"       # carried local deviations of the mirror
STATE_FILE = f"{PATCH_DIR}/UPSTREAM_BASE"
PIN_FILE = f"{PATCH_DIR}/PINNED_SUBMODULES"   # optional: submodule paths to hold back
SURFACE_FILE = "cmake/CoreMainSurface.cmake"  # generated build surface
FIBER_COMPANION = "fiber_core_main_replacment.cpp"  # set-version.sh's replacement file
FIBER_SRC = "src/core/libraries/fiber/fiber.cpp"
BRAND_SRC = "src/emulator.cpp"

# cmake/ files owned by the umbrella (never overwritten from upstream)
UMBRELLA_CMAKE = {
    "cmake/Combine.cmake",
    "cmake/export.map",
    "cmake/embed_cores.S.in",
    "cmake/embed_cores_win.S.in",
    "cmake/CoreMainSurface.cmake",
}

# submodules that exist only in the combined repo (never removed on sync)
KEEP_SUBMODULES = {"externals/sdl3_mixer"}

# libs Combine.cmake regenerates per-core under its own names — upstream's
# spellings of them are dropped from the computed LIBS
INTERNAL_LIBS = {"res::embedded", "embedded-resources"}

# target_compile_definitions(shadps4 ...) already handled by Combine.cmake/umbrella
KNOWN_TCD = {"IMGUI_USER_CONFIG", "ENABLE_DISCORD_RPC", "ENABLE_USERFAULTFD"}

# global add_compile_definitions already handled by Combine.cmake or the umbrella
KNOWN_GLOBAL_DEFINES = {
    "BOOST_ASIO_STANDALONE",
    "_CRT_SECURE_NO_WARNINGS", "_CRT_NONSTDC_NO_DEPRECATE", "_SCL_SECURE_NO_WARNINGS",
    "NOMINMAX", "WIN32_LEAN_AND_MEAN", "_TIMESPEC_DEFINED",
    "NTDDI_VERSION", "_WIN32_WINNT", "WINVER",
}

# the build toolchain lives in a distrobox (the Silverblue host has no cmake/clang);
# "host" disables the wrapper, --container overrides the name
BUILD_CONTAINER = "shadps4-dev"

# the production build (kept in sync with the user's canonical invocation)
BUILD_DIR = "build_opt"
CONFIGURE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER=clang",
    "-DCMAKE_CXX_COMPILER=clang++",
    "-DCMAKE_C_FLAGS=-march=znver4 -mtune=znver4 -O3 -flto=thin -DNDEBUG",
    "-DCMAKE_CXX_FLAGS=-march=znver4 -mtune=znver4 -O3 -flto=thin -DNDEBUG",
    "-DCMAKE_EXE_LINKER_FLAGS=-flto=thin -fuse-ld=lld",
    "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
    "-DCMAKE_DISABLE_FIND_PACKAGE_SDL3=ON",
    "-DCOMBINED_EMBED_CORES=ON",
]

# ------------------------------------------------------------------------ reporting

try:  # progress lines should stream when piped/backgrounded
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

USE_COLOR = sys.stdout.isatty()


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def say(msg):
    print(_c("1;36", "::") + f" {msg}")


def warn(msg):
    print(_c("1;33", "!!") + f" {msg}", file=sys.stderr)


def die(msg, extra=None):
    print(_c("1;31", "xx") + f" {msg}", file=sys.stderr)
    if extra:
        print(extra, file=sys.stderr)
    sys.exit(1)


class Report:
    """Collects everything the human needs to see at the end."""

    def __init__(self):
        self.sections = {}   # title -> [lines]
        self.problems = []   # blocking issues found (non-empty => exit 1)

    def add(self, section, line):
        self.sections.setdefault(section, []).append(line)

    def problem(self, section, line):
        self.add(section, _c("1;31", "PROBLEM: ") + line)
        self.problems.append(f"{section}: {line}")

    def dump(self):
        for title, lines in self.sections.items():
            print(f"\n=== {title} ===")
            for ln in lines:
                print(f"  {ln}")


REPORT = Report()

# ------------------------------------------------------------------------------ git


def run(args, check=True, capture=True, cwd=None, allow_fail=False):
    try:
        r = subprocess.run(args, cwd=cwd, text=True,
                           stdout=subprocess.PIPE if capture else None,
                           stderr=subprocess.PIPE if capture else None)
    except FileNotFoundError:
        die(f"'{args[0]}' not found on PATH (host is toolchain-less? see --container)")
    if check and r.returncode != 0 and not allow_fail:
        die(f"command failed ({r.returncode}): {' '.join(args)}",
            (r.stderr or "").strip())
    return r


def git(*args, **kw):
    return run(["git", *args], **kw)


_toolchain = None


def toolchain(container_opt):
    """argv prefix that provides cmake/clang: [] on a host with a toolchain,
    else `distrobox enter <name> --` (build box). Cached after first resolve."""
    global _toolchain
    if _toolchain is None:
        if container_opt == "host" or shutil.which("cmake"):
            _toolchain = []
        else:
            name = container_opt or BUILD_CONTAINER
            probe = subprocess.run(["distrobox", "enter", name, "--", "true"],
                                   capture_output=True)
            if probe.returncode == 0:
                _toolchain = ["distrobox", "enter", name, "--"]
                say(f"toolchain: distrobox container '{name}'")
            else:
                die(f"no cmake on the host and distrobox '{name}' is unavailable — "
                    f"pass --container <name> (or --container host).",
                    probe.stderr.decode(errors="replace")[:300])
    return _toolchain


def git_out(*args, **kw):
    return git(*args, **kw).stdout.strip()


def ls_tree_paths(ref, path):
    """{path: (mode, type)} for everything under `path` at `ref`."""
    out = git_out("ls-tree", "-r", "--full-tree", ref, "--", path)
    entries = {}
    for line in out.splitlines():
        meta, p = line.split("\t", 1)
        mode, typ, _ = meta.split(None, 2)
        entries[p] = (mode, typ)
    return entries


def tree_file_set(ref):
    out = git_out("ls-tree", "-r", "--name-only", "--full-tree", ref)
    return set(out.splitlines())


# =========================================================================== CMake
# A small condition-aware CMake evaluator: enough of the language to extract the
# build surface of upstream's root CMakeLists.txt under a fixed platform profile.
# Anything it cannot understand degrades to a WARNING, and anything it cannot
# understand that AFFECTS the shadps4 target becomes a blocking PROBLEM.

FALSE_CONSTANTS = {"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"}


def truthy(v):
    u = v.upper()
    return u not in FALSE_CONSTANTS and not u.endswith("-NOTFOUND")


def strip_comments(text):
    out = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "#":
            if text.startswith("#[[", i) or re.match(r"#\[=*\[", text[i:i + 8] or ""):
                m = re.match(r"#\[(=*)\[", text[i:])
                closer = "]" + (m.group(1) if m else "") + "]"
                j = text.find(closer, i)
                if j == -1:
                    break
                i = j + len(closer)
                continue
            j = text.find("\n", i)
            if j == -1:
                break
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


CMD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(")


def scan_commands(text):
    """Yield (name_lower, raw_args) for every top-of-statement command call."""
    cmds = []
    i, n = 0, len(text)
    while True:
        m = CMD_RE.search(text, i)
        if not m:
            break
        line_start = text.rfind("\n", 0, m.start()) + 1
        if text[line_start:m.start()].strip():
            i = m.end()
            continue
        depth, j, in_str = 1, m.end(), False
        while j < n and depth:
            ch = text[j]
            if in_str:
                if ch == "\\":
                    j += 2
                    continue
                if ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        cmds.append((m.group(1).lower(), text[m.end():j - 1]))
        i = j
    return cmds


def split_args(raw):
    """-> [(text, quoted)] ; parens are standalone tokens (for if() conditions)."""
    toks, cur, quoted = [], [], False
    i, n = 0, len(raw)
    in_str = False

    def flush():
        nonlocal cur, quoted
        if cur or quoted:
            toks.append(("".join(cur), quoted))
        cur, quoted = [], False

    while i < n:
        ch = raw[i]
        if in_str:
            if ch == "\\" and i + 1 < n:
                cur.append(raw[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
                i += 1
                continue
            cur.append(ch)
            i += 1
            continue
        if ch == '"':
            in_str = True
            quoted = True
            i += 1
            continue
        if ch in " \t\r\n":
            flush()
            i += 1
            continue
        if ch in "()":
            flush()
            toks.append((ch, False))
            i += 1
            continue
        cur.append(ch)
        i += 1
    flush()
    return toks


# ---- statement tree ----

@dataclass
class Cmd:
    name: str
    raw: str


@dataclass
class IfBlock:
    branches: list = field(default_factory=list)  # [(cond_raw_or_None, [stmts])]


@dataclass
class ForeachBlock:
    raw: str = ""
    body: list = field(default_factory=list)


BLOCK_OPEN = {"if": "endif", "foreach": "endforeach", "function": "endfunction",
              "macro": "endmacro", "while": "endwhile", "block": "endblock"}


def build_tree(cmds):
    root, stack = [], []
    cur = root
    for name, raw in cmds:
        if name == "if":
            blk = IfBlock(branches=[(raw, [])])
            cur.append(blk)
            stack.append((cur, blk, "if"))
            cur = blk.branches[-1][1]
        elif name == "elseif":
            parent, blk, kind = stack[-1]
            assert kind == "if"
            blk.branches.append((raw, []))
            cur = blk.branches[-1][1]
        elif name == "else":
            parent, blk, kind = stack[-1]
            assert kind == "if"
            blk.branches.append((None, []))
            cur = blk.branches[-1][1]
        elif name == "foreach":
            blk = ForeachBlock(raw=raw)
            cur.append(blk)
            stack.append((cur, blk, "foreach"))
            cur = blk.body
        elif name == "while":
            blk = ForeachBlock(raw="__WHILE__")  # evaluated zero times, warned
            cur.append(blk)
            stack.append((cur, blk, "while"))
            cur = blk.body
        elif name in ("function", "macro", "block"):
            blk = ForeachBlock(raw="__SKIP__")   # definitions: never evaluated
            cur.append(blk)
            stack.append((cur, blk, name))
            cur = blk.body
        elif name in ("endif", "endforeach", "endfunction", "endmacro",
                      "endwhile", "endblock"):
            parent, blk, kind = stack.pop()
            cur = parent
        else:
            cur.append(Cmd(name, raw))
    if stack:
        raise ValueError("unbalanced blocks in CMakeLists")
    return root


# ---- evaluator ----

class CMakeEval:
    def __init__(self, profile, tree_files, warn_sink):
        self.vars = dict(profile)
        self.tree_files = tree_files      # full repo file list at the target commit
        self.tree_dirs = {os.path.dirname(p) for p in tree_files}
        self.warn = warn_sink
        self.warned = set()
        self.targets = set()
        # collected surface
        self.find_packages = []           # (name, arg_tokens) in order
        self.sources = []                 # expanded tokens for target shadps4
        self.libs = []                    # TLL(shadps4)
        self.tcd = []                     # TCD(shadps4)
        self.global_defs = []             # add_compile_definitions
        self.includes = []                # TID(shadps4)
        self.custom_commands = []         # dicts
        self.cmrc = None                  # dict for embedded-resources
        self.subdirs = []
        self.configure_files = []
        self.sfp = []                     # set_source_files_properties records
        self.compile_opts = []            # add_compile_options
        self.link_opts = []               # add_link_options
        self.structural = []              # things that need a human/tool upgrade

    # -- warnings --
    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            self.warn(msg)

    # -- expansion --
    VAR_RE = re.compile(r"\$\{([A-Za-z0-9_./+\-]+)\}")
    ENV_RE = re.compile(r"\$ENV\{[^}]*\}")
    CACHE_RE = re.compile(r"\$CACHE\{[^}]*\}")

    def expand(self, s):
        for _ in range(10):
            s2 = self.ENV_RE.sub("", s)
            s2 = self.CACHE_RE.sub("", s2)
            s2 = self.VAR_RE.sub(lambda m: self.vars.get(m.group(1), ""), s2)
            if s2 == s:
                return s2
            s = s2
        return s

    def eval_args(self, raw, keep_parens=False):
        """expand + list-split unquoted tokens -> [(text, quoted)]"""
        out = []
        for text, quoted in split_args(raw):
            if text in ("(", ")") and not quoted:
                if keep_parens:
                    out.append((text, quoted))
                continue
            ex = self.expand(text)
            if quoted:
                out.append((ex, True))
            else:
                for part in ex.split(";"):
                    if part != "":
                        out.append((part, False))
        return out

    def toks(self, raw):
        return [t for t, _ in self.eval_args(raw)]

    # -- conditions --
    def eval_cond(self, raw):
        toks = self.eval_args(raw, keep_parens=True)
        pos = 0

        def peek():
            return toks[pos] if pos < len(toks) else (None, False)

        def take():
            nonlocal pos
            t = toks[pos]
            pos += 1
            return t

        def atom_value(tok, quoted):
            if not quoted and tok in self.vars:
                return truthy(self.vars[tok])
            if not quoted:
                u = tok.upper()
                if u in ("ON", "TRUE", "YES", "Y", "1"):
                    return True
                if u in FALSE_CONSTANTS or u.endswith("-NOTFOUND"):
                    return False
                if re.fullmatch(r"\d+", tok):
                    return tok != "0"
                return False  # undefined variable reference
            return truthy(tok)

        def resolve(tok, quoted):
            """value used by binary comparisons: auto-deref unquoted var names"""
            if not quoted and tok in self.vars:
                return self.vars[tok]
            return tok

        def parse_primary():
            nonlocal pos
            t, q = peek()
            if t == "(" and not q:
                take()
                v = parse_or()
                nt, _ = peek()
                if nt == ")":
                    take()
                return v
            if t == "NOT" and not q:
                take()
                return not parse_primary()
            for kw in ("DEFINED", "TARGET", "COMMAND", "POLICY", "EXISTS", "TEST"):
                if t == kw and not q:
                    take()
                    arg, aq = take()
                    if kw == "DEFINED":
                        return arg in self.vars
                    if kw == "TARGET":
                        return arg in self.targets
                    if kw == "EXISTS":
                        p = arg
                        if p.startswith("@ROOT@/"):
                            rel = p[len("@ROOT@/"):]
                            return rel in self.tree_files or rel in self.tree_dirs
                        self.warn_once(f"exists:{p}", f"EXISTS on unresolvable path treated FALSE: {p}")
                        return False
                    self.warn_once(f"cond:{kw}", f"condition '{kw}' treated FALSE")
                    return False
            tok, tq = take()
            # binary operator?
            op, oq = peek()
            BINOPS = {"STREQUAL", "MATCHES", "EQUAL", "LESS", "GREATER",
                      "STRLESS", "STRGREATER", "VERSION_LESS", "VERSION_GREATER",
                      "VERSION_EQUAL", "VERSION_GREATER_EQUAL", "VERSION_LESS_EQUAL",
                      "LESS_EQUAL", "GREATER_EQUAL", "IN_LIST", "PATH_EQUAL"}
            if op in BINOPS and not oq:
                take()
                rhs, rq = take()
                a, b = resolve(tok, tq), resolve(rhs, rq)
                if op == "STREQUAL":
                    return a == b
                if op == "MATCHES":
                    try:
                        return re.search(b, a) is not None
                    except re.error:
                        self.warn_once(f"re:{b}", f"unsupported regex in MATCHES: {b!r} -> FALSE")
                        return False
                if op == "EQUAL":
                    try:
                        return int(a) == int(b)
                    except ValueError:
                        return False
                if op == "IN_LIST":
                    return a in self.vars.get(rhs, "").split(";")
                self.warn_once(f"op:{op}", f"comparison '{op}' treated FALSE")
                return False
            return atom_value(tok, tq)

        def parse_and():
            v = parse_primary()
            while True:
                t, q = peek()
                if t == "AND" and not q:
                    take()
                    r = parse_primary()
                    v = v and r
                else:
                    return v

        def parse_or():
            v = parse_and()
            while True:
                t, q = peek()
                if t == "OR" and not q:
                    take()
                    r = parse_and()
                    v = v or r
                else:
                    return v

        try:
            return parse_or()
        except Exception as e:  # noqa
            self.warn_once(f"cond:{raw}", f"could not evaluate condition ({e}): if({raw.strip()}) -> FALSE")
            return False

    # -- statements --
    def eval_stmts(self, stmts):
        for st in stmts:
            if isinstance(st, IfBlock):
                for cond, body in st.branches:
                    if cond is None or self.eval_cond(cond):
                        self.eval_stmts(body)
                        break
            elif isinstance(st, ForeachBlock):
                if st.raw == "__SKIP__":
                    continue
                if st.raw == "__WHILE__":
                    self.warn_once("while", "while() blocks are skipped")
                    continue
                self.eval_foreach(st)
            else:
                self.eval_cmd(st)

    def eval_foreach(self, blk):
        toks = self.toks(blk.raw)
        if not toks:
            return
        var = toks[0]
        rest = toks[1:]
        items = []
        if rest[:2] == ["IN", "LISTS"]:
            for name in rest[2:]:
                if name == "ITEMS":
                    continue
                items += [x for x in self.vars.get(name, "").split(";") if x != ""]
        elif rest[:2] == ["IN", "ITEMS"]:
            items = rest[2:]
        elif rest[:1] == ["RANGE"]:
            self.warn_once("foreach-range", "foreach(RANGE) skipped")
            return
        else:
            items = rest
        saved = self.vars.get(var)
        for it in items:
            self.vars[var] = it
            self.eval_stmts(blk.body)
        if saved is None:
            self.vars.pop(var, None)
        else:
            self.vars[var] = saved

    def eval_cmd(self, cmd):
        n = cmd.name
        if n == "set":
            toks = self.eval_args(cmd.raw)
            if not toks:
                return
            name = toks[0][0]
            vals = []
            for t, q in toks[1:]:
                if t in ("CACHE", "PARENT_SCOPE") and not q:
                    break
                vals.append(t)
            if vals:
                self.vars[name] = ";".join(vals)
            else:
                self.vars.pop(name, None)
        elif n == "list":
            toks = self.toks(cmd.raw)
            if len(toks) >= 2:
                op, name = toks[0], toks[1]
                cur = [x for x in self.vars.get(name, "").split(";") if x != ""]
                if op == "APPEND":
                    cur += toks[2:]
                    self.vars[name] = ";".join(cur)
                elif op == "PREPEND":
                    self.vars[name] = ";".join(toks[2:] + cur)
                elif op == "REMOVE_ITEM":
                    cur = [x for x in cur if x not in toks[2:]]
                    self.vars[name] = ";".join(cur)
                elif op == "POP_BACK":
                    if cur:
                        cur.pop()
                    self.vars[name] = ";".join(cur)
                # other list() ops don't matter for surface extraction
        elif n == "option":
            toks = self.toks(cmd.raw)
            if toks and toks[0] not in self.vars:
                default = toks[2] if len(toks) > 2 else "OFF"
                self.vars[toks[0]] = "1" if truthy(default) else ""
        elif n == "cmake_dependent_option":
            toks = self.toks(cmd.raw)
            if toks and toks[0] not in self.vars:
                # approximation: depends-condition unmet on our profile -> force value
                force = toks[4] if len(toks) > 4 else "OFF"
                self.vars[toks[0]] = "1" if truthy(force) else ""
        elif n == "string":
            toks = self.toks(cmd.raw)
            if len(toks) >= 3 and toks[0] in ("TOLOWER", "TOUPPER"):
                src, dst = toks[1], toks[2]
                self.vars[dst] = src.lower() if toks[0] == "TOLOWER" else src.upper()
            elif len(toks) >= 4 and toks[0] == "REPLACE":
                self.vars[toks[3]] = ";".join(t.replace(toks[1], toks[2]) for t in toks[4:]) if len(toks) > 4 else ""
            elif len(toks) >= 2:
                # unknown string() op: define output var empty if it looks like one
                self.vars.setdefault(toks[-1], "")
        elif n in ("execute_process", "cmake_host_system_information"):
            toks = self.toks(cmd.raw)
            for i, t in enumerate(toks):
                if t in ("OUTPUT_VARIABLE", "RESULT_VARIABLE", "RESULT", "ERROR_VARIABLE") and i + 1 < len(toks):
                    self.vars.setdefault(toks[i + 1], "")
        elif n == "find_package":
            toks = self.toks(cmd.raw)
            if toks:
                self.find_packages.append(toks)
        elif n in ("add_executable", "add_library", "qt_add_executable"):
            toks = self.toks(cmd.raw)
            if not toks:
                return
            tgt = toks[0]
            self.targets.add(tgt)
            if n == "qt_add_executable":
                self.structural.append("qt_add_executable() appeared upstream — Qt is back?!")
            if tgt == "shadps4":
                for t in toks[1:]:
                    if t in ("WIN32", "MACOSX_BUNDLE", "EXCLUDE_FROM_ALL",
                             "STATIC", "SHARED", "MODULE", "OBJECT", "INTERFACE", "ALIAS"):
                        continue
                    self.sources.append(t)
            elif n != "add_library" or "IMPORTED" not in cmd.raw:
                if tgt not in ("shadps4",):
                    self.structural_target(tgt)
        elif n == "add_custom_target":
            toks = self.toks(cmd.raw)
            if toks:
                self.targets.add(toks[0])
        elif n == "target_sources":
            toks = self.toks(cmd.raw)
            if toks and toks[0] == "shadps4":
                for t in toks[1:]:
                    if t not in ("PRIVATE", "PUBLIC", "INTERFACE"):
                        self.sources.append(t)
        elif n == "target_link_libraries":
            toks = self.toks(cmd.raw)
            if toks and toks[0] == "shadps4":
                for t in toks[1:]:
                    if t not in ("PRIVATE", "PUBLIC", "INTERFACE"):
                        self.libs.append(t)
        elif n == "target_compile_definitions":
            toks = self.toks(cmd.raw)
            if toks and toks[0] == "shadps4":
                for t in toks[1:]:
                    if t not in ("PRIVATE", "PUBLIC", "INTERFACE"):
                        self.tcd.append(t)
        elif n == "add_compile_definitions":
            self.global_defs += self.toks(cmd.raw)
        elif n == "add_compile_options":
            self.compile_opts += self.toks(cmd.raw)
        elif n == "add_link_options":
            self.link_opts += self.toks(cmd.raw)
        elif n == "target_include_directories":
            toks = self.toks(cmd.raw)
            if toks and toks[0] == "shadps4":
                for t in toks[1:]:
                    if t not in ("PRIVATE", "PUBLIC", "INTERFACE", "SYSTEM", "BEFORE"):
                        self.includes.append(t)
        elif n == "add_custom_command":
            toks = self.toks(cmd.raw)
            rec = {"outputs": [], "command": [], "depends": []}
            mode = None
            for t in toks:
                if t in ("OUTPUT", "COMMAND", "DEPENDS", "COMMENT", "VERBATIM",
                         "MAIN_DEPENDENCY", "WORKING_DIRECTORY", "BYPRODUCTS"):
                    mode = t
                    continue
                if mode == "OUTPUT":
                    rec["outputs"].append(t)
                elif mode == "COMMAND":
                    rec["command"].append(t)
                elif mode == "DEPENDS":
                    rec["depends"].append(t)
            self.custom_commands.append(rec)
        elif n == "cmrc_add_resource_library":
            toks = self.toks(cmd.raw)
            rec = {"name": toks[0] if toks else "", "alias": "", "namespace": "",
                   "files": [], "extra_kw": []}
            i = 1
            while i < len(toks):
                t = toks[i]
                if t == "ALIAS":
                    rec["alias"] = toks[i + 1]
                    i += 2
                elif t == "NAMESPACE":
                    rec["namespace"] = toks[i + 1]
                    i += 2
                elif t in ("WHENCE", "PREFIX", "TYPE"):
                    rec["extra_kw"].append(t)
                    i += 2
                else:
                    rec["files"].append(t)
                    i += 1
            self.targets.add(rec["name"])
            if rec["name"] in ("embedded-resources",):
                self.cmrc = rec
            else:
                self.structural.append(f"unknown cmrc resource library: {rec['name']}")
        elif n == "add_subdirectory":
            toks = self.toks(cmd.raw)
            if toks:
                self.subdirs.append(toks[0])
                # targets defined inside known subdirs that the root references
                if "host_shaders" in toks[0]:
                    self.targets.add("host_shaders")
                if "imgui/renderer" in toks[0]:
                    self.targets.add("ImGui_Resources")
                if toks[0].endswith("externals"):
                    for t in ("Dear_ImGui", "discord-rpc", "protobuf::protoc"):
                        self.targets.add(t)
        elif n == "configure_file":
            toks = self.toks(cmd.raw)
            if len(toks) >= 2:
                self.configure_files.append((toks[0], toks[1]))
        elif n == "set_source_files_properties":
            self.sfp.append(self.toks(cmd.raw))
        # everything else (project, include, message, install, ...) is irrelevant

    def structural_target(self, tgt):
        if tgt not in ("shadps4",):
            self.structural.append(f"new root-level target upstream: {tgt}")


# ---- profiles ----

def make_profile(win=False):
    p = {
        "CMAKE_CURRENT_SOURCE_DIR": "@ROOT@",
        "CMAKE_SOURCE_DIR": "@ROOT@",
        "PROJECT_SOURCE_DIR": "@ROOT@",
        "CMAKE_CURRENT_BINARY_DIR": "@BIN@",
        "CMAKE_BINARY_DIR": "@BIN@",
        "PROJECT_BINARY_DIR": "@BIN@",
        "CMAKE_SYSTEM_PROCESSOR": "x86_64",
        "CMAKE_HOST_SYSTEM_PROCESSOR": "x86_64",
        "CMAKE_CXX_COMPILER_ID": "Clang",
        "CMAKE_MODULE_PATH": "@ROOT@/cmake",
        # options pinned regardless of upstream defaults:
        "ENABLE_TESTS": "",
        "ENABLE_QT_GUI": "",
        "ENABLE_SYSTEM_LIBRARIES": "1",   # keep the find_package mirror visible
        "ENABLE_DISCORD_RPC": "1",
        "ENABLE_USERFAULTFD": "",
        "ENABLE_SYSTEM_VULKAN": "",
        "ENABLE_UPDATER": "",
        "GIT_BRANCH": "main",
        "GIT_REMOTE_NAME": "origin",
    }
    if win:
        p.update({"WIN32": "1", "MSVC": "1", "CMAKE_SYSTEM_NAME": "Windows",
                  "CMAKE_HOST_WIN32": "1"})
    else:
        p.update({"UNIX": "1", "LINUX": "1", "CMAKE_SYSTEM_NAME": "Linux"})
    return p


# ---- surface computation ----

@dataclass
class Surface:
    sha: str
    version: str
    find_packages: list       # list of token-lists, order preserved, deduped
    excludes: list            # SRC-relative .cpp not compiled upstream
    libs: list                # core_main extra libs (upstream order)
    defines: list
    extra_sources: list       # SRC-relative compiled sources the glob misses
    cmrc_files: list          # resource-dir-relative embedded resources
    warnings: list
    structural: list
    win_libs: list
    win_defs: list
    compiled: set


def parse_upstream_surface(ref, combine_text):
    """Parse ref:CMakeLists.txt and compute the core_main surface."""
    text = git_out("show", f"{ref}:CMakeLists.txt")
    tree_files = tree_file_set(ref)
    warnings = []

    def parse_with(profile):
        ev = CMakeEval(profile, tree_files, lambda m: warnings.append(m))
        tree = build_tree(scan_commands(strip_comments(text)))
        ev.eval_stmts(tree)
        return ev

    ev = parse_with(make_profile(win=False))
    evw = parse_with(make_profile(win=True))

    version = ".".join(filter(None, (
        ev.vars.get("EMULATOR_VERSION_MAJOR", ""),
        ev.vars.get("EMULATOR_VERSION_MINOR", ""),
        ev.vars.get("EMULATOR_VERSION_PATCH", ""))))

    structural = list(ev.structural) + list(evw.structural)

    # ---- classify sources ----
    # EXCLUDES must be "on disk but compiled on NO platform" — a Windows-only .cpp
    # is #ifdef-guarded inside and must stay globbed on Linux (status-quo policy),
    # so the compiled set is the UNION of the Linux and Windows profiles.
    ondisk_cpp = {p for p in tree_files if p.startswith("src/") and p.endswith(".cpp")}
    compiled, generated = set(), set()
    for s in list(dict.fromkeys(ev.sources + evw.sources)):
        if s.startswith("@BIN@"):
            generated.add(s)
        elif s.startswith("@ROOT@/"):
            compiled.add(s[len("@ROOT@/"):])
        else:
            compiled.add(s)
    compiled = {c for c in compiled if not c.endswith(".h") and not c.endswith(".hpp")
                and not c.endswith(".inc") and not c.endswith(".inl")}

    # compiled files that don't exist in the tree (and aren't generated) are a parse bug
    for c in sorted(compiled):
        if c not in tree_files:
            structural.append(f"compiled source not in upstream tree (parser bug?): {c}")

    # generated compiled sources must match known recipes Combine.cmake regenerates
    known_gen = []
    for g in sorted(generated):
        base = os.path.basename(g)
        if base == "scm_rev.cpp":
            ok = any(src.endswith("scm_rev.cpp.in") for src, _ in ev.configure_files)
            (known_gen if ok else structural).append(
                g if ok else f"generated source with unknown recipe: {g}")
        elif base.endswith(".pb.cc"):
            protoc = [cc for cc in ev.custom_commands
                      if any("protoc" in t for t in cc["command"])
                      and any(o.endswith(base) for o in cc["outputs"])]
            if protoc and any(d.endswith("shadnet.proto") for d in protoc[0]["depends"]):
                if "shadnet_proto_gen" not in combine_text:
                    structural.append(
                        "upstream compiles shadnet protobuf sources but cmake/Combine.cmake "
                        "has no shadnet_proto_gen mirror — upgrade Combine.cmake")
                known_gen.append(g)
            else:
                structural.append(f"generated source with unknown recipe: {g}")
        else:
            structural.append(f"generated source with unknown recipe: {g}")

    # custom commands that feed the target but aren't understood
    for cc in ev.custom_commands:
        outs = set(cc["outputs"])
        feeding = [g for g in generated if g in outs]
        if feeding and not any(os.path.basename(f) in ("scm_rev.cpp",) or f.endswith(".pb.cc")
                               for f in feeding):
            structural.append(f"unknown codegen feeding shadps4: {cc['command'][:3]} -> {feeding}")

    excludes = sorted((ondisk_cpp - compiled) - {"src/main.cpp"})
    excludes = [p[len("src/"):] for p in excludes]

    extra = sorted(c for c in compiled
                   if c in tree_files
                   and os.path.splitext(c)[1] not in (".cpp", ".s", ".S")
                   and not c.endswith(".rc"))
    extra_sources = [p[len("src/"):] for p in extra if p.startswith("src/")]

    # ---- libs ----
    base_libs = set()
    for m in re.finditer(r"target_link_libraries\s*\(\s*core_\$\{name\}([^)]*)\)", combine_text):
        for t in m.group(1).split():
            if t not in ("PRIVATE", "PUBLIC", "INTERFACE"):
                base_libs.add(t)
    libs, seen = [], set()
    for l in ev.libs:
        if l.startswith("$") or l.startswith("/") or l.startswith("@"):
            warnings.append(f"link library skipped (path/genex): {l}")
            continue
        if l in base_libs or l in INTERNAL_LIBS or l in seen:
            continue
        seen.add(l)
        libs.append(l)

    # ---- defines ----
    defines, dseen = [], set()
    for d in ev.tcd + ev.global_defs:
        key = d.split("=")[0].strip('"')
        if key in KNOWN_TCD or key in KNOWN_GLOBAL_DEFINES or key in dseen:
            continue
        dseen.add(key)
        defines.append(d)

    # ---- find_package (dedup by name, keep first, order preserved) ----
    fps, fseen = [], set()
    for toks in ev.find_packages:
        if toks[0] in fseen:
            continue
        fseen.add(toks[0])
        fps.append(toks)

    # ---- cmrc ----
    cmrc_files = []
    if ev.cmrc:
        if ev.cmrc["extra_kw"]:
            structural.append(f"cmrc mechanism changed upstream (keywords {ev.cmrc['extra_kw']})")
        if ev.cmrc["alias"] != "res::embedded" or ev.cmrc["namespace"] != "res":
            structural.append(
                f"cmrc alias/namespace changed upstream: {ev.cmrc['alias']} / {ev.cmrc['namespace']}")
        for f in ev.cmrc["files"]:
            m = re.match(r"^src/(resources|images)/(.+)$", f)
            if m:
                cmrc_files.append(m.group(2))
            else:
                structural.append(f"embedded resource outside src/(resources|images): {f}")
    else:
        warnings.append("no cmrc_add_resource_library(embedded-resources) found upstream; "
                        "core_main falls back to Combine.cmake's derived resource set")

    # ---- windows report ----
    win_libs = []
    lseen = set(ev.libs)
    for l in evw.libs:
        if l not in lseen and not l.startswith("$"):
            win_libs.append(l)
    win_defs = sorted({d.split("=")[0] for d in evw.global_defs + evw.tcd}
                      - {d.split("=")[0] for d in ev.global_defs + ev.tcd})

    # unexpected subdirectories
    known_subdirs = ("externals", "host_shaders", "imgui/renderer", "tests")
    for sd in ev.subdirs:
        if not any(k in sd for k in known_subdirs):
            structural.append(f"unknown add_subdirectory upstream: {sd}")

    return Surface(sha=ref, version=version, find_packages=fps, excludes=excludes,
                   libs=libs, defines=defines, extra_sources=extra_sources,
                   cmrc_files=cmrc_files, warnings=warnings, structural=structural,
                   win_libs=win_libs, win_defs=win_defs, compiled=compiled)


def emit_surface_file(surface, sha_full, meta_line):
    fp_lines = []
    for toks in surface.find_packages:
        args = " ".join(t if " " not in t else f'"{t}"' for t in toks)
        fp_lines.append(f"find_package({args})")

    def cmake_list(name, items):
        if not items:
            return f'set({name} "")'
        inner = "\n    ".join(items)
        return f"set({name}\n    {inner})"

    fp_block = "\n".join("    " + l for l in fp_lines)
    return f"""# GENERATED FILE - do not edit; regenerate with ./upstream-merge.py
#
# core_main's upstream build surface, extracted from shadPS4's root
# CMakeLists.txt under the combined build's profile (Linux/x86_64, SDL-only,
# no tests). The umbrella includes this file; Combine.cmake consumes the
# CORE_MAIN_* lists via add_core(NAME main ...).
#
#   upstream commit : {meta_line}
#   emulator version: {surface.version or 'unknown'}

set(CORE_MAIN_UPSTREAM_SHA "{sha_full}")

# --- find_package mirror. Gated on ENABLE_SYSTEM_LIBRARIES exactly like upstream
#     (default OFF): with it off, the vendored externals/ provide every target and
#     the build cannot break when host/container dev packages come and go.
#     Apple/FreeBSD-only entries are intentionally absent. ---
if(ENABLE_SYSTEM_LIBRARIES)
{fp_block}
endif()

# --- on-disk .cpp files upstream does NOT compile (add_core globs, then removes these) ---
{cmake_list("CORE_MAIN_EXCLUDES", surface.excludes)}

# --- link libraries beyond Combine.cmake's shared base set (upstream order) ---
{cmake_list("CORE_MAIN_LIBS", surface.libs)}

# --- compile definitions beyond the shared base ---
{cmake_list("CORE_MAIN_DEFINES", surface.defines)}

# --- compiled sources the *.cpp/*.s/*.S glob would miss ---
{cmake_list("CORE_MAIN_EXTRA_SOURCES", surface.extra_sources)}

# --- embedded (cmrc) resources, relative to the tree's resources/ dir ---
{cmake_list("CORE_MAIN_CMRC_FILES", surface.cmrc_files)}
"""


# ------------------------------------------------------------- absorbed-patch check


def resolve_conflicts_ours(text):
    """Resolves merge-conflict markers by keeping only the 'ours' side.

    Returns the text as a list of lines. Handles the plain merge marker style
    and the diff3/zdiff3 styles (a '|||||||' base section inside the ours half).
    """
    out, mode = [], "keep"
    for line in text.splitlines():
        if mode == "keep" and line.startswith("<<<<<<<"):
            mode = "ours"
        elif mode == "ours" and line.startswith("|||||||"):
            mode = "base"
        elif mode in ("ours", "base") and line == "=======":
            mode = "theirs"
        elif mode == "theirs" and line.startswith(">>>>>>>"):
            mode = "keep"
        elif mode in ("keep", "ours"):
            out.append(line)
    return out


def patch_hunk_lines(body, sign):
    """Hunk payload lines carrying the given sign ('+' or '-'), sign stripped."""
    lines, in_hunk = [], False
    for line in body.splitlines():
        if line.startswith("@@"):
            in_hunk = True
        elif line.startswith("diff --git"):
            in_hunk = False
        elif in_hunk and line.startswith(sign):
            lines.append(line[1:])
    return lines


def absorbed_upstream(body, conflicted, target_text):
    """Judges whether a conflicted carried patch already exists in the new base.

    A pure-addition patch counts as absorbed when every added line is present
    in the target with at least the added multiplicity, and resolving every
    conflict in favor of the incoming upstream side reproduces the target
    exactly (so each cleanly merged hunk was a textual no-op). Patches with
    removals are never auto-judged. A fork addition that duplicates an existing
    target line verbatim could in principle satisfy both conditions, so the
    conflict dump is kept on disk and the drop is reported explicitly.
    """
    if patch_hunk_lines(body, "-"):
        return False
    added = patch_hunk_lines(body, "+")
    if not added:
        return False
    target_lines = target_text.splitlines()
    have = Counter(target_lines)
    for line, n in Counter(added).items():
        if have[line] < n:
            return False
    return resolve_conflicts_ours(conflicted) == target_lines


# ======================================================================== sync logic


class Sync:
    def __init__(self, args):
        self.args = args
        self.root = os.path.abspath(args.root)
        self.created_paths = []          # for the failure/rollback message
        self.patch_results = []          # (patch, status, detail)
        self.recovery = []               # (patch, fpath, dump) per FAILED patch

    # ---------- phases ----------

    def preflight(self):
        os.chdir(self.root)
        top = git_out("rev-parse", "--show-toplevel")
        os.chdir(top)
        self.root = top
        for p in (SRC_DIR, "src-gr2", "cmake", "externals", "CMakeLists.txt"):
            if not os.path.exists(p):
                die(f"not the combined repo (missing {p}) — run from {DEFAULT_ROOT}")

        if not git("remote", "get-url", UPSTREAM_REMOTE, check=False).returncode == 0:
            say(f"adding remote {UPSTREAM_REMOTE} -> {UPSTREAM_URL}")
            git("remote", "add", UPSTREAM_REMOTE, UPSTREAM_URL)
        if not self.args.no_fetch:
            say(f"fetching {UPSTREAM_REMOTE}/{UPSTREAM_BRANCH} ...")
            git("fetch", "--quiet", UPSTREAM_REMOTE, UPSTREAM_BRANCH)

        ref = self.args.commit or f"{UPSTREAM_REMOTE}/{UPSTREAM_BRANCH}"
        self.target = git_out("rev-parse", ref + "^{commit}")
        self.target_short = git_out("rev-parse", "--short", self.target)
        if not git("merge-base", "--is-ancestor", self.target,
                   f"{UPSTREAM_REMOTE}/{UPSTREAM_BRANCH}", check=False).returncode == 0:
            warn(f"target {self.target_short} is not on {UPSTREAM_REMOTE}/{UPSTREAM_BRANCH}")

        # old base: recorded, else merge-base with upstream
        self.old_base = None
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                first = f.readline().strip().split()[0]
            self.old_base = git_out("rev-parse", first + "^{commit}")
        else:
            r = git("merge-base", "HEAD", f"{UPSTREAM_REMOTE}/{UPSTREAM_BRANCH}",
                    check=False)
            if r.returncode != 0:
                die(f"no {STATE_FILE} and no common history with upstream (flattened "
                    f"repo?) — write the last synced upstream sha into {STATE_FILE} "
                    "and rerun.")
            self.old_base = r.stdout.strip()
        self.old_base_short = git_out("rev-parse", "--short", self.old_base)
        say(f"upstream base: {self.old_base_short}  ->  target: {self.target_short}")

        # owned paths must be clean vs HEAD (worktree + index)
        dirty = git_out("status", "--porcelain", "--", SRC_DIR, "cmake",
                        "externals", ".gitmodules", SURFACE_FILE)
        if dirty:
            die("uncommitted changes under the sync-owned paths — commit or stash first:",
                dirty)

        log = git_out("log", "--oneline", "--no-decorate",
                      f"{self.old_base}..{self.target}")
        self.upstream_log = log.splitlines()
        say(f"upstream commits to absorb: {len(self.upstream_log)}")
        if self.old_base == self.target:
            say("already at the target commit — will still re-verify the surface.")

    def load_patches(self):
        """Bootstrap upstream-patches/ from the current worktree drift if needed."""
        drift = git_out("diff", "--no-renames", "--name-status", self.old_base,
                        "HEAD", "--", SRC_DIR, "externals/CMakeLists.txt", "cmake")
        drift_files = {}
        for line in drift.splitlines():
            status, path = line.split("\t", 1)
            if path in UMBRELLA_CMAKE:      # umbrella-owned files are not "drift"
                continue
            drift_files[path] = status

        os.makedirs(PATCH_DIR, exist_ok=True)
        existing = {f for f in os.listdir(PATCH_DIR) if f.endswith(".patch")}

        def patch_name(path):
            return path.replace("/", "_") + ".patch"

        adopted = []
        for path, status in sorted(drift_files.items()):
            name = patch_name(path)
            if name in existing:
                continue
            if status == "D":
                REPORT.problem("carried patches",
                               f"{path} was deleted locally vs the upstream base — "
                               "deletions can't be carried; restore it or handle manually")
                continue
            # Raw stdout, not git_out(): .strip() drops a trailing whitespace-only context
            # line (a blank source line diffs as a single space), corrupting the hunk.
            diff = git("diff", self.old_base, "HEAD", "--", path).stdout
            header = (f"# carried local patch for the upstream mirror (auto-extracted)\n"
                      f"# file: {path}\n# original base: {self.old_base}\n")
            if not self.args.dry_run:
                with open(os.path.join(PATCH_DIR, name), "w") as f:
                    f.write(header + diff + ("\n" if not diff.endswith("\n") else ""))
                self.created_paths.append(os.path.join(PATCH_DIR, name))
            adopted.append(path)
        if adopted:
            for p in adopted:
                REPORT.add("carried patches", f"adopted local deviation as {PATCH_DIR}/{patch_name(p)}: {p}")
            if self.args.strict:
                die("--strict: refusing to adopt unreviewed local drift (see report)")

        self.patches = sorted(f for f in os.listdir(PATCH_DIR) if f.endswith(".patch")) \
            if os.path.isdir(PATCH_DIR) else []
        say(f"carried patches: {len(self.patches)}" +
            (f"  ({', '.join(self.patches)})" if self.patches else ""))

    def mirror(self):
        say(f"mirroring {SRC_DIR}/ -> {self.target_short} (replace, not merge) ...")
        shutil.rmtree(SRC_DIR)
        git("checkout", self.target, "--", SRC_DIR)
        git("add", "-A", "--", SRC_DIR)

        # ---- cmake/ : upstream files replace-synced, umbrella files preserved ----
        up_cmake = ls_tree_paths(self.target, "cmake")
        local_cmake = ls_tree_paths("HEAD", "cmake")
        if up_cmake:
            git("checkout", self.target, "--", "cmake")
            # restore umbrella-owned files that exist in HEAD
            for f in UMBRELLA_CMAKE:
                if f in local_cmake:
                    git("checkout", "HEAD", "--", f)
        removed = [p for p in local_cmake
                   if p not in up_cmake and p not in UMBRELLA_CMAKE]
        for p in removed:
            git("rm", "-f", "--quiet", "--", p)
            REPORT.add("cmake/", f"removed (gone upstream): {p}")
        git("add", "-A", "--", "cmake")

        # ---- externals/ ----
        up_ext = ls_tree_paths(self.target, "externals")
        local_ext = ls_tree_paths("HEAD", "externals")
        pins = set()
        if os.path.exists(PIN_FILE):
            with open(PIN_FILE) as f:
                pins = {l.strip() for l in f if l.strip() and not l.startswith("#")}

        git("checkout", self.target, "--", "externals")
        # restore pinned submodule gitlinks to their local SHAs
        for path in pins:
            if path in local_ext and local_ext[path][1] == "commit":
                sha = git_out("rev-parse", f"HEAD:{path}")
                git("update-index", "--cacheinfo", f"160000,{sha},{path}")
                REPORT.add("externals", f"pinned (held at local commit): {path}")

        self.new_submodules, self.moved_submodules, self.removed_submodules = [], [], []
        for path, (mode, typ) in sorted(up_ext.items()):
            if typ != "commit":
                continue
            if path not in local_ext:
                self.new_submodules.append(path)
            elif local_ext[path][1] == "commit":
                old = git_out("rev-parse", f"HEAD:{path}")
                new = git_out("rev-parse", f"{self.target}:{path}")
                if old != new and path not in pins:
                    self.moved_submodules.append((path, old[:12], new[:12]))
        for path, (mode, typ) in sorted(local_ext.items()):
            if typ == "commit" and path not in up_ext:
                if path in KEEP_SUBMODULES or path in pins:
                    continue
                self.removed_submodules.append(path)

        for path in self.removed_submodules:
            git("submodule", "deinit", "-f", "--", path, check=False)
            git("rm", "-f", "--cached", "--", path, check=False)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            REPORT.add("externals", f"submodule removed upstream: {path} "
                                    "(worktree dir deleted; .git/modules left in place)")
        git("add", "-A", "--", "externals")
        # add -A records each initialized submodule at its checked-out worktree HEAD,
        # which reverts gitlinks the checkout above just synced to the target. Re-assert
        # them; the materialize step below follows the index, so it must see the new SHAs.
        for path in self.new_submodules + [p for p, _, _ in self.moved_submodules]:
            sha = git_out("rev-parse", f"{self.target}:{path}")
            git("update-index", "--cacheinfo", f"160000,{sha},{path}")

        # ---- .gitmodules : upstream verbatim + local-only sections appended ----
        up_gm = git_out("show", f"{self.target}:.gitmodules") \
            if ".gitmodules" in tree_file_set(self.target) else ""
        local_gm = open(".gitmodules").read() if os.path.exists(".gitmodules") else ""
        sect_re = re.compile(r'(?ms)^\[submodule "([^"]+)"\][^\[]*')
        up_paths = set()
        for m in re.finditer(r'(?m)^\s*path\s*=\s*(\S+)', up_gm):
            up_paths.add(m.group(1))
        keep_sections = []
        for m in sect_re.finditer(local_gm):
            body = m.group(0)
            pm = re.search(r'(?m)^\s*path\s*=\s*(\S+)', body)
            path = pm.group(1) if pm else m.group(1)
            if path not in up_paths and (path in KEEP_SUBMODULES or path in pins
                                         or os.path.isdir(path)):
                keep_sections.append(body.rstrip() + "\n")
                if path not in KEEP_SUBMODULES and path not in pins:
                    REPORT.add(".gitmodules", f"kept local-only submodule entry: {path} "
                               "(add to KEEP_SUBMODULES in upstream-merge.py if intentional)")
        merged = up_gm.rstrip() + "\n" + "".join(keep_sections)
        with open(".gitmodules", "w") as f:
            f.write(merged)
        git("add", "--", ".gitmodules")

        # ---- materialize submodule worktrees ----
        need_update = self.new_submodules + [p for p, _, _ in self.moved_submodules]
        if need_update:
            git("submodule", "sync", "--quiet", "--", *need_update, check=False)
            say(f"initializing/updating submodules: {', '.join(need_update)} ...")
            r = git("submodule", "update", "--init", "--recursive", "--",
                    *need_update, check=False)
            if r.returncode != 0:
                REPORT.problem("externals",
                               f"git submodule update failed (offline?): {(r.stderr or '').strip()[-400:]} "
                               f"— rerun: git submodule update --init --recursive")
        for p in self.new_submodules:
            REPORT.add("externals", f"NEW submodule: {p}")
        for p, old, new in self.moved_submodules:
            REPORT.add("externals", f"submodule moved: {p}  {old} -> {new}  "
                                    "(shared with core_gr2 — watch its build)")

    def reapply_patches(self):
        target_files = tree_file_set(self.target)
        for patch in self.patches:
            ppath = os.path.join(PATCH_DIR, patch)
            body = open(ppath).read()
            m = re.search(r"(?m)^# file: (.+)$", body)
            fpath = m.group(1) if m else None
            # A patch with no hunks means its deviation was absorbed upstream at an earlier
            # sync (regeneration wrote a header-only file). Prune it rather than feed it to
            # git apply, which rejects an empty body with "No valid patches in input".
            if not re.search(r"(?m)^@@", body):
                os.remove(ppath)
                self.patch_results.append((patch, "obsolete",
                                           f"{fpath or patch}: empty (absorbed upstream) - pruned"))
                continue
            if fpath and fpath not in target_files and not os.path.exists(fpath):
                self.patch_results.append((patch, "ORPHANED",
                                           f"{fpath} no longer exists upstream — patch skipped"))
                continue
            r = git("apply", "--3way", "--whitespace=nowarn", ppath,
                    check=False)
            if r.returncode == 0:
                self.patch_results.append((patch, "applied", ""))
                if fpath:
                    git("add", "--", fpath)
                continue
            # conflict: save the conflicted state, then decide
            conflicted = None
            if fpath and os.path.exists(fpath):
                conflicted = open(fpath).read()
            if fpath:
                git("checkout", self.target, "--", fpath, check=False)
                git("add", "--", fpath, check=False)
            if fpath == BRAND_SRC and self.apply_brand_fallback():
                self.patch_results.append((patch, "applied (regex fallback)",
                                           "3-way merge failed; branding re-applied by transform"))
                continue
            detail = f"3-way merge failed: {(r.stderr or '').strip()[:300]}"
            dump = None
            if conflicted and "<<<<<<<" in conflicted:
                dump = os.path.join(tempfile.gettempdir(),
                                    "upstream-merge-conflict-" + patch.replace(".patch", ""))
                with open(dump, "w") as f:
                    f.write(conflicted)
                detail += f"; conflicted version saved to {dump}"
            # A conflict can also mean upstream adopted the deviation itself while the
            # surrounding context moved. The file is already restored to the target
            # above, so dropping the patch completes the absorption.
            if dump and fpath:
                target_text = git("show", f"{self.target}:{fpath}", check=False).stdout
                if target_text and absorbed_upstream(body, conflicted, target_text):
                    os.remove(ppath)
                    self.patch_results.append(
                        (patch, "absorbed upstream (context drift) - dropped",
                         "every added line already exists in the new base; "
                         f"conflicted merge kept at {dump}"))
                    continue
                if target_text:
                    added = patch_hunk_lines(body, "+")
                    have = Counter(target_text.splitlines())
                    present = sum(1 for ln in added if have[ln] > 0)
                    if added and present:
                        detail += (f"; {present}/{len(added)} added lines already exist "
                                   "in the new base (possible partial absorption)")
            self.recovery.append((patch, fpath, dump))
            self.patch_results.append((patch, "FAILED", detail))

        # refresh the patches against the NEW base + the fiber companion file
        for patch, status, _ in self.patch_results:
            if not status.startswith("applied"):
                continue
            body = open(os.path.join(PATCH_DIR, patch)).read()
            m = re.search(r"(?m)^# file: (.+)$", body)
            if not m:
                REPORT.add("carried patches",
                           f"{patch}: no '# file:' header — not regenerated against the new base")
                continue
            fpath = m.group(1)
            # Raw stdout, not git_out(): .strip() drops a trailing whitespace-only context
            # line (a blank source line diffs as a single space), corrupting the hunk.
            diff = git("diff", self.target, "--", fpath).stdout
            if not diff.strip():
                # Deviation fully absorbed upstream - prune rather than write a header-only
                # patch that would fail to apply next sync ("No valid patches in input").
                os.remove(os.path.join(PATCH_DIR, patch))
                REPORT.add("carried patches", f"{patch}: absorbed upstream - dropped")
                continue
            header = (f"# carried local patch for the upstream mirror\n"
                      f"# file: {fpath}\n# original base: {self.target}\n")
            with open(os.path.join(PATCH_DIR, patch), "w") as f:
                f.write(header + diff + ("\n" if diff and not diff.endswith("\n") else ""))
        with open(STATE_FILE, "w") as f:
            f.write(f"{self.target} shadPS4 upstream base (synced by upstream-merge.py)\n")
        git("add", "-A", "--", PATCH_DIR)

        if any(s.startswith("applied") for p, s, _ in self.patch_results
               if p.startswith("src_core_libraries_fiber")):
            shutil.copyfile(FIBER_SRC, FIBER_COMPANION)
            git("add", "--", FIBER_COMPANION)
            REPORT.add("carried patches",
                       f"refreshed {FIBER_COMPANION} (set-version.sh companion) from the "
                       "newly patched fiber.cpp")

        for patch, status, detail in self.patch_results:
            line = f"{patch}: {status}" + (f" — {detail}" if detail else "")
            if status in ("FAILED", "ORPHANED"):
                REPORT.problem("carried patches", line)
            else:
                REPORT.add("carried patches", line)

    def apply_brand_fallback(self):
        """Re-brand src/emulator.cpp window titles when the patch no longer applies."""
        try:
            gr2 = open("src-gr2/emulator.cpp").read()
            vm = re.search(r"GR2FORK v([0-9][0-9.]*)", gr2)
            ver = vm.group(1) if vm else None
        except OSError:
            ver = None
        if not ver:
            return False
        src = open(BRAND_SRC).read()
        if 'fmt::format("shadPS4 ' not in src:
            return False
        out = src.replace('fmt::format("shadPS4 ',
                          f'fmt::format("Junmin Lee GR2fork v{ver} - shadPS4 ')
        with open(BRAND_SRC, "w") as f:
            f.write(out)
        git("add", "--", BRAND_SRC)
        return True

    def regenerate_surface(self):
        combine_text = open("cmake/Combine.cmake").read()
        surface = parse_upstream_surface(self.target, combine_text)
        self.surface = surface

        for w in surface.warnings:
            REPORT.add("parser warnings", w)
        for s in surface.structural:
            REPORT.problem("structural", s)

        meta = git_out("log", "-1", "--format=%h %cs", self.target)
        content = emit_surface_file(surface, self.target, meta)

        old = open(SURFACE_FILE).read() if os.path.exists(SURFACE_FILE) else None
        if old != content:
            with open(SURFACE_FILE, "w") as f:
                f.write(content)
            git("add", "--", SURFACE_FILE)
        self.report_surface_delta(old, content)

        # ---- structural safety checks ----
        # imgui config ODR identity between the trees (shared Dear_ImGui build)
        a, b = "src/imgui/imgui_config.h", "src-gr2/imgui/imgui_config.h"
        if os.path.exists(a) and os.path.exists(b):
            if open(a).read() != open(b).read():
                REPORT.problem("ODR", "src/ and src-gr2/ imgui_config.h now DIFFER — the shared "
                               "Dear_ImGui build uses src/'s config; a divergence is an "
                               "ABI/ODR hazard. Reconcile or split the imgui build per-core.")
        # Windows surface drift (report-only; Combine.cmake owns the win32 sets)
        combine_win = set(re.findall(r"[A-Za-z0-9_.:\-]+", "\n".join(
            re.findall(r"target_link_libraries\(core_\$\{name\} PRIVATE([^)]*)\)", combine_text))))
        for l in self.surface.win_libs:
            if l not in combine_win and l not in INTERNAL_LIBS:
                REPORT.add("windows surface (report-only)",
                           f"upstream links '{l}' on Windows; Combine.cmake's WIN32 set doesn't — "
                           "verify on the next Windows build")
        for d in self.surface.win_defs:
            if d not in KNOWN_GLOBAL_DEFINES and d not in KNOWN_TCD:
                REPORT.add("windows surface (report-only)",
                           f"new Windows compile definition upstream: {d}")
        # shared base set removals: upstream stopped linking something the base still has
        combine_base = set()
        for m in re.finditer(r"target_link_libraries\s*\(\s*core_\$\{name\}([^)]*)\)", combine_text):
            combine_base.update(t for t in m.group(1).split()
                                if t not in ("PRIVATE", "PUBLIC", "INTERFACE")
                                and "$" not in t)
        up_all = set(self.surface.libs) | set(l for l in self.surfaces_all_libs())
        for l in sorted(combine_base - up_all):
            if l in ("uuid", "discord-rpc", "imgui_fonts_${name}", "res_${name}::embedded",
                     "ws2_32", "iphlpapi", "winmm", "bcrypt", "userenv", "version",
                     "mincore", "wepoll", "wbemuuid", "clang_rt.builtins-x86_64.lib",
                     "Tracy::TracyClient"):
                continue
            REPORT.add("shared base (report-only)",
                       f"Combine.cmake base links '{l}' but upstream no longer does — "
                       "harmless if unreferenced; consider removing when convenient")

    def surfaces_all_libs(self):
        # upstream linux libs including the ones subsumed by the base set
        text = git_out("show", f"{self.target}:CMakeLists.txt")
        ev = CMakeEval(make_profile(win=False), tree_file_set(self.target), lambda m: None)
        ev.eval_stmts(build_tree(scan_commands(strip_comments(text))))
        return ev.libs

    def report_surface_delta(self, old, new):
        def lists_of(txt):
            out = {}
            if not txt:
                return out
            for m in re.finditer(r"set\((CORE_MAIN_[A-Z_]+)([^)]*)\)", txt):
                out[m.group(1)] = [t for t in m.group(2).split() if t != '""']
            out["FIND_PACKAGES"] = re.findall(r"(?m)^find_package\((.+)\)$", txt)
            return out
        o, n = lists_of(old), lists_of(new)
        keys = sorted(set(o) | set(n))
        any_change = False
        for k in keys:
            ov, nv = o.get(k, []), n.get(k, [])
            if k == "CORE_MAIN_UPSTREAM_SHA":
                continue
            added = [x for x in nv if x not in ov]
            removed = [x for x in ov if x not in nv]
            for x in added:
                REPORT.add("surface delta", f"{k}: + {x}")
                any_change = True
            for x in removed:
                REPORT.add("surface delta", f"{k}: - {x}")
                any_change = True
        if old is None:
            REPORT.add("surface delta", f"(first generation of {SURFACE_FILE})")
        elif not any_change:
            REPORT.add("surface delta", "(no build-surface changes)")

    def cmake_root(self):
        """The source-root path form cmake must be invoked with. An existing build
        cache pins CMAKE_HOME_DIRECTORY as a string (e.g. /home/... vs /var/home/...
        on Silverblue) — reuse its spelling when it points at this repo."""
        bdir = self.args.build_dir
        cache = os.path.join(self.root if not os.path.isabs(bdir) else "", bdir,
                             "CMakeCache.txt")
        try:
            m = re.search(r"(?m)^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$",
                          open(cache).read())
            if m and os.path.realpath(m.group(1)) == os.path.realpath(self.root):
                return m.group(1)
        except OSError:
            pass
        if os.path.realpath(DEFAULT_ROOT) == os.path.realpath(self.root):
            return DEFAULT_ROOT
        return self.root

    def verify(self):
        if self.args.no_verify and not self.args.build:
            return
        tc = toolchain(self.args.container)
        sroot = self.cmake_root()
        if self.args.build:
            bdir = self.args.build_dir
            if not os.path.isabs(bdir):
                bdir = os.path.join(sroot, bdir)
            if self.args.clean_build and os.path.isdir(bdir):
                say(f"removing {bdir}/ for a clean build ...")
                shutil.rmtree(bdir)
            logf = os.path.join(self.root, "upstream-merge-build.log")
            say(f"configuring {bdir}/ (log: {logf}) ...")
            with open(logf, "w") as lf:
                r = subprocess.run([*tc, "cmake", "-S", sroot, "-B", bdir,
                                    *CONFIGURE_ARGS],
                                   stdout=lf, stderr=subprocess.STDOUT, text=True)
                if r.returncode != 0:
                    REPORT.problem("build", f"configure FAILED — tail of {logf}:\n" +
                                   open(logf).read()[-2500:])
                    return
                say(f"building {bdir}/ (this takes a while) ...")
                r = subprocess.run([*tc, "cmake", "--build", bdir, "--parallel",
                                    str(os.cpu_count())],
                                   stdout=lf, stderr=subprocess.STDOUT, text=True)
            if r.returncode != 0:
                out = open(logf).read()
                errs = "\n".join(l for l in out.splitlines()
                                 if re.search(r"error|FAILED", l))[-3000:]
                REPORT.problem("build", f"build FAILED — errors (full log: {logf}):\n" +
                               (errs or out[-3000:]))
            else:
                REPORT.add("build", f"full build OK: {bdir}/shadps4 (+ both cores); log: {logf}")
        else:
            scratch = os.path.join(sroot, ".upstream-merge-cfgcheck")
            say(f"configure smoke-test in {scratch} ...")
            r = run([*tc, "cmake", "-S", sroot, "-B", scratch, *CONFIGURE_ARGS],
                    check=False)
            if r.returncode != 0:
                REPORT.problem("verify", "cmake configure FAILED:\n" +
                               ((r.stderr or "") + (r.stdout or ""))[-2500:])
            else:
                REPORT.add("verify", "cmake configure OK (full build not run; use --build)")
            shutil.rmtree(scratch, ignore_errors=True)

    def print_recovery(self):
        """Copy-paste recovery steps for each patch the 3-way merge failed on."""
        for patch, fpath, dump in self.recovery:
            ppath = f"{PATCH_DIR}/{patch}"
            print(f"""
!! recovery for {ppath}
   The worktree copy of {fpath} is reset to the pure upstream side; the
   conflicted merge is saved at {dump or '(no dump)'}.
   a) If the deviation is fully absorbed upstream, drop the patch:
        git rm -f {ppath}
   b) If the deviation is still needed, re-apply it to {fpath} (the dump shows
      both sides), then refresh the patch against the new base:
        {{ printf '# carried local patch for the upstream mirror\\n# file: {fpath}\\n# original base: {self.target}\\n'; git diff {self.target} -- {fpath}; }} > {ppath}
        git add {fpath} {ppath}
   Then commit the staged sync by hand, or roll back (recipe below) and rerun
   after the fix.""", file=sys.stderr)

    def commit(self):
        if REPORT.problems:
            if not (self.args.no_commit or self.args.dry_run):
                warn("blocking problems found — NOT committing (worktree left for inspection).")
            self.print_recovery()
            return
        if self.args.no_commit or self.args.dry_run:
            return
        staged = git_out("diff", "--cached", "--name-only")
        if not staged:
            say("nothing changed — no commit needed.")
            return
        subject = f"core_main: sync upstream surface to shadPS4 @ {self.target_short}"
        patchlines = [f"  {p}: {s}" for p, s, _ in self.patch_results] or ["  (none)"]
        # Bare SHAs only - upstream subject lines carry (#NNNN) PR numbers, and a
        # pushed commit message creates a reference event on every GitHub PR it
        # mentions. A SHA range autolinks without creating references.
        body = (f"Replace-mirror of src/ + shared cmake//externals//.gitmodules to upstream "
                f"{self.target}, regenerated {SURFACE_FILE}.\n\n"
                f"Carried patches:\n" + "\n".join(patchlines) +
                f"\n\nUpstream range: {self.old_base}..{self.target} "
                f"({len(self.upstream_log)} commits)")
        git("commit", "--quiet", "-m", subject, "-m", body)
        say(f"committed: {git_out('rev-parse', '--short', 'HEAD')}")

        if not self.args.no_merge_history:
            if git("merge-base", "HEAD", self.target, check=False).returncode != 0:
                # flattened (orphan) repo: merging would re-import upstream's whole
                # history — skip; the sync base is tracked in STATE_FILE instead.
                say("flattened/unrelated history detected — skipping the -s ours "
                    f"history merge (sync base tracked in {STATE_FILE}).")
                return
            if git("diff", "--cached", "--quiet", check=False).returncode != 0:
                warn("index not clean; skipping the -s ours history merge")
                return
            r = git("merge", "-s", "ours", "--no-ff", "-m",
                    f"absorb upstream history @ {self.target_short} "
                    f"(content already mirrored by upstream-merge.py)",
                    self.target, check=False)
            if r.returncode == 0:
                say("history merge recorded (git merge -s ours) — merge-base now tracks the sync.")
            else:
                warn(f"history merge skipped: {(r.stderr or '').strip()[:200]}")

    # ---------- driver ----------

    def run(self):
        self.preflight()
        if self.args.dry_run:
            self.dry_run_report()
            return
        self.load_patches()
        self.mirror()
        self.reapply_patches()
        self.regenerate_surface()
        self.verify()
        REPORT.dump()
        self.commit()
        if REPORT.problems:
            print()
            die(f"{len(REPORT.problems)} blocking problem(s) — see the report above. "
                "Worktree left in the synced state for inspection; to roll back:\n"
                "  git restore --source=HEAD --staged --worktree -- src cmake externals .gitmodules\n"
                "  git submodule update --init --recursive")
        say("done.")

    def dry_run_report(self):
        delta = git_out("diff", "--no-renames", "--name-status", "HEAD",
                        self.target, "--", SRC_DIR)
        counts = {}
        for line in delta.splitlines():
            counts[line[0]] = counts.get(line[0], 0) + 1
        say(f"src/ delta: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
            if counts else "src/ identical")
        for line in delta.splitlines()[:40]:
            print(f"    {line}")
        if len(delta.splitlines()) > 40:
            print(f"    ... {len(delta.splitlines()) - 40} more")
        ext = git_out("diff", "--no-renames", "--name-status", "HEAD",
                      self.target, "--", "externals", "cmake", ".gitmodules")
        say("externals/cmake/.gitmodules delta:")
        for line in ext.splitlines():
            print(f"    {line}")
        combine_text = open("cmake/Combine.cmake").read()
        surface = parse_upstream_surface(self.target, combine_text)
        old = open(SURFACE_FILE).read() if os.path.exists(SURFACE_FILE) else None
        meta = git_out("log", "-1", "--format=%h %cs", self.target)
        self.surface = surface
        self.report_surface_delta(old, emit_surface_file(surface, self.target, meta))
        for w in surface.warnings:
            REPORT.add("parser warnings", w)
        for s in surface.structural:
            REPORT.problem("structural", s)
        REPORT.dump()
        say("dry run — nothing changed.")


# ----------------------------------------------------------------------- selftest


def selftest(ref, root):
    os.chdir(root)
    os.chdir(git_out("rev-parse", "--show-toplevel"))
    target = git_out("rev-parse", ref + "^{commit}")
    combine_text = open("cmake/Combine.cmake").read()
    s = parse_upstream_surface(target, combine_text)
    meta = git_out("log", "-1", "--format=%h %cs", target)
    print(f"# surface of {meta}")
    print(f"version        : {s.version}")
    print(f"compiled files : {len(s.compiled)}")
    print(f"find_package   : {len(s.find_packages)}")
    for t in s.find_packages:
        print(f"    {' '.join(t)}")
    print(f"EXCLUDES       : {s.excludes}")
    print(f"LIBS           : {s.libs}")
    print(f"DEFINES        : {s.defines}")
    print(f"EXTRA_SOURCES  : {s.extra_sources}")
    print(f"CMRC ({len(s.cmrc_files):2})      : {s.cmrc_files}")
    print(f"win-only libs  : {s.win_libs}")
    print(f"win-only defs  : {s.win_defs}")
    if s.warnings:
        print("warnings:")
        for w in s.warnings:
            print(f"    {w}")
    if s.structural:
        print("STRUCTURAL:")
        for x in s.structural:
            print(f"    {x}")
    sys.exit(2 if s.structural else 0)


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--commit", help="upstream commit/ref to sync to "
                                           "(default: upstream/main after fetch)")
    ap.add_argument("-n", "--dry-run", action="store_true", help="report only, change nothing")
    ap.add_argument("--no-fetch", action="store_true", help="use the already-fetched upstream")
    ap.add_argument("--no-commit", action="store_true", help="do everything but leave it staged")
    ap.add_argument("--no-merge-history", action="store_true",
                    help="skip the history-only `git merge -s ours <target>`")
    ap.add_argument("--build", action="store_true",
                    help=f"run the full production build into {BUILD_DIR}/ (implies verify)")
    ap.add_argument("--clean-build", action="store_true",
                    help=f"rm -rf {BUILD_DIR}/ first (implies --build)")
    ap.add_argument("--build-dir", default=BUILD_DIR)
    ap.add_argument("--container", default="",
                    help=f"distrobox with the toolchain (default: auto — host cmake if "
                         f"present, else '{BUILD_CONTAINER}'); 'host' forces the host")
    ap.add_argument("--no-verify", action="store_true", help="skip the cmake configure check")
    ap.add_argument("--strict", action="store_true",
                    help="fail instead of auto-adopting unrecognized local drift as a patch")
    ap.add_argument("--selftest", metavar="REF",
                    help="parse REF's CMakeLists.txt, print the computed surface, exit")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    if args.clean_build:
        args.build = True

    if args.selftest:
        selftest(args.selftest, args.root)

    Sync(args).run()


if __name__ == "__main__":
    main()
