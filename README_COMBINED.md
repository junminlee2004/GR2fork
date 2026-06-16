# Combined shadPS4 — two cores, one binary

One executable containing **two complete emulator cores**, one selected per launch:

- **`core_gr2`** — the editable gr2fork (all features + the accuracy work you port in).
- **`core_main`** — the **byte-untouched** prerelease (`6a20eba`), a clean A/B reference.

One file on disk. No second exe, no launcher-process toggle. Switching cores = relaunch.

---

## Layout

The combined repo is shaped **like a normal shadPS4 repo** — same shared `cmake/` and
`externals/`, the umbrella as the root `CMakeLists.txt` — except there are **two pure
source folders** side by side instead of one:

```
combined/                  <- build from here
├── CMakeLists.txt         the umbrella (replaces a tree's root CMakeLists)
├── cmake/                 SHARED: the trees' modules (CMakeRC, Find*, …) + the 3 umbrella files
│   ├── Combine.cmake          (umbrella) add_core(): one source tree -> libcore_<name>.so
│   ├── export.map             (umbrella) version script: export only core_entry
│   ├── embed_cores.S.in       (umbrella) .incbin template (release single-file)
│   ├── CMakeRC.cmake          (from the tree)
│   └── Find*.cmake            (from the tree)
├── externals/             SHARED vendored deps, built ONCE (the two trees' are identical)
├── src/                   prerelease SOURCE  ->  core_main   (common/, core/, main.cpp, … — NO CMakeLists)
├── src-gr2/               gr2fork SOURCE     ->  core_gr2    (common/, core/, main.cpp, … — NO CMakeLists)
├── dispatcher/            the selector/loader exe
├── verify_mechanism.sh
└── README_COMBINED.md
```

`src/` and `src-gr2/` are **pure source** — exactly the contents of a tree's `src/` folder,
no `CMakeLists.txt` inside (the root umbrella drives the build). They're separate trees: the
prerelease source differs from the gr2fork source. `cmake/` and `externals/` are shared
because the two trees vendor identical deps and modules.

### Setup — start from a prerelease checkout, then drop gr2fork's source in

The cleanest path: a recursive prerelease checkout already gives you `cmake/`, `externals/`,
and `src/`. Swap its root CMakeLists for the umbrella, add the dispatcher + the 3 umbrella
cmake files, then drop gr2fork's `src/` in as `src-gr2/`.

```
git clone --recursive <prerelease-remote> combined
cd combined

# umbrella + dispatcher + the 3 umbrella cmake files (added alongside the tree's cmake/)
cp   /path/to/delivered/CMakeLists.txt        .
cp -r /path/to/delivered/dispatcher           .
cp   /path/to/delivered/cmake/Combine.cmake   /path/to/delivered/cmake/export.map \
     /path/to/delivered/cmake/embed_cores.S.in  cmake/
cp   /path/to/delivered/verify_mechanism.sh   /path/to/delivered/README_COMBINED.md  .

# gr2fork's SOURCE goes in as src-gr2/  (just its src/ folder)
cp -r /path/to/gr2fork/src  src-gr2
```

(Or make `src-gr2` a git submodule pointing at gr2fork, if you'd rather keep it linked.)

---

## Build & run

The umbrella does **not** hardcode `-march` when you pass your own (it only applies an
`x86-64-v3` floor if `CMAKE_C/CXX_FLAGS` carry no `-march`), so your znver4 isn't clobbered.
Your usual invocation works as-is; add `-DCOMBINED_EMBED_CORES=ON` for the single file.

**znver4 (your daily build):**

```
rm -rf build_opt
cmake -S . -B build_opt \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_FLAGS="-march=znver4 -mtune=znver4 -O3 -flto=thin -DNDEBUG" \
  -DCMAKE_CXX_FLAGS="-march=znver4 -mtune=znver4 -O3 -flto=thin -DNDEBUG" \
  -DCMAKE_EXE_LINKER_FLAGS="-flto=thin -fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
  -DCOMBINED_EMBED_CORES=ON
cmake --build build_opt --parallel $(nproc)
./build_opt/shadps4 /path/to/game            # auto-picks the core from the game's title id
./build_opt/shadps4 --core=main /path/...    # force a core (overrides auto-detection)
```

**generic (x86-64-v3 floor, Deck-safe):** same with `-march=x86-64-v3 -mtune=znver4`.

Drop `-DCOMBINED_EMBED_CORES=ON` for a **dev** build — cores are then loose `libcore_*.so`
next to the exe (faster: touch one core, only it relinks). With the flag, `build_opt/shadps4`
is the *only* file to ship.

**Core selection** precedence: `--core=gr2|main` (consumed, not forwarded) → `$SHADPS4_CORE`
→ **auto-detect from the game's `TITLE_ID`** (a Gravity Rush 2 / Remastered id → `core_gr2`,
anything else → `core_main`; the dispatcher reads `sce_sys/param.sfo` itself before loading a
core, and also matches a bare title id passed on the command line) → default `core_main`.
`argv[0]` preserved, the rest forwarded verbatim. Invalid core exits 2.

---

## How it works

Each source tree is compiled into its own `-fvisibility=hidden` shared object exporting a
**single** symbol, `core_entry`. A tiny dispatcher (the only real `main`) loads exactly one
per run with `dlopen(..., RTLD_LOCAL)`. Only one core is mapped per run and every other
symbol is hidden + localized, so the two cores' **identical** symbol sets never collide.

- Each tree's `int main(...)` is retargeted to `core_<name>_main` via a per-TU
  `-Dmain=<name>_main` — **no source edit**. A 3-line shim calls it. The shim declares the
  renamed main with **normal C++ linkage** (not `extern "C"`) and forces `core_entry` to
  default visibility so `-fvisibility=hidden` doesn't drop it.
- `cmake/export.map` (`{ global: core_entry; local: *; }`) localizes everything else,
  including the statically-linked externals' symbols — that's what guarantees isolation.
- The dependency stack (`externals/`) is identical between the trees, so it's built **once**
  and shared; each core links only the subset it references.
- Embedded resources: the source hardcodes the alias `"src/images/…"`, so cmrc is given
  `WHENCE=<src>/images PREFIX=src/images` — the alias reads `src/images/…` for both cores
  even though gr2's folder is `src-gr2`.
- **Dev:** loose `libcore_*.so`, `dlopen`'d by path. **Release** (`-DCOMBINED_EMBED_CORES=ON`):
  both `.so` `.incbin`-embedded → at launch the chosen one is written to a `memfd` and
  `dlopen`'d via `/proc/self/fd/N` → one self-contained file.

---

## What's proven vs. what needs your toolchain

**Proven here**, by compile-and-run and by a real `cmake`+`ninja` build/configure:

- *Mechanism* (`verify_mechanism.sh`, 11/11 under both gcc and your clang/ThinLTO/lld):
  two `.so` with identical symbol sets load in one process with `RTLD_LOCAL` and neither
  interposes; the `-Dmain=` retarget + shim work; dispatcher selection/forwarding; release
  embed → memfd → a single file that runs both cores with **zero `.so` on disk**.
- *Umbrella, stub-source, your exact flags* (`clang++ -O3 -flto=thin -fuse-ld=lld`,
  `-DCOMBINED_EMBED_CORES=ON`): configures, builds, both cores export only `core_entry`,
  cmrc aliases come out `src/images/…` for both, single binary runs both cores.
- *Umbrella, **real** prerelease + gr2fork source*: configures and generates; each core gets
  exactly its tree's TU set (gr2 = 346, main = 362), the per-tree stray files are handled
  correctly (gr2 compiles `file.cpp`/`net_obj.cpp` and excludes `screenshot_service2`/
  `emit_spirv`; main does the reverse), and all per-core sub-targets are uniquely named.

**Needs your toolchain** (this sandbox has no Vulkan/Boost/SDL3 and the `externals/`
submodules aren't fetched, so the real TUs weren't compiled here): the actual per-core
compile + link of ~346/362 TUs against your vendored `externals/`. `find_package` floors
mirror the trees' CMakeLists; externals supply most via `OVERRIDE_FIND_PACKAGE`.

Run `verify_mechanism.sh` first — it re-proves the isolation contract on your box in seconds:

```
CXX=clang++ CXXFLAGS_EXTRA="-march=znver4 -flto=thin" LDFLAGS_EXTRA="-fuse-ld=lld" \
  bash verify_mechanism.sh
```

---

## Maintenance notes

- **Source lists are globbed** (`*.cpp`) minus each tree's `EXCLUDES` (set per-`add_core()` in
  the umbrella). The two trees have different non-compiled strays — gr2:
  `screenshot_service2.cpp` + `frontend/translate/emit_spirv.cpp`; main: `core/file_sys/file.cpp`
  + `network/net_obj.cpp`. New files you add are picked up automatically; if you add a `.cpp`
  meant to be `#include`d rather than compiled, add it to that core's `EXCLUDES`.
- **Shared-engine fixes drift.** A change to code in *both* trees lands in `src-gr2` only;
  `src/` stays the frozen reference. Re-port deliberately when you want the reference to move.
- **`libstdc++` stays dynamic/shared** across the `dlopen` boundary — don't statically bundle it.
- **One core per run.** Switching = relaunch; there's no in-process hot-swap (that's what the
  isolation buys).
- The **PREAMBLE SLOT** in the umbrella is where extra top-level flags go (sanitizer/LTO
  toggles, etc.) — it runs before externals and before any core.
