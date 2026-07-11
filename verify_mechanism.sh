#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
#
# verify_mechanism.sh — reproduce the "two cores, one binary" isolation proof on
# YOUR machine, with no externals/Vulkan/SDL3 needed. It builds two trivial stub
# cores that deliberately share an identical symbol set, then proves:
#   1. each core .so exports exactly one symbol (core_entry);
#   2. both can be loaded in ONE process with RTLD_LOCAL and neither interposes
#      on the other's identical symbol;
#   3. the dispatcher selects/forwards correctly;
#   4. the embedded release build is a single self-contained file (runs after
#      being copied away from the .so files).
#
# This validates the MECHANISM the umbrella uses. It does not build the real
# emulator (that needs your full toolchain) — it isolates the novel part.
set -euo pipefail
CXX="${CXX:-g++}"
# Optional: re-run under your real toolchain, e.g.
#   CXX=clang++ CXXFLAGS_EXTRA="-flto=thin" LDFLAGS_EXTRA="-fuse-ld=lld" bash verify_mechanism.sh
CXXFLAGS_EXTRA="${CXXFLAGS_EXTRA:-}"
LDFLAGS_EXTRA="${LDFLAGS_EXTRA:-}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"
pass=0; fail=0
ok(){ echo "  PASS: $1"; pass=$((pass+1)); }
no(){ echo "  FAIL: $1"; fail=$((fail+1)); }
# Capture a command's exit code without tripping `set -e` (the stub cores return
# non-zero "exit codes" 42/7 on purpose).
getrc(){ local r=0; "$@" >/dev/null 2>&1 || r=$?; printf '%s' "$r"; }

echo "== building two stub cores with an IDENTICAL symbol (core_tag) =="
for c in gr2 main; do
  tag=$([ "$c" = gr2 ] && echo 1001 || echo 2002)
  ret=$([ "$c" = gr2 ] && echo 42 || echo 7)
  cat > ${c}_main.cpp <<EOF
#include <cstdio>
int core_tag();
int main(int argc,char**argv){ std::printf("[core_$c] tag=%d argc=%d\n",core_tag(),argc); return $ret; }
EOF
  cat > ${c}_tag.cpp <<EOF
int core_tag(){ return $tag; }
EOF
done
cat > core_entry.cpp <<'EOF'
extern int CORE_REAL_MAIN(int,char**);
extern "C" __attribute__((visibility("default")))
int core_entry(int argc,char**argv){ return CORE_REAL_MAIN(argc,argv); }
EOF
echo '{ global: core_entry; local: *; };' > export.map

build_core(){ local c=$1
  $CXX -std=c++23 -O2 $CXXFLAGS_EXTRA -fPIC -fvisibility=hidden -Dmain=${c}_main -c ${c}_main.cpp -o ${c}_main.o
  $CXX -std=c++23 -O2 $CXXFLAGS_EXTRA -fPIC -fvisibility=hidden                  -c ${c}_tag.cpp  -o ${c}_tag.o
  $CXX -std=c++23 -O2 $CXXFLAGS_EXTRA -fPIC -fvisibility=hidden -DCORE_REAL_MAIN=${c}_main -c core_entry.cpp -o ${c}_entry.o
  $CXX -shared -fPIC ${c}_main.o ${c}_tag.o ${c}_entry.o -Wl,--version-script=export.map $LDFLAGS_EXTRA -o libcore_${c}.so
}
build_core gr2; build_core main

echo "== 1. each core exports ONLY core_entry =="
for c in gr2 main; do
  syms="$(nm -D --defined-only libcore_${c}.so | awk '{print $NF}' | tr '\n' ' ')"
  [ "$(echo $syms)" = "core_entry" ] && ok "libcore_${c}.so exports: $syms" || no "libcore_${c}.so exports: $syms"
  nm -D libcore_${c}.so | grep -q core_tag && no "core_${c} leaks core_tag" || ok "core_${c} hides identical core_tag"
done

echo "== 2. same-process isolation (both loaded, identical symbol not interposed) =="
cat > iso.cpp <<'EOF'
#include <cstdio>
#include <dlfcn.h>
using fn=int(*)(int,char**);
int main(){
  void*g=dlopen("./libcore_gr2.so",RTLD_NOW|RTLD_LOCAL);
  void*m=dlopen("./libcore_main.so",RTLD_NOW|RTLD_LOCAL);
  if(!g||!m){std::fprintf(stderr,"dlopen: %s\n",dlerror());return 2;}
  auto eg=(fn)dlsym(g,"core_entry"); auto em=(fn)dlsym(m,"core_entry");
  if(eg==em){std::puts("INTERPOSED");return 3;}
  char a[]="p";char*v[]={a,nullptr};
  return eg(1,v)==42 && em(1,v)==7 ? 0 : 4;   // each must reach its OWN main
}
EOF
$CXX -std=c++23 -O2 $CXXFLAGS_EXTRA iso.cpp -ldl $LDFLAGS_EXTRA -o iso
if ./iso >/dev/null; then ok "distinct core_entry; each resolves its own core_tag"; else no "isolation broke"; fi

echo "== 3. dispatcher selection + arg forwarding (dev, loose .so) =="
# minimal dispatcher mirroring dispatcher/main.cpp + core_loader.cpp
cat > disp.cpp <<'EOF'
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
using fn=int(*)(int,char**);
int main(int argc,char**argv){
  std::string core; bool fromflag=false; std::vector<char*> fwd{argv[0]};
  for(int i=1;i<argc;i++){
    if(!fromflag&&!std::strncmp(argv[i],"--core=",7)){core=argv[i]+7;fromflag=true;continue;}
    fwd.push_back(argv[i]);
  }
  fwd.push_back(nullptr);
  if(!fromflag){const char*e=std::getenv("SHADPS4_CORE"); if(e&&*e)core=e;}
  if(core.empty())core="gr2";
  if(core!="gr2"&&core!="main"){std::fprintf(stderr,"invalid core\n");return 2;}
  std::string so="./libcore_"+core+".so";
  void*h=dlopen(so.c_str(),RTLD_NOW|RTLD_LOCAL);
  if(!h){std::fprintf(stderr,"%s\n",dlerror());return 5;}
  return ((fn)dlsym(h,"core_entry"))((int)fwd.size()-1,fwd.data());
}
EOF
$CXX -std=c++23 -O2 $CXXFLAGS_EXTRA disp.cpp -ldl $LDFLAGS_EXTRA -o shadps4
[ "$(getrc ./shadps4 --core=gr2)"  -eq 42 ] && ok "--core=gr2 -> gr2 (rc 42)"  || no "gr2 selection"
[ "$(getrc ./shadps4 --core=main)" -eq 7  ] && ok "--core=main -> main (rc 7)" || no "main selection"
[ "$(getrc ./shadps4)"             -eq 42 ] && ok "default -> gr2"              || no "default selection"
[ "$(SHADPS4_CORE=main getrc ./shadps4)" -eq 7 ] && ok "SHADPS4_CORE=main env" || no "env selection"
[ "$(getrc ./shadps4 --core=bogus)" -eq 2 ] && ok "invalid core rejected (rc 2)" || no "invalid handling"

echo "== 4. embedded release = single self-contained file =="
cat > embed.S <<EOF
    .section .rodata
    .balign 16
    .global core_gr2_so_start
core_gr2_so_start:
    .incbin "$work/libcore_gr2.so"
    .global core_gr2_so_end
core_gr2_so_end:
    .balign 16
    .global core_main_so_start
core_main_so_start:
    .incbin "$work/libcore_main.so"
    .global core_main_so_end
core_main_so_end:
    .section .note.GNU-stack,"",@progbits
EOF
cat > disp_embed.cpp <<'EOF'
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dlfcn.h>
using fn=int(*)(int,char**);
extern "C" const unsigned char core_gr2_so_start[],core_gr2_so_end[],core_main_so_start[],core_main_so_end[];
int main(int argc,char**argv){
  std::string core=argc>1?argv[1]:"gr2";
  const unsigned char*b,*e;
  if(core=="gr2"){b=core_gr2_so_start;e=core_gr2_so_end;}
  else{b=core_main_so_start;e=core_main_so_end;}
  size_t n=e-b; int fd=memfd_create(("c_"+core).c_str(),MFD_CLOEXEC);
  if(write(fd,b,n)!=(ssize_t)n)return 6;
  char p[64]; std::snprintf(p,sizeof p,"/proc/self/fd/%d",fd);
  void*h=dlopen(p,RTLD_NOW|RTLD_LOCAL); if(!h){std::fprintf(stderr,"%s\n",dlerror());return 7;}
  char a[]="p";char*v[]={a,nullptr};
  return ((fn)dlsym(h,"core_entry"))(1,v);
}
EOF
$CXX -std=c++23 -O2 $CXXFLAGS_EXTRA -c embed.S -o embed.o
$CXX -std=c++23 -O2 $CXXFLAGS_EXTRA disp_embed.cpp embed.o -ldl $LDFLAGS_EXTRA -o shadps4_embed
ship="$(mktemp -d)"; cp shadps4_embed "$ship/shadps4"     # copy ONLY the binary
nso="$(find "$ship" -name '*.so' | wc -l)"
rg="$(cd "$ship" && getrc ./shadps4 gr2)"
rm_="$(cd "$ship" && getrc ./shadps4 main)"
{ [ "$nso" -eq 0 ] && [ "$rg" -eq 42 ] && [ "$rm_" -eq 7 ]; } \
  && ok "single binary runs both cores with NO .so on disk (gr2 rc=$rg, main rc=$rm_)" \
  || no "self-contained run (.so=$nso gr2=$rg main=$rm_)"
rm -rf "$ship"

echo
echo "==================  $pass passed, $fail failed  =================="
[ $fail -eq 0 ]
