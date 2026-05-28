#!/usr/bin/env bash
# build_ior.sh - build IOR from source against the cphqvsy OpenMPI variant
# (the one orangefs/2.10 depends on, so the two coexist at runtime without a
# module swap). Installs into $PREFIX (default ~/.local).
#
# One-time setup:
#   bash scripts/build_ior.sh
#
# Idempotent: skips clone/configure/make/install if $PREFIX/bin/ior already
# works. Force rebuild with REBUILD=1.
#
# Tunables (env vars):
#   PREFIX        install root                 (default: $HOME/.local)
#   IOR_SRC       source clone path            (default: $REPO_ROOT/external/libs/ior)
#   IOR_REF       git ref to check out         (default: main)
#   MODULE_MPI    openmpi module to build with (default: openmpi/5.0.5-cphqvsy
#                                              -- matches orangefs/2.10's dep)
#   SKIP_MODULES  1 to skip module loads       (default: 0)
#   REBUILD       1 to wipe build/ and rebuild (default: 0)
#   JOBS          make -j parallelism          (default: nproc)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PREFIX="${PREFIX:-$HOME/.local}"
IOR_SRC="${IOR_SRC:-$REPO_ROOT/external/libs/ior}"
IOR_REF="${IOR_REF:-main}"
MODULE_MPI="${MODULE_MPI:-openmpi/5.0.5-cphqvsy}"
SKIP_MODULES="${SKIP_MODULES:-0}"
REBUILD="${REBUILD:-0}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '  PASS: %s\n' "$*"; }
warn() { printf '  WARN: %s\n' "$*"; }
fail() { printf '  FAIL: %s\n' "$*" >&2; exit 1; }

# Smoke test: ior -h prints help and exits 1 (by design). Check output instead.
ior_works() {
  [ -x "$1" ] || return 1
  "$1" -h 2>&1 | grep -q "^Synopsis"
}

# -- Short-circuit if already installed --
if [ "$REBUILD" != "1" ] && ior_works "$PREFIX/bin/ior"; then
  ok "ior already installed at $PREFIX/bin/ior (set REBUILD=1 to rebuild)"
  "$PREFIX/bin/ior" -h 2>&1 | head -3
  exit 0
fi

# -- Load the right MPI module --
if [ "$SKIP_MODULES" != "1" ]; then
  if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
      [ -f "$init" ] && . "$init" >/dev/null 2>&1 && break
    done
  fi
  if command -v module >/dev/null 2>&1; then
    module --redirect purge 2>/dev/null
    module --redirect load "$MODULE_MPI" 2>/dev/null
    if [ $? -eq 0 ]; then ok "loaded $MODULE_MPI"; else fail "module load $MODULE_MPI"; fi
  else
    warn "no Lmod available; using whatever mpicc is on PATH"
  fi
fi

command -v mpicc >/dev/null || fail "mpicc not in PATH after module load"
ok "mpicc: $(command -v mpicc)"
ok "mpi version: $(mpicc --showme:version 2>&1 | head -1 || mpicc -v 2>&1 | head -1)"

# -- Toolchain sanity --
for tool in git autoconf automake libtool make; do
  command -v "$tool" >/dev/null || fail "$tool not in PATH; install via apt or load a module"
done

# -- Clone or update source --
if [ ! -d "$IOR_SRC/.git" ]; then
  log "cloning hpc/ior into $IOR_SRC"
  mkdir -p "$(dirname "$IOR_SRC")"
  git clone https://github.com/hpc/ior.git "$IOR_SRC" || fail "git clone"
fi
( cd "$IOR_SRC" && git fetch --depth=1 origin "$IOR_REF" 2>/dev/null && git checkout "$IOR_REF" >/dev/null 2>&1 ) || warn "could not switch to $IOR_REF; using current checkout"
log "ior source: $IOR_SRC @ $(cd "$IOR_SRC" && git rev-parse --short HEAD)"

# -- Bootstrap (if needed) --
if [ ! -f "$IOR_SRC/configure" ] || [ "$REBUILD" = "1" ]; then
  log "running ./bootstrap"
  ( cd "$IOR_SRC" && ./bootstrap ) >"$IOR_SRC/bootstrap.log" 2>&1 || fail "bootstrap failed; see $IOR_SRC/bootstrap.log"
  ok "bootstrap done"
fi

# -- Configure + build --
BUILD_DIR="$IOR_SRC/build"
if [ "$REBUILD" = "1" ]; then
  rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

if [ ! -f "$BUILD_DIR/Makefile" ]; then
  log "configuring with prefix=$PREFIX"
  ( cd "$BUILD_DIR" && CC=mpicc ../configure --prefix="$PREFIX" ) >"$BUILD_DIR/configure.log" 2>&1 \
    || fail "configure failed; see $BUILD_DIR/configure.log"
  ok "configure done"
fi

log "building (make -j$JOBS)"
( cd "$BUILD_DIR" && make -j"$JOBS" ) >"$BUILD_DIR/make.log" 2>&1 \
  || fail "make failed; see $BUILD_DIR/make.log"
ok "build done"

log "installing into $PREFIX"
mkdir -p "$PREFIX/bin"
( cd "$BUILD_DIR" && make install ) >"$BUILD_DIR/install.log" 2>&1 \
  || fail "make install failed; see $BUILD_DIR/install.log"
ok "installed: $PREFIX/bin/ior"

# -- Smoke test --
ior_works "$PREFIX/bin/ior" || fail "installed ior fails to run"
"$PREFIX/bin/ior" -h 2>&1 | head -3
log "done. Add to PATH: export PATH=$PREFIX/bin:\$PATH"
