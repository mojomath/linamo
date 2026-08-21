#!/bin/bash
# ===----------------------------------------------------------------------=== #
# Make the `decimo` arbitrary-precision package available to the Linamo build.
#
# Linamo does not compile without it. `Matrix` and `MatrixView` name
# `decimo.Numeric` in the `where` clauses that give a `Matrix[BInt]` its
# arithmetic, and Mojo has no conditional imports, so decimo is a hard
# dependency of the whole library rather than of one corner of it. The trait
# has to live in decimo because Mojo conformance is nominal and is declared
# where the struct is.
#
# Three sources, tried in order:
#
#   1. `DECIMO_PATH=/path/to/decimo` --- a working copy, for developing the two
#      libraries together. This is the path to use while a decimo change is
#      still uncommitted.
#   2. The conda package `decimo` from the modular-community channel, once it
#      ships a build carrying `decimo.Numeric` and `decimo.Parsable`.
#   3. The upstream git repository, pinned at $DECIMO_COMMIT.
#
# The package is always *precompiled with this workspace's own `mojo`*. A
# `.mojoc` built by another environment's compiler is not loadable here: it
# does not fail to import, it crashes the compiler, which is a long way to
# travel for a stale artefact.
#
# Usage:
#   bash tools/ensure_decimo.sh            # auto-detect
#
# Environment overrides:
#   DECIMO_PATH=<dir>      build from a local working copy
#   LINAMO_DECIMO=conda    require the environment-provided package
#   LINAMO_DECIMO=git      force the pinned git checkout
#   DECIMO_COMMIT=<sha>    use a different upstream commit
#   DECIMO_REPO=<url>      use a different upstream repository
# ===----------------------------------------------------------------------=== #

set -euo pipefail

DECIMO_REPO="${DECIMO_REPO:-https://github.com/forfudan/decimo.git}"
# The commit that carries both `decimo.Numeric` and `decimo.Parsable`. Update
# this when decimo releases and the conda package carries both traits, at which
# point source 2 takes over and this is only the fallback.
DECIMO_COMMIT="${DECIMO_COMMIT:-f0dbc4a690377f73322c14e34d47ab1900f3e0e5}"
MODE="${LINAMO_DECIMO:-auto}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p temp

PKG="temp/decimo.mojoc"
STAMP="temp/.decimo.stamp"

build_from() {
    local src="$1" stamp="$2"
    pixi run mojo precompile "$src" -o "$PKG"
    echo "$stamp" >"$STAMP"
    echo "decimo: built $PKG from $stamp"
}

# --- 1. A local working copy ---------------------------------------------- #
if [[ -n "${DECIMO_PATH:-}" ]]; then
    if [[ ! -d "$DECIMO_PATH/src/decimo" ]]; then
        echo "decimo: DECIMO_PATH=$DECIMO_PATH has no src/decimo." >&2
        exit 1
    fi
    build_from "$DECIMO_PATH/src/decimo" "local:$DECIMO_PATH"
    exit 0
fi

# --- 2. Is decimo already importable from the environment? ---------------- #
# Compile a two-line probe *without* `-I temp`, so only a package provided by
# the environment can satisfy the import.
#
# The probe is built in a scratch directory outside the repository, and that
# is load-bearing: Mojo searches the directory holding the file it is
# compiling. A probe written into `temp/` finds `temp/decimo.mojoc` sitting
# next to it, reports success, and the fallback build this script just made
# gets deleted as a stale shadow of a conda package that was never installed.
env_has_decimo() {
    local dir
    dir="$(mktemp -d)"
    printf 'from decimo import Numeric, Parsable\n\ndef main():\n    pass\n' >"$dir/probe.mojo"
    local ok=0
    pixi run mojo build -o "$dir/probe" "$dir/probe.mojo" >/dev/null 2>&1 || ok=1
    rm -rf "$dir"
    return $ok
}

if [[ "$MODE" != "git" ]]; then
    if env_has_decimo; then
        # A stale fallback build would shadow the conda package via `-I temp`.
        rm -f "$PKG" "$STAMP"
        echo "decimo: using the package provided by the environment (conda)."
        exit 0
    fi
    if [[ "$MODE" == "conda" ]]; then
        echo "decimo: LINAMO_DECIMO=conda, but no decimo package is installed." >&2
        exit 1
    fi
fi

# --- 3. Fallback: pinned git checkout -------------------------------------- #
if [[ -z "$DECIMO_COMMIT" ]]; then
    echo "decimo: not provided by the environment, and no upstream commit is" >&2
    echo "        pinned yet. Set DECIMO_PATH to a local checkout, or set" >&2
    echo "        DECIMO_COMMIT once the traits land upstream." >&2
    exit 1
fi

# Outside `temp/`, because `-I temp` is on every Linamo build: a directory
# named `temp/decimo` would resolve as the package `decimo` and shadow the
# `.mojoc` built from it.
CLONE_DIR=".decimo-src"
if [[ -f "$PKG" && -f "$STAMP" && "$(cat "$STAMP")" == "git:$DECIMO_COMMIT" ]]; then
    echo "decimo: reusing $PKG (commit ${DECIMO_COMMIT:0:8})."
    exit 0
fi

if [[ ! -d "$CLONE_DIR/.git" ]]; then
    rm -rf "$CLONE_DIR"
    # Blobless: file contents are fetched only for the commit checked out
    # below, which is the whole point on a CI runner with no warm cache.
    git clone --quiet --filter=blob:none --no-checkout "$DECIMO_REPO" "$CLONE_DIR"
fi
if ! git -C "$CLONE_DIR" cat-file -e "$DECIMO_COMMIT^{commit}" 2>/dev/null; then
    git -C "$CLONE_DIR" fetch --quiet --all --tags --prune
fi
git -C "$CLONE_DIR" checkout --quiet --detach "$DECIMO_COMMIT"
build_from "$CLONE_DIR/src/decimo" "git:$DECIMO_COMMIT"
