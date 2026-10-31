#!/usr/bin/env bash
# Builds the reference Python runtime (CPython + numpy + OpenCV as a wasip1 reactor) natively on Linux or macOS (Windows: inside WSL2).
# Output in the workspace's build/runtime/dist: runtime.wasm (what `viobench bundle` embeds your code into; `viobench build-runtime`
# runs this script, and `viobench bundle` does when runtime.wasm is missing) and python.wasm (a REPL for experiments).
# First build 10-15 minutes; fetches the toolchain into build/toolchain first (toolchain/download_dependencies.sh). Run from anywhere.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/../../toolchain/download_dependencies.sh"
bash "$HERE/cpython/build-wasm.sh" "$@"
