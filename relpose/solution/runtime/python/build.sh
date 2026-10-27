#!/usr/bin/env bash
# Builds the reference Python runtime (CPython + numpy + OpenCV as a wasip1 reactor) natively on Linux or macOS (Windows: inside WSL2).
# Output: build/runtime/dist/runtime.wasm (what `relpose bundle` embeds your code into) and python.wasm (a REPL for experiments).
# First build 10-15 minutes; fetches the toolchain into build/toolchain first (relpose/solution/toolchain/download_dependencies.sh). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
relpose/solution/toolchain/download_dependencies.sh
relpose/solution/runtime/python/cpython/build-wasm.sh "$@"
