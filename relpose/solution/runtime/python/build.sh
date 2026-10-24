#!/usr/bin/env bash
# Builds the reference Python runtime (CPython + numpy + OpenCV as a wasip1 reactor) inside the toolchain image.
# Output: build/runtime/dist/runtime.wasm (what `relpose bundle` embeds your code into) and python.wasm (a REPL for experiments).
# First build 10-15 minutes; needs the image from relpose/solution/toolchain/build-image.sh. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work relpose-toolchain relpose/solution/runtime/python/cpython/build-wasm.sh "$@"
