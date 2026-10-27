#!/usr/bin/env bash
# Builds a native (C/C++) solution directory with wasi-sdk:  relpose/solution/runtime/native/build.sh relpose/solution/examples/native/cpp_minimal
# The directory needs a CMakeLists.txt (see relpose/solution/examples/native/cpp_minimal); the module lands in <dir>/build/<target>.wasm.
# wasi-sdk comes from build/toolchain (relpose/solution/toolchain/download_dependencies.sh, run here first).
set -euo pipefail
if [ $# -lt 1 ]; then
    echo "usage: $0 <solution directory> [extra cmake arguments]" >&2
    exit 2
fi
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
SOLUTION=$(cd "$1" && pwd)
shift
"$REPO/relpose/solution/toolchain/download_dependencies.sh"
mkdir -p "$SOLUTION/build"
cmake -S "$SOLUTION" -B "$SOLUTION/build" -DCMAKE_TOOLCHAIN_FILE="$REPO/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake" -DRELPOSE_NATIVE_DIR="$REPO/relpose/solution/runtime/native" "$@" > "$SOLUTION/build/cmake.log" 2>&1 || (cat "$SOLUTION/build/cmake.log"; exit 1)
cmake --build "$SOLUTION/build"
ls -la "$SOLUTION"/build/*.wasm
