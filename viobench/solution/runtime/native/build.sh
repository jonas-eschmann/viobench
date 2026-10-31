#!/usr/bin/env bash
# Builds a native (C/C++) solution directory with wasi-sdk (`viobench bundle <dir> -o x.wasm` runs this):  runtime/native/build.sh <dir> [cmake arguments]
# The directory needs a CMakeLists.txt (see examples/native/cpp_minimal); the module lands in <dir>/build/<target>.wasm.
# wasi-sdk comes from the workspace's build/toolchain (toolchain/download_dependencies.sh native, run here first).
set -euo pipefail
if [ $# -lt 1 ]; then
    echo "usage: $0 <solution directory> [extra cmake arguments]" >&2
    exit 2
fi
HERE=$(cd "$(dirname "$0")" && pwd)
TOOLCHAIN=${TOOLCHAIN:-$HERE/../../build/toolchain}
SOLUTION=$(cd "$1" && pwd)
shift
bash "$HERE/../../toolchain/download_dependencies.sh" native
mkdir -p "$SOLUTION/build"
cmake -S "$SOLUTION" -B "$SOLUTION/build" -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN/wasi-sdk/share/cmake/wasi-sdk-p1.cmake" -DVIOBENCH_NATIVE_DIR="$HERE" "$@" > "$SOLUTION/build/cmake.log" 2>&1 || (cat "$SOLUTION/build/cmake.log"; exit 1)
cmake --build "$SOLUTION/build"
ls -la "$SOLUTION"/build/*.wasm
