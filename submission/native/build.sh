#!/usr/bin/env bash
# Builds a native (C/C++) solution directory inside the toolchain image:  submission/native/build.sh examples/native/cpp_minimal
# The directory needs a CMakeLists.txt (see examples/native/cpp_minimal); the module lands in <dir>/build/<target>.wasm.
set -euo pipefail
if [ $# -lt 1 ]; then
    echo "usage: $0 <solution directory> [extra cmake arguments]" >&2
    exit 2
fi
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SOLUTION=$(cd "$1" && pwd)
shift
case "$SOLUTION" in
    "$REPO"/*) INSIDE=/work/${SOLUTION#"$REPO"/} ;;
    *) echo "the solution directory must be inside the repository ($REPO) so the container can see it" >&2; exit 2 ;;
esac
docker run --rm --user "$(id -u):$(id -g)" -v "$REPO:/work" -w "$INSIDE" relpose-toolchain bash -c \
    "mkdir -p build && cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=/opt/wasi-sdk/share/cmake/wasi-sdk-p1.cmake -DRELPOSE_NATIVE_DIR=/work/submission/native $* > build/cmake.log 2>&1 || (cat build/cmake.log; exit 1); cmake --build build"
ls -la "$SOLUTION"/build/*.wasm
