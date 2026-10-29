#!/bin/sh
# python.wasm REPL or script runner; the current directory is /work in the guest
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD=${BUILD:-$HERE/../../../build/runtime}
exec "${WASMTIME:-$HERE/../../../build/toolchain/wasmtime/wasmtime}" run --allow-precompiled --dir "$PWD::/work" --env PYTHON_BASIC_REPL=1 "$BUILD/dist/python.cwasm" "$@"
