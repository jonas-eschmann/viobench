#!/usr/bin/env bash
# Downloads the toolchain for both solution flavours into the workspace's build/toolchain (Linux, macOS; Windows inside WSL2): wasi-sdk 34, Wasmtime 50,
# a venv with Cython, ninja and pytest, and the pristine CPython 3.14.4, numpy 2.3.4 and OpenCV 4.12.0 sources (build-wasm.sh defaults point here).
# Every archive is SHA256-checked; finished pieces are skipped, so the build scripts run this each time. Run from anywhere.
# With the argument `native` it fetches wasi-sdk only, all a native solution needs (runtime/native/build.sh).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TOOLCHAIN=${TOOLCHAIN:-$HERE/../build/toolchain}

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)
        HOST=x86_64-linux
        WASI_SDK_SHA256=b761e3a0721dbae9c09a0059e5fdb2bf917d1b4a8a7b430fb3b5aafb0984b2c4
        WASMTIME_SHA256=77c19ee0437991f9ac7837b9c269d9c7076744f85dad263fea5214cc2723c7ff ;;
    Linux-aarch64|Linux-arm64)
        HOST=arm64-linux
        WASI_SDK_SHA256=f7e243dff54d60bcc576e94d6166b69f410f2500ae4a9ceef34315be10e77971
        WASMTIME_SHA256=10f9ad1f8539d2e04e041cc622de6a270109ab1e43a6be1f69843c2082166832 ;;
    Darwin-x86_64)
        HOST=x86_64-macos
        WASI_SDK_SHA256=87d27fa8adc68dee59bfbf2e22a6d34ef717c34d6bf1d8af2a56fc929d9ce0eb
        WASMTIME_SHA256=ad14d9fb6a51d1633318426d6df21734127faddc81fe18a3afd70c5964eabeb5 ;;
    Darwin-arm64)
        HOST=arm64-macos
        WASI_SDK_SHA256=9c59398106b417f8f14913380fdf0097a8cc0ff4af9eb3ce0065a859e88d49e9
        WASMTIME_SHA256=323dc098dc0455d60f396c404627e967b7033ffcfa29abfffb243126cf82c9d6 ;;
    *)
        echo "no wasi-sdk/Wasmtime release for $(uname -s) $(uname -m); on Windows, clone the repository inside WSL2 (not under /mnt/c) and build there" >&2
        exit 1 ;;
esac
# release tarballs: the same trees as the v3.14.4 / v2.3.4 (sdist, submodules included) / 4.12.0 tags minus VCS and CI files
CPYTHON_URL=https://www.python.org/ftp/python/3.14.4/Python-3.14.4.tar.xz
CPYTHON_SHA256=d923c51303e38e249136fc1bdf3568d56ecb03214efdef48516176d3d7faaef8
NUMPY_URL=https://files.pythonhosted.org/packages/b5/f4/098d2270d52b41f1bd7db9fc288aaa0400cb48c2a3e2af6fa365d9720947/numpy-2.3.4.tar.gz
NUMPY_SHA256=a7d018bfedb375a8d979ac758b120ba846a7fe764911a64465fd87b8729f4a6a
OPENCV_URL=https://github.com/opencv/opencv/archive/refs/tags/4.12.0.tar.gz
OPENCV_SHA256=44c106d5bb47efec04e531fd93008b3fcd1d27138985c5baf4eafac0e1ec9e9d
REQUIREMENTS=(cython==3.3.0 ninja==1.13.2 pytest==9.1.1 pluggy==1.6.0 iniconfig==2.3.0 packaging==26.3 pygments==2.20.0)

# host tools: curl/tar here, git/make/cc for build-wasm.sh (CPython compiles a native build Python first), cmake for OpenCV and native solutions;
# GNU tar needs xz for .tar.xz, macOS tar has it built in
TOOLS="curl tar git make cc cmake"
if [ "$(uname -s)" = Linux ]; then
    TOOLS="$TOOLS xz"
fi
MISSING=""
for tool in $TOOLS; do
    command -v "$tool" > /dev/null || MISSING="$MISSING $tool"
done
if [ -n "$MISSING" ]; then
    echo "missing:$MISSING (Ubuntu/WSL: sudo apt install build-essential cmake curl git python3-venv xz-utils; macOS: xcode-select --install, brew install cmake)" >&2
    exit 1
fi
if command -v sha256sum > /dev/null; then
    SHA256SUM=sha256sum
else
    SHA256SUM="shasum -a 256"
fi

# fetch <directory> <url> <sha256>: download, verify and unpack without the archive's top-level directory, unless that already happened
fetch() {
    if [ "$(cat "$1/.sha256" 2> /dev/null)" = "$3" ]; then
        return
    fi
    echo "downloading $2"
    rm -rf "$1" "$1.download"
    mkdir -p "$1"
    curl -fL --retry 3 --progress-bar -o "$1.download" "$2"
    actual=$($SHA256SUM "$1.download" | cut -d ' ' -f 1)
    if [ "$actual" != "$3" ]; then
        echo "SHA256 mismatch for $2: expected $3, got $actual" >&2
        exit 1
    fi
    tar -xf "$1.download" -C "$1" --strip-components=1
    rm "$1.download"
    echo "$3" > "$1/.sha256"
}

mkdir -p "$TOOLCHAIN"
TOOLCHAIN=$(cd "$TOOLCHAIN" && pwd)
fetch "$TOOLCHAIN/wasi-sdk" "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-34/wasi-sdk-34.0-$HOST.tar.gz" "$WASI_SDK_SHA256"
if [ "${1:-}" = native ]; then
    exit 0
fi
fetch "$TOOLCHAIN/wasmtime" "https://github.com/bytecodealliance/wasmtime/releases/download/v50.0.0/wasmtime-v50.0.0-${HOST/arm64/aarch64}.tar.xz" "$WASMTIME_SHA256"
fetch "$TOOLCHAIN/src/cpython" "$CPYTHON_URL" "$CPYTHON_SHA256"
fetch "$TOOLCHAIN/src/numpy" "$NUMPY_URL" "$NUMPY_SHA256"
fetch "$TOOLCHAIN/src/opencv" "$OPENCV_URL" "$OPENCV_SHA256"

# OpenCV's configure accepts the cross-built CPython 3.14 only if it is at least as new as the interpreter running the build
PYTHON=""
for candidate in ${PYTHON3:-} python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" > /dev/null && "$candidate" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 14))' 2> /dev/null; then
        # the real interpreter, not an activated venv's symlink to it
        PYTHON=$("$candidate" -c 'import os, sys; print(os.path.realpath(sys.executable))')
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "no Python 3.10-3.14 found (set PYTHON3=/path/to/python3.x); macOS: the Command Line Tools' python3 is 3.9, install one from python.org or Homebrew" >&2
    exit 1
fi
VENV=$TOOLCHAIN/venv
if [ "$(cat "$VENV/.requirements" 2> /dev/null)" != "$PYTHON ${REQUIREMENTS[*]}" ]; then
    echo "creating $VENV with $PYTHON"
    if ! "$PYTHON" -m venv --clear "$VENV"; then
        echo "$PYTHON -m venv failed (Ubuntu/WSL: sudo apt install python3-venv)" >&2
        exit 1
    fi
    "$VENV/bin/python" -m pip install --quiet --no-cache-dir --disable-pip-version-check "${REQUIREMENTS[@]}"
    echo "$PYTHON ${REQUIREMENTS[*]}" > "$VENV/.requirements"
fi
