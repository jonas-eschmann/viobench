#!/usr/bin/env bash
# builds the toolchain image (wasi-sdk, wasmtime, pinned CPython/numpy/OpenCV sources) from Dockerfile, always as relpose-toolchain
set -euo pipefail
cd "$(dirname "$0")"
docker build --platform linux/amd64 -t relpose-toolchain .
