#!/usr/bin/env bash
# builds the toolchain image from Dockerfile, always as eecs116-wasm-python
set -euo pipefail
cd "$(dirname "$0")"
docker build --platform linux/amd64 -t eecs116-wasm-python .
