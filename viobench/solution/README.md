# viobench/solution: building a solution

A solution is a `wasm32-wasip1` reactor exporting the ABI of [`docs/ABI.md`](../../docs/ABI.md)
(`runtime/native/viobench_abi.h` as a C header). What runs inside is up to you; nothing here is required.

This directory is the checkout's *workspace*: `viobench init python|native <dir>` copies `toolchain/` and `runtime/` into
a new one, with a minimal example as `solution/`, for you to change. Every build goes to `build/` in the workspace
(here `viobench/solution/build`); the commands find the workspace of a solution directory by walking up from it.

| | |
|---|---|
| [`toolchain/`](toolchain/) | `download_dependencies.sh`: wasi-sdk, wasmtime and the pinned CPython, numpy and OpenCV sources into `build/toolchain`; both `build.sh` scripts run it (the native one for wasi-sdk only) |
| [`runtime/python/`](runtime/python/) | the Python runtime: `build.sh` (`viobench build-runtime`) builds `runtime.wasm`, `viobench bundle` embeds your directory into it |
| [`runtime/native/`](runtime/native/) | C, C++, Rust: `viobench_abi.h`, a CMake snippet, `build.sh` (what `viobench bundle` runs on a directory with a `CMakeLists.txt`) |
| [`examples/`](examples/) | complete solutions; copy one to start |
| [`workspace.py`](workspace.py) | `viobench init`, finding a workspace, running its builds |
