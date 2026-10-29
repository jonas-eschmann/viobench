# relpose/solution: building a solution

A solution is a `wasm32-wasip1` reactor exporting the ABI of [`docs/ABI.md`](../../docs/ABI.md)
(`runtime/native/relpose_abi.h` as a C header). What runs inside is up to you; nothing here is required.

| | |
|---|---|
| [`toolchain/`](toolchain/) | `download_dependencies.sh`: wasi-sdk, wasmtime and the pinned CPython, numpy and OpenCV sources into `build/toolchain`; both `build.sh` scripts run it |
| [`runtime/python/`](runtime/python/) | the Python runtime: `build.sh` builds `runtime.wasm`, `relpose bundle` embeds your directory into it |
| [`runtime/native/`](runtime/native/) | C, C++, Rust: `relpose_abi.h`, a CMake snippet, `build.sh` |
| [`examples/`](examples/) | complete solutions; copy one to start |
