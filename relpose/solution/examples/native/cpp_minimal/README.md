# Native (C++) solution template

The solution ABI implemented directly as a `wasm32-wasip1` reactor: `#include "relpose_abi.h"` and define `abi_version`,
`alloc`, `init`, `step` (see `relpose/solution/runtime/native/`). No Python runtime: the module is ~40 KB and a step costs a few
thousand fuel units of overhead instead of the interpreter's.

```bash
relpose/solution/runtime/native/build.sh relpose/solution/examples/native/cpp_minimal         # cmake with wasi-sdk (downloaded into build/toolchain) → build/solution.wasm
relpose check relpose/solution/examples/native/cpp_minimal/build/solution.wasm
relpose manifest relpose/solution/examples/native/cpp_minimal/build/solution.wasm --channels 3   # optional: ask for RGB (or --resolution 640x480)
```

`CMakeLists.txt` is two lines of substance: `include(${RELPOSE_NATIVE_DIR}/relpose_reactor.cmake)` and
`relpose_solution(solution solution.cpp)`. Configured with the host compiler instead (`cmake -S . -B build-host && cmake --build build-host`)
the same call builds `build-host/debug`, which replays a `relpose feed` directory through your `init`/`step` under gdb
(`docs/DEBUGGING.md`). Rules of the sandbox, OpenCV linking and Rust notes: `relpose/solution/runtime/native/README.md`.
