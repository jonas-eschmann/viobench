# Native (C++) solution template

The solution ABI implemented directly as a `wasm32-wasip1` reactor: `#include "relpose_abi.h"` and define `abi_version`,
`alloc`, `init`, `step` (see `relpose/solution/runtime/native/`). No Python runtime: the module is ~40 KB and a step costs a few
thousand fuel units of overhead instead of the interpreter's.

```bash
relpose/solution/runtime/native/build.sh relpose/solution/examples/native/cpp_minimal         # cmake inside the toolchain image → build/solution.wasm
relpose check relpose/solution/examples/native/cpp_minimal/build/solution.wasm
relpose manifest relpose/solution/examples/native/cpp_minimal/build/solution.wasm --channels 3   # optional: ask for RGB (or --resolution 640x480)
```

`CMakeLists.txt` is three lines of substance: `add_executable`, `include(${RELPOSE_NATIVE_DIR}/relpose_reactor.cmake)`,
`relpose_solution(solution)`. Rules of the sandbox, OpenCV linking and Rust notes: `relpose/solution/runtime/native/README.md`.
