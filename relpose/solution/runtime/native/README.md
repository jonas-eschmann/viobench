# Native solutions (C, C++, Rust)

A solution is any `wasm32-wasip1` reactor that exports the ABI in `relpose_abi.h` (`abi_version`, `alloc`, `init`,
`step`, plus `memory`). No Python runtime is involved; the module is typically tens of kilobytes and a step costs a
few thousand fuel units of overhead instead of the interpreter's.

```bash
relpose/solution/runtime/native/build.sh relpose/solution/examples/native/cpp_minimal        # downloads wasi-sdk 34 into build/toolchain (once), cmake → relpose/solution/examples/native/cpp_minimal/build/solution.wasm
relpose check relpose/solution/examples/native/cpp_minimal/build/solution.wasm
relpose manifest relpose/solution/examples/native/cpp_minimal/build/solution.wasm --resolution 640x480   # optional opt-in
```

Your own project: copy `relpose/solution/examples/native/cpp_minimal`, keep `#include "relpose_abi.h"` and the two CMake lines
(`include(${RELPOSE_NATIVE_DIR}/relpose_reactor.cmake)`, `relpose_solution(<target> <sources...>)`).

**Debugging on the host** (`docs/DEBUGGING.md`): the same `relpose_solution(...)` call, configured with your host compiler
instead of the wasi-sdk toolchain, builds an executable `debug` = your sources + `relpose_debug_main.c` (`-g -O1`) that
replays one episode's inputs through `init`/`step` exactly like the sandbox and writes `estimate.tum`:

```bash
relpose feed build/datasets/public/default-val/<name> -o build/feed      # raw arrays: config.json, frames.u8, imu.f64, ...
cmake -S relpose/solution/examples/native/cpp_minimal -B build-host && cmake --build build-host
build-host/debug build/feed estimate.tum && relpose score build/datasets/public/default-val/<name> estimate.tum
gdb --args build-host/debug build/feed
```

Rules of the sandbox (`docs/RUNTIME.md` lists the Python-specific ones): `exit()`, `abort()` and uncaught C++ exceptions
trap and end the episode, return a negative status instead; stdout/stderr are captured into the evaluation log (make
them unbuffered, a reactor never exits); the frame and IMU buffers are reused next step.

**OpenCV from C++**: the Python-runtime build leaves OpenCV's static libraries and headers in `build/runtime/opencv-build`
(`lib/libopencv_*.a`, `3rdparty/lib/*.a`, `install/include` after `cmake --install`). Compile with
`-fwasm-exceptions -mllvm -wasm-use-legacy-eh=false`, link in a separate step with `-fwasm-exceptions -lc++ -lc++abi -lunwind`;
OpenCV needs the `eh` libc++ variant, which `-fwasm-exceptions` selects.

**Rust**: `crate-type = ["cdylib"]`, target `wasm32-wasip1`, `#[no_mangle] pub extern "C" fn step(t: f64, frame: *const u8,
imu: *const f64, n: i32, out: *mut f64) -> i32` and friends; there is no `_initialize` then, the harness only calls it
when present.
