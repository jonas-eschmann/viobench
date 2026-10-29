# Native solutions (C, C++, Rust)

Implement the exports of `relpose_abi.h` (`docs/ABI.md`). Start from `relpose init native <workspace>` (the
`examples/native/cpp_minimal` example as `solution/`); its `CMakeLists.txt` needs only
`include(${RELPOSE_NATIVE_DIR}/relpose_reactor.cmake)` and `relpose_solution(<target> <sources...>)`.

```bash
relpose bundle <dir> -o solution.wasm   # runtime/native/build.sh: wasi-sdk (fetched into the workspace's build/toolchain) → <dir>/build/, copied to -o
relpose check solution.wasm
```

The same CMake call configured with the host compiler builds `build-host/debug` (`docs/DEBUGGING.md`).

- `exit()`, `abort()` and uncaught C++ exceptions trap: return `-1` instead. Make stdout and stderr unbuffered; a
  reactor never exits.
- **Other libraries**: cross-build them with the workspace's `build/toolchain/wasi-sdk` (fetch their sources in
  `toolchain/download_dependencies.sh` if you like) and link them in your `CMakeLists.txt`.
- **OpenCV**: the Python runtime build leaves static libraries and headers in `build/runtime/opencv-build`. Compile
  with `-fwasm-exceptions -mllvm -wasm-use-legacy-eh=false`; link with `-fwasm-exceptions -lc++ -lc++abi -lunwind`.
- **Rust**: `crate-type = ["cdylib"]`, target `wasm32-wasip1`, `#[no_mangle] pub extern "C" fn step(t: f64, frame:
  *const u8, imu: *const f64, n: i32, out: *mut f64) -> i32` and the other exports.
