# Native (C++) solution template

Implements the solution ABI (`docs/ABI.md`) directly as a `wasm32-wasip1` reactor: the four exports `abi_version`,
`alloc`, `init`, `step` plus `memory`. No Python runtime, so the module is ~100 KB and a step costs a few thousand
fuel units of overhead instead of the interpreter's.

Build with the course toolchain image (wasi-sdk 34):

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" -w /work/examples/cpp-minimal eecs116-wasm-python \
    bash -c 'cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=/opt/wasi-sdk/share/cmake/wasi-sdk-p1.cmake && cmake --build build'
python -m eecs116 check examples/cpp-minimal/build/solution.wasm
```

Notes:
- `exit()`, `abort()` and uncaught exceptions trap and end the episode; return a negative status instead.
- `stdout`/`stderr` are captured into the evaluation log; the template makes them unbuffered.
- Using OpenCV from C++: link the static libraries the runtime build produces (`build/wasm-python/opencv-build/lib/*.a`
  and `3rdparty/lib/*.a`) with the include directory of that build. Compile with
  `-fwasm-exceptions -mllvm -wasm-use-legacy-eh=false`, link in a separate step with `-fwasm-exceptions -lc++ -lc++abi -lunwind`.
- Rust works the same way: `crate-type = ["cdylib"]`, target `wasm32-wasip1`, `#[no_mangle] pub extern "C"` for the exports
  (there is no `_initialize` then; the harness only calls it when present).
