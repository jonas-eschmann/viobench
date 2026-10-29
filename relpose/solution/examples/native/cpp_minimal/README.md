# cpp_minimal

The ABI in C++ without the Python runtime: about 40 KB, a few thousand fuel units of overhead per step.

```bash
relpose/solution/runtime/native/build.sh relpose/solution/examples/native/cpp_minimal    # → build/solution.wasm
relpose check relpose/solution/examples/native/cpp_minimal/build/solution.wasm
```

Building, debugging on the host, OpenCV and Rust: `relpose/solution/runtime/native/README.md`.
