# cpp_minimal

The ABI in C++ without the Python runtime: about 40 KB, a few thousand fuel units of overhead per step.

```bash
viobench bundle <this directory> -o solution.wasm
viobench check solution.wasm
```

Building, debugging on the host, OpenCV and Rust: `runtime/native/README.md` in the workspace.
