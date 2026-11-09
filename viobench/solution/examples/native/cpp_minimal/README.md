# cpp_minimal

The smallest valid C++ baseline: ignores the config, image and IMU and writes only the identity pose and zero twist
(all 13 required output values). It has no logging, frame copies, sensor processing or persistent state.
Use `viobench check` to measure the per-step fuel for this constant output; initialization fuel is reported separately.

```bash
viobench bundle <this directory> -o solution.wasm
viobench check solution.wasm
```

Building, debugging on the host, OpenCV and Rust: `runtime/native/README.md` in the workspace.
