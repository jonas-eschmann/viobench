# The Python runtime (`runtime.wasm`)

`runtime.wasm` is CPython 3.14 with numpy and OpenCV compiled to `wasm32-wasip1` as a *reactor* that implements the
solution ABI (`docs/ABI.md`). `python -m eecs116 bundle <dir>` appends your directory (zipped) as a custom section;
nothing is compiled. At `init` the embedded shim (`runtime/py/_eecs116.py`) installs an importer over that zip, imports
`solution:Solution` and constructs it with the config; each `step` wraps the raw buffers in numpy arrays and calls
`Solution.step`.

## What is inside

| | |
|---|---|
| Python | 3.14, full stdlib except `zlib`, `bz2`, `lzma`, `ssl`, `ctypes`, `sqlite3`, `tkinter`, `ensurepip`, `venv` |
| numpy | 2.3 — scalar build (no SIMD, no BLAS): large `@` are slow; push heavy arithmetic into OpenCV |
| OpenCV | 4.12 `core`, `imgproc`, `features2d`, `flann`, `calib3d`, `video`, `dnn` (ONNX import), SIMD128 kernels; no `imgcodecs`/`highgui` |
| threads | none (`cv2.setNumThreads` is a no-op, `threading` works but runs one thread at a time) |
| files / network | none: the module gets no directory and no sockets; data files travel in the bundle |
| clocks | `time.perf_counter()` works (host monotonic clock) but fuel is the metric |
| randomness | `os.urandom`/`random` are seeded from host entropy: seed them yourself for reproducible fuel numbers |

## Rules of the sandbox

- `init` runs once per episode; a fresh instance is created for every episode (no state leaks between episodes).
- Exceptions in `Solution.__init__`/`step` are caught by the shim: traceback to stderr (shown by `check` and kept in
  the evaluation logs), status `-1`, the episode continues. Ten consecutive failures abandon the episode.
- `sys.exit()` is an exception like any other. `os._exit()`, a C-level `abort()` or a memory fault trap the instance:
  episode failed.
- Running out of fuel in one `step` (`Limits.step_fuel`) or exceeding the wall-clock deadline traps: episode failed.
  Budget your per-frame work; `check` prints fuel per step.
- Memory is capped (`Limits.memory_bytes`): a too-large allocation raises `MemoryError` (recoverable).
- The frame array handed to `step` is a read-only view into a buffer the harness overwrites with the next frame; copy
  it (`frame.copy()`) if you keep it. OpenCV accepts read-only inputs.
- `print` output goes to the log; keep it light, host calls are free in fuel but cost wall time.

## Bundle layout

```
my_solution/
  solution.py          class Solution (or a package solution/__init__.py)
  helpers/...          any modules and packages; relative imports work; __file__ is /bundle/<path>
  net.onnx, w.npz      data files: `import eecs116_bundle as bundle; bundle.read_bytes("net.onnx")`, `np.load(bundle.open("w.npz"))`
  eecs116.json         optional: {"entry": "mypkg.estimator:Estimator"}
```

Zips are stored uncompressed (there is no zlib in the guest), so `np.savez` rather than `np.savez_compressed`.
`python -m eecs116 unbundle solution.wasm -o dir` extracts a bundle again.

## Developing without rebundling

`python -m eecs116 check runtime.wasm --dir my_solution` mounts the directory into the guest (read-only) instead of a
bundle, so edits are picked up immediately and tracebacks show real file paths. `evaluate ... --dir my_solution`
works the same way with `runtime.wasm`.

## Measuring

`check` prints fuel and wall time per step; `evaluate` reports medians and 95th percentiles per episode. Fuel is
deterministic: the same code on the same frames consumes the same fuel on any machine (the first few steps of an
instance cost more while the interpreter warms up). Rough calibration on this build: the empty solution costs ~0.15 M
fuel per step, IMU dead reckoning in numpy ~3.6 M, the KLT/essential-matrix baseline ~150 M (≈ 8 ms native), the tiny
CNN through `cv2.dnn` ~12 M; 1 s of wall time corresponds to roughly 2 × 10¹⁰ fuel.

## Native solutions

C, C++ and Rust modules implement the same exports directly (`examples/cpp-minimal`); the bundle mechanism is only
needed by the Python runtime. The harness treats both identically.

## Building the runtime (course staff)

`./build_docker_image.sh` once, then `./build_wasm_python.sh` (`wasm-python/build.sh` in the toolchain container,
10–15 min): produces `build/wasm-python/dist/python.wasm` (command-line Python for experiments: `wasm-python/python.sh`)
and `runtime.wasm` (the reactor). Patches, module lists and the frozen-stdlib mechanism are documented in
`wasm-python/README.md`. `tests/` exercises the runtime through the harness (`pytest`).
