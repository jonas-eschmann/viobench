# Python runtime (`runtime.wasm`)

CPython with numpy and OpenCV as a `wasm32-wasip1` reactor implementing `docs/ABI.md`. `relpose bundle <dir> -o x.wasm`
embeds your directory into it as a zip; `relpose unbundle` extracts it again.

## Contents

| | |
|---|---|
| Python | 3.14 stdlib without `zlib`, `bz2`, `lzma`, `ssl`, `ctypes`, `sqlite3`, `tkinter` |
| numpy | 2.3, scalar build (no SIMD, no BLAS): large `@` are slow, use OpenCV |
| OpenCV | 4.12 `core`, `imgproc`, `features2d`, `flann`, `calib3d`, `video`, `dnn` (ONNX); no `imgcodecs`, `highgui` |
| threads, files, network | none; data files travel in the bundle |
| randomness | seeded from host entropy: seed it yourself for reproducible fuel |

## Bundle

```
my_solution/
  solution.py      class Solution (or solution/__init__.py)
  helpers/...      modules and packages; __file__ is /bundle/<path>
  net.onnx, w.npz  data: import solution_bundle as bundle; bundle.read_bytes("net.onnx"), np.load(bundle.open("w.npz"))
```

The zip is uncompressed (no zlib in the guest): save arrays with `np.savez`, not `np.savez_compressed`.

## Behaviour

- An exception, `sys.exit()` included, is an error: the traceback goes to stderr and the episode fails.
- `os._exit()`, `abort()` and memory faults trap.
- `MemoryError` past the memory cap can be caught.
- `frame` is a read-only view that the next step overwrites: `frame.copy()` to keep it.
- `print` goes to the log.

## Fuel

Deterministic for the same code and input; the first steps of an instance cost more. Per step at 320 × 240: empty
solution 0.09 M, numpy dead reckoning 4.1 M, tiny CNN via `cv2.dnn` 14 M, KLT baseline 50–130 M depending on the
image (p95 up to 500 M).

## Building

`relpose/solution/runtime/python/build.sh` (10–15 min, once) writes `build/runtime/dist/runtime.wasm` and `python.wasm`
(a REPL). Only the ABI is fixed: change the runtime freely (`relpose/solution/runtime/python/README.md`).
