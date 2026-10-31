# The Python runtime: build

`build.sh` (what `viobench build-runtime` runs: `toolchain/download_dependencies.sh`, then `cpython/build-wasm.sh`;
10–15 min) writes two modules to the workspace's `build/runtime/dist/`:

- `runtime.wasm`: CPython 3.14.4, numpy 2.3.4 and OpenCV 4.12.0 as a reactor implementing `docs/ABI.md`;
- `python.wasm`: the same as a command-line interpreter.

Both are single-threaded and need no directory: the stdlib and numpy's Python code are frozen in. OpenCV uses SIMD128
and wasm exceptions (wasi-sdk 34 `eh` runtime, Wasmtime ≥ 50).

```bash
runtime/python/build.sh                 # or viobench build-runtime
runtime/python/cpython/test.sh          # numpy's tests inside python.wasm
runtime/python/cpython/python.sh        # a REPL
```

Pure-Python packages need no rebuild: copy them into your solution directory. Extension modules are cross-built in
`cpython/build-wasm.sh` and linked in like numpy and OpenCV (below).

| file | role |
|---|---|
| `reactor.c` | the ABI exports (with `../native/viobench_abi.h`) and the 32-byte bundle descriptor that `viobench bundle` patches |
| `guest/viobench_guest.py` | frozen shim: imports the bundle, loads `Solution`, glues `step` |
| `cpython/build-wasm.sh`, `patches/`, `scripts/` | the cross-build |

Overrides (paths in the workspace): `TOOLCHAIN` (default `build/toolchain`), `SRC` (pristine sources), `WASI_SDK_PATH`, `VENV`, `WASMTIME`,
`BUILD` (default `build/runtime`). The patched sources in `build/runtime/src` are plain copies:
`diff -ru build/toolchain/src/numpy build/runtime/src/numpy` shows the changes; delete `build/runtime/src/<project>`
after editing a patch.

## How it links

- WASI has no dynamic loading. `scripts/collect_numpy.py` archives each extension module's objects, and
  `scripts/link.py` links them into CPython with a generated `main.c` that registers each module in the inittab under
  its dotted name (`numpy._core._multiarray_umath`).
- `scripts/freeze.py` compiles the `.py` files into one blob, embedded with `#embed` and installed as
  `PyImport_FrozenModules`. Tracebacks show files and lines but no source.
- `link.py --reactor reactor.c` links with `-mexec-model=reactor`: `_initialize` runs the C++ static constructors, and
  `reactor.c` starts the interpreter in its `frame_width` export (isolated config, unbuffered stdio, fixed hash seed).

| patch | reason |
|---|---|
| numpy `npy_cpu.h` | wasm CPU detection only under Emscripten |
| numpy `random/meson.build`, `static_link_rename.h` | `mtrand`'s copy of `distributions.c` collides with the other once linked into one binary |
| OpenCV `cv_cpu_dispatch.h`, `intrin_wasm.hpp` | wasm SIMD gated on Emscripten |
| OpenCV `stereo_geom.cpp`, `cv2.cpp` | missing `<iterator>` with libc++ 23 |

Left out: the `zlib`, `bz2`, `lzma`, `ssl`, `ctypes`, `sqlite3` modules; stdlib `test`, `idlelib`, `tkinter`,
`ensurepip`, `venv`, `pydoc_data`; numpy `tests`, `f2py`; OpenCV `imgcodecs`, `highgui`, `objdetect`. `dnn` needs
`BUILD_opencv_dnn=ON` explicitly (thread support is disabled). numpy never warns on overflow or invalid values (wasm
has no floating-point flags), and its default integer is 32-bit.

Compile `.cwasm` files on the machine that runs them, and never accept them from others: `--allow-precompiled` skips
validation.
