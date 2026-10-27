# relpose/solution/runtime/python — the reference Python runtime

`build.sh` (`relpose/solution/runtime/python/build.sh`, from anywhere; it downloads the toolchain into `build/toolchain` first) produces two modules in `build/runtime/dist/`: `python.wasm`, a self-contained command-line CPython 3.14.4 with numpy 2.3.4 and OpenCV 4.12.0 (core, imgproc, features2d, flann, calib3d, video, dnn with ONNX import, `cv2` bindings), and `runtime.wasm`, the same interpreter linked as a wasi reactor whose exports implement the challenge's solution ABI (`relpose/solution/runtime/python/reactor.c`, `docs/ABI.md`; the guest-side shim `relpose/solution/runtime/python/guest/relpose_guest.py` is frozen in as well). Both are single-threaded, OpenCV uses WASM SIMD128 universal intrinsics and C++ exceptions (wasi-sdk 34 `eh` runtime, Wasmtime ≥ 50), and neither needs a directory at runtime: the stdlib and numpy's Python code are frozen into them.

```bash
relpose/solution/runtime/python/build.sh                    # = relpose/solution/toolchain/download_dependencies.sh && relpose/solution/runtime/python/cpython/build-wasm.sh
relpose/solution/runtime/python/cpython/test.sh             # numpy's own tests inside python.wasm
relpose/solution/runtime/python/cpython/python.sh           # a REPL
```

Layout: `reactor.c` (the ABI exports, includes `../native/relpose_abi.h`), `guest/relpose_guest.py` (frozen shim: bundle
importer, `Solution` loading, step glue), `cpython/` (the cross-build: `build-wasm.sh`, patches, `scripts/{freeze,link,collect_numpy}.py`).

`build-wasm.sh`, `test.sh` and `python.sh` default to the toolchain in `build/toolchain` (`TOOLCHAIN`); override single pieces with `SRC` (directory with the pristine `cpython`, `numpy` and `opencv` sources), `WASI_SDK_PATH`, `VENV` (Cython, ninja, pytest), `WASMTIME` and `BUILD` (default `build/runtime`). The sources are the SHA256-pinned release tarballs (python.org, the numpy sdist, OpenCV's GitHub archive), so the patched copies in `build/runtime/src` are not git repositories: `diff -ru build/toolchain/src/numpy build/runtime/src/numpy` shows what changed, and after editing a patch delete `build/runtime/src/<project>` to have it copied and patched afresh. A full build takes 10–15 minutes with 5 parallel jobs.

WASI has no dynamic loading, so meson/CMake extension-module targets fail at their shared-library link; `scripts/collect_numpy.py` archives each module's objects instead, and `scripts/link.py` relinks CPython's own `python.wasm` link line with a generated `main.c` that registers every module in the inittab under its dotted name (`numpy._core._multiarray_umath`), which `BuiltinImporter` resolves. `scripts/freeze.py` compiles the `.py` files with the native build Python (same bytecode format) into one blob that `frozen.c` pulls in with `#embed`; `main.c` installs it as `PyImport_FrozenModules`, which CPython searches before the filesystem. Frozen modules report `__file__` under `/usr/local/lib/python3.14/` (numpy as `/usr/local/lib/python3.14/numpy/`); tracebacks show file and line but no source text.

| Patch | Reason |
|---|---|
| numpy `npy_cpu.h` | CPU detection knew WASM only under Emscripten |
| numpy `random/meson.build`, `static_link_rename.h` | `mtrand` compiles a `long` copy of `distributions.c` whose symbols collide with the `int64_t` copy once both modules share one binary |
| OpenCV `cv_cpu_dispatch.h`, `intrin_wasm.hpp` | WASM SIMD intrinsics were gated on Emscripten |
| OpenCV `stereo_geom.cpp`, `cv2.cpp` | missing `<iterator>` with libc++ 23 |

Not included: zlib, bz2, lzma, ssl, ctypes, sqlite3 modules; stdlib `test`, `idlelib`, `tkinter`, `ensurepip`, `venv`, `pydoc_data`; numpy `tests`, `f2py`; OpenCV `imgcodecs`/`highgui`/`objdetect` and the `cv2` Python loader package. `dnn` needs OpenCV's bundled protobuf, which compiles unchanged: wasi-sdk 34's libc++ has thread primitives backed by wasi-libc's single-threaded pthread stubs. `OPENCV_DISABLE_THREAD_SUPPORT` turns `dnn` and `python3` off by default, hence the explicit `BUILD_opencv_dnn=ON`/`BUILD_opencv_python3=ON`. WebAssembly has no floating-point exception flags, so numpy never raises or warns on overflow/invalid (`np.errstate`); the default integer is 32-bit. `.cwasm` files are Wasmtime-version- and CPU-specific native code: compile them from `.wasm` on the machine that runs them and never accept them from untrusted sources (`--allow-precompiled` skips validation).

`scripts/link.py --reactor relpose/solution/runtime/python/reactor.c` links the reactor condition (`reactor.c` also holds the 32-byte bundle descriptor that `relpose bundle` patches when it embeds a student directory as a data segment above the initial memory): `-mexec-model=reactor` (wasi-libc's `_initialize` runs the C++ static constructors; the harness calls it once), the generated `main.c` only registers the inittab and frozen table in `wasm_python_register_modules()`, and `reactor.c` initializes the interpreter inside its `init` export (isolated config, unbuffered stdio, fixed hash seed, no site). `tests/` drives the result through the host harness with wasmtime-py.
