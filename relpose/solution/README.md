# relpose/solution — guest side: how a solution `.wasm` is made (by you)

A submission is a `wasm32-wasip1` reactor exporting the ABI in [`docs/ABI.md`](../docs/ABI.md)
(`runtime/native/relpose_abi.h` is the same contract as a header). What runs inside is up to you; nothing here is a required
dependency. We ship two reference recipes and the toolchain image to build them. No binaries are distributed.

| | |
|---|---|
| [`toolchain/`](toolchain/) | one Docker image for both flavours: wasi-sdk 34, wasmtime 50 and the pinned CPython 3.14 / numpy 2.3 / OpenCV 4.12 sources. `build-image.sh` once per machine. |
| [`runtime/python/`](runtime/python/) | **Python flavour.** `build.sh` (10–15 min, once) compiles CPython + numpy + OpenCV (incl. `dnn` for ONNX) into `build/runtime/dist/runtime.wasm`, a reactor with a bundle descriptor. `relpose bundle my_dir -o my.wasm` then embeds your directory into it as a data segment: a self-contained module, no compiler involved per edit. `reactor.c` is the ABI glue, `guest/relpose_guest.py` the frozen shim that imports your `solution:Solution`, `cpython/` the cross-build. |
| [`runtime/native/`](runtime/native/) | **Native flavour.** `relpose_abi.h`, a CMake snippet and `build.sh <dir>` that compiles a C/C++ directory inside the image; Rust notes. Modules are tens of kilobytes and skip the interpreter's fuel. |

| [`examples/`](examples/) | complete solutions in both flavours; copy one to start (`examples/README.md` has the table). |

Either way the result goes through `relpose check` (synthetic frames) and `relpose evaluate` (episodes) unchanged: the
evaluator does not know which flavour it is running.

Opt-ins (RGB frames, 640 × 480) are declared in the manifest: `solution.json` next to `solution.py` for the Python
flavour, `relpose manifest x.wasm --channels 3 --resolution 640x480` for anything else.
