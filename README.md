# relpose: compute-aware relative pose estimation

Estimate a flying camera's pose relative to its first frame from camera frames and IMU data. A submission is one
WebAssembly module; the evaluator meters its compute deterministically (Wasmtime fuel) and places it on an
accuracy-vs-compute curve.

| | |
|---|---|
| [`relpose/tools/`](relpose/tools/) | host side: `abi`, `datasets`, `runner`, `server`, `cli` |
| [`relpose/solution/`](relpose/solution/) | guest side: toolchain, runtimes, examples; what `relpose init` copies into a workspace |
| [`docs/`](docs/) | [TASK](docs/TASK.md), [STUDENT_GUIDE](docs/STUDENT_GUIDE.md), [DEBUGGING](docs/DEBUGGING.md), [ABI](docs/ABI.md), [RUNTIME](docs/RUNTIME.md) |
| [`tests/`](tests/) | pytest suite |

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,data]"
relpose build-runtime                                          # relpose/solution/build/runtime/dist/runtime.wasm, 10–15 min once
pytest
relpose bundle relpose/solution/examples/python/klt_imu -o klt.wasm && relpose check klt.wasm
relpose evaluate klt.wasm data/val -o report-klt
relpose leaderboard report-* --plot leaderboard.png
```

`relpose --help` lists all commands. Evaluation needs only wasmtime: no Docker, no GPU.
