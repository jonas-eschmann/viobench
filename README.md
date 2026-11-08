# viobench: compute-aware relative state estimation

Estimate a flying camera's pose relative to its first frame, and its twist, from camera frames and IMU data. A submission is one
WebAssembly module; the evaluator meters its compute deterministically (Wasmtime fuel) and places it on an
accuracy-vs-compute curve: Sim(3)-aligned trajectory error versus maximum fuel per frame.

| | |
|---|---|
| [`viobench/tools/`](viobench/tools/) | host side: `abi`, `datasets`, `runner`, `server`, `cli` |
| [`viobench/episodes/`](viobench/episodes/__init__.py) | the published episode sets, as conta hashes: `public` is what `evaluate` runs by default |
| [`viobench/solution/`](viobench/solution/) | guest side: toolchain, runtimes, examples; what `viobench init` copies into a workspace |
| [`docs/`](docs/) | [TASK](docs/TASK.md), [STUDENT_GUIDE](docs/STUDENT_GUIDE.md), [DEBUGGING](docs/DEBUGGING.md), [ABI](docs/ABI.md), [RUNTIME](docs/RUNTIME.md) |
| [`tests/`](tests/) | pytest suite |

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
viobench build-runtime                                          # viobench/solution/build/runtime/dist/runtime.wasm, 10–15 min once
pytest
viobench bundle viobench/solution/examples/python/klt_imu -o klt.wasm && viobench check klt.wasm
viobench evaluate klt.wasm -o report-klt                        # the public set, downloaded once into ~/.cache/conta
viobench leaderboard report-* --plot leaderboard.png
```

`viobench --help` lists all commands. Evaluation needs only wasmtime: no Docker, no GPU.
