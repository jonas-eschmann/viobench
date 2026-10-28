# relpose: compute-aware relative pose estimation

This challenge is about estimating the pose of a flying camera relative to its first frame from camera frames + IMU data. A submission is one WebAssembly module that bundles everything needed to run it. We are unopinionated about what runs inside it. The evaluator meters compute deterministically (Wasmtime fuel) and places every submission on an accuracy-vs-compute curve.

| directory | role |
|---|---|
| [`relpose/tools/`](relpose/tools/) | host side (the importable package): `abi` (the contract), `datasets` (episodes, producers), `runner`, `server`, `cli` |
| [`relpose/solution/`](relpose/solution/) | guest side: how a solution `.wasm` is made, by you, `toolchain/`, `runtime/{native,python}`, `examples/{native,python}` |
| [`docs/`](docs/) | [TASK](docs/TASK.md) (what to estimate, scoring), [STUDENT_GUIDE](docs/STUDENT_GUIDE.md), [DEBUGGING](docs/DEBUGGING.md), contracts: [ABI](docs/ABI.md), [RUNTIME](docs/RUNTIME.md) |
| [`tests/`](tests/) | pytest suite driving a real `runtime.wasm` through the harness |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,data]"
relpose/solution/runtime/python/build.sh   # downloads the toolchain into build/toolchain; 10-15 min once → build/runtime/dist/runtime.wasm
pytest
relpose bundle relpose/solution/examples/python/klt_imu -o klt.wasm && relpose check klt.wasm
relpose evaluate klt.wasm data/val -o report-klt                                 # episodes on disk
relpose evaluate klt.wasm --simulate 8 -o report-klt-sim                          # fresh episodes from HyperDrone (GPU)
relpose leaderboard report-* --plot leaderboard.png
```

`relpose --help` lists the commands by role: solution (`bundle`, `unbundle`), runner (`check`, `evaluate`,
`leaderboard`, `debug`, `score`), datasets (`simulate`, `feed`). Evaluation needs only wasmtime: no
Docker, no GPU. Debugging happens outside the sandbox, in your own process: `docs/DEBUGGING.md`.
