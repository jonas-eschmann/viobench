# relpose: compute-aware relative pose estimation

This challenge is about estimating the pose of a flying camera relative to its first frame from camera frames + IMU data. A submission is one WebAssembly module that bundles everything needed to run it. We are unopinionated about what runs inside it. The evaluator meters compute deterministically (Wasmtime fuel) and places every submission on an accuracy-vs-compute curve.

| directory | role |
|---|---|
| [`relpose/tools/`](relpose/tools/) | host side (the importable package): `abi` (the contract), `datasets` (episodes, producers), `runner`, `server`, `cli` |
| [`relpose/solution/`](relpose/solution/) | guest side: how a solution `.wasm` is made, by you, `toolchain/`, `runtime/{native,python}`, `examples/{native,python}` |
| [`docs/`](docs/) | [TASK](docs/TASK.md) (what to estimate, scoring), [STUDENT_GUIDE](docs/STUDENT_GUIDE.md), [DEBUGGING](docs/DEBUGGING.md), [INSTRUCTOR_GUIDE](docs/INSTRUCTOR_GUIDE.md), contracts: [ABI](docs/ABI.md), [RUNTIME](docs/RUNTIME.md), [DATA](docs/DATA.md) |
| [`tests/`](tests/) | pytest suite driving a real `runtime.wasm` through the harness |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,data]"
relpose/solution/runtime/python/build.sh   # downloads the toolchain into build/toolchain; 10-15 min once → build/runtime/dist/runtime.wasm
pytest
relpose bundle relpose/solution/examples/python/klt_imu -o klt.wasm && relpose check klt.wasm
relpose evaluate klt.wasm build/datasets/public/default-val -o report-klt         # episodes on disk
relpose evaluate klt.wasm --simulate 8 --condition blur -o report-klt-blur        # fresh episodes from HyperDrone (GPU)
relpose leaderboard report-* --plot leaderboard.png
```

`relpose --help` lists the commands by role: solution (`bundle`, `unbundle`, `manifest`), runner (`check`, `evaluate`,
`leaderboard`, `debug`, `score`), datasets (`simulate`, `convert-euroc`, `feed`). Evaluation needs only wasmtime: no
Docker, no GPU. Debugging happens outside the sandbox, in your own process: `docs/DEBUGGING.md`.

## Baselines

Public validation sets (one-minute episodes, stored at 640 × 480, consumed at 320 × 240 RGB @ 20 Hz; median ATE
over episodes in metres, median fuel per frame). The numbers below were measured on the earlier 10 s sets with
grayscale input and the examples' former hover-based start, and are being re-measured on the one-minute sets (ATE
grows with the length, dead reckoning's most of all):

| solution | default | blur | lowlight | oversaturated | real (EuRoC V1_01) | fuel/frame |
|---|---|---|---|---|---|---|
| `examples/python/dead_reckoning` (IMU only) | 18.8 | 8.2 | 14.1 | 18.5 | 30.8 | 3.6 M |
| `examples/python/klt_imu` (KLT + essential matrix + IMU) | 4.3 | 4.1 | 4.1 | 4.2 | 4.8 | 140–600 M |
| `examples/python/tiny_cnn` (CNN via `cv2.dnn`, trained on 8 episodes) | 1.2 | 1.2 | 1.4 | 2.3 | 3.8 | 12 M |

Frames are always RGB; a submission opts into 640 × 480 through its manifest (`solution.json` or `relpose manifest`),
and the runner downscales every episode per submission.
