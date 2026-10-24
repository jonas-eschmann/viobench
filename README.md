# relpose — compute-aware relative pose estimation (EECS 116/216 final project)

Students estimate the pose of a flying camera relative to its first frame from camera frames + IMU. A submission is
one WebAssembly module; we are unopinionated about what runs inside it. The evaluator meters compute deterministically
(Wasmtime fuel) and places every submission on an accuracy-vs-compute curve.

```
 examples/ ──submission/ (Python runtime + relpose bundle | native toolchain)──▶ solution.wasm
                                                                                     │
 datasets/ recipes ──relpose simulate (HyperDrone, GPU) / relpose convert-euroc──▶ episodes on disk
                                                                                     ▼
                                              relpose evaluate (wasmtime sandbox) ──▶ report.json ──▶ relpose leaderboard
                                              relpose.server: upload → the same evaluate on hidden episodes → leaderboard
```

| directory | role |
|---|---|
| [`docs/`](docs/) | the task and scoring ([TASK](docs/TASK.md)), guides for [students](docs/STUDENT_GUIDE.md) and [instructors](docs/INSTRUCTOR_GUIDE.md), the contracts: [ABI](docs/ABI.md), [runtime](docs/RUNTIME.md), [data](docs/DATA.md) |
| [`submission/`](submission/) | how a solution `.wasm` is made, by you: one toolchain image, a reference Python flavour (CPython + numpy + OpenCV runtime that `relpose bundle` embeds your directory into) and a reference native flavour (C/C++/Rust against `relpose_abi.h`) |
| [`relpose/`](relpose/) | the host toolkit, `pip install -e .`: `abi` (the contract), `packaging` (directory ↔ `.wasm`), `evaluation` (`.wasm` → numbers), `datasets` (episodes: storage, generation, real data), `server` (submissions) |
| [`examples/`](examples/) | complete solutions in both flavours: identity, IMU dead reckoning, KLT + essential matrix + IMU, a tiny CNN via `cv2.dnn`, a C++ reactor |
| [`datasets/`](datasets/) | the recipes that produce the published and hidden episode sets (the data itself lives in `build/datasets/`) |
| [`tests/`](tests/) | pytest suite mirroring the package; drives a real `runtime.wasm` through the harness |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,data]"
submission/toolchain/build-image.sh && submission/python/build.sh      # Docker; 10-15 min once → build/runtime/dist/runtime.wasm
pytest                                                                 # ~50 tests through the harness
relpose bundle examples/python/klt_imu -o klt.wasm && relpose check klt.wasm
relpose evaluate klt.wasm build/datasets/public/default-val -o report-klt
relpose leaderboard report-* --plot leaderboard.png
```

`relpose --help` lists the commands by role: packaging (`bundle`, `unbundle`, `manifest`), evaluation (`check`,
`evaluate`, `leaderboard`), datasets (`simulate`, `convert-euroc`). Evaluation needs only wasmtime: no Docker, no GPU.

## Baselines

Public validation sets (10 s episodes, stored at 640 × 480, consumed at 320 × 240 grayscale @ 20 Hz; median ATE over
episodes in metres, median fuel per frame):

| solution | default | blur | lowlight | oversaturated | real (EuRoC V1_01) | fuel/frame |
|---|---|---|---|---|---|---|
| `examples/python/dead_reckoning` (IMU only) | 18.8 | 8.2 | 14.1 | 18.5 | 30.8 | 3.6 M |
| `examples/python/klt_imu` (KLT + essential matrix + IMU) | 4.3 | 4.1 | 4.1 | 4.2 | 4.8 | 140–600 M |
| `examples/python/tiny_cnn` (CNN via `cv2.dnn`, trained on 8 episodes) | 1.2 | 1.2 | 1.4 | 2.3 | 3.8 | 12 M |

A submission opts into RGB and/or 640 × 480 through its manifest (`solution.json` or `relpose manifest`); the harness
resamples every episode per submission.
