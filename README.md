# EECS 116/216 final project: compute-aware relative pose estimation

Students estimate the pose of a flying camera relative to the first frame from camera frames + IMU, inside a
WebAssembly sandbox that meters compute deterministically, and are placed on an accuracy-vs-compute curve. This
repository is the whole infrastructure: the sandboxed Python runtime, the bundler/harness, the data pipeline on
HyperDrone, the metrics, baselines, and the submission server.

| | |
|---|---|
| Task, scoring, limits | [`docs/TASK.md`](docs/TASK.md) |
| Student walkthrough | [`docs/STUDENT_GUIDE.md`](docs/STUDENT_GUIDE.md) |
| Solution ABI (exports, buffers, status codes, config JSON) | [`docs/ABI.md`](docs/ABI.md) |
| What the Python runtime offers / forbids | [`docs/RUNTIME.md`](docs/RUNTIME.md) |
| Datasets: format, generator, variants, real data | [`docs/DATA.md`](docs/DATA.md) |
| Building the wasm runtime | [`wasm-python/README.md`](wasm-python/README.md) |
| Submission server | [`eecs116/server/README.md`](eecs116/server/README.md) |

## How it fits together

```
 student directory ──bundle──▶ solution.wasm = runtime.wasm + zip (custom section)
                                   │
 episode on disk (EuRoC layout) ───┤ eecs116.Solution: wasmtime, fuel + wall-time per step, memory cap
   frames + IMU + ground truth     ▼
                              report.json: ATE / RPE / rotation, fuel per frame (median, p95), logs
                                   │
 HyperDrone (GPU) ──simulate──▶ default / blur / lowlight / oversaturated episodes      EuRoC ──convert-euroc──▶ real episodes
```

- `runtime.wasm` (`wasm-python/`, `runtime/`): CPython 3.14 + numpy + OpenCV (incl. `dnn` for ONNX) as a wasip1
  reactor exporting `init`/`step`; the student's directory is appended as a custom section, no toolchain needed.
- `eecs116/` (pip package): `bundle`, `check`, `evaluate`, `simulate`, `convert-euroc`, `leaderboard` commands; the
  `Solution` wrapper (wasmtime 50, fuel metering), `metrics`, the `data` package (episode container, ASL-layout I/O,
  camera/ISP model, HyperDrone generator), the FastAPI `server`.
- `examples/`: `python-minimal` (identity), `python-deadreckoning` (IMU only), `python-orb` (KLT + essential matrix +
  IMU scale), `python-tinynn` (PyTorch → ONNX → `cv2.dnn`), `cpp-minimal` (native reactor).
- `tests/`: pytest suite driving `runtime.wasm` through the harness (protocol, traps, bundling, ONNX, metrics, data I/O).

## Quick start (staff)

```bash
./build_docker_image.sh && ./build_wasm_python.sh            # toolchain image, then runtime.wasm (10-15 min, once)
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,data,sim]"
pytest                                                      # 39 tests against build/wasm-python/dist/runtime.wasm
python -m eecs116 simulate -o build/datasets/public/default-val --episodes 16 --seed 2000      # GPU
python -m eecs116 convert-euroc build/datasets/raw/euroc/V1_01_easy -o build/datasets/public/real-sample --max-snippets 3
python -m eecs116 bundle examples/python-orb -o orb.wasm && python -m eecs116 check orb.wasm
python -m eecs116 evaluate orb.wasm build/datasets/public/default-val -o report-orb
python -m eecs116 leaderboard report-* --plot leaderboard.png
```

Baselines on the public validation sets (10 s episodes, 320 × 240 @ 20 Hz; median ATE over episodes in metres,
median fuel per frame; `python -m eecs116 leaderboard` output):

| solution | default | blur | lowlight | oversaturated | real (EuRoC V1_01) | fuel/frame |
|---|---|---|---|---|---|---|
| `python-deadreckoning` (IMU only) | 18.8 | 7.4 | 14.1 | 18.5 | 30.8 | 3.6 M |
| `python-orb` (KLT + essential matrix + IMU) | 4.2 | 4.1 | 4.1 | 4.2 | 4.3 | 140–350 M (3.4 G on low light: noise → thousands of corners) |
| `python-tinynn` (CNN via `cv2.dnn`, trained on 8 episodes) | 1.2 | 1.1 | 1.6 | 2.4 | 3.8 | 12 M |

## Layout

```
Dockerfile, build_*.sh      toolchain image (wasi-sdk 34, wasmtime 50, CPython/numpy/OpenCV sources) and build entry points
wasm-python/                CPython + numpy + OpenCV → wasm32-wasip1 (patches, freeze/link scripts)
runtime/                    reactor.c (ABI exports) and the frozen guest-side shim _eecs116.py
eecs116/                    host package: harness, bundler, metrics, data pipeline, server
examples/                   baselines and templates
docs/                       task, ABI, runtime, data, student guide
tests/                      pytest
build/                      (ignored) wasm artefacts, datasets
```
