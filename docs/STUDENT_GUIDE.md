# Student guide

## 1. Setup

Linux or macOS; on Windows use WSL2 and clone into the WSL filesystem (not under `/mnt/c`), in a path without spaces. You need git,
curl, CMake, a C compiler, make and Python 3.10–3.14 with `venv` (Ubuntu/WSL: `sudo apt install build-essential cmake curl git
python3-venv xz-utils`; macOS: `xcode-select --install`, `brew install cmake` and a Python from Homebrew or python.org, the
Command Line Tools' python3 is too old). The first `build.sh` downloads everything else into `build/toolchain`.

```bash
git clone <repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate && pip install -e ".[data]"
relpose/solution/runtime/python/build.sh       # your Python runtime: build/runtime/dist/runtime.wasm (10-15 min, once per machine)
```

The runtime is yours to build, and to change (`relpose/solution/runtime/python/README.md`): nothing inside the `.wasm` is prescribed,
only the ABI it exports. Writing C/C++ instead? Skip `build.sh` and read `relpose/solution/runtime/native/README.md`. The
validation episodes come from the course page; unpack them to `data/val` (the paths below assume that).

On the lab machines (RTX 5080): `pip install -e ".[data,sim,train]"` adds the HyperDrone generator (first use compiles
for a minute) and PyTorch. There is no pre-rendered training set; you generate what you need:

```bash
relpose simulate -o data/train --episodes 200 --seed 1                         # ~2 min per 8 one-minute episodes (640x480)
relpose simulate -o data/train-short --episodes 64 --seed 900 --duration 15     # any length; the frame count is in the config
```

Read `docs/TASK.md` (what to estimate, how it is scored) and `docs/ABI.md` / `docs/RUNTIME.md` (what the sandbox offers).

## 2. Your first solution

```bash
cp -r relpose/solution/examples/python/dead_reckoning my_solution
relpose bundle my_solution -o my_solution.wasm             # your runtime + your directory in one file; rerun after every edit
relpose check my_solution.wasm                             # synthetic frames, prints status/fuel per step
relpose evaluate my_solution.wasm data/val
relpose evaluate my_solution.wasm --simulate 8                    # fresh episodes, GPU, nothing on disk
```

`check` catches crashes and protocol errors; `evaluate` gives ATE/RPE and fuel per frame on episodes
(`report-my_solution/report.json`, per-episode `estimate.tum` for plotting with `evo`). Tracebacks name your files
`/bundle/<path>`. To step through your code with a debugger, or to iterate without bundling, run it outside the sandbox:
`relpose debug my_solution data/val --max-episodes 1`, see `docs/DEBUGGING.md`.

## 3. Iterate

- Data: generate training data with `relpose simulate` (any amount); validate on the episodes from the course page.
- Debug: `relpose debug my_solution <episodes>` runs your `Solution` in your own Python process, so `breakpoint()`, prints
  and matplotlib work; `trace()` yields per-frame inputs and poses for plots (`docs/DEBUGGING.md`). No fuel there:
  `check`/`evaluate` remain the measure.
- Inspect an episode in Python: `from relpose.tools.datasets import read_episode; ep = read_episode("data/val/<name>")`,
  then `ep.frames`, `ep.imu`, `ep.ground_truth_relative()`, `ep.steps()`.
- Length: episodes are continuous flights of unannounced length (the config does not say): keep per-frame cost flat
  (do not accumulate every frame or grow a map without bound), and expect drift to grow with it. Nor does the config
  say how an episode starts: at rest or already flying.
- Compute: watch the fuel column. Python-level loops over pixels are expensive; OpenCV calls are cheap per pixel; a
  `cv2.dnn` forward of a small CNN costs about as much as one `goodFeaturesToTrack`.
- Learned models: `relpose/solution/examples/python/tiny_cnn` shows PyTorch → ONNX → `cv2.dnn` inside the runtime, with a numpy fallback,
  and `train.py --generate N` trains straight from the simulator.
- Frames: always RGB `(H, W, 3)`, 320 × 240 unless your class declares `frame_width = 640` (then 640 × 480;
  native solutions export `int32_t frame_width(void)`); convert to gray yourself if that is what your method uses
  (`cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)`). More pixels cost fuel in everything you run on them.

## 4. Submit

```bash
relpose bundle my_solution -o my_solution.wasm     # one self-contained file: your runtime + your directory
relpose check my_solution.wasm                     # exactly what the server runs first
curl -F file=@my_solution.wasm -F label="klt+imu v3" -H "X-Token: $TOKEN" https://<server>/submit
```

The server evaluates on its own episodes (not published), posts the result on the
leaderboard and keeps your stderr logs for you (`/submissions/<id>/logs`). Every submission is a point on the
accuracy-vs-fuel curve; your best points form your Pareto front. Quotas: 20 submissions per day, 128 MiB per file.

## 5. Going native

A C++ (or Rust) solution implements four exports and skips the interpreter entirely: `relpose/solution/runtime/native/` (header, CMake
snippet, `build.sh`) and `relpose/solution/examples/native/cpp_minimal`. `build.sh` uses the wasi-sdk in `build/toolchain`; the Python-runtime build leaves
OpenCV's static libraries behind if you want them. Configured with your host compiler, the same `CMakeLists.txt` builds a
`debug` executable that replays an episode (`relpose feed`) through your code under gdb (`docs/DEBUGGING.md`).

## 6. Common problems

| symptom | cause / fix |
|---|---|
| `… was configured at another path`, `CMakeCache.txt directory … is different` | a build directory from the former Docker toolchain (or a moved checkout): delete it once (`build/runtime`, `<solution>/build`) |
| `ModuleNotFoundError: scipy` | not available in the reference runtime; numpy + OpenCV only (`docs/RUNTIME.md`), or build your own |
| `trap OUT_OF_FUEL` | one step exceeded the fuel budget: reduce work per frame (resolution, feature count, iterations) |
| `MemoryError` | the 2 GiB cap; avoid keeping every frame |
| `step returned an unusable pose` | NaN/inf or an all-zero quaternion in the pose: guard divisions and normalise quaternions |
| frames look identical | you kept a view of the frame buffer; `frame.copy()` |
| `np.load` fails on `.npz` | compressed archive: save with `np.savez`, not `np.savez_compressed` |
| the IMU looks "upside down" | the IMU frame need not be gravity-aligned at the start; estimate gravity from the measurements |
