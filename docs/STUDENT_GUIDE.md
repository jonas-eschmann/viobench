# Student guide

## Setup

Linux or macOS; on Windows, WSL2 with your files inside the WSL filesystem. Needs git, curl, CMake, a C compiler, make
and Python 3.10–3.14 with venv (Ubuntu: `sudo apt install build-essential cmake curl git python3-venv xz-utils`;
macOS: `xcode-select --install`, `brew install cmake`, Python from Homebrew or python.org).

```bash
git clone <repo> relpose
python -m venv .venv && source .venv/bin/activate && pip install -e "./relpose[data]"   # lab machines: "./relpose[data,sim,train]"
relpose init python my_workspace && cd my_workspace                                   # C/C++: relpose init native
```

The workspace holds your code in `solution/` (the minimal example to start from) and your copy of the build recipe in
`toolchain/` and `runtime/`: change it to add a package to the Python runtime or a library for native code
(`runtime/python/README.md`, `runtime/native/README.md`). Everything built lands in `build/`. Other starting points
(`dead_reckoning`, `klt_imu`, `tiny_cnn`) are in `relpose/solution/examples/` of the clone: copy one over `solution/`.

Download the validation episodes from the course page into `data/val`. Generate training data on the lab GPUs:

```bash
relpose simulate -o data/train --episodes 200 --seed 1                          # ~2 min per 8 one-minute episodes
relpose simulate -o data/train-short --episodes 64 --seed 900 --duration 15
```

## Workflow

```bash
relpose bundle solution -o solution.wasm        # after every edit; Python: the first time also builds the runtime (10–15 min)
relpose check solution.wasm                     # a few synthetic frames: errors, fuel per step
relpose evaluate solution.wasm data/val         # metrics and fuel; --simulate 8 for fresh episodes
relpose evaluate solution.wasm data/val -o report-solution   # also save trajectory videos
relpose debug solution data/val                 # in-process, with breakpoint() (docs/DEBUGGING.md)
```

`bundle` embeds Python code into the runtime and compiles native code (a `CMakeLists.txt`) with wasi-sdk. `evaluate`
writes `report-solution/report.json` and, per episode, `estimate.tum` (readable by `evo`) and `twist.txt`.

## Submit

Your team gets a token from the instructors; only that token sees your submissions.

```bash
curl -F file=@solution.wasm -F label="klt+imu v3" -H "Authorization: Bearer $TOKEN" https://<server>/submissions
curl -H "Authorization: Bearer $TOKEN" https://<server>/submissions          # yours, newest first: id, times, status
curl -H "Authorization: Bearer $TOKEN" https://<server>/submissions/<id>     # one of them, with results once done
curl -H "Authorization: Bearer $TOKEN" https://<server>/config               # the rules: quota, limits, timeouts
```

The server evaluates on unpublished episodes and returns metrics only, no logs; `relpose evaluate` applies the same
limits, so reproduce failures locally. Quota (see `/config`): one submission at a time, one per day (from midnight
Pacific time) plus one retry after a failed one.

## Tips

- Python loops over pixels are expensive; OpenCV calls are cheap.
- Keep the per-frame cost flat: episodes have no announced length.
- Gray input: `cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)`.
- Learned models: `relpose/solution/examples/python/tiny_cnn`. C, C++, Rust: `relpose init native`, `runtime/native/README.md`.

## Common problems

| symptom | fix |
|---|---|
| `CMakeCache.txt directory … is different` | the workspace moved: `relpose build-runtime --force`, delete `solution/build` |
| `… is not inside a workspace` | run the command on `solution/` of a workspace (`relpose init`) |
| `ModuleNotFoundError: scipy` | the runtime has numpy and OpenCV only (`docs/RUNTIME.md`) |
| `trap OUT_OF_FUEL` | less work per step: resolution, feature count, iterations |
| `MemoryError` | do not keep every frame |
| `step returned an unusable state` | NaN/inf or a zero quaternion: guard divisions, normalise quaternions |
| `step() must return 13 values` | append the twist to the pose |
| frames look identical | you kept a view of the frame buffer: `frame.copy()` |
| `np.load` fails on `.npz` | save with `np.savez`, not `np.savez_compressed` |
| the IMU looks upside down | the IMU need not start gravity-aligned: estimate gravity |
