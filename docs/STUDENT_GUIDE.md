# Student guide

## Setup

Linux or macOS; on Windows, WSL2 with the clone inside the WSL filesystem. Needs git, curl, CMake, a C compiler, make
and Python 3.10–3.14 with venv (Ubuntu: `sudo apt install build-essential cmake curl git python3-venv xz-utils`;
macOS: `xcode-select --install`, `brew install cmake`, Python from Homebrew or python.org).

```bash
git clone <repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate && pip install -e ".[data]"   # lab machines: ".[data,sim,train]"
relpose/solution/runtime/python/build.sh                                        # build/runtime/dist/runtime.wasm, 10–15 min
```

Download the validation episodes from the course page into `data/val`. Generate training data on the lab GPUs:

```bash
relpose simulate -o data/train --episodes 200 --seed 1                          # ~2 min per 8 one-minute episodes
relpose simulate -o data/train-short --episodes 64 --seed 900 --duration 15
```

## Workflow

```bash
cp -r relpose/solution/examples/python/dead_reckoning my_solution
relpose bundle my_solution -o my_solution.wasm     # after every edit
relpose check my_solution.wasm                     # a few synthetic frames: errors, fuel per step
relpose evaluate my_solution.wasm data/val         # metrics and fuel; --simulate 8 for fresh episodes
relpose debug my_solution data/val                 # in-process, with breakpoint() (docs/DEBUGGING.md)
```

`evaluate` writes `report-my_solution/report.json` and, per episode, `estimate.tum` (readable by `evo`) and
`twist.txt`.

## Submit

Your team gets a token from the instructors; only that token sees your submissions.

```bash
curl -F file=@my_solution.wasm -F label="klt+imu v3" -H "Authorization: Bearer $TOKEN" https://<server>/submissions
curl -H "Authorization: Bearer $TOKEN" https://<server>/submissions          # yours, newest first: id, times, status
curl -H "Authorization: Bearer $TOKEN" https://<server>/submissions/<id>     # one of them, with results once done
curl -H "Authorization: Bearer $TOKEN" https://<server>/config               # the rules: quota, limits, timeouts
```

The server evaluates on unpublished episodes and returns metrics only: no stdout, stderr or failure messages. An
episode that fails shows up as `"completed": false`; reproduce it locally with `relpose evaluate`, under the server's
limits with `--limits "$(curl -s -H "Authorization: Bearer $TOKEN" https://<server>/config)"`. Quota: one submission
per team and day (`daily_quota` in `/config`); the quota resets at midnight Pacific time.

## Tips

- Python loops over pixels are expensive; OpenCV calls are cheap.
- Keep the per-frame cost flat: episodes have no announced length.
- Gray input: `cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)`.
- Learned models: `relpose/solution/examples/python/tiny_cnn`. C, C++, Rust: `relpose/solution/runtime/native/README.md`.

## Common problems

| symptom | fix |
|---|---|
| `CMakeCache.txt directory … is different` | delete the stale build directory (`build/runtime`, `<solution>/build`) |
| `ModuleNotFoundError: scipy` | the runtime has numpy and OpenCV only (`docs/RUNTIME.md`) |
| `trap OUT_OF_FUEL` | less work per step: resolution, feature count, iterations |
| `MemoryError` | do not keep every frame |
| `step returned an unusable state` | NaN/inf or a zero quaternion: guard divisions, normalise quaternions |
| `step() must return 13 values` | append the twist to the pose |
| frames look identical | you kept a view of the frame buffer: `frame.copy()` |
| `np.load` fails on `.npz` | save with `np.savez`, not `np.savez_compressed` |
| the IMU looks upside down | the IMU need not start gravity-aligned: estimate gravity |
