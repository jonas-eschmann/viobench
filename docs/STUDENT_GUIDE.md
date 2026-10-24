# Student guide

## 1. Setup

```bash
git clone <repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate && pip install -e ".[data]"
relpose/solution/toolchain/build-image.sh      # Docker image with wasi-sdk + wasmtime + the CPython/numpy/OpenCV sources (once)
relpose/solution/runtime/python/build.sh               # your Python runtime: build/runtime/dist/runtime.wasm (10-15 min, once per machine)
```

The runtime is yours to build, and to change (`relpose/solution/runtime/python/README.md`): nothing inside the `.wasm` is prescribed,
only the ABI it exports. Writing C/C++ instead? Skip `build.sh` and read `relpose/solution/runtime/native/README.md`. The public
datasets come as a tarball from the course page; unpack them to `build/datasets/public`.

On the lab machines (RTX 5080): `pip install -e ".[data,sim,train]"` adds the HyperDrone generator (first use compiles
for a minute) and PyTorch. There is no pre-rendered training set; you generate what you need:

```bash
relpose simulate -o data/train --episodes 200 --seed 1 --condition default     # ~20 s per 8 episodes (640x480), blur ~45 s
relpose simulate -o data/train-blur --episodes 64 --seed 500 --condition blur
```

Read `docs/TASK.md` (what to estimate, how it is scored) and `docs/ABI.md` / `docs/RUNTIME.md` (what the sandbox offers).

## 2. Your first solution

```bash
cp -r relpose/solution/examples/python/dead_reckoning my_solution
relpose check build/runtime/dist/runtime.wasm --dir my_solution          # synthetic frames, prints status/fuel per step
relpose evaluate build/runtime/dist/runtime.wasm --dir my_solution build/datasets/public/default-val -o report-dev
relpose evaluate build/runtime/dist/runtime.wasm --dir my_solution --simulate 8 --condition blur   # fresh episodes, GPU, nothing on disk
```

`check` catches crashes and protocol errors; `evaluate` gives ATE/RPE and fuel per frame on real episodes
(`report-dev/report.json`, per-episode `estimate.tum` for plotting with `evo`). `--dir` mounts your directory into the
guest, so edits take effect immediately and tracebacks show real paths; bundling comes last.

## 3. Iterate

- Data: generate training data with `relpose simulate` (any condition, any amount); `build/datasets/public/default-val`
  and the other `*-val` sets are for validation and show the hidden conditions, `real-sample` the real-data format. The
  hidden sets use the same generator with other seeds and scenes.
- Inspect an episode in Python: `from relpose.tools.datasets import read_episode; ep = read_episode("build/datasets/public/default-val/<name>")`,
  then `ep.frames`, `ep.imu`, `ep.ground_truth_relative()`, `ep.steps()`.
- Augmentation: `relpose.tools.datasets.generate.conditions.sample(name, rng)` + `relpose.tools.datasets.generate.camera.apply(...)` reproduce the
  blur / low-light / saturation pipeline on your own renders (apply them to linear frames: `srgb_to_linear(frame / 255)`).
- Compute: watch the fuel column. Python-level loops over pixels are expensive; OpenCV calls are cheap per pixel; a
  `cv2.dnn` forward of a small CNN costs about as much as one `goodFeaturesToTrack`.
- Learned models: `relpose/solution/examples/python/tiny_cnn` shows PyTorch → ONNX → `cv2.dnn` inside the runtime, with a numpy fallback,
  and `train.py --generate N` trains straight from the simulator.
- Colour and resolution: frames are 320 × 240 grayscale unless `solution.json` next to your `solution.py` says
  `{"channels": 3}` (RGB, `(H, W, 3)`) and/or `{"resolution": [640, 480]}` (VGA). Real-data episodes are grayscale
  (replicated into three channels); if a set only exists at 320 × 240 you receive an upscale
  (`config["image"]["upscaled"]`). More pixels cost fuel in everything you run on them.

## 4. Submit

```bash
relpose bundle my_solution -o my_solution.wasm     # one self-contained file: your runtime + your directory
relpose check my_solution.wasm                     # exactly what the server runs first
curl -F file=@my_solution.wasm -F label="klt+imu v3" -H "X-Token: $TOKEN" https://<server>/submit
```

The server evaluates on the hidden sets (default, blur, low light, oversaturation, real), posts the result on the
leaderboard and keeps your stderr logs for you (`/submissions/<id>/logs`). Every submission is a point on the
accuracy-vs-fuel curve; your best points form your Pareto front. Quotas: 20 submissions per day, 128 MiB per file.

## 5. Going native

A C++ (or Rust) solution implements four exports and skips the interpreter entirely: `relpose/solution/runtime/native/` (header, CMake
snippet, `build.sh`) and `relpose/solution/examples/native/cpp_minimal`. The toolchain image has wasi-sdk; the Python-runtime build leaves
OpenCV's static libraries behind if you want them.

## 6. Common problems

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: scipy` | not available in the reference runtime; numpy + OpenCV only (`docs/RUNTIME.md`), or build your own |
| `trap OUT_OF_FUEL` | one step exceeded the fuel budget: reduce work per frame (resolution, feature count, iterations) |
| `MemoryError` | the 2 GiB cap; avoid keeping every frame |
| status `not_finite` | NaN in the pose: guard divisions and normalise quaternions |
| frames look identical | you kept a view of the frame buffer; `frame.copy()` |
| `np.load` fails on `.npz` | compressed archive: save with `np.savez`, not `np.savez_compressed` |
| real data looks "upside down" in the IMU | the IMU frame of the real sensor is not gravity-aligned at start; estimate gravity from the measurements |
