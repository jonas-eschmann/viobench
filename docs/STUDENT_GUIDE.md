# Student guide

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install "eecs116[data] @ git+https://github.com/<course>/course-eecs116-216-final-project"   # or pip install -e ".[data]" from a clone
# runtime.wasm (48 MB) and the datasets are downloaded separately (links on the course page):
export EECS116_RUNTIME=~/eecs116/runtime.wasm
```

Optional: `pip install "eecs116[sim]"` for the HyperDrone generator (needs an NVIDIA GPU + CUDA; first use compiles
for a minute) and `pip install "eecs116[train]"` for PyTorch.

Read `docs/TASK.md` (what to estimate, how it is scored) and `docs/ABI.md`/`docs/RUNTIME.md` (what the sandbox offers).

## 2. Your first solution

```bash
cp -r examples/python-deadreckoning my_solution
python -m eecs116 check $EECS116_RUNTIME --dir my_solution          # synthetic frames, prints status/fuel per step
python -m eecs116 evaluate $EECS116_RUNTIME --dir my_solution datasets/default-val -o report-dev
```

`check` catches crashes and protocol errors; `evaluate` gives ATE/RPE and fuel per frame on real episodes
(`report-dev/report.json`, per-episode `estimate.tum` for plotting with `evo`).

## 3. Iterate

- Data: `datasets/default-train` for training/tuning, `datasets/default-val` for validation, the `*-sample` sets to see
  the hidden variants, `real-sample` for the real-data format. Generate more with `python -m eecs116 simulate`
  (GPU) — the hidden sets use the same generator with other seeds and scenes.
- Inspect an episode in Python: `from eecs116.data import read_episode; ep = read_episode("datasets/default-val/<name>")`,
  then `ep.frames`, `ep.imu`, `ep.ground_truth_relative()`, `ep.steps()`.
- Augmentation: `eecs116.data.variants.sample(name, rng)` + `eecs116.data.isp.apply(...)` reproduce the blur /
  low-light / saturation pipeline on your own renders (apply them to linear frames: `srgb_to_linear(frame / 255)`).
- Compute: watch the fuel column. Python-level loops over pixels are expensive; OpenCV calls are cheap per pixel;
  a `cv2.dnn` forward of a small CNN costs about as much as one `goodFeaturesToTrack`.
- Learned models: `examples/python-tinynn` shows PyTorch → ONNX → `cv2.dnn` inside the runtime, with a numpy fallback.

## 4. Submit

```bash
python -m eecs116 bundle my_solution -o my_solution.wasm     # one file: runtime + your directory
python -m eecs116 check my_solution.wasm                     # exactly what the server runs first
curl -F file=@my_solution.wasm -F label="klt+imu v3" -H "X-Token: $TOKEN" https://<server>/submit
```

The server evaluates on the hidden sets (default, blur, low light, oversaturation, real), posts the result on the
leaderboard and keeps your stderr logs for you (`/submissions/<id>/logs`). Every submission is a point on the
accuracy-vs-fuel curve; your best points form your Pareto front. Quotas: 20 submissions per day, 128 MiB per file.

## 5. Going native

A C++ (or Rust) solution implements four exports and skips the interpreter entirely: `examples/cpp-minimal`. The
course Docker image has the wasi-sdk toolchain and the OpenCV static libraries compiled for wasm.

## 6. Common problems

| symptom | cause / fix |
|---|---|
| `ModuleNotFoundError: scipy` | not available in the runtime; numpy + OpenCV only (`docs/RUNTIME.md`) |
| `trap OUT_OF_FUEL` | one step exceeded the fuel budget: reduce work per frame (resolution, feature count, iterations) |
| `MemoryError` | the 2 GiB cap; avoid keeping every frame |
| status `not_finite` | NaN in the pose: guard divisions and normalise quaternions |
| frames look identical | you kept a view of the frame buffer; `frame.copy()` |
| `np.load` fails on `.npz` | compressed archive: save with `np.savez`, not `np.savez_compressed` |
| real data looks "upside down" in the IMU | the IMU frame of the real sensor is not gravity-aligned at start; estimate gravity from the measurements |
