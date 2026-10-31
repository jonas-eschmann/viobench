# Task

For every frame of a camera and IMU stream from a quadrotor, estimate the camera's pose and twist relative to its
first frame. Submissions are ranked on accuracy and on compute per frame.

## Inputs and output

Per step: a frame, its timestamp and the IMU samples since the previous frame; the output is the 13-value state of
`docs/ABI.md` (`out`), in metric scale.

- Camera: pinhole, intrinsics in the config (they vary per episode); the shutter type is not given.
- IMU: consumer-grade, with white noise, turn-on biases and bias drift; no noise model is given.
- Not given: the orientation with respect to gravity, whether an episode starts at rest or in flight, its length.
- Causal: at step `k` you have frames `0..k` and the IMU up to `t_k`; the estimate is final once `step` returns.

## Scoring

Per episode (`viobench.tools.runner.metrics`):

- **RPE** (the headline): for all frame pairs `(i, j)` one second apart,
  `E = (T_gt,i⁻¹ T_gt,j)⁻¹ (T_est,i⁻¹ T_est,j)`; the RMSE of `|trans(E)|` in metres, and of the angle of `E`.
- ATE: position RMSE in frame-0 coordinates, without alignment.
- Rotation RMSE, velocity RMSE, angular-velocity RMSE.
- Diagnostics: Sim(3)-aligned ATE and scale, final drift per metre, held frames.
- Compute: the maximum fuel consumed by a single `step` call across all frames, episodes and test datasets.
  This is Wasmtime's deterministic fuel count for a fixed module, input and runtime version. Failed calls count too.
  Initialization fuel is reported separately and has its own budget.

Per dataset, a submission is the point (maximum fuel per frame across the entire test set, median RPE for that dataset).
Every dataset's point uses the same compute value for a submission. A failed episode is scored with its last pose held,
standing still; its attempted frames, including the failing call, contribute to the fuel maximum.

For analysis, each episode, dataset and overall report also records mean, median, p95 and p99 fuel per frame,
total step fuel and the number of attempted frames. Dataset and overall statistics pool all attempted frames, so the
mean is weighted by frame count and percentiles describe the full frame distribution. Frames held after a failure
add no samples. With no attempted frames, the count and fuel statistics are zero. Wall time is reported for reference.

## Limits (`viobench.tools.runner.sandbox.Limits`)

| | |
|---|---|
| fuel per step | 2 × 10¹⁰ (≈ 1 s) |
| fuel for init | 2 × 10¹¹ |
| memory | 2 GiB (one linear memory) |
| function table | one, up to 2²⁰ entries |
| wall clock per step | 60 s |
| submission size | 128 MiB (the Python runtime is 48 MiB) |

Set in `viobench/tools/config.json`; `viobench check` and `evaluate` apply the same limits as the server.

## Data

Train on `viobench simulate` output, validate on the episodes from the course page. The evaluation episodes are not
published. `viobench.tools.datasets.read_episode` loads an episode.
