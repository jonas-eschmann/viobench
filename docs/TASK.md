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

Per episode (`relpose.tools.runner.metrics`):

- **RPE** (the headline): for all frame pairs `(i, j)` one second apart,
  `E = (T_gt,i⁻¹ T_gt,j)⁻¹ (T_est,i⁻¹ T_est,j)`; the RMSE of `|trans(E)|` in metres, and of the angle of `E`.
- ATE: position RMSE in frame-0 coordinates, without alignment.
- Rotation RMSE, velocity RMSE, angular-velocity RMSE.
- Diagnostics: Sim(3)-aligned ATE and scale, final drift per metre, held frames.
- Compute: fuel per frame (median and p95), Wasmtime's instruction count, identical on every machine; wall time for
  reference.

Per dataset, a submission is the point (median fuel per frame, median RPE). A failed episode is scored with its last
pose held, standing still.

## Limits (`relpose.tools.runner.sandbox.Limits`)

| | |
|---|---|
| fuel per step | 2 × 10¹⁰ (≈ 1 s) |
| fuel for init | 2 × 10¹¹ |
| memory | 2 GiB (one linear memory) |
| function table | one, up to 2²⁰ entries |
| wall clock per step | 60 s |
| submission size | 128 MiB (the Python runtime is 48 MiB) |

Set in `relpose/tools/config.json`; `relpose check` and `evaluate` apply the same limits as the server.

## Data

Train on `relpose simulate` output, validate on the episodes from the course page. The evaluation episodes are not
published. `relpose.tools.datasets.read_episode` loads an episode.
