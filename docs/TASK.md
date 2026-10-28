# The task: compute-aware relative pose estimation

You receive a stream of camera frames and IMU samples from a quadrotor and must report, for every frame, the pose of
that camera frame **relative to the first camera frame of the episode**. Submissions are scored on accuracy *and* on the
compute they spend per frame, so the leaderboard is a curve, not a number.

## Inputs

| | |
|---|---|
| Frames | `uint8` RGB, `(H, W, 3)`, at 320 × 240 by default or 640 × 480 if the solution declares `frame_width = 640` (see `docs/ABI.md`, "Frame format"); nothing else about the format varies. Pinhole camera without distortion, intrinsics in the config (`fx, fy, cx, cy`; they vary between episodes), global or rolling shutter (unannounced, no readout time), timestamp = mid-exposure of the centre row |
| Frame rate | nominally 20 Hz; use the timestamps |
| IMU | nominally 200 Hz: specific force in m/s² (gravity included: a resting IMU reads ≈ 9.81 along the direction opposite to gravity) and angular rate in rad/s, in the IMU frame. Expect a consumer MEMS IMU with white noise, turn-on biases and slow bias drift; no noise model is given, and the initial orientation w.r.t. gravity is not given either. |
| Extrinsics | `T_cam_imu` (4 × 4): IMU coordinates → camera coordinates (OpenCV: x right, y down, z forward); exact for the episode |
| Timing | `step(t, frame, imu)`: `t` is the frame timestamp, `imu` holds all IMU rows with `t_previous_frame < t_row ≤ t`; the first call gets the rows of one frame interval before the first frame. Everything is one clock. |
| Start | Not announced: an episode may start at rest or in flight. |
| Duration | One continuous flight per episode, of unannounced length: never assume a frame count, and keep the per-frame cost flat. |

## Output

`[x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]`, the state of the current camera frame:

- its pose: translation (metres) and unit quaternion (scalar last) of `T_c0_ck`, the pose of the current camera frame
  in the first camera frame (`p_c0 = R p_ck + t`);
- its twist in its own axes: the velocity of the camera centre (m/s) and the angular velocity (rad/s), `v = Rᵀ dt/dt`
  and `[w]× = Rᵀ dR/dt` of the same `T_c0_ck` (`docs/ABI.md`, `out`, has an example and the conversion from IMU states).

Metric scale is expected: the IMU makes it observable. Every step must return a usable state, all 13 values: one that
raises, returns anything else or a value that is not finite ends the episode as failed, so until you have an estimate
return your best guess (at rest: identity and zero twist).

Everything is **causal**: at step `k` you have seen frames `0..k` and IMU samples up to `t_k`, nothing later. The
estimate for frame `k` is final once `step` returns.

## Scoring

Per episode (`relpose.tools.runner.metrics`):

- **RPE** (the headline) – for every pair of frames one second apart, compare your motion between them with the true
  motion, `E = (T_gt,i⁻¹ T_gt,j)⁻¹ (T_est,i⁻¹ T_est,j)`; the translational RPE is the RMSE of `|trans(E)|` over all
  such pairs, in metres per one-second window. It measures how well you track motion over the last second, so an
  early mistake costs only the windows it affects, not the rest of the episode; the rotational RPE (geodesic angle of
  `E`) is reported next to it;
- **ATE** – RMSE of the position error over all frames, in the frame-0 coordinates, *without* any alignment (frame 0
  defines the frame): the accumulated drift;
- rotation RMSE (geodesic angle);
- **twist** – RMSE of the velocity (m/s) and of the angular velocity (rad/s) against the ground-truth twist (central
  differences of the ground-truth camera poses);
- diagnostics: Sim(3)-aligned ATE and the fitted scale (separates "wrong scale" from "wrong shape"), final
  drift per metre travelled, held frames (after a failure).

Compute (`relpose.tools.runner.run`):

- **fuel per frame** (median and 95th percentile): Wasmtime's deterministic instruction count for your `step`
  (host calls are free, the interpreter's own work is not). It is identical on your laptop and on the server.
- wall time per frame (for intuition only; it depends on the machine).

Each dataset of the evaluation is summarised by the median RPE and median fuel per frame over its episodes. A
submission is a point in the (fuel/frame, RPE) plane per dataset; your submissions together form your curve. A
failure (an error from `init` or `step`, an unusable state, a trap) ends the episode and marks it failed: the frames
from there on repeat the last pose, standing still, so every window after it costs the distance actually flown in it,
and the episode stays on the record.

## Limits (default `Limits` in `relpose.tools.runner.sandbox`)

| | |
|---|---|
| fuel per step | 2 × 10¹⁰ (≈ 1 s of native-speed wasm) |
| fuel for init | 2 × 10¹¹ |
| memory | 2 GiB |
| wall clock per step | 60 s (safety net) |
| submission size | 128 MiB (the Python runtime is 48 MiB) |

## Data

Generate training data with `relpose simulate` (HyperDrone, on the lab GPUs) and validate on the episodes from the
course page (`docs/STUDENT_GUIDE.md`). The episodes the server evaluates on are not published. Episodes on disk and
how to read them: `relpose.tools.datasets` (`read_episode`, `episodes`; the format is described in
`relpose/tools/datasets/storage.py`).
