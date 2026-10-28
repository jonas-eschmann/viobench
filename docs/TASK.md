# The task: compute-aware relative pose estimation

You receive a stream of camera frames and IMU samples from a quadrotor and must report, for every frame, the pose of
that camera frame **relative to the first camera frame of the episode**. Submissions are scored on accuracy *and* on the
compute they spend per frame, so the leaderboard is a curve, not a number.

## Inputs

| | |
|---|---|
| Frames | `uint8` RGB, `(H, W, 3)`, at 320 × 240 by default or 640 × 480 if the solution declares `frame_width = 640` (see `docs/ABI.md`, "Frame format"); nothing else about the format varies. Some sets are grayscale-only and arrive with R = G = B, unannounced. Pinhole camera without distortion, intrinsics in the config (`fx, fy, cx, cy`; the horizontal field of view varies between roughly 48° and 80° across episodes), global or rolling shutter (unannounced, no readout time), timestamp = mid-exposure of the centre row |
| Frame rate | nominally 20 Hz; use the timestamps |
| IMU | nominally 200 Hz: specific force in m/s² (gravity included: a resting IMU reads ≈ 9.81 along the direction opposite to gravity) and angular rate in rad/s, in the IMU frame. Expect a consumer MEMS IMU with white noise, turn-on biases and slow bias drift; no noise model is given, and the initial orientation w.r.t. gravity is not given either. |
| Extrinsics | `T_cam_imu` (4 × 4): IMU coordinates → camera coordinates (OpenCV: x right, y down, z forward); exact for the episode |
| Timing | `step(t, frame, imu)`: `t` is the frame timestamp, `imu` holds all IMU rows with `t_previous_frame < t_row ≤ t`; the first call gets the rows of one frame interval before the first frame. Everything is one clock. |
| Start | Not announced: an episode may start at rest or in flight. |
| Duration | One continuous flight per episode, of unannounced length: never assume a frame count, and keep the per-frame cost flat. The shipped sets are about a minute long (1 200 frames, 12 000 IMU samples). |

## Output

`[x, y, z, qx, qy, qz, qw]`: translation (metres) and unit quaternion (scalar last) of `T_c0_ck`, the pose of the
current camera frame in the first camera frame (`p_c0 = R p_ck + t`). Metric scale is expected: the IMU makes it
observable. Every step must return a usable pose: one that raises or returns none ends the episode as failed, so
until you have an estimate return your best guess (at rest: identity).

Everything is **causal**: at step `k` you have seen frames `0..k` and IMU samples up to `t_k`, nothing later. The
estimate for frame `k` is final once `step` returns.

## Scoring

Per episode (`relpose.tools.runner.metrics`):

- **ATE** – RMSE of the position error over all frames, in the frame-0 coordinates, *without* any alignment (frame 0
  defines the frame);
- rotation RMSE (geodesic angle);
- **RPE** at Δ = 1 s – RMSE of the relative pose error between frames one second apart (drift rate, robust to a single
  early mistake);
- diagnostics: Sim(3)-aligned ATE and the fitted scale (separates "wrong scale" from "wrong shape"), final
  drift per metre travelled, held frames (after a failure).

Compute (`relpose.tools.runner.run`):

- **fuel per frame** (median and 95th percentile): Wasmtime's deterministic instruction count for your `step`
  (host calls are free, the interpreter's own work is not). It is identical on your laptop and on the server.
- wall time per frame (for intuition only; it depends on the machine).

Each dataset category (default simulation, motion blur, low light, oversaturation, real) is summarised by the median
ATE and median fuel per frame over its episodes. A submission is a point in the (fuel/frame, ATE) plane per category;
your submissions together form your curve. A failure (an error from `init` or `step`, an unusable pose, a trap)
ends the episode and marks it failed: the frames from there on repeat the last pose, and it stays on the record.

## Limits (default `Limits` in `relpose.tools.runner.sandbox`)

| | |
|---|---|
| fuel per step | 2 × 10¹⁰ (≈ 1 s of native-speed wasm) |
| fuel for init | 2 × 10¹¹ |
| memory | 2 GiB |
| wall clock per step | 60 s (safety net) |
| submission size | 128 MiB (the Python runtime is 48 MiB) |

## Datasets

See `docs/DATA.md`: the generator you run yourself on the lab GPUs (the hidden sets come from the same pipeline), small
public validation sets and samples of every hidden category, the parameter ranges of the degradations, and the on-disk
format.
