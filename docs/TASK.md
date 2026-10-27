# The task: compute-aware relative pose estimation

You receive a stream of camera frames and IMU samples from a quadrotor and must report, for every frame, the pose of
that camera frame **relative to the first camera frame of the episode**. Submissions are scored on accuracy *and* on the
compute they spend per frame, so the leaderboard is a curve, not a number.

## Inputs

| | |
|---|---|
| Frames | `uint8`, 320 × 240 grayscale by default. Your submission may opt into RGB (`{"channels": 3}`) and/or VGA (`{"resolution": [640, 480]}`) in `solution.json` (see `docs/ABI.md`): sets are stored at 640 × 480 and resampled to what you declare; real-data sets are grayscale (replicated into three channels, `config["image"]["color"]` says so) and if a set only exists at 320 × 240 you get a bilinear upscale (`config["image"]["upscaled"]`). Pinhole camera without distortion, intrinsics in the config (`fx, fy, cx, cy`; the horizontal field of view varies between roughly 48° and 80° across episodes), global shutter, timestamp = mid-exposure |
| Frame rate | 20 Hz (default; `config["episode"]["frame_rate_hz"]`) |
| IMU | 200 Hz, in the body/IMU frame: specific force in m/s² (gravity included: a resting IMU reads ≈ 9.81 along the direction opposite to gravity), angular rate in rad/s. In simulation the body frame is x forward, y left, z up and the IMU sits at the body origin; in real data it is whatever the sensor calibration says (`T_cam_imu` is exact, the initial orientation w.r.t. gravity is not given). Noise and bias magnitudes are listed in the config (`imu.*_noise_std` per sample, `*_bias_init_max`, `*_random_walk`). |
| Extrinsics | `imu.T_cam_imu` (4 × 4, row-major): body/IMU coordinates → camera coordinates (OpenCV: x right, y down, z forward). `imu.T_imu_cam` is the inverse. |
| Timing | `step(t, frame, imu)`: `t` is the frame timestamp, `imu` holds all IMU rows with `t_previous_frame < t_row ≤ t` (the first call also delivers the rows before the first frame). Everything is one clock. |
| Start | Episodes of the simulated sets begin with ≥ 1 s of hover (`config["episode"]["hover_seconds"]`) before the drone starts flying its route; real-data snippets may start mid-flight (`hover_seconds` is 0 then). |
| Duration | One continuous flight per episode. The length is in the config (`config["episode"]["num_frames"]`, `frame_rate_hz`) and may differ between sets, so never hard-code it; the shipped sets are about a minute long (1 200 frames, 12 000 IMU samples), and real-data snippets are cut to the same length as the simulated episodes |

## Output

`[x, y, z, qx, qy, qz, qw]`: translation (metres) and unit quaternion (scalar last) of `T_c0_ck`, the pose of the
current camera frame in the first camera frame (`p_c0 = R p_ck + t`). Metric scale is expected: the IMU makes it
observable. Return `None` while you have no estimate yet (counted as a missing frame).

Everything is **causal**: at step `k` you have seen frames `0..k` and IMU samples up to `t_k`, nothing later. The
estimate for frame `k` is final once `step` returns.

## Scoring

Per episode (`relpose.tools.runner.metrics`):

- **ATE** – RMSE of the position error over all frames, in the frame-0 coordinates, *without* any alignment (frame 0
  defines the frame; a missing/invalid estimate counts with the error of a stationary estimator, i.e. the ground-truth
  distance travelled);
- rotation RMSE (geodesic angle);
- **RPE** at Δ = 1 s – RMSE of the relative pose error between frames one second apart (drift rate, robust to a single
  early mistake);
- diagnostics: Sim(3)-aligned ATE and the fitted scale (separates "wrong scale" from "wrong shape"), coverage,
  final drift per metre travelled.

Compute (`relpose.tools.runner.run`):

- **fuel per frame** (median and 95th percentile): Wasmtime's deterministic instruction count for your `step`
  (host calls are free, the interpreter's own work is not). It is identical on your laptop and on the server.
- wall time per frame (for intuition only; it depends on the machine).

Each dataset category (default simulation, motion blur, low light, oversaturation, real) is summarised by the median
ATE and median fuel per frame over its episodes. A submission is a point in the (fuel/frame, ATE) plane per category;
your submissions together form your curve. Failures (trap, exceptions on 10 consecutive frames, protocol errors)
mark an episode failed: its ATE is the stationary-estimator error and it stays on the record.

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
