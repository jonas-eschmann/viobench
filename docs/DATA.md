# Data

## Episodes on disk

One directory per episode, EuRoC/ASL layout so that standard VIO tooling (evo, ORB-SLAM, OpenVINS) reads it too:

```
<episode>/
  episode.json                                  format, name, the solution config (docs/ABI.md), meta (source, condition, ISP parameters, ...)
  mav0/cam0/data.csv                            "#timestamp [ns],filename"
  mav0/cam0/data/<timestamp ns>.png             8-bit RGB (simulated) or grayscale (real)
  mav0/cam0/sensor.yaml                         Kalibr-style: T_BS (camera in body), intrinsics, resolution
  mav0/imu0/data.csv                            "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z"   (gyro first, like EuRoC)
  mav0/imu0/sensor.yaml                         rate, continuous-time noise densities
  mav0/state_groundtruth_estimate0/data.csv     "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z"    body (IMU) pose in the world
```

`relpose.tools.datasets.read_episode(dir)` loads it into an `Episode` (frames, frame times, IMU rows `[t, a, g]`, ground truth,
`T_body_camera`, config); `episode.ground_truth_relative()` gives the `T_c0_ck` poses a perfect solution would output
(ground truth interpolated to each frame's mid-exposure time, moved into the camera frame via the extrinsics, re-based
on frame 0). `episode.steps()` yields `(t, frame, imu_window)` exactly as the harness feeds a solution (the first
window is one frame interval, however much IMU the set recorded before frame 0); `episode.adapt(resolution)` returns
the frames and config a solution receives (`docs/ABI.md`, "Frame format": always RGB, grayscale replicated; 640 × 480 →
320 × 240 by area averaging with rescaled intrinsics, never upscaled). The `config` in `episode.json` is the set's full
description (rates, IMU noise model, hover, name, ...); solutions only get the frame size, intrinsics and `T_cam_imu`.

A set must be stored at 640 × 480 (or 320 × 240, which then only serves 320 × 240 solutions), RGB or grayscale,
rectified pinhole, with intrinsics, `T_cam_imu` and ground truth covering every frame on one clock. Simulated episodes
are stored as RGB at 640 × 480, real ones as grayscale at 640 × 480 (EuRoC's 752 × 480 cropped).

A one-minute episode at 640 × 480 is ~360 MB (RGB PNG); the format has no length limit, and `episode.json` records
`frames` and `duration_s`.

## Simulated episodes (HyperDrone)

`relpose simulate -o <dir> --episodes N --seed S --condition {default,blur,lowlight,oversaturated} --scenes {procthor-train,hssd-train,hssd-val} [--duration 60]`

- Platform: HyperDrone preset `x500_fpv_imu` (quadrotor dynamics, 200 Hz IMU with white noise, turn-on bias and
  random walk; camera mounted forward), visual-inertial localization task: the built-in RAPTOR autopilot flies a
  random route of waypoints through free space after a 1 s hover.
- Length: `--duration` seconds per episode (default 60: 12 000 IMU steps, 1 200 frames). The route length scales with
  it (`--waypoints`, default 0.8 per second, 48 for a minute); each distinct duration or route length JIT-compiles
  the environment once (about a minute). Frames are spooled to `$RELPOSE_SPOOL` (default: the system temp
  directory) while a batch renders, so memory does not grow with the duration.
- Rendering: ray-traced ProcTHOR (public sets) / HSSD scenes at 640 × 480 and the IMU rate (200 Hz); the camera frame is the
  average of the sub-frames inside the shutter interval (box shutter, linear radiance) → motion blur is physically
  consistent with the trajectory. The horizontal FOV is drawn per episode from 63.8° ± 15°; intrinsics are in the
  config. The camera is an ideal pinhole, no distortion, no rolling shutter.
- Camera/ISP model (`relpose.tools.datasets.generate.camera`): exposure scaling, per-channel photon shot noise (full-well capacity;
  Gaussian approximation of the Poisson count), Gaussian read noise, digital gain, clipping, sRGB encoding, 8-bit
  quantisation.
- Determinism: a seed fixes the scene, spawn, route, IMU noise and ISP draws *on one machine and build*
  (HyperDrone compiles with `-march=native -ffast-math`); the published sets are therefore distributed as files, not
  as seeds. `episode.json` keeps the spawn point and all parameters.
- Episodes in which the autopilot crashes (terminated by the simulator) are dropped.

### Conditions (public parameter ranges; hidden sets draw from the same ranges)

| condition | shutter | exposure scale | read noise | full well | note |
|---|---|---|---|---|---|
| `default` | 1–5 ms | 0.8–1.25 | 0.001–0.004 | 10k–40k e⁻ | the training distribution |
| `blur` | 25–50 ms | 0.8–1.25 | 0.001–0.004 | 10k–40k e⁻ | heavy motion blur (5–10 sub-frames) |
| `lowlight` | 5–20 ms | 0.02–0.1 | 0.005–0.02 | 500–3000 e⁻ | under-exposed then gained up: noise, crushed shadows |
| `oversaturated` | 1–5 ms | 2–5 | 0.001–0.004 | 10k–40k e⁻ | large clipped areas |

`relpose.tools.datasets.generate.conditions.sample(condition, rng)` draws a parameter set the way the generator does — use it (and
`relpose.tools.datasets.generate.camera.apply`) as training augmentation, or simply call `simulate(..., condition=...)`; the same code
produces the hidden sets with other seeds. There is no pre-rendered training set: generate what you need on the lab
machines (`relpose/solution/examples/python/tiny_cnn/train.py --generate N` shows training straight from the generator).

### Published sets

| set | content | purpose |
|---|---|---|
| `default-val` | 8 one-minute episodes, 3 ProcTHOR scenes | local validation, format reference |
| `blur-val`, `lowlight-val`, `oversaturated-val` | 4 episodes each | see what the hidden conditions look like |
| `real-sample` | two EuRoC V1_01 snippets | see the real-data format |

Hidden (server only): `default`, `blur`, `lowlight`, `oversaturated` (new seeds, HSSD validation scenes) and `real`
(EuRoC snippets from sequences other than the sample).

## Real episodes (EuRoC MAV)

`relpose convert-euroc <sequence_dir> -o <dir> [--snippet 60] [--stride 60] [--start 0]`

EuRoC MAV (ETH ASL; rights: "In Copyright – Non-Commercial Use Permitted"): cam0 752 × 480 grayscale global
shutter at 20 Hz, ADIS16448 IMU at 200 Hz, Vicon/Leica ground truth. The converter undistorts (radial-tangential),
crops a centred 4:3 window (640 × 480, no scaling), recomputes the intrinsics, copies the IMU noise
densities, keeps `T_BS` (camera in IMU frame) and cuts the sequence into non-overlapping snippets of `--snippet`
seconds (default one minute, like the simulated episodes; the sequences are 1.5–3 minutes long) inside the
ground-truth span. Snippet 0 starts at the sequence start (the MAV is static there: `hover_seconds` = 1 in
`episode.json`); later snippets start mid-flight (`hover_seconds` = 0 — no stationary window). Solutions are not told
either.

## Metrics

`relpose.tools.runner.metrics.evaluate_trajectory(ground_truth, estimate, timestamps)` → ATE (no alignment), rotation RMSE, RPE at
1 s, Sim(3)-aligned ATE and scale, coverage, final drift. `relpose evaluate solution.wasm <dataset dirs>`
writes `report.json` with per-episode metrics, fuel and wall-time statistics, plus `estimate.tum` per episode
(`t x y z qx qy qz qw`, loadable with `evo_traj tum`).
