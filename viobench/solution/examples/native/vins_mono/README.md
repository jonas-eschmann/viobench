# vins_mono

[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) as a native viobench submission. The example runs the
upstream monocular-inertial estimator: visual-inertial initialization, IMU preintegration, sliding-window Ceres
optimization and marginalization. Its pinhole front end follows the upstream KLT pipeline with OpenCV optical
flow, epipolar rejection and spatially distributed Shi-Tomasi features.

Build the example directly from the viobench repository root:

```bash
viobench build-runtime                     # once: supplies the wasm OpenCV libraries
viobench bundle viobench/solution/examples/native/vins_mono -o vins_mono.wasm
viobench check vins_mono.wasm
viobench evaluate vins_mono.wasm /path/to/episode -o vins_mono-results
```

The first CMake configure downloads VINS-Mono at commit `90dabb5ec79946ae42fd2e1e91d4e69aabe1e25d`,
Ceres 2.1.0 and Eigen 3.4.1. Archives are pinned by SHA256. OpenCV comes from the workspace's runtime build;
`viobench build-runtime` builds that complete runtime even though this native example only links its OpenCV libraries.

| File | Purpose |
|---|---|
| `solution.cpp` | viobench ABI, sensor calibration and camera pose/twist output |
| `feature_tracker.h` | pinhole KLT tracking and spatially distributed feature detection |
| `parameters.cpp` | estimator settings without ROS parameter loading |
| `CMakeLists.txt` | dependency downloads and native/WASI build |
| `patches/vins_mono.patch` | initialization, numerical and portability changes to the pinned estimator |
| `compat/` | ROS logging/message shims and the square-root marginalization helper |

## Estimation and output

Each step consumes RGB pixels and causal IMU rows, using the camera intrinsics and `T_cam_imu` supplied by `init`.
The calibration is fixed; the adapter supports the simulator's undistorted pinhole camera. The estimator runs
synchronously without a ROS runtime, pose graph, loop closure or background workers.

The tracker uses bias-corrected gyro rotation to compensate image-patch deformation before optical flow. It
also re-observes tracks directly against a recent image every 0.25 seconds to reduce accumulated tracking error.
Forward/backward checks, epipolar rejection and spatial feature selection still apply.

Initialization supports both the upstream visual-inertial alignment and a stationary bootstrap. The latter
requires agreement between low image displacement and stable IMU measurements; it estimates gravity and gyro
bias and supplies a zero-velocity constraint. Translation and feature depth become observable as motion begins.
If earlier motion remains in the keyframe window, a recent stationary interval can rebuild that window from its
original images and IMU samples. This handles settling or turning before coming to rest without assigning zero
velocity to the earlier moving states.
A sensor-bias prior prevents a weak initial window from explaining gravity error with an implausible accelerometer
bias. These priors enter once and are carried through marginalization, rather than being re-applied at every frame.

Before metric initialization, the example reports gyro-integrated attitude and angular velocity, with the camera/IMU
lever-arm contribution to position and velocity. Metric translation is unavailable during this interval. A warning
is emitted every five seconds until initialization completes. `viobench check` checks ABI behavior on synthetic
inputs; passing it does not establish tracking accuracy.

Buffered IMU measurements recover the first input camera's pose from a named optimized window state. Dynamic
initialization refines this reference until it leaves the sliding window before publishing metric translation;
the stationary bootstrap supplies its metric reference immediately. Every subsequent pose is relative to the
original input camera frame, including when early images had no usable features. Linear and angular velocities
use current camera axes and include the camera/IMU lever arm. A reset after metric output begins returns an ABI
error because a fresh estimator would lose the original reference frame.

Optimization uses iteration limits rather than wall-clock deadlines, and marginalization runs on the calling
thread so WASM execution does not require threads. The local sliding window accumulates drift over long runs.
Accuracy and fuel measurements should be obtained with `viobench evaluate` on representative recordings.

The IMU defaults in `parameters.cpp` match the simulator at 200 Hz. VINS-Mono's discrete preintegration uses
per-sample standard deviations: `ACC_N = 0.28` m/s² and `GYR_N = 0.07` rad/s. Its bias noise terms multiply by
the sample interval, so continuous-time random-walk densities become `ACC_W = 0.001 * sqrt(200)` and
`GYR_W = 0.0006 * sqrt(200)`, giving bias covariance increments proportional to elapsed time.

The patch rebuilds the gravity-refinement normal equations after each tangent-plane update, initializes gravity
and pointer state, and rejects nonfinite or nonpositive initialization scale and feature depths. Marginalization
uses column-scaled QR on the residual Jacobians, avoiding the squared condition number and cancellation of
normal-equation Schur subtraction. Unusable Ceres solves return an ABI error.

## Validation and limitations

The regressions in `tests/test_vins_mono.py` cover three invalid camera calibrations, featureless initialization,
and tracking an independently rendered analytic room trajectory with delayed visual input, a camera/IMU lever arm
and exposures between IMU samples. The moving trajectory checks post-initialization ATE below 0.5 m and translation
RPE below 0.25 m, plus rotation, velocity and metric-scale bounds. Two noisy hover/turn/translation trajectories
score the entire sequence, including startup, in the unaligned first-camera frame. They require translation RPE
below 0.15 m, ATE and final drift below 0.6 m, and rotation RMSE below 0.1 rad.

`tests/test_vins_marginalization.py` compiles the production QR helper and checks its information matrix, gradient
and objective against independent elimination methods. It also covers rank deficiency, gauge freedom and weak
metric-scale information alongside stiff constraints.

The final WASM build passes `viobench check` and all eight tests across those two test files.

### Recorded datasets

Measured on 2026-10-30 using all eight `sim-ProcTHOR-Train-1-*` recordings in the project's `datasets` directory:
2,400 frames per recording, about 120 seconds each. Both versions use Wasmtime 50.0.1 and the same default runner
limits, including 20 billion fuel per step and 2 GiB memory. Metrics include startup and use the original first
camera frame, without fitted trajectory alignment. The baseline is the example before these improvements;
failed runs are scored over their full recording with the runner's usual hold-last-state behavior.

| Metric | Baseline | Improved |
|---|---:|---:|
| Completed recordings | 4 / 8 | 8 / 8 |
| Held frames after failures | 9,099 | 0 |
| Median translation RPE, 1 second (m) | 0.313 | 0.093 |
| Median absolute trajectory RMSE (m) | 3.041 | 1.580 |
| Median rotation RMSE (rad) | 1.018 | 0.085 |

Translation RPE improves by 70% and absolute trajectory RMSE by 48%. The improved build uses a median
471 million fuel per frame; the median of per-recording 95th percentiles is 834 million. Fuel medians are not
directly comparable to the baseline because four baseline recordings stopped shortly after startup.

| Recording suffix | Completion, baseline → improved | Translation RPE (m) | Absolute trajectory RMSE (m) |
|---|---|---:|---:|
| `00000-00` | failed → complete | 0.459 → 0.111 | 4.466 → 3.162 |
| `00000-01` | complete → complete | 0.257 → 0.130 | 3.049 → 1.188 |
| `00000-02` | failed → complete | 0.342 → 0.063 | 3.033 → 0.785 |
| `00000-03` | failed → complete | 0.361 → 0.165 | 6.064 → 2.312 |
| `00001-00` | failed → complete | 1.481 → 0.057 | 17.364 → 1.111 |
| `00001-01` | complete → complete | 0.285 → 0.108 | 2.258 → 1.972 |
| `00001-02` | complete → complete | 0.067 → 0.078 | 2.426 → 2.800 |
| `00001-03` | complete → complete | 0.145 → 0.057 | 1.095 → 0.594 |

Seven recordings improve on both position metrics; `00001-02` regresses modestly. The estimator uses no recording
identifiers or ground truth. The input camera/IMU calibration and simulator-wide noise model apply to every run.

Reproduce the evaluation after building with `viobench evaluate vins_mono.wasm /path/to/datasets` (omit `-o` to
avoid rendering videos). Local reports from this comparison are `report-baseline/report.json` and
`build/report-final/report.json`; the latter includes an immutable module snapshot and SHA256
`4d21c6a500ed53b482592df0e0174c9b9d9ff6a43e0f3a5e550205ad914c3f31`.

There is no loop closure or global map correction, so long trajectories still accumulate drift. A textureless or
unobservable moving startup can delay metric output. A late initialization can produce an output discontinuity;
the warning and initialization logs distinguish this interval from established tracking. Stationary IMU averages
and startup preintegrations share samples, so the stationary likelihood is an approximation with correlated noise.

## Host debugging and tests

Install host OpenCV with `core`, `imgproc`, `features2d`, `video` and `calib3d` (Ubuntu: `sudo apt install libopencv-dev`), then:

```bash
cmake -S viobench/solution/examples/native/vins_mono -B viobench/solution/examples/native/vins_mono/build-host
cmake --build viobench/solution/examples/native/vins_mono/build-host
viobench feed /path/to/episode -o feed
viobench/solution/examples/native/vins_mono/build-host/debug feed
```

Run these commands from the viobench repository root. Set `OpenCV_DIR` at configure time if CMake cannot find it.
See `docs/DEBUGGING.md` in the viobench repository for the feed format and debugger setup.

From the viobench repository, run `pytest tests/test_vins_mono.py tests/test_vins_marginalization.py` after building.
Set `VIOBENCH_VINS_MONO` to the path of another `.wasm` build to test it. The marginalization test additionally needs
a host C++ compiler and the Eigen headers downloaded by the example build (or `VIOBENCH_EIGEN`).

VINS-Mono is GPL-3.0; Ceres is BSD-3-Clause and Eigen is MPL-2.0.
