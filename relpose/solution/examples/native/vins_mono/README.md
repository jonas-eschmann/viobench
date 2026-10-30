# vins_mono

[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) as a native relpose submission. The example runs the
upstream monocular-inertial estimator: visual-inertial initialization, IMU preintegration, sliding-window Ceres
optimization and marginalization. Its pinhole front end follows the upstream KLT pipeline with OpenCV optical
flow, epipolar rejection and spatially distributed Shi-Tomasi features.

Build the example directly from the relpose repository root:

```bash
relpose build-runtime                     # once: supplies the wasm OpenCV libraries
relpose bundle relpose/solution/examples/native/vins_mono -o vins_mono.wasm
relpose check vins_mono.wasm
relpose evaluate vins_mono.wasm /path/to/episode -o vins_mono-results
```

The first CMake configure downloads VINS-Mono at commit `90dabb5ec79946ae42fd2e1e91d4e69aabe1e25d`,
Ceres 2.1.0 and Eigen 3.4.1. Archives are pinned by SHA256. OpenCV comes from the workspace's runtime build;
`relpose build-runtime` builds that complete runtime even though this native example only links its OpenCV libraries.

| File | Purpose |
|---|---|
| `solution.cpp` | relpose ABI, sensor calibration and camera pose/twist output |
| `feature_tracker.h` | pinhole KLT tracking and spatially distributed feature detection |
| `parameters.cpp` | estimator settings without ROS parameter loading |
| `CMakeLists.txt` | dependency downloads and native/WASI build |
| `patches/vins_mono.patch` | portability fixes, synchronous marginalization and deterministic solver limits |
| `compat/` | the small ROS logging and message interfaces required by the estimator sources |

## Estimation and output

Each step consumes RGB pixels and causal IMU rows, using the camera intrinsics and `T_cam_imu` supplied by `init`.
The calibration is fixed; the adapter supports the simulator's undistorted pinhole camera. The estimator runs
synchronously without a ROS runtime, pose graph, loop closure or background workers.

Before metric initialization, the example returns identity pose and zero twist. It reports a warning every five
seconds while initialization remains incomplete. A stationary camera alone does not provide the parallax and
inertial excitation needed to recover monocular metric scale. `relpose check` checks ABI behavior on synthetic
inputs; passing it does not establish successful initialization or tracking accuracy.

Once initialized, buffered IMU measurements recover the first input camera's pose from the oldest optimized
window state. Every subsequent pose is relative to that original camera frame. The output includes the camera/IMU
lever arm, and linear and angular velocities use camera axes. A reset after initialization returns an ABI error
because a fresh estimator would lose the original reference frame.

Optimization uses iteration limits rather than wall-clock deadlines, and marginalization runs on the calling
thread so WASM execution does not require threads. The local sliding window accumulates drift over long runs.
Accuracy and fuel measurements should be obtained with `relpose evaluate` on representative recordings.

The IMU defaults in `parameters.cpp` match the simulator at 200 Hz. VINS-Mono's discrete preintegration uses
per-sample standard deviations: `ACC_N = 0.28` m/s² and `GYR_N = 0.07` rad/s. Its bias noise terms multiply by
the sample interval, so continuous-time random-walk densities become `ACC_W = 0.001 * sqrt(200)` and
`GYR_W = 0.0006 * sqrt(200)`, giving bias covariance increments proportional to elapsed time.

The patch rebuilds the gravity-refinement normal equations after each tangent-plane update, initializes gravity
and pointer state, and rejects nonfinite or nonpositive initialization scale. Unusable Ceres solves return an ABI
error, as do estimator resets after initialization.

## Validation and limitations

`relpose check` and all five tests in `tests/test_vins_mono.py` pass. The regressions cover three invalid camera
calibrations, featureless initialization, and tracking an independently rendered analytic room trajectory with
delayed visual input, a camera/IMU lever arm and exposures between IMU samples. The trajectory test checks
post-initialization ATE below 0.5 m and translation RPE below 0.25 m, plus rotation, velocity and metric-scale bounds.

A smoke evaluation on the recording `sim-ProcTHOR-Train-1-00000-00` initializes at 6.895 s but fails at 6.945 s:
the estimated accelerometer bias reaches 4.033 m/s², exceeding the upstream reset threshold of 2.5 m/s².
Recorded-episode tracking still needs tuning.

## Host debugging and tests

Install host OpenCV with `core`, `imgproc`, `features2d`, `video` and `calib3d` (Ubuntu: `sudo apt install libopencv-dev`), then:

```bash
cmake -S relpose/solution/examples/native/vins_mono -B relpose/solution/examples/native/vins_mono/build-host
cmake --build relpose/solution/examples/native/vins_mono/build-host
relpose feed /path/to/episode -o feed
relpose/solution/examples/native/vins_mono/build-host/debug feed
```

Run these commands from the relpose repository root. Set `OpenCV_DIR` at configure time if CMake cannot find it.
See `docs/DEBUGGING.md` in the relpose repository for the feed format and debugger setup.

From the relpose repository, run `pytest tests/test_vins_mono.py` after building. Set `RELPOSE_VINS_MONO` to the
path of another `.wasm` build to test it.

VINS-Mono is GPL-3.0; Ceres is BSD-3-Clause and Eigen is MPL-2.0.
