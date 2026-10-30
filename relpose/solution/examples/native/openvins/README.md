# openvins

[OpenVINS](https://github.com/rpng/open_vins) 2.7 as a native submission: `solution.cpp` drives `ov_msckf::VioManager`, a
monocular MSCKF with KLT tracking, rotation-aware stationary startup and the dynamic (Ceres) initializer, and
reports the filter's IMU state as the camera's pose and twist relative to the first frame.

```bash
relpose/solution/runtime/python/build.sh                                            # once: the wasm OpenCV it links against
relpose bundle relpose/solution/examples/native/openvins -o openvins.wasm
relpose check openvins.wasm
relpose evaluate openvins.wasm /path/to/sim-ProcTHOR-Train-1-00000-00 -o openvins-results
```

The first configure downloads OpenVINS 2.7, Ceres 2.1.0 and Eigen 3.4.1 (pinned by SHA256, 150 MB, cached in
`build/downloads`); the build takes about 2.5 minutes, 25 s with `CMAKE_BUILD_PARALLEL_LEVEL=8`.

| | |
|---|---|
| `solution.cpp` | the ABI around `VioManager`; sensor and estimator settings in `make_options` |
| `rotation_update.h` | calibrated visual rotation and stationary constraints during startup |
| `CMakeLists.txt` | fetches the sources, builds Ceres (no threads, miniglog) and OpenVINS' `ov_core`, `ov_init`, `ov_msckf` |
| `patches/open_vins.patch` | synchronous initialization, configurable static bias covariance, valid triangulation/refinement, rotation-compensated KLT with forward/backward checks, no `highgui`/`imgcodecs`, a silent Ceres log |
| `compat/boost/` | the three Boost headers OpenVINS includes: timing, χ² quantiles, filesystem |

## Porting to wasm32-wasip1

- **Threads.** wasi-sdk's libc++ builds `std::mutex` and `std::thread` on single-threaded pthread stubs: mutexes work,
  starting a thread throws. OpenVINS starts one per initialization attempt (the patch runs it inline); Ceres runs with
  `CERES_THREADING_MODEL=NO_THREADS`.
- **Exceptions.** Everything is compiled with `-fwasm-exceptions`, like the OpenCV it links; `init` and `step` catch and
  return `-1`. OpenVINS' own `std::exit` on fatal errors traps, which fails the episode the same way.
- **Boost.** `compat/boost` stands in for it: `posix_time` on `std::chrono`, `filesystem` as `std::filesystem` and the
  χ² quantile by inverting the incomplete gamma function (matches SciPy to 1e-14).
- **Determinism.** Fuel must not depend on the machine. Inside the module the `posix_time` clock stands still (OpenVINS
  formats some timing lines even when it does not print them), the Ceres solve has no time limit and logs nothing.
- **Versions.** OpenVINS 2.7 needs `ceres::LocalParameterization`, which Ceres 2.2 removed; Eigen 3.4.0's `JacobiSVD`
  reads out of bounds on non-finite input, which a degenerate dynamic initialization produces.

## From OpenVINS' state to the ABI

- Before initialization: the gyro-integrated attitude, standing still. If initialization takes more than five seconds,
  stderr reports that position and linear velocity are unavailable, repeating every five seconds until initialization.
- At initialization, the camera pose of the first frame in OpenVINS' gravity-aligned frame comes from integrating the
  buffered IMU back from the initial state. Every later pose is relative to it.
- Each step: the filter's state at the frame time (propagated with `fast_state_propagate` while the filter has not
  processed the frame), moved from the IMU to the camera with `T_cam_imu`. The angular velocity is the frame interval's
  mean gyro minus the estimated bias.

The simulator's white IMU noise is specified per sample at 200 Hz, so its standard deviations are divided by
`sqrt(200)` for OpenVINS. Its bias random walks are already continuous-time densities: they must **not** be multiplied
by `sqrt(200)`. Static initialization retains the accelerometer bias uncertainty of the simulator's uniform
`[-0.2, 0.2] m/s²` prior; gravity alone cannot distinguish horizontal bias from tilt.

Static initialization uses a 0.8-second window plus OpenVINS' 0.1-second history margin. This avoids estimating gyro
bias from the initial settling motion. Image-disparity and IMU-variance checks still reject moving input. Dynamic
initialization requires at least two observations per landmark and rejects nonfinite or nonpositive depths before
constructing the Ceres problem. Failed triangulations are not sent to the feature refiner.

KLT uses the integrated gyro rotation to warp each new image into the previous camera orientation before tracking.
This handles patch deformation during wide-angle turns. Forward/backward consistency rejects mismatches, and only
valid tracks enter epipolar RANSAC; stored observations remain in the original calibrated image coordinates.

A hovering camera can turn without providing a baseline for triangulation. During startup, `rotation_update.h` fits
calibrated rotations to spatially distributed feature tracks over half-second windows. Accepted fits update relative
orientation, relative position and zero linear velocity, allowing the drone to turn in place. Consumed measurements
are removed before normal VIO uses them. Once the rotation model fails on sufficient tracks, ordinary MSCKF/SLAM
updates take over permanently. A missing reference or temporary feature dropout gets a bounded one-second retry
without zero-motion constraints. The ordinary zero-velocity detector is disabled: its raw per-frame disparity test can
mistake slow translation for rest and incorrectly absorb that motion into the IMU biases.

`openvins: initialized at ...` in stdout confirms filter initialization. The evaluator's `ok` only means the module
returned usable numbers for every frame; it does not establish successful initialization or accurate tracking.

After building the example, `pytest tests/test_openvins.py` checks stationary startup, moving-camera initialization
with short feature tracks, the warning when images have no features, and tracking after a noisy hovering turn.
The last test renders a textured room from an independent analytic trajectory and checks translation error, orientation error and final drift. Set `RELPOSE_OPENVINS` to test another build.

## Results

On `sim-ProcTHOR-Train-1-00000-00` (120 s, 2,400 frames, 32.16 m travelled), the same evaluation before and after
these changes gives:

| Metric | Previous example | Fixed example |
|---|---:|---:|
| Translation RPE, 1 s | 109.37 m | 0.248 m |
| ATE | 4,386.31 m | 3.000 m |
| Final position drift | about 8,412 m | 4.057 m |

The fixed example initializes at 1.195 s and applies eight rotation-aware startup updates before full VIO at 5.695 s.
Median fuel is about 92 M per frame (p95 149 M). On the neighboring `sim-ProcTHOR-Train-1-00000-01` recording,
RPE improves from 14.554 m to 0.254 m, and ATE from 95.624 m to 1.976 m.

These are measurements on the named recordings, not an accuracy guarantee. Slow monocular motion still leaves
metric scale weakly constrained: the target recording has a fitted Sim(3) scale of 0.552, despite a much improved
trajectory shape. The reported RPE and ATE above do not apply that scale correction.

The host build (`docs/DEBUGGING.md`) needs OpenCV with `core`, `imgproc`, `features2d`, `video` and `calib3d`
(Ubuntu: `sudo apt install libopencv-dev`; elsewhere `-DOpenCV_DIR=...`).

OpenVINS is GPL-3.0, so a module built from it is too; Ceres is BSD-3-Clause, Eigen MPL-2.0.
