# openvins

[OpenVINS](https://github.com/rpng/open_vins) 2.7 as a native submission: KLT tracking, rotation-aware stationary
startup and fixed-lag visual-inertial optimization. Two OpenVINS frontends initialize three smoother hypotheses;
their robust consensus reports the camera pose and twist relative to the first frame.

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
| `solution.cpp` | the ABI, two independent `VioManager` frontends and three smoother hypotheses; sensor settings in `make_options` |
| `rotation_update.h` | calibrated visual rotation and stationary constraints during startup |
| `fixed_lag_vio.h` | Ceres visual-inertial optimization, IMU prediction and square-root marginalization of old measurements |
| `consensus_output.h` | geometric-median position/velocity consensus, rotation averaging and causal position correction |
| `CMakeLists.txt` | fetches the sources, builds Ceres (no threads, miniglog) and OpenVINS' `ov_core`, `ov_init`, `ov_msckf` |
| `patches/open_vins.patch` | synchronous initialization, configurable static bias covariance, valid triangulation/refinement, rotation-compensated KLT with forward/backward checks and direct keyframe tracking, no `highgui`/`imgcodecs` |
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
- During startup: the filter's state at the frame time, propagated with `fast_state_propagate` if needed.
- After startup: each smoother's state, propagated with IMU preintegration between keyframes. Every hypothesis
  moves the state from the IMU to the camera with `T_cam_imu` before combination. Angular velocity is the frame
  interval's mean gyro minus the corresponding estimated bias. If an exposure falls between IMU samples, the newest
  sample covers the remaining fractional interval, up to one IMU sampling period.

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
One frontend also matches each smoother keyframe directly against the previous keyframe, using the intervening
frame-to-frame tracks as initial guesses. Valid direct matches replace accumulated track coordinates; unsuccessful
matches retain the ordinary tracks. This reduces accumulated tracking error while preserving recovery between
keyframes. The other frontend retains ordinary frame-to-frame tracking.

A hovering camera can turn without providing a baseline for triangulation. During startup, `rotation_update.h` fits
calibrated rotations to spatially distributed feature tracks over half-second windows. Accepted fits update relative
orientation, relative position and zero linear velocity, allowing the drone to turn in place. Consumed measurements
are removed before normal VIO uses them. Once the rotation model fails on sufficient tracks, fixed-lag tracking
takes over permanently. A missing reference or temporary feature dropout gets a bounded one-second retry
without zero-motion constraints. The ordinary zero-velocity detector is disabled: its raw per-frame disparity test can
mistake slow translation for rest and incorrectly absorb that motion into the IMU biases.

Each smoother jointly refines poses, velocities, IMU biases and landmarks in a 2.4-second window: 13 keyframes at
5 Hz, up to 100 landmarks and six Ceres iterations per update. It uses calibrated reprojection factors with a robust
loss and continuous-time IMU preintegration. New tracks enter in spatially balanced cohorts, keeping visual constraints
available as old tracks leave the window. IMU prediction supplies every intervening output frame.

The three hypotheses use ordinary tracking, direct keyframe tracking, and ordinary tracking with weaker visual
weight. The first two use pixel sigma 0.75 and Huber scale 2; the third uses 1.5 and 1, giving one-quarter of the
visual objective with the same 1.5-pixel robust-loss transition. The third graph shares the ordinary frontend's
measurements and initialization but maintains its own states, landmarks and marginal prior.

Each EKF supplies its smoother's initial state and covariance once. After that, each smoother uses its own raw image
and IMU factors; repeatedly using the EKF posterior would count those measurements twice. The EKFs continue running
to provide the gyro biases used for image warping. Marginalization consumes complete landmark segments and the oldest
IMU factor into a prior in the same quaternion tangent coordinates as Ceres. Column-scaled QR elimination operates
directly on Jacobians, avoiding the squared conditioning of normal equations. Compression retains weak nonzero prior
rows and removes only constant residual rows; exact absent-state columns stay zero. Invalid integration or nonfinite
constraints return an ABI error instead of silently advancing an old estimate with stale velocity.

Consensus takes the equal-weight geometric median of positions and velocities in the common first-camera frame,
an intrinsic mean of rotations, and the mean angular velocity in that common frame. A causal observer predicts
position from velocity and applies position corrections with a two-second time constant. It predicts the current
timestamp, so this time constant is not a two-second trajectory delay. The returned linear twist includes the
position correction at the current endpoint, and both twists are expressed in the returned camera axes.
These are correlated hypotheses, not independent measurements: consensus never feeds a posterior or an output
correction back into any estimator. It uses no ground truth, future frames or episode-specific selection.

`openvins: initialized at ...` in stdout confirms filter initialization. The evaluator's `ok` only means the module
returned usable numbers for every frame; it does not establish successful initialization or accurate tracking.

After building the example, `pytest tests/test_openvins.py` checks stationary startup, moving-camera initialization
with short feature tracks, the warning when images have no features, and tracking after a noisy hovering turn both
with and without a one-second camera blackout, and with exposures between IMU samples. The three tracking cases
render a textured room from an independent analytic trajectory and check translation error, orientation error and
final drift. There are six integration tests. Set `RELPOSE_OPENVINS` to test another build.
`pytest tests/test_openvins_math.py` also checks coordinate invariance, outlier rejection, analytic motion without
velocity lag, and transactional failure handling; it needs a native C++ compiler and the Eigen headers fetched by
the example build.

## Results

The following comparison uses the previous single fixed-lag smoother at commit `695372a` as its baseline. It
already includes rotation-aware startup, gyro-compensated tracking and calibrated IMU noise. Each recording lasts
120 seconds (2,400 frames). Episode names below omit the common `sim-ProcTHOR-Train-1-` prefix. Translation RPE uses
one-second intervals; ATE is measured in the first camera frame. Neither metric applies fitted rotation, translation
or scale alignment.

| Episode | Previous RPE (m) | Consensus RPE (m) | Previous ATE (m) | Consensus ATE (m) |
|---|---:|---:|---:|---:|
| `00000-00` | 0.098 | 0.077 | 0.818 | 0.866 |
| `00000-01` | 0.150 | 0.082 | 1.342 | 0.686 |
| `00000-02` | 0.121 | 0.109 | 1.011 | 0.832 |
| `00000-03` | 0.184 | 0.122 | 2.042 | 1.562 |
| `00001-00` | 0.120 | 0.088 | 2.370 | 1.379 |
| `00001-01` | 0.252 | 0.100 | 2.699 | 0.790 |
| `00001-02` | 0.124 | 0.101 | 1.728 | 1.595 |
| `00001-03` | 0.111 | 0.073 | 0.760 | 0.666 |
| **Median** | **0.123** | **0.094** | **1.535** | **0.849** |

All eight episodes complete without held frames. Median RPE decreases 23% and median ATE decreases 45%.
Relative error improves on all eight recordings; absolute error improves on seven. On `00000-00`, ATE regresses
5.9%, from 0.818 m to 0.866 m, while RPE improves 21%. Median final drift decreases from 1.597 m to 1.138 m,
but final drift does not improve on every recording. There is no loop closure or global trajectory correction.

The extra accuracy costs computation. The median of per-episode median fuel rises from 96 M to 224 M per frame
(2.3 times); the median p95 rises from 1.224 B to 2.648 B. All recordings remain within the 20 B per-step limit.
Evaluation reports retain exact metrics and runtime measurements; these results are measurements on the named
recordings, not an accuracy guarantee.

The host build (`docs/DEBUGGING.md`) needs OpenCV with `core`, `imgproc`, `features2d`, `video` and `calib3d`
(Ubuntu: `sudo apt install libopencv-dev`; elsewhere `-DOpenCV_DIR=...`).

OpenVINS is GPL-3.0, so a module built from it is too; Ceres is BSD-3-Clause, Eigen MPL-2.0.
