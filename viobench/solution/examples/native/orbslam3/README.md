# orbslam3

[ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) as a native `viobench` solution: monocular ORB features,
IMU preintegration, inertial initialization and local bundle adjustment. Tracking and local mapping run
synchronously so the same code runs inside the single-threaded `wasm32-wasip1` sandbox and the host debugger.
This example disables loop closure, map merging, the viewer and atlas persistence.

## Build and run

Build the example directly from the `viobench` source repository root:

```bash
viobench/solution/runtime/python/build.sh     # once: builds the wasm OpenCV libraries
viobench bundle viobench/solution/examples/native/orbslam3 -o orbslam3.wasm
viobench check orbslam3.wasm
viobench evaluate orbslam3.wasm /path/to/episode -o orbslam3-results
pytest tests/test_orbslam3.py
```

The first configure downloads a pinned ORB-SLAM3 source archive and Eigen, verified by SHA256. The upstream
archive is large (about 572 MiB, including its examples and vocabulary); downloads and extracted sources stay
in the example's ignored `build/` directory. The build converts the upstream ORB vocabulary to a compact binary
and embeds it in the module. No vocabulary, calibration file, preopened directory or network access is needed
when evaluating the resulting `.wasm`.

The host build uses an installed OpenCV with `core`, `imgproc`, `features2d` and `calib3d`.
Starting from the same repository root:

```bash
cd viobench/solution/examples/native/orbslam3
cmake -S . -B build-host                    # add -DOpenCV_DIR=... if needed
cmake --build build-host --parallel
viobench feed /path/to/episode -o build-host/feed
build-host/debug build-host/feed
```

See the repository's `docs/DEBUGGING.md` for the native debugger workflow. `VIOBENCH_OPENCV_DIR` is a CMake
cache option for reusing an existing wasm OpenCV build when its location differs from the workspace default.

## Adapter and estimator behavior

- `solution.cpp` implements ABI version 1, requests 320 × 240 RGB images, validates calibration and constructs
  ORB-SLAM3's settings in memory. `Tbc` is the inverse of the ABI's `T_cam_imu`.
- `viobench_slam.h` and `viobench_slam.cpp` connect the upstream tracker, atlas, vocabulary and local mapper.
  The mapper processes its keyframe queue after each image; it never creates a worker thread.
- `patches/orbslam3.patch` makes upstream processing synchronous and removes unavailable display dependencies.
  `compat/` supplies the unused viewer/system interfaces and serialization declarations; atlas serialization
  is not part of this build.
- The full upstream vocabulary, ORB extractor, feature matching, visual initialization, IMU preintegration and
  optimization are used. This is not a reimplementation of their algorithms.

The camera runs at 20 Hz and IMU at 200 Hz, matching the challenge. White gyro/accelerometer noise is converted
from the simulator's per-sample standard deviation to continuous densities; bias random walks are already
continuous densities. A causal held sample covers the final fractional IMU interval when an exposure falls
between samples.

Before inertial initialization, visual bundle adjustment with only one fixed camera has an arbitrary scale.
`compat/ScaleGauge.h` preserves one existing camera baseline length in that optimization, allowing rotation
and translation direction to refine. This prevents repeated synchronous updates from shrinking the visual
map; IMU initialization subsequently establishes metres. Optimized keyframe poses are copied back to the
current tracker frame before returning an estimate.

The synchronous mapper can process closely spaced keyframes that a background mapper would leave queued.
Its two-keyframe short-baseline reset heuristic is disabled: a short pause after metric initialization keeps
the map and continues local inertial optimization, while later scale/bias refinements wait for enough motion.
Tracking-requested resets still take effect. The port also initializes the frame's IMU mutex before the
featureless-image early return, which matters in the native debug build.

Everything runs inside one metered `step`, so the cost of each bundle adjustment must stay bounded. Inertial
initialization runs the whole-map inertial bundle adjustment once, while the map is still small, with ten
iterations instead of upstream's hundred. The two later refinements (upstream's VIBA 1 and 2, after 5 s and 15 s
of motion) keep the inertial-only optimization of scale, gravity direction, biases and velocities over all
keyframes, but skip the whole-map bundle adjustment that follows it upstream: its cost grows with the map, and it
exceeded the step's fuel on EuRoC episodes where a refinement came late, e.g. after a long pause. Local bundle
adjustment then refines the latest keyframes with its upstream iteration schedule. Replacing the whole-map
adjustment by the 25-keyframe local inertial window instead lost tracking right after the refinement on most EuRoC
episodes.

Monocular translation has arbitrary scale until inertial initialization succeeds. Until then the adapter
returns identity pose and zero twist, with a warning every five seconds. Stationary scenes and pure rotation
cannot establish monocular metric scale. `orbslam3: initialized at t=...` confirms a metric estimate;
`viobench check` passing alone does not establish successful tracking.

After initialization, the output is the current camera in the **first input camera frame**, not the first
successfully tracked frame. The origin keyframe's IMU state is integrated backward through buffered startup
measurements to the first exposure. Recomputing this reference follows later scale, gravity and bias
refinements. Linear velocity includes the camera's lever arm about the IMU; both linear and angular velocity
are expressed in the current camera axes. Startup integration is an estimate and can accumulate error during
long periods without visual initialization.

A temporary tracking loss holds the last pose with zero twist and logs a warning. Losing the initialized map
returns an ABI error: an unrelated new map cannot preserve the original camera coordinate frame. There is no
loop closure or global map merging to remove long-term drift.

## Validation

`pytest tests/test_orbslam3.py` checks invalid calibration, featureless input and metric tracking in a rendered
textured room with an independent analytic trajectory, a nonzero camera–IMU offset and exposures between IMU
samples. Set `VIOBENCH_ORBSLAM3=/path/to/other.wasm` to test another build. Tests skip if no module is built.

The WASM 18-second room regression passes with 0.056 m ATE and 0.058 m one-second translation RPE over
seconds 10–18, measured in the original first-camera frame. Its maximum step uses 15.1 B fuel, below the
default 20 B limit. The initial half-second of images is blank while the camera moves, so this also exercises
reconstruction of the first exposure from a later visual keyframe.
The optimized native build passes the same trajectory bounds. An AddressSanitizer run also completes without
reported errors, with leak and container-overflow checks disabled for the mixed instrumented/uninstrumented
libraries. Native and WASM trajectories can differ numerically; use the WASM module for scored accuracy and
fuel measurements.

The recorded-episode smoke test `sim-ProcTHOR-Train-1-00000-00` processes 827 of 2,400 frames before losing
its initialized map (about 41 seconds). The adapter then returns the documented ABI error. Fuel over those
processed frames has a 1.08 B median and 2.12 B p95. This integration still needs estimator tuning for reliable
completion of the challenge recordings; the synthetic regression establishes correct operation on its test
trajectory. Use `viobench evaluate` to measure accuracy and completion on other recordings.

## License

ORB-SLAM3 and a module linked with it are GPL-3.0; its bundled dependencies retain their upstream licenses
(DBoW2, g2o and Sophus). Eigen is MPL-2.0 and OpenCV is Apache-2.0. The pinned source downloads include their
license files. See [ORB-SLAM3's license](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/LICENSE).
