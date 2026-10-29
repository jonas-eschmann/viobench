# openvins

[OpenVINS](https://github.com/rpng/open_vins) 2.7 as a native submission: `solution.cpp` drives `ov_msckf::VioManager`, a
monocular MSCKF with KLT tracking, a static initializer with zero-velocity updates and the dynamic (Ceres) initializer, and
reports the filter's IMU state as the camera's pose and twist relative to the first frame.

```bash
relpose/solution/runtime/python/build.sh                                            # once: the wasm OpenCV it links against
relpose/solution/runtime/native/build.sh relpose/solution/examples/native/openvins    # → build/solution.wasm, 8 MB
relpose check relpose/solution/examples/native/openvins/build/solution.wasm
```

The first configure downloads OpenVINS 2.7, Ceres 2.1.0 and Eigen 3.4.1 (pinned by SHA256, 150 MB, cached in
`build/downloads`); the build takes about 2.5 minutes, 25 s with `CMAKE_BUILD_PARALLEL_LEVEL=8`.

| | |
|---|---|
| `solution.cpp` | the ABI around `VioManager`; all tuning is in `make_options` |
| `CMakeLists.txt` | fetches the sources, builds Ceres (no threads, miniglog) and OpenVINS' `ov_core`, `ov_init`, `ov_msckf` |
| `patches/open_vins.patch` | synchronous initialization, valid dynamic-initialization landmarks, no `highgui`/`imgcodecs`, a silent Ceres log |
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

Zero-velocity updates stay on throughout. A hovering or creeping drone gives a monocular MSCKF too little parallax to
triangulate, and without them it would drift on the IMU alone.

Static initialization uses a 0.5-second window plus OpenVINS' 0.1-second history margin, so it can finish during the
simulator's one-second startup hover. Image-disparity and IMU-variance checks still reject moving input. Dynamic
initialization requires at least two observations per landmark and rejects nonfinite or nonpositive depths before
constructing the Ceres problem; a single monocular observation cannot determine depth. Synchronous initialization
reports success immediately so the same camera frame can use the initialized state.

`openvins: initialized at ...` in stdout confirms filter initialization. The evaluator's `ok` only means the module
returned usable numbers for every frame; it does not establish successful initialization or accurate tracking.

After building the example, `pytest tests/test_openvins.py` checks stationary startup, moving-camera initialization
with short feature tracks, and the warning when images have no features. Set `RELPOSE_OPENVINS` to test another build.

## Results

On the demo episode (10 s; the drone hovers, then creeps forward at under 0.4 m/s): RPE 0.126 m/1 s (`python/minimal`:
0.178), 33 M fuel per frame (p95 86 M). Slow, nearly straight flight gives monocular VIO neither the parallax nor the
acceleration to observe scale. `make_options` starts from OpenVINS' EuRoC configuration, adapted to 320 × 240 and a
hovering start: tune it on the training episodes.

The 120-second `sim-ProcTHOR-Train-0-00000-00` recording initializes at 0.995 s, but subsequently loses tracking and
drifts severely. Successful startup does not resolve this remaining tracking limitation.

The host build (`docs/DEBUGGING.md`) needs OpenCV with `core`, `imgproc`, `features2d`, `video` and `calib3d`
(Ubuntu: `sudo apt install libopencv-dev`; elsewhere `-DOpenCV_DIR=...`).

OpenVINS is GPL-3.0, so a module built from it is too; Ceres is BSD-3-Clause, Eigen MPL-2.0.
