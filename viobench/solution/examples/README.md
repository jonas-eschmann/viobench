# Examples

`viobench init python|native <dir>` starts a workspace with `python/minimal` or `native/cpp_minimal` as `solution/`;
copy the other examples over `solution/` to use them.
`viobench bundle <dir> -o x.wasm` packages the Python examples and compiles the native ones.
See each native example's README for library prerequisites and direct build commands.

| example | method | fuel/frame |
|---|---|---|
| `python/minimal` | identity pose, zero twist | 0.09 M |
| `python/dead_reckoning` | IMU only: gravity and gyro bias from the first half second, then integration | 4.1 M |
| `python/klt_imu` | KLT tracks and the essential matrix for direction, gyro for rotation, IMU for scale | 50–130 M, p95 up to 500 M |
| `python/tiny_cnn` | CNN trained in PyTorch, run through `cv2.dnn` | 14 M |
| `native/cpp_minimal` | the minimal solution in C++ (reads every pixel once) | 2.2 M |
| `native/openvins` | [OpenVINS](https://github.com/rpng/open_vins) (GPL-3.0): monocular MSCKF, KLT tracks, zero-velocity updates, Ceres initialization | 33 M, p95 100 M |
| [`native/orbslam3`](native/orbslam3/README.md) | ORB-SLAM3 (GPL-3.0): monocular-inertial ORB tracking and mapping | see README (partial run) |
| [`native/vins_mono`](native/vins_mono/README.md) | VINS-Mono (GPL-3.0): KLT tracks, IMU preintegration, sliding-window optimization | not yet measured |

Fuel per frame at 320 × 240 RGB: medians over simulated episodes with the current runtime (`openvins`: the demo
episode). Only `klt_imu`, `openvins`, `orbslam3`, and `vins_mono` depend on the image content (feature count); the others are constant.
