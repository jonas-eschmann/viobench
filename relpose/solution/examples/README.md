# Examples

Copy one to start. Python examples are packaged with `relpose bundle <dir> -o x.wasm`, the native one with
`relpose/solution/runtime/native/build.sh <dir>`.

| example | method | fuel/frame |
|---|---|---|
| `python/minimal` | identity pose, zero twist | 0.09 M |
| `python/dead_reckoning` | IMU only: gravity and gyro bias from the first half second, then integration | 4.1 M |
| `python/klt_imu` | KLT tracks and the essential matrix for direction, gyro for rotation, IMU for scale | 50–130 M, p95 up to 500 M |
| `python/tiny_cnn` | CNN trained in PyTorch, run through `cv2.dnn` | 14 M |
| `native/cpp_minimal` | the minimal solution in C++ (reads every pixel once) | 2.2 M |

Fuel per frame at 320 × 240 RGB: medians over simulated episodes with the current runtime. Only `klt_imu` depends on
the image content (feature count); the others are constant.
