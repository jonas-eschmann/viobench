# relpose/solution/examples/ — complete solutions in both flavours

Each directory is a submission as it would be handed in; copy one to start. Python ones are packaged with
`relpose bundle <dir> -o x.wasm`, the native one with `relpose/solution/runtime/native/build.sh <dir>`.

| example | what it shows | fuel/frame |
|---|---|---|
| `python/minimal` | the contract: `class Solution` with `__init__(config)` and `step(t, frame, imu)`; returns the identity pose and a zero twist | 0.15 M |
| `python/dead_reckoning` | IMU only: gravity (and, at rest, gyro bias) from the first half second, integrate; the twist from the integrated velocity and the gyro, moved to the camera centre; the baseline every visual method must beat | 3.6 M |
| `python/klt_imu` | classical: `goodFeaturesToTrack` + `calcOpticalFlowPyrLK` + `findEssentialMat`/`recoverPose` for direction, gyro for rotation, IMU for scale | 140–600 M |
| `python/tiny_cnn` | learned: PyTorch training (`train.py`, also straight from the simulator with `--generate N`), ONNX export, inference through `cv2.dnn` inside the runtime with a numpy fallback | 12 M |
| `native/cpp_minimal` | the same contract in C++ without the Python runtime; `relpose_abi.h` + two CMake lines | 0.7 M |

Fuel numbers are per frame at 320 × 240 (the default input), measured before frames became RGB. `relpose evaluate <wasm>
<episode dirs>` measures them, with the accuracy, on your episodes.

`relpose debug <dir> <episode dir>` runs a Python example in-process with the debugger of your choice; the native
example gets a host `debug` executable from the same `CMakeLists.txt` (`docs/DEBUGGING.md`).
