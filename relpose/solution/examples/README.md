# relpose/solution/examples/ — complete solutions in both flavours

Each directory is a submission as it would be handed in; copy one to start. Python ones are packaged with
`relpose bundle <dir> -o x.wasm`, the native one with `relpose/solution/runtime/native/build.sh <dir>`.

| example | what it shows | ATE default-val [m] | fuel/frame |
|---|---|---|---|
| `python/minimal` | the contract: `class Solution` with `__init__(config)` and `step(t, frame, imu)`; returns the identity | — | 0.15 M |
| `python/dead_reckoning` | IMU only: gravity and gyro bias from the initial hover, integrate; the baseline every visual method must beat | 18.8 | 3.6 M |
| `python/klt_imu` | classical: `goodFeaturesToTrack` + `calcOpticalFlowPyrLK` + `findEssentialMat`/`recoverPose` for direction, gyro for rotation, IMU for scale | 4.3 | 140–600 M |
| `python/tiny_cnn` | learned: PyTorch training (`train.py`, also straight from the simulator with `--generate N`), ONNX export, inference through `cv2.dnn` inside the runtime with a numpy fallback; `solution.json` manifest | 1.2 | 12 M |
| `native/cpp_minimal` | the same contract in C++ without the Python runtime; `relpose_abi.h` + two CMake lines | — | 0.7 M |

Fuel numbers are per frame at 320 × 240 grayscale (the default input); see `README.md` at the top for the full table
over all public conditions. `relpose evaluate <wasm> build/datasets/public/default-val` reproduces them.

Every Python example ends with a three-line `if __name__ == "__main__":` footer, so `python solution.py <episode dir>`
runs it in-process with the debugger of your choice; the native example gets a host `debug` executable from the same
`CMakeLists.txt` (`docs/DEBUGGING.md`).
