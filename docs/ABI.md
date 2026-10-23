# Solution ABI (version 1)

A submission is one WebAssembly file (`wasm32-wasip1`, *reactor*: no `_start`). The harness instantiates it once per
episode, calls `init` with a JSON config, then `step` once per camera frame, and reads a pose back after each step.
State lives inside the module between steps. The Python runtime (`runtime.wasm`) implements this ABI for you; C, C++
and Rust solutions implement it directly (`examples/cpp-minimal`).

## Exports

| Export | Wasm type | C prototype | Notes |
|---|---|---|---|
| `memory` | memory | – | the module's linear memory (default export of wasi-sdk / Rust) |
| `_initialize` | `() → ()` | – | emitted by wasi-libc for reactors; the harness calls it once if present |
| `abi_version` | `() → i32` | `int32_t abi_version(void)` | must return `1` |
| `alloc` | `(i32) → i32` | `void *alloc(uint32_t size)` | `malloc`; ≥ 8-byte aligned; the harness never frees |
| `load_bundle` | `(i32, i32) → i32` | `int32_t load_bundle(const uint8_t *zip, uint32_t size)` | optional; only the Python runtime uses it (the bundled directory) |
| `init` | `(i32, i32) → i32` | `int32_t init(const char *json, uint32_t size)` | once, before any `step` |
| `step` | `(f64, i32, i32, i32, i32) → i32` | `int32_t step(double t, const uint8_t *frame, const double *imu, int32_t imu_count, double *out)` | once per frame |

Call sequence per episode: `instantiate → _initialize → [load_bundle] → init → alloc × 3 → step × N → drop`.
There is no `reset`: every episode gets a fresh instance (instantiation costs ~0.1 s, Python init ~0.1 s).

## Manifest (optional custom section `eecs116.manifest`)

A JSON object the harness reads from the binary *before* `init`: `{"channels": 1 | 3, "entry": "module:Class"}`.
`channels` selects what `step` receives: grayscale (`1`, the default) or RGB (`3`). `python -m eecs116 bundle` writes it
from `eecs116.json` or `--channels`; native modules add it with `python -m eecs116 manifest solution.wasm --channels 3`.
The config then reports `image.channels`/`image.layout` accordingly and `image.color` tells whether the colour is real
(real-data sets are grayscale and are replicated into three channels for RGB solutions).

## Buffers

All numbers are little-endian; integers are `i32`, floats `f64`.

- **`frame`**: `height × width × channels` bytes of `uint8`, row-major, C-contiguous; channels is 1 (grayscale, default)
  or 3 (RGB, if the manifest asks for it; grayscale is the ITU-R 601 luma of the stored RGB, `cv2.COLOR_RGB2GRAY`). For
  `channels == 1` the Python runtime hands the solution a `(H, W)` array, otherwise `(H, W, 3)`. Size and channel
  count come from the config and are constant for the episode. The buffer is **reused for the next frame**: keep a copy
  if you need the old frame.
- **`imu`**: `imu_count` rows of 7 `f64`: `[t, ax, ay, az, gx, gy, gz]` — time in seconds, specific force in m/s²
  (includes gravity, body frame), angular rate in rad/s (body frame). The rows are all IMU samples with
  `t_previous_frame < t ≤ t_frame` in time order; `imu_count` may be 0 (the first frame typically carries the hover
  samples before it). Also reused next step.
- **`t`**: timestamp of the frame in seconds (mid-exposure), same clock as the IMU rows.
- **`out`**: 7 `f64` `[x, y, z, qx, qy, qz, qw]` — the pose of the *current camera frame* expressed in the
  *first camera frame* (`T_c0_ck`: `p_c0 = R p_ck + t`), unit quaternion with scalar last. Written on status `0`.

The harness writes `frame`/`imu` into memory it obtained from `alloc`, calls `step`, then reads `out`. The guest
owns nothing the host passes except the right to read it during the call; the config bytes given to `init` are
valid only during the call. Pointers are offsets into `memory`; the host re-reads `memory`'s base address after every
call because the memory may grow and move.

## Status codes

| code | name | meaning | harness reaction |
|---|---|---|---|
| `0` | ok | `out` holds a pose | recorded |
| `1` | no_estimate | the solution has no pose for this frame yet | recorded as missing (counts as a failed frame in ATE) |
| `-1` | exception | the solution raised; traceback on stderr | frame missing, episode continues; after 10 consecutive failures the episode is abandoned |
| `-2` | protocol | `step` before `init`, `init` twice, negative count | episode failed |
| `-3` | runtime | the runtime itself could not start | submission failed |
| `-4` | no_solution | runtime without bundled code | submission failed |
| `-5` | not_finite | pose contains NaN/inf | frame missing |

**Traps** (`unreachable`, out-of-bounds memory access, stack overflow, running out of fuel, wall-clock deadline,
`exit()`/`proc_exit`) kill the instance: the episode is marked failed and the next episode gets a new instance.
In Python, `sys.exit()` is caught like any exception (status `-1`); `os._exit()` is a trap.

## Config JSON

```json
{
  "abi_version": 1,
  "image": {"width": 320, "height": 240, "channels": 1, "dtype": "uint8", "layout": "HW", "color": false, "source_channels": 3},
  "camera": {"fx": 190.7, "fy": 190.7, "cx": 159.5, "cy": 119.5, "model": "pinhole", "distortion": []},
  "imu": {"rate_hz": 200.0,
          "accel_noise_std": 0.28, "gyro_noise_std": 0.07,
          "accel_bias_init_max": 0.2, "gyro_bias_init_max": 0.02,
          "accel_bias_random_walk": 0.001, "gyro_bias_random_walk": 0.0006,
          "gravity": [0.0, 0.0, -9.81],
          "T_cam_imu": [16 numbers, row-major 4x4: IMU/body coordinates -> camera coordinates],
          "T_imu_cam": [16 numbers, its inverse]},
  "episode": {"name": "...", "frame_rate_hz": 20.0, "num_frames": 400, "hover_seconds": 1.0},
  "pose": {"frame": "camera0", "order": "xyz_qxyzw", "timestamp": "mid_exposure"}
}
```

Camera convention is OpenCV's: x right, y down, z forward (optical axis). The body/IMU frame is the drone's: x
forward, y left, z up (FLU). Intrinsics vary per episode within the announced ranges — read them from the config.

## Resource accounting

Every `step` runs with a fresh fuel budget (`Limits.step_fuel`) and a wall-clock deadline. The leaderboard uses
**fuel per frame** (≈ one unit per executed wasm instruction; identical on any machine) as the compute axis and
reports wall time only for intuition. The memory of the instance is capped (`Limits.memory_bytes`); exceeding it
surfaces as an allocation failure inside the guest (Python: `MemoryError`), not as a trap.

## Python contract (runtime.wasm)

```python
class Solution:
    def __init__(self, config: dict): ...
    def step(self, t: float, frame: np.ndarray, imu: np.ndarray) -> Sequence[float] | None:
        """frame: uint8 (H, W) or (H, W, C), read-only view valid until the next step;
        imu: float64 (N, 7) rows [t, ax, ay, az, gx, gy, gz]. Return [x, y, z, qx, qy, qz, qw] or None."""
```

The entry point defaults to `solution:Solution` (`solution.py` or `solution/__init__.py` at the root of the
bundled directory) and can be changed with `eecs116.json` (`{"entry": "mypkg.estimator:Estimator"}`). Data files in
the directory are reachable through `import eecs116_bundle as bundle` (`bundle.read_bytes(name)`, `bundle.open(name)`,
`bundle.exists(name)`, `bundle.listdir(prefix)`), `pkgutil.get_data`, or the module's `__file__` under `/bundle/`.
Available inside the runtime: CPython 3.14 stdlib (no zlib/ssl/sqlite3/ctypes), numpy 2.3 (scalar build), OpenCV 4.12
(`core`, `imgproc`, `features2d`, `flann`, `calib3d`, `video`, `dnn` with ONNX import). No threads, no filesystem,
no network, no clock you should rely on for scoring.
