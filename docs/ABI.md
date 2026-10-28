# Solution ABI (version 1)

A submission is one WebAssembly file (`wasm32-wasip1`, *reactor*: no `_start`). The harness instantiates it once per
episode, calls `init` with a JSON config, then `step` once per camera frame, and reads the state of the camera (pose
and twist) back after each step.
State lives inside the module between steps. The Python runtime (`runtime.wasm`) implements this ABI for you; C, C++
and Rust solutions implement it directly (`relpose/solution/examples/native/cpp_minimal`).

## Exports

| Export | Wasm type | C prototype | Notes |
|---|---|---|---|
| `memory` | memory | – | the module's linear memory (default export of wasi-sdk / Rust) |
| `_initialize` | `() → ()` | – | emitted by wasi-libc for reactors; the harness calls it once if present |
| `abi_version` | `() → i32` | `int32_t abi_version(void)` | must return `1` |
| `alloc` | `(i32) → i32` | `void *alloc(uint32_t size)` | `malloc`; ≥ 8-byte aligned; the harness never frees |
| `frame_width` | `() → i32` | `int32_t frame_width(void)` | optional: `320` (the default without it) or `640`, see "Frame format"; negative: an error |
| `init` | `(i32, i32) → i32` | `int32_t init(const char *json, uint32_t size)` | once, before any `step` |
| `step` | `(f64, i32, i32, i32, i32) → i32` | `int32_t step(double t, const uint8_t *frame, const double *imu, int32_t imu_count, double *out)` | once per frame |

Call sequence per episode: `instantiate → _initialize → frame_width → init → alloc × 3 → step × N → drop` (exports
that are absent are skipped). The harness guarantees this order (`step` only after a successful `init`), so a solution
need not check it. All calls before the first `step` share the init fuel budget.
There is no `reset`: every episode gets a fresh instance (instantiation costs ~0.1 s, Python init ~0.1 s). The host
treats Python and native modules identically; it does not know what runs inside.

## How Python code travels (`relpose bundle`)

The student's directory is zipped and embedded into `runtime.wasm` as a **data segment** placed just above the
runtime's initial memory (the memory minimum grows by the zip's size), and a 32-byte descriptor inside the runtime
(`magic "RELPOSEBUNDLEv1", address, length`) is patched to point at it. The module is therefore self-contained: in
`frame_width` the embedded shim reads the descriptor, opens the zip in its own memory and imports `solution:Solution`
from it (import errors surface there, with the traceback on stderr); `init` constructs it. The bundler is pure Python section surgery; no compiler is involved, and the zip can be as large as the
memory cap allows. `relpose unbundle` extracts it again.

## Frame format

Frames are always **RGB**, `uint8`, `height × width × 3`, and the only choice a solution makes is the resolution:
**320 × 240** (default) or **640 × 480**. Nothing else about the format varies between episodes or sets, and the
config does not say where an episode comes from.

- **Resolution**: the solution selects the resolution it consumes (see below), e.g. the lower one to save compute:
  more pixels cost fuel in everything a solution runs on them. Any up- or downscaling from the resolution a set is
  stored at is left to the harness, which rescales the intrinsics to match (pixel-centre convention).
- **Camera**: rectified pinhole without distortion.

**Requesting the resolution: `frame_width`.** The solution declares it in its code, and the harness asks *before*
`init`: `640` gives 640 × 480 frames, `320` or no declaration 320 × 240; anything else fails the episode. Native
modules export `int32_t frame_width(void)`; Python solutions set a class attribute, `class Solution: frame_width = 640`
(the Python runtime exports the function and returns it).

## Buffers

All numbers are little-endian; integers are `i32`, floats `f64`.

- **`frame`**: `height × width × 3` bytes of `uint8` RGB, row-major, C-contiguous (see "Frame format"); the Python
  runtime hands the solution a `(H, W, 3)` array. The size comes from the config and is constant for the episode. The
  buffer is **reused for the next frame**: keep a copy if you need the old frame.
- **`imu`**: `imu_count` rows of 7 `f64`: `[t, ax, ay, az, gx, gy, gz]` — time in seconds, specific force in m/s²
  (includes gravity: at rest it reads ≈ 9.81 opposite to gravity) and angular rate in rad/s, both in the IMU frame that
  `T_cam_imu` refers to. The rows are all IMU samples with `t_previous_frame < t ≤ t_frame` in time order; the first
  frame gets one frame interval as well (`t_0 − 1/20 s < t ≤ t_0`), never more, whatever a set recorded before it.
  Frames arrive at 20 Hz and the IMU at 200 Hz, so `imu_count` is always **10**. The data pipeline guarantees this:
  a dropped frame is replaced by a repeat of the previous frame (at its own timestamp), dropped IMU samples by
  repeats of the previous sample, and an IMU recorded at another rate is linearly interpolated. Spacing between rows
  is only nominal (≈ 5 ms); go by the timestamps. Also reused next step.
- **`t`**: timestamp of the frame in seconds (mid-exposure of the centre row), same clock as the IMU rows.
- **`out`**: 13 `f64` `[x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]` — the state of the *current camera frame*
  at `t`, all 13 written on status `0` (the host fills `out` with NaN before every call, so a value left out fails the
  step):
  - `[x, y, z, qx, qy, qz, qw]`: its pose expressed in the *first camera frame* (`T_c0_ck`: `p_c0 = R p_ck + t`), unit
    quaternion with scalar last;
  - `[vx, vy, vz, wx, wy, wz]`: its twist in its *own* axes (x right, y down, z forward), the derivative of the same
    `T_c0_ck`: `v = Rᵀ dt/dt` is the velocity of the camera centre in m/s and `w` (with `[w]× = Rᵀ dR/dt`) the angular
    velocity in rad/s. Flying straight ahead at 2 m/s while yawing left at 0.5 rad/s is `v = (0, 0, 2)`,
    `w = (0, −0.5, 0)`, however the camera is oriented relative to frame 0. A filter that tracks the IMU's velocity
    `v_w` in a world frame converts with `v = R_cam_imu (R_wiᵀ v_w + w_i × r)` and `w = R_cam_imu w_i`, where `w_i`
    is the IMU's angular velocity and `r = −R_cam_imuᵀ t_cam_imu` the camera centre in IMU coordinates (lever arm).

The harness writes `frame`/`imu` into memory it obtained from `alloc`, calls `step`, then reads `out`. The guest
owns nothing the host passes except the right to read it during the call; the config bytes given to `init` are
valid only during the call. Pointers are offsets into `memory`; the host re-reads `memory`'s base address after every
call because the memory may grow and move.

## Status codes

| code | from `init` | from `step` |
|---|---|---|
| `0` | ready | `out` holds the pose and the twist |
| anything else (return `-1`) | error | error |

Explain an error on stderr (the Python runtime prints the traceback there).

**Every step must deliver a usable state.** An error from `init` or `step`, a state with a value that is not finite
(one left unwritten included) or a pose with an all-zero quaternion, or a trap ends the episode there and marks it
failed. It is still scored: the frames from the failure on repeat the last pose, standing still (identity and zero
twist if there was none). There is no "no estimate": until you have one, return your best guess (at rest: identity
and zero twist).

**Traps** (`unreachable`, out-of-bounds memory access, stack overflow, running out of fuel, wall-clock deadline,
`exit()`/`proc_exit`) kill the instance; the next episode gets a new instance. In Python, `sys.exit()` is caught like
any exception (an error); `os._exit()` is a trap.

## Config JSON

```json
{
  "width": 320, "height": 240,
  "fx": 190.7, "fy": 190.7, "cx": 159.5, "cy": 119.5,
  "T_cam_imu": [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
}
```

`width`/`height` are the delivered frame size (what `frame_width` asked for), `fx, fy, cx, cy` the pinhole intrinsics
at that size in pixels, `T_cam_imu` the 4 × 4 transform from IMU coordinates to camera coordinates (rows of the
matrix; camera axes as in OpenCV: x right, y down, z forward). The intrinsics vary per episode, so read them.

That is all a solution is told, on purpose: only what any calibrated camera-IMU dataset provides. Beyond the fixed
20 Hz / 200 Hz above there are no rates, no IMU noise model, no episode length, no statement about how an episode starts
(at rest or in flight), no shutter type or readout time, no colour flag and no name. A solution that needs any of these
estimates it from the stream.

## Resource accounting

Every `step` runs with a fresh fuel budget (`Limits.step_fuel`) and a wall-clock deadline. The leaderboard uses
**fuel per frame** (≈ one unit per executed wasm instruction; identical on any machine) as the compute axis and
reports wall time only for intuition. The memory of the instance is capped (`Limits.memory_bytes`); exceeding it
surfaces as an allocation failure inside the guest (Python: `MemoryError`), not as a trap.

## Python contract (runtime.wasm)

```python
class Solution:
    frame_width: int = 320  # optional; 640 for 640 x 480 frames
    def __init__(self, config: dict): ...
    def step(self, t: float, frame: np.ndarray, imu: np.ndarray) -> Sequence[float]:
        """frame: uint8 (H, W, 3) RGB, read-only view valid until the next step;
        imu: float64 (N, 7) rows [t, ax, ay, az, gx, gy, gz]. Return the 13 values of `out`
        [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]; anything else (None or a pose alone included) or raising
        is an error and ends the episode."""
```

The entry point is always `Solution` in `solution.py` (or `solution/__init__.py`) at the root of the bundled
directory; a class that lives elsewhere is re-exported there (`from mypkg.estimator import Estimator as Solution`).
Data files in the directory are reachable through `import solution_bundle as bundle` (`bundle.read_bytes(name)`,
`bundle.open(name)`, `bundle.exists(name)`, `bundle.listdir(prefix)`), `pkgutil.get_data`, or the module's `__file__`
under `/bundle/`.
Available inside the runtime: CPython 3.14 stdlib (no zlib/ssl/sqlite3/ctypes), numpy 2.3 (scalar build), OpenCV 4.12
(`core`, `imgproc`, `features2d`, `flann`, `calib3d`, `video`, `dnn` with ONNX import). No threads, no filesystem,
no network, no clock you should rely on for scoring.
