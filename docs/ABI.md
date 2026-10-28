# Solution ABI (version 1)

A submission is one WebAssembly file (`wasm32-wasip1`, *reactor*: no `_start`). The harness instantiates it once per
episode, calls `init` with a JSON config, then `step` once per camera frame, and reads a pose back after each step.
State lives inside the module between steps. The Python runtime (`runtime.wasm`) implements this ABI for you; C, C++
and Rust solutions implement it directly (`relpose/solution/examples/native/cpp_minimal`).

## Exports

| Export | Wasm type | C prototype | Notes |
|---|---|---|---|
| `memory` | memory | – | the module's linear memory (default export of wasi-sdk / Rust) |
| `_initialize` | `() → ()` | – | emitted by wasi-libc for reactors; the harness calls it once if present |
| `abi_version` | `() → i32` | `int32_t abi_version(void)` | must return `1` |
| `alloc` | `(i32) → i32` | `void *alloc(uint32_t size)` | `malloc`; ≥ 8-byte aligned; the harness never frees |
| `init` | `(i32, i32) → i32` | `int32_t init(const char *json, uint32_t size)` | once, before any `step` |
| `step` | `(f64, i32, i32, i32, i32) → i32` | `int32_t step(double t, const uint8_t *frame, const double *imu, int32_t imu_count, double *out)` | once per frame |

Call sequence per episode: `instantiate → _initialize → init → alloc × 3 → step × N → drop`.
There is no `reset`: every episode gets a fresh instance (instantiation costs ~0.1 s, Python init ~0.1 s). The host
treats Python and native modules identically; it does not know what runs inside.

## How Python code travels (`relpose bundle`)

The student's directory is zipped and embedded into `runtime.wasm` as a **data segment** placed just above the
runtime's initial memory (the memory minimum grows by the zip's size), and a 32-byte descriptor inside the runtime
(`magic "RELPOSEBUNDLEv1", address, length`) is patched to point at it. The module is therefore self-contained: at
`init` the embedded shim reads the descriptor, opens the zip in its own memory and imports `solution:Solution`
from it. The bundler is pure Python section surgery; no compiler is involved, and the zip can be as large as the
memory cap allows. `relpose unbundle` extracts it again.

## Frame format

Frames are always **RGB**, `uint8`, `height × width × 3`, and the only choice a solution makes is the resolution:
**320 × 240** (default) or **640 × 480**. Nothing else about the format varies between episodes or sets, and the
config does not say where an episode comes from.

- **Resolution**: the solution selects the resolution it consumes (see the manifest below), e.g. the lower one to save
  compute: more pixels cost fuel in everything a solution runs on them. Any up- or downscaling from the resolution a
  set is stored at is left to the harness, which rescales the intrinsics to match (pixel-centre convention).
- **Camera**: rectified pinhole without distortion.

**Requesting the resolution: the manifest.** An optional JSON object in the module's custom section
`relpose.manifest`, read by the harness *before* `init`: `{"resolution": [640, 480]}`; without it frames are
320 × 240. Unknown keys and other resolutions are rejected. Python solutions put it in `solution.json` next to
`solution.py` and `relpose bundle` copies it in; native modules use `relpose manifest solution.wasm --resolution 640x480`.

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
  "width": 320, "height": 240,
  "fx": 190.7, "fy": 190.7, "cx": 159.5, "cy": 119.5,
  "T_cam_imu": [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
}
```

`width`/`height` are the delivered frame size (what the manifest asked for), `fx, fy, cx, cy` the pinhole intrinsics
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
    def __init__(self, config: dict): ...
    def step(self, t: float, frame: np.ndarray, imu: np.ndarray) -> Sequence[float] | None:
        """frame: uint8 (H, W, 3) RGB, read-only view valid until the next step;
        imu: float64 (N, 7) rows [t, ax, ay, az, gx, gy, gz]. Return [x, y, z, qx, qy, qz, qw] or None."""
```

The entry point is always `Solution` in `solution.py` (or `solution/__init__.py`) at the root of the bundled
directory; a class that lives elsewhere is re-exported there (`from mypkg.estimator import Estimator as Solution`).
Data files in the directory are reachable through `import solution_bundle as bundle` (`bundle.read_bytes(name)`,
`bundle.open(name)`, `bundle.exists(name)`, `bundle.listdir(prefix)`), `pkgutil.get_data`, or the module's `__file__`
under `/bundle/`.
Available inside the runtime: CPython 3.14 stdlib (no zlib/ssl/sqlite3/ctypes), numpy 2.3 (scalar build), OpenCV 4.12
(`core`, `imgproc`, `features2d`, `flann`, `calib3d`, `video`, `dnn` with ONNX import). No threads, no filesystem,
no network, no clock you should rely on for scoring.
