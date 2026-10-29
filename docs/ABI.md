# Solution ABI (version 1)

A submission is one `wasm32-wasip1` reactor module. Each episode gets a fresh instance: `init` once with a JSON config,
then `step` once per frame, which writes the camera's state to `out`. The Python runtime implements the ABI for you
(`docs/RUNTIME.md`); native solutions implement it directly (`relpose/solution/examples/native/cpp_minimal`).

## Exports

| Export | Wasm type | C prototype | Notes |
|---|---|---|---|
| `memory` | memory | – | linear memory |
| `_initialize` | `() → ()` | – | optional, called once |
| `abi_version` | `() → i32` | `int32_t abi_version(void)` | returns `1` |
| `alloc` | `(i32) → i32` | `void *alloc(uint32_t size)` | ≥ 8-byte aligned, never freed |
| `frame_width` | `() → i32` | `int32_t frame_width(void)` | optional: `320` (default) or `640`; anything else fails the episode |
| `init` | `(i32, i32) → i32` | `int32_t init(const char *json, uint32_t size)` | once |
| `step` | `(f64, i32, i32, i32, i32) → i32` | `int32_t step(double t, const uint8_t *frame, const double *imu, int32_t imu_count, double *out)` | once per frame |

Call order: `_initialize → frame_width → init → alloc × 3 → step × N`; a host calls every one of these that the module
exports (the Python runtime starts the interpreter in `frame_width`). Everything before the first `step` shares the
init fuel budget.

## `step` arguments

Little-endian, row-major. `frame` and `imu` are overwritten for the next step and the config bytes are valid only
during `init`: copy what you keep.

- `t`: frame timestamp in seconds, mid-exposure.
- `frame`: `height × width × 3` `uint8`, RGB. `frame_width` selects 320 × 240 or 640 × 480; the smaller one saves
  fuel. Rectified pinhole camera, no distortion.
- `imu`: `imu_count × 7` `f64`, rows `[t, ax, ay, az, gx, gy, gz]` with `t_prev_frame < t ≤ t_frame` (the first frame:
  one frame interval): specific force in m/s² (gravity included) and angular rate in rad/s, in the IMU frame. Frames
  are 20 Hz and the IMU 200 Hz, so `imu_count` is always 10; use the timestamps for the spacing.
- `out`: 13 `f64` `[x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]`, all written:
  - the pose `T_c0_ck` of the current camera frame in the first camera frame (`p_c0 = R p_ck + t`), unit quaternion
    with scalar last;
  - the twist in the camera frame: `v = Rᵀ dt/dt` (m/s) and `w` with `[w]× = Rᵀ dR/dt` (rad/s).

## Status

`init` and `step` return `0` on success; anything else is an error (write the reason to stderr). An error, a value in
`out` that is not finite, an all-zero quaternion or a trap ends the episode as failed; its remaining frames hold the
last pose with zero twist. Until you have an estimate, return identity and zero twist.

Traps: `unreachable`, out-of-bounds access, stack overflow, out of fuel, the wall-clock deadline, `proc_exit`. An
allocation beyond the memory cap fails instead of trapping (Python: `MemoryError`).

## Config

```json
{
  "width": 320, "height": 240,
  "fx": 190.7, "fy": 190.7, "cx": 159.5, "cy": 119.5,
  "T_cam_imu": [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
}
```

The frame size, the intrinsics in pixels at that size (they vary per episode) and the transform from IMU to camera
coordinates (camera axes: x right, y down, z forward).

## Python

```python
class Solution:
    frame_width = 320              # optional: 320 or 640
    def __init__(self, config): ...
    def step(self, t, frame, imu): ...  # frame: (H, W, 3) uint8, read-only; imu: (10, 7) float64; returns the 13 values of out
```

`Solution` is defined or imported in `solution.py` (or `solution/__init__.py`) at the root of the bundled directory.
Raising, or returning anything other than 13 numbers, is an error.
