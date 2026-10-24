"""The smallest valid solution: reports the identity pose for every frame.

Use it as the template for your own: keep the class name `Solution`, the constructor signature and `step`.
Bundle and check it with
    relpose bundle relpose/solution/examples/python/minimal -o minimal.wasm
    relpose check minimal.wasm
"""
import numpy as np


class Solution:
    def __init__(self, config):
        # config: intrinsics, camera-IMU extrinsics, image size, IMU rates/noise (docs/ABI.md)
        self.config = config
        self.frames = 0

    def step(self, t, frame, imu):
        """t: frame timestamp [s]; frame: uint8 (H, W) (or (H, W, C)), read-only view valid until the next step;
        imu: float64 (N, 7) rows [t, ax, ay, az, gx, gy, gz] with t_prev < t <= t_frame.
        Return [x, y, z, qx, qy, qz, qw] of this camera frame in the frame of the first one, or None."""
        if self.frames == 0:
            print(f"first frame at t={t:.3f}s shape={frame.shape} dtype={frame.dtype} imu rows={len(imu)}")
        self.frames += 1
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

if __name__ == "__main__":  # debug outside the sandbox: python solution.py <episode dirs...>  (docs/DEBUGGING.md)
    from relpose.tools.runner.direct import main

    main(Solution)
