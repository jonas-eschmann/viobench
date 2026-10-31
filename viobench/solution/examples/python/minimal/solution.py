"""The smallest valid solution: reports the identity pose and a zero twist (standing still) for every frame.

Use it as the template for your own (`viobench init python <dir>` starts a workspace with it): keep the class name
`Solution`, the constructor signature and `step`.
Bundle and check it with
    viobench bundle <this directory> -o solution.wasm
    viobench check solution.wasm
"""
import numpy as np


class Solution:
    frame_width = 320  # optional (320 is the default); 640 asks for 640x480 frames

    def __init__(self, config):
        # config: frame size, intrinsics, camera-IMU extrinsics (docs/ABI.md)
        self.config = config
        self.frames = 0

    def step(self, t, frame, imu):
        """t: frame timestamp [s]; frame: uint8 (H, W, 3) RGB, read-only view valid until the next step;
        imu: float64 (N, 7) rows [t, ax, ay, az, gx, gy, gz] with t_prev < t <= t_frame.
        Return 13 values for every frame: [x, y, z, qx, qy, qz, qw], the pose of this camera frame in the frame of the
        first one, then [vx, vy, vz, wx, wy, wz], the velocity of the camera centre and the angular velocity, both in
        this camera frame's axes."""
        if self.frames == 0:
            print(f"first frame at t={t:.3f}s shape={frame.shape} dtype={frame.dtype} imu rows={len(imu)}")
        self.frames += 1
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
