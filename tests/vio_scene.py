"""Analytic trajectory and textured-room renderer shared by VIO regression tests."""
import numpy as np

from relpose.tools.geometry import matrix_to_quaternion


def room_motion(t):
    """Smooth, independently specified IMU trajectory with observable metric scale."""
    roll, pitch, yaw = 0.05 * np.sin(1.1 * t), 0.06 * np.cos(0.9 * t), 0.2 * np.sin(0.7 * t)
    roll_rate, pitch_rate, yaw_rate = 0.055 * np.cos(1.1 * t), -0.054 * np.sin(0.9 * t), 0.14 * np.cos(0.7 * t)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rotation = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @
                np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @
                np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    body_rate = np.array([roll_rate - yaw_rate * sp,
                          pitch_rate * cr + yaw_rate * sr * cp,
                          -pitch_rate * sr + yaw_rate * cr * cp])
    position = np.array([0.8 * np.sin(1.1 * t), 0.7 * np.sin(0.7 * t), 0.4 * np.cos(0.9 * t)])
    velocity = np.array([0.88 * np.cos(1.1 * t), 0.49 * np.cos(0.7 * t), -0.36 * np.sin(0.9 * t)])
    acceleration = np.array([-0.968 * np.sin(1.1 * t), -0.343 * np.sin(0.7 * t), -0.324 * np.cos(0.9 * t)])
    return rotation, position, velocity, acceleration, body_rate


def moving_room_inputs(config, seconds=18, imu_phase=0.0):
    """Render fixed textured walls and exact IMU, including the calibrated camera lever arm.

    Truth is always relative to the first *input* exposure, including the motion before
    the estimator initializes. Pose evaluation therefore cannot hide a late origin or a
    scale error through trajectory alignment.
    """
    import cv2

    rng = np.random.default_rng(31)
    lower, upper = np.array([-4, -4, -3]), np.array([5, 4, 3])
    textures = [cv2.GaussianBlur(cv2.resize(rng.integers(20, 235, (96, 96), dtype=np.uint8),
                                          (768, 768), interpolation=cv2.INTER_NEAREST), (3, 3), 0.6)
                for _ in range(6)]
    T_imu_cam = np.linalg.inv(config["T_cam_imu"])
    R_imu_cam, lever = T_imu_cam[:3, :3], T_imu_cam[:3, 3]
    yy, xx = np.mgrid[:config["height"], :config["width"]]
    rays = np.stack([(xx - config["cx"]) / config["fx"],
                     (yy - config["cy"]) / config["fy"], np.ones_like(xx)], axis=-1)
    rotation0, position0, *_ = room_motion(0.05)
    camera0 = rotation0 @ R_imu_cam
    centre0 = position0 + rotation0 @ lever
    for index in range(round(seconds * 20)):
        t = (index + 1) / 20.0
        rotation, position, velocity, _, rate = room_motion(t)
        camera = rotation @ R_imu_cam
        centre = position + rotation @ lever
        camera_velocity = velocity + rotation @ np.cross(rate, lever)
        directions = rays @ camera.T
        with np.errstate(divide="ignore"):
            distances = np.where(directions > 0, upper - centre, lower - centre) / directions
        axes = np.argmin(distances, axis=-1)
        points = centre + directions * np.min(distances, axis=-1)[..., None]
        gray = np.zeros(xx.shape, np.uint8)
        for axis in range(3):
            a, b = [other for other in range(3) if other != axis]
            map_x = ((points[..., a] - lower[a]) / (upper[a] - lower[a]) * 767).astype(np.float32)
            map_y = ((points[..., b] - lower[b]) / (upper[b] - lower[b]) * 767).astype(np.float32)
            for side in range(2):
                visible = (axes == axis) & ((directions[..., axis] > 0) == bool(side))
                texture = cv2.remap(textures[2 * axis + side], map_x, map_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
                gray[visible] = texture[visible]
        times = (index * 10 + np.arange(1, 11)) / 200.0 - imu_phase
        imu = np.zeros((10, 7))
        imu[:, 0] = times
        for j, time in enumerate(times):
            body_rotation, _, _, acceleration, body_rate = room_motion(time)
            imu[j, 1:4] = body_rotation.T @ (acceleration + [0, 0, 9.81])
            imu[j, 4:7] = body_rate
        truth = np.r_[camera0.T @ (centre - centre0), matrix_to_quaternion(camera0.T @ camera),
                      camera.T @ camera_velocity, R_imu_cam.T @ rate]
        yield t, np.repeat(gray[..., None], 3, axis=2), imu, truth

