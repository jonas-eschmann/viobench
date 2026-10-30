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


def hovering_turn_motion(t, settling_scale=1.0):
    """Settle, rotate while hovering, then translate with an analytic IMU trajectory.

    The first five seconds have no translational baseline. Small initial attitude
    oscillations prevent a test from assuming the first IMU sample is stationary.
    """
    roll = settling_scale * 0.015 * np.exp(-3 * t) * np.sin(10 * t)
    pitch = settling_scale * 0.01 * np.exp(-3 * t) * np.sin(8 * t)
    roll_rate = settling_scale * 0.015 * np.exp(-3 * t) * (10 * np.cos(10 * t) - 3 * np.sin(10 * t))
    pitch_rate = settling_scale * 0.01 * np.exp(-3 * t) * (8 * np.cos(8 * t) - 3 * np.sin(8 * t))
    u = np.clip((t - 1) / 2, 0, 1)
    yaw = -0.8 * (10 * u**3 - 15 * u**4 + 6 * u**5)
    yaw_rate = -0.4 * 30 * u**2 * (1 - u)**2
    s = max(t - 5, 0)
    yaw += 0.12 * (1 - np.cos(0.4 * s))
    yaw_rate += 0.048 * np.sin(0.4 * s)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rotation = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @
                np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @
                np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    body_rate = np.array([roll_rate - yaw_rate * sp,
                          pitch_rate * cr + yaw_rate * sr * cp,
                          -pitch_rate * sr + yaw_rate * cr * cp])
    position = np.array([0.3 * (s - np.sin(0.8 * s) / 0.8),
                         0.5 * (1 - np.cos(0.5 * s)), 0.15 * (1 - np.cos(0.6 * s))])
    velocity = np.array([0.3 * (1 - np.cos(0.8 * s)), 0.25 * np.sin(0.5 * s), 0.09 * np.sin(0.6 * s)])
    acceleration = np.array([0.24 * np.sin(0.8 * s), 0.125 * np.cos(0.5 * s), 0.054 * np.cos(0.6 * s)])
    if t <= 5:
        acceleration[:] = 0
    return rotation, position, velocity, acceleration, body_rate


def moving_room_inputs(config, seconds=18, imu_phase=0.0, motion=room_motion, noisy_imu=False, seed=31,
                       bounds=((-4, -4, -3), (5, 4, 3)), texture_shape=(96, 768), intensity_range=(20, 235)):
    """Render fixed textured walls and IMU, including the calibrated camera lever arm.

    Truth is always relative to the first *input* exposure, including the motion before
    the estimator initializes. Pose evaluation therefore cannot hide a late origin or a
    scale error through trajectory alignment.
    """
    import cv2

    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])
    coarse, fine = texture_shape
    textures = [cv2.GaussianBlur(cv2.resize(rng.integers(*intensity_range, (coarse, coarse), dtype=np.uint8),
                                          (fine, fine), interpolation=cv2.INTER_NEAREST), (3, 3), 0.6)
                for _ in range(6)]
    T_imu_cam = np.linalg.inv(config["T_cam_imu"])
    R_imu_cam, lever = T_imu_cam[:3, :3], T_imu_cam[:3, 3]
    yy, xx = np.mgrid[:config["height"], :config["width"]]
    rays = np.stack([(xx - config["cx"]) / config["fx"],
                     (yy - config["cy"]) / config["fy"], np.ones_like(xx)], axis=-1)
    rotation0, position0, *_ = motion(0.05)
    camera0 = rotation0 @ R_imu_cam
    centre0 = position0 + rotation0 @ lever
    gyro_bias = np.array([0.01, -0.015, -0.015])
    accel_bias = np.array([0.03, 0.08, 0.16])
    for index in range(round(seconds * 20)):
        t = (index + 1) / 20.0
        rotation, position, velocity, _, rate = motion(t)
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
            map_x = ((points[..., a] - lower[a]) / (upper[a] - lower[a]) * (fine - 1)).astype(np.float32)
            map_y = ((points[..., b] - lower[b]) / (upper[b] - lower[b]) * (fine - 1)).astype(np.float32)
            for side in range(2):
                visible = (axes == axis) & ((directions[..., axis] > 0) == bool(side))
                texture = cv2.remap(textures[2 * axis + side], map_x, map_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
                gray[visible] = texture[visible]
        times = (index * 10 + np.arange(1, 11)) / 200.0 - imu_phase
        imu = np.zeros((10, 7))
        imu[:, 0] = times
        for j, time in enumerate(times):
            body_rotation, _, _, acceleration, body_rate = motion(time)
            imu[j, 1:4] = body_rotation.T @ (acceleration + [0, 0, 9.81])
            imu[j, 4:7] = body_rate
            if noisy_imu:
                # White noise is per sample; random walks are continuous-time densities.
                accel_bias += rng.normal(0, 0.001 * np.sqrt(0.005), 3)
                gyro_bias += rng.normal(0, 0.0006 * np.sqrt(0.005), 3)
                imu[j, 1:4] += accel_bias + rng.normal(0, 0.28, 3)
                imu[j, 4:7] += gyro_bias + rng.normal(0, 0.07, 3)
        truth = np.r_[camera0.T @ (centre - centre0), matrix_to_quaternion(camera0.T @ camera),
                      camera.T @ camera_velocity, R_imu_cam.T @ rate]
        yield t, np.repeat(gray[..., None], 3, axis=2), imu, truth


def hovering_room_inputs(config, seconds=15, imu_phase=0.0025, settling_scale=1.0):
    """Noisy hovering turn in a larger room whose distant walls give weak parallax."""
    def motion(t):
        return hovering_turn_motion(t, settling_scale)

    yield from moving_room_inputs(config, seconds=seconds, imu_phase=imu_phase,
                                  motion=motion, noisy_imu=True, seed=19,
                                  bounds=((-4, -5, -3), (8, 5, 3)), texture_shape=(64, 512),
                                  intensity_range=(30, 230))

