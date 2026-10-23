"""IMU preintegration over one frame interval, shared by train.py and the solution (runs inside the runtime too)."""
import numpy as np


def exp_so3(omega):
    angle = float(np.linalg.norm(omega))
    if angle < 1e-12:
        return np.eye(3)
    k = omega / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def log_so3(R):
    cosine = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3)
    return angle / (2 * np.sin(angle)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def preintegrate(window, T_cam_imu):
    """window: (N, 7) rows [t, ax, ay, az, gx, gy, gz] since the previous frame; returns 9 values in the previous
    camera frame: rotation vector, velocity change and position change of the raw (gravity included) integration."""
    R_ci = np.asarray(T_cam_imu, dtype=float)[:3, :3]
    R = np.eye(3)
    dv = np.zeros(3)
    dp = np.zeros(3)
    previous = None
    for row in window:
        t, accel, gyro = row[0], row[1:4], row[4:7]
        if previous is not None:
            dt = t - previous
            if dt > 0:
                acceleration = R @ accel
                dp += dv * dt + 0.5 * acceleration * dt * dt
                dv += acceleration * dt
                R = R @ exp_so3(gyro * dt)
        previous = t
    # express in the previous camera frame
    R_camera = R_ci @ R @ R_ci.T
    return np.concatenate([log_so3(R_camera), R_ci @ dv, R_ci @ dp]).astype(np.float32)
