"""Configs: guest_config() is the JSON a solution's init() receives (docs/ABI.md): the frame format plus what any calibrated
camera-IMU dataset provides. make_config() is the full description stored with an episode (episode.json: rates, IMU
noise model, hover, name, ...), which the harness keeps to itself."""
import math

import numpy as np

# Camera mounted looking along the body/IMU +x axis (forward), OpenCV camera axes: x right, y down, z forward.
# Columns are the camera axes expressed in the body frame: cam x = -body y, cam y = -body z, cam z = +body x.
R_BODY_CAMERA_FORWARD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# x500_fpv_imu preset of HyperDrone (per-sample white noise at 200 Hz, biases and random walks)
DEFAULT_IMU = {
    "rate_hz": 200.0,
    "accel_noise_std": 0.28,  # m/s^2 per sample
    "gyro_noise_std": 0.07,  # rad/s per sample
    "accel_bias_init_max": 0.2,
    "gyro_bias_init_max": 0.02,
    "accel_bias_random_walk": 0.001,
    "gyro_bias_random_walk": 0.0006,
    "gravity": [0.0, 0.0, -9.81],
}


def pinhole_intrinsics(width, height, horizontal_fov_degrees):
    """Square pixels, principal point at the image centre; same convention as HyperDrone's calibration.h."""
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0))
    return {"fx": focal, "fy": focal, "cx": width / 2.0 - 0.5, "cy": height / 2.0 - 0.5}


def transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def guest_config(width, height, fx, fy, cx, cy, T_cam_imu):
    """What init() receives: the delivered frame size, the intrinsics for it and T_cam_imu (4x4, IMU -> camera coordinates)."""
    return {"width": int(width), "height": int(height),
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "T_cam_imu": np.asarray(T_cam_imu, dtype=float).reshape(4, 4).tolist()}


def default_guest_config(width=320, height=240, horizontal_fov_degrees=80.0):
    """A guest config for synthetic input (viobench check, tests): pinhole intrinsics, camera looking forward."""
    camera = pinhole_intrinsics(width, height, horizontal_fov_degrees)
    return guest_config(width, height, camera["fx"], camera["fy"], camera["cx"], camera["cy"],
                        np.linalg.inv(transform(R_BODY_CAMERA_FORWARD, [0.0, 0.0, 0.0])))


def make_config(width=640, height=480, channels=1, horizontal_fov_degrees=80.0, frame_rate_hz=20.0,
                num_frames=None, hover_seconds=1.0, T_body_camera=None, imu=None, intrinsics=None, episode_name=None):
    """The stored description of an episode. T_body_camera maps camera coordinates into the body (= IMU) frame."""
    if T_body_camera is None:
        T_body_camera = transform(R_BODY_CAMERA_FORWARD, [0.0, 0.0, 0.0])
    T_body_camera = np.asarray(T_body_camera, dtype=float)
    T_camera_body = np.linalg.inv(T_body_camera)
    camera = dict(intrinsics or pinhole_intrinsics(width, height, horizontal_fov_degrees))
    camera.update({"model": "pinhole", "distortion": []})
    imu_block = dict(DEFAULT_IMU)
    if imu:
        imu_block.update(imu)
    imu_block["T_cam_imu"] = T_camera_body.reshape(-1).tolist()  # row-major 4x4: maps IMU/body coordinates into the camera frame
    imu_block["T_imu_cam"] = T_body_camera.reshape(-1).tolist()
    return {
        "image": {"width": int(width), "height": int(height), "channels": int(channels), "dtype": "uint8",
                  "layout": "HW" if channels == 1 else "HWC"},
        "camera": camera,
        "imu": imu_block,
        "episode": {"name": episode_name, "frame_rate_hz": float(frame_rate_hz), "num_frames": num_frames, "hover_seconds": hover_seconds},
        "pose": {"frame": "camera0", "order": "xyz_qxyzw", "timestamp": "mid_exposure"},
    }
