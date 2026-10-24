"""In-memory episode: frames, IMU, ground truth and the config a solution receives."""
from dataclasses import dataclass, field

import copy

import cv2
import numpy as np

from .. import abi
from ..metrics import invert, matrices_to_poses, poses_to_matrices, quaternion_to_matrix


def slerp(q0, q1, fraction):
    """Vectorised quaternion slerp, [x, y, z, w]; q0, q1 (N, 4), fraction (N,)."""
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-8
    w0 = np.where(small, 1.0 - fraction, np.sin((1.0 - fraction) * theta) / np.where(small, 1.0, sin_theta))
    w1 = np.where(small, fraction, np.sin(fraction * theta) / np.where(small, 1.0, sin_theta))
    out = w0[:, None] * q0 + w1[:, None] * q1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def interpolate_poses(times, poses, query_times):
    """Linear position + slerp orientation of (K, 7) poses sampled at times (K,), evaluated at query_times (N,)."""
    times = np.asarray(times, dtype=float)
    poses = np.asarray(poses, dtype=float)
    query_times = np.asarray(query_times, dtype=float)
    if np.any(query_times < times[0] - 1e-9) or np.any(query_times > times[-1] + 1e-9):
        raise ValueError(f"query times [{query_times.min()}, {query_times.max()}] outside ground truth [{times[0]}, {times[-1]}]")
    upper = np.clip(np.searchsorted(times, query_times, side="right"), 1, len(times) - 1)
    lower = upper - 1
    span = times[upper] - times[lower]
    fraction = np.where(span > 0, (query_times - times[lower]) / np.where(span > 0, span, 1.0), 0.0)
    fraction = np.clip(fraction, 0.0, 1.0)
    position = poses[lower, :3] + fraction[:, None] * (poses[upper, :3] - poses[lower, :3])
    orientation = slerp(poses[lower, 3:7], poses[upper, 3:7], fraction)
    return np.concatenate([position, orientation], axis=1)


@dataclass
class Episode:
    """frames (N, H, W[, C]) uint8 at frame_times (N,) seconds; imu (M, 7) [t, ax, ay, az, gx, gy, gz];
    ground_truth (K, 8) [t, x, y, z, qx, qy, qz, qw] = pose of the body/IMU frame in the world frame, any rate;
    T_body_camera (4, 4) maps camera coordinates into the body frame. config is what the solution's init() gets."""

    name: str
    config: dict
    frame_times: np.ndarray
    frames: np.ndarray
    imu: np.ndarray
    ground_truth: np.ndarray
    T_body_camera: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.frame_times = np.asarray(self.frame_times, dtype=float)
        self.imu = np.asarray(self.imu, dtype=float).reshape(-1, abi.IMU_ROW_VALUES)
        self.ground_truth = np.asarray(self.ground_truth, dtype=float).reshape(-1, 8)
        self.T_body_camera = np.asarray(self.T_body_camera, dtype=float).reshape(4, 4)
        assert len(self.frames) == len(self.frame_times), "one timestamp per frame"
        assert np.all(np.diff(self.frame_times) > 0), "frame times must increase"
        assert np.all(np.diff(self.imu[:, 0]) > 0), "IMU times must increase"

    @property
    def num_frames(self):
        return len(self.frame_times)

    @property
    def duration(self):
        return float(self.frame_times[-1] - self.frame_times[0])

    TIME_TOLERANCE = 1e-6  # seconds; timestamps are ns integers on disk, so this only absorbs float round-off

    def imu_window(self, index):
        """IMU rows with t_{k-1} < t <= t_k (for k = 0: everything up to t_0)."""
        upper = np.searchsorted(self.imu[:, 0], self.frame_times[index] + self.TIME_TOLERANCE, side="right")
        lower = 0 if index == 0 else np.searchsorted(self.imu[:, 0], self.frame_times[index - 1] + self.TIME_TOLERANCE, side="right")
        return self.imu[lower:upper]

    def steps(self, frames=None):
        """(t, frame, imu_window) per frame, the way the harness feeds a solution (frames: optional override)."""
        frames = self.frames if frames is None else frames
        for index in range(self.num_frames):
            yield self.frame_times[index], frames[index], self.imu_window(index)

    @property
    def channels(self):
        return 1 if self.frames.ndim == 3 else int(self.frames.shape[3])

    @property
    def resolution(self):
        return int(self.frames.shape[2]), int(self.frames.shape[1])

    def as_channels(self, channels):
        return self.adapt(channels=channels)

    def adapt(self, channels=None, resolution=None):
        """(config, frames) for a solution that wants 1 or 3 channels and a given (width, height).

        RGB -> gray is the ITU-R 601 luma (cv2.COLOR_RGB2GRAY), gray -> RGB replicates (config["image"]["color"] says
        whether colour is real). Downscaling uses area averaging, upscaling bilinear interpolation
        (config["image"]["upscaled"] says so); the intrinsics are rescaled with the pixel-centre convention."""
        channels = self.channels if channels is None else int(channels)
        resolution = self.resolution if resolution is None else (int(resolution[0]), int(resolution[1]))
        if channels not in abi.SUPPORTED_CHANNELS:
            raise ValueError(f"channels must be one of {abi.SUPPORTED_CHANNELS}")
        source_channels, source_resolution = self.channels, self.resolution
        frames = self.frames
        if channels != source_channels:
            frames = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]) if channels == 1 else np.repeat(frames[..., None], 3, axis=3)
        config = copy.deepcopy(self.config)
        camera = config["camera"]
        if resolution != source_resolution:
            width, height = resolution
            scale_x, scale_y = width / source_resolution[0], height / source_resolution[1]
            if abs(scale_x - scale_y) > 1e-9:
                raise ValueError(f"aspect ratio change {source_resolution} -> {resolution} is not supported")
            interpolation = cv2.INTER_AREA if scale_x < 1 else cv2.INTER_LINEAR
            frames = np.stack([cv2.resize(frame, (width, height), interpolation=interpolation) for frame in frames])
            camera["fx"], camera["fy"] = camera["fx"] * scale_x, camera["fy"] * scale_y
            camera["cx"], camera["cy"] = (camera["cx"] + 0.5) * scale_x - 0.5, (camera["cy"] + 0.5) * scale_y - 0.5
        config["image"].update({"width": resolution[0], "height": resolution[1], "channels": channels, "layout": "HW" if channels == 1 else "HWC",
                                "color": source_channels == 3 and channels == 3, "source_channels": source_channels,
                                "source_resolution": list(source_resolution), "upscaled": resolution[0] > source_resolution[0]})
        return config, frames

    def camera_poses_world(self, times=None):
        """(N, 7) pose of the camera in the world frame at the given times (default: frame times)."""
        times = self.frame_times if times is None else np.asarray(times, dtype=float)
        body = interpolate_poses(self.ground_truth[:, 0], self.ground_truth[:, 1:8], times)
        return matrices_to_poses(poses_to_matrices(body) @ self.T_body_camera)

    def ground_truth_relative(self):
        """(N, 7) pose of camera k in the frame of camera 0 (T_c0_ck): what a perfect solution returns."""
        T_wc = poses_to_matrices(self.camera_poses_world())
        return matrices_to_poses(invert(T_wc[0]) @ T_wc)

    def rotation_body_camera(self):
        return self.T_body_camera[:3, :3]

    @staticmethod
    def config_extrinsics(T_body_camera):
        T_body_camera = np.asarray(T_body_camera, dtype=float)
        return {"T_cam_imu": invert(T_body_camera).reshape(-1).tolist(), "T_imu_cam": T_body_camera.reshape(-1).tolist()}

    def check_consistency(self):
        image = self.config["image"]
        expected = (image["height"], image["width"]) + (() if image.get("channels", 1) == 1 else (image["channels"],))
        assert self.frames.shape[1:] == expected, f"frames {self.frames.shape[1:]} vs config {expected}"
        assert self.frames.dtype == np.uint8
        np.testing.assert_allclose(np.asarray(self.config["imu"]["T_imu_cam"]).reshape(4, 4), self.T_body_camera, atol=1e-9)
        assert self.ground_truth[0, 0] <= self.frame_times[0] and self.ground_truth[-1, 0] >= self.frame_times[-1], "ground truth must cover all frames"
        np.testing.assert_allclose(np.linalg.norm(self.ground_truth[:, 4:8], axis=1), 1.0, atol=1e-4)
        return True
