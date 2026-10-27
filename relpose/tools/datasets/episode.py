"""In-memory episode: frames (materialised or streamed), IMU, ground truth and the config a solution receives."""
import copy
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import abi
from ..geometry import interpolate_poses, invert, matrices_to_poses, poses_to_matrices


class FrameSequence:
    """Frames decoded on demand: behaves like a read-only (N, H, W[, C]) uint8 array for len/indexing/iteration/shape.

    A one-minute RGB episode at 640x480 is a gigabyte materialised; the runner only ever needs one frame at a time.
    `map(transform)` returns a sequence that applies transform per frame (used by Episode.adapt); `np.asarray(seq)`
    or `seq.materialize()` loads everything. `paths` (optional) names the PNG file behind each frame, so that
    `storage.write_episode` can copy the files without decoding and re-encoding them; a mapped sequence has none.
    `link_paths` opts immutable simulator spool files into hard linking; persistent dataset files are copied.
    """

    def __init__(self, loader, count, frame_shape, dtype=np.uint8, paths=None, link_paths=False):
        self._loader = loader
        self._count = int(count)
        self.frame_shape = tuple(int(s) for s in frame_shape)
        self.dtype = np.dtype(dtype)
        self.paths = None if paths is None else list(paths)
        self.link_paths = link_paths
        assert self.paths is None or len(self.paths) == self._count, "one path per frame"
        assert not link_paths or self.paths is not None, "linking requires frame paths"

    @property
    def shape(self):
        return (self._count, *self.frame_shape)

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        return self._loader(index)

    def __iter__(self):
        for index in range(self._count):
            yield self._loader(index)

    def __array__(self, dtype=None, copy=None):
        array = self.materialize()
        return array.astype(dtype) if dtype is not None else array

    def materialize(self):
        return np.stack(list(self)) if self._count else np.zeros(self.shape, self.dtype)

    def map(self, transform):
        probe = transform(self._loader(0)) if self._count else None
        shape = probe.shape if probe is not None else self.frame_shape
        return FrameSequence(lambda index: transform(self._loader(index)), self._count, shape, probe.dtype if probe is not None else self.dtype)


@dataclass
class Episode:
    """frames (N, H, W[, C]) uint8 (ndarray or FrameSequence) at frame_times (N,) seconds; imu (M, 7) [t, ax, ay, az, gx, gy, gz];
    ground_truth (K, 8) [t, x, y, z, qx, qy, qz, qw] = pose of the body/IMU frame in the world frame, any rate;
    T_body_camera (4, 4) maps camera coordinates into the body frame. config is what the solution's init() gets."""

    name: str
    config: dict
    frame_times: np.ndarray
    frames: object
    imu: np.ndarray
    ground_truth: np.ndarray
    T_body_camera: np.ndarray
    meta: dict = field(default_factory=dict)

    TIME_TOLERANCE = 1e-6  # seconds; timestamps are ns integers on disk, so this only absorbs float round-off

    def __post_init__(self):
        self.frame_times = np.asarray(self.frame_times, dtype=float)
        self.imu = np.asarray(self.imu, dtype=float).reshape(-1, abi.IMU_ROW_VALUES)
        self.ground_truth = np.asarray(self.ground_truth, dtype=float).reshape(-1, 8)
        self.T_body_camera = np.asarray(self.T_body_camera, dtype=float).reshape(4, 4)
        if not isinstance(self.frames, FrameSequence):
            self.frames = np.asarray(self.frames)
        assert len(self.frames) == len(self.frame_times), "one timestamp per frame"
        assert np.all(np.diff(self.frame_times) > 0), "frame times must increase"
        assert np.all(np.diff(self.imu[:, 0]) > 0), "IMU times must increase"

    @property
    def num_frames(self):
        return len(self.frame_times)

    @property
    def duration(self):
        return float(self.frame_times[-1] - self.frame_times[0])

    @property
    def dataset(self):
        return self.meta.get("dataset", "episodes")

    @property
    def channels(self):
        return 1 if len(self.frames.shape) == 3 else int(self.frames.shape[3])

    @property
    def resolution(self):
        return int(self.frames.shape[2]), int(self.frames.shape[1])

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

    def as_channels(self, channels):
        return self.adapt(channels=channels)

    def adapt(self, channels=None, resolution=None):
        """(config, frames) for a solution that wants 1 or 3 channels and a given (width, height).

        RGB -> gray is the ITU-R 601 luma (cv2.COLOR_RGB2GRAY), gray -> RGB replicates (config["image"]["color"] says
        whether colour is real). Downscaling uses area averaging, upscaling bilinear interpolation
        (config["image"]["upscaled"] says so); the intrinsics are rescaled with the pixel-centre convention.
        Streamed frames stay streamed: the conversion is applied per frame as the runner consumes them."""
        channels = self.channels if channels is None else int(channels)
        resolution = self.resolution if resolution is None else (int(resolution[0]), int(resolution[1]))
        if channels not in abi.SUPPORTED_CHANNELS:
            raise ValueError(f"channels must be one of {abi.SUPPORTED_CHANNELS}")
        source_channels, source_resolution = self.channels, self.resolution
        config = copy.deepcopy(self.config)
        camera = config["camera"]
        transforms = []
        if channels != source_channels:
            transforms.append((lambda f: cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)) if channels == 1 else (lambda f: np.repeat(f[..., None], 3, axis=2)))
        if resolution != source_resolution:
            width, height = resolution
            scale_x, scale_y = width / source_resolution[0], height / source_resolution[1]
            if abs(scale_x - scale_y) > 1e-9:
                raise ValueError(f"aspect ratio change {source_resolution} -> {resolution} is not supported")
            interpolation = cv2.INTER_AREA if scale_x < 1 else cv2.INTER_LINEAR
            transforms.append(lambda f: cv2.resize(f, (width, height), interpolation=interpolation))
            camera["fx"], camera["fy"] = camera["fx"] * scale_x, camera["fy"] * scale_y
            camera["cx"], camera["cy"] = (camera["cx"] + 0.5) * scale_x - 0.5, (camera["cy"] + 0.5) * scale_y - 0.5

        def convert(frame):
            for transform in transforms:
                frame = transform(frame)
            return frame

        if not transforms:
            frames = self.frames
        elif isinstance(self.frames, FrameSequence):
            frames = self.frames.map(convert)
        else:
            frames = np.stack([convert(frame) for frame in self.frames])
        config["image"].update({"width": resolution[0], "height": resolution[1], "channels": channels, "layout": "HW" if channels == 1 else "HWC",
                                "color": source_channels == 3 and channels == 3, "source_channels": source_channels,
                                "source_resolution": list(source_resolution), "upscaled": resolution[0] > source_resolution[0]})
        return config, frames

    def check_consistency(self):
        image = self.config["image"]
        expected = (image["height"], image["width"]) + (() if image.get("channels", 1) == 1 else (image["channels"],))
        assert tuple(self.frames.shape[1:]) == expected, f"frames {self.frames.shape[1:]} vs config {expected}"
        assert np.dtype(self.frames.dtype) == np.uint8
        np.testing.assert_allclose(np.asarray(self.config["imu"]["T_imu_cam"]).reshape(4, 4), self.T_body_camera, atol=1e-9)
        assert self.ground_truth[0, 0] <= self.frame_times[0] and self.ground_truth[-1, 0] >= self.frame_times[-1], "ground truth must cover all frames"
        np.testing.assert_allclose(np.linalg.norm(self.ground_truth[:, 4:8], axis=1), 1.0, atol=1e-4)
        return True
