"""In-memory episode: frames (materialised or streamed), IMU, ground truth, its stored description and what a solution receives."""
from dataclasses import dataclass, field

import numpy as np

from .. import abi
from ..abi.config import guest_config
from ..geometry import (
    interpolate_poses,
    invert,
    matrices_to_poses,
    poses_to_matrices,
    rotation_vectors,
)


STORAGE_RESOLUTION = (640, 480)


def validate_storage_resolution(resolution):
    if tuple(resolution) != STORAGE_RESOLUTION:
        width, height = resolution
        raise ValueError(f"episodes must be stored at 640x480, got {width}x{height}; regenerate the episodes at 640x480")


class FrameSequence:
    """Frames decoded on demand: behaves like a read-only (N, H, W[, C]) uint8 array for len/indexing/iteration/shape.

    A one-minute RGB episode at 640x480 is a gigabyte materialised; the runner only ever needs one frame at a time.
    `map(transform)` returns a sequence that applies transform per frame (used by Episode.adapt); `np.asarray(seq)`
    or `seq.materialize()` loads everything. `encoded` (optional, index -> bytes) gives the PNG behind each frame as
    stored, so that `storage.write_episode` copies it without decoding and re-encoding; a mapped sequence has none.
    """

    def __init__(self, loader, count, frame_shape, dtype=np.uint8, encoded=None):
        self._loader = loader
        self._count = int(count)
        self.frame_shape = tuple(int(s) for s in frame_shape)
        self.dtype = np.dtype(dtype)
        self.encoded = encoded

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
    T_body_camera (4, 4) maps camera coordinates into the body frame. config is the stored description (episode.json);
    adapt() derives the frames and the config a solution receives."""

    name: str
    config: dict
    frame_times: np.ndarray
    frames: object
    imu: np.ndarray
    ground_truth: np.ndarray
    T_body_camera: np.ndarray
    meta: dict = field(default_factory=dict)

    TIME_TOLERANCE = 1e-6  # seconds; timestamps are ns integers on disk, so this only absorbs float round-off
    TWIST_STEP = 0.005  # seconds: half-width of the central difference of the ground truth that gives the twist

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
    def hash(self):
        """The sha1 of the archive the episode was read from: its identity. None for one without an archive."""
        return self.meta.get("hash")

    @property
    def key(self):
        """What logs and output directories call the episode: its hash; the name only for an episode without an archive
        (unzipped or in memory), which the server never evaluates."""
        return self.hash or self.name

    @property
    def channels(self):
        return 1 if len(self.frames.shape) == 3 else int(self.frames.shape[3])

    @property
    def resolution(self):
        return int(self.frames.shape[2]), int(self.frames.shape[1])

    @property
    def frame_interval(self):
        return 1.0 / float(self.config["episode"]["frame_rate_hz"])

    def imu_window(self, index):
        """IMU rows with t_{k-1} < t <= t_k; frame 0 gets one frame interval as well (t_0 - interval < t <= t_0), so no set
        reveals itself by how much IMU it recorded before its first frame."""
        previous = self.frame_times[index - 1] if index else self.frame_times[0] - self.frame_interval
        upper = np.searchsorted(self.imu[:, 0], self.frame_times[index] + self.TIME_TOLERANCE, side="right")
        lower = np.searchsorted(self.imu[:, 0], previous + self.TIME_TOLERANCE, side="right")
        return self.imu[lower:upper]

    def delivered_imu(self):
        """All IMU rows a solution receives over the episode (the concatenated windows)."""
        return np.concatenate([self.imu_window(index) for index in range(self.num_frames)])

    def steps(self, frames=None):
        """(t, frame, imu_window) per frame, the way the harness feeds a solution (frames: optional override)."""
        frames = self.frames if frames is None else frames
        for index in range(self.num_frames):
            yield self.frame_times[index], frames[index], self.imu_window(index)

    def camera_poses_world(self, times=None):
        """(N, 7) pose of the camera in the world frame at the given times (default: the frame times)."""
        times = self.frame_times if times is None else times
        body = interpolate_poses(self.ground_truth[:, 0], self.ground_truth[:, 1:8], times)
        return matrices_to_poses(poses_to_matrices(body) @ self.T_body_camera)

    def ground_truth_relative(self):
        """(N, 7) pose of camera k in the frame of camera 0 (T_c0_ck): the pose a perfect solution returns."""
        T_wc = poses_to_matrices(self.camera_poses_world())
        return matrices_to_poses(invert(T_wc[0]) @ T_wc)

    def ground_truth_twist(self):
        """(N, 6) twist of camera k [vx, vy, vz, wx, wy, wz]: velocity of the camera centre and angular velocity, both in
        camera-k axes (v = R^T dt/dt and [w]x = R^T dR/dt of T_c0_ck), by central differences of the ground truth."""
        before = np.maximum(self.frame_times - self.TWIST_STEP, self.ground_truth[0, 0])
        after = np.minimum(self.frame_times + self.TWIST_STEP, self.ground_truth[-1, 0])
        T_before, T_now, T_after = (poses_to_matrices(self.camera_poses_world(times)) for times in (before, self.frame_times, after))
        interval = (after - before)[:, None]
        velocity = np.einsum("nji,nj->ni", T_now[:, :3, :3], T_after[:, :3, 3] - T_before[:, :3, 3]) / interval
        angular_velocity = rotation_vectors(np.swapaxes(T_before[:, :3, :3], 1, 2) @ T_after[:, :3, :3]) / interval
        return np.concatenate([velocity, angular_velocity], axis=1)

    def ground_truth_state(self):
        """(N, 13) pose and twist of camera k (ground_truth_relative, ground_truth_twist): what a perfect solution returns."""
        return np.concatenate([self.ground_truth_relative(), self.ground_truth_twist()], axis=1)

    def adapt(self, resolution=None):
        """(guest config, frames) for a solution that declared 320x240 or 640x480 (docs/ABI.md, "step arguments").

        Frames are always RGB (H, W, 3): a grayscale source arrives with its gray value in all three channels, and
        nothing says so. Stored frames are 640x480; downscaling averages areas and rescales intrinsics with the pixel-centre convention. Streamed frames are converted as the runner consumes them."""
        resolution = self.resolution if resolution is None else (int(resolution[0]), int(resolution[1]))
        if resolution not in abi.SUPPORTED_RESOLUTIONS:
            raise ValueError(f"resolution must be one of {abi.SUPPORTED_RESOLUTIONS}, got {resolution}")
        source_resolution = self.resolution
        validate_storage_resolution(source_resolution)
        camera = self.config["camera"]
        fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]
        transforms = []
        if resolution != source_resolution:
            width, height = resolution
            scale_x, scale_y = width / source_resolution[0], height / source_resolution[1]
            import cv2  # not at the top: the CLI loads this module, and cv2 stays out of its startup

            transforms.append(lambda f: cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA))
            fx, fy = fx * scale_x, fy * scale_y
            cx, cy = (cx + 0.5) * scale_x - 0.5, (cy + 0.5) * scale_y - 0.5
        if self.channels == 1:  # after the resize: one channel to resample instead of three
            transforms.append(lambda f: np.repeat(f[..., None], abi.CHANNELS, axis=2))

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
        return guest_config(resolution[0], resolution[1], fx, fy, cx, cy, invert(self.T_body_camera)), frames

    def check_consistency(self):
        image = self.config["image"]
        expected = (image["height"], image["width"]) + (() if image["channels"] == 1 else (image["channels"],))
        assert tuple(self.frames.shape[1:]) == expected, f"frames {self.frames.shape[1:]} vs config {expected}"
        assert np.dtype(self.frames.dtype) == np.uint8
        np.testing.assert_allclose(np.asarray(self.config["imu"]["T_imu_cam"]).reshape(4, 4), self.T_body_camera, atol=1e-9)
        assert self.ground_truth[0, 0] <= self.frame_times[0] and self.ground_truth[-1, 0] >= self.frame_times[-1], "ground truth must cover all frames"
        np.testing.assert_allclose(np.linalg.norm(self.ground_truth[:, 4:8], axis=1), 1.0, atol=1e-4)
        assert np.all(np.abs(np.diff(self.frame_times) * abi.FRAME_RATE_HZ - 1) < 0.5), "frames must be one interval apart (datasets.regularize)"
        assert all(len(self.imu_window(index)) == abi.IMU_ROWS_PER_FRAME for index in range(self.num_frames)), \
            f"every frame needs {abi.IMU_ROWS_PER_FRAME} IMU rows (datasets.regularize)"
        return True
