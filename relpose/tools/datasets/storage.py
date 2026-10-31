"""Episode storage: one uncompressed zip per episode holding the EuRoC/ASL layout, read in place — the file-system Dataset provider.

    <name>.zip
      <name>/episode.json                               our config + metadata (format, name, frame count)
      <name>/mav0/cam0/data.csv                         "#timestamp [ns],filename"
      <name>/mav0/cam0/data/<timestamp ns>.png          8-bit RGB or grayscale PNG
      <name>/mav0/cam0/sensor.yaml                      Kalibr-style intrinsics and T_BS (camera in body)
      <name>/mav0/imu0/data.csv                         "#timestamp [ns],w_RS_S_x [rad s^-1],...,a_RS_S_x [m s^-2],..." (gyro first, like EuRoC)
      <name>/mav0/imu0/sensor.yaml                      noise densities, rate
      <name>/mav0/state_groundtruth_estimate0/data.csv  "#timestamp, p_RS_R_x [m], ..., q_RS_w [], q_RS_x [], ..." body in world
      <name>/trajectory.mp4                             attachments, if any (`relpose simulate` adds this video)

Members are stored, not deflated (PNGs are compressed already), in a fixed order with fixed timestamps, so the same
episode always gives the same bytes and the archive is content-addressable. `unzip` restores the directory layout that
evo / ORB-SLAM / OpenVINS tooling reads; `read_episode` accepts such a directory as well as the archive.
Timestamps are integer nanoseconds in the files and float seconds (relative to the first IMU sample) in memory. Frames
are decoded on demand by default (`FrameSequence`), so episodes of any length cost little memory.
"""
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

from .episode import Episode, FrameSequence

# cv2 and yaml (relpose[data]) are imported where they are used: the CLI loads this module, and has to start without them

FORMAT = 1
PNG_COMPRESSION = 6  # encoder level for frames from arrays (also the simulator's spool): 9 % smaller than level 1 at twice the time; level 9 gains 2 % more at six times
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)  # member timestamps are fixed: identical content gives identical archives
CAM_HEADER = "#timestamp [ns],filename"
IMU_HEADER = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
GT_HEADER = "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []"
READ_ONLY_META = ("dataset", "path", "origin_ns")  # set by read_episode, not stored in the manifest's meta


def _nanoseconds(seconds, origin_ns=0):
    return np.rint(np.asarray(seconds, dtype=float) * 1e9).astype(np.int64) + np.int64(origin_ns)


def encode_png(frame):
    """PNG bytes of one frame as the format stores it: (H, W) uint8 gray or (H, W, 3) RGB."""
    import cv2

    frame = np.ascontiguousarray(frame)
    image = frame if frame.ndim == 2 or frame.shape[2] == 1 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise OSError("could not encode a frame as PNG")
    return data.tobytes()


def decode_png(data, channels):
    """One frame from PNG bytes: (H, W) uint8 for 1 channel, (H, W, 3) RGB for 3."""
    import cv2

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
    if image is None:
        raise OSError("could not decode a PNG frame")
    return image if channels == 1 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_png(path, channels):
    return decode_png(Path(path).read_bytes(), channels)


def write_episode(episode, path, origin_ns=None, attachments=None):
    """Writes the episode as the archive `path` (a .zip, replaced atomically when complete).

    origin_ns is added to all timestamps (default: the origin the episode was read with, 0 for simulated data).
    attachments: {name: bytes or file path} stored next to episode.json, e.g. {"trajectory.mp4": video_path}.
    Frames that are already PNGs (`FrameSequence.encoded`) are copied as they are, others are encoded."""
    import yaml

    path = Path(path)
    if path.suffix != ".zip":
        raise ValueError(f"an episode is stored as one .zip archive, got {path}")
    if origin_ns is None:
        origin_ns = int(episode.meta.get("origin_ns", 0))
    root = path.stem + "/"
    frame_ns = _nanoseconds(episode.frame_times, origin_ns)
    encoded = getattr(episode.frames, "encoded", None)

    camera_lines = [CAM_HEADER] + [f"{stamp},{stamp}.png" for stamp in frame_ns]
    imu_ns = _nanoseconds(episode.imu[:, 0], origin_ns)
    imu_lines = [IMU_HEADER] + [f"{stamp},{row[4]:.9g},{row[5]:.9g},{row[6]:.9g},{row[1]:.9g},{row[2]:.9g},{row[3]:.9g}" for stamp, row in zip(imu_ns, episode.imu)]
    gt_ns = _nanoseconds(episode.ground_truth[:, 0], origin_ns)
    gt_lines = [GT_HEADER]
    for stamp, row in zip(gt_ns, episode.ground_truth):
        x, y, z, qx, qy, qz, qw = row[1:8]
        gt_lines.append(f"{stamp},{x:.9g},{y:.9g},{z:.9g},{qw:.12g},{qx:.12g},{qy:.12g},{qz:.12g}")

    camera = episode.config["camera"]
    image = episode.config["image"]
    cam_yaml = {
        "sensor_type": "camera",
        "comment": "camera in body/IMU frame; intrinsics as in episode.json",
        "T_BS": {"cols": 4, "rows": 4, "data": [float(v) for v in episode.T_body_camera.reshape(-1)]},
        "rate_hz": float(episode.config["episode"]["frame_rate_hz"]),
        "resolution": [int(image["width"]), int(image["height"])],
        "camera_model": camera["model"],
        "intrinsics": [float(camera["fx"]), float(camera["fy"]), float(camera["cx"]), float(camera["cy"])],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [float(v) for v in camera["distortion"]] or [0.0, 0.0, 0.0, 0.0],
    }
    imu_block = episode.config["imu"]
    rate = float(imu_block["rate_hz"])
    imu_yaml = {
        "sensor_type": "imu",
        "comment": "body/IMU frame; per-sample noise std in episode.json, continuous-time densities here (std / sqrt(rate))",
        "T_BS": {"cols": 4, "rows": 4, "data": [float(v) for v in np.eye(4).reshape(-1)]},
        "rate_hz": rate,
        "gyroscope_noise_density": float(imu_block["gyro_noise_std"] / np.sqrt(rate)),
        "gyroscope_random_walk": float(imu_block["gyro_bias_random_walk"] * np.sqrt(rate)),
        "accelerometer_noise_density": float(imu_block["accel_noise_std"] / np.sqrt(rate)),
        "accelerometer_random_walk": float(imu_block["accel_bias_random_walk"] * np.sqrt(rate)),
    }
    manifest = {
        "format": FORMAT,
        "name": episode.name,
        "frames": int(episode.num_frames),
        "duration_s": episode.duration,
        "origin_ns": int(origin_ns),
        "config": episode.config,
        "meta": {key: value for key, value in episode.meta.items() if key not in READ_ONLY_META},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial-{os.getpid()}")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            def add(name, data):
                info = zipfile.ZipInfo(root + name, date_time=ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3  # Unix, whatever the writer runs on
                info.external_attr = 0o644 << 16
                archive.writestr(info, data)

            add("episode.json", json.dumps(manifest, indent=1, sort_keys=True, default=_json_default))
            add("mav0/cam0/sensor.yaml", yaml.safe_dump(cam_yaml, sort_keys=False))
            add("mav0/cam0/data.csv", "\n".join(camera_lines) + "\n")
            for index, stamp in enumerate(frame_ns):
                add(f"mav0/cam0/data/{stamp}.png", encoded(index) if encoded is not None else encode_png(episode.frames[index]))
            add("mav0/imu0/sensor.yaml", yaml.safe_dump(imu_yaml, sort_keys=False))
            add("mav0/imu0/data.csv", "\n".join(imu_lines) + "\n")
            add("mav0/state_groundtruth_estimate0/data.csv", "\n".join(gt_lines) + "\n")
            for name in sorted(attachments or {}):
                value = attachments[name]
                add(name, bytes(value) if isinstance(value, (bytes, bytearray, memoryview)) else Path(value).read_bytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"cannot serialise {type(value)}")


class _Directory:
    """An unzipped episode."""

    def __init__(self, root):
        self.root = Path(root)

    def read_bytes(self, name):
        return (self.root / name).read_bytes()

    def has(self, name):
        return (self.root / name).is_file()

    def close(self):
        pass


class _Archive:
    """An episode archive: episode.json at the root or in one top-level directory (our writer's layout)."""

    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)  # stays open while a FrameSequence reads from it; member reads are thread-safe
        self.members = set(self.archive.namelist())
        manifests = [name for name in self.members if name == "episode.json" or (name.endswith("/episode.json") and name.count("/") == 1)]
        if len(manifests) != 1:
            self.archive.close()
            raise ValueError(f"{path}: expected one episode.json at the root or one directory down, found {len(manifests)}")
        self.prefix = manifests[0][:-len("episode.json")]

    def read_bytes(self, name):
        return self.archive.read(self.prefix + name)

    def has(self, name):
        return self.prefix + name in self.members

    def close(self):
        self.archive.close()


def _open(path):
    path = Path(path)
    return _Archive(path) if path.is_file() else _Directory(path)


def _read_csv(data):
    rows = np.genfromtxt(io.BytesIO(data), delimiter=",", comments="#", dtype=np.float64, ndmin=2)
    stamps = np.genfromtxt(io.BytesIO(data), delimiter=",", comments="#", usecols=0, dtype=np.int64, ndmin=1)
    return stamps, rows


def read_episode(path, frames="lazy", dataset=None):
    """path: an episode archive or an unzipped episode directory.
    frames: "lazy" (decode PNGs on demand, default), "memory" (load everything), "none" (metadata only)."""
    import yaml

    path = Path(path)
    tree = _open(path)
    manifest = json.loads(tree.read_bytes("episode.json").decode("utf-8"))
    if manifest.get("format") != FORMAT:
        tree.close()
        raise ValueError(f"{path}: unsupported episode format {manifest.get('format')}")
    config = manifest["config"]
    origin_ns = int(manifest["origin_ns"])
    names = np.genfromtxt(io.BytesIO(tree.read_bytes("mav0/cam0/data.csv")), delimiter=",", comments="#", dtype=str, ndmin=2, encoding="utf-8")
    frame_ns = names[:, 0].astype(np.int64)
    frame_times = (frame_ns - origin_ns) / 1e9
    channels = int(config["image"]["channels"])
    members = [f"mav0/cam0/data/{name}" for name in names[:, 1]]
    shape = (int(config["image"]["height"]), int(config["image"]["width"])) + (() if channels == 1 else (channels,))

    imu_ns, imu_rows = _read_csv(tree.read_bytes("mav0/imu0/data.csv"))
    imu = np.column_stack([(imu_ns - origin_ns) / 1e9, imu_rows[:, 4:7], imu_rows[:, 1:4]])  # file: gyro then accel
    gt_ns, gt_rows = _read_csv(tree.read_bytes("mav0/state_groundtruth_estimate0/data.csv"))
    ground_truth = np.column_stack([(gt_ns - origin_ns) / 1e9, gt_rows[:, 1:4], gt_rows[:, 5:8], gt_rows[:, 4]])  # file: qw first
    cam_yaml = yaml.safe_load(tree.read_bytes("mav0/cam0/sensor.yaml").decode("utf-8"))
    T_body_camera = np.asarray(cam_yaml["T_BS"]["data"], dtype=float).reshape(4, 4)

    if frames == "none":
        tree.close()
        frame_data = FrameSequence(lambda index: (_ for _ in ()).throw(RuntimeError("frames were not loaded")), len(members), shape)
    else:
        frame_data = FrameSequence(lambda index: decode_png(tree.read_bytes(members[index]), channels), len(members), shape,
                                   encoded=lambda index: tree.read_bytes(members[index]))
        if frames == "memory":
            frame_data = frame_data.materialize()
            tree.close()
    meta = dict(manifest["meta"])
    meta["dataset"] = dataset or path.parent.name
    meta["path"] = str(path)
    meta["origin_ns"] = origin_ns
    return Episode(manifest["name"], config, frame_times, frame_data, imu, ground_truth, T_body_camera, meta)


def episode_file(path, name):
    """Bytes of a file stored with the episode (an attachment such as trajectory.mp4), or None."""
    tree = _open(path)
    try:
        return tree.read_bytes(name) if tree.has(name) else None
    finally:
        tree.close()


def list_episodes(root):
    """Episodes under root (an episode archive or directory itself, or a directory of them), sorted by name."""
    root = Path(root)
    if root.is_file() or (root / "episode.json").is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted((path for path in root.iterdir() if (path.suffix == ".zip" and path.is_file()) or (path / "episode.json").is_file()),
                  key=lambda path: path.name)


def episodes(roots, frames="lazy"):
    """The file-system Dataset provider: yields the episodes under each root (dataset name = the root directory's name)."""
    for root in ([roots] if isinstance(roots, (str, Path)) else roots):
        root = Path(root)
        paths = list_episodes(root)
        if not paths:
            raise FileNotFoundError(f"no episodes under {root}")
        name = root.parent.name if paths == [root] else root.name
        for path in paths:
            yield read_episode(path, frames=frames, dataset=name)
