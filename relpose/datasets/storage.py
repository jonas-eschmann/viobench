"""On-disk episode format: the EuRoC/ASL directory layout plus episode.json.

    <episode>/episode.json                               our config + metadata (format, condition, source, frame count)
    <episode>/mav0/cam0/data.csv                         "#timestamp [ns],filename"
    <episode>/mav0/cam0/data/<timestamp ns>.png          8-bit grayscale (or RGB) PNG
    <episode>/mav0/cam0/sensor.yaml                      Kalibr-style intrinsics and T_BS (camera in body)
    <episode>/mav0/imu0/data.csv                         "#timestamp [ns],w_x,w_y,w_z [rad s^-1],a_x,a_y,a_z [m s^-2]"
    <episode>/mav0/imu0/sensor.yaml                      noise densities, rate
    <episode>/mav0/state_groundtruth_estimate0/data.csv  "#timestamp,p_x,p_y,p_z [m],q_w,q_x,q_y,q_z,..." body in world

Identical for simulated and real episodes, and readable by evo / ORB-SLAM / OpenVINS tooling. Timestamps are
integer nanoseconds in the files and float seconds (relative to the first IMU sample) in memory.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .episode import Episode

FORMAT = 1
CAM_HEADER = "#timestamp [ns],filename"
IMU_HEADER = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
GT_HEADER = "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []"


def _nanoseconds(seconds, origin_ns=0):
    return np.rint(np.asarray(seconds, dtype=float) * 1e9).astype(np.int64) + np.int64(origin_ns)


def write_episode(episode, directory, origin_ns=0, png_compression=1):
    """Writes the episode; origin_ns is added to all timestamps (0 for simulated data)."""
    directory = Path(directory)
    cam = directory / "mav0" / "cam0"
    imu = directory / "mav0" / "imu0"
    gt = directory / "mav0" / "state_groundtruth_estimate0"
    for path in (cam / "data", imu, gt):
        path.mkdir(parents=True, exist_ok=True)

    frame_ns = _nanoseconds(episode.frame_times, origin_ns)
    lines = [CAM_HEADER]
    for stamp, frame in zip(frame_ns, episode.frames):
        name = f"{stamp}.png"
        image = frame if frame.ndim == 2 or frame.shape[2] == 1 else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(cam / "data" / name), image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression]):
            raise OSError(f"could not write {cam / 'data' / name}")
        lines.append(f"{stamp},{name}")
    (cam / "data.csv").write_text("\n".join(lines) + "\n")

    imu_ns = _nanoseconds(episode.imu[:, 0], origin_ns)
    with open(imu / "data.csv", "w") as file:
        file.write(IMU_HEADER + "\n")
        for stamp, row in zip(imu_ns, episode.imu):
            file.write(f"{stamp},{row[4]:.9g},{row[5]:.9g},{row[6]:.9g},{row[1]:.9g},{row[2]:.9g},{row[3]:.9g}\n")

    gt_ns = _nanoseconds(episode.ground_truth[:, 0], origin_ns)
    with open(gt / "data.csv", "w") as file:
        file.write(GT_HEADER + "\n")
        for stamp, row in zip(gt_ns, episode.ground_truth):
            x, y, z, qx, qy, qz, qw = row[1:8]
            file.write(f"{stamp},{x:.9g},{y:.9g},{z:.9g},{qw:.12g},{qx:.12g},{qy:.12g},{qz:.12g}\n")

    camera = episode.config["camera"]
    image = episode.config["image"]
    cam_yaml = {
        "sensor_type": "camera",
        "comment": "camera in body/IMU frame; intrinsics as in episode.json",
        "T_BS": {"cols": 4, "rows": 4, "data": [float(v) for v in episode.T_body_camera.reshape(-1)]},
        "rate_hz": float(episode.config["episode"]["frame_rate_hz"]),
        "resolution": [int(image["width"]), int(image["height"])],
        "camera_model": camera.get("model", "pinhole"),
        "intrinsics": [float(camera["fx"]), float(camera["fy"]), float(camera["cx"]), float(camera["cy"])],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [float(v) for v in camera.get("distortion", [])] or [0.0, 0.0, 0.0, 0.0],
    }
    (cam / "sensor.yaml").write_text(yaml.safe_dump(cam_yaml, sort_keys=False))
    imu_block = episode.config["imu"]
    rate = float(imu_block["rate_hz"])
    imu_yaml = {
        "sensor_type": "imu",
        "comment": "body/IMU frame; per-sample noise std in episode.json, continuous-time densities here (std / sqrt(rate))",
        "T_BS": {"cols": 4, "rows": 4, "data": [float(v) for v in np.eye(4).reshape(-1)]},
        "rate_hz": rate,
        "gyroscope_noise_density": float(imu_block.get("gyro_noise_std", 0.0) / np.sqrt(rate)),
        "gyroscope_random_walk": float(imu_block.get("gyro_bias_random_walk", 0.0) * np.sqrt(rate)),
        "accelerometer_noise_density": float(imu_block.get("accel_noise_std", 0.0) / np.sqrt(rate)),
        "accelerometer_random_walk": float(imu_block.get("accel_bias_random_walk", 0.0) * np.sqrt(rate)),
    }
    (imu / "sensor.yaml").write_text(yaml.safe_dump(imu_yaml, sort_keys=False))

    manifest = {
        "format": FORMAT,
        "name": episode.name,
        "frames": int(episode.num_frames),
        "duration_s": episode.duration,
        "origin_ns": int(origin_ns),
        "config": episode.config,
        "meta": episode.meta,
    }
    (directory / "episode.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=_json_default))
    return directory


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"cannot serialise {type(value)}")


def _read_csv(path):
    rows = np.genfromtxt(path, delimiter=",", comments="#", dtype=np.float64, ndmin=2)
    stamps = np.genfromtxt(path, delimiter=",", comments="#", usecols=0, dtype=np.int64, ndmin=1)
    return stamps, rows


def read_episode(directory, load_frames=True):
    directory = Path(directory)
    manifest = json.loads((directory / "episode.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{directory}: unsupported episode format {manifest.get('format')}")
    config = manifest["config"]
    origin_ns = int(manifest.get("origin_ns", 0))
    cam = directory / "mav0" / "cam0"
    names = np.genfromtxt(cam / "data.csv", delimiter=",", comments="#", dtype=str, ndmin=2)
    frame_ns = names[:, 0].astype(np.int64)
    frame_times = (frame_ns - origin_ns) / 1e9
    channels = int(config["image"].get("channels", 1))
    if load_frames:
        frames = []
        for name in names[:, 1]:
            image = cv2.imread(str(cam / "data" / name), cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"could not read {cam / 'data' / name}")
            frames.append(image if channels == 1 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames = np.stack(frames)
    else:
        frames = np.zeros((len(frame_times), 0), dtype=np.uint8)

    imu_ns, imu_rows = _read_csv(directory / "mav0" / "imu0" / "data.csv")
    imu = np.column_stack([(imu_ns - origin_ns) / 1e9, imu_rows[:, 4:7], imu_rows[:, 1:4]])  # file: gyro then accel

    gt_ns, gt_rows = _read_csv(directory / "mav0" / "state_groundtruth_estimate0" / "data.csv")
    ground_truth = np.column_stack([(gt_ns - origin_ns) / 1e9, gt_rows[:, 1:4], gt_rows[:, 5:8], gt_rows[:, 4]])  # file: qw first

    cam_yaml = yaml.safe_load((cam / "sensor.yaml").read_text())
    T_body_camera = np.asarray(cam_yaml["T_BS"]["data"], dtype=float).reshape(4, 4)
    return Episode(manifest.get("name", directory.name), config, frame_times, frames, imu, ground_truth, T_body_camera, manifest.get("meta", {}))


def list_episodes(root):
    """Episode directories under root (an episode directory itself, or a directory of them), sorted."""
    root = Path(root)
    if (root / "episode.json").is_file():
        return [root]
    return sorted(path.parent for path in root.glob("*/episode.json"))
