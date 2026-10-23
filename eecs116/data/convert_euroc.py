"""Convert a EuRoC MAV sequence (ASL format) into challenge episodes (the sim2real test data).

EuRoC: cam0 752x480 grayscale global shutter at 20 Hz, ADIS16448 IMU at 200 Hz, Vicon/Leica ground truth as
`state_groundtruth_estimate0` (body = IMU pose in the world at 200 Hz). Per snippet of `snippet_seconds`, the frames
are undistorted (radial-tangential), cropped to 4:3 around the principal point and resized to the challenge
resolution; intrinsics are recomputed accordingly; the IMU noise is copied from the sensor description; the episode's
reference frame is the first frame of the snippet, like in simulation.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from ..config import make_config
from .episode import Episode

SOURCE_NAME = "euroc"


def read_sensor_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def read_timestamped_csv(path):
    data = np.genfromtxt(path, delimiter=",", comments="#", dtype=np.float64, ndmin=2)
    stamps = np.genfromtxt(path, delimiter=",", comments="#", usecols=0, dtype=np.int64, ndmin=1)
    return stamps, data


def rectification(sensor, width, height):
    """Undistortion maps + new intrinsics: undistort with the original K, crop a centred 4:3 window, scale to (width, height)."""
    fu, fv, cu, cv_ = sensor["intrinsics"]
    source_width, source_height = sensor["resolution"]
    K = np.array([[fu, 0, cu], [0, fv, cv_], [0, 0, 1]], dtype=np.float64)
    distortion = np.asarray(sensor["distortion_coefficients"], dtype=np.float64)
    if sensor.get("distortion_model", "radial-tangential") not in ("radial-tangential", "radtan", "plumb_bob"):
        raise ValueError(f"unsupported distortion model {sensor.get('distortion_model')}")
    # crop window with the target aspect ratio, as large as fits, centred on the principal point
    aspect = width / height
    crop_height = source_height
    crop_width = int(round(crop_height * aspect))
    if crop_width > source_width:
        crop_width = source_width
        crop_height = int(round(crop_width / aspect))
    x0 = int(round(np.clip(cu - crop_width / 2.0, 0, source_width - crop_width)))
    y0 = int(round(np.clip(cv_ - crop_height / 2.0, 0, source_height - crop_height)))
    scale_x, scale_y = width / crop_width, height / crop_height
    # pixel-centre convention: a pixel index i covers [i, i+1); the centre of the cropped+scaled pixel maps accordingly
    new_K = np.array([[fu * scale_x, 0, (cu - x0 + 0.5) * scale_x - 0.5],
                      [0, fv * scale_y, (cv_ - y0 + 0.5) * scale_y - 0.5],
                      [0, 0, 1]])
    # one map that undistorts into the cropped+scaled image directly
    map_x, map_y = cv2.initUndistortRectifyMap(K, distortion, None, new_K, (width, height), cv2.CV_32FC1)
    return map_x, map_y, new_K


def load_sequence(sequence):
    """Reads the ASL directory (…/mav0 or its parent) into raw arrays."""
    sequence = Path(sequence)
    mav0 = sequence / "mav0" if (sequence / "mav0").is_dir() else sequence
    cam = mav0 / "cam0"
    cam_sensor = read_sensor_yaml(cam / "sensor.yaml")
    imu_sensor = read_sensor_yaml(mav0 / "imu0" / "sensor.yaml")
    names = np.genfromtxt(cam / "data.csv", delimiter=",", comments="#", dtype=str, ndmin=2)
    frame_ns = names[:, 0].astype(np.int64)
    imu_ns, imu_rows = read_timestamped_csv(mav0 / "imu0" / "data.csv")
    gt_ns, gt_rows = read_timestamped_csv(mav0 / "state_groundtruth_estimate0" / "data.csv")
    return {
        "mav0": mav0,
        "name": sequence.name if sequence.name != "mav0" else sequence.parent.name,
        "cam_sensor": cam_sensor,
        "imu_sensor": imu_sensor,
        "frame_ns": frame_ns,
        "frame_files": [cam / "data" / name for name in names[:, 1]],
        "imu_ns": imu_ns,
        "imu": np.column_stack([imu_rows[:, 4:7], imu_rows[:, 1:4]]),  # accel then gyro
        "gt_ns": gt_ns,
        "gt": np.column_stack([gt_rows[:, 1:4], gt_rows[:, 5:8] / np.linalg.norm(gt_rows[:, 4:8], axis=1, keepdims=True),
                               gt_rows[:, 4] / np.linalg.norm(gt_rows[:, 4:8], axis=1)]),  # xyz, qxyz, qw (file rounds to ~1e-5)
    }


def convert(sequence, width=320, height=240, snippet_seconds=10.0, max_snippets=None, start_seconds=0.0, stride_seconds=None,
            hover_seconds_first=1.0, log=print):
    """Yields Episodes cut from the sequence. Only frames inside the ground-truth time span are used."""
    raw = load_sequence(sequence)
    map_x, map_y, new_K = rectification(raw["cam_sensor"], width, height)
    T_body_camera = np.asarray(raw["cam_sensor"]["T_BS"]["data"], dtype=float).reshape(4, 4)
    imu_rate = float(raw["imu_sensor"].get("rate_hz", 200.0))
    imu_noise = {
        "rate_hz": imu_rate,
        "accel_noise_std": float(raw["imu_sensor"]["accelerometer_noise_density"]) * np.sqrt(imu_rate),
        "gyro_noise_std": float(raw["imu_sensor"]["gyroscope_noise_density"]) * np.sqrt(imu_rate),
        "accel_bias_random_walk": float(raw["imu_sensor"]["accelerometer_random_walk"]) / np.sqrt(imu_rate),
        "gyro_bias_random_walk": float(raw["imu_sensor"]["gyroscope_random_walk"]) / np.sqrt(imu_rate),
        "accel_bias_init_max": 0.0,
        "gyro_bias_init_max": 0.0,
    }
    frame_rate = float(raw["cam_sensor"].get("rate_hz", 20.0))
    origin_ns = int(raw["imu_ns"][0])
    frame_t = (raw["frame_ns"] - origin_ns) / 1e9
    imu_t = (raw["imu_ns"] - origin_ns) / 1e9
    gt_t = (raw["gt_ns"] - origin_ns) / 1e9
    valid_from, valid_to = max(gt_t[0], imu_t[0]), min(gt_t[-1], imu_t[-1])
    stride = stride_seconds or snippet_seconds
    first = max(frame_t[0], valid_from) + start_seconds
    count = 0
    snippet_start = first
    while snippet_start + snippet_seconds <= valid_to + 1e-6:
        if max_snippets is not None and count >= max_snippets:
            break
        snippet_end = snippet_start + snippet_seconds
        selected = np.flatnonzero((frame_t >= snippet_start - 1e-6) & (frame_t < snippet_end - 1e-6))
        if len(selected) < 2:
            snippet_start += stride
            continue
        times = frame_t[selected]
        frames = []
        for index in selected:
            image = cv2.imread(str(raw["frame_files"][index]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise OSError(f"could not read {raw['frame_files'][index]}")
            frames.append(cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR))
        imu_mask = (imu_t > times[0] - 1.5) & (imu_t <= times[-1] + 1e-6)  # include a lead-in before the first frame
        imu = np.column_stack([imu_t[imu_mask], raw["imu"][imu_mask]])
        gt_mask = (gt_t >= times[0] - 2.0) & (gt_t <= times[-1] + 0.5)
        ground_truth = np.column_stack([gt_t[gt_mask], raw["gt"][gt_mask]])
        hover = hover_seconds_first if count == 0 and start_seconds == 0.0 else 0.0
        name = f"euroc-{raw['name']}-{count:02d}"
        config = make_config(width=width, height=height, channels=1, frame_rate_hz=frame_rate, num_frames=len(selected), hover_seconds=hover,
                             T_body_camera=T_body_camera, imu=imu_noise, episode_name=name,
                             intrinsics={"fx": float(new_K[0, 0]), "fy": float(new_K[1, 1]), "cx": float(new_K[0, 2]), "cy": float(new_K[1, 2])})
        config["camera"]["horizontal_fov_degrees"] = float(np.degrees(2 * np.arctan(width / (2 * new_K[0, 0]))))
        episode = Episode(name, config, times, np.stack(frames), imu, ground_truth, T_body_camera,
                          {"source": SOURCE_NAME, "sequence": raw["name"], "snippet": count, "start_s": float(snippet_start),
                           "origin_ns": origin_ns, "variant": "real", "hover_seconds": hover})
        episode.check_consistency()
        log(f"{name}: {len(selected)} frames {times[0]:.2f}-{times[-1]:.2f}s, {len(imu)} IMU rows")
        yield episode, origin_ns
        count += 1
        snippet_start += stride
