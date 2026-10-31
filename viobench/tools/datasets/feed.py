"""`viobench feed`: one episode's inputs as raw little-endian files, so a native debug build can read them with fread.

    <dir>/config.json      the exact bytes to pass to init()
    <dir>/frames.u8        N x H x W x 3 uint8 RGB, already adapted to the requested resolution
    <dir>/frame_times.f64  N doubles (seconds, mid-exposure)
    <dir>/imu.f64          M x 7 doubles [t, ax, ay, az, gx, gy, gz], the rows the harness delivers (from frame 0's window on)
    <dir>/gt.f64           N x 7 doubles, camera k in camera 0 (what a perfect solution returns)
    <dir>/shape.txt        "N H W C M"
"""
import json
from pathlib import Path

import numpy as np

from .. import abi


def write_feed(episode, directory, resolution=abi.DEFAULT_RESOLUTION):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config, frames = episode.adapt(resolution)
    height, width = config["height"], config["width"]
    (directory / "config.json").write_text(json.dumps(config))
    with open(directory / "frames.u8", "wb") as file:
        for frame in frames:  # streamed: one frame in memory at a time
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            assert frame.shape[:2] == (height, width)
            file.write(frame.tobytes())
    np.asarray(episode.frame_times, dtype="<f8").tofile(directory / "frame_times.f64")
    imu = np.ascontiguousarray(episode.delivered_imu(), dtype="<f8")
    imu.tofile(directory / "imu.f64")
    np.ascontiguousarray(episode.ground_truth_relative(), dtype="<f8").tofile(directory / "gt.f64")
    (directory / "shape.txt").write_text(f"{episode.num_frames} {height} {width} {abi.CHANNELS} {len(imu)}\n")
    return directory
