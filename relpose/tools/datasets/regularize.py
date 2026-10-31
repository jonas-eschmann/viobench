"""Puts an episode on the ABI's timeline before it is stored (docs/ABI.md, "step arguments"): a frame every 1/20 s and exactly
IMU_ROWS_PER_FRAME IMU rows in each frame's window (t_{k-1}, t_k], the first frame's window being one interval long.

- A dropped frame (a gap of n frame intervals) is filled with n - 1 repeats of the frame before it.
- A window that already holds IMU_ROWS_PER_FRAME rows at the nominal IMU spacing is kept as recorded (HyperDrone after
  frame 0). Any other window is resampled onto IMU_ROWS_PER_FRAME evenly spaced slots ending at the frame time:
  linear between neighbouring samples (an IMU at another rate), the previous sample across a gap (dropped samples),
  the first sample before the recording starts (HyperDrone's frame 0 window).

Resampling frames from another frame rate is not implemented; such a source is rejected. Idempotent: an episode that is
already regular comes back with the same frames and IMU rows.
"""
import dataclasses

import numpy as np

from .. import abi
from .episode import Episode, FrameSequence

GAP = 1.5  # a spacing above GAP nominal intervals (frames or IMU samples) is a gap


def regularize(episode: Episode) -> Episode:
    interval = 1.0 / abi.FRAME_RATE_HZ
    source, frame_times = fill_frames(episode.frame_times, interval)
    frames = repeat_frames(episode.frames, source)
    imu, resampled = imu_windows(episode.imu, frame_times, interval)
    config = {**episode.config, "episode": {**episode.config["episode"], "num_frames": len(frame_times)}}
    meta = {**episode.meta, "regularized": {"filled_frames": len(frame_times) - episode.num_frames, "resampled_imu_windows": resampled}}
    return dataclasses.replace(episode, config=config, frame_times=frame_times, frames=frames, imu=imu, meta=meta)


def fill_frames(frame_times, interval):
    """(source index, time) per output frame: a gap of n intervals after frame i adds n - 1 repeats of frame i."""
    spacing = np.diff(frame_times) / interval
    if len(spacing) and abs(np.median(spacing) - 1) > 0.1:
        raise ValueError(f"frames arrive at {1 / (np.median(spacing) * interval):.1f} Hz; the ABI delivers {abi.FRAME_RATE_HZ:g} Hz")
    steps = np.rint(spacing).astype(int)
    if np.any(steps < 1):
        raise ValueError("frames closer than half a frame interval")
    counts = np.append(steps, 1)
    source = np.repeat(np.arange(len(frame_times)), counts)
    repeat = np.arange(len(source)) - np.repeat(np.cumsum(counts) - counts, counts)
    return source, frame_times[source] + repeat * interval


def repeat_frames(frames, source):
    if np.array_equal(source, np.arange(len(frames))):
        return frames
    if isinstance(frames, FrameSequence):
        encoded = None if frames.encoded is None else (lambda index: frames.encoded(int(source[index])))
        return FrameSequence(lambda index: frames[int(source[index])], len(source), frames.frame_shape, frames.dtype, encoded)
    return np.asarray(frames)[source]


def imu_windows(imu, frame_times, interval):
    """(rows, number of resampled windows): IMU_ROWS_PER_FRAME rows per frame, concatenated in frame order."""
    rows_per_frame = abi.IMU_ROWS_PER_FRAME
    tolerance = Episode.TIME_TOLERANCE
    previous = np.append(frame_times[0] - interval, frame_times[:-1])
    lower = np.searchsorted(imu[:, 0], previous + tolerance, side="right")
    upper = np.searchsorted(imu[:, 0], frame_times + tolerance, side="right")
    slots = previous[:, None] + (frame_times - previous)[:, None] * np.arange(1, rows_per_frame + 1) / rows_per_frame
    windows = resample(imu, slots.reshape(-1)).reshape(len(frame_times), rows_per_frame, abi.IMU_ROW_VALUES)
    resampled = 0
    for index, (start, end) in enumerate(zip(lower, upper)):
        recorded = imu[start:end]
        if len(recorded) == rows_per_frame and np.all(np.abs(np.diff(recorded[:, 0]) * abi.IMU_RATE_HZ - 1) < 0.5):
            windows[index] = recorded
        else:
            resampled += 1
    return windows.reshape(-1, abi.IMU_ROW_VALUES), resampled


def resample(imu, slots):
    """IMU rows at the slot times: a sample at the slot as is, linear between neighbours, the previous sample across a gap,
    the first (last) sample before (after) the recording."""
    t = imu[:, 0]
    nominal = np.median(np.diff(t)) if len(t) > 1 else np.inf
    after = np.searchsorted(t, slots + Episode.TIME_TOLERANCE, side="right")
    left = np.clip(after - 1, 0, len(t) - 1)
    right = np.clip(after, 0, len(t) - 1)
    span = t[right] - t[left]
    hold = (span <= 0) | (span > GAP * nominal) | (np.abs(slots - t[left]) <= Episode.TIME_TOLERANCE)
    weight = np.where(hold, 0.0, (slots - t[left]) / np.where(hold, 1.0, span))
    values = imu[left, 1:] + weight[:, None] * (imu[right, 1:] - imu[left, 1:])
    return np.column_stack([slots, values])
