"""The data pipeline's regularization: dropped frames and IMU samples, other IMU rates, the window before the IMU starts."""
import dataclasses

import numpy as np
import pytest

from viobench.tools import abi
from viobench.tools.datasets.regularize import regularize
from viobench.tools.datasets.storage import read_episode, write_episode
from tests.test_datasets import synthetic_episode


def windows(episode):
    return [imu for _, _, imu in episode.steps()]


def linear_imu(times):
    """IMU rows whose values are linear in time, so interpolation reproduces them exactly."""
    return np.column_stack([times] + [value + 0.5 * times for value in (0.1, 0.2, 9.81, 0.01, 0.02, 0.03)])


def test_regular_episode_passes_through():
    episode = synthetic_episode()
    regular = regularize(episode)
    assert regular.frames is episode.frames
    np.testing.assert_array_equal(regular.frame_times, episode.frame_times)
    np.testing.assert_array_equal(regular.imu, episode.delivered_imu())
    assert regular.meta["regularized"] == {"filled_frames": 0, "resampled_imu_windows": 0}
    regular.check_consistency()


def test_dropped_frame_repeats_the_previous_one(tmp_path):
    episode = synthetic_episode(n_frames=8, width=640, height=480)
    keep = np.r_[0:3, 4:8]
    dropped = dataclasses.replace(episode, frame_times=episode.frame_times[keep], frames=episode.frames[keep])
    with pytest.raises(AssertionError):
        dropped.check_consistency()
    regular = regularize(read_episode(write_episode(dropped, tmp_path / "dropped.zip")))  # streamed frames with stored PNGs
    regular.check_consistency()
    assert regular.num_frames == 8 and regular.config["episode"]["num_frames"] == 8
    np.testing.assert_allclose(regular.frame_times, episode.frame_times, atol=1e-9)
    np.testing.assert_array_equal(regular.frames[3], episode.frames[2])
    np.testing.assert_array_equal(regular.frames[4], episode.frames[4])
    assert regular.meta["regularized"]["filled_frames"] == 1
    np.testing.assert_allclose(regular.imu, episode.delivered_imu(), rtol=1e-6, atol=1e-9)  # the IMU was complete (CSV keeps 9 digits)
    written = read_episode(write_episode(regular, tmp_path / "regular.zip"))
    np.testing.assert_array_equal(written.frames[3], episode.frames[2])


def test_dropped_imu_samples_repeat_the_previous_one():
    episode = synthetic_episode()
    window = episode.imu_window(4)
    gap = (episode.imu[:, 0] > window[3, 0] + 1e-9) & (episode.imu[:, 0] < window[7, 0] - 1e-9)  # samples 4..6 of frame 4
    regular = regularize(dataclasses.replace(episode, imu=episode.imu[~gap]))
    regular.check_consistency()
    rows = regular.imu_window(4)
    np.testing.assert_allclose(np.diff(rows[:, 0]), 1 / abi.IMU_RATE_HZ, atol=1e-9)
    np.testing.assert_array_equal(rows[:4, 1:], window[:4, 1:])
    np.testing.assert_array_equal(rows[4:7, 1:], np.tile(window[3, 1:], (3, 1)))
    np.testing.assert_array_equal(rows[7:, 1:], window[7:, 1:])
    assert regular.meta["regularized"]["resampled_imu_windows"] == 1
    np.testing.assert_array_equal(np.delete(regular.imu, slice(40, 50), axis=0), np.delete(episode.delivered_imu(), slice(40, 50), axis=0))


@pytest.mark.parametrize("rate", [100.0, 333.0])
def test_other_imu_rate_is_interpolated(rate):
    episode = synthetic_episode()
    source = linear_imu(np.arange(1, int(episode.frame_times[-1] * rate) + 2) / rate)
    regular = regularize(dataclasses.replace(episode, imu=source))
    regular.check_consistency()
    assert all(len(rows) == abi.IMU_ROWS_PER_FRAME for rows in windows(regular))
    np.testing.assert_allclose(np.diff(regular.imu[:, 0]), 1 / abi.IMU_RATE_HZ, atol=1e-9)
    inside = regular.imu[:, 0] >= source[0, 0]  # before the first sample the first one is held
    np.testing.assert_allclose(regular.imu[inside], linear_imu(regular.imu[inside, 0]), atol=1e-9)
    np.testing.assert_array_equal(regular.imu[~inside, 1:], np.tile(source[0, 1:], (int((~inside).sum()), 1)))


def test_first_window_before_the_imu_starts():
    episode = synthetic_episode()
    late = episode.imu[episode.imu[:, 0] > episode.frame_times[0] - 0.02 + 1e-9]  # like the simulator: no IMU at the start
    regular = regularize(dataclasses.replace(episode, imu=late))
    regular.check_consistency()
    first = regular.imu_window(0)
    np.testing.assert_allclose(first[:, 0], episode.imu_window(0)[:, 0], atol=1e-9)
    np.testing.assert_array_equal(first[:6, 1:], np.tile(late[0, 1:], (6, 1)))
    np.testing.assert_array_equal(first[6:, 1:], late[:4, 1:])
    np.testing.assert_array_equal(regular.imu[10:], episode.delivered_imu()[10:])


def test_idempotent():
    episode = synthetic_episode(imu_rate=333.0)
    once = regularize(episode)
    twice = regularize(once)
    np.testing.assert_array_equal(twice.imu, once.imu)
    np.testing.assert_array_equal(twice.frame_times, once.frame_times)
    assert twice.meta["regularized"] == {"filled_frames": 0, "resampled_imu_windows": 0}


def test_other_frame_rate_is_rejected():
    with pytest.raises(ValueError, match="30.0 Hz"):
        regularize(synthetic_episode(rate=30.0))
