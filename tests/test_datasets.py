import numpy as np
import pytest

from eecs116.config import make_config
from eecs116.data.episode import Episode, interpolate_poses
from eecs116.data.euroc_io import list_episodes, read_episode, write_episode
from eecs116.data.isp import ISPParams, apply, integrate_exposure, linear_to_srgb, srgb_to_linear
from eecs116.metrics import matrices_to_poses, poses_to_matrices, quaternion_to_matrix


def synthetic_episode(n_frames=8, rate=20.0, imu_rate=200.0, width=32, height=24):
    rng = np.random.default_rng(0)
    config = make_config(width=width, height=height, channels=1, frame_rate_hz=rate, num_frames=n_frames)
    T_body_camera = np.asarray(config["imu"]["T_imu_cam"]).reshape(4, 4)
    duration = n_frames / rate
    gt_t = np.arange(0, duration + 1 / imu_rate, 1 / imu_rate)
    # smooth trajectory: circle with yaw
    angle = 0.5 * gt_t
    position = np.column_stack([np.cos(angle), np.sin(angle), 0.1 * gt_t])
    quaternion = np.column_stack([np.zeros_like(angle), np.zeros_like(angle), np.sin(angle / 2), np.cos(angle / 2)])
    ground_truth = np.column_stack([gt_t, position, quaternion])
    imu_t = gt_t[1:]
    imu = np.column_stack([imu_t, rng.normal(0, 0.1, (len(imu_t), 3)) + [0, 0, 9.81], rng.normal(0, 0.01, (len(imu_t), 3))])
    frame_times = (np.arange(n_frames) + 1) / rate
    frames = rng.integers(0, 256, (n_frames, height, width), dtype=np.uint8)
    return Episode("synthetic", config, frame_times, frames, imu, ground_truth, T_body_camera, {"variant": "test"})


def test_episode_windows_and_ground_truth():
    episode = synthetic_episode()
    episode.check_consistency()
    windows = [imu for _, _, imu in episode.steps()]
    assert len(windows) == 8
    assert all(len(w) == 10 for w in windows[1:])  # 200 Hz / 20 Hz
    assert windows[0][-1, 0] <= episode.frame_times[0] and all(w[0, 0] > t_prev for w, t_prev in zip(windows[1:], episode.frame_times[:-1]))
    relative = episode.ground_truth_relative()
    np.testing.assert_allclose(relative[0], [0, 0, 0, 0, 0, 0, 1], atol=1e-12)
    # consistency with a direct computation at frame 3
    T_wc = poses_to_matrices(episode.camera_poses_world())
    expected = np.linalg.inv(T_wc[0]) @ T_wc[3]
    np.testing.assert_allclose(poses_to_matrices(relative)[3], expected, atol=1e-12)


def test_interpolation_matches_samples():
    times = np.array([0.0, 1.0, 2.0])
    poses = np.array([[0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, np.sin(0.5), np.cos(0.5)], [2, 0, 0, 0, 0, np.sin(1.0), np.cos(1.0)]])
    mid = interpolate_poses(times, poses, [0.5])[0]
    assert mid[0] == pytest.approx(0.5)
    np.testing.assert_allclose(quaternion_to_matrix(mid[3:]), quaternion_to_matrix([0, 0, np.sin(0.25), np.cos(0.25)]), atol=1e-12)
    with pytest.raises(ValueError):
        interpolate_poses(times, poses, [2.5])


def test_write_read_roundtrip(tmp_path):
    episode = synthetic_episode()
    directory = write_episode(episode, tmp_path / "ep0", origin_ns=1_400_000_000_000_000_000)
    assert list_episodes(tmp_path) == [directory]
    loaded = read_episode(directory)
    loaded.check_consistency()
    np.testing.assert_array_equal(loaded.frames, episode.frames)
    np.testing.assert_allclose(loaded.frame_times, episode.frame_times, atol=1e-9)
    np.testing.assert_allclose(loaded.imu, episode.imu, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(loaded.ground_truth, episode.ground_truth, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(loaded.T_body_camera, episode.T_body_camera)
    assert loaded.config == episode.config and loaded.meta == {"variant": "test"}
    assert (directory / "mav0" / "cam0" / "sensor.yaml").is_file() and (directory / "mav0" / "imu0" / "sensor.yaml").is_file()


def test_isp_identity_and_effects():
    rng = np.random.default_rng(0)
    srgb = rng.uniform(0, 1, (4, 16, 16, 3)).astype(np.float32)
    linear = integrate_exposure(srgb)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(srgb)), srgb, atol=1e-5)
    clean = apply(linear, ISPParams(full_well=np.inf, read_noise=0.0), rng)
    assert clean.shape == (16, 16) and clean.dtype == np.uint8
    dark = apply(linear, ISPParams(exposure_scale=0.05, full_well=np.inf, read_noise=0.0), rng)
    bright = apply(linear, ISPParams(exposure_scale=20.0, full_well=np.inf, read_noise=0.0), rng)
    assert dark.mean() < clean.mean() < bright.mean() and (bright == 255).mean() > 0.9
    noisy = apply(linear, ISPParams(full_well=200.0, read_noise=0.05), rng)
    assert np.abs(noisy.astype(float) - clean.astype(float)).mean() > 2.0
    # a static stack blurs to itself
    np.testing.assert_allclose(integrate_exposure(np.repeat(srgb[:1], 5, axis=0)), srgb_to_linear(srgb[0]), atol=1e-6)


def test_as_channels_conversions():
    episode = synthetic_episode(n_frames=3, width=16, height=12)
    config, frames = episode.as_channels(3)
    assert frames.shape == (3, 12, 16, 3) and config["image"]["channels"] == 3 and config["image"]["color"] is False
    np.testing.assert_array_equal(frames[..., 0], episode.frames)
    rgb = type(episode)(episode.name, {**episode.config, "image": {**episode.config["image"], "channels": 3, "layout": "HWC"}},
                        episode.frame_times, frames, episode.imu, episode.ground_truth, episode.T_body_camera)
    config, gray = rgb.as_channels(1)
    assert gray.shape == (3, 12, 16) and config["image"]["channels"] == 1 and config["image"]["source_channels"] == 3
    np.testing.assert_array_equal(gray, episode.frames)  # replicated channels -> identical luma
    assert rgb.as_channels(3)[1] is frames


def test_adapt_resolution_rescales_frames_and_intrinsics():
    episode = synthetic_episode(n_frames=2, width=64, height=48)
    config, frames = episode.adapt(resolution=(32, 24))
    assert frames.shape == (2, 24, 32) and config["image"]["width"] == 32 and config["image"]["upscaled"] is False
    assert config["camera"]["fx"] == pytest.approx(episode.config["camera"]["fx"] / 2)
    assert config["camera"]["cx"] == pytest.approx((episode.config["camera"]["cx"] + 0.5) / 2 - 0.5)
    # a frame-centre ray keeps its direction: the principal point stays at the image centre
    assert config["camera"]["cx"] == pytest.approx(32 / 2 - 0.5) and config["camera"]["cy"] == pytest.approx(24 / 2 - 0.5)
    config_up, frames_up = episode.adapt(channels=3, resolution=(128, 96))
    assert frames_up.shape == (2, 96, 128, 3) and config_up["image"]["upscaled"] is True and config_up["image"]["source_resolution"] == [64, 48]
    assert config_up["camera"]["fx"] == pytest.approx(episode.config["camera"]["fx"] * 2)
    with pytest.raises(ValueError, match="aspect"):
        episode.adapt(resolution=(64, 24))
