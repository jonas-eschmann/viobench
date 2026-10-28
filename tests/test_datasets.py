import gc
import os

import numpy as np
import pytest

from relpose.tools.abi.config import make_config
from relpose.tools.datasets.episode import Episode, FrameSequence, interpolate_poses
from relpose.tools.datasets.storage import list_episodes, read_episode, write_episode
from relpose.tools.geometry import poses_to_matrices, quaternion_to_matrix


def synthetic_episode(n_frames=8, rate=20.0, imu_rate=200.0, width=32, height=24, channels=1, lead_in=0.0):
    """lead_in: seconds of IMU (and ground truth) recorded before the first frame."""
    rng = np.random.default_rng(0)
    config = make_config(width=width, height=height, channels=channels, frame_rate_hz=rate, num_frames=n_frames)
    T_body_camera = np.asarray(config["imu"]["T_imu_cam"]).reshape(4, 4)
    duration = n_frames / rate
    gt_t = np.arange(-lead_in, duration + 1 / imu_rate, 1 / imu_rate)
    # smooth trajectory: circle with yaw
    angle = 0.5 * gt_t
    position = np.column_stack([np.cos(angle), np.sin(angle), 0.1 * gt_t])
    quaternion = np.column_stack([np.zeros_like(angle), np.zeros_like(angle), np.sin(angle / 2), np.cos(angle / 2)])
    ground_truth = np.column_stack([gt_t, position, quaternion])
    imu_t = gt_t[1:]
    imu = np.column_stack([imu_t, rng.normal(0, 0.1, (len(imu_t), 3)) + [0, 0, 9.81], rng.normal(0, 0.01, (len(imu_t), 3))])
    frame_times = (np.arange(n_frames) + 1) / rate
    frames = rng.integers(0, 256, (n_frames, height, width) + (() if channels == 1 else (channels,)), dtype=np.uint8)
    return Episode("synthetic", config, frame_times, frames, imu, ground_truth, T_body_camera, {"condition": "test"})


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


def test_ground_truth_twist_of_an_offset_camera():
    """The synthetic body circles at 0.5 rad/s (yaw = angle), climbing at 0.1 m/s; the camera sits off the body origin, rotated."""
    import dataclasses

    episode = synthetic_episode()
    R_bc = quaternion_to_matrix([0.1, -0.2, 0.3, 0.9])
    T_body_camera = np.eye(4)
    T_body_camera[:3, :3], T_body_camera[:3, 3] = R_bc, [0.05, -0.02, 0.1]
    episode = dataclasses.replace(episode, T_body_camera=T_body_camera)
    angle = 0.5 * episode.frame_times
    velocity_world = np.column_stack([-0.5 * np.sin(angle), 0.5 * np.cos(angle), np.full_like(angle, 0.1)])
    R_wb = quaternion_to_matrix(np.column_stack([np.zeros_like(angle), np.zeros_like(angle), np.sin(angle / 2), np.cos(angle / 2)]))
    omega_body = np.array([0.0, 0.0, 0.5])
    velocity_body = np.einsum("nji,nj->ni", R_wb, velocity_world) + np.cross(omega_body, T_body_camera[:3, 3])  # camera centre: v + w x r
    expected = np.column_stack([velocity_body @ R_bc, np.tile(R_bc.T @ omega_body, (len(angle), 1))])
    twist = episode.ground_truth_twist()
    np.testing.assert_allclose(twist[:-1], expected[:-1], atol=1e-5)
    np.testing.assert_allclose(twist[-1], expected[-1], atol=2e-3)  # the ground truth ends at the last frame: one-sided there
    state = episode.ground_truth_state()
    assert state.shape == (8, 13)
    np.testing.assert_array_equal(state[:, :7], episode.ground_truth_relative())


def test_first_imu_window_is_one_frame_interval():
    episode = synthetic_episode(lead_in=1.5)  # a recording may hold IMU from long before its first frame, simulated episodes barely
    first = episode.imu_window(0)
    assert len(first) == 10 and first[0, 0] > episode.frame_times[0] - episode.frame_interval
    np.testing.assert_array_equal(episode.delivered_imu(), np.concatenate([imu for _, _, imu in episode.steps()]))


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
    assert loaded.config == episode.config and loaded.meta["condition"] == "test" and loaded.meta["dataset"] == tmp_path.name
    assert (directory / "mav0" / "cam0" / "sensor.yaml").is_file() and (directory / "mav0" / "imu0" / "sensor.yaml").is_file()


@pytest.mark.parametrize("channels", [1, 3])
def test_write_episode_copies_existing_pngs(tmp_path, channels):
    episode = synthetic_episode(n_frames=4, width=16, height=12, channels=channels)
    source = read_episode(write_episode(episode, tmp_path / "src"))
    assert isinstance(source.frames, FrameSequence) and len(source.frames.paths) == 4
    copy = write_episode(source, tmp_path / "dst")
    copied = read_episode(copy)
    for original, duplicate in zip(source.frames.paths, copied.frames.paths):
        assert original.read_bytes() == duplicate.read_bytes()
        assert not original.samefile(duplicate), "persistent datasets must have independent images"
    np.testing.assert_array_equal(np.asarray(read_episode(copy).frames), episode.frames)
    modified = read_episode(copy, frames="memory")
    modified.frames[:] = 0
    write_episode(modified, copy)
    np.testing.assert_array_equal(np.asarray(source.frames), episode.frames)
    assert not np.asarray(read_episode(copy).frames).any()
    write_episode(source, copy)
    np.testing.assert_array_equal(np.asarray(read_episode(copy).frames), episode.frames)


@pytest.mark.parametrize("origin_ns", [0, 1_400_000_000_000_000_000])
def test_write_episode_back_to_same_directory(tmp_path, origin_ns):
    episode = synthetic_episode(n_frames=4, width=16, height=12)
    directory = write_episode(episode, tmp_path / "episode", origin_ns=origin_ns)
    loaded = read_episode(directory)
    original_pngs = [path.read_bytes() for path in loaded.frames.paths]
    loaded.meta["condition"] = "corrected"
    write_episode(loaded, directory / ".." / directory.name, origin_ns=origin_ns)
    assert [path.read_bytes() for path in loaded.frames.paths] == original_pngs
    saved = read_episode(directory)
    assert saved.meta["condition"] == "corrected"
    np.testing.assert_array_equal(np.asarray(saved.frames), episode.frames)


@pytest.mark.parametrize("frames", ["lazy", "memory"])
def test_write_episode_detaches_existing_hardlinks(tmp_path, frames):
    episode = synthetic_episode(n_frames=4, width=16, height=12)
    source = read_episode(write_episode(episode, tmp_path / "src"))
    destination = write_episode(source, tmp_path / "dst")
    paths = read_episode(destination).frames.paths
    for original, duplicate in zip(source.frames.paths, paths):
        duplicate.unlink()
        os.link(original, duplicate)  # an export made by an older writer, or a still-live simulator spool
    rewritten = read_episode(tmp_path / "src", frames=frames)
    if frames == "memory":
        rewritten.frames[:] = 0
    write_episode(rewritten, destination)
    assert all(not original.samefile(duplicate) for original, duplicate in zip(source.frames.paths, paths))
    np.testing.assert_array_equal(np.asarray(source.frames), episode.frames)
    np.testing.assert_array_equal(np.asarray(read_episode(destination).frames), np.asarray(rewritten.frames))


@pytest.mark.parametrize("cross_volume", [False, True])
def test_write_episode_exports_spool_and_survives_cleanup(tmp_path, monkeypatch, cross_volume):
    import errno

    from relpose.tools.datasets.generate.simulation import Spool, spooled_frames

    episode = synthetic_episode(n_frames=4, width=16, height=12, channels=3)  # the simulator spools RGB
    expected = episode.frames.copy()
    spool = Spool(tmp_path / "spool")
    spool_path = spool.path
    stored = read_episode(write_episode(episode, spool.path / "episode"))
    episode.frames = spooled_frames(stored.frames.paths, expected.shape[1:], spool)
    mapped = episode.frames.map(np.copy)
    assert mapped.paths is None and not mapped.link_paths
    del mapped, spool

    if cross_volume:
        def cannot_link(source, target):
            raise OSError(errno.EXDEV, "cross-device link")

        monkeypatch.setattr(os, "link", cannot_link)
    destination = write_episode(episode, tmp_path / "dst")
    paths = read_episode(destination).frames.paths
    assert all(original.samefile(duplicate) == (not cross_volume) for original, duplicate in zip(stored.frames.paths, paths))
    write_episode(episode, destination)  # exporting again replaces existing links safely
    np.testing.assert_array_equal(np.asarray(episode.frames), expected)
    del episode
    gc.collect()
    assert not spool_path.exists()
    np.testing.assert_array_equal(np.asarray(read_episode(destination).frames), expected)


@pytest.mark.parametrize("frames", ["lazy", "memory"])
def test_failed_image_write_preserves_destination(tmp_path, monkeypatch, frames):
    from pathlib import Path

    from relpose.tools.datasets import storage

    episode = synthetic_episode(n_frames=4, width=16, height=12)
    source = read_episode(write_episode(episode, tmp_path / "src"), frames=frames)
    episode.frames[:] = 0
    destination = write_episode(episode, tmp_path / "dst")
    paths = read_episode(destination).frames.paths
    original_pngs = [path.read_bytes() for path in paths]

    def fail_copy(source, target):
        Path(target).write_bytes(b"partial PNG")
        raise OSError("copy failed")

    def fail_encode(target, *args):
        Path(target).write_bytes(b"partial PNG")
        return False

    monkeypatch.setattr(storage.shutil, "copyfile", fail_copy)
    monkeypatch.setattr(storage.cv2, "imwrite", fail_encode)
    with pytest.raises(OSError):
        write_episode(source, destination)
    assert [path.read_bytes() for path in paths] == original_pngs
    assert not list((destination / "mav0" / "cam0" / "data").glob(".relpose-write-*"))


def test_simulation_spool_lives_as_long_as_its_frames(tmp_path):
    from relpose.tools.datasets.generate.simulation import Spool, spooled_frames

    spool = Spool(tmp_path / "spool")
    path = spool.path
    assert path.is_dir() and path.parent == tmp_path / "spool"
    frames = spooled_frames([], (2, 2, 3), spool)
    del spool
    gc.collect()
    assert path.is_dir(), "the frame sequence keeps the spool alive"
    del frames
    gc.collect()
    assert not path.exists(), "the last reference removes the directory"


def test_sim_config_scales_with_duration(tmp_path, monkeypatch):
    from relpose.tools.datasets.generate.simulation import SimConfig

    monkeypatch.setenv("RELPOSE_CACHE", str(tmp_path))
    minute = SimConfig()
    assert minute.duration_seconds == 60.0 and minute.step_count(200.0) == 12000 and minute.num_waypoints == 48
    assert SimConfig(duration_seconds=10).num_waypoints == 8 and SimConfig(duration_seconds=10, waypoints=5).num_waypoints == 5
    header = minute.render_header(200.0).read_text()
    assert "NUM_WAYPOINTS = 48;" in header and "INITIALIZATION_HOLD_STEPS = 200;" in header and "@" not in header
    assert SimConfig(duration_seconds=30).render_header(200.0) != minute.render_header(200.0)  # a different route length is a different JIT key


def test_adapt_always_delivers_rgb():
    episode = synthetic_episode(n_frames=3, width=320, height=240)  # a grayscale-only set
    config, frames = episode.adapt()
    assert frames.shape == (3, 240, 320, 3)
    for channel in range(3):
        np.testing.assert_array_equal(frames[..., channel], episode.frames)  # replicated, and nothing announces it
    assert set(config) == {"width", "height", "fx", "fy", "cx", "cy", "T_cam_imu"}
    np.testing.assert_allclose(np.asarray(config["T_cam_imu"]) @ episode.T_body_camera, np.eye(4), atol=1e-12)
    rgb = synthetic_episode(n_frames=3, width=320, height=240, channels=3)
    assert rgb.adapt()[1] is rgb.frames  # RGB at the stored resolution passes through untouched


def test_adapt_rescales_frames_and_intrinsics():
    episode = synthetic_episode(n_frames=2, width=640, height=480)
    config, frames = episode.adapt(resolution=(320, 240))
    assert frames.shape == (2, 240, 320, 3) and (config["width"], config["height"]) == (320, 240)
    assert config["fx"] == pytest.approx(episode.config["camera"]["fx"] / 2)
    assert config["cx"] == pytest.approx((episode.config["camera"]["cx"] + 0.5) / 2 - 0.5)
    # a frame-centre ray keeps its direction: the principal point stays at the image centre
    assert config["cx"] == pytest.approx(320 / 2 - 0.5) and config["cy"] == pytest.approx(240 / 2 - 0.5)
    small = synthetic_episode(n_frames=2, width=320, height=240)
    config, frames = small.adapt(resolution=(640, 480))
    assert frames.shape == (2, 480, 640, 3) and (config["width"], config["height"]) == (640, 480)
    assert config["fx"] == pytest.approx(small.config["camera"]["fx"] * 2)
    assert config["cx"] == pytest.approx((small.config["camera"]["cx"] + 0.5) * 2 - 0.5)
    with pytest.raises(ValueError, match="resolution must be one of"):
        episode.adapt(resolution=(64, 48))
    with pytest.raises(ValueError, match="sets must be stored at"):
        synthetic_episode(n_frames=2, width=64, height=48).adapt(resolution=(320, 240))
