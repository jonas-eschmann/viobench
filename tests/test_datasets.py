import gc
import json
import zipfile

import numpy as np
import pytest

from viobench.tools.abi.config import make_config
from viobench.tools.datasets.episode import Episode, FrameSequence, interpolate_poses
from viobench.tools.datasets.storage import (
    episode_file,
    episodes,
    list_episodes,
    read_episode,
    write_episode,
)
from viobench.tools.geometry import poses_to_matrices, quaternion_to_matrix


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


def _pngs(path):
    """The PNG members of an episode archive, in archive order (frame order)."""
    with zipfile.ZipFile(path) as archive:
        return [archive.read(name) for name in archive.namelist() if name.endswith(".png")]


def test_write_read_roundtrip(tmp_path):
    episode = synthetic_episode(width=640, height=480)
    path = write_episode(episode, tmp_path / "ep0.zip", origin_ns=1_400_000_000_000_000_000)
    assert list_episodes(tmp_path) == [path] and list_episodes(path) == [path]
    loaded = read_episode(path)
    loaded.check_consistency()
    np.testing.assert_array_equal(loaded.frames, episode.frames)
    np.testing.assert_allclose(loaded.frame_times, episode.frame_times, atol=1e-9)
    np.testing.assert_allclose(loaded.imu, episode.imu, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(loaded.ground_truth, episode.ground_truth, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(loaded.T_body_camera, episode.T_body_camera)
    assert loaded.config == episode.config and loaded.meta["condition"] == "test" and loaded.meta["dataset"] == tmp_path.name
    assert loaded.meta["origin_ns"] == 1_400_000_000_000_000_000 and loaded.meta["path"] == str(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert names[0] == "ep0/episode.json" and "ep0/mav0/cam0/sensor.yaml" in names and "ep0/mav0/imu0/sensor.yaml" in names
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    # the same episode is the same archive, byte for byte: the sets are content-addressed
    assert write_episode(episode, tmp_path / "again" / "ep0.zip", origin_ns=1_400_000_000_000_000_000).read_bytes() == path.read_bytes()
    assert write_episode(loaded, tmp_path / "resaved" / "ep0.zip").read_bytes() == path.read_bytes()  # keeps the origin, copies the PNGs
    with pytest.raises(ValueError):
        write_episode(episode, tmp_path / "ep0")


def test_unzipped_episode_reads_the_same(tmp_path):
    episode = synthetic_episode(n_frames=4, width=640, height=480, channels=3)
    path = write_episode(episode, tmp_path / "set" / "ep.zip", attachments={"trajectory.mp4": b"not really a video"})
    assert episode_file(path, "trajectory.mp4") == b"not really a video" and episode_file(path, "notes.txt") is None
    with zipfile.ZipFile(path) as archive:
        archive.extractall(tmp_path / "unzipped")
    directory = tmp_path / "unzipped" / "ep"
    assert list_episodes(tmp_path / "unzipped") == [directory] and (directory / "mav0" / "cam0" / "data.csv").is_file()
    unzipped = read_episode(directory)
    np.testing.assert_array_equal(np.asarray(unzipped.frames), episode.frames)
    assert unzipped.meta["dataset"] == "unzipped" and episode_file(directory, "trajectory.mp4") == b"not really a video"
    assert [e.meta["dataset"] for e in episodes([tmp_path / "set", directory])] == ["set", "unzipped"]
    repacked = write_episode(unzipped, tmp_path / "repacked" / "ep.zip", attachments={"trajectory.mp4": directory / "trajectory.mp4"})
    assert repacked.read_bytes() == path.read_bytes()


@pytest.mark.parametrize("channels", [1, 3])
def test_write_episode_copies_existing_pngs(tmp_path, channels):
    episode = synthetic_episode(n_frames=4, width=640, height=480, channels=channels)
    source = read_episode(write_episode(episode, tmp_path / "src.zip"))
    assert isinstance(source.frames, FrameSequence) and source.frames.encoded is not None
    copy = write_episode(source, tmp_path / "dst.zip")
    assert _pngs(copy) == _pngs(tmp_path / "src.zip")
    np.testing.assert_array_equal(np.asarray(read_episode(copy).frames), episode.frames)
    modified = read_episode(copy, frames="memory")
    modified.frames[:] = 0
    write_episode(modified, copy)
    np.testing.assert_array_equal(np.asarray(source.frames), episode.frames)
    assert not np.asarray(read_episode(copy).frames).any()
    write_episode(source, copy)
    np.testing.assert_array_equal(np.asarray(read_episode(copy).frames), episode.frames)


@pytest.mark.parametrize("origin_ns", [0, 1_400_000_000_000_000_000])
def test_write_episode_back_to_its_own_archive(tmp_path, origin_ns):
    episode = synthetic_episode(n_frames=4, width=640, height=480)
    path = write_episode(episode, tmp_path / "episode.zip", origin_ns=origin_ns)
    original_pngs = _pngs(path)
    loaded = read_episode(path)  # streams its frames from the archive that is about to be replaced
    loaded.meta["condition"] = "corrected"
    write_episode(loaded, path)
    assert _pngs(path) == original_pngs
    saved = read_episode(path)
    assert saved.meta["condition"] == "corrected" and saved.meta["origin_ns"] == origin_ns
    np.testing.assert_array_equal(np.asarray(saved.frames), episode.frames)
    np.testing.assert_array_equal(np.asarray(loaded.frames), episode.frames)  # the open reader still sees the archive it opened


def test_write_episode_exports_spool_and_survives_cleanup(tmp_path):
    from viobench.tools.datasets.generate.simulation import Spool, spooled_frames
    from viobench.tools.datasets.storage import encode_png

    episode = synthetic_episode(n_frames=4, width=640, height=480, channels=3)  # the simulator spools RGB
    expected = episode.frames.copy()
    spool = Spool(tmp_path / "spool")
    spool_path = spool.path
    paths = [spool_path / f"{index}.png" for index in range(len(expected))]
    for png, frame in zip(paths, expected):
        png.write_bytes(encode_png(frame))
    spooled = [png.read_bytes() for png in paths]
    episode.frames = spooled_frames(paths, expected.shape[1:], spool)
    mapped = episode.frames.map(np.copy)
    assert mapped.encoded is None
    del mapped, spool
    destination = write_episode(episode, tmp_path / "dst.zip")
    assert _pngs(destination) == spooled  # copied byte for byte, not re-encoded
    write_episode(episode, destination)  # exporting again replaces the archive
    np.testing.assert_array_equal(np.asarray(episode.frames), expected)
    del episode
    gc.collect()
    assert not spool_path.exists()
    np.testing.assert_array_equal(np.asarray(read_episode(destination).frames), expected)


@pytest.mark.parametrize("frames", ["lazy", "memory"])
def test_failed_frame_write_preserves_destination(tmp_path, monkeypatch, frames):
    from viobench.tools.datasets import storage

    episode = synthetic_episode(n_frames=4, width=640, height=480)
    source = read_episode(write_episode(episode, tmp_path / "src.zip"), frames=frames)
    episode.frames[:] = 0
    destination = write_episode(episode, tmp_path / "dst.zip")
    before = destination.read_bytes()

    def fail(*args):
        raise OSError("frame unavailable")

    if frames == "lazy":
        source.frames.encoded = fail  # the stored PNG cannot be read
    else:
        monkeypatch.setattr(storage, "encode_png", fail)  # the frame cannot be encoded
    with pytest.raises(OSError, match="frame unavailable"):
        write_episode(source, destination)
    assert destination.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["dst.zip", "src.zip"]  # no partial archive left behind


def test_simulation_spool_lives_as_long_as_its_frames(tmp_path):
    from viobench.tools.datasets.generate.simulation import Spool, spooled_frames

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
    from viobench.tools.datasets.generate.simulation import SimConfig

    monkeypatch.setenv("VIOBENCH_CACHE", str(tmp_path))
    minute = SimConfig()
    assert minute.duration_seconds == 60.0 and minute.step_count(200.0) == 12000 and minute.num_waypoints == 48
    assert SimConfig(duration_seconds=10).num_waypoints == 8 and SimConfig(duration_seconds=10, waypoints=5).num_waypoints == 5
    header = minute.render_header(200.0).read_text()
    assert "NUM_WAYPOINTS = 48;" in header and "INITIALIZATION_HOLD_STEPS = 200;" in header and "@" not in header
    assert SimConfig(duration_seconds=30).render_header(200.0) != minute.render_header(200.0)  # a different route length is a different JIT key
    with pytest.raises(ValueError, match="stored at 640x480"):
        SimConfig(width=320, height=240)



def test_simulate_runs_scene_by_scene_and_names_episodes_by_order(tmp_path, monkeypatch):
    """Rounds of sim.instances parallel flights per scene: crashed flights are dropped, the surplus of the last round is
    discarded, survivors are named {name}-{scene}-{index:04d} by order within the scene, and a scene whose flights
    keep crashing is given up with an error."""
    import sys
    import types

    from viobench.tools.datasets.generate import simulation

    monkeypatch.setenv("VIOBENCH_CACHE", str(tmp_path))
    environments = []

    class FakeHyperDrone:
        def __init__(self, references, config=None, seed=0, parameters=None):
            self.reference, self.seed, self.dt, self.closed = references[0], seed, 0.005, False
            environments.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "hyperdrone", types.SimpleNamespace(HyperDrone=FakeHyperDrone, Config=lambda **kwargs: kwargs))
    crashes = {("B", 7, 0)}  # (scene, seed, instance) flights that crash; every flight in scene C crashes
    rounds = []

    def fake_round(env, sim, seed, scene_name, camera, name, log, progress=(0, None)):
        rounds.append((scene_name, seed, progress))
        for instance in range(sim.instances):
            if (scene_name, seed, instance) in crashes or scene_name == "C":
                continue
            episode = synthetic_episode(n_frames=2)
            episode.name = episode.config["episode"]["name"] = f"{name}-{scene_name}-{seed:05d}-{instance:02d}"  # the provisional name
            episode.meta.update(scene=scene_name, seed=seed, instance=instance)
            yield episode

    monkeypatch.setattr(simulation, "_run_batch", fake_round)
    sim = simulation.SimConfig(instances=2)

    episodes = list(simulation.simulate(sim, {"A": "ref-a", "B": "ref-b"}, 3, seed=5, name="sim"))
    assert [episode.name for episode in episodes] == ["sim-A-0000", "sim-A-0001", "sim-A-0002", "sim-B-0000", "sim-B-0001", "sim-B-0002"]
    assert all(episode.config["episode"]["name"] == episode.name for episode in episodes)
    # A's second round has one surplus flight, B's first round one crashed flight: neither is an episode
    assert [(episode.meta["seed"], episode.meta["instance"]) for episode in episodes] == [(5, 0), (5, 1), (6, 0), (7, 1), (8, 0), (8, 1)]
    assert rounds == [("A", 5, (0, 3)), ("A", 6, (2, 3)), ("B", 7, (0, 3)), ("B", 8, (1, 3))]
    assert [(env.reference, env.seed) for env in environments] == [("ref-a", 5), ("ref-b", 7)]
    assert all(env.closed for env in environments)

    generator = simulation.simulate(sim, {"A": "ref-a", "C": "ref-c"}, 1, seed=0)
    assert next(generator).name == "sim-A-0000"
    with pytest.raises(RuntimeError, match=r"C: 0 of 1 episodes after 4 flights \(4 crashed\)"):
        next(generator)
    assert environments[-1].reference == "ref-c" and environments[-1].closed
    with pytest.raises(ValueError):
        list(simulation.simulate(sim, {"A": "ref-a"}, 0))

def test_adapt_always_delivers_rgb():
    episode = synthetic_episode(n_frames=3, width=640, height=480)  # a grayscale-only set
    config, frames = episode.adapt()
    assert frames.shape == (3, 480, 640, 3)
    for channel in range(3):
        np.testing.assert_array_equal(frames[..., channel], episode.frames)  # replicated, and nothing announces it
    assert set(config) == {"width", "height", "fx", "fy", "cx", "cy", "T_cam_imu"}
    np.testing.assert_allclose(np.asarray(config["T_cam_imu"]) @ episode.T_body_camera, np.eye(4), atol=1e-12)
    rgb = synthetic_episode(n_frames=3, width=640, height=480, channels=3)
    assert rgb.adapt()[1] is rgb.frames  # RGB at the stored resolution passes through untouched


def test_adapt_rescales_frames_and_intrinsics():
    episode = synthetic_episode(n_frames=2, width=640, height=480)
    config, frames = episode.adapt(resolution=(320, 240))
    assert frames.shape == (2, 240, 320, 3) and (config["width"], config["height"]) == (320, 240)
    assert config["fx"] == pytest.approx(episode.config["camera"]["fx"] / 2)
    assert config["cx"] == pytest.approx((episode.config["camera"]["cx"] + 0.5) / 2 - 0.5)
    # a frame-centre ray keeps its direction: the principal point stays at the image centre
    assert config["cx"] == pytest.approx(320 / 2 - 0.5) and config["cy"] == pytest.approx(240 / 2 - 0.5)
    with pytest.raises(ValueError, match="resolution must be one of"):
        episode.adapt(resolution=(64, 48))
    with pytest.raises(ValueError, match="stored at 640x480"):
        synthetic_episode(n_frames=2, width=64, height=48).adapt(resolution=(320, 240))


@pytest.mark.parametrize("resolution", [(320, 240), (640, 480)])
def test_adapt_rejects_low_resolution_sources(resolution):
    with pytest.raises(ValueError, match="stored at 640x480"):
        synthetic_episode(n_frames=2, width=320, height=240).adapt(resolution)


@pytest.mark.parametrize("invalid", ["metadata", "array", "encoded"])
def test_write_rejects_low_resolution_without_replacing_archive(tmp_path, invalid):
    from viobench.tools.datasets.storage import encode_png

    episode = synthetic_episode(n_frames=2, width=640, height=480)
    path = write_episode(episode, tmp_path / "episode.zip")
    before = path.read_bytes()
    if invalid == "metadata":
        episode.config["image"].update(width=320, height=240)
    elif invalid == "array":
        episode.frames = episode.frames[:, ::2, ::2]
    else:
        small = encode_png(episode.frames[0, ::2, ::2])
        episode.frames = FrameSequence(lambda index: pytest.fail("encoded PNGs should not be decoded"), 2, (480, 640), encoded=lambda index: small)
    with pytest.raises(ValueError, match="stored at 640x480"):
        write_episode(episode, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def _alter_episode(tmp_path, layout, alter):
    path = write_episode(synthetic_episode(n_frames=2, width=640, height=480), tmp_path / "episode.zip")
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    alter(members)
    if layout == "archive":
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        return path
    root = tmp_path / "unzipped"
    for name, data in members.items():
        member = root / name
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(data)
    return root / "episode"


@pytest.mark.parametrize("layout", ["archive", "directory"])
@pytest.mark.parametrize("frames", ["lazy", "memory", "none"])
@pytest.mark.parametrize("metadata", ["manifest", "sensor"])
def test_read_rejects_low_resolution_metadata(tmp_path, layout, frames, metadata):
    def alter(members):
        if metadata == "manifest":
            manifest = json.loads(members["episode/episode.json"])
            manifest["config"]["image"].update(width=320, height=240)
            members["episode/episode.json"] = json.dumps(manifest).encode()
        else:
            import yaml

            sensor = yaml.safe_load(members["episode/mav0/cam0/sensor.yaml"])
            sensor["resolution"] = [320, 240]
            members["episode/mav0/cam0/sensor.yaml"] = yaml.safe_dump(sensor).encode()

    path = _alter_episode(tmp_path, layout, alter)
    with pytest.raises(ValueError, match="stored at 640x480"):
        read_episode(path, frames=frames)


@pytest.mark.parametrize("layout", ["archive", "directory"])
@pytest.mark.parametrize("frames", ["lazy", "memory"])
def test_read_rejects_low_resolution_png_with_full_resolution_metadata(tmp_path, layout, frames):
    from viobench.tools.datasets.storage import encode_png

    def alter(members):
        name = next(name for name in members if name.endswith(".png"))
        members[name] = encode_png(np.zeros((240, 320), dtype=np.uint8))

    path = _alter_episode(tmp_path, layout, alter)
    with pytest.raises(ValueError, match="stored at 640x480"):
        loaded = read_episode(path, frames=frames)
        loaded.frames[0]


@pytest.mark.parametrize("channels", [1, 3])
def test_disk_frames_downsample_lazily_and_preserve_full_resolution(tmp_path, monkeypatch, channels):
    from viobench.tools.datasets import storage

    episode = synthetic_episode(n_frames=3, width=640, height=480, channels=channels)
    episode.config["camera"].update(fx=470.0, fy=462.0, cx=303.25, cy=227.75)
    path = write_episode(episode, tmp_path / "episode.zip")
    before = path.read_bytes()
    decoded = []
    decode_png = storage.decode_png

    def decode(data, channels):
        decoded.append(data)
        return decode_png(data, channels)

    monkeypatch.setattr(storage, "decode_png", decode)
    loaded = read_episode(path)
    assert decoded == []
    config, frames = loaded.adapt((320, 240))
    assert isinstance(frames, FrameSequence) and frames.shape == (3, 240, 320, 3)
    assert len(decoded) == 1
    expected = episode.frames[2].astype(np.uint16).reshape((240, 2, 320, 2) + (() if channels == 1 else (channels,))).sum(axis=(1, 3))
    expected = ((expected + 2) // 4).astype(np.uint8)
    if channels == 1:
        expected = np.repeat(expected[..., None], 3, axis=2)
    np.testing.assert_array_equal(frames[2], expected)
    assert len(decoded) == 2
    assert (config["fx"], config["fy"], config["cx"], config["cy"]) == (235.0, 231.0, 151.375, 113.625)
    full_config, full_frames = loaded.adapt((640, 480))
    expected_full = episode.frames[2] if channels == 3 else np.repeat(episode.frames[2, ..., None], 3, axis=2)
    np.testing.assert_array_equal(full_frames[2], expected_full)
    assert (full_config["fx"], full_config["fy"], full_config["cx"], full_config["cy"]) == (470.0, 462.0, 303.25, 227.75)
    np.testing.assert_array_equal(config["T_cam_imu"], full_config["T_cam_imu"])
    assert loaded.config == episode.config and path.read_bytes() == before
