"""End-to-end: synthetic episodes on disk -> evaluate a bundled solution -> report with metrics and fuel."""
import json

import numpy as np

from relpose.solution.runtime.python.bundler.bundle import bundle
from relpose.tools.datasets.storage import write_episode
from relpose.tools.runner.run import evaluate
from tests.test_datasets import synthetic_episode

PERFECT = """
import numpy as np
import solution_bundle as bundle
class Solution:
    # cheats: replays the ground truth shipped in the bundle (tests the plumbing, not an estimator)
    def __init__(self, config):
        self.poses = np.load(bundle.open("gt.npy"))
        self.k = 0
    def step(self, t, frame, imu):
        pose = self.poses[self.k]
        self.k += 1
        return pose
"""


def test_evaluate_report(engine, runtime_path, solution_dir, tmp_path):
    episode = synthetic_episode(n_frames=12, width=320, height=240)
    dataset = tmp_path / "set"
    write_episode(episode, dataset / episode.name)
    gt = episode.ground_truth_relative()
    import io

    buffer = io.BytesIO()
    np.save(buffer, gt)
    directory = solution_dir(PERFECT, {"gt.npy": buffer.getvalue()})
    wasm = tmp_path / "perfect.wasm"
    bundle(directory, wasm, runtime=runtime_path)
    report = evaluate(wasm, [dataset], output_dir=tmp_path / "report", engine=engine, log=lambda *a: None)
    assert report["summary"]["overall"]["episodes"] == 1 and report["summary"]["overall"]["completed"] == 1
    episode_report = report["episodes"][0]
    assert episode_report["metrics"]["ate_rmse"] < 1e-9 and episode_report["metrics"]["coverage"] == 1.0
    assert episode_report["fuel_per_frame_median"] > 0
    assert (tmp_path / "report" / "report.json").is_file()
    assert json.loads((tmp_path / "report" / "report.json").read_text())["episodes"][0]["name"] == episode.name
    estimate = np.loadtxt(tmp_path / "report" / "logs" / "set" / episode.name / "estimate.tum")
    np.testing.assert_allclose(estimate[:, 1:], gt, atol=1e-6)


def test_evaluate_delivers_rgb_from_gray_and_rgb_sets(engine, runtime_path, solution_dir, tmp_path):
    gray = synthetic_episode(n_frames=6, width=320, height=240)
    rgb = synthetic_episode(n_frames=6, width=320, height=240, channels=3)
    write_episode(gray, tmp_path / "gray" / gray.name)
    write_episode(rgb, tmp_path / "rgb" / rgb.name)
    probe = """
import numpy as np
class Solution:
    def __init__(self, config):
        assert "channels" not in config and "color" not in config
    def step(self, t, frame, imu):
        replicated = np.array_equal(frame[..., 0], frame[..., 1]) and np.array_equal(frame[..., 1], frame[..., 2])
        return [frame.ndim, frame.shape[2], float(replicated), 0, 0, 0, 1]
"""
    wasm = tmp_path / "probe.wasm"
    bundle(solution_dir(probe), wasm, runtime=runtime_path)
    report = evaluate(wasm, [tmp_path / "gray", tmp_path / "rgb"], output_dir=tmp_path / "report", engine=engine, log=lambda *a: None)
    assert report["summary"]["overall"]["completed"] == 2
    for dataset, replicated in (("gray", 1.0), ("rgb", 0.0)):
        estimate = np.loadtxt(tmp_path / "report" / "logs" / dataset / gray.name / "estimate.tum")
        assert estimate[0, 1] == 3 and estimate[0, 2] == 3 and estimate[0, 3] == replicated


def test_evaluate_resolution_opt_in(engine, runtime_path, solution_dir, tmp_path):
    episode = synthetic_episode(n_frames=4, width=640, height=480)
    dataset = tmp_path / "set"
    write_episode(episode, dataset / episode.name)
    probe = """
class Solution:
    def __init__(self, config):
        self.config = config
    def step(self, t, frame, imu):
        return [frame.shape[1], frame.shape[0], self.config["fx"], 0, 0, 0, 1]
"""
    import pytest

    # only the two supported resolutions are accepted by the manifest validation
    with pytest.raises(Exception, match="resolution"):
        bundle(solution_dir(probe, {"solution.json": '{"resolution": [128, 96]}'}, name="bad"), tmp_path / "bad.wasm", runtime=runtime_path)
    for name, manifest, size in (("qvga", "{}", (320, 240)), ("vga", '{"resolution": "640x480"}', (640, 480))):
        wasm = tmp_path / f"{name}.wasm"
        assert bundle(solution_dir(probe, {"solution.json": manifest}, name=name), wasm, runtime=runtime_path)["resolution"] == list(size)
        report = evaluate(wasm, [dataset], output_dir=tmp_path / name, engine=engine, log=lambda *a: None)
        estimate = np.loadtxt(tmp_path / name / "logs" / "set" / episode.name / "estimate.tum")
        assert (estimate[0, 1], estimate[0, 2]) == size and report["summary"]["overall"]["completed"] == 1
        assert estimate[0, 3] == pytest.approx(episode.config["camera"]["fx"] * size[0] / 640)
    # a set stored at 320x240 serves a 640x480 solution: the harness upscales
    small = synthetic_episode(n_frames=4, width=320, height=240)
    write_episode(small, tmp_path / "small" / small.name)
    report = evaluate(tmp_path / "vga.wasm", [tmp_path / "small"], output_dir=tmp_path / "upscaled", engine=engine, log=lambda *a: None)
    estimate = np.loadtxt(tmp_path / "upscaled" / "logs" / "small" / small.name / "estimate.tum")
    assert (estimate[0, 1], estimate[0, 2]) == (640, 480) and report["summary"]["overall"]["completed"] == 1
    assert estimate[0, 3] == pytest.approx(small.config["camera"]["fx"] * 2)


def test_evaluate_accepts_an_in_memory_dataset(engine, runtime_path, solution_dir, tmp_path):
    """The ad-hoc provider path: a list of Episodes, nothing on disk (this is what --simulate feeds the runner)."""
    import io

    episode = synthetic_episode(n_frames=5, width=320, height=240)
    episode.meta["dataset"] = "adhoc"
    gt = episode.ground_truth_relative()
    buffer = io.BytesIO()
    np.save(buffer, gt)
    wasm = tmp_path / "perfect.wasm"
    bundle(solution_dir(PERFECT, {"gt.npy": buffer.getvalue()}, name="perfect"), wasm, runtime=runtime_path)
    report = evaluate(wasm, [episode, episode], output_dir=tmp_path / "report", engine=engine, log=lambda *a: None)
    assert report["summary"]["overall"]["episodes"] == 2 and list(report["summary"]["datasets"]) == ["adhoc"]
    assert report["episodes"][0]["metrics"]["ate_rmse"] < 1e-9


def test_streamed_frames_from_disk(tmp_path):
    from relpose.tools.datasets.episode import FrameSequence
    from relpose.tools.datasets.storage import episodes, read_episode

    episode = synthetic_episode(n_frames=6, width=320, height=240)
    write_episode(episode, tmp_path / "set" / episode.name)
    lazy = read_episode(tmp_path / "set" / episode.name)
    assert isinstance(lazy.frames, FrameSequence) and lazy.frames.shape == (6, 240, 320) and lazy.channels == 1
    np.testing.assert_array_equal(lazy.frames[3], episode.frames[3])
    np.testing.assert_array_equal(np.asarray(lazy.frames), episode.frames)
    config, frames = lazy.adapt()
    assert isinstance(frames, FrameSequence) and frames.shape == (6, 240, 320, 3)
    assert next(iter(episodes([tmp_path / "set"]))).dataset == "set"
