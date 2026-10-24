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
    episode = synthetic_episode(n_frames=12, width=64, height=48)
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


def test_evaluate_converts_rgb_episode_per_solution(engine, runtime_path, solution_dir, tmp_path):
    episode = synthetic_episode(n_frames=6, width=64, height=48)
    rgb = np.stack([episode.frames, episode.frames // 2, episode.frames // 4], axis=3)
    episode = type(episode)(episode.name, {**episode.config, "image": {**episode.config["image"], "channels": 3, "layout": "HWC"}},
                            episode.frame_times, rgb, episode.imu, episode.ground_truth, episode.T_body_camera, episode.meta)
    dataset = tmp_path / "set"
    write_episode(episode, dataset / episode.name)
    probe = """
import numpy as np
class Solution:
    def __init__(self, config):
        self.channels = config["image"]["channels"]
        self.color = config["image"].get("color")
    def step(self, t, frame, imu):
        return [frame.ndim, self.channels, float(self.color), 0, 0, 0, 1]
"""
    for channels in (1, 3):
        directory = solution_dir(probe, {"solution.json": '{"channels": %d}' % channels}, name=f"probe{channels}")
        wasm = tmp_path / f"probe{channels}.wasm"
        bundle(directory, wasm, runtime=runtime_path)
        report = evaluate(wasm, [dataset], output_dir=tmp_path / f"report{channels}", engine=engine, log=lambda *a: None)
        estimate = np.loadtxt(tmp_path / f"report{channels}" / "logs" / "set" / episode.name / "estimate.tum")
        assert estimate[0, 1] == (2 if channels == 1 else 3) and estimate[0, 2] == channels and estimate[0, 3] == (1.0 if channels == 3 else 0.0)
        assert report["summary"]["overall"]["completed"] == 1


def test_evaluate_resolution_opt_in(engine, runtime_path, solution_dir, tmp_path):
    episode = synthetic_episode(n_frames=4, width=64, height=48)
    dataset = tmp_path / "set"
    write_episode(episode, dataset / episode.name)
    probe = """
class Solution:
    def __init__(self, config):
        self.config = config
    def step(self, t, frame, imu):
        image = self.config["image"]
        return [frame.shape[1], frame.shape[0], self.config["camera"]["fx"], float(image["upscaled"]), 0, 0, 1]
"""
    import pytest

    # only the two supported resolutions are accepted by the manifest validation

    with pytest.raises(Exception, match="resolution"):
        bundle(solution_dir(probe, {"solution.json": '{"resolution": [128, 96]}'}, name="bad"), tmp_path / "bad.wasm", runtime=runtime_path)
    directory = solution_dir(probe, {"solution.json": '{"resolution": "640x480"}'}, name="vga")
    wasm = tmp_path / "vga.wasm"
    assert bundle(directory, wasm, runtime=runtime_path)["resolution"] == [640, 480]
    report = evaluate(wasm, [dataset], output_dir=tmp_path / "report", engine=engine, log=lambda *a: None)
    estimate = np.loadtxt(tmp_path / "report" / "logs" / "set" / episode.name / "estimate.tum")
    assert estimate[0, 1] == 640 and estimate[0, 2] == 480 and estimate[0, 4] == 1.0  # upscaled from 64x48
    assert estimate[0, 3] == pytest.approx(episode.config["camera"]["fx"] * 10)
    assert report["summary"]["overall"]["completed"] == 1


def test_evaluate_accepts_an_in_memory_dataset(engine, runtime_path, solution_dir, tmp_path):
    """The ad-hoc provider path: a list of Episodes, nothing on disk (this is what --simulate feeds the runner)."""
    import io

    episode = synthetic_episode(n_frames=5, width=64, height=48)
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

    episode = synthetic_episode(n_frames=6, width=32, height=24)
    write_episode(episode, tmp_path / "set" / episode.name)
    lazy = read_episode(tmp_path / "set" / episode.name)
    assert isinstance(lazy.frames, FrameSequence) and lazy.frames.shape == (6, 24, 32) and lazy.channels == 1
    np.testing.assert_array_equal(lazy.frames[3], episode.frames[3])
    np.testing.assert_array_equal(np.asarray(lazy.frames), episode.frames)
    config, frames = lazy.adapt(channels=3, resolution=(64, 48))
    assert isinstance(frames, FrameSequence) and frames.shape == (6, 48, 64, 3)
    assert next(iter(episodes([tmp_path / "set"]))).dataset == "set"
