"""End-to-end: synthetic episodes on disk -> evaluate a bundled solution -> report with metrics and fuel."""
import json

import numpy as np

from eecs116.bundle import bundle
from eecs116.data.euroc_io import write_episode
from eecs116.evaluate import evaluate
from tests.test_data import synthetic_episode

PERFECT = """
import numpy as np
import eecs116_bundle as bundle
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


def test_evaluate_report(runtime, runtime_path, solution_dir, tmp_path):
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
    report = evaluate(wasm, [dataset], output_dir=tmp_path / "report", runtime=runtime, log=lambda *a: None)
    assert report["summary"]["overall"]["episodes"] == 1 and report["summary"]["overall"]["completed"] == 1
    episode_report = report["episodes"][0]
    assert episode_report["metrics"]["ate_rmse"] < 1e-9 and episode_report["metrics"]["coverage"] == 1.0
    assert episode_report["fuel_per_frame_median"] > 0
    assert (tmp_path / "report" / "report.json").is_file()
    assert json.loads((tmp_path / "report" / "report.json").read_text())["episodes"][0]["name"] == episode.name
    estimate = np.loadtxt(tmp_path / "report" / "logs" / "set" / episode.name / "estimate.tum")
    np.testing.assert_allclose(estimate[:, 1:], gt, atol=1e-6)
