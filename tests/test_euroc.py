"""The EuRoC converter on a synthetic ASL-layout sequence (our writer produces the same layout)."""
import numpy as np

from relpose.datasets.euroc import convert, rectification
from relpose.datasets.storage import write_episode
from tests.test_datasets import synthetic_episode


def test_rectification_keeps_center_and_scales():
    sensor = {"intrinsics": [458.0, 457.0, 367.2, 248.4], "resolution": [752, 480], "distortion_model": "radial-tangential",
              "distortion_coefficients": [-0.28, 0.07, 0.0002, 0.00002]}
    map_x, map_y, K = rectification(sensor, 320, 240)
    assert map_x.shape == (240, 320)
    # the principal point of the new camera lands on the principal point of the old one (no distortion there)
    assert abs(map_x[int(round(K[1, 2])), int(round(K[0, 2]))] - 367.2) < 1.5
    assert abs(map_y[int(round(K[1, 2])), int(round(K[0, 2]))] - 248.4) < 1.5
    assert abs(K[0, 0] - 458.0 / 2) < 1e-9  # 640-wide crop scaled to 320


def test_convert_synthetic_sequence(tmp_path):
    episode = synthetic_episode(n_frames=60, rate=20.0, width=48, height=36)
    write_episode(episode, tmp_path / "seq", origin_ns=1_403_636_579_763_555_584)
    episodes = list(convert(tmp_path / "seq", width=32, height=24, snippet_seconds=1.0, log=lambda *a: None))
    assert len(episodes) == 2  # 3 s of frames inside the ground truth span -> two full 1 s snippets (the last one is short)
    first, origin = episodes[0][0], episodes[0][1]
    assert first.frames.shape == (20, 24, 32) and first.frames.dtype == np.uint8
    assert first.config["image"]["width"] == 32 and first.config["camera"]["fx"] > 0
    assert first.meta["source"] == "euroc" and first.meta["hover_seconds"] == 1.0
    assert episodes[1][0].meta["hover_seconds"] == 0.0
    np.testing.assert_allclose(first.ground_truth_relative()[0], [0, 0, 0, 0, 0, 0, 1], atol=1e-9)
    assert first.imu[0, 0] < first.frame_times[0] < first.imu[-1, 0] + 1e-9
