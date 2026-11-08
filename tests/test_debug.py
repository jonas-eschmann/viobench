"""The out-of-sandbox debugging facilities: `viobench debug` (Python, in-process), `viobench feed` + the native host
`debug` executable, `viobench score`."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from viobench.tools.cli import main as cli
from viobench.tools.datasets.feed import write_feed
from viobench.tools.datasets.storage import archive_hash, write_episode
from viobench.tools.runner import direct
from viobench.tools.runner.score import load_estimates, score
from tests.test_datasets import synthetic_episode

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "viobench" / "solution" / "examples"


def test_direct_run_matches_the_sandbox(engine, runtime_path, tmp_path):
    """dead_reckoning on the host and inside the runtime give the same poses (same code, same inputs, host numpy)."""
    from viobench.solution.runtime.python.bundler.bundle import bundle
    from viobench.tools.runner.run import evaluate

    episode = synthetic_episode(n_frames=30, width=640, height=480)  # 1.5 s: dead reckoning calibrates on the first 0.5 s (identity), then estimates
    dataset = tmp_path / "set"
    digest = archive_hash(write_episode(episode, dataset / f"{episode.name}.zip"))
    solution = direct.PythonSolution(EXAMPLES / "python" / "dead_reckoning")
    assert solution.resolution == (320, 240)
    [(_, estimates, metrics)] = direct.run(solution, [dataset], output_dir=tmp_path / "direct", log=lambda *a: None)
    assert estimates.shape == (30, 13) and metrics.filled_frames == 0
    np.testing.assert_array_equal(estimates[0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert np.abs(estimates[-1, 7:]).max() > 0  # dead reckoning reports its twist once it has calibrated
    written = load_estimates(tmp_path / "direct" / digest / "estimate.tum", episode.frame_times)
    np.testing.assert_allclose(written, estimates, atol=1e-8)

    wasm = tmp_path / "dr.wasm"
    bundle(EXAMPLES / "python" / "dead_reckoning", wasm, runtime=runtime_path)
    evaluate(wasm, [dataset], output_dir=tmp_path / "report", engine=engine, log=lambda *a: None)
    sandboxed = load_estimates(tmp_path / "report" / "logs" / digest / "estimate.tum", episode.frame_times)
    np.testing.assert_allclose(estimates, sandboxed, atol=1e-6)


def test_direct_frame_width_and_host_solution_bundle(solution_dir, tmp_path):
    """frame_width picks the resolution; solution.py may re-export a class from elsewhere; data files next to the code come through `solution_bundle`."""
    directory = solution_dir(
        "from estimator import Est as Solution\n",
        {
            "estimator.py": (
                "import numpy as np\nimport solution_bundle as bundle\n"
                "class Est:\n"
                "    frame_width = 640\n"
                "    def __init__(self, config):\n"
                "        assert config['width'] == 640 and 'channels' not in config\n"
                "        self.offset = np.load(bundle.open('data/offset.npy'))\n"
                "        assert bundle.exists('data/offset.npy') and bundle.listdir('data') == ['offset.npy'] and bundle.read_text('note.txt') == 'hi'\n"
                "    def step(self, t, frame, imu):\n"
                "        assert frame.shape == (480, 640, 3)\n"
                "        return np.concatenate([self.offset, [0, 0, 0, 1], np.zeros(6)])\n"
            ),
            "note.txt": "hi",
        },
        name="vga",
    )
    import io

    buffer = io.BytesIO()
    np.save(buffer, np.array([1.0, 2.0, 3.0]))
    (directory / "data").mkdir()
    (directory / "data" / "offset.npy").write_bytes(buffer.getvalue())
    solution = direct.PythonSolution(directory)
    assert solution.resolution == (640, 480)
    episode = synthetic_episode(n_frames=3, width=640, height=480)  # grayscale: delivered as RGB like the sandbox does
    steps = list(direct.trace(solution, episode))
    assert len(steps) == 3 and steps[0].frame.shape == (480, 640, 3)
    np.testing.assert_allclose(steps[-1].pose[:3], [1.0, 2.0, 3.0])


def test_direct_cli(tmp_path, capsys):
    episode = synthetic_episode(n_frames=4, width=640, height=480)
    digest = archive_hash(write_episode(episode, tmp_path / "set" / f"{episode.name}.zip"))
    assert cli(["debug", str(EXAMPLES / "python" / "minimal"), str(tmp_path / "set"), "-o", str(tmp_path / "out")]) == 0
    assert f"set/{digest}: Sim(3) ATE" in capsys.readouterr().out
    assert (tmp_path / "out" / digest / "estimate.tum").is_file()
    assert (tmp_path / "out" / digest / "twist.txt").is_file()
    from tests.test_video import read_video

    fps, frames = read_video(tmp_path / "out" / digest / "trajectory.mp4")
    assert fps == pytest.approx(20.0) and len(frames) == episode.num_frames


def test_feed_and_score(tmp_path, capsys):
    episode = synthetic_episode(n_frames=6, width=640, height=480, lead_in=1.0)
    write_episode(episode, tmp_path / "set" / f"{episode.name}.zip")
    feed = write_feed(episode, tmp_path / "feed", resolution=(320, 240))
    assert (feed / "shape.txt").read_text().split() == ["6", "240", "320", "3", "60"]  # IMU: six windows of one frame interval
    frames = np.fromfile(feed / "frames.u8", dtype=np.uint8).reshape(6, 240, 320, 3)
    assert frames.std() > 0
    config = json.loads((feed / "config.json").read_text())
    assert config["width"] == 320 and "channels" not in config
    imu = np.fromfile(feed / "imu.f64").reshape(-1, 7)
    np.testing.assert_allclose(imu, episode.delivered_imu())
    gt = np.fromfile(feed / "gt.f64").reshape(6, 7)
    np.testing.assert_allclose(gt, episode.ground_truth_relative())
    # a perfect estimate scores zero; a frame missing from the files uses origin, identity rotation and zero twist
    tum = tmp_path / "perfect.tum"
    np.savetxt(tum, np.column_stack([episode.frame_times, gt]), fmt="%.12g")
    np.savetxt(tmp_path / "twist.txt", np.column_stack([episode.frame_times, episode.ground_truth_twist()]), fmt="%.12g")
    metrics = score(tmp_path / "set" / f"{episode.name}.zip", tum)
    assert metrics.ate_rmse < 1e-9 and metrics.velocity_rmse < 1e-6 and metrics.filled_frames == 0
    assert "ATE 0.000" in capsys.readouterr().out
    rows = np.column_stack([episode.frame_times, gt])[1:]  # first frame missing
    np.savetxt(tum, rows, fmt="%.12g")
    assert np.isnan(load_estimates(tum, episode.frame_times)[0, :7]).all()
    assert score(tmp_path / "set" / f"{episode.name}.zip", tum, log=lambda *a: None).filled_frames == 1
    assert cli(["score", str(tmp_path / "set" / f"{episode.name}.zip"), str(tum)]) == 0


@pytest.mark.skipif(shutil.which("cmake") is None or (shutil.which("cc") is None and shutil.which("gcc") is None and shutil.which("clang") is None),
                    reason="needs cmake and a host C/C++ compiler")
def test_native_host_debug_build(tmp_path):
    """cpp_minimal configured for the host builds `debug`, which replays a feed and writes identity poses."""
    source = EXAMPLES / "native" / "cpp_minimal"
    build = tmp_path / "build-host"
    configure = subprocess.run(["cmake", "-S", str(source), "-B", str(build)], capture_output=True, text=True)
    assert configure.returncode == 0, configure.stderr
    assert "debug" in configure.stdout
    compile_ = subprocess.run(["cmake", "--build", str(build)], capture_output=True, text=True)
    assert compile_.returncode == 0, compile_.stderr
    episode = synthetic_episode(n_frames=5, width=640, height=480)
    write_episode(episode, tmp_path / "set" / f"{episode.name}.zip")
    feed = write_feed(episode, tmp_path / "feed")
    tum = tmp_path / "estimate.tum"
    run = subprocess.run([str(build / "debug"), str(feed), str(tum)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "5/5 frames ok" in run.stdout
    estimates = load_estimates(tum, episode.frame_times)
    np.testing.assert_allclose(estimates, np.tile([0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0], (5, 1)))
    metrics = score(tmp_path / "set" / f"{episode.name}.zip", tum, log=lambda *a: None)
    assert metrics.filled_frames == 0


def test_debug_never_asks_for_an_episode_beyond_max_episodes():
    """A Dataset's next episode may be a download (a conta reference): a run that stops early must not ask for it."""
    episode = synthetic_episode(n_frames=4, width=640, height=480)
    episode.meta["dataset"] = "adhoc"

    def dataset():
        yield episode
        pytest.fail("asked for an episode beyond max_episodes")

    [(read, _, _)] = direct.run(direct.PythonSolution(EXAMPLES / "python" / "minimal"), dataset(), max_episodes=1, log=lambda *a: None)
    assert read is episode
