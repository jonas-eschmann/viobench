"""The out-of-sandbox debugging facilities: `relpose debug` (Python, in-process), `relpose feed` + the native host
`debug` executable, `relpose score`."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from relpose.tools.cli import main as cli
from relpose.tools.datasets.feed import write_feed
from relpose.tools.datasets.storage import write_episode
from relpose.tools.runner import direct
from relpose.tools.runner.score import load_tum, score
from tests.test_datasets import synthetic_episode

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "relpose" / "solution" / "examples"


def test_direct_run_matches_the_sandbox(engine, runtime_path, tmp_path):
    """dead_reckoning on the host and inside the runtime give the same poses (same code, same inputs, host numpy)."""
    from relpose.solution.runtime.python.bundler.bundle import bundle
    from relpose.tools.runner.run import evaluate

    episode = synthetic_episode(n_frames=30, width=320, height=240)  # 1.5 s: dead reckoning calibrates on the first 0.5 s, then estimates
    dataset = tmp_path / "set"
    write_episode(episode, dataset / episode.name)
    solution = direct.PythonSolution(EXAMPLES / "python" / "dead_reckoning")
    assert solution.resolution == (320, 240)
    [(_, estimates, metrics)] = direct.run(solution, [dataset], output_dir=tmp_path / "direct", log=lambda *a: None)
    assert estimates.shape == (30, 7) and np.isfinite(estimates[-1]).all() and 0 < metrics.coverage < 1
    written = np.loadtxt(tmp_path / "direct" / "set" / episode.name / "estimate.tum")
    np.testing.assert_allclose(written[:, 1:], estimates, atol=1e-8)

    wasm = tmp_path / "dr.wasm"
    bundle(EXAMPLES / "python" / "dead_reckoning", wasm, runtime=runtime_path)
    evaluate(wasm, [dataset], output_dir=tmp_path / "report", engine=engine, log=lambda *a: None)
    sandboxed = np.loadtxt(tmp_path / "report" / "logs" / "set" / episode.name / "estimate.tum")
    np.testing.assert_allclose(estimates, sandboxed[:, 1:], atol=1e-6)


def test_direct_manifest_and_host_solution_bundle(solution_dir, tmp_path):
    """solution.json picks entry/resolution; data files next to the code come through `solution_bundle`."""
    directory = solution_dir(
        "",
        {
            "solution.json": json.dumps({"entry": "estimator:Est", "resolution": "640x480"}),
            "estimator.py": (
                "import numpy as np\nimport solution_bundle as bundle\n"
                "class Est:\n"
                "    def __init__(self, config):\n"
                "        assert config['width'] == 640 and 'channels' not in config\n"
                "        self.offset = np.load(bundle.open('data/offset.npy'))\n"
                "        assert bundle.exists('data/offset.npy') and bundle.listdir('data') == ['offset.npy'] and bundle.read_text('note.txt') == 'hi'\n"
                "    def step(self, t, frame, imu):\n"
                "        assert frame.shape == (480, 640, 3)\n"
                "        return np.concatenate([self.offset, [0, 0, 0, 1]])\n"
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
    episode = synthetic_episode(n_frames=4, width=320, height=240)
    write_episode(episode, tmp_path / "set" / episode.name)
    assert cli(["debug", str(EXAMPLES / "python" / "minimal"), str(tmp_path / "set"), "-o", str(tmp_path / "out")]) == 0
    assert "ATE" in capsys.readouterr().out
    assert (tmp_path / "out" / "set" / episode.name / "estimate.tum").is_file()


def test_feed_and_score(tmp_path, capsys):
    episode = synthetic_episode(n_frames=6, width=640, height=480, lead_in=1.0)
    write_episode(episode, tmp_path / "set" / episode.name)
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
    # a perfect pose file scores zero; one with a missing frame has coverage < 1
    tum = tmp_path / "perfect.tum"
    np.savetxt(tum, np.column_stack([episode.frame_times, gt]), fmt="%.12g")
    metrics = score(tmp_path / "set" / episode.name, tum)
    assert metrics.ate_rmse < 1e-9 and metrics.coverage == 1.0
    assert "ATE 0.000" in capsys.readouterr().out
    rows = np.column_stack([episode.frame_times, gt])[1:]  # first frame missing
    np.savetxt(tum, rows, fmt="%.12g")
    assert np.isnan(load_tum(tum, episode.frame_times)[0]).all()
    assert cli(["score", str(tmp_path / "set" / episode.name), str(tum)]) == 0


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
    episode = synthetic_episode(n_frames=5, width=320, height=240)
    write_episode(episode, tmp_path / "set" / episode.name)
    feed = write_feed(episode, tmp_path / "feed")
    tum = tmp_path / "estimate.tum"
    run = subprocess.run([str(build / "debug"), str(feed), str(tum)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "5/5 frames estimated" in run.stdout
    estimates = load_tum(tum, episode.frame_times)
    np.testing.assert_allclose(estimates, np.tile([0, 0, 0, 0, 0, 0, 1.0], (5, 1)))
    metrics = score(tmp_path / "set" / episode.name, tum, log=lambda *a: None)
    assert metrics.coverage == 1.0


def test_python_example_footers_are_inert_when_imported():
    """Importing an example (as the runtime does) must not start the debug driver."""
    for name in ("minimal", "dead_reckoning", "klt_imu"):
        source = (EXAMPLES / "python" / name / "solution.py").read_text()
        assert 'if __name__ == "__main__":' in source and "main(Solution)" in source
    result = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import solution; print(solution.Solution.__name__)",
                             str(EXAMPLES / "python" / "minimal")], capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == "Solution"
