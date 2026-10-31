"""Replay videos: real encoding/decoding, lazy frames and CLI output behaviour."""
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import wasmtime

from relpose.tools.cli import main as cli
from relpose.tools.datasets.episode import FrameSequence
from relpose.tools.datasets.storage import write_episode
from relpose.tools.runner.video import write_video
from tests.test_datasets import synthetic_episode


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened(), path
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        return fps, frames
    finally:
        capture.release()


@pytest.mark.parametrize("channels,n_frames", [(1, 1), (3, 8)])
@pytest.mark.parametrize("with_estimates", [False, True])
def test_video_replays_lazy_frames_in_colour_and_at_source_rate(tmp_path, channels, n_frames, with_estimates):
    episode = synthetic_episode(n_frames=n_frames, channels=channels, rate=10.0)
    loaded = []

    def frame(index):
        loaded.append(index)
        colour = 90 if channels == 1 else [180 + index * 5, 90, 30]
        return np.full((24, 32) + (() if channels == 1 else (channels,)), colour, dtype=np.uint8)

    episode.frames = FrameSequence(frame, n_frames, episode.frames.shape[1:])
    estimates = episode.ground_truth_state()
    estimates[n_frames // 2:] = np.nan  # failed estimates, including a failure at the very first frame
    path = write_video(episode, estimates if with_estimates else None, tmp_path / "trajectory.mp4")
    assert loaded == list(range(n_frames)), "read each frame once, without materialising the episode"
    fps, frames = read_video(path)
    assert fps == pytest.approx(10.0) and len(frames) == n_frames
    for index, decoded in enumerate(frames):
        assert decoded.shape == (480, 860, 3)
        expected_bgr = [90, 90, 90] if channels == 1 else [30, 90, 180 + index * 5]
        # Maps and status must leave the entire camera feed unobstructed, including its edges.
        np.testing.assert_allclose(decoded[:, :640], np.broadcast_to(expected_bgr, (480, 640, 3)), atol=6)
    assert path.stat().st_size > 0


def test_video_keeps_previous_output_if_frame_loading_fails(tmp_path):
    episode = synthetic_episode(n_frames=3)
    estimates = episode.ground_truth_state()

    def missing(index):
        raise FileNotFoundError("missing camera frame")

    episode.frames = FrameSequence(missing, 3, (24, 32))
    path = tmp_path / "trajectory.mp4"
    path.write_bytes(b"previous video")
    with pytest.raises(FileNotFoundError, match="missing camera frame"):
        write_video(episode, estimates, path)
    assert path.read_bytes() == b"previous video"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("no_video", [False, True])
def test_simulate_cli_writes_video_in_each_episode_by_default(tmp_path, monkeypatch, no_video):
    episodes = [synthetic_episode(n_frames=3, rate=10.0, channels=3) for _ in range(2)]  # the provider yields exactly --episodes per scene
    for index, episode in enumerate(episodes):
        episode.name = f"sim-{index}"
        episode.config["camera"]["horizontal_fov_degrees"] = 60.0
    monkeypatch.setattr("relpose.tools.datasets.generate.simulation.dataset_from_arguments", lambda args: iter(episodes))
    if no_video:
        def unexpected(*args, **kwargs):
            pytest.fail("video rendered with --no-video")

        monkeypatch.setattr("relpose.tools.runner.video.write_video", unexpected)
    output = tmp_path / "dataset"
    args = ["simulate", "-o", str(output), "--episodes", "2"]
    if no_video:
        args.append("--no-video")
    assert cli(args) == 0
    assert sorted(path.name for path in output.iterdir()) == ["sim-0.zip", "sim-1.zip"]  # the video scratch directory is gone
    for episode in episodes:
        with zipfile.ZipFile(output / f"{episode.name}.zip") as archive:
            names = archive.namelist()
            assert f"{episode.name}/episode.json" in names
            assert sum(name.endswith(".png") for name in names) == episode.num_frames
            assert (f"{episode.name}/trajectory.mp4" in names) == (not no_video)
            if not no_video:
                video = tmp_path / f"{episode.name}.mp4"
                video.write_bytes(archive.read(f"{episode.name}/trajectory.mp4"))
                fps, frames = read_video(video)
                assert fps == pytest.approx(10.0) and len(frames) == episode.num_frames


# A native identity solution exercises the full evaluator without building the Python WASM runtime.
IDENTITY = """
(module
  (memory (export "memory") 16)
  (global $heap (mut i32) (i32.const 1024))
  (func (export "abi_version") (result i32) (i32.const 1))
  (func (export "alloc") (param $size i32) (result i32)
    (local $old i32)
    (local.set $old (global.get $heap))
    (global.set $heap (i32.add (global.get $heap) (local.get $size)))
    (local.get $old))
  (func (export "init") (param i32 i32) (result i32) (i32.const 0))
  (func (export "step") (param f64 i32 i32 i32) (param $out i32) (result i32)
    (memory.fill (local.get $out) (i32.const 0) (i32.const 104))
    (f64.store offset=48 (local.get $out) (f64.const 1))
    (i32.const 0)))
"""


@pytest.mark.parametrize("output", [False, True])
def test_evaluate_cli_only_writes_video_with_explicit_output(tmp_path, monkeypatch, output):
    episode = synthetic_episode(n_frames=4, width=320, height=240)
    write_episode(episode, tmp_path / "set" / f"{episode.name}.zip")
    wasm = tmp_path / "identity.wasm"
    wasm.write_bytes(wasmtime.wat2wasm(IDENTITY))
    monkeypatch.chdir(tmp_path)
    args = ["evaluate", str(wasm), str(tmp_path / "set")]
    if output:
        args += ["-o", str(tmp_path / "out")]
    assert cli(args) == 0
    report = tmp_path / ("out" if output else "report-identity")
    directory = report / "logs" / "set" / episode.name
    assert (directory / "estimate.tum").is_file() and (report / "report.json").is_file()
    assert (directory / "twist.txt").is_file()
    video = directory / "trajectory.mp4"
    assert video.is_file() == output
    if output:
        fps, frames = read_video(video)
        assert fps == pytest.approx(20.0) and len(frames) == episode.num_frames


def test_evaluate_failure_still_produces_complete_video(tmp_path):
    episode = synthetic_episode(n_frames=4, width=320, height=240)
    write_episode(episode, tmp_path / "set" / f"{episode.name}.zip")
    wasm = tmp_path / "broken.wasm"
    # No solution exports: evaluation fails before it can produce any poses.
    wasm.write_bytes(wasmtime.wat2wasm("(module)"))
    assert cli(["evaluate", str(wasm), str(tmp_path / "set"), "-o", str(tmp_path / "out")]) == 1
    video = tmp_path / "out" / "logs" / "set" / episode.name / "trajectory.mp4"
    _, frames = read_video(video)
    assert len(frames) == episode.num_frames


def test_debug_without_output_does_not_render_video(tmp_path, monkeypatch):
    episode = synthetic_episode(n_frames=4, width=320, height=240)
    write_episode(episode, tmp_path / "set" / f"{episode.name}.zip")
    solution = Path(__file__).resolve().parents[1] / "relpose/solution/examples/python/minimal"

    def unexpected(*args, **kwargs):
        pytest.fail("video rendered without -o")

    monkeypatch.setattr("relpose.tools.runner.video.write_video", unexpected)
    assert cli(["debug", str(solution), str(tmp_path / "set")]) == 0
