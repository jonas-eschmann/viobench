"""`viobench evaluate`: run a solution .wasm over a Dataset (any iterable of Episodes), meter it, score it, write a report.

The Dataset may be episodes on disk or in the conta cache (`datasets.sources.episodes`: paths and conta hashes) or fresh
episodes from the simulator (`datasets.generate.simulation.simulate`); the runner does not care, and reads frames one at
a time either way.
"""
import json
import platform
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np

from .. import abi
from .engine import WASMTIME_VERSION, Engine
from .metrics import evaluate_trajectory
from .sandbox import STDERR_FILE, Limits, Sandbox, SandboxError, SandboxTrap, tail
from .score import write_estimates


@dataclass
class EpisodeReport:
    hash: str | None  # the sha1 of the episode's archive, its identity (Episode.hash); None for one without an archive
    dataset: str
    frames: int
    frames_attempted: int  # includes a failing step, excludes frames held after the failure
    duration_s: float  # first to last frame; episodes of one set share it, sets may differ
    completed: bool  # all frames were fed; otherwise the episode failed and the remaining frames repeat the last pose
    failure: str | None  # why it ended early: init failed, a step failed or returned an unusable state, a trap
    fuel_init: int
    fuel_per_frame_max: int
    fuel_per_frame_median: float
    fuel_per_frame_p95: float
    fuel_per_frame_p99: float
    fuel_per_frame_mean: float
    fuel_total: int
    wall_init_s: float
    wall_per_frame_median_ms: float
    wall_per_frame_p95_ms: float
    metrics: dict
    stderr_tail: str = ""

    def as_dict(self):
        return asdict(self)


def fuel_statistics(fuel):
    """Statistics over actual attempted frames; no samples are added for unprocessed frames or init failures."""
    median, p95, p99 = np.percentile(fuel, [50, 95, 99]) if fuel.size else (0.0, 0.0, 0.0)
    return {
        "frames_attempted": int(fuel.size),
        "fuel_per_frame_max": int(fuel.max(initial=0)),
        "fuel_per_frame_mean": float(fuel.mean()) if fuel.size else 0.0,
        "fuel_per_frame_median": float(median),
        "fuel_per_frame_p95": float(p95),
        "fuel_per_frame_p99": float(p99),
        "fuel_total": int(fuel.sum()),
    }


def run_episode(wasm_path, episode, engine, limits, log_dir):
    """Returns (EpisodeReport, estimates (N, 13) with NaN rows after failure, fuel for attempted frames)."""
    estimates = np.full((episode.num_frames, abi.STATE_VALUES), np.nan)
    fuel = np.zeros(episode.num_frames)
    wall = np.zeros(episode.num_frames)
    failure = None
    completed = False
    fuel_init = 0
    wall_init = 0.0
    fed = 0
    try:
        sandbox = Sandbox(wasm_path, engine, limits=limits, log_dir=log_dir)
        config, frames = episode.adapt(sandbox.resolution)
        sandbox.init(config)
        fuel_init, wall_init = sandbox.init_fuel, sandbox.init_wall_seconds
        for index, (t, frame, imu) in enumerate(episode.steps(frames)):
            fed = index + 1
            try:
                result = sandbox.step(t, frame, imu)
            except SandboxError as error:
                fuel[index], wall[index] = error.fuel, error.wall_seconds
                raise
            fuel[index], wall[index] = result.fuel, result.wall_seconds
            estimates[index] = np.concatenate([result.pose, result.twist])
        completed = True
    except SandboxTrap as trap:
        failure = f"trap {trap.trap_code}: {trap.trap_message}"
    except SandboxError as error:
        failure = str(error).splitlines()[0]
    metrics = evaluate_trajectory(episode.ground_truth_state(), estimates, episode.frame_times)
    per_frame_fuel = fuel[:fed]
    per_frame_wall = wall[:max(fed, 1)]
    return EpisodeReport(
        hash=episode.hash,
        dataset=episode.dataset,
        frames=episode.num_frames,
        duration_s=float(episode.duration),
        completed=completed,
        failure=failure,
        fuel_init=int(fuel_init),
        **fuel_statistics(per_frame_fuel),
        wall_init_s=float(wall_init),
        wall_per_frame_median_ms=float(np.median(per_frame_wall) * 1e3),
        wall_per_frame_p95_ms=float(np.percentile(per_frame_wall, 95) * 1e3),
        metrics=metrics.as_dict(),
        stderr_tail=tail(Path(log_dir) / STDERR_FILE, 20),  # also when the Sandbox could not even be constructed
    ), estimates, per_frame_fuel


def summarize(reports, frame_fuels):
    """Per-dataset and overall fuel distributions pooled across attempted frames; accuracy uses episode medians."""
    entries = list(zip(reports, frame_fuels, strict=True))

    def summary(entries):
        group, fuel = zip(*entries)
        completed = [r for r in group if r.completed]
        return {
            "episodes": len(group),
            "completed": len(completed),
            "duration_s_median": float(np.median([r.duration_s for r in group])),
            "rpe_translation_rmse_median": float(np.nanmedian([r.metrics["rpe_translation_rmse"] for r in group])),  # the headline
            "rpe_rotation_rmse_median": float(np.nanmedian([r.metrics["rpe_rotation_rmse"] for r in group])),
            "ate_rmse_median": float(np.median([r.metrics["ate_rmse"] for r in group])),
            "rotation_rmse_median": float(np.median([r.metrics["rotation_rmse"] for r in group])),
            "velocity_rmse_median": float(np.median([r.metrics["velocity_rmse"] for r in group])),
            "angular_velocity_rmse_median": float(np.median([r.metrics["angular_velocity_rmse"] for r in group])),
            **fuel_statistics(np.concatenate(fuel)),
            "wall_per_frame_median_ms": float(np.median([r.wall_per_frame_median_ms for r in group])),
        }

    datasets = sorted({r.dataset for r in reports})
    return {"overall": summary(entries), "datasets": {name: summary([entry for entry in entries if entry[0].dataset == name]) for name in datasets}}


def as_dataset(dataset):
    """Anything a user may pass as a dataset -> an iterable of Episodes: sources (paths, conta references) go through
    `datasets.sources`, anything else is taken as the Dataset itself."""
    from ..datasets.sources import episodes

    source_types = (str, Path, dict)
    if isinstance(dataset, source_types):
        return episodes([dataset])
    dataset = list(dataset) if not hasattr(dataset, "__next__") else dataset
    if isinstance(dataset, list) and dataset and all(isinstance(item, source_types) for item in dataset):
        return episodes(dataset)
    return dataset


def evaluate(wasm_path, dataset, output_dir=None, limits=None, engine=None, max_episodes=None, log=print):
    """Evaluate a Dataset and write report.json; an explicit output_dir also enables per-episode videos."""
    wasm_path = Path(wasm_path)
    engine = engine or Engine()
    limits = limits or Limits()
    save_video = output_dir is not None
    output_dir = Path(output_dir) if output_dir else Path.cwd() / f"report-{wasm_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    frame_fuels = []
    hashes = set()
    started = time.perf_counter()
    for episode in islice(as_dataset(dataset), max_episodes):  # never asks for one more: it may be a download
        if episode.hash in hashes:  # one episode (copies of an archive are one): it would count twice, in one log directory
            raise ValueError(f"episode {episode.hash} given twice")
        if episode.hash is not None:
            hashes.add(episode.hash)
        log_dir = output_dir / "logs" / episode.key
        report, estimates, fuel = run_episode(wasm_path, episode, engine, limits, log_dir)
        reports.append(report)
        frame_fuels.append(fuel)
        write_estimates(log_dir, episode.frame_times, estimates)
        np.save(log_dir / "fuel.npy", fuel, allow_pickle=False)  # private samples for exact aggregation of episode jobs
        m = report.metrics
        log(f"{episode.dataset}/{episode.key}: {'ok' if report.completed else 'FAILED'} RPE {m['rpe_translation_rmse']:.3f} m/1 s  ATE {m['ate_rmse']:.3f} m  "
            f"rot {np.degrees(m['rotation_rmse']):.1f} deg  vel {m['velocity_rmse']:.3f} m/s  held {m['held']}  fuel/frame max {report.fuel_per_frame_max:,.0f} (p95 {report.fuel_per_frame_p95:,.0f})  "
            f"wall/frame {report.wall_per_frame_median_ms:.1f} ms" + (f"  [{report.failure}]" if report.failure else ""))
        if save_video:
            from .video import write_video

            log(f"video: {write_video(episode, estimates, log_dir / 'trajectory.mp4')}")
    if not reports:
        raise ValueError("the dataset yielded no episodes")
    result = {
        "solution": str(wasm_path),
        "solution_bytes": wasm_path.stat().st_size,
        "wasmtime": WASMTIME_VERSION,
        "host": platform.platform(),
        "limits": asdict(limits),
        "elapsed_s": time.perf_counter() - started,
        "summary": summarize(reports, frame_fuels),
        "episodes": [report.as_dict() for report in reports],
    }
    (output_dir / "report.json").write_text(json.dumps(result, indent=1, default=lambda v: float(v) if isinstance(v, np.floating) else str(v)))
    log(f"report: {output_dir / 'report.json'}")
    return result
