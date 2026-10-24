"""`relpose evaluate`: run a solution .wasm over a Dataset (any iterable of Episodes), meter it, score it, write a report.

The Dataset may be episode directories on disk (`datasets.storage.episodes`) or fresh episodes from the simulator
(`datasets.generate.simulation.simulate`); the runner does not care, and reads frames one at a time either way.
"""
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import abi
from .engine import WASMTIME_VERSION, Engine
from .metrics import evaluate_trajectory
from .sandbox import Limits, Sandbox, SandboxError, SandboxTrap

MAX_CONSECUTIVE_FAILURES = 10


@dataclass
class EpisodeReport:
    name: str
    dataset: str
    frames: int
    completed: bool  # all frames were fed (no trap / abandonment)
    failure: str | None  # trap or protocol error text
    statuses: dict  # status name -> count
    fuel_init: int
    fuel_per_frame_median: float
    fuel_per_frame_p95: float
    fuel_per_frame_mean: float
    fuel_total: int
    wall_init_s: float
    wall_per_frame_median_ms: float
    wall_per_frame_p95_ms: float
    metrics: dict
    stderr_tail: str = ""

    def as_dict(self):
        return asdict(self)


def null_pose_estimates(n):
    return np.full((n, abi.POSE_VALUES), np.nan)


def run_episode(wasm_path, episode, engine, limits, log_dir, dev_dir=None, progress=None):
    """One episode through a fresh Sandbox; returns (EpisodeReport, estimates (N, 7) with NaN rows for missing frames)."""
    estimates = null_pose_estimates(episode.num_frames)
    fuel = np.zeros(episode.num_frames)
    wall = np.zeros(episode.num_frames)
    statuses = {}
    failure = None
    completed = False
    fuel_init = 0
    wall_init = 0.0
    sandbox = None
    try:
        sandbox = Sandbox(wasm_path, engine, limits=limits, log_dir=log_dir, dev_dir=dev_dir)
        config, frames = episode.adapt(sandbox.channels, sandbox.resolution)
        sandbox.init(config)
        fuel_init, wall_init = sandbox.init_fuel, sandbox.init_wall_seconds
        consecutive = 0
        for index, (t, frame, imu) in enumerate(episode.steps(frames)):
            result = sandbox.step(t, frame, imu)
            name = abi.status_name(result.status)
            statuses[name] = statuses.get(name, 0) + 1
            fuel[index], wall[index] = result.fuel, result.wall_seconds
            if result.ok:
                estimates[index] = result.pose
            consecutive = consecutive + 1 if result.status < 0 else 0
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                failure = f"abandoned after {consecutive} consecutive failing steps (last: {name})"
                break
            if progress:
                progress(index, episode.num_frames, result)
        else:
            completed = True
    except SandboxTrap as trap:
        failure = f"trap {trap.trap_code}: {trap.trap_message}"
    except SandboxError as error:
        failure = str(error).splitlines()[0]
    fed = episode.num_frames if completed else max(int(np.count_nonzero(fuel)), 1)
    metrics = evaluate_trajectory(episode.ground_truth_relative(), estimates, episode.frame_times)
    per_frame_fuel = fuel[:fed]
    per_frame_wall = wall[:fed]
    return EpisodeReport(
        name=episode.name,
        dataset=episode.dataset,
        frames=episode.num_frames,
        completed=completed,
        failure=failure,
        statuses=statuses,
        fuel_init=int(fuel_init),
        fuel_per_frame_median=float(np.median(per_frame_fuel)),
        fuel_per_frame_p95=float(np.percentile(per_frame_fuel, 95)),
        fuel_per_frame_mean=float(per_frame_fuel.mean()),
        fuel_total=int(fuel.sum()),
        wall_init_s=float(wall_init),
        wall_per_frame_median_ms=float(np.median(per_frame_wall) * 1e3),
        wall_per_frame_p95_ms=float(np.percentile(per_frame_wall, 95) * 1e3),
        metrics=metrics.as_dict(),
        stderr_tail=sandbox.stderr_tail(20) if sandbox is not None else "",
    ), estimates


def summarize(reports):
    """Per-dataset medians of the headline numbers, plus overall."""
    def summary(group):
        completed = [r for r in group if r.completed]
        return {
            "episodes": len(group),
            "completed": len(completed),
            "ate_rmse_median": float(np.median([r.metrics["ate_rmse"] for r in group])) if group else float("nan"),
            "rpe_translation_rmse_median": float(np.nanmedian([r.metrics["rpe_translation_rmse"] for r in group])) if group else float("nan"),
            "rotation_rmse_median": float(np.nanmedian([r.metrics["rotation_rmse"] for r in group])) if group else float("nan"),
            "coverage_mean": float(np.mean([r.metrics["coverage"] for r in group])) if group else float("nan"),
            "fuel_per_frame_median": float(np.median([r.fuel_per_frame_median for r in group])) if group else float("nan"),
            "fuel_per_frame_p95": float(np.median([r.fuel_per_frame_p95 for r in group])) if group else float("nan"),
            "wall_per_frame_median_ms": float(np.median([r.wall_per_frame_median_ms for r in group])) if group else float("nan"),
        }

    datasets = sorted({r.dataset for r in reports})
    return {"overall": summary(reports), "datasets": {name: summary([r for r in reports if r.dataset == name]) for name in datasets}}


def as_dataset(dataset):
    """Anything a user may pass as a dataset -> an iterable of Episodes: paths go through the file-system provider."""
    from ..datasets.storage import episodes

    if isinstance(dataset, (str, Path)):
        return episodes([dataset])
    dataset = list(dataset) if not hasattr(dataset, "__next__") else dataset
    if isinstance(dataset, list) and dataset and all(isinstance(item, (str, Path)) for item in dataset):
        return episodes(dataset)
    return dataset


def evaluate(wasm_path, dataset, output_dir=None, limits=None, engine=None, dev_dir=None, max_episodes=None,
             save_estimates=True, progress=None, log=print):
    """dataset: episode directories / dataset roots, or any iterable of Episodes. Returns the report dict (also written as report.json)."""
    wasm_path = Path(wasm_path)
    engine = engine or Engine()
    limits = limits or Limits()
    output_dir = Path(output_dir) if output_dir else Path.cwd() / f"report-{wasm_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    started = time.perf_counter()
    for count, episode in enumerate(as_dataset(dataset)):
        if max_episodes is not None and count >= max_episodes:
            break
        log_dir = output_dir / "logs" / episode.dataset / episode.name
        report, estimates = run_episode(wasm_path, episode, engine, limits, log_dir, dev_dir=dev_dir, progress=progress)
        reports.append(report)
        if save_estimates:
            np.savetxt(log_dir / "estimate.tum", np.column_stack([episode.frame_times, estimates]), fmt="%.9g",
                       header="t x y z qx qy qz qw (camera k in camera 0 frame)", comments="# ")
        m = report.metrics
        log(f"{episode.dataset}/{episode.name}: {'ok' if report.completed else 'FAILED'} ATE {m['ate_rmse']:.3f} m  RPE {m['rpe_translation_rmse']:.3f} m/s  "
            f"rot {np.degrees(m['rotation_rmse']):.1f} deg  coverage {m['coverage']:.2f}  fuel/frame {report.fuel_per_frame_median:,.0f} (p95 {report.fuel_per_frame_p95:,.0f})  "
            f"wall/frame {report.wall_per_frame_median_ms:.1f} ms" + (f"  [{report.failure}]" if report.failure else ""))
    if not reports:
        raise ValueError("the dataset yielded no episodes")
    result = {
        "solution": str(wasm_path),
        "solution_bytes": wasm_path.stat().st_size,
        "wasmtime": WASMTIME_VERSION,
        "host": platform.platform(),
        "limits": asdict(limits),
        "elapsed_s": time.perf_counter() - started,
        "summary": summarize(reports),
        "episodes": [report.as_dict() for report in reports],
    }
    (output_dir / "report.json").write_text(json.dumps(result, indent=1, default=lambda v: float(v) if isinstance(v, np.floating) else str(v)))
    log(f"report: {output_dir / 'report.json'}")
    return result
