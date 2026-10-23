"""Run a solution .wasm over episodes on disk, meter it, score it, and write a report."""
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import abi
from .data.euroc_io import list_episodes, read_episode
from .engine import WASMTIME_VERSION, Runtime
from .metrics import evaluate_trajectory
from .solution import Limits, Solution, SolutionError, SolutionTrap

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


def run_episode(wasm_path, episode, runtime, limits, log_dir, dataset_name="", dev_dir=None, progress=None):
    estimates = null_pose_estimates(episode.num_frames)
    fuel = np.zeros(episode.num_frames)
    wall = np.zeros(episode.num_frames)
    statuses = {}
    failure = None
    completed = False
    fuel_init = 0
    wall_init = 0.0
    solution = None
    try:
        solution = Solution(wasm_path, runtime, limits=limits, log_dir=log_dir, dev_dir=dev_dir)
        config, frames = episode.as_channels(solution.channels)
        solution.init(config)
        fuel_init, wall_init = solution.init_fuel, solution.init_wall_seconds
        consecutive = 0
        for index, (t, frame, imu) in enumerate(episode.steps(frames)):
            result = solution.step(t, frame, imu)
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
    except SolutionTrap as trap:
        failure = f"trap {trap.trap_code}: {trap.trap_message}"
    except SolutionError as error:
        failure = str(error).splitlines()[0]
    fed = int(np.count_nonzero(fuel)) if not completed else episode.num_frames
    fed = max(fed, 1)
    metrics = evaluate_trajectory(episode.ground_truth_relative(), estimates, episode.frame_times)
    per_frame_fuel = fuel[:fed] if fed else np.zeros(1)
    per_frame_wall = wall[:fed] if fed else np.zeros(1)
    return EpisodeReport(
        name=episode.name,
        dataset=dataset_name,
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
        stderr_tail=solution.stderr_tail(20) if solution is not None else "",
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


def evaluate(wasm_path, dataset_roots, output_dir=None, limits=None, runtime=None, dev_dir=None, max_episodes=None,
             save_estimates=True, progress=None, log=print):
    """dataset_roots: episode directories or directories of episodes. Returns the report dict (also written as report.json)."""
    wasm_path = Path(wasm_path)
    runtime = runtime or Runtime()
    limits = limits or Limits()
    output_dir = Path(output_dir) if output_dir else Path.cwd() / f"report-{wasm_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    started = time.perf_counter()
    for root in dataset_roots:
        root = Path(root)
        episodes = list_episodes(root)
        if not episodes:
            raise FileNotFoundError(f"no episodes under {root}")
        if max_episodes:
            episodes = episodes[:max_episodes]
        dataset_name = root.name if (root / "episode.json").is_file() is False else root.parent.name
        for episode_dir in episodes:
            episode = read_episode(episode_dir)
            log_dir = output_dir / "logs" / dataset_name / episode.name
            report, estimates = run_episode(wasm_path, episode, runtime, limits, log_dir, dataset_name=dataset_name, dev_dir=dev_dir, progress=progress)
            reports.append(report)
            if save_estimates:
                np.savetxt(log_dir / "estimate.tum", np.column_stack([episode.frame_times, estimates]), fmt="%.9g",
                           header="t x y z qx qy qz qw (camera k in camera 0 frame)", comments="# ")
            m = report.metrics
            log(f"{dataset_name}/{episode.name}: {'ok' if report.completed else 'FAILED'} ATE {m['ate_rmse']:.3f} m  RPE {m['rpe_translation_rmse']:.3f} m/s  "
                f"rot {np.degrees(m['rotation_rmse']):.1f} deg  coverage {m['coverage']:.2f}  fuel/frame {report.fuel_per_frame_median:,.0f} (p95 {report.fuel_per_frame_p95:,.0f})  "
                f"wall/frame {report.wall_per_frame_median_ms:.1f} ms" + (f"  [{report.failure}]" if report.failure else ""))
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
