"""`relpose score`: the metrics of a pose file (TUM format: t x y z qx qy qz qw, NaN rows allowed) against an episode."""
import numpy as np

from .. import abi
from ..datasets.storage import read_episode
from .metrics import evaluate_trajectory


def load_tum(path, frame_times, tolerance=1e-3):
    """(N, 7) estimates aligned to the episode's frame times (by nearest timestamp; NaN where none is within tolerance)."""
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.size == 0:
        return np.full((len(frame_times), abi.POSE_VALUES), np.nan)
    if rows.shape[1] != 8:
        raise ValueError(f"{path}: expected 8 columns (t x y z qx qy qz qw), found {rows.shape[1]}")
    estimates = np.full((len(frame_times), abi.POSE_VALUES), np.nan)
    order = np.argsort(rows[:, 0])
    times, poses = rows[order, 0], rows[order, 1:8]
    index = np.clip(np.searchsorted(times, frame_times), 0, len(times) - 1)
    before = np.clip(index - 1, 0, len(times) - 1)
    nearest = np.where(np.abs(times[before] - frame_times) < np.abs(times[index] - frame_times), before, index)
    hit = np.abs(times[nearest] - frame_times) <= tolerance
    estimates[hit] = poses[nearest[hit]]
    return estimates


def score(episode_dir, tum_path, log=print):
    episode = read_episode(episode_dir, frames="none")
    estimates = load_tum(tum_path, episode.frame_times)
    metrics = evaluate_trajectory(episode.ground_truth_relative(), estimates, episode.frame_times)
    log(f"{episode.dataset}/{episode.name}: ATE {metrics.ate_rmse:.3f} m  RPE {metrics.rpe_translation_rmse:.3f} m/s  "
        f"rot {np.degrees(metrics.rotation_rmse):.1f} deg  coverage {metrics.coverage:.2f}  sim3 scale {metrics.sim3_scale:.3f}")
    return metrics
