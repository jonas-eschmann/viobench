"""`viobench score`: the metrics of an estimate against an episode. An estimate is two files side by side: estimate.tum
(TUM format: t x y z qx qy qz qw, loadable with evo) and twist.txt (t vx vy vz wx wy wz); frames without a usable
state (NaN rows, gaps in either file) repeat the previous pose, like the frames after a failure in `viobench evaluate`."""
from pathlib import Path

import numpy as np

from .. import abi
from .metrics import evaluate_trajectory

POSE_FILE = "estimate.tum"
TWIST_FILE = "twist.txt"  # always next to the pose file


def write_estimates(directory, frame_times, states):
    """Writes (N, 13) states as estimate.tum and twist.txt into directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(directory / POSE_FILE, np.column_stack([frame_times, states[:, :abi.POSE_VALUES]]), fmt="%.9g",
               header="t x y z qx qy qz qw (camera k in camera 0 frame)", comments="# ")
    np.savetxt(directory / TWIST_FILE, np.column_stack([frame_times, states[:, abi.POSE_VALUES:]]), fmt="%.9g",
               header="t vx vy vz wx wy wz (velocity and angular velocity of camera k in camera k axes)", comments="# ")


def load_rows(path, frame_times, values, tolerance=1e-3):
    """(N, values) rows of a "t v1 v2 ..." file aligned to the frame times (by nearest timestamp; NaN where none is within tolerance)."""
    rows = np.loadtxt(path, comments="#", ndmin=2)
    aligned = np.full((len(frame_times), values), np.nan)
    if rows.size == 0:
        return aligned
    if rows.shape[1] != values + 1:
        raise ValueError(f"{path}: expected {values + 1} columns (t and {values} values), found {rows.shape[1]}")
    order = np.argsort(rows[:, 0])
    times, data = rows[order, 0], rows[order, 1:]
    index = np.clip(np.searchsorted(times, frame_times), 0, len(times) - 1)
    before = np.clip(index - 1, 0, len(times) - 1)
    nearest = np.where(np.abs(times[before] - frame_times) < np.abs(times[index] - frame_times), before, index)
    hit = np.abs(times[nearest] - frame_times) <= tolerance
    aligned[hit] = data[nearest[hit]]
    return aligned


def load_estimates(pose_path, frame_times):
    """(N, 13) states from a pose file and the twist.txt next to it."""
    pose_path = Path(pose_path)
    return np.concatenate([load_rows(pose_path, frame_times, abi.POSE_VALUES),
                           load_rows(pose_path.parent / TWIST_FILE, frame_times, abi.TWIST_VALUES)], axis=1)


def score(episode_dir, pose_path, log=print):
    from ..datasets.storage import read_episode  # cv2, yaml: only where an episode is read

    episode = read_episode(episode_dir, frames="none")
    estimates = load_estimates(pose_path, episode.frame_times)
    metrics = evaluate_trajectory(episode.ground_truth_state(), estimates, episode.frame_times)
    log(f"{episode.dataset}/{episode.name}: RPE {metrics.rpe_translation_rmse:.3f} m/1 s  ATE {metrics.ate_rmse:.3f} m  "
        f"rot {np.degrees(metrics.rotation_rmse):.1f} deg  vel {metrics.velocity_rmse:.3f} m/s  "
        f"ang vel {np.degrees(metrics.angular_velocity_rmse):.1f} deg/s  held {metrics.held}  sim3 scale {metrics.sim3_scale:.3f}")
    return metrics
