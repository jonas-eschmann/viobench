"""Trajectory error metrics for poses given relative to the first camera frame.

Poses are (N, 7) arrays [x, y, z, qx, qy, qz, qw] (scalar-last quaternions), one row per frame; missing estimates
are rows of NaN. Because the task fixes the reference frame (frame 0), ATE is computed *without* alignment; a
Sim(3)-aligned ATE is reported as a diagnostic that separates scale/gravity errors from structural ones.
"""
from dataclasses import asdict, dataclass

import numpy as np

from ..geometry import invert, poses_to_matrices, rotation_angles

MAX_ERROR_ROTATION = np.pi


def umeyama_sim3(source, target):
    """Similarity (s, R, t) with target ≈ s R source + t in the least-squares sense (Umeyama 1991)."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    mean_source, mean_target = source.mean(0), target.mean(0)
    centred_source, centred_target = source - mean_source, target - mean_target
    covariance = centred_target.T @ centred_source / len(source)
    U, D, Vt = np.linalg.svd(covariance)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    variance_source = (centred_source ** 2).sum() / len(source)
    scale = np.trace(np.diag(D) @ S) / variance_source if variance_source > 0 else 1.0
    t = mean_target - scale * R @ mean_source
    return scale, R, t


@dataclass
class TrajectoryMetrics:
    frames: int
    estimated: int  # frames with a finite estimate
    coverage: float  # estimated / frames
    ate_rmse: float  # m, no alignment (missing frames count with max_position_error)
    ate_rmse_valid: float  # m, over estimated frames only
    rotation_rmse: float  # rad, over estimated frames
    rotation_max: float  # rad
    rpe_translation_rmse: float  # m per delta_seconds, over estimated frame pairs
    rpe_rotation_rmse: float  # rad per delta_seconds
    rpe_pairs: int
    sim3_scale: float  # diagnostic: scale factor that best aligns the estimate to ground truth
    sim3_ate_rmse: float  # m, after Sim(3) alignment of the estimated frames
    distance_travelled: float  # m, ground truth
    final_drift: float  # m, position error at the last estimated frame
    drift_per_metre: float  # final_drift / distance_travelled

    def as_dict(self):
        return {key: (float(value) if isinstance(value, (np.floating, float)) else value) for key, value in asdict(self).items()}


def evaluate_trajectory(ground_truth, estimate, timestamps, delta_seconds=1.0, max_position_error=None):
    """ground_truth, estimate: (N, 7) poses relative to frame 0; estimate rows may be NaN.

    max_position_error: substituted for missing frames in ate_rmse (default: the ground truth's distance travelled,
    the error of an estimator that stays at the origin would be of that order).
    """
    ground_truth = np.asarray(ground_truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    n = len(ground_truth)
    assert estimate.shape == (n, 7) and timestamps.shape == (n,)
    valid = np.all(np.isfinite(estimate), axis=1)
    T_gt = poses_to_matrices(ground_truth)
    steps = np.linalg.norm(np.diff(ground_truth[:, :3], axis=0), axis=1)
    distance = float(steps.sum())
    if max_position_error is None:
        max_position_error = max(distance, 1.0)

    position_error = np.full(n, max_position_error)
    rotation_error = np.full(n, MAX_ERROR_ROTATION)
    if valid.any():
        T_est = poses_to_matrices(np.where(valid[:, None], estimate, [0, 0, 0, 0, 0, 0, 1]))
        delta = invert(T_gt) @ T_est  # ground truth -> estimate
        position_error[valid] = np.linalg.norm(delta[valid, :3, 3], axis=1)
        rotation_error[valid] = rotation_angles(delta[valid, :3, :3])
    ate_rmse = float(np.sqrt(np.mean(position_error ** 2)))
    ate_rmse_valid = float(np.sqrt(np.mean(position_error[valid] ** 2))) if valid.any() else float("nan")
    rotation_rmse = float(np.sqrt(np.mean(rotation_error[valid] ** 2))) if valid.any() else float("nan")
    rotation_max = float(rotation_error[valid].max()) if valid.any() else float("nan")

    # relative pose error over pairs (i, j) with t_j - t_i ≈ delta_seconds and both frames estimated
    rpe_translation, rpe_rotation = [], []
    if valid.any():
        for i in range(n):
            j = i + int(np.searchsorted(timestamps[i:], timestamps[i] + delta_seconds))
            if j >= n or not (valid[i] and valid[j]):
                continue
            relative_gt = invert(T_gt[i]) @ T_gt[j]
            relative_est = invert(T_est[i]) @ T_est[j]
            error = invert(relative_gt) @ relative_est
            rpe_translation.append(np.linalg.norm(error[:3, 3]))
            rpe_rotation.append(rotation_angles(error[:3, :3]))
    rpe_translation = np.asarray(rpe_translation)
    rpe_rotation = np.asarray(rpe_rotation)

    sim3_scale, sim3_ate = float("nan"), float("nan")
    if valid.sum() >= 3:
        scale, R, t = umeyama_sim3(estimate[valid, :3], ground_truth[valid, :3])
        aligned = (scale * (R @ estimate[valid, :3].T)).T + t
        sim3_scale = float(scale)
        sim3_ate = float(np.sqrt(np.mean(np.sum((aligned - ground_truth[valid, :3]) ** 2, axis=1))))

    last_valid = np.flatnonzero(valid)
    final_drift = float(position_error[last_valid[-1]]) if len(last_valid) else max_position_error
    return TrajectoryMetrics(
        frames=n,
        estimated=int(valid.sum()),
        coverage=float(valid.mean()),
        ate_rmse=ate_rmse,
        ate_rmse_valid=ate_rmse_valid,
        rotation_rmse=rotation_rmse,
        rotation_max=rotation_max,
        rpe_translation_rmse=float(np.sqrt(np.mean(rpe_translation ** 2))) if len(rpe_translation) else float("nan"),
        rpe_rotation_rmse=float(np.sqrt(np.mean(rpe_rotation ** 2))) if len(rpe_rotation) else float("nan"),
        rpe_pairs=int(len(rpe_translation)),
        sim3_scale=sim3_scale,
        sim3_ate_rmse=sim3_ate,
        distance_travelled=distance,
        final_drift=final_drift,
        drift_per_metre=final_drift / distance if distance > 0 else float("nan"),
    )
