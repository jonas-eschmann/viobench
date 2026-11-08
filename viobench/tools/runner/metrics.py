"""Trajectory error metrics for states given relative to the first camera frame.

States are (N, 13) arrays [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz], one row per frame: the pose T_c0_ck
(scalar-last quaternion), then the twist of camera k in camera-k axes. A row without a usable state (NaN/inf, as
recorded from the first failed step onward, or a zero quaternion) is replaced by the origin, identity rotation and
zero twist, so every metric runs over the complete trajectory.

The headline is position RMSE after one Sim(3) alignment of the complete estimated trajectory to ground truth.
Unaligned ATE, one-second relative pose error (RPE), rotation and twist errors are also reported as diagnostics.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .. import abi
from ..geometry import invert, poses_to_matrices, rotation_angles

IDENTITY_STATE = np.array([0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0])


def valid_states(states):
    """(N,) mask of the rows of (N, 13) states that are usable: finite, with a non-zero quaternion."""
    states = np.asarray(states, dtype=float)
    return np.all(np.isfinite(states), axis=1) & np.any(states[:, 3:7] != 0, axis=1)


def fill_missing_states(states):
    """Replace unusable rows with origin, identity rotation and zero twist without changing the raw estimates."""
    filled = np.array(states, dtype=float, copy=True)
    filled[~valid_states(filled)] = IDENTITY_STATE
    return filled


def umeyama_sim3(source, target):
    """Similarity (s, R, t) with target ≈ s R source + t in the least-squares sense (Umeyama 1991)."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    mean_source, mean_target = source.mean(0), target.mean(0)
    if np.all(source == source[0]):
        # Rotation and scale are undetermined for a constant prediction; align that point to the target centroid.
        return 1.0, np.eye(3), mean_target - source[0]
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
    filled_frames: int  # diagnostic: frames replaced by origin, identity rotation and zero twist
    rpe_translation_rmse: float  # m over windows of delta_seconds
    rpe_rotation_rmse: float  # rad over windows of delta_seconds
    rpe_pairs: int  # windows (frame pairs delta_seconds apart)
    ate_rmse: float  # m, no alignment
    rotation_rmse: float  # rad
    rotation_max: float  # rad
    velocity_rmse: float  # m/s, velocity of the camera centre in camera-k axes
    angular_velocity_rmse: float  # rad/s, in camera-k axes
    sim3_scale: float  # diagnostic: scale factor that best aligns the estimate to ground truth
    sim3_ate_rmse: float  # headline: m, after Sim(3) alignment of the complete trajectory
    distance_travelled: float  # m, ground truth
    final_drift: float  # m, position error at the last frame
    drift_per_metre: float  # final_drift / distance_travelled

    def as_dict(self):
        return {key: (float(value) if isinstance(value, (np.floating, float)) else value) for key, value in asdict(self).items()}


def evaluate_trajectory(ground_truth, estimate, timestamps, delta_seconds=1.0):
    """Score complete (N, 13) trajectories, filling unusable estimates with origin, identity rotation and zero twist."""
    ground_truth = np.asarray(ground_truth, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    n = len(ground_truth)
    assert ground_truth.shape == estimate.shape == (n, abi.STATE_VALUES) and timestamps.shape == (n,)
    filled_frames = int(n - valid_states(estimate).sum())
    estimate = fill_missing_states(estimate)
    T_gt = poses_to_matrices(ground_truth)
    T_est = poses_to_matrices(estimate)
    steps = np.linalg.norm(np.diff(ground_truth[:, :3], axis=0), axis=1)
    distance = float(steps.sum())

    delta = invert(T_gt) @ T_est  # ground truth -> estimate
    position_error = np.linalg.norm(delta[:, :3, 3], axis=1)
    rotation_error = rotation_angles(delta[:, :3, :3])
    ate_rmse = float(np.sqrt(np.mean(position_error ** 2)))
    rotation_rmse = float(np.sqrt(np.mean(rotation_error ** 2)))
    rotation_max = float(rotation_error.max())
    twist_error = estimate[:, abi.POSE_VALUES:] - ground_truth[:, abi.POSE_VALUES:]
    velocity_rmse = float(np.sqrt(np.mean(np.sum(twist_error[:, :3] ** 2, axis=1))))
    angular_velocity_rmse = float(np.sqrt(np.mean(np.sum(twist_error[:, 3:] ** 2, axis=1))))

    # relative pose error over pairs (i, j) with t_j - t_i = delta_seconds, to within half a frame interval (the
    # timestamps are floats: without the slack, rounding sometimes picks the frame after)
    slack = 0.5 * float(np.median(np.diff(timestamps))) if n > 1 else 0.0
    j = np.searchsorted(timestamps, timestamps + delta_seconds - slack)
    i = np.flatnonzero(j < n)
    j = j[i]
    error = invert(invert(T_gt[i]) @ T_gt[j]) @ (invert(T_est[i]) @ T_est[j])
    rpe_translation = np.linalg.norm(error[:, :3, 3], axis=1)
    rpe_rotation = rotation_angles(error[:, :3, :3])

    sim3_scale, sim3_ate = float("nan"), float("nan")
    if n >= 3:
        scale, R, t = umeyama_sim3(estimate[:, :3], ground_truth[:, :3])
        aligned = (scale * (R @ estimate[:, :3].T)).T + t
        sim3_scale = float(scale)
        sim3_ate = float(np.sqrt(np.mean(np.sum((aligned - ground_truth[:, :3]) ** 2, axis=1))))

    final_drift = float(position_error[-1])
    return TrajectoryMetrics(
        frames=n,
        filled_frames=filled_frames,
        rpe_translation_rmse=float(np.sqrt(np.mean(rpe_translation ** 2))) if len(rpe_translation) else float("nan"),
        rpe_rotation_rmse=float(np.sqrt(np.mean(rpe_rotation ** 2))) if len(rpe_rotation) else float("nan"),
        rpe_pairs=int(len(rpe_translation)),
        ate_rmse=ate_rmse,
        rotation_rmse=rotation_rmse,
        rotation_max=rotation_max,
        velocity_rmse=velocity_rmse,
        angular_velocity_rmse=angular_velocity_rmse,
        sim3_scale=sim3_scale,
        sim3_ate_rmse=sim3_ate,
        distance_travelled=distance,
        final_drift=final_drift,
        drift_per_metre=final_drift / distance if distance > 0 else float("nan"),
    )
