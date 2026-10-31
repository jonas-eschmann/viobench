"""SE(3) and quaternion helpers shared by the runner (metrics) and the datasets (ground truth handling).

Poses are [x, y, z, qx, qy, qz, qw] (scalar-last quaternions); transforms are 4x4 matrices.
"""
import numpy as np


def quaternion_to_matrix(q):
    """(..., 4) [x, y, z, w] -> (..., 3, 3)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def matrix_to_quaternion(R):
    """(..., 3, 3) -> (..., 4) [x, y, z, w] with w >= 0 (Shepperd's method, vectorised)."""
    R = np.asarray(R, dtype=float)
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22
    candidates = np.stack([
        np.stack([m21 - m12, m02 - m20, m10 - m01, 1 + trace], -1),  # w largest
        np.stack([1 + m00 - m11 - m22, m01 + m10, m02 + m20, m21 - m12], -1),  # x largest
        np.stack([m01 + m10, 1 + m11 - m00 - m22, m12 + m21, m02 - m20], -1),  # y largest
        np.stack([m02 + m20, m12 + m21, 1 + m22 - m00 - m11, m10 - m01], -1),  # z largest
    ], -2)
    pivots = np.stack([trace, m00, m11, m22], -1)
    best = np.argmax(pivots, axis=-1)
    q = np.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., 3:4] < 0, -q, q)


def poses_to_matrices(poses):
    poses = np.asarray(poses, dtype=float)
    T = np.tile(np.eye(4), (len(poses), 1, 1))
    T[:, :3, :3] = quaternion_to_matrix(poses[:, 3:7])
    T[:, :3, 3] = poses[:, :3]
    return T


def matrices_to_poses(T):
    T = np.asarray(T, dtype=float)
    return np.concatenate([T[:, :3, 3], matrix_to_quaternion(T[:, :3, :3])], axis=1)


def invert(T):
    T = np.asarray(T, dtype=float)
    out = np.tile(np.eye(4), T.shape[:-2] + (1, 1))
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    return out


def rotation_angles(R):
    """Geodesic rotation angle of each (..., 3, 3) matrix."""
    trace = np.einsum("...ii->...", R)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def rotation_vectors(R):
    """Rotation vector (axis times angle, the SO(3) logarithm) of each (..., 3, 3) matrix; exact up to angles near pi."""
    R = np.asarray(R, dtype=float)
    skew = 0.5 * np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    sine = np.linalg.norm(skew, axis=-1)
    angle = np.arctan2(sine, (np.einsum("...ii->...", R) - 1.0) / 2.0)
    return skew * np.where(sine > 0, angle / np.where(sine > 0, sine, 1.0), 1.0)[..., None]


def slerp(q0, q1, fraction):
    """Vectorised quaternion slerp, [x, y, z, w]; q0, q1 (N, 4), fraction (N,)."""
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-8
    w0 = np.where(small, 1.0 - fraction, np.sin((1.0 - fraction) * theta) / np.where(small, 1.0, sin_theta))
    w1 = np.where(small, fraction, np.sin(fraction * theta) / np.where(small, 1.0, sin_theta))
    out = w0[:, None] * q0 + w1[:, None] * q1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def interpolate_poses(times, poses, query_times):
    """Linear position + slerp orientation of (K, 7) poses sampled at times (K,), evaluated at query_times (N,)."""
    times = np.asarray(times, dtype=float)
    poses = np.asarray(poses, dtype=float)
    query_times = np.asarray(query_times, dtype=float)
    if np.any(query_times < times[0] - 1e-9) or np.any(query_times > times[-1] + 1e-9):
        raise ValueError(f"query times [{query_times.min()}, {query_times.max()}] outside ground truth [{times[0]}, {times[-1]}]")
    upper = np.clip(np.searchsorted(times, query_times, side="right"), 1, len(times) - 1)
    lower = upper - 1
    span = times[upper] - times[lower]
    fraction = np.where(span > 0, (query_times - times[lower]) / np.where(span > 0, span, 1.0), 0.0)
    fraction = np.clip(fraction, 0.0, 1.0)
    position = poses[lower, :3] + fraction[:, None] * (poses[upper, :3] - poses[lower, :3])
    orientation = slerp(poses[lower, 3:7], poses[upper, 3:7], fraction)
    return np.concatenate([position, orientation], axis=1)
