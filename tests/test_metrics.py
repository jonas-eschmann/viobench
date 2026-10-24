import numpy as np
import pytest

from relpose.tools.runner.metrics import evaluate_trajectory, invert, matrices_to_poses, matrix_to_quaternion, poses_to_matrices, quaternion_to_matrix, umeyama_sim3


def random_poses(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1
    return np.concatenate([rng.normal(size=(n, 3)), q], axis=1)


def test_quaternion_matrix_roundtrip():
    rng = np.random.default_rng(0)
    poses = random_poses(rng, 50)
    np.testing.assert_allclose(matrix_to_quaternion(quaternion_to_matrix(poses[:, 3:])), poses[:, 3:], atol=1e-12)
    np.testing.assert_allclose(matrices_to_poses(poses_to_matrices(poses)), poses, atol=1e-12)
    T = poses_to_matrices(poses)
    np.testing.assert_allclose(invert(T) @ T, np.tile(np.eye(4), (50, 1, 1)), atol=1e-12)


def test_perfect_estimate_has_zero_error():
    rng = np.random.default_rng(1)
    gt = random_poses(rng, 40)
    t = np.arange(40) * 0.05
    m = evaluate_trajectory(gt, gt.copy(), t)
    assert m.ate_rmse == pytest.approx(0, abs=1e-12) and m.rotation_rmse == pytest.approx(0, abs=1e-7)
    assert m.rpe_translation_rmse == pytest.approx(0, abs=1e-12) and m.rpe_pairs == 20
    assert m.coverage == 1.0 and m.sim3_scale == pytest.approx(1.0)


def test_constant_offset_and_missing_frames():
    rng = np.random.default_rng(2)
    gt = random_poses(rng, 30)
    t = np.arange(30) * 0.05
    est = gt.copy()
    est[:, 0] += 0.5  # half a metre off everywhere
    est[5] = np.nan
    m = evaluate_trajectory(gt, est, t, max_position_error=2.0)
    assert m.estimated == 29
    assert m.ate_rmse_valid == pytest.approx(0.5)
    assert m.ate_rmse == pytest.approx(np.sqrt((29 * 0.25 + 4.0) / 30))
    assert m.final_drift == pytest.approx(0.5)


def test_scaled_estimate_is_fixed_by_sim3():
    rng = np.random.default_rng(3)
    gt = random_poses(rng, 25)
    t = np.arange(25) * 0.1
    est = gt.copy()
    est[:, :3] *= 0.5  # monocular scale error
    m = evaluate_trajectory(gt, est, t)
    assert m.sim3_scale == pytest.approx(2.0)
    assert m.sim3_ate_rmse == pytest.approx(0, abs=1e-9)
    assert m.ate_rmse > 0.1


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(20, 3))
    R = quaternion_to_matrix(random_poses(rng, 1)[0, 3:])
    target = 1.7 * source @ R.T + [1, 2, 3]
    scale, R_hat, t_hat = umeyama_sim3(source, target)
    assert scale == pytest.approx(1.7)
    np.testing.assert_allclose(R_hat, R, atol=1e-9)
    np.testing.assert_allclose(t_hat, [1, 2, 3], atol=1e-9)
