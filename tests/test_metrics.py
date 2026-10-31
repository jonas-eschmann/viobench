import numpy as np
import pytest

from viobench.tools.geometry import invert, matrices_to_poses, matrix_to_quaternion, poses_to_matrices, quaternion_to_matrix, rotation_vectors
from viobench.tools.runner.metrics import evaluate_trajectory, hold_last_state, umeyama_sim3


def random_poses(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1
    return np.concatenate([rng.normal(size=(n, 3)), q], axis=1)


def random_states(rng, n):
    return np.concatenate([random_poses(rng, n), rng.normal(size=(n, 6))], axis=1)


def test_quaternion_matrix_roundtrip():
    rng = np.random.default_rng(0)
    poses = random_poses(rng, 50)
    np.testing.assert_allclose(matrix_to_quaternion(quaternion_to_matrix(poses[:, 3:])), poses[:, 3:], atol=1e-12)
    np.testing.assert_allclose(matrices_to_poses(poses_to_matrices(poses)), poses, atol=1e-12)
    T = poses_to_matrices(poses)
    np.testing.assert_allclose(invert(T) @ T, np.tile(np.eye(4), (50, 1, 1)), atol=1e-12)


def test_rotation_vectors_invert_the_exponential():
    rng = np.random.default_rng(5)
    axes = rng.normal(size=(50, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.concatenate([[0.0, 1e-9, 1e-4], rng.uniform(0, 3.0, 47)])
    q = np.column_stack([axes * np.sin(angles / 2)[:, None], np.cos(angles / 2)])
    np.testing.assert_allclose(rotation_vectors(quaternion_to_matrix(q)), axes * angles[:, None], atol=1e-9)


def test_perfect_estimate_has_zero_error():
    rng = np.random.default_rng(1)
    gt = random_states(rng, 40)
    t = np.arange(40) * 0.05
    m = evaluate_trajectory(gt, gt.copy(), t)
    assert m.ate_rmse == pytest.approx(0, abs=1e-12) and m.rotation_rmse == pytest.approx(0, abs=1e-7)
    assert m.velocity_rmse == 0 and m.angular_velocity_rmse == 0
    assert m.rpe_translation_rmse == pytest.approx(0, abs=1e-12) and m.rpe_pairs == 20
    assert m.held == 0 and m.sim3_scale == pytest.approx(1.0)


def test_constant_offset_and_held_frames():
    rng = np.random.default_rng(2)
    gt = random_states(rng, 30)
    t = np.arange(30) * 0.05
    est = gt.copy()
    est[:, 0] += 0.5  # half a metre off everywhere
    m = evaluate_trajectory(gt, est, t)
    assert m.held == 0 and m.ate_rmse == pytest.approx(0.5) and m.final_drift == pytest.approx(0.5)
    est[0] = np.nan  # no state yet: identity, standing still
    est[5] = np.nan  # a failed step: repeats the pose of frame 4, standing still
    est[6, 3:7] = 0  # a zero quaternion is no state either
    est[7, 12] = np.inf  # nor is a twist that is not finite
    held = hold_last_state(est)
    np.testing.assert_array_equal(held[0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(held[5:8, :7], [est[4, :7]] * 3)
    np.testing.assert_array_equal(held[5:8, 7:], np.zeros((3, 6)))
    np.testing.assert_array_equal(np.delete(held, [0, 5, 6, 7], axis=0), np.delete(est, [0, 5, 6, 7], axis=0))
    m = evaluate_trajectory(gt, est, t)
    assert m.held == 4 and m.ate_rmse == pytest.approx(evaluate_trajectory(gt, held, t).ate_rmse) and m.ate_rmse > 0.5


def test_scaled_estimate_is_fixed_by_sim3():
    rng = np.random.default_rng(3)
    gt = random_states(rng, 25)
    t = np.arange(25) * 0.1
    est = gt.copy()
    est[:, :3] *= 0.5  # monocular scale error
    m = evaluate_trajectory(gt, est, t)
    assert m.sim3_scale == pytest.approx(2.0)
    assert m.sim3_ate_rmse == pytest.approx(0, abs=1e-9)
    assert m.ate_rmse > 0.1


def test_twist_errors():
    rng = np.random.default_rng(6)
    gt = random_states(rng, 20)
    t = np.arange(20) * 0.05
    est = gt.copy()
    est[:, 7:10] += [0.0, 0.3, 0.4]  # 0.5 m/s off
    est[:, 12] -= 0.2  # 0.2 rad/s off
    m = evaluate_trajectory(gt, est, t)
    assert m.velocity_rmse == pytest.approx(0.5) and m.angular_velocity_rmse == pytest.approx(0.2)
    assert m.ate_rmse == pytest.approx(0, abs=1e-12) and m.rotation_rmse == pytest.approx(0, abs=1e-7)  # the twist does not enter the pose metrics


def test_rpe_pairs_frames_exactly_one_second_apart():
    """Float timestamps at 20 Hz: every window is 20 frames, never 21 (rounding once made t_i + 1 s land past t_i+20)."""
    rng = np.random.default_rng(6)
    n = 1200
    t = 0.025 + np.arange(n) / 20.0  # mid-exposure stamps, as the episodes have them
    gt = random_states(rng, n)
    est = gt.copy()
    est[:, :3] += np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0)  # drift
    m = evaluate_trajectory(gt, est, t)
    G, E = poses_to_matrices(gt), poses_to_matrices(est)
    errors = [np.linalg.norm((invert(invert(G[i]) @ G[i + 20]) @ (invert(E[i]) @ E[i + 20]))[:3, 3]) for i in range(n - 20)]
    assert m.rpe_pairs == n - 20
    assert m.rpe_translation_rmse == pytest.approx(np.sqrt(np.mean(np.square(errors))))


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(20, 3))
    R = quaternion_to_matrix(random_poses(rng, 1)[0, 3:])
    target = 1.7 * source @ R.T + [1, 2, 3]
    scale, R_hat, t_hat = umeyama_sim3(source, target)
    assert scale == pytest.approx(1.7)
    np.testing.assert_allclose(R_hat, R, atol=1e-9)
    np.testing.assert_allclose(t_hat, [1, 2, 3], atol=1e-9)
