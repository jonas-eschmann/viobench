"""OpenVINS startup and tracking regressions; build first or set VIOBENCH_OPENVINS to its .wasm."""
import os
from pathlib import Path

import numpy as np
import pytest

from viobench import Sandbox
from viobench.solution import workspace
from viobench.tools.abi.config import default_guest_config


@pytest.fixture(scope="module")
def openvins_path():
    path = Path(os.environ.get("VIOBENCH_OPENVINS") or
                workspace.PACKAGE / "examples/native/openvins/build/solution.wasm")
    if not path.is_file():
        pytest.skip(f"OpenVINS not built ({path}); build examples/native/openvins or set VIOBENCH_OPENVINS")
    return path


def stationary_inputs(config, textured, seconds):
    width, height = config["width"], config["height"]
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, (height, width), dtype=np.uint8) if textured else np.zeros((height, width), np.uint8)
    frame = np.repeat(gray[:, :, None], 3, axis=2)
    for index in range(round(seconds * 20)):
        times = (index * 10 + np.arange(1, 11)) / 200.0
        # Gravity along body +z, with the simulator's noise and small sensor biases.
        acceleration = rng.normal(0, 0.28, (10, 3)) + [0.02, -0.03, 9.85]
        gyro = rng.normal(0, 0.07, (10, 3)) + [0.01, -0.01, 0.005]
        yield float(times[-1]), frame, np.column_stack([times, acceleration, gyro])


def dynamic_inputs(config, seconds):
    """Project a fixed 3D corner cloud while translating and turning; borders create short tracks."""
    from viobench.tools.abi.config import R_BODY_CAMERA_FORWARD

    rng = np.random.default_rng(42)
    landmarks = np.column_stack([rng.uniform(3, 10, 900), rng.uniform(-5, 5, 900), rng.uniform(-4, 4, 900)])
    intensities = rng.integers(100, 255, len(landmarks))

    def rotation(t):
        theta = 0.25 * np.sin(1.7 * t)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    for index in range(round(seconds * 20)):
        t = (index + 1) / 20.0
        position = np.array([0.8 * np.sin(1.3 * t), 0.5 * np.sin(1.1 * t), 0.2 * np.cos(1.7 * t)])
        camera = (landmarks - position) @ rotation(t) @ R_BODY_CAMERA_FORWARD
        uv = camera[:, :2] / camera[:, 2, None] * [config["fx"], config["fy"]] + [config["cx"], config["cy"]]
        gray = np.full((config["height"], config["width"]), 30, dtype=np.uint8)
        for i, (u, v) in enumerate(np.rint(uv).astype(int)):
            if camera[i, 2] > 0 and 3 <= u < config["width"] - 3 and 3 <= v < config["height"] - 3:
                gray[v - 2:v + 3, u - 2:u + 3] = intensities[i]
                gray[v:v + 3, u:u + 3] = 0

        times = (index * 10 + np.arange(1, 11)) / 200.0
        imu = np.zeros((10, 7))
        imu[:, 0] = times
        for j, time in enumerate(times):
            # Specific force and angular velocity from the same analytic camera motion.
            acceleration = np.array([-0.8 * 1.3**2 * np.sin(1.3 * time), -0.5 * 1.1**2 * np.sin(1.1 * time),
                                     -0.2 * 1.7**2 * np.cos(1.7 * time)])
            imu[j, 1:4] = rotation(time).T @ (acceleration + [0, 0, 9.81])
            imu[j, 6] = 0.25 * 1.7 * np.cos(1.7 * time)
        yield t, np.repeat(gray[:, :, None], 3, axis=2), imu


def run_inputs(wasm, engine, directory, inputs):
    sandbox = Sandbox(wasm, engine, log_dir=directory)
    width, height = sandbox.resolution
    config = default_guest_config(width=width, height=height)
    sandbox.init(config)
    for t, frame, imu in inputs(config):
        result = sandbox.step(t, frame, imu)
        assert np.isfinite(result.pose).all()
        assert np.isfinite(result.twist).all()
    return (directory / "stdout.txt").read_text(), (directory / "stderr.txt").read_text()


def test_stationary_initialization(openvins_path, engine, tmp_path):
    # The simulator's startup hover lasts one second. Initialize before motion begins, despite OpenVINS' history margin.
    output, _ = run_inputs(openvins_path, engine, tmp_path, lambda config: stationary_inputs(config, True, 0.95))
    assert "openvins: initialized at t=" in output, output


def test_dynamic_initialization_with_short_tracks(openvins_path, engine, tmp_path):
    output, errors = run_inputs(openvins_path, engine, tmp_path, lambda config: dynamic_inputs(config, 5))
    # One-observation landmarks previously reached Ceres with zero/NaN depth. A moving scene must
    # initialize without sending an invalid residual/Jacobian problem to the optimizer first.
    invalid_residuals = "Residual and Jacobian evaluation failed" in output + errors
    assert not invalid_residuals, f"Invalid initialization residuals; see {tmp_path / 'stderr.txt'}"
    assert "openvins: initialized at t=" in output, output


def test_missing_features_reports_uninitialized_state(openvins_path, engine, tmp_path):
    output, errors = run_inputs(openvins_path, engine, tmp_path, lambda config: stationary_inputs(config, False, 5.5))
    assert "openvins: initialized at t=" not in output, output
    assert "still initializing after" in errors, errors


def low_parallax_motion(t):
    """Body pose, velocity, acceleration and body angular rate, all from one smooth trajectory.

    A hovering drone settles, turns in place, then starts translating. In particular, the first
    five seconds have rotational image motion but no translational baseline for triangulation.
    """
    roll = 0.015 * np.exp(-3 * t) * np.sin(10 * t)
    pitch = 0.01 * np.exp(-3 * t) * np.sin(8 * t)
    roll_rate = 0.015 * np.exp(-3 * t) * (10 * np.cos(10 * t) - 3 * np.sin(10 * t))
    pitch_rate = 0.01 * np.exp(-3 * t) * (8 * np.cos(8 * t) - 3 * np.sin(8 * t))
    u = np.clip((t - 1) / 2, 0, 1)
    yaw = -0.8 * (10 * u**3 - 15 * u**4 + 6 * u**5)
    yaw_rate = -0.4 * 30 * u**2 * (1 - u)**2
    s = max(t - 5, 0)
    yaw += 0.12 * (1 - np.cos(0.4 * s))
    yaw_rate += 0.048 * np.sin(0.4 * s)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rotation = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @
                np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @
                np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    angular_velocity = np.array([roll_rate - yaw_rate * sp,
                                 pitch_rate * cr + yaw_rate * sr * cp,
                                 -pitch_rate * sr + yaw_rate * cr * cp])
    position = np.array([0.3 * (s - np.sin(0.8 * s) / 0.8),
                         0.5 * (1 - np.cos(0.5 * s)), 0.15 * (1 - np.cos(0.6 * s))])
    velocity = np.array([0.3 * (1 - np.cos(0.8 * s)), 0.25 * np.sin(0.5 * s), 0.09 * np.sin(0.6 * s)])
    acceleration = np.array([0.24 * np.sin(0.8 * s), 0.125 * np.cos(0.5 * s), 0.054 * np.cos(0.6 * s)])
    if t <= 5:
        acceleration[:] = 0
    return rotation, position, velocity, acceleration, angular_velocity


def low_parallax_inputs(config, seconds=15, imu_phase=0.0):
    """Ray-cast fixed textured room walls, with noisy IMU and a separate analytic truth oracle.

    Unlike screen-space corner sprites, wall texture undergoes the actual perspective warp,
    including subpixel movement. The solution sees only RGB frames, calibration and IMU rows.
    """
    import cv2

    from viobench.tools.abi.config import R_BODY_CAMERA_FORWARD
    from viobench.tools.geometry import matrix_to_quaternion

    rng = np.random.default_rng(19)
    lower, upper = np.array([-4, -5, -3]), np.array([8, 5, 3])
    textures = [cv2.GaussianBlur(cv2.resize(rng.integers(30, 230, (64, 64), dtype=np.uint8),
                                          (512, 512), interpolation=cv2.INTER_NEAREST), (3, 3), 0.6)
                for _ in range(6)]
    yy, xx = np.mgrid[:config["height"], :config["width"]]
    rays = np.stack([(xx - config["cx"]) / config["fx"],
                     (yy - config["cy"]) / config["fy"], np.ones_like(xx)], axis=-1)
    rotation0, position0, *_ = low_parallax_motion(0.05)
    camera0 = rotation0 @ R_BODY_CAMERA_FORWARD
    gyro_bias, accel_bias = np.array([0.01, -0.015, -0.015]), np.array([0.03, 0.08, 0.16])
    for index in range(round(seconds * 20)):
        t = (index + 1) / 20.0
        rotation, position, velocity, _, angular_velocity = low_parallax_motion(t)
        camera = rotation @ R_BODY_CAMERA_FORWARD
        directions = rays @ camera.T
        # The camera stays inside the room: the nearest forward intersection is its visible wall.
        with np.errstate(divide="ignore"):
            distances = np.where(directions > 0, upper - position, lower - position) / directions
        axes = np.argmin(distances, axis=-1)
        points = position + directions * np.min(distances, axis=-1)[..., None]
        gray = np.zeros(xx.shape, np.uint8)
        for axis in range(3):
            a, b = [other for other in range(3) if other != axis]
            map_x = ((points[..., a] - lower[a]) / (upper[a] - lower[a]) * 511).astype(np.float32)
            map_y = ((points[..., b] - lower[b]) / (upper[b] - lower[b]) * 511).astype(np.float32)
            for side in range(2):
                visible = (axes == axis) & ((directions[..., axis] > 0) == bool(side))
                texture = cv2.remap(textures[2 * axis + side], map_x, map_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
                gray[visible] = texture[visible]

        # Camera exposures need not coincide with the newest causal IMU sample.
        times = (index * 10 + np.arange(1, 11)) / 200.0 - imu_phase
        imu = np.zeros((10, 7))
        imu[:, 0] = times
        for j, time in enumerate(times):
            body_rotation, _, _, acceleration, body_rate = low_parallax_motion(time)
            # Random walks are continuous-time densities; measurement noise is per sample.
            accel_bias += rng.normal(0, 0.001 * np.sqrt(0.005), 3)
            gyro_bias += rng.normal(0, 0.0006 * np.sqrt(0.005), 3)
            imu[j, 1:4] = body_rotation.T @ (acceleration + [0, 0, 9.81]) + accel_bias + rng.normal(0, 0.28, 3)
            imu[j, 4:7] = body_rate + gyro_bias + rng.normal(0, 0.07, 3)
        truth = np.r_[camera0.T @ (position - position0), matrix_to_quaternion(camera0.T @ camera),
                      camera.T @ velocity, R_BODY_CAMERA_FORWARD.T @ angular_velocity]
        yield t, np.repeat(gray[..., None], 3, axis=2), imu, truth


@pytest.mark.parametrize("blank_images,imu_phase", [(False, 0.0), (True, 0.0), (False, 0.0025)],
                         ids=["normal", "image_dropout", "camera_between_imu_samples"])
def test_tracking_after_turning_in_place(openvins_path, engine, tmp_path, blank_images, imu_phase):
    from viobench.tools.runner.metrics import evaluate_trajectory

    sandbox = Sandbox(openvins_path, engine, log_dir=tmp_path)
    width, height = sandbox.resolution
    # A wide-angle camera makes slow motion subpixel, as in the ProcTHOR recordings.
    config = default_guest_config(width=width, height=height, horizontal_fov_degrees=120)
    sandbox.init(config)
    times, estimates, truth = [], [], []
    for t, frame, imu, expected in low_parallax_inputs(config, imu_phase=imu_phase):
        if blank_images and 9.0 <= t < 10.0:
            frame = np.zeros_like(frame)
        result = sandbox.step(t, frame, imu)
        times.append(t)
        estimates.append(np.r_[result.pose, result.twist])
        truth.append(expected)
    metrics = evaluate_trajectory(truth, estimates, times)
    assert metrics.filled_frames == 0, metrics
    # Initialization alone is insufficient: bias drift used to destroy the trajectory after
    # the rotation-only interval. These bounds also reject an always-stationary solution.
    # A complete one-second image dropout must survive window marginalization
    # and recover when textured images return, without losing metric scale.
    assert metrics.rpe_translation_rmse < 0.15, metrics
    assert metrics.ate_rmse < 0.6, metrics
    assert metrics.final_drift < 0.6, metrics
    assert metrics.rotation_rmse < 0.1, metrics
