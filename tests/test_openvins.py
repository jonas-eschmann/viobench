"""OpenVINS startup regressions; build the example first or set RELPOSE_OPENVINS to its .wasm."""
import os
from pathlib import Path

import numpy as np
import pytest

from relpose import Sandbox
from relpose.solution import workspace
from relpose.tools.abi.config import default_guest_config


@pytest.fixture(scope="module")
def openvins_path():
    path = Path(os.environ.get("RELPOSE_OPENVINS") or
                workspace.PACKAGE / "examples/native/openvins/build/solution.wasm")
    if not path.is_file():
        pytest.skip(f"OpenVINS not built ({path}); build examples/native/openvins or set RELPOSE_OPENVINS")
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
    from relpose.tools.abi.config import R_BODY_CAMERA_FORWARD

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
