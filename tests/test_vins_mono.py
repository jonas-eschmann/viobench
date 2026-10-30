"""VINS-Mono ABI and metric tracking regressions; build first or set RELPOSE_VINS_MONO."""
import os
from pathlib import Path
import re

import numpy as np
import pytest

from relpose import Sandbox
from relpose.solution import workspace
from relpose.tools.abi.config import default_guest_config
from relpose.tools.runner.metrics import IDENTITY_STATE, evaluate_trajectory
from relpose.tools.runner.sandbox import SandboxError

from vio_scene import moving_room_inputs


@pytest.fixture(scope="module")
def vins_mono_path():
    path = Path(os.environ.get("RELPOSE_VINS_MONO") or
                workspace.PACKAGE / "examples/native/vins_mono/build/solution.wasm")
    if not path.is_file():
        pytest.skip(f"VINS-Mono not built ({path}); build examples/native/vins_mono or set RELPOSE_VINS_MONO")
    return path


def new_sandbox(wasm, engine, directory):
    sandbox = Sandbox(wasm, engine, log_dir=directory)
    width, height = sandbox.resolution
    return sandbox, default_guest_config(width=width, height=height)


@pytest.mark.parametrize("invalid", ["missing_focal_length", "negative_focal_length", "singular_extrinsics"])
def test_invalid_calibration_is_rejected(vins_mono_path, engine, tmp_path, invalid):
    sandbox, config = new_sandbox(vins_mono_path, engine, tmp_path)
    if invalid == "missing_focal_length":
        del config["fx"]
    elif invalid == "negative_focal_length":
        config["fx"] = -1
    else:
        config["T_cam_imu"][0][:3] = [0, 0, 0]
    with pytest.raises(SandboxError, match="init failed"):
        sandbox.init(config)


def test_no_features_holds_identity_and_reports_initialization(vins_mono_path, engine, tmp_path):
    sandbox, config = new_sandbox(vins_mono_path, engine, tmp_path)
    sandbox.init(config)
    frame = np.zeros((config["height"], config["width"], 3), dtype=np.uint8)
    for index in range(110):
        times = (index * 10 + np.arange(1, 11)) / 200.0
        imu = np.zeros((10, 7))
        imu[:, 0] = times
        imu[:, 3] = 9.81
        result = sandbox.step(float(times[-1]), frame, imu)
        np.testing.assert_allclose(np.r_[result.pose, result.twist], IDENTITY_STATE, atol=1e-12)
    assert "still initializing after" in sandbox.stderr_path.read_text()
    assert "vins_mono: initialized at t=" not in sandbox.stdout_path.read_text()


def test_metric_tracking_with_camera_imu_offset(vins_mono_path, engine, tmp_path):
    sandbox, config = new_sandbox(vins_mono_path, engine, tmp_path)
    # The common scene oracle derives exact camera pose and twist from an analytic
    # body trajectory. Its render, IMU and truth do not use estimator outputs.
    # A lever arm exercises extrinsic direction and camera-centre velocity; each
    # exposure also falls 2.5 ms after the newest causal IMU sample.
    T_imu_cam = np.linalg.inv(config["T_cam_imu"])
    T_imu_cam[:3, 3] = [0.08, -0.02, 0.04]
    config["T_cam_imu"] = np.linalg.inv(T_imu_cam).tolist()
    sandbox.init(config)
    times, estimates, truth = [], [], []
    for t, frame, imu, expected in moving_room_inputs(config, imu_phase=0.0025):
        # A delayed first visual observation must not change the ABI origin from
        # the first input exposure. Recovering it requires the startup IMU history.
        if t <= 0.5:
            frame = np.zeros_like(frame)
        result = sandbox.step(t, frame, imu)
        times.append(t)
        estimates.append(np.r_[result.pose, result.twist])
        truth.append(expected)
    output = sandbox.stdout_path.read_text()
    initialization = re.search(r"vins_mono: initialized at t=([\d.]+)", output)
    assert initialization is not None, sandbox.stdout_tail()
    assert 0.5 < float(initialization.group(1)) < 10, initialization.group(0)
    np.testing.assert_allclose(estimates[0][:7], IDENTITY_STATE[:7], atol=1e-6)
    # Omit the initialization interval for accuracy, retaining the unaligned
    # first-camera coordinate frame and metric scale. A stationary output fails.
    tail = np.asarray(times) >= 10
    metrics = evaluate_trajectory(np.asarray(truth)[tail], np.asarray(estimates)[tail], np.asarray(times)[tail])
    assert metrics.held == 0, metrics
    assert metrics.ate_rmse < 0.5, metrics
    assert metrics.rpe_translation_rmse < 0.25, metrics
    assert metrics.rotation_rmse < 0.15, metrics
    assert metrics.velocity_rmse < 0.4, metrics
    assert metrics.angular_velocity_rmse < 0.05, metrics
    assert 0.8 < metrics.sim3_scale < 1.2, metrics
