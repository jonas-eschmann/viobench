"""ORB-SLAM3 ABI and metric tracking regressions; build first or set VIOBENCH_ORBSLAM3."""
import os
from pathlib import Path
import re

import numpy as np
import pytest

from viobench import Sandbox
from viobench.solution import workspace
from viobench.tools.abi.config import default_guest_config
from viobench.tools.geometry import matrix_to_quaternion
from viobench.tools.runner.metrics import IDENTITY_STATE, evaluate_trajectory
from viobench.tools.runner.sandbox import SandboxError


@pytest.fixture(scope="module")
def orbslam3_path():
    path = Path(os.environ.get("VIOBENCH_ORBSLAM3") or
                workspace.PACKAGE / "examples/native/orbslam3/build/solution.wasm")
    if not path.is_file():
        pytest.skip(f"ORB-SLAM3 not built ({path}); build examples/native/orbslam3 or set VIOBENCH_ORBSLAM3")
    return path


def new_sandbox(wasm, engine, directory):
    sandbox = Sandbox(wasm, engine, log_dir=directory)
    width, height = sandbox.resolution
    return sandbox, default_guest_config(width=width, height=height)


@pytest.mark.parametrize("invalid", ["missing_focal_length", "negative_focal_length", "singular_extrinsics"])
def test_invalid_calibration_is_rejected(orbslam3_path, engine, tmp_path, invalid):
    sandbox, config = new_sandbox(orbslam3_path, engine, tmp_path)
    if invalid == "missing_focal_length":
        del config["fx"]
    elif invalid == "negative_focal_length":
        config["fx"] = -1
    else:
        config["T_cam_imu"][0][:3] = [0, 0, 0]
    with pytest.raises(SandboxError, match="init failed"):
        sandbox.init(config)


def test_no_features_holds_identity_and_reports_initialization(orbslam3_path, engine, tmp_path):
    sandbox, config = new_sandbox(orbslam3_path, engine, tmp_path)
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
    assert "orbslam3: initialized at t=" not in sandbox.stdout_path.read_text()


def room_motion(t):
    """Smooth, independently specified IMU trajectory with observable metric scale."""
    roll, pitch, yaw = 0.05 * np.sin(1.1 * t), 0.06 * np.cos(0.9 * t), 0.2 * np.sin(0.7 * t)
    roll_rate, pitch_rate, yaw_rate = 0.055 * np.cos(1.1 * t), -0.054 * np.sin(0.9 * t), 0.14 * np.cos(0.7 * t)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rotation = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @
                np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @
                np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    body_rate = np.array([roll_rate - yaw_rate * sp,
                          pitch_rate * cr + yaw_rate * sr * cp,
                          -pitch_rate * sr + yaw_rate * cr * cp])
    position = np.array([0.8 * np.sin(1.1 * t), 0.7 * np.sin(0.7 * t), 0.4 * np.cos(0.9 * t)])
    velocity = np.array([0.88 * np.cos(1.1 * t), 0.49 * np.cos(0.7 * t), -0.36 * np.sin(0.9 * t)])
    acceleration = np.array([-0.968 * np.sin(1.1 * t), -0.343 * np.sin(0.7 * t), -0.324 * np.cos(0.9 * t)])
    return rotation, position, velocity, acceleration, body_rate


def moving_room_inputs(config, seconds=18, imu_phase=0.0):
    """Render fixed textured walls and exact IMU, including the calibrated camera lever arm.

    Truth is always relative to the first *input* exposure, including the motion before
    ORB-SLAM3 initializes. Pose evaluation therefore cannot hide a late origin or a
    scale error through trajectory alignment.
    """
    import cv2

    rng = np.random.default_rng(31)
    lower, upper = np.array([-4, -4, -3]), np.array([5, 4, 3])
    textures = [cv2.GaussianBlur(cv2.resize(rng.integers(20, 235, (96, 96), dtype=np.uint8),
                                          (768, 768), interpolation=cv2.INTER_NEAREST), (3, 3), 0.6)
                for _ in range(6)]
    T_imu_cam = np.linalg.inv(config["T_cam_imu"])
    R_imu_cam, lever = T_imu_cam[:3, :3], T_imu_cam[:3, 3]
    yy, xx = np.mgrid[:config["height"], :config["width"]]
    rays = np.stack([(xx - config["cx"]) / config["fx"],
                     (yy - config["cy"]) / config["fy"], np.ones_like(xx)], axis=-1)
    rotation0, position0, *_ = room_motion(0.05)
    camera0 = rotation0 @ R_imu_cam
    centre0 = position0 + rotation0 @ lever
    for index in range(round(seconds * 20)):
        t = (index + 1) / 20.0
        rotation, position, velocity, _, rate = room_motion(t)
        camera = rotation @ R_imu_cam
        centre = position + rotation @ lever
        camera_velocity = velocity + rotation @ np.cross(rate, lever)
        directions = rays @ camera.T
        with np.errstate(divide="ignore"):
            distances = np.where(directions > 0, upper - centre, lower - centre) / directions
        axes = np.argmin(distances, axis=-1)
        points = centre + directions * np.min(distances, axis=-1)[..., None]
        gray = np.zeros(xx.shape, np.uint8)
        for axis in range(3):
            a, b = [other for other in range(3) if other != axis]
            map_x = ((points[..., a] - lower[a]) / (upper[a] - lower[a]) * 767).astype(np.float32)
            map_y = ((points[..., b] - lower[b]) / (upper[b] - lower[b]) * 767).astype(np.float32)
            for side in range(2):
                visible = (axes == axis) & ((directions[..., axis] > 0) == bool(side))
                texture = cv2.remap(textures[2 * axis + side], map_x, map_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
                gray[visible] = texture[visible]
        times = (index * 10 + np.arange(1, 11)) / 200.0 - imu_phase
        imu = np.zeros((10, 7))
        imu[:, 0] = times
        for j, time in enumerate(times):
            body_rotation, _, _, acceleration, body_rate = room_motion(time)
            imu[j, 1:4] = body_rotation.T @ (acceleration + [0, 0, 9.81])
            imu[j, 4:7] = body_rate
        truth = np.r_[camera0.T @ (centre - centre0), matrix_to_quaternion(camera0.T @ camera),
                      camera.T @ camera_velocity, R_imu_cam.T @ rate]
        yield t, np.repeat(gray[..., None], 3, axis=2), imu, truth


def test_metric_tracking_with_camera_imu_offset(orbslam3_path, engine, tmp_path):
    sandbox, config = new_sandbox(orbslam3_path, engine, tmp_path)
    # Nonzero extrinsic translation exercises the camera-centre velocity as well
    # as the IMU-to-camera direction; exposures fall between IMU sample times.
    # The moving camera initially sees no features, so the first visual keyframe
    # is later than camera0 and reconstructing the ABI origin requires IMU history.
    T_imu_cam = np.linalg.inv(config["T_cam_imu"])
    T_imu_cam[:3, 3] = [0.08, -0.02, 0.04]
    config["T_cam_imu"] = np.linalg.inv(T_imu_cam).tolist()
    sandbox.init(config)
    times, estimates, truth = [], [], []
    for t, frame, imu, expected in moving_room_inputs(config, imu_phase=0.0025):
        if t <= 0.5:
            frame = np.zeros_like(frame)
        result = sandbox.step(t, frame, imu)
        times.append(t)
        estimates.append(np.r_[result.pose, result.twist])
        truth.append(expected)
    output = sandbox.stdout_path.read_text()
    initialization = re.search(r"orbslam3: initialized at t=[\d.]+ \(origin t=([\d.]+)\)", output)
    assert initialization is not None, sandbox.stdout_tail()
    assert float(initialization.group(1)) > 0.5, initialization.group(0)
    np.testing.assert_allclose(estimates[0][:7], IDENTITY_STATE[:7], atol=1e-6)
    # Discard startup for accuracy, but retain the original first-frame origin.
    tail = np.asarray(times) >= 10
    metrics = evaluate_trajectory(np.asarray(truth)[tail], np.asarray(estimates)[tail], np.asarray(times)[tail])
    assert metrics.held == 0, metrics
    assert metrics.ate_rmse < 0.5, metrics
    assert metrics.rpe_translation_rmse < 0.25, metrics
    assert metrics.rotation_rmse < 0.15, metrics
    assert metrics.velocity_rmse < 0.4, metrics
    assert metrics.angular_velocity_rmse < 0.05, metrics
    assert 0.8 < metrics.sim3_scale < 1.2, metrics
