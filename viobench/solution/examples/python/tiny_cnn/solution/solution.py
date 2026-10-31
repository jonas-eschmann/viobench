"""Learned baseline: a tiny CNN on consecutive frame pairs plus IMU preintegration predicts the relative camera
translation (and a rotation residual on top of the gyroscope) per frame; the motions are chained into the pose
relative to frame 0, and the last one over the frame interval gives the twist (its mean over the interval, so it lags
half a frame).

Runs the ONNX graph through cv2.dnn (SIMD kernels) when available and falls back to the numpy implementation in
tinynn.py otherwise. Trained with ../train.py, which also writes net.onnx / weights.npz / meta.json next to this file.
"""
import json

import cv2
import numpy as np

from preintegrate import exp_so3, log_so3, preintegrate
from tinynn import TinyNet


def quaternion_from_matrix(R):
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.empty(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q / np.linalg.norm(q)


class Solution:
    def __init__(self, config):
        import solution_bundle as bundle  # the files next to this one (in the runtime, or on the host under `viobench debug`)

        self.meta = json.loads(bundle.read_text("meta.json"))
        self.size = (int(self.meta["input_width"]), int(self.meta["input_height"]))
        self.imu_mean = np.asarray(self.meta["imu_mean"], dtype=np.float32)
        self.imu_std = np.asarray(self.meta["imu_std"], dtype=np.float32)
        self.target_std = np.asarray(self.meta["target_std"], dtype=np.float32)
        self.T_cam_imu = np.asarray(config["T_cam_imu"], dtype=float)
        self.net = None
        if hasattr(cv2, "dnn") and bundle.exists("net.onnx"):
            try:
                self.net = cv2.dnn.readNetFromONNX(np.frombuffer(bundle.read_bytes("net.onnx"), np.uint8))
                print("using cv2.dnn")
            except cv2.error as error:  # fall back to numpy
                print("cv2.dnn unavailable:", str(error).strip().splitlines()[-1][:100])
        if self.net is None:
            self.fallback = TinyNet(np.load(bundle.open("weights.npz")))
            print("using the numpy forward pass")
        self.previous = None
        self.previous_t = None
        self.T = np.eye(4)

    def _predict(self, pair, imu_vector):
        imu_normalised = ((imu_vector - self.imu_mean) / self.imu_std).astype(np.float32)
        if self.net is not None:
            self.net.setInput(pair[None], "images")
            self.net.setInput(imu_normalised[None], "imu")
            output = self.net.forward()[0]
        else:
            output = self.fallback(pair, imu_normalised)
        return output * self.target_std

    def step(self, t, frame, imu):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # frames are RGB; the network was trained on gray
        small = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if self.previous is None or len(imu) == 0:
            self.previous, self.previous_t = small, t
            return np.array([0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0])
        integrated = preintegrate(imu, self.T_cam_imu)
        motion = self._predict(np.stack([self.previous, small]), integrated)
        delta = np.eye(4)
        delta[:3, :3] = exp_so3(integrated[:3].astype(np.float64)) @ exp_so3(motion[3:6].astype(np.float64))  # gyro rotation + learned residual
        delta[:3, 3] = motion[:3]
        self.T = self.T @ delta
        interval = t - self.previous_t
        twist = np.concatenate([delta[:3, :3].T @ delta[:3, 3], log_so3(delta[:3, :3])]) / interval  # in the current camera's axes
        self.previous, self.previous_t = small, t
        return np.concatenate([self.T[:3, 3], quaternion_from_matrix(self.T[:3, :3]), twist])
