"""Classical visual-inertial baseline: KLT tracking + essential matrix for the translation direction, gyroscope for
rotation, accelerometer for scale and gravity.

Per frame pair:
  1. the gyroscope (bias from the initial hover) is integrated to the body rotation R_wb;
  2. features tracked from the previous frame (goodFeaturesToTrack + calcOpticalFlowPyrLK) give an essential matrix,
     recoverPose yields the unit translation direction of the camera motion;
  3. the IMU displacement over the frame interval (dead reckoning with the current velocity) provides the scale and
     the fallback when tracking fails; the fused displacement updates the velocity so dead-reckoning drift is damped.
It is a baseline, not a VIO system: no bundle adjustment, no bias estimation beyond the hover. Fuel per frame is
dominated by the OpenCV calls, which run as SIMD wasm.
"""
import cv2
import numpy as np

GRAVITY = 9.81
MIN_TRACKS = 30
MIN_INLIERS = 15
MIN_PARALLAX_PIXELS = 1.0


def matrix_to_quaternion(R):
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
        camera = config["camera"]
        self.K = np.array([[camera["fx"], 0, camera["cx"]], [0, camera["fy"], camera["cy"]], [0, 0, 1]])
        self.T_cam_imu = np.asarray(config["imu"]["T_cam_imu"], dtype=float).reshape(4, 4)
        self.T_imu_cam = np.linalg.inv(self.T_cam_imu)
        self.R_bc = self.T_imu_cam[:3, :3]
        self.hover_until = float(config["episode"].get("hover_seconds") or 0.5)
        self.hover_samples = []
        self.initialized = False
        self.R_wb = np.eye(3)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.gravity_world = np.array([0.0, 0.0, -GRAVITY])
        self.last_t = None
        self.T_w_c0 = None
        self.previous_frame = None
        self.previous_points = None
        self.frame_count = 0

    # --- IMU ---------------------------------------------------------------------------------------------------

    def _initialize(self, t):
        samples = np.asarray(self.hover_samples)
        mean_accel = samples[:, 1:4].mean(axis=0)
        self.gyro_bias = samples[:, 4:7].mean(axis=0)
        z_world = mean_accel / np.linalg.norm(mean_accel)  # "up" in body coordinates
        x_world = np.array([1.0, 0.0, 0.0]) - z_world[0] * z_world
        x_world /= np.linalg.norm(x_world)
        self.R_wb = np.stack([x_world, np.cross(z_world, x_world), z_world])
        self.last_t = t
        self.initialized = True
        self.T_w_c0 = self._pose_camera()

    def _integrate(self, imu):
        """Dead-reckon over the window; returns the IMU-only displacement and the rotation at the end."""
        start = self.p.copy()
        for row in imu:
            t, accel, gyro = row[0], row[1:4], row[4:7]
            dt = t - self.last_t
            if dt > 0:
                self.R_wb = self.R_wb @ cv2.Rodrigues((gyro - self.gyro_bias) * dt)[0]
                acceleration = self.R_wb @ accel + self.gravity_world
                self.p += self.v * dt + 0.5 * acceleration * dt * dt
                self.v += acceleration * dt
            self.last_t = t
        return self.p - start

    def _pose_camera(self):
        T_wb = np.eye(4)
        T_wb[:3, :3] = self.R_wb
        T_wb[:3, 3] = self.p
        return T_wb @ self.T_imu_cam

    # --- vision ------------------------------------------------------------------------------------------------

    def _track(self, frame):
        """Unit translation direction of the camera (in the previous camera frame) or None."""
        if self.previous_frame is None or self.previous_points is None or len(self.previous_points) < MIN_TRACKS:
            return None
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame, frame, self.previous_points, None, winSize=(21, 21), maxLevel=3)
        good = status.ravel() == 1
        p0 = self.previous_points[good].reshape(-1, 2)
        p1 = tracked[good].reshape(-1, 2)
        if len(p0) < MIN_TRACKS or np.median(np.linalg.norm(p1 - p0, axis=1)) < MIN_PARALLAX_PIXELS:
            return None
        E, inliers = cv2.findEssentialMat(p0, p1, self.K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3) or inliers is None or int(inliers.sum()) < MIN_INLIERS:
            return None
        count, R, t, _ = cv2.recoverPose(E, p0, p1, self.K, mask=inliers)
        if count < MIN_INLIERS:
            return None
        # recoverPose: x1 = R x0 + t maps points from camera 0 to camera 1 coordinates -> camera 1 sits at -R^T t in camera 0
        return (-R.T @ t).ravel()

    def _detect(self, frame):
        points = cv2.goodFeaturesToTrack(frame, maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
        return None if points is None else points.astype(np.float32)

    # --- protocol ----------------------------------------------------------------------------------------------

    def step(self, t, frame, imu):
        self.frame_count += 1
        if not self.initialized:
            self.hover_samples.extend(imu.tolist())
            if t < self.hover_until or len(self.hover_samples) < 20:
                return None
            self._initialize(imu[-1, 0] if len(imu) else t)
            self.previous_frame = frame.copy()
            self.previous_points = self._detect(frame)
            T_c0_c = np.linalg.inv(self.T_w_c0) @ self._pose_camera()
            return np.concatenate([T_c0_c[:3, 3], matrix_to_quaternion(T_c0_c[:3, :3])])

        R_wb_before = self.R_wb.copy()
        displacement_imu = self._integrate(imu)
        direction_camera = self._track(frame)
        if direction_camera is not None:
            R_wc_previous = R_wb_before @ self.R_bc
            direction_world = R_wc_previous @ direction_camera
            scale = np.linalg.norm(displacement_imu)
            displacement = scale * direction_world
            correction = displacement - displacement_imu
            self.p += correction
            dt = max(imu[-1, 0] - imu[0, 0] + (imu[1, 0] - imu[0, 0] if len(imu) > 1 else 0.0), 1e-3) if len(imu) else 0.05
            self.v = 0.5 * self.v + 0.5 * displacement / dt  # damp dead-reckoning drift with the visual direction
        self.previous_frame = frame.copy()
        self.previous_points = self._detect(frame)
        T_c0_c = np.linalg.inv(self.T_w_c0) @ self._pose_camera()
        return np.concatenate([T_c0_c[:3, 3], matrix_to_quaternion(T_c0_c[:3, :3])])
