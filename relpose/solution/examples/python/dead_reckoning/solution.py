"""IMU-only baseline: integrate the gyroscope for attitude and the accelerometer twice for position.

Nothing says how an episode starts, so the first half second of IMU calibrates: the mean accelerometer reading gives
the gravity direction in the body frame (and absorbs the accelerometer bias along it) and, if both sensors only show
noise there (the platform was at rest), the mean gyroscope reading gives the gyro bias; a moving start keeps it at
zero. Everything is expressed in the camera frame of the first frame via the camera-IMU extrinsics from the config;
the twist is the integrated velocity and the latest bias-corrected gyroscope reading, moved from the IMU to the camera
centre (lever arm) and into camera axes. Expect the position to drift within a few seconds; that is the baseline every
visual method has to beat.
"""
import numpy as np

GRAVITY = 9.81
CALIBRATION_SECONDS = 0.5
# at rest a MEMS IMU only shows noise (per-sample standard deviations of ~0.03-0.3 m/s^2 and ~0.002-0.07 rad/s)
REST_ACCEL_STD = 1.0  # m/s^2, norm of the per-axis standard deviations
REST_GYRO_STD = 0.2  # rad/s


def quaternion_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


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


def exp_so3(omega):
    angle = np.linalg.norm(omega)
    if angle < 1e-12:
        return np.eye(3)
    k = omega / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class Solution:
    def __init__(self, config):
        self.T_cam_imu = np.asarray(config["T_cam_imu"], dtype=float)
        self.T_imu_cam = np.linalg.inv(self.T_cam_imu)
        self.calibration_samples = []
        self.initialized = False
        # state of the IMU/body frame in the "world" = body frame at the first frame (gravity-aligned later)
        self.R_wb = np.eye(3)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.omega = np.zeros(3)  # latest angular velocity of the IMU/body frame, body axes
        self.gyro_bias = np.zeros(3)
        self.gravity_world = np.array([0.0, 0.0, -GRAVITY])
        self.last_t = None
        self.T_w_c0 = None

    def _initialize(self):
        samples = np.asarray(self.calibration_samples)
        mean_accel = samples[:, 1:4].mean(axis=0)
        at_rest = np.linalg.norm(samples[:, 1:4].std(axis=0)) < REST_ACCEL_STD and np.linalg.norm(samples[:, 4:7].std(axis=0)) < REST_GYRO_STD
        self.gyro_bias = samples[:, 4:7].mean(axis=0) if at_rest else np.zeros(3)
        # choose the world frame so that gravity points down: measured specific force at rest = -g in body coordinates
        up_body = mean_accel / np.linalg.norm(mean_accel)
        z_world = up_body
        x_candidate = np.array([1.0, 0.0, 0.0])
        x_world = x_candidate - np.dot(x_candidate, z_world) * z_world
        x_world /= np.linalg.norm(x_world)
        y_world = np.cross(z_world, x_world)
        self.R_wb = np.stack([x_world, y_world, z_world])  # rows: world axes in body coordinates -> R_wb maps body to world
        self.v[:] = 0
        self.p[:] = 0
        self.initialized = True

    def _integrate(self, imu):
        for row in imu:
            t, accel, gyro = row[0], row[1:4], row[4:7]
            self.omega = gyro - self.gyro_bias
            if self.last_t is not None:
                dt = t - self.last_t
                if dt > 0:
                    self.R_wb = self.R_wb @ exp_so3((gyro - self.gyro_bias) * dt)
                    acceleration = self.R_wb @ accel + self.gravity_world
                    self.p += self.v * dt + 0.5 * acceleration * dt * dt
                    self.v += acceleration * dt
            self.last_t = t

    def _pose_camera(self):
        T_wb = np.eye(4)
        T_wb[:3, :3] = self.R_wb
        T_wb[:3, 3] = self.p
        return T_wb @ self.T_imu_cam

    def _twist_camera(self):
        """[v, w] of the camera in camera axes: the camera centre sits at r in IMU coordinates, so it moves with v + w x r."""
        R_cam_imu, r = self.T_cam_imu[:3, :3], self.T_imu_cam[:3, 3]
        velocity = R_cam_imu @ (self.R_wb.T @ self.v + np.cross(self.omega, r))  # world -> IMU axes, + lever arm, -> camera axes
        return np.concatenate([velocity, R_cam_imu @ self.omega])  # the angular velocity is the same everywhere on the body

    def step(self, t, frame, imu):
        if not self.initialized:
            self.calibration_samples.extend(imu.tolist())
            samples = self.calibration_samples
            if len(samples) < 2 or samples[-1][0] - samples[0][0] < CALIBRATION_SECONDS:
                return [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]  # still calibrating at rest: identity, standing still
            self._initialize()
            self.last_t = imu[-1, 0] if len(imu) else t
            self.T_w_c0 = self._pose_camera()
        else:
            self._integrate(imu)
        T_c0_c = np.linalg.inv(self.T_w_c0) @ self._pose_camera()
        return np.concatenate([T_c0_c[:3, 3], matrix_to_quaternion(T_c0_c[:3, :3]), self._twist_camera()])
