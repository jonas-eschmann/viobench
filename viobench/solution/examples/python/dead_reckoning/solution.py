"""Conservative IMU-only dead reckoning in the first camera frame.

The gyro tracks turns starting at frame zero. The opening half second estimates gravity and, for a quiet start,
gyro bias. Accelerometer readings near gravity gently correct tilt during the flight. Velocity decays without
supporting acceleration, trading accuracy during sustained motion for less drift from noisy accelerations.
"""
import numpy as np

GRAVITY = 9.81
CALIBRATION_SECONDS = 0.5
# at rest a MEMS IMU only shows noise (per-sample standard deviations of ~0.03-0.3 m/s^2 and ~0.002-0.07 rad/s)
REST_ACCEL_STD = 1.0  # m/s^2, norm of the per-axis standard deviations
REST_GYRO_STD = 0.2  # rad/s
MAX_GYRO_BIAS = 0.05  # rad/s; a larger mean rate is evidence of a turn even if it is steady
GRAVITY_TOLERANCE = 0.1 * GRAVITY  # trust tilt observations within 10% of gravity
TILT_SECONDS = 2.0
VELOCITY_SECONDS = 1.0


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
        self.R_wb = np.eye(3)  # "world" is the IMU frame at the first camera timestamp
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.omega = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.gravity = None  # specific force measured at rest, in world axes
        self.calibration_samples = []
        self.calibration_force = np.zeros(3)
        self.calibration_time = 0.0
        self.last_t = None

    def _calibrate(self, imu, force, dt):
        self.calibration_samples.extend(imu.tolist())
        self.calibration_force += force * dt
        self.calibration_time += dt
        if self.calibration_time < CALIBRATION_SECONDS:
            return
        mean_force = self.calibration_force / self.calibration_time
        magnitude = np.linalg.norm(mean_force)
        if abs(magnitude - GRAVITY) < GRAVITY_TOLERANCE:
            self.gravity = mean_force * (GRAVITY / magnitude)
            samples = np.asarray(self.calibration_samples)
            mean_gyro = samples[:, 4:7].mean(axis=0)
            if (np.linalg.norm(mean_gyro) < MAX_GYRO_BIAS and
                    np.linalg.norm(samples[:, 4:7].std(axis=0)) < REST_GYRO_STD and
                    np.linalg.norm(samples[:, 1:4].std(axis=0)) < REST_ACCEL_STD):
                self.gyro_bias = mean_gyro
                self.gravity = mean_force  # at rest this also absorbs the accelerometer offset along gravity
        # Try a fresh window if the specific force was too far from gravity to initialize tilt.
        self.calibration_samples.clear()
        self.calibration_force[:] = 0
        self.calibration_time = 0.0

    def _integrate(self, imu):
        start = self.last_t
        force = np.zeros(3)
        for row in imu:
            dt = row[0] - self.last_t
            if dt <= 0:
                continue
            self.last_t = row[0]
            self.omega = row[4:7] - self.gyro_bias
            self.R_wb = self.R_wb @ exp_so3(self.omega * dt)
            force += (self.R_wb @ row[1:4]) * dt
        dt = self.last_t - start
        if dt <= 0:
            return
        force /= dt  # frame-average specific force, rotated before averaging
        if self.gravity is None:
            self._calibrate(imu, force, dt)
            return

        magnitude = np.linalg.norm(force)
        if abs(magnitude - GRAVITY) < GRAVITY_TOLERANCE:
            # Complement the gyro's tilt estimate with the accelerometer. Gravity supplies no yaw observation.
            axis = np.cross(force / magnitude, self.gravity / np.linalg.norm(self.gravity))
            correction = exp_so3(axis * -np.expm1(-dt / TILT_SECONDS))
            self.R_wb = correction @ self.R_wb
            force = correction @ force

        acceleration = force - self.gravity
        gain = -np.expm1(-dt / VELOCITY_SECONDS)
        next_v = (1.0 - gain) * self.v + gain * VELOCITY_SECONDS * acceleration
        self.p += 0.5 * dt * (self.v + next_v)
        self.v = next_v

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
        if self.last_t is None:
            self.last_t = t  # frame zero defines the origin; its IMU samples precede it
            return [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]
        self._integrate(imu)
        T_c0_c = self.T_cam_imu @ self._pose_camera()
        return np.concatenate([T_c0_c[:3, 3], matrix_to_quaternion(T_c0_c[:3, :3]), self._twist_camera()])
