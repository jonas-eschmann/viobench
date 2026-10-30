// ORB-SLAM3 monocular-inertial tracking through the relpose ABI. See README.md.
#include "relpose_abi.h"
#include "relpose_slam.h"

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Vector = Eigen::Vector3d;
using Rotation = Eigen::Matrix3d;
using Measurement = ORB_SLAM3::IMU::Point;

struct Pose {
    Rotation rotation = Rotation::Identity();  // frame -> world
    Vector position = Vector::Zero();
};

struct State {
    std::unique_ptr<relpose_orbslam3::RelposeSlam> slam;
    Rotation R_camera_imu = Rotation::Identity();
    Vector p_camera_in_imu = Vector::Zero();
    int width = 0, height = 0;
    double first_time = 0, last_time = -std::numeric_limits<double>::infinity();
    double last_imu_time = -std::numeric_limits<double>::infinity();
    double next_warning = 5;
    bool started = false, anchored = false, lost = false;
    unsigned long map_id = 0;
    ORB_SLAM3::KeyFrame* origin = nullptr;
    std::vector<Measurement> history;
    std::array<double, RELPOSE_STATE_VALUES> last = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
};
State state;

double scalar(const cv::FileNode& node, const char* name) {
    if (!node.isInt() && !node.isReal()) throw std::runtime_error(std::string("missing or invalid ") + name);
    const double value = static_cast<double>(node);
    if (!std::isfinite(value)) throw std::runtime_error(std::string("nonfinite ") + name);
    return value;
}

std::string settings(const std::string& json) {
    cv::FileStorage config(json, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    const double width = scalar(config["width"], "width"), height = scalar(config["height"], "height");
    if (width != 320 || height != 240) throw std::runtime_error("expected 320 x 240 RGB input");
    state.width = static_cast<int>(width);
    state.height = static_cast<int>(height);
    const double fx = scalar(config["fx"], "fx"), fy = scalar(config["fy"], "fy");
    const double cx = scalar(config["cx"], "cx"), cy = scalar(config["cy"], "cy");
    if (fx <= 0 || fy <= 0) throw std::runtime_error("focal lengths must be positive");
    const auto transform = config["T_cam_imu"];
    if (!transform.isSeq() || transform.size() != 4) throw std::runtime_error("T_cam_imu must be a 4 x 4 matrix");
    Eigen::Matrix4d T_camera_imu;
    for (int row = 0; row < 4; ++row) {
        const auto entries = transform[row];
        if (!entries.isSeq() || entries.size() != 4) throw std::runtime_error("T_cam_imu must be a 4 x 4 matrix");
        for (int col = 0; col < 4; ++col) T_camera_imu(row, col) = scalar(entries[col], "T_cam_imu");
    }
    state.R_camera_imu = T_camera_imu.topLeftCorner<3, 3>();
    if ((T_camera_imu.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-6 ||
        (state.R_camera_imu.transpose() * state.R_camera_imu - Rotation::Identity()).norm() > 1e-5 ||
        std::abs(state.R_camera_imu.determinant() - 1) > 1e-5)
        throw std::runtime_error("T_cam_imu must be a rigid transform with a proper rotation");
    // ORB-SLAM3's legacy Tbc is camera -> body/IMU, the inverse of the ABI calibration.
    state.p_camera_in_imu = -state.R_camera_imu.transpose() * T_camera_imu.topRightCorner<3, 1>();
    Eigen::Matrix4d T_imu_camera = Eigen::Matrix4d::Identity();
    T_imu_camera.topLeftCorner<3, 3>() = state.R_camera_imu.transpose();
    T_imu_camera.topRightCorner<3, 1>() = state.p_camera_in_imu;
    std::ostringstream yaml;
    yaml.imbue(std::locale::classic());
    yaml << std::setprecision(17) << std::showpoint
         << "%YAML:1.0\nCamera.type: \"PinHole\"\n"
         << "Camera.fx: " << fx << "\nCamera.fy: " << fy << "\nCamera.cx: " << cx << "\nCamera.cy: " << cy
         << "\nCamera.k1: 0.0\nCamera.k2: 0.0\nCamera.p1: 0.0\nCamera.p2: 0.0\n"
         << "Camera.width: 320\nCamera.height: 240\nCamera.fps: 20.0\nCamera.RGB: 1\n"
         << "Tbc: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: f\n   data: [";
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col)
            yaml << (row || col ? ", " : "") << T_imu_camera(row, col);
    // The simulator specifies white noise per sample; ORB expects continuous noise densities.
    // Its bias random walks already are continuous densities (no sqrt(rate) conversion).
    yaml << "]\nIMU.Frequency: 200\nIMU.NoiseGyro: " << 0.07 / std::sqrt(200.0)
         << "\nIMU.NoiseAcc: " << 0.28 / std::sqrt(200.0)
         << "\nIMU.GyroWalk: 0.0006\nIMU.AccWalk: 0.001\n"
         << "ORBextractor.nFeatures: 1000\nORBextractor.scaleFactor: 1.2\nORBextractor.nLevels: 8\n"
         << "ORBextractor.iniThFAST: 20\nORBextractor.minThFAST: 7\n";
    return yaml.str();
}

Rotation rotation_increment(const Vector& angle) {
    const double norm = angle.norm();
    if (norm < 1e-12) return Rotation::Identity();
    return Eigen::AngleAxisd(norm, angle / norm).toRotationMatrix();
}

Pose camera_pose(const Rotation& R_world_imu, const Vector& p_world_imu) {
    return {R_world_imu * state.R_camera_imu.transpose(), p_world_imu + R_world_imu * state.p_camera_in_imu};
}

// Recover the actual episode's first exposure, which can precede ORB's first visual keyframe.
// Reading the origin keyframe again also follows later inertial scale/gravity/bias refinements.
Pose first_camera_pose() {
    auto& keyframe = *state.origin;
    Rotation rotation = keyframe.GetImuRotation().cast<double>();
    Vector position = keyframe.GetImuPosition().cast<double>();
    Vector velocity = keyframe.GetVelocity().cast<double>();
    const auto bias = keyframe.GetImuBias();
    const Vector bias_g(bias.bwx, bias.bwy, bias.bwz), bias_a(bias.bax, bias.bay, bias.baz);
    const Vector gravity(0, 0, -9.81);
    double cursor = keyframe.mTimeStamp;
    for (size_t j = state.history.size(); j-- > 1;) {
        const auto& before = state.history[j - 1];
        const auto& after = state.history[j];
        const double start = std::max(before.t, state.first_time), end = std::min(after.t, cursor);
        if (end <= start) continue;
        if (cursor - end > 1e-7) throw std::runtime_error("missing startup IMU interval");
        const double dt = end - start;
        const double fraction = (0.5 * (start + end) - before.t) / (after.t - before.t);
        const Vector omega = (1 - fraction) * before.w.cast<double>() + fraction * after.w.cast<double>() - bias_g;
        const Vector specific_force = (1 - fraction) * before.a.cast<double>() + fraction * after.a.cast<double>() - bias_a;
        const Rotation earlier = rotation * rotation_increment(-omega * dt);
        const Vector acceleration = 0.5 * (earlier + rotation) * specific_force + gravity;
        position -= velocity * dt - 0.5 * acceleration * dt * dt;
        velocity -= acceleration * dt;
        rotation = earlier;
        cursor = start;
    }
    if (cursor - state.first_time > 1e-7) throw std::runtime_error("startup IMU does not reach the first camera frame");
    return camera_pose(rotation, position);
}

void hold(double* out) {
    std::fill(state.last.begin() + RELPOSE_POSE_VALUES, state.last.end(), 0.0);
    std::copy(state.last.begin(), state.last.end(), out);
}
}  // namespace

extern "C" {
RELPOSE_EXPORT(abi_version) int32_t abi_version() { return RELPOSE_ABI_VERSION; }
RELPOSE_EXPORT(frame_width) int32_t frame_width() { return 320; }
RELPOSE_EXPORT(alloc) void* alloc(uint32_t size) { return std::malloc(size); }

RELPOSE_EXPORT(init) int32_t init(const char* json, uint32_t length) {
    try {
        if (state.slam) throw std::runtime_error("init called twice");
        if (!json || !length) throw std::runtime_error("empty configuration");
        const auto yaml = settings(std::string(json, length));
        cv::setNumThreads(1);
        cv::setRNGSeed(0);
        state.slam = std::make_unique<relpose_orbslam3::RelposeSlam>(yaml);
        return RELPOSE_STATUS_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "orbslam3: init: %s\n", error.what());
        return RELPOSE_STATUS_ERROR;
    }
}

RELPOSE_EXPORT(step) int32_t step(double time, const uint8_t* pixels, const double* imu, int32_t count, double* out) {
    try {
        if (!state.slam || !pixels || !out || !imu || count < 1 || !std::isfinite(time) || time <= state.last_time)
            throw std::runtime_error("invalid frame, timestamp, or empty IMU batch");
        if (!state.started) {
            state.first_time = time;
            state.started = true;
        }
        std::vector<Measurement> measurements;
        measurements.reserve(count + 1);
        for (int i = 0; i < count; ++i) {
            const double* row = imu + i * RELPOSE_IMU_ROW_VALUES;
            for (int j = 0; j < RELPOSE_IMU_ROW_VALUES; ++j)
                if (!std::isfinite(row[j])) throw std::runtime_error("nonfinite IMU measurement");
            if (row[0] <= state.last_imu_time || row[0] <= state.last_time || row[0] > time + 1e-8)
                throw std::runtime_error("IMU timestamps must increase and not exceed the exposure");
            measurements.emplace_back(row[1], row[2], row[3], row[4], row[5], row[6], row[0]);
            state.last_imu_time = row[0];
        }
        // Supply the fractional interval causally if exposure falls between samples. The next real
        // sample follows this synthetic endpoint, so ORB's queue remains strictly chronological.
        if (time - measurements.back().t > 0.005001) throw std::runtime_error("IMU does not cover the exposure");
        if (time - measurements.back().t > 1e-8) {
            auto endpoint = measurements.back();
            endpoint.t = time;
            measurements.push_back(endpoint);
        }
        if (!state.anchored) state.history.insert(state.history.end(), measurements.begin(), measurements.end());
        cv::Mat rgb(state.height, state.width, CV_8UC3, const_cast<uint8_t*>(pixels)), gray;
        cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
        state.slam->track(gray, time, measurements);
        state.last_time = time;
        auto& tracker = state.slam->tracker();
        auto* map = state.slam->atlas().GetCurrentMap();
        if (state.anchored && (map->GetId() != state.map_id || !map->isImuInitialized()))
            throw std::runtime_error("lost the initialized map; cannot preserve the first-camera coordinate frame");
        auto& frame = tracker.mCurrentFrame;
        const bool tracked = tracker.mState == ORB_SLAM3::Tracking::OK && frame.HasPose() && frame.HasVelocity();
        if (!map->isImuInitialized() || !tracked) {
            if (!state.anchored && time - state.first_time >= state.next_warning) {
                std::fprintf(stderr, "orbslam3: still initializing after %.1f s; metric position and velocity unavailable\n",
                             time - state.first_time);
                state.next_warning += 5;
            }
            if (state.anchored && !state.lost) {
                std::fprintf(stderr, "orbslam3: tracking lost at t=%.3f; holding the last pose\n", time);
                state.lost = true;
            }
            hold(out);
            return RELPOSE_STATUS_OK;
        }
        if (!state.anchored) {
            state.origin = map->GetOriginKF();
            if (!state.origin || state.origin->isBad()) throw std::runtime_error("missing map origin keyframe");
            state.origin->SetNotErase();
            state.map_id = map->GetId();
            state.anchored = true;
            // Retain only the startup history needed to reconstruct camera0 from this keyframe.
            const auto end = std::upper_bound(state.history.begin(), state.history.end(), state.origin->mTimeStamp,
                                             [](double t, const Measurement& sample) { return t < sample.t; });
            state.history.erase(end, state.history.end());
            std::printf("orbslam3: initialized at t=%.3f (origin t=%.3f)\n", time, state.origin->mTimeStamp);
        }
        if (state.lost) std::printf("orbslam3: tracking recovered at t=%.3f\n", time);
        state.lost = false;
        const Pose camera0 = first_camera_pose();
        const Rotation R_world_imu = frame.GetImuRotation().cast<double>();
        const Pose camera = camera_pose(R_world_imu, frame.GetImuPosition().cast<double>());
        const Vector position = camera0.rotation.transpose() * (camera.position - camera0.position);
        Eigen::Quaterniond quaternion(camera0.rotation.transpose() * camera.rotation);
        quaternion.normalize();
        if (quaternion.w() < 0) quaternion.coeffs() *= -1;
        const auto bias = frame.mImuBias;
        const Vector omega = measurements.back().w.cast<double>() - Vector(bias.bwx, bias.bwy, bias.bwz);
        const Vector velocity = camera.rotation.transpose() * frame.GetVelocity().cast<double>() +
                                state.R_camera_imu * omega.cross(state.p_camera_in_imu);
        const Vector angular_velocity = state.R_camera_imu * omega;
        for (int i = 0; i < 3; ++i) {
            state.last[i] = position[i];
            state.last[i + 7] = velocity[i];
            state.last[i + 10] = angular_velocity[i];
        }
        for (int i = 0; i < 4; ++i) state.last[i + 3] = quaternion.coeffs()[i];
        for (double value : state.last)
            if (!std::isfinite(value)) throw std::runtime_error("estimator returned a nonfinite state");
        std::copy(state.last.begin(), state.last.end(), out);
        return RELPOSE_STATUS_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "orbslam3: step: %s\n", error.what());
        return RELPOSE_STATUS_ERROR;
    }
}
}  // extern "C"
