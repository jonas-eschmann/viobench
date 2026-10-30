// VINS-Mono sliding-window visual-inertial odometry through the relpose ABI (GPL-3.0).
#include "relpose_abi.h"
#include "estimator.h"
#include "feature_tracker.h"

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

struct Measurement {
    double time;
    Vec3 acceleration, gyro;
};
struct Pose {
    Mat3 rotation = Mat3::Identity();  // camera -> estimator world
    Vec3 position = Vec3::Zero();
};
struct State {
    std::unique_ptr<Estimator> estimator;
    relpose_vins_mono::FeatureTracker tracker;
    Mat3 R_camera_imu = Mat3::Identity();
    Vec3 p_camera_in_imu = Vec3::Zero();
    int width = 0, height = 0;
    double first_time = 0, last_time = -std::numeric_limits<double>::infinity();
    double integration_time = -std::numeric_limits<double>::infinity();
    double last_imu_time = -std::numeric_limits<double>::infinity();
    double next_warning = 5;
    bool started = false, anchored = false, output_ready = false;
    bool have_previous_imu = false;
    Measurement previous_imu;
    Mat3 tracking_rotation = Mat3::Identity();  // previous IMU bearings -> current IMU bearings
    Mat3 initial_rotation = Mat3::Identity();   // current IMU axes -> first exposure IMU axes
    double reference_time = 0;
    bool refining_reference = false;
    Pose camera0;
    std::vector<Measurement> history;
};
State state;

double scalar(const cv::FileNode& node, const char* name) {
    if (!node.isInt() && !node.isReal()) throw std::runtime_error(std::string("missing or invalid ") + name);
    const double value = static_cast<double>(node);
    if (!std::isfinite(value)) throw std::runtime_error(std::string("nonfinite ") + name);
    return value;
}

void configure(const std::string& json) {
    cv::FileStorage config(json, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    if (scalar(config["width"], "width") != 320 || scalar(config["height"], "height") != 240)
        throw std::runtime_error("expected 320 x 240 RGB input");
    state.width = 320;
    state.height = 240;
    auto& tracker = state.tracker;
    tracker.fx = scalar(config["fx"], "fx");
    tracker.fy = scalar(config["fy"], "fy");
    tracker.cx = scalar(config["cx"], "cx");
    tracker.cy = scalar(config["cy"], "cy");
    if (tracker.fx <= 0 || tracker.fy <= 0) throw std::runtime_error("focal lengths must be positive");
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
        (state.R_camera_imu.transpose() * state.R_camera_imu - Mat3::Identity()).norm() > 1e-5 ||
        std::abs(state.R_camera_imu.determinant() - 1) > 1e-5)
        throw std::runtime_error("T_cam_imu must be a rigid transform with a proper rotation");
    state.p_camera_in_imu = -state.R_camera_imu.transpose() * T_camera_imu.topRightCorner<3, 1>();
    RIC = {state.R_camera_imu.transpose()};
    TIC = {state.p_camera_in_imu};
    ROW = state.height;
    COL = state.width;
    ESTIMATE_EXTRINSIC = 0;
    ESTIMATE_TD = 0;
    ROLLING_SHUTTER = 0;
    TD = TR = 0;
    MIN_PARALLAX = 8.0 / std::min(tracker.fx, tracker.fy);
}

Mat3 increment(const Vec3& angle) {
    const double norm = angle.norm();
    if (norm < 1e-12) return Mat3::Identity();
    return Eigen::AngleAxisd(norm, angle / norm).toRotationMatrix();
}

Pose camera_pose(const Mat3& rotation, const Vec3& position) {
    return {rotation * state.R_camera_imu.transpose(), position + rotation * state.p_camera_in_imu};
}

// Initialization may discard early images. Integrate backwards from a retained optimized
// IMU state to the first input exposure, so the ABI origin never becomes a later keyframe.
Pose first_camera_pose(int reference) {
    const auto& estimator = *state.estimator;
    Mat3 rotation = estimator.Rs[reference];
    Vec3 position = estimator.Ps[reference], velocity = estimator.Vs[reference];
    const Vec3 bias_g = estimator.Bgs[reference], bias_a = estimator.Bas[reference];
    double cursor = estimator.Headers[reference].stamp.toSec();
    for (size_t j = state.history.size(); j-- > 1;) {
        const auto& before = state.history[j - 1];
        const auto& after = state.history[j];
        const double start = std::max(before.time, state.first_time), end = std::min(after.time, cursor);
        if (end <= start) continue;
        if (cursor - end > 1e-7) throw std::runtime_error("missing startup IMU interval");
        const double dt = end - start;
        const double fraction = (0.5 * (start + end) - before.time) / (after.time - before.time);
        const Vec3 omega = (1 - fraction) * before.gyro + fraction * after.gyro - bias_g;
        const Vec3 force = (1 - fraction) * before.acceleration + fraction * after.acceleration - bias_a;
        const Mat3 earlier = rotation * increment(-omega * dt);
        const Vec3 acceleration = 0.5 * (earlier + rotation) * force - estimator.g;
        position -= velocity * dt - 0.5 * acceleration * dt * dt;
        velocity -= acceleration * dt;
        rotation = earlier;
        cursor = start;
    }
    if (cursor - state.first_time > 1e-7) throw std::runtime_error("startup IMU does not reach the first exposure");
    return camera_pose(rotation, position);
}

void feed_imu(const Measurement& measurement) {
    const double dt = std::isfinite(state.integration_time) ? measurement.time - state.integration_time : 0.0;
    Vec3 bias = Vec3::Zero();
    if (state.estimator->solver_flag == Estimator::NON_LINEAR) bias = state.estimator->Bgs[WINDOW_SIZE];
    if (state.have_previous_imu) {
        const Vec3 omega = 0.5 * (state.previous_imu.gyro + measurement.gyro) - bias;
        state.tracking_rotation = increment(-omega * dt) * state.tracking_rotation;
        const double attitude_dt = measurement.time - std::max(state.integration_time, state.first_time);
        if (!state.output_ready && attitude_dt > 0) state.initial_rotation *= increment(omega * attitude_dt);
    }
    state.estimator->processIMU(dt, measurement.acceleration, measurement.gyro);
    state.integration_time = measurement.time;
    state.previous_imu = measurement;
    state.have_previous_imu = true;
    if (!state.output_ready) state.history.push_back(measurement);
}
}  // namespace

extern "C" {
RELPOSE_EXPORT(abi_version) int32_t abi_version() { return RELPOSE_ABI_VERSION; }
RELPOSE_EXPORT(frame_width) int32_t frame_width() { return 320; }
RELPOSE_EXPORT(alloc) void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }

RELPOSE_EXPORT(init) int32_t init(const char* json, uint32_t length) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    try {
        if (state.estimator) throw std::runtime_error("init called twice");
        if (!json || !length) throw std::runtime_error("empty configuration");
        configure(std::string(json, length));
        cv::setNumThreads(1);
        cv::setRNGSeed(0);
        state.estimator = std::make_unique<Estimator>();
        state.estimator->setParameter();
        // VINS residuals use normalized image coordinates; apply this episode's intrinsics.
        ProjectionFactor::sqrt_info = Eigen::Vector2d(state.tracker.fx / 1.5, state.tracker.fy / 1.5).asDiagonal();
        ProjectionTdFactor::sqrt_info = ProjectionFactor::sqrt_info;
        std::printf("vins_mono: %dx%d fx=%.1f fy=%.1f\n", state.width, state.height, state.tracker.fx, state.tracker.fy);
        return RELPOSE_STATUS_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "vins_mono: init: %s\n", error.what());
        return RELPOSE_STATUS_ERROR;
    }
}

RELPOSE_EXPORT(step) int32_t step(double time, const uint8_t* pixels, const double* imu, int32_t count, double* out) {
    try {
        if (!state.estimator || !pixels || !out || !imu || count < 1 || !std::isfinite(time) || time <= state.last_time)
            throw std::runtime_error("invalid frame, timestamp, or empty IMU batch");
        if (!state.started) {
            state.first_time = time;
            state.started = true;
        }
        Measurement latest;
        Vec3 gyro_mean = Vec3::Zero();
        for (int i = 0; i < count; ++i) {
            const double* row = imu + i * RELPOSE_IMU_ROW_VALUES;
            for (int j = 0; j < RELPOSE_IMU_ROW_VALUES; ++j)
                if (!std::isfinite(row[j])) throw std::runtime_error("nonfinite IMU measurement");
            if (row[0] <= state.last_imu_time || row[0] <= state.last_time || row[0] > time + 1e-8)
                throw std::runtime_error("IMU timestamps must increase and not exceed the exposure");
            latest = {row[0], Vec3(row[1], row[2], row[3]), Vec3(row[4], row[5], row[6])};
            gyro_mean += latest.gyro / count;
            feed_imu(latest);
            state.last_imu_time = row[0];
        }
        // Use only causal measurements; holding the newest sample covers at most one IMU period.
        if (time - latest.time > 0.005001) throw std::runtime_error("IMU does not cover the exposure");
        if (time - latest.time > 1e-8) {
            latest.time = time;
            feed_imu(latest);
        }
        cv::Mat rgb(state.height, state.width, CV_8UC3, const_cast<uint8_t*>(pixels)), gray;
        cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
        const auto observations = state.tracker.process(gray, time,
            state.R_camera_imu * state.tracking_rotation * state.R_camera_imu.transpose());
        state.tracking_rotation.setIdentity();
        std_msgs::Header header;
        header.stamp = ros::Time(time);
        auto& estimator = *state.estimator;
        estimator.processImage(observations, header);
        state.last_time = time;
        if (estimator.solver_flag != Estimator::NON_LINEAR) {
            if (state.output_ready) throw std::runtime_error("estimator reset after initialization; original camera frame was lost");
            state.anchored = false;
            state.refining_reference = false;
        } else {
            if (!state.anchored) {
                // Dynamic initialization's first scale/velocity estimate is tentative. Keep
                // a named reference in the lag window and refine camera0 with that state's
                // posterior until it leaves the window. Its integration interval never grows.
                const int reference = estimator.static_initialized ? 0 : WINDOW_SIZE - 2;
                state.reference_time = estimator.Headers[reference].stamp.toSec();
                state.camera0 = first_camera_pose(reference);
                state.anchored = true;
                state.refining_reference = !estimator.static_initialized;
            }
            if (state.refining_reference) {
                int reference = -1;
                for (int i = 0; i <= WINDOW_SIZE; ++i)
                    if (estimator.Headers[i].stamp.toSec() == state.reference_time) reference = i;
                if (reference >= 0) state.camera0 = first_camera_pose(reference);
                else state.refining_reference = false;
            }
            if (!state.refining_reference && !state.output_ready) {
                // A stationary initializer has a metric zero-velocity constraint. A dynamic
                // initializer instead waits for its reference to finish fixed-lag smoothing;
                // publishing its provisional scale would create large fictitious motion.
                state.output_ready = true;
                state.history.clear();
                state.history.shrink_to_fit();
                std::printf("vins_mono: initialized at t=%.3f (origin t=%.3f)\n", time, state.first_time);
            }
        }
        if (!state.output_ready) {
            if (time - state.first_time >= state.next_warning) {
                std::fprintf(stderr, "vins_mono: still initializing after %.1f s; metric translation unavailable, reporting gyro attitude\n",
                             time - state.first_time);
                state.next_warning += 5;
            }
            std::fill(out, out + RELPOSE_STATE_VALUES, 0.0);
            Eigen::Quaterniond orientation(state.R_camera_imu * state.initial_rotation * state.R_camera_imu.transpose());
            orientation.normalize();
            const Vec3 lever_position = state.R_camera_imu * (state.initial_rotation * state.p_camera_in_imu - state.p_camera_in_imu);
            Vec3 omega = gyro_mean;
            if (estimator.solver_flag == Estimator::NON_LINEAR) omega -= estimator.Bgs[WINDOW_SIZE];
            const Vec3 angular_velocity = state.R_camera_imu * omega;
            const Vec3 lever_velocity = state.R_camera_imu * omega.cross(state.p_camera_in_imu);
            for (int i = 0; i < 3; ++i) {
                out[i] = lever_position[i];
                out[i + 7] = lever_velocity[i];
                out[i + 10] = angular_velocity[i];
            }
            for (int i = 0; i < 4; ++i) out[i + 3] = orientation.coeffs()[i];
            return RELPOSE_STATUS_OK;
        }
        const Pose camera = camera_pose(estimator.Rs[WINDOW_SIZE], estimator.Ps[WINDOW_SIZE]);
        const Vec3 position = state.camera0.rotation.transpose() * (camera.position - state.camera0.position);
        Eigen::Quaterniond orientation(state.camera0.rotation.transpose() * camera.rotation);
        orientation.normalize();
        if (orientation.w() < 0) orientation.coeffs() *= -1;
        const Vec3 omega = gyro_mean - estimator.Bgs[WINDOW_SIZE];
        const Vec3 velocity = camera.rotation.transpose() * estimator.Vs[WINDOW_SIZE] +
                                state.R_camera_imu * omega.cross(state.p_camera_in_imu);
        const Vec3 angular_velocity = state.R_camera_imu * omega;
        for (int i = 0; i < 3; ++i) {
            out[i] = position[i];
            out[i + 7] = velocity[i];
            out[i + 10] = angular_velocity[i];
        }
        for (int i = 0; i < 4; ++i) out[i + 3] = orientation.coeffs()[i];
        for (int i = 0; i < RELPOSE_STATE_VALUES; ++i)
            if (!std::isfinite(out[i])) throw std::runtime_error("estimator returned a nonfinite state");
        return RELPOSE_STATUS_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "vins_mono: step at t=%.3f: %s\n", time, error.what());
        return RELPOSE_STATUS_ERROR;
    }
}
}  // extern "C"
