// OpenVINS (github.com/rpng/open_vins, GPL-3.0) as a viobench solution: KLT tracking, rotation-aware startup,
// and Ceres fixed-lag visual-inertial smoothing. This file adapts the ABI (viobench_abi.h) to it: the
// configuration from the init() JSON, IMU rows and the frame per step, and the estimated IMU state re-expressed as the pose and
// twist of the camera relative to the first frame.
// Build:  viobench/solution/runtime/native/build.sh viobench/solution/examples/native/openvins   ->   build/solution.wasm
#include "viobench_abi.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include "consensus_output.h"
#include <opencv2/imgproc.hpp>

#include "cam/CamRadtan.h"
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "rotation_update.h"
#include "fixed_lag_vio.h"
#include "track/TrackBase.h"
#include "track/TrackKLT.h"
#include "types/IMU.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

namespace {

// The numbers that follow `"key":` in the flat config JSON, nested arrays flattened (T_cam_imu row by row).
std::vector<double> numbers(const std::string& json, const char* key, size_t count) {
    std::vector<double> values;
    size_t at = json.find(std::string("\"") + key + "\"");
    if (at == std::string::npos || (at = json.find(':', at)) == std::string::npos) return values;
    const char* cursor = json.c_str() + at + 1;
    while (values.size() < count && *cursor) {
        if (*cursor == '-' || *cursor == '.' || (*cursor >= '0' && *cursor <= '9')) {
            char* end;
            values.push_back(std::strtod(cursor, &end));
            cursor = end;
        } else if (std::strchr("[], \t\r\n", *cursor)) {
            cursor++;
        } else {
            break;
        }
    }
    return values;
}

// The simulator's IMU preset (viobench/tools/abi/config.py DEFAULT_IMU). White noise
// is per sample at 200 Hz; bias random walks are already continuous-time densities.
ov_msckf::VioManagerOptions make_options(int width, int height, const double intrinsics[4], const Eigen::Matrix4d& T_cam_imu) {
    ov_msckf::VioManagerOptions options;
    const double imu_rate = 200.0;
    options.imu_noises.sigma_w = 0.07 / std::sqrt(imu_rate);
    options.imu_noises.sigma_a = 0.28 / std::sqrt(imu_rate);
    options.imu_noises.sigma_wb = 0.0006;
    options.imu_noises.sigma_ab = 0.001;
    options.imu_noises.sigma_w_2 = std::pow(options.imu_noises.sigma_w, 2);
    options.imu_noises.sigma_a_2 = std::pow(options.imu_noises.sigma_a, 2);
    options.imu_noises.sigma_wb_2 = std::pow(options.imu_noises.sigma_wb, 2);
    options.imu_noises.sigma_ab_2 = std::pow(options.imu_noises.sigma_ab, 2);
    for (ov_msckf::UpdaterOptions* updater : {&options.msckf_options, &options.slam_options, &options.aruco_options}) {
        updater->sigma_pix = 0.5;
        updater->sigma_pix_sq = updater->sigma_pix * updater->sigma_pix;
        updater->chi2_multipler = 1.0;
    }
    options.zupt_options.chi2_multipler = 0.0;  // zero-velocity detection by disparity only

    // estimator (OpenVINS' EuRoC configuration, without online calibration: the intrinsics and extrinsics are exact)
    ov_msckf::StateOptions& state = options.state_options;
    state.num_cameras = 1;
    state.do_fej = true;
    state.integration_method = ov_msckf::StateOptions::IntegrationMethod::RK4;
    state.max_clone_size = 11;
    state.max_slam_features = 50;
    state.max_slam_in_update = 25;
    state.max_msckf_in_update = 40;
    state.feat_rep_msckf = ov_type::LandmarkRepresentation::Representation::GLOBAL_3D;
    state.feat_rep_slam = ov_type::LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    options.dt_slam_delay = 1.0;
    options.gravity_mag = 9.81;
    // Permit static initialization without waiting for an acceleration jerk. The
    // wrapper replaces the ordinary ZUPT with a rotation-aware startup update.
    options.try_zupt = true;
    options.zupt_only_at_beginning = true;

    // tracking
    options.use_stereo = false;
    options.use_klt = true;
    options.use_aruco = false;
    options.num_opencv_threads = 1;
    options.use_multi_threading_pubs = false;
    options.use_multi_threading_subs = false;
    options.num_pts = 150;
    options.fast_threshold = 15;
    options.grid_x = 5;
    options.grid_y = 4;
    options.min_px_dist = 8;
    options.histogram_method = ov_core::TrackBase::HistogramMethod::NONE;

    // ideal IMU intrinsics: identity scale and misalignment (Kalibr layout), no gravity sensitivity
    options.vec_dw << 1, 0, 0, 1, 0, 1;
    options.vec_da << 1, 0, 0, 1, 0, 1;
    options.vec_tg.setZero();
    options.q_ACCtoIMU << 0, 0, 0, 1;
    options.q_GYROtoIMU << 0, 0, 0, 1;

    // camera: pinhole (radtan without distortion), extrinsics as OpenVINS stores them: JPL q_ItoC and p_IinC
    auto camera = std::make_shared<ov_core::CamRadtan>(width, height);
    Eigen::VectorXd calibration(8);
    calibration << intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], 0, 0, 0, 0;
    camera->set_value(calibration);
    Eigen::VectorXd extrinsics(7);
    extrinsics << ov_core::rot_2_quat(T_cam_imu.block<3, 3>(0, 0)), T_cam_imu.block<3, 1>(0, 3);
    options.camera_intrinsics.insert({0, camera});
    options.camera_extrinsics.insert({0, extrinsics});

    // initialization: static while hovering (both halves of the window still), else dynamic (Ceres MLE, single-threaded and
    // bounded by iterations only, so the fuel does not depend on the machine)
    ov_init::InertialInitializerOptions& init = options.init_options;
    // OpenVINS also requires 0.1 s of history before trying this window. Leave time to initialize during the
    // simulator's one-second startup hover; both image disparity and IMU variance must still pass the static checks.
    init.init_window_time = 0.8;
    init.init_imu_thresh = 1.0;
    init.init_max_disparity = 1.5;
    init.init_max_features = 75;
    // A uniform initial bias in [-0.2, 0.2] m/s^2 cannot be distinguished from tilt
    // in one stationary pose; retain its prior uncertainty for subsequent updates.
    init.init_static_sigma_ba = 0.2 / std::sqrt(3.0);
    init.init_dyn_use = true;
    init.init_dyn_mle_opt_calib = false;
    init.init_dyn_mle_max_iter = 50;
    init.init_dyn_mle_max_time = 1e9;
    init.init_dyn_mle_max_threads = 1;
    init.init_dyn_num_pose = 6;
    init.init_dyn_min_deg = 5.0;
    init.init_dyn_inflation_orientation = 10.0;
    init.init_dyn_inflation_velocity = 100.0;
    init.init_dyn_inflation_bias_gyro = 10.0;
    init.init_dyn_inflation_bias_accel = 100.0;
    init.init_dyn_min_rec_cond = 1e-12;
    init.sigma_w = options.imu_noises.sigma_w;
    init.sigma_wb = options.imu_noises.sigma_wb;
    init.sigma_a = options.imu_noises.sigma_a;
    init.sigma_ab = options.imu_noises.sigma_ab;
    init.sigma_pix = 1.0;
    init.gravity_mag = options.gravity_mag;
    init.num_cameras = 1;
    init.use_stereo = false;
    init.camera_intrinsics = options.camera_intrinsics;
    init.camera_extrinsics = options.camera_extrinsics;
    return options;
}

// VioManager keeps "has an initial state" protected; initialized() becomes true only after the first visual update
struct Vio : ov_msckf::VioManager {
    Vio(ov_msckf::VioManagerOptions& options, bool use_anchor, bool use_conservative)
        : VioManager(options), anchored_tracking(use_anchor) {
        if (use_conservative) {
            conservative = std::make_unique<viobench_openvins::FixedLagVio>();
            conservative->pixel_sigma = 1.5;
            conservative->huber_scale = 1.0;
        }
        for (auto* graph : {&smoother, conservative.get()}) {
            if (!graph) continue;
            graph->sigma_w = options.imu_noises.sigma_w;
            graph->sigma_wb = options.imu_noises.sigma_wb;
            graph->sigma_a = options.imu_noises.sigma_a;
            graph->sigma_ab = options.imu_noises.sigma_ab;
            graph->gravity = Eigen::Vector3d(0, 0, options.gravity_mag);
        }
    }
    bool started() const { return is_initialized_vio; }
    RotationOnlyStartup rotation_startup;
    viobench_openvins::FixedLagVio smoother;
    std::unique_ptr<viobench_openvins::FixedLagVio> conservative;
    bool anchored_tracking = false;
    // Rotation of IMU bearings between images; remove estimated gyro bias after initialization.
    Eigen::Matrix3d tracking_rotation = Eigen::Matrix3d::Identity();
    ov_core::ImuData previous_imu;
    bool have_previous_imu = false;
    void feed_measurement_imu(const ov_core::ImuData& measurement) {
        smoother.feed_imu(measurement);
        if (conservative) conservative->feed_imu(measurement);
        if (have_previous_imu) {
            Eigen::Vector3d bias = Eigen::Vector3d::Zero();
            if (is_initialized_vio) bias = get_state()->_imu->bias_g();
            const Eigen::Vector3d omega = 0.5 * (previous_imu.wm + measurement.wm) - bias;
            tracking_rotation = ov_core::exp_so3(-omega * (measurement.timestamp - previous_imu.timestamp)) * tracking_rotation;
        }
        previous_imu = measurement;
        have_previous_imu = true;
        VioManager::feed_measurement_imu(measurement);
    }
    void feed_measurement_camera(const ov_core::CameraData& message) {
        rT1 = boost::posix_time::microsec_clock::local_time();
        const Eigen::Matrix3d R_ItoC = get_state()->_calib_IMUtoCAM.at(0)->Rot();
        std::dynamic_pointer_cast<ov_core::TrackKLT>(trackFEATS)->set_temporal_rotation(
            0, R_ItoC * tracking_rotation * R_ItoC.transpose());
        trackFEATS->feed_new_camera(message);
        tracking_rotation.setIdentity();
        rT2 = boost::posix_time::microsec_clock::local_time();
        if (!is_initialized_vio) {
            is_initialized_vio = try_to_initialize(message);
            if (!is_initialized_vio) return;
        }
        has_moved_since_zupt = true;
        updaterZUPT.reset();
        const bool was_active = rotation_startup.active();
        if (rotation_startup.update(get_state(), get_propagator(), trackFEATS->get_feature_database(), message.timestamp)) return;
        if (was_active) {
            // Give translating tracks a full window before adding persistent landmarks.
            startup_time = message.timestamp;
            std::printf("openvins: full VIO at %.3f after %d rotation updates (rms %.3f, inliers %.2f)\n",
                        message.timestamp, rotation_startup.updates(), rotation_startup.fit_rms_pixels(),
                        rotation_startup.inlier_fraction());
        }
        // Bootstrap once, then optimize raw IMU and image factors independently.
        // The fast EKF remains active to supply the gyro bias for image warping.
        if (!smoother.initialized()) {
            // Both chain graphs copy the same untouched EKF state/covariance.
            // Subsequent optimization only consumes their own raw factor copies.
            smoother.initialize(get_state());
            if (conservative) conservative->initialize(get_state());
            if (anchored_tracking)
                std::dynamic_pointer_cast<ov_core::TrackKLT>(trackFEATS)->enable_temporal_anchor(message.timestamp, smoother.keyframe_interval);
        }
        smoother.feed_camera(message.timestamp, trackFEATS->get_feature_database());
        if (conservative) conservative->feed_camera(message.timestamp, trackFEATS->get_feature_database());
        if (smoother.failed() || (conservative && conservative->failed()))
            throw std::runtime_error("fixed-lag estimator could not maintain finite state constraints");
        do_feature_propagate_update(message);
    }
};

struct Pose {  // a frame in a world frame: p_world = R p_frame + p
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

struct State {
    std::unique_ptr<Vio> vio;
    Eigen::Matrix3d R_ItoC;  // camera axes from IMU axes
    Eigen::Vector3d p_CinI;  // camera centre in the IMU frame
    cv::Mat mask;            // all zero: every pixel may hold a feature
    int width = 0, height = 0;
    double gravity = 9.81;
    long frames = 0;
    double t0 = 0;
    double next_init_warning = 5.0;
    std::vector<ov_core::ImuData> imu;  // every sample since the first frame, until the first camera pose is known
    Eigen::Matrix3d R_I0_I = Eigen::Matrix3d::Identity();  // gyro attitude before initialization
    bool anchored = false;
    Pose camera0;  // the first camera frame in the world frame: the first IMU frame until initialization, then OpenVINS' global frame
    double last[VIOBENCH_STATE_VALUES] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
    double last_conservative[VIOBENCH_STATE_VALUES] = {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
};

State chain_state, anchor_state;
ConsensusOutput consensus;

Pose camera_pose(const State& state, const Eigen::Matrix3d& R_world_imu, const Eigen::Vector3d& p_imu) {
    return {R_world_imu * state.R_ItoC.transpose(), p_imu + R_world_imu * state.p_CinI};
}

// The IMU pose at the first frame, integrating the measurements back from the initial state (time, R_GtoI, p_IinG, v_IinG,
// biases) with the propagator's mean model; OpenVINS starts its window later than the first frame.
Pose imu_pose_at_first_frame(const State& state, double time, Eigen::Matrix3d R_GtoI, Eigen::Vector3d p, Eigen::Vector3d v, const Eigen::Vector3d& bias_g,
                             const Eigen::Vector3d& bias_a, double gravity) {
    const Eigen::Vector3d g(0, 0, gravity);
    for (size_t j = state.imu.size(); j-- > 1;) {
        const ov_core::ImuData &before = state.imu[j - 1], &after = state.imu[j];
        const double start = std::max(before.timestamp, state.t0), end = std::min(after.timestamp, time);
        if (end <= start) continue;
        const double dt = end - start;
        const Eigen::Vector3d w = 0.5 * (before.wm + after.wm) - bias_g, a = 0.5 * (before.am + after.am) - bias_a;
        R_GtoI = ov_core::exp_so3(w * dt) * R_GtoI;  // inverse of the forward step R_GtoI <- exp(-w dt) R_GtoI
        const Eigen::Vector3d acceleration = R_GtoI.transpose() * a - g;
        v -= acceleration * dt;
        p -= v * dt + 0.5 * acceleration * dt * dt;
    }
    return {R_GtoI.transpose(), p};
}

void write(State& state, double* out, const Pose& camera, const Eigen::Vector3d& v_cam, const Eigen::Vector3d& w_cam, bool conservative = false) {
    double* last = conservative ? state.last_conservative : state.last;
    // relative to the first camera: p_c0 = R p_ck + t
    const Eigen::Matrix3d R = state.camera0.R.transpose() * camera.R;
    const Eigen::Vector3d t = state.camera0.R.transpose() * (camera.p - state.camera0.p);
    Eigen::Quaterniond q(R);
    q.normalize();
    const double values[VIOBENCH_STATE_VALUES] = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w(),
                                                 v_cam.x(), v_cam.y(), v_cam.z(), w_cam.x(), w_cam.y(), w_cam.z()};
    bool finite = true;
    for (double value : values) finite = finite && std::isfinite(value);
    if (!finite) throw std::runtime_error("nonfinite estimator output candidate");
    for (int i = 0; i < VIOBENCH_STATE_VALUES; i++) last[i] = values[i];
    for (int i = 0; i < VIOBENCH_STATE_VALUES; i++) out[i] = last[i];
}

int32_t estimate(State& state, double t, const uint8_t* frame, const double* imu, int32_t imu_count, double* out, double* conservative_out = nullptr) {
    Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
    for (int32_t i = 0; i < imu_count; i++) {
        const double* row = imu + VIOBENCH_IMU_ROW_VALUES * i;
        ov_core::ImuData sample;
        sample.timestamp = row[0];
        sample.am << row[1], row[2], row[3];
        sample.wm << row[4], row[5], row[6];
        gyro_mean += sample.wm / imu_count;
        state.vio->feed_measurement_imu(sample);
        if (!state.anchored) {
            if (state.frames > 0 && !state.imu.empty()) {  // gyro attitude for the outputs before initialization
                const ov_core::ImuData& previous = state.imu.back();
                state.R_I0_I = state.R_I0_I * ov_core::exp_so3(0.5 * (previous.wm + sample.wm) * (sample.timestamp - previous.timestamp));
            }
            state.imu.push_back(sample);
        }
    }
    if (state.frames++ == 0) state.t0 = t;

    ov_core::CameraData message;
    message.timestamp = t;
    message.sensor_ids.push_back(0);
    cv::Mat gray;
    cv::cvtColor(cv::Mat(state.height, state.width, CV_8UC3, const_cast<uint8_t*>(frame)), gray, cv::COLOR_RGB2GRAY);
    message.images.push_back(gray);
    message.masks.push_back(state.mask);
    state.vio->feed_measurement_camera(message);

    if (!state.vio->started()) {  // the gyro attitude, in place
        if (t - state.t0 >= state.next_init_warning) {
            std::fprintf(stderr, "openvins: still initializing after %.1f s; position and linear velocity are unavailable "
                                 "(returning zero translation and gyro-only rotation)\n", t - state.t0);
            state.next_init_warning = t - state.t0 + 5.0;
        }
        write(state, out, camera_pose(state, state.R_I0_I, Eigen::Vector3d::Zero()), Eigen::Vector3d::Zero(), state.R_ItoC * gyro_mean);
        if (conservative_out) {
            std::copy(out, out + VIOBENCH_STATE_VALUES, conservative_out);
            std::copy(out, out + VIOBENCH_STATE_VALUES, state.last_conservative);
        }
        return VIOBENCH_STATUS_OK;
    }
    std::shared_ptr<ov_msckf::State> filter = state.vio->get_state();
    if (!state.anchored) {
        std::shared_ptr<ov_type::IMU> initial = filter->_imu;
        Pose imu0 = imu_pose_at_first_frame(state, filter->_timestamp, initial->Rot(), initial->pos(), initial->vel(), initial->bias_g(),
                                            initial->bias_a(), state.gravity);
        state.camera0 = camera_pose(state, imu0.R, imu0.p);
        state.anchored = true;
        state.imu.clear();
        state.imu.shrink_to_fit();
        std::printf("openvins: initialized at t=%.3f (frame %ld), state at t=%.3f, |v|=%.3f m/s\n", t, state.frames - 1,
                    filter->_timestamp, initial->vel().norm());
    }
    // the IMU state at t: the filter's own once it has processed this frame, else propagated to it
    Eigen::Matrix3d R_GtoI;
    Eigen::Vector3d p_IinG, v_IinI;
    if (filter->_timestamp == t) {
        R_GtoI = filter->_imu->Rot();
        p_IinG = filter->_imu->pos();
        v_IinI = R_GtoI * filter->_imu->vel();
    } else {
        Eigen::Matrix<double, 13, 1> propagated;  // q_GtoI (JPL), p_IinG, v_IinI, w_IinI
        Eigen::Matrix<double, 12, 12> covariance;
        if (!state.vio->get_propagator()->fast_state_propagate(filter, t, propagated, covariance))
            throw std::runtime_error("EKF output propagation failed");
        R_GtoI = ov_core::quat_2_Rot(propagated.block<4, 1>(0, 0));
        p_IinG = propagated.block<3, 1>(4, 0);
        v_IinI = propagated.block<3, 1>(7, 0);
    }
    // Re-express each independent graph in this frontend's first-camera frame.
    // The conservative graph never changes the EKF, tracker, or primary graph.
    auto write_graph = [&](viobench_openvins::FixedLagVio& graph, double* destination, bool is_conservative) {
        Eigen::Matrix3d rotation = R_GtoI;
        Eigen::Vector3d position = p_IinG, velocity = v_IinI, gyro_bias = filter->_imu->bias_g();
        Eigen::Matrix<double, 16, 1> smoothed;
        if (graph.state_at(t, smoothed)) {
            rotation = ov_core::quat_2_Rot(smoothed.head<4>());
            position = smoothed.segment<3>(4);
            velocity = rotation * smoothed.segment<3>(7);
            gyro_bias = smoothed.segment<3>(10);
        }
        if (graph.failed()) throw std::runtime_error("fixed-lag IMU propagation failed");
        const Eigen::Vector3d w_IinI = gyro_mean - gyro_bias;
        const Eigen::Vector3d v_cam = state.R_ItoC * (velocity + w_IinI.cross(state.p_CinI));
        write(state, destination, camera_pose(state, rotation.transpose(), position), v_cam, state.R_ItoC * w_IinI, is_conservative);
    };
    write_graph(state.vio->smoother, out, false);
    if (conservative_out) {
        if (state.vio->conservative) write_graph(*state.vio->conservative, conservative_out, true);
        else std::copy(out, out + VIOBENCH_STATE_VALUES, conservative_out);
    }
    return VIOBENCH_STATUS_OK;
}

void initialize_state(State& state, int width, int height, const double intrinsics[4], const Eigen::Matrix4d& T_cam_imu,
                      bool use_anchor, bool use_conservative) {
    state = State{};
    state.width = width;
    state.height = height;
    state.R_ItoC = T_cam_imu.block<3, 3>(0, 0);
    state.p_CinI = -state.R_ItoC.transpose() * T_cam_imu.block<3, 1>(0, 3);
    state.mask = cv::Mat::zeros(height, width, CV_8UC1);
    state.camera0 = camera_pose(state, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
    // Construct fresh calibration objects for each EKF/frontend context.
    ov_msckf::VioManagerOptions options = make_options(width, height, intrinsics, T_cam_imu);
    state.gravity = options.gravity_mag;
    state.vio = std::make_unique<Vio>(options, use_anchor, use_conservative);
}

Output output_from_abi(double timestamp, const double* values) {
    Output result;
    result.timestamp = timestamp;
    result.position = Eigen::Map<const Eigen::Vector3d>(values);
    result.rotation = Eigen::Quaterniond(values[6], values[3], values[4], values[5]).toRotationMatrix();
    result.linear_velocity = Eigen::Map<const Eigen::Vector3d>(values + 7);
    result.angular_velocity = Eigen::Map<const Eigen::Vector3d>(values + 10);
    return result;
}

void output_to_abi(const Output& value, double* out) {
    const Eigen::Quaterniond q = Eigen::Quaterniond(value.rotation).normalized();
    Eigen::Map<Eigen::Vector3d> position(out);
    Eigen::Map<Eigen::Vector4d> quaternion(out + 3);
    Eigen::Map<Eigen::Vector3d> velocity(out + 7), angular_velocity(out + 10);
    position = value.position;
    quaternion << q.x(), q.y(), q.z(), q.w();
    velocity = value.linear_velocity;
    angular_velocity = value.angular_velocity;
}

}  // namespace

int32_t abi_version() { return VIOBENCH_ABI_VERSION; }

void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }

int32_t frame_width() { return 320; }

int32_t init(const char* json, uint32_t length) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    try {
        const std::string text(json, length);
        const std::vector<double> size = numbers(text, "width", 1), height = numbers(text, "height", 1);
        const std::vector<double> fx = numbers(text, "fx", 1), fy = numbers(text, "fy", 1), cx = numbers(text, "cx", 1),
                                  cy = numbers(text, "cy", 1), T = numbers(text, "T_cam_imu", 16);
        if (size.empty() || height.empty() || fx.empty() || fy.empty() || cx.empty() || cy.empty() || T.size() != 16) {
            std::fprintf(stderr, "config without width, height, fx, fy, cx, cy or T_cam_imu\n");
            return VIOBENCH_STATUS_ERROR;
        }
        const int width = static_cast<int>(size[0]), image_height = static_cast<int>(height[0]);
        Eigen::Matrix4d T_cam_imu;
        for (int i = 0; i < 16; i++) T_cam_imu(i / 4, i % 4) = T[i];
        ov_core::Printer::setPrintLevel(ov_core::Printer::PrintLevel::WARNING);
        const double intrinsics[4] = {fx[0], fy[0], cx[0], cy[0]};
        initialize_state(chain_state, width, image_height, intrinsics, T_cam_imu, false, true);
        initialize_state(anchor_state, width, image_height, intrinsics, T_cam_imu, true, false);
        consensus.reset();
        std::printf("openvins consensus: %dx%d fx=%.1f fy=%.1f cx=%.1f cy=%.1f\n", width, image_height, fx[0], fy[0], cx[0], cy[0]);
        return VIOBENCH_STATUS_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "init: %s\n", error.what());
        return VIOBENCH_STATUS_ERROR;
    }
}

int32_t step(double t, const uint8_t* frame, const double* imu, int32_t imu_count, double* out) {
    try {
        double chain_values[VIOBENCH_STATE_VALUES], anchor_values[VIOBENCH_STATE_VALUES], conservative_values[VIOBENCH_STATE_VALUES];
        const int32_t chain_status = estimate(chain_state, t, frame, imu, imu_count, chain_values, conservative_values);
        if (chain_status != VIOBENCH_STATUS_OK) return chain_status;
        const int32_t anchor_status = estimate(anchor_state, t, frame, imu, imu_count, anchor_values);
        if (anchor_status != VIOBENCH_STATUS_OK) return anchor_status;
        const Output chain = output_from_abi(t, chain_values), anchor = output_from_abi(t, anchor_values),
                     conservative = output_from_abi(t, conservative_values);
        Output combined;
        if (!consensus.update(chain, anchor, conservative, combined))
            throw std::runtime_error("consensus output rejected invalid candidates");
        output_to_abi(combined, out);
        return VIOBENCH_STATUS_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "step at t=%.3f: %s\n", t, error.what());
        return VIOBENCH_STATUS_ERROR;
    }
}
