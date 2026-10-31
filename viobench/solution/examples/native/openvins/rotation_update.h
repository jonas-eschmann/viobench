// Startup aid for a monocular camera that hovers and turns before translating.
// A pure rotation does not determine landmark depths. Use its observable rotation
// directly, with a zero-linear-velocity constraint, until measurable parallax appears.
#ifndef VIOBENCH_OPENVINS_ROTATION_UPDATE_H
#define VIOBENCH_OPENVINS_ROTATION_UPDATE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/quat_ops.h"

class RotationOnlyStartup {
public:
    // Pixel thresholds are deliberately independent of the focal length. A short
    // adjacent-frame test mistakes slow translation for a stationary camera.
    double window_seconds = 0.5;
    double max_fit_rms_pixels = 0.7;
    double minimum_inlier_fraction = 0.65;
    int minimum_inliers = 20;
    double max_missing_seconds = 1.0; // bounded propagation-only retry when visual evidence is unavailable
    double rotation_sigma = 0.003;  // radians; includes tracking/model uncertainty
    double position_sigma = 0.02;  // metres; <= 0 disables the relative-position constraint
    double velocity_sigma = 0.05;  // m/s; small hover motion is not exactly zero

    bool active() const { return active_; }
    int updates() const { return updates_; }
    double fit_rms_pixels() const { return last_rms_; }
    double inlier_fraction() const { return last_fraction_; }

    // Call after tracking the current image and successful initialization, before
    // normal ZUPT or visual updates. True means this frame has been handled. False
    // hands over permanently to the ordinary MSCKF/SLAM update. Missing tracks
    // permit a bounded propagation-only retry; only a supported rotation fit adds
    // zero-motion constraints, regardless of the initial velocity estimate. This
    // routine never resets the state or covariance or re-enters after handoff.
    bool update(const std::shared_ptr<ov_msckf::State>& state,
                const std::shared_ptr<ov_msckf::Propagator>& propagator,
                const std::shared_ptr<ov_core::FeatureDatabase>& database, double timestamp) {
        if (!active_) return false;
        if (state->_timestamp > timestamp) return finish();
        if (state->_timestamp < timestamp) propagator->propagate_and_clone(state, timestamp);
        if (state->_timestamp != timestamp) return finish();
        if (!state->_clones_IMU.count(timestamp))
            ov_msckf::StateHelper::augment_clone(state, Eigen::Vector3d::Zero());
        prune_intermediate_clones(state, timestamp);
        if (reference_time_ < 0) {
            if (missing_timeout(timestamp)) return finish();
            // Initialization can consume every observation of its current image.
            // A clone alone is not a usable visual reference.
            if (!current_frame_available(state, database, timestamp))
                return retry_missing(state, database, timestamp);
            reset_reference(state, database, timestamp);
            return true;
        }
        if (timestamp - reference_time_ < window_seconds - 1e-6)
            return missing_timeout(timestamp) ? finish() : true;
        auto reference = state->_clones_IMU.find(reference_time_);
        if (reference == state->_clones_IMU.end()) return retry_missing(state, database, timestamp);

        const auto camera = state->_cam_intrinsics_cameras.at(0);
        const double pixel_scale = 320.0 / camera->w();
        const cv::Matx33d calibration = camera->get_K();
        Eigen::Matrix3d K;
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) K(row, col) = calibration(row, col);
        K.topRows<2>() *= pixel_scale;
        std::vector<Match> matches;
        for (const auto& entry : database->get_internal_data()) {
            const auto& feature = entry.second;
            auto times_it = feature->timestamps.find(0);
            if (times_it == feature->timestamps.end()) continue;
            const auto& times = times_it->second;
            if (times.empty() || times.back() != timestamp) continue;
            auto old = std::lower_bound(times.begin(), times.end(), reference_time_);
            if (old == times.end() || *old != reference_time_) continue;
            const size_t i = static_cast<size_t>(old - times.begin());
            const auto& uv0 = feature->uvs_norm.at(0).at(i);
            const auto& uv1 = feature->uvs.at(0).back();
            Eigen::Vector3d bearing(uv0(0), uv0(1), 1.0);
            const Eigen::Vector2d pixel = pixel_scale * Eigen::Vector2d(uv1(0), uv1(1));
            if (bearing.allFinite() && pixel.allFinite())
                matches.push_back({entry.first, bearing.normalized(), pixel});
        }
        // Stable sampling order, independent of unordered-map implementation.
        std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) { return a.id < b.id; });
        const int image_height = static_cast<int>(camera->h() * pixel_scale);
        if (!sufficiently_distributed(matches, image_height))
            return retry_missing(state, database, timestamp);
        Fit fit = fit_rotation(matches, K);
        last_rms_ = fit.rms;
        last_fraction_ = matches.empty() ? 0.0 : static_cast<double>(fit.inliers.size()) / matches.size();
        if (!fit.valid) return retry_missing(state, database, timestamp);
        // Distributed correspondences that reject the rotation model are evidence
        // for handoff. A missing/degenerate fit is only an absence of evidence.
        if (last_fraction_ < minimum_inlier_fraction || fit.rms > max_fit_rms_pixels)
            return finish();
        if (static_cast<int>(fit.inliers.size()) < minimum_inliers ||
            !distributed(matches, fit.inliers, 320, image_height))
            return retry_missing(state, database, timestamp);
        if (!state->_imu->vel().allFinite()) return finish();

        const Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(0)->Rot();
        const Eigen::Matrix3d measured = R_ItoC.transpose() * fit.rotation * R_ItoC;
        const Eigen::Matrix3d predicted = state->_imu->Rot() * reference->second->Rot().transpose();
        const Eigen::Vector3d rotation_residual = ov_core::log_so3(measured * predicted.transpose());
        if (!rotation_residual.allFinite() || rotation_residual.norm() > 0.2) return finish();

        // JPL left errors obey R_true = Exp(-dtheta) R_est. Hence
        // Log(Q_true Q_est^T) = Q_est*dtheta_reference - dtheta_current,
        // where Q = R_current R_reference^T. FEJ preserves the global yaw gauge.
        const Eigen::Matrix3d Q_jac = state->_options.do_fej
            ? state->_imu->Rot_fej() * reference->second->Rot_fej().transpose() : predicted;
        std::vector<std::shared_ptr<ov_type::Type>> order = {
            reference->second->q(), state->_imu->q(), reference->second->p(), state->_imu->p(), state->_imu->v()};
        const bool constrain_position = position_sigma > 0;
        const int velocity_row = constrain_position ? 6 : 3;
        const int residual_size = velocity_row + 3;
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(residual_size, 15);
        H.block<3, 3>(0, 0) = Q_jac;
        H.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
        H.block<3, 3>(velocity_row, 12) = Eigen::Matrix3d::Identity();
        Eigen::VectorXd residual(residual_size);
        residual.head<3>() = rotation_residual;
        residual.segment<3>(velocity_row) = -state->_imu->vel();
        Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(residual_size, residual_size);
        noise.block<3, 3>(0, 0) = rotation_sigma * rotation_sigma * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(velocity_row, velocity_row) = velocity_sigma * velocity_sigma * Eigen::Matrix3d::Identity();
        if (constrain_position) {
            // Keep inertial integration error from inventing a triangulation baseline
            // in a window whose images support only a rotation. Both positions are in G.
            residual.segment<3>(3) = reference->second->pos() - state->_imu->pos();
            H.block<3, 3>(3, 6) = -Eigen::Matrix3d::Identity();
            H.block<3, 3>(3, 9) = Eigen::Matrix3d::Identity();
            noise.block<3, 3>(3, 3) = position_sigma * position_sigma * Eigen::Matrix3d::Identity();
        }
        ov_msckf::StateHelper::EKFUpdate(state, order, H, residual, noise);
        propagator->invalidate_cache();
        ++updates_;
        missing_since_ = -1;

        // Nonoverlapping windows avoid repeatedly counting the same visual pair.
        // Discard consumed observations before ordinary VIO can use them again.
        reset_reference(state, database, timestamp);
        return true;
    }

private:
    struct Match {
        size_t id;
        Eigen::Vector3d bearing;
        Eigen::Vector2d pixel;
    };
    struct Fit {
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
        std::vector<size_t> inliers;
        double rms = std::numeric_limits<double>::infinity();
        bool valid = false;
    };
    bool active_ = true;
    double reference_time_ = -1;
    double missing_since_ = -1;
    int updates_ = 0;
    double last_rms_ = std::numeric_limits<double>::infinity();
    double last_fraction_ = 0;

    bool finish() { active_ = false; return false; }

    bool missing_timeout(double timestamp) const {
        return missing_since_ >= 0 && timestamp - missing_since_ >= max_missing_seconds - 1e-6;
    }

    bool sufficiently_distributed(const std::vector<Match>& matches, int height) const {
        if (static_cast<int>(matches.size()) < minimum_inliers) return false;
        std::vector<size_t> indices;
        for (size_t i = 0; i < matches.size(); ++i) indices.push_back(i);
        return distributed(matches, indices, 320, height);
    }

    bool current_frame_available(const std::shared_ptr<ov_msckf::State>& state,
                                 const std::shared_ptr<ov_core::FeatureDatabase>& database, double timestamp) const {
        const auto camera = state->_cam_intrinsics_cameras.at(0);
        const double pixel_scale = 320.0 / camera->w();
        std::vector<Match> observations;
        for (const auto& entry : database->get_internal_data()) {
            const auto& feature = entry.second;
            auto times = feature->timestamps.find(0);
            if (times == feature->timestamps.end() || times->second.empty() || times->second.back() != timestamp) continue;
            const auto& uv = feature->uvs.at(0).back();
            const Eigen::Vector2d pixel = pixel_scale * Eigen::Vector2d(uv(0), uv(1));
            if (pixel.allFinite()) observations.push_back({entry.first, Eigen::Vector3d::Zero(), pixel});
        }
        return sufficiently_distributed(observations, static_cast<int>(camera->h() * pixel_scale));
    }

    void reset_reference(const std::shared_ptr<ov_msckf::State>& state,
                         const std::shared_ptr<ov_core::FeatureDatabase>& database, double timestamp) {
        reference_time_ = timestamp;
        // cleanup_measurements is inclusive: preserve the new reference image.
        database->cleanup_measurements(std::nextafter(timestamp, -std::numeric_limits<double>::infinity()));
        auto clone = state->_clones_IMU.begin();
        while (clone != state->_clones_IMU.end()) {
            if (clone->first == timestamp) { ++clone; continue; }
            ov_msckf::StateHelper::marginalize(state, clone->second);
            clone = state->_clones_IMU.erase(clone);
        }
    }

    bool retry_missing(const std::shared_ptr<ov_msckf::State>& state,
                       const std::shared_ptr<ov_core::FeatureDatabase>& database, double timestamp) {
        last_rms_ = std::numeric_limits<double>::infinity();
        last_fraction_ = 0;
        if (missing_since_ < 0) missing_since_ = timestamp;
        if (missing_timeout(timestamp)) return finish();
        const bool available = current_frame_available(state, database, timestamp);
        // New tracks cannot reconnect to a lost reference. Rebase both the clone
        // and observations together; never constrain the unobserved interval.
        reset_reference(state, database, timestamp);
        if (!available) reference_time_ = -1;
        return true;
    }

    void prune_intermediate_clones(const std::shared_ptr<ov_msckf::State>& state, double timestamp) const {
        const size_t limit = static_cast<size_t>(std::max(2, state->_options.max_clone_size));
        auto clone = state->_clones_IMU.begin();
        while (clone != state->_clones_IMU.end() && state->_clones_IMU.size() > limit) {
            if (clone->first == reference_time_ || clone->first == timestamp) { ++clone; continue; }
            ov_msckf::StateHelper::marginalize(state, clone->second);
            clone = state->_clones_IMU.erase(clone);
        }
    }

    static Eigen::Matrix3d wahba(const std::vector<Match>& matches, const std::vector<size_t>& indices,
                                const Eigen::Matrix3d& inverse_K) {
        Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();
        for (size_t i : indices) {
            const Eigen::Vector3d observed = inverse_K * Eigen::Vector3d(matches[i].pixel.x(), matches[i].pixel.y(), 1);
            cross.noalias() += observed.normalized() * matches[i].bearing.transpose();
        }
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(cross, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d sign = Eigen::Matrix3d::Identity();
        sign(2, 2) = (svd.matrixU() * svd.matrixV().transpose()).determinant() < 0 ? -1 : 1;
        return svd.matrixU() * sign * svd.matrixV().transpose();
    }

    static Fit score(const std::vector<Match>& matches, const Eigen::Matrix3d& K, const Eigen::Matrix3d& rotation) {
        Fit fit;
        fit.rotation = rotation;
        if (!rotation.allFinite()) return fit;
        fit.valid = true;
        double sum = 0;
        for (size_t i = 0; i < matches.size(); ++i) {
            const Eigen::Vector3d point = K * rotation * matches[i].bearing;
            if (point.z() <= 0) continue;
            const double error_sq = (point.head<2>() / point.z() - matches[i].pixel).squaredNorm();
            if (error_sq < 1.0) {
                fit.inliers.push_back(i);
                sum += error_sq;
            }
        }
        if (!fit.inliers.empty()) fit.rms = std::sqrt(sum / fit.inliers.size());
        return fit;
    }

    static Fit fit_rotation(const std::vector<Match>& matches, const Eigen::Matrix3d& K) {
        if (matches.size() < 3) return {};
        const Eigen::Matrix3d inverse_K = K.inverse();
        std::vector<size_t> all;
        for (size_t i = 0; i < matches.size(); ++i) all.push_back(i);
        Fit best = score(matches, K, wahba(matches, all, inverse_K));
        uint32_t random = 0x61943a27;
        for (int trial = 0; trial < 64; ++trial) {
            std::vector<size_t> indices;
            while (indices.size() < 3) {
                random = 1664525u * random + 1013904223u;
                size_t i = random % matches.size();
                if (std::find(indices.begin(), indices.end(), i) == indices.end()) indices.push_back(i);
            }
            Fit candidate = score(matches, K, wahba(matches, indices, inverse_K));
            if (candidate.inliers.size() > best.inliers.size() ||
                (candidate.inliers.size() == best.inliers.size() && candidate.rms < best.rms)) best = std::move(candidate);
        }
        if (best.inliers.size() < 3) return best;

        // Refine pixel reprojection error, not angular error: wide-angle image
        // corners magnify the same bearing error more than the image centre.
        Eigen::Matrix3d rotation = best.rotation;
        for (int iteration = 0; iteration < 6; ++iteration) {
            Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
            Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
            for (size_t i : best.inliers) {
                const Eigen::Vector3d point = rotation * matches[i].bearing;
                if (point.z() <= 0) continue;
                const double inverse_z = 1.0 / point.z();
                Eigen::Matrix<double, 2, 3> projection;
                projection << K(0, 0) * inverse_z, 0, -K(0, 0) * point.x() * inverse_z * inverse_z,
                              0, K(1, 1) * inverse_z, -K(1, 1) * point.y() * inverse_z * inverse_z;
                const Eigen::Matrix<double, 2, 3> J = -projection * ov_core::skew_x(point);
                const Eigen::Vector3d pixel = K * point;
                const Eigen::Vector2d error = matches[i].pixel - pixel.head<2>() / pixel.z();
                information.noalias() += J.transpose() * J;
                rhs.noalias() += J.transpose() * error;
            }
            Eigen::LDLT<Eigen::Matrix3d> factor(information);
            if (factor.info() != Eigen::Success || factor.vectorD().minCoeff() <= 1e-8) return {};
            const Eigen::Vector3d delta = factor.solve(rhs);
            if (!delta.allFinite() || delta.norm() > 0.1) return {};
            rotation = ov_core::exp_so3(delta) * rotation;
            best = score(matches, K, rotation);
            if (best.inliers.size() < 3 || delta.norm() < 1e-7) break;
        }
        return best;
    }

    static bool distributed(const std::vector<Match>& matches, const std::vector<size_t>& indices, int width, int height) {
        if (indices.empty()) return false;
        Eigen::Vector2d minimum = matches[indices.front()].pixel, maximum = minimum, mean = Eigen::Vector2d::Zero();
        for (size_t i : indices) {
            minimum = minimum.cwiseMin(matches[i].pixel);
            maximum = maximum.cwiseMax(matches[i].pixel);
            mean += matches[i].pixel;
        }
        if ((maximum - minimum).x() < 0.25 * width || (maximum - minimum).y() < 0.15 * height) return false;
        mean /= indices.size();
        Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
        for (size_t i : indices) {
            const Eigen::Vector2d delta = matches[i].pixel - mean;
            covariance.noalias() += delta * delta.transpose();
        }
        covariance /= indices.size();
        return Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(covariance).eigenvalues().minCoeff() > 64.0;
    }
};

#endif
