// Fixed-lag visual-inertial smoothing with OpenVINS GPL-3.0 preintegration and camera factors.
// The graph owns its state and measurement history; it never overwrites the front-end EKF.
#ifndef RELPOSE_OPENVINS_FIXED_LAG_VIO_H
#define RELPOSE_OPENVINS_FIXED_LAG_VIO_H

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "ceres/Factor_ImageReprojCalib.h"
#include "ceres/Factor_ImuCPIv1.h"
#include "ceres/State_JPLQuatLocal.h"
#include "cpi/CpiV1.h"
#include "feat/FeatureDatabase.h"
#include "types/IMU.h"
#include "feat/Feature.h"
#include "utils/sensor_data.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/quat_ops.h"

namespace relpose_openvins {

// Normalize diagonal information before deciding numerical rank. Applying one
// eigenvalue threshold directly to metres, radians and bias units can discard a
// weak landmark direction beside a stiff IMU factor, falsely strengthening the
// Schur prior. A zero diagonal of a PSD matrix has a zero row; leave its scale 1.
inline Eigen::VectorXd information_scaling(const Eigen::MatrixXd& H) {
    return H.diagonal().unaryExpr([](double diagonal) {
        return diagonal > 0 ? 1.0 / std::sqrt(diagonal) : 1.0;
    });
}

// State order throughout this header: local q, p, v, bg, ba (15 tangent dimensions).
// The upstream generic prior uses a different orientation tangent and computes its
// nonzero-gradient term with the wrong triangular transpose. Do not substitute it.
class LocalPrior final : public ceres::CostFunction {
public:
    Eigen::VectorXd linearization, offset;
    Eigen::MatrixXd root_information;
    bool valid = false;
    int rank = 0;
    double minimum_eigenvalue = 0; // dimensionless, before clipping
    double minimum_retained_eigenvalue = 0;
    explicit LocalPrior(const Eigen::VectorXd& x, const Eigen::MatrixXd& H, const Eigen::VectorXd& g)
        : linearization(x) {
        if (x.size() == 0 || x.size() % 16 != 0 || H.rows() != 15 * (x.size() / 16) || H.cols() != H.rows() ||
            g.size() != H.rows() || !x.allFinite() || !H.allFinite() || !g.allFinite()) return;
        const Eigen::VectorXd scaling = information_scaling(H);
        const Eigen::MatrixXd normalized = scaling.asDiagonal() * (0.5 * (H + H.transpose())) * scaling.asDiagonal();
        if (!scaling.allFinite() || !normalized.allFinite()) return;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(normalized);
        if (eig.info() != Eigen::Success || !eig.eigenvalues().allFinite() || !eig.eigenvectors().allFinite()) return;
        const double cutoff = std::max(1e-12, eig.eigenvalues().maxCoeff() * 1e-12);
        Eigen::VectorXd roots = eig.eigenvalues().unaryExpr([&](double v) { return v > cutoff ? std::sqrt(v) : 0.0; });
        minimum_eigenvalue = eig.eigenvalues().minCoeff();
        for (int i = 0; i < roots.size(); ++i) {
            if (roots(i) <= 0) continue;
            if (rank++ == 0) minimum_retained_eigenvalue = eig.eigenvalues()(i);
        }
        // A = D H D = U L U^T, so S = sqrt(L) U^T D^-1 obeys S^T S = H.
        // Also S^T b = g requires b = sqrt(L)^-1 U^T D g.
        root_information = roots.asDiagonal() * eig.eigenvectors().transpose() * scaling.cwiseInverse().asDiagonal();
        // Eigensolvers can put roundoff into structurally absent columns. Do not
        // let the next diagonal equilibration turn those into apparent modes.
        for (int column = 0; column < H.cols(); ++column)
            if ((H.row(column).array() == 0.0).all()) root_information.col(column).setZero();
        Eigen::VectorXd transformed = eig.eigenvectors().transpose() * scaling.asDiagonal() * g;
        offset = transformed;
        for (int i = 0; i < roots.size(); ++i) offset(i) = roots(i) > 0 ? transformed(i) / roots(i) : 0.0;
        if (!root_information.allFinite() || !offset.allFinite()) return;
        set_num_residuals(H.rows());
        for (int i = 0; i < x.size() / 16; ++i)
            for (int size : {4, 3, 3, 3, 3}) mutable_parameter_block_sizes()->push_back(size);
        valid = true;
    }
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        if (!valid) return false;
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(num_residuals());
        for (int i = 0; i < linearization.size() / 16; ++i) {
            Eigen::Vector4d q = Eigen::Map<const Eigen::Vector4d>(parameters[5 * i]);
            Eigen::Vector4d q0 = linearization.segment<4>(16 * i);
            Eigen::Vector3d d = ov_core::log_so3(ov_core::quat_2_Rot(q0) * ov_core::quat_2_Rot(q).transpose());
            delta.segment<3>(15 * i) = d;
            if (jacobians && jacobians[5 * i]) {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>> J(jacobians[5 * i], num_residuals(), 4);
                J.setZero();
                // State_JPLQuatLocal deliberately expects local derivatives in the first 3 columns.
                J.leftCols<3>() = root_information.middleCols(15 * i, 3) * ov_core::Jr_so3(d).inverse();
            }
            for (int block = 1; block < 5; ++block) {
                const int local = 15 * i + 3 * block, ambient = 16 * i + 1 + 3 * block;
                delta.segment<3>(local) = Eigen::Map<const Eigen::Vector3d>(parameters[5 * i + block]) - linearization.segment<3>(ambient);
                if (jacobians && jacobians[5 * i + block]) {
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> J(jacobians[5 * i + block], num_residuals(), 3);
                    J = root_information.middleCols(local, 3);
                }
            }
        }
        Eigen::Map<Eigen::VectorXd> r(residuals, num_residuals());
        r = root_information * delta + offset;
        return r.allFinite();
    }
};

class FixedLagVio {
public:
    double keyframe_interval = 0.2; // 5 Hz; outputs between keyframes are IMU predictions.
    size_t maximum_nodes = 13;     // 2.4 s lag, full velocity and biases at each node.
    size_t maximum_features = 100;
    size_t maximum_new_features = 25; // stagger track lifetimes and later marginalization
    int maximum_iterations = 6;
    double pixel_sigma = 0.75;
    double sigma_w = 0.07 / std::sqrt(200.0), sigma_a = 0.28 / std::sqrt(200.0);
    double sigma_wb = 0.0006, sigma_ab = 0.001;
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, 9.81);
    int optimizations = 0;
    double last_initial_cost = 0, last_final_cost = 0;
    size_t last_visual_factors = 0;
    size_t last_triangulated = 0, last_consumed = 0;
    int last_prior_rank = 0;
    double last_prior_min_eigenvalue = 0, last_prior_min_retained_eigenvalue = 0;

    void feed_imu(const ov_core::ImuData& sample) {
        if (failed_) return;
        if (!std::isfinite(sample.timestamp) || !sample.wm.allFinite() || !sample.am.allFinite() ||
            (!imu_.empty() && sample.timestamp <= imu_.back().timestamp)) { fail(); return; }
        imu_.push_back(sample);
        // Initialization starts at the current handoff image, so older startup
        // history is unnecessary. Preserve one sample before the cutoff.
        if (!initialized()) trim_imu_before(sample.timestamp - 1.0);
    }
    bool initialized() const { return !nodes_.empty(); }
    bool failed() const { return failed_; }
    bool healthy() const { return !failed_; }
    Eigen::Vector3d gyro_bias() const {
        if (nodes_.empty()) return Eigen::Vector3d::Zero();
        return nodes_.rbegin()->second.x.segment<3>(10);
    }

    // Call exactly once after the rotation-startup handoff, before ordinary VIO
    // consumes the current tracks. Feed IMU samples before this call as well.
    void initialize(const std::shared_ptr<ov_msckf::State>& state) {
        if (failed_) return;
        nodes_.clear(); features_.clear(); prior_times_.clear();
        Node first;
        first.x = state->_imu->value();
        const double t = state->_timestamp;
        if (!std::isfinite(t) || !first.x.allFinite()) { fail(); return; }
        nodes_.emplace(t, first);
        calibration_ = state->_cam_intrinsics_cameras.at(0)->get_value();
        image_width_ = state->_cam_intrinsics_cameras.at(0)->w();
        image_height_ = state->_cam_intrinsics_cameras.at(0)->h();
        extrinsics_ = state->_calib_IMUtoCAM.at(0)->value();
        if (!calibration_.allFinite() || !extrinsics_.allFinite() || image_width_ <= 0 || image_height_ <= 0) { fail(); return; }
        prior_times_.push_back(t);
        prior_x_ = first.x;
        Eigen::MatrixXd covariance = ov_msckf::StateHelper::get_marginal_covariance(state, {state->_imu});
        if (!pseudo_inverse(covariance, prior_H_)) { fail(); return; }
        prior_g_ = Eigen::VectorXd::Zero(15);
        last_frame_ = -1;
    }

    // Copy only current-keyframe observations. Never pass this graph an EKF
    // posterior as a recurring prior: that posterior already used these images.
    bool feed_camera(double timestamp, const std::shared_ptr<ov_core::FeatureDatabase>& database) {
        if (failed_) return false;
        if (!std::isfinite(timestamp)) return fail();
        if (!initialized() || timestamp <= last_frame_) return false;
        const double latest = nodes_.rbegin()->first;
        if (timestamp < latest || (timestamp > latest && timestamp - latest < keyframe_interval - 1e-6)) return false;
        if (timestamp > latest) {
            auto cpi = integrate(latest, timestamp, nodes_.rbegin()->second);
            if (!cpi) return fail();
            const Node predicted = predict(nodes_.rbegin()->second, *cpi);
            if (!predicted.x.allFinite()) return fail();
            nodes_.emplace(timestamp, predicted);
        }
        last_frame_ = timestamp;
        // Retain existing tracks. New cohorts should cover the image rather than
        // inherit the detector's grid/ID order.
        std::map<size_t, Eigen::Vector2d> observations;
        for (const auto& entry : database->get_internal_data()) {
            auto found = entry.second->timestamps.find(0);
            if (found == entry.second->timestamps.end() || found->second.empty() || found->second.back() != timestamp) continue;
            Eigen::Vector2d uv = entry.second->uvs.at(0).back().cast<double>();
            if (uv.allFinite()) observations.emplace(entry.first, uv);
        }
        for (const auto& observation : observations) {
            auto found = features_.find(observation.first);
            if (found == features_.end()) continue;
            found->second.observations[timestamp] = observation.second;
        }
        std::array<size_t, 20> occupancy{}, next_candidate{};
        std::array<std::vector<size_t>, 20> candidates;
        for (const auto& observation : observations) {
            const int cell = image_cell(observation.second);
            if (features_.count(observation.first)) ++occupancy[cell];
            else candidates[cell].push_back(observation.first); // observations is sorted by ID
        }
        for (size_t admitted = 0; admitted < maximum_new_features && features_.size() < maximum_features; ++admitted) {
            int selected = -1;
            for (int cell = 0; cell < 20; ++cell) {
                if (next_candidate[cell] == candidates[cell].size()) continue;
                if (selected < 0 || occupancy[cell] < occupancy[selected] ||
                    (occupancy[cell] == occupancy[selected] &&
                     candidates[cell][next_candidate[cell]] < candidates[selected][next_candidate[selected]])) selected = cell;
            }
            if (selected < 0) break;
            const size_t id = candidates[selected][next_candidate[selected]++];
            features_[id].observations[timestamp] = observations.at(id);
            ++occupancy[selected];
        }
        for (auto it = features_.begin(); it != features_.end();) {
            if (!it->second.triangulated && timestamp - it->second.observations.rbegin()->first > 0.5)
                it = features_.erase(it);
            else { if (!it->second.triangulated) triangulate(it->second); ++it; }
            if (failed_) return false;
        }
        if (nodes_.size() < 3) return false;
        return optimize();
    }

    bool state_at(double timestamp, Eigen::Matrix<double, 16, 1>& state) const {
        if (failed_) return false;
        if (!std::isfinite(timestamp)) return fail();
        if (nodes_.empty() || timestamp < nodes_.rbegin()->first) return false;
        if (timestamp == nodes_.rbegin()->first) {
            if (!nodes_.rbegin()->second.x.allFinite()) return fail();
            state = nodes_.rbegin()->second.x;
            return true;
        }
        auto cpi = integrate(nodes_.rbegin()->first, timestamp, nodes_.rbegin()->second);
        if (!cpi) return fail();
        const Node predicted = predict(nodes_.rbegin()->second, *cpi);
        if (!predicted.x.allFinite()) return fail();
        state = predicted.x;
        return true;
    }

private:
    struct Node {
        Eigen::Matrix<double, 16, 1> x;
        double* q() { return x.data(); }
        double* p() { return x.data() + 4; }
        double* v() { return x.data() + 7; }
        double* bg() { return x.data() + 10; }
        double* ba() { return x.data() + 13; }
        std::vector<double*> blocks() { return {q(), p(), v(), bg(), ba()}; }
    };
    struct Feature {
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        std::map<double, Eigen::Vector2d> observations;
        bool triangulated = false;
    };
    std::map<double, Node> nodes_;
    std::map<size_t, Feature> features_;
    std::vector<ov_core::ImuData> imu_;
    Eigen::Matrix<double, 8, 1> calibration_;
    Eigen::Matrix<double, 7, 1> extrinsics_;
    int image_width_ = 0, image_height_ = 0;
    std::vector<double> prior_times_;
    Eigen::VectorXd prior_x_, prior_g_;
    Eigen::MatrixXd prior_H_;
    double last_frame_ = -1;
    mutable bool failed_ = false; // state_at can also discover a hard integration failure

    bool fail() const { failed_ = true; return false; }

    void trim_imu_before(double timestamp) {
        auto retain = std::lower_bound(imu_.begin(), imu_.end(), timestamp,
            [](const ov_core::ImuData& sample, double t) { return sample.timestamp < t; });
        if (retain != imu_.begin()) --retain;
        imu_.erase(imu_.begin(), retain);
    }

    int image_cell(const Eigen::Vector2d& pixel) const {
        const int column = std::clamp(static_cast<int>(5 * pixel.x() / image_width_), 0, 4);
        const int row = std::clamp(static_cast<int>(4 * pixel.y() / image_height_), 0, 3);
        return 5 * row + column;
    }

    static bool pseudo_inverse(const Eigen::MatrixXd& H, Eigen::MatrixXd& inverse) {
        if (H.rows() == 0 || H.rows() != H.cols() || !H.allFinite()) return false;
        const Eigen::VectorXd scaling = information_scaling(H);
        const Eigen::MatrixXd normalized = scaling.asDiagonal() * (0.5 * (H + H.transpose())) * scaling.asDiagonal();
        if (!scaling.allFinite() || !normalized.allFinite()) return false;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(normalized);
        if (eig.info() != Eigen::Success || !eig.eigenvalues().allFinite() || !eig.eigenvectors().allFinite()) return false;
        const double cutoff = std::max(1e-12, eig.eigenvalues().maxCoeff() * 1e-12);
        // H^-1 = D (D H D)^-1 D. In the rank-deficient case this is a
        // generalized inverse in normalized coordinates, suitable for the
        // compatible PSD blocks of a Schur complement.
        Eigen::MatrixXd candidate = scaling.asDiagonal() * eig.eigenvectors() *
            eig.eigenvalues().unaryExpr([&](double v) { return v > cutoff ? 1.0 / v : 0.0; }).asDiagonal() *
            eig.eigenvectors().transpose() * scaling.asDiagonal();
        if (!candidate.allFinite()) return false;
        inverse.swap(candidate);
        return true;
    }
    std::shared_ptr<ov_core::CpiV1> integrate(double from, double to, const Node& node) const {
        if (!std::isfinite(from) || !std::isfinite(to) || to <= from || !node.x.allFinite() || imu_.size() < 2 ||
            from < imu_.front().timestamp - 1e-9) return nullptr;
        const auto& newest = imu_.back();
        const double sample_interval = newest.timestamp - imu_[imu_.size() - 2].timestamp;
        // The ABI permits a camera timestamp between IMU samples. Extend only
        // the causal fractional-sample tail; larger gaps are missing input.
        if (to - newest.timestamp > sample_interval + 1e-9) return nullptr;
        auto cpi = std::make_shared<ov_core::CpiV1>(sigma_w, sigma_wb, sigma_a, sigma_ab, true);
        Eigen::Vector3d bg = node.x.segment<3>(10), ba = node.x.segment<3>(13);
        cpi->setLinearizationPoints(bg, ba);
        // Clip each sample interval explicitly: the initializer's helper omits
        // the final sample when the requested end equals the newest IMU time.
        for (size_t i = 1; i < imu_.size(); ++i) {
            const auto& a = imu_[i - 1]; const auto& b = imu_[i];
            const double lo = std::max(from, a.timestamp), hi = std::min(to, b.timestamp);
            if (hi <= lo) continue;
            const double u = (lo - a.timestamp) / (b.timestamp - a.timestamp);
            const double v = (hi - a.timestamp) / (b.timestamp - a.timestamp);
            cpi->feed_IMU(lo, hi, (1 - u) * a.wm + u * b.wm, (1 - u) * a.am + u * b.am,
                                 (1 - v) * a.wm + v * b.wm, (1 - v) * a.am + v * b.am);
        }
        const double tail_start = std::max(from, newest.timestamp);
        if (to > tail_start)
            cpi->feed_IMU(tail_start, to, newest.wm, newest.am, newest.wm, newest.am);
        if (!std::isfinite(cpi->DT) || std::abs(cpi->DT - (to - from)) > 1e-6 || !cpi->alpha_tau.allFinite() ||
            !cpi->beta_tau.allFinite() || !cpi->q_k2tau.allFinite() || !cpi->P_meas.allFinite() ||
            !cpi->J_q.allFinite() || !cpi->J_b.allFinite() || !cpi->J_a.allFinite() ||
            !cpi->H_b.allFinite() || !cpi->H_a.allFinite()) return nullptr;
        // The upstream IMU factor assumes a valid Cholesky decomposition.
        Eigen::LLT<Eigen::Matrix<double, 15, 15>> covariance_factor(cpi->P_meas);
        if (covariance_factor.info() != Eigen::Success) return nullptr;
        return cpi;
    }
    Node predict(const Node& before, const ov_core::CpiV1& cpi) const {
        Node after = before;
        const Eigen::Matrix3d R = ov_core::quat_2_Rot(before.x.head<4>());
        after.x.head<4>() = ov_core::quat_multiply(cpi.q_k2tau, before.x.head<4>());
        after.x.segment<3>(4) = before.x.segment<3>(4) + cpi.DT * before.x.segment<3>(7) - 0.5 * cpi.DT * cpi.DT * gravity + R.transpose() * cpi.alpha_tau;
        after.x.segment<3>(7) = before.x.segment<3>(7) - cpi.DT * gravity + R.transpose() * cpi.beta_tau;
        return after;
    }
    bool triangulate(Feature& feature) const {
        if (feature.observations.size() < 3) return false;
        const Eigen::Matrix3d Ric = ov_core::quat_2_Rot(extrinsics_.head<4>());
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero(); Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
        double maximum_baseline = 0; Eigen::Vector3d first_center = Eigen::Vector3d::Zero();
        bool first = true;
        for (const auto& obs : feature.observations) {
            const auto& node = nodes_.at(obs.first);
            const Eigen::Matrix3d R = Ric * ov_core::quat_2_Rot(node.x.head<4>());
            const Eigen::Vector3d center = node.x.segment<3>(4) - R.transpose() * extrinsics_.tail<3>();
            if (first) { first_center = center; first = false; }
            maximum_baseline = std::max(maximum_baseline, (center - first_center).norm());
            Eigen::Vector3d ray((obs.second.x() - calibration_(2)) / calibration_(0), (obs.second.y() - calibration_(3)) / calibration_(1), 1);
            ray = R.transpose() * ray.normalized();
            Eigen::Matrix3d A = Eigen::Matrix3d::Identity() - ray * ray.transpose(); H += A; rhs += A * center;
        }
        if (!H.allFinite() || !rhs.allFinite()) return fail();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(H);
        if (eig.info() != Eigen::Success || !eig.eigenvalues().allFinite()) return fail();
        if (maximum_baseline < 0.04 || eig.eigenvalues().minCoeff() < 1e-5) return false;
        Eigen::LDLT<Eigen::Matrix3d> factor(H);
        if (factor.info() != Eigen::Success) return fail();
        Eigen::Vector3d position = factor.solve(rhs);
        if (!position.allFinite()) return fail();
        double squared_error = 0;
        for (const auto& obs : feature.observations) {
            const auto& node = nodes_.at(obs.first);
            Eigen::Vector3d in_camera = Ric * ov_core::quat_2_Rot(node.x.head<4>()) * (position - node.x.segment<3>(4)) + extrinsics_.tail<3>();
            if (!in_camera.allFinite() || in_camera.z() < 0.1 || in_camera.z() > 100) return false;
            Eigen::Vector2d uv(calibration_(0) * in_camera.x() / in_camera.z() + calibration_(2), calibration_(1) * in_camera.y() / in_camera.z() + calibration_(3));
            squared_error += (uv - obs.second).squaredNorm();
        }
        if (squared_error / feature.observations.size() > 16.0) return false;
        feature.position = position; feature.triangulated = true; return true;
    }

    bool optimize() {
        if (failed_) return false;
        last_consumed = 0;
        ceres::Problem problem;
        auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
        const auto nodes_before = nodes_; const auto features_before = features_;
        for (auto& entry : nodes_) {
            auto& n = entry.second;
            problem.AddParameterBlock(n.q(), 4, new ov_init::State_JPLQuatLocal());
            for (double* p : {n.p(), n.v(), n.bg(), n.ba()}) problem.AddParameterBlock(p, 3);
            for (double* p : n.blocks()) ordering->AddElementToGroup(p, 1);
        }
        problem.AddParameterBlock(extrinsics_.data(), 4, new ov_init::State_JPLQuatLocal());
        problem.AddParameterBlock(extrinsics_.data() + 4, 3);
        problem.AddParameterBlock(calibration_.data(), 8);
        for (double* p : {extrinsics_.data(), extrinsics_.data() + 4, calibration_.data()}) problem.SetParameterBlockConstant(p);
        std::vector<double*> prior_parameters;
        for (double t : prior_times_) { auto blocks = nodes_.at(t).blocks(); prior_parameters.insert(prior_parameters.end(), blocks.begin(), blocks.end()); }
        auto* prior = new LocalPrior(prior_x_, prior_H_, prior_g_);
        if (!prior->valid) { delete prior; return fail(); }
        last_prior_rank = prior->rank;
        last_prior_min_eigenvalue = prior->minimum_eigenvalue;
        last_prior_min_retained_eigenvalue = prior->minimum_retained_eigenvalue;
        auto prior_id = problem.AddResidualBlock(prior, nullptr, prior_parameters);
        std::map<double, ceres::ResidualBlockId> imu_factors;
        auto previous = nodes_.begin();
        for (auto current = std::next(previous); current != nodes_.end(); ++current) {
            auto cpi = integrate(previous->first, current->first, previous->second);
            if (!cpi) return fail();
            auto& a = previous->second; auto& b = current->second;
            auto* factor = new ov_init::Factor_ImuCPIv1(cpi->DT, gravity, cpi->alpha_tau, cpi->beta_tau, cpi->q_k2tau, cpi->b_a_lin, cpi->b_w_lin, cpi->J_q, cpi->J_b, cpi->J_a, cpi->H_b, cpi->H_a, cpi->P_meas);
            imu_factors[previous->first] = problem.AddResidualBlock(factor, nullptr, a.q(), a.bg(), a.v(), a.ba(), a.p(), b.q(), b.bg(), b.v(), b.ba(), b.p());
            previous = current;
        }
        std::map<size_t, std::vector<ceres::ResidualBlockId>> image_factors;
        last_visual_factors = 0;
        last_triangulated = 0;
        for (auto& entry : features_) {
            auto& f = entry.second; if (!f.triangulated) continue;
            ++last_triangulated;
            problem.AddParameterBlock(f.position.data(), 3); ordering->AddElementToGroup(f.position.data(), 0);
            for (const auto& obs : f.observations) {
                auto& n = nodes_.at(obs.first);
                auto* factor = new ov_init::Factor_ImageReprojCalib(obs.second, pixel_sigma, false);
                image_factors[entry.first].push_back(problem.AddResidualBlock(factor, new ceres::HuberLoss(2.0), n.q(), n.p(), f.position.data(), extrinsics_.data(), extrinsics_.data() + 4, calibration_.data()));
                ++last_visual_factors;
            }
        }
        ceres::Solver::Options options;
        options.linear_solver_type = image_factors.empty() ? ceres::DENSE_QR : ceres::DENSE_SCHUR;
        if (!image_factors.empty()) options.linear_solver_ordering = ordering;
        options.max_num_iterations = maximum_iterations; options.num_threads = 1;
        options.max_solver_time_in_seconds = 1e9; options.logging_type = ceres::SILENT;
        options.function_tolerance = 1e-5;
        ceres::Solver::Summary summary; ceres::Solve(options, &problem, &summary);
        last_initial_cost = summary.initial_cost; last_final_cost = summary.final_cost;
        bool usable = summary.IsSolutionUsable() && std::isfinite(summary.initial_cost) &&
            std::isfinite(summary.final_cost) && summary.final_cost <= summary.initial_cost;
        for (const auto& entry : nodes_) usable = usable && entry.second.x.allFinite();
        for (const auto& entry : features_) usable = usable && entry.second.position.allFinite();
        if (!usable) {
            for (auto& entry : nodes_) entry.second.x = nodes_before.at(entry.first).x;
            for (auto& entry : features_) entry.second.position = features_before.at(entry.first).position;
            // A rejected nonlinear step is recoverable if the restored graph
            // evaluates. Its finite linearization can still be marginalized.
            double restored_cost = 0;
            ceres::Problem::EvaluateOptions restored_options;
            restored_options.num_threads = 1;
            if (!problem.Evaluate(restored_options, &restored_cost, nullptr, nullptr, nullptr) || !std::isfinite(restored_cost))
                return fail();
        } else ++optimizations;
        if (nodes_.size() > maximum_nodes && !marginalize_oldest(problem, prior_id, imu_factors, image_factors)) return fail();
        return usable;
    }

    bool marginalize_oldest(ceres::Problem& problem, ceres::ResidualBlockId prior_id,
                            const std::map<double, ceres::ResidualBlockId>& imu_factors,
                            const std::map<size_t, std::vector<ceres::ResidualBlockId>>& image_factors) {
        const double oldest = nodes_.begin()->first;
        std::set<size_t> consumed;
        for (const auto& entry : features_) if (entry.second.observations.count(oldest)) consumed.insert(entry.first);
        ceres::Problem::EvaluateOptions options;
        options.apply_loss_function = true; options.num_threads = 1;
        options.residual_blocks = {prior_id, imu_factors.at(oldest)};
        options.parameter_blocks = nodes_.begin()->second.blocks();
        int eliminated = 15;
        for (size_t id : consumed) {
            auto found = image_factors.find(id); if (found == image_factors.end()) continue;
            options.residual_blocks.insert(options.residual_blocks.end(), found->second.begin(), found->second.end());
            options.parameter_blocks.push_back(features_.at(id).position.data()); eliminated += 3;
        }
        std::vector<double> kept_times; Eigen::VectorXd kept_x(16 * (nodes_.size() - 1)); int ambient = 0;
        for (auto it = std::next(nodes_.begin()); it != nodes_.end(); ++it) {
            auto blocks = it->second.blocks(); options.parameter_blocks.insert(options.parameter_blocks.end(), blocks.begin(), blocks.end());
            kept_times.push_back(it->first); kept_x.segment(ambient, 16) = it->second.x; ambient += 16;
        }
        double cost = 0; std::vector<double> residuals; ceres::CRSMatrix sparse;
        if (!problem.Evaluate(options, &cost, &residuals, nullptr, &sparse) || !std::isfinite(cost)) return false;
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(sparse.num_rows, sparse.num_cols);
        for (int r = 0; r < sparse.num_rows; ++r)
            for (int k = sparse.rows[r]; k < sparse.rows[r + 1]; ++k) J(r, sparse.cols[k]) = sparse.values[k];
        const Eigen::MatrixXd H = J.transpose() * J;
        const Eigen::VectorXd g = J.transpose() * Eigen::Map<Eigen::VectorXd>(residuals.data(), residuals.size());
        if (!H.allFinite() || !g.allFinite() || !kept_x.allFinite()) return false;
        const int kept = H.rows() - eliminated;
        Eigen::MatrixXd eliminated_inverse;
        if (!pseudo_inverse(H.topLeftCorner(eliminated, eliminated), eliminated_inverse)) return false;
        Eigen::MatrixXd transfer = H.bottomLeftCorner(kept, eliminated) * eliminated_inverse;
        Eigen::MatrixXd candidate_H = H.bottomRightCorner(kept, kept) - transfer * H.topRightCorner(eliminated, kept);
        candidate_H = (0.5 * (candidate_H + candidate_H.transpose())).eval();
        Eigen::VectorXd candidate_g = g.tail(kept) - transfer * g.head(eliminated);
        if (!candidate_H.allFinite() || !candidate_g.allFinite()) return false;
        // Only the first retained node receives the marginalized IMU factor.
        // All later nodes participate through image poses only. Preserve these
        // exact structural zeros rather than equilibrating numerical remnants.
        for (size_t node = 1; node < kept_times.size(); ++node) {
            const int offset = static_cast<int>(15 * node + 6);
            candidate_H.middleRows(offset, 9).setZero();
            candidate_H.middleCols(offset, 9).setZero();
            candidate_g.segment(offset, 9).setZero();
        }
        // Commit the complete prior together; failures above preserve the old
        // prior and graph. The caller makes such a failure sticky and stops input.
        prior_H_.swap(candidate_H);
        prior_g_.swap(candidate_g);
        prior_x_.swap(kept_x);
        prior_times_.swap(kept_times);
        last_consumed = consumed.size(); // includes untriangulated tracks discarded with the old node
        // All factors of a consumed landmark were incorporated above. Future
        // observations of this same tracker ID form a fresh independent segment;
        // old observations must not be inserted again into that new segment.
        for (size_t id : consumed) {
            if (problem.HasParameterBlock(features_.at(id).position.data())) problem.RemoveParameterBlock(features_.at(id).position.data());
            features_.erase(id);
        }
        for (double* p : nodes_.begin()->second.blocks()) problem.RemoveParameterBlock(p);
        nodes_.erase(oldest);
        trim_imu_before(nodes_.begin()->first);
        return true;
    }
};
} // namespace relpose_openvins
#endif
