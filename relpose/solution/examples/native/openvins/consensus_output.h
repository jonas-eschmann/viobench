#ifndef RELPOSE_CONSENSUS_OUTPUT_H
#define RELPOSE_CONSENSUS_OUTPUT_H

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Pose maps current camera coordinates into the first camera's reference frame.
// Both velocity vectors are expressed in the current camera's axes.
struct Output {
    double timestamp = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d linear_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};

namespace consensus_output_detail {

inline bool valid_rotation(const Eigen::Matrix3d& R) {
    return R.allFinite() &&
        (R.transpose() * R - Eigen::Matrix3d::Identity()).norm() <= 1e-6 &&
        std::abs(R.determinant() - 1.0) <= 1e-6;
}

inline bool valid_output(const Output& value) {
    return std::isfinite(value.timestamp) && value.position.allFinite() &&
        valid_rotation(value.rotation) && value.linear_velocity.allFinite() &&
        value.angular_velocity.allFinite();
}

// For three points, the equal-weight geometric median is a >=120-degree
// triangle vertex or the interior Fermat point. Interior barycentric weights
// a/sin(A+60deg) are proportional to 1/[4*area+sqrt(3)*(b*b+c*c-a*a)].
// Distances and area make this construction independent of coordinate axes.
inline bool geometric_median(const std::array<Eigen::Vector3d, 3>& points,
                             Eigen::Vector3d& result) {
    for (const auto& point : points) if (!point.allFinite()) return false;
    constexpr int order[3][3] = {{0, 1, 2}, {1, 0, 2}, {2, 0, 1}};
    bool at_vertex = false;
    Eigen::Vector3d candidate = Eigen::Vector3d::Zero();
    for (const auto& indices : order) {
        const Eigen::Vector3d u = points[indices[1]] - points[indices[0]];
        const Eigen::Vector3d v = points[indices[2]] - points[indices[0]];
        const double nu = u.norm(), nv = v.norm(), dot = u.dot(v);
        if (!std::isfinite(nu) || !std::isfinite(nv) || !std::isfinite(dot) ||
            !std::isfinite(nu * nv)) return false;
        if (dot <= -0.5 * nu * nv || nu < 1e-12 || nv < 1e-12) {
            candidate = points[indices[0]];
            at_vertex = true;
        }
    }
    if (at_vertex) { result = candidate; return true; }
    Eigen::Vector3d sides_squared;
    sides_squared << (points[1] - points[2]).squaredNorm(),
                     (points[0] - points[2]).squaredNorm(),
                     (points[0] - points[1]).squaredNorm();
    const double four_area = 2.0 * (points[1] - points[0]).cross(points[2] - points[0]).norm();
    const Eigen::Vector3d denominators = Eigen::Vector3d::Constant(four_area) +
        std::sqrt(3.0) * (Eigen::Vector3d::Constant(sides_squared.sum()) - 2.0 * sides_squared);
    if (!denominators.allFinite() || denominators.minCoeff() <= 0.0) return false;
    // Normalize before summing; near a 120-degree vertex one weight dominates.
    const Eigen::Vector3d weights = denominators.minCoeff() * denominators.cwiseInverse();
    candidate = (weights(0) * points[0] + weights(1) * points[1] + weights(2) * points[2]) / weights.sum();
    if (!candidate.allFinite()) return false;
    result = candidate;
    return true;
}

inline Eigen::Vector3d rotation_log(const Eigen::Matrix3d& R) {
    Eigen::Vector3d skew;
    skew << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
    skew *= 0.5;
    const double sine = skew.norm(), cosine = 0.5 * (R.trace() - 1.0);
    // The regular expression matches the offline observer. Near pi the skew
    // part vanishes, so recover the axis from the symmetric part instead.
    if (sine < 1e-8 && cosine < 0.0) {
        const Eigen::AngleAxisd angle_axis(R);
        return angle_axis.angle() * angle_axis.axis();
    }
    if (sine > 0.0) return (std::atan2(sine, cosine) / sine) * skew;
    return skew;
}

inline Eigen::Matrix3d rotation_exp(const Eigen::Vector3d& vector) {
    Eigen::Matrix3d skew;
    skew << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(), 0.0;
    const double angle = vector.norm();
    if (angle < 1e-8) return Eigen::Matrix3d::Identity() + skew + 0.5 * skew * skew;
    return Eigen::Matrix3d::Identity() + (std::sin(angle) / angle) * skew +
        ((1.0 - std::cos(angle)) / (angle * angle)) * skew * skew;
}

inline bool rotation_mean(const std::array<Eigen::Matrix3d, 3>& rotations,
                          Eigen::Matrix3d& result) {
    for (const auto& R : rotations) if (!valid_rotation(R)) return false;
    Eigen::Matrix3d mean = rotations[0];
    for (int iteration = 0; iteration < 30; ++iteration) {
        Eigen::Vector3d delta = Eigen::Vector3d::Zero();
        for (const auto& R : rotations) delta += rotation_log(mean.transpose() * R);
        delta /= 3.0;
        if (!delta.allFinite()) return false;
        mean = mean * rotation_exp(delta);
        if (!valid_rotation(mean)) return false;
        if (delta.norm() < 1e-12) { result = mean; return true; }
    }
    // Widely inconsistent rotations may have no locally unambiguous mean.
    // Never turn a failed numerical solve into an apparently valid ABI state.
    return false;
}

} // namespace consensus_output_detail

// This observer changes only returned output. It never feeds corrections back
// into any estimator, and uses no ground truth or hypothesis-quality weights.
class ConsensusOutput {
public:
    void reset() {
        initialized_ = false;
        previous_timestamp_ = std::numeric_limits<double>::quiet_NaN();
        previous_position_.setZero();
        previous_raw_velocity_.setZero();
    }

    // On false, both the observer and out retain their previous values.
    bool update(const Output& chain, const Output& anchor, const Output& conservative, Output& out) {
        using namespace consensus_output_detail;
        const std::array<const Output*, 3> inputs = {&chain, &anchor, &conservative};
        for (const auto* input : inputs)
            if (!valid_output(*input) || input->timestamp != chain.timestamp) return false;
        const double dt = initialized_ ? chain.timestamp - previous_timestamp_ : 0.0;
        if (initialized_ && (!std::isfinite(dt) || dt <= 0.0)) return false;

        std::array<Eigen::Vector3d, 3> positions, velocities;
        std::array<Eigen::Matrix3d, 3> rotations;
        Eigen::Vector3d angular_world = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = *inputs[i];
            positions[i] = input.position;
            rotations[i] = input.rotation;
            velocities[i] = input.rotation * input.linear_velocity;
            angular_world += input.rotation * input.angular_velocity;
        }
        angular_world /= 3.0;
        Output candidate;
        candidate.timestamp = chain.timestamp;
        Eigen::Vector3d raw_position, raw_velocity;
        if (!geometric_median(positions, raw_position) || !geometric_median(velocities, raw_velocity) ||
            !rotation_mean(rotations, candidate.rotation) || !angular_world.allFinite()) return false;
        Eigen::Vector3d endpoint_velocity = raw_velocity;
        candidate.position = raw_position;
        if (initialized_) {
            const Eigen::Vector3d predicted = previous_position_ + 0.5 * (previous_raw_velocity_ + raw_velocity) * dt;
            const double alpha = -std::expm1(-dt / 2.0);
            candidate.position = predicted + alpha * (raw_position - predicted);
            // Endpoint velocity, not the backward interval-average derivative.
            endpoint_velocity = raw_velocity + (candidate.position - predicted) / dt;
        }
        candidate.linear_velocity = candidate.rotation.transpose() * endpoint_velocity;
        candidate.angular_velocity = candidate.rotation.transpose() * angular_world;
        if (!valid_output(candidate)) return false;

        out = candidate;
        previous_timestamp_ = candidate.timestamp;
        previous_position_ = candidate.position;
        previous_raw_velocity_ = raw_velocity;
        initialized_ = true;
        return true;
    }

private:
    bool initialized_ = false;
    double previous_timestamp_ = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d previous_position_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d previous_raw_velocity_ = Eigen::Vector3d::Zero();
};

#endif // RELPOSE_CONSENSUS_OUTPUT_H
