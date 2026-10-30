// Native numerical checks for the estimate-only consensus output observer.
// Expected motion comes from analytic trajectories; no recorded ground truth is used.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "consensus_output.h"

namespace {

using Eigen::Matrix3d;
using Eigen::Vector3d;
namespace detail = consensus_output_detail;

int checks = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void near_vector(const Vector3d& actual, const Vector3d& expected,
                 double tolerance, const char* message) {
    require((actual - expected).norm() <= tolerance, message);
}

void near_rotation(const Matrix3d& actual, const Matrix3d& expected,
                   double tolerance, const char* message) {
    require((actual - expected).norm() <= tolerance, message);
}

void unchanged(const Output& actual, const Output& expected) {
    require(actual.timestamp == expected.timestamp, "rejection changed output timestamp");
    near_vector(actual.position, expected.position, 0.0, "rejection changed output position");
    near_rotation(actual.rotation, expected.rotation, 0.0, "rejection changed output rotation");
    near_vector(actual.linear_velocity, expected.linear_velocity, 0.0,
                "rejection changed output linear velocity");
    near_vector(actual.angular_velocity, expected.angular_velocity, 0.0,
                "rejection changed output angular velocity");
}

Output state(double timestamp, const Vector3d& position = Vector3d::Zero(),
             const Vector3d& world_velocity = Vector3d::Zero(),
             const Matrix3d& rotation = Matrix3d::Identity()) {
    Output result;
    result.timestamp = timestamp;
    result.position = position;
    result.rotation = rotation;
    result.linear_velocity = rotation.transpose() * world_velocity;
    return result;
}

void identical_update(ConsensusOutput& observer, const Output& input, Output& output) {
    require(observer.update(input, input, input, output), "identical update rejected");
}

void check_geometric_median() {
    Vector3d median;
    require(detail::geometric_median(
                {Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(0, 1, 0)}, median),
            "right-triangle median rejected");
    const double coordinate = (3.0 - std::sqrt(3.0)) / 6.0;
    near_vector(median, Vector3d(coordinate, coordinate, 0), 1e-14,
                "incorrect known Fermat point");

    require(detail::geometric_median(
                {Vector3d(0, 0, 0), Vector3d(1, 0, 0), Vector3d(1000, 0, 0)}, median),
            "collinear median rejected");
    near_vector(median, Vector3d(1, 0, 0), 0.0,
                "collinear outlier must leave the middle point as median");

    require(detail::geometric_median(
                {Vector3d(2, 3, 4), Vector3d(2, 3, 4), Vector3d(-1e4, 8e3, 9e3)}, median),
            "duplicate-majority median rejected");
    near_vector(median, Vector3d(2, 3, 4), 0.0,
                "duplicate majority must reject an arbitrary outlier");

    std::mt19937_64 generator(561);
    std::normal_distribution<double> normal;
    const Matrix3d rotation = detail::rotation_exp(Vector3d(0.4, -0.7, 0.2));
    const Vector3d translation(1.2, -3.0, 0.4);
    double maximum_invariance_error = 0.0;
    double maximum_subgradient_residual = 0.0;

    for (int sample = 0; sample < 10000; ++sample) {
        std::array<Vector3d, 3> points;
        for (auto& point : points)
            for (int coordinate_index = 0; coordinate_index < 3; ++coordinate_index)
                point(coordinate_index) = normal(generator);
        require(detail::geometric_median(points, median), "random triangle rejected");

        std::array<Vector3d, 3> transformed_points;
        for (int i = 0; i < 3; ++i)
            transformed_points[i] = rotation * points[i] + translation;
        Vector3d transformed_median;
        require(detail::geometric_median(transformed_points, transformed_median),
                "transformed triangle rejected");
        maximum_invariance_error = std::max(
            maximum_invariance_error,
            (transformed_median - (rotation * median + translation)).norm());

        // For the convex sum-of-distances objective, zero must belong to its
        // subgradient. A coincident input contributes a unit ball instead of
        // a normalized displacement vector.
        Vector3d gradient = Vector3d::Zero();
        int coincident = 0;
        for (const auto& point : points) {
            const Vector3d displacement = median - point;
            if (displacement.norm() < 1e-10)
                ++coincident;
            else
                gradient += displacement.normalized();
        }
        maximum_subgradient_residual = std::max(
            maximum_subgradient_residual, std::max(0.0, gradient.norm() - coincident));

        std::swap(points[0], points[2]);
        Vector3d permuted_median;
        require(detail::geometric_median(points, permuted_median),
                "permuted triangle rejected");
        near_vector(permuted_median, median, 1e-10,
                    "geometric median depends on hypothesis order");
    }

    require(maximum_invariance_error < 1e-10, "geometric median depends on coordinate frame");
    require(maximum_subgradient_residual < 1e-9, "geometric median is not a convex optimum");
    std::cout << "geometry: 10000 triangles, max rigid-frame error="
              << maximum_invariance_error << ", max subgradient residual="
              << maximum_subgradient_residual << '\n';
}

void check_rotation_mean() {
    const Matrix3d frame = detail::rotation_exp(Vector3d(0.4, 0.2, -0.1));
    std::array<Matrix3d, 3> rotations = {
        frame * detail::rotation_exp(Vector3d(0, 0, -0.2)),
        frame,
        frame * detail::rotation_exp(Vector3d(0, 0, 0.5)),
    };
    Matrix3d mean;
    require(detail::rotation_mean(rotations, mean), "rotation mean did not converge");
    near_rotation(mean, frame * detail::rotation_exp(Vector3d(0, 0, 0.1)), 1e-12,
                  "incorrect equal-weight mean for rotations about a common axis");

    rotations = {
        detail::rotation_exp(Vector3d(0.2, 0.1, -0.1)),
        detail::rotation_exp(Vector3d(-0.1, 0.3, 0.05)),
        detail::rotation_exp(Vector3d(0.05, -0.1, 0.2)),
    };
    require(detail::rotation_mean(rotations, mean), "noncommuting rotation mean failed");
    Vector3d gradient = Vector3d::Zero();
    for (const auto& rotation : rotations)
        gradient += detail::rotation_log(mean.transpose() * rotation);
    near_vector(gradient, Vector3d::Zero(), 1e-11,
                "intrinsic rotation mean is not stationary");

    const Matrix3d camera_frame = detail::rotation_exp(Vector3d(0.1, -0.2, 0.4));
    std::array<Matrix3d, 3> transformed;
    for (int i = 0; i < 3; ++i)
        transformed[i] = frame * rotations[i] * camera_frame;
    Matrix3d transformed_mean;
    require(detail::rotation_mean(transformed, transformed_mean),
            "transformed rotation mean failed");
    near_rotation(transformed_mean, frame * mean * camera_frame, 1e-11,
                  "rotation mean depends on reference or camera coordinate frame");

    std::swap(rotations[0], rotations[2]);
    Matrix3d permuted;
    require(detail::rotation_mean(rotations, permuted), "permuted rotation mean failed");
    near_rotation(permuted, mean, 1e-11, "rotation mean depends on hypothesis order");

    const Matrix3d half_turn =
        Eigen::AngleAxisd(std::acos(-1.0), Vector3d::UnitX()).toRotationMatrix();
    near_rotation(detail::rotation_exp(detail::rotation_log(half_turn)), half_turn, 1e-12,
                  "SO(3) logarithm is not defined near pi");
}

void check_analytic_motion() {
    const Vector3d initial_position(0.1, -0.2, 0.3);
    const Vector3d initial_velocity(0.4, -0.3, 0.2);
    const Vector3d acceleration(0.1, 0.2, -0.07);
    const Matrix3d rotation = detail::rotation_exp(Vector3d(0.4, -0.2, 0.3));

    for (bool accelerating : {false, true}) {
        ConsensusOutput observer;
        Output output;
        Vector3d actual_acceleration = Vector3d::Zero();
        if (accelerating) actual_acceleration = acceleration;
        for (int frame = 0; frame <= 100; ++frame) {
            const double time = frame * 0.05;
            const Vector3d velocity = initial_velocity + actual_acceleration * time;
            const Vector3d position = initial_position + initial_velocity * time +
                                     0.5 * actual_acceleration * time * time;
            Output input = state(time, position, velocity, rotation);
            input.angular_velocity = Vector3d(0.1, -0.2, 0.3);
            identical_update(observer, input, output);
            near_vector(output.position, position, 2e-13,
                        "observer delays an exact constant-acceleration trajectory");
            near_vector(output.linear_velocity, rotation.transpose() * velocity, 2e-12,
                        "endpoint twist does not retain the current acceleration velocity");
            near_vector(output.angular_velocity, input.angular_velocity, 1e-13,
                        "angular velocity is expressed in the wrong camera frame");
        }
    }

    ConsensusOutput observer;
    Output output;
    const Output first = state(0, Vector3d::Zero(), Vector3d(1, 2, 3));
    identical_update(observer, first, output);
    const double dt = 0.05;
    const Vector3d velocity(1.1, 2.2, 2.9);
    const Vector3d predicted = 0.5 * (first.linear_velocity + velocity) * dt;
    const Vector3d innovation(0.2, -0.1, 0.3);
    const Output second = state(dt, predicted + innovation, velocity, rotation);
    identical_update(observer, second, output);
    const double alpha = -std::expm1(-dt / 2.0);
    near_vector(output.position, predicted + alpha * innovation, 1e-14,
                "incorrect causal position correction");
    near_vector(output.linear_velocity,
                rotation.transpose() * (velocity + alpha * innovation / dt), 1e-12,
                "incorrect endpoint velocity correction");
    require((rotation * output.linear_velocity - output.position / dt).norm() > 0.01,
            "endpoint twist was replaced by a backward interval-average velocity");

    // The next prediction uses the previous raw velocity, not the previous
    // correction-inflated output velocity. Corrections remain output-only.
    const Vector3d next_velocity(1.2, 2.4, 2.8);
    const Vector3d next_prediction = output.position + 0.5 * (velocity + next_velocity) * dt;
    const Output third = state(2 * dt, next_prediction, next_velocity, rotation);
    identical_update(observer, third, output);
    near_vector(output.position, next_prediction, 1e-14,
                "output correction contaminated raw-velocity prediction");
    near_vector(output.linear_velocity, rotation.transpose() * next_velocity, 1e-12,
                "zero innovation does not retain the raw endpoint velocity");

    // Camera-axis velocities must be transformed into a common frame before
    // their geometric median is computed.
    ConsensusOutput world_velocity_observer;
    const Output a = state(0, Vector3d::Zero(), Vector3d(1, 0, 0));
    const Output b = state(0, Vector3d::Zero(), Vector3d(2, 0, 0), rotation);
    const Output c = state(0, Vector3d::Zero(), Vector3d(100, 0, 0), rotation.transpose());
    require(world_velocity_observer.update(a, b, c, output),
            "world-frame velocity consensus failed");
    near_vector(output.rotation * output.linear_velocity, Vector3d(2, 0, 0), 1e-12,
                "velocity median was computed in mismatched camera frames");
}

void check_failure_guards_and_reset() {
    ConsensusOutput observer;
    ConsensusOutput control;
    Output output = state(-1, Vector3d(8, 9, 10));
    const Output sentinel = output;
    const Output first = state(1, Vector3d(1, 2, 3), Vector3d(0.1, 0.2, 0.3));

    Output bad = first;
    bad.position.x() = std::numeric_limits<double>::quiet_NaN();
    require(!observer.update(first, bad, first, output), "nonfinite position accepted");
    unchanged(output, sentinel);
    bad = first;
    bad.timestamp += 0.001;
    require(!observer.update(first, bad, first, output), "mismatched timestamps accepted");
    unchanged(output, sentinel);
    bad = first;
    bad.timestamp = std::numeric_limits<double>::infinity();
    require(!observer.update(bad, bad, bad, output), "nonfinite timestamps accepted");
    unchanged(output, sentinel);
    bad = first;
    bad.rotation(0, 0) = 2;
    require(!observer.update(first, bad, first, output), "nonorthogonal rotation accepted");
    unchanged(output, sentinel);
    bad = first;
    bad.rotation.col(0) *= -1;
    require(!observer.update(first, bad, first, output), "reflection accepted as rotation");
    unchanged(output, sentinel);
    bad = first;
    bad.angular_velocity.z() = std::numeric_limits<double>::infinity();
    require(!observer.update(first, bad, first, output), "nonfinite angular velocity accepted");
    unchanged(output, sentinel);

    Output expected;
    identical_update(observer, first, output);
    identical_update(control, first, expected);
    unchanged(output, expected);
    const Output saved = output;
    require(!observer.update(first, first, first, output), "repeated timestamp accepted");
    unchanged(output, saved);
    bad = first;
    bad.timestamp = 0.5;
    require(!observer.update(bad, bad, bad, output), "backward timestamp accepted");
    unchanged(output, saved);
    bad = first;
    bad.timestamp = 1.05;
    bad.linear_velocity.y() = std::numeric_limits<double>::quiet_NaN();
    require(!observer.update(bad, bad, bad, output), "nonfinite linear velocity accepted");
    unchanged(output, saved);

    // Compare with a fresh observer that never saw the rejected calls. This
    // checks hidden history as well as the unchanged output argument.
    const Output next = state(1.05, Vector3d(1.1, 2.2, 3.3), Vector3d(0.3, 0.4, 0.5));
    identical_update(observer, next, output);
    identical_update(control, next, expected);
    unchanged(output, expected);

    observer.reset();
    const Output restart = state(0, Vector3d(-1, 4, 2), Vector3d(1, -3, 0.5));
    identical_update(observer, restart, output);
    near_vector(output.position, restart.position, 0.0, "reset retained previous position");
    near_vector(output.linear_velocity, restart.linear_velocity, 0.0,
                "reset retained previous raw velocity");

    Output alias = restart;
    ConsensusOutput alias_observer;
    require(alias_observer.update(alias, alias, alias, alias), "aliased output rejected");
    unchanged(alias, restart);
}

} // namespace

int main() {
    try {
        check_geometric_median();
        check_rotation_mean();
        check_analytic_motion();
        check_failure_guards_and_reset();
        std::cout << "PASS " << checks << " native numerical checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
