// Verify the production square-root marginalizer against independent projections.
#include "marginalization_qr.h"
#include <cmath>
#include <iostream>
#include <random>

using Eigen::MatrixXd;
using Eigen::VectorXd;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

MatrixXd random_matrix(int rows, int columns) {
    static std::mt19937 generator(214);
    static std::normal_distribution<double> normal;
    MatrixXd result(rows, columns);
    for (int col = 0; col < columns; ++col)
        for (int row = 0; row < rows; ++row) result(row, col) = normal(generator);
    return result;
}

void check_projection(const MatrixXd& J, const VectorXd& r, int marginalized) {
    const auto prior = relpose_vins_mono::marginalize_square_root(J, r, marginalized);
    const MatrixXd old = J.leftCols(marginalized), keep = J.rightCols(J.cols() - marginalized);
    const auto decomposition = old.completeOrthogonalDecomposition();
    const MatrixXd projected_J = keep - old * decomposition.solve(keep);
    const VectorXd projected_r = r - old * decomposition.solve(r);
    const MatrixXd expected_H = projected_J.transpose() * projected_J;
    const VectorXd expected_g = projected_J.transpose() * projected_r;
    require((prior.jacobian.transpose() * prior.jacobian - expected_H).norm() < 1e-10 * (1 + expected_H.norm()),
            "marginalized information differs from projection");
    require((prior.jacobian.transpose() * prior.residual - expected_g).norm() < 1e-10 * (1 + expected_g.norm()),
            "marginalized gradient differs from projection");
    // The discarded residual is constant with respect to retained variables.
    const double constant = projected_r.squaredNorm() - prior.residual.squaredNorm();
    for (int trial = 0; trial < 8; ++trial) {
        const VectorXd x = random_matrix(keep.cols(), 1);
        const double full_cost = (projected_J * x + projected_r).squaredNorm();
        const double prior_cost = (prior.jacobian * x + prior.residual).squaredNorm() + constant;
        require(std::abs(full_cost - prior_cost) < 1e-10 * (1 + full_cost), "marginalized objective changed");
    }
}

int main() {
    const MatrixXd regular = random_matrix(48, 17);
    const VectorXd regular_r = random_matrix(48, 1);
    check_projection(regular, regular_r, 7);
    const auto regular_prior = relpose_vins_mono::marginalize_square_root(regular, regular_r, 7);
    const MatrixXd H = regular.transpose() * regular;
    const VectorXd g = regular.transpose() * regular_r;
    const auto old_inverse = H.topLeftCorner(7, 7).ldlt();
    const MatrixXd schur_H = H.bottomRightCorner(10, 10) -
        H.bottomLeftCorner(10, 7) * old_inverse.solve(H.topRightCorner(7, 10));
    const VectorXd schur_g = g.tail(10) - H.bottomLeftCorner(10, 7) * old_inverse.solve(g.head(7));
    require((regular_prior.jacobian.transpose() * regular_prior.jacobian - schur_H).norm() < 1e-10,
            "well-conditioned Schur information differs");
    require((regular_prior.jacobian.transpose() * regular_prior.residual - schur_g).norm() < 1e-10,
            "well-conditioned Schur gradient differs");
    MatrixXd deficient = random_matrix(42, 15);
    deficient.col(3) = 2 * deficient.col(0);
    deficient.col(4).setZero();
    deficient.col(8).setZero();
    const VectorXd r = random_matrix(42, 1);
    check_projection(deficient, r, 5);
    const auto rank_prior = relpose_vins_mono::marginalize_square_root(deficient, r, 5);
    require(rank_prior.jacobian.col(3).norm() == 0, "absent retained variable acquired information");

    // A retained direction that can be fully explained by marginalized state
    // must remain a gauge freedom, even when its columns are individually nonzero.
    MatrixXd gauge = random_matrix(36, 9);
    gauge.col(7) = gauge.col(5) + 2 * gauge.col(1);
    const auto gauge_prior = relpose_vins_mono::marginalize_square_root(gauge, random_matrix(36, 1), 4);
    VectorXd null = VectorXd::Zero(5);
    null[1] = 1;
    null[3] = -1;
    require((gauge_prior.jacobian * null).norm() < 1e-12, "marginalization constrained a gauge freedom");

    // A weak retained direction follows a million-unit component explained by
    // an old state. Normal-equation subtraction loses its 1e-10 information.
    const MatrixXd basis = random_matrix(32, 8).householderQr().householderQ() * MatrixXd::Identity(32, 8);
    MatrixXd mixed(32, 8);
    mixed.leftCols(4) = basis.leftCols(4) * VectorXd::LinSpaced(4, 1e5, 1e8).asDiagonal();
    mixed.col(4) = 1e6 * basis.col(0) + 1e-5 * basis.col(4);
    mixed.col(5) = 1e7 * basis.col(5);
    mixed.col(6).setZero();
    mixed.col(7) = 0.2 * basis.col(6);
    const auto weak = relpose_vins_mono::marginalize_square_root(mixed, basis.col(4), 4);
    const MatrixXd information = weak.jacobian.transpose() * weak.jacobian;
    const VectorXd gradient = weak.jacobian.transpose() * weak.residual;
    require(std::abs(information(0, 0) / 1e-10 - 1) < 1e-3, "weak scale information lost");
    require(std::abs(gradient[0] / 1e-5 - 1) < 1e-3, "weak scale gradient lost");
    require(weak.jacobian.col(2).norm() == 0, "zero column changed by compression");
    require(std::abs(information(1, 1) / 1e14 - 1) < 1e-12, "stiff information changed");

    const MatrixXd direct_J = random_matrix(20, 6);
    const VectorXd direct_r = random_matrix(20, 1);
    const auto direct = relpose_vins_mono::marginalize_square_root(direct_J, direct_r, 0);
    require((direct.jacobian.transpose() * direct.jacobian - direct_J.transpose() * direct_J).norm() < 1e-10,
            "compression without elimination changed information");
    bool rejected = false;
    MatrixXd invalid = direct_J;
    invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
    try { relpose_vins_mono::marginalize_square_root(invalid, direct_r, 2); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "nonfinite factor accepted");
    std::cout << "VINS marginalization: projection, rank, weak scale and finite checks passed\n";
}
