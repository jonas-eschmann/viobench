#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>

namespace viobench_vins_mono {

struct SquareRootPrior {
    Eigen::MatrixXd jacobian;
    Eigen::VectorXd residual;
};

// Eliminate the first `marginal_columns` tangent variables from ||J dx + r||².
// Operate on Jacobians, avoiding the squared conditioning and cancellation of
// normal-equation Schur subtraction. Column normalization preserves weak scale
// information alongside stiff IMU/bias constraints, including exact zero columns.
inline SquareRootPrior marginalize_square_root(const Eigen::MatrixXd& jacobian,
                                              const Eigen::VectorXd& residual,
                                              int marginal_columns) {
    const int rows = jacobian.rows(), columns = jacobian.cols();
    if (rows != residual.size() || marginal_columns < 0 || marginal_columns > columns)
        throw std::invalid_argument("invalid VINS marginalization dimensions");
    if (!jacobian.allFinite() || !residual.allFinite())
        throw std::runtime_error("nonfinite VINS marginalization factor");
    const int retained_columns = columns - marginal_columns;
    Eigen::MatrixXd normalized = jacobian;
    Eigen::VectorXd scales = Eigen::VectorXd::Ones(columns);
    for (int col = 0; col < columns; ++col) {
        const double norm = normalized.col(col).norm();
        if (norm > 0) scales[col] = norm;
        normalized.col(col) /= scales[col];
    }

    int eliminated_rank = 0;
    Eigen::MatrixXd projected(rows, retained_columns + 1);
    projected.leftCols(retained_columns) = normalized.rightCols(retained_columns);
    projected.col(retained_columns) = residual;
    if (marginal_columns > 0 && rows > 0) {
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> elimination(normalized.leftCols(marginal_columns));
        elimination.setThreshold(1e-12);
        eliminated_rank = elimination.rank();
        projected = (elimination.householderQ().adjoint() * projected).eval();
    }
    projected = projected.bottomRows(rows - eliminated_rank).eval();

    SquareRootPrior prior{Eigen::MatrixXd::Zero(retained_columns, retained_columns),
                          Eigen::VectorXd::Zero(retained_columns)};
    if (retained_columns > 0 && projected.rows() > 0) {
        // Compression removes only constant residual rows. Do not threshold the
        // retained system: its tiny nonzero rows can encode the metric scale.
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> compression(projected.leftCols(retained_columns));
        const int retained_rows = std::min<int>(retained_columns, projected.rows());
        Eigen::MatrixXd upper = compression.matrixQR().topRows(retained_rows)
                                   .template triangularView<Eigen::Upper>();
        prior.jacobian.topRows(retained_rows) = upper * compression.colsPermutation().transpose() *
                                               scales.tail(retained_columns).asDiagonal();
        const Eigen::VectorXd transformed = compression.householderQ().adjoint() * projected.col(retained_columns);
        prior.residual.head(retained_rows) = transformed.head(retained_rows);
    }
    if (!prior.jacobian.allFinite() || !prior.residual.allFinite())
        throw std::runtime_error("nonfinite VINS marginalization prior");
    return prior;
}
}  // namespace viobench_vins_mono
