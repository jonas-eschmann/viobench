#pragma once

#include <cmath>
#include <stdexcept>

#include <g2o/core/base_unary_edge.h>
#include <g2o/types/types_six_dof_expmap.h>

namespace relpose_orbslam3 {

// Monocular bundle adjustment with one fixed camera still has a free scale.
// Preserve one existing camera-to-camera distance without constraining the
// second camera's orientation or the direction of its translation. This is a
// choice of visual-map units, never a metric measurement or ground-truth prior.
class ScaleGauge : public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ScaleGauge(const Eigen::Vector3d& fixed_center, double baseline, double information = 1e6)
        : fixed_center_(fixed_center) {
        if (!fixed_center.allFinite() || !std::isfinite(baseline) || baseline <= 0 ||
            !std::isfinite(information) || information <= 0)
            throw std::invalid_argument("invalid monocular scale gauge");
        setMeasurement(baseline);
        this->information()(0, 0) = information;
    }

    void computeError() override {
        const auto* camera = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const Eigen::Vector3d difference = camera->estimate().inverse().translation() - fixed_center_;
        _error[0] = difference.norm() / _measurement - 1.0;
    }

    void linearizeOplus() override {
        const auto* camera = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        const Eigen::Vector3d difference = camera->estimate().inverse().translation() - fixed_center_;
        const double distance = difference.norm();
        _jacobianOplusXi.setZero();
        // g2o applies left increments to T_camera_world, ordered [rotation, translation].
        // The world camera centre is unchanged by a left rotation; d centre/d v = -R^T.
        if (distance > 0)
            _jacobianOplusXi.template block<1, 3>(0, 3) =
                -difference.transpose() * camera->estimate().rotation().toRotationMatrix().transpose() /
                (distance * _measurement);
    }

    // This adapter does not serialize optimization graphs.
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }

private:
    Eigen::Vector3d fixed_center_;
};

}  // namespace relpose_orbslam3
