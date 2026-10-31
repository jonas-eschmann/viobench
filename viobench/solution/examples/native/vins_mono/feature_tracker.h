// Pinhole-only adaptation of VINS-Mono's feature tracker (GPL-3.0).
// The ABI provides rectified images and intrinsics directly, so no camodocal/ROS files are needed.
#pragma once

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

namespace viobench_vins_mono {
using Observations = std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;

class FeatureTracker {
    struct Track {
        cv::Point2f pixel, previous;
        int id, age;
    };
    std::vector<cv::Mat> previous_pyramid, anchor_pyramid;
    std::vector<Track> tracks;
    std::map<int, cv::Point2f> anchor_points;
    Eigen::Matrix3d anchor_rotation = Eigen::Matrix3d::Identity();
    double previous_time = 0, anchor_time = 0;
    int next_id = 0;

    // Estimate residual translation after undoing rotation-induced patch deformation.
    // All arguments and returned observations use actual calibrated image coordinates.
    std::vector<unsigned char> flow(const std::vector<cv::Mat>& reference,
                                   const std::vector<cv::Mat>& current,
                                   const std::vector<cv::Point2f>& before,
                                   std::vector<cv::Point2f>& after,
                                   const Eigen::Matrix3d& rotation_prior, bool use_guess) const {
        const std::vector<cv::Mat>* target = &current;
        std::vector<cv::Mat> rotated_pyramid;
        cv::Matx33d H_current_reference = cv::Matx33d::eye();
        const bool compensate = rotation_prior.allFinite() &&
                                (rotation_prior - Eigen::Matrix3d::Identity()).norm() > 1e-10;
        if (compensate) {
            cv::Matx33d rotation;
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) rotation(row, col) = rotation_prior(row, col);
            const cv::Matx33d K(fx, 0, cx, 0, fy, cy, 0, 0, 1);
            const cv::Matx33d H_reference_current = K * rotation.t() * K.inv();
            H_current_reference = K * rotation * K.inv();
            cv::Mat rotated;
            cv::warpPerspective(current.front(), rotated, H_reference_current, reference.front().size(),
                                cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::buildOpticalFlowPyramid(rotated, rotated_pyramid, cv::Size(21, 21), 3);
            target = &rotated_pyramid;
            if (use_guess) {
                for (size_t i = 0; i < after.size(); ++i) {
                    const cv::Vec3d mapped = H_reference_current * cv::Vec3d(after[i].x, after[i].y, 1.0);
                    after[i] = mapped[2] > 1e-8 && std::isfinite(mapped[0]) && std::isfinite(mapped[1]) && std::isfinite(mapped[2])
                                   ? cv::Point2f(mapped[0] / mapped[2], mapped[1] / mapped[2]) : before[i];
                }
            }
        }
        if (!use_guess) after = before;
        std::vector<cv::Point2f> back = before;
        std::vector<unsigned char> forward_status, backward_status;
        std::vector<float> error;
        const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
        cv::calcOpticalFlowPyrLK(reference, *target, before, after, forward_status, error,
                                cv::Size(21, 21), 3, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
        cv::calcOpticalFlowPyrLK(*target, reference, after, back, backward_status, error,
                                cv::Size(21, 21), 3, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
        for (size_t i = 0; i < after.size(); ++i) {
            if (compensate) {
                const cv::Vec3d mapped = H_current_reference * cv::Vec3d(after[i].x, after[i].y, 1.0);
                if (!std::isfinite(mapped[0]) || !std::isfinite(mapped[1]) ||
                    !std::isfinite(mapped[2]) || mapped[2] <= 1e-8) {
                    forward_status[i] = 0;
                    after[i] = cv::Point2f(-1, -1);
                } else {
                    after[i] = cv::Point2f(mapped[0] / mapped[2], mapped[1] / mapped[2]);
                }
            }
            const auto& p = after[i];
            forward_status[i] = forward_status[i] && backward_status[i] &&
                                cv::norm(before[i] - back[i]) < 1.0 &&
                                p.x >= 1 && p.y >= 1 && p.x < current.front().cols - 1 && p.y < current.front().rows - 1;
        }
        return forward_status;
    }

public:
    double fx = 1, fy = 1, cx = 0, cy = 0;
    // Direct re-observation every quarter second limits chained LK drift. Zero disables it.
    double anchor_interval = 0.25;
    static constexpr int max_features = 180;
    static constexpr int min_distance = 10;

    // R_current_previous rotates a bearing from the preceding camera into this camera.
    Observations process(const cv::Mat& image, double time,
                         const Eigen::Matrix3d& R_current_previous = Eigen::Matrix3d::Identity()) {
        std::vector<cv::Mat> image_pyramid;
        cv::buildOpticalFlowPyramid(image, image_pyramid, cv::Size(21, 21), 3);
        anchor_rotation = R_current_previous * anchor_rotation;
        const bool anchor_due = anchor_interval > 0 &&
                                (anchor_pyramid.empty() || time - anchor_time >= anchor_interval - 1e-6);
        std::vector<Track> continued;
        if (!tracks.empty()) {
            std::vector<cv::Point2f> before, after;
            for (const auto& track : tracks) before.push_back(track.pixel);
            const auto status = flow(previous_pyramid, image_pyramid, before, after, R_current_previous, false);
            for (size_t i = 0; i < tracks.size(); ++i)
                if (status[i]) continued.push_back({after[i], before[i], tracks[i].id, tracks[i].age + 1});

            // The adjacent-frame track is only an initial guess. Successful direct matches
            // replace its accumulated coordinates; failed matches keep the ordinary track.
            if (anchor_due && !anchor_pyramid.empty()) {
                before.clear();
                after.clear();
                std::vector<size_t> indices;
                for (size_t i = 0; i < continued.size(); ++i) {
                    const auto match = anchor_points.find(continued[i].id);
                    if (match != anchor_points.end()) {
                        before.push_back(match->second);
                        after.push_back(continued[i].pixel);
                        indices.push_back(i);
                    }
                }
                if (!indices.empty()) {
                    const auto anchor_status = flow(anchor_pyramid, image_pyramid, before, after, anchor_rotation, true);
                    for (size_t i = 0; i < indices.size(); ++i) {
                        auto& track = continued[indices[i]];
                        if (anchor_status[i] && cv::norm(after[i] - track.pixel) < 3.0) track.pixel = after[i];
                    }
                }
            }
            if (continued.size() >= 8) {
                before.clear();
                after.clear();
                for (const auto& track : continued) {
                    before.push_back(track.previous);
                    after.push_back(track.pixel);
                }
                std::vector<unsigned char> inliers;
                const auto fundamental = cv::findFundamentalMat(before, after, cv::FM_RANSAC, 1.0, 0.99, inliers);
                if (!fundamental.empty() && inliers.size() == continued.size()) {
                    size_t keep = 0;
                    for (size_t i = 0; i < continued.size(); ++i)
                        if (inliers[i]) continued[keep++] = continued[i];
                    continued.resize(keep);
                }
            }
        }

        // Prefer established tracks while reserving space for new corners across the image.
        std::stable_sort(continued.begin(), continued.end(), [](const Track& a, const Track& b) { return a.age > b.age; });
        cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
        tracks.clear();
        for (const auto& track : continued) {
            const cv::Point pixel(cvRound(track.pixel.x), cvRound(track.pixel.y));
            if (mask.at<unsigned char>(pixel)) {
                tracks.push_back(track);
                cv::circle(mask, pixel, min_distance, cv::Scalar(0), -1);
            }
        }
        std::vector<cv::Point2f> corners;
        const int needed = max_features - static_cast<int>(tracks.size());
        if (needed > 0) cv::goodFeaturesToTrack(image, corners, needed, 0.01, min_distance, mask);
        for (const auto& corner : corners) tracks.push_back({corner, corner, next_id++, 1});

        Observations observations;
        for (const auto& track : tracks) {
            if (track.age < 2) continue;  // upstream publishes only features that survived a second image
            Eigen::Matrix<double, 7, 1> measurement;
            measurement << (track.pixel.x - cx) / fx, (track.pixel.y - cy) / fy, 1.0,
                           track.pixel.x, track.pixel.y,
                           (track.pixel.x - track.previous.x) / (fx * (time - previous_time)),
                           (track.pixel.y - track.previous.y) / (fy * (time - previous_time));
            observations[track.id].emplace_back(0, measurement);
        }
        previous_pyramid = std::move(image_pyramid);
        previous_time = time;
        if (anchor_due) {
            anchor_pyramid = previous_pyramid;
            anchor_points.clear();
            for (const auto& track : tracks) anchor_points.emplace(track.id, track.pixel);
            anchor_rotation.setIdentity();
            anchor_time = time;
        }
        return observations;
    }
};
}  // namespace viobench_vins_mono
