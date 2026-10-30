// Pinhole-only adaptation of VINS-Mono's feature tracker (GPL-3.0).
// The ABI provides rectified images and intrinsics directly, so no camodocal/ROS files are needed.
#pragma once

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <map>
#include <vector>

namespace relpose_vins_mono {
using Observations = std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;

class FeatureTracker {
    struct Track {
        cv::Point2f pixel, previous;
        int id, age;
    };
    cv::Mat previous_image;
    std::vector<Track> tracks;
    double previous_time = 0;
    int next_id = 0;

public:
    double fx = 1, fy = 1, cx = 0, cy = 0;
    static constexpr int max_features = 180;
    static constexpr int min_distance = 10;

    Observations process(const cv::Mat& image, double time) {
        std::vector<Track> continued;
        if (!tracks.empty()) {
            std::vector<cv::Point2f> before, after, back;
            for (const auto& track : tracks) before.push_back(track.pixel);
            std::vector<unsigned char> forward_status, backward_status;
            std::vector<float> error;
            cv::calcOpticalFlowPyrLK(previous_image, image, before, after, forward_status, error, cv::Size(21, 21), 3);
            cv::calcOpticalFlowPyrLK(image, previous_image, after, back, backward_status, error, cv::Size(21, 21), 3);
            for (size_t i = 0; i < tracks.size(); ++i) {
                const auto& p = after[i];
                if (forward_status[i] && backward_status[i] && cv::norm(before[i] - back[i]) < 1.0 &&
                    p.x >= 1 && p.y >= 1 && p.x < image.cols - 1 && p.y < image.rows - 1)
                    continued.push_back({p, before[i], tracks[i].id, tracks[i].age + 1});
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
        previous_image = image.clone();
        previous_time = time;
        return observations;
    }
};
}  // namespace relpose_vins_mono
