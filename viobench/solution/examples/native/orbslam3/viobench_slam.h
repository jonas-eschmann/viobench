#pragma once
#include <memory>
#include <string>
#include <vector>
#include "Tracking.h"
#include "Atlas.h"
namespace viobench_orbslam3 {
// Deterministic single-threaded monocular-inertial ORB-SLAM3 front end and mapper.
class ViobenchSlam {
public:
    explicit ViobenchSlam(const std::string& yaml);
    ~ViobenchSlam();
    ViobenchSlam(const ViobenchSlam&) = delete;
    ViobenchSlam& operator=(const ViobenchSlam&) = delete;
    void track(const cv::Mat& image, double time, const std::vector<ORB_SLAM3::IMU::Point>& imu);
    ORB_SLAM3::Tracking& tracker();
    ORB_SLAM3::Atlas& atlas();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
