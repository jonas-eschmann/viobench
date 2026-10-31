#ifndef VIOBENCH_HEADLESS_MAPDRAWER_H
#define VIOBENCH_HEADLESS_MAPDRAWER_H
#include <sophus/se3.hpp>
namespace ORB_SLAM3 {
class MapDrawer {
public:
    void SetCurrentCameraPose(const Sophus::SE3f&) {}
};
}

#endif
