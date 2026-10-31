#ifndef VIOBENCH_HEADLESS_FRAMEDRAWER_H
#define VIOBENCH_HEADLESS_FRAMEDRAWER_H
namespace ORB_SLAM3 {
class Tracking;
class FrameDrawer {
public:
    bool both = false;
    void Update(Tracking*) {}
};
}

#endif
