#ifndef RELPOSE_HEADLESS_FRAMEDRAWER_H
#define RELPOSE_HEADLESS_FRAMEDRAWER_H
namespace ORB_SLAM3 {
class Tracking;
class FrameDrawer {
public:
    bool both = false;
    void Update(Tracking*) {}
};
}

#endif
