#ifndef VIOBENCH_HEADLESS_VIEWER_H
#define VIOBENCH_HEADLESS_VIEWER_H
namespace ORB_SLAM3 {
class Viewer {
public:
    void RequestStop() {}
    bool isStopped() const { return true; }
    void Release() {}
};
}

#endif
