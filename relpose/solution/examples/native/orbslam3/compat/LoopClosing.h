// Local visual-inertial mapping only: no background loop-closure worker.
#ifndef RELPOSE_HEADLESS_LOOPCLOSING_H
#define RELPOSE_HEADLESS_LOOPCLOSING_H
#include <map>
#include "System.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
namespace ORB_SLAM3 {
class Map;
class KeyFrame;
class KeyFrameDatabase;
class LoopClosing {
public:
    using KeyFrameAndPose = std::map<KeyFrame*, g2o::Sim3, std::less<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<KeyFrame* const, g2o::Sim3>>>;
    KeyFrameDatabase* database = nullptr;
    void InsertKeyFrame(KeyFrame*);
    void RequestReset() {}
    void RequestResetActiveMap(Map*) {}
};
}

#endif
