#include "relpose_slam.h"
#include "KeyFrameDatabase.h"
#include "LocalMapping.h"
#include "System.h"
#include <cstring>
#include <stdexcept>

extern "C" const unsigned char orb_vocabulary[];

void ORB_SLAM3::LoopClosing::InsertKeyFrame(KeyFrame* keyframe) {
    database->add(keyframe);
}

namespace relpose_orbslam3 {
namespace {
// Build-time conversion preserves all upstream descriptors and double weights.
// Fields are little-endian; memcpy avoids alignment-dependent loads.
class Vocabulary : public ORB_SLAM3::ORBVocabulary {
public:
    Vocabulary() {
        const unsigned char* cursor = orb_vocabulary;
        auto u32 = [&]() { uint32_t x; std::memcpy(&x, cursor, 4); cursor += 4; return x; };
        m_k = u32(); m_L = u32();
        m_scoring = static_cast<DBoW2::ScoringType>(u32());
        m_weighting = static_cast<DBoW2::WeightingType>(u32());
        const uint32_t count = u32();
        cursor += 4; // each descriptor is 16-byte aligned for upstream int32_t loads
        createScoringObject();
        m_nodes.resize(count + 1);
        m_nodes[0].id = 0;
        m_words.reserve(count);
        for (uint32_t id = 1; id <= count; ++id) {
            auto& node = m_nodes[id];
            node.id = id;
            node.parent = u32();
            const bool leaf = *cursor; cursor += 4;
            if (node.parent >= id) throw std::runtime_error("invalid embedded ORB vocabulary");
            node.descriptor = cv::Mat(1, 32, CV_8U, const_cast<unsigned char*>(cursor));
            cursor += 32;
            std::memcpy(&node.weight, cursor, sizeof(double)); cursor += sizeof(double);
            m_nodes[node.parent].children.push_back(id);
            if (leaf) { node.word_id = m_words.size(); m_words.push_back(&node); }
            else node.children.reserve(m_k);
        }
    }
};
}
struct RelposeSlam::Impl {
    Vocabulary vocabulary;
    ORB_SLAM3::System system;
    ORB_SLAM3::FrameDrawer frame_drawer;
    ORB_SLAM3::MapDrawer map_drawer;
    ORB_SLAM3::LoopClosing loop_closer;
    ORB_SLAM3::Atlas atlas{0};
    ORB_SLAM3::KeyFrameDatabase database{vocabulary};
    std::unique_ptr<ORB_SLAM3::Tracking> tracker;
    std::unique_ptr<ORB_SLAM3::LocalMapping> mapper;
    explicit Impl(const std::string& yaml) {
        loop_closer.database = &database;
        atlas.SetInertialSensor();
        atlas.SetORBVocabulary(&vocabulary);
        atlas.SetKeyFrameDababase(&database);
        tracker = std::make_unique<ORB_SLAM3::Tracking>(&system, &vocabulary, &frame_drawer, &map_drawer,
            &atlas, &database, yaml, ORB_SLAM3::System::IMU_MONOCULAR, nullptr);
        mapper = std::make_unique<ORB_SLAM3::LocalMapping>(&system, &atlas, true, true, "relpose");
        mapper->mInitFr = 0;
        mapper->mThFarPoints = 0;
        mapper->mbFarPoints = false;
        tracker->SetLocalMapper(mapper.get());
        tracker->SetLoopClosing(&loop_closer);
        mapper->SetTracker(tracker.get());
        mapper->SetLoopCloser(&loop_closer);
    }
};
RelposeSlam::RelposeSlam(const std::string& yaml) : impl_(std::make_unique<Impl>(yaml)) {}
RelposeSlam::~RelposeSlam() = default;
ORB_SLAM3::Tracking& RelposeSlam::tracker() { return *impl_->tracker; }
ORB_SLAM3::Atlas& RelposeSlam::atlas() { return impl_->atlas; }
void RelposeSlam::track(const cv::Mat& image, double time, const std::vector<ORB_SLAM3::IMU::Point>& imu) {
    auto& p = *impl_;
    if (p.system.reset_requested) {
        p.tracker->ResetActiveMap();
        p.system.reset_requested = false;
    }
    for (const auto& sample : imu) p.tracker->GrabImuData(sample);
    p.tracker->GrabImageMonocular(image, time, "");
    // A failed track can request a reset after queuing a keyframe. Drain that
    // request first: mapping it could initialize a map that tracking has already
    // rejected and hide the failure by changing mState back to OK.
    if (p.system.reset_requested) {
        p.tracker->ResetActiveMap();
        p.system.reset_requested = false;
        return;
    }
    while (p.mapper->KeyframesInQueue() > 0) {
        p.mapper->RunOnce();
        if (p.system.reset_requested) {
            p.tracker->ResetActiveMap();
            p.system.reset_requested = false;
            return;
        }
    }
    // Synchronous BA finishes before this output. The frame that became the last
    // keyframe must use that keyframe's optimized inertial state, including the
    // full inertial BA performed after upstream UpdateFrameIMU during startup.
    auto* keyframe = p.tracker->GetLastKeyFrame();
    auto& current = p.tracker->mCurrentFrame;
    if (keyframe && keyframe->mnFrameId == current.mnId) {
        auto refresh = [&](ORB_SLAM3::Frame& frame) {
            if (p.atlas.isImuInitialized()) {
                frame.SetImuPoseVelocity(keyframe->GetImuRotation(), keyframe->GetImuPosition(), keyframe->GetVelocity());
                frame.SetNewBias(keyframe->GetImuBias());
            } else {
                frame.SetPose(keyframe->GetPose());
            }
        };
        refresh(current);
        if (p.tracker->mLastFrame.mnId == current.mnId) refresh(p.tracker->mLastFrame);
    }
}
}
