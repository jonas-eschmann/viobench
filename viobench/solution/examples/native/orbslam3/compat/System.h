// Headless integration surface used in place of upstream's threaded System.
#ifndef VIOBENCH_HEADLESS_SYSTEM_H
#define VIOBENCH_HEADLESS_SYSTEM_H
#include <iostream>
#include <string>
#include <stdexcept>
namespace ORB_SLAM3 {
class Map;
class Verbose {
public:
    enum eLevel { VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE, VERBOSITY_VERY_VERBOSE, VERBOSITY_DEBUG };
    inline static eLevel th = VERBOSITY_QUIET;
    static void PrintMess(const std::string& text, eLevel level) { if (level <= th) std::cout << text << std::endl; }
    static void SetTh(eLevel level) { th = level; }
};
class System {
public:
    enum eSensor { MONOCULAR, STEREO, RGBD, IMU_MONOCULAR, IMU_STEREO, IMU_RGBD };
    bool reset_requested = false;
    void ResetActiveMap() { reset_requested = true; }
    void SaveTrajectoryEuRoC(const std::string&, Map* = nullptr) { throw std::logic_error("trajectory files are unavailable in a viobench reactor"); }
    void SaveKeyFrameTrajectoryEuRoC(const std::string&, Map* = nullptr) { throw std::logic_error("trajectory files are unavailable in a viobench reactor"); }
};
}

#endif
