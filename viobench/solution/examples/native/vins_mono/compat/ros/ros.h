#pragma once

// Only the timestamp and diagnostic interfaces used by the estimator. No ROS runtime.
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace ros {
class NodeHandle;
class Time {
  public:
    explicit Time(double seconds = 0.0) : seconds_(seconds) {}
    double toSec() const { return seconds_; }
  private:
    double seconds_;
};
}

#define ROS_DEBUG(...) ((void)0)
#define ROS_INFO(...) ((void)0)
#define ROS_DEBUG_STREAM(...) ((void)0)
#define ROS_INFO_STREAM(...) ((void)0)
#define ROS_WARN(...) do { std::fprintf(stderr, "vins_mono: "); std::fprintf(stderr, __VA_ARGS__); std::fputc('\n', stderr); } while (0)
#define ROS_ERROR(...) ROS_WARN(__VA_ARGS__)
#define ROS_WARN_STREAM(value) do { std::ostringstream stream; stream << value; ROS_WARN("%s", stream.str().c_str()); } while (0)
#define ROS_ERROR_STREAM(value) ROS_WARN_STREAM(value)
#define ROS_ASSERT(condition) do { if (!(condition)) throw std::runtime_error("VINS assertion: " #condition); } while (0)
#define ROS_ASSERT_MSG(condition, ...) ROS_ASSERT(condition)
#define ROS_BREAK() throw std::runtime_error("VINS assertion failed")
