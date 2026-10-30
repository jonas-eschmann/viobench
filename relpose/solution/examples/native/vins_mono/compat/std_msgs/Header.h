#pragma once
#include <ros/ros.h>
#include <cstdint>
#include <string>

namespace std_msgs {
struct Header {
    std::uint32_t seq = 0;
    ros::Time stamp;
    std::string frame_id;
};
}
