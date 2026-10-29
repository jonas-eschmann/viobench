// Minimal native solution: a wasm32-wasip1 reactor implementing the relpose ABI (runtime/native/relpose_abi.h) without
// Python. It reports the identity pose and a zero twist and shows where to put state and how to read the config.
// Build:  relpose bundle <this directory> -o solution.wasm
#include "relpose_abi.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {


struct Config {
    int width = 0, height = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
};

// A deliberately tiny JSON number lookup: finds `"key": <number>`. Good enough for the flat config's scalars; use a
// real JSON parser (e.g. nlohmann/json, header-only, works with wasi-sdk) for T_cam_imu.
bool find_number(const std::string& json, const char* key, double& out) {
    std::string needle = std::string("\"") + key + "\"";
    size_t at = json.find(needle);
    if (at == std::string::npos) return false;
    at = json.find(':', at + needle.size());
    if (at == std::string::npos) return false;
    out = std::strtod(json.c_str() + at + 1, nullptr);
    return true;
}

struct State {
    Config config;
    long frames = 0;
    std::vector<uint8_t> previous_frame;  // keep the previous frame: the host reuses its buffer every step
    double position[3] = {0, 0, 0};
    double quaternion[4] = {0, 0, 0, 1};  // x, y, z, w
    double velocity[3] = {0, 0, 0};       // of the camera centre, in camera axes
    double angular_velocity[3] = {0, 0, 0};  // in camera axes
};

State state;

}  // namespace

int32_t abi_version() { return RELPOSE_ABI_VERSION; }

void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }

// Optional (320 is the default without it); return 640 for 640x480 frames. Called once, before init().
int32_t frame_width() { return 320; }

// Called once per instance; step() is only called after init() succeeded.
int32_t init(const char* json, uint32_t length) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a reactor never exits: unbuffered prints reach the host log
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::string text(json, length);
    double value;
    if (find_number(text, "width", value)) state.config.width = static_cast<int>(value);
    if (find_number(text, "height", value)) state.config.height = static_cast<int>(value);
    find_number(text, "fx", state.config.fx);
    find_number(text, "fy", state.config.fy);
    find_number(text, "cx", state.config.cx);
    find_number(text, "cy", state.config.cy);
    if (state.config.width <= 0 || state.config.height <= 0) {
        std::fprintf(stderr, "config without image size\n");
        return RELPOSE_STATUS_ERROR;
    }
    state.previous_frame.resize(static_cast<size_t>(state.config.width) * state.config.height * 3);  // RGB
    std::printf("init: %dx%d fx=%.1f cx=%.1f\n", state.config.width, state.config.height, state.config.fx, state.config.cx);
    return RELPOSE_STATUS_OK;
}

// frame: height*width*3 uint8 RGB; imu: imu_count rows [t, ax, ay, az, gx, gy, gz];
// out: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz] (all 13 values, every step)
int32_t step(double t, const uint8_t* frame, const double* imu, int32_t imu_count, double* out) {
    state.frames++;
    double mean = 0;
    for (size_t i = 0; i < state.previous_frame.size(); i++) mean += frame[i];
    mean /= static_cast<double>(state.previous_frame.size());
    double gyro_norm = 0;
    for (int32_t i = 0; i < imu_count; i++) {
        const double* row = imu + 7 * i;
        gyro_norm += std::sqrt(row[4] * row[4] + row[5] * row[5] + row[6] * row[6]);
    }
    if (state.frames == 1) std::printf("first frame t=%.3f mean=%.1f imu rows=%d\n", t, mean, imu_count);
    std::memcpy(state.previous_frame.data(), frame, state.previous_frame.size());
    // identity pose, standing still: replace with your estimate of the current camera frame in the first camera frame
    // and of its twist
    for (int i = 0; i < 3; i++) out[i] = state.position[i];
    for (int i = 0; i < 4; i++) out[3 + i] = state.quaternion[i];
    for (int i = 0; i < 3; i++) out[7 + i] = state.velocity[i];
    for (int i = 0; i < 3; i++) out[10 + i] = state.angular_velocity[i];
    return RELPOSE_STATUS_OK;
}
