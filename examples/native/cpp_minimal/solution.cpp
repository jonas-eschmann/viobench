// Minimal native solution implementing the ABI (docs/ABI.md) without Python: a wasm32-wasip1 reactor.
// It reports the identity pose and shows where to put state and how to read the config.
// Build (inside the toolchain image, see README.md):
//   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=/opt/wasi-sdk/share/cmake/wasi-sdk-p1.cmake && cmake --build build
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define EXPORT(name) __attribute__((export_name(#name)))

namespace {

constexpr int32_t STATUS_OK = 0;
constexpr int32_t STATUS_NO_ESTIMATE = 1;
constexpr int32_t STATUS_EXCEPTION = -1;
constexpr int32_t STATUS_PROTOCOL = -2;

struct Config {
    int width = 0, height = 0, channels = 1;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    double imu_rate_hz = 200;
};

// A deliberately tiny JSON number lookup: finds `"key": <number>` after an optional object prefix. Good enough for the
// flat config; use a real JSON parser (e.g. nlohmann/json, header-only, works with wasi-sdk) for anything more.
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
    bool initialized = false;
    long frames = 0;
    std::vector<uint8_t> previous_frame;  // keep the previous frame: the host reuses its buffer every step
    double position[3] = {0, 0, 0};
    double quaternion[4] = {0, 0, 0, 1};  // x, y, z, w
};

State state;

}  // namespace

EXPORT(abi_version) int32_t abi_version() { return 1; }

EXPORT(alloc) void* eecs116_alloc(uint32_t size) { return std::malloc(size ? size : 1); }

EXPORT(init) int32_t init(const char* json, uint32_t length) {
    if (state.initialized) return STATUS_PROTOCOL;
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a reactor never exits: unbuffered prints reach the host log
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::string text(json, length);
    double value;
    if (find_number(text, "width", value)) state.config.width = static_cast<int>(value);
    if (find_number(text, "height", value)) state.config.height = static_cast<int>(value);
    if (find_number(text, "channels", value)) state.config.channels = static_cast<int>(value);
    find_number(text, "fx", state.config.fx);
    find_number(text, "fy", state.config.fy);
    find_number(text, "cx", state.config.cx);
    find_number(text, "cy", state.config.cy);
    find_number(text, "rate_hz", state.config.imu_rate_hz);
    if (state.config.width <= 0 || state.config.height <= 0) {
        std::fprintf(stderr, "config without image size\n");
        return STATUS_EXCEPTION;
    }
    state.previous_frame.resize(static_cast<size_t>(state.config.width) * state.config.height * state.config.channels);
    std::printf("init: %dx%dx%d fx=%.1f cx=%.1f imu %.0f Hz\n", state.config.width, state.config.height, state.config.channels, state.config.fx, state.config.cx, state.config.imu_rate_hz);
    state.initialized = true;
    return STATUS_OK;
}

// frame: height*width*channels uint8; imu: imu_count rows [t, ax, ay, az, gx, gy, gz]; out: [x, y, z, qx, qy, qz, qw]
EXPORT(step) int32_t step(double t, const uint8_t* frame, const double* imu, int32_t imu_count, double* out) {
    if (!state.initialized || imu_count < 0) return STATUS_PROTOCOL;
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
    // identity pose: replace with your estimate of the current camera frame in the first camera frame
    for (int i = 0; i < 3; i++) out[i] = state.position[i];
    for (int i = 0; i < 4; i++) out[3 + i] = state.quaternion[i];
    return STATUS_OK;
}
