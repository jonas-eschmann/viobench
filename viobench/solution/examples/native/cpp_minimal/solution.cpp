// Minimal native solution: a wasm32-wasip1 reactor implementing the viobench ABI (runtime/native/viobench_abi.h) without
// Python. It ignores all inputs and only writes the required identity pose and zero twist.
// Build:  viobench bundle <this directory> -o solution.wasm
#include "viobench_abi.h"

#include <cstdlib>

int32_t abi_version() { return VIOBENCH_ABI_VERSION; }

void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }

// Optional (320 is the default without it); return 640 for 640x480 frames. Called once, before init().
int32_t frame_width() { return 320; }

// No setup or config parsing is needed for a constant output.
int32_t init(const char*, uint32_t) {
    return VIOBENCH_STATUS_OK;
}

// frame: height*width*3 uint8 RGB; imu: imu_count rows [t, ax, ay, az, gx, gy, gz];
// out: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz] (all 13 values, every step)
int32_t step(double, const uint8_t*, const double*, int32_t, double* out) {
    // The host fills out[] with NaNs before each call: all 13 values must be written every time.
    for (int i = 0; i < VIOBENCH_STATE_VALUES; i++) out[i] = 0.0;
    out[6] = 1.0;  // qw: identity quaternion
    return VIOBENCH_STATUS_OK;
}
