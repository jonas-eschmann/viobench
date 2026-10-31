/* viobench solution ABI, version 1 (docs/ABI.md) — the whole contract between a solution module and the harness.
 *
 * A solution is a wasm32-wasip1 *reactor* that exports the functions declared below (plus `memory`). The harness
 * calls, per episode: _initialize (if present) -> frame_width (if present) -> init(config_json) -> alloc x3 -> step(...) x N.
 * Anything may run inside; this header is a convenience for C/C++ solutions, the Python runtime
 * (viobench/solution/runtime/python/reactor.c) uses it too, so status codes exist in exactly one place.
 */
#ifndef VIOBENCH_ABI_H
#define VIOBENCH_ABI_H

#include <stdint.h>

#define VIOBENCH_ABI_VERSION 1

/* Mark a function as a wasm export under its own name (a no-op in a host debug build). */
#if defined(__wasm__)
#define VIOBENCH_EXPORT(name) __attribute__((export_name(#name)))
#else
#define VIOBENCH_EXPORT(name)
#endif

/* Buffer layouts: everything little-endian; pointers are offsets into the module's linear memory. */
#define VIOBENCH_IMU_ROW_VALUES 7 /* [t_s, ax, ay, az, gx, gy, gz] as double */
#define VIOBENCH_POSE_VALUES 7    /* [x, y, z, qx, qy, qz, qw] as double, camera k in camera 0 */
#define VIOBENCH_TWIST_VALUES 6   /* [vx, vy, vz, wx, wy, wz] as double, velocity and angular velocity of camera k in its own axes */
#define VIOBENCH_STATE_VALUES 13  /* out[]: the pose, then the twist */

/* Status codes returned by init() and step(): anything but OK is an error (explain it on stderr). */
#define VIOBENCH_STATUS_OK 0     /* init: ready; step: out[] holds the pose and the twist */
#define VIOBENCH_STATUS_ERROR -1 /* the episode ends and is marked failed */

#ifdef __cplusplus
extern "C" {
#endif

/* Exports every solution must provide (`memory` is exported by the toolchain automatically). */
VIOBENCH_EXPORT(abi_version) int32_t abi_version(void);                   /* returns VIOBENCH_ABI_VERSION */
VIOBENCH_EXPORT(alloc) void *alloc(uint32_t size);                        /* malloc; never freed by the host */
VIOBENCH_EXPORT(frame_width) int32_t frame_width(void);                  /* optional: 320 (the default without it) or 640 (640x480) */
VIOBENCH_EXPORT(init) int32_t init(const char *config_json, uint32_t length); /* once; the bytes are valid during the call */
VIOBENCH_EXPORT(step) int32_t step(double t, const uint8_t *frame, const double *imu, int32_t imu_count, double *out);
/* frame: height*width*3 uint8 RGB (row-major), size per the config; imu: imu_count (always 10) rows of VIOBENCH_IMU_ROW_VALUES
 * doubles with t_prev_frame < t <= t_frame; out: VIOBENCH_STATE_VALUES doubles, all written on VIOBENCH_STATUS_OK (the host fills
 * them with NaN before the call, so a value left out fails the step). The frame and IMU buffers are reused for the next
 * step: copy what you want to keep. */

#ifdef __cplusplus
}
#endif

#endif /* VIOBENCH_ABI_H */
