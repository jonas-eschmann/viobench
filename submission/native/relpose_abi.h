/* relpose solution ABI, version 1 (docs/ABI.md) — the whole contract between a solution module and the harness.
 *
 * A solution is a wasm32-wasip1 *reactor* that exports the functions declared below (plus `memory`). The harness
 * calls, per episode: _initialize (if present) -> init(config_json) -> alloc x3 -> step(...) x N. Anything may run
 * inside; this header is a convenience for C/C++ solutions, the Python runtime (submission/python/reactor.c) uses it
 * too, so status codes exist in exactly one place.
 */
#ifndef RELPOSE_ABI_H
#define RELPOSE_ABI_H

#include <stdint.h>

#define RELPOSE_ABI_VERSION 1

/* Mark a function as a wasm export under its own name. */
#define RELPOSE_EXPORT(name) __attribute__((export_name(#name)))

/* Buffer layouts: everything little-endian; pointers are offsets into the module's linear memory. */
#define RELPOSE_IMU_ROW_VALUES 7 /* [t_s, ax, ay, az, gx, gy, gz] as double */
#define RELPOSE_POSE_VALUES 7    /* [x, y, z, qx, qy, qz, qw] as double, camera k in camera 0 */

/* Status codes returned by init() and step(). */
#define RELPOSE_STATUS_OK 0           /* step: out[] holds a pose */
#define RELPOSE_STATUS_NO_ESTIMATE 1  /* step: no pose for this frame (yet) */
#define RELPOSE_STATUS_EXCEPTION -1   /* an error inside the solution; message on stderr; the episode continues */
#define RELPOSE_STATUS_PROTOCOL -2    /* step before init, init twice, bad arguments */
#define RELPOSE_STATUS_RUNTIME -3     /* the module itself could not start */
#define RELPOSE_STATUS_NO_SOLUTION -4 /* a Python runtime without embedded code */
#define RELPOSE_STATUS_NOT_FINITE -5  /* pose contains NaN or inf */

#ifdef __cplusplus
extern "C" {
#endif

/* Exports every solution must provide (`memory` is exported by the toolchain automatically). */
RELPOSE_EXPORT(abi_version) int32_t abi_version(void);                   /* returns RELPOSE_ABI_VERSION */
RELPOSE_EXPORT(alloc) void *alloc(uint32_t size);                        /* malloc; never freed by the host */
RELPOSE_EXPORT(init) int32_t init(const char *config_json, uint32_t length); /* once; the bytes are valid during the call */
RELPOSE_EXPORT(step) int32_t step(double t, const uint8_t *frame, const double *imu, int32_t imu_count, double *out);
/* frame: height*width*channels uint8 (row-major) per the config; imu: imu_count rows of RELPOSE_IMU_ROW_VALUES doubles
 * with t_prev_frame < t <= t_frame; out: RELPOSE_POSE_VALUES doubles, written on RELPOSE_STATUS_OK. The frame and IMU
 * buffers are reused for the next step: copy what you want to keep. */

#ifdef __cplusplus
}
#endif

#endif /* RELPOSE_ABI_H */
