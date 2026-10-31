/* Host-side driver for debugging a native solution outside the wasm sandbox.
 *
 * Linked with your solution's sources by viobench_reactor.cmake whenever the configure is not WASI
 * (`cmake -S . -B build-host && cmake --build build-host` -> build-host/debug). It reads a feed directory written by
 * `viobench feed <episode> -o <dir>` (raw arrays, no PNG/JSON parsing needed), calls init() and step() exactly like the
 * sandbox would, writes estimate.tum and twist.txt next to it and prints the position RMSE. Score it fully with
 * `viobench score <episode> estimate.tum`.
 * Run under gdb/lldb:  gdb --args build-host/debug <feed dir>
 */
#include "viobench_abi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TIME_TOLERANCE 1e-6 /* seconds: the same slack the host harness uses when selecting IMU rows */

/* The harness's test for a usable state (pose and twist): all finite, with a non-zero quaternion. */
static int usable(const double *state) {
    for (int i = 0; i < VIOBENCH_STATE_VALUES; i++) {
        if (!isfinite(state[i])) return 0;
    }
    return state[3] != 0 || state[4] != 0 || state[5] != 0 || state[6] != 0;
}

static void *read_file(const char *directory, const char *name, size_t *size) {
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", directory, name);
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(2);
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    void *data = malloc((size_t)length + 1);
    if (fread(data, 1, (size_t)length, file) != (size_t)length) {
        fprintf(stderr, "short read on %s\n", path);
        exit(2);
    }
    ((char *)data)[length] = 0;
    fclose(file);
    *size = (size_t)length;
    return data;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <feed directory from `viobench feed`> [estimate.tum]\n", argv[0]);
        return 2;
    }
    const char *directory = argv[1];
    const char *output = argc > 2 ? argv[2] : "estimate.tum";

    size_t size;
    char *shape = read_file(directory, "shape.txt", &size);
    long frames, height, width, channels, imu_rows;
    if (sscanf(shape, "%ld %ld %ld %ld %ld", &frames, &height, &width, &channels, &imu_rows) != 5) {
        fprintf(stderr, "bad shape.txt\n");
        return 2;
    }
    char *config = read_file(directory, "config.json", &size);
    size_t config_length = size;
    size_t frame_bytes = (size_t)height * (size_t)width * (size_t)channels;
    uint8_t *pixels = read_file(directory, "frames.u8", &size);
    if (size != frame_bytes * (size_t)frames) {
        fprintf(stderr, "frames.u8 has %zu bytes, expected %zu\n", size, frame_bytes * (size_t)frames);
        return 2;
    }
    double *frame_times = read_file(directory, "frame_times.f64", &size);
    double *imu = read_file(directory, "imu.f64", &size);
    double *ground_truth = read_file(directory, "gt.f64", &size);

    printf("feed: %ld frames %ldx%ldx%ld, %ld IMU rows, abi %d\n", frames, width, height, channels, imu_rows, abi_version());
    int32_t status = init(config, (uint32_t)config_length);
    if (status != VIOBENCH_STATUS_OK) {
        fprintf(stderr, "init returned %d\n", status);
        return 1;
    }

    char twist_output[4096]; /* twist.txt next to the pose file */
    const char *slash = strrchr(output, '/');
    snprintf(twist_output, sizeof twist_output, "%.*stwist.txt", slash ? (int)(slash - output + 1) : 0, output);
    FILE *tum = fopen(output, "w");
    FILE *twist = fopen(twist_output, "w");
    if (tum == NULL || twist == NULL) {
        fprintf(stderr, "cannot write %s\n", tum == NULL ? output : twist_output);
        return 2;
    }
    fprintf(tum, "# t x y z qx qy qz qw (camera k in camera 0 frame)\n");
    fprintf(twist, "# t vx vy vz wx wy wz (velocity and angular velocity of camera k in camera k axes)\n");
    double state[VIOBENCH_STATE_VALUES];
    double previous = -INFINITY;
    long window_start = 0;
    long k = 0;
    double squared_error = 0.0, total_seconds = 0.0;
    for (; k < frames; k++) {
        double t = frame_times[k];
        /* IMU rows with t_prev < t_row <= t (the feed starts at frame 0's window, so frame 0 gets everything up to its own time) */
        while (window_start < imu_rows && imu[window_start * VIOBENCH_IMU_ROW_VALUES] <= previous + TIME_TOLERANCE) window_start++;
        long window_end = window_start;
        while (window_end < imu_rows && imu[window_end * VIOBENCH_IMU_ROW_VALUES] <= t + TIME_TOLERANCE) window_end++;
        for (int i = 0; i < VIOBENCH_STATE_VALUES; i++) state[i] = NAN; /* like the host: a value not written fails the step */
        double started = now_seconds();
        status = step(t, pixels + (size_t)k * frame_bytes, imu + window_start * VIOBENCH_IMU_ROW_VALUES, (int32_t)(window_end - window_start), state);
        double seconds = now_seconds() - started;
        if (status != VIOBENCH_STATUS_OK || !usable(state)) {
            printf("frame %ld: %s; the harness ends the episode here and marks it failed\n", k,
                   status != VIOBENCH_STATUS_OK ? "step returned an error"
                                               : "state not usable (not all 13 values written, NaN/inf, or all-zero quaternion)");
            break;
        }
        total_seconds += seconds;
        fprintf(tum, "%.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", t, state[0], state[1], state[2], state[3], state[4], state[5], state[6]);
        fprintf(twist, "%.9g %.9g %.9g %.9g %.9g %.9g %.9g\n", t, state[7], state[8], state[9], state[10], state[11], state[12]);
        const double *truth = ground_truth + k * VIOBENCH_POSE_VALUES;
        double dx = state[0] - truth[0], dy = state[1] - truth[1], dz = state[2] - truth[2];
        squared_error += dx * dx + dy * dy + dz * dz;
        previous = t;
        window_start = window_end;
    }
    fclose(tum);
    fclose(twist);
    printf("%ld/%ld frames ok, position RMSE over them %.3f m, %.2f ms/frame (host, no fuel); poses in %s, twists in %s\n", k, frames,
           sqrt(squared_error / (double)k), 1e3 * total_seconds / (double)k, output, twist_output);
    return k < frames;
}
