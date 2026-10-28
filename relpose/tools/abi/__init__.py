"""Constants of the solution ABI (docs/ABI.md), shared by the host harness, relpose/solution/runtime/python/reactor.c and relpose/solution/runtime/python/guest/relpose_guest.py."""

ABI_VERSION = 1

BUNDLE_MAGIC = b"RELPOSEBUNDLEv1\0"  # 16-byte head of the runtime's bundle descriptor (see relpose/solution/runtime/python/reactor.c)
MANIFEST_SECTION = "relpose.manifest"  # wasm custom section with the manifest JSON (read by the harness before init)
BUNDLE_MANIFEST = "solution.json"  # optional file in the directory: {"resolution": [w, h]}
CHANNELS = 3  # frames are always RGB (H, W, 3); a grayscale-only set arrives with its gray value in all three channels
DEFAULT_RESOLUTION = (320, 240)  # (width, height) unless the manifest opts into VGA
SUPPORTED_RESOLUTIONS = ((320, 240), (640, 480))  # also the only resolutions a set may be stored at; either serves both
SOLUTION_MODULE = "solution"  # the entry point is always class Solution in solution.py (or solution/__init__.py)
SOLUTION_CLASS = "Solution"

IMU_ROW_VALUES = 7  # [t_s, ax, ay, az, gx, gy, gz]
FRAME_RATE_HZ = 20.0  # every set is delivered at these rates (relpose.tools.datasets.regularize)
IMU_RATE_HZ = 200.0
IMU_ROWS_PER_FRAME = 10  # IMU_RATE_HZ / FRAME_RATE_HZ: imu_count of every step
POSE_VALUES = 7  # [x, y, z, qx, qy, qz, qw]
F64_BYTES = 8

REQUIRED_EXPORTS = ("memory", "abi_version", "alloc", "init", "step")  # plus `_initialize`, called if the toolchain emitted it

STATUS_OK = 0  # init: ready; step: `out` holds the pose
STATUS_ERROR = -1  # what our guests return on an error; any status but STATUS_OK is one (explained on stderr)


class AbiError(Exception):
    """A module or manifest violates the contract (malformed manifest, unsupported option, inconsistent descriptor)."""
