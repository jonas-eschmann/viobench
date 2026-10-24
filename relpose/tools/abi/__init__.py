"""Constants of the solution ABI (docs/ABI.md), shared by the host harness, relpose/solution/runtime/python/reactor.c and relpose/solution/runtime/python/guest/relpose_guest.py."""

ABI_VERSION = 1

BUNDLE_MAGIC = b"RELPOSEBUNDLEv1\0"  # 16-byte head of the runtime's bundle descriptor (see relpose/solution/runtime/python/reactor.c)
BUNDLE_DESCRIPTOR_BYTES = 32  # magic[16] | address u32 | length u32 | version u32 | reserved u32
MANIFEST_SECTION = "relpose.manifest"  # wasm custom section with the manifest JSON (read by the harness before init)
BUNDLE_MANIFEST = "solution.json"  # optional file in the directory: {"entry": "module:Class", "channels": 1 | 3}
DEFAULT_CHANNELS = 1  # grayscale unless the manifest opts into RGB
SUPPORTED_CHANNELS = (1, 3)
DEFAULT_RESOLUTION = (320, 240)  # (width, height) unless the manifest opts into VGA
SUPPORTED_RESOLUTIONS = ((320, 240), (640, 480))
DEFAULT_ENTRY = "solution:Solution"
SOLUTION_DIR_VARIABLE = "RELPOSE_SOLUTION_DIR"  # dev mode: guest path of the preopened solution directory

IMU_ROW_VALUES = 7  # [t_s, ax, ay, az, gx, gy, gz]
POSE_VALUES = 7  # [x, y, z, qx, qy, qz, qw]
F64_BYTES = 8

REQUIRED_EXPORTS = ("memory", "alloc", "init", "step")
OPTIONAL_EXPORTS = ("_initialize", "abi_version")

STATUS_OK = 0
STATUS_NO_ESTIMATE = 1
STATUS_EXCEPTION = -1
STATUS_PROTOCOL = -2
STATUS_RUNTIME = -3
STATUS_NO_SOLUTION = -4
STATUS_NOT_FINITE = -5

STATUS_NAMES = {
    STATUS_OK: "ok",
    STATUS_NO_ESTIMATE: "no_estimate",
    STATUS_EXCEPTION: "exception",
    STATUS_PROTOCOL: "protocol",
    STATUS_RUNTIME: "runtime",
    STATUS_NO_SOLUTION: "no_solution",
    STATUS_NOT_FINITE: "not_finite",
}


def status_name(status):
    return STATUS_NAMES.get(status, f"unknown({status})")


class AbiError(Exception):
    """A module or manifest violates the contract (malformed manifest, unsupported option, inconsistent descriptor)."""
