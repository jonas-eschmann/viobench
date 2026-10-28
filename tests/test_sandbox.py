"""End-to-end tests of runtime.wasm driven through relpose.Solution: protocol, errors, traps, bundling."""
import io

import numpy as np
import pytest
from wasmtime import TrapCode

from relpose import Limits, Sandbox, SandboxError, SandboxTrap, abi
from relpose.solution.runtime.python.bundler.bundle import bundle
from relpose.tools.abi.config import default_guest_config

CONFIG = default_guest_config()  # 320x240, the default declaration
GRAY = (np.arange(240 * 320, dtype=np.uint32).reshape(240, 320) % 251).astype(np.uint8)
FRAME = np.stack([GRAY, GRAY // 2, GRAY // 3], axis=2)  # RGB with distinct channels
IMU = np.column_stack([np.linspace(0.005, 0.05, 10), np.tile([0.1, 0.0, 9.81, 0.0, 0.0, 0.01], (10, 1))])

IDENTITY = """
import numpy as np
class Solution:
    def __init__(self, config):
        self.config = config
    def step(self, t, frame, imu):
        return np.array([0, 0, 0, 0, 0, 0, 1.0])
"""


def make_sandbox(engine, runtime_path, directory, tmp_path, limits=None, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    wasm = tmp_path / "solution.wasm"
    bundle(directory, wasm, runtime=runtime_path)
    return Sandbox(wasm, engine, limits=limits or Limits(), log_dir=tmp_path / "logs", **kwargs)


def test_identity_solution_round_trip(engine, runtime_path, solution_dir, tmp_path):
    sandbox = make_sandbox(engine, runtime_path, solution_dir(IDENTITY), tmp_path)
    sandbox.init(CONFIG)
    assert sandbox.init_fuel > 0
    results = [sandbox.step(0.05 * (k + 1), FRAME, IMU) for k in range(3)]
    assert all(result.status == abi.STATUS_OK for result in results)
    np.testing.assert_array_equal(results[0].pose, [0, 0, 0, 0, 0, 0, 1])
    assert all(result.fuel > 0 for result in results)
    # fuel is deterministic for deterministic code: a second instance repeats the exact per-step numbers
    # (consecutive steps of one instance differ because the interpreter warms up: specialization, caches)
    again = make_sandbox(engine, runtime_path, solution_dir.directory, tmp_path / "again")
    again.init(CONFIG)
    assert [again.step(0.05 * (k + 1), FRAME, IMU).fuel for k in range(3)] == [result.fuel for result in results]


def test_frame_and_imu_arrive_intact(engine, runtime_path, solution_dir, tmp_path):
    source = """
import numpy as np
class Solution:
    def __init__(self, config):
        assert set(config) == {"abi_version", "width", "height", "fx", "fy", "cx", "cy", "T_cam_imu"} and config["width"] == 320
    def step(self, t, frame, imu):
        assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8 and not frame.flags.writeable
        assert imu.shape == (10, 7) and abs(imu[-1, 0] - 0.05) < 1e-12 and abs(imu[3, 3] - 9.81) < 1e-12
        return [float(frame[7, 9, 1]), float(imu[0, 1]), t, 0, 0, 0, 1]
"""
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source), tmp_path)
    sandbox.init(CONFIG)
    result = sandbox.step(0.05, FRAME, IMU)
    assert result.status == abi.STATUS_OK, sandbox.stderr_tail()
    assert result.pose[0] == FRAME[7, 9, 1] and result.pose[1] == 0.1 and result.pose[2] == 0.05


def test_empty_imu_window(engine, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "assert imu.shape == (0, 7)\n        return np.array")
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source), tmp_path)
    sandbox.init(CONFIG)
    assert sandbox.step(0.05, FRAME, np.zeros((0, 7))).status == abi.STATUS_OK


def test_none_nan_and_bad_shape(engine, runtime_path, solution_dir, tmp_path):
    source = """
import numpy as np
class Solution:
    def __init__(self, config):
        self.k = 0
    def step(self, t, frame, imu):
        self.k += 1
        return [None, [0, 0, float("nan"), 0, 0, 0, 1], [1, 2, 3]][self.k - 1]
"""
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source), tmp_path)
    sandbox.init(CONFIG)
    assert sandbox.step(0.05, FRAME, IMU).status == abi.STATUS_NO_ESTIMATE
    assert sandbox.step(0.10, FRAME, IMU).status == abi.STATUS_NOT_FINITE
    assert sandbox.step(0.15, FRAME, IMU).status == abi.STATUS_EXCEPTION
    assert "must return 7 values" in sandbox.stderr_tail()


def test_exception_shows_bundle_traceback_and_episode_continues(engine, runtime_path, solution_dir, tmp_path):
    source = """
import numpy as np
from helpers.math import blow_up
class Solution:
    def __init__(self, config):
        self.k = 0
    def step(self, t, frame, imu):
        self.k += 1
        if self.k == 2:
            blow_up()
        return np.array([0, 0, 0, 0, 0, 0, 1.0])
"""
    helpers = {"helpers/__init__.py": "", "helpers/math.py": "def blow_up():\n    raise RuntimeError('kaboom in helper')\n"}
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source, helpers), tmp_path)
    sandbox.init(CONFIG)
    assert sandbox.step(0.05, FRAME, IMU).status == abi.STATUS_OK
    assert sandbox.step(0.10, FRAME, IMU).status == abi.STATUS_EXCEPTION
    assert sandbox.step(0.15, FRAME, IMU).status == abi.STATUS_OK  # interpreter is still consistent
    stderr = sandbox.stderr_tail()
    assert 'File "/bundle/helpers/math.py", line 2' in stderr
    assert "raise RuntimeError('kaboom in helper')" in stderr  # source line, not just the location
    assert "RuntimeError: kaboom in helper" in stderr


def test_bundle_data_files(engine, runtime_path, solution_dir, tmp_path):
    weights = io.BytesIO()
    np.savez(weights, w=np.arange(6, dtype=np.float64).reshape(2, 3))
    source = """
import numpy as np, json, pkgutil
import solution_bundle as bundle
import mypkg
class Solution:
    def __init__(self, config):
        self.w = np.load(bundle.open("model/weights.npz"))["w"]
        assert json.loads(bundle.read_text("model/meta.json"))["layers"] == 2
        assert bundle.exists("model/meta.json") and not bundle.exists("nope")
        assert pkgutil.get_data("mypkg", "data.txt") == b"hi"
        assert mypkg.__file__ == "/bundle/mypkg/__init__.py"
    def step(self, t, frame, imu):
        return [self.w.sum(), 0, 0, 0, 0, 0, 1]
"""
    extra = {"model/weights.npz": weights.getvalue(), "model/meta.json": '{"layers": 2}', "mypkg/__init__.py": "", "mypkg/data.txt": "hi"}
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source, extra), tmp_path)
    sandbox.init(CONFIG)
    result = sandbox.step(0.05, FRAME, IMU)
    assert result.status == abi.STATUS_OK, sandbox.stderr_tail()
    assert result.pose[0] == 15.0


def test_init_exception_reported(engine, runtime_path, solution_dir, tmp_path):
    sandbox = make_sandbox(engine, runtime_path, solution_dir("class Solution:\n    def __init__(self, config):\n        raise ValueError('bad init')\n"), tmp_path)
    with pytest.raises(SandboxError, match="exception"):
        sandbox.init(CONFIG)
    assert "ValueError: bad init" in sandbox.stderr_tail()


def test_sys_exit_is_an_exception_not_a_trap(engine, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "import sys; sys.exit(3)\n        return np.array")
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source), tmp_path)
    sandbox.init(CONFIG)
    assert sandbox.step(0.05, FRAME, IMU).status == abi.STATUS_EXCEPTION
    assert sandbox.step(0.10, FRAME, IMU).status == abi.STATUS_EXCEPTION


def test_os_exit_kills_the_instance(engine, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "import os; os._exit(3)\n        return np.array")
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source), tmp_path)
    sandbox.init(CONFIG)
    with pytest.raises(SandboxTrap) as info:
        sandbox.step(0.05, FRAME, IMU)
    assert info.value.trap_code == "exit"
    assert sandbox.dead
    with pytest.raises(SandboxError):
        sandbox.step(0.10, FRAME, IMU)


def test_out_of_fuel_traps(engine, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "while True: pass\n        return np.array")
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source), tmp_path, limits=Limits(step_fuel=50_000_000))
    sandbox.init(CONFIG)
    with pytest.raises(SandboxTrap) as info:
        sandbox.step(0.05, FRAME, IMU)
    assert info.value.trap_code == TrapCode.OUT_OF_FUEL
    assert sandbox.dead


def test_wall_clock_deadline_traps(engine, runtime_path, solution_dir, tmp_path):
    # a host-call loop burns little fuel per second; the epoch deadline is the safety net
    source = IDENTITY.replace("return np.array", "import time\n        while True: time.sleep(0.01)\n        return np.array")
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source), tmp_path, limits=Limits(step_wall_seconds=0.5))
    sandbox.init(CONFIG)
    with pytest.raises(SandboxTrap) as info:
        sandbox.step(0.05, FRAME, IMU)
    assert info.value.trap_code == TrapCode.INTERRUPT


def test_deep_recursion_is_a_python_error(engine, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "def f(n): return f(n + 1)\n        f(0)\n        return np.array")
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source), tmp_path)
    sandbox.init(CONFIG)
    assert sandbox.step(0.05, FRAME, IMU).status == abi.STATUS_EXCEPTION
    assert "RecursionError" in sandbox.stderr_tail()


def test_memory_limit_is_a_python_error(engine, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "if t < 0.08: big = np.ones(400 * 2**20, np.uint8)\n        return np.array")
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source), tmp_path, limits=Limits(memory_bytes=256 << 20))
    sandbox.init(CONFIG)
    assert sandbox.step(0.05, FRAME, IMU).status == abi.STATUS_EXCEPTION
    assert "MemoryError" in sandbox.stderr_tail()
    assert sandbox.step(0.10, FRAME, IMU).status == abi.STATUS_OK


def test_opencv_on_the_read_only_frame(engine, runtime_path, solution_dir, tmp_path):
    source = """
import numpy as np, cv2
class Solution:
    def __init__(self, config):
        self.orb = cv2.ORB_create(100)
        self.previous = None
    def step(self, t, frame, imu):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(frame, 50, 0.01, 3)
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)
        blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
        if self.previous is not None:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, frame, corners, None)
        self.previous = frame.copy()
        return [0 if corners is None else len(corners), float(blurred.mean()), 0, 0, 0, 0, 1]
"""
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source), tmp_path)
    sandbox.init(CONFIG)
    rng = np.random.default_rng(1)
    blocks = np.kron(rng.integers(0, 255, (30, 40)), np.ones((8, 8))).astype(np.uint8)  # blocky: has corners
    frame = np.repeat(blocks[:, :, None], 3, axis=2)
    for k in range(2):
        result = sandbox.step(0.05 * (k + 1), frame, IMU)
        assert result.status == abi.STATUS_OK, sandbox.stderr_tail()
    assert result.pose[0] > 0 and result.pose[1] > 0


def test_dev_mode_mounts_the_directory(engine, runtime_path, solution_dir, tmp_path):
    directory = solution_dir(IDENTITY.replace("self.config = config", "self.config = config\n        import solution; assert solution.__file__.startswith('/solution/')"))
    sandbox = Sandbox(runtime_path, engine, log_dir=tmp_path / "logs", dev_dir=directory)
    sandbox.init(CONFIG)
    assert sandbox.step(0.05, FRAME, IMU).status == abi.STATUS_OK, sandbox.stderr_tail()


def test_bare_runtime_without_code_is_refused(engine, runtime_path, tmp_path):
    sandbox = Sandbox(runtime_path, engine, log_dir=tmp_path / "logs")
    assert sandbox.descriptor is not None and sandbox.descriptor.length == 0
    with pytest.raises(SandboxError, match="bundle"):
        sandbox.init(CONFIG)


def test_embedded_bundle_survives_heap_growth(engine, runtime_path, solution_dir, tmp_path):
    """The zip lives above the runtime's initial memory; the heap must grow past it, never into it."""
    blob = np.random.default_rng(3).integers(0, 256, 3 << 20, dtype=np.uint8).tobytes()
    source = """
import hashlib
import numpy as np
import solution_bundle as bundle
EXPECTED = %r
class Solution:
    def __init__(self, config):
        self.keep = []
    def step(self, t, frame, imu):
        for _ in range(8):
            self.keep.append(np.ones(16 << 20, np.uint8))  # 128 MiB per step, retained
        digest = hashlib.sha256(bundle.read_bytes("blob.bin")).hexdigest()
        return [float(digest == EXPECTED), len(self.keep), 0, 0, 0, 0, 1]
""" % __import__("hashlib").sha256(blob).hexdigest()
    sandbox = make_sandbox(engine, runtime_path, solution_dir(source, {"blob.bin": blob}), tmp_path, limits=Limits(memory_bytes=1 << 30))
    sandbox.init(CONFIG)
    for k in range(4):  # ~512 MiB of heap growth in total, far beyond the zip
        result = sandbox.step(0.05 * (k + 1), FRAME, IMU)
        assert result.status == abi.STATUS_OK, sandbox.stderr_tail()
        assert result.pose[0] == 1.0, "bundle bytes were corrupted by heap growth"


def test_unbundle_roundtrip(runtime_path, solution_dir, tmp_path):
    from relpose.solution.runtime.python.bundler.bundle import unbundle
    from relpose.solution.runtime.python.bundler.embed import extract_bundle, runtime_descriptor, strip_bundle

    directory = solution_dir(IDENTITY, {"data/w.npy": b"\x93NUMPY" + bytes(100), "pkg/__init__.py": "X = 1\n"})
    wasm = tmp_path / "s.wasm"
    bundle(directory, wasm, runtime=runtime_path)
    bundled = wasm.read_bytes()
    descriptor = runtime_descriptor(bundled)
    assert descriptor.length == len(extract_bundle(bundled)) and descriptor.address % 65536 == 0
    names = unbundle(wasm, tmp_path / "out")
    assert set(names) == {"solution.py", "data/w.npy", "pkg/__init__.py", "solution.json"}
    assert (tmp_path / "out" / "data" / "w.npy").read_bytes() == b"\x93NUMPY" + bytes(100)
    # stripping gives back the clean runtime byte for byte (the manifest section aside), and bundling twice is idempotent
    assert strip_custom_manifest(strip_bundle(bundled)) == strip_custom_manifest(runtime_path.read_bytes())
    bundle(directory, tmp_path / "again.wasm", runtime=wasm)  # an already-bundled file as the runtime is re-stripped first
    assert (tmp_path / "again.wasm").read_bytes() == bundled


def strip_custom_manifest(wasm):
    from relpose.tools.abi.wasm_binary import strip_custom

    return strip_custom(wasm, abi.MANIFEST_SECTION)


def test_abi_constants_match_the_guest_shim_and_the_native_header():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    shim = (repo / "relpose" / "solution" / "runtime" / "python" / "guest" / "relpose_guest.py").read_text()
    header = (repo / "relpose" / "solution" / "runtime" / "native" / "relpose_abi.h").read_text()
    for name in ("STATUS_OK", "STATUS_NO_ESTIMATE", "STATUS_EXCEPTION", "STATUS_PROTOCOL", "STATUS_RUNTIME", "STATUS_NO_SOLUTION", "STATUS_NOT_FINITE"):
        assert f"{name} = {getattr(abi, name)}" in shim
        assert f"#define RELPOSE_{name} {getattr(abi, name)}" in header
    assert f"#define RELPOSE_ABI_VERSION {abi.ABI_VERSION}" in header


def test_resolution_opt_in_via_manifest(engine, runtime_path, solution_dir, tmp_path):
    source = """
import numpy as np
class Solution:
    def __init__(self, config):
        assert (config["width"], config["height"]) == (640, 480)
    def step(self, t, frame, imu):
        assert frame.shape == (480, 640, 3)
        return [float(frame[1, 2, 0]), float(frame[1, 2, 1]), float(frame[1, 2, 2]), 0, 0, 0, 1]
"""
    directory = solution_dir(source, {"solution.json": '{"resolution": [640, 480]}'})
    wasm = tmp_path / "vga.wasm"
    summary = bundle(directory, wasm, runtime=runtime_path)
    assert summary["resolution"] == [640, 480]
    sandbox = Sandbox(wasm, engine, log_dir=tmp_path / "logs")
    assert sandbox.resolution == (640, 480) and sandbox.manifest["resolution"] == [640, 480]
    with pytest.raises(SandboxError, match="declares 640x480"):
        sandbox.init(CONFIG)  # a 320x240 config does not match the declaration
    sandbox = Sandbox(wasm, engine, log_dir=tmp_path / "logs2")
    sandbox.init(default_guest_config(640, 480))
    vga = np.kron(FRAME, np.ones((2, 2, 1), dtype=np.uint8))
    result = sandbox.step(0.05, vga, IMU)
    assert result.status == abi.STATUS_OK, sandbox.stderr_tail()
    np.testing.assert_array_equal(result.pose[:3], vga[1, 2])
    # the manifest command rewrites the section; bundling again from a --resolution flag overrides solution.json
    from relpose.tools.abi.manifest import read_manifest, write_manifest

    assert write_manifest(wasm, {"resolution": "320x240"})["resolution"] == [320, 240] and read_manifest(wasm.read_bytes())["resolution"] == [320, 240]
    assert bundle(directory, wasm, runtime=runtime_path, resolution="320x240")["resolution"] == [320, 240]
    with pytest.raises(Exception, match="resolution"):
        write_manifest(wasm, {"resolution": [64, 48]})
    with pytest.raises(Exception, match="unknown manifest keys"):  # there is no colour choice: frames are always RGB
        write_manifest(wasm, {"channels": 3})
    assert "entry" not in write_manifest(wasm, {"entry": "solution:Solution"})  # what older bundlers wrote: accepted and dropped
    with pytest.raises(Exception, match="from mypkg.estimator import Estimator as Solution"):  # the entry point is fixed
        write_manifest(wasm, {"entry": "mypkg.estimator:Estimator"})
