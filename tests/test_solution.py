"""End-to-end tests of runtime.wasm driven through eecs116.Solution: protocol, errors, traps, bundling."""
import io

import numpy as np
import pytest
from wasmtime import TrapCode

from eecs116 import Limits, Solution, SolutionError, SolutionTrap, abi
from eecs116.bundle import bundle
from eecs116.config import make_config

CONFIG = make_config(width=64, height=48, channels=1, num_frames=10)
FRAME = (np.arange(48 * 64, dtype=np.uint32).reshape(48, 64) % 251).astype(np.uint8)
IMU = np.column_stack([np.linspace(0.005, 0.05, 10), np.tile([0.1, 0.0, 9.81, 0.0, 0.0, 0.01], (10, 1))])

IDENTITY = """
import numpy as np
class Solution:
    def __init__(self, config):
        self.config = config
    def step(self, t, frame, imu):
        return np.array([0, 0, 0, 0, 0, 0, 1.0])
"""


def make_solution(runtime, runtime_path, directory, tmp_path, limits=None, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    wasm = tmp_path / "solution.wasm"
    bundle(directory, wasm, runtime=runtime_path)
    return Solution(wasm, runtime, limits=limits or Limits(), log_dir=tmp_path / "logs", **kwargs)


def test_identity_solution_round_trip(runtime, runtime_path, solution_dir, tmp_path):
    solution = make_solution(runtime, runtime_path, solution_dir(IDENTITY), tmp_path)
    solution.init(CONFIG)
    assert solution.init_fuel > 0
    results = [solution.step(0.05 * (k + 1), FRAME, IMU) for k in range(3)]
    assert all(result.status == abi.STATUS_OK for result in results)
    np.testing.assert_array_equal(results[0].pose, [0, 0, 0, 0, 0, 0, 1])
    assert all(result.fuel > 0 for result in results)
    # fuel is deterministic for deterministic code: a second instance repeats the exact per-step numbers
    # (consecutive steps of one instance differ because the interpreter warms up: specialization, caches)
    again = make_solution(runtime, runtime_path, solution_dir.directory, tmp_path / "again")
    again.init(CONFIG)
    assert [again.step(0.05 * (k + 1), FRAME, IMU).fuel for k in range(3)] == [result.fuel for result in results]


def test_frame_and_imu_arrive_intact(runtime, runtime_path, solution_dir, tmp_path):
    source = """
import numpy as np
class Solution:
    def __init__(self, config):
        assert config["image"]["width"] == 64 and config["camera"]["model"] == "pinhole"
    def step(self, t, frame, imu):
        assert frame.shape == (48, 64) and frame.dtype == np.uint8 and not frame.flags.writeable
        assert imu.shape == (10, 7) and abs(imu[-1, 0] - 0.05) < 1e-12 and abs(imu[3, 3] - 9.81) < 1e-12
        return [float(frame[7, 9]), float(imu[0, 1]), t, 0, 0, 0, 1]
"""
    solution = make_solution(runtime, runtime_path, solution_dir(source), tmp_path)
    solution.init(CONFIG)
    result = solution.step(0.05, FRAME, IMU)
    assert result.status == abi.STATUS_OK, solution.stderr_tail()
    assert result.pose[0] == FRAME[7, 9] and result.pose[1] == 0.1 and result.pose[2] == 0.05


def test_empty_imu_window(runtime, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "assert imu.shape == (0, 7)\n        return np.array")
    solution = make_solution(runtime, runtime_path, solution_dir(source), tmp_path)
    solution.init(CONFIG)
    assert solution.step(0.05, FRAME, np.zeros((0, 7))).status == abi.STATUS_OK


def test_none_nan_and_bad_shape(runtime, runtime_path, solution_dir, tmp_path):
    source = """
import numpy as np
class Solution:
    def __init__(self, config):
        self.k = 0
    def step(self, t, frame, imu):
        self.k += 1
        return [None, [0, 0, float("nan"), 0, 0, 0, 1], [1, 2, 3]][self.k - 1]
"""
    solution = make_solution(runtime, runtime_path, solution_dir(source), tmp_path)
    solution.init(CONFIG)
    assert solution.step(0.05, FRAME, IMU).status == abi.STATUS_NO_ESTIMATE
    assert solution.step(0.10, FRAME, IMU).status == abi.STATUS_NOT_FINITE
    assert solution.step(0.15, FRAME, IMU).status == abi.STATUS_EXCEPTION
    assert "must return 7 values" in solution.stderr_tail()


def test_exception_shows_bundle_traceback_and_episode_continues(runtime, runtime_path, solution_dir, tmp_path):
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
    solution = make_solution(runtime, runtime_path, solution_dir(source, helpers), tmp_path)
    solution.init(CONFIG)
    assert solution.step(0.05, FRAME, IMU).status == abi.STATUS_OK
    assert solution.step(0.10, FRAME, IMU).status == abi.STATUS_EXCEPTION
    assert solution.step(0.15, FRAME, IMU).status == abi.STATUS_OK  # interpreter is still consistent
    stderr = solution.stderr_tail()
    assert 'File "/bundle/helpers/math.py", line 2' in stderr
    assert "raise RuntimeError('kaboom in helper')" in stderr  # source line, not just the location
    assert "RuntimeError: kaboom in helper" in stderr


def test_bundle_data_files(runtime, runtime_path, solution_dir, tmp_path):
    weights = io.BytesIO()
    np.savez(weights, w=np.arange(6, dtype=np.float64).reshape(2, 3))
    source = """
import numpy as np, json, pkgutil
import eecs116_bundle as bundle
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
    solution = make_solution(runtime, runtime_path, solution_dir(source, extra), tmp_path)
    solution.init(CONFIG)
    result = solution.step(0.05, FRAME, IMU)
    assert result.status == abi.STATUS_OK, solution.stderr_tail()
    assert result.pose[0] == 15.0


def test_init_exception_reported(runtime, runtime_path, solution_dir, tmp_path):
    solution = make_solution(runtime, runtime_path, solution_dir("class Solution:\n    def __init__(self, config):\n        raise ValueError('bad init')\n"), tmp_path)
    with pytest.raises(SolutionError, match="exception"):
        solution.init(CONFIG)
    assert "ValueError: bad init" in solution.stderr_tail()


def test_sys_exit_is_an_exception_not_a_trap(runtime, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "import sys; sys.exit(3)\n        return np.array")
    solution = make_solution(runtime, runtime_path, solution_dir(source), tmp_path)
    solution.init(CONFIG)
    assert solution.step(0.05, FRAME, IMU).status == abi.STATUS_EXCEPTION
    assert solution.step(0.10, FRAME, IMU).status == abi.STATUS_EXCEPTION


def test_os_exit_kills_the_instance(runtime, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "import os; os._exit(3)\n        return np.array")
    solution = make_solution(runtime, runtime_path, solution_dir(source), tmp_path)
    solution.init(CONFIG)
    with pytest.raises(SolutionTrap) as info:
        solution.step(0.05, FRAME, IMU)
    assert info.value.trap_code == "exit"
    assert solution.dead
    with pytest.raises(SolutionError):
        solution.step(0.10, FRAME, IMU)


def test_out_of_fuel_traps(runtime, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "while True: pass\n        return np.array")
    solution = make_solution(runtime, runtime_path, solution_dir(source), tmp_path, limits=Limits(step_fuel=50_000_000))
    solution.init(CONFIG)
    with pytest.raises(SolutionTrap) as info:
        solution.step(0.05, FRAME, IMU)
    assert info.value.trap_code == TrapCode.OUT_OF_FUEL
    assert solution.dead


def test_wall_clock_deadline_traps(runtime, runtime_path, solution_dir, tmp_path):
    # a host-call loop burns little fuel per second; the epoch deadline is the safety net
    source = IDENTITY.replace("return np.array", "import time\n        while True: time.sleep(0.01)\n        return np.array")
    solution = make_solution(runtime, runtime_path, solution_dir(source), tmp_path, limits=Limits(step_wall_seconds=0.5))
    solution.init(CONFIG)
    with pytest.raises(SolutionTrap) as info:
        solution.step(0.05, FRAME, IMU)
    assert info.value.trap_code == TrapCode.INTERRUPT


def test_deep_recursion_is_a_python_error(runtime, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "def f(n): return f(n + 1)\n        f(0)\n        return np.array")
    solution = make_solution(runtime, runtime_path, solution_dir(source), tmp_path)
    solution.init(CONFIG)
    assert solution.step(0.05, FRAME, IMU).status == abi.STATUS_EXCEPTION
    assert "RecursionError" in solution.stderr_tail()


def test_memory_limit_is_a_python_error(runtime, runtime_path, solution_dir, tmp_path):
    source = IDENTITY.replace("return np.array", "if t < 0.08: big = np.ones(400 * 2**20, np.uint8)\n        return np.array")
    solution = make_solution(runtime, runtime_path, solution_dir(source), tmp_path, limits=Limits(memory_bytes=256 << 20))
    solution.init(CONFIG)
    assert solution.step(0.05, FRAME, IMU).status == abi.STATUS_EXCEPTION
    assert "MemoryError" in solution.stderr_tail()
    assert solution.step(0.10, FRAME, IMU).status == abi.STATUS_OK


def test_opencv_on_the_read_only_frame(runtime, runtime_path, solution_dir, tmp_path):
    source = """
import numpy as np, cv2
class Solution:
    def __init__(self, config):
        self.orb = cv2.ORB_create(100)
        self.previous = None
    def step(self, t, frame, imu):
        corners = cv2.goodFeaturesToTrack(frame, 50, 0.01, 3)
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)
        blurred = cv2.GaussianBlur(frame, (5, 5), 1.0)
        if self.previous is not None:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, frame, corners, None)
        self.previous = frame.copy()
        return [0 if corners is None else len(corners), float(blurred.mean()), 0, 0, 0, 0, 1]
"""
    solution = make_solution(runtime, runtime_path, solution_dir(source), tmp_path)
    solution.init(CONFIG)
    rng = np.random.default_rng(1)
    frame = np.kron(rng.integers(0, 255, (6, 8)), np.ones((8, 8))).astype(np.uint8)  # blocky: has corners
    for k in range(2):
        result = solution.step(0.05 * (k + 1), frame, IMU)
        assert result.status == abi.STATUS_OK, solution.stderr_tail()
    assert result.pose[0] > 0 and result.pose[1] > 0


def test_dev_mode_mounts_the_directory(runtime, runtime_path, solution_dir, tmp_path):
    directory = solution_dir(IDENTITY.replace("self.config = config", "self.config = config\n        import solution; assert solution.__file__.startswith('/solution/')"))
    solution = Solution(runtime_path, runtime, log_dir=tmp_path / "logs", dev_dir=directory)
    solution.init(CONFIG)
    assert solution.step(0.05, FRAME, IMU).status == abi.STATUS_OK, solution.stderr_tail()


def test_bare_runtime_without_code_is_refused(runtime, runtime_path, tmp_path):
    solution = Solution(runtime_path, runtime, log_dir=tmp_path / "logs")
    with pytest.raises(SolutionError, match="bundle"):
        solution.init(CONFIG)


def test_abi_constants_match_the_shim(runtime_path):
    shim = (runtime_path.parent.parent.parent.parent / "runtime" / "py" / "_eecs116.py").read_text()
    for name in ("STATUS_OK", "STATUS_NO_ESTIMATE", "STATUS_EXCEPTION", "STATUS_PROTOCOL", "STATUS_RUNTIME", "STATUS_NO_SOLUTION", "STATUS_NOT_FINITE"):
        assert f"{name} = {getattr(abi, name)}" in shim
