"""Host-side wrapper around one solution .wasm: instantiate, init, step, with fuel and wall-time metering."""
import ctypes
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wasmtime

from . import abi
from .bundle import BundleError, read_manifest, validate_manifest
from .engine import Runtime
from .wasm_sections import split_bundle, strip_custom


@dataclass
class Limits:
    init_fuel: int = 200_000_000_000  # covers _initialize, load_bundle and init (interpreter start + imports)
    step_fuel: int = 20_000_000_000  # per step; exceeding it traps and ends the episode
    memory_bytes: int = 2 << 30
    init_wall_seconds: float = 300.0  # wall-clock safety nets only (fuel is the metric)
    step_wall_seconds: float = 60.0
    imu_capacity: int = 1024  # rows; grows on demand


@dataclass
class StepResult:
    status: int
    pose: np.ndarray | None  # [x, y, z, qx, qy, qz, qw] when status == STATUS_OK
    fuel: int
    wall_seconds: float

    @property
    def ok(self):
        return self.status == abi.STATUS_OK


class SolutionError(Exception):
    """The solution broke the protocol or failed to initialize; the message carries the stderr tail."""


class SolutionTrap(SolutionError):
    """The guest trapped (out of fuel, wall-clock deadline, abort, memory fault, exit()); the instance is dead."""

    def __init__(self, trap_code, message, stderr_tail):
        super().__init__(f"trap {trap_code}: {message}\n--- stderr tail ---\n{stderr_tail}")
        self.trap_code = trap_code
        self.trap_message = message
        self.stderr_tail = stderr_tail


class Solution:
    """One instance of a solution module for one episode.

    solution = Solution("student.wasm", runtime)
    solution.init(config)
    result = solution.step(t, frame, imu)   # StepResult with status, pose, fuel, wall_seconds
    Fresh instance per episode: nothing is reset between episodes, and a trap kills the instance for good.
    """

    def __init__(self, wasm_path, runtime=None, limits=None, log_dir=None, dev_dir=None, inherit_stdio=False):
        self.path = Path(wasm_path)
        self.limits = limits or Limits()
        self.runtime = runtime or Runtime()
        self.dev_dir = Path(dev_dir) if dev_dir else None
        self.dead = False
        self.initialized = False
        wasm = self.path.read_bytes()
        try:
            self.manifest = read_manifest(wasm)
            if self.dev_dir is not None and (self.dev_dir / abi.BUNDLE_MANIFEST).is_file():  # dev mode: the directory's eecs116.json
                self.manifest = {**self.manifest, **json.loads((self.dev_dir / abi.BUNDLE_MANIFEST).read_text())}
            validate_manifest(self.manifest)
        except (BundleError, ValueError) as error:
            raise SolutionError(str(error)) from None
        self.channels = int(self.manifest.get("channels", abi.DEFAULT_CHANNELS))  # 1 (default) or 3: what step() receives
        base, self.bundle = split_bundle(wasm, abi.BUNDLE_SECTION)
        base = strip_custom(base, abi.MANIFEST_SECTION)  # every submission on the same runtime shares one compilation
        module = self.runtime.module(base)
        export_names = {export.name for export in module.exports}
        missing = [name for name in abi.REQUIRED_EXPORTS if name not in export_names]
        if missing:
            raise SolutionError(f"{self.path.name} is not a solution module: missing exports {missing} (has {sorted(export_names)})")

        if log_dir is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="eecs116-")
            log_dir = self._temporary.name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = self.log_dir / "stdout.txt"
        self.stderr_path = self.log_dir / "stderr.txt"

        self.store = wasmtime.Store(self.runtime.engine)
        self.store.set_limits(memory_size=self.limits.memory_bytes)
        wasi = wasmtime.WasiConfig()
        wasi.argv = ["solution"]
        environment = [("EECS116", "1")]
        if self.dev_dir is not None:
            wasi.preopen_dir(str(self.dev_dir), "/solution", fs_mutable=False)
            environment.append((abi.SOLUTION_DIR_VARIABLE, "/solution"))
        wasi.env = environment
        if inherit_stdio:
            wasi.inherit_stdout()
            wasi.inherit_stderr()
        else:
            wasi.stdout_file = str(self.stdout_path)
            wasi.stderr_file = str(self.stderr_path)
        self.store.set_wasi(wasi)
        linker = wasmtime.Linker(self.runtime.engine)
        linker.define_wasi()

        self._arm(self.limits.init_fuel, self.limits.init_wall_seconds)
        self.instance = self._guard(lambda: linker.instantiate(self.store, module))
        exports = self.instance.exports(self.store)
        self.memory = exports["memory"]
        self._alloc = exports["alloc"]
        self._init = exports["init"]
        self._step = exports["step"]
        self._load_bundle = exports["load_bundle"] if "load_bundle" in export_names else None
        if "abi_version" in export_names:
            version = self._guard(lambda: exports["abi_version"](self.store))
            if version != abi.ABI_VERSION:
                raise SolutionError(f"{self.path.name} implements ABI version {version}, the harness speaks {abi.ABI_VERSION}")
        if "_initialize" in export_names:
            self._guard(lambda: exports["_initialize"](self.store))

    # --- plumbing -------------------------------------------------------------------------------------------

    def _arm(self, fuel, wall_seconds):
        if self.runtime.fuel:
            self.store.set_fuel(fuel)
        if self.runtime.epochs:
            self.store.set_epoch_deadline(self.runtime.ticks(wall_seconds))

    def _consumed(self, fuel):
        return fuel - self.store.get_fuel() if self.runtime.fuel else 0

    def _guard(self, call):
        try:
            return call()
        except wasmtime.Trap as trap:
            self.dead = True
            raise SolutionTrap(trap.trap_code, trap.message, self.stderr_tail()) from None
        except wasmtime.ExitTrap as trap:
            self.dead = True
            raise SolutionTrap("exit", f"the guest called exit({trap.code})", self.stderr_tail()) from None

    def alloc(self, size):
        pointer = self._guard(lambda: self._alloc(self.store, int(size))) & 0xFFFFFFFF
        if pointer == 0:
            raise SolutionError(f"the guest could not allocate {size} bytes")
        return pointer

    def write(self, pointer, array):
        """Single memmove from a contiguous numpy array (or bytes) into guest memory at pointer."""
        array = np.ascontiguousarray(np.frombuffer(array, np.uint8) if isinstance(array, (bytes, bytearray)) else array)
        size = array.nbytes
        if pointer + size > self.memory.data_len(self.store):
            raise SolutionError(f"write of {size} bytes at {pointer} exceeds guest memory")
        base = ctypes.cast(self.memory.data_ptr(self.store), ctypes.c_void_p).value  # re-fetched: memory may have moved
        ctypes.memmove(base + pointer, array.ctypes.data, size)

    def read(self, pointer, size):
        return bytes(self.memory.read(self.store, pointer, pointer + size))

    def _tail(self, path, lines=40):
        try:
            text = path.read_text(errors="replace")
        except FileNotFoundError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def stderr_tail(self, lines=40):
        return self._tail(self.stderr_path, lines)

    def stdout_tail(self, lines=40):
        return self._tail(self.stdout_path, lines)

    # --- protocol -------------------------------------------------------------------------------------------

    def init(self, config):
        """Hands the bundle (if any) and the config JSON to the guest; raises SolutionError on failure."""
        if self.initialized:
            raise SolutionError("init() called twice")
        if self.bundle is not None:
            if self._load_bundle is None:
                raise SolutionError(f"{self.path.name} carries a {abi.BUNDLE_SECTION} section but exports no load_bundle")
            pointer = self.alloc(len(self.bundle))
            self.write(pointer, self.bundle)
            status = self._guard(lambda: self._load_bundle(self.store, pointer, len(self.bundle)))
            if status != abi.STATUS_OK:
                raise SolutionError(f"load_bundle failed: {abi.status_name(status)}\n{self.stderr_tail()}")
        elif self._load_bundle is not None and self.dev_dir is None:
            raise SolutionError(f"{self.path.name} is a bare runtime without solution code: run `python -m eecs116 bundle` or pass --dir")
        self.config = dict(config)
        self.config.setdefault("abi_version", abi.ABI_VERSION)
        if int(self.config["image"].get("channels", 1)) != self.channels:
            raise SolutionError(f"config promises {self.config['image'].get('channels', 1)} channel(s) but the solution declares {self.channels}; "
                                "adapt the episode with Episode.as_channels() first")
        encoded = json.dumps(self.config).encode("utf-8")
        pointer = self.alloc(len(encoded))
        self.write(pointer, encoded)
        started = time.perf_counter()
        status = self._guard(lambda: self._init(self.store, pointer, len(encoded)))
        self.init_wall_seconds = time.perf_counter() - started
        self.init_fuel = self._consumed(self.limits.init_fuel)
        if status != abi.STATUS_OK:
            raise SolutionError(f"init failed: {abi.status_name(status)}\n--- stderr tail ---\n{self.stderr_tail()}")
        image = self.config["image"]
        self.height, self.width = int(image["height"]), int(image["width"])
        self.channels = int(image.get("channels", 1))
        self.frame_bytes = self.height * self.width * self.channels
        self.frame_pointer = self.alloc(self.frame_bytes)
        self.out_pointer = self.alloc(abi.POSE_VALUES * abi.F64_BYTES)
        self.imu_capacity = self.limits.imu_capacity
        self.imu_pointer = self.alloc(self.imu_capacity * abi.IMU_ROW_VALUES * abi.F64_BYTES)
        self.initialized = True

    def step(self, t, frame, imu):
        if not self.initialized:
            raise SolutionError("step() before init()")
        if self.dead:
            raise SolutionError("the instance trapped earlier; create a new Solution")
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.nbytes != self.frame_bytes:
            raise SolutionError(f"frame has {frame.nbytes} bytes, the config promises {self.height}x{self.width}x{self.channels}")
        imu = np.ascontiguousarray(imu, dtype=np.float64).reshape(-1, abi.IMU_ROW_VALUES)
        if len(imu) > self.imu_capacity:
            self.imu_capacity = max(len(imu), 2 * self.imu_capacity)
            self.imu_pointer = self.alloc(self.imu_capacity * abi.IMU_ROW_VALUES * abi.F64_BYTES)  # the old block just stays allocated
        self.write(self.frame_pointer, frame)
        if len(imu):
            self.write(self.imu_pointer, imu)
        self._arm(self.limits.step_fuel, self.limits.step_wall_seconds)
        started = time.perf_counter()
        status = self._guard(lambda: self._step(self.store, float(t), self.frame_pointer, self.imu_pointer, len(imu), self.out_pointer))
        wall = time.perf_counter() - started
        fuel = self._consumed(self.limits.step_fuel)
        pose = None
        if status == abi.STATUS_OK:
            pose = np.frombuffer(self.read(self.out_pointer, abi.POSE_VALUES * abi.F64_BYTES), dtype=np.float64).copy()
        return StepResult(status, pose, fuel, wall)
