"""Sandbox: one running instance of a solution module (Python or native alike): init, step, fuel and wall time per step."""
import ctypes
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import wasmtime

from .. import abi
from ..abi import AbiError
from ..abi.module import runtime_descriptor
from .engine import Engine
from .metrics import valid_states

IMU_CAPACITY_ROWS = 1024  # initial IMU buffer; grows on demand
UNWRITTEN = np.full(abi.STATE_VALUES, np.nan)  # `out` before every step: whatever the step does not write is unusable
STDOUT_FILE = "stdout.txt"  # in the log directory
STDERR_FILE = "stderr.txt"


def tail(path, lines=40):
    """The last lines of a log file ("" while it does not exist)."""
    try:
        text = Path(path).read_text(errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


CHALLENGE_LIMITS = json.loads((Path(__file__).parents[1] / "config.json").read_text())["limits"]  # the server's too


@dataclass
class Limits:
    """Defaults: the challenge's limits (relpose/tools/config.json), so local runs and the server apply the same."""
    init_fuel: int = int(CHALLENGE_LIMITS["init_fuel"])  # covers _initialize and init (interpreter start + imports)
    step_fuel: int = int(CHALLENGE_LIMITS["step_fuel"])  # per step; exceeding it traps and ends the episode
    memory_bytes: int = int(CHALLENGE_LIMITS["memory_bytes"])  # the module's one linear memory
    table_elements: int = int(CHALLENGE_LIMITS["table_elements"])  # its one table (function references)
    init_wall_seconds: float = float(CHALLENGE_LIMITS["init_wall_seconds"])  # wall-clock safety nets only (fuel is the metric)
    step_wall_seconds: float = float(CHALLENGE_LIMITS["step_wall_seconds"])

    @classmethod
    def from_dict(cls, values):
        """Limits from a JSON object: any subset of the fields (the rest keep their defaults), numbers converted."""
        defaults = asdict(cls())
        unknown = sorted(set(values) - set(defaults))
        if unknown:
            raise ValueError(f"unknown limits {unknown}; known: {sorted(defaults)}")
        return cls(**{key: type(defaults[key])(value) for key, value in values.items()})


@dataclass
class StepResult:
    pose: np.ndarray  # [x, y, z, qx, qy, qz, qw]
    twist: np.ndarray  # [vx, vy, vz, wx, wy, wz]
    fuel: int
    wall_seconds: float


class SandboxError(Exception):
    """The solution failed. Failed step calls carry their consumed fuel and wall time for the report."""

    def __init__(self, message):
        super().__init__(message)
        self.fuel = 0
        self.wall_seconds = 0.0


class SandboxTrap(SandboxError):
    """The guest trapped (out of fuel, wall-clock deadline, abort, memory fault, exit()); the instance is dead."""

    def __init__(self, trap_code, message, stderr_tail):
        super().__init__(f"trap {trap_code}: {message}\n--- stderr tail ---\n{stderr_tail}")
        self.trap_code = trap_code
        self.trap_message = message
        self.stderr_tail = stderr_tail


class Sandbox:
    """One running instance of a solution module (Python or native alike) for one episode.

    sandbox = Sandbox("student.wasm", engine)
    sandbox.init(config)
    result = sandbox.step(t, frame, imu)   # StepResult with pose, twist, fuel, wall_seconds; SandboxError on a failed step
    Fresh instance per episode: nothing is reset between episodes, and a trap kills the instance for good.
    """

    def __init__(self, wasm_path, engine=None, limits=None, log_dir=None, inherit_stdio=False):
        self.path = Path(wasm_path)
        self.limits = limits or Limits()
        self.engine = engine or Engine()
        self.dead = False
        self.initialized = False
        if log_dir is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="relpose-")
            log_dir = self._temporary.name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = self.log_dir / STDOUT_FILE
        self.stderr_path = self.log_dir / STDERR_FILE
        for path in (self.stdout_path, self.stderr_path):
            path.unlink(missing_ok=True)  # an earlier run's logs must not pass for this one's

        wasm = self.path.read_bytes()
        descriptor = runtime_descriptor(wasm)  # None for native modules; .length == 0 for a bare Python runtime
        if descriptor is not None and descriptor.length == 0:
            raise SandboxError(f"{self.path.name} is a bare Python runtime without solution code: run `relpose bundle` first")
        module = self.engine.module(wasm)
        export_names = {export.name for export in module.exports}
        missing = [name for name in abi.REQUIRED_EXPORTS if name not in export_names]
        if missing:
            raise SandboxError(f"{self.path.name} is not a solution module: missing exports {missing} (has {sorted(export_names)})")

        self.store = wasmtime.Store(self.engine.wasmtime)
        # per-memory and per-table caps bound the total only with one of each (and one instance)
        self.store.set_limits(memory_size=self.limits.memory_bytes, table_elements=self.limits.table_elements,
                              instances=1, memories=1, tables=1)
        wasi = wasmtime.WasiConfig()
        wasi.argv = ["solution"]
        wasi.env = [("RELPOSE", "1")]
        if inherit_stdio:
            wasi.inherit_stdout()
            wasi.inherit_stderr()
        else:
            wasi.stdout_file = str(self.stdout_path)
            wasi.stderr_file = str(self.stderr_path)
        self.store.set_wasi(wasi)
        linker = wasmtime.Linker(self.engine.wasmtime)
        linker.define_wasi()

        self._arm(self.limits.init_fuel, self.limits.init_wall_seconds)
        try:
            self.instance = self._guard(lambda: linker.instantiate(self.store, module))
        except wasmtime.WasmtimeError as error:  # not a trap: beyond the limits above, or an unknown import
            raise SandboxError(f"{self.path.name} cannot be instantiated: {str(error).splitlines()[0]}") from None
        exports = self.instance.exports(self.store)
        self.memory = exports["memory"]
        self._alloc = exports["alloc"]
        self._init = exports["init"]
        self._step = exports["step"]
        version = self._guard(lambda: exports["abi_version"](self.store))
        if version != abi.ABI_VERSION:
            raise SandboxError(f"{self.path.name} implements ABI version {version}, the harness speaks {abi.ABI_VERSION}")
        if "_initialize" in export_names:
            self._guard(lambda: exports["_initialize"](self.store))
        frame_width = None
        if abi.FRAME_WIDTH in export_names:  # the Python runtime starts the interpreter and imports solution.py here
            frame_width = self._guard(lambda: exports[abi.FRAME_WIDTH](self.store))
            if frame_width < 0:
                raise SandboxError(f"frame_width() failed (status {frame_width})\n--- stderr tail ---\n{self.stderr_tail()}")
        try:
            self.resolution = abi.resolution(frame_width)  # (width, height) of the frames
        except AbiError as error:
            raise SandboxError(str(error)) from None

    # --- plumbing -------------------------------------------------------------------------------------------

    def _arm(self, fuel, wall_seconds):
        self.store.set_fuel(fuel)
        self.store.set_epoch_deadline(self.engine.ticks(wall_seconds))

    def _consumed(self, fuel):
        return fuel - self.store.get_fuel()

    def _guard(self, call):
        try:
            return call()
        except wasmtime.Trap as trap:
            self.dead = True
            raise SandboxTrap(trap.trap_code, trap.message, self.stderr_tail()) from None
        except wasmtime.ExitTrap as trap:
            self.dead = True
            raise SandboxTrap("exit", f"the guest called exit({trap.code})", self.stderr_tail()) from None

    def alloc(self, size):
        pointer = self._guard(lambda: self._alloc(self.store, int(size))) & 0xFFFFFFFF
        if pointer == 0:
            raise SandboxError(f"the guest could not allocate {size} bytes")
        return pointer

    def write(self, pointer, array):
        """Single memmove from a contiguous numpy array (or bytes) into guest memory at pointer."""
        array = np.ascontiguousarray(np.frombuffer(array, np.uint8) if isinstance(array, (bytes, bytearray)) else array)
        size = array.nbytes
        if pointer + size > self.memory.data_len(self.store):
            raise SandboxError(f"write of {size} bytes at {pointer} exceeds guest memory")
        base = ctypes.cast(self.memory.data_ptr(self.store), ctypes.c_void_p).value  # re-fetched: memory may have moved
        ctypes.memmove(base + pointer, array.ctypes.data, size)

    def read(self, pointer, size):
        return bytes(self.memory.read(self.store, pointer, pointer + size))

    def stderr_tail(self, lines=40):
        return tail(self.stderr_path, lines)

    def stdout_tail(self, lines=40):
        return tail(self.stdout_path, lines)

    # --- protocol -------------------------------------------------------------------------------------------

    def init(self, config):
        """Hands the bundle (if any) and the config JSON to the guest; raises SandboxError on failure."""
        if self.initialized:
            raise SandboxError("init() called twice")
        self.config = dict(config)
        delivered = int(self.config["width"]), int(self.config["height"])
        if delivered != self.resolution:  # the frame size the solution declared
            raise SandboxError(f"config delivers {delivered[0]}x{delivered[1]} but the solution declares {self.resolution[0]}x{self.resolution[1]}; "
                               "adapt the episode with Episode.adapt() first")
        encoded = json.dumps(self.config).encode("utf-8")
        pointer = self.alloc(len(encoded))
        self.write(pointer, encoded)
        started = time.perf_counter()
        status = self._guard(lambda: self._init(self.store, pointer, len(encoded)))
        self.init_wall_seconds = time.perf_counter() - started
        self.init_fuel = self._consumed(self.limits.init_fuel)
        if status != abi.STATUS_OK:
            raise SandboxError(f"init failed (status {status})\n--- stderr tail ---\n{self.stderr_tail()}")
        self.width, self.height = self.resolution
        self.frame_bytes = self.height * self.width * abi.CHANNELS
        self.frame_pointer = self.alloc(self.frame_bytes)
        self.out_pointer = self.alloc(abi.STATE_VALUES * abi.F64_BYTES)
        self.imu_capacity = IMU_CAPACITY_ROWS
        self.imu_pointer = self.alloc(self.imu_capacity * abi.IMU_ROW_VALUES * abi.F64_BYTES)
        self.initialized = True

    def step(self, t, frame, imu):
        if not self.initialized:
            raise SandboxError("step() before init()")
        if self.dead:
            raise SandboxError("the instance trapped earlier; create a new Sandbox")
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.nbytes != self.frame_bytes:
            raise SandboxError(f"frame has {frame.nbytes} bytes, the config promises {self.height}x{self.width}x{abi.CHANNELS}")
        imu = np.ascontiguousarray(imu, dtype=np.float64).reshape(-1, abi.IMU_ROW_VALUES)
        if len(imu) > self.imu_capacity:
            self.imu_capacity = max(len(imu), 2 * self.imu_capacity)
            self.imu_pointer = self.alloc(self.imu_capacity * abi.IMU_ROW_VALUES * abi.F64_BYTES)  # the old block just stays allocated
        self.write(self.frame_pointer, frame)
        if len(imu):
            self.write(self.imu_pointer, imu)
        self.write(self.out_pointer, UNWRITTEN)
        self._arm(self.limits.step_fuel, self.limits.step_wall_seconds)
        started = time.perf_counter()
        try:
            status = self._guard(lambda: self._step(self.store, float(t), self.frame_pointer, self.imu_pointer, len(imu), self.out_pointer))
            wall = time.perf_counter() - started
            fuel = self._consumed(self.limits.step_fuel)
            if status != abi.STATUS_OK:
                raise SandboxError(f"step failed (status {status})\n--- stderr tail ---\n{self.stderr_tail()}")
            state = np.frombuffer(self.read(self.out_pointer, abi.STATE_VALUES * abi.F64_BYTES), dtype=np.float64).copy()
            if not valid_states(state[None])[0]:
                raise SandboxError(f"step returned an unusable state {state.tolist()} (not all {abi.STATE_VALUES} values written, NaN/inf, "
                                   "or an all-zero quaternion)")
        except SandboxError as error:
            error.fuel = self._consumed(self.limits.step_fuel)
            error.wall_seconds = time.perf_counter() - started
            raise
        return StepResult(state[:abi.POSE_VALUES], state[abi.POSE_VALUES:], fuel, wall)
