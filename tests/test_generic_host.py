"""A bundled solution is a self-contained wasip1 reactor: a host that knows nothing about eecs116 can drive it.

This is the whole protocol in 25 lines of plain wasmtime-py: instantiate, _initialize, init(config), step(...).
"""
import ctypes
import json

import numpy as np
import wasmtime

from eecs116.bundle import bundle
from eecs116.config import make_config

SOLUTION = """
import numpy as np
import eecs116_bundle as bundle
class Solution:
    def __init__(self, config):
        self.offset = float(bundle.read_text("offset.txt"))
    def step(self, t, frame, imu):
        return [self.offset, float(frame.mean()), t, 0, 0, 0, 1]
"""


def test_plain_wasmtime_host_runs_a_bundle(runtime_path, solution_dir, tmp_path):
    wasm = tmp_path / "s.wasm"
    bundle(solution_dir(SOLUTION, {"offset.txt": "2.5"}), wasm, runtime=runtime_path)

    config = wasmtime.Config()
    config.wasm_exceptions = True
    config.cache = True
    engine = wasmtime.Engine(config)
    store = wasmtime.Store(engine)
    wasi = wasmtime.WasiConfig()
    wasi.inherit_stdout()
    wasi.inherit_stderr()
    store.set_wasi(wasi)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    instance = linker.instantiate(store, wasmtime.Module.from_file(engine, str(wasm)))
    exports = instance.exports(store)
    memory, alloc, init, step = exports["memory"], exports["alloc"], exports["init"], exports["step"]
    exports["_initialize"](store)

    def write(pointer, data):
        ctypes.memmove(ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value + pointer, data, len(data))

    config_bytes = json.dumps(make_config(width=32, height=24)).encode()
    pointer = alloc(store, len(config_bytes))
    write(pointer, config_bytes)
    assert init(store, pointer, len(config_bytes)) == 0

    frame = np.full(32 * 24, 100, np.uint8).tobytes()
    frame_pointer, imu_pointer, out_pointer = alloc(store, len(frame)), alloc(store, 7 * 8), alloc(store, 7 * 8)
    write(frame_pointer, frame)
    assert step(store, 0.5, frame_pointer, imu_pointer, 0, out_pointer) == 0
    pose = np.frombuffer(bytes(memory.read(store, out_pointer, out_pointer + 56)), np.float64)
    np.testing.assert_allclose(pose, [2.5, 100.0, 0.5, 0, 0, 0, 1])
