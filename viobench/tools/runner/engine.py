"""Wasmtime engine configuration shared by all evaluations: fuel metering, epoch watchdog and a compile cache."""
import hashlib
import os
import threading
from contextlib import contextmanager
from importlib.metadata import version as package_version

import wasmtime
from wasmtime import _bindings

WASMTIME_VERSION = package_version("wasmtime")
EPOCH_TICK_SECONDS = 0.1  # one epoch tick; wall-clock deadlines are expressed in ticks
MAX_WASM_STACK = 16 << 20  # native stack for guest frames (CPython's own WASI test runner uses 16 MiB)


@contextmanager
def _compilation_lock():
    """Workers share one lock across evaluator processes; ordinary local runs need none."""
    path = os.environ.get("VIOBENCH_COMPILE_LOCK")
    if path is None:
        yield
        return
    import fcntl  # the server runs on Unix; local evaluation does not require this module

    # Never unlink this file: all evaluators must lock the same inode, including after a retry.
    # Closing the descriptor (also on errors or process death) releases the lock.
    with open(path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def make_config():
    config = wasmtime.Config()
    config.wasm_exceptions = True  # the runtime is built with -fwasm-exceptions (new EH proposal)
    config.wasm_simd = True
    config.wasm_bulk_memory = True
    config.consume_fuel = True
    config.epoch_interruption = True
    config.cranelift_opt_level = "speed"
    # the C API validates max_wasm_stack <= async_stack_size (default 2 MiB) even though nothing here runs async
    _bindings.wasmtime_config_async_stack_size_set(config.ptr(), 2 * MAX_WASM_STACK)
    config.max_wasm_stack = MAX_WASM_STACK
    config.cache = True  # wasmtime's own on-disk cache, keyed by module bytes and config
    return config


class Engine:
    """One wasmtime engine with its epoch ticker plus a per-process cache of compiled modules keyed by the binary's hash.

    The bundle is a data segment, so every distinct bundle is its own compilation.
    """

    def __init__(self):
        self.config = make_config()
        self.wasmtime = wasmtime.Engine(self.config)
        self._modules = {}
        self._start_ticker()

    def _start_ticker(self):
        engine = self.wasmtime

        def tick():
            while True:
                threading.Event().wait(EPOCH_TICK_SECONDS)
                engine.increment_epoch()

        self._ticker = threading.Thread(target=tick, name="wasmtime-epoch-ticker", daemon=True)
        self._ticker.start()

    @staticmethod
    def ticks(seconds):
        return max(1, int(seconds / EPOCH_TICK_SECONDS + 0.999))

    def module(self, wasm):
        key = hashlib.sha256(wasm).hexdigest()
        module = self._modules.get(key)
        if module is None:
            # Serialize cache lookup and compilation together so waiters reuse the first process's disk cache.
            with _compilation_lock():
                module = wasmtime.Module(self.wasmtime, wasm)
            self._modules[key] = module
        return module
