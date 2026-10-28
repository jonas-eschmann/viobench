"""`relpose check`: instantiate a module and push a few synthetic frames through it (protocol, exceptions, fuel)."""
import numpy as np

from .. import abi
from ..abi.config import default_guest_config
from .sandbox import Limits, Sandbox, SandboxError

FRAME_RATE_HZ = 20.0  # the nominal rates of every set (docs/ABI.md); solutions go by the timestamps
IMU_RATE_HZ = 200.0


def check(wasm, frames=5, logs=None, verbose=False, seed=0, log=print):
    """Returns 0 when every step succeeded, 1 otherwise. Prints one line per step."""
    rng = np.random.default_rng(seed)
    try:
        sandbox = Sandbox(wasm, limits=Limits(), log_dir=logs, inherit_stdio=verbose)
    except SandboxError as error:
        log(f"load failed: {error}")
        return 1
    width, height = sandbox.resolution
    config = default_guest_config(width=width, height=height)
    shape = (height, width, abi.CHANNELS)
    dt = 1.0 / FRAME_RATE_HZ
    imu_dt = 1.0 / IMU_RATE_HZ
    rows_per_frame = int(round(dt / imu_dt))
    log(f"module: frames {shape}, manifest {sandbox.manifest or '{}'}")
    try:
        sandbox.init(config)
    except SandboxError as error:
        log(f"init failed: {error}")
        return 1
    log(f"init: fuel {sandbox.init_fuel:,} wall {sandbox.init_wall_seconds * 1e3:.0f} ms")
    failures = 0
    for k in range(frames):
        t = (k + 1) * dt
        base = np.linspace(0, 255, width, dtype=np.float32)[None, :] + rng.normal(0, 20, (height, width))
        frame = np.clip(base, 0, 255).astype(np.uint8)
        frame = np.repeat(frame[:, :, None], abi.CHANNELS, axis=2)
        times = t - dt + imu_dt * np.arange(1, rows_per_frame + 1)
        imu = np.column_stack([times, rng.normal(0, 0.3, (rows_per_frame, 3)) + [0, 0, 9.81], rng.normal(0, 0.07, (rows_per_frame, 3))])
        try:
            result = sandbox.step(t, frame, imu)
        except SandboxError as error:
            log(f"step {k}: {error}")
            return 1
        pose = "-" if result.pose is None else np.array2string(result.pose, precision=3, suppress_small=True)
        log(f"step {k}: {abi.status_name(result.status):12s} fuel {result.fuel:>14,} wall {result.wall_seconds * 1e3:8.2f} ms pose {pose}")
        failures += result.status < 0
    if not verbose:
        out, err = sandbox.stdout_tail(), sandbox.stderr_tail()
        if out:
            log("--- stdout (tail) ---\n" + out)
        if err:
            log("--- stderr (tail) ---\n" + err)
    log("check: " + ("FAILED" if failures else "ok") + f" ({failures} failing steps of {frames})")
    return 1 if failures else 0
