"""Episode generator on HyperDrone: random autopilot flights through ProcTHOR/HSSD scenes, rendered at IMU rate.

    from eecs116.data.simulate import SimConfig, simulate
    for episode in simulate(SimConfig(), seeds=range(4), variant="default"):
        ...

Needs `pip install hyperdrone` and a GPU (the environment module is JIT-compiled per configuration on first use;
the rendered spec header's hash is the key). Frames are RGB uint8 after the camera/ISP model (eecs116.data.isp)
with per-episode parameters drawn from the variant's public ranges (eecs116.data.variants); solutions receive
grayscale unless they opt into RGB (Episode.as_channels).
"""
import hashlib
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import abi
from ..config import make_config, pinhole_intrinsics
from ..metrics import matrix_to_quaternion, quaternion_to_matrix
from .episode import Episode
from .isp import apply, luminance, srgb_to_linear
from .variants import sample as sample_variant

TEMPLATE = Path(__file__).with_name("spec_header.h.in")
CACHE_VARIABLE = "EECS116_CACHE"
NOMINAL_FOV_DEGREES = 63.78166175396324  # preset camera, horizontal


@dataclass(frozen=True)
class SimConfig:
    width: int = 640  # stored resolution; the harness resamples to what a solution asks for (320x240 by default)
    height: int = 480
    instances: int = 8  # episodes rendered in parallel (one reset = one batch); part of the JIT key
    frame_rate_hz: float = 20.0
    fov_range: float = 15.0  # horizontal FOV = nominal (63.8 deg) + U(-fov_range, fov_range) degrees per episode
    hold_seconds: float = 1.0  # hover before the route starts
    aa_grid: int = 1
    channels: int = 3  # stored frames; the harness converts to what a solution asks for (grayscale by default)

    def render_header(self, imu_rate_hz):
        text = TEMPLATE.read_text()
        values = {"INSTANCES": self.instances, "WIDTH": self.width, "HEIGHT": self.height, "AA_GRID": self.aa_grid,
                  "FOV_RANGE": repr(float(self.fov_range)), "HOLD_STEPS": int(round(self.hold_seconds * imu_rate_hz))}
        for key, value in values.items():
            text = text.replace(f"@{key}@", str(value))
        assert "@" not in text, "unrendered placeholder in spec header template"
        cache = Path(os.environ.get(CACHE_VARIABLE, Path.home() / ".cache" / "eecs116"))
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"spec-{hashlib.sha256(text.encode()).hexdigest()[:12]}.h"
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
        return path


def mount_to_body_camera(mount):
    """T_body_camera (OpenCV camera axes) from a HyperDrone camera mount {"pose": {"position", "orientation" (w, x, y, z)}}.

    The mount quaternion maps the camera's FLU axes (x forward, y left, z up) into the body frame; OpenCV's camera
    frame is x right, y down, z forward — the same construction as calibration.h's camera_to_body_transform.
    """
    w, x, y, z = mount["pose"]["orientation"]
    R_body_camflu = quaternion_to_matrix([x, y, z, w])
    forward = R_body_camflu[:, 0]
    up = R_body_camflu[:, 2]
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = right, down, forward
    T[:3, 3] = mount["pose"]["position"]
    return T


def _catalog_scenes(catalog, names=None):
    """{name: conta reference} of a hyperlux scene catalog (indexable by position and by scene name)."""
    scenes = {}
    for index in range(len(catalog)):
        entry = catalog[index]
        scenes[str(getattr(entry, "id", None) or f"{catalog.name}-{index}")] = entry.reference
    if names:
        missing = [name for name in names if name not in scenes]
        if missing:
            raise KeyError(f"unknown scenes {missing}; known: {list(scenes)[:5]}...")
        scenes = {name: scenes[name] for name in names}
    return scenes


def procthor_scenes(names=None):
    """conta references of ProcTHOR training scenes (all 200 by default)."""
    import hyperlux

    return _catalog_scenes(hyperlux.datasets.procthor.Train200(), names)


def hssd_scenes(split="val", names=None):
    import hyperlux

    return _catalog_scenes(hyperlux.datasets.hssd.Val43() if split == "val" else hyperlux.datasets.hssd.Train125(), names)


# a user-spec World exposes unnamed observation blocks: privileged = [position 3 | rotation matrix 9 | linear velocity 3 |
# angular velocity 3 | waypoint 3], IMU = [accelerometer 3 | gyroscope 3 | frame_age | new_frame]
BLOCKS = {"position": (0, 3), "orientation_rotation_matrix": (3, 9), "linear_velocity": (12, 3), "angular_velocity": (15, 3),
          "accelerometer": (0, 3), "gyroscope": (3, 3)}


def _block(vector, layout, name):
    offset, size = layout.blocks.get(name, BLOCKS[name])
    return vector[:, offset:offset + size]


def simulate(sim, seeds, scenes=None, variant="default", scene_names=None, log=print):
    """Yields Episodes: for each seed one batch of `sim.instances` flights in one scene (round-robin over scenes).

    scenes: {name: reference} (default: ProcTHOR Train200). Terminated flights (collisions) are skipped.
    """
    import hyperdrone

    if scenes is None:
        scenes = procthor_scenes(scene_names)
    scene_items = list(scenes.items())
    header = None
    env = None
    current_scene = None
    for seed in seeds:
        scene_name, reference = scene_items[seed % len(scene_items)]
        if env is None or scene_name != current_scene:
            if env is not None:
                env.close()
            if header is None:
                header = sim.render_header(200.0)
            env = hyperdrone.HyperDrone([reference], config=hyperdrone.Config(num_worlds=1, spec_header=str(header)), seed=int(seed))
            current_scene = scene_name
            imu_rate = float(round(1.0 / float(env.dt)))
            if header.name != sim.render_header(imu_rate).name:  # the preset's IMU rate differs from the assumed 200 Hz
                raise RuntimeError(f"IMU rate {imu_rate} Hz: re-render the spec header for it")
        yield from _run_batch(env, sim, int(seed), scene_name, variant, log)
    if env is not None:
        env.close()


def _run_batch(env, sim, seed, scene_name, variant, log):
    import time

    imu_rate = float(round(1.0 / float(env.dt)))  # env.dt is float32 (0.00499999989)
    dt = 1.0 / imu_rate
    steps = int(env.episode_step_limit)
    frame_every = int(round(1.0 / (sim.frame_rate_hz * dt)))
    total = env.total_instances
    layout = env.observation_layout_privileged
    layout_imu = env.observation_layout_imu
    rng = np.random.default_rng([seed, abs(hash(variant)) % (2**32)])
    isp = [sample_variant(variant, rng, imu_rate_hz=imu_rate) for _ in range(total)]
    sub_frames = max(k for _, k in isp)  # ring buffer length

    env.reset(np.ones(total, dtype=bool))
    parameters = env.instance_parameters()
    spawns = env.spawn_points()
    state = env.observe_privileged()
    ground_truth = [np.column_stack([np.zeros(total), _block(state, layout, "position"),
                                     matrix_to_quaternion(_block(state, layout, "orientation_rotation_matrix").reshape(-1, 3, 3))])]
    imu_rows = []
    frames = [[] for _ in range(total)]
    frame_times = []
    sub_frames = min(sub_frames, frame_every)  # a shutter longer than the frame interval would overlap the previous frame
    ring = deque(maxlen=sub_frames)
    terminated = np.zeros(total, dtype=bool)
    instance_rngs = [np.random.default_rng(rng.integers(2**63)) for _ in range(total)]
    pool = ThreadPoolExecutor(max_workers=min(total, os.cpu_count() or 1))  # numpy releases the GIL on large arrays

    def to_linear(rendered):
        convert = (lambda x: luminance(srgb_to_linear(x))) if sim.channels == 1 else srgb_to_linear
        return np.stack(list(pool.map(convert, rendered)))

    def develop(instance):
        params, k = isp[instance]
        stack = [linear[instance] for linear in list(ring)[-k:]]
        return apply(np.mean(stack, axis=0) if len(stack) > 1 else stack[-1], params, instance_rngs[instance], grayscale=False)

    started = time.perf_counter()
    for step in range(steps):
        # only sub-frames inside the longest shutter window before a frame's closing step are needed: converting
        # every render to linear radiance would dominate the run time at 640x480
        if step % frame_every >= frame_every - sub_frames:
            rendered = env.frames()  # (total, H, W, 3) sRGB float32 in [0, 1], camera at the pre-step state (time step*dt)
            ring.append(to_linear(rendered))
        if step % frame_every == frame_every - 1:
            for instance, frame in enumerate(pool.map(develop, range(total))):
                frames[instance].append(frame)
            frame_times.append(step * dt)  # provisional: the per-instance mid-exposure correction is applied below
        env.step()
        sample = env.observe_imu()  # the sample produced by this step, stamped at its end
        imu_rows.append(np.column_stack([np.full(total, (step + 1) * dt), _block(sample, layout_imu, "accelerometer"), _block(sample, layout_imu, "gyroscope")]))
        state = env.observe_privileged()
        ground_truth.append(np.column_stack([np.full(total, (step + 1) * dt), _block(state, layout, "position"),
                                             matrix_to_quaternion(_block(state, layout, "orientation_rotation_matrix").reshape(-1, 3, 3))]))
        terminated |= env.terminated()
    elapsed = time.perf_counter() - started
    pool.shutdown()
    fovs = [float((p["cameras"].get("front") or next(iter(p["cameras"].values())))["fov"]) for p in parameters]
    log(f"seed {seed} scene {scene_name}: {steps} steps x {total} instances in {elapsed:.1f}s ({steps * total / elapsed:.0f} steps/s), "
        f"{int(terminated.sum())} terminated, fov {np.round(fovs, 1).tolist()}")

    imu_rows = np.stack(imu_rows)  # (steps, total, 7)
    ground_truth = np.stack(ground_truth)  # (steps + 1, total, 8)
    frame_times = np.asarray(frame_times)
    for instance in range(total):
        if terminated[instance]:
            continue
        params, k = isp[instance]
        cameras = parameters[instance]["cameras"]
        camera = cameras.get("front") or next(iter(cameras.values()))
        fov = float(camera["fov"])
        T_body_camera = mount_to_body_camera(camera["mount"])
        intrinsics = pinhole_intrinsics(sim.width, sim.height, fov)
        config = make_config(width=sim.width, height=sim.height, channels=sim.channels, frame_rate_hz=sim.frame_rate_hz,
                             num_frames=len(frame_times), hover_seconds=sim.hold_seconds, T_body_camera=T_body_camera,
                             intrinsics=intrinsics, episode_name=f"{variant}-{scene_name}-{seed:05d}-{instance:02d}")
        config["camera"]["horizontal_fov_degrees"] = fov
        config["episode"]["exposure_seconds"] = k * dt
        # frame k integrates the renders at steps c-k+1..c (c = closing step): mid-exposure is (c - (k-1)/2) dt
        times = frame_times - (k - 1) / 2.0 * dt
        episode = Episode(
            name=config["episode"]["name"],
            config=config,
            frame_times=times,
            frames=np.stack(frames[instance]),
            imu=imu_rows[:, instance, :],
            ground_truth=ground_truth[:, instance, :],
            T_body_camera=T_body_camera,
            meta={"source": "hyperdrone", "scene": scene_name, "seed": seed, "instance": instance, "variant": variant,
                  "isp": params.as_dict(), "sub_frames": k, "spawn": spawns[instance], "hyperdrone": _versions(),
                  "sim": asdict(sim)},
        )
        yield episode


def _versions():
    try:
        from importlib.metadata import version

        return {"hyperdrone": version("hyperdrone"), "hyperlux": version("hyperlux")}
    except Exception:  # noqa: BLE001
        return {}
