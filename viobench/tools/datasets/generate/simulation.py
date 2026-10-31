"""Episode generator on HyperDrone: random autopilot flights through ProcTHOR/HSSD scenes, rendered at IMU rate.

    from viobench.tools.datasets.generate.simulation import SimConfig, scene_references, simulate
    for episode in simulate(SimConfig(), scene_references(["ProcTHOR-Train-1"]), episodes=8):
        ...

Needs `pip install hyperdrone` and a GPU (the environment module is JIT-compiled per configuration on first use;
the rendered spec header's hash is the key, so every distinct resolution, duration or route length compiles once).
Frames are RGB uint8: by default each one is the render at its time (`nominal_camera`); another camera model plugs in
through `simulate(camera=...)`. Solutions receive frames as stored or rescaled (Episode.adapt). The episode length is
`SimConfig.duration_seconds` (default one minute); frames are spooled to a temporary directory while a round of
parallel flights renders, so memory does not grow with it.
"""
import hashlib
import json
import os
import shutil
import tempfile
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ... import abi
from ...abi.config import make_config, pinhole_intrinsics
from ...geometry import matrix_to_quaternion, quaternion_to_matrix
from ..episode import Episode, FrameSequence
from ..regularize import regularize
from ..storage import PNG_COMPRESSION, read_png

TEMPLATE = Path(__file__).with_name("hyperdrone_world.h.in")
CACHE_VARIABLE = "VIOBENCH_CACHE"
SPOOL_VARIABLE = "VIOBENCH_SPOOL"
NOMINAL_FOV_DEGREES = 63.78166175396324  # preset camera, horizontal
DEFAULT_FOV_DEGREES = 120.0  # base horizontal FOV of generated episodes (SimConfig.fov, --fov): a wide FPV lens, not the preset camera
DEFAULT_WAYPOINTS = 48  # --waypoints default: the route length of a one-minute flight, fixed so that --duration alone does not change the route
DEFAULT_SCENE_NAMES = ["ProcTHOR-Train-1"]  # --scene-names default
DEFAULT_EPISODES = 8  # --episodes default: episodes per scene
PARALLEL_FLIGHTS = 4  # flights the CLI simulates at once (SimConfig.instances); sized for the lab GPUs, not exposed as an option
MAX_FLIGHTS_PER_EPISODE = 3  # a scene that has not delivered its episodes after this many flights per episode is given up
WAYPOINTS_PER_SECOND = 0.8  # route length per second of flight unless SimConfig.waypoints is given
MIN_WAYPOINTS = 8


@dataclass(frozen=True)
class SimConfig:
    width: int = 640  # stored resolution; the harness resamples to what a solution asks for (320x240 by default)
    height: int = 480
    instances: int = PARALLEL_FLIGHTS  # flights rendered in parallel (one reset = one round); part of the JIT key
    frame_rate_hz: float = 20.0
    duration_seconds: float = 60.0  # flight time per episode; frames = duration x frame rate (HyperDrone's own step limit is informational)
    waypoints: int | None = None  # route length; None scales it with the duration (num_waypoints), the route cycles when exhausted
    fov_range: float = 15.0  # horizontal FOV = fov + U(-fov_range, fov_range) degrees per episode
    hold_seconds: float = 1.0  # hover before the route starts
    spool: str | None = None  # directory for the frames while a batch renders (default: $VIOBENCH_SPOOL or the system temp dir)
    fov: float = DEFAULT_FOV_DEGREES  # base horizontal FOV in degrees, before per-episode randomization

    @property
    def num_waypoints(self):
        return int(self.waypoints) if self.waypoints else max(MIN_WAYPOINTS, int(round(self.duration_seconds * WAYPOINTS_PER_SECOND)))

    def step_count(self, imu_rate_hz):
        return int(round(self.duration_seconds * imu_rate_hz))

    def render_header(self, imu_rate_hz):
        text = TEMPLATE.read_text()
        values = {"INSTANCES": self.instances, "WIDTH": self.width, "HEIGHT": self.height,
                  "FOV_RANGE": repr(float(self.fov_range)), "HOLD_STEPS": int(round(self.hold_seconds * imu_rate_hz)),
                  "WAYPOINTS": self.num_waypoints}
        for key, value in values.items():
            text = text.replace(f"@{key}@", str(value))
        assert "@" not in text, "unrendered placeholder in spec header template"
        cache = Path(os.environ.get(CACHE_VARIABLE, Path.home() / ".cache" / "viobench"))
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"hyperdrone_world-{hashlib.sha256(text.encode()).hexdigest()[:12]}.h"
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


def hssd_scenes(split="train", names=None):
    import hyperlux

    return _catalog_scenes(hyperlux.datasets.hssd.Val43() if split == "val" else hyperlux.datasets.hssd.Train125(), names)


CATALOGS = {"procthor-train": procthor_scenes, "hssd-train": lambda names=None: hssd_scenes("train", names)}  # the public scene catalogs


def scene_references(names, catalogs=CATALOGS):
    """{scene name: conta reference} for scene names and/or catalog names (a catalog name stands for all of its scenes),
    in the given order; the catalogs are searched in order and loaded only as needed."""
    names = [str(name) for name in names]
    if not names:
        raise ValueError("no scenes given")
    loaded = {}
    resolved = {}
    for name in names:
        if name in catalogs:
            loaded.setdefault(name, catalogs[name]())
            resolved.update(loaded[name])
            continue
        for catalog, load in catalogs.items():
            if catalog not in loaded:
                loaded[catalog] = load()
            if name in loaded[catalog]:
                resolved[name] = loaded[catalog][name]
                break
        else:
            raise ValueError(f"unknown scene {name}: not in the catalogs {list(catalogs)} (scene names look like ProcTHOR-Train-7; a catalog name selects all of its scenes)")
    return resolved


def nominal_camera(rng, imu_rate_hz):
    """The default camera of simulate(): each frame is the render at its time, quantised to 8 bits.

    A camera is called once per flight and returns (sub_frames, develop, meta): develop(renders, rng) turns the last
    `sub_frames` renders up to a frame ((sub_frames, H, W, 3) sRGB float32 in [0, 1], one per IMU step, oldest first)
    into that (H, W, 3) uint8 frame, and meta goes into the episode's metadata.
    """
    return 1, quantise, {}


def quantise(renders, rng):
    return np.clip(np.rint(renders[-1] * 255.0), 0, 255).astype(np.uint8)


# a user-spec World exposes unnamed observation blocks: privileged = [position 3 | rotation matrix 9 | linear velocity 3 |
# angular velocity 3 | waypoint 3], IMU = [accelerometer 3 | gyroscope 3 | frame_age | new_frame]
BLOCKS = {"position": (0, 3), "orientation_rotation_matrix": (3, 9), "linear_velocity": (12, 3), "angular_velocity": (15, 3),
          "waypoint": (18, 3), "accelerometer": (0, 3), "gyroscope": (3, 3)}


class Spool:
    """A temporary directory for one batch's frames, removed when the last reference to it is gone (weakref.finalize).

    Developed frames go to disk as PNGs instead of RAM (a batch of 8 one-minute episodes at 640 x 480 would be ~9 GB).
    The yielded Episodes' FrameSequences hold the Spool, so `viobench simulate` (writes an episode, moves on) and
    a library caller that keeps one episode and drops the generator both see their frames for as long as they need them.
    """

    def __init__(self, root=None, prefix="viobench-sim-"):
        root = root or os.environ.get(SPOOL_VARIABLE) or None
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
        self._finalizer = weakref.finalize(self, shutil.rmtree, str(self.path), ignore_errors=True)

    def remove(self):
        self._finalizer()


def spooled_frames(paths, frame_shape, spool):
    """FrameSequence over RGB PNGs in a Spool; the loader keeps the spool alive."""
    return FrameSequence(lambda index, _spool=spool: read_png(paths[index], abi.CHANNELS), len(paths), frame_shape,
                         encoded=lambda index, _spool=spool: Path(paths[index]).read_bytes())


def _block(vector, layout, name):
    offset, size = layout.blocks.get(name, BLOCKS[name])
    return vector[:, offset:offset + size]


def simulate(sim, scenes, episodes, seed=0, camera=nominal_camera, name="sim", log=print):
    """Yields `episodes` Episodes per scene, scene by scene: rounds of `sim.instances` parallel flights (consecutive seeds
    from `seed`) until the scene has its episodes. Flights that crash (terminated by the simulator) are dropped, surplus
    flights of the last round are discarded; a scene that has not delivered after MAX_FLIGHTS_PER_EPISODE flights per
    episode raises RuntimeError, after yielding what it had.

    scenes: {name: reference} (scene_references). camera: how renders become frames (see nominal_camera).
    name: the episodes' dataset and the prefix of their names, `{name}-{scene}-{index:04d}` by order within the scene;
    the round's seed and the flight's slot in it are rendering details and stay in the episode's meta.
    """
    import hyperdrone

    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes per scene must be positive")
    header = sim.render_header(200.0)
    seed = int(seed)
    env = None
    try:
        for scene_name, reference in scenes.items():
            if env is not None:
                env.close()
            # A custom spec_header exposes indexed camera names, regardless of the preset's named mounts.
            env = hyperdrone.HyperDrone([reference], config=hyperdrone.Config(num_worlds=1, spec_header=str(header)), seed=seed,
                                       parameters={"cameras": {"camera0": {"fov": sim.fov}}})
            imu_rate = float(round(1.0 / float(env.dt)))
            if header.name != sim.render_header(imu_rate).name:  # the preset's IMU rate differs from the assumed 200 Hz
                raise RuntimeError(f"IMU rate {imu_rate} Hz: re-render the spec header for it")
            written = flown = crashed = 0
            while written < episodes:
                if flown >= MAX_FLIGHTS_PER_EPISODE * episodes:
                    raise RuntimeError(f"{scene_name}: {written} of {episodes} episodes after {flown} flights ({crashed} crashed)")
                landed = 0
                for episode in _run_batch(env, sim, seed, scene_name, camera, name, log, progress=(written, episodes)):
                    landed += 1
                    if written < episodes:
                        episode.name = episode.config["episode"]["name"] = f"{name}-{scene_name}-{written:04d}"
                        written += 1
                        yield episode
                flown += sim.instances
                crashed += sim.instances - landed
                seed += 1
    finally:
        if env is not None:
            env.close()

def _run_batch(env, sim, seed, scene_name, camera, name, log, progress=(0, None)):
    """One round of `env.total_instances` parallel flights; yields the ones that did not crash (simulate() names them).
    progress: (episodes of the scene before this round, episodes wanted) for the log line."""
    import time

    import cv2  # not at the top: the CLI loads this module for its arguments, without viobench[data]

    imu_rate = float(round(1.0 / float(env.dt)))  # env.dt is float32 (0.00499999989)
    dt = 1.0 / imu_rate
    steps = sim.step_count(imu_rate)  # not env.episode_step_limit: the preset's 10 s limit is informational, nothing ends a flight at it
    frame_every = int(round(1.0 / (sim.frame_rate_hz * dt)))
    total = env.total_instances
    layout = env.observation_layout_privileged
    layout_imu = env.observation_layout_imu
    rng = np.random.default_rng([seed, abs(hash(name)) % (2**32)])
    cameras = [camera(rng, imu_rate) for _ in range(total)]  # (sub_frames, develop, meta) per flight
    sub_frames = max(k for k, _, _ in cameras)  # ring buffer length

    env.reset(np.ones(total, dtype=bool))
    parameters = env.instance_parameters()
    spawns = env.spawn_points()
    state = env.observe_privileged()
    ground_truth = [np.column_stack([np.zeros(total), _block(state, layout, "position"),
                                     matrix_to_quaternion(_block(state, layout, "orientation_rotation_matrix").reshape(-1, 3, 3))])]
    imu_rows = []
    spool = Spool(sim.spool, prefix=f"viobench-sim-{seed:05d}-")
    frame_paths = [[] for _ in range(total)]
    frame_times = []
    sub_frames = min(sub_frames, frame_every)  # a shutter longer than the frame interval would overlap the previous frame
    ring = deque(maxlen=sub_frames)
    terminated = np.zeros(total, dtype=bool)
    waypoint = _block(state, layout, "waypoint").copy()
    waypoints_reached = np.zeros(total, dtype=int)  # diagnostic: how far along the route each flight got (calibrates WAYPOINTS_PER_SECOND)
    instance_rngs = [np.random.default_rng(rng.integers(2**63)) for _ in range(total)]
    pool = ThreadPoolExecutor(max_workers=min(total, os.cpu_count() or 1))  # numpy and OpenCV release the GIL on large arrays
    for instance in range(total):
        (spool.path / f"{instance:02d}").mkdir()

    def develop(instance, index):
        """One instance's frame through its camera, written straight to the spool (PNG, BGR like storage.write_episode); returns the path."""
        k, develop_frame, _ = cameras[instance]
        frame = develop_frame(np.stack([renders[instance] for renders in list(ring)[-k:]]), instance_rngs[instance])
        path = spool.path / f"{instance:02d}" / f"{index:05d}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
            raise OSError(f"could not write {path}")
        return path

    started = time.perf_counter()
    for step in range(steps):
        # only renders inside the longest exposure before a frame's closing step are kept: (total, H, W, 3) sRGB
        # float32 in [0, 1], camera at the pre-step state (time step*dt), copied in case the renderer reuses its buffer
        if step % frame_every >= frame_every - sub_frames:
            ring.append(np.array(env.frames()))
        if step % frame_every == frame_every - 1:
            index = len(frame_times)
            for instance, path in enumerate(pool.map(lambda i: develop(i, index), range(total))):
                frame_paths[instance].append(path)
            frame_times.append(step * dt)  # provisional: the per-instance mid-exposure correction is applied below
        env.step()
        sample = env.observe_imu()  # the sample produced by this step, stamped at its end
        imu_rows.append(np.column_stack([np.full(total, (step + 1) * dt), _block(sample, layout_imu, "accelerometer"), _block(sample, layout_imu, "gyroscope")]))
        state = env.observe_privileged()
        ground_truth.append(np.column_stack([np.full(total, (step + 1) * dt), _block(state, layout, "position"),
                                             matrix_to_quaternion(_block(state, layout, "orientation_rotation_matrix").reshape(-1, 3, 3))]))
        terminated |= env.terminated()
        current = _block(state, layout, "waypoint")
        waypoints_reached += np.any(current != waypoint, axis=1)
        waypoint = current.copy()
    elapsed = time.perf_counter() - started
    pool.shutdown()
    fovs = [float((p["cameras"].get("front") or next(iter(p["cameras"].values())))["fov"]) for p in parameters]
    crashed = int(terminated.sum())
    before, wanted = progress
    last = min(before + total - crashed, before + total if wanted is None else wanted) - 1
    span = (f"episodes {before}-{last}" if last >= before else "no episode") + ("" if wanted is None else f" of {wanted}")
    log(f"{scene_name} {span}: {steps} steps ({sim.duration_seconds:g} s) x {total} flights in {elapsed:.1f}s ({steps * total / elapsed:.0f} steps/s), "
        f"{crashed} crashed, waypoints reached {waypoints_reached.tolist()} of {sim.num_waypoints}, fov {np.round(fovs, 1).tolist()}")

    imu_rows = np.stack(imu_rows)  # (steps, total, 7)
    ground_truth = np.stack(ground_truth)  # (steps + 1, total, 8)
    frame_times = np.asarray(frame_times)
    frame_shape = (sim.height, sim.width, abi.CHANNELS)
    for instance in range(total):
        if terminated[instance]:
            continue
        k, _, camera_meta = cameras[instance]
        mounted = parameters[instance]["cameras"]  # HyperDrone's camera descriptions
        front = mounted.get("front") or next(iter(mounted.values()))
        fov = float(front["fov"])
        T_body_camera = mount_to_body_camera(front["mount"])
        intrinsics = pinhole_intrinsics(sim.width, sim.height, fov)
        config = make_config(width=sim.width, height=sim.height, channels=abi.CHANNELS, frame_rate_hz=sim.frame_rate_hz,
                             num_frames=len(frame_times), hover_seconds=sim.hold_seconds, T_body_camera=T_body_camera,
                             intrinsics=intrinsics, episode_name=f"{name}-{scene_name}-{seed:05d}-{instance:02d}")  # provisional: simulate() names by order
        config["camera"]["horizontal_fov_degrees"] = fov
        config["episode"]["exposure_seconds"] = k * dt
        # frame k integrates the renders at steps c-k+1..c (c = closing step): mid-exposure is (c - (k-1)/2) dt
        times = frame_times - (k - 1) / 2.0 * dt
        episode = Episode(
            name=config["episode"]["name"],
            config=config,
            frame_times=times,
            frames=spooled_frames(frame_paths[instance], frame_shape, spool),
            imu=imu_rows[:, instance, :],
            ground_truth=ground_truth[:, instance, :],
            T_body_camera=T_body_camera,
            meta={"dataset": name, "source": "hyperdrone", "scene": scene_name, "seed": seed, "instance": instance, "sub_frames": k,
                  **camera_meta, "spawn": spawns[instance], "hyperdrone": _versions(), "sim": asdict(sim)},
        )
        yield regularize(episode)  # frame 0's window starts before the first IMU sample


def _versions():
    try:
        from importlib.metadata import version

        return {"hyperdrone": version("hyperdrone"), "hyperlux": version("hyperlux")}
    except Exception:  # noqa: BLE001
        return {}


# --- command-line plumbing of `viobench simulate`

def add_simulation_arguments(parser):
    """Adds the generator's options to an argparse parser: what to fly, how long, and how to film it."""
    parser.add_argument("--scene-names", nargs="+", default=DEFAULT_SCENE_NAMES, metavar="SCENE",
                        help="scenes to fly; a catalog name (procthor-train, hssd-train) stands for all of its scenes (default: ProcTHOR-Train-1)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="episodes per scene (default: 8)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds of flight per episode (frames = duration x frame rate)")
    parser.add_argument("--waypoints", type=int, default=DEFAULT_WAYPOINTS,
                        help="route length (default: 48; 0 scales it with --duration, 0.8 per second)")
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV_DEGREES, help="base horizontal FOV in degrees (default: 120)")
    parser.add_argument("--fov-range", type=float, default=15.0, help="horizontal FOV = --fov + U(-range, range) degrees (default: 15; 0 fixes the FOV)")
    parser.add_argument("--hover", type=float, default=1.0, help="seconds of hover at the start of each episode")
    parser.add_argument("--spool", help="directory for the frames while they render (default: $VIOBENCH_SPOOL or the system temp dir)")


def dataset_from_arguments(arguments, camera=nominal_camera, name="sim"):
    """The Dataset provider configured from add_simulation_arguments() options."""
    sim = SimConfig(instances=min(PARALLEL_FLIGHTS, int(arguments.episodes)), duration_seconds=arguments.duration, waypoints=arguments.waypoints,
                    fov_range=arguments.fov_range, hold_seconds=arguments.hover, spool=arguments.spool, fov=arguments.fov)
    return simulate(sim, scene_references(arguments.scene_names), arguments.episodes, seed=arguments.seed, camera=camera, name=name)
