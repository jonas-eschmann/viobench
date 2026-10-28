"""`relpose debug`: run a Python solution directory in-process — no sandbox — on episodes, for debuggers and plots.

The solution is imported and driven exactly like the runtime drives it (`Solution(config)` then `step(t, frame, imu)`
per frame, inputs adapted to its `Solution.frame_width`), so a `breakpoint()` in `step` stops
under pdb or VS Code and the returned states can be plotted. Same metrics and estimate files as `relpose evaluate`,
but no fuel and the host's numpy/OpenCV: `relpose check` remains the authority on what the sandbox accepts.
"""
import importlib
import itertools
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import abi
from .metrics import evaluate_trajectory, valid_states
from .score import write_estimates

BUNDLE_MODULE = "solution_bundle"


class HostBundle:
    """What `import solution_bundle` gives a solution inside the runtime, backed by its directory on the host."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.names = frozenset(str(path.relative_to(self.directory)).replace(os.sep, "/") for path in self.directory.rglob("*")
                               if path.is_file() and "__pycache__" not in path.parts)

    def exists(self, name):
        return name in self.names

    def read_bytes(self, name):
        try:
            return (self.directory / name).read_bytes()
        except FileNotFoundError:
            raise FileNotFoundError(f"{name!r} is not in the solution directory {self.directory}") from None

    def read_text(self, name, encoding="utf-8"):
        return self.read_bytes(name).decode(encoding)

    def open(self, name):
        import io

        return io.BytesIO(self.read_bytes(name))

    def listdir(self, prefix=""):
        prefix = prefix.strip("/")
        prefix = prefix + "/" if prefix else ""
        return sorted(name[len(prefix):] for name in self.names if name.startswith(prefix))


def install_solution_bundle(directory):
    bundle = HostBundle(directory)
    module = types.ModuleType(BUNDLE_MODULE, HostBundle.__doc__)
    for attribute in ("exists", "read_bytes", "read_text", "open", "listdir"):
        setattr(module, attribute, getattr(bundle, attribute))
    module.names = bundle.names
    module.bundle = bundle
    sys.modules[BUNDLE_MODULE] = module
    return module


class PythonSolution:
    """A solution directory loaded on the host: a factory for instances and the resolution its frame_width declares."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        install_solution_bundle(self.directory)
        if str(self.directory) not in sys.path:
            sys.path.insert(0, str(self.directory))
        cached = sys.modules.get(abi.SOLUTION_MODULE)
        if cached is not None and not str(getattr(cached, "__file__", "") or "").startswith(str(self.directory)):
            del sys.modules[abi.SOLUTION_MODULE]  # a same-named module from another solution directory (one process, several solutions)
        self.factory = getattr(importlib.import_module(abi.SOLUTION_MODULE), abi.SOLUTION_CLASS)
        self.resolution = abi.resolution(getattr(self.factory, abi.FRAME_WIDTH, None))

    def create(self, config):
        return self.factory(config)


@dataclass
class Step:
    index: int
    t: float
    frame: np.ndarray
    imu: np.ndarray
    pose: np.ndarray
    twist: np.ndarray
    seconds: float


def trace(solution, episode):
    """Drives one instance over one episode; yields a Step per frame (the hook for your own loops and plots)."""
    config, frames = episode.adapt(solution.resolution)
    estimator = solution.create(config)
    for index, (t, frame, imu) in enumerate(episode.steps(frames)):
        started = time.perf_counter()
        state = estimator.step(float(t), frame, imu)
        seconds = time.perf_counter() - started
        if state is None:
            raise ValueError("step() must return 13 values [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz], got None "
                             "(return the identity pose and zero twist until you have an estimate)")
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.shape != (abi.STATE_VALUES,):
            raise ValueError(f"step() must return 13 values [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz], got shape {state.shape}")
        if not valid_states(state[None])[0]:
            raise ValueError(f"step() returned an unusable state {state.tolist()} (NaN/inf or an all-zero quaternion)")
        yield Step(index, float(t), frame, imu, state[:abi.POSE_VALUES], state[abi.POSE_VALUES:], seconds)


def run(solution, dataset, output_dir=None, max_episodes=None, log=print):
    """Runs the solution over a Dataset; returns [(episode, estimates (N, 13), TrajectoryMetrics)]. A failing step raises."""
    from .run import as_dataset

    results = []
    for count, episode in enumerate(as_dataset(dataset)):
        if max_episodes is not None and count >= max_episodes:
            break
        estimates = np.empty((episode.num_frames, abi.STATE_VALUES))
        seconds = []
        for step in trace(solution, episode):
            estimates[step.index] = np.concatenate([step.pose, step.twist])
            seconds.append(step.seconds)
        metrics = evaluate_trajectory(episode.ground_truth_state(), estimates, episode.frame_times)
        if output_dir is not None:
            write_estimates(Path(output_dir) / episode.dataset / episode.name, episode.frame_times, estimates)
        log(f"{episode.dataset}/{episode.name}: RPE {metrics.rpe_translation_rmse:.3f} m/1 s  ATE {metrics.ate_rmse:.3f} m  "
            f"rot {np.degrees(metrics.rotation_rmse):.1f} deg  vel {metrics.velocity_rmse:.3f} m/s  held {metrics.held}  "
            f"{np.median(seconds) * 1e3:.1f} ms/frame (host, no fuel)")
        results.append((episode, estimates, metrics))
    if not results:
        raise ValueError("the dataset yielded no episodes")
    return results


def add_arguments(parser):
    """The `relpose debug` options."""
    from ..datasets.generate.simulation import add_simulation_arguments

    parser.add_argument("solution", help="solution directory (with solution.py)")
    parser.add_argument("datasets", nargs="*", help="episode directories or directories of episodes")
    parser.add_argument("--simulate", type=int, metavar="N", help="also run on N fresh HyperDrone episodes (GPU; options below)")
    parser.add_argument("--max-episodes", type=int, help="stop after this many episodes")
    parser.add_argument("-o", "--output", help="write estimate.tum and twist.txt per episode under this directory")
    add_simulation_arguments(parser)


def run_arguments(arguments):
    from ..datasets.storage import episodes

    if not arguments.datasets and not arguments.simulate:
        raise SystemExit("debug needs episode directories and/or --simulate N")
    solution = PythonSolution(arguments.solution)

    def dataset():
        if arguments.datasets:
            yield from episodes(arguments.datasets)
        if arguments.simulate:
            from ..datasets.generate.simulation import dataset_from_arguments

            yield from itertools.islice(dataset_from_arguments(arguments), arguments.simulate)

    run(solution, dataset(), output_dir=arguments.output, max_episodes=arguments.max_episodes)
    return 0
