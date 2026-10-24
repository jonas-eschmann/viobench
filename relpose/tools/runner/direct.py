"""`relpose debug`: run a Python solution directory in-process — no sandbox — on episodes, for debuggers and plots.

The solution is imported and driven exactly like the runtime drives it (`Solution(config)` then `step(t, frame, imu)`
per frame, inputs adapted to the channels/resolution of its `solution.json`), so a `breakpoint()` in `step` stops
under pdb or VS Code and the returned poses can be plotted. Same metrics and `estimate.tum` as `relpose evaluate`,
but no fuel and the host's numpy/OpenCV: `relpose check` remains the authority on what the sandbox accepts.

Every example ends with
    if __name__ == "__main__":
        from relpose.tools.runner.direct import main
        main(Solution)
so `python solution.py <episode dirs>` debugs it directly.
"""
import argparse
import importlib
import inspect
import itertools
import json
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import abi
from ..abi.manifest import parse_resolution, validate_manifest
from .metrics import evaluate_trajectory

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
    """A solution directory loaded on the host: its manifest (channels, resolution, entry) and a factory for instances."""

    def __init__(self, directory, entry=None, solution_class=None):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / abi.BUNDLE_MANIFEST
        self.manifest = validate_manifest(json.loads(manifest_path.read_text()) if manifest_path.is_file() else {})
        self.channels = int(self.manifest.get("channels", abi.DEFAULT_CHANNELS))
        self.resolution = tuple(parse_resolution(self.manifest.get("resolution", abi.DEFAULT_RESOLUTION)))
        install_solution_bundle(self.directory)
        if str(self.directory) not in sys.path:
            sys.path.insert(0, str(self.directory))
        if solution_class is not None:
            self.factory = solution_class
        else:
            module_name, _, class_name = (entry or self.manifest.get("entry", abi.DEFAULT_ENTRY)).partition(":")
            cached = sys.modules.get(module_name)
            if cached is not None and not str(getattr(cached, "__file__", "") or "").startswith(str(self.directory)):
                del sys.modules[module_name]  # a same-named module from another solution directory (one process, several solutions)
            self.factory = getattr(importlib.import_module(module_name), class_name or "Solution")

    @classmethod
    def from_class(cls, solution_class):
        """For the examples' `main(Solution)` footer: the directory is the one holding the class's source file."""
        return cls(Path(inspect.getfile(solution_class)).resolve().parent, solution_class=solution_class)

    def create(self, config):
        return self.factory(config)


@dataclass
class Step:
    index: int
    t: float
    frame: np.ndarray
    imu: np.ndarray
    pose: np.ndarray | None
    seconds: float


def trace(solution, episode):
    """Drives one instance over one episode; yields a Step per frame (the hook for your own loops and plots)."""
    config, frames = episode.adapt(solution.channels, solution.resolution)
    estimator = solution.create(config)
    for index, (t, frame, imu) in enumerate(episode.steps(frames)):
        started = time.perf_counter()
        pose = estimator.step(float(t), frame, imu)
        seconds = time.perf_counter() - started
        if pose is not None:
            pose = np.asarray(pose, dtype=float).reshape(abi.POSE_VALUES)
        yield Step(index, float(t), frame, imu, pose, seconds)


def run(solution, dataset, output_dir=None, episode_limit=None, log=print):
    """Runs the solution over a Dataset; returns [(episode, estimates (N, 7) with NaN for missing, TrajectoryMetrics)]."""
    from .run import as_dataset

    results = []
    for count, episode in enumerate(as_dataset(dataset)):
        if episode_limit is not None and count >= episode_limit:
            break
        estimates = np.full((episode.num_frames, abi.POSE_VALUES), np.nan)
        seconds = []
        for step in trace(solution, episode):
            if step.pose is not None:
                estimates[step.index] = step.pose
            seconds.append(step.seconds)
        metrics = evaluate_trajectory(episode.ground_truth_relative(), estimates, episode.frame_times)
        if output_dir is not None:
            directory = Path(output_dir) / episode.dataset / episode.name
            directory.mkdir(parents=True, exist_ok=True)
            np.savetxt(directory / "estimate.tum", np.column_stack([episode.frame_times, estimates]), fmt="%.9g",
                       header="t x y z qx qy qz qw (camera k in camera 0 frame)", comments="# ")
        log(f"{episode.dataset}/{episode.name}: ATE {metrics.ate_rmse:.3f} m  RPE {metrics.rpe_translation_rmse:.3f} m/s  "
            f"rot {np.degrees(metrics.rotation_rmse):.1f} deg  coverage {metrics.coverage:.2f}  "
            f"{np.median(seconds) * 1e3:.1f} ms/frame (host, no fuel)")
        results.append((episode, estimates, metrics))
    if not results:
        raise ValueError("the dataset yielded no episodes")
    return results


def add_arguments(parser, solution_class=None):
    """The `relpose debug` options; without a solution_class the solution directory is the first positional."""
    from ..datasets.generate.simulation import add_simulation_arguments

    if solution_class is None:
        parser.add_argument("solution", help="solution directory (with solution.py / solution.json)")
    parser.add_argument("datasets", nargs="*", help="episode directories or directories of episodes")
    parser.add_argument("--simulate", type=int, metavar="N", help="also run on N fresh HyperDrone episodes (GPU; options below)")
    parser.add_argument("--episode-limit", type=int, help="stop after this many episodes")
    parser.add_argument("-o", "--output", help="write estimate.tum per episode under this directory")
    parser.add_argument("--entry", help="module:Class (default: solution.json or solution:Solution)")
    add_simulation_arguments(parser)


def run_arguments(arguments, solution_class=None):
    from ..datasets.storage import episodes

    if not arguments.datasets and not arguments.simulate:
        raise SystemExit("debug needs episode directories and/or --simulate N")
    solution = PythonSolution.from_class(solution_class) if solution_class is not None else PythonSolution(arguments.solution, entry=arguments.entry)

    def dataset():
        if arguments.datasets:
            yield from episodes(arguments.datasets)
        if arguments.simulate:
            from ..datasets.generate.simulation import dataset_from_arguments

            yield from itertools.islice(dataset_from_arguments(arguments), arguments.simulate)

    run(solution, dataset(), output_dir=arguments.output, episode_limit=arguments.episode_limit)
    return 0


def main(solution_class=None, argv=None):
    """`python solution.py <episodes...>` (the examples' footer passes their class, so the target is implied)."""
    parser = argparse.ArgumentParser(prog="relpose debug" if solution_class is None else Path(sys.argv[0]).name,
                                     description="run a Python solution in this process (no sandbox) on episodes: breakpoints, prints and plots work")
    add_arguments(parser, solution_class)
    return run_arguments(parser.parse_args(argv), solution_class)
