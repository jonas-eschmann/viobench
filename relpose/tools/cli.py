"""The `relpose` command line (also `python -m relpose`).

solution     bundle, unbundle, manifest        Python runtime + directory <-> solution.wasm; declare RGB / VGA (any module)
runner       check, evaluate, leaderboard      run a module on synthetic frames / on a dataset (disk or freshly simulated); aggregate reports
datasets     simulate, convert-euroc           write episodes with HyperDrone (GPU); cut EuRoC sequences into episodes

The CLI is the one host component that also reaches into the guest side (for `bundle`); the runner library never does.
"""
import argparse
import itertools
import sys
import threading

from . import abi

THREAD_STACK_BYTES = 64 << 20  # guest frames run on the calling thread's native stack: give it room


def run_with_large_stack(function):
    """Runs function on a thread with a 64 MiB stack and re-raises its exception or returns its result."""
    outcome = {}

    def target():
        try:
            outcome["result"] = function()
        except BaseException as error:  # noqa: BLE001 - propagated below
            outcome["error"] = error

    threading.stack_size(THREAD_STACK_BYTES)
    thread = threading.Thread(target=target, name="relpose-guest")
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def simulated_dataset(arguments):
    """The ad-hoc Dataset provider from CLI arguments (--simulate N)."""
    from .datasets.generate.simulation import SimConfig, hssd_scenes, procthor_scenes, simulate

    sim = SimConfig(width=arguments.width, height=arguments.height, instances=arguments.instances, frame_rate_hz=arguments.frame_rate,
                    fov_range=arguments.fov_range, hold_seconds=arguments.hover, aa_grid=arguments.aa_grid, channels=arguments.channels)
    scenes = {"procthor-train": lambda: procthor_scenes(arguments.scene_names), "hssd-val": lambda: hssd_scenes("val", arguments.scene_names),
              "hssd-train": lambda: hssd_scenes("train", arguments.scene_names)}[arguments.scenes]()
    seeds = range(arguments.seed, arguments.seed + arguments.batches)
    return simulate(sim, seeds, scenes=scenes, condition=arguments.condition)


def add_simulation_arguments(parser, prefix=""):
    parser.add_argument(f"--{prefix}seed", dest="seed", type=int, default=0, help="first seed; seeds also pick the scene round-robin")
    parser.add_argument(f"--{prefix}batches", dest="batches", type=int, default=1_000_000, help="maximum number of seeds/batches to run")
    parser.add_argument("--condition", default="default", choices=["default", "blur", "lowlight", "oversaturated"])
    parser.add_argument("--scenes", default="procthor-train", choices=["procthor-train", "hssd-train", "hssd-val"])
    parser.add_argument("--scene-names", nargs="*", help="restrict to these scene names")
    parser.add_argument("--width", type=int, default=640, help="rendered resolution (the runner resamples to what a solution asks for)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--instances", type=int, default=8, help="flights per batch")
    parser.add_argument("--frame-rate", type=float, default=20.0)
    parser.add_argument("--fov-range", type=float, default=15.0, help="horizontal FOV = 63.8 + U(-range, range) degrees")
    parser.add_argument("--hover", type=float, default=1.0, help="seconds of hover at the start of each episode")
    parser.add_argument("--aa-grid", type=int, default=1)
    parser.add_argument("--channels", type=int, default=3, choices=abi.SUPPORTED_CHANNELS, help="render RGB (default) or grayscale")


# --- solution (guest side) -------------------------------------------------------------------------------------

def command_bundle(arguments):
    from ..solution.runtime.python.bundler import BundleError
    from ..solution.runtime.python.bundler.bundle import bundle

    try:
        summary = bundle(arguments.directory, arguments.output, runtime=arguments.runtime, entry=arguments.entry, channels=arguments.channels,
                         resolution=arguments.resolution)
    except (BundleError, abi.AbiError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {summary['output']} ({summary['bytes'] / 2**20:.1f} MiB, bundle {summary['bundle_bytes'] / 1024:.0f} KiB, {len(summary['files'])} files, "
          f"entry {summary['entry']}, {'RGB' if summary['channels'] == 3 else 'grayscale'} {summary['resolution'][0]}x{summary['resolution'][1]} frames)")
    for warning in summary["warnings"]:
        print(f"warning: {warning}")
    print(f"next: relpose check {summary['output']}")
    return 0


def command_unbundle(arguments):
    from ..solution.runtime.python.bundler import BundleError
    from ..solution.runtime.python.bundler.bundle import unbundle

    try:
        names = unbundle(arguments.wasm, arguments.output)
    except (BundleError, abi.AbiError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"extracted {len(names)} files to {arguments.output}")
    return 0


def command_manifest(arguments):
    from .abi.manifest import read_manifest, write_manifest

    try:
        updates = {}
        if arguments.channels is not None:
            updates["channels"] = arguments.channels
        if arguments.entry is not None:
            updates["entry"] = arguments.entry
        if arguments.resolution is not None:
            updates["resolution"] = arguments.resolution
        manifest = write_manifest(arguments.wasm, updates) if updates else read_manifest(open(arguments.wasm, "rb").read())
    except abi.AbiError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(manifest)
    return 0


# --- runner ----------------------------------------------------------------------------------------------------

def command_check(arguments):
    from .runner.check import check

    return run_with_large_stack(lambda: check(arguments.wasm, dev_dir=arguments.dir, frames=arguments.frames, channels=arguments.channels,
                                              width=arguments.width, height=arguments.height, logs=arguments.logs, verbose=arguments.verbose))


def command_evaluate(arguments):
    from .runner.run import evaluate
    from .runner.sandbox import Limits

    limits = Limits()
    if arguments.step_fuel:
        limits.step_fuel = arguments.step_fuel
    if arguments.memory_mib:
        limits.memory_bytes = arguments.memory_mib << 20
    if not arguments.datasets and not arguments.simulate:
        raise SystemExit("evaluate needs dataset directories and/or --simulate N")
    max_episodes = arguments.max_episodes

    def dataset():
        from .datasets.storage import episodes

        if arguments.datasets:
            yield from episodes(arguments.datasets)
        if arguments.simulate:
            yield from itertools.islice(simulated_dataset(arguments), arguments.simulate)

    def run():
        result = evaluate(arguments.wasm, dataset(), output_dir=arguments.output, limits=limits, dev_dir=arguments.dir, max_episodes=max_episodes)
        summary = result["summary"]["overall"]
        print(f"overall: {summary['completed']}/{summary['episodes']} episodes completed, ATE median {summary['ate_rmse_median']:.3f} m, "
              f"fuel/frame median {summary['fuel_per_frame_median']:,.0f}")
        return 0 if summary["completed"] == summary["episodes"] else 1

    return run_with_large_stack(run)


def command_leaderboard(arguments):
    from .runner.leaderboard import load_reports, markdown_table, plot, rows

    table = rows(load_reports(arguments.reports))
    if not table:
        print("no reports found", file=sys.stderr)
        return 1
    text = markdown_table(table)
    print(text)
    if arguments.output:
        from pathlib import Path

        Path(arguments.output).write_text(text)
    if arguments.plot:
        try:
            plot(table, arguments.plot)
            print(f"plot: {arguments.plot}")
        except ImportError:
            print("plot skipped: matplotlib is not installed", file=sys.stderr)
    return 0


# --- datasets --------------------------------------------------------------------------------------------------

def command_simulate(arguments):
    from pathlib import Path

    from .datasets.storage import write_episode

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for episode in itertools.islice(simulated_dataset(arguments), arguments.episodes):
        write_episode(episode, output / episode.name)
        written += 1
        print(f"wrote {output / episode.name} ({episode.num_frames} frames, {episode.duration:.1f}s, fov {episode.config['camera']['horizontal_fov_degrees']:.1f} deg)")
    print(f"{written} episodes in {output}")
    return 0 if written >= arguments.episodes else 1


def command_convert_euroc(arguments):
    from pathlib import Path

    from .datasets.generate.euroc import convert
    from .datasets.storage import write_episode

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for episode, origin_ns in convert(arguments.sequence, width=arguments.width, height=arguments.height, snippet_seconds=arguments.snippet,
                                      max_snippets=arguments.max_snippets, start_seconds=arguments.start, stride_seconds=arguments.stride):
        write_episode(episode, output / episode.name, origin_ns=origin_ns)
        count += 1
    print(f"{count} episodes in {output}")
    return 0 if count else 1


# --- parser ----------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="relpose", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    bundle = commands.add_parser("bundle", help="[solution] embed a directory of Python code into the Python runtime -> solution .wasm")
    bundle.add_argument("directory")
    bundle.add_argument("-o", "--output", required=True, help="output .wasm")
    bundle.add_argument("--runtime", help="runtime.wasm to build on (default: $RELPOSE_RUNTIME or build/runtime/dist/runtime.wasm)")
    bundle.add_argument("--entry", help=f"module:Class of the solution (default {abi.DEFAULT_ENTRY} or solution.json)")
    bundle.add_argument("--channels", type=int, choices=abi.SUPPORTED_CHANNELS, help="frames the solution wants: 1 grayscale (default) or 3 RGB (or solution.json)")
    bundle.add_argument("--resolution", help="frames the solution wants: 320x240 (default) or 640x480 (or solution.json)")
    bundle.set_defaults(function=command_bundle)

    unbundle = commands.add_parser("unbundle", help="[solution] extract the embedded directory from a solution .wasm")
    unbundle.add_argument("wasm")
    unbundle.add_argument("-o", "--output", required=True, help="output directory")
    unbundle.set_defaults(function=command_unbundle)

    manifest = commands.add_parser("manifest", help="[solution] show or change a module's manifest (native modules declare RGB / VGA this way)")
    manifest.add_argument("wasm")
    manifest.add_argument("--channels", type=int, choices=abi.SUPPORTED_CHANNELS)
    manifest.add_argument("--resolution", help="320x240 or 640x480")
    manifest.add_argument("--entry")
    manifest.set_defaults(function=command_manifest)

    check = commands.add_parser("check", help="[runner] instantiate a module and run a few synthetic frames through it")
    check.add_argument("wasm", help="solution .wasm (or a bare runtime.wasm together with --dir)")
    check.add_argument("--dir", help="dev mode: mount this solution directory instead of a bundle")
    check.add_argument("--frames", type=int, default=5)
    check.add_argument("--channels", type=int, choices=abi.SUPPORTED_CHANNELS, help="override the manifest's channel choice")
    check.add_argument("--width", type=int, help="override the manifest's resolution")
    check.add_argument("--height", type=int)
    check.add_argument("--logs", help="directory for stdout.txt/stderr.txt (default: temporary)")
    check.add_argument("-v", "--verbose", action="store_true", help="pass the guest's stdout/stderr through instead of capturing them")
    check.set_defaults(function=command_check)

    evaluate = commands.add_parser("evaluate", help="[runner] run a module over episodes (directories and/or freshly simulated) and write report.json")
    evaluate.add_argument("wasm")
    evaluate.add_argument("datasets", nargs="*", help="episode directories or directories of episodes")
    evaluate.add_argument("-o", "--output", help="report directory (default: ./report-<name>)")
    evaluate.add_argument("--dir", help="dev mode: mount this solution directory instead of a bundle")
    evaluate.add_argument("--max-episodes", type=int)
    evaluate.add_argument("--step-fuel", type=int, help="fuel budget per step (default from Limits)")
    evaluate.add_argument("--memory-mib", type=int, help="guest memory cap in MiB")
    evaluate.add_argument("--simulate", type=int, metavar="N", help="also evaluate on N fresh HyperDrone episodes (GPU; options below)")
    add_simulation_arguments(evaluate)
    evaluate.set_defaults(function=command_evaluate)

    leaderboard = commands.add_parser("leaderboard", help="[runner] aggregate report.json files into the ATE-vs-fuel table (and plot)")
    leaderboard.add_argument("reports", nargs="+", help="report.json files or directories containing them")
    leaderboard.add_argument("-o", "--output", help="write the markdown table here")
    leaderboard.add_argument("--plot", help="write an ATE-vs-fuel PNG here (needs matplotlib)")
    leaderboard.set_defaults(function=command_leaderboard)

    simulate = commands.add_parser("simulate", help="[datasets] write episodes generated with HyperDrone (needs a GPU and pip install relpose[sim])")
    simulate.add_argument("-o", "--output", required=True, help="directory that receives one sub-directory per episode")
    simulate.add_argument("--episodes", type=int, default=8, help="episodes to write (a batch of --instances flights per seed)")
    add_simulation_arguments(simulate)
    simulate.set_defaults(function=command_simulate)

    convert = commands.add_parser("convert-euroc", help="[datasets] cut a EuRoC MAV sequence (ASL format directory) into episodes")
    convert.add_argument("sequence", help="sequence directory (containing mav0/)")
    convert.add_argument("-o", "--output", required=True)
    convert.add_argument("--snippet", type=float, default=10.0, help="episode length in seconds")
    convert.add_argument("--stride", type=float, help="seconds between snippet starts (default: --snippet, no overlap)")
    convert.add_argument("--start", type=float, default=0.0, help="skip this many seconds at the beginning")
    convert.add_argument("--max-snippets", type=int)
    convert.add_argument("--width", type=int, default=640, help="EuRoC's 752x480 crops to 640x480 without scaling")
    convert.add_argument("--height", type=int, default=480)
    convert.set_defaults(function=command_convert_euroc)

    arguments = parser.parse_args(argv)
    return arguments.function(arguments)


if __name__ == "__main__":
    sys.exit(main())
