"""The `viobench` command line (also `python -m viobench`).

solution     init                              new workspace: an example solution with its own toolchain
             bundle, unbundle                  solution directory (Python, or native with a CMakeLists.txt) <-> solution.wasm
             build-runtime                     (re)build the workspace's Python runtime; bundle does when it is missing
runner       check, evaluate, leaderboard      run a module on synthetic frames / on a dataset; aggregate reports
             debug, score                      run a Python solution in-process (debugger-friendly); score an estimate
datasets     simulate, pack, feed              write episodes with HyperDrone (GPU); re-pack episodes as archives; export raw inputs for native debugging

The CLI is the one host component that also reaches into the guest side (for `bundle`, `init` and the builds); the
runner library never does.
"""
import argparse
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
    thread = threading.Thread(target=target, name="viobench-guest")
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


# --- solution (guest side) -------------------------------------------------------------------------------------

def command_init(arguments):
    from ..solution.workspace import WorkspaceError, init

    try:
        init(arguments.flavour, arguments.directory)
    except WorkspaceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"created {arguments.directory}: solution/ (yours), toolchain/ and runtime/ (the build recipe, yours to change)")
    print(f"next: cd {arguments.directory} && viobench bundle solution -o solution.wasm && viobench check solution.wasm")
    if arguments.flavour == "python":
        print("      (the first bundle builds the runtime into build/: 10-15 min)")
    return 0


def command_bundle(arguments):
    from pathlib import Path

    from ..solution.runtime.python.bundler import BundleError
    from ..solution.runtime.python.bundler.bundle import bundle
    from ..solution.workspace import WorkspaceError, build_native

    try:
        if (Path(arguments.directory) / "CMakeLists.txt").is_file():  # native: compiled with the workspace's wasi-sdk
            build_native(arguments.directory, arguments.output)
            print(f"wrote {arguments.output} ({Path(arguments.output).stat().st_size / 1024:.0f} KiB)")
        else:
            summary = bundle(arguments.directory, arguments.output, runtime=arguments.runtime)
            print(f"wrote {summary['output']} ({summary['bytes'] / 2**20:.1f} MiB, bundle {summary['bundle_bytes'] / 1024:.0f} KiB, {len(summary['files'])} files)")
            for warning in summary["warnings"]:
                print(f"warning: {warning}")
    except (BundleError, WorkspaceError, abi.AbiError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"next: viobench check {arguments.output}")
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


def command_build_runtime(arguments):
    from ..solution.workspace import WorkspaceError, build_runtime, find

    try:
        path = build_runtime(find(arguments.directory), force=arguments.force)
    except WorkspaceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"built {path}")
    return 0


# --- runner ----------------------------------------------------------------------------------------------------

def command_check(arguments):
    from .runner.check import check

    return run_with_large_stack(lambda: check(arguments.wasm, frames=arguments.frames, logs=arguments.logs,
                                              verbose=arguments.verbose))


def command_evaluate(arguments):
    from .runner.run import evaluate

    max_episodes = arguments.max_episodes

    def dataset():
        from .datasets.storage import episodes

        yield from episodes(arguments.datasets)

    def run():
        result = evaluate(arguments.wasm, dataset(), output_dir=arguments.output, max_episodes=max_episodes)
        summary = result["summary"]["overall"]
        print(f"overall: {summary['completed']}/{summary['episodes']} episodes completed, RPE median {summary['rpe_translation_rmse_median']:.3f} m/1 s, "
              f"fuel/frame max (all datasets) {summary['fuel_per_frame_max']:,.0f} "
              f"(mean {summary['fuel_per_frame_mean']:,.0f}, median {summary['fuel_per_frame_median']:,.0f}, "
              f"p95 {summary['fuel_per_frame_p95']:,.0f}, p99 {summary['fuel_per_frame_p99']:,.0f})")
        return 0 if summary["completed"] == summary["episodes"] else 1

    return run_with_large_stack(run)


def command_debug(arguments):
    from .runner.direct import run_arguments

    return run_arguments(arguments)  # plain Python on the main thread: breakpoint() and Ctrl-C behave


def command_score(arguments):
    from .runner.score import score

    score(arguments.episode, arguments.tum)
    return 0


def command_leaderboard(arguments):
    from .runner.leaderboard import load_reports, markdown_table, plot, rows

    try:
        table = rows(load_reports(arguments.reports))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
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
    import tempfile
    from pathlib import Path

    from .datasets.generate.simulation import dataset_from_arguments
    from .datasets.storage import write_episode

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with tempfile.TemporaryDirectory(prefix=".viobench-simulate-", dir=output) as scratch:  # the video before it goes into the archive
            for episode in dataset_from_arguments(arguments):
                attachments = {}
                if not arguments.no_video:
                    from .runner.video import write_video

                    attachments["trajectory.mp4"] = write_video(episode, None, Path(scratch) / f"{episode.name}.mp4")
                path = write_episode(episode, output / f"{episode.name}.zip", attachments=attachments)
                for video in attachments.values():
                    Path(video).unlink()
                written += 1
                print(f"wrote {path} ({episode.num_frames} frames, {episode.duration:.1f}s, fov {episode.config['camera']['horizontal_fov_degrees']:.1f} deg"
                      f"{', with trajectory.mp4' if attachments else ''})")
    except (ValueError, RuntimeError) as error:  # unknown scene, or a scene whose flights keep crashing
        print(f"error: {error}", file=sys.stderr)
        print(f"{written} episodes in {output}")
        return 1
    print(f"{written} episodes in {output}")
    return 0


def command_pack(arguments):
    """Writes episodes (archives or unzipped directories) as one archive each: converts sets made before the archive format."""
    from pathlib import Path

    from .datasets.storage import (
        episode_file,
        list_episodes,
        read_episode,
        write_episode,
    )

    output = Path(arguments.output)
    count = 0
    for root in arguments.episodes:
        sources = list_episodes(root)
        if not sources:
            print(f"error: no episodes under {root}", file=sys.stderr)
            return 1
        for source in sources:
            episode = read_episode(source)
            video = episode_file(source, "trajectory.mp4")
            path = write_episode(episode, output / f"{episode.name}.zip", attachments={"trajectory.mp4": video} if video else None)
            print(f"packed {source} -> {path}")
            count += 1
    print(f"{count} episodes in {output}")
    return 0


def command_feed(arguments):
    from .datasets.feed import write_feed
    from .datasets.storage import episodes

    episode = next(iter(episodes([arguments.episode])))
    resolution = tuple(int(v) for v in arguments.resolution.split("x"))
    directory = write_feed(episode, arguments.output, resolution=resolution)
    print(f"feed of {episode.name} ({episode.num_frames} frames at {resolution[0]}x{resolution[1]}) in {directory}")
    return 0


# --- parser ----------------------------------------------------------------------------------------------------

def main(argv=None):
    from ..solution.workspace import EXAMPLES
    from .datasets.generate.simulation import add_simulation_arguments
    from .runner.direct import add_arguments as add_debug_arguments

    parser = argparse.ArgumentParser(prog="viobench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    init = commands.add_parser("init", help="[solution] create a workspace: minimal Python/C++ with its own toolchain")
    init.add_argument("flavour", choices=list(EXAMPLES))
    init.add_argument("directory", help="new (or empty) directory")
    init.set_defaults(function=command_init)

    bundle = commands.add_parser("bundle", help="[solution] solution directory -> .wasm: Python code embedded into the Python runtime, "
                                                "native code (a CMakeLists.txt) compiled with wasi-sdk")
    bundle.add_argument("directory")
    bundle.add_argument("-o", "--output", required=True, help="output .wasm")
    bundle.add_argument("--runtime", help="Python: runtime.wasm to build on (default: $VIOBENCH_RUNTIME, else the workspace's "
                                          "build/runtime/dist/runtime.wasm, built first if missing)")
    bundle.set_defaults(function=command_bundle)

    unbundle = commands.add_parser("unbundle", help="[solution] extract the embedded directory from a solution .wasm")
    unbundle.add_argument("wasm")
    unbundle.add_argument("-o", "--output", required=True, help="output directory")
    unbundle.set_defaults(function=command_unbundle)

    build_runtime = commands.add_parser("build-runtime", help="[solution] build the workspace's Python runtime (runtime.wasm), e.g. after changing runtime/")
    build_runtime.add_argument("directory", nargs="?", default=".", help="a directory inside the workspace (default: the current one)")
    build_runtime.add_argument("--force", action="store_true", help="delete the previous build first instead of building incrementally")
    build_runtime.set_defaults(function=command_build_runtime)

    check = commands.add_parser("check", help="[runner] instantiate a module and run a few synthetic frames through it")
    check.add_argument("wasm", help="solution .wasm")
    check.add_argument("--frames", type=int, default=5)
    check.add_argument("--logs", help="directory for stdout.txt/stderr.txt (default: temporary)")
    check.add_argument("-v", "--verbose", action="store_true", help="pass the guest's stdout/stderr through instead of capturing them")
    check.set_defaults(function=command_check)

    evaluate = commands.add_parser("evaluate", help="[runner] run a module over episodes and write report.json")
    evaluate.add_argument("wasm")
    evaluate.add_argument("datasets", nargs="+", help="episodes (.zip or unzipped) or directories of them")
    evaluate.add_argument("-o", "--output", help="report directory; also write trajectory.mp4 per episode (default: ./report-<name>, without video)")
    evaluate.add_argument("--max-episodes", type=int)
    evaluate.set_defaults(function=command_evaluate)

    debug = commands.add_parser("debug", help="[runner] run a Python solution directory in this process (no sandbox): breakpoints, prints, plots")
    add_debug_arguments(debug)
    debug.set_defaults(function=command_debug)

    score = commands.add_parser("score", help="[runner] metrics of an estimate (estimate.tum + twist.txt next to it) against an episode")
    score.add_argument("episode", help="episode (.zip or unzipped directory)")
    score.add_argument("tum", help="estimate.tum (TUM: t x y z qx qy qz qw) written by a debug build or your own script; twist.txt (t vx vy vz wx wy wz) must sit next to it")
    score.set_defaults(function=command_score)

    leaderboard = commands.add_parser("leaderboard", help="[runner] aggregate report.json files into the RPE-vs-fuel table (and plot)")
    leaderboard.add_argument("reports", nargs="+", help="report.json files or directories containing them")
    leaderboard.add_argument("-o", "--output", help="write the markdown table here")
    leaderboard.add_argument("--plot", help="write an RPE-vs-fuel PNG here (needs matplotlib)")
    leaderboard.set_defaults(function=command_leaderboard)

    simulate = commands.add_parser("simulate", help="[datasets] write episodes generated with HyperDrone (needs an NVIDIA GPU)")
    simulate.add_argument("-o", "--output", default="datasets", help="directory that receives one .zip per episode, with trajectory.mp4 inside by default (default: datasets)")
    simulate.add_argument("--no-video", action="store_true", help="skip the camera and ground-truth trajectory MP4 in each episode")
    add_simulation_arguments(simulate)
    simulate.set_defaults(function=command_simulate)

    pack = commands.add_parser("pack", help="[datasets] write episodes (archives or unzipped directories) as one .zip each, e.g. to convert sets made before the archive format")
    pack.add_argument("episodes", nargs="+", help="episodes or directories of episodes")
    pack.add_argument("-o", "--output", required=True, help="directory that receives one .zip per episode")
    pack.set_defaults(function=command_pack)

    feed = commands.add_parser("feed", help="[datasets] export one episode's inputs as raw files for a native debug build (viobench_debug_main.c)")
    feed.add_argument("episode", help="episode (.zip or unzipped directory)")
    feed.add_argument("-o", "--output", required=True, help="feed directory")
    feed.add_argument("--resolution", default="x".join(map(str, abi.DEFAULT_RESOLUTION)), choices=[f"{w}x{h}" for w, h in abi.SUPPORTED_RESOLUTIONS],
                      help="what your solution declares")
    feed.set_defaults(function=command_feed)

    arguments = parser.parse_args(argv)
    return arguments.function(arguments)


if __name__ == "__main__":
    sys.exit(main())
