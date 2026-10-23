"""Command line: python -m eecs116 {bundle,unbundle,check,...}"""
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
    thread = threading.Thread(target=target, name="eecs116-guest")
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def command_bundle(arguments):
    from .bundle import BundleError, bundle

    try:
        summary = bundle(arguments.directory, arguments.output, runtime=arguments.runtime, entry=arguments.entry)
    except BundleError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {summary['output']} ({summary['bytes'] / 2**20:.1f} MiB, bundle {summary['bundle_bytes'] / 1024:.0f} KiB, {len(summary['files'])} files, entry {summary['entry']})")
    for warning in summary["warnings"]:
        print(f"warning: {warning}")
    print(f"next: python -m eecs116 check {summary['output']}")
    return 0


def command_unbundle(arguments):
    from .bundle import BundleError, unbundle

    try:
        names = unbundle(arguments.wasm, arguments.output)
    except BundleError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"extracted {len(names)} files to {arguments.output}")
    return 0


def command_check(arguments):
    import numpy as np

    from .config import make_config
    from .solution import Limits, Solution, SolutionError

    config = make_config(width=arguments.width, height=arguments.height, channels=arguments.channels, num_frames=arguments.frames)
    rng = np.random.default_rng(0)
    shape = (arguments.height, arguments.width) if arguments.channels == 1 else (arguments.height, arguments.width, arguments.channels)
    dt = 1.0 / config["episode"]["frame_rate_hz"]
    imu_dt = 1.0 / config["imu"]["rate_hz"]
    rows_per_frame = int(round(dt / imu_dt))

    def run():
        solution = Solution(arguments.wasm, limits=Limits(), log_dir=arguments.logs, dev_dir=arguments.dir, inherit_stdio=arguments.verbose)
        try:
            solution.init(config)
        except SolutionError as error:
            print(f"init failed: {error}")
            return 1
        print(f"init: fuel {solution.init_fuel:,} wall {solution.init_wall_seconds * 1e3:.0f} ms")
        failures = 0
        for k in range(arguments.frames):
            t = (k + 1) * dt
            base = np.linspace(0, 255, arguments.width, dtype=np.float32)[None, :] + rng.normal(0, 20, (arguments.height, arguments.width))
            frame = np.clip(base, 0, 255).astype(np.uint8)
            if arguments.channels > 1:
                frame = np.repeat(frame[:, :, None], arguments.channels, axis=2)
            times = t - dt + imu_dt * np.arange(1, rows_per_frame + 1)
            imu = np.column_stack([times, rng.normal(0, 0.3, (rows_per_frame, 3)) + [0, 0, 9.81], rng.normal(0, 0.07, (rows_per_frame, 3))])
            try:
                result = solution.step(t, frame, imu)
            except SolutionError as error:
                print(f"step {k}: {error}")
                return 1
            pose = "-" if result.pose is None else np.array2string(result.pose, precision=3, suppress_small=True)
            print(f"step {k}: {abi.status_name(result.status):12s} fuel {result.fuel:>14,} wall {result.wall_seconds * 1e3:8.2f} ms pose {pose}")
            failures += result.status < 0
        if not arguments.verbose:
            out, err = solution.stdout_tail(), solution.stderr_tail()
            if out:
                print("--- stdout (tail) ---\n" + out)
            if err:
                print("--- stderr (tail) ---\n" + err)
        print("check:", "FAILED" if failures else "ok", f"({failures} failing steps of {arguments.frames})")
        return 1 if failures else 0

    return run_with_large_stack(run)


def command_evaluate(arguments):
    from .evaluate import evaluate
    from .solution import Limits

    limits = Limits()
    if arguments.step_fuel:
        limits.step_fuel = arguments.step_fuel
    if arguments.memory_mib:
        limits.memory_bytes = arguments.memory_mib << 20

    def run():
        result = evaluate(arguments.wasm, arguments.datasets, output_dir=arguments.output, limits=limits, dev_dir=arguments.dir, max_episodes=arguments.max_episodes)
        summary = result["summary"]["overall"]
        print(f"overall: {summary['completed']}/{summary['episodes']} episodes completed, ATE median {summary['ate_rmse_median']:.3f} m, "
              f"fuel/frame median {summary['fuel_per_frame_median']:,.0f}")
        return 0 if summary["completed"] == summary["episodes"] else 1

    return run_with_large_stack(run)


def command_simulate(arguments):
    from pathlib import Path

    from .data.euroc_io import write_episode
    from .data.simulate import SimConfig, hssd_scenes, procthor_scenes, simulate

    sim = SimConfig(width=arguments.width, height=arguments.height, instances=arguments.instances, frame_rate_hz=arguments.frame_rate,
                    fov_range=arguments.fov_range, hold_seconds=arguments.hover, aa_grid=arguments.aa_grid)
    if arguments.scenes == "procthor-train":
        scenes = procthor_scenes(arguments.scene_names)
    elif arguments.scenes == "hssd-val":
        scenes = hssd_scenes("val", arguments.scene_names)
    elif arguments.scenes == "hssd-train":
        scenes = hssd_scenes("train", arguments.scene_names)
    else:
        raise SystemExit(f"unknown scene set {arguments.scenes}")
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = range(arguments.seed, arguments.seed + arguments.batches)
    written = 0
    for episode in simulate(sim, seeds, scenes=scenes, variant=arguments.variant):
        if written >= arguments.episodes:
            break
        write_episode(episode, output / episode.name)
        written += 1
        print(f"wrote {output / episode.name} ({episode.num_frames} frames, {episode.duration:.1f}s, fov {episode.config['camera']['horizontal_fov_degrees']:.1f} deg)")
    print(f"{written} episodes in {output}")
    return 0 if written >= arguments.episodes else 1


def command_convert_euroc(arguments):
    from pathlib import Path

    from .data.convert_euroc import convert
    from .data.euroc_io import write_episode

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for episode, origin_ns in convert(arguments.sequence, width=arguments.width, height=arguments.height, snippet_seconds=arguments.snippet,
                                      max_snippets=arguments.max_snippets, start_seconds=arguments.start, stride_seconds=arguments.stride):
        write_episode(episode, output / episode.name, origin_ns=origin_ns)
        count += 1
    print(f"{count} episodes in {output}")
    return 0 if count else 1


def command_leaderboard(arguments):
    from .leaderboard import load_reports, markdown_table, plot, rows

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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m eecs116", description="EECS 116/216 relative pose challenge tools")
    commands = parser.add_subparsers(dest="command", required=True)

    bundle = commands.add_parser("bundle", help="embed a directory of Python code into a solution .wasm")
    bundle.add_argument("directory")
    bundle.add_argument("-o", "--output", required=True, help="output .wasm")
    bundle.add_argument("--runtime", help="runtime.wasm to build on (default: $EECS116_RUNTIME or the repository build)")
    bundle.add_argument("--entry", help=f"module:Class of the solution (default {abi.DEFAULT_ENTRY} or eecs116.json)")
    bundle.set_defaults(function=command_bundle)

    unbundle = commands.add_parser("unbundle", help="extract the bundled directory from a solution .wasm")
    unbundle.add_argument("wasm")
    unbundle.add_argument("-o", "--output", required=True, help="output directory")
    unbundle.set_defaults(function=command_unbundle)

    check = commands.add_parser("check", help="instantiate a solution and run a few synthetic frames through it")
    check.add_argument("wasm", help="solution .wasm (or a bare runtime.wasm together with --dir)")
    check.add_argument("--dir", help="dev mode: mount this solution directory instead of a bundle")
    check.add_argument("--frames", type=int, default=5)
    check.add_argument("--width", type=int, default=320)
    check.add_argument("--height", type=int, default=240)
    check.add_argument("--channels", type=int, default=1)
    check.add_argument("--logs", help="directory for stdout.txt/stderr.txt (default: temporary)")
    check.add_argument("-v", "--verbose", action="store_true", help="pass the guest's stdout/stderr through instead of capturing them")
    check.set_defaults(function=command_check)

    evaluate = commands.add_parser("evaluate", help="run a solution over episode directories and write report.json")
    evaluate.add_argument("wasm")
    evaluate.add_argument("datasets", nargs="+", help="episode directories or directories of episodes")
    evaluate.add_argument("-o", "--output", help="report directory (default: ./report-<name>)")
    evaluate.add_argument("--dir", help="dev mode: mount this solution directory instead of a bundle")
    evaluate.add_argument("--max-episodes", type=int)
    evaluate.add_argument("--step-fuel", type=int, help="fuel budget per step (default from Limits)")
    evaluate.add_argument("--memory-mib", type=int, help="guest memory cap in MiB")
    evaluate.set_defaults(function=command_evaluate)

    simulate = commands.add_parser("simulate", help="generate episodes with HyperDrone (needs a GPU and pip install hyperdrone)")
    simulate.add_argument("-o", "--output", required=True, help="directory that receives one sub-directory per episode")
    simulate.add_argument("--episodes", type=int, default=8, help="episodes to write (a batch of --instances flights per seed)")
    simulate.add_argument("--seed", type=int, default=0, help="first seed; seeds also pick the scene round-robin")
    simulate.add_argument("--batches", type=int, default=1_000_000, help="maximum number of seeds/batches to run")
    simulate.add_argument("--variant", default="default", choices=["default", "blur", "lowlight", "oversaturated"])
    simulate.add_argument("--scenes", default="procthor-train", choices=["procthor-train", "hssd-train", "hssd-val"])
    simulate.add_argument("--scene-names", nargs="*", help="restrict to these scene names")
    simulate.add_argument("--width", type=int, default=320)
    simulate.add_argument("--height", type=int, default=240)
    simulate.add_argument("--instances", type=int, default=8)
    simulate.add_argument("--frame-rate", type=float, default=20.0)
    simulate.add_argument("--fov-range", type=float, default=15.0, help="horizontal FOV = 63.8 + U(-range, range) degrees")
    simulate.add_argument("--hover", type=float, default=1.0, help="seconds of hover at the start of each episode")
    simulate.add_argument("--aa-grid", type=int, default=1)
    simulate.set_defaults(function=command_simulate)

    convert = commands.add_parser("convert-euroc", help="cut a EuRoC MAV sequence (ASL format directory) into challenge episodes")
    convert.add_argument("sequence", help="sequence directory (containing mav0/)")
    convert.add_argument("-o", "--output", required=True)
    convert.add_argument("--snippet", type=float, default=10.0, help="episode length in seconds")
    convert.add_argument("--stride", type=float, help="seconds between snippet starts (default: --snippet, no overlap)")
    convert.add_argument("--start", type=float, default=0.0, help="skip this many seconds at the beginning")
    convert.add_argument("--max-snippets", type=int)
    convert.add_argument("--width", type=int, default=320)
    convert.add_argument("--height", type=int, default=240)
    convert.set_defaults(function=command_convert_euroc)

    leaderboard = commands.add_parser("leaderboard", help="aggregate report.json files into the ATE-vs-fuel table (and plot)")
    leaderboard.add_argument("reports", nargs="+", help="report.json files or directories containing them")
    leaderboard.add_argument("-o", "--output", help="write the markdown table here")
    leaderboard.add_argument("--plot", help="write an ATE-vs-fuel PNG here (needs matplotlib)")
    leaderboard.set_defaults(function=command_leaderboard)

    arguments = parser.parse_args(argv)
    return arguments.function(arguments)


if __name__ == "__main__":
    sys.exit(main())
