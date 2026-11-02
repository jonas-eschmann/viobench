"""Evaluation worker: polls the queue, runs `viobench evaluate` on EPISODES, stores the team's results.

    python -m viobench.tools.server.worker            # exactly one; resolves EPISODES first (downloads what the cache lacks)
    python -m viobench.tools.server.worker --fetch    # only download EPISODES into the conta cache, then exit

EPISODES, below, is the one place that says what the server evaluates: change it and restart the worker. The episodes
come through conta's cache (CONTA_CACHE, default ~/.cache/conta). A worker without network (compose.yaml) evaluates
from a cache that `--fetch` filled; it fails at start, not at the first submission, if an episode is missing there.
An episode is its hash: the results list the episodes by hash, in the order of EPISODES, and nothing else names them.

Runs each evaluation in a subprocess (own process = own memory, killed on timeout). Run the worker as an unprivileged
user in a container without network: the guest is sandboxed by wasmtime, but the evaluation host process is not.

Students get metrics only: nothing the guest wrote (stdout, stderr, failure messages) leaves the server, since it could
carry the hidden episodes out. The full report and logs stay in the report directory for the instructors.
"""
import argparse
import json
import subprocess
import sys
import time

from ... import episodes
from ..datasets.sources import argument, reference_hash, resolve
from .store import Store

EPISODES = episodes.public  # what every submission is evaluated on: conta references (viobench.episodes)
POLL_SECONDS = 3.0
EPISODE_FIELDS = ("hash", "dataset", "frames", "frames_attempted", "duration_s", "completed", "metrics", "fuel_init", "fuel_per_frame_max", "fuel_per_frame_median",
                  "fuel_per_frame_p95", "fuel_per_frame_p99", "fuel_per_frame_mean", "fuel_total", "wall_init_s", "wall_per_frame_median_ms",
                  "wall_per_frame_p95_ms")  # allowlist: numbers, no text from the guest


def public_results(report):
    """The part of a report.json the team sees."""
    return {"summary": report["summary"], "episodes": [{key: episode[key] for key in EPISODE_FIELDS} for episode in report["episodes"]]}


def prepare(references):
    """The cached archives of references, downloading what the cache lacks; exits with conta's reason if one cannot be had,
    and before that if the references are not distinct episodes."""
    import conta

    if not references:
        sys.exit("worker: no evaluation episodes (EPISODES is empty)")
    digests = [reference_hash(reference) for reference in references]
    if None in digests:
        sys.exit(f"worker: not a conta reference in EPISODES: {references[digests.index(None)]!r}")
    repeated = sorted({digest for digest in digests if digests.count(digest) > 1})
    if repeated:  # one episode, counted twice
        sys.exit(f"worker: episodes listed twice in EPISODES: {', '.join(repeated)}")
    try:
        return resolve(references)
    except conta.ContaError as error:
        sys.exit(f"worker: cannot provide the evaluation episodes\n{error}")


def evaluate_submission(store, submission, references, log=print):
    wasm = store.upload_path(submission["id"])
    report_dir = store.report_dir(submission["id"])
    report_dir.mkdir(parents=True, exist_ok=True)
    config = store.config
    command = [sys.executable, "-m", "viobench", "evaluate", str(wasm), *map(argument, references), "-o", str(report_dir)]  # limits: config.json
    log(f"submission {submission['id']} by {submission['name']}: {' '.join(command)}")
    with open(report_dir / "worker.txt", "w") as output:
        try:
            completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=config.evaluation_timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            store.finish(submission["id"], error=f"evaluation exceeded {config.evaluation_timeout_seconds:.0f} s")
            return
    report_path = report_dir / "report.json"
    if not report_path.is_file():
        store.finish(submission["id"], error=f"evaluate exited with {completed.returncode} without a report")  # its output: worker.txt
        return
    report = json.loads(report_path.read_text())
    hashes = [episode.get("hash") for episode in report["episodes"]]
    if hashes != [reference_hash(reference) for reference in references]:  # the results are what was asked for, or none
        log(f"submission {submission['id']}: the report's episodes {hashes} are not the requested ones")
        store.finish(submission["id"], error="internal error")
        return
    overall = report["summary"]["overall"]
    if overall["completed"] == 0:  # a broken submission: failed, without results (it may fall under the failed quota)
        store.finish(submission["id"], error="no episode completed")
    else:
        store.finish(submission["id"], results=public_results(report))
    log(f"submission {submission['id']}: {overall['completed']}/{overall['episodes']} episodes completed")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m viobench.tools.server.worker", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="download EPISODES into the conta cache and exit")
    arguments = parser.parse_args(argv)
    paths = prepare(EPISODES)
    if arguments.fetch:
        print(f"worker: {len(paths)} episodes in the conta cache")
        return
    store = Store()
    requeued = store.requeue_running()  # interrupted by a restart: there is only ever one worker
    print(f"worker: {len(paths)} episodes, polling {store.path}, {requeued} interrupted submission(s) requeued")
    while True:
        submission = store.claim_next()
        if submission is None:
            time.sleep(POLL_SECONDS)
            continue
        try:
            evaluate_submission(store, submission, EPISODES)
        except Exception as error:  # noqa: BLE001 - keep the worker alive
            print(f"submission {submission['id']}: worker error: {error!r}")
            store.finish(submission["id"], error="internal error")  # the details (host paths, ...) stay in the worker's log


if __name__ == "__main__":
    main()
