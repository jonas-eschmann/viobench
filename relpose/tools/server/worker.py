"""Evaluation worker: polls the queue, runs `relpose evaluate` on the server's datasets, stores the team's results.

    RELPOSE_DATASETS=/data/test/first:/data/test/second python -m relpose.tools.server.worker

Runs each evaluation in a subprocess (own process = own memory, killed on timeout). Run the worker as an unprivileged
user in a container without network: the guest is sandboxed by wasmtime, but the evaluation host process is not.

Students get metrics only: nothing the guest wrote (stdout, stderr, failure messages) leaves the server, since it could
carry the hidden episodes out. The full report and logs stay in the report directory for the instructors.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .store import Store

POLL_SECONDS = 3.0
EPISODE_FIELDS = ("dataset", "name", "frames", "duration_s", "completed", "metrics", "fuel_init", "fuel_per_frame_median",
                  "fuel_per_frame_p95", "fuel_per_frame_mean", "fuel_total", "wall_init_s", "wall_per_frame_median_ms",
                  "wall_per_frame_p95_ms")  # allowlist: numbers, no text from the guest


def public_results(report):
    """The part of a report.json the team sees."""
    return {"summary": report["summary"], "episodes": [{key: episode[key] for key in EPISODE_FIELDS} for episode in report["episodes"]]}


def datasets_from_environment():
    value = os.environ.get("RELPOSE_DATASETS", "")
    paths = [Path(p) for p in value.split(os.pathsep) if p]
    if not paths:
        sys.exit("RELPOSE_DATASETS must list the dataset directories (os.pathsep separated)")
    for path in paths:
        if not path.is_dir():
            sys.exit(f"dataset directory {path} does not exist")
    return paths


def evaluate_submission(store, submission, datasets, log=print):
    wasm = store.upload_path(submission["id"])
    report_dir = store.report_dir(submission["id"])
    report_dir.mkdir(parents=True, exist_ok=True)
    config = store.config
    command = [sys.executable, "-m", "relpose", "evaluate", str(wasm), *map(str, datasets), "-o", str(report_dir)]  # limits: config.json
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
    overall = report["summary"]["overall"]
    if overall["completed"] == 0:  # a broken submission: failed, without results (it may fall under the failed quota)
        store.finish(submission["id"], error="no episode completed")
    else:
        store.finish(submission["id"], results=public_results(report))
    log(f"submission {submission['id']}: {overall['completed']}/{overall['episodes']} episodes completed")


def main():
    store = Store()
    datasets = datasets_from_environment()
    requeued = store.requeue_running()  # interrupted by a restart: there is only ever one worker
    print(f"worker: {len(datasets)} dataset directories, polling {store.path}, {requeued} interrupted submission(s) requeued")
    while True:
        submission = store.claim_next()
        if submission is None:
            time.sleep(POLL_SECONDS)
            continue
        try:
            evaluate_submission(store, submission, datasets)
        except Exception as error:  # noqa: BLE001 - keep the worker alive
            print(f"submission {submission['id']}: worker error: {error!r}")
            store.finish(submission["id"], error="internal error")  # the details (host paths, ...) stay in the worker's log


if __name__ == "__main__":
    main()
