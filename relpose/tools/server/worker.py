"""Evaluation worker: polls the queue, runs `relpose evaluate` on the hidden datasets, stores the summary.

    RELPOSE_DATASETS=/data/hidden/default:/data/hidden/blur:... relpose.tools.server.worker

Runs each evaluation in a subprocess (own process = own memory, killed on timeout). Run the worker as an unprivileged
user in a container without network: the guest is sandboxed by wasmtime, but the evaluation host process is not.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .store import Store

POLL_SECONDS = 3.0
TIMEOUT_SECONDS = float(os.environ.get("RELPOSE_EVAL_TIMEOUT", "14400"))  # ~50 one-minute episodes per submission


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
    command = [sys.executable, "-m", "relpose", "evaluate", str(wasm), *map(str, datasets), "-o", str(report_dir)]
    log(f"submission {submission['id']} by {submission['user']}: {' '.join(command)}")
    with open(report_dir / "worker.txt", "w") as output:
        try:
            completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired:
            store.finish(submission["id"], error=f"evaluation exceeded {TIMEOUT_SECONDS:.0f} s")
            return
    report_path = report_dir / "report.json"
    if not report_path.is_file():
        tail = "\n".join((report_dir / "worker.txt").read_text(errors="replace").splitlines()[-20:])
        store.finish(submission["id"], error=f"evaluate exited with {completed.returncode}:\n{tail}")
        return
    report = json.loads(report_path.read_text())
    report["label"] = submission["label"]
    store.finish(submission["id"], summary=report["summary"])
    log(f"submission {submission['id']}: done ({report['summary']['overall']['completed']}/{report['summary']['overall']['episodes']} episodes)")


def main():
    store = Store()
    datasets = datasets_from_environment()
    print(f"worker: {len(datasets)} dataset directories, polling {store.path}")
    while True:
        submission = store.claim_next()
        if submission is None:
            time.sleep(POLL_SECONDS)
            continue
        try:
            evaluate_submission(store, submission, datasets)
        except Exception as error:  # noqa: BLE001 - keep the worker alive
            store.finish(submission["id"], error=f"worker error: {error}")


if __name__ == "__main__":
    main()
