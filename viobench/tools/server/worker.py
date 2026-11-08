"""Evaluation worker: polls SQLite for episode jobs and stores their metrics.

    python -m viobench.tools.server.worker --jobs 2   # one supervisor, two concurrent evaluations

Each job evaluates one episode. Conta resolves its hash when the evaluation reads it, downloading into the writable
cache if needed. The subprocess timeout covers both downloading and evaluation.

Runs each evaluation in a subprocess (own process = own memory, killed on timeout). The worker and evaluation host
have network access. Run as an unprivileged container user: wasmtime sandboxes the guest, not the evaluation host.

Students get metrics only: nothing the guest wrote (stdout, stderr, failure messages) leaves the server, since it could
carry the hidden episodes out. The report and capped logs stay in the report directory for the instructors.
"""
import argparse
import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event

import numpy as np

from ..datasets.sources import argument
from ..runner.logs import MAX_LOG_BYTES
from .store import Store

POLL_SECONDS = 3.0
EPISODE_FIELDS = ("hash", "dataset", "frames", "frames_attempted", "duration_s", "completed", "metrics", "fuel_init", "fuel_per_frame_max", "fuel_per_frame_median",
                  "fuel_per_frame_p95", "fuel_per_frame_p99", "fuel_per_frame_mean", "fuel_total", "wall_init_s", "wall_per_frame_median_ms",
                  "wall_per_frame_p95_ms")  # allowlist: numbers, no text from the guest


def public_episode(episode):
    """An episode's metrics without guest-written text."""
    return {key: episode[key] for key in EPISODE_FIELDS}


class EvaluationStopped(Exception):
    """The supervisor is stopping; leave the job unfinished for recovery on restart."""


def run_evaluation(command, log_path, timeout, stop=None):
    """Keep a bounded output prefix while draining the child, with a deadline even when it is silent."""
    stop = stop or Event()
    if stop.is_set():
        raise EvaluationStopped
    # Let the supervisor handle terminal signals, then stop the evaluator through the cancellation event.
    with open(log_path, "wb", buffering=0) as output, subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, start_new_session=True
    ) as process:
        deadline = time.monotonic() + timeout
        remaining = MAX_LOG_BYTES
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    if stop.is_set():
                        raise EvaluationStopped
                    time_left = deadline - time.monotonic()
                    if time_left <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    if not selector.get_map():  # stdout may close before the evaluator exits
                        if process.poll() is not None:
                            return subprocess.CompletedProcess(command, process.returncode)
                        stop.wait(min(time_left, 0.25))
                        continue
                    if not selector.select(min(time_left, 0.25)):
                        continue
                    data = os.read(process.stdout.fileno(), 65536)
                    if not data:
                        selector.unregister(process.stdout)
                        continue
                    if remaining:
                        kept = data[:remaining]
                        output.write(kept)
                        remaining -= len(kept)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def evaluate_job(store, job, log=print, stop=None):
    """Resolve and evaluate one episode within a single subprocess deadline."""
    wasm = store.upload_path(job["submission_id"])
    digest = job["episode_hash"]
    report_dir = store.report_dir(job["submission_id"], digest)
    if report_dir.exists():
        shutil.rmtree(report_dir)  # a retry must never accept a report or fuel samples from the previous attempt
    report_dir.mkdir(parents=True, exist_ok=True)
    config = store.config
    command = [sys.executable, "-m", "viobench", "evaluate", str(wasm), argument(digest), "-o", str(report_dir)]  # limits: config.json
    log(f"job {job['id']}, submission {job['submission_id']} by {job['name']}: {' '.join(command)}")
    try:
        completed = run_evaluation(command, report_dir / "worker.txt", config.episode_timeout_seconds, stop=stop)
    except subprocess.TimeoutExpired:
        store.finish_job(job["id"], error=f"episode evaluation exceeded {config.episode_timeout_seconds:.0f} s")
        return
    report_path = report_dir / "report.json"
    if not report_path.is_file():
        store.finish_job(job["id"], error=f"evaluate exited with {completed.returncode} without a report")  # its output: worker.txt
        return
    try:
        report = json.loads(report_path.read_text())
        [episode] = report["episodes"]
        if episode["hash"] != digest:
            raise ValueError("the report does not match the requested episode")
        fuel = np.load(report_dir / "logs" / digest / "fuel.npy", allow_pickle=False)
        if (fuel.ndim != 1 or fuel.dtype.kind not in "iuf" or len(fuel) != episode["frames_attempted"]
                or not np.isfinite(fuel).all() or (fuel < 0).any()):
            raise ValueError("invalid attempted-frame fuel samples")
        result = public_episode(episode)
    except (OSError, ValueError, TypeError, KeyError) as error:
        log(f"job {job['id']}: invalid evaluation report: {error}")
        store.finish_job(job["id"], error="internal error")
        return
    store.finish_job(job["id"], result=result)  # a guest failure still has useful metrics and attempted-frame fuel
    log(f"job {job['id']}, episode {digest}: {'completed' if result['completed'] else 'guest failed'}")


def work(store, stop):
    """One job slot: claim only when free, and stop without failing interrupted jobs."""
    while not stop.is_set():
        job = store.claim_next()
        if job is None:
            stop.wait(POLL_SECONDS)
            continue
        try:
            evaluate_job(store, job, stop=stop)
        except EvaluationStopped:
            return
        except Exception as error:  # noqa: BLE001 - keep the worker alive
            print(f"job {job['id']}: worker error: {error!r}")
            store.finish_job(job["id"], error="internal error")  # the details (host paths, ...) stay in the worker's log


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m viobench.tools.server.worker", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=1, help="concurrent episode evaluations (default: 1)")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    store = Store()
    stop = Event()

    def request_stop(signum, frame):
        stop.set()

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        requeued = store.requeue_running()  # exactly one supervisor, before any job slots start
        print(f"worker: polling {store.path}, {args.jobs} job slot(s), {requeued} interrupted episode job(s) requeued")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            try:
                futures = [pool.submit(work, store, stop) for _ in range(args.jobs)]
                for future in as_completed(futures):
                    future.result()  # a broken slot must stop the supervisor, not silently reduce its capacity
            finally:
                stop.set()  # stop the other slots before the executor waits for them
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
