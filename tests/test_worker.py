"""The evaluation worker, with bounded subprocesses and offline conta resolution."""
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from viobench.tools.datasets.storage import write_episode
from viobench.tools.server import worker
from tests.test_datasets import synthetic_episode

MODULE = b"\0asm\x01\0\0\0"
HASH = "a" * 40


def report(completed=True, hashes=(HASH,)):
    episodes = [{field: 0 for field in worker.EPISODE_FIELDS} | {
        "hash": digest, "dataset": "val", "completed": completed, "frames": 2, "frames_attempted": 2,
        "fuel_total": 30, "fuel_per_frame_max": 20, "metrics": {}, "failure": "guest text", "stderr_tail": "guest text"}
                for digest in hashes]
    return {"summary": {"overall": {"episodes": len(hashes), "completed": completed}, "datasets": {}}, "episodes": episodes}


@pytest.fixture
def queued(tmp_path):
    store = SimpleNamespace(config=SimpleNamespace(episode_timeout_seconds=1440), finished=[])
    store.upload_path = lambda submission_id: tmp_path / f"{submission_id}.wasm"
    store.report_dir = lambda submission_id, digest: tmp_path / "reports" / submission_id / digest
    store.finish_job = lambda job_id, result=None, error=None: store.finished.append((job_id, result, error))
    job = {"id": 1, "submission_id": "submission", "episode_hash": HASH, "name": "team"}
    store.upload_path(job["submission_id"]).write_bytes(MODULE)
    return store, job


def fake_evaluate(monkeypatch, result, commands=None, fuels=(10, 20)):
    """An evaluation subprocess that writes its report and private attempted-frame fuel samples."""
    def run(command, log_path, timeout):
        if commands is not None:
            commands.append((command, timeout))
        output = Path(command[command.index("-o") + 1])
        (output / "report.json").write_text(json.dumps(result))
        if fuels is not None:
            logs = output / "logs" / HASH
            logs.mkdir(parents=True)
            np.save(logs / "fuel.npy", np.asarray(fuels))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(worker, "run_evaluation", run)


def test_worker_output_is_capped_and_drained(tmp_path):
    output = tmp_path / "worker.txt"
    command = [sys.executable, "-c", "import os; os.write(1, b'prefix\\n'); "
               "[(os.write(1, b'x' * 65536), os.write(2, b'y' * 65536)) for _ in range(32)]"]
    completed = worker.run_evaluation(command, output, timeout=10)
    assert completed.returncode == 0
    assert output.stat().st_size == worker.MAX_LOG_BYTES
    assert output.read_bytes().startswith(b"prefix\n")


@pytest.mark.parametrize("body", ["time.sleep(30)", "os.close(1); os.close(2); time.sleep(30)", "while True: os.write(1, b'x' * 65536)"])
def test_worker_times_out_and_reaps_children(tmp_path, body):
    pid_path = tmp_path / "pid"
    command = [sys.executable, "-c", f"import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid()))\n{body}", str(pid_path)]
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        worker.run_evaluation(command, tmp_path / "worker.txt", timeout=0.5)
    assert time.monotonic() - start < 5
    with pytest.raises(ChildProcessError):
        os.waitpid(int(pid_path.read_text()), os.WNOHANG)


def test_worker_marks_timed_out_job_failed(queued, monkeypatch):
    store, job = queued

    def timeout(command, log_path, timeout):
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(worker, "run_evaluation", timeout)
    worker.evaluate_job(store, job, log=lambda *a: None)
    assert store.finished == [(job["id"], None, f"episode evaluation exceeded {store.config.episode_timeout_seconds:.0f} s")]


@pytest.mark.parametrize("completed", [True, False])
def test_worker_evaluates_one_episode_and_keeps_guest_failures_for_aggregation(queued, monkeypatch, completed):
    store, job = queued
    commands = []
    fake_evaluate(monkeypatch, report(completed), commands)
    worker.evaluate_job(store, job, log=lambda *a: None)
    [(command, timeout)] = commands
    assert command[3:] == ["evaluate", str(store.upload_path(job["submission_id"])), "conta:" + HASH,
                           "-o", str(store.report_dir(job["submission_id"], HASH))]
    assert timeout == store.config.episode_timeout_seconds
    [(job_id, result, error)] = store.finished
    assert job_id == job["id"] and error is None
    assert result["hash"] == HASH and result["completed"] is completed
    assert set(result) == set(worker.EPISODE_FIELDS) and "guest text" not in json.dumps(result)


@pytest.mark.parametrize("hashes", [[], ["b" * 40], [HASH, HASH], [None]])
def test_worker_fails_a_report_of_other_episodes(queued, monkeypatch, hashes):
    """An episode is its hash: results of anything but the requested hashes, in order, are not shown."""
    store, job = queued
    fake_evaluate(monkeypatch, report(True, hashes))
    worker.evaluate_job(store, job, log=lambda *a: None)
    assert store.finished == [(job["id"], None, "internal error")]


@pytest.mark.parametrize("fuels", [None, (10,), [[10], [20]], (float("nan"), 20)])
def test_worker_requires_usable_fuel_samples_for_every_attempted_frame(queued, monkeypatch, fuels):
    store, job = queued
    fake_evaluate(monkeypatch, report(), fuels=fuels)
    worker.evaluate_job(store, job, log=lambda *a: None)
    assert store.finished == [(job["id"], None, "internal error")]


def test_worker_does_not_reuse_a_stale_report_after_retry(queued, monkeypatch):
    store, job = queued
    output = store.report_dir(job["submission_id"], HASH)
    logs = output / "logs" / HASH
    logs.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report()))
    np.save(logs / "fuel.npy", np.asarray([10, 20]))
    monkeypatch.setattr(worker, "run_evaluation", lambda command, *args: subprocess.CompletedProcess(command, 17))
    worker.evaluate_job(store, job, log=lambda *a: None)
    [(job_id, result, error)] = store.finished
    assert job_id == job["id"] and result is None and "17" in error and "without a report" in error
    assert not (output / "report.json").exists() and not (logs / "fuel.npy").exists()


def test_worker_resolves_cold_cache_and_continues_after_missing_episode(tmp_path, conta_store, monkeypatch):
    pytest.importorskip("httpx")
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    from viobench.tools.server import app as server
    from viobench.tools.server.store import Store
    from tests.test_fuel import metered_module

    archive = write_episode(synthetic_episode(n_frames=3, width=640, height=480), tmp_path / "episode.zip")
    digest = conta_store(archive)
    source = Path(os.environ["CONTA_ROOT"]) / "data" / digest
    source.with_suffix(".gz").write_bytes(gzip.compress(source.read_bytes()))
    source.unlink()
    cache = Path(os.environ["CONTA_CACHE"]) / digest
    assert not cache.exists()
    missing = "0" * 40
    monkeypatch.setattr("viobench.tools.server.store.default_hashes", lambda: [missing, digest])
    store = Store(tmp_path / "server")
    monkeypatch.setattr(server, "_store", store)
    client = TestClient(server.app)
    client.headers["Authorization"] = f"Bearer {store.add_team('team', is_participant=True)}"
    wasm = metered_module(tmp_path)
    created = client.post("/submissions", files={"file": ("native.wasm", wasm.read_bytes())})
    assert created.status_code == 201 and created.json()["status"] == "queued"
    submission_id = created.json()["id"]
    failed_job = store.claim_next()
    assert failed_job["episode_hash"] == missing
    worker.evaluate_job(store, failed_job, log=lambda *args: None)
    failed_result = client.get(f"/submissions/{submission_id}").json()
    assert failed_result["jobs"][0]["status"] == "failed"
    assert failed_result["progress"]["failed"] == 1 and failed_result["progress"]["queued"] == 1
    job = store.claim_next()
    assert job["episode_hash"] == digest
    worker.evaluate_job(store, job, log=lambda *args: None)
    assert cache.read_bytes() == archive.read_bytes()
    detail = client.get(f"/submissions/{submission_id}")
    assert detail.status_code == 200
    result = detail.json()
    assert result["status"] == "done" and result["results"]["partial"] is True
    assert result["progress"]["failed"] == 1 and result["progress"]["completed"] == 1
    assert not {"initial_status", "started", "finished"} & result.keys()
    assert "initial" not in result["jobs"][0]
    assert result["jobs"][0]["started"] is not None and result["jobs"][0]["finished"] is not None
    assert result["results"]["episodes"][0]["hash"] == digest
    fuel = np.load(store.report_dir(submission_id, digest) / "logs" / digest / "fuel.npy", allow_pickle=False)
    overall = result["results"]["summary"]["overall"]
    assert len(fuel) == overall["frames_attempted"] == 3
    assert int(fuel.sum()) == overall["fuel_total"]
    assert float(np.median(fuel)) == overall["fuel_per_frame_median"]
    assert client.get("/submissions").json()[0]["overall"] == overall
    assert store.claim_next() is None
