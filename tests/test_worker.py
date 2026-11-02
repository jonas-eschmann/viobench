"""The evaluation worker without a real evaluation: subprocess.run stands in for `viobench evaluate`."""
import json
import subprocess
from pathlib import Path

import pytest

from viobench import episodes as published
from viobench.tools.server import worker
from viobench.tools.server.store import Store

MODULE = b"\0asm\x01\0\0\0"
REFERENCES = [{"description": "val/a.zip", "hash": "a" * 40}, {"description": "val/b.zip", "hash": "B" * 40}]
HASHES = ["a" * 40, "b" * 40]  # the episodes of REFERENCES


def report(completed, hashes=HASHES):
    episodes = [{field: 0 for field in worker.EPISODE_FIELDS} | {"hash": digest, "dataset": "val", "completed": bool(completed), "stderr_tail": "guest text"}
                for digest in hashes]
    return {"summary": {"overall": {"episodes": len(hashes), "completed": completed}, "datasets": {}}, "episodes": episodes}


@pytest.fixture
def queued(tmp_path):
    store = Store(tmp_path / "server")
    token = store.add_team("team")
    submission_id = store.add(token, "v1", "solution.wasm", MODULE)
    return store, token, submission_id, store.claim_next()


def test_the_server_evaluates_the_public_set():
    assert worker.EPISODES is published.public


def fake_evaluate(monkeypatch, report, commands=None):
    """subprocess.run as `viobench evaluate` that writes report."""
    def run(command, **options):
        if commands is not None:
            commands.append(command)
        (Path(command[command.index("-o") + 1]) / "report.json").write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(worker.subprocess, "run", run)


@pytest.mark.parametrize("completed", [1, 0])
def test_worker_evaluates_its_references(queued, monkeypatch, completed):
    store, token, submission_id, submission = queued
    commands = []
    fake_evaluate(monkeypatch, report(completed), commands)
    worker.evaluate_submission(store, submission, REFERENCES, log=lambda *a: None)
    [command] = commands
    assert command[3:6] == ["evaluate", str(store.upload_path(submission_id)), "conta:" + "a" * 40]
    assert command[6] == "conta:" + "b" * 40 and command[7] == "-o"
    row = store.get(token, submission_id)
    if completed:
        assert row["status"] == "done" and "guest text" not in row["results"]
        assert [episode["hash"] for episode in json.loads(row["results"])["episodes"]] == HASHES
    else:
        assert row["status"] == "failed" and row["error"] == "no episode completed"


@pytest.mark.parametrize("hashes", [HASHES[:1], HASHES[::-1], HASHES + ["c" * 40], [HASHES[0], None]])
def test_worker_fails_a_report_of_other_episodes(queued, monkeypatch, hashes):
    """An episode is its hash: results of anything but the requested hashes, in order, are not shown."""
    store, token, submission_id, submission = queued
    fake_evaluate(monkeypatch, report(1, hashes))
    messages = []
    worker.evaluate_submission(store, submission, REFERENCES, log=messages.append)
    row = store.get(token, submission_id)
    assert row["status"] == "failed" and row["error"] == "internal error" and row["results"] is None
    assert "not the requested ones" in messages[-1]


def test_worker_does_not_start_without_distinct_episodes():
    with pytest.raises(SystemExit, match="listed twice in EPISODES: " + "a" * 40):
        worker.prepare([{"description": "a.zip", "hash": "a" * 40}, {"description": "A.zip", "hash": "A" * 40}])
    with pytest.raises(SystemExit, match="not a conta reference"):
        worker.prepare([{"description": "short.zip", "hash": "a" * 39}])


def test_worker_does_not_start_without_its_episodes(tmp_path, conta_store):
    archive = tmp_path / "a.zip"
    archive.write_bytes(b"an episode")
    digest = conta_store(archive)
    [path] = worker.prepare([{"description": "a.zip", "hash": digest}])
    assert path.read_bytes() == b"an episode"
    with pytest.raises(SystemExit, match="0{40}"):
        worker.prepare([{"description": "gone.zip", "hash": "0" * 40}])
    with pytest.raises(SystemExit, match="EPISODES is empty"):
        worker.prepare([])


def test_fetch_only_fills_the_cache(tmp_path, conta_store, monkeypatch, capsys):
    archive = tmp_path / "a.zip"
    archive.write_bytes(b"an episode")
    monkeypatch.setattr(worker, "EPISODES", [{"description": "a.zip", "hash": conta_store(archive)}])
    monkeypatch.setattr(worker, "Store", lambda *a: pytest.fail("--fetch must not open the store"))
    worker.main(["--fetch"])
    assert "1 episodes in the conta cache" in capsys.readouterr().out
