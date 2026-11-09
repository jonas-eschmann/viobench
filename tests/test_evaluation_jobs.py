"""Evaluation selection, durable per-episode jobs, and upload quota accounting; no datasets or network."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from threading import Barrier
import time

import numpy as np
import pytest

from viobench.tools.runner.run import fuel_statistics
from viobench.tools.server import store as store_module
from viobench.tools.server.evaluations import default_hashes
from viobench.tools.server.store import Refused, Store

A, B, C, D = (letter * 40 for letter in "abcd")
MODULE = b"\0asm\x01\0\0\0"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "default_hashes", lambda: [A])
    return Store(tmp_path)


def submit(store, token):
    return store.add(token, "v1", "solution.wasm", MODULE)


def jobs(store, token, submission_id):
    return {job["episode_hash"]: job for job in store.get(token, submission_id)["jobs"]}


def start_job(store, job_id):
    """Set up a particular queued job for tests that deliberately complete jobs out of FIFO order."""
    with store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("UPDATE evaluation_jobs SET status = 'running', started = ?, evaluation_config = ? "
                           "WHERE id = ? AND status = 'queued'", (time.time(), json.dumps(asdict(store.config.evaluation)), job_id))


def succeed(store, submission_id, job, fuel=(1, 3), rpe=0.1, sim3=0.2, completed=True):
    """Persist the same numeric result and fuel artifact that the isolated worker commits."""
    samples = np.asarray(fuel, dtype=np.float64)
    directory = store.report_dir(submission_id, job["episode_hash"]) / "logs" / job["episode_hash"]
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "fuel.npy", samples, allow_pickle=False)
    result = {
        "hash": job["episode_hash"], "completed": completed,
        "frames": len(samples), "duration_s": 1.0, "fuel_init": 0, "wall_init_s": 0.1,
        "wall_per_frame_median_ms": 1.0, "wall_per_frame_p95_ms": 2.0,
        "metrics": {"sim3_ate_rmse": sim3, "rpe_translation_rmse": rpe, "rpe_rotation_rmse": 0.1, "rotation_rmse": 0.1, "ate_rmse": 0.2,
                    "velocity_rmse": 0.3, "angular_velocity_rmse": 0.4},
        **fuel_statistics(samples),
    }
    start_job(store, job["id"])
    store.finish_job(job["id"], result=result)
    return result


def test_default_selection_is_seeded_once_and_reopening_preserves_state(tmp_path):
    first = Store(tmp_path)
    seeded = default_hashes()
    assert seeded and first.evaluation_hashes() == seeded
    with first._connect() as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert "evaluation_episodes" not in tables
        assert {row["table"] for row in connection.execute("PRAGMA foreign_key_list(evaluation_jobs)")} == {"submissions"}
    first.set_evaluations([D, A])
    first.set_pause_message("Maintenance")

    reopened = Store(tmp_path)
    assert reopened.evaluation_hashes() == [D, A]
    assert reopened.pause_message() == "Maintenance"
    assert set(reopened.evaluations()) == {"hashes", "presets", "outdated_jobs", "outdated_running"}


@pytest.mark.parametrize("invalid", [[], [A, A.upper()], ["x" * 40], ["a" * 39], [7], A])
def test_invalid_hash_selection_does_not_change_selection_or_jobs(store, invalid):
    token = store.add_team("team")
    submission_id = submit(store, token)
    original = jobs(store, token, submission_id)
    for change in (store.set_evaluations, store.schedule_evaluations):
        with pytest.raises(ValueError):
            change(invalid)
        assert store.evaluation_hashes() == [A]
        assert jobs(store, token, submission_id) == original


def test_cold_hash_is_claimable_and_restart_requeues_only_unfinished_jobs(store):
    token = store.add_team("team")
    submission_id = submit(store, token)
    running = store.claim_next()  # No archive download or readiness record is needed to claim a job.
    assert running["status"] == "running" and running["episode_hash"] == A
    reopened = Store(store.directory)
    assert reopened.claim_next() is None
    assert reopened.requeue_running() == 1
    view = reopened.get(token, submission_id)
    assert view["status"] == "queued"
    assert not {"initial_status", "started", "finished"} & view.keys()
    assert view["jobs"][0]["started"] is None and view["jobs"][0]["finished"] is None
    reclaimed = reopened.claim_next()
    assert reclaimed["id"] == running["id"]
    reopened.finish_job(reclaimed["id"], error="solution failed")
    assert reopened.requeue_running() == 0
    assert reopened.set_evaluations([A], backfill=True) == 0
    assert reopened.schedule_evaluations() == 0
    assert reopened.claim_next() is None
    assert reopened.get(token, submission_id)["progress"]["finished"] == 1


def test_selection_changes_during_execution_preserve_queued_work_and_reuse_pairs(store):
    store.set_evaluations([A, B])
    token = store.add_team("team")
    submission_id = submit(store, token)
    running = store.claim_next()
    assert running["episode_hash"] == A
    original_ids = {digest: job["id"] for digest, job in jobs(store, token, submission_id).items()}

    assert store.set_evaluations([B, C]) == 0
    view = store.get(token, submission_id)
    assert view["progress"] == {"total": 2, "finished": 0, "completed": 0, "failed": 0,
                                "queued": 1, "running": 0, "missing": 1, "outdated": 0}
    preserved = jobs(store, token, submission_id)
    assert not preserved[A]["selected"] and preserved[A]["status"] == "running"
    assert preserved[B]["status"] == "queued"
    store.finish_job(running["id"], error="failure in removed episode")
    assert store.get(token, submission_id)["status"] == "queued"
    remaining = store.claim_next()
    assert remaining["episode_hash"] == B
    store.finish_job(remaining["id"], error="failure")
    view = store.get(token, submission_id)
    assert view["status"] == "incomplete"
    assert view["progress"]["finished"] == 1 and view["progress"]["missing"] == 1

    assert store.set_evaluations([A, B], backfill=True) == 0
    assert {digest: job["id"] for digest, job in jobs(store, token, submission_id).items()} == original_ids
    assert store.get(token, submission_id)["progress"]["finished"] == 2
    assert store.claim_next() is None


def test_manual_backfill_scopes_and_automatic_backfill_only_insert_missing_pairs(store):
    tokens = [store.add_team(name) for name in ("one", "two")]
    submissions = [submit(store, token) for token in tokens]
    assert store.set_evaluations([A, B]) == 0
    assert all(store.get(token, sid)["progress"]["missing"] == 1 for token, sid in zip(tokens, submissions))
    assert store.schedule_evaluations(submission_id=submissions[0]) == 1
    assert store.schedule_evaluations([C.upper()], submission_id=submissions[0]) == 1
    assert not jobs(store, tokens[0], submissions[0])[C]["selected"]
    assert C not in jobs(store, tokens[1], submissions[1])
    assert store.evaluation_hashes() == [A, B]  # Explicit ad-hoc jobs do not change the shared selection.
    assert store.schedule_evaluations() == 1
    assert store.schedule_evaluations() == 0
    assert store.set_evaluations([A, B, C], backfill=True) == 1
    for token, submission_id in zip(tokens, submissions):
        selected = jobs(store, token, submission_id)
        assert set(selected) == {A, B, C}
        assert all("initial" not in job for job in selected.values())
    before = store.evaluations()
    with pytest.raises(LookupError):
        store.schedule_evaluations([D], submission_id="does-not-exist")
    assert store.evaluations() == before


def test_concurrent_scheduling_is_idempotent_and_claims_are_unique(store):
    tokens = [store.add_team(name) for name in ("one", "two")]
    submissions = [submit(store, token) for token in tokens]
    store.set_evaluations([A, B, C])
    barrier = Barrier(6)

    def schedule(_):
        barrier.wait(timeout=10)
        return store.schedule_evaluations()

    with ThreadPoolExecutor(max_workers=6) as pool:
        counts = list(pool.map(schedule, range(6)))
    assert sorted(counts) == [0, 0, 0, 0, 0, 4]
    def claim(_):
        barrier.wait(timeout=10)
        return store.claim_next()

    with ThreadPoolExecutor(max_workers=6) as pool:
        claimed = list(pool.map(claim, range(6)))
    assert all(job is not None and job["status"] == "running" for job in claimed)
    assert len({job["id"] for job in claimed}) == 6
    assert {(job["submission_id"], job["episode_hash"]) for job in claimed} == {
        (submission_id, digest) for submission_id in submissions for digest in (A, B, C)}
    assert store.claim_next() is None


def test_jobs_run_fifo_including_backfills(store):
    token = store.add_team("older")
    old = submit(store, token)
    store.finish_job(store.claim_next()["id"], error="failure")
    store.set_evaluations([B], backfill=True)
    newcomer = store.add_team("newer")
    new = submit(store, newcomer)
    first = store.claim_next()
    assert first["submission_id"] == old
    store.finish_job(first["id"], error="failure")
    second = store.claim_next()
    assert second["submission_id"] == new


def test_upload_quota_counts_accepted_uploads_independently_of_jobs(store):
    store.set_config(replace(store.config, uploads=replace(store.config.uploads, daily_quota=2)))
    token = store.add_team("team")
    first = submit(store, token)
    store.finish_job(store.claim_next()["id"], error="failure")
    store.set_evaluations([A, B], backfill=True)
    assert store.get(token, first)["status"] == "queued"
    second = submit(store, token)  # Pending jobs allow another upload when quota remains.
    with pytest.raises(Refused, match="quota"):
        submit(store, token)

    succeed(store, first, jobs(store, token, first)[B])
    assert store.get(token, first)["status"] == "done"
    second_jobs = jobs(store, token, second)
    for job in second_jobs.values():
        start_job(store, job["id"])
        store.finish_job(job["id"], error="failure")
    assert store.get(token, second)["status"] == "failed"
    with pytest.raises(Refused, match="quota"):
        submit(store, token)

    unlimited = store.add_team("unlimited", unlimited=True)
    unlimited_submissions = [submit(store, unlimited) for _ in range(3)]
    assert len(set(unlimited_submissions)) == 3
    assert all(store.get(unlimited, sid)["progress"]["queued"] == 2 for sid in unlimited_submissions)


def test_concurrent_uploads_cannot_exceed_the_daily_quota(store):
    store.set_config(replace(store.config, uploads=replace(store.config.uploads, daily_quota=2)))
    token = store.add_team("team")
    barrier = Barrier(6)

    def upload(_):
        barrier.wait(timeout=10)
        try:
            return submit(store, token)
        except Refused as exc:
            assert "quota" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(upload, range(6)))

    accepted = {submission_id for submission_id in outcomes if submission_id is not None}
    assert len(accepted) == 2 and outcomes.count(None) == 4
    assert {submission["id"] for submission in store.submissions(token)} == accepted
    assert all(store.get(token, sid)["progress"]["queued"] == 1 for sid in accepted)


def test_aggregates_pool_unequal_frame_distributions_and_follow_current_selection(store):
    store.set_evaluations([A, B])
    token = store.add_team("team")
    submission_id = submit(store, token)
    selected_jobs = jobs(store, token, submission_id)
    succeed(store, submission_id, selected_jobs[A], fuel=[1, 3], rpe=0.1, sim3=2)
    partial = store.get(token, submission_id)
    assert partial["results"]["partial"] and partial["progress"]["finished"] == 1
    succeed(store, submission_id, selected_jobs[B], fuel=[10, 30, 50, 70, 90, 200], rpe=0.3, sim3=8)
    complete = store.get(token, submission_id)
    assert complete["progress"]["finished"] == complete["progress"]["completed"] == 2
    assert not complete["results"]["partial"]
    summary = complete["results"]["summary"]
    assert summary["frames_attempted"] == 8 and summary["fuel_total"] == 454
    assert summary["fuel_per_frame_median"] == 40  # Neither mean nor median of the episode medians (2 and 60).
    assert summary["fuel_per_frame_p95"] == pytest.approx(161.5)
    assert summary["fuel_per_frame_p99"] == pytest.approx(192.3)
    assert summary["fuel_per_frame_mean"] == pytest.approx(56.75)
    assert summary["rpe_translation_rmse_median"] == pytest.approx(0.2)
    assert summary["sim3_ate_rmse_mean"] == 5
    assert not {"overall", "datasets"} & summary.keys()

    assert store.set_evaluations([B], backfill=True) == 0
    narrowed = store.get(token, submission_id)
    assert [episode["hash"] for episode in narrowed["results"]["episodes"]] == [B]
    assert narrowed["results"]["summary"]["fuel_per_frame_median"] == 60
    assert store.set_evaluations([B, A], backfill=True) == 0
    assert store.get(token, submission_id)["results"]["summary"] == summary


def test_wrong_hash_and_duplicate_completion_do_not_mutate_the_job(store):
    token = store.add_team("team")
    submission_id = submit(store, token)
    queued = jobs(store, token, submission_id)[A]
    with pytest.raises(ValueError, match="running"):
        store.finish_job(queued["id"], error="not claimed")
    assert jobs(store, token, submission_id)[A] == queued
    running = store.claim_next()
    with pytest.raises(ValueError, match="hash"):
        store.finish_job(running["id"], result={"hash": B})
    assert jobs(store, token, submission_id)[A]["status"] == "running"
    store.finish_job(running["id"], error="failure")
    finished = jobs(store, token, submission_id)[A]
    with pytest.raises(ValueError, match="running"):
        store.finish_job(running["id"], error="different failure")
    assert jobs(store, token, submission_id)[A] == finished
