"""Baseline deletion removes only idle owned submissions, including their evaluation artifacts."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tests.test_server import MODULE, api, finish_submission, results

A, B, C = (letter * 40 for letter in "abc")


def baseline(store):
    token = store.add_team("reference", team_type="baseline")
    return token, {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("authorization", [None, "Bearer wrong"])
def test_delete_requires_authentication(api, authorization):
    client, _ = api
    client.headers.pop("Authorization")
    headers = {"Authorization": authorization} if authorization else {}
    response = client.delete("/submissions/missing", headers=headers)
    assert response.status_code == 401 and response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("kind", ["test", "participant", "admin"])
def test_only_current_baseline_accounts_can_delete(api, kind):
    client, store = api
    token, headers = baseline(store)
    submission_id = store.add(token, "old baseline", "solution.wasm", MODULE)
    store.update_team(token, team_type=kind)
    response = client.delete(f"/submissions/{submission_id}", headers=headers)
    assert response.status_code == 403
    assert store.get(token, submission_id) is not None
    assert store.upload_path(submission_id).read_bytes() == MODULE
    if kind != "admin":
        assert client.post("/submissions", files={"file": ("next.wasm", MODULE)}, headers=headers).status_code == 429


def test_baseline_cannot_delete_foreign_or_missing_submissions(api):
    client, store = api
    _, headers = baseline(store)
    foreign = client.post("/submissions", files={"file": ("other.wasm", MODULE)}).json()["id"]
    for submission_id in (foreign, "missing"):
        assert client.delete(f"/submissions/{submission_id}", headers=headers).status_code == 404
    assert client.get(f"/submissions/{foreign}").status_code == 200
    assert store.upload_path(foreign).read_bytes() == MODULE


def test_delete_removes_all_jobs_and_artifacts_and_baseline_comparison_point(api, tmp_path):
    client, store = api
    token, headers = baseline(store)
    submission_id = store.add(token, "reference", "solution.wasm", MODULE)
    finish_submission(store, submission_id, results=results(0.1))
    store.schedule_evaluations([B, C], submission_id=submission_id)
    with store._connect() as connection:
        failed_id = connection.execute("SELECT id FROM evaluation_jobs WHERE submission_id = ? AND episode_hash = ?",
                                       (submission_id, B)).fetchone()[0]
    store.finish_job(failed_id, error="download failed")
    for digest in (B, C):
        directory = store.report_dir(submission_id, digest)
        directory.mkdir(parents=True)
        (directory / "worker.txt").write_text("previous attempt")
    other = client.post("/submissions", files={"file": ("other.wasm", MODULE)}).json()["id"]
    other_report = store.report_dir(other, A) / "worker.txt"
    other_report.parent.mkdir(parents=True)
    other_report.write_text("keep other report")
    cache_file = tmp_path / "conta-cache" / A
    cache_file.parent.mkdir()
    cache_file.write_bytes(b"shared archive")
    assert client.get("/comparison").json()["datasets"][0]["points"][0]["id"] == submission_id

    response = client.delete(f"/submissions/{submission_id}", headers=headers)
    assert response.status_code == 200 and response.json() == {"deleted": submission_id}
    assert store.get(token, submission_id) is None
    with store._connect() as connection:
        assert connection.execute("SELECT 1 FROM evaluation_jobs WHERE submission_id = ?", (submission_id,)).fetchone() is None
    assert not store.upload_path(submission_id).exists()
    assert not (store.reports / submission_id).exists()
    assert client.get("/comparison").json()["datasets"] == []
    assert client.get(f"/submissions/{submission_id}", headers=headers).status_code == 404
    assert client.get(f"/submissions/{other}").status_code == 200
    assert store.upload_path(other).read_bytes() == MODULE
    assert other_report.read_text() == "keep other report" and cache_file.read_bytes() == b"shared archive"
    assert store.claim_next()["submission_id"] == other


def test_deleting_queued_submission_removes_queue_entry_and_frees_baseline_quota(api):
    client, store = api
    _, headers = baseline(store)
    first = client.post("/submissions", files={"file": ("solution.wasm", MODULE)}, headers=headers)
    assert first.status_code == 201
    submission_id = first.json()["id"]
    assert client.post("/submissions", files={"file": ("solution.wasm", MODULE)}, headers=headers).status_code == 429
    assert client.delete(f"/submissions/{submission_id}", headers=headers).status_code == 200
    assert store.claim_next() is None
    replacement = client.post("/submissions", files={"file": ("replacement.wasm", MODULE)}, headers=headers)
    assert replacement.status_code == 201 and replacement.json()["id"] != submission_id


def test_failed_submission_with_missing_files_can_be_deleted(api):
    client, store = api
    token, headers = baseline(store)
    submission_id = store.add(token, "failed", "solution.wasm", MODULE)
    finish_submission(store, submission_id, error="evaluation failed")
    store.upload_path(submission_id).unlink()
    assert not (store.reports / submission_id).exists()
    response = client.delete(f"/submissions/{submission_id}", headers=headers)
    assert response.status_code == 200 and response.json() == {"deleted": submission_id}
    assert store.submissions(token) == [] and store.claim_next() is None


def test_file_cleanup_permission_error_is_sanitized_after_database_deletion(api, monkeypatch):
    client, store = api
    token, headers = baseline(store)
    submission_id = store.add(token, "queued", "solution.wasm", MODULE)
    directory = store.report_dir(submission_id, A)
    directory.mkdir(parents=True)

    def denied(path):
        raise PermissionError(f"private server path: {path}")

    with monkeypatch.context() as patch:
        patch.setattr("viobench.tools.server.store.shutil.rmtree", denied)
        response = client.delete(f"/submissions/{submission_id}", headers=headers)
    assert response.status_code == 500
    assert response.json() == {"detail": "Submission deleted, but server file cleanup failed."}
    assert str(store.directory) not in response.text
    assert store.get(token, submission_id) is None and store.claim_next() is None
    assert not store.upload_path(submission_id).exists() and directory.exists()


@pytest.mark.parametrize("selected", [True, False])
def test_any_running_job_prevents_deletion_even_outside_current_selection(api, selected):
    client, store = api
    token, headers = baseline(store)
    submission_id = store.add(token, "running", "solution.wasm", MODULE)
    job = store.claim_next()
    report = store.report_dir(submission_id, A) / "worker.txt"
    report.parent.mkdir(parents=True)
    report.write_text("running evaluation")
    if not selected:
        store.set_evaluations([B])
    response = client.delete(f"/submissions/{submission_id}", headers=headers)
    assert response.status_code == 409
    assert store.get(token, submission_id)["jobs"][0]["status"] == "running"
    assert store.upload_path(submission_id).read_bytes() == MODULE and report.read_text() == "running evaluation"
    store.finish_job(job["id"], error="finished")
    assert client.delete(f"/submissions/{submission_id}", headers=headers).status_code == 200


def test_worker_claim_and_deletion_cannot_both_win(api):
    client, store = api
    token, headers = baseline(store)
    submission_id = store.add(token, "queued", "solution.wasm", MODULE)
    barrier = Barrier(2)

    def claim():
        barrier.wait(timeout=5)
        return store.claim_next()

    def delete():
        barrier.wait(timeout=5)
        return client.delete(f"/submissions/{submission_id}", headers=headers)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed, deleted = pool.submit(claim), pool.submit(delete)
        job, response = claimed.result(timeout=5), deleted.result(timeout=5)
    if job is None:
        assert response.status_code == 200 and store.get(token, submission_id) is None
        assert not store.upload_path(submission_id).exists()
    else:
        assert response.status_code == 409 and job["submission_id"] == submission_id
        assert store.get(token, submission_id)["jobs"][0]["status"] == "running"
        assert store.upload_path(submission_id).read_bytes() == MODULE
