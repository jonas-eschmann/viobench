"""Authenticated baseline comparisons use the same complete current episode set."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from viobench.tools.server import app as server
from viobench.tools.server.store import Store
from tests.test_evaluation_jobs import A, B, C, MODULE, succeed


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.set_evaluations([A, B])
    monkeypatch.setattr(server, "_store", store)
    return store, TestClient(server.app)


def submit(store, token, scores=(1, 3), fuel=(2, 5), datasets=("val", "val"), completed=True, label=""):
    submission_id = store.add(token, label, "solution.wasm", MODULE)
    with store._connect() as connection:
        jobs = connection.execute("SELECT * FROM evaluation_jobs WHERE submission_id = ? ORDER BY id", (submission_id,)).fetchall()
    for job, score, used, dataset in zip(jobs, scores, fuel, datasets):
        succeed(store, submission_id, job, fuel=(used,), dataset=dataset, sim3=score, completed=completed)
    return submission_id


def view(client, token):
    response = client.get("/comparison", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    return response.json()


def test_comparison_shares_only_baseline_scores_even_with_admin_viewer(comparison):
    store, client = comparison
    own, other, test, baseline, admin = [store.add_team(name, team_type=kind, unlimited=True)
                                       for name, kind in [("own", "participant"), ("other", "participant"),
                                                          ("debug", "test"), ("reference", "baseline"), ("admin", "test")]]
    own_id = submit(store, own, scores=(0, 0), fuel=(0, 0), label="zero")
    hidden = [submit(store, token) for token in (other, test)]
    baseline_id = submit(store, baseline, scores=(2, 4), fuel=(1, 9), completed=False)
    admin_id = submit(store, admin)
    store.update_team(admin, team_type="admin")  # Existing uploads remain owned when the account changes type.
    for token, expected_own in [(own, own_id), (admin, admin_id), (baseline, baseline_id)]:
        response = view(client, token)
        assert response["evaluation_hashes"] == [A, B]
        [dataset] = response["datasets"]
        assert dataset["name"] == "val" and dataset["frontier"] == [baseline_id]
        points = {point["id"]: point for point in dataset["points"]}
        assert len(dataset["points"]) == len(points)  # A baseline viewer's own submission is not duplicated.
        assert set(points) == {expected_own, baseline_id}
        assert points[expected_own]["own"] is True
        assert points[baseline_id] == {
            "id": baseline_id, "label": baseline_id, "team_name": "reference", "team_type": "baseline",
            "own": token == baseline, "fuel_per_frame_max": 9, "sim3_ate_rmse_median": 3,
        }
        assert not any(secret in str(response) for secret in (own, other, test, baseline, admin, *hidden))
    own_point = view(client, own)["datasets"][0]["points"][0]
    assert own_point["label"] == "zero" and own_point["fuel_per_frame_max"] == own_point["sim3_ate_rmse_median"] == 0
    headers = {"Authorization": f"Bearer {own}"}
    assert client.get(f"/submissions/{baseline_id}", headers=headers).status_code == 404
    assert [row["id"] for row in client.get("/submissions", headers=headers).json()] == [own_id]


def test_comparison_tracks_account_type_and_current_selection(comparison):
    store, client = comparison
    viewer = store.add_team("viewer")
    token = store.add_team("reference", unlimited=True)
    first = submit(store, token, scores=(2, 8), fuel=(3, 11), datasets=("a", "b"))
    second = submit(store, token, scores=(1,))  # B has not produced a report.
    assert view(client, viewer)["datasets"] == []
    store.update_team(token, team_type="baseline")
    before = view(client, viewer)
    assert [dataset["name"] for dataset in before["datasets"]] == ["a", "b"]
    assert [dataset["points"][0]["fuel_per_frame_max"] for dataset in before["datasets"]] == [11, 11]
    assert [dataset["points"][0]["sim3_ate_rmse_median"] for dataset in before["datasets"]] == [2, 8]
    assert all([point["id"] for point in dataset["points"]] == [first] for dataset in before["datasets"])
    store.set_evaluations([A])
    selected = view(client, viewer)
    assert selected["evaluation_hashes"] == [A]
    assert {point["id"] for dataset in selected["datasets"] for point in dataset["points"]} == {first, second}
    assert selected["datasets"][0]["points"][0]["fuel_per_frame_max"] == 3
    store.update_team(token, team_type="participant")
    assert view(client, viewer)["datasets"] == []
    store.update_team(token, team_type="baseline")
    store.set_evaluations([C])
    assert view(client, viewer) == {"evaluation_hashes": [C], "datasets": []}


def test_comparison_excludes_unscored_or_invalid_selected_reports(comparison):
    store, client = comparison
    token = store.add_team("reference", team_type="baseline", unlimited=True)
    valid = submit(store, token)
    submit(store, token, scores=(1,))  # Still queued.
    failed = submit(store, token, scores=(1,))
    with store._connect() as connection:
        job_id = connection.execute("SELECT id FROM evaluation_jobs WHERE submission_id = ? AND episode_hash = ?", (failed, B)).fetchone()[0]
    store.finish_job(job_id, error="download failed")
    for score in (float("nan"), float("inf"), -1, None):
        submit(store, token, scores=(1, score), datasets=("a", "b"))
    [dataset] = view(client, token)["datasets"]
    assert [point["id"] for point in dataset["points"]] == [valid]


def test_baseline_frontier_retains_ties_and_zero_coordinates(comparison):
    store, client = comparison
    token = store.add_team("reference", team_type="baseline", unlimited=True)
    zero_fuel = submit(store, token, scores=(4, 4), fuel=(0, 0))
    first = submit(store, token, scores=(2, 2), fuel=(1, 1))
    tied = submit(store, token, scores=(2, 2), fuel=(1, 1))
    zero_error = submit(store, token, scores=(0, 0), fuel=(2, 2))
    dominated = submit(store, token, scores=(3, 3), fuel=(2, 2))
    [dataset] = view(client, token)["datasets"]
    assert {point["id"] for point in dataset["points"]} == {zero_fuel, first, tied, zero_error, dominated}
    assert dataset["frontier"] == [zero_fuel, first, tied, zero_error]
