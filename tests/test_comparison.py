"""Authenticated baseline comparisons use the same complete current episode set."""
from dataclasses import replace
from math import log

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from viobench.tools.server import app as server
from viobench.tools.server.evaluations import dominated_area
from viobench.tools.server.store import Store
from tests.test_evaluation_jobs import A, B, C, MODULE, succeed


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.config = replace(store.config, limits=replace(store.config.limits, step_fuel=10))
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
    baseline_area = log(12 / 9) * log(3.6 / 3)
    claimed = {own: log(12) * log(3.6 / 0.001) - baseline_area,
               admin: log(12 / 5) * log(3.6 / 2) - baseline_area, baseline: 0}
    for token, expected_own in [(own, own_id), (admin, admin_id), (baseline, baseline_id)]:
        response = view(client, token)
        assert response["evaluation_hashes"] == [A, B]
        assert response["fuel_reference"] == 12
        assert response["fuel_floor"] == 1 and response["error_floor"] == 0.001
        [dataset] = response["datasets"]
        assert dataset["name"] == "val" and dataset["baseline_frontier"] == [baseline_id]
        assert dataset["frontier"] == [expected_own]
        assert dataset["error_reference"] == pytest.approx(3.6)
        assert dataset["claimed_area"] == pytest.approx(claimed[token])
        rectangle = log(response["fuel_reference"] / response["fuel_floor"]) * log(dataset["error_reference"] / response["error_floor"])
        assert 0 <= dataset["claimed_area"] / rectangle <= 1
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
    assert [dataset["error_reference"] for dataset in before["datasets"]] == pytest.approx([2.4, 9.6])
    assert all(dataset["claimed_area"] == 0 for dataset in before["datasets"])
    assert all([point["id"] for point in dataset["points"]] == [first] for dataset in before["datasets"])
    store.set_evaluations([A])
    selected = view(client, viewer)
    assert selected["evaluation_hashes"] == [A]
    assert {point["id"] for dataset in selected["datasets"] for point in dataset["points"]} == {first, second}
    assert selected["datasets"][0]["points"][0]["fuel_per_frame_max"] == 3
    assert [dataset["error_reference"] for dataset in selected["datasets"]] == pytest.approx([2.4, 1.2])
    store.update_team(token, team_type="participant")
    assert view(client, viewer)["datasets"] == []
    store.update_team(token, team_type="baseline")
    store.set_evaluations([C])
    assert view(client, viewer) == {"evaluation_hashes": [C], "fuel_reference": 12,
                                    "fuel_floor": 1, "error_floor": 0.001, "datasets": []}


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
    assert dataset["frontier"] == [first, tied, zero_error]  # Fuel 0 and 1 share the floor, so error 4 is dominated.
    assert dataset["baseline_frontier"] == dataset["frontier"]
    assert dataset["error_reference"] == pytest.approx(4.8) and dataset["claimed_area"] == 0


@pytest.mark.parametrize("points, expected", [
    ([], 0),
    ([(2, 3)], log(10 / 2) * log(5 / 3)),
    ([(2, 3), (6, 1)], log(6 / 2) * log(5 / 3) + log(10 / 6) * log(5 / 1)),  # The final step extends to fuel 10.
    ([(2, 3), (2, 3), (4, 4)], log(10 / 2) * log(5 / 3)),  # Ties and dominated points add no area.
    ([(11, 1), (2, 6)], 0),  # Points outside the reference rectangle claim nothing.
    ([(0, 0), (10, 0), (0, 5)], log(10 / 1) * log(5 / 0.001)),
])
def test_dominated_area_is_the_clipped_union_of_log_rectangles(points, expected):
    assert dominated_area(points, max_fuel=10, max_error=5) == pytest.approx(expected)


def test_log_area_rewards_equal_multiplicative_improvements_equally():
    expected = log(10) * log(2)
    for old_error, new_error in [(2, 1), (20, 10)]:
        before = dominated_area([(10, old_error)], 100, 100)
        after = dominated_area([(10, old_error), (10, new_error)], 100, 100)
        assert after - before == pytest.approx(expected)
    for old_fuel, new_fuel in [(20, 10), (200, 100)]:
        before = dominated_area([(old_fuel, 2)], 1000, 20)
        after = dominated_area([(old_fuel, 2), (new_fuel, 2)], 1000, 20)
        assert after - before == pytest.approx(expected)


def test_log_area_is_invariant_to_units_when_floors_and_bounds_scale_together():
    points = [(2, 0.002), (8, 0.001)]
    expected = dominated_area(points, 10, 1)
    converted = [(fuel * 1000, error * 100) for fuel, error in points]
    assert dominated_area(converted, 10_000, 100, min_fuel=1000, min_error=0.1) == pytest.approx(expected)


def test_claimed_area_credits_high_fuel_tail_with_viewer_independent_references(comparison):
    store, client = comparison
    viewer = store.add_team("viewer", unlimited=True)
    observer = store.add_team("observer")
    baseline = store.add_team("reference", team_type="baseline", unlimited=True)
    best = submit(store, baseline, scores=(4, 4), fuel=(4, 4))
    submit(store, baseline, scores=(6, 6), fuel=(6, 6))  # Even a dominated baseline sets the worst baseline error.
    before = view(client, viewer)
    [dataset] = before["datasets"]
    assert dataset["baseline_frontier"] == [best] and dataset["frontier"] == [best]
    assert dataset["error_reference"] == pytest.approx(7.2) and dataset["claimed_area"] == 0

    own = submit(store, viewer, scores=(2, 2), fuel=(10, 10))
    [dataset] = view(client, viewer)["datasets"]
    assert dataset["frontier"] == [best, own]
    assert dataset["claimed_area"] == pytest.approx(log(1.2) * log(2))  # The fuel-limit tail still earns credit.
    submit(store, viewer, scores=(2, 2), fuel=(10, 10))  # A duplicate own point earns no extra credit.
    submit(store, viewer, scores=(100, 100), fuel=(0, 0))  # Own uploads cannot enlarge the error reference.
    submit(store, viewer, scores=(0, 0), fuel=(12, 12))  # No width remains at the reference boundary.
    after = view(client, viewer)
    assert after["fuel_reference"] == before["fuel_reference"] == view(client, observer)["fuel_reference"] == 12
    assert after["datasets"][0]["error_reference"] == pytest.approx(7.2)
    assert after["datasets"][0]["claimed_area"] == pytest.approx(log(1.2) * log(2))
    assert view(client, observer)["datasets"][0]["error_reference"] == pytest.approx(7.2)
    assert view(client, observer)["datasets"][0]["claimed_area"] == 0

    store.update_team(baseline, team_type="test")
    after = view(client, viewer)
    [dataset] = after["datasets"]
    assert after["fuel_reference"] == 12 and dataset["baseline_frontier"] == []
    assert dataset["frontier"]  # Own points remain visible, but have no baseline reference to compare against.
    assert dataset["error_reference"] is None and dataset["claimed_area"] is None


def test_claimed_area_and_reference_follow_current_baselines_and_episode_selection(comparison):
    store, client = comparison
    viewer = store.add_team("viewer")
    baseline = store.add_team("reference", team_type="baseline", unlimited=True)
    submit(store, viewer, scores=(2, 6), fuel=(6, 6))
    submit(store, baseline, scores=(4, 10), fuel=(3, 3))
    for hashes, reference, area in [([A, B], 8.4, log(2) * log(7 / 4)),
                                     ([A], 4.8, log(2) ** 2), ([B], 12, log(2) * log(10 / 6))]:
        store.set_evaluations(hashes)
        result = view(client, viewer)
        assert result["evaluation_hashes"] == hashes
        [dataset] = result["datasets"]
        assert dataset["error_reference"] == pytest.approx(reference)
        assert dataset["claimed_area"] == pytest.approx(area)
    submit(store, baseline, scores=(20,), fuel=(9, 9))  # Current selection contains only B.
    [dataset] = view(client, viewer)["datasets"]
    assert dataset["error_reference"] == 24 and dataset["claimed_area"] == pytest.approx(log(2) * log(10 / 6))


def test_zero_baseline_error_still_allows_fuel_improvements_at_the_error_floor(comparison):
    store, client = comparison
    viewer = store.add_team("viewer")
    baseline = store.add_team("reference", team_type="baseline")
    submit(store, baseline, scores=(0, 0), fuel=(4, 4))
    own = submit(store, viewer, scores=(0, 0), fuel=(0, 0))
    [dataset] = view(client, viewer)["datasets"]
    assert dataset["frontier"] == [own]
    assert dataset["error_reference"] == pytest.approx(0.0012)
    assert dataset["claimed_area"] == pytest.approx(log(4) * log(1.2))


def test_zero_and_subfloor_scores_tie_for_scoring_without_changing_raw_points(comparison):
    store, client = comparison
    viewer = store.add_team("viewer")
    baseline = store.add_team("reference", team_type="baseline", unlimited=True)
    zero = submit(store, baseline, scores=(0, 0), fuel=(0, 0))
    subfloor = submit(store, baseline, scores=(0.0005, 0.0005), fuel=(1, 1))
    own = submit(store, viewer, scores=(0.0008, 0.0008), fuel=(0, 0))
    [dataset] = view(client, viewer)["datasets"]
    assert dataset["baseline_frontier"] == [zero, subfloor]
    assert dataset["frontier"] == [zero, subfloor, own]
    assert dataset["error_reference"] == pytest.approx(0.0012) and dataset["claimed_area"] == 0
    assert [(point["fuel_per_frame_max"], point["sim3_ate_rmse_median"]) for point in dataset["points"]] == [
        (0, 0), (1, 0.0005), (0, 0.0008)]
