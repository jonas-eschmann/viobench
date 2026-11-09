"""Authenticated baseline comparisons use the same complete current episode set."""
import json
from dataclasses import replace
from math import log

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from viobench.tools.server import app as server
from viobench.tools.server.evaluations import dominated_area
from viobench.tools.server.store import Store
from tests.test_evaluation_jobs import A, B, C, MODULE, start_job, succeed


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.set_config(replace(store.config, evaluation=replace(store.config.evaluation,
                             limits=replace(store.config.evaluation.limits, step_fuel=10))))
    store.set_evaluations([A, B])
    monkeypatch.setattr(server, "_store", store)
    return store, TestClient(server.app)


def submit(store, token, scores=(1, 3), fuel=(2, 5), completed=True, label=""):
    submission_id = store.add(token, label, "solution.wasm", MODULE)
    with store._connect() as connection:
        jobs = connection.execute("SELECT * FROM evaluation_jobs WHERE submission_id = ? ORDER BY id", (submission_id,)).fetchall()
    for job, score, used in zip(jobs, scores, fuel):
        succeed(store, submission_id, job, fuel=(used,), sim3=score, completed=completed)
    return submission_id


def view(client, token):
    response = client.get("/comparison", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"evaluation_hashes", "fuel_floor", "error_floor", "fuel_reference", "points",
                           "baseline_frontier", "frontier", "error_reference", "claimed_area"}
    return result


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
        assert response["baseline_frontier"] == [baseline_id]
        assert response["frontier"] == [expected_own]
        assert response["error_reference"] == pytest.approx(3.6)
        assert response["claimed_area"] == pytest.approx(claimed[token])
        rectangle = log(response["fuel_reference"] / response["fuel_floor"]) * log(response["error_reference"] / response["error_floor"])
        assert 0 <= response["claimed_area"] / rectangle <= 1
        points = {point["id"]: point for point in response["points"]}
        assert len(response["points"]) == len(points)  # A baseline viewer's own submission is not duplicated.
        assert set(points) == {expected_own, baseline_id}
        assert points[expected_own]["own"] is True
        assert points[baseline_id] == {
            "id": baseline_id, "label": baseline_id, "team_name": "reference", "team_type": "baseline",
            "own": token == baseline, "fuel_per_frame_max": 9, "sim3_ate_rmse_mean": 3,
        }
        assert not any(secret in str(response) for secret in (own, other, test, baseline, admin, *hidden))
    own_point = view(client, own)["points"][0]
    assert own_point["label"] == "zero" and own_point["fuel_per_frame_max"] == own_point["sim3_ate_rmse_mean"] == 0
    headers = {"Authorization": f"Bearer {own}"}
    assert client.get(f"/submissions/{baseline_id}", headers=headers).status_code == 404
    assert [row["id"] for row in client.get("/submissions", headers=headers).json()] == [own_id]


def test_comparison_aggregates_legacy_sources_once_and_tracks_roles_and_selection(comparison):
    store, client = comparison
    viewer = store.add_team("viewer")
    token = store.add_team("reference", unlimited=True)
    first = submit(store, token, scores=(2, 8), fuel=(3, 11))
    second = submit(store, token, scores=(1,))  # B has not produced a report.
    with store._connect() as connection:
        old_jobs = connection.execute("SELECT id, result FROM evaluation_jobs WHERE submission_id = ? ORDER BY id", (first,)).fetchall()
        for index, job in enumerate(old_jobs):
            legacy_result = json.loads(job["result"]) | {"dataset": f"legacy-source-{index}"}
            connection.execute("UPDATE evaluation_jobs SET result = ? WHERE id = ?", (json.dumps(legacy_result), job["id"]))
    assert view(client, viewer)["points"] == []
    store.update_team(token, team_type="baseline")
    before = view(client, viewer)
    [point] = before["points"]
    assert point["id"] == first and point["fuel_per_frame_max"] == 11
    assert point["sim3_ate_rmse_mean"] == 5
    assert before["error_reference"] == 6 and before["claimed_area"] == 0
    store.set_evaluations([A])
    selected = view(client, viewer)
    assert selected["evaluation_hashes"] == [A]
    assert [(point["id"], point["fuel_per_frame_max"], point["sim3_ate_rmse_mean"])
            for point in selected["points"]] == [(first, 3, 2), (second, 2, 1)]
    assert selected["error_reference"] == pytest.approx(2.4)
    headers = {"Authorization": f"Bearer {token}"}
    for path in ("/submissions", f"/submissions/{first}", "/comparison"):
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert all(key not in response.text for key in ('"dataset"', '"datasets"', '"overall"', "legacy-source-"))
    detail = client.get(f"/submissions/{first}", headers=headers).json()
    assert detail["results"]["summary"]["sim3_ate_rmse_mean"] == 2
    assert [episode["hash"] for episode in detail["results"]["episodes"]] == [A]
    assert len(detail["jobs"]) == 2  # Old jobs remain stored without influencing the selected aggregate.
    store.set_evaluations([B, A])
    restored = view(client, viewer)
    assert restored["points"] == before["points"] and restored["error_reference"] == 6
    store.update_team(token, team_type="participant")
    assert view(client, viewer)["points"] == []
    store.update_team(token, team_type="baseline")
    store.set_evaluations([C])
    assert view(client, viewer) == {"evaluation_hashes": [C], "fuel_reference": 12,
                                    "fuel_floor": 1, "error_floor": 0.001, "points": [],
                                    "baseline_frontier": [], "frontier": [], "error_reference": None, "claimed_area": None}


def test_comparison_excludes_unscored_or_invalid_selected_reports(comparison):
    store, client = comparison
    token = store.add_team("reference", team_type="baseline", unlimited=True)
    valid = submit(store, token)
    submit(store, token, scores=(1,))  # Still queued.
    failed = submit(store, token, scores=(1,))
    with store._connect() as connection:
        job_id = connection.execute("SELECT id FROM evaluation_jobs WHERE submission_id = ? AND episode_hash = ?", (failed, B)).fetchone()[0]
    start_job(store, job_id)
    store.finish_job(job_id, error="download failed")
    for score in (float("nan"), float("inf"), -1, None):
        submit(store, token, scores=(1, score))
    result = view(client, token)
    assert [point["id"] for point in result["points"]] == [valid]


def test_baseline_frontier_retains_ties_and_zero_coordinates(comparison):
    store, client = comparison
    token = store.add_team("reference", team_type="baseline", unlimited=True)
    zero_fuel = submit(store, token, scores=(4, 4), fuel=(0, 0))
    first = submit(store, token, scores=(2, 2), fuel=(1, 1))
    tied = submit(store, token, scores=(2, 2), fuel=(1, 1))
    zero_error = submit(store, token, scores=(0, 0), fuel=(2, 2))
    dominated = submit(store, token, scores=(3, 3), fuel=(2, 2))
    result = view(client, token)
    assert {point["id"] for point in result["points"]} == {zero_fuel, first, tied, zero_error, dominated}
    assert result["frontier"] == [first, tied, zero_error]  # Fuel 0 and 1 share the floor, so error 4 is dominated.
    assert result["baseline_frontier"] == result["frontier"]
    assert result["error_reference"] == pytest.approx(4.8) and result["claimed_area"] == 0


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
    assert before["baseline_frontier"] == [best] and before["frontier"] == [best]
    assert before["error_reference"] == pytest.approx(7.2) and before["claimed_area"] == 0

    own = submit(store, viewer, scores=(2, 2), fuel=(10, 10))
    result = view(client, viewer)
    assert result["frontier"] == [best, own]
    assert result["claimed_area"] == pytest.approx(log(1.2) * log(2))  # The fuel-limit tail still earns credit.
    submit(store, viewer, scores=(2, 2), fuel=(10, 10))  # A duplicate own point earns no extra credit.
    submit(store, viewer, scores=(100, 100), fuel=(0, 0))  # Own uploads cannot enlarge the error reference.
    submit(store, viewer, scores=(0, 0), fuel=(12, 12))  # No width remains at the reference boundary.
    after = view(client, viewer)
    assert after["fuel_reference"] == before["fuel_reference"] == view(client, observer)["fuel_reference"] == 12
    assert after["error_reference"] == pytest.approx(7.2)
    assert after["claimed_area"] == pytest.approx(log(1.2) * log(2))
    assert view(client, observer)["error_reference"] == pytest.approx(7.2)
    assert view(client, observer)["claimed_area"] == 0

    store.update_team(baseline, team_type="test")
    after = view(client, viewer)
    assert after["fuel_reference"] == 12 and after["baseline_frontier"] == []
    assert after["frontier"]  # Own points remain visible, but have no baseline reference to compare against.
    assert after["error_reference"] is None and after["claimed_area"] is None


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
        assert result["error_reference"] == pytest.approx(reference)
        assert result["claimed_area"] == pytest.approx(area)
    submit(store, baseline, scores=(20,), fuel=(9, 9))  # Current selection contains only B.
    result = view(client, viewer)
    assert result["error_reference"] == 24 and result["claimed_area"] == pytest.approx(log(2) * log(10 / 6))


def test_mean_ate_weights_selected_episodes_equally_for_points_references_and_area(comparison):
    store, client = comparison
    store.set_evaluations([A, B, C])
    baseline = store.add_team("reference", team_type="baseline")
    viewer = store.add_team("viewer")
    baseline_id = store.add(baseline, "", "solution.wasm", MODULE)
    for score, frames, fuel in [(1, 1, 1), (1, 2, 2), (10, 7, 3)]:
        succeed(store, baseline_id, store.claim_next(), fuel=(fuel,) * frames, sim3=score)
    own_id = submit(store, viewer, scores=(1, 2, 3), fuel=(2, 4, 6))

    result = view(client, viewer)
    points = {point["id"]: point for point in result["points"]}
    assert points[baseline_id]["sim3_ate_rmse_mean"] == 4  # Median is 1; frame-weighted mean is 7.3.
    assert points[own_id]["sim3_ate_rmse_mean"] == 2
    assert result["error_reference"] == pytest.approx(4.8)
    assert result["claimed_area"] == pytest.approx(log(2) ** 2)
    headers = {"Authorization": f"Bearer {baseline}"}
    [listed] = client.get("/submissions", headers=headers).json()
    summary = client.get(f"/submissions/{baseline_id}", headers=headers).json()["results"]["summary"]
    assert summary == listed["summary"] and summary["frames_attempted"] == 10
    assert summary["sim3_ate_rmse_mean"] == 4 and "sim3_ate_rmse_median" not in summary

    store.set_evaluations([A, B])
    narrowed = view(client, viewer)
    assert {point["id"]: point["sim3_ate_rmse_mean"] for point in narrowed["points"]} == {baseline_id: 1, own_id: 1.5}
    assert narrowed["error_reference"] == pytest.approx(1.2) and narrowed["claimed_area"] == 0


def test_zero_baseline_error_still_allows_fuel_improvements_at_the_error_floor(comparison):
    store, client = comparison
    viewer = store.add_team("viewer")
    baseline = store.add_team("reference", team_type="baseline")
    submit(store, baseline, scores=(0, 0), fuel=(4, 4))
    own = submit(store, viewer, scores=(0, 0), fuel=(0, 0))
    result = view(client, viewer)
    assert result["frontier"] == [own]
    assert result["error_reference"] == pytest.approx(0.0012)
    assert result["claimed_area"] == pytest.approx(log(4) * log(1.2))


def test_zero_and_subfloor_scores_tie_for_scoring_without_changing_raw_points(comparison):
    store, client = comparison
    viewer = store.add_team("viewer")
    baseline = store.add_team("reference", team_type="baseline", unlimited=True)
    zero = submit(store, baseline, scores=(0, 0), fuel=(0, 0))
    subfloor = submit(store, baseline, scores=(0.0005, 0.0005), fuel=(1, 1))
    own = submit(store, viewer, scores=(0.0008, 0.0008), fuel=(0, 0))
    result = view(client, viewer)
    assert result["baseline_frontier"] == [zero, subfloor]
    assert result["frontier"] == [zero, subfloor, own]
    assert result["error_reference"] == pytest.approx(0.0012) and result["claimed_area"] == 0
    assert [(point["fuel_per_frame_max"], point["sim3_ate_rmse_mean"]) for point in result["points"]] == [
        (0, 0), (1, 0.0005), (0, 0.0008)]
