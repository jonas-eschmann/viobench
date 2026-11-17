"""Comparisons show all eligible points and score the best single submission against shared baselines."""
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


def submit(store, token, scores=(1, 3), fuel=(2, 5), completed=True, label="", baseline=False, sim3_scores=None):
    submission_id = store.add(token, label, "solution.wasm", MODULE)
    if baseline:
        store.set_baseline(submission_id, True)
    with store._connect() as connection:
        jobs = connection.execute("SELECT * FROM evaluation_jobs WHERE submission_id = ? ORDER BY id", (submission_id,)).fetchall()
    sim3_scores = scores if sim3_scores is None else sim3_scores
    for job, score, used, sim3 in zip(jobs, scores, fuel, sim3_scores):
        succeed(store, submission_id, job, fuel=(used,), ate=score, sim3=sim3, completed=completed)
    return submission_id


def view(client, token):
    response = client.get("/comparison", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"evaluation_hashes", "fuel_floor", "error_floor", "fuel_reference", "points",
                           "baseline_frontier", "frontier", "error_reference", "claimed_area",
                           "best_submission_id", "best_frontier"}
    return result


def listed_best(store, client, token):
    response = client.get("/submissions", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    rows = response.json()
    assert all(type(row["is_best"]) is bool for row in rows)
    assert {row["id"]: row["is_best"] for row in rows} == {
        row["id"]: row["is_best"] for row in store.submissions(token)}
    return {row["id"] for row in rows if row["is_best"]}


def test_comparison_shares_only_baseline_scores_even_with_admin_viewer(comparison):
    store, client = comparison
    own, other, test, baseline, admin = [store.add_team(name, team_type=kind, unlimited=True)
                                       for name, kind in [("own", "participant"), ("other", "participant"),
                                                          ("debug", "test"), ("reference", "participant"), ("admin", "test")]]
    own_id = submit(store, own, scores=(0, 0), fuel=(0, 0), label="zero")
    hidden = [submit(store, token) for token in (other, test)]
    baseline_id = submit(store, baseline, scores=(2, 4), fuel=(1, 9), completed=False, baseline=True)
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
        assert response["best_submission_id"] == expected_own
        assert response["best_frontier"] == [expected_own]
        assert response["error_reference"] == pytest.approx(3.6)
        assert response["claimed_area"] == pytest.approx(claimed[token])
        rectangle = log(response["fuel_reference"] / response["fuel_floor"]) * log(response["error_reference"] / response["error_floor"])
        assert 0 <= response["claimed_area"] / rectangle <= 1
        points = {point["id"]: point for point in response["points"]}
        assert len(response["points"]) == len(points)  # A baseline viewer's own submission is not duplicated.
        assert set(points) == {expected_own, baseline_id}
        assert points[expected_own]["own"] is True
        assert points[baseline_id] == {
            "id": baseline_id, "label": baseline_id, "team_name": "reference", "team_type": "participant", "is_baseline": True,
            "own": token == baseline, "fuel_per_frame_max": 9, "ate_rmse_mean": 3,
        }
        assert not any(secret in str(response) for secret in (own, other, test, baseline, admin, *hidden))
    own_point = view(client, own)["points"][0]
    assert own_point["label"] == "zero" and own_point["fuel_per_frame_max"] == own_point["ate_rmse_mean"] == 0
    headers = {"Authorization": f"Bearer {own}"}
    assert client.get(f"/submissions/{baseline_id}", headers=headers).status_code == 404
    assert [row["id"] for row in client.get("/submissions", headers=headers).json()] == [own_id]


@pytest.mark.parametrize("team_type", ["test", "participant", "admin"])
def test_baseline_flag_shares_only_selected_submission_independently_of_account_type(comparison, team_type):
    store, client = comparison
    owner = store.add_team("owner", team_type=team_type, unlimited=True)
    observer = store.add_team("observer", team_type="participant")
    selected = submit(store, owner, scores=(3, 3), baseline=True)
    private = submit(store, owner, scores=(1, 1))
    [shared] = view(client, observer)["points"]
    assert shared["id"] == selected and shared["is_baseline"] is True and shared["own"] is False
    assert shared["team_type"] == team_type
    owned = view(client, owner)["points"]
    assert len(owned) == 2 and {point["id"] for point in owned} == {selected, private}
    assert all(point["own"] is True for point in owned)
    assert {point["id"]: point["is_baseline"] for point in owned} == {selected: True, private: False}
    headers = {"Authorization": f"Bearer {observer}"}
    assert client.get(f"/submissions/{selected}", headers=headers).status_code == 404
    assert client.get(f"/submissions/{private}", headers=headers).status_code == 404
    store.update_team(owner, team_type="test" if team_type == "admin" else "admin")
    assert [point["id"] for point in view(client, observer)["points"]] == [selected]
    store.set_baseline(selected, False)
    assert view(client, observer)["points"] == []
    assert len(view(client, owner)["points"]) == 2


def test_comparison_aggregates_legacy_sources_once_and_tracks_baselines_and_selection(comparison):
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
    for submission_id in (first, second):
        store.set_baseline(submission_id, True)
    before = view(client, viewer)
    [point] = before["points"]
    assert point["id"] == first and point["fuel_per_frame_max"] == 11
    assert point["ate_rmse_mean"] == 5
    assert before["error_reference"] == 6 and before["claimed_area"] == 0
    store.set_evaluations([A])
    selected = view(client, viewer)
    assert selected["evaluation_hashes"] == [A]
    assert [(point["id"], point["fuel_per_frame_max"], point["ate_rmse_mean"])
            for point in selected["points"]] == [(first, 3, 2), (second, 2, 1)]
    assert selected["error_reference"] == pytest.approx(1.2)  # The second submission is now the fastest eligible baseline.
    headers = {"Authorization": f"Bearer {token}"}
    for path in ("/submissions", f"/submissions/{first}", "/comparison"):
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert all(key not in response.text for key in ('"dataset"', '"datasets"', '"overall"', "legacy-source-"))
    detail = client.get(f"/submissions/{first}", headers=headers).json()
    assert detail["results"]["summary"]["ate_rmse_mean"] == 2
    assert [episode["hash"] for episode in detail["results"]["episodes"]] == [A]
    assert len(detail["jobs"]) == 2  # Old jobs remain stored without influencing the selected aggregate.
    store.set_evaluations([B, A])
    restored = view(client, viewer)
    assert restored["points"] == before["points"] and restored["error_reference"] == 6
    for submission_id in (first, second):
        store.set_baseline(submission_id, False)
    assert view(client, viewer)["points"] == []
    for submission_id in (first, second):
        store.set_baseline(submission_id, True)
    store.set_evaluations([C])
    assert view(client, viewer) == {"evaluation_hashes": [C], "fuel_reference": 12,
                                    "fuel_floor": 1, "error_floor": 0.001, "points": [],
                                    "baseline_frontier": [], "frontier": [], "error_reference": None, "claimed_area": None,
                                    "best_submission_id": None, "best_frontier": []}


def test_comparison_excludes_unscored_or_invalid_selected_reports(comparison):
    store, client = comparison
    token = store.add_team("reference", team_type="participant", unlimited=True)
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
    token = store.add_team("reference", team_type="participant", unlimited=True)
    zero_fuel = submit(store, token, scores=(4, 4), fuel=(0, 0), baseline=True)
    first = submit(store, token, scores=(2, 2), fuel=(1, 1), baseline=True)
    tied = submit(store, token, scores=(2, 2), fuel=(1, 1), baseline=True)
    zero_error = submit(store, token, scores=(0, 0), fuel=(2, 2), baseline=True)
    dominated = submit(store, token, scores=(3, 3), fuel=(2, 2), baseline=True)
    result = view(client, token)
    assert {point["id"] for point in result["points"]} == {zero_fuel, first, tied, zero_error, dominated}
    assert result["frontier"] == [first, tied, zero_error]  # Fuel 0 and 1 share the floor, so error 4 is dominated.
    assert result["baseline_frontier"] == result["frontier"]
    assert result["best_submission_id"] == zero_fuel  # Every owned baseline has exactly zero added area.
    assert result["best_frontier"] == result["baseline_frontier"]  # The winning baseline is not duplicated.
    assert result["error_reference"] == pytest.approx(2.4) and result["claimed_area"] == 0


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
    baseline = store.add_team("reference", team_type="participant", unlimited=True)
    best = submit(store, baseline, scores=(4, 4), fuel=(4, 4), baseline=True)
    submit(store, baseline, scores=(6, 6), fuel=(6, 6), baseline=True)  # A slower dominated baseline does not move the reference.
    before = view(client, viewer)
    assert before["baseline_frontier"] == [best] and before["frontier"] == [best]
    assert before["error_reference"] == pytest.approx(4.8) and before["claimed_area"] == 0
    assert before["best_submission_id"] is None and before["best_frontier"] == [best]

    own = submit(store, viewer, scores=(2, 2), fuel=(10, 10))
    result = view(client, viewer)
    assert result["frontier"] == [best, own]
    assert result["claimed_area"] == pytest.approx(log(1.2) * log(2))  # The fuel-limit tail still earns credit.
    submit(store, viewer, scores=(2, 2), fuel=(10, 10))  # A duplicate own point earns no extra credit.
    submit(store, viewer, scores=(100, 100), fuel=(0, 0))  # Own uploads cannot enlarge the error reference.
    submit(store, viewer, scores=(0, 0), fuel=(12, 12))  # No width remains at the reference boundary.
    after = view(client, viewer)
    assert after["fuel_reference"] == before["fuel_reference"] == view(client, observer)["fuel_reference"] == 12
    assert after["error_reference"] == pytest.approx(4.8)
    assert after["claimed_area"] == pytest.approx(log(1.2) * log(2))
    assert after["best_submission_id"] == own and after["best_frontier"] == [best, own]
    assert view(client, observer)["error_reference"] == pytest.approx(4.8)
    assert view(client, observer)["claimed_area"] == 0

    for submission in store.submissions(baseline):
        store.set_baseline(submission["id"], False)
    after = view(client, viewer)
    assert after["fuel_reference"] == 12 and after["baseline_frontier"] == []
    assert after["frontier"]  # Own points remain visible, but have no baseline reference to compare against.
    assert after["error_reference"] is None and after["claimed_area"] is None
    assert after["best_submission_id"] is None and after["best_frontier"] == []


def test_claimed_area_and_reference_follow_current_baselines_and_episode_selection(comparison):
    store, client = comparison
    viewer = store.add_team("viewer")
    baseline = store.add_team("reference", team_type="participant", unlimited=True)
    submit(store, viewer, scores=(2, 6), fuel=(6, 6))
    submit(store, baseline, scores=(4, 10), fuel=(3, 3), baseline=True)
    for hashes, reference, area in [([A, B], 8.4, log(2) * log(7 / 4)),
                                     ([A], 4.8, log(2) ** 2), ([B], 12, log(2) * log(10 / 6))]:
        store.set_evaluations(hashes)
        result = view(client, viewer)
        assert result["evaluation_hashes"] == hashes
        assert result["error_reference"] == pytest.approx(reference)
        assert result["claimed_area"] == pytest.approx(area)
    submit(store, baseline, scores=(20,), fuel=(9, 9), baseline=True)  # Current selection contains only B.
    result = view(client, viewer)
    assert result["error_reference"] == 12 and result["claimed_area"] == pytest.approx(log(2) * log(10 / 6))


def test_mean_ate_weights_selected_episodes_equally_for_points_references_and_area(comparison):
    store, client = comparison
    store.set_evaluations([A, B, C])
    baseline = store.add_team("reference", team_type="participant")
    viewer = store.add_team("viewer")
    baseline_id = store.add(baseline, "", "solution.wasm", MODULE)
    store.set_baseline(baseline_id, True)
    for score, frames, fuel in [(1, 1, 1), (1, 2, 2), (10, 7, 3)]:
        succeed(store, baseline_id, store.claim_next(), fuel=(fuel,) * frames, ate=score, sim3=score / 100)
    own_id = submit(store, viewer, scores=(1, 2, 3), fuel=(2, 4, 6))

    result = view(client, viewer)
    points = {point["id"]: point for point in result["points"]}
    assert points[baseline_id]["ate_rmse_mean"] == 4  # Median is 1; frame-weighted mean is 7.3.
    assert points[own_id]["ate_rmse_mean"] == 2
    assert result["error_reference"] == pytest.approx(4.8)
    assert result["claimed_area"] == pytest.approx(log(2) ** 2)
    headers = {"Authorization": f"Bearer {baseline}"}
    [listed] = client.get("/submissions", headers=headers).json()
    summary = client.get(f"/submissions/{baseline_id}", headers=headers).json()["results"]["summary"]
    assert summary == listed["summary"] and summary["frames_attempted"] == 10
    assert summary["ate_rmse_mean"] == 4 and summary["ate_rmse_median"] == 1
    assert summary["sim3_ate_rmse_mean"] == pytest.approx(0.04)  # Aligned error remains a separate diagnostic.

    store.set_evaluations([A, B])
    narrowed = view(client, viewer)
    assert {point["id"]: point["ate_rmse_mean"] for point in narrowed["points"]} == {baseline_id: 1, own_id: 1.5}
    assert narrowed["error_reference"] == pytest.approx(1.2) and narrowed["claimed_area"] == 0


def test_zero_baseline_error_still_allows_fuel_improvements_at_the_error_floor(comparison):
    store, client = comparison
    viewer = store.add_team("viewer")
    baseline = store.add_team("reference", team_type="participant")
    submit(store, baseline, scores=(0, 0), fuel=(4, 4), baseline=True)
    own = submit(store, viewer, scores=(0, 0), fuel=(0, 0))
    result = view(client, viewer)
    assert result["frontier"] == [own]
    assert result["error_reference"] == pytest.approx(0.0012)
    assert result["claimed_area"] == pytest.approx(log(4) * log(1.2))


def test_zero_and_subfloor_scores_tie_for_scoring_without_changing_raw_points(comparison):
    store, client = comparison
    viewer = store.add_team("viewer")
    baseline = store.add_team("reference", team_type="participant", unlimited=True)
    zero = submit(store, baseline, scores=(0, 0), fuel=(0, 0), baseline=True)
    subfloor = submit(store, baseline, scores=(0.0005, 0.0005), fuel=(1, 1), baseline=True)
    own = submit(store, viewer, scores=(0.0008, 0.0008), fuel=(0, 0))
    result = view(client, viewer)
    assert result["baseline_frontier"] == [zero, subfloor]
    assert result["frontier"] == [zero, subfloor, own]
    assert result["error_reference"] == pytest.approx(0.0012) and result["claimed_area"] == 0
    assert result["best_submission_id"] == own and result["best_frontier"] == [zero, subfloor, own]
    assert [(point["fuel_per_frame_max"], point["ate_rmse_mean"]) for point in result["points"]] == [
        (0, 0), (1, 0.0005), (0, 0.0008)]


def test_complementary_uploads_show_combined_frontier_but_only_best_single_area_counts(comparison):
    store, client = comparison
    baseline_owner = store.add_team("reference")
    viewer = store.add_team("viewer", unlimited=True)
    baseline = submit(store, baseline_owner, scores=(4, 4), fuel=(4, 4), baseline=True)
    winner = submit(store, viewer, scores=(1.5, 1.5), fuel=(1, 1))
    more_accurate = submit(store, viewer, scores=(1, 1), fuel=(3, 3))
    result = view(client, viewer)

    baseline_area = log(3) * log(1.2)
    winning_gain = log(12) * log(4.8 / 1.5) - baseline_area
    other_gain = log(4) * log(4.8) - baseline_area
    combined_gain = log(3) * log(4.8 / 1.5) + log(4) * log(4.8) - baseline_area
    assert combined_gain > winning_gain > other_gain
    assert {point["id"] for point in result["points"]} == {baseline, winner, more_accurate}
    assert result["frontier"] == [winner, more_accurate]
    assert result["best_submission_id"] == winner  # The later upload has lower ATE but a smaller individual gain.
    assert result["best_frontier"] == [winner]
    assert result["claimed_area"] == pytest.approx(winning_gain)

    duplicate = submit(store, viewer, scores=(1.5, 1.5), fuel=(1, 1))
    dominated = submit(store, viewer, scores=(2, 2), fuel=(2, 2))
    after = view(client, viewer)
    assert after["best_submission_id"] == winner
    assert after["best_frontier"] == [winner]
    assert after["claimed_area"] == pytest.approx(winning_gain)
    assert set(after["frontier"]) == {winner, more_accurate, duplicate}
    assert dominated in {point["id"] for point in after["points"]}


def test_exact_best_area_ties_use_creation_time_then_submission_id(comparison):
    store, client = comparison
    baseline_owner = store.add_team("reference")
    viewer = store.add_team("viewer", unlimited=True)
    submit(store, baseline_owner, scores=(4, 4), fuel=(4, 4), baseline=True)
    tied = [submit(store, viewer, scores=(2, 2), fuel=(2, 2)) for _ in range(2)]
    assert view(client, viewer)["best_submission_id"] == tied[0]
    with store._connect() as connection:
        connection.executemany("UPDATE submissions SET created = 100 WHERE id = ?", [(sid,) for sid in tied])
    assert view(client, viewer)["best_submission_id"] == min(tied)
    with store._connect() as connection:
        connection.execute("UPDATE submissions SET created = 99 WHERE id = ?", (max(tied),))
    result = view(client, viewer)
    assert result["best_submission_id"] == max(tied)
    assert result["best_frontier"] == [max(tied)]


def test_best_submission_ignores_missing_failed_and_outdated_evaluations(comparison):
    store, client = comparison
    baseline_owner = store.add_team("reference")
    viewer = store.add_team("viewer", unlimited=True)
    baseline = submit(store, baseline_owner, scores=(4, 4), fuel=(4, 4), baseline=True)
    valid = submit(store, viewer, scores=(3, 3), fuel=(3, 3))
    missing = submit(store, viewer, scores=(0.1,), fuel=(1, 1))
    outdated = submit(store, viewer, scores=(0.2, 0.2), fuel=(1, 1))
    failed = submit(store, viewer, scores=(0.3,), fuel=(1, 1))
    with store._connect() as connection:
        connection.execute("UPDATE evaluation_jobs SET evaluation_config = NULL WHERE submission_id = ?", (outdated,))
        failed_job = connection.execute("SELECT id FROM evaluation_jobs WHERE submission_id = ? AND episode_hash = ?",
                                        (failed, B)).fetchone()[0]
    start_job(store, failed_job)
    store.finish_job(failed_job, error="no report")
    result = view(client, viewer)
    assert {point["id"] for point in result["points"]} == {baseline, valid}
    assert result["best_submission_id"] == valid and result["best_frontier"] == [valid]
    assert not {missing, outdated, failed} & set(result["best_frontier"])
    assert result["claimed_area"] == pytest.approx(log(4) * log(4.8 / 3) - log(3) * log(1.2))


def test_submission_list_marks_each_owners_best_individual_gain_without_grouping_by_name(comparison):
    store, client = comparison
    reference, first, second = [store.add_team("same name", unlimited=True) for _ in range(3)]
    admin = store.add_team("instructors", team_type="admin")
    baseline = submit(store, reference, scores=(4, 4), fuel=(4, 4), baseline=True)
    first_best = submit(store, first, scores=(1.5, 1.5), fuel=(1, 1))
    first_other = submit(store, first, scores=(1, 1), fuel=(3, 3))  # Lower ATE, smaller individual area gain.
    second_other = submit(store, second, scores=(3, 3), fuel=(3, 3))
    second_best = submit(store, second, scores=(2, 2), fuel=(2, 2))
    assert listed_best(store, client, admin) == {baseline, first_best, second_best}
    for token, expected, own_ids in ((reference, baseline, {baseline}),
                                     (first, first_best, {first_best, first_other}),
                                     (second, second_best, {second_best, second_other})):
        assert listed_best(store, client, token) == {expected}
        assert view(client, token)["best_submission_id"] == expected
        response = client.get("/submissions", headers={"Authorization": f"Bearer {token}"})
        assert {row["id"] for row in response.json()} == own_ids
        assert all(secret not in response.text for secret in (reference, first, second, admin))
    admin_response = client.get("/submissions", headers={"Authorization": f"Bearer {admin}"})
    assert all(secret not in admin_response.text for secret in (reference, first, second, admin))
    assert client.get(f"/submissions/{second_best}", headers={"Authorization": f"Bearer {first}"}).status_code == 404


def test_submission_best_flags_follow_current_eligibility_selection_and_baselines(comparison):
    store, client = comparison
    reference = store.add_team("reference", unlimited=True)
    viewer = store.add_team("viewer", unlimited=True)
    baseline = submit(store, reference, scores=(4, 4), fuel=(4, 4))
    valid = submit(store, viewer, scores=(3, 3), fuel=(3, 3))
    assert listed_best(store, client, viewer) == set()  # Scores alone do not supply a baseline reference.
    store.set_baseline(baseline, True)
    assert listed_best(store, client, viewer) == {valid}
    submit(store, reference, scores=(4, None), fuel=(4, 4), baseline=True)

    partial = submit(store, viewer, scores=(0.1,), fuel=(1, 1))
    failed = submit(store, viewer, scores=(0.2,), fuel=(1, 1))
    for invalid in (-1, float("nan"), float("inf")):
        submit(store, viewer, scores=(0.3, invalid), fuel=(1, 1))
    outdated = submit(store, viewer, scores=(0.05, 0.05), fuel=(1, 1))
    with store._connect() as connection:
        connection.execute("UPDATE evaluation_jobs SET evaluation_config = NULL WHERE submission_id = ?", (outdated,))
        failed_job = connection.execute("SELECT id FROM evaluation_jobs WHERE submission_id = ? AND episode_hash = ?",
                                        (failed, B)).fetchone()[0]
    start_job(store, failed_job)
    store.finish_job(failed_job, error="no report")
    assert listed_best(store, client, viewer) == {valid}
    store.set_evaluations([A])
    assert listed_best(store, client, viewer) == {partial}
    assert view(client, viewer)["best_submission_id"] == partial
    store.set_evaluations([A, B])
    assert listed_best(store, client, viewer) == {valid}

    original = store.config
    store.set_config(replace(original, evaluation=replace(original.evaluation,
                     limits=replace(original.evaluation.limits, step_fuel=11))))
    assert listed_best(store, client, viewer) == set()  # Outdated references and candidates cannot select a winner.
    store.set_config(original)
    assert listed_best(store, client, viewer) == {valid}
    store.set_baseline(baseline, False)
    assert listed_best(store, client, viewer) == set()


def test_submission_list_zero_gain_ties_include_owned_baselines_and_use_stable_order(comparison):
    store, client = comparison
    viewer = store.add_team("viewer", unlimited=True)
    baseline = submit(store, viewer, scores=(4, 4), fuel=(4, 4), baseline=True)
    dominated = submit(store, viewer, scores=(5, 5), fuel=(5, 5))
    assert view(client, viewer)["claimed_area"] == 0
    assert listed_best(store, client, viewer) == {baseline}
    tied = [baseline, dominated]
    with store._connect() as connection:
        connection.executemany("UPDATE submissions SET created = 100 WHERE id = ?", [(sid,) for sid in tied])
    assert listed_best(store, client, viewer) == {min(tied)}
    with store._connect() as connection:
        connection.execute("UPDATE submissions SET created = 99 WHERE id = ?", (max(tied),))
    assert listed_best(store, client, viewer) == {max(tied)}
    assert view(client, viewer)["best_submission_id"] == max(tied)


def test_raw_ate_reverses_aligned_ranking_for_frontiers_area_and_best_markers(comparison):
    store, client = comparison
    reference = store.add_team("reference")
    viewer = store.add_team("viewer", unlimited=True)
    admin = store.add_team("admin", team_type="admin")
    baseline = submit(store, reference, scores=(4, 4), sim3_scores=(0.5, 0.5), fuel=(4, 4), baseline=True)
    aligned_winner = submit(store, viewer, scores=(3, 3), sim3_scores=(0.01, 0.01), fuel=(2, 2))
    raw_winner = submit(store, viewer, scores=(1, 1), sim3_scores=(0.2, 0.2), fuel=(2, 2))
    result = view(client, viewer)
    assert result["baseline_frontier"] == [baseline]
    assert result["frontier"] == result["best_frontier"] == [raw_winner]
    assert result["best_submission_id"] == raw_winner
    assert result["error_reference"] == pytest.approx(4.8)
    assert result["claimed_area"] == pytest.approx(log(6) * log(4.8) - log(3) * log(1.2))
    assert {point["id"]: point["ate_rmse_mean"] for point in result["points"]} == {
        baseline: 4, aligned_winner: 3, raw_winner: 1}
    assert all("sim3_ate_rmse_mean" not in point for point in result["points"])
    assert listed_best(store, client, viewer) == {raw_winner}
    assert listed_best(store, client, admin) == {baseline, raw_winner}
    summaries = {row["id"]: row["summary"] for row in client.get(
        "/submissions", headers={"Authorization": f"Bearer {viewer}"}).json()}
    assert summaries[aligned_winner]["sim3_ate_rmse_mean"] == 0.01
    assert summaries[raw_winner]["sim3_ate_rmse_mean"] == 0.2


@pytest.mark.parametrize("invalid_sim3", [None, float("nan"), float("inf"), -1])
def test_raw_scores_control_eligibility_even_when_aligned_diagnostics_are_invalid(comparison, invalid_sim3):
    store, client = comparison
    reference = store.add_team("reference")
    viewer = store.add_team("viewer", unlimited=True)
    baseline = submit(store, reference, scores=(4, 4), sim3_scores=(0.5, 0.5), fuel=(4, 4), baseline=True)
    valid = submit(store, viewer, scores=(1, 1), sim3_scores=(invalid_sim3, invalid_sim3), fuel=(2, 2))
    for invalid_raw in (None, float("nan"), float("inf"), -1):
        submit(store, viewer, scores=(1, invalid_raw), sim3_scores=(0.01, 0.01), fuel=(1, 1))
    result = view(client, viewer)
    assert {point["id"] for point in result["points"]} == {baseline, valid}
    assert result["best_submission_id"] == valid and result["best_frontier"] == [valid]
    assert listed_best(store, client, viewer) == {valid}


def test_slow_high_error_baseline_cannot_inflate_reference_area_or_best_marker(comparison):
    store, client = comparison
    reference = store.add_team("reference", unlimited=True)
    viewer = store.add_team("viewer", unlimited=True)
    admin = store.add_team("admin", team_type="admin")
    baseline = submit(store, reference, scores=(4, 4), fuel=(4, 4), baseline=True)
    fast_but_inaccurate = submit(store, viewer, scores=(5, 5), fuel=(1, 1))
    winner = submit(store, viewer, scores=(1, 1), fuel=(5, 5))
    before = view(client, viewer)
    expected_area = log(12 / 5) * log(4)
    assert before["error_reference"] == pytest.approx(4.8)
    assert before["claimed_area"] == pytest.approx(expected_area)
    assert before["best_submission_id"] == winner

    slow = submit(store, reference, scores=(100, 100), fuel=(9, 9), baseline=True)
    after = view(client, viewer)
    assert {point["id"] for point in after["points"]} == {baseline, fast_but_inaccurate, winner, slow}
    assert after["error_reference"] == before["error_reference"]
    assert after["claimed_area"] == pytest.approx(expected_area)
    assert after["best_submission_id"] == winner and after["best_frontier"] == [baseline, winner]
    assert listed_best(store, client, viewer) == {winner}
    assert listed_best(store, client, admin) == {baseline, winner}

    faster = submit(store, reference, scores=(8, 8), fuel=(2, 2), baseline=True)
    changed = view(client, viewer)
    assert changed["error_reference"] == pytest.approx(9.6)  # The anchor changes when an actually faster baseline arrives.
    assert changed["baseline_frontier"] == [faster, baseline]
    assert changed["claimed_area"] == pytest.approx(expected_area)
    assert changed["best_submission_id"] == winner
    assert listed_best(store, client, viewer) == {winner}


@pytest.mark.parametrize("reverse_order", [False, True])
def test_fastest_baseline_fuel_ties_use_lower_error_independently_of_upload_order(comparison, reverse_order):
    store, client = comparison
    reference = store.add_team("reference", unlimited=True)
    viewer = store.add_team("viewer")
    tied = {}
    for error in ((2, 8) if reverse_order else (8, 2)):
        tied[error] = submit(store, reference, scores=(error, error), fuel=(4, 4), baseline=True)
    slower = submit(store, reference, scores=(0.5, 0.5), fuel=(6, 6), baseline=True)
    own = submit(store, viewer, scores=(1, 1), fuel=(2, 2))
    result = view(client, viewer)
    assert result["error_reference"] == pytest.approx(2.4)
    assert result["baseline_frontier"] == [tied[2], slower]
    assert result["best_submission_id"] == own
    assert listed_best(store, client, viewer) == {own}


def test_fastest_baseline_ties_use_plotted_fuel_floor_before_raw_speed(comparison):
    store, client = comparison
    reference = store.add_team("reference", unlimited=True)
    viewer = store.add_team("viewer")
    submit(store, reference, scores=(8, 8), fuel=(0, 0), baseline=True)
    submit(store, reference, scores=(4, 4), fuel=(0, 0), baseline=True)
    leftmost = submit(store, reference, scores=(2, 2), fuel=(1, 1), baseline=True)
    slower = submit(store, reference, scores=(0.1, 0.1), fuel=(2, 2), baseline=True)
    result = view(client, viewer)
    assert result["fuel_floor"] == 1 and result["error_reference"] == pytest.approx(2.4)
    assert result["baseline_frontier"] == [leftmost, slower]
    assert [(point["fuel_per_frame_max"], point["ate_rmse_mean"]) for point in result["points"]] == [
        (0, 8), (0, 4), (1, 2), (2, 0.1)]


def test_fastest_baseline_reference_excludes_incomplete_and_outdated_results(comparison):
    store, client = comparison
    reference = store.add_team("reference", unlimited=True)
    viewer = store.add_team("viewer")
    admin = store.add_team("admin", team_type="admin")
    baseline = submit(store, reference, scores=(4, 4), fuel=(4, 4), baseline=True)
    incomplete = submit(store, reference, scores=(1,), fuel=(1, 1), baseline=True)
    outdated = submit(store, reference, scores=(2, 2), fuel=(2, 2), baseline=True)
    with store._connect() as connection:
        connection.execute("UPDATE evaluation_jobs SET evaluation_config = NULL WHERE submission_id = ?", (outdated,))
    own = submit(store, viewer, scores=(3, 3), fuel=(3, 3))
    result = view(client, viewer)
    assert {point["id"] for point in result["points"]} == {baseline, own}
    assert result["error_reference"] == pytest.approx(4.8)
    assert listed_best(store, client, viewer) == {own}
    assert listed_best(store, client, admin) == {baseline, own}
    store.set_evaluations([A])
    changed = view(client, viewer)
    assert changed["error_reference"] == pytest.approx(1.2)
    assert changed["baseline_frontier"] == [incomplete]
    assert changed["claimed_area"] == 0 and changed["best_submission_id"] == own
    assert listed_best(store, client, viewer) == {own}
