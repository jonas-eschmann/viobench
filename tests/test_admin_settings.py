"""Admin settings, per-team daily allowances, and evaluation snapshots/reruns through the API."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from threading import Barrier

import pytest

from tests.test_evaluation_jobs import A, B, C, jobs, start_job, succeed
from tests.test_server import MODULE, api
from viobench.tools.server import store as store_module
from viobench.tools.server.store import Refused, Store


@pytest.fixture
def settings(api):
    client, store = api
    admin = {"Authorization": f"Bearer {store.add_team('instructors', team_type='admin')}"}
    return client, store, admin


def upload(client):
    response = client.post("/submissions", files={"file": ("solution.wasm", MODULE)})
    assert response.status_code == 201
    return response.json()["id"]


@pytest.mark.parametrize("authorization,status", [(None, 401), ("Bearer wrong", 401), ("regular", 403)])
def test_new_admin_controls_require_admin_authentication(settings, authorization, status):
    client, store, admin = settings
    team_id = client.get("/teams", headers=admin).json()[0]["id"]
    original = store.config.as_dict()
    if authorization == "regular":
        authorization = client.headers["Authorization"]
    client.headers.pop("Authorization")
    headers = {"Authorization": authorization} if authorization else {}
    for method, path, body in [
        ("PUT", "/config", original), ("GET", "/teams", None),
        ("PUT", f"/teams/{team_id}/allowance", {"daily_quota": 9, "unlimited": True, "extra_uploads": 5}),
        ("POST", "/evaluations/rerun-outdated", None),
    ]:
        response = client.request(method, path, headers=headers, json=body)
        assert response.status_code == status
    assert store.config.as_dict() == original
    assert store.team_allowances()[0]["daily_quota"] is None


def test_config_is_nested_persistent_and_applied_without_restarting(settings):
    client, store, admin = settings
    body = store.config.as_dict()
    body["uploads"].update(daily_quota=3, max_upload_mib=1, quota_time_zone="UTC")
    body["evaluation"]["episode_timeout_seconds"] += 1
    response = client.put("/config", json=body, headers=admin)
    assert response.status_code == 200
    assert {key: response.json()[key] for key in body} == body
    assert Store(store.directory).config.as_dict() == body
    own_config = client.get("/config").json()
    assert own_config["uploads"] == body["uploads"] and own_config["evaluation"] == body["evaluation"]
    assert own_config["allowance"]["daily_quota"] == own_config["allowance"]["remaining"] == 3
    assert client.post("/submissions", files={"file": ("large.wasm", MODULE + bytes(1 << 20))}).status_code == 413
    assert client.get("/config").json()["allowance"]["used"] == 0


@pytest.mark.parametrize("change", ["missing", "unknown", "coerced", "invalid_limit", "timezone"])
def test_invalid_config_is_rejected_atomically(settings, change):
    client, store, admin = settings
    before = store.config.as_dict()
    body = deepcopy(before)
    body["uploads"]["daily_quota"] = 9  # A valid sibling must not be committed on error.
    if change == "missing":
        del body["evaluation"]
    elif change == "unknown":
        body["evaluation"]["limit"] = 1
    elif change == "coerced":
        body["uploads"]["daily_quota"] = "9"
    elif change == "invalid_limit":
        body["evaluation"]["limits"]["step_fuel"] = 0
    else:
        body["uploads"]["quota_time_zone"] = "Not/A_Zone"
    assert client.put("/config", json=body, headers=admin).status_code == 400
    assert store.config.as_dict() == before


def test_team_allowances_are_private_persisted_and_reset_at_local_midnight(settings, monkeypatch):
    client, store, admin = settings
    now = [datetime(2026, 11, 9, 7, 59, tzinfo=timezone.utc).timestamp()]
    monkeypatch.setattr(store_module.time, "time", lambda: now[0])
    store.set_config(replace(store.config, uploads=replace(store.config.uploads, daily_quota=1,
                                                          quota_time_zone="America/Los_Angeles")))
    response = client.get("/teams", headers=admin)
    assert response.status_code == 200
    assert all(token not in response.text for token, *_ in store.teams())
    team = next(row for row in response.json() if row["name"] == "team")
    assert set(team) == {"id", "name", "type", "daily_quota", "effective_daily_quota", "used", "extra_uploads",
                         "unlimited", "remaining", "day"}
    assert team["daily_quota"] is None and team["effective_daily_quota"] == 1 and team["day"] == "2026-11-08"
    path = f"/teams/{team['id']}/allowance"
    body = {"daily_quota": 2, "unlimited": False, "extra_uploads": 1}
    changed = client.put(path, headers=admin, json=body)
    assert changed.status_code == 200 and changed.json()["remaining"] == 3
    for _ in range(3):
        upload(client)
    assert client.post("/submissions", files={"file": ("solution.wasm", MODULE)}).status_code == 429
    allowance = client.get("/config").json()["allowance"]
    assert allowance == {"daily_quota": 2, "used": 3, "extra_uploads": 1, "remaining": 0,
                          "unlimited": False, "day": "2026-11-08"}
    assert next(row for row in Store(store.directory).team_allowances() if row["id"] == team["id"])["used"] == 3
    assert client.put(path, headers=admin, json=body | {"unlimited": True}).json()["remaining"] is None
    upload(client)
    now[0] += 120
    reset = client.get("/config").json()["allowance"]
    assert reset == {"daily_quota": 2, "used": 0, "extra_uploads": 0, "remaining": None,
                     "unlimited": True, "day": "2026-11-09"}
    inherited = client.put(path, headers=admin, json={"daily_quota": None, "unlimited": False, "extra_uploads": 0}).json()
    assert inherited["daily_quota"] is None and inherited["effective_daily_quota"] == inherited["remaining"] == 1
    config = store.config.as_dict()
    config["uploads"]["daily_quota"] = 4
    assert client.put("/config", headers=admin, json=config).status_code == 200
    assert client.get("/config").json()["allowance"]["remaining"] == 4


@pytest.mark.parametrize("changes", [{"daily_quota": True}, {"unlimited": "false"}, {"extra_uploads": -1}, {"extra": 1}])
def test_invalid_team_allowance_does_not_change_account(settings, changes):
    client, store, admin = settings
    before = store.team_allowances()
    body = {"daily_quota": None, "unlimited": False, "extra_uploads": 0} | changes
    assert client.put(f"/teams/{before[0]['id']}/allowance", headers=admin, json=body).status_code == 422
    assert store.team_allowances() == before
    assert client.put("/teams/999999/allowance", headers=admin,
                      json={"daily_quota": None, "unlimited": False, "extra_uploads": 0}).status_code == 404


def test_concurrent_uploads_respect_override_and_extra_allowance(settings):
    client, store, admin = settings
    token = client.headers["Authorization"].removeprefix("Bearer ")
    team_id = next(row["id"] for row in store.team_allowances() if row["name"] == "team")
    store.set_config(replace(store.config, uploads=replace(store.config.uploads, daily_quota=0)))
    store.set_team_allowance(team_id, daily_quota=1, unlimited=False, extra_uploads=1)
    barrier = Barrier(4)

    def attempt(_):
        barrier.wait(timeout=5)
        try:
            return store.add(token, "", "solution.wasm", MODULE)
        except Refused:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(attempt, range(4)))
    assert sum(value is not None for value in accepted) == 2
    allowance = client.get("/config").json()["allowance"]
    assert allowance["used"] == 2 and allowance["remaining"] == 0


def test_evaluation_snapshots_survive_settings_changes_and_exclude_outdated_scores(settings):
    client, store, admin = settings
    token = client.headers["Authorization"].removeprefix("Bearer ")
    own = upload(client)
    own_job = store.claim_next()
    succeed(store, own, own_job)
    baseline = store.add_team("reference", team_type="baseline")
    baseline_id = store.add(baseline, "", "solution.wasm", MODULE)
    baseline_job = store.claim_next()
    succeed(store, baseline_id, baseline_job)
    original = store.config.as_dict()
    assert own_job["evaluation_config"] == original["evaluation"]
    assert len(client.get("/comparison").json()["points"]) == 2

    uploads_only = deepcopy(original)
    uploads_only["uploads"]["daily_quota"] += 1
    assert client.put("/config", headers=admin, json=uploads_only).status_code == 200
    assert client.get(f"/submissions/{own}").json()["results"]["outdated"] is False
    assert len(client.get("/comparison").json()["points"]) == 2
    changed = deepcopy(uploads_only)
    changed["evaluation"]["episode_timeout_seconds"] += 1
    assert client.put("/config", headers=admin, json=changed).status_code == 200
    detail = client.get(f"/submissions/{own}").json()
    assert detail["results"]["outdated"] and detail["progress"]["outdated"] == 1
    assert detail["jobs"][0]["evaluation_config"] == original["evaluation"]
    assert client.get("/submissions").json()[0]["outdated"] is True
    assert client.get("/comparison").json()["points"] == []
    assert client.get("/evaluations", headers=admin).json()["outdated_jobs"] == 2
    reopened = Store(store.directory)
    assert reopened.config.as_dict() == changed and reopened.get(token, own)["results"]["outdated"]

    assert client.put("/config", headers=admin, json=uploads_only).status_code == 200
    assert len(client.get("/comparison").json()["points"]) == 2
    assert client.get("/evaluations", headers=admin).json()["outdated_jobs"] == 0
    with store._connect() as connection:
        connection.execute("UPDATE evaluation_jobs SET evaluation_config = NULL WHERE id = ?", (baseline_job["id"],))
    comparison = client.get("/comparison").json()
    assert [point["id"] for point in comparison["points"]] == [own]
    assert comparison["baseline_frontier"] == [] and comparison["error_reference"] is None
    assert store.get(baseline, baseline_id)["results"]["outdated"] is True


def test_bulk_rerun_replaces_only_selected_outdated_terminal_jobs_at_queue_tail(settings):
    client, store, admin = settings
    tokens = [store.add_team(name, team_type=kind) for name, kind in
              [("reference", "baseline"), ("failed", "test"), ("running", "participant"), ("queued", "test")]]
    baseline, failed, running, queued = [store.add(token, "", "solution.wasm", MODULE) for token in tokens]
    done_job = store.claim_next()
    succeed(store, baseline, done_job)
    failed_job = store.claim_next()
    store.finish_job(failed_job["id"], error="previous failure")
    running_job = store.claim_next()
    store.schedule_evaluations([B], submission_id=baseline)
    unselected = jobs(store, tokens[0], baseline)[B]
    start_job(store, unselected["id"])
    store.finish_job(unselected["id"], error="outside current set")
    store.set_evaluations([A, C])  # C is missing; rerun is deliberately separate from scheduling missing jobs.
    changed = store.config.as_dict()
    changed["evaluation"]["limits"]["step_fuel"] += 1
    client.put("/config", headers=admin, json=changed).raise_for_status()
    counts = client.get("/evaluations", headers=admin).json()
    assert counts["outdated_jobs"] == 2 and counts["outdated_running"] == 1
    assert store.get(tokens[2], running)["progress"]["outdated"] == 0
    path = "/evaluations/rerun-outdated"
    assert client.post(path, headers=admin).json() == {"scheduled": 2, "skipped_running": 1}
    assert client.post(path, headers=admin).json() == {"scheduled": 0, "skipped_running": 1}
    assert jobs(store, tokens[0], baseline)[B]["id"] == unselected["id"]
    assert all(store.get(token, sid)["progress"]["missing"] == 1 for token, sid in zip(tokens, [baseline, failed, running, queued]))
    assert all(row["used"] == 1 for row in store.team_allowances() if row["name"] in {"reference", "failed", "running", "queued"})
    next_jobs = [store.claim_next() for _ in range(3)]
    assert [job["submission_id"] for job in next_jobs] == [queued, baseline, failed]
    assert all(job["evaluation_config"] == changed["evaluation"] for job in next_jobs)
    assert jobs(store, tokens[2], running)[A]["evaluation_config"] == running_job["evaluation_config"]
    succeed(store, running, running_job)
    assert store.get(tokens[2], running)["results"]["outdated"] is True
    assert client.post(path, headers=admin).json() == {"scheduled": 1, "skipped_running": 0}
    assert store.claim_next()["submission_id"] == running
    assert client.post(path, headers=admin).json() == {"scheduled": 0, "skipped_running": 0}
