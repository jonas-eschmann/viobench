"""The submission server through FastAPI's test client: the API as the web UI uses it, and the UI's files.
No worker runs: the tests finish submissions themselves."""
import re
import sys
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from viobench.tools.server import app as server  # noqa: E402
from viobench.tools.server import teams  # noqa: E402
from viobench.tools.server.store import Store  # noqa: E402

MODULE = b"\0asm\x01\0\0\0" + bytes([1, 4, 1, 0x60, 0, 0])  # one empty function type: a module, not a Python bundle
NAN = float("nan")


@pytest.fixture
def api(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.set_evaluations(["a" * 40])
    monkeypatch.setattr(server, "_store", store)
    client = TestClient(server.app)
    client.headers["Authorization"] = f"Bearer {store.add_team('team')}"
    return client, store


def results(rpe):
    return {"hash": "a" * 40, "dataset": "val", "completed": True, "frames": 1, "frames_attempted": 1,
            "duration_s": 1, "fuel_per_frame_median": 4.1e6, "wall_per_frame_median_ms": 1,
            "metrics": {"rpe_translation_rmse": rpe, "rpe_rotation_rmse": 0, "ate_rmse": 0, "rotation_rmse": 0,
                        "velocity_rmse": 0, "angular_velocity_rmse": 0, "drift_per_metre": float("inf")}}


def finish_submission(store, submission_id, results=None, error=None):
    with store._connect() as connection:
        [job] = connection.execute("SELECT * FROM evaluation_jobs WHERE submission_id = ?", (submission_id,)).fetchall()
    if results is not None:
        directory = store.report_dir(submission_id, job["episode_hash"]) / "logs" / job["episode_hash"]
        directory.mkdir(parents=True)
        np.save(directory / "fuel.npy", np.array([4.1e6]))
    store.finish_job(job["id"], result=results, error=error)


@pytest.mark.parametrize("path, media_type", [("/", "text/html"), ("/app.js", "text/javascript"), ("/style.css", "text/css")])
def test_ui_files_are_public_and_confined(path, media_type):
    response = TestClient(server.app).get(path)
    assert response.status_code == 200 and response.headers["content-type"].startswith(media_type)
    policy = response.headers["content-security-policy"]
    assert "default-src 'none'" in policy and "script-src 'self'" in policy and "frame-ancestors 'none'" in policy
    assert response.headers["x-content-type-options"] == "nosniff"


def test_ui_never_parses_server_text_as_html():
    script = server.UI_FILES["app.js"].decode()
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function", script)
    for match in re.finditer(r'(?:src|href)="([^"]+)"', server.UI_FILES["index.html"].decode()):  # same origin, relative
        assert match.group(1) in ("style.css", "app.js", "data:,")


def test_non_finite_metrics_come_back_as_null(api):
    client, store = api
    created = client.post("/submissions", files={"file": ("x.wasm", MODULE)}, data={"label": "v1"})
    assert created.status_code == 201
    finish_submission(store, created.json()["id"], results=results(NAN))
    [listed] = client.get("/submissions").json()
    assert listed["status"] == "done" and "results" not in listed
    assert listed["overall"]["rpe_translation_rmse_median"] is None and listed["overall"]["fuel_per_frame_median"] == 4.1e6
    detail = client.get(f"/submissions/{listed['id']}")
    assert detail.status_code == 200
    assert detail.json()["results"]["episodes"][0]["metrics"]["rpe_translation_rmse"] is None
    assert detail.json()["results"]["episodes"][0]["metrics"]["drift_per_metre"] is None


def test_errors_carry_a_detail_the_ui_shows(api):
    client, store = api
    assert client.get("/submissions", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/submissions", files={"file": ("x.wasm", b"not wasm")}).json()["detail"].startswith("not a WebAssembly module")
    first = client.post("/submissions", files={"file": ("x.wasm", MODULE)}).json()
    refused = client.post("/submissions", files={"file": ("x.wasm", MODULE)})
    assert refused.status_code == 429 and "today's quota" in refused.json()["detail"]
    finish_submission(store, first["id"], error="no episode completed")
    [listed] = client.get("/submissions").json()
    assert listed["status"] == "failed" and listed["overall"] is None
    assert client.post("/submissions", files={"file": ("x.wasm", MODULE)}).status_code == 429  # Failed evaluations still count.


def test_unlimited_team_has_no_daily_quota(api):
    client, store = api
    unlimited = {"Authorization": f"Bearer {store.add_team('debug', unlimited=True)}"}
    for headers in ({}, unlimited):
        for _ in range(store.config.daily_quota):
            assert client.post("/submissions", files={"file": ("x.wasm", MODULE)}, headers=headers).status_code == 201
    assert client.post("/submissions", files={"file": ("x.wasm", MODULE)}).status_code == 429
    assert client.post("/submissions", files={"file": ("x.wasm", MODULE)}, headers=unlimited).status_code == 201
    assert [unlimited for _, _, unlimited, _, _ in store.teams()] == [False, True]


@pytest.mark.parametrize("running", [False, True])
def test_daily_quota_allows_another_upload_while_jobs_are_pending(api, running):
    client, store = api
    store.config = replace(store.config, daily_quota=2)
    first = client.post("/submissions", files={"file": ("first.wasm", MODULE)})
    assert first.status_code == 201
    if running:
        assert store.claim_next()["submission_id"] == first.json()["id"]
    second = client.post("/submissions", files={"file": ("second.wasm", MODULE)})
    assert second.status_code == 201
    assert client.post("/submissions", files={"file": ("third.wasm", MODULE)}).status_code == 429
    detail = client.get(f"/submissions/{first.json()['id']}").json()
    assert detail["status"] == ("running" if running else "queued")
    assert not {"initial_status", "started", "finished"} & detail.keys()
    assert "initial" not in detail["jobs"][0]
    assert "daily_failed_quota" not in client.get("/config").json()


def test_admin_can_read_all_submissions_without_exposing_tokens(api):
    client, store = api
    team_token = client.headers["Authorization"].removeprefix("Bearer ")
    # Neither the account name nor an unlimited quota grants admin access.
    other_token = store.add_team("admin", unlimited=True, is_participant=True)
    other = {"Authorization": f"Bearer {other_token}"}
    admin_token = store.add_team("instructors", is_admin=True)
    admin = {"Authorization": f"Bearer {admin_token}"}
    for headers, name, is_admin in [({}, "team", False), (other, "admin", False), (admin, "instructors", True)]:
        config = client.get("/config", headers=headers).json()
        assert config["team_name"] == name and config["is_admin"] is is_admin  # Even before any submissions exist.
    first = client.post("/submissions", files={"file": ("one.wasm", MODULE)}).json()["id"]
    finish_submission(store, first, results=results(0.1))
    second = client.post("/submissions", files={"file": ("two.wasm", MODULE)}, headers=other).json()["id"]

    for headers, own, foreign, name, participant in [({}, first, second, "team", False), (other, second, first, "admin", True)]:
        [row] = client.get("/submissions", headers=headers).json()
        assert row["id"] == own and row["team_name"] == name and row["is_participant"] is participant
        detail = client.get(f"/submissions/{own}", headers=headers)
        assert detail.status_code == 200
        assert detail.json()["team_name"] == name and detail.json()["is_participant"] is participant
        assert client.get(f"/submissions/{foreign}", headers=headers).status_code == 404

    listed = client.get("/submissions", headers=admin)
    assert [(row["id"], row["team_name"]) for row in listed.json()] == [(second, "admin"), (first, "team")]
    assert listed.json()[0]["is_participant"] is True and listed.json()[1]["is_participant"] is False
    assert listed.json()[1]["overall"]["rpe_translation_rmse_median"] == 0.1
    details = [client.get(f"/submissions/{submission_id}", headers=admin) for submission_id in (first, second)]
    assert all(response.status_code == 200 for response in details)
    assert [response.json()["team_name"] for response in details] == ["team", "admin"]
    assert details[0].json()["is_participant"] is False and details[1].json()["is_participant"] is True
    assert details[0].json()["results"]["summary"]["overall"]["rpe_translation_rmse_median"] == 0.1
    assert client.get("/submissions/missing", headers=admin).status_code == 404
    for response in [listed, *details, client.get("/config", headers=admin)]:
        for secret in (team_token, other_token, admin_token):
            assert secret not in response.text
    assert store.is_admin("unknown") is False
    assert store.submissions("unknown") == []
    assert store.get("unknown", first) is None


@pytest.mark.parametrize("unlimited", [False, True])
def test_admin_cannot_upload(api, unlimited):
    client, store = api
    headers = {"Authorization": f"Bearer {store.add_team('admin', unlimited=unlimited, is_admin=True)}"}
    response = client.post("/submissions", files={"file": ("x.wasm", MODULE)}, headers=headers)
    assert response.status_code == 403
    assert client.get("/submissions", headers=headers).json() == []
    assert list(store.uploads.iterdir()) == []


@pytest.mark.parametrize("authorization", [None, "Bearer wrong"])
@pytest.mark.parametrize("method,path", [("GET", "/config"), ("GET", "/submissions"),
                                         ("GET", "/submissions/missing"), ("POST", "/submissions"),
                                         ("PUT", "/submission-pause"), ("GET", "/evaluations"),
                                         ("PUT", "/evaluations"), ("POST", "/evaluations/schedule")])
def test_api_requires_a_valid_token(api, authorization, method, path):
    client, _ = api
    client.headers.pop("Authorization")
    headers = {"Authorization": authorization} if authorization else {}
    kwargs = {"files": {"file": ("x.wasm", MODULE)}} if method == "POST" else {}
    if path == "/evaluations/schedule":
        kwargs = {"json": {}}
    elif method == "PUT":
        kwargs = {"json": {"hashes": ["a" * 40]} if path == "/evaluations" else {"message": "Maintenance"}}
    response = client.request(method, path, headers=headers, **kwargs)
    assert response.status_code == 401 and response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("flags,kind,unlimited,is_admin,is_participant", [
    ([], "test", False, False, False),
    (["--admin"], "admin", False, True, False),
    (["--participant"], "participant", False, False, True),
    (["--participant", "--unlimited"], "participant", True, False, True),
])
def test_cli_creates_and_lists_account_types(api, monkeypatch, capsys, flags, kind, unlimited, is_admin, is_participant):
    _, store = api
    monkeypatch.setattr(teams, "Store", lambda: store)
    monkeypatch.setattr(sys, "argv", ["teams", "add", *flags, "account"])
    teams.main()
    token, name = capsys.readouterr().out.split()
    assert name == "account" and store.is_admin(token) is is_admin
    assert store.teams()[-1] == (token, "account", unlimited, is_admin, is_participant)
    monkeypatch.setattr(sys, "argv", ["teams", "list"])
    teams.main()
    [listed] = [line for line in capsys.readouterr().out.splitlines() if line.startswith(token)]
    assert f"({kind})" in listed
    assert ("(unlimited)" in listed) is unlimited


def test_admin_and_participant_are_mutually_exclusive(api, monkeypatch, capsys):
    _, store = api
    before = store.teams()
    with pytest.raises(ValueError):
        store.add_team("invalid", is_admin=True, is_participant=True)
    monkeypatch.setattr(teams, "Store", lambda: store)
    monkeypatch.setattr(sys, "argv", ["teams", "add", "--admin", "--participant", "invalid"])
    with pytest.raises(SystemExit) as error:
        teams.main()
    assert error.value.code == 2
    assert "--admin" in capsys.readouterr().err
    assert store.teams() == before


def test_cli_updates_roles_and_access_without_changing_the_account(api, monkeypatch):
    client, store = api
    # Duplicate names are allowed; updates must select only the exact token.
    token = store.add_team("team", unlimited=True)
    headers = {"Authorization": f"Bearer {token}"}
    foreign = client.post("/submissions", files={"file": ("other.wasm", MODULE)}).json()["id"]
    own = client.post("/submissions", files={"file": ("own.wasm", MODULE)}, headers=headers).json()["id"]
    finish_submission(store, own, results=results(0.1))
    original = client.get(f"/submissions/{own}", headers=headers).json()
    unchanged_team = store.teams()[0]
    monkeypatch.setattr(teams, "Store", lambda: store)

    for flags, admin, participant in [
        (["--participant"], False, True),
        (["--admin", "--no-participant"], True, False),
        (["--participant", "--no-admin"], False, True),
        (["--no-participant"], False, False),
    ]:
        monkeypatch.setattr(sys, "argv", ["teams", "update", token, *flags])
        teams.main()
        assert store.teams() == [unchanged_team, (token, "team", True, admin, participant)]
        assert client.get("/config", headers=headers).json()["is_admin"] is admin
        assert client.get(f"/submissions/{own}", headers=headers).json() == {**original, "is_participant": participant}
        visible = {row["id"] for row in client.get("/submissions", headers=headers).json()}
        assert visible == ({own, foreign} if admin else {own})
        assert client.get(f"/submissions/{foreign}", headers=headers).status_code == (200 if admin else 404)
        if admin:
            assert client.post("/submissions", files={"file": ("x.wasm", MODULE)}, headers=headers).status_code == 403
    # Revocation restores uploads immediately, using the same token and existing client.
    assert client.post("/submissions", files={"file": ("x.wasm", MODULE)}, headers=headers).status_code == 201


def test_invalid_role_updates_leave_accounts_unchanged(api, monkeypatch, capsys):
    _, store = api
    token = store.add_team("participant", is_participant=True)
    before = store.teams()
    monkeypatch.setattr(teams, "Store", lambda: store)
    for selected, options, flags in [
        ("unknown-token", {"is_admin": True}, ["--admin"]),
        (token, {}, []),
        (token, {"is_admin": True}, ["--admin"]),  # Omitted participant flag must retain True.
    ]:
        with pytest.raises(ValueError) as error:
            store.update_team(selected, **options)
        assert selected not in str(error.value)
        monkeypatch.setattr(sys, "argv", ["teams", "update", selected, *flags])
        with pytest.raises(SystemExit) as error:
            teams.main()
        assert error.value.code == 2
        assert selected not in capsys.readouterr().err
        assert store.teams() == before


def test_pause_control_requires_admin_and_a_valid_message(api):
    client, store = api
    admin = {"Authorization": f"Bearer {store.add_team('admin', is_admin=True)}"}
    assert client.get("/config").json()["pause_message"] is None
    assert client.put("/submission-pause", json={"message": "Maintenance"}).status_code == 403
    assert store.pause_message() is None
    assert client.put("/submission-pause", json={"message": "Maintenance"}, headers=admin).status_code == 200
    assert client.put("/submission-pause", json={"message": None}).status_code == 403
    for body, status in [({"message": "   \n"}, 400), ({"message": "x" * 1001}, 422),
                         ({"message": 7}, 422), ({}, 422)]:
        assert client.put("/submission-pause", json=body, headers=admin).status_code == status
        assert store.pause_message() == "Maintenance"
    assert client.post("/submissions", files={"file": ("x.wasm", MODULE)}, headers=admin).status_code == 403


def test_pause_persists_and_blocks_uploads_until_resumed(api, monkeypatch):
    client, store = api
    admin = {"Authorization": f"Bearer {store.add_team('admin', is_admin=True)}"}
    unlimited = {"Authorization": f"Bearer {store.add_team('debug', unlimited=True)}"}
    queued_token = store.add_team("already queued")
    queued_headers = {"Authorization": f"Bearer {queued_token}"}
    queued = store.add(queued_token, "queued", "queued.wasm", MODULE)

    async def unexpected_read(*args, **kwargs):
        pytest.fail("a paused upload must be rejected before reading the uploaded file")

    for message in ("Maintenance", "Maintenance: please try again tomorrow"):
        response = client.put("/submission-pause", json={"message": f"  {message}  "}, headers=admin)
        assert response.status_code == 200 and response.json() == {"pause_message": message}
        assert Store(store.directory).pause_message() == message
        for headers in ({}, unlimited, admin):
            assert client.get("/config", headers=headers).json()["pause_message"] == message
        with monkeypatch.context() as patch:
            patch.setattr("starlette.datastructures.UploadFile.read", unexpected_read)
            for headers in ({}, unlimited):
                rejected = client.post("/submissions", files={"file": ("x.wasm", b"not wasm")}, headers=headers)
                assert rejected.status_code == 503 and rejected.json()["detail"] == message
        assert [row["id"] for row in client.get("/submissions", headers=admin).json()] == [queued]
        assert client.get(f"/submissions/{queued}", headers=queued_headers).status_code == 200
        assert list(store.uploads.iterdir()) == [store.upload_path(queued)]

    assert store.claim_next()["submission_id"] == queued  # Pausing uploads does not stop work already queued.
    resumed = client.put("/submission-pause", json={"message": None}, headers=admin)
    assert resumed.status_code == 200 and resumed.json() == {"pause_message": None}
    assert Store(store.directory).pause_message() is None
    assert client.get("/config").json()["pause_message"] is None
    for headers in ({}, unlimited):
        assert client.post("/submissions", files={"file": ("x.wasm", MODULE)}, headers=headers).status_code == 201


def test_pause_after_upload_precheck_still_prevents_enqueue(api, monkeypatch):
    client, store = api
    add = store.add

    def pause_then_add(*args, **kwargs):
        store.set_pause_message("Maintenance began during upload")
        return add(*args, **kwargs)

    monkeypatch.setattr(store, "add", pause_then_add)
    response = client.post("/submissions", files={"file": ("x.wasm", MODULE)})
    assert response.status_code == 503 and response.json()["detail"] == "Maintenance began during upload"
    assert client.get("/submissions").json() == []
    assert list(store.uploads.iterdir()) == []


def test_evaluation_controls_require_admin(api):
    client, store = api
    before = store.evaluations()
    for method, path, body in [("GET", "/evaluations", None), ("PUT", "/evaluations", {"hashes": ["b" * 40]}),
                               ("POST", "/evaluations/schedule", {})]:
        assert client.request(method, path, json=body).status_code == 403
    assert store.evaluations() == before


def test_admin_selects_custom_hashes_and_schedules_missing_jobs(api):
    client, store = api
    admin = {"Authorization": f"Bearer {store.add_team('instructors', is_admin=True)}"}
    submission = client.post("/submissions", files={"file": ("x.wasm", MODULE)}).json()["id"]
    catalog = client.get("/evaluations", headers=admin).json()
    assert set(catalog) == {"hashes", "presets"}
    assert catalog["presets"][0]["name"] == "public" and catalog["presets"][0]["hashes"]
    selected = client.put("/evaluations", json={"hashes": ["B" * 40]}, headers=admin)
    assert selected.status_code == 200 and selected.json()["hashes"] == ["b" * 40]
    assert set(selected.json()) == {"hashes", "presets", "scheduled"}
    assert selected.json()["scheduled"] == 0
    assert client.get("/config").json()["evaluation_hashes"] == ["b" * 40]
    detail = client.get(f"/submissions/{submission}").json()
    assert detail["status"] == "incomplete"
    assert detail["progress"]["missing"] == 1 and detail["jobs"][0]["selected"] is False
    for expected in (1, 0):
        response = client.post("/evaluations/schedule", json={"submission_id": submission, "hashes": ["c" * 40]}, headers=admin)
        assert response.status_code == 200 and response.json()["scheduled"] == expected
    assert store.evaluation_hashes() == ["b" * 40]  # Ad hoc evaluation does not change the current set.
    for expected in (1, 0):
        response = client.put("/evaluations", json={"hashes": ["b" * 40], "backfill": True}, headers=admin)
        assert response.json()["scheduled"] == expected
    detail = client.get(f"/submissions/{submission}").json()
    assert detail["progress"]["queued"] == 1
    assert [job["episode_hash"] for job in detail["jobs"]] == ["a" * 40, "c" * 40, "b" * 40]
    assert client.post("/evaluations/schedule", json={}, headers=admin).json() == {"scheduled": 0}
    assert client.post("/evaluations/schedule", json={"submission_id": "missing"}, headers=admin).status_code == 404


@pytest.mark.parametrize("body,status", [({"hashes": []}, 400), ({"hashes": ["x" * 40]}, 400),
    ({"hashes": ["a" * 40, "A" * 40]}, 400), ({"hashes": [7]}, 422), ({"hashes": "a" * 40}, 422),
    ({"hashes": ["a" * 40], "backfill": "false"}, 422), ({"preset": "public"}, 422)])
def test_invalid_evaluation_selection_does_not_change_state(api, body, status):
    client, store = api
    admin = {"Authorization": f"Bearer {store.add_team('instructors', is_admin=True)}"}
    before = store.evaluations()
    assert client.put("/evaluations", json=body, headers=admin).status_code == status
    assert store.evaluations() == before
    # A misspelled scope must not silently schedule jobs for every submission.
    assert client.post("/evaluations/schedule", json={"submission": "missing"}, headers=admin).status_code == 422
