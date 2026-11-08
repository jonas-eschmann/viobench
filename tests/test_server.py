"""The submission server through FastAPI's test client: the API as the web UI uses it, and the UI's files.
No worker runs: the tests finish submissions themselves."""
import re
import sqlite3
import sys

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
    monkeypatch.setattr(server, "_store", store)
    client = TestClient(server.app)
    client.headers["Authorization"] = f"Bearer {store.add_team('team')}"
    return client, store


def results(rpe):
    overall = {"episodes": 2, "completed": 1, "rpe_translation_rmse_median": rpe, "fuel_per_frame_median": 4.1e6}
    episode = {"hash": "0" * 40, "dataset": "val", "completed": False, "metrics": {"rpe_translation_rmse": NAN, "drift_per_metre": float("inf")}}
    return {"summary": {"overall": overall, "datasets": {"val": overall}}, "episodes": [episode]}


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
    store.finish(created.json()["id"], results=results(NAN))
    [listed] = client.get("/submissions").json()
    assert listed["status"] == "done" and "results" not in listed
    assert listed["overall"]["rpe_translation_rmse_median"] is None and listed["overall"]["fuel_per_frame_median"] == 4.1e6
    detail = client.get(f"/submissions/{listed['id']}")
    assert detail.status_code == 200
    assert detail.json()["results"]["episodes"][0]["metrics"] == {"rpe_translation_rmse": None, "drift_per_metre": None}


def test_errors_carry_a_detail_the_ui_shows(api):
    client, store = api
    assert client.get("/submissions", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/submissions", files={"file": ("x.wasm", b"not wasm")}).json()["detail"].startswith("not a WebAssembly module")
    first = client.post("/submissions", files={"file": ("x.wasm", MODULE)}).json()
    refused = client.post("/submissions", files={"file": ("x.wasm", MODULE)})
    assert refused.status_code == 429 and first["id"] in refused.json()["detail"]
    store.finish(first["id"], error="no episode completed")
    [listed] = client.get("/submissions").json()
    assert listed["status"] == "failed" and listed["overall"] is None


def test_unlimited_team_has_no_daily_quota(api):
    client, store = api
    unlimited = {"Authorization": f"Bearer {store.add_team('debug', unlimited=True)}"}
    for headers in ({}, unlimited):
        for _ in range(store.config.daily_quota):
            store.finish(client.post("/submissions", files={"file": ("x.wasm", MODULE)}, headers=headers).json()["id"], results=results(0.1))
    assert client.post("/submissions", files={"file": ("x.wasm", MODULE)}).status_code == 429
    assert client.post("/submissions", files={"file": ("x.wasm", MODULE)}, headers=unlimited).status_code == 201
    assert [unlimited for _, _, unlimited, _, _ in store.teams()] == [False, True]


def test_admin_can_read_all_submissions_without_exposing_tokens(api):
    client, store = api
    team_token = client.headers["Authorization"].removeprefix("Bearer ")
    # Neither the account name nor an unlimited quota grants admin access.
    other_token = store.add_team("admin", unlimited=True, is_participant=True)
    other = {"Authorization": f"Bearer {other_token}"}
    admin_token = store.add_team("instructors", is_admin=True)
    admin = {"Authorization": f"Bearer {admin_token}"}
    first = client.post("/submissions", files={"file": ("one.wasm", MODULE)}).json()["id"]
    store.finish(first, results=results(0.1))
    second = client.post("/submissions", files={"file": ("two.wasm", MODULE)}, headers=other).json()["id"]

    assert client.get("/config").json()["is_admin"] is False
    assert client.get("/config", headers=other).json()["is_admin"] is False
    assert client.get("/config", headers=admin).json()["is_admin"] is True
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
                                         ("PUT", "/submission-pause")])
def test_api_requires_a_valid_token(api, authorization, method, path):
    client, _ = api
    client.headers.pop("Authorization")
    headers = {"Authorization": authorization} if authorization else {}
    kwargs = {"files": {"file": ("x.wasm", MODULE)}} if method == "POST" else {}
    if method == "PUT":
        kwargs = {"json": {"message": "Maintenance"}}
    response = client.request(method, path, headers=headers, **kwargs)
    assert response.status_code == 401 and response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("has_unlimited", [False, True])
def test_existing_database_is_upgraded_without_losing_data(tmp_path, has_unlimited):
    # The pre-merge schema has no account flags; later databases may already grant unlimited quotas.
    with sqlite3.connect(tmp_path / "submissions.sqlite") as connection:
        connection.executescript("""
            CREATE TABLE teams (token TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE submissions (
                id TEXT PRIMARY KEY, token TEXT NOT NULL REFERENCES teams(token), label TEXT,
                filename TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, created REAL NOT NULL,
                status TEXT NOT NULL, started REAL, finished REAL, results TEXT, error TEXT);
            INSERT INTO teams VALUES ('existing-token', 'admin', 1);
            INSERT INTO submissions (id, token, label, filename, sha256, bytes, created, status, error)
                VALUES ('existing-id', 'existing-token', 'v1', 'x.wasm', 'hash', 8, 2, 'failed', 'old error');
        """)
        if has_unlimited:
            connection.execute("ALTER TABLE teams ADD COLUMN unlimited INTEGER NOT NULL DEFAULT 0")
            connection.execute("UPDATE teams SET unlimited = 1 WHERE token = 'existing-token'")
    for _ in range(2):  # Restarting the app/worker must not repeat the migration or alter existing accounts.
        store = Store(tmp_path)
        assert store.teams() == [("existing-token", "admin", has_unlimited, False, False)]
        assert store.is_team("existing-token") and not store.is_admin("existing-token")
        [submission] = store.submissions("existing-token")
        assert submission["id"] == "existing-id" and submission["error"] == "old error"
        assert store.get("existing-token", "existing-id") == submission
    admin = store.add_team("instructors", is_admin=True)
    assert store.get(admin, "existing-id")["team_name"] == "admin"
    participant = store.add_team("participant", is_participant=True)
    assert Store(tmp_path).teams()[-1] == (participant, "participant", False, False, True)


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
    store.finish(own, results=results(0.1))
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

    assert store.claim_next()["id"] == queued  # Pausing uploads does not stop work already queued.
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
