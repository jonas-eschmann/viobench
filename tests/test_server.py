"""The submission server through FastAPI's test client: the API as the web UI uses it, and the UI's files.
No worker runs: the tests finish submissions themselves."""
import re

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from viobench.tools.server import app as server  # noqa: E402
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
    episode = {"dataset": "val", "name": "e0", "completed": False, "metrics": {"rpe_translation_rmse": NAN, "drift_per_metre": float("inf")}}
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
