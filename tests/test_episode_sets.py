"""Admin presets preserve neutral private labels, canonical datasets, and the public default."""
import pytest

from viobench import episodes as published
from viobench.tools.datasets import sources


def test_private_catalog_is_neutral_and_combined_preset_preserves_canonical_datasets():
    entries = published.private
    assert len(entries) == 14
    assert len({entry["hash"] for entry in entries}) == 14
    assert all(sources.reference_hash(entry) == entry["hash"] for entry in entries)
    assert [entry["description"] for entry in entries] == [f"private/episode-{index:04d}.zip" for index in range(14)]
    listed = sources._listed()
    for name, entries in [("public", published.public), ("private", published.private)]:
        assert all(listed[entry["hash"]] == (name, entry) for entry in entries)


def test_admin_combined_preset_keeps_order_without_changing_fresh_default(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    from viobench.tools.server import app as server
    from viobench.tools.server.store import Store

    store = Store(tmp_path)
    monkeypatch.setattr(server, "_store", store)
    client = TestClient(server.app)
    token = store.add_team("admin", team_type="admin")
    response = client.get("/evaluations", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    catalog = response.json()
    presets = {preset["name"]: preset["hashes"] for preset in catalog["presets"]}
    combined = presets["public+private-10"]
    assert len(combined) == len(set(combined)) == 20
    assert combined == [entry["hash"] for entry in published.public + published.private[:10]]
    assert presets["private"] == [entry["hash"] for entry in published.private]
    assert catalog["hashes"] == presets["public"] == store.evaluation_hashes() == [entry["hash"] for entry in published.public]
