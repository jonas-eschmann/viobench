"""Explicit migration copies legacy baseline accounts into submission flags without losing evidence."""
import hashlib
import json
from pathlib import Path
import sqlite3
import stat

import numpy as np
import pytest

from tests.test_server import MODULE, results
from viobench.tools.server.config import load
from viobench.tools.server.migrate import migrate
from viobench.tools.server.store import Store


LEGACY_SCHEMA = """
CREATE TABLE teams (
    id INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE, name TEXT NOT NULL, created REAL NOT NULL,
    unlimited INTEGER NOT NULL DEFAULT 0, daily_quota INTEGER CHECK (daily_quota >= 0),
    extra_uploads INTEGER NOT NULL DEFAULT 0 CHECK (extra_uploads >= 0), extra_day REAL,
    type TEXT NOT NULL DEFAULT 'test' CHECK (type IN ('test','participant','baseline','admin'))
);
CREATE TABLE submissions (
    id TEXT PRIMARY KEY, token TEXT NOT NULL REFERENCES teams(token), label TEXT,
    filename TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, created REAL NOT NULL
);
CREATE INDEX submissions_token ON submissions(token, created);
CREATE TABLE submission_control (
    id INTEGER PRIMARY KEY CHECK (id=1), pause_message TEXT, evaluation_hashes TEXT NOT NULL, config TEXT NOT NULL
);
CREATE TABLE evaluation_jobs (
    id INTEGER PRIMARY KEY, submission_id TEXT NOT NULL REFERENCES submissions(id), episode_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','done','failed')),
    created REAL NOT NULL, started REAL, finished REAL, result TEXT, error TEXT, evaluation_config TEXT,
    UNIQUE(submission_id,episode_hash)
);
CREATE INDEX evaluation_queue ON evaluation_jobs(status,id);
"""
HASH = "a" * 40
BASELINES = ("1" * 16, "2" * 16, "3" * 16)
PRIVATE = "4" * 16


def files(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


@pytest.fixture
def legacy(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    config = load().as_dict()
    config["uploads"]["daily_quota"] = 9
    config["evaluation"]["episode_timeout_seconds"] += 17
    done = results(0.25)
    with sqlite3.connect(source / "submissions.sqlite") as connection:
        connection.executescript(LEGACY_SCHEMA)
        connection.execute("INSERT INTO submission_control VALUES (1,?,?,?)",
                           ("Migration maintenance", json.dumps([HASH]), json.dumps(config)))
        connection.execute("INSERT INTO teams VALUES (1,'reference-token','references',1,1,7,3,100,'baseline')")
        connection.execute("INSERT INTO teams VALUES (2,'participant-token','baseline',2,0,NULL,0,NULL,'participant')")
        connection.execute("INSERT INTO teams VALUES (3,'admin-token','instructors',3,0,NULL,0,NULL,'admin')")
        for index, submission_id in enumerate((*BASELINES, PRIVATE), start=1):
            token = "reference-token" if submission_id in BASELINES else "participant-token"
            connection.execute("INSERT INTO submissions VALUES (?,?,?,?,?,?,?)",
                               (submission_id, token, f"solution {index}", f"solution-{index}.wasm",
                                hashlib.sha256(MODULE).hexdigest(), len(MODULE), index))
        connection.execute("INSERT INTO evaluation_jobs VALUES (10,?,?, 'done',1,2,3,?,NULL,?)",
                           (BASELINES[0], HASH, json.dumps(done), json.dumps(config["evaluation"])))
        connection.execute("INSERT INTO evaluation_jobs VALUES (11,?,?, 'failed',1,2,3,NULL,?,?)",
                           (BASELINES[1], HASH, "old failure", json.dumps(config["evaluation"])))
        for job_id, submission_id in ((12, BASELINES[2]), (13, PRIVATE)):
            connection.execute("INSERT INTO evaluation_jobs VALUES (?,?,?, 'queued',1,NULL,NULL,NULL,NULL,NULL)",
                               (job_id, submission_id, HASH))
    (source / "uploads").mkdir()
    for submission_id in (*BASELINES, PRIVATE):
        (source / "uploads" / f"{submission_id}.wasm").write_bytes(MODULE)
    reports = source / "reports" / BASELINES[0] / HASH
    logs = reports / "logs" / HASH
    logs.mkdir(parents=True)
    np.save(logs / "fuel.npy", np.array([4.1e6]))
    (reports / "report.json").write_text(json.dumps({"episodes": [done]}))
    (reports / "worker.txt").write_text("private evaluator output")
    (source / "submissions.sqlite.backup").write_bytes(b"unrelated backup to preserve")
    (source / "submissions.sqlite").chmod(0o600)
    return source, config


def test_migration_preserves_accounts_jobs_settings_and_artifacts_and_marks_all_baselines(legacy, tmp_path):
    source, config = legacy
    before = files(source)
    with sqlite3.connect(source / "submissions.sqlite") as connection:
        old_jobs = connection.execute("SELECT * FROM evaluation_jobs ORDER BY id").fetchall()
        old_submissions = connection.execute("SELECT * FROM submissions ORDER BY id").fetchall()
    destination = tmp_path / "migrated"
    migrate(source, destination)
    assert files(source) == before
    assert stat.S_IMODE((destination / "submissions.sqlite").stat().st_mode) == 0o600
    store = Store(destination)
    assert store.config.as_dict() == config and store.evaluation_hashes() == [HASH]
    assert store.pause_message() == "Migration maintenance"
    assert store.teams() == [("reference-token", "references", True, "participant"),
                             ("participant-token", "baseline", False, "participant"),
                             ("admin-token", "instructors", False, "admin")]
    with store._connect() as connection:
        team = connection.execute("SELECT * FROM teams WHERE token='reference-token'").fetchone()
        assert (team["id"], team["daily_quota"], team["extra_uploads"], team["extra_day"]) == (1, 7, 3, 100)
        rows = connection.execute("SELECT * FROM submissions ORDER BY id").fetchall()
        assert [tuple(row)[:-1] for row in rows] == old_submissions
        assert [bool(row["baseline"]) for row in rows] == [True, True, True, False]
        assert [tuple(row) for row in connection.execute("SELECT * FROM evaluation_jobs ORDER BY id")] == old_jobs
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    for relative, data in before.items():
        if relative != Path("submissions.sqlite"):
            assert (destination / relative).read_bytes() == data
    points = store.comparison("participant-token")["points"]
    assert [point["id"] for point in points] == [BASELINES[0]] and points[0]["is_baseline"]
    assert store.get("participant-token", BASELINES[0]) is None
    assert store.get("reference-token", BASELINES[0])["results"]["summary"]["fuel_per_frame_max"] == 4.1e6


def test_migration_refuses_existing_destinations_and_repeat_migration(legacy, tmp_path):
    source, _ = legacy
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("keep me")
    with pytest.raises(ValueError):
        migrate(source, destination)
    assert (destination / "keep.txt").read_text() == "keep me"
    target = tmp_path / "missing-target"
    link = tmp_path / "redirect"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        migrate(source, link)
    assert link.is_symlink() and not target.exists()
    migrated = tmp_path / "migrated"
    migrate(source, migrated)
    before = files(migrated)
    repeated = tmp_path / "repeated"
    with pytest.raises(ValueError):
        migrate(migrated, repeated)
    assert files(migrated) == before and not repeated.exists()


def test_legacy_store_requires_explicit_migration_and_unknown_schema_is_rejected(legacy, tmp_path):
    source, _ = legacy
    before = files(source)
    with pytest.raises(RuntimeError, match="migrat"):
        Store(source)
    assert files(source) == before
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    with sqlite3.connect(unknown / "submissions.sqlite") as connection:
        connection.execute("CREATE TABLE arbitrary(value TEXT)")
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        migrate(unknown, destination)
    assert not destination.exists()
