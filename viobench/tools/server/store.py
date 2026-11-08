"""SQLite store shared by the web app, the worker and `teams`: the team tokens and the submission queue.

A team is its token: submissions belong to a token, and the name next to it is only a note for the instructors."""
import hashlib
import json
import os
import secrets
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import load as load_config

DEFAULT_DIR = Path(os.environ.get("VIOBENCH_SERVER_DIR", "server-data"))
TOKEN_BYTES = 24  # a token is 32 base64url characters
ID_BYTES = 8  # a submission id is 16 hex digits, random: it says nothing about how many submissions exist
COLUMNS = "id, label, filename, sha256, bytes, created, started, finished, status, error"  # what a team sees of a submission

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    token TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created REAL NOT NULL,
    unlimited INTEGER NOT NULL DEFAULT 0  -- 1: no daily quota, e.g. the instructors' debug team
);
CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY,
    token TEXT NOT NULL REFERENCES teams(token),
    label TEXT,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    created REAL NOT NULL,
    status TEXT NOT NULL,            -- queued | running | done | failed
    started REAL,
    finished REAL,
    results TEXT,                    -- JSON: the metrics the team gets back (worker.public_results)
    error TEXT
);
CREATE INDEX IF NOT EXISTS submissions_token ON submissions(token, created);
"""


class Refused(Exception):
    """The rules do not admit an upload right now; the message says why."""


def day_start(now, time_zone):
    """Unix time of the last midnight in `time_zone` (e.g. "America/Los_Angeles") at or before the Unix time `now`."""
    return datetime.fromtimestamp(now, ZoneInfo(time_zone)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


class Store:
    def __init__(self, directory=DEFAULT_DIR):
        self.directory = Path(directory)
        self.uploads = self.directory / "uploads"
        self.reports = self.directory / "reports"
        for path in (self.uploads, self.reports):
            path.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "submissions.sqlite"
        with self._connect() as connection:
            connection.executescript(SCHEMA)
        self.config = load_config()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    # --- teams ----------------------------------------------------------------------------------------------

    def add_team(self, name, unlimited=False):
        """Creates a team, without a daily quota if `unlimited`; returns its token."""
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with self._connect() as connection:
            connection.execute("INSERT INTO teams (token, name, created, unlimited) VALUES (?, ?, ?, ?)", (token, name, time.time(), unlimited))
        return token

    def teams(self):
        """[(token, name, unlimited)] in the order they were created."""
        with self._connect() as connection:
            return [(row["token"], row["name"], bool(row["unlimited"])) for row in connection.execute("SELECT token, name, unlimited FROM teams ORDER BY created")]

    def is_team(self, token):
        with self._connect() as connection:
            return connection.execute("SELECT 1 FROM teams WHERE token = ?", (token,)).fetchone() is not None

    # --- submissions, as a team sees them --------------------------------------------------------------------

    def add(self, token, label, filename, data):
        """Queues a submission and returns its id. Refused while the team has one queued or running, or once today's
        quota is used up: config.daily_quota submissions, where up to config.daily_failed_quota failed ones do not count
        (further failures count like any other submission). An unlimited team has no daily quota.

        One transaction: concurrent uploads cannot both pass these checks, and the worker cannot claim the row before
        the file is written."""
        submission_id = secrets.token_hex(ID_BYTES)
        now = time.time()
        config = self.config
        with self._connect() as connection:  # leaving it with an exception rolls back
            connection.execute("BEGIN IMMEDIATE")
            pending = connection.execute("SELECT id, status FROM submissions WHERE token = ? AND status IN ('queued', 'running')", (token,)).fetchone()
            if pending is not None:
                raise Refused(f"submission {pending['id']} is still {pending['status']}; submit again once it has finished")
            unlimited = connection.execute("SELECT unlimited FROM teams WHERE token = ?", (token,)).fetchone()["unlimited"]
            today = connection.execute("SELECT COUNT(*) AS n, COALESCE(SUM(status = 'failed'), 0) AS failed FROM submissions "
                                       "WHERE token = ? AND created >= ?", (token, day_start(now, config.quota_time_zone))).fetchone()
            counted = today["n"] - today["failed"] + max(0, today["failed"] - config.daily_failed_quota)
            if counted >= config.daily_quota and not unlimited:
                raise Refused(f"today's quota of {config.daily_quota} submission(s) is used up (up to {config.daily_failed_quota} "
                              f"failed one(s) do not count); it resets at midnight ({config.quota_time_zone})")
            connection.execute(
                "INSERT INTO submissions (id, token, label, filename, sha256, bytes, created, status) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued')",
                (submission_id, token, label, filename, hashlib.sha256(data).hexdigest(), len(data), now))
            self.upload_path(submission_id).write_bytes(data)
            connection.execute("COMMIT")
        return submission_id

    def get(self, token, submission_id):
        """One of the team's submissions with its results; None when there is none (or it is another team's)."""
        with self._connect() as connection:
            row = connection.execute(f"SELECT {COLUMNS}, results FROM submissions WHERE id = ? AND token = ?", (submission_id, token)).fetchone()
        return dict(row) if row else None

    def submissions(self, token):
        """The team's submissions with their results, newest first."""
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(f"SELECT {COLUMNS}, results FROM submissions WHERE token = ? ORDER BY created DESC", (token,))]

    # --- the worker's side ----------------------------------------------------------------------------------

    def claim_next(self):
        """Atomically moves the oldest queued submission to running; returns it or None."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT submissions.*, teams.name FROM submissions JOIN teams USING (token) "
                                     "WHERE status = 'queued' ORDER BY submissions.created LIMIT 1").fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            connection.execute("UPDATE submissions SET status = 'running', started = ? WHERE id = ?", (time.time(), row["id"]))
            connection.execute("COMMIT")
        return dict(row)

    def requeue_running(self):
        """Puts submissions left running by a dead worker back in the queue; only while no worker runs. Returns how many."""
        with self._connect() as connection:
            return connection.execute("UPDATE submissions SET status = 'queued', started = NULL WHERE status = 'running'").rowcount

    def finish(self, submission_id, results=None, error=None):
        with self._connect() as connection:
            connection.execute("UPDATE submissions SET status = ?, finished = ?, results = ?, error = ? WHERE id = ?",
                               ("done" if error is None else "failed", time.time(), json.dumps(results) if results else None, error, submission_id))

    def upload_path(self, submission_id):
        return self.uploads / f"{submission_id}.wasm"

    def report_dir(self, submission_id):
        return self.reports / submission_id
