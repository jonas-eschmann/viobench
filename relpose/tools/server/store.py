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

from .config import CONFIG_FILE, load as load_config

DEFAULT_DIR = Path(os.environ.get("RELPOSE_SERVER_DIR", "server-data"))
TOKEN_BYTES = 24  # a token is 32 base64url characters
ID_BYTES = 8  # a submission id is 16 hex digits, random: it says nothing about how many submissions exist
COLUMNS = "id, label, filename, sha256, bytes, created, started, finished, status, error"  # what a team sees of a submission

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    token TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created REAL NOT NULL
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
        self.config = load_config(self.directory / CONFIG_FILE)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    # --- teams ----------------------------------------------------------------------------------------------

    def add_team(self, name):
        """Creates a team; returns its token."""
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with self._connect() as connection:
            connection.execute("INSERT INTO teams (token, name, created) VALUES (?, ?, ?)", (token, name, time.time()))
        return token

    def teams(self):
        """[(token, name)] in the order they were created."""
        with self._connect() as connection:
            return [tuple(row) for row in connection.execute("SELECT token, name FROM teams ORDER BY created")]

    def is_team(self, token):
        with self._connect() as connection:
            return connection.execute("SELECT 1 FROM teams WHERE token = ?", (token,)).fetchone() is not None

    # --- submissions, as a team sees them --------------------------------------------------------------------

    def add(self, token, label, filename, data):
        """Queues a submission; returns its id, or None when the team has used up today's quota (config.daily_quota).

        One transaction: concurrent uploads cannot both pass the quota, and the worker cannot claim the row before the
        file is written."""
        submission_id = secrets.token_hex(ID_BYTES)
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            today = connection.execute("SELECT COUNT(*) AS n FROM submissions WHERE token = ? AND created >= ?",
                                       (token, day_start(now, self.config.quota_time_zone))).fetchone()
            if today["n"] >= self.config.daily_quota:
                connection.execute("ROLLBACK")
                return None
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
        """The team's submissions, newest first, without results."""
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(f"SELECT {COLUMNS} FROM submissions WHERE token = ? ORDER BY created DESC", (token,))]

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
