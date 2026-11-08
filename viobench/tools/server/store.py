"""SQLite store shared by the web app, the worker and `teams`: the team tokens and the submission queue.

A team is its token: submissions belong to a token. Participants are marked separately from test accounts;
admin accounts can see every team's submissions and pause new uploads."""
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
MAX_PAUSE_MESSAGE = 1000
COLUMNS = ("id, label, filename, sha256, bytes, submissions.created, started, finished, status, error, "
           "results, teams.name AS team_name, teams.is_participant")  # API fields; never include tokens

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    token TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created REAL NOT NULL,
    unlimited INTEGER NOT NULL DEFAULT 0,  -- 1: no daily quota, e.g. the instructors' debug team
    is_admin INTEGER NOT NULL DEFAULT 0,   -- 1: view every team's submissions and control new uploads
    is_participant INTEGER NOT NULL DEFAULT 0  -- 1: a regular participant, rather than a test or admin account
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
CREATE TABLE IF NOT EXISTS submission_control (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    pause_message TEXT  -- NULL: accepting submissions; otherwise the message shown to teams
);
INSERT OR IGNORE INTO submission_control (id, pause_message) VALUES (1, NULL);
"""


class Refused(Exception):
    """The rules do not admit an upload right now; the message says why."""


class Paused(Refused):
    """New submissions are paused by an admin."""


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
            connection.execute("BEGIN IMMEDIATE")  # app, worker and CLI may start together
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(teams)")}
            for column in ("unlimited", "is_admin", "is_participant"):
                if column not in columns:
                    connection.execute(f"ALTER TABLE teams ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
        self.config = load_config()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    # --- teams ----------------------------------------------------------------------------------------------

    def add_team(self, name, unlimited=False, is_admin=False, is_participant=False):
        """Creates a test account, participant or admin; returns its token. `unlimited` affects quotas only."""
        if is_admin and is_participant:
            raise ValueError("admin accounts cannot be participants")
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with self._connect() as connection:
            connection.execute("INSERT INTO teams (token, name, created, unlimited, is_admin, is_participant) VALUES (?, ?, ?, ?, ?, ?)",
                               (token, name, time.time(), unlimited, is_admin, is_participant))
        return token

    def update_team(self, token, *, is_admin=None, is_participant=None):
        """Changes the specified account flags; None leaves a flag unchanged. Keeps the token and submissions."""
        if is_admin is None and is_participant is None:
            raise ValueError("specify --admin/--no-admin or --participant/--no-participant")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            team = connection.execute("SELECT is_admin, is_participant FROM teams WHERE token = ?", (token,)).fetchone()
            if team is None:
                raise ValueError("no such team")
            is_admin = team["is_admin"] if is_admin is None else is_admin
            is_participant = team["is_participant"] if is_participant is None else is_participant
            if is_admin and is_participant:
                raise ValueError("admin accounts cannot be participants; clear the other flag in the same command")
            connection.execute("UPDATE teams SET is_admin = ?, is_participant = ? WHERE token = ?", (is_admin, is_participant, token))

    def teams(self):
        """[(token, name, unlimited, is_admin, is_participant)] in the order they were created."""
        with self._connect() as connection:
            return [(row["token"], row["name"], bool(row["unlimited"]), bool(row["is_admin"]), bool(row["is_participant"]))
                    for row in connection.execute("SELECT token, name, unlimited, is_admin, is_participant FROM teams ORDER BY created")]

    def is_team(self, token):
        with self._connect() as connection:
            return connection.execute("SELECT 1 FROM teams WHERE token = ?", (token,)).fetchone() is not None

    def is_admin(self, token):
        with self._connect() as connection:
            row = connection.execute("SELECT is_admin FROM teams WHERE token = ?", (token,)).fetchone()
        return bool(row and row["is_admin"])

    # --- submission availability ---------------------------------------------------------------------------

    def pause_message(self):
        with self._connect() as connection:
            return connection.execute("SELECT pause_message FROM submission_control WHERE id = 1").fetchone()["pause_message"]

    def set_pause_message(self, message):
        """Pause with a message, or resume with None; returns the saved message."""
        if message is not None:
            message = message.strip()
            if not message or len(message) > MAX_PAUSE_MESSAGE:
                raise ValueError(f"pause message must contain 1 to {MAX_PAUSE_MESSAGE} characters")
        with self._connect() as connection:
            connection.execute("UPDATE submission_control SET pause_message = ? WHERE id = 1", (message,))
        return message

    # --- submissions, as a team or admin sees them -----------------------------------------------------------

    def add(self, token, label, filename, data):
        """Queues a submission and returns its id. Refused while paused, while the team has one queued or running, or once today's
        quota is used up: config.daily_quota submissions, where up to config.daily_failed_quota failed ones do not count
        (further failures count like any other submission). An unlimited team has no daily quota.

        One transaction: concurrent uploads cannot both pass these checks, and the worker cannot claim the row before
        the file is written."""
        submission_id = secrets.token_hex(ID_BYTES)
        now = time.time()
        config = self.config
        with self._connect() as connection:  # leaving it with an exception rolls back
            connection.execute("BEGIN IMMEDIATE")
            message = connection.execute("SELECT pause_message FROM submission_control WHERE id = 1").fetchone()["pause_message"]
            if message is not None:
                raise Paused(message)
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
        """A visible submission with results and team metadata, or None. Admins can see any team's submission."""
        is_admin = self.is_admin(token)
        with self._connect() as connection:
            row = connection.execute(f"SELECT {COLUMNS} FROM submissions JOIN teams USING (token) "
                                     "WHERE id = ? AND (submissions.token = ? OR ?)", (submission_id, token, is_admin)).fetchone()
        return dict(row) if row else None

    def submissions(self, token):
        """Visible submissions with results and team metadata, newest first; all teams for admins."""
        is_admin = self.is_admin(token)
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(f"SELECT {COLUMNS} FROM submissions JOIN teams USING (token) "
                                                           "WHERE submissions.token = ? OR ? ORDER BY submissions.created DESC",
                                                           (token, is_admin))]

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
