"""SQLite accounts, current evaluation selection, and one job per submission/episode.

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
from .evaluations import default_hashes, normalize_hashes, presets, results_for

DEFAULT_DIR = Path(os.environ.get("VIOBENCH_SERVER_DIR", "server-data"))
TOKEN_BYTES = 24  # a token is 32 base64url characters
ID_BYTES = 8  # a submission id is 16 hex digits, random: it says nothing about how many submissions exist
MAX_PAUSE_MESSAGE = 1000
COLUMNS = ("submissions.id, label, filename, sha256, bytes, submissions.created, "
           "teams.name AS team_name, teams.is_participant")  # API fields; never include tokens

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    token TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL,
    unlimited INTEGER NOT NULL DEFAULT 0,
    is_admin INTEGER NOT NULL DEFAULT 0,
    is_participant INTEGER NOT NULL DEFAULT 0,
    CHECK (NOT (is_admin AND is_participant))
);
CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY, token TEXT NOT NULL REFERENCES teams(token), label TEXT,
    filename TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS submissions_token ON submissions(token, created);
CREATE TABLE IF NOT EXISTS submission_control (
    id INTEGER PRIMARY KEY CHECK (id = 1), pause_message TEXT, evaluation_hashes TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluation_jobs (
    id INTEGER PRIMARY KEY,
    submission_id TEXT NOT NULL REFERENCES submissions(id),
    episode_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
    created REAL NOT NULL, started REAL, finished REAL, result TEXT, error TEXT,
    UNIQUE (submission_id, episode_hash)
);
CREATE INDEX IF NOT EXISTS evaluation_queue ON evaluation_jobs(status, id);
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
            connection.executescript("BEGIN IMMEDIATE;" + SCHEMA)
            hashes = default_hashes()
            connection.execute("INSERT OR IGNORE INTO submission_control (id, evaluation_hashes) VALUES (1, ?)",
                               (json.dumps(hashes),))
        self.config = load_config()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
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

    def team_name(self, token):
        with self._connect() as connection:
            row = connection.execute("SELECT name FROM teams WHERE token = ?", (token,)).fetchone()
        return row["name"] if row else None

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

    # --- evaluation selection and scheduling ---------------------------------------------------------------

    @staticmethod
    def _evaluation_hashes(connection):
        return json.loads(connection.execute("SELECT evaluation_hashes FROM submission_control WHERE id = 1").fetchone()[0])

    def evaluation_hashes(self):
        with self._connect() as connection:
            return self._evaluation_hashes(connection)

    def evaluations(self):
        with self._connect() as connection:
            connection.execute("BEGIN")
            return {"hashes": self._evaluation_hashes(connection), "presets": presets()}

    @staticmethod
    def _schedule(connection, hashes, submission_id=None):
        """Insert missing pairs only. Call inside the selection/upload transaction."""
        ids = [submission_id] if submission_id is not None else [row[0] for row in connection.execute("SELECT id FROM submissions")]
        now, count = time.time(), 0
        for selected in ids:
            for digest in hashes:
                count += connection.execute(
                    "INSERT OR IGNORE INTO evaluation_jobs (submission_id, episode_hash, created) VALUES (?, ?, ?)",
                    (selected, digest, now)).rowcount
        return count

    def set_evaluations(self, hashes, backfill=False):
        hashes = normalize_hashes(hashes)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE submission_control SET evaluation_hashes = ? WHERE id = 1", (json.dumps(hashes),))
            return self._schedule(connection, hashes) if backfill else 0

    def schedule_evaluations(self, hashes=None, submission_id=None):
        hashes = normalize_hashes(hashes) if hashes is not None else None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if submission_id is not None and connection.execute("SELECT 1 FROM submissions WHERE id = ?", (submission_id,)).fetchone() is None:
                raise LookupError("no such submission")
            hashes = self._evaluation_hashes(connection) if hashes is None else hashes
            return self._schedule(connection, hashes, submission_id)

    # --- uploads: admission depends only on pause state and daily upload count ------------------------------

    def add(self, token, label, filename, data):
        """Atomically admit an upload and schedule its jobs for the current hashes."""
        submission_id = secrets.token_hex(ID_BYTES)
        now, config = time.time(), self.config
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            message = connection.execute("SELECT pause_message FROM submission_control WHERE id = 1").fetchone()[0]
            if message is not None:
                raise Paused(message)
            unlimited = connection.execute("SELECT unlimited FROM teams WHERE token = ?", (token,)).fetchone()[0]
            count = connection.execute("SELECT COUNT(*) FROM submissions WHERE token = ? AND created >= ?",
                                       (token, day_start(now, config.quota_time_zone))).fetchone()[0]
            if count >= config.daily_quota and not unlimited:
                raise Refused(f"today's quota of {config.daily_quota} upload(s) is used up; "
                              f"it resets at midnight ({config.quota_time_zone})")
            connection.execute(
                "INSERT INTO submissions (id, token, label, filename, sha256, bytes, created) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (submission_id, token, label, filename, hashlib.sha256(data).hexdigest(), len(data), now))
            self._schedule(connection, self._evaluation_hashes(connection), submission_id)
            self.upload_path(submission_id).write_bytes(data)
        return submission_id

    # --- visibility, progress and metrics for the current selection -----------------------------------------

    def _view(self, connection, row, hashes, details):
        submission = dict(row)
        jobs = {}
        for job_row in connection.execute("SELECT id, episode_hash, status, created, started, finished, result, error "
                                          "FROM evaluation_jobs WHERE submission_id = ? ORDER BY id", (row["id"],)):
            job = dict(job_row)
            job["result"] = json.loads(job["result"]) if job["result"] else None
            job["selected"] = job["episode_hash"] in hashes
            jobs[job["episode_hash"]] = job
        selected = [jobs[digest] for digest in hashes if digest in jobs]
        terminal = [job for job in selected if job["status"] in ("done", "failed")]
        completed = sum(bool(job["result"] and job["result"]["completed"]) for job in terminal)
        queued = [job for job in selected if job["status"] == "queued"]
        progress = {"total": len(hashes), "finished": len(terminal), "completed": completed,
                    "failed": len(terminal) - completed, "queued": len(queued),
                    "running": sum(job["status"] == "running" for job in selected),
                    "missing": len(hashes) - len(selected)}
        submission["progress"] = progress
        submission["status"] = ("running" if progress["running"] else "queued" if progress["queued"] else
                                "incomplete" if progress["missing"] else "done" if completed else "failed")
        submission["error"] = "no episode completed" if submission["status"] == "failed" else None
        submission["results"] = results_for(self, row["id"], hashes, jobs)
        if details:
            submission["jobs"] = list(jobs.values())
        return submission

    def get(self, token, submission_id):
        is_admin = self.is_admin(token)
        with self._connect() as connection:
            connection.execute("BEGIN")
            row = connection.execute(f"SELECT {COLUMNS} FROM submissions JOIN teams USING (token) "
                                     "WHERE submissions.id = ? AND (submissions.token = ? OR ?)", (submission_id, token, is_admin)).fetchone()
            return self._view(connection, row, self._evaluation_hashes(connection), details=True) if row else None

    def submissions(self, token):
        is_admin = self.is_admin(token)
        with self._connect() as connection:
            connection.execute("BEGIN")
            hashes = self._evaluation_hashes(connection)
            rows = connection.execute(f"SELECT {COLUMNS} FROM submissions JOIN teams USING (token) "
                                      "WHERE submissions.token = ? OR ? ORDER BY submissions.created DESC", (token, is_admin)).fetchall()
            return [self._view(connection, row, hashes, details=False) for row in rows]

    # --- single worker -----------------------------------------------------------------------------

    def claim_next(self):
        """Claim the oldest queued job; conta resolves its episode when evaluation reads it."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT evaluation_jobs.*, teams.name FROM evaluation_jobs "
                "JOIN submissions ON submission_id = submissions.id JOIN teams USING (token) "
                "WHERE evaluation_jobs.status = 'queued' "
                "ORDER BY evaluation_jobs.id LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            connection.execute("UPDATE evaluation_jobs SET status = 'running', started = ? WHERE id = ?", (now, row["id"]))
            return dict(row) | {"status": "running", "started": now}

    def requeue_running(self):
        """Only at startup of the single worker: finished jobs are immutable."""
        with self._connect() as connection:
            return connection.execute("UPDATE evaluation_jobs SET status = 'queued', started = NULL WHERE status = 'running'").rowcount

    def finish_job(self, job_id, result=None, error=None):
        if (result is None) == (error is None):
            raise ValueError("supply either an episode result or an error")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute("SELECT * FROM evaluation_jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None or job["status"] not in ("queued", "running"):
                raise ValueError("job is not pending")
            if result is not None and result["hash"] != job["episode_hash"]:
                raise ValueError("result hash does not match job")
            now = time.time()
            connection.execute("UPDATE evaluation_jobs SET status = ?, finished = ?, result = ?, error = ? WHERE id = ?",
                               ("done" if error is None else "failed", now, json.dumps(result) if result is not None else None, error, job_id))

    def upload_path(self, submission_id):
        return self.uploads / f"{submission_id}.wasm"

    def report_dir(self, submission_id, episode_hash):
        return self.reports / submission_id / episode_hash
