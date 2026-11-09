"""SQLite accounts, current evaluation selection, and one job per submission/episode.

A team is its token: submissions belong to a token. Baseline accounts share comparison scores;
admin accounts can see every team's submissions and pause new uploads."""
import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import load as load_config
from .evaluations import comparison_for, default_hashes, normalize_hashes, presets, public_episode, results_for

DEFAULT_DIR = Path(os.environ.get("VIOBENCH_SERVER_DIR", "server-data"))
TOKEN_BYTES = 24  # a token is 32 base64url characters
ID_BYTES = 8  # a submission id is 16 hex digits, random: it says nothing about how many submissions exist
MAX_PAUSE_MESSAGE = 1000
TEAM_TYPES = ("test", "participant", "baseline", "admin")
COLUMNS = ("submissions.id, label, filename, sha256, bytes, submissions.created, "
           "teams.name AS team_name, teams.type AS team_type")  # API fields; never include tokens

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    token TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL,
    unlimited INTEGER NOT NULL DEFAULT 0,
    type TEXT NOT NULL DEFAULT 'test' CHECK (type IN ('test', 'participant', 'baseline', 'admin'))
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

    def add_team(self, name, team_type="test", unlimited=False):
        """Create an account and return its token. `unlimited` affects quotas only."""
        if team_type not in TEAM_TYPES:
            raise ValueError(f"account type must be one of: {', '.join(TEAM_TYPES)}")
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with self._connect() as connection:
            connection.execute("INSERT INTO teams (token, name, created, unlimited, type) VALUES (?, ?, ?, ?, ?)",
                               (token, name, time.time(), unlimited, team_type))
        return token

    def update_team(self, token, *, team_type):
        """Change the account type, keeping its token, quota setting and submissions."""
        if team_type not in TEAM_TYPES:
            raise ValueError(f"account type must be one of: {', '.join(TEAM_TYPES)}")
        with self._connect() as connection:
            if not connection.execute("UPDATE teams SET type = ? WHERE token = ?", (team_type, token)).rowcount:
                raise ValueError("no such team")

    def teams(self):
        """[(token, name, unlimited, team_type)] in the order they were created."""
        with self._connect() as connection:
            return [(row["token"], row["name"], bool(row["unlimited"]), row["type"])
                    for row in connection.execute("SELECT token, name, unlimited, type FROM teams ORDER BY created")]

    def is_team(self, token):
        with self._connect() as connection:
            return connection.execute("SELECT 1 FROM teams WHERE token = ?", (token,)).fetchone() is not None

    def team_name(self, token):
        with self._connect() as connection:
            row = connection.execute("SELECT name FROM teams WHERE token = ?", (token,)).fetchone()
        return row["name"] if row else None

    def team_type(self, token):
        with self._connect() as connection:
            row = connection.execute("SELECT type FROM teams WHERE token = ?", (token,)).fetchone()
        return row["type"] if row else None

    def is_admin(self, token):
        return self.team_type(token) == "admin"

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

    def delete_submission(self, token, submission_id):
        """Delete a baseline account's own submission unless any of its jobs is running."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            team = connection.execute("SELECT type FROM teams WHERE token = ?", (token,)).fetchone()
            if team is None or team["type"] != "baseline":
                raise PermissionError("only baseline accounts can delete submissions")
            if connection.execute("SELECT 1 FROM submissions WHERE id = ? AND token = ?", (submission_id, token)).fetchone() is None:
                raise LookupError("no such submission")
            if connection.execute("SELECT 1 FROM evaluation_jobs WHERE submission_id = ? AND status = 'running' LIMIT 1",
                                  (submission_id,)).fetchone() is not None:
                raise Refused("cannot delete a submission while an evaluation job is running")
            connection.execute("DELETE FROM evaluation_jobs WHERE submission_id = ?", (submission_id,))
            connection.execute("DELETE FROM submissions WHERE id = ?", (submission_id,))

        # Commit after existing readers finish and before removing the files they may still need.
        try:
            self.upload_path(submission_id).unlink(missing_ok=True)
            try:
                shutil.rmtree(self.reports / submission_id)
            except FileNotFoundError:
                pass
        except OSError as error:
            # A filesystem PermissionError is a cleanup failure, distinct from the authorization error above.
            raise OSError("submission deleted but file cleanup failed") from error

    # --- visibility, progress and metrics for the current selection -----------------------------------------

    @staticmethod
    def _jobs(connection, submission_id, hashes):
        jobs = {}
        for job_row in connection.execute("SELECT id, episode_hash, status, created, started, finished, result, error "
                                          "FROM evaluation_jobs WHERE submission_id = ? ORDER BY id", (submission_id,)):
            job = dict(job_row)
            job["result"] = public_episode(json.loads(job["result"])) if job["result"] else None
            job["selected"] = job["episode_hash"] in hashes
            jobs[job["episode_hash"]] = job
        return jobs

    def _view(self, connection, row, hashes, details):
        submission = dict(row)
        jobs = self._jobs(connection, row["id"], hashes)
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

    def comparison(self, token):
        """Own and baseline scores from one snapshot, including for admin viewers."""
        with self._connect() as connection:
            connection.execute("BEGIN")
            hashes = self._evaluation_hashes(connection)
            rows = connection.execute(
                "SELECT submissions.id, label, teams.name AS team_name, teams.type AS team_type, "
                "submissions.token = ? AS own FROM submissions JOIN teams USING (token) "
                "WHERE submissions.token = ? OR teams.type = 'baseline' ORDER BY submissions.created, submissions.id",
                (token, token)).fetchall()
            return comparison_for(self, hashes, [(dict(row), self._jobs(connection, row["id"], hashes)) for row in rows])

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
