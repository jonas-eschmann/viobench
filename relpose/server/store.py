"""SQLite-backed submission store shared by the web app and the worker."""
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

DEFAULT_DIR = Path(os.environ.get("EECS116_SERVER_DIR", "server-data"))
MAX_UPLOAD_BYTES = int(os.environ.get("EECS116_MAX_UPLOAD_MIB", "128")) << 20
DAILY_QUOTA = int(os.environ.get("EECS116_DAILY_QUOTA", "20"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    label TEXT,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    created REAL NOT NULL,
    status TEXT NOT NULL,            -- queued | running | done | failed
    started REAL,
    finished REAL,
    summary TEXT,                    -- JSON: report["summary"]
    error TEXT
);
CREATE INDEX IF NOT EXISTS submissions_user ON submissions(user, created);
"""


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

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def tokens(self):
        """{token: user} from $EECS116_TOKENS (lines "token user") or <dir>/tokens.txt."""
        path = Path(os.environ.get("EECS116_TOKENS", self.directory / "tokens.txt"))
        if not path.is_file():
            return {}
        mapping = {}
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and not line.startswith("#"):
                mapping[parts[0]] = parts[1]
        return mapping

    def submissions_today(self, user):
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS n FROM submissions WHERE user = ? AND created > ?", (user, time.time() - 86400)).fetchone()
        return int(row["n"])

    def add(self, user, label, filename, data):
        digest = hashlib.sha256(data).hexdigest()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO submissions (user, label, filename, sha256, bytes, created, status) VALUES (?, ?, ?, ?, ?, ?, 'queued')",
                (user, label, filename, digest, len(data), time.time()))
            submission_id = cursor.lastrowid
        (self.uploads / f"{submission_id}.wasm").write_bytes(data)
        return submission_id

    def get(self, submission_id):
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
        return dict(row) if row else None

    def list(self, user=None, status=None, limit=500):
        query = "SELECT * FROM submissions"
        clauses, values = [], []
        if user:
            clauses.append("user = ?")
            values.append(user)
        if status:
            clauses.append("status = ?")
            values.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created DESC LIMIT ?"
        values.append(limit)
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, values).fetchall()]

    def claim_next(self):
        """Atomically moves the oldest queued submission to running; returns it or None."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM submissions WHERE status = 'queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            connection.execute("UPDATE submissions SET status = 'running', started = ? WHERE id = ?", (time.time(), row["id"]))
            connection.execute("COMMIT")
        return dict(row)

    def finish(self, submission_id, summary=None, error=None):
        with self._connect() as connection:
            connection.execute("UPDATE submissions SET status = ?, finished = ?, summary = ?, error = ? WHERE id = ?",
                               ("done" if error is None else "failed", time.time(), json.dumps(summary) if summary else None, error, submission_id))

    def upload_path(self, submission_id):
        return self.uploads / f"{submission_id}.wasm"

    def report_dir(self, submission_id):
        return self.reports / str(submission_id)

    def leaderboard(self):
        """Best (lowest ATE) completed submission per user and dataset, plus every point for the curve."""
        points = {}
        for row in self.list(status="done", limit=100000):
            summary = json.loads(row["summary"] or "{}")
            for dataset, values in summary.get("datasets", {}).items():
                points.setdefault(dataset, []).append({
                    "id": row["id"], "user": row["user"], "label": row["label"], "created": row["created"],
                    "ate": values["ate_rmse_median"], "rpe": values["rpe_translation_rmse_median"],
                    "fuel": values["fuel_per_frame_median"], "fuel_p95": values["fuel_per_frame_p95"],
                    "ms": values["wall_per_frame_median_ms"], "completed": values["completed"], "episodes": values["episodes"],
                })
        return points
