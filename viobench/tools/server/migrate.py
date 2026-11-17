"""Migrate a legacy server into the submission-baseline schema.

Stop the API and worker first, then run:
    python -m viobench.tools.server.migrate server-data server-data-baselines

By default the source is never modified; the destination must not exist. Point
both services at the verified copy using VIOBENCH_SERVER_DIR.

With an existing backup, migrate the original database in one transaction:
    python -m viobench.tools.server.migrate --in-place server-data

Both modes preserve evaluation settings, jobs, report paths, and ownership.
"""
import argparse
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

from .store import SCHEMA, SCHEMA_VERSION


LEGACY_COLUMNS = {
    "teams": {"id", "token", "name", "created", "unlimited", "daily_quota", "extra_uploads", "extra_day", "type"},
    "submissions": {"id", "token", "label", "filename", "sha256", "bytes", "created"},
    "submission_control": {"id", "pause_message", "evaluation_hashes", "config"},
    "evaluation_jobs": {"id", "submission_id", "episode_hash", "status", "created", "started", "finished", "result", "error", "evaluation_config"},
}


def _check_legacy(connection):
    if connection.execute("PRAGMA user_version").fetchone()[0] != 0:
        raise ValueError("database is already migrated or has an unsupported schema version")
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")}
    if tables != set(LEGACY_COLUMNS):
        raise ValueError("unsupported legacy schema: expected the current account-baseline server tables")
    for table, expected in LEGACY_COLUMNS.items():
        if {row[1] for row in connection.execute(f"PRAGMA table_info({table})")} != expected:
            raise ValueError(f"unsupported legacy schema in {table}")
    if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise ValueError("source database failed SQLite integrity_check")
    if connection.execute("PRAGMA foreign_key_check").fetchone():
        raise ValueError("source database contains broken foreign keys")
    if connection.execute("SELECT 1 FROM submission_control WHERE id = 1").fetchone() is None:
        raise ValueError("source database has no saved server configuration")


def _convert(connection):
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("BEGIN IMMEDIATE")
    try:
        _check_legacy(connection)
        counts = {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in LEGACY_COLUMNS}
        baseline_count = connection.execute(
            "SELECT COUNT(*) FROM submissions JOIN teams USING (token) WHERE teams.type = 'baseline'").fetchone()[0]
        team_count = connection.execute("SELECT COUNT(*) FROM teams WHERE type = 'baseline'").fetchone()[0]
        connection.execute("ALTER TABLE submissions ADD COLUMN baseline INTEGER NOT NULL DEFAULT 0 CHECK (baseline IN (0, 1))")
        connection.execute("UPDATE submissions SET baseline = 1 WHERE token IN (SELECT token FROM teams WHERE type = 'baseline')")
        # Rebuild only teams to remove the retired account type from its CHECK constraint.
        team_schema = re.search(r"CREATE TABLE IF NOT EXISTS teams \(.*?\n\);", SCHEMA, re.DOTALL).group()
        connection.execute(team_schema.replace("EXISTS teams (", "EXISTS teams_migrated (", 1))
        connection.execute("INSERT INTO teams_migrated (id, token, name, created, unlimited, daily_quota, extra_uploads, extra_day, type) "
                           "SELECT id, token, name, created, unlimited, daily_quota, extra_uploads, extra_day, "
                           "CASE WHEN type = 'baseline' THEN 'participant' ELSE type END FROM teams")
        connection.execute("DROP TABLE teams")
        connection.execute("ALTER TABLE teams_migrated RENAME TO teams")
        if any(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] != count for table, count in counts.items()):
            raise ValueError("migration changed an existing row count")
        if connection.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("migration failed foreign_key_check")
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")
    return {**counts, "baseline_submissions": baseline_count, "converted_teams": team_count}


def migrate_in_place(source):
    """Update an existing offline database atomically, without copying or moving artifacts."""
    database = Path(source).resolve() / "submissions.sqlite"
    if not database.is_file():
        raise ValueError("source must contain submissions.sqlite")
    # mode=rw refuses to create a database if the path disappears before opening it.
    with sqlite3.connect(database.as_uri() + "?mode=rw", uri=True) as connection:
        return _convert(connection)


def migrate(source, destination):
    """Migrate a new copy of an offline server directory; never overwrite a source or destination."""
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if not source.is_dir() or not (source / "submissions.sqlite").is_file():
        raise ValueError("source must contain submissions.sqlite")
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination already exists; choose a new directory")
    destination = destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("source and destination must be separate directories")
    with sqlite3.connect((source / "submissions.sqlite").as_uri() + "?mode=ro", uri=True) as original:
        _check_legacy(original)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
        try:
            def ignore(directory, names):
                sqlite_files = {"submissions.sqlite", "submissions.sqlite-wal", "submissions.sqlite-shm", "submissions.sqlite-journal"}
                return [name for name in names if name in sqlite_files] if Path(directory) == source else []

            shutil.copytree(source, temporary, dirs_exist_ok=True, ignore=ignore)
            with sqlite3.connect(temporary / "submissions.sqlite") as copied:
                original.backup(copied)
                result = _convert(copied)
            shutil.copymode(source / "submissions.sqlite", temporary / "submissions.sqlite")
            if destination.exists() or destination.is_symlink():
                raise ValueError("destination appeared during migration; refusing to replace it")
            temporary.rename(destination)
        except BaseException:
            shutil.rmtree(temporary)
            raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?", help="new directory for a migrated copy")
    parser.add_argument("--in-place", action="store_true", help="update the source database; requires stopped services and an existing backup")
    args = parser.parse_args()
    if args.in_place and args.destination is not None:
        parser.error("--in-place does not accept a destination")
    if not args.in_place and args.destination is None:
        parser.error("provide a destination for a copy, or use --in-place")
    try:
        result = migrate_in_place(args.source) if args.in_place else migrate(args.source, args.destination)
    except (ValueError, sqlite3.Error, OSError) as error:
        parser.error(str(error))
    location = f"in place at {args.source}" if args.in_place else f"to {args.destination}"
    note = "Uploads and reports were not changed." if args.in_place else "The source was not modified."
    print(f"Migrated {location}: {result['submissions']} submissions, {result['baseline_submissions']} baselines; "
          f"{result['converted_teams']} baseline teams are now participants. {note}")


if __name__ == "__main__":
    main()
