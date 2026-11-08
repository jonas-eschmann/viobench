# Submission server

The app queues uploads in SQLite and shows each team its own submissions; admins can view all submissions.
Each submission has one job per requested episode. One worker supervises evaluation subprocesses, downloading
uncached archives as needed. Both processes use `VIOBENCH_SERVER_DIR` (default `./server-data`).

This per-episode schema requires a fresh database; older submission databases are not migrated. For an existing
installation, use an empty server-data directory or a new Compose server-data volume, retaining the old data separately.

```bash
pip install -e ".[server]"
python -m viobench.tools.server.teams add --participant "team a" "team b"  # prints a token per team; `teams list` repeats them
python -m viobench.tools.server.worker &                        # exactly one worker
uvicorn viobench.tools.server.app:app --host 0.0.0.0 --port 8000
```

The current evaluation set is stored as an ordered JSON list in SQLite's `submission_control.evaluation_hashes`,
initially the public set from [`viobench.episodes`](../../episodes/__init__.py).
Admins choose a named preset for new uploads and can check **Also evaluate all existing submissions** or use
**Fill missing for current set** later. A submission's details also accept individual episode hashes. Scheduling
adds only missing jobs, including for test accounts, and keeps existing results. The API accepts arbitrary hash
sets; signing in with a set without a matching preset shows **Custom current set** in the dropdown.
The dropdown is a local edit form initialized at sign-in. Polling updates the saved-set label without changing
your selection, even when another admin changes the saved set. Reload the page to load new presets.

The worker's `--jobs N` option controls concurrent evaluations (default `1`), each in its own subprocess supervised
by a thread; for example, `python -m viobench.tools.server.worker --jobs 2`. Run exactly one worker process/container;
do not scale worker replicas. Jobs are claimed in queue order but can finish out of order. SIGINT or SIGTERM stops
the evaluation subprocesses; interrupted running jobs are requeued at the next worker startup.
Each evaluator reads its archive through conta (`CONTA_CACHE`, default
`~/.cache/conta`), downloading it if absent. A download failure fails that job with a sanitized public error;
details stay in the private log. Saving or scheduling again does not rerun existing jobs, including failed ones.

A team is its token: every API request sends `Authorization: Bearer <token>` and sees only that team's submissions.
Teams can also open the server's root in a browser: a page ([`ui/`](ui/), no dependencies) that signs in with the token
and uses the API below.

Use `teams add --participant` for real participants. Accounts created without `--participant` or `--admin` are test
accounts; for example, `python -m viobench.tools.server.teams add --unlimited debug` creates a test account without a
daily quota. Participant status is a label for distinguishing submissions and does not change upload permissions.

Create an admin account with the same token system:

```bash
python -m viobench.tools.server.teams add --admin admin
# With Docker Compose, from the viobench repository root:
docker compose -f viobench/tools/server/compose.yaml exec app python -m viobench.tools.server.teams add --admin admin
```

Use the printed token in the same sign-in form. Admins see all submissions and their results, with the submitting
team's name and Participant or Test classification in the list and details. Admin accounts cannot upload
(`POST /submissions` returns 403); use a participant or test account for submissions. `teams list` marks each account
as `(participant)`, `(test)` or `(admin)`, with `(unlimited)` separately where applicable. `--admin` and `--participant`
are mutually exclusive; `--unlimited` is independent. Naming a team "admin" alone does not grant admin access.

Change an existing team's flags using its token from `teams list`:

```bash
python -m viobench.tools.server.teams update "$TOKEN" --participant
python -m viobench.tools.server.teams update "$TOKEN" --no-participant
python -m viobench.tools.server.teams update "$TOKEN" --admin --no-participant
python -m viobench.tools.server.teams update "$TOKEN" --no-admin
```

Omitted flags keep their values. Switching an admin to a participant uses `--no-admin --participant`; the two flags
cannot both be true. Updates keep the token, submissions and quota setting. API permissions change immediately;
the browser's account view updates on its next refresh. With Compose, run the same command through `exec app` as above.

Admins can pause new submissions from **Submission access** by entering a message (up to 1000 characters), update
that message while paused, or resume submissions. All teams see the saved message and their upload button is disabled.
Visible browser tabs refresh every 15 seconds, including while idle. Paused uploads return 503 with the message;
participant, test and unlimited accounts are all blocked. Queued/running evaluations continue, and submissions and
results remain readable. The pause survives server restarts until an admin resumes submissions; it does not stop
admin scheduling or existing evaluation jobs.

Submission responses include the submitting account's `team_name` and `is_participant` for both teams and admins.

| endpoint | |
|---|---|
| `GET /` (`/app.js`, `/style.css`) | the web UI; public files, strict Content-Security-Policy |
| `POST /submissions` | multipart `file` (and `label`) → 201 |
| `GET /submissions` | the team's submissions (all submissions for admins), newest first, with current-set `status`, `progress` and `overall` |
| `GET /submissions/<id>` | current-set `results` (`summary`, `episodes`, `partial`) plus all scheduled `jobs`, including episodes outside the current set; admins can access any submission |
| `GET /config` | the rules below, `team_name`, `is_admin`, `pause_message` (null when accepting uploads), and current `evaluation_hashes` |
| `PUT /submission-pause` | admin only: JSON `{"message": "reason"}` pauses or updates the message; `{"message": null}` resumes; returns `pause_message` |
| `GET /evaluations` | admin only: current `hashes` and named `presets` |
| `PUT /evaluations` | admin only: `{"hashes": ["<sha1>", "..."], "backfill": true}` saves the set and optionally schedules missing jobs for all submissions; returns selection and `scheduled` count |
| `POST /evaluations/schedule` | admin only: optional `submission_id` and `hashes`; defaults to all submissions and the current set; returns `scheduled` count |

Rules: [`viobench/tools/config.json`](../config.json), checked in; every key is required, and its `limits` are what
`viobench check` and `evaluate` apply locally too (restart app and worker after editing).

A new upload schedules jobs for the current set. Every accepted upload counts toward `daily_quota`, resetting at
midnight in `quota_time_zone`; failed evaluations do not refund uploads. Queued or running jobs do not block another
upload. Accounts created with `teams add --unlimited` have no daily quota. Scheduling more jobs for existing uploads
does not affect the quota. Each episode job has a 24-minute timeout (`episode_timeout_seconds: 1440`), including
any archive download.

Displayed status and summaries always use the current set: `queued`, `running`, `incomplete` (missing jobs), `done`
or `failed`. Progress counts finished, completed, failed, queued, running and missing episodes.
Partial summaries update as selected episodes finish. Submissions have upload timestamps; start/finish times belong
to individual jobs and remain available in the details JSON. The job table marks jobs in the current set.

Teams never see the guest's output: full `report.json`, logs and per-frame `fuel.npy` stay private under
`reports/<submission-id>/<episode-hash>/`. Each episode's `stdout.txt` and `stderr.txt` keeps its first 1 MiB,
and that job's `worker.txt` keeps the first 1 MiB of evaluation
output. Excess output is discarded while evaluation continues. Report tails read at most 16 KiB from each log.
Local `--verbose` streams guest output directly to the console without these capture limits.

Results include maximum, mean, median, p95 and p99 fuel per attempted frame, total step fuel and attempted-frame counts
per episode, per dataset and overall. Current-set dataset and overall statistics pool the original per-frame fuel
samples from their finished jobs; they do not average per-episode percentiles. Judging uses the overall maximum;
the other statistics are diagnostics. Initialization fuel is reported separately per episode.

Deployment: [`compose.yaml`](compose.yaml) (usage in its header) runs the app behind a Cloudflare tunnel (no open port;
Cloudflare terminates TLS) and the worker unprivileged. Set `WORKER_JOBS=2` for two concurrent evaluations (default `1`).
The worker's memory limit (`WORKER_MEMORY`, default `8g`) is shared across all evaluations. The worker has network access and a writable
`conta-cache` volume for archive downloads. Wasmtime sandboxes the guest, while each evaluation subprocess has a time
limit and capped logs. New selections and scheduled jobs need no service restart.
Keep the `server-data` volume private (the sqlite holds the tokens) and back it up. Behind a
path prefix, serve the page with a trailing slash (`/prefix/`): its requests are relative to it.

When upgrading an existing Compose deployment, remove the obsolete `episodes` container with
`docker compose -f viobench/tools/server/compose.yaml up -d --build --remove-orphans` (from the repository root).
Use the fresh database described above; this command does not migrate the old database.

[`backup.sh`](backup.sh) copies uploads, reports and SQLite (including team tokens) into a dated local snapshot.
It uses the active Docker context and finds `compose.yaml` next to the script, so it can run from any directory.
The app and worker stop during the local copy; services that were running restart before the optional
rsync, including if the copy fails. An interrupted episode job is requeued when the worker starts.

```bash
viobench/tools/server/backup.sh  # defaults to ~/viobench-backups
viobench/tools/server/backup.sh /path/to/backups
viobench/tools/server/backup.sh /path/to/backups user@backup-host:/path/to/viobench-backups/
```

Each snapshot is a full copy of `server-data`; rsync transfers all completed snapshots and skips unchanged files.
Failed copies remain in `.partial.*` directories and are excluded from rsync. The downloadable episode cache is
separate from these backups. Keep the snapshots private, like the original volume.
