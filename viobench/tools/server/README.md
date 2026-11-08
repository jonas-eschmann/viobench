# Submission server

The app queues uploads in SQLite and shows each team its own submissions; admins can view all submissions.
The worker evaluates them one at a time.
Both use `VIOBENCH_SERVER_DIR` (default `./server-data`).

```bash
pip install -e ".[server]"
python -m viobench.tools.server.teams add --participant "team a" "team b"  # prints a token per team; `teams list` repeats them
python -m viobench.tools.server.worker &                        # exactly one worker
uvicorn viobench.tools.server.app:app --host 0.0.0.0 --port 8000
```

The worker evaluates every submission on `EPISODES` in [`worker.py`](worker.py), conta references to episode archives
(the public set, [`viobench.episodes`](../../episodes/__init__.py), for now). That constant is the only place the
evaluation set is configured: change it and restart the worker. On start the worker resolves the references through
conta, downloading what its cache (`CONTA_CACHE`, default `~/.cache/conta`) lacks, and exits if one cannot be had
(or if a hash is listed twice); `--fetch` does only that, for a worker without network. An episode is its hash, the
sha1 of its archive: results list the episodes by hash, in the order of `EPISODES`, and the worker fails a submission
(internal error) whose report lists any others. Nothing from inside an archive (its manifest's name) identifies it.

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
Existing databases add missing flags automatically. Existing accounts default to nonparticipant and retain any admin access.

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
results remain readable. The pause survives server restarts until an admin resumes submissions.

Submission responses include the submitting account's `team_name` and `is_participant` for both teams and admins.

| endpoint | |
|---|---|
| `GET /` (`/app.js`, `/style.css`) | the web UI; public files, strict Content-Security-Policy |
| `POST /submissions` | multipart `file` (and `label`) → 201 |
| `GET /submissions` | the team's submissions (all submissions for admins), newest first: id, label, sha256, times (UTC), `status` (`queued`, `running`, `done`, `failed`), `error`, `overall` (headline medians once done) |
| `GET /submissions/<id>` | one of them, with `results` (metrics only) once done; admins can access any submission |
| `GET /config` | the rules below, `is_admin` for the authenticated account, and `pause_message` (null when accepting submissions) |
| `PUT /submission-pause` | admin only: JSON `{"message": "reason"}` pauses or updates the message; `{"message": null}` resumes; returns `pause_message` |

Rules: [`viobench/tools/config.json`](../config.json), checked in; every key is required, and its `limits` are what
`viobench check` and `evaluate` apply locally too (restart app and worker after editing).

A team has one submission in flight at a time and `daily_quota` per day (from midnight in `quota_time_zone`); up to
`daily_failed_quota` failed ones (crashed, timed out or no episode completed; no results) do not count. A team created
with `teams add --unlimited` (e.g. the instructors' debug team; the `unlimited` column of `teams`) has no daily quota,
but still one submission at a time.
Teams never see the guest's output: `report.json` and the logs stay in `reports/<id>/`.
Each episode's `stdout.txt` and `stderr.txt` keeps its first 1 MiB, and `worker.txt` keeps the first 1 MiB of evaluation
output. Excess output is discarded while evaluation continues. Report tails read at most 16 KiB from each log.
Local `--verbose` streams guest output directly to the console without these capture limits.

Results include maximum, mean, median, p95 and p99 fuel per attempted frame, total step fuel and attempted-frame counts
per episode, per dataset and overall. Dataset and overall statistics pool the frames. Judging uses the overall maximum;
the other statistics are diagnostics. Initialization fuel is reported separately per episode.

Deployment: [`compose.yaml`](compose.yaml) (usage in its header) runs the app behind a Cloudflare tunnel (no open port;
Cloudflare terminates TLS) and the worker unprivileged, without network and with a memory limit (wasmtime sandboxes
the guest, not the host process); a one-shot `episodes` service fills the `conta-cache` volume (`worker --fetch`)
before the worker starts. Keep the `server-data` volume private (the sqlite holds the tokens) and back it up. Behind a
path prefix, serve the page with a trailing slash (`/prefix/`): its requests are relative to it.
