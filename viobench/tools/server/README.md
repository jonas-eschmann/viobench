# Submission server

The app queues uploads in SQLite and shows each team its own submissions; admins can view all submissions.
Each submission has one job per requested episode. One worker supervises evaluation subprocesses, downloading
uncached archives as needed. Both processes use `VIOBENCH_SERVER_DIR` (default `./server-data`).

This schema, including saved competition settings and team allowances, requires a fresh database; older submission
databases are not migrated. Existing deployments must explicitly choose an empty server-data directory or a new
Compose server-data volume, retaining the old data and a backup separately. Run fresh evaluations for the
zero-prediction scoring rule; results computed with the previous held-pose rule are not converted.

```bash
pip install -e ".[server]"
python -m viobench.tools.server.teams add --type participant "team a" "team b"  # prints tokens; `teams list` repeats them
python -m viobench.tools.server.worker &                        # exactly one worker
uvicorn viobench.tools.server.app:app --host 0.0.0.0 --port 8000
```

To seed a fresh local server that is already running with its worker, activate the Python environment and run
[`seed.sh`](seed.sh) from the directory containing `server-data`. For example, from the parent project directory:

```bash
bash viobench/viobench/tools/server/seed.sh baselines
```

It creates `participant`, `baseline` (unlimited uploads), and `admin`, prints their tokens, and uploads
every `*.wasm` in the given directory through the baseline account. Labels are filenames without `.wasm`.
The directory defaults to `./baselines`; set `SERVER_URL` to override `http://localhost:8000`.
Account creation uses the existing `python -m viobench.tools.server.teams add` CLI; uploads use curl.
The script does not inspect, reset or edit the database directly. It keeps the current working directory, so the
CLI uses `./server-data` (or `VIOBENCH_SERVER_DIR`, if set), just like the app and worker.
The fixed tokens are `test-participant`, `test-baseline`, and `test-admin`, respectively; they stay the same across
fresh database setups. These predictable credentials are for local testing only. An existing token causes an error;
the script does not replace or reuse an existing account.
Uploads join the worker queue; on an error it stops and leaves any accounts/uploads already created intact.

The current evaluation set is stored as an ordered JSON list in SQLite's `submission_control.evaluation_hashes`,
initially the public set from [`viobench.episodes`](../../episodes/__init__.py).
The `public+private-10` preset combines all public episodes with the first ten entries of the neutrally labeled
`private` catalog. The individual `public` and `private` sets are also available as presets.
Preset names are admin-only shortcuts for selecting hashes. Episodes have no dataset membership: every team's
summary, comparison point, and claimed area use the entire current evaluation set.
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
**Rerun outdated for current set** is the explicit exception described below.

For additional local archives, set `CONTA_SEARCH_PATHS` to directories containing `<hash>` or `<hash>.gz` files
(for a shard checkout, use its `data` directory). Separate directories with `:` on Unix or `;` on Windows.
They are searched after the normal cache and before downloads, and remain read-only; downloads and extracted
files still go into the normal cache. Leave `CONTA_ROOT` unset to retain online fallback.

```bash
export CONTA_SEARCH_PATHS="/datasets/private/data:/datasets/shared/data"
```

A team is its token: every API request sends `Authorization: Bearer <token>`. Submission lists and details are
restricted to their owner and admins; comparison plots also show shared baseline scores.
Teams can also open the server's root in a browser: a page ([`ui/`](ui/), no dependencies) that signs in with the token
and uses the API below.

Accounts have one type: `test` (the default), `participant`, `baseline`, or `admin`. Use `teams add --type participant`
for real participants. `--unlimited` independently removes the daily quota; for example,
`python -m viobench.tools.server.teams add --unlimited debug` creates an unlimited test account.
Tokens are random by default. For a single test account, `teams add --token test-participant --type participant participant`
supplies a fixed token. Explicit tokens use ASCII letters, digits, `_` and `-`; they must be nonempty and unique.

Create an admin account with the same token system:

```bash
python -m viobench.tools.server.teams add --type admin admin
# With Docker Compose, from the viobench repository root:
docker compose -f viobench/tools/server/compose.yaml exec app python -m viobench.tools.server.teams add --type admin admin
```

Use the printed token in the same sign-in form. Admins see all submissions and their results, with the submitting
team's name and account type in the list and details. Admin accounts cannot upload (`POST /submissions` returns 403).
`teams list` shows each account's type and quota exemption. Naming a team "admin" alone does not grant admin access.

Admins edit **Competition settings** in the browser: the global daily quota, reset time zone, upload size,
episode timeout and guest limits. Settings persist in SQLite; saving needs no restart. The checked-in
[`config.json`](../config.json) seeds a fresh server and supplies local CLI defaults, rather than overriding saved
server settings on restart. These controls do not change deployment settings such as worker concurrency or container memory.
Forms initialize at sign-in; polling updates the saved values beside them without replacing local edits.
To reproduce server limits locally, save the `evaluation.limits` object as JSON and pass it to
`viobench evaluate solution.wasm --limits limits.json`. `viobench check` always uses the packaged defaults.

**Team allowances** lists account names/types and usage without exposing tokens. Set a daily override (blank means
the global quota), unlimited uploads, or extra uploads for the current quota day, then save that team's form.
Extra uploads is the day's total extra allowance, not an increment; it expires at the next quota-day boundary.
Changing the reset time zone recomputes today's counted uploads and applicable extras using the new local midnight.
The saved status shows the effective base quota, extras, usage and remaining allowance. Teams see their own allowance
beside the upload form. Unlimited accounts still obey the upload pause and size limit.

Change an existing team's type using its token from `teams list`:

```bash
python -m viobench.tools.server.teams update --type participant -- "$TOKEN"
python -m viobench.tools.server.teams update --type admin -- "$TOKEN"
python -m viobench.tools.server.teams update --type test -- "$TOKEN"  # also revokes admin/baseline status
```

Updates keep the token, submissions and quota setting. API permissions change immediately;
the browser's account view updates on its next refresh. With Compose, run the same command through `exec app` as above.

Create a baseline uploader and sign in with its token to upload labeled reference solutions:

```bash
python -m viobench.tools.server.teams add --type baseline --unlimited "Baselines"
```

Every eligible upload from a baseline account appears in the comparison plots of all signed-in teams. Use the
submission label for the method and configuration, such as `OpenVINS, 320px`; labels need not be unique. Changing an
existing account to baseline publishes its previous eligible submissions too; changing it away removes them from
other teams' plots. Baseline accounts have no admin permissions and use the same upload pause and job queue.

Baseline accounts can delete their own submissions through **Details → Delete submission**, with confirmation.
Deletion permanently removes the submission, its queued/finished/failed jobs, uploaded WASM, reports and comparison
points. Wait until every running job finishes, including jobs outside the current evaluation set. Other account
types, including admins, cannot delete submissions. Deleting an upload counted in the current quota day frees its
daily upload slot.

One plot compares each submission's mean episode Sim(3)-aligned ATE (metres) with its maximum fuel per frame,
both across the entire current evaluation set. The arithmetic mean gives each selected hash equal weight,
regardless of its frame count. Baselines are squares, the viewer's other submissions are circles, and the dashed
line marks the baseline Pareto frontier. The solid frontier combines baselines with the viewer's submissions;
shading shows the additional area they cover beyond the baselines (lower is better on both axes).
Both axes and the area calculation use logarithms, giving equal proportional improvements equal weight. For the
plot, frontiers and area only, fuel is clamped to at least `1` per frame and ATE to `0.001 m`; labels and reported
metrics keep their raw values. The fuel reference is `X = 1.2 × step_fuel`; the error reference is
`Y = 1.2 ×` the worst eligible baseline's mean ATE after clamping. Thus all-zero baseline errors give `Y = 0.0012 m`,
and fuel improvements can still earn area. Nonbaseline uploads do not change these references.
The claimed area is the combined dominated area minus baseline dominated area in natural-log coordinates.
Its percentage is `100 × claimed area / (ln(X / 1) × ln(Y / 0.001))`, unavailable until a baseline exists.
The plot zooms each lower bound to one tenth of the visible minimum, never below its scoring floor.
This changes only the display: the full scoring reference and denominator, and upper fuel bound `1.2 × step_fuel`, stay unchanged.
The fuel interval from `step_fuel` to `1.2 × step_fuel` still contributes `ln(1.2)` width, with less relative
weight than on linear axes. The comparison table provides raw values and names. Baseline details, files, and
historical jobs remain private.

Only submissions with reports for every selected episode, finite scores and current evaluation settings appear. Guest errors still produce
scores: the failed and remaining frames use zero position/twist and identity rotation, and alignment includes all
frames. Infrastructure failures without a report remain unavailable. Selecting additional episodes hides points
until the existing fill-missing workflow supplies those results. Comparison data refreshes every 15 seconds.

Admins can pause new submissions from **Submission access** by entering a message (up to 1000 characters), update
that message while paused, or resume submissions. All teams see the saved message and their upload button is disabled.
Visible browser tabs refresh every 15 seconds, including while idle. Paused uploads return 503 with the message;
participant, test, baseline and unlimited accounts are all blocked. Queued/running evaluations continue, and submissions and
results remain readable. The pause survives server restarts until an admin resumes submissions; it does not stop
admin scheduling or existing evaluation jobs.

Submission responses include the submitting account's `team_name` and `team_type` for both teams and admins.

| endpoint | |
|---|---|
| `GET /` (`/app.js`, `/style.css`) | the web UI; public files, strict Content-Security-Policy |
| `POST /submissions` | multipart `file` (and `label`) → 201 |
| `GET /submissions` | the team's submissions (all submissions for admins), newest first, with current-set `status`, `progress`, `summary`, `partial` and `outdated` |
| `GET /submissions/<id>` | current-set `results` (`summary`, `episodes`, `partial`, `outdated`) plus all scheduled `jobs`, including episodes outside the current set; admins can access any submission |
| `DELETE /submissions/<id>` | baseline accounts' own submissions only: 200 `{"deleted": "<id>"}`; 403 for other account types, 404 if missing or not owned, 409 if any job is running |
| `GET /config` | nested `uploads` and `evaluation` rules, `team_name`, `team_type`, `pause_message`, `evaluation_hashes`, and the caller's resolved `allowance` (base daily quota, used, extras, remaining, unlimited, day) |
| `PUT /config` | admin only: complete `{uploads, evaluation}` settings; returns the same shape as `GET /config` |
| `GET /teams` | admin only: account id/name/type, nullable daily override, unlimited flag, today's extras/usage/remaining, effective daily base quota and day; no tokens |
| `PUT /teams/<id>/allowance` | admin only: `{daily_quota: null or nonnegative integer, unlimited: boolean, extra_uploads: nonnegative integer}`; sets today's total extras and returns the team row |
| `GET /comparison` | current `evaluation_hashes`, one aggregate point per eligible submission from the viewer or shared baselines, baseline and combined frontiers, and improvement area; no other teams' private submissions |
| `PUT /submission-pause` | admin only: JSON `{"message": "reason"}` pauses or updates the message; `{"message": null}` resumes; returns `pause_message` |
| `GET /evaluations` | admin only: current `hashes`, named `presets`, and current-set `outdated_jobs` (finished) / `outdated_running` counts across all teams |
| `PUT /evaluations` | admin only: `{"hashes": ["<sha1>", "..."], "backfill": true}` saves the set and optionally schedules missing jobs for all submissions; returns selection and `scheduled` count |
| `POST /evaluations/schedule` | admin only: optional `submission_id` and `hashes`; defaults to all submissions and the current set; returns `scheduled` count |
| `POST /evaluations/rerun-outdated` | admin only: replaces outdated done/failed jobs for the current set across all teams, including baselines; returns `scheduled` and `skipped_running`; queued jobs stay unchanged |

`uploads` contains `daily_quota`, `quota_time_zone` and `max_upload_mib`. `evaluation` contains
`episode_timeout_seconds` and `limits` (`init_fuel`, `step_fuel`, `memory_bytes`, `table_elements`,
`init_wall_seconds`, `step_wall_seconds`). Each settings update supplies every key; quotas may be zero,
while sizes, timeouts and guest limits must be positive.

A new upload schedules jobs for the current set. Every accepted upload counts toward the team's daily quota plus
today's extras, resetting at midnight in `uploads.quota_time_zone`; failed evaluations do not refund uploads.
Queued or running jobs do not block another upload. Unlimited accounts have no daily quota; their `remaining` is null.
Scheduling or rerunning jobs for existing uploads does not affect the quota. The default episode timeout is
24 minutes (`evaluation.episode_timeout_seconds: 1440`), including any archive download.

Each job snapshots the evaluation settings when it starts. Running jobs finish under their snapshot; queued jobs
use the saved settings when claimed. Results computed with different settings remain readable and are marked
**outdated**, but neither they nor outdated baselines enter comparison points or claimed area. The list and details
show the outdated count; each job's `evaluation_config` snapshot and `outdated` flag appear in Raw JSON.
**Rerun outdated for current set** replaces outdated finished jobs and skips running ones; repeat after those finish
if needed. Upload-only settings and team allowances do not make results outdated.

Displayed status and summaries always use the current set: `queued`, `running`, `incomplete` (missing jobs), `done`
or `failed`. Progress counts finished, completed, failed, queued, running, missing and terminal outdated episodes.
Partial summaries update as selected episodes finish. Submissions have upload timestamps; start/finish times belong
to individual jobs and remain available in the details JSON. The job table marks jobs in the current set.

Teams never see the guest's output: full `report.json`, logs and per-frame `fuel.npy` stay private under
`reports/<submission-id>/<episode-hash>/`. Each episode's `stdout.txt` and `stderr.txt` keeps its first 1 MiB,
and that job's `worker.txt` keeps the first 1 MiB of evaluation
output. Excess output is discarded while evaluation continues. Report tails read at most 16 KiB from each log.
Local `--verbose` streams guest output directly to the console without these capture limits.

Results include maximum, mean, median, p95 and p99 fuel per attempted frame, total step fuel and attempted-frame counts
per episode and for the entire current evaluation set. Current-set statistics pool the original per-frame fuel
samples from their finished jobs; they do not average per-episode percentiles. Judging uses the current-set maximum;
the other statistics are diagnostics. Initialization fuel is reported separately per episode.

Deployment: [`compose.yaml`](compose.yaml) (usage in its header) runs the app behind a Cloudflare tunnel (no open port;
Cloudflare terminates TLS) and the worker unprivileged. Compose defaults to 16 concurrent evaluations; set `WORKER_JOBS` to override.
The worker's memory limit (`WORKER_MEMORY`, default `8g`) is shared across all evaluations. The worker has network access and a writable
`conta-cache` volume for archive downloads. Wasmtime sandboxes the guest, while each evaluation subprocess has a time
limit and capped logs. New selections and scheduled jobs need no service restart.
Compose mounts `evaluation-data/viobench-private/data` beside `compose.yaml` read-only at `/srv/private-episodes`
and sets the worker's `CONTA_SEARCH_PATHS` to it. Provision this directory with `<hash>` or `<hash>.gz` archives
readable by container UID `10001` before starting the worker; a missing directory fails startup.
The entire `evaluation-data` directory is excluded from Git and Docker builds. The normal `conta-cache` volume
remains writable for public downloads and gzip extraction.
Keep the `server-data` volume private (the sqlite holds the tokens) and back it up. Behind a
path prefix, serve the page with a trailing slash (`/prefix/`): its requests are relative to it.

When upgrading an existing Compose deployment, remove the obsolete `episodes` container with
`docker compose -f viobench/tools/server/compose.yaml up -d --build --remove-orphans` (from the repository root).
Use the fresh database described above; this command does not migrate the old database.

[`backup.sh`](backup.sh) copies uploads, reports and SQLite (including team tokens) into a dated local snapshot.
It uses the active Docker context and finds `compose.yaml` next to the script, so it can run from any directory.
The app and worker stop during the local copy; services that were running restart before the optional
rsync, including if the copy fails. An interrupted episode job is requeued when the worker starts.
It also works after `docker compose down`, as long as the `server-data` volume remains. In that case,
Python 3 resolves the volume name from the Compose configuration, and a temporary, never-started container
mounts it read-only using the locally built app image on the active Docker host. The helper is removed
after the copy; the server stays down. A missing volume fails the backup instead of creating an empty one.

```bash
viobench/tools/server/backup.sh  # defaults to ~/viobench-backups
viobench/tools/server/backup.sh /path/to/backups
viobench/tools/server/backup.sh /path/to/backups user@backup-host:/path/to/viobench-backups/
```

Each snapshot is a full copy of `server-data`; rsync transfers all completed snapshots and skips unchanged files.
Failed copies remain in `.partial.*` directories and are excluded from rsync. The downloadable episode cache is
separate from these backups. Keep the snapshots private, like the original volume.
