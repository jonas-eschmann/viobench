# Submission server

The app queues uploads in SQLite and shows each team its own submissions; the worker evaluates them one at a time.
Both use `VIOBENCH_SERVER_DIR` (default `./server-data`).

```bash
pip install -e ".[server]"
python -m viobench.tools.server.teams add "team a" "team b"     # prints a token per team; `teams list` repeats them
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

| endpoint | |
|---|---|
| `GET /` (`/app.js`, `/style.css`) | the web UI; public files, strict Content-Security-Policy |
| `POST /submissions` | multipart `file` (and `label`) → 201 |
| `GET /submissions` | the team's submissions, newest first: id, label, sha256, times (UTC), `status` (`queued`, `running`, `done`, `failed`), `error`, `overall` (headline medians once done) |
| `GET /submissions/<id>` | one of them, with `results` (metrics only) once done |
| `GET /config` | the rules below |

Rules: [`viobench/tools/config.json`](../config.json), checked in; every key is required, and its `limits` are what
`viobench check` and `evaluate` apply locally too (restart app and worker after editing).

A team has one submission in flight at a time and `daily_quota` per day (from midnight in `quota_time_zone`); up to
`daily_failed_quota` failed ones (crashed, timed out or no episode completed; no results) do not count. A team created
with `teams add --unlimited` (e.g. the instructors' debug team; the `unlimited` column of `teams`) has no daily quota,
but still one submission at a time.
Teams never see the guest's output: `report.json` and the logs stay in `reports/<id>/`.

Results include maximum, mean, median, p95 and p99 fuel per attempted frame, total step fuel and attempted-frame counts
per episode, per dataset and overall. Dataset and overall statistics pool the frames. Judging uses the overall maximum;
the other statistics are diagnostics. Initialization fuel is reported separately per episode.

Deployment: [`compose.yaml`](compose.yaml) (usage in its header) runs the app behind a Cloudflare tunnel (no open port;
Cloudflare terminates TLS) and the worker unprivileged, without network and with a memory limit (wasmtime sandboxes
the guest, not the host process); a one-shot `episodes` service fills the `conta-cache` volume (`worker --fetch`)
before the worker starts. Keep the `server-data` volume private (the sqlite holds the tokens) and back it up. Behind a
path prefix, serve the page with a trailing slash (`/prefix/`): its requests are relative to it.

[`backup.sh`](backup.sh) copies uploads, reports and SQLite (including team tokens) into a dated local snapshot.
It uses the active Docker context and finds `compose.yaml` next to the script, so it can run from any directory.
The app and worker stop during the local copy; services that were running restart before the optional rsync,
including if the copy fails. An interrupted evaluation is requeued when the worker starts.

```bash
viobench/tools/server/backup.sh  # defaults to ~/viobench-backups
viobench/tools/server/backup.sh /path/to/backups
viobench/tools/server/backup.sh /path/to/backups user@backup-host:/path/to/viobench-backups/
```

Each snapshot is a full copy of `server-data`; rsync transfers all completed snapshots and skips unchanged files.
Failed copies remain in `.partial.*` directories and are excluded from rsync. The downloadable episode cache is
separate from these backups. Keep the snapshots private, like the original volume.
