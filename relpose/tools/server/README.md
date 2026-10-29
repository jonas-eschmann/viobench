# Submission server

The app queues uploads in SQLite and shows each team its own submissions; the worker evaluates them one at a time.
Both use `RELPOSE_SERVER_DIR` (default `./server-data`).

```bash
pip install -e ".[server,data]"
python -m relpose.tools.server.teams add "team a" "team b"     # prints a token per team; `teams list` repeats them
RELPOSE_DATASETS=/data/test/first:/data/test/second python -m relpose.tools.server.worker &   # exactly one worker
uvicorn relpose.tools.server.app:app --host 0.0.0.0 --port 8000
```

A team is its token: every request sends `Authorization: Bearer <token>` and sees only that team's submissions.

| endpoint | |
|---|---|
| `POST /submissions` | multipart `file` (and `label`) → 201 |
| `GET /submissions` | the team's submissions, newest first: id, label, sha256, times (UTC), `status` (`queued`, `running`, `done`, `failed`), `error` |
| `GET /submissions/<id>` | one of them, with `results` (metrics only) once done |
| `GET /config` | the rules below |

Rules: [`relpose/tools/config.json`](../config.json), checked in; every key is required, and its `limits` are what
`relpose check` and `evaluate` apply locally too (restart app and worker after editing).

A team has one submission in flight at a time and `daily_quota` per day (from midnight in `quota_time_zone`); up to
`daily_failed_quota` failed ones (crashed, timed out or no episode completed; no results) do not count. Teams never
see the guest's output: `report.json` and the logs stay in `reports/<id>/`.

Deployment: [`compose.yaml`](compose.yaml) (usage in its header) runs the app on 0.0.0.0:8000 and the worker
unprivileged, without network and with a memory limit (wasmtime sandboxes the guest, not the host process); put TLS in
front of the app; keep the `server-data` volume private (the sqlite holds the tokens) and back it up.
