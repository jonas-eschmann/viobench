# Submission server

The app takes uploads into an SQLite queue and shows each team its own submissions and results; the worker evaluates
submissions one at a time. Both share `RELPOSE_SERVER_DIR` (default `./server-data`).

```bash
pip install -e ".[server,data]"
python -m relpose.tools.server.teams add "team a" "team b"     # prints a token per team; `teams list` shows them again
RELPOSE_DATASETS=/data/test/first:/data/test/second python -m relpose.tools.server.worker &
uvicorn relpose.tools.server.app:app --host 0.0.0.0 --port 8000
```

A team is its token (32 random base64url characters). Every request carries `Authorization: Bearer <token>` and sees
only that token's submissions.

| endpoint | |
|---|---|
| `POST /submissions` | `curl -F file=@solution.wasm -F label=v1 -H "Authorization: Bearer $TOKEN" https://server/submissions` → 201 |
| `GET /submissions` | the team's submissions, newest first: `id`, `label`, `filename`, `sha256`, `bytes`, `created`, `started`, `finished` (UTC), `status` (`queued`, `running`, `done`, `failed`), `error` |
| `GET /submissions/<id>` | one of them, plus `results` once done: per-dataset summary and per-episode metrics, fuel and wall time |
| `GET /config` | the rules in force (below), defaults filled in |

Submission ids are random (16 hex digits), so they say nothing about how many submissions exist. Teams get metrics
only, never the guest's stdout, stderr or failure messages (they could carry the hidden episodes out): the worker stores
an allowlist of numeric fields as `results`, and the full `report.json`, the logs and `worker.txt` stay in
`reports/<id>/`. There is no leaderboard for now; `relpose leaderboard server-data/reports` aggregates the reports
offline (labelled by submission id).

The rules live in `server-data/config.json`; every key is optional, and a missing file means the defaults. Restart
the app and the worker after editing it.

```json
{"daily_quota": 1, "quota_time_zone": "America/Los_Angeles", "max_upload_mib": 128, "evaluation_timeout_seconds": 14400,
 "limits": {"init_fuel": 200000000000, "step_fuel": 20000000000, "memory_bytes": 2147483648, "table_elements": 1048576,
            "init_wall_seconds": 300, "step_wall_seconds": 60}}
```

The quota counts per calendar day in `quota_time_zone`; `limits` are the sandbox's (`relpose.tools.runner.sandbox.Limits`),
which the worker hands to `relpose evaluate --limits`. Teams can evaluate under the same limits locally with
`relpose evaluate my.wasm data --limits "$(curl -s -H "Authorization: Bearer $TOKEN" https://server/config)"`.
The environment only says where things are: `RELPOSE_SERVER_DIR` and the worker's `RELPOSE_DATASETS`.

Run exactly one worker: on start it requeues whatever a previous worker left running.

Deployment: run the worker unprivileged, in a container without network and with a memory limit (wasmtime sandboxes
the guest, not the host process); TLS in front of the app; keep `submissions.sqlite` private (it holds the tokens) and
back it up with `reports/`.
