# Submission server

The app accepts uploads into an SQLite queue and serves the leaderboard; the worker evaluates submissions one at a
time. Both share `RELPOSE_SERVER_DIR` (default `./server-data`).

```bash
pip install -e ".[server,data]"
mkdir -p server-data && printf "token-1 alice\ntoken-2 bob\n" > server-data/tokens.txt
RELPOSE_DATASETS=/data/test/first:/data/test/second python -m relpose.tools.server.worker &
uvicorn relpose.tools.server.app:app --host 0.0.0.0 --port 8000
```

| endpoint | |
|---|---|
| `POST /submit` | `curl -F file=@solution.wasm -F label=v1 -H "X-Token: token-1" http://server:8000/submit` |
| `GET /submissions/<id>` | status and summary |
| `GET /submissions/<id>/metrics` | per-episode and per-dataset metrics, fuel and wall time |
| `GET /leaderboard`, `/leaderboard.json` | every completed submission per dataset, Pareto front marked |

Students get metrics only, never the guest's stdout, stderr or failure messages (they could carry the hidden episodes
out). The worker writes `reports/<id>/metrics.json` from an allowlist of numeric fields; the full `report.json`, the
logs and `worker.txt` stay on the server.

Settings: `RELPOSE_TOKENS`, `RELPOSE_MAX_UPLOAD_MIB` (128), `RELPOSE_DAILY_QUOTA` (20).

Deployment: run the worker unprivileged, in a container without network and with a memory limit (wasmtime sandboxes
the guest, not the host process); TLS in front of the app; back up `submissions.sqlite` and `reports/`.
