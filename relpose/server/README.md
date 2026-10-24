# Submission server

Two processes share one directory (`RELPOSE_SERVER_DIR`, default `./server-data`): the web app accepts uploads into
an SQLite queue, the worker evaluates them one at a time on the hidden datasets and the app serves the leaderboard.

```bash
pip install -e ".[server,data]"
mkdir -p server-data && printf "s3cr3t-token-1 alice\ns3cr3t-token-2 bob\n" > server-data/tokens.txt
RELPOSE_DATASETS=/data/hidden/default:/data/hidden/blur:/data/hidden/lowlight:/data/hidden/oversaturated:/data/hidden/real \
    python -m relpose.server.worker &
uvicorn relpose.server.app:app --host 0.0.0.0 --port 8000
```

Students:

```bash
curl -F file=@solution.wasm -F label=orb-v2 -H "X-Token: s3cr3t-token-1" http://server:8000/submit
curl -H "X-Token: s3cr3t-token-1" http://server:8000/submissions/7          # status + summary
curl -H "X-Token: s3cr3t-token-1" http://server:8000/submissions/7/logs     # stderr tails per episode
curl -H "X-Token: s3cr3t-token-1" http://server:8000/submissions/7/report   # full report.json
```

`GET /leaderboard` (HTML) and `/leaderboard.json` list every completed submission per dataset category with the Pareto
front highlighted; the hidden datasets never leave the server (students see metrics and their own logs).

Hardening checklist for a public deployment: run the worker as a dedicated user in a container without network and
with a memory limit (the wasm guest is sandboxed by wasmtime and capped by `Limits`, the host Python process is not),
put the app behind TLS, rotate tokens, keep `RELPOSE_DAILY_QUOTA` modest, and back up `submissions.sqlite`
plus `reports/`.
