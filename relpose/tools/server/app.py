"""FastAPI submission server.  Run:  uvicorn relpose.tools.server.app:app --host 0.0.0.0 --port 8000

Environment: RELPOSE_SERVER_DIR (uploads, reports, sqlite; default ./server-data), RELPOSE_TOKENS (file with
"token user" lines; default <dir>/tokens.txt), RELPOSE_MAX_UPLOAD_MIB (128), RELPOSE_DAILY_QUOTA (20).
The evaluation itself is done by `relpose.tools.server.worker`, which polls the queue.
"""
import html
import json
import os
import time

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..abi import AbiError
from ..abi.module import runtime_descriptor
from ..abi.wasm_binary import iter_sections
from ..runner.leaderboard import pareto_front
from .store import DAILY_QUOTA, MAX_UPLOAD_BYTES, Store

app = FastAPI(title="relpose challenge")
_store = None


def get_store():
    global _store
    if _store is None:
        _store = Store()
    return _store


def authenticate(token):
    user = get_store().tokens().get(token or "")
    if user is None:
        raise HTTPException(status_code=401, detail="unknown token")
    return user


@app.post("/submit")
async def submit(file: UploadFile = File(...), label: str = Form(""), x_token: str | None = Header(default=None)):
    user = authenticate(x_token)
    if get_store().submissions_today(user) >= DAILY_QUOTA:
        raise HTTPException(status_code=429, detail=f"daily quota of {DAILY_QUOTA} submissions reached")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"file larger than {MAX_UPLOAD_BYTES >> 20} MiB")
    try:
        sections = list(iter_sections(data))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"not a WebAssembly module: {error}") from None
    if not sections:
        raise HTTPException(status_code=400, detail="empty module")
    try:
        descriptor = runtime_descriptor(data)
    except AbiError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    if descriptor is not None and descriptor.length == 0:
        raise HTTPException(status_code=400, detail="this is the bare Python runtime without solution code: run `relpose bundle`")
    submission_id = get_store().add(user, label[:80], file.filename or "solution.wasm", data)
    return {"id": submission_id, "user": user, "bytes": len(data), "kind": "python" if descriptor else "native", "status": "queued"}


@app.get("/submissions/{submission_id}")
def submission(submission_id: int, x_token: str | None = Header(default=None)):
    user = authenticate(x_token)
    row = get_store().get(submission_id)
    if row is None or row["user"] != user:
        raise HTTPException(status_code=404, detail="no such submission")
    row["summary"] = json.loads(row["summary"]) if row["summary"] else None
    return row


@app.get("/submissions/{submission_id}/report")
def report(submission_id: int, x_token: str | None = Header(default=None)):
    user = authenticate(x_token)
    row = get_store().get(submission_id)
    if row is None or row["user"] != user:
        raise HTTPException(status_code=404, detail="no such submission")
    path = get_store().report_dir(submission_id) / "report.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no report yet")
    return FileResponse(path, media_type="application/json")


@app.get("/submissions/{submission_id}/logs")
def logs(submission_id: int, x_token: str | None = Header(default=None), lines: int = 200):
    user = authenticate(x_token)
    row = get_store().get(submission_id)
    if row is None or row["user"] != user:
        raise HTTPException(status_code=404, detail="no such submission")
    out = {}
    report_dir = get_store().report_dir(submission_id)
    for path in sorted(report_dir.rglob("*.txt")):
        text = path.read_text(errors="replace").splitlines()
        if text:
            out[str(path.relative_to(report_dir))] = "\n".join(text[-lines:])
    return JSONResponse(out)


@app.get("/mine")
def mine(x_token: str | None = Header(default=None)):
    user = authenticate(x_token)
    rows = get_store().list(user=user)
    for row in rows:
        row["summary"] = json.loads(row["summary"]) if row["summary"] else None
    return rows


@app.get("/leaderboard.json")
def leaderboard_json():
    return get_store().leaderboard()


@app.get("/leaderboard", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def leaderboard():
    points = get_store().leaderboard()
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Pose challenge leaderboard</title>",
             "<style>body{font-family:system-ui,sans-serif;margin:2em;max-width:1100px}table{border-collapse:collapse;margin-bottom:2em}",
             "td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}th{background:#eee}td:first-child,td:nth-child(2){text-align:left}",
             ".pareto{font-weight:bold}</style></head><body><h1>Relative pose challenge</h1>",
             f"<p>{len(get_store().list(limit=100000))} submissions, {len(get_store().list(status='queued'))} queued, {len(get_store().list(status='running'))} running. ",
             "Submit with <code>curl -F file=@solution.wasm -F label=v1 -H 'X-Token: ...' /submit</code>.</p>"]
    for dataset, rows in sorted(points.items()):
        front = {rows[i]["id"] for i in pareto_front([(row["fuel"], row["ate"]) for row in rows])}
        parts.append(f"<h2>{html.escape(dataset)}</h2><table><tr><th>user</th><th>label</th><th>ATE [m]</th><th>RPE [m/s]</th>"
                     "<th>fuel/frame</th><th>fuel p95</th><th>ms/frame</th><th>episodes</th><th>id</th><th>when</th></tr>")
        for row in sorted(rows, key=lambda r: r["ate"]):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["created"]))
            css = " class='pareto'" if row["id"] in front else ""
            parts.append(f"<tr{css}><td>{html.escape(row['user'])}</td><td>{html.escape(row['label'] or '')}</td><td>{row['ate']:.3f}</td>"
                         f"<td>{row['rpe']:.3f}</td><td>{row['fuel']:,.0f}</td><td>{row['fuel_p95']:,.0f}</td><td>{row['ms']:.1f}</td>"
                         f"<td>{row['completed']}/{row['episodes']}</td><td>{row['id']}</td><td>{when}</td></tr>")
        parts.append("</table>")
    parts.append("<p>Bold rows are on the Pareto front (no other submission has both lower fuel and lower ATE).</p></body></html>")
    return "".join(parts)
