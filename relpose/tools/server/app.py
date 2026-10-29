"""FastAPI submission server.  Run:  uvicorn relpose.tools.server.app:app --host 0.0.0.0 --port 8000

    POST /submissions        multipart: file (the .wasm), label (optional)  -> 201 and the new submission
    GET  /submissions        the team's submissions, newest first
    GET  /submissions/{id}   one of them, with its results once evaluated
    GET  /config             the rules: daily quota, upload size, evaluation timeout, sandbox limits (`server.config`)

Every request carries "Authorization: Bearer <token>" (tokens: `python -m relpose.tools.server.teams`) and sees only
its own team's submissions. RELPOSE_SERVER_DIR: uploads, reports, sqlite and config.json (default ./server-data).
The evaluation itself is done by `relpose.tools.server.worker`, which polls the queue.
"""
import json
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from ..abi import AbiError
from ..abi.module import runtime_descriptor
from ..abi.wasm_binary import iter_sections
from .store import Store

app = FastAPI(title="relpose submissions", docs_url=None, redoc_url=None, openapi_url=None)  # the routes below, nothing else
_store = None


def get_store():
    global _store
    if _store is None:
        _store = Store()
    return _store


def authenticate(authorization):
    """The token of an "Authorization: Bearer <token>" header, i.e. the team; 401 unless it is one."""
    scheme, _, token = (authorization or "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not get_store().is_team(token):
        raise HTTPException(status_code=401, detail="unknown token", headers={"WWW-Authenticate": "Bearer"})
    return token


def public(row):
    """A submission as the API shows it: times in ISO 8601 (UTC), results parsed."""
    for key in ("created", "started", "finished"):
        if row[key] is not None:
            row[key] = datetime.fromtimestamp(row[key], timezone.utc).isoformat(timespec="seconds")
    if "results" in row:
        row["results"] = json.loads(row["results"]) if row["results"] else None
    return row


@app.post("/submissions", status_code=201)
async def submit(file: UploadFile = File(...), label: str = Form(""), authorization: str | None = Header(default=None)):
    token = authenticate(authorization)
    config = get_store().config
    data = await file.read()
    if len(data) > config.max_upload_mib << 20:
        raise HTTPException(status_code=413, detail=f"file larger than {config.max_upload_mib} MiB")
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
    submission_id = get_store().add(token, label[:80], file.filename or "solution.wasm", data)
    if submission_id is None:
        raise HTTPException(status_code=429, detail=f"quota of {config.daily_quota} submission(s) per day reached; "
                                                    f"it resets at midnight ({config.quota_time_zone})")
    return public(get_store().get(token, submission_id))


@app.get("/submissions")
def submissions(authorization: str | None = Header(default=None)):
    return [public(row) for row in get_store().submissions(authenticate(authorization))]


@app.get("/submissions/{submission_id}")
def submission(submission_id: str, authorization: str | None = Header(default=None)):
    row = get_store().get(authenticate(authorization), submission_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such submission")
    return public(row)


@app.get("/config")
def configuration(authorization: str | None = Header(default=None)):
    authenticate(authorization)
    return get_store().config.as_dict()
