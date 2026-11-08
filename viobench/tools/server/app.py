"""FastAPI submission server.  Run:  uvicorn viobench.tools.server.app:app --host 0.0.0.0 --port 8000

    POST /submissions        multipart: file (the .wasm), label (optional)  -> 201 and the new submission
    GET  /submissions        the team's submissions (all for admins), newest first, with overall summaries once evaluated
    GET  /submissions/{id}   one of them, with its results once evaluated
    DELETE /submissions/{id} baseline: delete an own submission when none of its jobs is running
    GET  /config             the rules (`server.config`), team name/type and submission pause message
    GET  /comparison         own and baseline scores for the current evaluation selection
    PUT  /submission-pause   admin: {"message": "reason"} to pause, {"message": null} to resume
    GET/PUT /evaluations    admin: current hashes and presets; save a selection with optional backfill
    POST /evaluations/schedule  admin: schedule missing jobs for all/one submission and current/explicit hashes
    GET  /, /app.js, /style.css   the web UI (ui/): a page on top of the routes above

Every API request carries "Authorization: Bearer <token>" (tokens: `python -m viobench.tools.server.teams`). Submission
lists and details show the team's own submissions, or all submissions for admins; comparisons also show baseline scores.
The UI's files are public.
VIOBENCH_SERVER_DIR: uploads, reports and sqlite (default ./server-data); the rules are
the checked-in viobench/tools/config.json.
The evaluation itself is done by `viobench.tools.server.worker`, which polls the queue.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict

from ..abi import AbiError
from ..abi.module import runtime_descriptor
from ..abi.wasm_binary import iter_sections
from .store import MAX_PAUSE_MESSAGE, Paused, Refused, Store

app = FastAPI(title="viobench submissions", docs_url=None, redoc_url=None, openapi_url=None)  # the routes below, nothing else
_store = None
UI_FILES = {name: (Path(__file__).with_name("ui") / name).read_bytes() for name in ("index.html", "app.js", "style.css")}
UI_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}
UI_HEADERS = {  # the page runs only its own script and talks only to this server
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src data:; "
                               "form-action 'none'; frame-ancestors 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}


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
    """API timestamps are UTC; non-finite metrics are JSON null."""
    for item in [row, *row.get("jobs", [])]:
        for key in ("created", "started", "finished"):
            if item.get(key) is not None:
                item[key] = datetime.fromtimestamp(item[key], timezone.utc).isoformat(timespec="seconds")
    return json.loads(json.dumps(row), parse_constant=lambda constant: None)


@app.post("/submissions", status_code=201)
async def submit(file: UploadFile = File(...), label: str = Form(""), authorization: str | None = Header(default=None)):
    token = authenticate(authorization)
    if get_store().is_admin(token):
        raise HTTPException(status_code=403, detail="admin accounts cannot upload submissions")
    message = get_store().pause_message()
    if message is not None:
        raise HTTPException(status_code=503, detail=message)
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
        raise HTTPException(status_code=400, detail="this is the bare Python runtime without solution code: run `viobench bundle`")
    try:
        submission_id = get_store().add(token, label[:80], file.filename or "solution.wasm", data)
    except Paused as error:  # the admin may have paused while the upload was being read and checked
        raise HTTPException(status_code=503, detail=str(error)) from None
    except Refused as error:  # today's upload quota is used up
        raise HTTPException(status_code=429, detail=str(error)) from None
    return public(get_store().get(token, submission_id))


@app.get("/submissions")
def submissions(authorization: str | None = Header(default=None)):
    rows = [public(row) for row in get_store().submissions(authenticate(authorization))]
    for row in rows:  # the headline numbers only; GET /submissions/{id} has the rest
        results = row.pop("results")
        row["overall"] = (results["summary"] or {}).get("overall")
        row["partial"] = results["partial"]
    return rows


@app.get("/submissions/{submission_id}")
def submission(submission_id: str, authorization: str | None = Header(default=None)):
    row = get_store().get(authenticate(authorization), submission_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such submission")
    return public(row)


@app.delete("/submissions/{submission_id}")
def delete_submission(submission_id: str, authorization: str | None = Header(default=None)):
    token = authenticate(authorization)
    try:
        get_store().delete_submission(token, submission_id)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from None
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except Refused as error:
        raise HTTPException(status_code=409, detail=str(error)) from None
    except OSError:
        logging.exception("Submission %s was deleted, but file cleanup failed", submission_id)
        raise HTTPException(status_code=500, detail="Submission deleted, but server file cleanup failed.") from None
    return {"deleted": submission_id}


@app.get("/config")
def configuration(authorization: str | None = Header(default=None)):
    token = authenticate(authorization)
    return {**get_store().config.as_dict(), "team_name": get_store().team_name(token),
            "team_type": get_store().team_type(token), "pause_message": get_store().pause_message(),
            "evaluation_hashes": get_store().evaluation_hashes()}


@app.get("/comparison")
def comparison(authorization: str | None = Header(default=None)):
    return get_store().comparison(authenticate(authorization))


def require_admin(authorization):
    if not get_store().is_admin(authenticate(authorization)):
        raise HTTPException(status_code=403, detail="admin access required")


class EvaluationSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    hashes: list[str]
    backfill: bool = False


class EvaluationSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    submission_id: str | None = None
    hashes: list[str] | None = None


@app.get("/evaluations")
def evaluations(authorization: str | None = Header(default=None)):
    require_admin(authorization)
    return get_store().evaluations()


@app.put("/evaluations")
def select_evaluations(selection: EvaluationSelection, authorization: str | None = Header(default=None)):
    require_admin(authorization)
    try:
        scheduled = get_store().set_evaluations(selection.hashes, backfill=selection.backfill)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    return {**get_store().evaluations(), "scheduled": scheduled}


@app.post("/evaluations/schedule")
def schedule_evaluations(selection: EvaluationSchedule, authorization: str | None = Header(default=None)):
    require_admin(authorization)
    try:
        scheduled = get_store().schedule_evaluations(selection.hashes, submission_id=selection.submission_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    return {"scheduled": scheduled}


@app.put("/submission-pause")
def submission_pause(message: str | None = Body(..., embed=True, max_length=MAX_PAUSE_MESSAGE, strict=True),
                     authorization: str | None = Header(default=None)):
    require_admin(authorization)
    try:
        return {"pause_message": get_store().set_pause_message(message)}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


def ui_file(name):
    return Response(UI_FILES[name], media_type=UI_TYPES[Path(name).suffix], headers=UI_HEADERS)


@app.get("/")
def page():
    return ui_file("index.html")


@app.get("/app.js")
def script():
    return ui_file("app.js")


@app.get("/style.css")
def style():
    return ui_file("style.css")
