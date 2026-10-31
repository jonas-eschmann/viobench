"use strict";
// The web UI on top of the submission API (app.py): plain DOM, no dependencies, nothing loaded from elsewhere.
// Everything the server sends is inserted as text (textContent), never parsed as HTML.

const TOKEN_KEY = "relpose-token";
const REQUEST_TIMEOUT_MS = 30000;
const UPLOAD_STALL_MS = 120000;  // an upload that makes no progress for this long is given up
const POLL_MS = 15000;  // while a submission is queued or running
const MAX_RETRY_MS = 120000;
const DASH = "—";
const STATUS_TEXT = {0: "the server cannot be reached", 413: "the file is too large for the server", 502: "the server is unavailable",
                     503: "the server is unavailable", 504: "the server did not answer in time"};

const $ = (id) => document.getElementById(id);
let token = null;
let config = null;
let pollTimer = null;
let waiting = false;  // a refresh came due while the tab was hidden
let retryMs = POLL_MS;
let generation = 0;  // only the newest list request renders
let uploading = false;
let shown = null;  // {id, status} of the submission in the details section
const finished = new Map();  // id -> details of done and failed submissions, which no longer change

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// --- the token: for this tab (sessionStorage) or remembered (localStorage); either may be unavailable ------------

function loadToken() {
  for (const kind of ["sessionStorage", "localStorage"]) {
    try {
      const value = window[kind].getItem(TOKEN_KEY);
      if (value) return value;
    } catch (error) { /* storage disabled */ }
  }
  return null;
}

function forgetToken() {
  for (const kind of ["sessionStorage", "localStorage"]) {
    try {
      window[kind].removeItem(TOKEN_KEY);
    } catch (error) { /* storage disabled */ }
  }
}

function storeToken(value, remember) {
  forgetToken();
  try {
    window[remember ? "localStorage" : "sessionStorage"].setItem(TOKEN_KEY, value);
  } catch (error) { /* storage disabled: signed in until the page is reloaded */ }
}

// --- requests --------------------------------------------------------------------------------------------------

function answer(status, statusText, text) {
  // The JSON body of a successful response, else an ApiError with the server's detail, or with the status when the
  // body is not the API's (a proxy's HTML error page, say).
  let body = null;
  try {
    body = JSON.parse(text);
  } catch (error) { /* not JSON */ }
  if (status >= 200 && status < 300) {
    if (body === null || typeof body !== "object") throw new ApiError(status, "the server sent an unreadable answer");
    return body;
  }
  if (body && typeof body.detail === "string") throw new ApiError(status, body.detail);
  throw new ApiError(status, STATUS_TEXT[status] || `the server answered ${status} ${statusText || ""}`.trim());
}

async function api(path) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(path, {headers: {Authorization: `Bearer ${token}`}, cache: "no-store", signal: controller.signal});
    return answer(response.status, response.statusText, await response.text());
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(0, error.name === "AbortError" ? "the server did not answer in time" : STATUS_TEXT[0]);
  } finally {
    clearTimeout(timer);
  }
}

function upload(file, label, onProgress) {
  // XMLHttpRequest rather than fetch: only it reports upload progress.
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    let lastProgress = Date.now();
    const watchdog = setInterval(() => {
      if (Date.now() - lastProgress > UPLOAD_STALL_MS) request.abort();
    }, 5000);
    const interrupted = () => {
      clearInterval(watchdog);
      reject(new ApiError(0, "the upload was interrupted: check the list below before submitting again"));
    };
    request.upload.onprogress = (event) => {
      lastProgress = Date.now();
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onload = () => {
      clearInterval(watchdog);
      try {
        resolve(answer(request.status, request.statusText, request.responseText));
      } catch (error) {
        reject(error);
      }
    };
    request.onerror = interrupted;
    request.onabort = interrupted;
    request.open("POST", "submissions");
    request.setRequestHeader("Authorization", `Bearer ${token}`);
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("label", label);
    request.send(form);
  });
}

// --- formatting and DOM ----------------------------------------------------------------------------------------

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function decimal(value) {
  if (!isNumber(value)) return DASH;
  return Math.abs(value) >= 1000 ? Math.round(value).toLocaleString() : String(Number(value.toPrecision(3)));
}

function fuel(value) {
  if (!isNumber(value)) return DASH;
  for (const [scale, unit] of [[1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"]]) {
    if (Math.abs(value) >= scale) return `${(value / scale).toFixed(1)} ${unit}`;
  }
  return String(Math.round(value));
}

function time(iso) {
  const date = new Date(iso);
  return iso && !Number.isNaN(date.getTime()) ? date.toLocaleString() : DASH;
}

function size(bytes) {
  if (!isNumber(bytes)) return DASH;
  return bytes < 2 ** 20 ? `${Math.max(1, Math.round(bytes / 1024))} KiB` : `${(bytes / 2 ** 20).toFixed(1)} MiB`;
}

function get(object, ...keys) {
  for (const key of keys) object = object !== null && typeof object === "object" ? object[key] : undefined;
  return object;
}

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined && text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}

function tableRow(cells, tag = "td") {
  const row = document.createElement("tr");
  for (const cell of cells) {
    const element = document.createElement(tag);
    if (cell instanceof Node) element.append(cell);
    else element.textContent = cell === null || cell === undefined ? "" : String(cell);
    row.append(element);
  }
  return row;
}

function table(headers, rows) {
  const element = document.createElement("table");
  element.createTHead().append(tableRow(headers, "th"));
  const body = element.createTBody();
  for (const cells of rows) body.append(tableRow(cells));
  const wrapper = node("div", null, "scroll");
  wrapper.append(element);
  return wrapper;
}

function notice(message, kind = "error") {
  const element = $("notice");
  element.textContent = message || "";
  element.className = kind;
  element.hidden = !message;
}

// --- signing in and out ----------------------------------------------------------------------------------------

function showSignIn(message, kind = "error") {
  clearTimeout(pollTimer);
  generation++;  // drop answers still on their way
  token = null;
  config = null;
  shown = null;
  finished.clear();
  for (const id of ["team", "sign-out", "details", "connection"]) $(id).hidden = true;
  $("sign-in").hidden = false;
  notice(message, kind);
  $("token").focus();
}

async function signIn(value, remember) {
  // remember: true/false from the form, null for a token loaded from storage (left where it is)
  token = value.trim();
  const button = $("token-form").querySelector("button");
  button.disabled = true;
  try {
    config = await api("config");
  } catch (error) {
    const tried = token;
    if (error.status === 401) {
      forgetToken();
      showSignIn("Unknown token: check it with your instructors.");
    } else {
      showSignIn(`Cannot sign in: ${error.message}. Try again in a moment.`);
      $("token").value = tried;
    }
    return;
  } finally {
    button.disabled = false;
  }
  if (remember !== null) storeToken(token, remember);
  $("token").value = "";
  $("sign-in").hidden = true;
  $("team").hidden = false;
  $("sign-out").hidden = false;
  notice("");
  renderRules(config);
  refresh();
}

// --- the list of submissions -----------------------------------------------------------------------------------

function schedule(ms) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(() => {
    if (document.hidden) waiting = true;
    else refresh();
  }, ms);
}

async function refresh() {
  clearTimeout(pollTimer);
  const mine = ++generation;
  let rows;
  try {
    rows = await api("submissions");
    if (!Array.isArray(rows)) throw new ApiError(200, "the server sent an unreadable answer");
  } catch (error) {
    if (mine !== generation) return;
    if (error.status === 401) return showSignIn("The token is no longer valid.");
    const connection = $("connection");
    connection.textContent = `Cannot update the list: ${error.message}. Trying again in ${Math.round(retryMs / 1000)} s.`;
    connection.hidden = false;
    schedule(retryMs);
    retryMs = Math.min(retryMs * 2, MAX_RETRY_MS);
    return;
  }
  if (mine !== generation) return;
  retryMs = POLL_MS;
  $("connection").hidden = true;
  renderList(rows);
  $("updated").textContent = `updated ${new Date().toLocaleTimeString()}`;
  const pending = rows.find((row) => row && (row.status === "queued" || row.status === "running"));
  renderUploadState(pending);
  if (pending) schedule(POLL_MS);
  const current = shown && rows.find((row) => row && row.id === shown.id);
  if (current && current.status !== shown.status) showDetails(current.id, false);
}

function renderList(rows) {
  const body = $("submissions").tBodies[0];
  body.replaceChildren();
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const overall = row.overall || {};
    const status = node("span", row.status === "failed" && row.error ? `failed: ${row.error}` : row.status, `status ${row.status}`);
    const button = node("button", "Details");
    button.type = "button";
    button.setAttribute("aria-label", `Details of ${row.label || row.id}`);
    button.addEventListener("click", () => showDetails(row.id, true));
    const episodes = isNumber(overall.episodes) ? `${overall.completed}/${overall.episodes}` : DASH;
    body.append(tableRow([time(row.created), row.label || DASH, status, decimal(overall.rpe_translation_rmse_median),
                          fuel(overall.fuel_per_frame_median), episodes, button]));
  }
  $("empty").hidden = body.rows.length > 0;
}

// --- submitting ------------------------------------------------------------------------------------------------

function renderUploadState(pending) {
  $("upload-button").disabled = uploading || Boolean(pending);
  const hint = $("upload-hint");
  if (pending) {
    hint.textContent = `"${pending.label || pending.id}" is ${pending.status}: submit the next one once it has finished.`;
  } else if (isNumber(get(config, "daily_quota"))) {
    hint.textContent = `One submission at a time and ${config.daily_quota} per day (from midnight, ${config.quota_time_zone}); ` +
                       `up to ${config.max_upload_mib} MiB.`;
  } else {
    hint.textContent = "";
  }
  hint.hidden = !hint.textContent;
}

async function submit(event) {
  event.preventDefault();
  const file = $("file").files[0];
  if (!file) return;
  const limit = get(config, "max_upload_mib");
  if (isNumber(limit) && file.size > limit * 2 ** 20) return notice(`${file.name} is ${size(file.size)}; the limit is ${limit} MiB.`);
  if (file.size === 0) return notice(`${file.name} is empty.`);
  uploading = true;
  $("upload-button").disabled = true;
  const progress = $("upload-progress");
  progress.value = 0;
  progress.hidden = false;
  notice(`Uploading ${file.name} (${size(file.size)}) …`, "info");
  try {
    const submission = await upload(file, $("label").value.trim(), (fraction) => { progress.value = fraction; });
    notice(`Submitted ${file.name}: it is ${submission.status || "queued"}.`, "info");
    $("upload-form").reset();
  } catch (error) {
    if (error.status === 401) return showSignIn("The token is no longer valid.");
    notice(error.status === 0 ? `${error.message}.` : `Not submitted: ${error.message}`);
  } finally {
    uploading = false;
    progress.hidden = true;
    if (token !== null) refresh();
  }
}

// --- one submission --------------------------------------------------------------------------------------------

const SUMMARY_COLUMNS = [
  ["Episodes", (s) => (isNumber(s.episodes) ? `${s.completed}/${s.episodes}` : DASH)],
  ["RPE (m)", (s) => decimal(s.rpe_translation_rmse_median)],
  ["RPE (rad)", (s) => decimal(s.rpe_rotation_rmse_median)],
  ["ATE (m)", (s) => decimal(s.ate_rmse_median)],
  ["Velocity (m/s)", (s) => decimal(s.velocity_rmse_median)],
  ["Fuel/frame", (s) => fuel(s.fuel_per_frame_median)],
  ["Fuel/frame p95", (s) => fuel(s.fuel_per_frame_p95)],
  ["Wall/frame (ms)", (s) => decimal(s.wall_per_frame_median_ms)],
];
const EPISODE_COLUMNS = [
  ["Dataset", (e) => e.dataset],
  ["Episode", (e) => e.name],
  ["Frames", (e) => e.frames],
  ["Completed", (e) => (e.completed === true ? "yes" : e.completed === false ? "no" : DASH)],
  ["RPE (m)", (e) => decimal(get(e, "metrics", "rpe_translation_rmse"))],
  ["RPE (rad)", (e) => decimal(get(e, "metrics", "rpe_rotation_rmse"))],
  ["ATE (m)", (e) => decimal(get(e, "metrics", "ate_rmse"))],
  ["Drift/m", (e) => decimal(get(e, "metrics", "drift_per_metre"))],
  ["Fuel/frame", (e) => fuel(e.fuel_per_frame_median)],
  ["Fuel/frame p95", (e) => fuel(e.fuel_per_frame_p95)],
];

async function showDetails(id, scroll) {
  let submission = finished.get(id);
  if (!submission) {
    try {
      submission = await api(`submissions/${encodeURIComponent(id)}`);
    } catch (error) {
      if (error.status === 401) return showSignIn("The token is no longer valid.");
      return notice(`Cannot show the submission: ${error.message}`);
    }
    if (token === null) return;
    if (submission.status === "done" || submission.status === "failed") finished.set(id, submission);
  }
  shown = {id, status: submission.status};
  renderDetails(submission);
  if (scroll) $("details").scrollIntoView({block: "start"});
}

function renderDetails(submission) {
  $("details-title").textContent = submission.label || submission.id || "Submission";
  const facts = $("details-facts");
  facts.replaceChildren();
  for (const [label, value] of [["Id", submission.id], ["File", `${submission.filename || DASH} (${size(submission.bytes)})`],
                                ["SHA-256", submission.sha256], ["Status", submission.status], ["Error", submission.error],
                                ["Submitted", time(submission.created)], ["Started", time(submission.started)],
                                ["Finished", time(submission.finished)]]) {
    if (value !== null && value !== undefined) facts.append(node("dt", label), node("dd", value));
  }
  const results = $("details-results");
  results.replaceChildren();
  const summary = get(submission, "results", "summary");
  if (summary && typeof summary === "object") {
    const groups = [["overall", summary.overall], ...Object.entries(summary.datasets || {})];
    results.append(node("h3", "Summary (medians over episodes)"),
                   table(["", ...SUMMARY_COLUMNS.map(([title]) => title)],
                         groups.map(([name, values]) => [name, ...SUMMARY_COLUMNS.map(([, format]) => format(values || {}))])));
  }
  const episodes = get(submission, "results", "episodes");
  if (Array.isArray(episodes) && episodes.length) {
    results.append(node("h3", "Episodes"), table(EPISODE_COLUMNS.map(([title]) => title),
                                                 episodes.map((episode) => EPISODE_COLUMNS.map(([, format]) => format(episode || {})))));
  }
  $("details-json").textContent = JSON.stringify(submission, null, 2);
  $("details").hidden = false;
}

// --- the rules (GET /config) -----------------------------------------------------------------------------------

const RULES = {
  daily_quota: ["Submissions per day", String],
  daily_failed_quota: ["Failed ones per day that do not count", String],
  quota_time_zone: ["A day starts at midnight in", String],
  max_upload_mib: ["Largest upload", (v) => `${v} MiB`],
  evaluation_timeout_seconds: ["Time for the whole evaluation", (v) => `${decimal(v / 3600)} h`],
  init_fuel: ["Fuel for init", fuel],
  step_fuel: ["Fuel per step", fuel],
  memory_bytes: ["Memory", (v) => `${decimal(v / 2 ** 30)} GiB`],
  table_elements: ["Table elements", (v) => v.toLocaleString()],
  init_wall_seconds: ["Wall-clock time for init", (v) => `${v} s`],
  step_wall_seconds: ["Wall-clock time per step", (v) => `${v} s`],
};

function renderRules(rules) {
  const list = $("rules");
  list.replaceChildren();
  const flat = Object.entries(rules || {}).flatMap(([key, value]) => (value && typeof value === "object" ? Object.entries(value) : [[key, value]]));
  for (const [key, value] of flat) {
    const [label, format] = RULES[key] || [key, String];
    list.append(node("dt", label), node("dd", isNumber(value) ? format(value) : String(value)));
  }
}

// --- start -----------------------------------------------------------------------------------------------------

window.addEventListener("error", () => notice("Something went wrong on this page: reloading it should help."));
window.addEventListener("unhandledrejection", () => notice("Something went wrong on this page: reloading it should help."));
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && waiting && token !== null) {
    waiting = false;
    refresh();
  }
});
$("token-form").addEventListener("submit", (event) => {
  event.preventDefault();
  signIn($("token").value, $("remember").checked);
});
$("upload-form").addEventListener("submit", submit);
$("refresh").addEventListener("click", () => {
  retryMs = POLL_MS;
  refresh();
});
$("sign-out").addEventListener("click", () => {
  forgetToken();
  showSignIn("Signed out.", "info");
});

const stored = loadToken();
if (stored) signIn(stored, null);
else showSignIn("");
