"use strict";
// The web UI on top of the submission API (app.py): plain DOM, no dependencies, nothing loaded from elsewhere.
// Everything the server sends is inserted as text (textContent), never parsed as HTML.

const TOKEN_KEY = "viobench-token";
const REQUEST_TIMEOUT_MS = 30000;
const UPLOAD_STALL_MS = 120000;  // an upload that makes no progress for this long is given up
const POLL_MS = 15000;  // submissions and pause status, including while idle
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
let session = 0;  // discard details requested before signing out
let uploading = false;
let deletingSubmission = false;
let adminSaving = false;
let evaluationPresets = null;
let detailGeneration = 0;
let detailPending = false;
let shown = null;  // {id} of the submission in the details section; results change as jobs finish

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

async function api(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(path, {...options, headers: {Authorization: `Bearer ${token}`, ...options.headers},
                                        cache: "no-store", signal: controller.signal});
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

function episodeHash(hash) {
  // an episode is the sha1 of its archive: its first 12 digits here, all of them on hover
  if (typeof hash !== "string") return DASH;
  const element = node("code", hash.slice(0, 12));
  element.title = hash;
  return element;
}

function progressText(progress) {
  if (!progress) return DASH;
  const counts = [["completed", "completed"], ["failed", "failed"], ["queued", "queued"], ["running", "running"],
                  ["missing", "not scheduled"]];
  return `${progress.finished}/${progress.total} finished` + counts.filter(([key]) => progress[key]).map(([key, label]) => `; ${progress[key]} ${label}`).join("");
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
  session++;
  waiting = false;
  retryMs = POLL_MS;
  token = null;
  config = null;
  adminSaving = false;
  deletingSubmission = false;
  evaluationPresets = null;
  detailGeneration++;
  detailPending = false;
  $("pause-message").value = "";
  $("evaluation-backfill").checked = false;
  $("schedule-hashes").value = "";
  shown = null;
  $("submissions").tBodies[0].replaceChildren();
  for (const id of ["details-title", "details-facts", "details-results", "details-jobs", "details-progress", "details-json", "updated", "rules",
                    "team-name", "evaluation-preset", "evaluation-status", "evaluation-summary", "comparison-charts"]) $(id).replaceChildren();
  for (const id of ["team", "account", "details", "connection", "pause-controls", "pause-banner", "evaluation-controls", "schedule-form"]) $(id).hidden = true;
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
  $("account").hidden = false;
  $("pause-message").value = config.pause_message || "";
  renderConfiguration(config);
  notice("");
  refresh();
}

function renderConfiguration(next) {
  if (config && config.team_type !== next.team_type) {
    session++;  // discard details requested with the previous permissions
    shown = null;
    evaluationPresets = null;
    detailGeneration++;
    detailPending = false;
    $("submissions").tBodies[0].replaceChildren();
    $("details").hidden = true;
    $("comparison-charts").replaceChildren();
  } else if (config && JSON.stringify(config.evaluation_hashes) !== JSON.stringify(next.evaluation_hashes)) {
    detailGeneration++;
    detailPending = false;
    $("details").hidden = true;
    $("comparison-charts").replaceChildren();
  }
  config = next;
  const admin = config.team_type === "admin";
  $("team-name").textContent = `Signed in as ${config.team_name}`;
  $("upload-section").hidden = admin;
  $("pause-controls").hidden = !admin;
  $("evaluation-controls").hidden = !admin;
  $("schedule-form").hidden = !admin;
  $("delete-submission").hidden = config.team_type !== "baseline";
  $("team-column").hidden = !admin;
  $("account-column").hidden = !admin;
  $("submissions-title").textContent = admin ? "All submissions" : "Submissions";
  $("evaluation-summary").textContent = `Results and progress use the current set of ${config.evaluation_hashes.length} episode(s).`;
  renderPause();
  renderRules(config);
}

function renderPause() {
  const message = config.pause_message;
  $("pause-banner").textContent = message ? `Submissions are paused: ${message}` : "";
  $("pause-banner").hidden = !message;
  $("pause-status").textContent = message ? "New submissions are paused." : "Accepting new submissions.";
  $("pause-button").textContent = message ? "Update message" : "Pause submissions";
  renderAdminButtons();
  if (message) $("upload-button").disabled = true;
}

async function setPause(message) {
  if (adminSaving || config.team_type !== "admin") return;
  if (message !== null && !message.trim()) return notice("Enter a message for the teams before pausing submissions.");
  const mine = session;
  adminSaving = true;
  clearTimeout(pollTimer);
  generation++;  // discard config/list responses from before this update
  renderPause();
  try {
    const state = await api("submission-pause", {method: "PUT", headers: {"Content-Type": "application/json"},
                                                 body: JSON.stringify({message: message === null ? null : message.trim()})});
    if (mine !== session) return;
    config.pause_message = state.pause_message;
    notice(state.pause_message ? "Submission pause saved." : "Submissions resumed.", "info");
  } catch (error) {
    if (mine !== session) return;
    if (error.status === 401) return showSignIn("The token is no longer valid.");
    notice(`Cannot update submission access: ${error.message}`);
  } finally {
    if (mine === session) {
      adminSaving = false;
      renderPause();
      refresh();
    }
  }
}

// --- selecting and scheduling evaluations ----------------------------------------------------------------------

function renderAdminButtons() {
  $("pause-button").disabled = adminSaving;
  $("resume-button").disabled = adminSaving || !config.pause_message;
  $("evaluation-save").disabled = adminSaving || !evaluationPresets || !$("evaluation-preset").value;
  $("evaluation-fill").disabled = adminSaving || !evaluationPresets;
  $("schedule-button").disabled = adminSaving;
}

function renderEvaluations(state) {
  const current = state.presets.find((preset) => JSON.stringify(preset.hashes) === JSON.stringify(state.hashes));
  if (evaluationPresets === null) {
    // This is a local edit form: keep both the options and their hashes until the next sign-in.
    evaluationPresets = state.presets;
    const select = $("evaluation-preset");
    select.replaceChildren();
    if (!current) {
      const custom = node("option", "Custom current set");
      custom.value = "";
      custom.disabled = true;
      select.append(custom);
    }
    for (const preset of evaluationPresets) {
      const option = node("option", `${preset.name} (${preset.hashes.length} episodes)`);
      option.value = preset.name;
      select.append(option);
    }
    select.value = current?.name || "";
  }
  $("evaluation-status").textContent = `Current set: ${current?.name || "custom"}, ${state.hashes.length} episode(s).`;
  renderAdminButtons();
}

async function updateEvaluations(path, method, body, savedSet = false) {
  if (adminSaving || config.team_type !== "admin") return;
  const mine = session;
  adminSaving = true;
  clearTimeout(pollTimer);
  generation++;
  detailGeneration++;  // discard details computed before new evaluations were scheduled
  detailPending = false;
  renderAdminButtons();
  try {
    const result = await api(path, {method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    if (mine !== session) return;
    notice(`${savedSet ? "Evaluation set saved. " : ""}${result.scheduled} evaluation(s) scheduled.`, "info");
  } catch (error) {
    if (mine !== session) return;
    if (error.status === 401) return showSignIn("The token is no longer valid.");
    notice(`Cannot update evaluations: ${error.message}`);
  } finally {
    if (mine === session) {
      adminSaving = false;
      renderAdminButtons();
      refresh();
    }
  }
}

function saveEvaluations(event) {
  event.preventDefault();
  const preset = evaluationPresets?.find((entry) => entry.name === $("evaluation-preset").value);
  if (preset) return updateEvaluations("evaluations", "PUT", {hashes: preset.hashes, backfill: $("evaluation-backfill").checked}, true);
}

function scheduleSubmission(event) {
  event.preventDefault();
  if (!shown) return;
  const body = {submission_id: shown.id};
  const hashes = $("schedule-hashes").value.trim();
  if (hashes) body.hashes = hashes.split(/[\s,]+/).filter(Boolean);
  return updateEvaluations("evaluations/schedule", "POST", body);
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
  if (token === null || adminSaving || deletingSubmission) return;
  clearTimeout(pollTimer);
  const mine = ++generation;
  let rows, evaluations, comparison;
  try {
    const next = await api("config");
    if (mine !== generation) return;
    renderConfiguration(next);
    [rows, evaluations, comparison] = await Promise.all([api("submissions"), config.team_type === "admin" ? api("evaluations") : null, api("comparison")]);
    if (!Array.isArray(rows)) throw new ApiError(200, "the server sent an unreadable answer");
  } catch (error) {
    if (mine !== generation) return;
    if (error.status === 401) return showSignIn("The token is no longer valid.");
    const connection = $("connection");
    connection.textContent = `Cannot refresh: ${error.message}. Trying again in ${Math.round(retryMs / 1000)} s.`;
    connection.hidden = false;
    schedule(retryMs);
    retryMs = Math.min(retryMs * 2, MAX_RETRY_MS);
    return;
  }
  if (mine !== generation) return;
  retryMs = POLL_MS;
  $("connection").hidden = true;
  if (evaluations) renderEvaluations(evaluations);
  renderList(rows);
  renderComparison(comparison);
  $("updated").textContent = `updated ${new Date().toLocaleTimeString()}`;
  if (config.team_type !== "admin") renderUploadState();
  schedule(POLL_MS);
  if (shown && !detailPending) showDetails(shown.id, false);
}

function renderList(rows) {
  const body = $("submissions").tBodies[0];
  body.replaceChildren();
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const overall = row.overall || {};
    const statusText = row.status === "failed" && row.error ? `failed: ${row.error}` : row.status;
    const status = node("span", statusText + (row.partial && row.overall ? " (partial results)" : ""), `status ${row.status}`);
    const button = node("button", "Details");
    button.type = "button";
    button.setAttribute("aria-label", `Details of ${row.label || row.id}`);
    button.addEventListener("click", () => showDetails(row.id, true));
    const episodes = row.progress ? `${row.progress.finished}/${row.progress.total} finished` : DASH;
    const progress = node("span", episodes);
    progress.title = progressText(row.progress);
    body.append(tableRow([time(row.created), ...(config.team_type === "admin" ? [row.team_name || DASH, row.team_type] : []),
                          row.label || DASH, status, decimal(overall.sim3_ate_rmse_median),
                          fuel(overall.fuel_per_frame_max), progress, button]));
  }
  $("empty").hidden = body.rows.length > 0;
}

// --- current-set comparison with baselines ---------------------------------------------------------------------

function svgNode(tag, attributes, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes || {})) element.setAttribute(name, String(value));
  if (text !== undefined) element.textContent = text;
  return element;
}

function logTicks(min, max) {
  const low = Math.log10(min), high = Math.log10(max);
  const stride = Math.max(1, Math.ceil((high - low) / 6));
  const ticks = [min];
  for (let exponent = Math.ceil(low / stride) * stride; exponent < high; exponent += stride) {
    const position = (exponent - low) / (high - low);
    if (position > 0.08 && position < 0.92) ticks.push(10 ** exponent); // leave room for endpoint labels
  }
  return [...ticks, max];
}

function renderComparison(comparison) {
  const container = $("comparison-charts");
  container.replaceChildren();
  if (JSON.stringify(comparison.evaluation_hashes) !== JSON.stringify(config.evaluation_hashes)) {
    container.append(node("p", "The evaluation set changed. Comparison will update on the next refresh.", "hint"));
    return;
  }
  if (!comparison.datasets.length) {
    container.append(node("p", "No comparable results for the current set yet.", "hint"));
    return;
  }
  for (const dataset of comparison.datasets) {
    container.append(node("h3", dataset.name));
    const points = dataset.points.filter((point) => isNumber(point.fuel_per_frame_max) && point.fuel_per_frame_max >= 0 &&
                                                    isNumber(point.sim3_ate_rmse_median) && point.sim3_ate_rmse_median >= 0);
    if (!points.length) {
      container.append(node("p", "No comparable results for this dataset yet.", "hint"));
      continue;
    }
    const fuelFloor = comparison.fuel_floor, errorFloor = comparison.error_floor, maxX = comparison.fuel_reference;
    const visiblePoints = points.filter((point) => point.fuel_per_frame_max <= maxX);
    // Zoom the display to one decade below the data, without changing the scoring reference.
    const minX = visiblePoints.length ? Math.max(fuelFloor, Math.min(...visiblePoints.map((point) => point.fuel_per_frame_max)) / 10) : fuelFloor;
    const minY = visiblePoints.length ? Math.max(errorFloor, Math.min(...visiblePoints.map((point) => point.sim3_ate_rmse_median)) / 10) : errorFloor;
    const maxY = Math.max(errorFloor * 1.2, dataset.error_reference || 0, ...visiblePoints.map((point) => point.sim3_ate_rmse_median * 1.1));
    const x = (used) => 76 + Math.log(Math.max(fuelFloor, used) / minX) / Math.log(maxX / minX) * 580;
    const y = (error) => 280 - Math.log(Math.max(errorFloor, error) / minY) / Math.log(maxY / minY) * 256;
    const frontierIds = new Set(dataset.frontier);
    const baselineIds = new Set(dataset.baseline_frontier);
    // The envelope begins at its ceiling and drops only when a submission improves it.
    // Include the final horizontal segment through the fixed fuel reference.
    const steps = (ids, ceiling) => {
      const coordinates = [[minX, ceiling]];
      const ordered = visiblePoints.filter((point) => ids.has(point.id))
        .map((point) => [Math.max(fuelFloor, point.fuel_per_frame_max), Math.max(errorFloor, point.sim3_ate_rmse_median)])
        .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      for (const [used, error] of ordered) {
        if (error >= ceiling) continue;
        coordinates.push([used, ceiling], [used, error]);
        ceiling = error;
      }
      coordinates.push([maxX, ceiling]);
      return coordinates;
    };
    const svgPoints = (coordinates) => coordinates.map(([used, error]) => `${x(used)},${y(error)}`).join(" ");
    const chart = svgNode("svg", {viewBox: "0 0 680 340", class: "comparison-chart", role: "group",
                                  "aria-label": `${dataset.name}: Sim(3) ATE versus maximum fuel per frame, both on logarithmic scales. Lower is better.`});
    chart.append(svgNode("title", {}, `${dataset.name}: your submissions and baselines`));
    for (const tick of logTicks(minX, maxX)) {
      const px = x(tick);
      chart.append(svgNode("line", {x1: px, y1: 24, x2: px, y2: 280, class: "comparison-grid"}),
                   svgNode("text", {x: px, y: 300, "text-anchor": "middle"}, fuel(tick)));
    }
    for (const tick of logTicks(minY, maxY)) {
      const py = y(tick);
      chart.append(svgNode("line", {x1: 76, y1: py, x2: 656, y2: py, class: "comparison-grid"}),
                   svgNode("text", {x: 65, y: py + 4, "text-anchor": "end"}, decimal(tick)));
    }
    chart.append(svgNode("text", {x: 366, y: 330, "text-anchor": "middle"}, "Maximum fuel/frame (log scale)"),
                 svgNode("text", {x: 18, y: 152, transform: "rotate(-90 18 152)", "text-anchor": "middle"}, "Median Sim(3) ATE (m, log scale)"));
    if (dataset.claimed_area > 0) {
      const boundary = steps(baselineIds, dataset.error_reference)
        .concat(steps(frontierIds, dataset.error_reference).reverse());
      chart.append(svgNode("polygon", {points: svgPoints(boundary), class: "comparison-area"}));
    }
    for (const [ids, className] of [[frontierIds, "comparison-frontier"], [baselineIds, "comparison-frontier baseline"]]) {
      if (visiblePoints.some((point) => ids.has(point.id))) {
        chart.append(svgNode("polyline", {points: svgPoints(steps(ids, maxY)), class: className}));
      }
    }
    const rows = [];
    for (const point of points) {
      const name = point.label || point.id;
      const onFrontier = frontierIds.has(point.id);
      const description = `${point.team_name}: ${name}; maximum fuel/frame ${point.fuel_per_frame_max}; median Sim(3) ATE ${point.sim3_ate_rmse_median} m${onFrontier ? "; on Pareto frontier" : ""}`;
      const canOpen = point.own || config.team_type === "admin";
      const mark = svgNode("g", {class: `comparison-point ${point.own ? "own" : "baseline"}${onFrontier ? " frontier" : ""}`,
                                 role: canOpen ? "button" : "img", tabindex: 0, "aria-label": description});
      mark.append(svgNode("title", {}, description), point.team_type === "baseline" ?
                  svgNode("rect", {x: x(point.fuel_per_frame_max) - 5, y: y(point.sim3_ate_rmse_median) - 5, width: 10, height: 10}) :
                  svgNode("circle", {cx: x(point.fuel_per_frame_max), cy: y(point.sim3_ate_rmse_median), r: 5}));
      let button = "";
      if (canOpen) {
        mark.addEventListener("click", () => showDetails(point.id, true));
        mark.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            showDetails(point.id, true);
          }
        });
        button = node("button", "Details");
        button.type = "button";
        button.setAttribute("aria-label", `Details of ${point.team_name}: ${name}`);
        button.addEventListener("click", () => showDetails(point.id, true));
      }
      if (point.fuel_per_frame_max <= maxX) chart.append(mark);
      rows.push([point.team_name, name, point.own ? `Yours (${point.team_type})` : "Baseline",
                 point.fuel_per_frame_max, point.sim3_ate_rmse_median, button]);
    }
    container.append(chart);
    if (dataset.claimed_area === null) {
      container.append(node("p", "Claimed area is unavailable until baseline results are available for this dataset.", "hint"));
    } else {
      const fraction = dataset.claimed_area / (Math.log(maxX / fuelFloor) * Math.log(dataset.error_reference / errorFloor));
      container.append(node("p", `Claimed area: ${decimal(fraction * 100)}% of the log–log reference rectangle. ` +
        `Scoring reference: ${fuel(fuelFloor)}–${fuel(maxX)} fuel/frame and ${decimal(errorFloor)}–${decimal(dataset.error_reference)} m Sim(3) ATE.`, "hint"));
    }
    if (points.some((point) => point.fuel_per_frame_max > maxX)) {
      container.append(node("p", "Submissions beyond the fuel axis are listed in Exact values below.", "hint"));
    }
    const values = node("details", null, "comparison-values");
    values.append(node("summary", "Exact values"), table(["Team", "Submission", "Account", "Max fuel/frame", "Sim(3) ATE (m)", ""], rows));
    container.append(values);
  }
}

// --- submitting ------------------------------------------------------------------------------------------------

function renderUploadState() {
  $("upload-button").disabled = uploading || Boolean(config.pause_message);
  const hint = $("upload-hint");
  if (isNumber(get(config, "daily_quota"))) {
    hint.textContent = `${config.daily_quota} accepted upload(s) per day (from midnight, ${config.quota_time_zone}; unlimited accounts exempt); ` +
                       `up to ${config.max_upload_mib} MiB.`;
  } else {
    hint.textContent = "";
  }
  hint.hidden = !hint.textContent;
}

async function submit(event) {
  event.preventDefault();
  if (config.pause_message) return notice(config.pause_message);
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
  ["Sim(3) ATE (m)", (s) => decimal(s.sim3_ate_rmse_median)],
  ["RPE (rad)", (s) => decimal(s.rpe_rotation_rmse_median)],
  ["ATE (m)", (s) => decimal(s.ate_rmse_median)],
  ["Velocity (m/s)", (s) => decimal(s.velocity_rmse_median)],
  ["Max fuel/frame", (s) => fuel(s.fuel_per_frame_max)],
  ["Fuel/frame p95", (s) => fuel(s.fuel_per_frame_p95)],
  ["Wall/frame (ms)", (s) => decimal(s.wall_per_frame_median_ms)],
];
const EPISODE_COLUMNS = [
  ["Dataset", (e) => e.dataset],
  ["Episode", (e) => episodeHash(e.hash)],
  ["Frames", (e) => e.frames],
  ["Completed", (e) => (e.completed === true ? "yes" : e.completed === false ? "no" : DASH)],
  ["Filled frames", (e) => get(e, "metrics", "filled_frames") ?? DASH],
  ["Sim(3) ATE (m)", (e) => decimal(get(e, "metrics", "sim3_ate_rmse"))],
  ["RPE (rad)", (e) => decimal(get(e, "metrics", "rpe_rotation_rmse"))],
  ["ATE (m)", (e) => decimal(get(e, "metrics", "ate_rmse"))],
  ["Drift/m", (e) => decimal(get(e, "metrics", "drift_per_metre"))],
  ["Max fuel/frame", (e) => fuel(e.fuel_per_frame_max)],
  ["Fuel/frame p95", (e) => fuel(e.fuel_per_frame_p95)],
];

async function showDetails(id, scroll) {
  if (deletingSubmission) return;
  const mine = session;
  const request = ++detailGeneration;
  detailPending = true;
  try {
    const submission = await api(`submissions/${encodeURIComponent(id)}`);
    if (mine !== session || request !== detailGeneration || token === null) return;
    if (shown?.id !== id) $("schedule-hashes").value = "";
    shown = {id};
    renderDetails(submission);
    if (scroll) $("details").scrollIntoView({block: "start"});
  } catch (error) {
    if (mine !== session || request !== detailGeneration) return;
    if (error.status === 401) return showSignIn("The token is no longer valid.");
    if (error.status === 404 && shown?.id === id) {
      shown = null;
      $("details").hidden = true;
    }
    notice(`Cannot show the submission: ${error.message}`);
  } finally {
    if (request === detailGeneration) detailPending = false;
  }
}

async function deleteSubmission(submission) {
  if (deletingSubmission || config?.team_type !== "baseline") return;
  if (!window.confirm(`Permanently delete “${submission.label || submission.id}”? This removes the upload, all evaluation jobs and results, and its points from everyone's comparisons.`)) return;
  const mine = session;
  deletingSubmission = true;
  clearTimeout(pollTimer);
  generation++; // discard list/comparison responses from before deletion
  detailGeneration++;
  detailPending = false;
  $("delete-submission").disabled = true;
  try {
    await api(`submissions/${encodeURIComponent(submission.id)}`, {method: "DELETE"});
    if (mine !== session) return;
    shown = null;
    $("details").hidden = true;
    notice("Submission deleted.", "info");
  } catch (error) {
    if (mine !== session) return;
    if (error.status === 401) return showSignIn("The token is no longer valid.");
    if (error.status === 0 || error.status === 404 || error.status >= 500) {
      shown = null; // deletion may have committed even if cleanup or the response failed
      $("details").hidden = true;
    }
    notice(`Delete request failed: ${error.message}`);
  } finally {
    if (mine === session) {
      deletingSubmission = false;
      $("delete-submission").disabled = false;
      refresh();
    }
  }
}

function renderDetails(submission) {
  $("details-title").textContent = submission.label || submission.id || "Submission";
  const baseline = config.team_type === "baseline";
  const running = submission.jobs.some((job) => job.status === "running");
  $("delete-submission").hidden = !baseline;
  $("delete-submission").disabled = deletingSubmission || running;
  $("delete-submission").onclick = () => deleteSubmission(submission);
  $("delete-hint").hidden = !baseline || !running;
  const facts = $("details-facts");
  facts.replaceChildren();
  const account = config.team_type === "admin" ? [["Team", submission.team_name], ["Account", submission.team_type]] : [];
  for (const [label, value] of [["Id", submission.id], ...account,
                                ["File", `${submission.filename || DASH} (${size(submission.bytes)})`],
                                ["SHA-256", submission.sha256], ["Status", submission.status], ["Error", submission.error],
                                ["Submitted", time(submission.created)]]) {
    if (value !== null && value !== undefined) facts.append(node("dt", label), node("dd", value));
  }
  const results = $("details-results");
  results.replaceChildren();
  $("details-progress").textContent = `Current set: ${progressText(submission.progress)}.`;
  $("schedule-form").hidden = config.team_type !== "admin";
  if (get(submission, "results", "partial")) {
    results.append(node("p", "Partial results for the current set. Summary numbers will update as more episodes finish.", "hint"));
  }
  const summary = get(submission, "results", "summary");
  if (summary && typeof summary === "object") {
    const groups = [["overall", summary.overall], ...Object.entries(summary.datasets || {})];
    results.append(node("h3", "Current-set summary"),
                   table(["", ...SUMMARY_COLUMNS.map(([title]) => title)],
                         groups.map(([name, values]) => [name, ...SUMMARY_COLUMNS.map(([, format]) => format(values || {}))])));
  }
  const episodes = get(submission, "results", "episodes");
  if (Array.isArray(episodes) && episodes.length) {
    results.append(node("h3", "Current-set results"), table(EPISODE_COLUMNS.map(([title]) => title),
                                                 episodes.map((episode) => EPISODE_COLUMNS.map(([, format]) => format(episode || {})))));
  }
  const jobs = $("details-jobs");
  jobs.replaceChildren();
  jobs.append(table(["Episode", "Status", "Current set", "Sim(3) ATE (m)", "Max fuel/frame", "Error"],
                    submission.jobs.map((job) => [episodeHash(job.episode_hash), node("span", job.status, `status ${job.status}`),
                      job.selected ? "yes" : "no",
                      decimal(get(job, "result", "metrics", "sim3_ate_rmse")), fuel(get(job, "result", "fuel_per_frame_max")), job.error || DASH])));
  $("details-json").textContent = JSON.stringify(submission, null, 2);
  $("details").hidden = false;
}

// --- the rules (GET /config) -----------------------------------------------------------------------------------

const RULES = {
  daily_quota: ["Submissions per day", String],
  quota_time_zone: ["A day starts at midnight in", String],
  max_upload_mib: ["Largest upload", (v) => `${v} MiB`],
  episode_timeout_seconds: ["Time per episode", (v) => `${decimal(v / 60)} min`],
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
  const flat = Object.entries(rules || {}).filter(([key]) => !["team_type", "pause_message", "team_name", "evaluation_hashes"].includes(key))
    .flatMap(([key, value]) => (value && typeof value === "object" ? Object.entries(value) : [[key, value]]));
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
$("pause-form").addEventListener("submit", (event) => {
  event.preventDefault();
  setPause($("pause-message").value);
});
$("resume-button").addEventListener("click", () => setPause(null));
$("evaluation-form").addEventListener("submit", saveEvaluations);
$("evaluation-fill").addEventListener("click", () => updateEvaluations("evaluations/schedule", "POST", {}));
$("evaluation-preset").addEventListener("change", renderAdminButtons);
$("schedule-form").addEventListener("submit", scheduleSubmission);
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
