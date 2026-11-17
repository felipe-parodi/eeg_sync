// New-session page: step form, live checks, timeline preview, chunked upload.
// CHUNK_SIZE, CSRF_TOKEN, MIN_PART_S and VIDEO_END_TOLERANCE_S come from the
// inline config script in new_session.html.
//
// The rows are only an editor: before every check and on submit they are
// written into the hidden textareas in the format the server has always
// parsed ("name,start,end" and "start,end,reason", one per line). The server
// repeats every check below (portal/app.py), so these are for fast feedback.

const form = document.getElementById("upload-form");
const statusEl = document.getElementById("upload-status");
const progressEl = document.getElementById("upload-progress");
const CAMERAS = {
  camera_a: { label: "Video A", letter: "a" },
  camera_b: { label: "Video B", letter: "b" },
};
const TASK_COLORS = ["#0072b2", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#d55e00"];
const TIME_HINT = "Use seconds (90), M:SS (1:30) or H:MM:SS (1:01:30).";
const state = {
  videos: { camera_a: null, camera_b: null }, // {name, size, duration|null}
  session: null, // last /sessions/{id}/exists answer
  showAll: false, // after a submit attempt, show every error
};
let uploadInProgress = false;

// ---- Time helpers (mirror portal/jobs.py parse_timecode / format_clock) ----

function parseTime(raw) {
  const text = raw.trim();
  if (!text) return { error: "Time is empty." };
  const secondsRe = /^\d+(\.\d+)?$/;
  const clockRe = /^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$/;
  const rest = text.slice(1);
  if (text.startsWith("-") && (secondsRe.test(rest) || clockRe.test(rest))) {
    return { error: `"${text}" is negative. Times count from the start of the video.` };
  }
  if (secondsRe.test(text)) return { value: parseFloat(text) };
  const match = clockRe.exec(text);
  if (!match) return { error: `"${text}" is not a time. ${TIME_HINT}` };
  const seconds = parseFloat(match[3]);
  const minutes = parseInt(match[2], 10);
  if (seconds >= 60) {
    return { error: `"${text}" has ${match[3].replace(/^0+(?=\d)/, "")} seconds. Seconds must be under 60.` };
  }
  if (match[1] === undefined) return { value: minutes * 60 + seconds };
  if (minutes >= 60) return { error: `"${text}" has ${minutes} minutes. Minutes must be under 60.` };
  return { value: parseInt(match[1], 10) * 3600 + minutes * 60 + seconds };
}

function formatClock(totalSeconds) {
  const total = Math.round(totalSeconds * 10) / 10;
  const whole = Math.floor(total);
  const fraction = Math.round((total - whole) * 10) / 10;
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  const secText = fraction === 0 ? String(secs).padStart(2, "0") : (secs + fraction).toFixed(1).padStart(4, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${secText}` : `${minutes}:${secText}`;
}

function formatLength(seconds) {
  if (seconds < 60) return `${Math.round(seconds * 10) / 10} s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.round(seconds - minutes * 60);
  return secs ? `${minutes} min ${secs} s` : `${minutes} min`;
}

function slug(value) {
  return value.trim().replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
}

// Mirror of split_blocks_for_exclusions + split_blocks_with_min_part.
function splitTask(task, windows) {
  let fragments = [[task.start, task.end]];
  const sorted = [...windows].sort((x, y) => x.start - y.start);
  for (const win of sorted) {
    const next = [];
    for (const [start, end] of fragments) {
      const overlapStart = Math.max(start, win.start);
      const overlapEnd = Math.min(end, win.end);
      if (overlapStart >= overlapEnd) {
        next.push([start, end]);
        continue;
      }
      if (overlapStart - start >= 1e-9) next.push([start, overlapStart]);
      if (end - overlapEnd >= 1e-9) next.push([overlapEnd, end]);
    }
    fragments = next;
  }
  return fragments.map(([start, end], index) => ({
    name: fragments.length === 1 ? task.slug : `${task.slug}_part${index + 1}`,
    start,
    end,
    dropped: end - start < MIN_PART_S,
  }));
}

// ---- Rows -------------------------------------------------------------------

function field(labelText, className, attrs = "") {
  const id = `f${Math.random().toString(36).slice(2, 9)}`;
  return `<div class="row-field ${className}-field"><label for="${id}">${labelText}</label>` +
    `<input id="${id}" class="${className}" type="text" autocomplete="off" ${attrs}></div>`;
}

function addRow(list, values = {}) {
  const kind = list.dataset.kind;
  const row = document.createElement("div");
  row.className = "time-row";
  row.innerHTML = kind === "task"
    ? field("Task", "row-name", 'list="task-name-suggestions" placeholder="Name"') +
      field("Start", "row-start", 'placeholder="M:SS"') +
      field("End", "row-end", 'placeholder="M:SS"') +
      '<button type="button" class="button--quiet row-remove" aria-label="Remove this task">Remove</button>' +
      '<p class="field-error row-error" hidden></p><p class="field-note row-note" hidden></p>'
    : field("Start", "row-start", 'placeholder="M:SS"') +
      field("End", "row-end", 'placeholder="M:SS"') +
      field("Reason", "row-reason", 'placeholder="Optional"') +
      '<button type="button" class="button--quiet row-remove" aria-label="Remove this window">Remove</button>' +
      '<p class="field-error row-error" hidden></p>';
  for (const [key, value] of Object.entries(values)) {
    const input = row.querySelector(`.row-${key}`);
    if (input) input.value = value;
  }
  list.appendChild(row);
  return row;
}

function rowValues(row) {
  const get = (name) => (row.querySelector(`.row-${name}`) || { value: "" }).value;
  return { name: get("name"), start: get("start"), end: get("end"), reason: get("reason") };
}

function rowsFromText(list, text) {
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    const parts = line.split(",").map((part) => part.trim());
    if (list.dataset.kind === "task") {
      addRow(list, { name: parts[0] || "", start: parts[1] || "", end: parts[2] || "" });
    } else {
      addRow(list, { start: parts[0] || "", end: parts[1] || "", reason: parts.slice(2).join(", ") });
    }
  }
}

// ---- Checks -------------------------------------------------------------------

function videoHint(end) {
  const hours = Math.floor(end / 3600);
  const minutes = Math.floor((end % 3600) / 60);
  if (!hours || end % 60 !== 0) return "";
  return ` For ${hours} minutes ${minutes} seconds, type ${hours}:${String(minutes).padStart(2, "0")}.`;
}

// Check one camera: returns {tasks, windows, parts, errors} and marks rows.
function planCamera(cameraId) {
  const label = CAMERAS[cameraId].label;
  const video = state.videos[cameraId];
  const plan = { tasks: [], windows: [], parts: [], errors: [] };
  const windowRows = document.querySelectorAll(`#windows-${cameraId} .time-row`);
  windowRows.forEach((row, index) => {
    const v = rowValues(row);
    const errors = [];
    if (v.start.trim() || v.end.trim() || v.reason.trim()) {
      const start = parseTime(v.start);
      const end = parseTime(v.end);
      if (start.error) errors.push(`Start time: ${start.error}`);
      if (end.error) errors.push(`End time: ${end.error}`);
      if (!errors.length && end.value <= start.value) errors.push("The end must be after the start.");
      if (!errors.length) plan.windows.push({ row, start: start.value, end: end.value, text: v });
    }
    row.errors = errors.map((e) => `${label} extra-person window ${index + 1}: ${e}`);
    row.rowErrors = errors;
  });

  const seen = new Map();
  document.querySelectorAll(`#tasks-${cameraId} .time-row`).forEach((row) => {
    const v = rowValues(row);
    const errors = [];
    const notes = [];
    row.task = null;
    if (!(v.name.trim() || v.start.trim() || v.end.trim())) {
      row.rowErrors = [];
      row.rowNotes = [];
      return;
    }
    const name = v.name.trim();
    const safe = slug(name);
    if (!name) errors.push("Enter a task name.");
    else if (name.includes(",")) errors.push("Task names cannot contain commas.");
    else if (!safe) errors.push("Use at least one letter or number in the task name.");
    else if (seen.has(safe)) {
      const other = seen.get(safe);
      errors.push(other === name
        ? `Another ${label} task is also named "${name}". Give each task a different name.`
        : `"${other}" and "${name}" both become "${safe}" in file names. Give each task a different name.`);
    }
    if (safe) seen.set(safe, name);
    const start = parseTime(v.start);
    const end = parseTime(v.end);
    if (start.error) errors.push(`Start time: ${start.error}`);
    if (end.error) errors.push(`End time: ${end.error}`);
    if (!start.error && !end.error) {
      const length = end.value - start.value;
      if (length <= 0) errors.push("The end must be after the start.");
      else if (length < MIN_PART_S) errors.push(`This task is only ${Math.round(length * 10) / 10} s long. Tasks must be at least ${MIN_PART_S} s.`);
      if (video && video.duration && end.value > video.duration + VIDEO_END_TOLERANCE_S) {
        errors.push(`Ends at ${formatClock(end.value)}, but ${label} is only ${formatClock(video.duration)} long.${videoHint(end.value)}`);
      }
    }
    row.rowErrors = errors;
    row.rowNotes = notes;
    // Only rows without errors reach the split and the timeline.
    if (!errors.length) {
      row.task = { row, name, slug: safe || "task", start: start.value, end: end.value, text: v };
      plan.tasks.push(row.task);
    }
  });

  // Split around windows, then check names against the generated parts.
  const generated = new Map();
  plan.tasks.forEach((task, index) => {
    task.color = TASK_COLORS[index % TASK_COLORS.length];
    task.parts = splitTask(task, plan.windows);
    if (!task.parts.length) {
      task.row.rowErrors.push("This task is entirely inside an extra-person window, so none of it can be analysed.");
    } else if (task.parts.every((part) => part.dropped)) {
      task.row.rowErrors.push(`Every piece left after the extra-person windows is shorter than ${MIN_PART_S} s.`);
    }
    if (task.parts.length > 1) task.parts.forEach((part) => generated.set(part.name, task));
    task.parts.forEach((part) => plan.parts.push({ ...part, task }));
    plan.tasks.forEach((other) => {
      if (other !== task && other.start < task.end && task.start < other.end && index < plan.tasks.indexOf(other)) {
        other.row.rowNotes.push(`Overlaps task "${task.name}". Both will be analysed, so these seconds count twice.`);
      }
    });
  });
  plan.tasks.forEach((task) => {
    const owner = generated.get(task.slug);
    if (owner && owner !== task) {
      task.row.rowErrors.push(`This name clashes with a piece of task "${owner.name}" (${task.slug}). Rename it.`);
    }
  });

  document.querySelectorAll(`#tasks-${cameraId} .time-row`).forEach((row) => {
    const v = rowValues(row);
    row.errors = (row.rowErrors || []).map((e) => `${label} task "${v.name.trim() || "(no name)"}": ${e}`);
  });
  const allRows = document.querySelectorAll(`#tasks-${cameraId} .time-row, #windows-${cameraId} .time-row`);
  allRows.forEach((row) => plan.errors.push(...(row.errors || [])));
  if (video && !plan.tasks.length && !plan.errors.length) plan.errors.push(`Add at least one task for ${label}.`);
  return plan;
}

function showRowMessages(row) {
  const errorEl = row.querySelector(".row-error");
  const noteEl = row.querySelector(".row-note");
  const visible = state.showAll || row.dataset.touched === "yes";
  const errors = visible ? row.rowErrors || [] : [];
  errorEl.textContent = errors.join(" ");
  errorEl.hidden = !errors.length;
  row.classList.toggle("has-error", errors.length > 0);
  if (noteEl) {
    noteEl.textContent = (row.rowNotes || []).join(" ");
    noteEl.hidden = !(row.rowNotes || []).length;
  }
}

function sessionErrors() {
  const input = document.getElementById("session_id");
  const box = document.getElementById("session-exists");
  const text = document.getElementById("session-exists-text");
  const replace = document.getElementById("replace_existing");
  const errors = [];
  const info = state.session;
  box.hidden = !(info && info.valid && info.exists && !info.active_job);
  if (!box.hidden) {
    text.textContent = `Session "${info.session_id}" already has results. Starting again under the same ID would mix old and new results.`;
  }
  if (!input.value.trim()) errors.push("Enter a session ID.");
  else if (info && !info.valid) errors.push(info.message);
  else if (info && info.active_job) errors.push(`Session "${info.session_id}" is already being processed. Wait for that job to finish, or choose a new session ID.`);
  else if (info && info.exists && !replace.checked) errors.push(`Session "${info.session_id}" already has results. Tick "Replace the earlier results" or choose a new session ID.`);
  return errors;
}

function setFieldError(id, errors, visible) {
  const el = document.getElementById(id);
  el.textContent = visible ? errors.join(" ") : "";
  el.hidden = !(visible && errors.length);
}

// ---- Timeline preview -----------------------------------------------------------

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderTimeline(cameraId, plan) {
  const box = document.getElementById(`timeline-${cameraId}`);
  const label = CAMERAS[cameraId].label;
  const video = state.videos[cameraId];
  if (!plan.tasks.length) {
    box.innerHTML = `<p class="muted">The ${label} timeline appears here once a task has a start and an end.</p>`;
    return;
  }
  const lastTime = Math.max(...plan.tasks.map((t) => t.end), ...plan.windows.map((w) => w.end));
  const length = video && video.duration ? Math.max(video.duration, lastTime) : lastTime * 1.05;
  const pct = (t) => `${((t / length) * 100).toFixed(3)}%`;
  const width = (a, b) => `${(((b - a) / length) * 100).toFixed(3)}%`;
  let bars = "";
  for (const part of plan.parts) {
    const cls = part.dropped ? "tl-part tl-part--dropped" : "tl-part";
    const style = part.dropped ? "" : `background:${part.task.color};`;
    bars += `<div class="${cls}" style="left:${pct(part.start)};width:${width(part.start, part.end)};${style}" title="${escapeHtml(part.name)}"></div>`;
  }
  for (const win of plan.windows) {
    bars += `<div class="tl-window" style="left:${pct(win.start)};width:${width(win.start, Math.min(win.end, length))}" title="Extra-person window"></div>`;
  }
  if (video && video.duration && lastTime > video.duration) {
    bars += `<div class="tl-past" style="left:${pct(video.duration)};width:${width(video.duration, length)}" title="After the end of the video"></div>`;
  }
  const endLabel = video && video.duration ? `${formatClock(video.duration)} (video end)` : `${formatClock(length)} (video length not known yet)`;
  let items = "";
  for (const part of plan.parts) {
    const span = `${formatClock(part.start)} to ${formatClock(part.end)} (${formatLength(part.end - part.start)})`;
    items += part.dropped
      ? `<li class="tl-item tl-item--dropped"><span class="tl-swatch tl-swatch--dropped" aria-hidden="true"></span><span><strong>${escapeHtml(part.name)}</strong> ${span}. Skipped, shorter than ${MIN_PART_S} s.</span></li>`
      : `<li class="tl-item"><span class="tl-swatch" style="background:${part.task.color}" aria-hidden="true"></span><span><strong>${escapeHtml(part.name)}</strong> ${span}</span></li>`;
  }
  const split = plan.tasks.some((t) => t.parts.length > 1);
  box.innerHTML =
    `<div class="tl-track" role="img" aria-label="${label} timeline: ${plan.parts.length} pieces, ${plan.windows.length} extra-person windows">${bars}</div>` +
    `<div class="tl-axis"><span>0:00</span><span>${endLabel}</span></div>` +
    `<div class="tl-legend"><span><span class="tl-swatch tl-swatch--window" aria-hidden="true"></span>Extra-person window (removed)</span>` +
    `<span><span class="tl-swatch tl-swatch--dropped" aria-hidden="true"></span>Too short, skipped</span></div>` +
    `<ul class="tl-list">${items}</ul>` +
    (split ? '<p class="muted">A task cut by an extra-person window is measured in pieces (_part1, _part2, ...). The pieces are joined back together in the task summary.</p>' : "");
}

// ---- Review and the whole-form check ---------------------------------------------

function writeTextMirrors(plans) {
  for (const [cameraId, info] of Object.entries(CAMERAS)) {
    const hasVideo = Boolean(state.videos[cameraId]);
    const taskLines = [];
    const windowLines = [];
    if (hasVideo) {
      document.querySelectorAll(`#tasks-${cameraId} .time-row`).forEach((row) => {
        const v = rowValues(row);
        if (v.name.trim() || v.start.trim() || v.end.trim()) taskLines.push(`${v.name.trim()},${v.start.trim()},${v.end.trim()}`);
      });
      plans[cameraId].windows.forEach((w) => windowLines.push(`${w.text.start.trim()},${w.text.end.trim()},${w.text.reason.trim()}`));
    }
    document.getElementById(`blocks_text_${info.letter}`).value = taskLines.join("\n");
    document.getElementById(`exclusions_text_${info.letter}`).value = windowLines.join("\n");
  }
}

function renderReview(plans, sessionErrs) {
  const review = document.getElementById("review");
  const info = state.session;
  const id = document.getElementById("session_id").value.trim();
  let html = `<dt>Session</dt><dd>${id ? escapeHtml(info && info.session_id ? info.session_id : id) : "Not entered yet"}` +
    (info && info.exists && document.getElementById("replace_existing").checked ? " (earlier results will be archived)" : "") + "</dd>";
  for (const [cameraId, cam] of Object.entries(CAMERAS)) {
    const video = state.videos[cameraId];
    if (!video) {
      html += `<dt>${cam.label}</dt><dd>Not used</dd>`;
      continue;
    }
    const plan = plans[cameraId];
    const kept = plan.parts.filter((p) => !p.dropped);
    const dropped = plan.parts.length - kept.length;
    const total = kept.reduce((sum, p) => sum + p.end - p.start, 0);
    html += `<dt>${cam.label}</dt><dd>${escapeHtml(video.name)}` +
      (video.duration ? ` (${formatClock(video.duration)} long)` : " (length could not be read in the browser; the server checks it after upload)") +
      `. ${plan.tasks.length} task${plan.tasks.length === 1 ? "" : "s"}, ${kept.length} piece${kept.length === 1 ? "" : "s"} to analyse, ${formatLength(total)} in total` +
      (dropped ? `, ${dropped} short piece${dropped === 1 ? "" : "s"} skipped` : "") + ".</dd>";
  }
  html += `<dt>Gaze</dt><dd>${form.include_gaze.checked ? "Yes" : "No"}</dd>`;
  review.innerHTML = html;
}

function checkForm() {
  const plans = {};
  const errors = [];
  const sessionErrs = sessionErrors();
  const sessionTouched = state.showAll || document.getElementById("session_id").dataset.touched === "yes";
  setFieldError("session_id-error", sessionErrs, sessionTouched);
  errors.push(...sessionErrs);
  const videoErrs = state.videos.camera_a || state.videos.camera_b ? [] : ["Choose at least one video."];
  setFieldError("videos-error", videoErrs, state.showAll);
  errors.push(...videoErrs);
  for (const cameraId of Object.keys(CAMERAS)) {
    const plan = planCamera(cameraId);
    plans[cameraId] = plan;
    document.querySelectorAll(`#tasks-${cameraId} .time-row, #windows-${cameraId} .time-row`).forEach(showRowMessages);
    const hasVideo = Boolean(state.videos[cameraId]);
    document.querySelector(`[data-no-video="${cameraId}"]`).hidden = hasVideo;
    const needTask = `Add at least one task for ${CAMERAS[cameraId].label}.`;
    setFieldError(`tasks-${cameraId}-error`, plan.errors.includes(needTask) ? [needTask] : [], state.showAll);
    if (hasVideo) errors.push(...plan.errors);
    renderTimeline(cameraId, plan);
  }
  writeTextMirrors(plans);
  renderReview(plans, sessionErrs);
  const summary = document.getElementById("form-errors");
  document.getElementById("form-errors-list").innerHTML = errors.map((e) => `<li>${escapeHtml(e)}</li>`).join("");
  summary.hidden = !(state.showAll && errors.length);
  return errors;
}

// ---- Session ID lookup --------------------------------------------------------------

let sessionTimer = null;
async function lookUpSession() {
  const value = document.getElementById("session_id").value.trim();
  if (!value) {
    state.session = null;
    checkForm();
    return;
  }
  try {
    const response = await fetch(`/sessions/${encodeURIComponent(value)}/exists`, {
      headers: { accept: "application/json" },
      credentials: "same-origin",
    });
    if (document.getElementById("session_id").value.trim() !== value) return;
    state.session = response.ok ? await response.json() : null;
  } catch (error) {
    state.session = null; // The server checks again when the upload starts.
  }
  checkForm();
}

// ---- Videos ------------------------------------------------------------------------------

function readDuration(file) {
  return new Promise((resolve) => {
    const probe = document.createElement("video");
    const url = URL.createObjectURL(file);
    const done = (value) => {
      URL.revokeObjectURL(url);
      resolve(Number.isFinite(value) && value > 0 ? value : null);
    };
    probe.preload = "metadata";
    probe.muted = true;
    probe.addEventListener("loadedmetadata", () => done(probe.duration));
    probe.addEventListener("error", () => done(null));
    setTimeout(() => done(null), 15000);
    probe.src = url;
  });
}

async function setVideoFile(input) {
  const cameraId = input.id === "video_a" ? "camera_a" : "camera_b";
  const file = input.files[0];
  const nameEl = document.getElementById(`${input.id}_selected`);
  const durationEl = document.getElementById(`${input.id}_duration`);
  if (!file) {
    state.videos[cameraId] = null;
    nameEl.textContent = "No file selected";
    durationEl.textContent = "";
    checkForm();
    return;
  }
  nameEl.textContent = file.name;
  durationEl.textContent = "Reading the video length...";
  state.videos[cameraId] = { name: file.name, size: file.size, duration: null };
  checkForm();
  const duration = await readDuration(file);
  if (input.files[0] !== file) return;
  state.videos[cameraId].duration = duration;
  durationEl.textContent = duration
    ? `Length ${formatClock(duration)}`
    : "This browser could not read the length. The server checks it after upload.";
  checkForm();
}

document.querySelectorAll(".file-drop-zone").forEach((zone) => {
  const input = document.getElementById(zone.dataset.fileInput);
  input.addEventListener("change", () => setVideoFile(input));
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("dragover");
    if (!event.dataTransfer.files.length) return;
    const transfer = new DataTransfer();
    transfer.items.add(event.dataTransfer.files[0]);
    input.files = transfer.files;
    setVideoFile(input);
  });
});

// ---- Wiring ---------------------------------------------------------------------------------

document.querySelectorAll(".row-list").forEach((list) => {
  const letter = CAMERAS[list.dataset.camera].letter;
  const mirror = document.getElementById(list.dataset.kind === "task" ? `blocks_text_${letter}` : `exclusions_text_${letter}`);
  rowsFromText(list, mirror.value); // Restores rows after the browser's Back button.
  if (list.dataset.kind === "task" && !list.children.length) addRow(list);
  list.addEventListener("input", checkForm);
  list.addEventListener("focusout", (event) => {
    const row = event.target.closest(".time-row");
    if (row && event.target.value.trim()) row.dataset.touched = "yes";
    checkForm();
  });
  list.addEventListener("click", (event) => {
    if (!event.target.classList.contains("row-remove")) return;
    event.target.closest(".time-row").remove();
    checkForm();
  });
});

document.querySelectorAll("[data-add-row]").forEach((button) => {
  button.addEventListener("click", () => {
    const row = addRow(document.getElementById(button.dataset.addRow));
    row.querySelector("input").focus();
    checkForm();
  });
});

document.getElementById("copy-tasks-a-to-b").addEventListener("click", () => {
  const source = [...document.querySelectorAll("#tasks-camera_a .time-row")].map(rowValues).filter((v) => v.name.trim() || v.start.trim() || v.end.trim());
  const target = document.getElementById("tasks-camera_b");
  const status = document.getElementById("copy-status");
  if (!source.length) {
    status.textContent = "Video A has no tasks to copy yet.";
    return;
  }
  const hasB = [...target.querySelectorAll(".time-row")].some((row) => Object.values(rowValues(row)).some((v) => v.trim()));
  if (hasB && !window.confirm("Replace the Video B tasks with a copy of the Video A tasks?")) return;
  target.innerHTML = "";
  source.forEach((v) => addRow(target, { name: v.name, start: v.start, end: v.end }));
  status.textContent = `Copied ${source.length} task${source.length === 1 ? "" : "s"}. Now change each time to match Video B's clock.`;
  target.querySelector(".row-start").focus();
  checkForm();
});

const sessionInput = document.getElementById("session_id");
sessionInput.addEventListener("input", () => {
  clearTimeout(sessionTimer);
  sessionTimer = setTimeout(lookUpSession, 400);
  checkForm();
});
sessionInput.addEventListener("blur", () => {
  if (sessionInput.value.trim()) sessionInput.dataset.touched = "yes";
  checkForm();
});
document.getElementById("replace_existing").addEventListener("change", checkForm);
form.include_gaze.addEventListener("change", checkForm);

// ---- Upload (unchanged protocol: init, chunks, finish) --------------------------------------

function setStatus(message) {
  statusEl.textContent = message;
}

function formatBytes(bytes) {
  const gib = bytes / (1024 * 1024 * 1024);
  return gib >= 1 ? `${gib.toFixed(2)} GB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function errorText(text, status) {
  try {
    const detail = JSON.parse(text).detail;
    if (typeof detail === "string") return detail;
  } catch (error) {
    // Not JSON; fall through.
  }
  return text || `Request failed (${status}).`;
}

async function postFormData(url, formData) {
  const response = await fetch(url, {
    method: "POST",
    body: formData,
    headers: { "X-CSRF-Token": CSRF_TOKEN },
    credentials: "same-origin",
  });
  if (!response.ok) throw new Error(errorText(await response.text(), response.status));
  return await response.json();
}

function postChunkFormData(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", url);
    request.withCredentials = true;
    request.setRequestHeader("X-CSRF-Token", CSRF_TOKEN);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded, event.total);
    };
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        resolve(JSON.parse(request.responseText));
        return;
      }
      reject(new Error(errorText(request.responseText, request.status)));
    };
    request.onerror = () => reject(new Error("The connection dropped during the upload."));
    request.send(formData);
  });
}

function updateUploadProgress(cameraId, cameraLabel, chunkNumber, totalChunks, visibleCameraBytes, uploadState) {
  uploadState[cameraId] = visibleCameraBytes;
  const visibleBytes = uploadState.camera_a + uploadState.camera_b;
  progressEl.value = Math.round((visibleBytes / uploadState.totalBytes) * 100);
  setStatus(
    `${cameraLabel}: uploading part ${chunkNumber} of ${totalChunks} ` +
    `(${formatBytes(visibleBytes)} of ${formatBytes(uploadState.totalBytes)} total). Keep this tab open.`
  );
}

async function uploadCamera(jobId, cameraId, file, uploadState) {
  const totalChunks = Math.max(1, Math.ceil(file.size / CHUNK_SIZE));
  const cameraLabel = CAMERAS[cameraId].label;
  let uploadedCameraBytes = 0;
  for (let index = 0; index < totalChunks; index += 1) {
    const start = index * CHUNK_SIZE;
    const chunk = file.slice(start, Math.min(file.size, start + CHUNK_SIZE));
    const formData = new FormData();
    formData.append("camera_id", cameraId);
    formData.append("chunk_index", String(index));
    formData.append("total_chunks", String(totalChunks));
    formData.append("chunk", chunk, file.name);
    await postChunkFormData(`/uploads/${jobId}/chunk`, formData, (loaded) => {
      updateUploadProgress(cameraId, cameraLabel, index + 1, totalChunks, uploadedCameraBytes + loaded, uploadState);
    });
    uploadedCameraBytes += chunk.size;
    updateUploadProgress(cameraId, cameraLabel, index + 1, totalChunks, uploadedCameraBytes, uploadState);
  }
}

window.addEventListener("beforeunload", (event) => {
  if (!uploadInProgress) return;
  event.preventDefault();
  event.returnValue = "";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  state.showAll = true;
  const errors = checkForm();
  if (errors.length) {
    document.getElementById("form-errors").focus();
    return;
  }
  const submitButton = document.getElementById("submit-button");
  submitButton.disabled = true;
  submitButton.textContent = "Uploading...";
  uploadInProgress = true;
  document.title = "Uploading videos";
  progressEl.style.display = "block";
  progressEl.value = 0;
  try {
    const videoA = document.getElementById("video_a").files[0];
    const videoB = document.getElementById("video_b").files[0];
    const initData = new FormData(form);
    initData.append("video_a_name", videoA ? videoA.name : "");
    initData.append("video_b_name", videoB ? videoB.name : "");
    initData.append("video_a_size", videoA ? String(videoA.size) : "0");
    initData.append("video_b_size", videoB ? String(videoB.size) : "0");
    initData.delete("video_a");
    initData.delete("video_b");

    setStatus("Checking the form on the server.");
    const init = await postFormData("/uploads/init", initData);
    const uploadState = { camera_a: 0, camera_b: 0, totalBytes: (videoA ? videoA.size : 0) + (videoB ? videoB.size : 0) };
    const uploads = [];
    if (videoA) uploads.push(uploadCamera(init.job_id, "camera_a", videoA, uploadState));
    if (videoB) uploads.push(uploadCamera(init.job_id, "camera_b", videoB, uploadState));
    await Promise.all(uploads);

    setStatus("Upload complete. Checking the videos and starting the job.");
    document.title = "Processing queued";
    const finish = await postFormData(`/uploads/${init.job_id}/finish`, new FormData());
    uploadInProgress = false;
    window.location.href = finish.status_url;
  } catch (error) {
    setStatus(`Could not start: ${error.message}`);
    submitButton.disabled = false;
    submitButton.textContent = "Start processing";
    uploadInProgress = false;
    document.title = "New session";
  }
});

// Hooks for previews and manual testing: set a video without a real file, or
// a session-ID answer, then re-run the checks.
window.portalForm = {
  setVideo(cameraId, name, duration) {
    state.videos[cameraId] = { name, size: 1, duration };
    const letter = CAMERAS[cameraId].letter;
    document.getElementById(`video_${letter}_selected`).textContent = name;
    document.getElementById(`video_${letter}_duration`).textContent = duration ? `Length ${formatClock(duration)}` : "";
    checkForm();
  },
  setSession(info) {
    state.session = info;
    checkForm();
  },
  showAllErrors() {
    state.showAll = true;
    return checkForm();
  },
  parseTime,
  splitTask,
};

checkForm();
