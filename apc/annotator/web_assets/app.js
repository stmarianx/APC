const $ = (id) => document.getElementById(id);
const state = { project: null, selected: null, suggestion: null, drawnBox: null, dragStart: null };
const BOX_COLORS = { table: "#72e6ff", seat: "#b8f171", hero_card: "#f3c86a", board_card: "#f3c86a", pot: "#ff9f68", action_button: "#d79cff", turn_clock: "#ff7db8" };
const DEFAULT_FIELDS = {
  table: {},
  seat: { seat_no: 1, occupied: true, is_hero: false, has_dealer_button: false, player_name: "Player", stack_bb: "100", raw_stack_text: "100 BB", status: "active", visibility: "clear" },
  hero_card: { rank: "A", suit: "s", visibility: "clear" },
  board_card: { rank: "A", suit: "s", visibility: "clear" },
  pot: { amount_bb: "0", raw_text: "0 BB", visibility: "clear" },
  action_button: { action: "check", enabled: true, raw_text: "Check", visibility: "clear" },
  turn_clock: { remaining_ms: 30000, raw_text: "30", visibility: "clear" },
};

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function message(text, kind = "") {
  $("message").textContent = text;
  $("message").className = `message ${kind}`;
}

function readAnnotation() {
  return JSON.parse($("annotationEditor").value);
}

function writeAnnotation(annotation) {
  $("annotationEditor").value = JSON.stringify(annotation, null, 2);
  renderOverlay();
}

function boxLabelRows(annotation) {
  const objects = annotation.objects || {};
  const rows = [];
  if (objects.table) rows.push({ type: "table", label: "table", box: objects.table });
  (objects.seats || []).forEach((item, index) => rows.push({ type: "seat", label: `seat ${item.seat_no || index + 1}`, box: item.box }));
  (objects.hero_cards || []).forEach((item, index) => rows.push({ type: "hero_card", label: `hero ${index + 1}`, box: item.box }));
  (objects.board_cards || []).forEach((item, index) => rows.push({ type: "board_card", label: `board ${index + 1}`, box: item.box }));
  if (objects.pot?.box) rows.push({ type: "pot", label: "pot", box: objects.pot.box });
  (objects.action_buttons || []).forEach((item, index) => rows.push({ type: "action_button", label: item.action || `action ${index + 1}`, box: item.box }));
  if (objects.turn_clock?.box) rows.push({ type: "turn_clock", label: "turn clock", box: objects.turn_clock.box });
  return rows.filter((row) => row.box);
}

function sizeCanvas() {
  const image = $("frameImage");
  const canvas = $("boxCanvas");
  if (!image.clientWidth || !image.clientHeight) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.style.width = `${image.clientWidth}px`;
  canvas.style.height = `${image.clientHeight}px`;
  canvas.width = Math.round(image.clientWidth * ratio);
  canvas.height = Math.round(image.clientHeight * ratio);
  renderOverlay();
}

function drawBox(context, box, color, label, width, height) {
  const x = box.x * width, y = box.y * height, w = box.width * width, h = box.height * height;
  context.strokeStyle = color; context.lineWidth = 2; context.strokeRect(x, y, w, h);
  context.font = "11px Segoe UI";
  const labelWidth = context.measureText(label).width + 8;
  context.fillStyle = color; context.fillRect(x, Math.max(0, y - 18), labelWidth, 18);
  context.fillStyle = "#07100c"; context.fillText(label, x + 4, Math.max(12, y - 5));
}

function renderOverlay() {
  const canvas = $("boxCanvas");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.width / ratio, height = canvas.height / ratio;
  if (!width || !height) return;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
  try { boxLabelRows(readAnnotation()).forEach((row) => drawBox(context, row.box, BOX_COLORS[row.type], row.label, width, height)); }
  catch (_) { /* Keep JSON syntax errors in the editor without breaking drawing. */ }
  if (state.drawnBox) drawBox(context, state.drawnBox, "#ffffff", "new", width, height);
}

function setDefaultFields() {
  const type = $("objectType").value;
  const fields = structuredClone(DEFAULT_FIELDS[type]);
  if (type === "seat") fields.seat_no = Number($("objectIndex").value);
  $("objectFields").value = JSON.stringify(fields, null, 2);
}

function normalizedPointer(event) {
  const rect = $("boxCanvas").getBoundingClientRect();
  return { x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)) };
}

function syncCanonicalState(annotation) {
  const objects = annotation.objects || {};
  annotation.state = annotation.state || {};
  const hero = (objects.seats || []).find((seat) => seat.is_hero);
  const dealer = (objects.seats || []).find((seat) => seat.has_dealer_button);
  if (hero) annotation.state.hero_seat = hero.seat_no;
  if (dealer) annotation.state.dealer_seat = dealer.seat_no;
  if (objects.pot?.amount_bb != null) annotation.state.pot_bb = objects.pot.amount_bb;
  if (objects.turn_clock?.remaining_ms != null) {
    annotation.state.decision_time_remaining_ms = Number(objects.turn_clock.remaining_ms);
    annotation.state.decision_deadline_source = "visible_timer";
    annotation.state.hero_to_act = true;
  }
  annotation.state.legal_actions = (objects.action_buttons || []).filter((button) => button.enabled).map((button) => button.action);
  const boardCount = (objects.board_cards || []).length;
  annotation.state.street = ({ 0: "preflop", 3: "flop", 4: "turn", 5: "river" })[boardCount] || annotation.state.street;
  return annotation;
}

function applySuggestionToDraft(annotation, suggestion) {
  if (!suggestion || suggestion.review_required !== true || suggestion.auto_applied !== false) {
    throw new Error("Only review-required, non-auto-applied suggestions are supported.");
  }
  const visible = suggestion.suggested_visible_state;
  if (!visible || typeof visible !== "object") throw new Error("This suggestion has no visible-state fields.");
  annotation.state = annotation.state || {
    game: "holdem_no_limit", table_id: "REVIEW_REQUIRED", hand_id: "REVIEW_REQUIRED",
    action_history: [],
  };
  ["street", "hero_seat", "dealer_seat"].forEach((field) => {
    if (visible[field] != null) annotation.state[field] = visible[field];
  });
  ["pot_bb", "to_call_bb"].forEach((field) => {
    if (visible[field] != null) annotation.state[field] = String(visible[field]);
  });
  if (Array.isArray(visible.legal_actions)) annotation.state.legal_actions = [...visible.legal_actions];
  if (!Array.isArray(annotation.state.action_history)) annotation.state.action_history = [];

  annotation.objects = annotation.objects || {};
  annotation.objects.seats = Array.isArray(annotation.objects.seats) ? annotation.objects.seats : [];
  (visible.seat_stacks_bb || []).forEach((row) => {
    if (!row || !Number.isInteger(row.seat_no)) return;
    let seat = annotation.objects.seats.find((candidate) => candidate.seat_no === row.seat_no);
    if (!seat && row.seat_box) {
      seat = {
        seat_no: row.seat_no, box: structuredClone(row.seat_box), occupied: true,
        is_hero: false, has_dealer_button: false, status: "unknown", visibility: "uncertain",
      };
      annotation.objects.seats.push(seat);
    }
    if (seat && row.stack_bb != null) {
      seat.stack_bb = String(row.stack_bb);
      seat.raw_stack_text = null;
    }
  });
  annotation.objects.seats.forEach((seat) => {
    seat.is_hero = seat.seat_no === annotation.state.hero_seat;
    seat.has_dealer_button = seat.seat_no === annotation.state.dealer_seat;
  });
  if (annotation.objects.pot && visible.pot_bb != null) {
    annotation.objects.pot.amount_bb = String(visible.pot_bb);
  }

  const suggestedObjects = suggestion.suggested_objects;
  if (suggestedObjects && typeof suggestedObjects === "object") {
    if (suggestedObjects.table) annotation.objects.table = structuredClone(suggestedObjects.table);
    (suggestedObjects.seats || []).forEach((row) => {
      if (!row || !Number.isInteger(row.seat_no) || !row.box) return;
      let seat = annotation.objects.seats.find((candidate) => candidate.seat_no === row.seat_no);
      if (!seat) {
        seat = {
          seat_no: row.seat_no, occupied: row.seat_no === annotation.state.hero_seat,
          is_hero: row.seat_no === annotation.state.hero_seat, has_dealer_button: false,
          status: "unknown", visibility: "uncertain",
        };
        annotation.objects.seats.push(seat);
      }
      seat.box = structuredClone(row.box);
    });
    if (suggestedObjects.pot?.box) {
      annotation.objects.pot = annotation.objects.pot || {
        amount_bb: "0", raw_text: "", visibility: "uncertain",
      };
      annotation.objects.pot.box = structuredClone(suggestedObjects.pot.box);
    }
    if (suggestedObjects.turn_clock?.box) {
      annotation.objects.turn_clock = annotation.objects.turn_clock || {
        remaining_ms: 0, raw_text: "", visibility: "uncertain",
      };
      annotation.objects.turn_clock.box = structuredClone(suggestedObjects.turn_clock.box);
    }
    ["hero_cards", "board_cards", "action_buttons"].forEach((collection) => {
      const rows = suggestedObjects[collection] || [];
      annotation.objects[collection] = Array.isArray(annotation.objects[collection]) ? annotation.objects[collection] : [];
      rows.forEach((row, index) => {
        if (!row?.box) return;
        if (!annotation.objects[collection][index]) {
          annotation.objects[collection][index] = collection === "action_buttons"
            ? { enabled: false, raw_text: "", visibility: "uncertain" }
            : { rank: "unknown", suit: "unknown", visibility: "uncertain" };
        }
        annotation.objects[collection][index].box = structuredClone(row.box);
      });
    });
  }

  annotation.provenance = annotation.provenance || {
    annotator: "human-review", annotation_version: 1, created_at: new Date().toISOString(),
  };
  annotation.provenance.verified = false;
  annotation.provenance.reviewer = null;
  const marker = `suggestion_sha256=${suggestion.suggestion_sha256}`;
  const notes = annotation.provenance.notes || "";
  if (!notes.includes(marker)) annotation.provenance.notes = `${notes}${notes ? "\n" : ""}${marker}; human review required`;
  return annotation;
}

function renderSuggestion(suggestion) {
  state.suggestion = suggestion;
  const available = Boolean(suggestion);
  $("suggestionPanel").hidden = !available;
  $("applySuggestionButton").disabled = !available;
  if (!available) return;
  const confidence = Number(suggestion.minimum_supported_confidence);
  $("suggestionConfidence").textContent = Number.isFinite(confidence) ? `${(confidence * 100).toFixed(1)}% raw` : "unscored";
  $("suggestionPreview").textContent = JSON.stringify({
    status: suggestion.model_status,
    suggestion_sha256: suggestion.suggestion_sha256,
    checkpoint_provenance: suggestion.checkpoint_provenance,
    visible_state: suggestion.suggested_visible_state,
    suggested_objects: suggestion.suggested_objects,
    abstentions: suggestion.perception_abstentions,
  }, null, 2);
}

function renderProject(payload) {
  state.project = payload;
  const s = payload.status;
  $("projectSummary").textContent = `${s.project_id} · ${s.frames} frames · ${s.verified_annotations} verified · ${s.capture_sessions} sessions`;
  const rows = payload.records.map((record) => {
    const button = document.createElement("button");
    button.className = `frame-row ${state.selected === record.sample_id ? "active" : ""}`;
    const title = document.createElement("strong"); title.textContent = record.sample_id;
    const meta = document.createElement("small"); meta.textContent = `${record.capture_session_id} · ${record.width}×${record.height} · ${record.verified ? "verified" : record.annotated ? "draft" : record.suggested ? "suggestion" : "pending"}`;
    button.append(title, meta);
    button.addEventListener("click", () => selectFrame(record.sample_id));
    return button;
  });
  $("frameList").replaceChildren(...rows);
}

async function refresh() {
  renderProject(await api("/api/project"));
}

async function selectFrame(sampleId) {
  state.selected = sampleId;
  const result = await api(`/api/annotations/${sampleId}`);
  const record = state.project.records.find((row) => row.sample_id === sampleId);
  $("selectedTitle").textContent = sampleId;
  $("frameImage").src = `/api/frames/${sampleId}`;
  $("annotationEditor").value = JSON.stringify(result.annotation, null, 2);
  renderSuggestion(result.suggestion);
  $("frameBadge").textContent = record.verified ? "verified" : result.saved ? "saved" : "template";
  $("frameBadge").className = `badge ${record.verified ? "verified" : ""}`;
  renderProject(state.project);
  message(result.saved ? "Saved annotation loaded." : "Base template loaded. Add state, objects and provenance before saving.");
  state.drawnBox = null; $("applyBoxButton").disabled = true; $("boxReadout").textContent = "No box drawn"; renderOverlay();
}

$("refreshButton").addEventListener("click", () => refresh().catch((error) => message(error.message, "error")));
$("reloadButton").addEventListener("click", () => state.selected && selectFrame(state.selected).catch((error) => message(error.message, "error")));
$("formatButton").addEventListener("click", () => {
  try { writeAnnotation(readAnnotation()); message("JSON formatted.", "ok"); }
  catch (error) { message(error.message, "error"); }
});
$("annotationEditor").addEventListener("input", renderOverlay);
$("applySuggestionButton").addEventListener("click", () => {
  try {
    writeAnnotation(applySuggestionToDraft(readAnnotation(), state.suggestion));
    message("Suggested fields copied into the unsaved draft. Review every field and complete all missing boxes before verification.", "ok");
  } catch (error) { message(error.message, "error"); }
});
$("frameImage").addEventListener("load", sizeCanvas);
new ResizeObserver(sizeCanvas).observe($("imageSurface"));
$("objectType").addEventListener("change", setDefaultFields);
$("objectIndex").addEventListener("change", () => { if ($("objectType").value === "seat") setDefaultFields(); });
$("boxCanvas").addEventListener("pointerdown", (event) => { state.dragStart = normalizedPointer(event); $("boxCanvas").setPointerCapture(event.pointerId); });
$("boxCanvas").addEventListener("pointermove", (event) => {
  if (!state.dragStart) return;
  const end = normalizedPointer(event), start = state.dragStart;
  state.drawnBox = { x: Math.min(start.x, end.x), y: Math.min(start.y, end.y), width: Math.abs(end.x - start.x), height: Math.abs(end.y - start.y) };
  renderOverlay();
});
$("boxCanvas").addEventListener("pointerup", () => {
  state.dragStart = null;
  const box = state.drawnBox;
  const valid = box && box.width >= 0.002 && box.height >= 0.002;
  $("applyBoxButton").disabled = !valid;
  $("boxReadout").textContent = valid ? `${box.x.toFixed(4)}, ${box.y.toFixed(4)} · ${box.width.toFixed(4)} × ${box.height.toFixed(4)}` : "Draw a larger box";
});
$("applyBoxButton").addEventListener("click", () => {
  try {
    if (!state.drawnBox) throw new Error("Draw a box first.");
    const annotation = readAnnotation(), type = $("objectType").value, index = Number($("objectIndex").value) - 1;
    const fields = JSON.parse($("objectFields").value || "{}");
    annotation.objects = annotation.objects || {};
    const object = { box: state.drawnBox, ...fields };
    if (type === "table") annotation.objects.table = state.drawnBox;
    else if (type === "pot") annotation.objects.pot = object;
    else if (type === "turn_clock") annotation.objects.turn_clock = object;
    else {
      const collection = ({ seat: "seats", hero_card: "hero_cards", board_card: "board_cards", action_button: "action_buttons" })[type];
      annotation.objects[collection] = annotation.objects[collection] || [];
      annotation.objects[collection][index] = object;
      annotation.objects[collection] = annotation.objects[collection].filter(Boolean);
    }
    writeAnnotation(annotation); state.drawnBox = null; $("applyBoxButton").disabled = true; $("boxReadout").textContent = "Applied to JSON"; renderOverlay();
    message("Box and fields applied. Sync state, then validate and save.", "ok");
  } catch (error) { message(error.message, "error"); }
});
$("syncStateButton").addEventListener("click", () => {
  try { writeAnnotation(syncCanonicalState(readAnnotation())); message("Hero, dealer, pot, street and legal actions synchronized from objects.", "ok"); }
  catch (error) { message(error.message, "error"); }
});
$("saveButton").addEventListener("click", async () => {
  if (!state.selected) return message("Choose a frame first.", "error");
  try {
    const annotation = JSON.parse($("annotationEditor").value);
    await api(`/api/annotations/${state.selected}`, { method: "POST", body: JSON.stringify(annotation) });
    await refresh();
    await selectFrame(state.selected);
    message("Annotation validated and saved.", "ok");
  } catch (error) { message(error.message, "error"); }
});
$("importButton").addEventListener("click", async () => {
  const paths = $("framePaths").value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  try {
    const result = await api("/api/import", { method: "POST", body: JSON.stringify({ paths, capture_session_id: $("sessionId").value.trim(), timestamp_ms: Number($("timestampMs").value) }) });
    renderProject(result.project); $("framePaths").value = ""; message(`Imported ${result.frames.filter((row) => row.inserted).length} new frame(s).`, "ok");
  } catch (error) { message(error.message, "error"); }
});
$("exportButton").addEventListener("click", async () => {
  try {
    const result = await api("/api/export", { method: "POST", body: JSON.stringify({ dataset_version: $("datasetVersion").value.trim() }) });
    message(`Manifest exported: ${result.manifest}\nMinimum dataset ready: ${result.validation.minimum_dataset.ready}`, "ok");
  } catch (error) { message(error.message, "error"); }
});

setDefaultFields();
refresh().catch((error) => message(error.message, "error"));
