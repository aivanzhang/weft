"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const fragment = new URLSearchParams(location.hash.slice(1));
  const token = fragment.get("token") || sessionStorage.getItem("weft-lab-token") || "";
  if (token) sessionStorage.setItem("weft-lab-token", token);
  history.replaceState(null, "", location.pathname);
  let current = null, scene = null, snapshot = null, running = false, usePolicy = false;
  let action = 0, requestPending = false, recording = false, activePanel = "evidence";
  let statusTimer = null, playbackTimer = null, lastEvents = "", lastFiles = "", loadedReady = false;
  let playbackGeneration = 0, stepPromise = Promise.resolve(), controlBusy = false, mutationEpoch = 0;
  let pollPending = false, pollAgain = false, referenceFrames = [], referenceState = null;
  let mode = "manual", trainedJoint = null, trainedActor = null, capabilities = null, exporting = false, referenceTimer = null;
  let selectedMotor = null, wholeBodyPolicyReady = false, simulationFailureMessage = null, learnedPolicyAvailable = true;
  let coverageLoaded = false;
  let poseCoverageLoaded = false, poseCoveragePending = false, poseCoverageAttempts = 0;
  let poseIssues = [], poseIssueIndex = -1, pendingSourceSeek = null;
  let inspection = null, inspectionLoaded = false, inspectionPending = false, inspectionAttempts = 0;
  let evidenceScorecardSignature = null;
  let exerciseTask = null, exerciseEnvelope = null, exerciseIdentity = null, exerciseActions = null, exerciseContractReady = false, exerciseStateReady = false;
  let feetPanel = null, feetSummary = null, feetCoverage = null, feetDetails = null, feetRows = null, feetDetailSignature = "";
  let exerciseDetailSignature = "", exercisePolicy = null, cameraPolicyView = null;
  let fittedReference = null, fittedButton = null, fittedSummary = null, fittedComparison = null;
  let mediaClockReady = false;
  let recordingDiagnosis = null, requestedRecording = null, recordingReadPending = false, recordingRetryTimer = null;
  let recordingReadAttempts = 0;
  let requestedRecordingEvidence = null, recordingEvidenceRequest = null, recordingEvidenceTimer = null;
  let recordingEvidenceGeneration = 0, recordingEvidenceAttempts = 0, recordingEvidenceSuspended = false;
  let recordingEvidenceSignature = null;
  let appearanceView = null;
  const inspectionActors = new Map();
  const actorLabel = (id) => inspectionActors.get(id)?.label || scene?.objects?.find(obj => obj.id === id)?.label || id;
  const wholeBody = () => scene?.profile === "exercise-imitation-v1";
  const spatialRigid = () => scene?.profile === "spatial-rigid-v1";
  const sharedFeet = () => scene?.profile === "shared-world-v5";
  const cameraWorld = () => scene?.profile === "shared-world-v6";
  const sharedWorld = () => ["shared-world-v1", "shared-world-v2", "shared-world-v3", "shared-world-v4", "shared-world-v5", "shared-world-v6"].includes(scene?.profile);
  const baseSceneProfile = () => sharedFeet() ? scene.base_profile : scene?.profile;
  const personOnly = () => baseSceneProfile() === "shared-world-v3";
  const staticSupport = () => baseSceneProfile() === "shared-world-v4";
  const sharedExercise = () => current?.control_profile === "shared_exercise" || staticSupport() || cameraWorld();
  const engineState = state => ["shared-exercise-state-v1", "shared-exercise-feet-state-v1", "camera-shared-state-v1"].includes(state?.profile) ? state.engine_state : state;
  const exerciseMetrics = () => sharedExercise() && exerciseStateReady ? snapshot.task_metrics : null;
  const sharedHumanTarget = () => sharedWorld() && (sharedExercise() || (snapshot?.controlled_id || scene?.controlled_id) === "human");
  const sharedObjectTarget = () => sharedWorld() && !sharedHumanTarget();
  const dynamicObjects = () => (scene?.objects || []).filter((obj) => obj.role !== "static");
  const number = (value, places = 3) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(places) : "unavailable";
  const held = new Set();
  const keyActions = {ArrowLeft: 1, a: 1, ArrowRight: 2, d: 2, ArrowUp: 3, w: 3, ArrowDown: 4, s: 4, " ": 5};
  const spatialKeys = {ArrowLeft: 1, a: 1, A: 1, ArrowRight: 2, d: 2, D: 2, ArrowUp: 3, w: 3, W: 3, ArrowDown: 4, s: 4, S: 4, q: 5, Q: 5, PageUp: 5, e: 6, E: 6, PageDown: 6, " ": 7};
  const keyAction = (key) => (spatialRigid() || sharedObjectTarget()) ? spatialKeys[key] : (wholeBody() || sharedHumanTarget()) ? (key === " " ? 0 : ["ArrowLeft", "a", "ArrowUp", "w"].includes(key) ? 1 : ["ArrowRight", "d", "ArrowDown", "s"].includes(key) ? 2 : undefined) : keyActions[key] === undefined ? undefined : current?.profile === "exercise" ? (key === " " ? 3 : ["ArrowLeft", "a", "ArrowUp", "w"].includes(key) ? 1 : 2) : keyActions[key];
  let actionNames = ["coast", "left", "right", "up", "down", "brake"];
  const fileURL = (name) => `/files/${name.split("/").map(encodeURIComponent).join("/")}?token=${encodeURIComponent(token)}`;
  const text = (element, value) => { element.textContent = String(value); };
  function windowText(id, value) { if ($(id).textContent !== value) text($(id), value); }
  function updateWindowClocks() {
    if ($("source-simulation-window").hidden) return;
    const video = $("source"), state = engineState(snapshot);
    windowText("window-media", mediaClockReady && Number.isFinite(video.currentTime)
      ? `Original video player: ${number(video.currentTime)}s${Number.isFinite(video.duration) ? ` / ${number(video.duration)}s media duration` : ""}. Playback is independent of the simulation.`
      : "Original video player clock unavailable until media metadata loads. No source duration is inferred from frame count.");
    const valid = !sharedExercise() || exerciseStateReady;
    const stale = sharedExercise() ? snapshot?.engine_state_status === "last_valid_control_boundary" : Boolean(state?.simulation_failure);
    windowText("window-live", valid && Number.isFinite(state?.time_s)
      ? `${stale ? "Last valid simulation geometry" : "Displayed simulation geometry"}: ${number(state.time_s)}s elapsed; ${Number.isFinite(state.source_time_s) ? `${number(state.source_time_s)}s on the declared reference clock` : "reference-clock position unavailable"}. ${stale ? "Simulation failed; this is not a new terminal pose." : "This is a simulated state, not the recorded video frame."}`
      : "Simulation/reference-clock position unavailable. No zero time or synchronized source pose is inferred.");
  }
  function showSourceWindow() {
    const timeline = inspection?.timeline;
    $("source-simulation-window").hidden = !sharedWorld() && timeline?.status !== "available";
    if ($("source-simulation-window").hidden) return;
    const pair = value => Array.isArray(value) && value.length === 2 && value.every(Number.isFinite) && value[0] <= value[1];
    const reference = timeline?.reference_span_s, task = timeline?.task_window_s;
    windowText("window-reference", pair(reference)
      ? `Retained body reference: ${number(reference[0])}–${number(reference[1])}s (${Number.isInteger(timeline.reference_poses) ? timeline.reference_poses : "unknown"} poses), on its declared reference clock.`
      : "Retained reference span unavailable in this saved summary; the source video is not assumed fully reconstructed.");
    windowText("window-task", pair(task) && Number.isFinite(timeline.task_duration_s) && Number.isInteger(timeline.task_steps)
      ? `Selected simulation window: ${number(task[0])}–${number(task[1])}s on that reference clock (${number(timeline.task_duration_s)}s, ${timeline.task_steps} control steps). ${pair(reference) && (task[0] > reference[0] + 1e-9 || task[1] < reference[1] - 1e-9) ? "Only part of the retained reference is tested." : "Covering a reference window does not establish complete source-video coverage."}`
      : "Selected reference-time window unavailable. The live task's declared interval is reported separately below.");
    if (inspection?.fullspan_preparation?.status === "preparation_only") {
      windowText("window-task", `${$("window-task").textContent} Full-reference preparation changes the task interval, not the recovered geometry. Inspect fullspan.json in Files for the preserved inputs and limits.`);
    }
    const slots = timeline?.source_slots;
    windowText("window-coverage", slots && [slots.with_reference, slots.without_reference, slots.leading_without_reference, slots.trailing_without_reference].every(Number.isInteger)
      ? `Source slots: ${slots.with_reference} with a retained body reference; ${slots.without_reference} without one, including ${slots.leading_without_reference} leading and ${slots.trailing_without_reference} trailing slots. These are reference-coverage counts, not detection or visibility labels; gaps are not filled.`
      : "Source-to-reference slot coverage unavailable here. Pose counts alone do not identify missing source frames.");
    windowText("window-clocks", timeline?.alignment_method === "pose_frame_index_over_fps_prior_v1"
      ? "Reference times use an authored frame-index / FPS prior. Native source timestamps are not read by this summary. Video, kinematic reference and live simulation are not synchronized."
      : "Reference times use the saved declared clock; its mapping to native source timestamps is unavailable here. Video, kinematic reference and live simulation are not synchronized.");
    if (cameraWorld()) {
      const clock = scene.camera_coupling?.clock;
      windowText("window-clocks", `Original source grid: ${number(clock?.source_fps, 6)} frames/s; control period ${number(scene.dt, 6)}s; unchanged body-reference period ${number(scene.human_spec?.dt, 6)}s. Source start ${number(clock?.source_time_offset_s)}s. The clock hash is an identity, not a timestamp. Video, kinematic replay and simulation play independently.`);
    }
    updateWindowClocks();
  }
  function updateCapabilities(doc) {
    if (!doc) return;
    capabilities = doc;
    text($("capabilities-json"), JSON.stringify(doc, null, 2));
    if (doc.controllers?.trained_target_id && !doc.entities?.articulated_body_present) {
      trainedActor = doc.controllers.trained_target_id;
    }
  }
  const error = (message) => { simulationFailureMessage = null; text($("error"), message); $("error").hidden = false; };
  const clearError = () => { simulationFailureMessage = null; $("error").hidden = true; };
  async function request(path, data, signal) {
    const response = await fetch(path, {headers: {Authorization: `Bearer ${token}`, ...(data ? {"Content-Type": "application/json"} : {})},
      ...(data ? {method: "POST", body: JSON.stringify(data)} : {}), signal: signal || AbortSignal.timeout(10000)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
    return result;
  }
  function link(label, path) {
    const a = document.createElement("a"); a.textContent = label; a.href = fileURL(path); a.download = path.split("/").pop(); return a;
  }
  function cell(row, value) { const td = document.createElement("td"); td.textContent = value; row.append(td); return td; }
  function showPanel(name) {
    pause(); stopReference(); activePanel = name;
    for (const panel of document.querySelectorAll(".panel")) panel.hidden = panel.id !== name;
    for (const button of $("stages").querySelectorAll("button")) {
      button.classList.toggle("selected", button.dataset.panel === name);
      button.setAttribute("aria-current", button.dataset.panel === name ? "step" : "false");
    }
    if (name === "world") showGeometryPanel();
    appearanceView?.setVisible(name === "play" && !document.hidden);
  }
  for (const button of $("stages").querySelectorAll("button")) button.onclick = () => showPanel(button.dataset.panel);

  function draw(canvas, state) {
    if (canvas.id === "live" && appearanceView?.update(state)) return;
    state = engineState(state);
    if (!scene || !state) return;
    if (staticSupport() && !supportObjectsMatch(state.objects)) return;
    if (sharedWorld()) { drawSharedWorld(canvas, state); return; }
    if (state.human) { drawHuman(canvas, state); return; }
    if (spatialRigid()) { drawSpatial(canvas, state); return; }
    const width = 960, height = Math.round(width * scene.height / (scene.width || 1));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext("2d"), scale = width / (scene.width || 1);
    ctx.fillStyle = "#FBF7F1"; ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#E9DED1"; ctx.lineWidth = 1;
    for (let x = 0; x < width; x += scale / 10) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
    for (let y = 0; y < height; y += scale / 10) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }
    if (scene.support) { ctx.strokeStyle = "#214F50"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(0, scene.support.intercept * scale); ctx.lineTo(width, (scene.support.intercept + scene.support.slope) * scale); ctx.stroke(); }
    const target = state.target || scene.target;
    if (target) { ctx.strokeStyle = "#965644"; ctx.setLineDash([8, 6]); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(target[0] * scale, target[1] * scale, (scene.task.position_tolerance || .06) * scale, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); ctx.font = "15px system-ui"; ctx.fillStyle = "#965644"; ctx.fillText("Authored target", Math.min(width - 130, target[0] * scale + 10), Math.max(20, target[1] * scale - 16)); }
    for (const body of state.objects) {
      const model = scene.objects.find((item) => item.id === body.id) || body;
      const [x, y] = body.position; ctx.save(); ctx.translate(x * scale, y * scale); ctx.rotate(body.angle || 0);
      ctx.fillStyle = model.color || "#C5A059"; ctx.strokeStyle = "#214F50"; ctx.lineWidth = 2;
      ctx.beginPath();
      if (model.shape === "box") ctx.rect(-model.size[0] * scale / 2, -model.size[1] * scale / 2, model.size[0] * scale, model.size[1] * scale);
      else ctx.arc(0, 0, model.radius * scale, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      if (body.id === scene.controlled_id) { ctx.strokeStyle = "#965644"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(0, 0, (model.radius || Math.max(...model.size) / 2) * scale + 7, 0, Math.PI * 2); ctx.stroke(); }
      ctx.restore(); ctx.font = "13px system-ui"; ctx.fillStyle = "#214F50";
      ctx.fillText(model.label || body.id, Math.min(width - 100, Math.max(5, x * scale + 14)), Math.max(17, y * scale - 18));
    }
  }
  function drawHuman(canvas, state) {
    const width = 960, height = 640;
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext("2d"), human = state.human;
    const yaw = Number($("camera-yaw").value), pitch = 0.2, cy = Math.cos(yaw), sy = Math.sin(yaw);
    const viewHeight = human.view_height || human.height, origin = human.view_origin || [0, 0, 0];
    const scale = Math.min(width / (viewHeight * 1.3), height / (viewHeight * 1.2));
    const project = ([x, y, z]) => { x -= origin[0]; y -= origin[1]; z -= origin[2];
      return [width / 2 + (x * cy - y * sy) * scale, height * (human.view_origin ? .5 : .88) - (z * Math.cos(pitch) - (x * sy + y * cy) * Math.sin(pitch)) * scale]; };
    ctx.fillStyle = "#FBF7F1"; ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#D8C9BA"; ctx.lineWidth = 1;
    const floor = human.mobility?.floor_height || 0;
    const normal = human.mobility?.floor_normal || [0, 0, 1];
    const grid = human.view_mode === "body_follow" ? origin.slice(0, 2).map(value => Math.round(value * 5) / 5) : [0, 0];
    const floorPoint = (x, y) => { x += grid[0]; y += grid[1]; return [x, y, floor - (normal[0] * x + normal[1] * y) / normal[2]]; };
    for (let i = -5; i <= 5; i++) {
      for (const [a, b] of [[floorPoint(i * .2, -1), floorPoint(i * .2, 1)], [floorPoint(-1, i * .2), floorPoint(1, i * .2)]]) {
        const p = project(a), q = project(b); ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q); ctx.stroke();
      }
    }
    const segments = [...human.segments].sort((a, b) => (a.a[0] + a.b[0] - b.a[0] - b.b[0]) * sy + (a.a[1] + a.b[1] - b.a[1] - b.b[1]) * cy);
    ctx.lineCap = "round";
    for (const segment of segments) {
      const a = project(segment.a), b = project(segment.b);
      ctx.strokeStyle = "#214F50"; ctx.lineWidth = Math.max(4, segment.radius * scale * 2 + 3); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
      ctx.strokeStyle = segment.id === state.controlled_id ? "#C6A58A" : "#9BB3B0"; ctx.lineWidth = Math.max(2, segment.radius * scale * 2); ctx.stroke();
      for (const point of [a, b]) { ctx.fillStyle = "#F5EADC"; ctx.beginPath(); ctx.arc(...point, 3, 0, 2 * Math.PI); ctx.fill(); }
    }
    if ($("show-muscles").checked) {
      for (const muscle of human.muscles) {
        const a = project(muscle.a), b = project(muscle.b), value = muscle.activation;
        ctx.strokeStyle = `rgb(${Math.round(150 + value * 60)},${Math.round(110 - value * 55)},${Math.round(90 - value * 45)})`; ctx.lineWidth = 4 + value * 5; ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
      }
    }
    ctx.fillStyle = "#214F50"; ctx.font = "15px system-ui";
    for (const contact of human.mobility?.contacts || []) {
      const point = project(contact.position); ctx.fillStyle = "#965644";
      ctx.beginPath(); ctx.arc(...point, 4, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.fillStyle = "#214F50";
    ctx.fillText(human.reference ? "Captured pose · estimated from video" : human.control_mode === "direct_motor_actions" ? "Explicit motor commands · no automatic support" : human.mobility ? "Mobile simulation · assumed floor and support" : "Simulated body · fixed at the pelvis", 20, 27);
    ctx.font = "12px system-ui"; ctx.fillText((human.view_mode === "body_follow" ? "View follows body · " : "") + "Illustrative muscles · not measured activation", 20, height - 18);
  }
  const spatialFaces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]];
  function rotateQuat([x, y, z], [w, a, b, c]) {
    return [
      (1 - 2 * (b * b + c * c)) * x + 2 * (a * b - w * c) * y + 2 * (a * c + w * b) * z,
      2 * (a * b + w * c) * x + (1 - 2 * (a * a + c * c)) * y + 2 * (b * c - w * a) * z,
      2 * (a * c - w * b) * x + 2 * (b * c + w * a) * y + (1 - 2 * (a * a + b * b)) * z,
    ];
  }
  function spatialCorners(obj) {
    const result = [];
    for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
      const r = rotateQuat([x * obj.size[0] / 2, y * obj.size[1] / 2, z * obj.size[2] / 2], obj.quaternion);
      result.push(r.map((v, i) => v + obj.position[i]));
    }
    return result;
  }
  function composeQuat([w, x, y, z], [a, b, c, d]) {
    return [w * a - x * b - y * c - z * d, w * b + x * a + y * d - z * c,
      w * c - x * d + y * a + z * b, w * d + x * c - y * b + z * a];
  }
  function sharedGeometry(objects) {
    const primitives = [], extents = new Map();
    const vector = (value, length, bound) => Array.isArray(value) && value.length === length && value.every(v => Number.isFinite(v) && Math.abs(v) <= bound);
    const quaternion = value => vector(value, 4, 1) && Math.abs(Math.hypot(...value) - 1) <= 1e-8;
    const unavailable = () => ({primitives: [], extents, error: "Rigid geometry unavailable · unsupported or invalid compound; no substitute drawn"});
    if (!Array.isArray(objects) || objects.length > 12) return unavailable();
    for (const obj of objects) {
      if (obj.shape !== "compound") { primitives.push(obj); continue; }
      const geometry = obj.geometry;
      if (geometry?.profile !== "rigid-compound-v1" || !Array.isArray(geometry.parts) || !geometry.parts.length || geometry.parts.length > 8
        || !vector(obj.position, 3, 100) || !quaternion(obj.quaternion)) return unavailable();
      let extent = 0;
      for (const part of geometry.parts) {
        if (!part || !vector(part.position, 3, 4) || !quaternion(part.quaternion)) return unavailable();
        let radius;
        if (part.shape === "box" && vector(part.size, 3, 4) && part.size.every(v => v >= .01)) radius = Math.hypot(...part.size) / 2;
        else if ((part.shape === "sphere" || part.shape === "capsule") && Number.isFinite(part.radius) && part.radius >= .01 && part.radius <= .5) {
          radius = part.radius;
          if (part.shape === "capsule") {
            if (!Number.isFinite(part.length) || part.length < .01 || part.length > 4) return unavailable();
            radius += part.length / 2;
          }
        } else return unavailable();
        const offset = rotateQuat(part.position, obj.quaternion);
        primitives.push({...part, id: obj.id, compoundPart: true, role: obj.role, color: obj.color,
          position: offset.map((v, i) => v + obj.position[i]), quaternion: composeQuat(obj.quaternion, part.quaternion)});
        extent = Math.max(extent, Math.hypot(...part.position) + radius);
      }
      extents.set(obj.id, extent);
    }
    if (primitives.length > 32) return unavailable();
    // Transform once per frame; all three projections share the same part geometry.
    return {extents, primitives: primitives.map(obj => {
      if (obj.shape === "box") return {obj, vertices: spatialCorners(obj)};
      if (obj.shape === "capsule") {
        const axis = rotateQuat([0, 0, obj.length / 2], obj.quaternion);
        return {obj, a: axis.map((v, i) => obj.position[i] - v), b: axis.map((v, i) => obj.position[i] + v)};
      }
      return {obj};
    })};
  }
  function capsulePath(ctx, a, b, radius) {
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    // Orthographic projection of a capsule is a stadium, including an end-on circle.
    ctx.beginPath();
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-9) ctx.arc(...a, radius, 0, 2 * Math.PI);
    else {
      ctx.moveTo(a[0] + Math.sin(angle) * radius, a[1] - Math.cos(angle) * radius);
      ctx.lineTo(b[0] + Math.sin(angle) * radius, b[1] - Math.cos(angle) * radius);
      ctx.arc(...b, radius, angle - Math.PI / 2, angle + Math.PI / 2);
      ctx.lineTo(a[0] - Math.sin(angle) * radius, a[1] + Math.cos(angle) * radius);
      ctx.arc(...a, radius, angle + Math.PI / 2, angle + 3 * Math.PI / 2);
    }
    ctx.closePath();
  }
  function geometryLabel(obj) {
    const parts = obj.geometry?.parts;
    return obj.shape === "compound" ? Array.isArray(parts) && parts.length >= 1 && parts.length <= 8 && parts.every(part => part && typeof part.id === "string")
      ? `compound · ${parts.length} authored parts (${parts.map(part => `${part.id}: ${part.shape}`).join(", ")})`
      : "compound · geometry unavailable" : obj.shape;
  }
  function drawSpatial(canvas, state) {
    const width = 1080, height = 400;
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#FBF7F1"; ctx.fillRect(0, 0, width, height);
    const yaw = Number($("spatial-yaw")?.value ?? 0.5), zoom = Number($("spatial-zoom")?.value ?? 1);
    const follow = Boolean($("spatial-follow")?.checked);
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(0.5), sp = Math.sin(0.5), pane = width / 3;
    const scale = zoom * Math.min(pane * 0.85, height * 0.76 / Math.max(...scene.bounds));
    const movingObjects = state.objects.filter((o) => o.role !== "static");
    const center = follow && movingObjects.length
      ? [0, 1, 2].map((i) => movingObjects.reduce((sum, o) => sum + o.position[i], 0) / movingObjects.length)
      : scene.bounds.map((v) => v / 2);
    const views = [
      {name: "Front: x / height", matrix: [[1, 0, 0], [0, 0, -1]], depth: [0, 1, 0]},
      {name: "Top: x / depth", matrix: [[1, 0, 0], [0, -1, 0]], depth: [0, 0, 1]},
      {name: "3D: same state", matrix: [[cy, -sy, 0], [sy * sp, cy * sp, -cp]], depth: [sy * cp, cy * cp, sp]},
    ];
    const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
    const controlledId = state.controlled_id || scene.controlled_id;
    views.forEach((view, index) => {
      ctx.save();
      ctx.beginPath(); ctx.rect(index * pane, 0, pane, height); ctx.clip();
      if (index > 0) { ctx.strokeStyle = "#E9DED1"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(index * pane, 0); ctx.lineTo(index * pane, height); ctx.stroke(); }
      const project = (p) => {
        const q = p.map((v, i) => v - center[i]);
        return [(index + 0.5) * pane + dot(q, view.matrix[0]) * scale, height * 0.55 + dot(q, view.matrix[1]) * scale];
      };
      ctx.fillStyle = "#214F50"; ctx.font = "15px system-ui"; ctx.fillText(view.name, index * pane + 10, 22);
      const shapes = [];
      for (const body of state.objects) {
        const model = scene.objects.find((item) => item.id === body.id) || body;
        const obj = {...model, ...body};
        const moving = obj.role !== "static";
        if (obj.shape === "sphere") { shapes.push({moving, depth: dot(obj.position, view.depth), object: obj}); continue; }
        const vertices = spatialCorners(obj);
        for (const face of spatialFaces) {
          const points = face.map((i) => vertices[i]);
          shapes.push({moving, depth: points.reduce((sum, p) => sum + dot(p, view.depth), 0) / 4, points, color: obj.color || "#C5A059"});
        }
      }
      for (const shape of shapes.sort((a, b) => Number(a.moving) - Number(b.moving) || a.depth - b.depth)) {
        ctx.beginPath();
        ctx.strokeStyle = shape.moving ? "#214F50" : "#AAA399";
        ctx.lineWidth = 1;
        if (shape.object) {
          const obj = shape.object, [x, y] = project(obj.position);
          ctx.fillStyle = obj.color || "#C5A059";
          ctx.arc(x, y, obj.radius * scale, 0, 2 * Math.PI);
        } else {
          ctx.fillStyle = shape.color;
          shape.points.forEach((p, i) => { const [x, y] = project(p); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
          ctx.closePath();
        }
        if (shape.moving) ctx.fill();
        ctx.stroke();
      }
      const controlled = state.objects.find((item) => item.id === controlledId);
      if (controlled) {
        const model = scene.objects.find((item) => item.id === controlled.id) || controlled;
        const [cx, cyPos] = project(controlled.position);
        const extent = (model.radius || Math.max(...(model.size || [0.05])) / 2) * scale + 5;
        ctx.beginPath(); ctx.strokeStyle = "#965644"; ctx.lineWidth = 2; ctx.arc(cx, cyPos, extent, 0, 2 * Math.PI); ctx.stroke();
      }
      const target = state.target || scene.target;
      if (target) {
        const [tx, ty] = project(target);
        ctx.beginPath(); ctx.strokeStyle = "#965644"; ctx.setLineDash([6, 4]); ctx.lineWidth = 2;
        ctx.arc(tx, ty, (scene.task.position_tolerance || 0.03) * scale, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
      }
      for (const contact of state.contact_details || []) {
        const [x, y] = project(contact.position);
        ctx.beginPath(); ctx.fillStyle = "#965644"; ctx.arc(x, y, 2.5, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.restore();
    });
  }
  function drawSharedWorld(canvas, state) {
    const width = 1080, height = 420;
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#FBF7F1"; ctx.fillRect(0, 0, width, height);
    const yaw = Number($("spatial-yaw")?.value ?? 0.5), zoom = Number($("spatial-zoom")?.value ?? 1);
    const follow = Boolean($("spatial-follow")?.checked), showMuscles = Boolean($("show-muscles")?.checked);
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(0.5), sp = Math.sin(0.5), pane = width / 3;
    const scale = zoom * Math.min(pane * 0.84 / Math.max(scene.bounds[0], scene.bounds[1]), height * 0.76 / Math.max(...scene.bounds));
    const movingObjects = state.objects.filter((o) => o.role !== "static");
    const center = follow && movingObjects.length
      ? [0, 1, 2].map((i) => movingObjects.reduce((sum, o) => sum + o.position[i], 0) / movingObjects.length)
      : scene.bounds.map((v) => v / 2);
    const floorZ = state.human?.mobility?.floor_height ?? 0;
    const goalTol = scene.task?.object_goal?.position_tolerance || 0.05;
    const controlledId = state.controlled_id || scene.controlled_id;
    const models = new Map(scene.objects.map(obj => [obj.id, obj]));
    const geometry = sharedGeometry(state.objects.map(body => ({...models.get(body.id), ...body})));
    const views = [
      {name: "Front: x / height", matrix: [[1, 0, 0], [0, 0, -1]], depth: [0, 1, 0]},
      {name: "Top: x / depth", matrix: [[1, 0, 0], [0, -1, 0]], depth: [0, 0, 1]},
      {name: cameraWorld() ? "3D: camera body and supplied ground" : personOnly() ? "3D: human and ground" : "3D: coupled world", matrix: [[cy, -sy, 0], [sy * sp, cy * sp, -cp]], depth: [sy * cp, cy * cp, sp]},
    ];
    const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
    views.forEach((view, index) => {
      ctx.save();
      ctx.beginPath(); ctx.rect(index * pane, 0, pane, height); ctx.clip();
      if (index > 0) { ctx.strokeStyle = "#E9DED1"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(index * pane, 0); ctx.lineTo(index * pane, height); ctx.stroke(); }
      const project = (p) => {
        const q = p.map((v, i) => v - center[i]);
        return [(index + 0.5) * pane + dot(q, view.matrix[0]) * scale, height * 0.54 + dot(q, view.matrix[1]) * scale];
      };
      ctx.fillStyle = "#214F50"; ctx.font = "15px system-ui"; ctx.fillText(view.name, index * pane + 10, 22);
      if (geometry.error) { ctx.font = "11px system-ui"; ctx.fillText(geometry.error, index * pane + 10, 40, pane - 20); }
      ctx.strokeStyle = "#E6D8C8"; ctx.lineWidth = 1;
      for (const f of [0.2, 0.5, 0.8]) {
        const gx = scene.bounds[0] * f, gy = scene.bounds[1] * f;
        for (const [a, b] of [[[gx, 0, floorZ], [gx, scene.bounds[1], floorZ]], [[0, gy, floorZ], [scene.bounds[0], gy, floorZ]]]) {
          const p1 = project(a), p2 = project(b); ctx.beginPath(); ctx.moveTo(...p1); ctx.lineTo(...p2); ctx.stroke();
        }
      }
      const items = [];
      for (const primitive of geometry.primitives) {
        const {obj} = primitive;
        const moving = obj.role !== "static";
        if (obj.shape === "sphere") { items.push({tier: moving ? 1 : 0, depth: dot(obj.position, view.depth), kind: "sphere", obj}); continue; }
        if (obj.shape === "capsule") { items.push({tier: moving ? 1 : 0, depth: dot(obj.position, view.depth), kind: "rigid-capsule", ...primitive}); continue; }
        const {vertices} = primitive;
        for (const face of spatialFaces) {
          const points = face.map((i) => vertices[i]);
          items.push({tier: moving ? 1 : 0, depth: points.reduce((sum, p) => sum + dot(p, view.depth), 0) / 4, kind: "poly", moving, points, color: obj.color || "#C5A059"});
        }
      }
      for (const seg of state.human?.segments || []) {
        const mid = [0, 1, 2].map((i) => (seg.a[i] + seg.b[i]) / 2);
        items.push({tier: 1, depth: dot(mid, view.depth), kind: "capsule", seg});
      }
      for (const item of items.sort((a, b) => a.tier - b.tier || a.depth - b.depth)) {
        if (item.kind === "poly") {
          ctx.beginPath(); ctx.strokeStyle = item.moving ? "#214F50" : "#AAA399"; ctx.lineWidth = 1; ctx.fillStyle = item.color;
          item.points.forEach((p, i) => { const [x, y] = project(p); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
          ctx.closePath(); if (item.moving) ctx.fill(); ctx.stroke();
        } else if (item.kind === "sphere") {
          const obj = item.obj, [x, y] = project(obj.position);
          ctx.beginPath(); ctx.strokeStyle = obj.role !== "static" ? "#214F50" : "#AAA399"; ctx.lineWidth = 1;
          ctx.fillStyle = obj.color || "#C5A059"; ctx.arc(x, y, obj.compoundPart ? obj.radius * scale : Math.max(2, obj.radius * scale), 0, 2 * Math.PI);
          if (obj.role !== "static") ctx.fill(); ctx.stroke();
        } else if (item.kind === "rigid-capsule") {
          const moving = item.obj.role !== "static";
          capsulePath(ctx, project(item.a), project(item.b), item.obj.radius * scale);
          ctx.strokeStyle = moving ? "#214F50" : "#AAA399"; ctx.lineWidth = 1; ctx.fillStyle = item.obj.color || "#C5A059";
          if (moving) ctx.fill(); ctx.stroke();
        } else {
          const pa = project(item.seg.a), pb = project(item.seg.b), w = Math.max(2, 2 * item.seg.radius * scale);
          ctx.lineCap = "round"; ctx.strokeStyle = "#214F50"; ctx.lineWidth = w + 2;
          ctx.beginPath(); ctx.moveTo(...pa); ctx.lineTo(...pb); ctx.stroke();
          ctx.strokeStyle = controlledId === "human" ? "#C6A58A" : "#9BB3B0"; ctx.lineWidth = w; ctx.stroke();
          ctx.lineCap = "butt";
        }
      }
      if (showMuscles) {
        for (const m of state.human?.muscles || []) {
          const pa = project(m.a), pb = project(m.b), act = m.activation;
          ctx.strokeStyle = `rgb(${Math.round(150 + 55 * act)},${Math.round(92 - 30 * act)},${Math.round(72 - 20 * act)})`;
          ctx.lineWidth = 1.5 + 2.5 * act; ctx.beginPath(); ctx.moveTo(...pa); ctx.lineTo(...pb); ctx.stroke();
        }
      }
      if (controlledId !== "human" && !geometry.error) {
        const controlled = state.objects.find((item) => item.id === controlledId);
        if (controlled) {
          const model = scene.objects.find((item) => item.id === controlled.id) || controlled;
          const [cx, cyPos] = project(controlled.position);
          const extent = (geometry.extents.get(controlled.id) ?? (model.radius || Math.max(...(model.size || [0.05])) / 2)) * scale + 5;
          ctx.beginPath(); ctx.strokeStyle = "#965644"; ctx.lineWidth = 2; ctx.arc(cx, cyPos, extent, 0, 2 * Math.PI); ctx.stroke();
        }
      }
      const target = state.target || scene.target;
      if (target && !sharedExercise() && !staticSupport()) {
        const [tx, ty] = project(target);
        ctx.beginPath(); ctx.strokeStyle = "#965644"; ctx.setLineDash([6, 4]); ctx.lineWidth = 2;
        ctx.arc(tx, ty, Math.max(3, goalTol * scale), 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
      }
      for (const c of state.contact_details || []) {
        const [x, y] = project(c.position), coupled = c.kind === "human_object";
        ctx.beginPath(); ctx.fillStyle = coupled ? "#C7795D" : "#965644";
        ctx.arc(x, y, coupled ? 4.5 : 2.5, 0, 2 * Math.PI); ctx.fill();
        if (coupled) { ctx.strokeStyle = "#214F50"; ctx.lineWidth = 1; ctx.stroke(); }
      }
      ctx.restore();
    });
  }
  const supportSegments = new Set(["torso", "head", "pelvis", "shoulders", "left_upper_arm", "left_forearm", "left_thigh", "left_shin", "right_upper_arm", "right_forearm", "right_thigh", "right_shin"]);
  const supportRequirementFields = ["object_id", "human_segment", "direction_world", "minimum_load_body_weight_fraction", "start_step", "end_step"];
  const supportDigest = value => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
  const nonnegative = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
  const nullableNonnegative = value => value === null || nonnegative(value);
  const feetIdentityKeys = ["scene_sha256", "base_scene_sha256", "base_task_sha256", "projection_contract_sha256", "adapter_implementation_sha256", "foot_reference_sha256"];
  const sameArray = (value, expected) => Array.isArray(value) && value.length === expected.length && value.every((item, i) => item === expected[i]);
  const cameraPins = ["scene_sha256", "camera_coupling_sha256", "source_sha256", "body_sha256", "source_clock_sha256",
    "metric_context_sha256", "kernel_evaluator_sha256", "adapter_implementation_sha256", "evaluator_sha256"];
  // Presentation joins are not a substitute for the server's sealed task/state validation.
  function validCameraTask(world, task, identity) {
    const clock = world?.camera_coupling?.clock, context = task?.metric_context, scope = task?.scope;
    const empty = value => Array.isArray(value) && value.length === 0;
    return world?.profile === "shared-world-v6" && task?.schema_version === 1 && task.profile === "camera-shared-task-v1"
      && identity && cameraPins.every(key => supportDigest(task[key]) && identity[key] === task[key]) && supportDigest(identity.task_sha256)
      && Array.isArray(world.bounds) && world.bounds.length === 3 && world.bounds.every(value => Number.isFinite(value) && value > 0)
      && empty(world.objects) && world.target === null && world.task?.object_goal === null && world.learned_policy_available === false
      && world.foot_extension === undefined && task.base_task === undefined && task.equipment_ids === undefined && task.contact_requirements === undefined
      && world.action_contract?.human_motor_count === 22 && world.action_contract.object_force_channels === 0
      && empty(world.action_contract.dynamic_object_ids) && Array.isArray(world.action_contract.names) && world.action_contract.names.length === 22
      && new Set(world.action_contract.names).size === 22 && world.action_contract.names.every(name => typeof name === "string" && name.length > 0 && name.length < 160)
      && ["disabled", "explicit_3_axis"].includes(task.control?.root_assistance) && task.control.object_forces === "forbidden"
      && Number.isSafeInteger(task.interval?.horizon_steps) && task.interval.horizon_steps > 0 && task.interval.horizon_steps <= 10000
      && task.interval.horizon_steps === world.task.horizon_steps && task.interval.start_offset_s === world.task.start_offset_s
      && task.interval.start_offset_s === 0 && nonnegative(clock?.source_time_offset_s)
      && Number.isFinite(world.dt) && world.dt > 0 && world.dt === clock.control_dt_s
      && Number.isFinite(clock.source_fps) && clock.source_fps > 0 && Math.abs(world.dt * clock.source_fps - 1) < 1e-8
      && Number.isInteger(world.substeps) && world.substeps >= 8 && world.substeps <= 64
      && ["source_sha256", "body_sha256", "source_clock_sha256"].every(key => task[key] === world.camera_coupling[key])
      && context?.profile === "camera-shared-metric-context-v1" && context.scene_sha256 === task.scene_sha256
      && context.valid_legacy_task === false && context.kernel_branch === "shared-exercise-task-v2"
      && context.kernel === "weft.shared_exercise_task._ExerciseMetrics" && context.projection === "identity_22_motor_samples_without_equipment"
      && context.control_dt_s === world.dt && context.source_body_dt_s === world.human_spec?.dt && context.source_body_dt_s > 0
      && context.substeps === world.substeps && supportDigest(context.kernel_arguments_sha256)
      && context.kernel_arguments?.profile === "shared-exercise-task-v2" && empty(context.kernel_arguments.equipment_ids) && empty(context.kernel_arguments.contact_requirements)
      && context.kernel_arguments.interval?.start_offset_s === task.interval.start_offset_s && context.kernel_arguments.interval?.horizon_steps === task.interval.horizon_steps
      && context.kernel_arguments.control?.root_assistance === task.control.root_assistance && context.kernel_arguments.control.object_forces === "forbidden"
      && scope && Object.keys(scope).every(key => context.kernel_arguments.source_scope?.[key] === scope[key])
      && task.gate_profile === "shared-exercise-gates-v1" && supportDigest(task.gates_sha256)
      && scope?.success_scope === "authored_simulated_camera_tracking_criteria" && scope.physical_validation === "unverified"
      && scope.source_contact_verified === false && scope.ground_support_verified === false && scope.equipment_scoring === "unsupported"
      && scope.foot_controls === "unsupported" && scope.learning === "unqualified"
      && scope.clock_alignment === "original_source_grid_with_unchanged_body_period";
  }
  function validCameraFrame(world, task, identity, geometry, m, status) {
    const empty = value => Array.isArray(value) && value.length === 0, count = value => Number.isSafeInteger(value) && value >= 0;
    const close = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-8;
    const maximum = task.interval.horizon_steps * world.substeps, tick = world.dt / world.substeps;
    const vector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
    const line = value => value && vector(value.a) && vector(value.b);
    if (geometry?.profile !== "shared-world-v6" || geometry.camera_coupling_sha256 !== identity.camera_coupling_sha256
      || geometry.controlled_id !== "human" || geometry.human?.reference === true || geometry.target !== null || !empty(geometry.objects)
      || !sameArray(geometry.human?.action_names, world.action_contract.names) || !nonnegative(geometry.time_s)
      || !Array.isArray(geometry.human.segments) || geometry.human.segments.length === 0 || geometry.human.segments.length > 32
      || !geometry.human.segments.every(value => line(value) && nonnegative(value.radius))
      || !Array.isArray(geometry.human.muscles) || geometry.human.muscles.length > 38
      || !geometry.human.muscles.every(value => line(value) && nonnegative(value.activation) && value.activation <= 1)
      || !Array.isArray(geometry.contact_details) || geometry.contact_details.length > 256
      || !geometry.contact_details.every(value => value && vector(value.position))
      || !close(geometry.source_time_s, world.camera_coupling.clock.source_time_offset_s + geometry.time_s)
      || !Array.isArray(geometry.normalized_action) || geometry.normalized_action.length !== 22
      || !geometry.normalized_action.every(value => Number.isFinite(value) && Math.abs(value) <= 1)
      || task.control.root_assistance === "disabled" && geometry.normalized_action.slice(19).some(value => value !== 0)
      || !["current", "last_valid_control_boundary"].includes(status)
      || m?.profile !== "camera-shared-metrics-v1" || m.kernel_branch !== "shared-exercise-task-v2" || m.task_sha256 !== identity.task_sha256
      || !cameraPins.filter(key => key !== "source_sha256").every(key => m[key] === identity[key])
      || m.gate_profile !== task.gate_profile || m.gates_sha256 !== task.gates_sha256
      || m.reward_profile !== "human_tracking_effort_v1" || m.root_assistance !== task.control.root_assistance || m.virtual_object_forces !== "forbidden"
      || m.gate_applicability?.human_tracking !== "applied" || m.gate_applicability.equipment_stability !== "not_applicable"
      || m.gate_applicability.contact_requirements !== "not_applicable" || !empty(m.equipment) || !empty(m.contact_requirements)
      || !Object.keys(task.scope).every(key => m.source_scope?.[key] === task.scope[key])
      || typeof m.success !== "boolean" || typeof m.completed_interval !== "boolean" || !(m.failure === null || typeof m.failure === "string" && m.failure.length <= 240)
      || !Array.isArray(m.failed_gates) || m.failed_gates.length > 40 || !m.failed_gates.every(value => typeof value === "string" && value.length < 200)
      || !count(m.observed_substeps) || m.observed_substeps > maximum || !count(m.valid_substeps) || m.valid_substeps > m.observed_substeps
      || !["simulated_duration_s", "valid_observed_duration_s", "declared_duration_s", "peak_angular_rmse_rad", "peak_root_rmse_model_units",
        "max_floor_penetration_model_units", "joint_effort_mean_squared", "root_assistance_mean_squared"].every(key => nonnegative(m[key]))
      || !close(m.declared_duration_s, task.interval.horizon_steps * world.dt) || !close(m.valid_observed_duration_s, m.valid_substeps * tick)
      || m.simulated_duration_s < m.valid_observed_duration_s - 1e-8 || m.simulated_duration_s > m.observed_substeps * tick + 1e-8
      || !close(m.observed_source_time_s, world.camera_coupling.clock.source_time_offset_s + m.simulated_duration_s)
      || !["observed", "invalid_substep_clock", "unavailable_after_last_observed_time"].includes(m.clock_status)
      || m.clock_status !== "observed" && m.failure === null || status === "last_valid_control_boundary" && (m.failure === null || geometry.time_s > m.simulated_duration_s + 1e-8)
      || status === "current" && !close(geometry.time_s, m.simulated_duration_s)
      || !["angular_rmse_rad", "root_rmse_model_units"].every(key => m.valid_substeps ? nonnegative(m[key]) && m[key] <= m[`peak_${key}`] + 1e-8 : m[key] === null)
      || !["reward_integral", "mean_reward", "failure_reward_penalty"].every(key => Number.isFinite(m[key]))
      || m.joint_effort_mean_squared > 1 + 1e-8 || m.root_assistance_mean_squared > 1 + 1e-8
      || task.control.root_assistance === "disabled" && m.root_assistance_mean_squared !== 0
      || m.completed_interval && (m.valid_substeps !== maximum || m.observed_substeps !== maximum || m.failure !== null)
      || m.success && (!m.completed_interval || m.failed_gates.length || m.failure !== null)) return false;
    return true;
  }
  function validCameraEnvelope(data) {
    const count = exerciseTask?.control?.root_assistance === "explicit_3_axis" ? 22 : 19, capability = data.capabilities;
    const modes = capability?.controllers;
    const policyReady = modes?.learned_policy_available === true && modes.trained_target_id === "human"
      && validCameraPolicyView(modes.camera_policy_admission);
    const noPolicy = modes?.learned_policy_available === false && modes.trained_target_id === null
      && modes.camera_policy_admission == null;
    return validCameraTask(scene, exerciseTask, exerciseIdentity) && data.control_profile === "shared_exercise"
      && capability?.scene_profile === scene.profile && capability.task_contract?.task_type === exerciseTask.profile
      && (policyReady || noPolicy) && !(modes.fitted_reference_controller === true && policyReady)
      && capability.controllers.manual === true && capability.controllers.coast === true && capability.controllers.authored_reference_controller === true
      && sameArray(exerciseActions?.names, scene.action_contract.names.slice(0, count))
      && sameArray(exerciseActions.joint_channels, Array.from({length: 19}, (_, i) => i))
      && sameArray(exerciseActions.root_assistance_channels, count === 22 ? [19, 20, 21] : [])
      && capability.action_contract?.dimension === count && sameArray(capability.action_contract.actuator_names, exerciseActions.names)
      && sameArray(capability.action_contract.object_force_channels, []);
  }
  function validCameraPolicyView(view) {
    const summary = view?.summary;
    return view?.profile === "camera-shared-policy-admission-v1" && view.status === "admitted"
      && view.privileged_simulator_features === true
      && cameraPins.every(key => supportDigest(view[key]) && view[key] === exerciseIdentity?.[key])
      && view.task_sha256 === exerciseIdentity?.task_sha256
      && ["admission_sha256", "policy_sha256", "training_manifest_sha256", "compatibility_sha256",
        "original_runtime_identity_sha256", "execution_runtime_identity_sha256"].every(key => supportDigest(view[key]))
      && (view.evaluation_manifest_sha256 === null || supportDigest(view.evaluation_manifest_sha256))
      && Number.isSafeInteger(summary?.nonzero_coefficients) && summary.nonzero_coefficients >= 0 && summary.nonzero_coefficients <= 12
      && summary.selected_feedback_kind === (summary.nonzero_coefficients ? "searched_phase_feedback_residuals" : "authored_unchanged")
      && summary.root_assistance === exerciseTask?.control?.root_assistance && summary.learning_qualification === "unqualified_research";
  }
  function validCameraState(state) {
    const count = exerciseActions.names.length;
    return state.schema_version === 1 && cameraPins.every(key => state[key] === exerciseIdentity[key]) && state.task_sha256 === exerciseIdentity.task_sha256
      && state.terminated === Boolean(state.task_metrics?.failure || state.task_metrics?.success)
      && state.truncated === Boolean(state.task_metrics?.completed_interval && !state.task_metrics?.success) && Array.isArray(state.last_action) && state.last_action.length === count
      && state.last_action.every(value => Number.isFinite(value) && Math.abs(value) <= 1)
      && (state.engine_state_status !== "current" || sameArray(state.engine_state?.normalized_action, [...state.last_action, ...(count === 19 ? [0, 0, 0] : [])]))
      && validCameraFrame(scene, exerciseTask, exerciseIdentity, state.engine_state, state.task_metrics, state.engine_state_status);
  }
  function validFeetEnvelope() {
    const task = exerciseEnvelope, identity = exerciseIdentity, assisted = exerciseTask?.control?.root_assistance === "explicit_3_axis";
    const indices = [...Array.from({length: 19}, (_, i) => i), ...Array.from({length: 6}, (_, i) => i + 22), ...(assisted ? [19, 20, 21] : [])];
    return task?.schema_version === 1 && task.profile === "shared-exercise-feet-task-v1"
      && ["shared-world-v2", "shared-world-v3", "shared-world-v4"].includes(scene.base_profile)
      && identity && feetIdentityKeys.every(key => supportDigest(task[key]) && identity[key] === task[key]) && supportDigest(identity.task_sha256)
      && task.projection_profile === "shared-exercise-feet-projection-v1" && identity.projection_profile === task.projection_profile
      && task.base_scene_sha256 === exerciseTask?.scene_sha256 && task.foot_reference_sha256 === scene.foot_extension?.reference_sha256
      && task.control?.root_assistance === exerciseTask.control.root_assistance && task.control.object_forces === "forbidden"
      && task.control.ankle_target_mode === "authored_initial_angle_hold" && exerciseActions?.ankle_target_mode === task.control.ankle_target_mode
      && task.interval?.horizon_steps === exerciseTask.interval?.horizon_steps
      && task.scope?.success_scope === "unchanged_base_task_simulated_criteria" && task.scope.ankle_metrics === "unscored_diagnostics"
      && task.scope.foot_support_credit === false && task.scope.ground_load_requirements === "unsupported"
      && task.scope.learning === "unsupported" && task.scope.physical_validation === "unverified"
      && sameArray(exerciseActions.engine_motor_indices, indices)
      && sameArray(exerciseActions.names, indices.map(i => scene.action_contract?.names?.[i]))
      && sameArray(exerciseActions.joint_channels, Array.from({length: 25}, (_, i) => i))
      && sameArray(exerciseActions.body_joint_channels, Array.from({length: 19}, (_, i) => i))
      && sameArray(exerciseActions.ankle_channels, [19, 20, 21, 22, 23, 24])
      && sameArray(exerciseActions.root_assistance_channels, assisted ? [25, 26, 27] : []);
  }
  function validFeetDiagnostics(d, m) {
    const count = value => Number.isSafeInteger(value) && value >= 0;
    if (d?.profile !== "shared-exercise-feet-diagnostics-v1" || d.source !== "simulated_engine_substeps"
      || d.target_mode !== "authored_initial_angle_hold" || d.scored !== false || d.foot_support_credit !== false
      || d.ground_load_requirements !== "unsupported" || d.retention !== "latest_observed_substep_plus_episode_aggregates"
      || !count(d.observed_substeps) || d.observed_substeps !== m.observed_substeps || !count(d.valid_substeps)
      || d.valid_substeps !== m.valid_substeps || d.valid_substeps > d.observed_substeps
      || d.observed_substeps > exerciseTask.interval.horizon_steps * scene.substeps
      || !nonnegative(d.valid_observed_duration_s) || d.valid_observed_duration_s !== m.valid_observed_duration_s
      || !count(d.excluded_foot_contact_records) || !Array.isArray(d.contact_aggregates)
      || d.contact_aggregates.length !== scene.objects.length * 2 || d.contact_aggregates.length > 24) return false;
    const errors = [d.ankle_hold_error_rmse_rad, d.peak_ankle_hold_error_rmse_rad, d.ankle_effort_mean_squared];
    if (!(d.valid_substeps ? errors.every(nonnegative) && errors[0] <= errors[1] + 1e-8 : errors.every(value => value === null))) return false;
    if (!d.contact_aggregates.every((row, i) => row?.object_id === scene.objects[Math.floor(i / 2)].id
      && row.human_segment === (i % 2 ? "right_foot" : "left_foot") && count(row.contact_records)
      && (row.contact_records ? Number.isFinite(row.first_contact_time_s) && Number.isFinite(row.last_contact_time_s) && nonnegative(row.peak_normal_force)
        : [row.first_contact_time_s, row.last_contact_time_s, row.peak_normal_force].every(value => value === null)))
      || d.contact_aggregates.reduce((sum, row) => sum + row.contact_records, 0) !== d.excluded_foot_contact_records) return false;
    const latest = d.latest_substep;
    return d.observed_substeps === 0 ? latest === null : latest && Number.isSafeInteger(latest.step_index)
      && Number.isSafeInteger(latest.substep_index) && Number.isFinite(latest.interval_start_s) && Number.isFinite(latest.contact_time_s)
      && (latest.interval_end_s === null || Number.isFinite(latest.interval_end_s)) && typeof latest.valid === "boolean"
      && (latest.failure === null || typeof latest.failure === "string" && latest.failure.length <= 240)
      && Array.isArray(latest.excluded_foot_contacts) && latest.excluded_foot_contacts.length <= 256
      && ((!latest.valid || latest.failure !== null || latest.interval_end_s === null) ? !!m.failure : true);
  }
  function validFeetState(state) {
    return feetIdentityKeys.every(key => state[key] === exerciseIdentity[key]) && state.task_sha256 === exerciseIdentity.task_sha256
      && state.projection_profile === exerciseEnvelope.projection_profile
      && state.engine_state?.profile === scene.profile
      && state.engine_state.human?.feet?.reference_sha256 === exerciseIdentity.foot_reference_sha256
      && state.task_metrics?.scene_sha256 === exerciseIdentity.base_scene_sha256 && state.task_metrics.task_sha256 === exerciseIdentity.base_task_sha256
      && sameArray(state.engine_state.human?.action_names, scene.action_contract.names.slice(0, 28))
      && validFeetDiagnostics(state.ankle_diagnostics, state.task_metrics);
  }
  function configureFeetDiagnostics() {
    if (feetPanel) return;
    feetPanel = document.createElement("section");
    const heading = document.createElement("h3"), summary = document.createElement("summary");
    text(heading, "Unscored ankles and foot contacts"); text(summary, "Episode foot/object contact aggregates");
    feetSummary = document.createElement("p"); feetCoverage = document.createElement("p");
    feetDetails = document.createElement("details"); feetRows = document.createElement("pre");
    feetDetails.append(summary, feetRows); feetPanel.append(heading, feetSummary, feetCoverage, feetDetails);
    $("exercise-results").append(feetPanel);
    feetDetails.ontoggle = () => updateFeetDiagnostics(snapshot);
  }
  function updateFeetDiagnostics(state) {
    if (!feetPanel) return;
    if (!exerciseStateReady) {
      text(feetSummary, "Ankle diagnostics unavailable; no zero error or contact is inferred.");
      text(feetCoverage, "Controls paused because the task/state contract is inconsistent."); text(feetRows, "Unavailable."); feetDetailSignature = ""; return;
    }
    const d = state.ankle_diagnostics, latest = d.latest_substep;
    const unchangedText = (element, value) => { if (element.textContent !== value) text(element, value); };
    unchangedText(feetSummary, `Original-start ankle hold · error mean ${number(d.ankle_hold_error_rmse_rad)} / peak ${number(d.peak_ankle_hold_error_rmse_rad)} rad · mean squared normalized ankle command ${number(d.ankle_effort_mean_squared)}. Unscored simulated readings; no foot support credit or ground-load qualification.`);
    unchangedText(feetCoverage, `${d.valid_substeps} accepted / ${d.observed_substeps} observed substeps · ${number(d.valid_observed_duration_s)}s accepted coverage. ${latest
      ? `Latest contact clock ${number(latest.contact_time_s)}s; sample ${number(latest.interval_start_s)}–${number(latest.interval_end_s)}s (${latest.valid && latest.failure === null ? "valid engine sample" : "failed/invalid engine sample"}).`
      : "No substep observed yet."} Geometry clock ${number(state.engine_state.time_s)}s${state.engine_state_status === "last_valid_control_boundary" ? " (last valid boundary)" : ""}. ${d.excluded_foot_contact_records} excluded foot/object contact records; only the latest sample and episode aggregates are retained, not a full trace.`);
    if (!feetDetails.open) return;
    const signature = JSON.stringify(d.contact_aggregates);
    if (signature === feetDetailSignature) return;
    feetDetailSignature = signature;
    text(feetRows, d.contact_aggregates.length ? d.contact_aggregates.map(row => `${row.human_segment.replaceAll("_", " ")} / ${actorLabel(row.object_id)}: ${row.contact_records} records · first ${number(row.first_contact_time_s)}s · last ${number(row.last_contact_time_s)}s · peak normal force ${number(row.peak_normal_force)} model force units`).join("\n")
      : "No objects are declared. Ground-load readings are unsupported; this is not evidence of absent foot/ground contact.");
  }
  const supportDirection = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite) && Math.abs(Math.hypot(...value) - 1) <= 1e-9;
  const supportVector = (value, length) => Array.isArray(value) && value.length === length && value.every(Number.isFinite);
  const validStaticSupports = objects => Array.isArray(objects) && objects.length > 0 && objects.length <= 12
    && objects.every(obj => obj?.role === "static" && typeof obj.id === "string" && supportVector(obj.position, 3)
      && supportVector(obj.quaternion, 4) && Math.abs(Math.hypot(...obj.quaternion) - 1) <= 1e-8)
    && new Set(objects.map(obj => obj.id)).size === objects.length;
  const supportObjectsMatch = objects => validStaticSupports(scene?.objects) && validStaticSupports(objects)
    && objects.length === scene.objects.length && objects.every(obj => scene.objects.some(source => source.id === obj.id
      && ["position", "quaternion"].every(key => obj[key].every((value, i) => Math.abs(value - source[key][i]) <= 1e-8))));
  function validSupportTask(task) {
    const scope = task?.source_scope, supports = scope?.supports, requirements = task?.support_requirements;
    return task?.schema_version === 1 && task.authored_addition === true && task.gate_profile === "shared-support-gates-v1"
      && task.reward_profile === "tracking-times-declared-support-v1" && supportDigest(task.scene_sha256) && supportDigest(task.gates_sha256)
      && task.contact_requirements === undefined && task.equipment_ids === undefined && scene.target === null && scene.task?.object_goal === null
      && scene.task?.type === "human_support_v1" && scene.task.authored_addition === true
      && validStaticSupports(scene.objects)
      && scope?.success_scope === "authored_simulated_support_task" && scope.physical_validation === "unverified"
      && scope.source_contact_verified === false && scope.grasp_supported === false
      && scope.geometry_scope === "declared_static_collision_geometry_not_recovered_contact" && scope.clock_alignment === "preserved_scene_clocks_not_retimed"
      && Array.isArray(supports) && supports.length === scene.objects.length && new Set(supports.map(row => row?.object_id)).size === supports.length
      && supports.every(row => typeof row?.object_id === "string" && row.object_id.length <= 128 && supportDigest(row.object_sha256)
        && scene.objects.some(obj => obj.id === row.object_id))
      && Array.isArray(requirements) && requirements.length > 0 && requirements.length <= 24
      && requirements.every(row => supports.some(obj => obj.object_id === row?.object_id) && supportSegments.has(row.human_segment)
        && supportDirection(row.direction_world) && nonnegative(row.minimum_load_body_weight_fraction)
        && row.minimum_load_body_weight_fraction > 0 && row.minimum_load_body_weight_fraction <= 2
        && Number.isInteger(row.start_step) && Number.isInteger(row.end_step) && 0 <= row.start_step
        && row.start_step < row.end_step && row.end_step <= task.interval?.horizon_steps);
  }
  function validSupportMetrics(m, geometry) {
    const rows = m?.support_requirements, required = exerciseTask.support_requirements;
    return m?.contact_requirements === undefined && m.equipment.length === 0
      && m.gate_applicability?.equipment_stability === "not_applicable" && m.gate_applicability?.declared_support === "required"
      && m.gate_profile === exerciseTask.gate_profile && m.gates_sha256 === exerciseTask.gates_sha256
      && m.reward_profile === exerciseTask.reward_profile && m.scene_sha256 === exerciseTask.scene_sha256 && supportDigest(m.task_sha256)
      && m.root_assistance === exerciseTask.control.root_assistance && m.virtual_object_forces === "forbidden"
      && m.source_scope?.physical_validation === "unverified" && m.source_scope?.source_contact_verified === false && m.source_scope?.grasp_supported === false
      && Array.isArray(m.source_scope?.supports) && m.source_scope.supports.length === exerciseTask.source_scope.supports.length
      && m.source_scope.supports.every((row, index) => row?.object_id === exerciseTask.source_scope.supports[index].object_id
        && row.object_sha256 === exerciseTask.source_scope.supports[index].object_sha256)
      && supportObjectsMatch(geometry.objects)
      && Array.isArray(m.failed_gates) && m.failed_gates.length <= 40 && m.failed_gates.every(value => typeof value === "string" && value.length < 200)
      && Array.isArray(rows) && rows.length === required.length && rows.every((row, index) => row && supportRequirementFields.every(key =>
        JSON.stringify(row[key]) === JSON.stringify(required[index][key]))
        && ["required_duration_s", "observed_duration_s", "qualifying_duration_s", "current_gap_s", "max_gap_s"].every(key => nonnegative(row[key]))
        && ["load_body_weight_fraction", "peak_load_body_weight_fraction", "qualifying_fraction"].every(key => nullableNonnegative(row[key]))
        && (row.qualifying_fraction === null || row.qualifying_fraction <= 1 + 1e-8)
        && ["active_now", "qualifying_now", "satisfied"].every(key => typeof row[key] === "boolean")
        && (!row.qualifying_now || row.active_now && row.load_body_weight_fraction !== null)
        && (!row.satisfied || row.qualifying_fraction !== null && Math.abs(row.observed_duration_s - row.required_duration_s) <= 1e-8))
      && (!m.success || m.completed_interval && !m.failure && m.failed_gates.length === 0 && rows.every(row => row.satisfied));
  }
  function configureSupportLabels() {
    text($("exercise-task-heading"), "Body tracking + authored static support");
    const details = $("exercise-metrics-details"), summary = details.querySelector?.("summary");
    if (summary) text(summary, "Body limits and declared support loads");
    const equipmentTable = $("exercise-equipment").closest?.(".table-wrap");
    if (equipmentTable) equipmentTable.hidden = true;
    const header = $("exercise-contacts").closest?.("table")?.querySelector("thead tr");
    if (header) {
      header.replaceChildren();
      for (const label of ["Body segment / static support", "Authored load requirement", "Observed coverage", "Unqualified gaps", "Current load / result"]) {
        const th = document.createElement("th"); th.setAttribute("scope", "col"); text(th, label); header.append(th);
      }
    }
    const hint = $("exercise-results").querySelector?.(".hint");
    if (hint) text(hint, "Loads come from simulated contacts, expressed as fractions of simulated body weight. Coverage and gaps use the declared windows; an early stop leaves the interval incomplete. These authored checks do not verify real contact or grasp.");
  }
  function configureExerciseTask(data) {
    exerciseEnvelope = data.exercise_task; exerciseTask = sharedFeet() ? exerciseEnvelope?.base_task : exerciseEnvelope;
    exerciseIdentity = data.capabilities?.task_contract?.identity; exerciseActions = data.action_contract;
    const admission = data.capabilities?.controllers?.camera_policy_admission;
    cameraPolicyView = cameraWorld() && validCameraPolicyView(admission) ? admission : null;
    const joints = sharedFeet() ? 25 : 19;
    const rootMode = exerciseTask?.control?.root_assistance, expected = rootMode === "disabled" ? joints : rootMode === "explicit_3_axis" ? joints + 3 : 0;
    const names = exerciseActions?.names;
    exerciseContractReady = (!sharedFeet() || validFeetEnvelope()) && (!cameraWorld() || validCameraEnvelope(data)) && exerciseTask?.profile === (cameraWorld() ? "camera-shared-task-v1" : staticSupport() ? "shared-support-task-v1" : personOnly() ? "shared-exercise-task-v2" : "shared-exercise-task-v1") && exerciseTask.control?.object_forces === "forbidden"
      && expected > 0 && Array.isArray(names) && names.length === expected && new Set(names).size === expected
      && names.every(name => typeof name === "string" && name.length > 0 && name.length < 160)
      && exerciseActions.root_assistance === rootMode && Array.isArray(exerciseActions.object_force_channels) && exerciseActions.object_force_channels.length === 0
      && Number.isInteger(exerciseTask.interval?.horizon_steps) && exerciseTask.interval.horizon_steps > 0
      && (cameraWorld() || (staticSupport() ? validSupportTask(exerciseTask) : Array.isArray(exerciseTask.contact_requirements) && exerciseTask.contact_requirements.length <= 24))
      && (!personOnly() || (Array.isArray(exerciseTask.equipment_ids) && exerciseTask.equipment_ids.length === 0
        && exerciseTask.contact_requirements.length === 0 && scene.objects.length === 0 && scene.target === null));
    $("exercise-task-card").hidden = $("exercise-results").hidden = $("mode-coast").hidden = false;
    if (staticSupport()) configureSupportLabels();
    const duration = exerciseTask?.interval?.horizon_steps * scene.dt;
    text($("exercise-task-scope"), exerciseContractReady
      ? staticSupport() ? `Track the body reference and meet the authored segment/load requirements over ${number(duration, 2)} seconds (${exerciseTask.interval.horizon_steps} control steps). Supports stay fixed; equipment-drift scoring does not apply. This is a simulated task, not recovered physical contact.`
        : personOnly() ? `Track the body reference over ${number(duration, 2)} seconds (${exerciseTask.interval.horizon_steps} control steps). Only human and ground are modeled; equipment and contact-requirement checks do not apply. Unsampled source motion remains unknown.`
        : `Track the body reference while every equipment object stays stable over ${number(duration, 2)} seconds (${exerciseTask.interval.horizon_steps} control steps). This declared task interval is not necessarily the whole source video.`
      : "The saved human-only task/control contract is unavailable or inconsistent. Task controls are disabled; no legacy task is substituted.");
    text($("exercise-assistance"), rootMode === "disabled" ? `Root assistance: OFF · ${joints} joint motors${sharedFeet() ? " (19 body + 6 ankle)" : ""}. The body has no virtual root-force controls.`
      : rootMode === "explicit_3_axis" ? `Root assistance: 3 explicit virtual controls · ${joints} joint motors${sharedFeet() ? " (19 body + 6 ankle)" : ""} + 3 root-force channels. This is not unassisted balance.`
      : "Root assistance: unavailable.");
    const requirements = staticSupport() ? exerciseTask?.support_requirements : exerciseTask?.contact_requirements;
    text($("exercise-contact-scope"), staticSupport() ? exerciseContractReady
      ? `${requirements.length} authored support requirement(s). Each names a body segment, static object, world-space load direction and minimum fraction of simulated body weight. Only these declared load windows count; contact dots alone do not qualify.`
      : "Static-support requirements unavailable; no contact or load is inferred."
      : Array.isArray(requirements) ? requirements.length
      ? `${requirements.length} declared wrist-region requirements. Only these checks count; other body contacts do not establish hand support or grasp.`
      : personOnly() ? "Equipment and wrist-region requirements: not applicable. This declared inventory is not verified object absence. Passing checks simulated body tracking, not real exercise completion."
        : "No wrist-region contact is required. Passing means body tracking and equipment stability only—not hand support, a completed exercise or grasp."
      : "Contact requirements are unavailable; none are inferred from the image.");
    if (cameraWorld()) {
      text($("exercise-task-heading"), "Camera-conditioned body task");
      text($("exercise-task-scope"), exerciseContractReady
        ? `Track the body reference over ${number(duration, 2)} seconds (${exerciseTask.interval.horizon_steps} source-clock control steps). Camera scale and ground are supplied priors; unchanged body tracking/floor limits are simulated criteria, not physical validation.`
        : "Camera task identities or controls are inconsistent. Controls are paused; no legacy task is substituted.");
      text($("exercise-contact-scope"), "Feet and ground/equipment contact are unqualified. Equipment and contact requirements add no reward or success credit. The empty simulated inventory is not evidence that the video contains no objects.");
      $("exercise-equipment").closest("div").hidden = true;
      $("exercise-contacts").closest("div").hidden = true;
    }
    if (sharedFeet()) {
      configureFeetDiagnostics();
      text($("exercise-task-heading"), "Original body task + unscored ankle controls");
      text($("exercise-task-scope"), $("exercise-task-scope").textContent + " The original 19-joint score is unchanged. Six ankle motors hold authored original-start angles, not later video foot poses. Foot contacts receive no support credit.");
    }
  }
  const exerciseFeedbackLabels = {coast_unchanged: "Selected coast baseline", authored_unchanged: "Selected authored baseline", searched_gains: "Searched human feedback", searched_joint_residuals: "Searched joint corrections", searched_feedback_residuals: "Searched joint feedback", searched_phase_feedback_residuals: "Experimental trained controller"};
  const exercisePolicyKinds = {"shared-exercise-policy-v1": ["coast_unchanged", "authored_unchanged", "searched_gains"],
    "shared-support-policy-v1": ["authored_unchanged", "searched_joint_residuals"],
    "shared-exercise-policy-v2": ["authored_unchanged", "searched_joint_residuals"],
    "shared-exercise-policy-v3": ["authored_unchanged", "searched_joint_residuals"],
    "shared-exercise-policy-v4": ["authored_unchanged", "searched_feedback_residuals"],
    "shared-exercise-policy-v5": ["authored_unchanged", "searched_feedback_residuals"]};
  function exerciseFeedbackLabel() {
    return exercisePolicy?.profile === "shared-exercise-policy-v5" && exercisePolicy.selected_feedback_kind === "searched_feedback_residuals"
      ? "Separate arm/leg feedback" : exerciseFeedbackLabels[exercisePolicy?.selected_feedback_kind];
  }
  function configureExercisePolicy(data) {
    const policy = data.exercise_policy;
    if (cameraWorld()) {
      const view = cameraPolicyView;
      exercisePolicy = data.learned_policy_available === true && view && policy?.profile === "camera-shared-policy-v1"
        && policy.admission_profile === view.profile && policy.admission_sha256 === view.admission_sha256
        && policy.policy_sha256 === view.policy_sha256 && policy.selected_feedback_kind === view.summary.selected_feedback_kind
        && policy.root_assistance === view.summary.root_assistance && policy.learning_qualification === "unqualified_research"
        && policy.privileged_simulator_features === true && Number.isFinite(policy.training_score)
        && typeof policy.training_score_improved_over_authored === "boolean"
        && ["evaluated", "not_attached"].includes(policy.evaluation_status)
        && (policy.evaluation_status === "evaluated") === (view.evaluation_manifest_sha256 !== null) ? policy : null;
      return exerciseContractReady && exercisePolicy !== null;
    }
    exercisePolicy = !sharedFeet() && !cameraWorld() && data.learned_policy_available === true && typeof policy?.profile === "string"
      && Object.hasOwn(exercisePolicyKinds, policy.profile)
      && (staticSupport() ? policy.profile === "shared-support-policy-v1" : policy.profile !== "shared-support-policy-v1")
      && typeof policy.policy_sha256 === "string" && /^[0-9a-f]{64}$/.test(policy.policy_sha256)
      && typeof policy.selected_feedback_kind === "string" && Object.hasOwn(exerciseFeedbackLabels, policy.selected_feedback_kind)
      && exercisePolicyKinds[policy.profile].includes(policy.selected_feedback_kind)
      && (!["shared-exercise-policy-v3", "shared-exercise-policy-v4", "shared-exercise-policy-v5"].includes(policy.profile) || typeof policy.training_feasible === "boolean")
      && policy.privileged_simulator_features === true ? policy : null;
    return exerciseContractReady && exercisePolicy !== null;
  }
  const fittedReferencePins = ["reference_controller_sha256", "reference_trial_manifest_sha256", "reference_proposal_manifest_sha256"];
  const fittedReferenceFields = ["profile", ...fittedReferencePins, "kind", "learned", "root_assistance", "saved_trial_outcomes",
    "saved_replay_scope", "physical_validation", "scope"];
  function validFittedReference(value) {
    const outcomes = value?.saved_trial_outcomes;
    return value && Object.keys(value).length === fittedReferenceFields.length && fittedReferenceFields.every(key => Object.hasOwn(value, key))
      && value.profile === "camera-reference-lab-attachment-v1" && fittedReferencePins.every(key => supportDigest(value[key]))
      && value.kind === "source_fit_authored_controller" && value.learned === false && value.physical_validation === "unverified"
      && value.root_assistance === exerciseTask?.control?.root_assistance
      && typeof value.saved_replay_scope === "string" && value.saved_replay_scope.length <= 512
      && typeof value.scope === "string" && value.scope.length <= 1024
      && outcomes && Object.keys(outcomes).length === 2 && ["authored", "reference_proposal"].every(name => {
        const row = outcomes[name];
        return row && Object.keys(row).length === 3 && ["steps", "success", "mean_reward"].every(key => Object.hasOwn(row, key))
          && Number.isSafeInteger(row.steps) && row.steps > 0 && row.steps <= exerciseTask.interval.horizon_steps
          && typeof row.success === "boolean" && (row.mean_reward === null || Number.isFinite(row.mean_reward));
      });
  }
  function fittedReferenceKey(value) {
    return JSON.stringify([...fittedReferenceFields.filter(key => key !== "saved_trial_outcomes").map(key => value[key]),
      ...["authored", "reference_proposal"].flatMap(name => ["steps", "success", "mean_reward"].map(key => value.saved_trial_outcomes[name][key]))]);
  }
  function fittedDescription() {
    return "Fitted joint targets follow the attached angle proposal. They are not learned and are not necessarily better than original controls. "
      + (fittedReference?.root_assistance === "disabled" ? "Body-position assistance is disabled. " : "Body-position assistance stays programmed. ")
      + "The original body, floor and task stay unchanged; contact is not verified.";
  }
  function configureFittedReference(data) {
    fittedReference = null;
    if (!cameraWorld()) return;
    const view = data.capabilities?.fitted_reference, declared = data.capabilities?.controllers?.fitted_reference_controller;
    if (exerciseContractReady && declared === true && data.learned_policy_available === false && data.exercise_policy == null
      && !cameraPolicyView && data.capabilities.controllers.learned_policy_available === false
      && validFittedReference(view) && validFittedReference(data.exercise_reference)
      && fittedReferenceKey(view) === fittedReferenceKey(data.exercise_reference)) fittedReference = view;
    if (!fittedButton) {
      const controls = document.createElement("div"); controls.classList.add("controls");
      fittedButton = document.createElement("button"); fittedButton.id = "mode-fitted"; fittedButton.type = "button";
      fittedButton.setAttribute("aria-pressed", "false"); fittedButton.onclick = () => selectMode("fitted"); controls.append(fittedButton);
      fittedSummary = document.createElement("p"); fittedSummary.id = "fitted-reference-summary"; fittedSummary.classList.add("hint");
      fittedComparison = document.createElement("p"); fittedComparison.id = "fitted-reference-comparison"; fittedComparison.classList.add("hint");
      $("exercise-task-card").append(controls, fittedSummary, fittedComparison);
    }
    text($("reference-servo"), "Original joint targets");
    text(fittedButton, fittedReference ? "Fitted joint targets" : "Fitted targets unavailable");
    text(fittedSummary, fittedReference ? fittedDescription() : declared || view != null || data.exercise_reference != null
      ? "Fitted targets are unavailable: attachment details are missing or inconsistent. Other valid controls remain available."
      : "No fitted controller is attached to this lab. Original joint targets and manual controls remain separate from learning.");
    fittedComparison.hidden = !fittedReference;
    if (fittedReference) {
      const describe = (name, label) => { const row = fittedReference.saved_trial_outcomes[name];
        return `${label}: ${row.success ? "passed" : "not passed"}, mean reward ${number(row.mean_reward)} over ${row.steps} steps`; };
      text(fittedComparison, `Saved comparison, not this trial — ${describe("authored", "Original")}; ${describe("reference_proposal", "fitted")}. A fitted result is not necessarily an improvement.`);
    }
  }
  function exerciseFeedbackDescription() {
    if (!exercisePolicy || !learnedPolicyAvailable) return "No validated human-only policy is available.";
    const kind = exercisePolicy.selected_feedback_kind;
    if (cameraWorld()) return `${kind === "authored_unchanged" ? "The search selected unchanged programmed feedback; it did not learn new joint corrections."
      : "The experimental trained controller makes small corrections to the 19 joint commands."} It uses the simulated body and known motion reference, not video pixels. ${exerciseTask.control.root_assistance === "disabled"
      ? "Body-position assistance is disabled." : "Body-position assistance remains programmed, not learned."} Ground/equipment contact is not verified. This does not establish learned balance or physical accuracy.`;
    if (exercisePolicy.profile === "shared-exercise-policy-v5" && kind === "searched_feedback_residuals") {
      return `${exerciseFeedbackLabel()}. Training searched separate torso, shoulder, elbow, hip and knee corrections using the same current position and velocity errors. Five groups use ten bounded feedback settings; each joint correction stays within ±0.25 of its normalized motor command. Any permitted root assistance remains programmed, not learned. Equipment has no force controls.`;
    }
    return `${exerciseFeedbackLabel()}. ${kind === "coast_unchanged" ? "Training selected the unchanged zero-command baseline, not new learned gains."
      : kind === "authored_unchanged" ? "Training selected unchanged programmed feedback, not new learned gains."
      : kind === "searched_joint_residuals" ? "Training searched small phase-dependent joint corrections on top of programmed feedback. Any permitted root assistance remains programmed, not learned."
      : kind === "searched_feedback_residuals" ? "Training searched small joint corrections using current position and velocity errors on top of programmed feedback. Any permitted root assistance remains programmed, not learned."
      : "Training searched feedback gains using simulator state and the known body reference."} Equipment has no force controls; root assistance remains the selected task setting.`;
  }
  function exerciseResetDescription() {
    const reset = exercisePolicy?.evaluation_reset_protocol;
    const values = [reset?.human_jitter_scale, reset?.object_position_jitter_world, reset?.object_velocity_jitter_world_s];
    if (!values.every(value => typeof value === "number" && Number.isFinite(value) && value >= 0)) return "Reset protocol unspecified in this saved comparison.";
    if (cameraWorld() && reset.profile === "shared-exercise-human-jitter-v1"
      && reset.equipment_reset_state === "unchanged_scene_pose_and_velocity"
      && reset.object_position_jitter_world === 0 && reset.object_velocity_jitter_world_s === 0) {
      return `Changed starts: human-only perturbations (scale ${number(reset.human_jitter_scale, 2)}); the supplied camera and ground priors remain fixed. This is not a new-video or changed-ground test.`;
    }
    if (staticSupport() && reset.profile === "shared-support-reset-v1"
      && reset.static_support_reset_state === "unchanged_scene_pose_and_velocity"
      && reset.object_position_jitter_world === 0 && reset.object_velocity_jitter_world_s === 0) {
      return `Changed starts: human-only perturbations (scale ${number(reset.human_jitter_scale, 2)}); static supports keep their exact scene poses. This is not a changed-support test.`;
    }
    if (reset.profile === "shared-exercise-human-jitter-v1" && reset.equipment_reset_state === "unchanged_scene_pose_and_velocity"
      && reset.object_position_jitter_world === 0 && reset.object_velocity_jitter_world_s === 0) {
      return personOnly() ? `Changed starts: human-only perturbations (scale ${number(reset.human_jitter_scale, 2)}); the ground remains unchanged. No equipment is declared.`
        : `Changed starts: human-only perturbations (scale ${number(reset.human_jitter_scale, 2)}); equipment keeps its exact scene pose and velocity. This is not a changed-equipment test.`;
    }
    if (reset.profile === "shared-exercise-reset-v1") return `Legacy reset: human perturbation scale ${number(reset.human_jitter_scale, 2)}, equipment position jitter ${number(reset.object_position_jitter_world)} model units and velocity jitter ${number(reset.object_velocity_jitter_world_s)} model units/s. Support-preserving starts are not established.`;
    return "Reset protocol unspecified in this saved comparison.";
  }
  function showExerciseLearning(data) {
    const available = learnedPolicyAvailable && exercisePolicy !== null;
    const feasibilityFirst = available && ["shared-exercise-policy-v3", "shared-exercise-policy-v4", "shared-exercise-policy-v5"].includes(exercisePolicy.profile);
    text($("policy"), available ? exerciseFeedbackLabel() : "Learned controller · unavailable");
    text($("start-learned"), available ? `Try ${exerciseFeedbackLabel().toLowerCase()}` : "Learned controller unavailable");
    text($("exercise-policy-summary"), exerciseFeedbackDescription());
    text($("learning-description"), available ? `${exerciseFeedbackDescription()} The policy interface uses privileged simulator-state and reference features, not video pixels.`
      : "No learned human-only exercise policy is available. The live task panel reports independent whole-interval results, not the legacy target task or a training score.");
    text($("training-caption"), feasibilityFirst
      ? `Training prioritized task limits before reward. ${exercisePolicy.training_feasible ? "Both training starts passed every task limit."
        : "At least one training start still failed a task limit."} Reward: ${number(exercisePolicy.training_score)}. Independent results are shown separately.`
      : available ? `Training selection score: ${number(exercisePolicy.training_score)}. ${exercisePolicy.training_score_improved_over_authored === true
      ? "The training score was above the authored baseline; this is not an independent evaluation result."
      : exercisePolicy.training_score_improved_over_authored === false ? "No higher training score than the authored baseline was reported."
      : "Training comparison unavailable."}`
      : "Compare manual, coast and authored feedback under the selected task. Inherited scene checks are development evidence, not exercise-task successes.");
    $("learning-curve").hidden = true;
    $("exercise-evaluation-summary").hidden = false;
    $("scores").replaceChildren();
    const scores = exercisePolicy?.evaluation_successes, names = ["coast", "random", "authored", "learned"];
    const evaluated = available && exercisePolicy.evaluation_status === "evaluated" && scores && names.every(name => {
      const row = scores[name];
      return Number.isInteger(row?.episodes) && row.episodes > 0 && row.episodes <= 6 && row.episodes === scores.coast?.episodes
        && Number.isInteger(row.successes) && row.successes >= 0 && row.successes <= row.episodes;
    });
    text($("exercise-evaluation-summary"), evaluated
      ? `Attached independent changed-start evaluation: matched starts, frozen policy and the same declared task interval. ${exerciseResetDescription()} Counts below are task trials passed, not physical or full-exercise validation.`
      : available && exercisePolicy.evaluation_status !== "not_attached"
        ? "Attached evaluation results unavailable or incomplete. No evaluation pass is inferred from training."
        : "Independent evaluation unavailable · no verified evaluation is attached. Training is not an evaluation pass.");
    if (evaluated) metrics($("scores"), names.map(name => [`${scores[name].successes}/${scores[name].episodes}`,
      `${name === "learned" ? exerciseFeedbackLabel() : {coast: "Coast baseline", random: "Random human actions", authored: "Authored feedback"}[name]} · declared-interval task trials passed`]));
    text($("learning-boundary"), "Changed starts reuse the same fixed source interval; they are not a new-video test. Simulator results do not verify grasp, anatomy, measured muscles, unassisted balance or physical accuracy. The live task panel reports this trial separately.");
    $("exercise-policy-links").replaceChildren();
    const policyFile = available && (data.artifacts || []).some(file => file.name === "policy.json");
    $("exercise-policy-links").hidden = !policyFile;
    if (policyFile) $("exercise-policy-links").append(link("Download attached policy", "run/policy.json"));
  }
  function exerciseOutcome(m) {
    if (!m) return "Task result unavailable";
    if (m.failure) return "Not passed · simulation failed";
    if (m.completed_interval) return m.success ? "Passed · declared interval complete" : "Not passed · interval complete with failed gates";
    return "In progress · task interval incomplete";
  }
  function exerciseDetails(m) {
    if (!$("exercise-metrics-details").open || !m) return;
    const signature = JSON.stringify([m.equipment, m.contact_requirements, m.support_requirements, m.failed_gates, m.gates_sha256]);
    if (signature === exerciseDetailSignature) return;
    exerciseDetailSignature = signature;
    text($("exercise-gates"), `Reported gate failures: ${m.failed_gates?.length ? m.failed_gates.join(", ") : "none reported so far"}. Gate identity: ${m.gates_sha256 || "unavailable"}. Passing still requires the complete declared interval.`);
    $("exercise-equipment").replaceChildren();
    if (cameraWorld()) { $("exercise-contacts").replaceChildren(); return; }
    if (personOnly()) {
      const row = document.createElement("tr"); cell(row, "Not applicable — no equipment declared"); $("exercise-equipment").append(row);
    }
    for (const item of m.equipment) {
      const row = document.createElement("tr");
      for (const value of [actorLabel(item.object_id), `${number(item.displacement_model_units)} model units`, `${number(item.orientation_rad)} rad`,
        `${number(item.speed_model_units_s)} model units/s`, `${number(item.angular_speed_rad_s)} rad/s`,
        item.within_limits === true ? "Yes, so far" : item.within_limits === false ? "No" : "Unavailable"]) cell(row, value);
      $("exercise-equipment").append(row);
    }
    $("exercise-contacts").replaceChildren();
    if (staticSupport()) {
      for (const item of m.support_requirements) {
        const row = document.createElement("tr"), name = cell(row, `${item.human_segment.replaceAll("_", " ")} / ${actorLabel(item.object_id)}`);
        const identity = document.createElement("details"), summary = document.createElement("summary"), hash = document.createElement("p");
        text(summary, "Declared support identity"); text(hash, exerciseTask.source_scope.supports.find(obj => obj.object_id === item.object_id).object_sha256);
        identity.append(summary, hash); name.append(identity);
        cell(row, `Direction [${item.direction_world.map(value => number(value, 2)).join(", ")}] in world axes; minimum ${number(item.minimum_load_body_weight_fraction * 100, 1)}% of simulated body weight · steps ${item.start_step}–${item.end_step} (end excluded)`);
        cell(row, `${item.qualifying_fraction === null ? "Qualifying fraction unavailable" : `${number(item.qualifying_fraction * 100, 1)}% qualifying`} · ${number(item.observed_duration_s)} / ${number(item.required_duration_s)}s observed`);
        cell(row, `Current ${number(item.current_gap_s)}s · longest ${number(item.max_gap_s)}s`);
        cell(row, `Current ${item.load_body_weight_fraction === null ? "unavailable" : `${number(item.load_body_weight_fraction * 100, 1)}%`} · peak ${item.peak_load_body_weight_fraction === null ? "unavailable" : `${number(item.peak_load_body_weight_fraction * 100, 1)}%`} of simulated body weight · ${item.active_now ? item.qualifying_now ? "qualifying now" : "active, not qualifying" : "outside active window / not observed"} · ${item.satisfied ? "Satisfied interval" : "Not yet satisfied / failed"}`);
        $("exercise-contacts").append(row);
      }
      return;
    }
    for (const item of m.contact_requirements) {
      const row = document.createElement("tr");
      const fraction = Number.isFinite(item.qualifying_fraction) ? `${number(item.qualifying_fraction * 100, 1)}% qualifying` : "Qualifying fraction unavailable";
      for (const value of [`${item.wrist} / ${actorLabel(item.object_id)}`, `${(item.kind || "unknown").replaceAll("_", " ")} · steps ${item.start_step}–${item.end_step} (end excluded)`,
        `${fraction} · ${number(item.observed_duration_s)} / ${number(item.required_duration_s)}s observed`,
        `${number(item.max_gap_s)}s`, item.satisfied === true ? "Satisfied" : "Not yet satisfied / failed"]) cell(row, value);
      $("exercise-contacts").append(row);
    }
    if (!m.contact_requirements.length) {
      const row = document.createElement("tr"); cell(row, personOnly() ? "Not applicable — human and ground only" : "No contact requirements in this task"); $("exercise-contacts").append(row);
    }
  }
  $("exercise-metrics-details").ontoggle = () => exerciseDetails(exerciseMetrics());
  function updateExerciseSnapshot(state) {
    snapshot = state;
    const m = state?.task_metrics, geometry = state?.engine_state;
    exerciseStateReady = exerciseContractReady && state?.profile === (cameraWorld() ? "camera-shared-state-v1" : sharedFeet() ? "shared-exercise-feet-state-v1" : "shared-exercise-state-v1") && !!geometry?.human && Array.isArray(geometry?.objects)
      && (!sharedFeet() || validFeetState(state)) && (!cameraWorld() || validCameraState(state))
      && ["current", "last_valid_control_boundary"].includes(state.engine_state_status)
      && typeof state.terminated === "boolean" && typeof state.truncated === "boolean"
      && m?.profile === (cameraWorld() ? "camera-shared-metrics-v1" : exerciseTask?.profile) && typeof m.completed_interval === "boolean" && typeof m.success === "boolean"
      && Number.isFinite(m.simulated_duration_s) && Number.isFinite(m.declared_duration_s)
      && Number.isInteger(m.observed_substeps) && Number.isInteger(m.valid_substeps)
      && Array.isArray(m.equipment) && m.equipment.length <= 12
      && (staticSupport() ? validSupportMetrics(m, geometry) : Array.isArray(m.contact_requirements) && m.contact_requirements.length <= 24)
      && (!personOnly() || (m.equipment.length === 0 && m.contact_requirements.length === 0 && geometry.objects.length === 0
        && m.reward_profile === "human_tracking_effort_v1" && m.gate_applicability?.human_tracking === "applied"
        && m.gate_applicability?.equipment_stability === "not_applicable" && m.gate_applicability?.contact_requirements === "not_applicable"));
    if (!exerciseStateReady) {
      pause(); text($("exercise-outcome"), "Task result unavailable · controls paused");
      for (const id of ["exercise-duration", "exercise-body-errors", "exercise-equipment-summary", "exercise-gates"]) text($(id), "Unavailable; no zero values or success are inferred.");
      $("exercise-equipment").replaceChildren(); $("exercise-contacts").replaceChildren(); exerciseDetailSignature = "";
      text($("exercise-failure"), "The selected task, control contract or state metrics are unavailable. Any visible geometry is the previous display, not a new verified state."); $("exercise-failure").hidden = false;
      text($("readout"), "Exercise task state unavailable. Reset or reload before continuing.");
      if (sharedFeet()) updateFeetDiagnostics(state);
      updateWindowClocks();
      return;
    }
    draw($("live"), geometry);
    text($("exercise-outcome"), exerciseOutcome(m));
    text($("exercise-duration"), `Reported elapsed: ${number(m.simulated_duration_s)} / ${number(m.declared_duration_s)} seconds · ${m.valid_substeps} valid / ${m.observed_substeps} observed substeps · clock: ${m.clock_status || "unavailable"}.`);
    text($("exercise-body-errors"), `Body angular error: mean ${number(m.angular_rmse_rad)} rad / peak ${number(m.peak_angular_rmse_rad)} rad. Root error: mean ${number(m.root_rmse_model_units)} / peak ${number(m.peak_root_rmse_model_units)} model units. Peak floor penetration: ${number(m.max_floor_penetration_model_units)} model units.`);
    const drifts = m.equipment.map(item => item.displacement_model_units);
    const drift = drifts.length && drifts.every(Number.isFinite) ? Math.max(...drifts) : null;
    windowText("exercise-equipment-summary", cameraWorld() ? `Source clock: ${number(m.observed_source_time_s)}s at the observed task boundary; geometry source clock ${number(geometry.source_time_s)}s. Accepted coverage ${number(m.valid_observed_duration_s)}s. Feet and ground/equipment contact are unqualified; no equipment/contact score applies.`
      : staticSupport() ? `Static supports are authored and cannot be actuated. ${m.support_requirements.filter(row => row.active_now).length} active load window(s); ${m.support_requirements.filter(row => row.load_body_weight_fraction !== null).length} current simulated loads available. Equipment drift is not a task metric. Load fractions, coverage and gaps are in the details below.`
      : personOnly() ? "Equipment and contact requirements: not applicable — none are declared. They add no reward or success credit."
      : `Equipment stays passive. Maximum displacement since reset: ${number(drift)} model units across ${m.equipment.length} objects. Detailed rotation, speed and contact checks are below.`);
    const stale = state.engine_state_status === "last_valid_control_boundary";
    $("exercise-failure").hidden = !stale && !m.failure;
    const failure = stale
      ? `Showing last valid geometry at ${number(geometry.time_s)}s after simulation failure; task clock reports ${number(m.simulated_duration_s)}s (${m.clock_status || "unavailable"}). No terminal pose was fabricated. ${m.failure || "Inspect the saved failure."}`
      : m.failure ? `Simulation stopped: ${m.failure}. Reported task time ${number(m.simulated_duration_s)}s; current geometry time ${number(geometry.time_s)}s.` : "";
    text($("exercise-failure"), failure);
    if (failure) { error(failure); simulationFailureMessage = failure; }
    else if (simulationFailureMessage !== null) { if ($("error").textContent === simulationFailureMessage) clearError(); simulationFailureMessage = null; }
    text($("readout"), `${exerciseOutcome(m)} · task ${number(m.simulated_duration_s)}s · geometry ${number(geometry.time_s)}s · ${cameraWorld() ? "supplied camera/ground priors" : staticSupport() ? "authored static supports" : personOnly() ? "human and ground only" : "equipment is not actuated"}.`);
    updateWindowClocks();
    text($("whole-body-status"), exerciseTask?.control?.root_assistance === "disabled" ? "Root assistance disabled; all three root-force commands stay zero."
      : `Explicit virtual root forces [${(geometry.human.mobility?.support_force || []).map(value => number(value, 1)).join(", ")}] · model force units, not measured human forces.`);
    text($("muscle-readout"), cameraWorld() ? "Illustrative muscle bands show simulated motor commands, not measured activation. Feet and ground/equipment contact are unqualified."
      : staticSupport() ? "Illustrative muscle bands show motor commands, not measured activation. Contact dots alone do not meet the declared simulated support-load requirements."
      : "Illustrative muscle bands show motor commands, not measured activation. Contact dots do not automatically qualify as wrist support.");
    if (sharedFeet()) updateFeetDiagnostics(state);
    $("spatial-readout").hidden = true;
    exerciseDetails(m); updateControls();
    if (state.terminated || state.truncated || m.failure) pause();
  }
  function updateSnapshot(state) {
    if (sharedExercise()) { updateExerciseSnapshot(state); return; }
    if (state.simulation_failure) {
      const message = `Simulation failed: ${state.simulation_failure.reason || "unknown dynamics failure"}. Showing the last valid state, not a fabricated terminal pose. Reset before continuing.`;
      error(message); simulationFailureMessage = message;
    } else if (simulationFailureMessage !== null) {
      if ($("error").textContent === simulationFailureMessage) clearError();
      simulationFailureMessage = null;
    }
    snapshot = state; draw($("live"), state); updateWindowClocks();
    updateControls();
    if (sharedWorld()) {
      const h = state.human, cm = state.coupled_metrics || {}, root = h?.mobility;
      const hoCount = state.contact_summary?.human_object ?? 0;
      const targetId = scene.task?.object_goal?.target_object_id || state.controlled_id;
      const suffix = cm.failure || (state.success ? "TASK SUCCESS — reset to continue" : state.terminated || state.truncated ? "TRIAL ENDED — reset to continue" : `target: ${state.controlled_id}`);
      text($("readout"), personOnly()
        ? `t ${number(state.time_s, 2)}s · step ${state.steps} · angular RMSE ${number(cm.angular_rmse_rad)} rad · root RMSE ${number(cm.root_rmse_model_units)} · ${state.contact_count ?? 0} simulated ground contacts · equipment not applicable · ${suffix}`
        : `t ${number(state.time_s, 2)}s · step ${state.steps} · angular RMSE ${number(cm.angular_rmse_rad)} rad · root RMSE ${number(cm.root_rmse_model_units)} · ${targetId} dist ${number(cm.object_distance_world)} · contacts ${state.contact_count ?? 0} (${hoCount} human↔object) · ${suffix}`);
      const forceUnits = cm.contact_force_units || scene.action_contract?.units?.[19] || "uncalibrated model force units";
      text($("whole-body-status"), `Virtual root forces [${(root?.support_force || []).map((v) => number(v, 1)).join(", ")}] ${forceUnits} · ${root?.contact_count ?? 0} simulated floor contacts · ${cm.human_object_contacts_total ?? 0} cumulative human↔object contacts.`);
      text($("muscle-readout"), `${h?.muscles?.length ?? 38} illustrative muscle bands show joint motor commands, not measured activation.`);
      const objLines = state.objects.map((o) => `${o.id} (${o.role}, ${geometryLabel(o)}): pos [${o.position.map((v) => v.toFixed(3)).join(", ")}] · vel [${o.velocity.map((v) => v.toFixed(3)).join(", ")}] · quat [${o.quaternion.map((v) => v.toFixed(3)).join(", ")}]${o.virtual_force ? ` · force [${o.virtual_force.map((v) => v.toFixed(2)).join(", ")}] ${forceUnits}` : ""}`);
      const contactLines = (state.contact_details || []).length
        ? state.contact_details.map((c) => `contact [${c.kind}] ${c.objects.map((id, i) => c.parts?.[i] ? `${id}[${c.parts[i]}]` : id).join("↔")} @ [${c.position.map((v) => v.toFixed(3)).join(", ")}] normal force ${c.normal_force.toFixed(2)} ${forceUnits}`)
        : ["contacts: none"];
      $("spatial-readout").hidden = false;
      text($("spatial-readout"), [...objLines, ...contactLines].join("\n"));
      if (state.terminated || state.truncated) pause();
      return;
    }
    if (state.human) {
      const h = state.human;
      if (wholeBody()) {
        const values = state.whole_body_metrics || {}, root = h.mobility;
        text($("readout"), `t ${number(state.time_s, 2)}s · step ${state.episode_steps ?? state.steps} · angular RMSE ${number(values.angular_rmse_rad)} rad · root RMSE ${number(values.root_rmse_model_units)} model units · maximum floor penetration ${number(values.max_floor_penetration_model_units)} model units · ${values.failure || (state.success ? "TASK SUCCESS — reset to continue" : state.terminated || state.truncated ? "TRIAL ENDED — reset to continue" : "explicit motor control")}`);
        text($("whole-body-status"), `Virtual root forces [${(root?.support_force || []).map(value => number(value, 1)).join(", ")}] N · ${root?.contact_count ?? 0} simulated floor contacts · no automatic support. Scale, depth and forces are not measured from the video.`);
        text($("muscle-readout"), `${h.muscles.length} illustrative muscle bands show joint motor commands, not measured activation. Root forces are separate virtual controls.`);
        if (state.terminated || state.truncated) pause();
        return;
      }
      const angle = h.angles[h.selected_joint];
      if (h.mobility) text($("mobile-status"), `Virtual support ${h.mobility.support_enabled ? "on" : "off"} · ${h.mobility.contact_count} simulated floor contacts · body displacement ${h.mobility.translation.map(value => Number(value).toFixed(2)).join(", ")} model units. Depth and scale are not calibrated.`);
      text($("readout"), `t ${(state.time_s || 0).toFixed(2)}s · ${h.selected_joint.replaceAll("_", " ")} · angle ${(angle * 180 / Math.PI).toFixed(1)}° · target ${(h.target_angle * 180 / Math.PI).toFixed(1)}° · ${state.success ? "TASK SUCCESS — reset to continue" : state.truncated ? "TRIAL ENDED — reset to continue" : actionNames[action]}`);
      text($("muscle-readout"), h.muscles.filter((m) => m.name.includes(h.selected_joint.split("_")[0])).map((m) => `${m.name.replaceAll("_", " ")}: ${Math.round(m.activation * 100)}% illustrative drive`).join(" · "));
      if (state.terminated || state.truncated) pause();
      return;
    }
    if (spatialRigid()) {
      const controlledId = state.controlled_id || scene.controlled_id;
      const body = state.objects.find((item) => item.id === controlledId) || state.objects[0];
      const lengthUnits = scene.units?.length || "scene_width", forceUnits = `${scene.units?.mass || "relative_mass"}*${lengthUnits}/second^2`;
      const suffix = state.success ? "TASK SUCCESS — reset to continue" : state.truncated ? "TIME LIMIT — reset to continue" : state.terminated ? "TRIAL ENDED — reset to continue" : `action: ${usePolicy ? "learned policy" : actionNames[action]}`;
      text($("readout"), `t ${(state.time_s || 0).toFixed(3)}s · step ${state.steps} · ${body.id} pos [${body.position.map((v) => v.toFixed(3)).join(", ")}] ${lengthUnits} · speed ${Math.hypot(...body.velocity).toFixed(3)} ${lengthUnits}/s · quat [${body.quaternion.map((v) => v.toFixed(3)).join(", ")}] · spin [${body.angular_velocity.map((v) => v.toFixed(2)).join(", ")}] rad/s · contacts ${state.contact_count ?? 0} · substeps ${state.substeps ?? 0} · ${suffix}`);
      const objectLines = state.objects.map((o) => `${o.id} (${o.role}, ${o.shape}): pos [${o.position.map((v) => v.toFixed(3)).join(", ")}] · vel [${o.velocity.map((v) => v.toFixed(3)).join(", ")}] · quat [${o.quaternion.map((v) => v.toFixed(3)).join(", ")}] · spin [${o.angular_velocity.map((v) => v.toFixed(2)).join(", ")}]`);
      const contactLines = (state.contact_details || []).length
        ? state.contact_details.map((c) => `contact ${c.objects.join("↔")} @ [${c.position.map((v) => v.toFixed(3)).join(", ")}] dist ${c.distance.toFixed(4)} ${lengthUnits} · normal force ${c.normal_force.toFixed(2)} ${forceUnits}`)
        : ["contacts: none"];
      $("spatial-readout").hidden = false;
      text($("spatial-readout"), [`virtual force (${controlledId}): [${(state.virtual_force || [0, 0, 0]).map((v) => v.toFixed(2)).join(", ")}] ${forceUnits}`, ...objectLines, ...contactLines].join("\n"));
      if (state.terminated || state.truncated) pause();
      return;
    }
    if (state.objects.length > 1) {
      const objectLines = state.objects.map((o) => `${o.id} (${o.role || "body"}, ${o.shape || "shape"}): pos [${o.position.map((v) => v.toFixed(3)).join(", ")}] · vel [${(o.velocity || [0, 0]).map((v) => v.toFixed(3)).join(", ")}] · angle ${number(o.angle ?? 0, 2)} rad`);
      const contacts = (state.contacts || []).length ? `contacts: ${state.contacts.map((pair) => pair.join("↔")).join(", ")}` : "contacts: none";
      $("spatial-readout").hidden = false;
      text($("spatial-readout"), [...objectLines, contacts].join("\n"));
    } else {
      $("spatial-readout").hidden = true;
    }
    const body = state.objects.find((item) => item.id === (state.controlled_id || scene.controlled_id)) || state.objects[0];
    const suffix = state.success ? "TASK SUCCESS — reset to continue" : state.truncated ? "TIME LIMIT — reset to continue" : state.terminated ? "TRIAL ENDED — reset to continue" : `action: ${usePolicy ? "learned policy" : actionNames[action]}`;
    text($("readout"), `t ${(state.time_s || 0).toFixed(2)}s · step ${state.steps} · position ${body.position.map((v) => v.toFixed(3)).join(", ")} · speed ${Math.hypot(...body.velocity).toFixed(3)} · ${suffix}`);
    if (state.terminated || state.truncated) pause();
  }
  function updateControls() {
    const taskMetrics = exerciseMetrics(), blockedTask = sharedExercise() && (!exerciseContractReady || !exerciseStateReady || mode === "fitted" && !fittedReference);
    const ended = snapshot?.terminated || snapshot?.truncated || (sharedExercise() && !!taskMetrics?.failure);
    const success = sharedExercise() ? taskMetrics?.success && taskMetrics.completed_interval && !taskMetrics.failure : snapshot?.success;
    text($("toggle"), running ? "Pause" : ended ? "Start again" : "Start");
    text($("play-state"), !scene ? "Preparing environment" : blockedTask ? "Task unavailable · controls paused" : running ? "Running" : success ? "Goal reached" : ended ? "Trial finished" : "Paused — ready to try");
    if (sharedExercise()) {
      actionNames = ["Release all motors", "Motor +0.25", "Motor −0.25"];
      for (const button of document.querySelectorAll("[data-action]")) {
        if (actionNames[Number(button.dataset.action)]) text(button, actionNames[Number(button.dataset.action)]);
      }
      text($("goal-summary"), cameraWorld() ? "Track the camera-conditioned body reference across the complete source-clock interval within unchanged body/floor limits. Camera scale and ground are supplied assumptions; passing is not physical accuracy, foot support or exercise completion."
        : staticSupport() ? "Across the complete declared interval: track the body reference, stay within body/floor limits and meet every authored segment/load window. Static supports have no controls or equipment-drift score. These are simulated criteria, not physical contact verification."
        : personOnly() ? "Across the complete declared interval: track the body reference and stay within the body/floor limits. Equipment and wrist-contact requirements do not apply. This is simulated tracking, not verified exercise completion."
        : "Across the complete declared interval: track the body reference, stay within the body/floor limits, and keep equipment stable. Only explicitly required wrist-region checks count. This is not a target-object task or proof of grasp.");
      windowText("mode-description", mode === "coast" ? cameraWorld() ? "Coast: all joint and root-force commands are zero. Gravity and the supplied simulated ground still act; contact is not physically qualified." : staticSupport() ? "Coast: every joint and root-force command is zero. Authored supports remain fixed; gravity and simulated contacts still act."
        : personOnly() ? "Coast: every joint and root-force command is zero. Gravity and simulated ground contacts still act." : "Coast: every joint and root-force command is zero. Equipment remains passive; gravity and simulated contacts still act."
        : mode === "reference" ? `Authored feedback follows the body reference using ${exerciseActions?.names?.length || "the declared"} human channels.${sharedFeet() ? " Six ankle motors hold original-start angles; they do not follow video foot motion." : ""} This is programmed control, not learned behavior; equipment has no force controls.`
        : mode === "learned" ? exerciseFeedbackDescription()
        : mode === "fitted" ? fittedDescription()
        : `Move one of ${exerciseActions?.names?.length || "the declared"} human motors with +0.25 or −0.25; all other commands stay zero. Root assistance is ${exerciseTask?.control?.root_assistance === "disabled" ? "disabled" : "explicit, not automatic"}. Equipment has no force controls.`);
    } else if (sharedWorld()) {
      const goalObj = scene.task?.object_goal?.target_object_id || "ball";
      actionNames = sharedHumanTarget()
        ? ["Release all motors", "Motor +0.25", "Motor −0.25"]
        : ["Coast", "← Left (−x)", "Right (+x) →", "↑ Forward (+y)", "↓ Back (−y)", "Up (+z)", "Down (−z)", "Slow down"];
      for (const button of document.querySelectorAll("[data-action]")) {
        const code = Number(button.dataset.action);
        if (code >= 6 && button.dataset.spatial !== "true") continue;
        if (actionNames[code]) text(button, actionNames[code]);
      }
      text($("goal-summary"), personOnly() ? "Authored terminal task: match the body reference within configured body and floor limits. No equipment or object target is declared; simulated tracking is not verified exercise completion."
        : `Authored terminal task: match the body reference within configured limits and stop ${actorLabel(goalObj)} in the dashed target at the end of the trial.${scene.task?.object_goal?.require_human_contact ? " A configured simulated human–object contact check also applies." : ""} This is not exercise or grasp success.`);
      text($("mode-description"), mode === "manual" ? (sharedHumanTarget() ? "Human selected: choose any of the 22 human motors and apply +0.25 or −0.25." : `${actorLabel(snapshot?.controlled_id || scene.controlled_id)} selected: use the eight 3D force controls to push or slow it in the shared world.`) : mode === "reference" ? "Authored human feedback drives the 22 body motors along the reference while rigid objects respond purely to gravity and 3D collisions." : "The learned coupled controller coordinates all 22 human motors and bounded 3D object guidance in one MuJoCo model.");
      if (personOnly()) text($("mode-description"), mode === "manual" ? "Choose one of 22 explicit human motors and apply +0.25 or −0.25; every other command stays zero."
        : mode === "reference" ? "Authored feedback drives the body along its reference. Only human and ground are modeled; root forces remain explicit, not unassisted balance."
        : "The attached human policy controls the declared human motors. No object forces or object target are present.");
    } else if (snapshot?.human) {
      const h = snapshot.human, degrees = value => (value * 180 / Math.PI).toFixed(0);
      text($("goal-summary"), wholeBody() ? "Track the known full-body reference using 19 joint motors and 3 virtual root-force controls. Tracking and floor-contact errors determine success; this is not learned balance." : mode === "reference" ? "Follow the video’s estimated joint movement. This controller is programmed, not learned." : `${h.selected_joint.replaceAll("_", " ")}: ${degrees(h.angles[h.selected_joint])}° now → ${degrees(h.target_angle)}° target. Reach the target and slow to a stop.`);
    } else if (spatialRigid()) text($("goal-summary"), `Move ${actorLabel(snapshot?.controlled_id || scene.controlled_id)} into the 3D dashed target [${(snapshot?.target || scene.target).map((v) => v.toFixed(2)).join(", ")}] m and slow below ${scene.task.speed_tolerance} m/s.`);
    else if (scene) text($("goal-summary"), dynamicObjects().length > 1 ? `Move ${actorLabel(snapshot?.controlled_id || scene.controlled_id)} into the dashed target and slow to a stop.` : "Move the highlighted object into the dashed target and slow to a stop.");
    if (!sharedWorld()) text($("mode-description"), wholeBody() ? (mode === "manual" ? "Choose one of 22 motors. Apply +0.25 or −0.25 of its force limit; every other motor stays at zero. Release sets all motors to zero." : mode === "reference" ? "Authored feedback follows the known full-body reference. Every joint torque and virtual root force is explicit; this baseline is not learned." : "The learned full-body controller follows a known reference using 19 joint torques and 3 virtual root forces. It has not learned unassisted balance or human intent.") : mode === "manual" ? "You control the movement. Tap a direction for a small move, or start and hold a direction." : mode === "reference" ? "A programmed controller follows the video’s estimated poses. This is simulated movement, not learned exercise behavior." : current?.profile === "exercise" ? `The learned controller moves the ${(trainedJoint || "trained joint").replaceAll("_", " ")} toward one target angle. It has not learned the whole exercise.` : "The learned controller attempts the target task. Watch its choices, or pause to inspect a step.");
    for (const [id, value] of [["mode-manual", "manual"], ["mode-coast", "coast"], ["reference-servo", "reference"], ["policy", "learned"]]) {
      $(id).setAttribute("aria-pressed", String(mode === value));
      $(id).disabled = !scene || controlBusy || blockedTask || (value === "coast" && !sharedExercise()) || (value === "learned" && (!learnedPolicyAvailable || (current?.profile === "exercise" && (wholeBody() ? !wholeBodyPolicyReady : !trainedJoint))));
    }
    if (sharedExercise()) $("start-learned").disabled = controlBusy || blockedTask || !learnedPolicyAvailable;
    if (fittedButton) {
      const disabled = !fittedReference || controlBusy || blockedTask, pressed = String(mode === "fitted");
      if (fittedButton.disabled !== disabled) fittedButton.disabled = disabled;
      if (fittedButton.getAttribute("aria-pressed") !== pressed) fittedButton.setAttribute("aria-pressed", pressed);
    }
    for (const button of document.querySelectorAll("[data-action]")) {
      const code = Number(button.dataset.action);
      const isSpatialExtra = button.dataset.spatial === "true";
      const spatialMismatch = isSpatialExtra && !(spatialRigid() || sharedObjectTarget());
      const mobileOnly = !isSpatialExtra && code >= 6 && scene?.profile !== "exercise-v3";
      const incompatible = (wholeBody() || sharedHumanTarget()) && ![0, 1, 2].includes(code);
      const exerciseServo = current?.profile === "exercise" && code === 5;
      button.disabled = !scene || controlBusy || blockedTask || mode !== "manual" || !!ended || spatialMismatch || mobileOnly || incompatible || (current?.profile === "rolling" && code > 2) || exerciseServo;
      button.hidden = spatialMismatch || mobileOnly || incompatible || exerciseServo;
    }
    $("joint").disabled = (recording && !(wholeBody() || sharedWorld())) || controlBusy || blockedTask || mode !== "manual" || sharedObjectTarget();
    $("actor").disabled = recording || controlBusy || mode !== "manual" || sharedExercise();
    $("record").disabled = recording || exporting || !scene || controlBusy || blockedTask;
    $("stop-record").disabled = !recording || controlBusy;
    for (const id of ["toggle", "reset", "step"]) $(id).disabled = !scene || controlBusy || (blockedTask && id !== "reset") || (id === "step" && !!ended);
    const recordedTime = sharedExercise() ? taskMetrics?.simulated_duration_s : snapshot?.time_s || 0;
    text($("record-status"), recording ? `● Recording · ${number(recordedTime, 1)}s of simulation` : exporting ? "Saving recording…" : "Not recording");
  }
  function setRecording(value) {
    recording = value; $("record-status").classList.toggle("active", value); updateControls();
  }
  function pause() {
    playbackGeneration++; running = false; held.clear(); action = 0;
    clearTimeout(playbackTimer); updateControls();
  }
  function step(count = 1) {
    if (requestPending || controlBusy || !scene || document.hidden) return;
    if (sharedExercise() && (!exerciseContractReady || !exerciseStateReady || snapshot.terminated || snapshot.truncated || exerciseMetrics()?.failure || mode === "learned" && !learnedPolicyAvailable || mode === "fitted" && !fittedReference)) return;
    requestPending = true;
    stepPromise = (async () => {
    try {
      let payload = {action, steps: count, policy: usePolicy};
      if (sharedExercise()) {
        const vector = exerciseActions.names.map(name => mode === "manual" && name === selectedMotor ? (action === 1 ? .25 : action === 2 ? -.25 : 0) : 0);
        payload = mode === "learned" ? {steps: count, policy: true} : mode === "reference" ? {steps: count, controller: "reference"} : mode === "fitted" ? {steps: count, controller: "reference_proposal"} : mode === "coast" ? {steps: count, controller: "coast"} : {steps: count, action: vector};
      } else if (wholeBody()) {
        const names = scene.action_contract.names;
        const vector = names.map(name => name === selectedMotor ? (action === 1 ? .25 : action === 2 ? -.25 : 0) : 0);
        payload = mode === "learned" ? {steps: count, policy: true} : mode === "reference" ? {steps: count, controller: "reference"} : {steps: count, action: vector};
      } else if (sharedWorld()) {
        if (mode === "learned") payload = {steps: count, policy: true};
        else if (mode === "reference") payload = {steps: count, controller: "reference"};
        else if (sharedHumanTarget()) {
          const names = scene.action_contract.names;
          const vector = names.map((name) => name === selectedMotor ? (action === 1 ? 0.25 : action === 2 ? -0.25 : 0) : 0);
          payload = {steps: count, action: vector};
        } else {
          payload = {steps: count, action};
        }
      }
      const data = await request("/api/step", payload);
      updateSnapshot(data.snapshot);
      if (data.warning) { pause(); if (!data.snapshot?.simulation_failure) error(data.warning); setRecording(false); scheduleStatus(0); }
    } catch (e) { pause(); error(e.message); scheduleStatus(0); }
    finally { requestPending = false; }
    })();
    return stepPromise;
  }
  async function tick(generation) {
    if (!running || document.hidden || activePanel !== "play" || generation !== playbackGeneration) return;
    const begin = performance.now();
    await step(3);
    if (running && generation === playbackGeneration) playbackTimer = setTimeout(() => tick(generation), Math.max(0, scene.dt * 3000 - (performance.now() - begin)));
  }
  function start() {
    if (!scene || controlBusy || document.hidden || activePanel !== "play" || snapshot.terminated || snapshot.truncated) return;
    if (sharedExercise() && (!exerciseContractReady || !exerciseStateReady || exerciseMetrics()?.failure || mode === "learned" && !learnedPolicyAvailable || mode === "fitted" && !fittedReference)) return;
    usePolicy = mode === "learned"; action = mode === "reference" ? 5 : 0;
    running = true; clearError(); updateControls(); tick(++playbackGeneration);
  }
  function selectMode(value) {
    if (controlBusy) return;
    if (sharedExercise() ? !["manual", "coast", "reference", "learned", "fitted"].includes(value) || value === "learned" && !learnedPolicyAvailable || value === "fitted" && !fittedReference : ["coast", "fitted"].includes(value)) return;
    pause(); mode = value; usePolicy = mode === "learned"; updateControls();
  }
  async function mutate(path, data = {}, prepareLearned = false) {
    if (controlBusy) return;
    if (sharedExercise() && (["/api/actor", "/api/joint"].includes(path) || (path === "/api/record/start" && (!exerciseContractReady || !exerciseStateReady)))) return;
    pause(); controlBusy = true; mutationEpoch++; updateControls();
    try {
      await stepPromise; requestPending = true; if (path !== "/api/reset") clearError();
      if (prepareLearned && mode === "learned" && current?.profile === "exercise" && !wholeBody()) {
        if (!trainedJoint) throw new Error("The trained joint is unavailable. Try manual controls instead.");
        if (snapshot.human.selected_joint !== trainedJoint) {
          if (recording) throw new Error("Stop & save this recording before switching to the trained joint.");
          const joint = await request("/api/joint", {joint: trainedJoint});
          scene = joint.scene; if (joint.capabilities) updateCapabilities(joint.capabilities);
          updateSnapshot(joint.snapshot); $("joint").value = trainedJoint;
        }
      }
      if (prepareLearned && mode === "learned" && !sharedExercise() && !snapshot?.human && trainedActor && (snapshot?.controlled_id || scene?.controlled_id) !== trainedActor) {
        if (recording) throw new Error("Stop & save this recording before switching to the trained object.");
        const switched = await request("/api/actor", {actor: trainedActor});
        scene = switched.scene; if (switched.capabilities) updateCapabilities(switched.capabilities);
        updateSnapshot(switched.snapshot); $("actor").value = trainedActor;
      }
      const result = await request(path, data);
      if (result.capabilities) updateCapabilities(result.capabilities);
      if (result.scene) {
        scene = result.scene;
        if ((!snapshot?.human || sharedWorld()) && $("actor").children.length) $("actor").value = result.scene.controlled_id;
        if (sharedWorld()) action = 0;
        if (!referenceState && (result.snapshot || snapshot)) draw($("world-canvas"), result.snapshot || snapshot);
      }
      if (result.snapshot) updateSnapshot(result.snapshot);
      if ("recording" in result) setRecording(result.recording);
      if (result.export === "running") exporting = true;
      scheduleStatus(0); return result;
    } catch (e) {
      if (snapshot?.human && !wholeBody() && !sharedWorld()) $("joint").value = snapshot.human.selected_joint;
      else if ((!snapshot?.human || sharedWorld()) && scene?.controlled_id && $("actor").children.length) $("actor").value = snapshot?.controlled_id || scene.controlled_id;
      error(e.message);
    }
    finally { requestPending = false; controlBusy = false; if (sharedWorld() && mode === "manual") chooseAction(action); updateControls(); }
  }
  async function begin({fresh = false, record = false} = {}) {
    if (controlBusy) return;
    try { await appearanceView?.flush(); } catch (e) { error(e.message); return; }
    if (controlBusy) return;
    const needsJoint = mode === "learned" && current?.profile === "exercise" && !wholeBody() && snapshot?.human?.selected_joint !== trainedJoint;
    const needsActor = mode === "learned" && !sharedExercise() && !snapshot?.human && Boolean(trainedActor) && (snapshot?.controlled_id || scene?.controlled_id) !== trainedActor;
    if (fresh || record || needsJoint || needsActor || snapshot?.terminated || snapshot?.truncated) {
      const pending = mutate(record ? "/api/record/start" : "/api/reset", {}, true);
      const generation = playbackGeneration;
      if (!await pending || generation !== playbackGeneration) return;
    }
    start();
  }
  $("toggle").onclick = () => running ? pause() : begin();
  $("mode-manual").onclick = () => selectMode("manual");
  $("mode-coast").onclick = () => selectMode("coast");
  $("policy").onclick = () => selectMode("learned");
  $("reference-servo").onclick = () => selectMode("reference");
  $("step").onclick = async () => {
    pause();
    const needsJoint = mode === "learned" && current?.profile === "exercise" && !wholeBody() && snapshot?.human?.selected_joint !== trainedJoint;
    const needsActor = mode === "learned" && !sharedExercise() && !snapshot?.human && Boolean(trainedActor) && (snapshot?.controlled_id || scene?.controlled_id) !== trainedActor;
    if (needsJoint || needsActor) {
      const pending = mutate("/api/reset", {}, true), generation = playbackGeneration;
      if (!await pending || generation !== playbackGeneration) return;
    }
    usePolicy = mode === "learned";
    action = mode === "reference" ? 5 : Number(document.querySelector("[data-action][aria-pressed=true]")?.dataset.action || 0); step();
  };
  $("reset").onclick = () => mutate("/api/reset");
  $("record").onclick = () => begin({record: true});
  $("stop-record").onclick = () => mutate("/api/record/stop");
  $("joint").onchange = () => {
    if (!wholeBody() && !sharedWorld()) return mutate("/api/joint", {joint: $("joint").value});
    if (controlBusy || mode !== "manual" || (sharedExercise() && !exerciseActions?.names?.includes($("joint").value))) { $("joint").value = selectedMotor; return; }
    pause(); selectedMotor = $("joint").value; chooseAction(0); updateControls();
  };
  $("actor").onchange = () => mutate("/api/actor", {actor: $("actor").value});
  $("start-manual").onclick = () => { showPanel("play"); selectMode("manual"); $("live").focus(); };
  $("start-learned").onclick = () => { showPanel("play"); selectMode("learned"); begin({fresh: true}); };
  $("start-reference").onclick = () => showPanel("world");
  function redrawViews() {
    if (snapshot) draw($("live"), snapshot);
    if (referenceState) draw($("world-canvas"), referenceState);
    else if ((spatialRigid() || sharedWorld()) && snapshot) draw($("world-canvas"), snapshot);
  }
  $("camera-yaw").oninput = () => { $("world-yaw").value = $("camera-yaw").value; redrawViews(); };
  $("world-yaw").oninput = () => { $("camera-yaw").value = $("world-yaw").value; redrawViews(); };
  $("show-muscles").oninput = redrawViews;
  $("spatial-yaw").oninput = () => { $("spatial-world-yaw").value = $("spatial-yaw").value; redrawViews(); };
  $("spatial-world-yaw").oninput = () => { $("spatial-yaw").value = $("spatial-world-yaw").value; redrawViews(); };
  $("spatial-zoom").oninput = () => { $("spatial-world-zoom").value = $("spatial-zoom").value; redrawViews(); };
  $("spatial-world-zoom").oninput = () => { $("spatial-zoom").value = $("spatial-world-zoom").value; redrawViews(); };
  $("spatial-follow").onchange = () => { $("spatial-world-follow").checked = $("spatial-follow").checked; redrawViews(); };
  $("spatial-world-follow").onchange = () => { $("spatial-follow").checked = $("spatial-world-follow").checked; redrawViews(); };
  function showReference(index) {
    $("reference-frame").value = index; referenceState = referenceFrames[index];
    if (referenceState) { draw($("world-canvas"), referenceState); text($("reference-time"), `${(referenceState.source_time_s ?? referenceState.time_s ?? 0).toFixed(2)} seconds · estimated pose`); }
  }
  function stopReference() { clearTimeout(referenceTimer); referenceTimer = null; text($("reference-play"), "Play captured movement"); }
  function referenceTick() {
    if (document.hidden || activePanel !== "world") { stopReference(); return; }
    const index = Number($("reference-frame").value);
    if (index >= referenceFrames.length - 1) { stopReference(); return; }
    const time = frame => frame.source_time_s ?? frame.time_s ?? 0;
    const delay = Math.max(16, (time(referenceFrames[index + 1]) - time(referenceFrames[index])) * 1000);
    referenceTimer = setTimeout(() => { showReference(index + 1); referenceTick(); }, delay);
  }
  $("reference-play").onclick = () => {
    if (referenceTimer !== null) { stopReference(); return; }
    if (Number($("reference-frame").value) >= referenceFrames.length - 1) showReference(0);
    text($("reference-play"), "Pause captured movement"); referenceTick();
  };
  $("reference-frame").oninput = () => { stopReference(); showReference(Number($("reference-frame").value)); };
  function chooseAction(value) {
    if (mode !== "manual" || controlBusy || ((wholeBody() || sharedHumanTarget()) && ![0, 1, 2].includes(value))) return;
    action = value; usePolicy = false;
    for (const button of document.querySelectorAll("[data-action]")) button.setAttribute("aria-pressed", String(!button.hidden && Number(button.dataset.action) === value));
  }
  for (const button of document.querySelectorAll("[data-action]")) {
    button.onpointerdown = (event) => { if (mode !== "manual") return; button.setPointerCapture(event.pointerId); chooseAction(Number(button.dataset.action)); };
    button.onpointerup = () => { if (running) chooseAction(0); };
    button.onpointercancel = () => chooseAction(0);
    button.onclick = () => { if (!running && mode === "manual") { chooseAction(Number(button.dataset.action)); step(3); } };
  }
  $("live").onkeydown = (event) => {
    if (event.key === "Escape") { pause(); return; }
    const value = keyAction(event.key);
    if (value === undefined || mode !== "manual" || snapshot?.terminated || snapshot?.truncated) return;
    if (current?.profile === "rolling" && value > 2) return;
    event.preventDefault(); held.add(event.key); chooseAction(value); if (!running && !event.repeat) step(3);
  };
  $("live").onkeyup = (event) => { if (keyAction(event.key) === undefined) return; event.preventDefault(); held.delete(event.key); chooseAction(held.size ? (keyAction([...held].at(-1)) ?? 0) : 0); };
  if (rootAppearanceAvailable()) {
    $("avatar-live").onkeydown = $("live").onkeydown;
    $("avatar-live").onkeyup = $("live").onkeyup;
  }
  const pauseAll = () => { pause(); stopReference(); };
  window.addEventListener("blur", pauseAll);
  document.addEventListener("visibilitychange", () => { appearanceView?.setVisible(!document.hidden && activePanel === "play"); if (document.hidden) { pauseAll(); clearTimeout(statusTimer); suspendRecordingDiagnosis(); suspendRecordingEvidence(); } else scheduleStatus(0); });
  window.addEventListener("pagehide", () => { appearanceView?.setVisible(false); pauseAll(); clearTimeout(statusTimer); suspendRecordingDiagnosis(); suspendRecordingEvidence(); });

  function rootAppearanceAvailable() { return typeof globalThis.WeftLabAppearance?.create === "function"; }
  function configureAppearance(data) {
    if (sharedFeet() || cameraWorld() || !data.appearance || !rootAppearanceAvailable()) return;
    appearanceView = globalThis.WeftLabAppearance.create({token, pause,
      send: async settings => (await request("/api/appearance", settings)).appearance,
      redraw: () => { if (snapshot) draw($("live"), snapshot); },
      camera: () => ({yaw: Number($("spatial-yaw").value), zoom: Number($("spatial-zoom").value), follow: $("spatial-follow").checked}),
      visible: () => activePanel === "play" && !document.hidden});
    appearanceView.configure(data.appearance, scene);
  }

  async function artifact(name) {
    const response = await fetch(fileURL(`run/${name}`), {signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error(`Artifact is not available yet: ${name}`);
    if (Number(response.headers.get("Content-Length")) > (name === "human-reference.json" ? 8_000_000 : 2_000_000)) throw new Error("This artifact is large; download it from the Files tab.");
    return response.json();
  }
  async function inspect(name) {
    try { const data = await artifact(name); text($("inspect-title"), name); text($("json"), JSON.stringify(data, null, 2)); $("inspector").hidden = false; $("inspector").scrollIntoView({block: "start"}); }
    catch (e) { error(e.message); }
  }
  for (const button of document.querySelectorAll("[data-json]")) button.onclick = () => inspect(button.dataset.json);
  $("close-inspector").onclick = () => { $("inspector").hidden = true; };
  function metrics(element, entries) {
    element.replaceChildren();
    for (const [value, label] of entries) { const box = document.createElement("div"); box.className = "metric"; const strong = document.createElement("strong"), span = document.createElement("span"); text(strong, value); text(span, label); box.append(strong, span); element.append(box); }
  }
  function curve(training) {
    const canvas = $("learning-curve"), ctx = canvas.getContext("2d"), history = training.history || [];
    ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = "#596562"; ctx.font = "13px system-ui";
    const fullBody = wholeBody() || sharedWorld();
    ctx.fillText(fullBody ? "Best training return by generation (not evaluation)" : "Training success · last 100–500 episodes (not evaluation)", 20, 23);
    if (!history.length) return;
    ctx.strokeStyle = "#D8C9BA"; ctx.beginPath(); ctx.moveTo(45, 40); ctx.lineTo(45, 205); ctx.lineTo(940, 205); ctx.stroke();
    ctx.strokeStyle = "#965644"; ctx.lineWidth = 2; ctx.beginPath();
    const max = fullBody ? history.at(-1).generation + 1 : history.at(-1).episode;
    const peak = fullBody ? Math.max(1, ...history.map((r) => r.best_training_return || 0)) : 1;
    history.forEach((row, i) => { const x = 45 + (fullBody ? row.generation + 1 : row.episode) / max * 890, y = 205 - Math.min(1, Math.max(0, (fullBody ? row.best_training_return / peak : row.recent_training_success_rate))) * 160; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
    ctx.fillText(fullBody ? "0" : "0%", 8, 205); ctx.fillText(fullBody ? peak.toFixed(1).replace(/\.0$/, "") : "100%", 4, 45); ctx.fillText(`${max} ${fullBody ? "generations" : "episodes"}`, 820, 230);
  }
  async function readyDetails(data) {
    scene = data.scene;
    configureAppearance(data);
    showSourceWindow();
    if (sharedExercise()) configureExerciseTask(data);
    const noLearnedPolicy = sharedExercise() ? !configureExercisePolicy(data) : data.learned_policy_available === false || scene.learned_policy_available === false;
    learnedPolicyAvailable = !noLearnedPolicy;
    if (data.capabilities) updateCapabilities(data.capabilities);
    configureFittedReference(data);
    if (data.evidence_provenance?.layers) {
      const prov = data.evidence_provenance, layers = prov.layers, src = prov.source_evidence || {};
      const missing = src.missing_or_uncertain_summary ? ` · Gaps/uncertainty: ${src.missing_or_uncertain_summary}.` : "";
      const muscles = layers.muscle_display && layers.muscle_display !== "not_applicable" ? ` · Muscles: ${layers.muscle_display.replaceAll("_", " ")}.` : "";
      $("evidence-layers").hidden = false;
      text($("evidence-layers-text"), `Layer separation — Source: ${layers.recorded_source}${src.frame_count ? ` (${src.frame_count} frames)` : ""}.${missing} Kinematic projection: ${layers.kinematic_projection}. Controllers: ${(layers.controllers || []).join(", ")}. Physics: ${layers.physical_simulation}.${muscles}`);
    }
    if (!sharedExercise() && !data.snapshot?.human && learnedPolicyAvailable && !trainedActor) trainedActor = scene.controlled_id;
    const controllable = sharedExercise() ? [{id: "human", label: cameraWorld() ? "Human · supplied camera/ground priors" : staticSupport() ? "Human · supports stay fixed" : personOnly() ? "Human · ground only" : "Human · equipment stays passive"}] : sharedWorld()
      ? [{id: "human", label: "Articulated human (22 motors)"}, ...dynamicObjects()]
      : dynamicObjects();
    $("actor-controls").hidden = sharedExercise() || (!sharedWorld() && Boolean(data.snapshot?.human)) || controllable.length < 2;
    $("actor").replaceChildren();
    for (const obj of controllable) {
      const option = document.createElement("option");
      option.value = obj.id;
      text(option, obj.id === "human" ? obj.label : `${obj.label || obj.id} (${obj.id})`);
      $("actor").append(option);
    }
    if (controllable.length) $("actor").value = data.snapshot?.controlled_id || scene.controlled_id;
    if ((data.profile === "exercise" && (wholeBody() || noLearnedPolicy)) || data.profile === "shared-world") {
      const feasibility = data.reference_feasibility, summary = feasibility?.summary;
      $("reference-feasibility").hidden = false;
      const isShared = data.profile === "shared-world";
      const label = feasibility?.status === "infeasible"
        ? (isShared ? "Reference has floor or static conflicts; this sandbox is a diagnostic simulation." : "Reference has floor conflicts; this sandbox is a diagnostic simulation.")
        : feasibility?.status === "feasible"
          ? (isShared ? "Reference passed the configured coupled floor/static checks; this sandbox is a diagnostic simulation, not validated physical reconstruction." : "Reference passed the configured floor checks; this sandbox is a diagnostic simulation, not validated physical reconstruction.")
          : "Reference floor feasibility is unavailable; this sandbox is a diagnostic simulation.";
      const detail = summary ? ` Maximum reference penetration: ${number(summary.max_penetration_model_units)} model units. Conflicting source poses: ${summary.failed_source_poses}/${summary.source_pose_checks}; conflicting sampled control poses: ${summary.failed_control_grid_poses}/${summary.control_grid_checks}. ${noLearnedPolicy ? "No learned policy is available for this candidate." : (isShared ? "Selected RL results remain separate." : "Selected RL window results remain separate.")}` : "";
      const gap = data.ground_clearance?.summary;
      const clearance = gap ? ` Reference body–ground gap: minimum ${number(gap.minimum)}, median ${number(gap.median)}, maximum ${number(gap.maximum)} model units. Contact timing has not been inferred.` : "";
      text($("feasibility-notice"), label + detail + clearance);
    }
    updateSnapshot(data.snapshot); if (!sharedExercise() || exerciseStateReady) draw($("world-canvas"), data.snapshot);
    $("build-details").open = false; text($("build-status"), "· complete");
    text($("start-help"), "Ready. Start with the captured movement, or jump straight into the sandbox.");
    for (const id of ["start-reference", "start-manual", "start-learned", "mode-manual"]) $(id).disabled = id === "start-learned" && (!learnedPolicyAvailable || data.profile === "exercise");
    $("scene-json").dataset.json = data.profile === "rolling" ? "world.json" : "scene.json";
    $("reconstruction-json").dataset.json = data.profile === "rolling" ? "report.json" : "reconstruction.json";
    if (data.profile === "rolling") text($("control-help"), "Left/right arrows or A/D to push. Release to coast. This profile has horizontal control only.");
    if (data.profile === "shared-world") {
      $("spatial-controls").hidden = false; $("spatial-world-controls").hidden = false;
      $("human-controls").hidden = false; $("human-reference-controls").hidden = false;
      $("whole-body-status").hidden = false; $("reference-servo").hidden = false;
      text($("joint-label"), sharedExercise() ? "Human motor to move" : "Human motor (when human selected)");
      const humanMotorNames = sharedExercise() ? (exerciseContractReady ? exerciseActions.names : []) : (scene.action_contract?.names || []).slice(0, 22);
      $("joint").replaceChildren();
      for (const name of humanMotorNames) {
        const option = document.createElement("option");
        option.value = name;
        text(option, name.replaceAll("_", " "));
        $("joint").append(option);
      }
      selectedMotor = humanMotorNames[0] || null;
      if (selectedMotor) $("joint").value = selectedMotor;
      text($("world-tab"), "Coupled 3D world");
      text($("world-heading"), "One coupled 3D world: articulated human and rigid objects.");
      text($("world-description"), "An articulated 22-motor human body and 3D rigid objects share one MuJoCo model, coordinate frame, clock, and contact solver. Muscle bands remain illustrative motor-command proxies, scale bridges are explicit, and physical validation is unverified.");
      text($("evidence-boundary"), `Coupled ${scene.profile} run. Source slot conservation, coordinate scale bridge, and kinematic preflight are explicit; physical validation remains unverified.`);
      text($("control-help"), "Select 'Articulated human' to drive any of the 22 body motors (+0.25 / −0.25), or select a dynamic 3D object to apply the eight 3D virtual forces. Choose Follow video for programmed human reference feedback.");
      text($("play-boundary"), "Human limbs and 3D rigid objects exchange real simulated MuJoCo contact impulses on one shared clock. Virtual root motors remain explicit assumptions, not unassisted balance.");
      $("live").setAttribute("aria-label", "Coupled 3D human and object simulation. Switch between human motors and 3D rigid objects; Escape pauses.");
      const hasVideo = (data.artifacts || []).some((file) => /^source\.(mp4|webm|mov|m4v|avi|mkv)$/.test(file.name));
      if (!hasVideo && $("tracks").hidden) text($("tracks-empty"), "No source video was sealed with this shared-world run. Inspect scene.json, reference-feasibility.json, and Files for provenance.");
      try {
        const reference = await artifact("human-reference.json");
        referenceFrames = reference.frames;
        $("reference-frame").max = Math.max(0, referenceFrames.length - 1);
        showReference(0);
        $("reference-play").disabled = referenceFrames.length < 2;
      } catch (e) { error(e.message); }
      if (noLearnedPolicy) {
        text($("start-help"), "Coupled 3D world ready. Inspect the front/top/3D views, scrub kinematic poses, or switch between human motors and 3D objects in the sandbox. No learned controller has been trained yet.");
        text($("learning-description"), "No learned coupled policy was packaged with this shared-world run.");
        text($("training-caption"), "Not trained · inspect interventions.json for verified human-feedback and 8-action object probes.");
      } else {
        text($("start-help"), "Coupled 3D world and learned policy ready. Try manual human/object control, authored human reference feedback, or the trained coupled controller.");
        text($("learning-description"), "Compare the learned coupled controller against zero force, random actions, and authored human feedback on identical seeded resets.");
      }
      if (sharedExercise()) {
        text($("start-help"), `Human-only task selected. Inspect the source and assumptions, then try manual motors, coast or authored body feedback${learnedPolicyAvailable ? ", or the attached selected policy" : ""}. Equipment is passive; record a trial to inspect the complete task result.`);
        text($("world-description"), "The same articulated body and equipment share one simulated world. In this task only human motors are controlled; equipment responds to gravity and contacts. Geometry, scale and physical accuracy remain unverified.");
        text($("control-help"), `Choose one of ${humanMotorNames.length || "the declared"} human motors. Tap + or −, or start and hold. Release sets all commands to zero. Equipment cannot be selected or pushed with virtual forces.`);
        text($("play-boundary"), "Human-only simulator task: passive equipment, explicit root-assistance setting and frozen whole-interval gates. Contact dots are not automatically qualifying wrist support. This does not verify grasp, anatomy, muscles, balance or real exercise completion.");
        showExerciseLearning(data);
        $("live").setAttribute("aria-label", "Human-only exercise simulation. Select a human motor; equipment remains passive. Escape pauses.");
      }
      if (personOnly()) {
        text($("world-tab"), "Human and ground");
        text($("world-heading"), "One 3D world: human body and ground.");
        text($("world-description"), "The articulated body and ground share one simulated world. No task-relevant equipment or obstacles are declared; their absence is not visually verified. Geometry, scale and physical accuracy remain unverified.");
        text($("evidence-boundary"), "Person-only declared inventory. Source slots, clocks and body proportions remain explicit; this is not proof that the source contains no equipment.");
        text($("start-help"), `Person-only world ready. Inspect the source and assumptions, then try human motors, coast or authored feedback${learnedPolicyAvailable ? ", or the attached selected policy" : ""}. Record a trial to inspect the complete task result.`);
        text($("control-help"), `Choose one of ${humanMotorNames.length || "the declared"} human motors. Tap + or −, or start and hold. Release sets all commands to zero. No object can be selected or pushed.`);
        text($("play-boundary"), "Human-and-ground simulator task with explicit root assistance and unchanged body/floor gates. Equipment and contact-requirement gates are not applicable and add no reward. This does not verify anatomy, muscles, balance or real exercise completion.");
        $("live").setAttribute("aria-label", "Person-only exercise simulation with ground. Select a human motor. Escape pauses.");
        if (!sharedExercise() && noLearnedPolicy) {
          text($("learning-description"), "No learned human policy was packaged with this person-only run.");
          text($("training-caption"), "Not trained · only human and ground are modeled.");
        }
      }
      if (staticSupport()) {
        text($("world-tab"), "Human and static supports");
        text($("world-heading"), "One 3D world: human body and authored static supports.");
        text($("world-description"), "The articulated body contacts fixed, authored collision shapes in one simulated world. Only human motors are controlled. Support shapes, load directions and thresholds are declared assumptions, not recovered physical contact.");
        text($("evidence-boundary"), "Authored static-support task. Original body/source clocks remain separate from these added support assumptions; physical validation is unverified.");
        text($("start-help"), `Inspect the authored support requirements, then try manual human motors, coast or programmed reference feedback${learnedPolicyAvailable ? ", or the attached selected policy" : ""}. Record a trial to inspect its simulated loads and complete-interval result.`);
        text($("control-help"), `Choose one of ${humanMotorNames.length || "the declared"} human motors. Tap + or −, or start and hold. Release sets all commands to zero. Static supports cannot be selected, moved or pushed.`);
        text($("play-boundary"), "Body tracking and authored segment/load windows are evaluated over the fixed interval. Root assistance is explicit; equipment drift is not scored. This does not establish wrist support, grasp, anatomy, unassisted balance or physical accuracy.");
        $("live").setAttribute("aria-label", "Human simulation with authored static supports. Only human motors are controlled. Escape pauses.");
      }
      if (cameraWorld()) {
        text($("world-tab"), "Camera-conditioned world");
        text($("world-heading"), "One simulated body with supplied camera and ground priors");
        text($("world-description"), "The camera-conditioned body keeps the original monocular shape and authored mass. Camera scale and ground placement are supplied assumptions, not independently measured anatomy or physical accuracy. No equipment inventory is inferred.");
        text($("evidence-boundary"), "Original video, kinematic body reference and live simulated movement are separate. Missing feet, ground support and equipment contact remain unqualified.");
        text($("start-help"), learnedPolicyAvailable ? "Try manual controls, coast, programmed feedback or the experimental trained controller. It adjusts joints; body-position assistance stays programmed when enabled. Record a trial to inspect actions and replay results. Ground/equipment contact is not verified."
          : "Inspect the source and camera assumptions, then try manual motors, coast or authored body feedback. Record a trial to inspect actions, saved states and exact replay evidence. Learning and avatars are unsupported.");
        text($("control-help"), `Choose one of ${humanMotorNames.length || "the declared"} human motors. Release sets every command to zero. No object-force, foot or avatar controls are available.`);
        text($("play-boundary"), "Actual camera-conditioned simulation with unchanged body tracking/floor gates and explicit root assistance. Feet, equipment contact and ground support are unqualified. Passing does not establish physical accuracy or exercise completion.");
        if (!learnedPolicyAvailable) {
          text($("learning-description"), "No trained controller is attached. Manual, coast and programmed body feedback remain available. Ground/equipment contact is not verified; avatars are unsupported.");
          text($("training-caption"), "No trained controller attached · manual and programmed controls are available.");
          text($("exercise-evaluation-summary"), "Recorded trials report the authored simulated body task. No verified learned-policy evaluation or physical validation is attached.");
        }
        $("live").setAttribute("aria-label", "Camera-conditioned human simulation. Select an explicit human motor; Escape pauses.");
      }
      if (sharedFeet()) {
        text($("world-heading"), "One 3D world: body with explicit ankle controls");
        text($("world-description"), `The body and feet share the same simulated world${staticSupport() ? " with fixed authored supports" : personOnly() ? " and ground; no equipment is declared" : " with passive equipment"}. Six added ankle motors use authored shapes and original-start angle holds, not recovered anatomy or video foot motion.`);
        text($("start-help"), "Try manual body/ankle motors, coast or authored feedback, then record a trial. Original body criteria remain unchanged. Ankle errors and foot contacts are unscored; learned policies and avatars are unsupported for this task.");
        text($("play-boundary"), "The original 19-joint body score and existing task requirements are unchanged. Ankle effort and foot contacts add no score or support credit. Ground-load qualification, learned policies and avatars are unsupported; physical validation is unverified.");
        text($("learning-description"), "Learning is unsupported for this feet task. Manual, coast and authored original-start ankle holds are available.");
        text($("training-caption"), "Not trained · no existing policy is qualified for these added ankle controls.");
        text($("exercise-evaluation-summary"), "No learned-policy evaluation is available for this feet task. Recorded trial results use the unchanged original body task.");
        $("live").setAttribute("aria-label", "Body and foot simulation. Select one explicit body or ankle motor. Escape pauses.");
      }
    }
    if (data.profile === "spatial-rigid") {
      actionNames = ["Coast", "← Left (−x)", "Right (+x) →", "↑ Forward (+y)", "↓ Back (−y)", "Up (+z)", "Down (−z)", "Slow down"];
      for (const button of document.querySelectorAll("[data-action]")) {
        if (Number(button.dataset.action) >= 6 && button.dataset.spatial !== "true") continue;
        const label = actionNames[Number(button.dataset.action)];
        if (label) text(button, label);
      }
      $("spatial-controls").hidden = false; $("spatial-world-controls").hidden = false;
      text($("world-tab"), "3D scene");
      text($("world-heading"), "One 3D world. Front, top and angled views.");
      text($("world-description"), "Rigid 3D objects share relative scene-width coordinates, rotations, contacts and MuJoCo dynamics. Depth, shape, materials, controls and task remain explicit assumptions; physical validation is unverified.");
      text($("evidence-boundary"), "Imported from a sealed 3D scene run. Depth, unseen surfaces, mass, friction and task forces are assumptions—not verified physical measurements.");
      const hasVideo = (data.artifacts || []).some((file) => /^source\.(mp4|webm|mov|m4v|avi|mkv)$/.test(file.name));
      if (!hasVideo && $("tracks").hidden) text($("tracks-empty"), "No source video or tracking preview was sealed with this 3D scene run. Inspect scene.json, reconstruction.json and Files for provenance.");
      text($("control-help"), "Tap one of the eight 3D controls, or start and hold. With the scene focused: A/D or Left/Right push x, W/S or Up/Down push depth y, Q/E (or PageUp/PageDown) push height z, Space slows movement, Escape pauses.");
      text($("play-boundary"), "Bounded virtual translational forces act on the controlled object; rotation and contact forces come from 3D collisions. Physical validation remains unverified.");
      $("live").setAttribute("aria-label", "3D rigid simulation. Arrows and WASD push x and depth y; Q and E push height z; Space slows movement; Escape pauses.");
      if (noLearnedPolicy) {
        text($("start-help"), "3D scene ready. Inspect the front/top/3D views and assumptions, then try manual 3D controls and recording. No learned controller has been trained for this run.");
        text($("learning-description"), "No learned controller was trained or imported for this 3D scene run.");
        text($("learning-boundary"), "Manual controls and 8-action intervention probes verify simulator responsiveness only, not learned behavior or physical accuracy.");
        text($("training-caption"), "Not trained · inspect interventions.json for verified 8-action MuJoCo control probes.");
      } else {
        text($("start-help"), "3D scene and learned controller ready. Inspect the 3D world, try manual controls, or watch the trained controller.");
        text($("learning-description"), "Compare the learned 3D policy against matched baselines on the same 3D rigid scene.");
      }
    }
    if (data.profile === "appearance") {
      text($("evidence-boundary"), "The rectangles were selected in the source video. Tracking compares their appearance; a match does not prove visibility or identity. Missing or ambiguous matches remain untracked.");
      text($("world-description"), "Selected regions become simple 2D shapes. Gravity, collisions and controls are explicit assumptions, checked against later tracked positions.");
      text($("play-boundary"), "This is an authored 2D simulation, not recovered 3D physics. Tracking a region does not establish what object it is or which forces act on it.");
    }
    if (noLearnedPolicy && ["planar", "appearance"].includes(data.profile)) {
      text($("start-help"), "2D scene ready. Inspect the video, tracks and assumptions, then try manual controls and recording. Learning was not requested for this run.");
      text($("learning-description"), "No learned controller or policy evaluation was requested for this run.");
      text($("learning-boundary"), "Manual controls and recorded trials do not establish learned behavior or physical accuracy.");
      text($("training-caption"), "Not trained · manual controls, reset and recording are available.");
    }
    if (data.profile === "exercise") {
      actionNames = ["Release", "Bend", "Straighten", "Slow down", "Hold position", "Follow video"];
      text($("world-tab"), "3D body");
      text($("evidence-boundary"), "The colored landmarks are estimated joints. Hidden limbs and depth are uncertain; inspect the source before trusting the inferred pose.");
      const identity = data.result?.pose_identity;
      if (identity?.mode === "temporal") text($("evidence-boundary"), `Landmarks and anatomical labels are estimates. ${identity.repaired_frames} frames had left/right labels reassigned using nearby poses. Raw detections and reasons are preserved in identity-audit.json; unresolved identities are not verified.`);
      text($("repair-description"), "Body proportions are fitted to estimated poses. Automatic physical repair of the human model is not implemented; an empty repair ledger does not mean the body is validated.");
      text($("learning-boundary"), "The learned task is one selected joint reaching an authored target angle—not learning the whole exercise. These results apply only to this simplified simulation.");
      text($("play-boundary"), "This body is fixed at the pelvis, with simplified joints and no balance or ground-contact model. Muscle colors illustrate motor commands—not measured muscle activation or force.");
      $("reference-servo").hidden = false;
      if (!noLearnedPolicy) {
        try { const policy = await artifact("policy.json"); trainedJoint = policy.selected_joint; wholeBodyPolicyReady = policy.algorithm === "whole_body_cem_v1"; } catch (e) { error(e.message); }
      }
      $("start-learned").disabled = wholeBody() ? !wholeBodyPolicyReady : !trainedJoint;
      text($("world-heading"), "An articulated body, with its assumptions visible.");
      text($("world-description"), "Body proportions and joint poses are inferred from the video. The simplified model has an anchored pelvis and eight hinge joints; it is not an anatomical scan or a full-body balance model.");
      text($("control-help"), "Choose a joint, then tap Bend or Straighten. Or start and hold a button. With the scene focused: left/up bends, right/down straightens, Space slows.");
      if (["exercise-v2", "exercise-v3"].includes(scene.profile)) {
        actionNames = ["Release", "Rotate +", "Rotate −", "Slow down", "Hold position", "Follow video"];
        text($("world-description"), "This experimental body has 19 rotation axes, including sideways limb motion, twist and torso rotation. Its pelvis stays fixed. Estimated depth, hidden limbs and twist remain uncertain.");
        text($("control-help"), "Choose a rotation axis, then tap Rotate + or Rotate −. Start and hold for continuous movement. With the scene focused: left/up increases, right/down decreases, Space slows.");
      }
      if (scene.profile === "exercise-v3") {
        actionNames.push("Release body support", "Push body left", "Push body right", "Restore support");
        $("mobile-controls").hidden = false;
        text($("world-description"), "This body has 19 rotation axes and three translation axes. Pelvis movement comes from image estimates under a fixed-camera, fixed-scale assumption. Depth stays at an assumed reference value; the floor is placed under the first pose.");
        text($("play-boundary"), "The body can move, fall and touch the assumed floor. Virtual motors help it follow the video; this is not recovered balance, measured contact forces or anatomical muscle activation.");
        text($("control-help"), "Rotate a joint or use the body buttons below. Release body support disables all motors; pushes do not turn support back on. Restore support holds the current pose. Follow video restores support and follows the current video time.");
      }
      if (wholeBody()) {
        actionNames = ["Release all motors", "Motor +0.25", "Motor −0.25"];
        $("mobile-controls").hidden = true; $("whole-body-status").hidden = false;
        text($("joint-label"), "Motor to control");
        text($("world-description"), "This body has 19 rotation axes and three translation axes. Every joint torque and virtual root force is an explicit controller action. Source-derived depth, scale and floor contact remain assumptions.");
        text($("play-boundary"), "The policy tracks a known reference with privileged simulator feedback and virtual root forces. There is no automatic support, recovered human intent or claim of unassisted balance. Muscle colors are illustrative motor commands.");
        text($("control-help"), "Choose any joint or root motor. Tap +0.25 or −0.25; all other motors remain zero. With the scene focused: left/up applies +0.25, right/down applies −0.25, and Space releases every motor.");
        text($("learning-boundary"), "All four controllers use the same 22 explicit motors and matched starts. Later windows are development evidence from an already-exposed reference, not physical transfer or a final holdout.");
        text($("learning-description"), "Compare the learned full-body policy, zero motor force, random actions and authored feedback on matched simulated trials.");
        $("live").setAttribute("aria-label", "Whole-body simulation. Arrow keys control the selected motor; Space releases all motors; Escape pauses.");
        $("learning-curve").setAttribute("aria-label", "Best whole-body training return by completed generation");
      }
      if (noLearnedPolicy) {
        text($("start-help"), "Camera-aware candidate. Inspect the captured poses, then try manual control and recording. Training has not been run.");
        text($("world-description"), "Supplied camera data separates camera movement from estimated body travel. The view follows the body; recorded positions keep the complete world movement. Pose scale, camera accuracy and floor contact remain uncertain.");
        text($("learning-description"), "No learned controller was created or transferred for this candidate.");
        text($("learning-boundary"), "Manual controls use an authored joint-target task. Follow video is programmed feedback, not learned exercise behavior.");
        text($("training-caption"), "Not trained · inspect camera-reference.json for every estimated, missing and unsampled source frame.");
        if (data.ground_clearance) text($("world-description"), "Supplied cameras and ground height share one coordinate system. The view follows the body without changing its world position. Gaps above the ground remain visible; no contact or body-height correction is applied.");
        if (scene.human?.root_method === "two-view-root-body-v1") {
          text($("world-description"), "Two camera views set body position; limb shape and joint angles still come from the single-view model. Ground height is supplied, not fitted. The view follows the body without hiding gaps or intersections.");
          text($("training-caption"), "Not trained · inspect two-view-root.json for position sources and missing frames. camera-reference.json keeps the original single-view estimates.");
        }
        if (scene.human?.geometry_mode) {
          const modes = {directions: "available joint directions", dimensions: "body lengths", both: "body lengths and available joint directions"};
          const coverage = scene.human.geometry_coverage || {};
          const partialSegments = scene.human.root_method === "two-view-body-v2";
          const counts = Object.entries({left_shoulder: "left arm", right_shoulder: "right arm", left_hip: "left leg", right_hip: "right leg"}).map(([key, label]) => {
            const values = coverage[key] || {}, total = Object.values(values).reduce((a, b) => a + b, 0);
            return partialSegments ? `${label} ${values.two_view_estimate || 0} complete + ${values.two_view_proximal_monocular_distal_and_twist || 0} upper-only / ${total}` : `${label} ${values.two_view_estimate || 0}/${total}`;
          });
          const sourceDescription = partialSegments ? "Upper-only limbs use the supported shoulder-to-elbow or hip-to-knee direction; the missing endpoint, bend and twist around the bone remain single-view estimates." : "Missing limb directions retain single-view estimates.";
          text($("world-description"), `Two views set position and ${modes[scene.human.geometry_mode]}. ${sourceDescription} Left/right lengths share a symmetry assumption. Ground and contact are not fitted. Available ${partialSegments ? "limb" : "complete limb"} estimates: ${counts.join("; ")}.`);
          text($("training-caption"), "Not trained · two-view-body.json shows which geometry uses two views and which retains single-view estimates. Muscles remain illustrative.");
        }
      }
      $("human-controls").hidden = false; $("human-reference-controls").hidden = false;
      $("joint").replaceChildren(); for (const name of (wholeBody() ? scene.action_contract.names : scene.joint_names)) { const option = document.createElement("option"); option.value = name; text(option, name.replaceAll("_", " ")); $("joint").append(option); }
      selectedMotor = wholeBody() ? scene.action_contract.names[0] : null;
      $("joint").value = wholeBody() ? selectedMotor : data.snapshot.human.selected_joint;
      for (const button of document.querySelectorAll("[data-action]")) {
        if (button.dataset.spatial === "true") continue;
        const label = actionNames[Number(button.dataset.action)]; if (label) text(button, label);
      }
      try { const reference = await artifact("human-reference.json"); referenceFrames = reference.frames; $("reference-frame").max = Math.max(0, referenceFrames.length - 1); showReference(0); $("reference-play").disabled = referenceFrames.length < 2; } catch (e) { error(e.message); }
    }
    renderObjectDetails();
    $("assumptions").replaceChildren(); for (const assumption of scene.assumptions || []) { const li = document.createElement("li"); text(li, assumption); $("assumptions").append(li); }
    const skipLearningArtifacts = sharedExercise() || noLearnedPolicy;
    if (noLearnedPolicy) { $("learning-curve").hidden = true; $("scores").replaceChildren(); }
    const results = await Promise.allSettled([skipLearningArtifacts ? Promise.resolve(null) : artifact("training.json"), skipLearningArtifacts ? Promise.resolve(null) : artifact("rl-evaluation.json"), artifact("repair-ledger.json"), artifact(data.profile === "rolling" ? "report.json" : "reconstruction.json")]);
    if (results[0].status === "fulfilled" && results[0].value) { const training = results[0].value; curve(training); text($("training-caption"), (wholeBody() || sharedWorld()) ? `${training.completed_generations} completed generations · ${number(training.seconds, 2)} seconds${training.budget_exhausted ? " · stopped at local time budget" : ""}` : `${training.completed_episodes ?? training.episodes} training episodes · ${training.steps} steps · ${Number(training.seconds).toFixed(2)} seconds${training.budget_exhausted ? " · stopped at local time budget" : ""}`); }
    if (results[1].status === "fulfilled" && results[1].value) metrics($("scores"), Object.entries(results[1].value.policies).map(([name, score]) => {
      if (sharedWorld()) {
        return [`${score.successes}/${score.episodes}`, `${({learned: "Learned coupled controller", zero: "Zero force", random: "Random actions", authored_feedback: "Authored human feedback"})[name] || name.replaceAll("_", " ")} · coupled passes · mean return ${number(score.mean_return)} · obj dist ${number(score.mean_final_object_distance_world)}`];
      }
      if (wholeBody()) {
        const trials = score.trials || [], successes = trials.filter(trial => trial.success).length;
        return [`${successes}/${trials.length}`, `${({learned: "Learned full-body controller", zero: "Zero motor force", random: "Random actions", authored_feedback: "Authored feedback (not learned)"})[name] || name.replaceAll("_", " ")} · full-body trials passed · mean return ${number(score.mean_return)}`];
      }
      const distanceUnit = data.profile === "spatial-rigid" ? "scene-width units" : "m";
      const extra = typeof score.mean_final_distance_m === "number" ? ` · mean return ${number(score.mean_return)} · final dist ${number(score.mean_final_distance_m)} ${distanceUnit}` : "";
      return [`${score.successes}/${score.episodes}`, `${({q_learning: "Learned controller", learned: "Learned controller", random: "Random actions", coast: "No control", brake: "Braking baseline", heuristic_feedback: "Target feedback (not learned)"})[name] || name.replaceAll("_", " ")} · goals reached${extra}`];
    }));
    if (results[2].status === "fulfilled") {
      const ledger = results[2].value; const rows = Array.isArray(ledger) ? ledger : (ledger.proposals || ledger.repairs || ledger.candidates || []);
      $("repairs").replaceChildren();
      for (const change of rows) { const row = document.createElement("tr"); cell(row, JSON.stringify(change.proposal || change.patch || change.parameters || change.candidate || change.method)); cell(row, change.accepted ? "Accepted" : "Rejected"); cell(row, change.reason || "Inspect the ledger for fixed gate measurements"); $("repairs").append(row); }
      if (data.profile !== "exercise") metrics($("repair-summary"), [[rows.length, "Bounded proposals"], [rows.filter((r) => r.accepted).length, "Accepted changes"]]);
    }
    if (results[3].status === "fulfilled") {
      const result = results[3].value, scores = data.profile === "rolling" ? result.reconstruction : result;
      const before = scores.before?.rmse_px ?? scores.initial?.rmse_px ?? scores.baseline?.rmse_px;
      const after = scores.after?.rmse_px ?? scores.selected?.rmse_px;
      if (before !== undefined && after !== undefined) metrics($("repair-summary"), [[Number(before).toFixed(2) + " px", "Initial development error"], [Number(after).toFixed(2) + " px", "Selected development error"]]);
    }
    updateControls(); refreshArtifactButtons(data);
  }
  function renderObjectDetails() {
    if (!scene) return;
    $("objects").replaceChildren();
    if (staticSupport() && !validStaticSupports(scene.objects)) {
      const row = document.createElement("tr"); cell(row, "Static-support geometry unavailable; no substitute inferred."); $("objects").append(row); return;
    }
    for (const obj of scene.objects) {
      const row = document.createElement("tr");
      cell(row, `${actorLabel(obj.id)} · ${obj.id}`); cell(row, obj.role); cell(row, geometryLabel(obj));
      cell(row, inspectionActors.get(obj.id)?.scope || (current?.profile === "exercise"
        ? "Simplified body segment; inferred proportions and authored physics."
        : "Geometry and physics are assumptions; inspect the model data for provenance."));
      $("objects").append(row);
    }
  }
  function showGeometryPanel() {
    if (activePanel !== "world") return;
    const panels = inspection?.geometry?.panels || [], index = Number($("geometry-panel").value);
    const panel = panels[index];
    if (!panel) return;
    if ($("geometry-preview").dataset.path === panel.name) return;
    $("geometry-preview").dataset.path = panel.name;
    $("geometry-preview").src = fileURL(`run/${panel.name}`); $("geometry-preview").hidden = false;
    text($("geometry-caption"), `${index + 1}/${panels.length} · ${panel.name.split("/").pop()} · Saved source slots and authored hypotheses. This image is not synchronized to video playback and does not show a live physics state.`);
    $("geometry-download").replaceChildren(link("Download full-size comparison", `run/${panel.name}`));
  }
  $("geometry-panel").onchange = showGeometryPanel;
  $("geometry-preview").onerror = () => {
    $("geometry-preview").hidden = true;
    text($("geometry-caption"), "The saved comparison could not be displayed. Download it below; no geometry result is inferred.");
  };
  for (const [id, panel, target] of [["original", "evidence", "source"], ["tracking", "evidence", "tracking-summary"],
    ["geometry", "world", "geometry-summary"], ["sandbox", "play", "live"], ["recordings", "play", "record-box"]]) {
    $("inspect-" + id).onclick = () => { showPanel(panel); $(target).scrollIntoView({block: "nearest"}); };
  }
  const evidenceDimensions = [
    ["execution_actions", "Execution and action validity"], ["perception_identity", "Perception and identity"],
    ["geometry_camera", "Geometry and camera"], ["temporal_dynamics", "Temporal dynamics"],
    ["physical_relationships", "Physical relationships"], ["hidden_state_continuity", "Hidden-state continuity"],
    ["parameter_validity", "Parameter validity"], ["causal_response", "Causal response"],
    ["long_run_stability", "Long-run stability"], ["robustness_coverage", "Robustness and coverage"],
    ["uncertainty_abstention", "Uncertainty and abstention"], ["task_learning_utility", "Task and learning utility"],
    ["reproducibility_integrity", "Reproducibility and integrity"], ["practical_efficiency", "Practical efficiency"],
  ];
  const evidenceStatuses = {unavailable: "Unavailable — not measured here", evidence_available: "Evidence available — not a pass", failure_reported: "Failures recorded"};
  const evidenceText = (value, fallback = "Unknown") => typeof value === "string" && value.length > 0 && value.length <= 2048 ? value : fallback;
  const evidenceLabel = value => evidenceText(value).replaceAll("_", " ");
  const evidenceArray = (value, maximum) => Array.isArray(value) ? value.slice(0, maximum) : [];
  const evidenceNumber = value => typeof value === "number" && Number.isFinite(value);
  function evidenceValue(metric) {
    if (!evidenceNumber(metric.value)) return "Unavailable";
    const ratio = Number.isSafeInteger(metric.numerator) && Number.isSafeInteger(metric.denominator)
      && metric.numerator >= 0 && metric.denominator >= 0 && metric.numerator <= metric.denominator;
    return `${ratio ? `${metric.numerator} / ${metric.denominator}` : metric.value} ${evidenceLabel(metric.unit)}`;
  }
  function evidenceLocalPath(value) {
    return typeof value === "string" && value.length <= 512 && !/[\\:\x00-\x1f]/.test(value)
      && value.split("/").every(part => part !== "" && part !== "." && part !== "..");
  }
  function evidencePointers(parent, pointers, verified, manifestSHA) {
    const entries = evidenceArray(pointers, 8);
    if (!entries.length) { text(parent, "Exact evidence unavailable"); return; }
    const list = document.createElement("ul"); list.classList.add("evidence-pointers");
    for (const pointer of entries) {
      const item = document.createElement("li"), identity = document.createElement("small");
      const name = evidenceText(pointer?.source_name, "Unknown artifact"), sha = pointer?.sha256;
      const match = verified.get(name);
      const manifest = name === "manifest.json" && pointer?.path === null && sha === manifestSHA;
      const path = match && match.sha256 === sha && (manifest || match.path === pointer?.path) ? match.path : null;
      if (path) item.append(link(`${name} · download`, path));
      else { const label = document.createElement("span"); text(label, `${name} · verified local download unavailable`); item.append(label); }
      text(identity, typeof sha === "string" && /^[0-9a-f]{64}$/.test(sha) ? `SHA-256: ${sha}` : "SHA-256 unavailable");
      item.append(identity); list.append(item);
    }
    parent.append(list);
  }
  function evidenceRowDetails(parent, row, verified, manifestSHA) {
    const scope = document.createElement("p"); scope.classList.add("hint");
    text(scope, `Evidence kind: ${evidenceLabel(row?.evidence_kind)}. Partition: ${evidenceLabel(row?.partition)}. Individual measurements retain their own labels below.`);
    parent.append(scope);
    const limitsHeading = document.createElement("h4"), limits = document.createElement("ul");
    text(limitsHeading, "Uncertainty and missing measurements");
    const missing = evidenceArray(row?.uncertainty_and_missing_measurements, 16);
    for (const value of missing.length ? missing : ["No saved limits or independently measured result are available in this category."]) {
      const item = document.createElement("li"); text(item, evidenceText(value, "Measurement details unavailable")); limits.append(item);
    }
    parent.append(limitsHeading, limits);
    const metrics = evidenceArray(row?.metrics, 16).filter(metric => metric && typeof metric === "object");
    if (metrics.length) {
      const wrap = document.createElement("div"), table = document.createElement("table"), caption = document.createElement("caption");
      wrap.classList.add("table-wrap"); text(caption, "Saved measurements; recorded limits are not independently retested here.");
      const head = document.createElement("thead"), headers = document.createElement("tr"), body = document.createElement("tbody");
      for (const label of ["Measurement", "Saved value / limit", "Evidence kind / partition", "Exact evidence"]) {
        const th = document.createElement("th"); th.setAttribute("scope", "col"); text(th, label); headers.append(th);
      }
      head.append(headers);
      for (const metric of metrics) {
        const tr = document.createElement("tr"), name = document.createElement("th");
        name.setAttribute("scope", "row"); text(name, evidenceText(metric.label, "Unnamed measurement")); tr.append(name);
        const value = cell(tr, evidenceValue(metric)), gate = metric.recorded_gate;
        if (gate != null) {
          const limit = document.createElement("p"); limit.classList.add("hint");
          text(limit, evidenceNumber(gate.value) ? `Recorded limit: ${evidenceLabel(gate.name)} ${evidenceText(gate.operator, "operator unavailable")} ${gate.value} ${evidenceLabel(gate.unit)}` : "Recorded limit unavailable");
          value.append(limit);
        }
        cell(tr, `Kind: ${evidenceLabel(metric.evidence_kind)}. Partition: ${evidenceLabel(metric.partition)}.`);
        evidencePointers(cell(tr, ""), metric.evidence, verified, manifestSHA); body.append(tr);
      }
      table.append(caption, head, body); wrap.append(table); parent.append(wrap);
    } else { const empty = document.createElement("p"); text(empty, "Measurements unavailable in this saved category."); parent.append(empty); }
    const examples = evidenceArray(row?.failure_examples, 8);
    if (examples.length) {
      const heading = document.createElement("h4"), note = document.createElement("p"), saved = document.createElement("pre");
      text(heading, "Saved failure examples"); note.classList.add("hint");
      text(note, "Frame IDs and times retain their saved clock labels; they are not synchronized to playback.");
      text(saved, JSON.stringify(examples, null, 2)); parent.append(heading, note, saved);
    }
    if (Number.isSafeInteger(row?.failure_examples_omitted) && row.failure_examples_omitted > 0) {
      const omitted = document.createElement("p"); text(omitted, `${row.failure_examples_omitted} additional examples omitted; the saved counts retain them.`); parent.append(omitted);
    }
    const sourcesHeading = document.createElement("h4"), sources = document.createElement("div");
    text(sourcesHeading, "Category evidence"); evidencePointers(sources, row?.evidence, verified, manifestSHA); parent.append(sourcesHeading, sources);
  }
  function showEvidenceScorecard(result) {
    const card = result?.evidence_scorecard, received = evidenceArray(card?.rows, 15);
    const manifestSHA = result?.source_manifest_sha256;
    const manifestBound = typeof manifestSHA === "string" && /^[0-9a-f]{64}$/.test(manifestSHA)
      && card?.source_manifest_sha256 === manifestSHA;
    const knownIds = new Set(evidenceDimensions.map(([id]) => id));
    const supported = manifestBound && card?.schema_version === 1 && card.profile === "weft-evidence-scorecard-v1"
      && Array.isArray(card.rows) && received.length <= 14 && received.every(row => knownIds.has(row?.id))
      && new Set(received.map(row => row.id)).size === received.length;
    const metadata = evidenceArray(result?.verified_metadata, 128);
    const signature = supported ? JSON.stringify([manifestSHA, card, metadata]) : "unavailable";
    if (signature === evidenceScorecardSignature) return;
    evidenceScorecardSignature = signature;
    $("evidence-scorecard-rows").replaceChildren();
    if (!supported) {
      text($("evidence-scorecard-status"), "Evidence summary unavailable for this saved run. No missing measurement is treated as zero or a pass.");
      text($("evidence-scorecard-scope"), "Original files and simulation controls are unchanged."); return;
    }
    const verified = new Map();
    for (const item of metadata) {
      if (evidenceLocalPath(item?.source_name) && item.path === `run/${item.source_name}` && /^[0-9a-f]{64}$/.test(item.sha256)) verified.set(item.source_name, item);
    }
    text($("evidence-scorecard-status"), "14 evaluation categories. Expand a category for saved measurements, exact evidence and remaining gaps.");
    text($("evidence-scorecard-scope"), `${evidenceText(card.scope, "Saved evidence only; no overall grade.")} ${evidenceText(card.identity_scope, "Independent replay and physical validation are unavailable.")}`);
    const rows = new Map(received.map(row => [row.id, row]));
    for (const [id, label] of evidenceDimensions) {
      const candidate = rows.get(id), row = Object.hasOwn(evidenceStatuses, candidate?.status) ? candidate : null;
      const details = document.createElement("details"), summary = document.createElement("summary"), title = document.createElement("strong"), status = document.createElement("span");
      details.dataset.category = id; details.dataset.status = row?.status || "unavailable"; details.open = false;
      text(title, label); text(status, evidenceStatuses[details.dataset.status]); status.classList.add("evidence-row-status");
      summary.append(title, status); details.append(summary);
      let populated = false;
      // Keep hidden tables out of the initial render and preserve them across unchanged refreshes.
      details.ontoggle = () => { if (details.open && !populated) { populated = true; evidenceRowDetails(details, row, verified, manifestSHA); } };
      $("evidence-scorecard-rows").append(details);
    }
  }
  function applyInspection(data) {
    const source = inspection.source, tracking = inspection.tracking;
    showEvidenceScorecard(inspection);
    text($("inspection-status"), "Explore the saved source, estimates and assumptions, then interact and record. This is a prepared simulation candidate, not verified automatic reconstruction.");
    text($("inspection-source"), `${source.decoded_frames === null ? "Decoded frame count unavailable" : `${source.decoded_frames} decoded source frames`}. ${source.sha256 ? `Source SHA256: ${source.sha256}.` : "Source identity unavailable."} ${source.scope}`);
    const poses = tracking.pose_status_counts;
    const body = poses ? `Body estimates: ${(poses.tracked || 0) + (poses.tracked_partial || 0)} usable, including ${poses.tracked_partial || 0} partial. Saved frame statuses: ${Object.entries(poses).map(([key, count]) => `${count} ${key.replaceAll("_", " ")}`).join("; ")}. `
      : inspection.actors.some(actor => actor.id === "human") ? "No compact body-status summary is saved here; check the pose-coverage section if available. " : "";
    text($("inspection-tracking"), `${body}${tracking.reference_poses === null ? "" : `${tracking.reference_poses} body poses retained for model replay. `}${tracking.scope}`);
    $("inspection-actors").replaceChildren();
    for (const actor of inspection.actors) {
      inspectionActors.set(actor.id, actor);
      const card = document.createElement("article"), title = document.createElement("h4"), identity = document.createElement("p"), detail = document.createElement("p"), scope = document.createElement("p");
      text(title, actor.label); text(identity, `ID: ${actor.id} · ${actor.geometry} · ${actor.label_origin}`); identity.classList.add("hint");
      const binding = actor.source_binding, counts = binding?.counts;
      text(detail, counts && binding.kind !== "authored_addition"
        ? `${counts.observed_slots} construction-observation slots · ${counts.missing_slots} not used or unavailable · ${counts.pre_init_slots} pre-initialization. These are not true-visibility or accuracy counts.`
        : "No source-qualified tracking counts for this model entity.");
      text(scope, actor.scope); card.append(title, identity, detail, scope); $("inspection-actors").append(card);
    }
    $("inspection-evidence-links").replaceChildren();
    for (const file of inspection.artifact_links.filter(file => file.label === "Source observation document" || ["evidence.json", "observations.json", "coverage-summary.json", "pose-quality.json"].includes(file.name))) {
      $("inspection-evidence-links").append(link(`${file.label} · ${file.sha256.slice(0, 12)} · download`, `run/${file.name}`));
    }
    if ($("tracks").hidden) text($("tracks-empty"), "No tracking movie is saved in this run. Review the coverage below and the selected source/geometry comparisons in stage 3. Raw observation documents are available to download.");
    for (const option of $("actor").children) {
      const actor = inspectionActors.get(option.value);
      if (actor) text(option, actor.id === "human" ? actor.label : `${actor.label} (${actor.id})`);
    }
    text($("inspection-geometry"), inspection.geometry.status === "initial_hypothesis"
      ? inspection.geometry.scope
      : "No saved source-to-geometry comparison is available. Inspect the model below and its assumptions; physical geometry is unverified.");
    const panels = inspection.geometry.panels;
    $("geometry-gallery").hidden = !panels.length; $("geometry-panel").replaceChildren();
    for (const [index, panel] of panels.entries()) {
      const option = document.createElement("option"); option.value = String(index);
      text(option, panel.name.split("/").pop()); $("geometry-panel").append(option);
    }
    $("geometry-panel").value = "0";
    $("geometry-assumptions").replaceChildren();
    for (const assumption of inspection.geometry.assumptions) {
      const li = document.createElement("li"); text(li, assumption); $("geometry-assumptions").append(li);
    }
    const labels = {available: "Saved evidence available", unavailable: "No saved artifact", initial_hypothesis: "Initial hypotheses · inspect assumptions",
      model_only: "Model only · no comparison", simulation: "Authored simulation · interact", local_recording: "Record, save and replay"};
    for (const stage of inspection.stages) text($("stage-" + stage.id), labels[stage.state] || "Inspect saved evidence");
    exerciseDetailSignature = ""; exerciseDetails(exerciseMetrics());
    renderObjectDetails(); files(data); showGeometryPanel(); updateControls(); showSourceWindow();
  }
  async function showInspection(data) {
    if (inspectionLoaded || data.status !== "ready") return;
    for (const id of ["original", "tracking", "geometry", "sandbox", "recordings"]) $("inspect-" + id).disabled = false;
    try {
      inspectionAttempts++;
      const result = await request("/api/inspection");
      const loading = ["pending", "loading"].includes(result.status);
      inspectionPending = loading && inspectionAttempts < 10;
      if (loading) {
        text($("inspection-status"), inspectionPending ? "Reading a compact summary of the saved run…" : "The saved summary is still being prepared. You can use the lab now or read it again.");
        $("inspection-retry").hidden = inspectionPending;
        inspectionLoaded = !inspectionPending;
        return;
      }
      inspectionLoaded = true;
      if (result.status !== "available") {
        showEvidenceScorecard(null);
        text($("inspection-status"), `Saved inspection summary unavailable: ${result.reason || "no supported metadata"}. Original files and simulation controls are unchanged.`);
        text($("inspection-tracking"), "No validated compact tracking summary is available. Use the saved source files; no completeness claim is made.");
        return;
      }
      inspection = result; applyInspection(data);
    } catch (_) {
      showEvidenceScorecard(null);
      inspectionLoaded = true; inspectionPending = false; $("inspection-retry").hidden = false;
      text($("inspection-status"), "The saved summary could not be read. The lab and saved files remain available.");
    }
  }
  $("inspection-retry").onclick = () => {
    inspectionLoaded = false; inspectionAttempts = 0; $("inspection-retry").hidden = true; scheduleStatus(0);
  };
  function artifactEntries(data) {
    return [...new Map([...(data.artifacts || []), ...(inspection?.artifact_links || [])].map(file => [file.name, file])).values()];
  }
  const inspectLimit = name => name === "human-reference.json" ? 8_000_000 : 2_000_000;
  function refreshArtifactButtons(data) {
    const entries = new Map(artifactEntries(data).map(file => [file.name, file]));
    for (const button of document.querySelectorAll("[data-json]")) {
      const file = entries.get(button.dataset.json);
      button.disabled = !file || file.bytes > inspectLimit(file.name);
      button.title = !file ? "Not saved in this run. Use the stage summary or Files." : button.disabled ? "Large artifact: download from Files." : "Inspect saved artifact";
    }
  }
  function files(data) {
    const entries = artifactEntries(data), signature = entries.map((item) => `${item.name}:${item.bytes}`).join("|");
    if (signature === lastFiles) return; lastFiles = signature;
    $("files").replaceChildren();
    for (const file of entries) {
      const row = document.createElement("tr"); const name = cell(row, ""); name.append(link(file.name, `run/${file.name}`));
      cell(row, `${(file.bytes / 1024).toFixed(1)} KB`); const inspectCell = cell(row, "");
      if (file.name.endsWith(".json")) {
        if (file.bytes <= inspectLimit(file.name)) { const button = document.createElement("button"); button.textContent = "Inspect"; button.onclick = () => inspect(file.name); inspectCell.append(button); }
        else text(inspectCell, "Large document · download to inspect");
      }
      $("files").append(row);
    }
    const names = entries.map((item) => item.name);
    const sourceName = names.find((name) => /^source\.(mp4|webm|mov|m4v|avi|mkv)$/.test(name));
    if (sourceName && !$("source").getAttribute("src")) $("source").src = fileURL(`run/${sourceName}`);
    const preview = names.includes("tracks.gif") ? "tracks.gif" : names.includes("replay-slow.gif") ? "replay-slow.gif" : null;
    if (preview) { $("tracks").src = fileURL(`run/${preview}`); $("tracks").hidden = false; $("tracks-empty").hidden = true; }
    const replay = names.includes("reconstruction.gif") ? "reconstruction.gif" : names.includes("replay-slow.gif") ? "replay-slow.gif" : names.includes("tracks-contact-sheet.png") ? "tracks-contact-sheet.png" : null;
    if (replay) { $("reconstruction-preview").src = fileURL(`run/${replay}`); $("reconstruction-preview").hidden = false; }
    refreshArtifactButtons(data);
  }
  function sessions(data) {
    const signature = JSON.stringify(data.sessions);
    if ($("recordings").dataset.signature !== signature) {
      $("recordings").dataset.signature = signature; $("recordings").replaceChildren();
      if (!data.sessions?.length) { const empty = document.createElement("li"); text(empty, "No recordings yet. Choose Record & start, try a few moves, then Stop & save."); $("recordings").append(empty); }
      for (const [index, session] of (data.sessions || []).entries()) {
        const li = document.createElement("li"); text(li, `Trial ${index + 1}` + (session.complete ? " · saved" : " · actions saved; replay not yet exported"));
        li.append(link("Action data", `sessions/${session.name}/events.jsonl`));
        if (session.complete) for (const [label, file] of [[session.appearance_recorded ? "Physics-body replay" : "Interactive replay", "replay.html"], [session.appearance_recorded ? "Physics-body video" : "Video", "video.mp4"], ["Preview", "preview.gif"]]) li.append(link(label, `sessions/${session.name}/${file}`));
        if (session.appearance_complete) for (const [label, file] of [["Avatar replay", "avatar-replay.html"], ["Avatar video", "avatar-video.mp4"], ["Appearance changes", "appearance-events.jsonl"]]) li.append(link(label, `sessions/${session.name}/${file}`));
        else if (session.appearance_recorded) { const note = document.createElement("span"); text(note, " · Avatar export not complete; physics-body files do not include the chosen character."); li.append(note); }
        if (sharedExercise() && session.complete && /^session-[0-9a-f]{12}$/.test(session.name)) {
          const explain = document.createElement("button"); text(explain, "Explain this trial");
          explain.onclick = () => selectRecordingDiagnosis(session.name); li.append(explain);
        }
        if (session.complete && /^session-[0-9a-f]{12}$/.test(session.name)) {
          const evidence = document.createElement("button"); text(evidence, "Recording evidence");
          evidence.setAttribute("aria-controls", "recording-evidence");
          evidence.setAttribute("aria-label", `Recording evidence for trial ${index + 1}`);
          evidence.onclick = () => selectRecordingEvidence(session.name); li.append(evidence);
        }
        $("recordings").append(li);
      }
    }
    exporting = data.export?.status === "running";
    if (data.export) { text($("export-status"), data.export.status === "ready" ? "Recording exported and verified. Download the offline replay or video below." : data.export.status === "failed" ? `Export failed: ${data.export.error}. The action journal is preserved.` : "Exporting recorded states to video and an offline replay…"); }
  }
  const equipmentMetricLabels = {
    displacement_model_units: ["displacement", "model units"], orientation_rad: ["rotation", "rad"],
    speed_model_units_s: ["speed", "model units/s"], angular_speed_rad_s: ["spin", "rad/s"],
  };
  const humanMetricLabels = {
    angular_rmse_rad: ["Average body-angle error", "rad", "running_rms"],
    root_rmse_model_units: ["Average body-position error", "model units", "running_rms"],
    peak_angular_rmse_rad: ["Peak body-angle error", "rad", "cumulative_peak"],
    peak_root_rmse_model_units: ["Peak body-position error", "model units", "cumulative_peak"],
    max_floor_penetration_model_units: ["Peak floor overlap", "model units", "cumulative_peak"],
  };
  function humanTimelineValid(value) {
    if (value === undefined) return true;
    const metric = row => Object.hasOwn(humanMetricLabels, row?.metric)
      && row.aggregation === humanMetricLabels[row.metric][2] && Number.isFinite(row.threshold) && row.threshold > 0;
    const crossing = row => metric(row) && Number.isFinite(row.value) && row.value > row.threshold + 1e-12
      && (row.previous_value === null || Number.isFinite(row.previous_value) && row.previous_value >= 0)
      && (row.previous_safe_elapsed_s === null || Number.isFinite(row.previous_safe_elapsed_s) && row.previous_safe_elapsed_s >= 0)
      && (row.previous_value !== null || row.previous_safe_elapsed_s === null)
      && Number.isFinite(row.first_failed_elapsed_s) && row.first_failed_elapsed_s >= 0
      && (row.previous_safe_elapsed_s === null || row.previous_safe_elapsed_s <= row.first_failed_elapsed_s)
      && Number.isInteger(row.event_index) && row.event_index >= 0 && row.event_index <= 1800
      && Number.isInteger(row.control_step) && row.control_step >= 0 && row.control_step <= 1800
      && ["failed_at_reset", "first_observed_boundary", "first_reported_boundary"].includes(row.status);
    return value?.profile === "shared-exercise-human-tracking-timeline-v1"
      && Array.isArray(value.terminal) && value.terminal.length === 5
      && new Set(value.terminal.map(row => row.metric)).size === 5
      && value.terminal.every(row => metric(row) && (row.status === "unavailable" ? row.value === null
        : Number.isFinite(row.value) && row.value >= 0 && row.status === (row.value > row.threshold + 1e-12 ? "above_limit" : "within_limit")))
      && Array.isArray(value.first_crossings) && value.first_crossings.length <= 5
      && new Set(value.first_crossings.map(row => row.metric)).size === value.first_crossings.length
      && value.first_crossings.every(crossing)
      && (value.first_failure === null ? value.first_crossings.length === 0
        : value.first_failure && Array.isArray(value.first_failure.crossings) && value.first_failure.crossings.length > 0
          && value.first_failure.crossings.length <= 5 && value.first_failure.crossings.every(row => crossing(row)
            && row.event_index === value.first_failure.event_index && row.first_failed_elapsed_s === value.first_failure.first_failed_elapsed_s
            && value.first_crossings.some(saved => JSON.stringify(saved) === JSON.stringify(row))));
  }
  function recordingTimelineValid(value) {
    const digest = v => typeof v === "string" && /^[0-9a-f]{64}$/.test(v);
    return value?.profile === "shared-exercise-failure-timeline-v1" && [value.scene_sha256, value.task_sha256, value.gates_sha256].every(digest)
      && digest(value.integrity?.manifest_sha256) && value.integrity?.verified_scope === "scene_task_policy_journal_metadata_only"
      && value.integrity?.media_verified === false && value.integrity?.action_replay_performed === false
      && value.source_clock?.kind === "declared_human_reference_clock" && value.source_clock?.native_video_time_available === false
      && (value.equipment_stability_applicability === undefined || ["applied", "not_applicable"].includes(value.equipment_stability_applicability))
      && Array.isArray(value.episodes) && value.episodes.length > 0 && value.episodes.length <= 64
      && value.episodes.every(episode => Number.isInteger(episode.reset_event_index) && episode.reset_event_index >= 0
        && Array.isArray(episode.first_crossings) && episode.first_crossings.length <= 48 && episode.terminal
        && typeof episode.terminal.completed_interval === "boolean" && typeof episode.terminal.success === "boolean"
        && humanTimelineValid(episode.human_tracking)
        && (value.equipment_stability_applicability !== "not_applicable" || episode.first_equipment_failure === null && episode.first_crossings.length === 0)
        && episode.first_crossings.every(crossing => Object.hasOwn(equipmentMetricLabels, crossing.metric)
          && [crossing.value, crossing.threshold, crossing.first_failed_elapsed_s].every(Number.isFinite)
          && Array.isArray(crossing.boundary_contacts) && crossing.boundary_contacts.length <= 64));
  }
  function renderRecordingEpisode() {
    const timeline = recordingDiagnosis?.timeline, index = Number($("recording-diagnosis-episode").value);
    const episode = timeline?.episodes[index];
    if (!episode) return;
    const terminal = episode.terminal, first = episode.first_equipment_failure, human = episode.human_tracking;
    const noEquipment = timeline.equipment_stability_applicability === "not_applicable";
    text($("recording-diagnosis-heading"), noEquipment ? "Saved trial: body tracking" : "Saved trial: tracking and equipment stability");
    const counts = Object.entries(episode.controller_counts || {}).map(([controller, count]) => `${controller}: ${count}`).join(", ");
    text($("recording-diagnosis-outcome"), `Episode ${index + 1}: ${terminal.completed_interval
      ? terminal.success ? "Declared task passed." : "Declared interval complete; task did not pass."
      : "Recording ended before a complete passing task interval."}`);
    text($("recording-diagnosis-episode-evidence"), `Reset event ${episode.reset_event_index} · seed ${episode.seed}. ${counts ? `Recorded controls: ${counts}. ` : ""}Reported task failures: ${terminal.failed_gates?.length ? terminal.failed_gates.join(", ") : "none reported"}. ${terminal.failure ? `Simulation failure: ${terminal.failure}.` : ""}`);
    const crossings = episode.first_crossings;
    const firstCrossings = first?.crossings || [];
    const earliest = firstCrossings[0];
    const window = value => Number.isFinite(value.previous_safe_elapsed_s)
      ? `${number(value.previous_safe_elapsed_s)}–${number(value.first_failed_elapsed_s)}s`
      : `first reported ${number(value.first_failed_elapsed_s)}s; previous safe boundary unavailable`;
    const humanFirst = human?.first_failure;
    const failedHuman = human?.terminal.filter(row => row.status === "above_limit") || [];
    const describe = row => `${humanMetricLabels[row.metric][0]}: ${number(row.value)} ${humanMetricLabels[row.metric][1]}; allowed ${number(row.threshold)} ${humanMetricLabels[row.metric][1]}`;
    const bodyMessage = !human ? "Body-tracking diagnosis is unavailable in this older summary; no body pass or failure timing is inferred."
      : failedHuman.length ? `Body tracking exceeded ${failedHuman.length === 1 ? "a limit" : "limits"}: ${failedHuman.map(describe).join(". ")}.`
      : human.terminal.some(row => row.status === "unavailable") ? "Some body-tracking values are unavailable. No missing value is treated as zero or a pass."
      : "Body-tracking values are within their limits at the end of this episode. The complete task result remains separate.";
    const mean = human?.terminal.find(row => row.metric === "angular_rmse_rad" && row.status === "within_limit");
    text($("recording-diagnosis-human"), bodyMessage + (failedHuman.some(row => row.metric === "peak_angular_rmse_rad") && mean
      ? ` Average body-angle error ${number(mean.value)} rad ended below ${number(mean.threshold)} rad${human.first_crossings.some(row => row.metric === mean.metric) ? " after an earlier crossing" : ""}.` : ""));
    text($("recording-diagnosis-human-first"), humanFirst
      ? `First recorded body-limit crossing: ${window(humanFirst)} after reset · step ${humanFirst.control_step}, event ${humanFirst.event_index}. ${humanFirst.crossings.some(row => row.aggregation === "running_rms")
          ? "Running averages can recover; check the end-of-episode results below." : "Peak values retain earlier deviations even if the pose later improves."}`
      : human ? "No body-limit crossing was recorded. This alone does not establish task success." : "");
    text($("recording-diagnosis-first"), first && earliest
      ? `First equipment limit reported at ${number(first.first_failed_elapsed_s)}s after reset. ${firstCrossings.map(value => {
          const [label, unit] = equipmentMetricLabels[value.metric] || [value.metric, ""];
          return `${actorLabel(value.object_id)} ${label}: reported peak ${number(value.value)} ${unit}, allowed ${number(value.threshold)} ${unit}`;
        }).join("; ")}.`
      : noEquipment ? "Equipment checks: not applicable — none are declared; no success credit is added."
        : "No equipment-limit crossing was recorded in this episode. This does not mean the task passed.");
    const bracket = first?.source_frame_bracket, lower = bracket?.lower?.source_frame_index, upper = bracket?.upper?.source_frame_index;
    const frames = Number.isInteger(lower) && Number.isInteger(upper)
      ? `Retained reference frames ${lower}–${upper}${Array.isArray(bracket.frames_without_reference_between) ? `; ${bracket.frames_without_reference_between.length} intervening source frames have no reference pose` : ""}. ` : "Source-frame bracket unavailable. ";
    text($("recording-diagnosis-clock"), first
      ? `First crossing window: ${window(first)} (step ${first.control_step}, recording event ${first.event_index}). Saved task time: ${number(first.first_failed_elapsed_s)}s; declared reference clock: ${number(first.reference_clock_s)}s. ${frames}${first.engine_state_status === "last_valid_control_boundary"
          ? `Geometry is the last valid boundary at ${number(first.last_valid_geometry_elapsed_s)}s, not a terminal pose. ` : ""}Native video time is unavailable; this does not seek or synchronize the video or live simulation.`
      : "No failure clock is inferred. Video, declared reference and current live time remain separate.");
    const bodyBracket = humanFirst?.source_frame_bracket, bodyLower = bodyBracket?.lower?.source_frame_index, bodyUpper = bodyBracket?.upper?.source_frame_index;
    text($("recording-diagnosis-human-clock"), humanFirst
      ? `Body crossing window: ${window(humanFirst)}. Declared reference clock: ${number(humanFirst.reference_clock_s)}s. ${Number.isInteger(bodyLower) && Number.isInteger(bodyUpper)
          ? `Retained reference frames ${bodyLower}–${bodyUpper}; ${bodyBracket.frames_without_reference_between?.length ?? "unknown"} intervening frames have no reference pose. ` : "Source-frame bracket unavailable. "}${humanFirst.engine_state_status === "last_valid_control_boundary"
          ? `Geometry is the last valid boundary at ${number(humanFirst.last_valid_geometry_elapsed_s)}s, not a terminal pose. ` : ""}No exact substep or cause is inferred. Native video time is unavailable; this does not seek or synchronize the original video or live simulation.` : "");
    const pairs = new Map();
    for (const crossing of firstCrossings) for (const contact of crossing.boundary_contacts || []) {
      if (!Array.isArray(contact.objects) || !contact.objects.includes(crossing.object_id)) continue;
      const names = contact.objects.map(id => id === contact.human_geom && contact.human_segment
        ? `${contact.human_segment.replaceAll("_", " ")} collision shape` : id === "mobile_floor" ? "model floor" : actorLabel(id));
      pairs.set(names.join(" ↔ "), true);
    }
    text($("recording-diagnosis-contacts"), first
      ? pairs.size ? `Contacts at the saved boundary: ${[...pairs.keys()].join("; ")}. Co-occurrence, not proof of cause.`
        : "No matching contact is saved at this boundary. Contact between recorded boundaries may be missing; no cause is inferred."
      : "No first-failure contact is inferred.");
    $("recording-diagnosis-contacts").hidden = noEquipment;
    $("recording-diagnosis-clock").hidden = noEquipment && !first;
    text($("recording-diagnosis-limits"), "Saved trial, not the current simulation. Body-angle error combines the evaluated joints; it does not identify a culprit joint. Model units are uncalibrated.");
    text($("recording-diagnosis-scope"), "Body averages can recover; body and equipment peaks retain earlier deviations. Values are relative to the authored reference and frozen simulator limits, not measured anatomy or physical accuracy. Saved control boundaries bracket crossings but do not retain every physics substep or establish a cause.");
    $("recording-diagnosis-human-metrics").replaceChildren();
    for (const value of human?.terminal || []) {
      const row = document.createElement("tr");
      for (const entry of [humanMetricLabels[value.metric][0], `${number(value.value)} / ${number(value.threshold)} ${humanMetricLabels[value.metric][1]}`,
        {within_limit: "Within limit", above_limit: "Above limit", unavailable: "Unavailable — not a pass"}[value.status]]) cell(row, entry);
      $("recording-diagnosis-human-metrics").append(row);
    }
    $("recording-diagnosis-crossings").replaceChildren();
    for (const value of human?.first_crossings || []) {
      const row = document.createElement("tr");
      for (const entry of [humanMetricLabels[value.metric][0], `${number(value.value)} / ${number(value.threshold)} ${humanMetricLabels[value.metric][1]}`,
        window(value), `${value.event_index} · step ${value.control_step}${value.aggregation === "running_rms" ? " · average may recover" : ""}`]) cell(row, entry);
      $("recording-diagnosis-crossings").append(row);
    }
    for (const value of crossings) {
      const row = document.createElement("tr"), [label, unit] = equipmentMetricLabels[value.metric];
      for (const entry of [`${actorLabel(value.object_id)} · ${label}`, `${number(value.value)} / ${number(value.threshold)} ${unit}`,
        window(value), `${value.event_index} · step ${value.control_step}`]) cell(row, entry);
      $("recording-diagnosis-crossings").append(row);
    }
    const replayEvent = [humanFirst, first].filter(value => Number.isInteger(value?.event_index)).sort((a, b) => a.event_index - b.event_index)[0];
    const replayLabel = replayEvent ? `Interactive replay · select state ${replayEvent.event_index + 1}${Number.isInteger(timeline.events) ? ` of ${timeline.events}` : ""} (event ${replayEvent.event_index})` : "Interactive replay · inspect recorded states";
    $("recording-diagnosis-links").replaceChildren(link(replayLabel, `sessions/${recordingDiagnosis.session}/replay.html`),
      link("Exact actions and states", `sessions/${recordingDiagnosis.session}/events.jsonl`), link("Saved task and scene", `sessions/${recordingDiagnosis.session}/scene.json`));
  }
  $("recording-diagnosis-episode").onchange = renderRecordingEpisode;
  function suspendRecordingDiagnosis() {
    clearTimeout(recordingRetryTimer);
    if (requestedRecording && !recordingDiagnosis) $("recording-diagnosis-retry").hidden = false;
  }
  function applyRecordingDiagnosis(data) {
    recordingDiagnosis = data;
    const timeline = data.timeline, samePolicy = timeline.policy_sha256 && timeline.policy_sha256 === exercisePolicy?.policy_sha256;
    text($("recording-diagnosis-status"), "Saved trial verified and summarized. No physics or media decoding was run.");
    text($("recording-diagnosis-provenance"), `${data.session} · sealed journal ${timeline.integrity.manifest_sha256.slice(0, 12)}. Same scene and task as this lab; ${samePolicy ? "same attached policy" : timeline.policy_sha256 ? "a different or no currently attached policy" : "no policy recorded"}. Journal/scene/task/policy metadata checked; media and action replay are not reverified here.`);
    const selector = $("recording-diagnosis-episode"); selector.replaceChildren();
    for (const [index, episode] of timeline.episodes.entries()) {
      const option = document.createElement("option"); option.value = String(index);
      text(option, `Episode ${index + 1} · ${index ? "after reset" : "recording start"} · event ${episode.reset_event_index}`); selector.append(option);
    }
    selector.value = String(timeline.episodes.length - 1);
    $("recording-diagnosis-content").hidden = false; $("recording-diagnosis-retry").hidden = true;
    renderRecordingEpisode();
  }
  async function readRecordingDiagnosis() {
    if (!requestedRecording || recordingReadPending || document.hidden) return;
    const name = requestedRecording;
    recordingReadPending = true; recordingReadAttempts++;
    try {
      const result = await request(`/api/recording-inspection?session=${encodeURIComponent(name)}`);
      if (requestedRecording !== name) return;
      if (result.status === "loading" && recordingReadAttempts < 12) {
        text($("recording-diagnosis-status"), "Reading this sealed trial in the background. Live controls are unchanged.");
        if (!document.hidden) recordingRetryTimer = setTimeout(readRecordingDiagnosis, 900);
      } else if (result.status === "available" && result.session === name && recordingTimelineValid(result.timeline)) applyRecordingDiagnosis(result);
      else {
        text($("recording-diagnosis-status"), result.status === "loading" ? "The saved summary is still being prepared. Read this trial again when ready."
          : `Saved failure summary unavailable: ${result.reason || "incomplete or incompatible diagnostic metadata"}. No time or success is inferred.`);
        $("recording-diagnosis-retry").hidden = false;
      }
    } catch (failure) {
      if (requestedRecording === name) { text($("recording-diagnosis-status"), `Saved summary could not be read: ${failure.message}. Live controls are unchanged.`); $("recording-diagnosis-retry").hidden = false; }
    } finally {
      recordingReadPending = false;
      if (requestedRecording !== name) readRecordingDiagnosis();
    }
  }
  function selectRecordingDiagnosis(name) {
    clearTimeout(recordingRetryTimer); requestedRecording = name; recordingReadAttempts = 0;
    $("recording-diagnosis").hidden = false;
    if (recordingDiagnosis?.session === name) { applyRecordingDiagnosis(recordingDiagnosis); return; }
    recordingDiagnosis = null; $("recording-diagnosis-content").hidden = true; $("recording-diagnosis-retry").hidden = true;
    text($("recording-diagnosis-status"), "Reading the selected saved trial; not the current live simulation.");
    readRecordingDiagnosis();
  }
  $("recording-diagnosis-retry").onclick = () => { recordingReadAttempts = 0; clearTimeout(recordingRetryTimer); readRecordingDiagnosis(); };
  const recordingIdentityLabels = {
    source_run_manifest_sha256: "Source run manifest · file bytes", recording_manifest_sha256: "Recording manifest · file bytes",
    receipt_manifest_sha256: "Receipt manifest · file bytes", receipt_sha256: "Replay receipt · file bytes",
    recording_scene_sha256: "Recording scene · canonical JSON", engine_scene_sha256: "Engine scene · canonical JSON",
    task_sha256: "Task · canonical JSON", policy_sha256: "Policy · file bytes",
  };
  const feetRecordingIdentityLabels = {base_scene_sha256: "Original scoring scene · canonical JSON", base_task_sha256: "Original task · canonical JSON",
    projection_contract_sha256: "Scoring projection contract", adapter_implementation_sha256: "Feet adapter implementation", foot_reference_sha256: "Exact foot reference"};
  const cameraRecordingIdentityLabels = {camera_coupling_sha256: "Camera coupling", source_sha256: "Source video · file bytes",
    body_sha256: "Original body reference", source_clock_sha256: "Original source clock identity", metric_context_sha256: "Explicit camera metric context",
    kernel_evaluator_sha256: "Unchanged body scorer", adapter_implementation_sha256: "Camera task adapter", evaluator_sha256: "Combined evaluator identity"};
  function recordingEvidencePointer(data, path, digest) {
    const matches = data.verified_metadata.filter(row => row?.source_name === path && row.path === path);
    return matches.length === 1 && /^[0-9a-f]{64}$/.test(matches[0].sha256) && (digest === undefined || matches[0].sha256 === digest)
      && Number.isSafeInteger(matches[0].bytes) && matches[0].bytes >= 0 ? path : null;
  }
  function recordingEvidenceValid(data, name) {
    const digest = value => typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
    const count = value => Number.isSafeInteger(value) && value >= 0;
    if (data?.schema_version !== 1 || data.profile !== "recording-evidence-v1" || data.session !== name
      || data.partition !== "exposed_development" || data.physical_validation !== "unverified"
      || !data.identities || !Array.isArray(data.verified_metadata) || data.verified_metadata.length > 16
      || !["events", "actions", "resets"].every(key => count(data.recorded?.[key]))
      || data.recorded.events !== 1 + data.recorded.actions + data.recorded.resets) return false;
    if (!Object.keys(recordingIdentityLabels).every(key => digest(data.identities[key])
      || ["task_sha256", "policy_sha256"].includes(key) && data.identities[key] === null)) return false;
    if (sharedFeet() && (!exerciseIdentity || data.identities.policy_sha256 !== null
      || data.identities.engine_scene_sha256 !== exerciseIdentity.scene_sha256 || data.identities.task_sha256 !== exerciseIdentity.task_sha256
      || !Object.keys(feetRecordingIdentityLabels).every(key => digest(data.identities[key]) && data.identities[key] === exerciseIdentity[key]))) return false;
    if (cameraWorld() && (!exerciseIdentity
      || (cameraPolicyView ? data.identities.policy_sha256 !== cameraPolicyView.policy_sha256
        || data.identities.policy_admission_sha256 !== cameraPolicyView.admission_sha256
        : data.identities.policy_sha256 !== null || data.identities.policy_admission_sha256 !== undefined)
      || data.identities.engine_scene_sha256 !== exerciseIdentity.scene_sha256 || data.identities.task_sha256 !== exerciseIdentity.task_sha256
      || !Object.keys(cameraRecordingIdentityLabels).every(key => digest(data.identities[key]) && data.identities[key] === exerciseIdentity[key]))) return false;
    if (fittedReference ? data.identities.reference_controller_sha256 !== fittedReference.reference_controller_sha256
      : data.identities.reference_controller_sha256 !== undefined) return false;
    const paths = {source_run_manifest_sha256: "run/manifest.json", recording_manifest_sha256: `sessions/${name}/manifest.json`,
      receipt_manifest_sha256: `sessions/${name}/evidence/manifest.json`, receipt_sha256: `sessions/${name}/evidence/receipt.json`,
      recording_scene_sha256: `sessions/${name}/scene.json`, engine_scene_sha256: "run/scene.json",
      task_sha256: "run/exercise-task.json", policy_sha256: "run/policy.json",
      ...(cameraWorld() && cameraPolicyView ? {policy_admission_sha256: "run/camera-policy-admission.json"} : {}),
      ...(fittedReference ? {reference_controller_sha256: "run/reference-controller.json"} : {})};
    // Canonical JSON identities differ from exact file-byte hashes; the reader checks both.
    if (!Object.entries(paths).every(([key, path]) => data.identities[key] === null || recordingEvidencePointer(data, path,
      ["recording_scene_sha256", "engine_scene_sha256", "task_sha256", "policy_admission_sha256"].includes(key) ? undefined : data.identities[key]))) return false;
    if (fittedReference && (!recordingEvidencePointer(data, "run/reference-trial/manifest.json", fittedReference.reference_trial_manifest_sha256)
      || !recordingEvidencePointer(data, "run/reference-attachment.json"))) return false;
    const replay = data.replay;
    if (!fittedReference && replay?.reference_actions_verified !== undefined) return false;
    if (replay?.status === "not_performed") return replay.protocol === null
      && ["events_checked", "actions_checked", "resets_checked", "learned_actions_verified"].every(key => replay[key] === null)
      && (!fittedReference || replay.reference_actions_verified === null);
    const spatialReplay = scene?.profile === "spatial-rigid-v1" && replay?.protocol === "spatial-rigid-exact-action-state-v1"
      && data.identities.task_sha256 === null && data.identities.policy_sha256 === null && replay.learned_actions_verified === null;
    const spatialPolicyReplay = scene?.profile === "spatial-rigid-v1" && replay?.protocol === "spatial-rigid-exact-action-state-policy-v2"
      && data.identities.task_sha256 === null && (data.identities.policy_sha256 === null
        ? replay.learned_actions_verified === null
        : digest(data.identities.policy_sha256) && count(replay.learned_actions_verified));
    const exerciseReplay = sharedExercise() && !sharedFeet() && !cameraWorld() && replay?.protocol === "shared-exercise-exact-action-state-task-v1"
      && digest(data.identities.task_sha256);
    const feetReplay = sharedFeet() && replay?.protocol === "shared-exercise-feet-full-state-replay-v1" && replay.learned_actions_verified === null;
    const cameraReplay = cameraWorld() && !cameraPolicyView && !fittedReference && replay?.protocol === "camera-shared-full-state-replay-v1" && replay.learned_actions_verified === null;
    const cameraPolicyReplay = cameraWorld() && Boolean(cameraPolicyView) && replay?.protocol === "camera-shared-policy-full-state-replay-v1"
      && count(replay.learned_actions_verified);
    const fittedReplay = cameraWorld() && Boolean(fittedReference) && !cameraPolicyView && replay?.protocol === "camera-shared-reference-full-state-replay-v1"
      && replay.learned_actions_verified === null && count(replay.reference_actions_verified) && replay.reference_actions_verified <= data.recorded.actions;
    return (spatialReplay || spatialPolicyReplay || exerciseReplay || feetReplay || cameraReplay || cameraPolicyReplay || fittedReplay) && replay.status === "passed"
      && ["events", "actions", "resets"].every(key => replay[`${key}_checked`] === data.recorded[key])
      && (replay.learned_actions_verified == null || count(replay.learned_actions_verified)
        && replay.learned_actions_verified <= data.recorded.actions && digest(data.identities.policy_sha256));
  }
  function clearRecordingEvidence() {
    recordingEvidenceSignature = null;
    $("recording-evidence-content").hidden = true;
    $("recording-evidence-identities").replaceChildren(); $("recording-evidence-links").replaceChildren();
    $("recording-evidence-details").open = false;
  }
  function applyRecordingEvidence(data) {
    const signature = JSON.stringify([data.session, data.recorded, data.replay, data.identities]);
    if (signature !== recordingEvidenceSignature) {
      recordingEvidenceSignature = signature;
      const counts = data.recorded, replay = data.replay;
      text($("recording-evidence-counts"), `Recorded: ${counts.actions} actions · ${counts.resets} resets · ${counts.events} total events, including the start event.`);
      const policyProof = replay.protocol === "camera-shared-policy-full-state-replay-v1"
        ? `Exact external policy decisions matched: ${replay.learned_actions_verified}. Manual actions are not policy-checked. Original policy bytes and current execution admission remain distinct; root assistance stays authored and physical accuracy remains unverified.`
        : replay.protocol === "camera-shared-reference-full-state-replay-v1"
        ? `Fitted decisions matched: ${replay.reference_actions_verified}. Manual and original-target actions are not fitted-decision checks. This is not a learned policy or proof of improved control; body-position assistance stays programmed when enabled. Physical accuracy remains unverified.`
        : replay.protocol === "camera-shared-full-state-replay-v1"
        ? "Actual camera state and body task results matched. No learned-policy check was performed; supplied priors and physical accuracy remain unverified."
        : replay.protocol === "shared-exercise-feet-full-state-replay-v1"
        ? "Actual feet state, original body scores and unscored ankle diagnostics matched. Learned policies and foot support qualification are unsupported."
        : replay.protocol === "spatial-rigid-exact-action-state-v1"
        ? "Recorded actions were replayed; learned-policy decisions were not re-evaluated."
        : replay.protocol === "spatial-rigid-exact-action-state-policy-v2"
          ? replay.learned_actions_verified === null
            ? "No learned policy was bound to this recording; manual actions have no policy-decision check."
            : `Learned decisions matched the recorded policy: ${replay.learned_actions_verified}. Manual actions are not policy-checked.`
        : `Learned actions verified: ${replay.learned_actions_verified ?? "unavailable"}.`;
      text($("recording-evidence-replay"), replay.status === "passed"
        ? `Exact action replay: verified in the saved receipt. Checked ${replay.actions_checked} actions, ${replay.resets_checked} resets and ${replay.events_checked} events against saved states and task results. ${policyProof}`
        : "Exact action replay: not performed. Replay-check counts are unavailable, not zero. A saved-state export is not an exact action replay check.");
      text($("recording-evidence-session"), `Selected recording: ${data.session}`);
      $("recording-evidence-identities").replaceChildren();
      for (const [key, label] of Object.entries({...recordingIdentityLabels, ...(sharedFeet() ? feetRecordingIdentityLabels : cameraWorld() ? cameraRecordingIdentityLabels : {}),
        ...(cameraWorld() && cameraPolicyView ? {policy_admission_sha256: "External policy admission · canonical JSON"} : {}),
        ...(fittedReference ? {reference_controller_sha256: "Fitted controller recipe · file bytes"} : {})})) {
        const row = document.createElement("tr"), heading = document.createElement("th");
        heading.setAttribute("scope", "row"); text(heading, label); row.append(heading);
        cell(row, data.identities[key] === null ? "Unavailable — not recorded" : data.identities[key]);
        $("recording-evidence-identities").append(row);
      }
      $("recording-evidence-links").replaceChildren(
        link("Saved replay receipt", recordingEvidencePointer(data, `sessions/${data.session}/evidence/receipt.json`, data.identities.receipt_sha256)),
        link("Receipt manifest", recordingEvidencePointer(data, `sessions/${data.session}/evidence/manifest.json`, data.identities.receipt_manifest_sha256)));
    }
    text($("recording-evidence-status"), "Saved receipt and bound file hashes checked. Replay was not rerun by this view.");
    $("recording-evidence-content").hidden = false; $("recording-evidence-retry").hidden = true;
  }
  function suspendRecordingEvidence() {
    const reading = recordingEvidenceRequest || recordingEvidenceTimer !== null;
    clearTimeout(recordingEvidenceTimer); recordingEvidenceTimer = null;
    recordingEvidenceSuspended = true; recordingEvidenceGeneration++;
    recordingEvidenceRequest?.controller?.abort();
    if (requestedRecordingEvidence && (reading || $("recording-evidence-content").hidden)) {
      text($("recording-evidence-status"), "Recording evidence read paused. Read this recording again when ready.");
      $("recording-evidence-retry").hidden = false;
    }
  }
  async function readRecordingEvidence() {
    if (!requestedRecordingEvidence || recordingEvidenceRequest || recordingEvidenceSuspended || document.hidden) return;
    clearTimeout(recordingEvidenceTimer); recordingEvidenceTimer = null;
    const name = requestedRecordingEvidence, generation = recordingEvidenceGeneration;
    const job = {controller: typeof AbortController === "function" ? new AbortController() : null};
    // Keep one request even when cancellation is unavailable or a late response ignores it.
    recordingEvidenceRequest = job; recordingEvidenceAttempts++;
    const timeout = job.controller ? setTimeout(() => job.controller.abort(), 10000) : null;
    try {
      const result = await request(`/api/recording-evidence?session=${encodeURIComponent(name)}`, undefined, job.controller?.signal);
      if (generation !== recordingEvidenceGeneration) return;
      if (result?.status === "loading" && result.session === name && recordingEvidenceAttempts < 12) {
        text($("recording-evidence-status"), "Checking the selected saved receipt in the background. Live controls are unchanged.");
        recordingEvidenceTimer = setTimeout(readRecordingEvidence, 900);
      } else if (result?.status === "available" && recordingEvidenceValid(result, name)) applyRecordingEvidence(result);
      else {
        clearRecordingEvidence();
        text($("recording-evidence-status"), result?.status === "loading" && result.session === name
          ? "The saved receipt is still being checked. Read this recording again when ready."
          : `Recording evidence unavailable: ${result?.status === "unavailable" && result.session === name && typeof result.reason === "string" ? result.reason.slice(0, 500) : "missing, incompatible or mismatched receipt metadata"}. Exact action replay is not verified.`);
        $("recording-evidence-retry").hidden = false;
      }
    } catch (_) {
      if (generation === recordingEvidenceGeneration) {
        clearRecordingEvidence();
        text($("recording-evidence-status"), "Recording evidence could not be read within the request limit. Exact action replay is unavailable; live controls are unchanged.");
        $("recording-evidence-retry").hidden = false;
      }
    } finally {
      clearTimeout(timeout);
      recordingEvidenceRequest = null;
      if (generation !== recordingEvidenceGeneration && !recordingEvidenceSuspended) readRecordingEvidence();
    }
  }
  function selectRecordingEvidence(name) {
    if (!/^session-[0-9a-f]{12}$/.test(name)) return;
    if (name === requestedRecordingEvidence && recordingEvidenceRequest && !recordingEvidenceSuspended) return;
    clearTimeout(recordingEvidenceTimer); recordingEvidenceTimer = null;
    recordingEvidenceGeneration++; recordingEvidenceAttempts = 0; recordingEvidenceSuspended = false;
    recordingEvidenceRequest?.controller?.abort();
    if (name !== requestedRecordingEvidence) clearRecordingEvidence();
    requestedRecordingEvidence = name; $("recording-evidence").hidden = false; $("recording-evidence-retry").hidden = true;
    text($("recording-evidence-status"), "Reading the selected recording's saved receipt; not replaying or moving the simulation.");
    readRecordingEvidence();
  }
  $("recording-evidence-retry").onclick = () => selectRecordingEvidence(requestedRecordingEvidence);
  function scheduleStatus(delay = 900) { clearTimeout(statusTimer); if (!document.hidden) statusTimer = setTimeout(poll, delay); }
  function seekSource(time) {
    pendingSourceSeek = time;
    if ($("source").readyState >= 1) {
      try { $("source").pause?.(); $("source").currentTime = time; pendingSourceSeek = null; }
      catch (_) { text($("pose-issue"), "The source video could not seek to this frame. Download it from Files to inspect it."); }
    }
  }
  $("source").addEventListener("loadedmetadata", () => { mediaClockReady = true; if (pendingSourceSeek !== null) seekSource(pendingSourceSeek); updateWindowClocks(); });
  $("source").addEventListener("timeupdate", updateWindowClocks);
  for (const event of ["emptied", "error"]) $("source").addEventListener(event, () => { mediaClockReady = false; updateWindowClocks(); });
  $("pose-next-issue").onclick = () => {
    if (!poseIssues.length) return;
    poseIssueIndex = (poseIssueIndex + 1) % poseIssues.length;
    const issue = poseIssues[poseIssueIndex];
    const label = {unusable_pose: "Unusable pose estimates", long_movement_gap: "Movement gap", unfilled_body_position: "Missing body position"}[issue.kind];
    text($("pose-issue"), ` ${poseIssueIndex + 1}/${poseIssues.length} · ${label} · frames ${issue.first_frame}–${issue.last_frame} · ${number(issue.seek_s, 2)}s`);
    seekSource(issue.seek_s);
  };
  async function showExerciseCoverage(data) {
    if (poseCoverageLoaded || data.profile !== "exercise" || !["ready", "failed", "cancelled"].includes(data.status)) return;
    $("pose-coverage").hidden = false;
    text($("evidence-boundary"), "Body landmarks and proportions are estimates. Muscle bands are illustrative; anatomy, muscle activity and physical contact are not verified.");
    try {
      poseCoverageAttempts++;
      const coverage = await request("/api/exercise-coverage");
      poseCoveragePending = ["pending", "loading"].includes(coverage.status) && poseCoverageAttempts < 10;
      if (poseCoveragePending) { text($("pose-coverage-text"), "Reading the saved tracking evidence…"); return; }
      poseCoverageLoaded = true;
      if (coverage.status !== "available") {
        text($("pose-coverage-text"), "No bounded pose-coverage summary is available. Inspect the source video and saved frame measurements.");
        $("pose-timeline").hidden = $("pose-coverage-legend").hidden = true;
        return;
      }
      const root = coverage.movement;
      const gaps = root.status === "not_available" ? "No saved movement-gap check is available."
        : `Saved image-based movement checks: ${root.long_gaps} long gaps and ${root.unfilled_gaps} unfilled body-position gaps.`;
      text($("pose-coverage-text"), `${coverage.usable_samples}/${coverage.sampled_frames} analyzed frames have usable pose estimates; ${coverage.partial_samples} are partial and ${coverage.unusable_samples} are unusable (${coverage.competing_samples} with competing hypotheses). Decoded ${coverage.decoded_frames} source frames; ${coverage.unsampled_frames} were not analyzed. ${gaps}`);
      const canvas = $("pose-timeline"), ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#965644"; ctx.fillStyle = "#214F50";
      for (const [frame, , kind] of coverage.markers) {
        const x = frame / Math.max(1, coverage.decoded_frames - 1) * (canvas.width - 4) + 2;
        if (["tracked", "tracked_partial"].includes(kind)) {
          const height = kind === "tracked" ? 42 : 21;
          ctx.fillRect(x, canvas.height - height - 10, 1, height);
        } else {
          ctx.beginPath(); ctx.moveTo(x - 2, 22); ctx.lineTo(x + 2, 30);
          ctx.moveTo(x + 2, 22); ctx.lineTo(x - 2, 30); ctx.stroke();
        }
      }
      poseIssues = coverage.issues;
      $("pose-next-issue").disabled = !poseIssues.length;
      if (!poseIssues.length) text($("pose-issue"), " No tracking issues reported; this does not establish physical accuracy.");
    } catch (_) {
      poseCoverageLoaded = true; poseCoveragePending = false;
      text($("pose-coverage-text"), "The coverage summary could not be read. Your source video and saved evidence are unchanged.");
    }
  }
  async function showCoverage(data) {
    if (coverageLoaded || !["appearance", "planar"].includes(data.profile)) return;
    const available = (data.artifacts || []).some(file => file.name === "coverage-summary.json");
    if (!available && !["ready", "failed", "cancelled"].includes(data.status)) return;
    coverageLoaded = true;
    $("coverage-details").hidden = false;
    $("coverage-inspect").hidden = !available;
    if (!available) {
      text($("coverage-text"), "This run has no full-interval coverage report. A preview may represent only part of the video; inspect its saved evidence before judging completeness.");
      return;
    }
    try {
      const coverage = await artifact("coverage-summary.json");
      const [first, last] = coverage.requested_interval;
      const span = coverage.model_interval;
      const model = coverage.status === "planned" && span
        ? `Simulation planned for frames ${span[0]}–${span[1]}, with no restart at gaps.`
        : "No supported simulation interval; tracked evidence is still available.";
      text($("coverage-text"), `Requested: frames ${first}–${last} (${coverage.requested_frames} frames). ${model} ${coverage.pre_model_frames} frames precede model initialization. Measured object positions: ${coverage.measured_observation_slots}/${coverage.observation_slots}; ${coverage.missing_observation_slots} unavailable. Missing positions are not filled in or counted as zero error. Coverage alone does not mean the reconstruction passed.`);
    } catch (_) {
      text($("coverage-text"), "Coverage summary could not be read. Inspect every frame or the saved files; no completeness claim is available.");
    }
  }
  async function poll() {
    if (pollPending) { pollAgain = true; return; }
    pollPending = true; const epoch = mutationEpoch;
    try {
      const data = await request("/api/status"); current = data; text($("status"), data.status === "ready" ? "Local · ready" : `Local · ${data.status}`);
      const interval = data.requested_source_interval;
      $("source-selection").hidden = !interval;
      if (interval) {
        const label = `Requested section: original frames ${interval.first_frame}–${interval.last_frame}. Tracking starts at frame ${interval.initialization_frame}. Earlier frames in this section remain untracked evidence. The full source is preserved; check the decode audit in Files for completion.`;
        if ($("source-selection").textContent !== label) text($("source-selection"), label);
      }
      text($("directory"), data.directory); $("worker-log").href = fileURL("worker.log");
      const events = data.events || [], last = events.at(-1);
      if (last) { text($("stage-label"), last.stage.replaceAll("_", " ")); text($("stage-detail"), last.message || (last.episode ? `Episode ${last.episode}/${last.requested_episodes}` : last.policy ? `${last.policy}: ${last.successes}/${last.episodes}` : "")); }
      const signature = `${events.length}:${last?.index}`;
      if (signature !== lastEvents) { lastEvents = signature; $("events").replaceChildren(); for (const event of events) { const li = document.createElement("li"); text(li, `${typeof event.elapsed_seconds === "number" && Number.isFinite(event.elapsed_seconds) ? event.elapsed_seconds.toFixed(2) + "s · " : ""}${event.stage} · ${event.message || JSON.stringify(event)}`); $("events").append(li); } }
      const stages = {source: 1, observe: 2, tracking: 2, world: 3, repair: 4, evaluate: 5, evaluation: 5, train: 6, training: 6, render: 6, ready: 7};
      $("progress").value = data.status === "ready" ? 7 : stages[last?.stage] || 0;
      files(data); sessions(data); await showCoverage(data); await showExerciseCoverage(data); if (epoch === mutationEpoch && !controlBusy) setRecording(data.recording);
      if (data.status === "ready" && !loadedReady) { loadedReady = true; await readyDetails(data); if (epoch === mutationEpoch && !controlBusy) setRecording(data.recording); }
      await showInspection(data);
      if (data.status === "failed" || data.status === "cancelled") {
        error(data.error?.message || "Pipeline stopped. Inspect the worker log and preserved artifacts.");
        text($("build-status"), "· stopped"); text($("start-help"), "Generation stopped before an interactive environment was ready. Inspect Video for tracking issues and Files for the saved evidence.");
        if ($("tracks").hidden) text($("tracks-empty"), "No tracking preview was saved. Inspect the original video, tracking issues and frame measurements instead.");
      }
      if (!["ready", "failed", "cancelled"].includes(data.status) || data.export?.status === "running" || poseCoveragePending || inspectionPending) scheduleStatus();
    } catch (e) { error(`Cannot reach the local lab: ${e.message}. Keep the CLI process running.`); pause(); }
    finally { pollPending = false; if (pollAgain) { pollAgain = false; scheduleStatus(0); } }
  }
  poll();
})();
