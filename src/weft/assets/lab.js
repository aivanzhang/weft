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
  let exerciseTask = null, exerciseActions = null, exerciseContractReady = false, exerciseStateReady = false;
  let exerciseDetailSignature = "", exercisePolicy = null;
  const inspectionActors = new Map();
  const actorLabel = (id) => inspectionActors.get(id)?.label || scene?.objects?.find(obj => obj.id === id)?.label || id;
  const wholeBody = () => scene?.profile === "exercise-imitation-v1";
  const spatialRigid = () => scene?.profile === "spatial-rigid-v1";
  const sharedWorld = () => scene?.profile === "shared-world-v1";
  const sharedExercise = () => current?.control_profile === "shared_exercise";
  const engineState = state => state?.profile === "shared-exercise-state-v1" ? state.engine_state : state;
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
  async function request(path, data) {
    const response = await fetch(path, {headers: {Authorization: `Bearer ${token}`, ...(data ? {"Content-Type": "application/json"} : {})},
      ...(data ? {method: "POST", body: JSON.stringify(data)} : {}), signal: AbortSignal.timeout(10000)});
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
  }
  for (const button of $("stages").querySelectorAll("button")) button.onclick = () => showPanel(button.dataset.panel);

  function draw(canvas, state) {
    state = engineState(state);
    if (!scene || !state) return;
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
      {name: "3D: coupled world", matrix: [[cy, -sy, 0], [sy * sp, cy * sp, -cp]], depth: [sy * cp, cy * cp, sp]},
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
      if (target && !sharedExercise()) {
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
  function configureExerciseTask(data) {
    exerciseTask = data.exercise_task; exerciseActions = data.action_contract;
    const rootMode = exerciseTask?.control?.root_assistance, expected = rootMode === "disabled" ? 19 : rootMode === "explicit_3_axis" ? 22 : 0;
    const names = exerciseActions?.names;
    exerciseContractReady = exerciseTask?.profile === "shared-exercise-task-v1" && exerciseTask.control.object_forces === "forbidden"
      && expected > 0 && Array.isArray(names) && names.length === expected && new Set(names).size === expected
      && names.every(name => typeof name === "string" && name.length > 0 && name.length < 160)
      && exerciseActions.root_assistance === rootMode && Array.isArray(exerciseActions.object_force_channels) && exerciseActions.object_force_channels.length === 0
      && Number.isInteger(exerciseTask.interval?.horizon_steps) && exerciseTask.interval.horizon_steps > 0
      && Array.isArray(exerciseTask.contact_requirements) && exerciseTask.contact_requirements.length <= 24;
    $("exercise-task-card").hidden = $("exercise-results").hidden = $("mode-coast").hidden = false;
    const duration = exerciseTask?.interval?.horizon_steps * scene.dt;
    text($("exercise-task-scope"), exerciseContractReady
      ? `Track the body reference while every equipment object stays stable over ${number(duration, 2)} seconds (${exerciseTask.interval.horizon_steps} control steps). This declared task interval is not necessarily the whole source video.`
      : "The saved human-only task/control contract is unavailable or inconsistent. Task controls are disabled; no legacy task is substituted.");
    text($("exercise-assistance"), rootMode === "disabled" ? "Root assistance: OFF · 19 joint motors. The body has no virtual root-force controls."
      : rootMode === "explicit_3_axis" ? "Root assistance: 3 explicit virtual controls · 19 joint motors + 3 root-force channels. This is not unassisted balance."
      : "Root assistance: unavailable.");
    const requirements = exerciseTask?.contact_requirements;
    text($("exercise-contact-scope"), Array.isArray(requirements) ? requirements.length
      ? `${requirements.length} declared wrist-region requirements. Only these checks count; other body contacts do not establish hand support or grasp.`
      : "No wrist-region contact is required. Passing means body tracking and equipment stability only—not hand support, a completed exercise or grasp."
      : "Contact requirements are unavailable; none are inferred from the image.");
  }
  const exerciseFeedbackLabels = {coast_unchanged: "Selected coast baseline", authored_unchanged: "Selected authored baseline", searched_gains: "Searched human feedback", searched_joint_residuals: "Searched joint corrections"};
  const exercisePolicyKinds = {"shared-exercise-policy-v1": ["coast_unchanged", "authored_unchanged", "searched_gains"],
    "shared-exercise-policy-v2": ["authored_unchanged", "searched_joint_residuals"]};
  function configureExercisePolicy(data) {
    const policy = data.exercise_policy;
    exercisePolicy = data.learned_policy_available === true && typeof policy?.profile === "string"
      && Object.hasOwn(exercisePolicyKinds, policy.profile)
      && typeof policy.policy_sha256 === "string" && /^[0-9a-f]{64}$/.test(policy.policy_sha256)
      && typeof policy.selected_feedback_kind === "string" && Object.hasOwn(exerciseFeedbackLabels, policy.selected_feedback_kind)
      && exercisePolicyKinds[policy.profile].includes(policy.selected_feedback_kind)
      && policy.privileged_simulator_features === true ? policy : null;
    return exerciseContractReady && exercisePolicy !== null;
  }
  function exerciseFeedbackDescription() {
    if (!exercisePolicy || !learnedPolicyAvailable) return "No validated human-only policy is available.";
    const kind = exercisePolicy.selected_feedback_kind;
    return `${exerciseFeedbackLabels[kind]}. ${kind === "coast_unchanged" ? "Training selected the unchanged zero-command baseline, not new learned gains."
      : kind === "authored_unchanged" ? "Training selected unchanged programmed feedback, not new learned gains."
      : kind === "searched_joint_residuals" ? "Training searched small phase-dependent joint corrections on top of programmed feedback. Any permitted root assistance remains programmed, not learned."
      : "Training searched feedback gains using simulator state and the known body reference."} Equipment has no force controls; root assistance remains the selected task setting.`;
  }
  function exerciseResetDescription() {
    const reset = exercisePolicy?.evaluation_reset_protocol;
    const values = [reset?.human_jitter_scale, reset?.object_position_jitter_world, reset?.object_velocity_jitter_world_s];
    if (!values.every(value => typeof value === "number" && Number.isFinite(value) && value >= 0)) return "Reset protocol unspecified in this saved comparison.";
    if (reset.profile === "shared-exercise-human-jitter-v1" && reset.equipment_reset_state === "unchanged_scene_pose_and_velocity"
      && reset.object_position_jitter_world === 0 && reset.object_velocity_jitter_world_s === 0) {
      return `Changed starts: human-only perturbations (scale ${number(reset.human_jitter_scale, 2)}); equipment keeps its exact scene pose and velocity. This is not a changed-equipment test.`;
    }
    if (reset.profile === "shared-exercise-reset-v1") return `Legacy reset: human perturbation scale ${number(reset.human_jitter_scale, 2)}, equipment position jitter ${number(reset.object_position_jitter_world)} model units and velocity jitter ${number(reset.object_velocity_jitter_world_s)} model units/s. Support-preserving starts are not established.`;
    return "Reset protocol unspecified in this saved comparison.";
  }
  function showExerciseLearning(data) {
    const available = learnedPolicyAvailable && exercisePolicy !== null;
    text($("policy"), available ? exerciseFeedbackLabels[exercisePolicy.selected_feedback_kind] : "Learned controller · unavailable");
    text($("start-learned"), available ? `Try ${exerciseFeedbackLabels[exercisePolicy.selected_feedback_kind].toLowerCase()}` : "Learned controller unavailable");
    text($("exercise-policy-summary"), exerciseFeedbackDescription());
    text($("learning-description"), available ? `${exerciseFeedbackDescription()} The policy interface uses privileged simulator-state and reference features, not video pixels.`
      : "No learned human-only exercise policy is available. The live task panel reports independent whole-interval results, not the legacy target task or a training score.");
    text($("training-caption"), available ? `Training selection score: ${number(exercisePolicy.training_score)}. ${exercisePolicy.training_score_improved_over_authored === true
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
      `${name === "learned" ? exerciseFeedbackLabels[exercisePolicy.selected_feedback_kind] : {coast: "Coast baseline", random: "Random human actions", authored: "Authored feedback"}[name]} · declared-interval task trials passed`]));
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
    const signature = JSON.stringify([m.equipment, m.contact_requirements, m.failed_gates, m.gates_sha256]);
    if (signature === exerciseDetailSignature) return;
    exerciseDetailSignature = signature;
    text($("exercise-gates"), `Reported gate failures: ${m.failed_gates?.length ? m.failed_gates.join(", ") : "none reported so far"}. Gate identity: ${m.gates_sha256 || "unavailable"}. Passing still requires the complete declared interval.`);
    $("exercise-equipment").replaceChildren();
    for (const item of m.equipment) {
      const row = document.createElement("tr");
      for (const value of [actorLabel(item.object_id), `${number(item.displacement_model_units)} model units`, `${number(item.orientation_rad)} rad`,
        `${number(item.speed_model_units_s)} model units/s`, `${number(item.angular_speed_rad_s)} rad/s`,
        item.within_limits === true ? "Yes, so far" : item.within_limits === false ? "No" : "Unavailable"]) cell(row, value);
      $("exercise-equipment").append(row);
    }
    $("exercise-contacts").replaceChildren();
    for (const item of m.contact_requirements) {
      const row = document.createElement("tr");
      const fraction = Number.isFinite(item.qualifying_fraction) ? `${number(item.qualifying_fraction * 100, 1)}% qualifying` : "Qualifying fraction unavailable";
      for (const value of [`${item.wrist} / ${actorLabel(item.object_id)}`, `${(item.kind || "unknown").replaceAll("_", " ")} · steps ${item.start_step}–${item.end_step} (end excluded)`,
        `${fraction} · ${number(item.observed_duration_s)} / ${number(item.required_duration_s)}s observed`,
        `${number(item.max_gap_s)}s`, item.satisfied === true ? "Satisfied" : "Not yet satisfied / failed"]) cell(row, value);
      $("exercise-contacts").append(row);
    }
    if (!m.contact_requirements.length) {
      const row = document.createElement("tr"); cell(row, "No contact requirements in this task"); $("exercise-contacts").append(row);
    }
  }
  $("exercise-metrics-details").ontoggle = () => exerciseDetails(exerciseMetrics());
  function updateExerciseSnapshot(state) {
    snapshot = state;
    const m = state?.task_metrics, geometry = state?.engine_state;
    exerciseStateReady = exerciseContractReady && state?.profile === "shared-exercise-state-v1" && !!geometry?.human && Array.isArray(geometry?.objects)
      && ["current", "last_valid_control_boundary"].includes(state.engine_state_status)
      && typeof state.terminated === "boolean" && typeof state.truncated === "boolean"
      && m?.profile === "shared-exercise-task-v1" && typeof m.completed_interval === "boolean" && typeof m.success === "boolean"
      && Number.isFinite(m.simulated_duration_s) && Number.isFinite(m.declared_duration_s)
      && Number.isInteger(m.observed_substeps) && Number.isInteger(m.valid_substeps)
      && Array.isArray(m.equipment) && m.equipment.length <= 12 && Array.isArray(m.contact_requirements) && m.contact_requirements.length <= 24;
    if (!exerciseStateReady) {
      pause(); text($("exercise-outcome"), "Task result unavailable · controls paused");
      for (const id of ["exercise-duration", "exercise-body-errors", "exercise-equipment-summary", "exercise-gates"]) text($(id), "Unavailable; no zero values or success are inferred.");
      $("exercise-equipment").replaceChildren(); $("exercise-contacts").replaceChildren(); exerciseDetailSignature = "";
      text($("exercise-failure"), "The selected task, control contract or state metrics are unavailable. Any visible geometry is the previous display, not a new verified state."); $("exercise-failure").hidden = false;
      text($("readout"), "Exercise task state unavailable. Reset or reload before continuing.");
      return;
    }
    draw($("live"), geometry);
    text($("exercise-outcome"), exerciseOutcome(m));
    text($("exercise-duration"), `Reported elapsed: ${number(m.simulated_duration_s)} / ${number(m.declared_duration_s)} seconds · ${m.valid_substeps} valid / ${m.observed_substeps} observed substeps · clock: ${m.clock_status || "unavailable"}.`);
    text($("exercise-body-errors"), `Body angular error: mean ${number(m.angular_rmse_rad)} rad / peak ${number(m.peak_angular_rmse_rad)} rad. Root error: mean ${number(m.root_rmse_model_units)} / peak ${number(m.peak_root_rmse_model_units)} model units. Peak floor penetration: ${number(m.max_floor_penetration_model_units)} model units.`);
    const drifts = m.equipment.map(item => item.displacement_model_units);
    const drift = drifts.length && drifts.every(Number.isFinite) ? Math.max(...drifts) : null;
    text($("exercise-equipment-summary"), `Equipment stays passive. Maximum displacement since reset: ${number(drift)} model units across ${m.equipment.length} objects. Detailed rotation, speed and contact checks are below.`);
    const stale = state.engine_state_status === "last_valid_control_boundary";
    $("exercise-failure").hidden = !stale && !m.failure;
    const failure = stale
      ? `Showing last valid geometry at ${number(geometry.time_s)}s after simulation failure; task clock reports ${number(m.simulated_duration_s)}s (${m.clock_status || "unavailable"}). No terminal pose was fabricated. ${m.failure || "Inspect the saved failure."}`
      : m.failure ? `Simulation stopped: ${m.failure}. Reported task time ${number(m.simulated_duration_s)}s; current geometry time ${number(geometry.time_s)}s.` : "";
    text($("exercise-failure"), failure);
    if (failure) { error(failure); simulationFailureMessage = failure; }
    else if (simulationFailureMessage !== null) { if ($("error").textContent === simulationFailureMessage) clearError(); simulationFailureMessage = null; }
    text($("readout"), `${exerciseOutcome(m)} · task ${number(m.simulated_duration_s)}s · geometry ${number(geometry.time_s)}s · equipment is not actuated.`);
    text($("whole-body-status"), exerciseTask?.control?.root_assistance === "disabled" ? "Root assistance disabled; all three root-force commands stay zero."
      : `Explicit virtual root forces [${(geometry.human.mobility?.support_force || []).map(value => number(value, 1)).join(", ")}] · model force units, not measured human forces.`);
    text($("muscle-readout"), "Illustrative muscle bands show motor commands, not measured activation. Contact dots do not automatically qualify as wrist support.");
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
    snapshot = state; draw($("live"), state);
    updateControls();
    if (sharedWorld()) {
      const h = state.human, cm = state.coupled_metrics || {}, root = h?.mobility;
      const hoCount = state.contact_summary?.human_object ?? 0;
      const targetId = scene.task?.object_goal?.target_object_id || state.controlled_id;
      const suffix = cm.failure || (state.success ? "TASK SUCCESS — reset to continue" : state.terminated || state.truncated ? "TRIAL ENDED — reset to continue" : `target: ${state.controlled_id}`);
      text($("readout"), `t ${number(state.time_s, 2)}s · step ${state.steps} · angular RMSE ${number(cm.angular_rmse_rad)} rad · root RMSE ${number(cm.root_rmse_model_units)} · ${targetId} dist ${number(cm.object_distance_world)} · contacts ${state.contact_count ?? 0} (${hoCount} human↔object) · ${suffix}`);
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
    const taskMetrics = exerciseMetrics(), blockedTask = sharedExercise() && (!exerciseContractReady || !exerciseStateReady);
    const ended = snapshot?.terminated || snapshot?.truncated || (sharedExercise() && !!taskMetrics?.failure);
    const success = sharedExercise() ? taskMetrics?.success && taskMetrics.completed_interval && !taskMetrics.failure : snapshot?.success;
    text($("toggle"), running ? "Pause" : ended ? "Start again" : "Start");
    text($("play-state"), !scene ? "Preparing environment" : blockedTask ? "Task unavailable · controls paused" : running ? "Running" : success ? "Goal reached" : ended ? "Trial finished" : "Paused — ready to try");
    if (sharedExercise()) {
      actionNames = ["Release all motors", "Motor +0.25", "Motor −0.25"];
      for (const button of document.querySelectorAll("[data-action]")) {
        if (actionNames[Number(button.dataset.action)]) text(button, actionNames[Number(button.dataset.action)]);
      }
      text($("goal-summary"), "Across the complete declared interval: track the body reference, stay within the body/floor limits, and keep equipment stable. Only explicitly required wrist-region checks count. This is not a target-object task or proof of grasp.");
      text($("mode-description"), mode === "coast" ? "Coast: every joint and root-force command is zero. Equipment remains passive; gravity and simulated contacts still act."
        : mode === "reference" ? `Authored feedback follows the body reference using ${exerciseActions?.names?.length || "the declared"} human channels. This is programmed control, not learned behavior; equipment has no force controls.`
        : mode === "learned" ? exerciseFeedbackDescription()
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
      text($("goal-summary"), `Authored terminal task: match the body reference within configured limits and stop ${actorLabel(goalObj)} in the dashed target at the end of the trial.${scene.task?.object_goal?.require_human_contact ? " A configured simulated human–object contact check also applies." : ""} This is not exercise or grasp success.`);
      text($("mode-description"), mode === "manual" ? (sharedHumanTarget() ? "Human selected: choose any of the 22 human motors and apply +0.25 or −0.25." : `${actorLabel(snapshot?.controlled_id || scene.controlled_id)} selected: use the eight 3D force controls to push or slow it in the shared world.`) : mode === "reference" ? "Authored human feedback drives the 22 body motors along the reference while rigid objects respond purely to gravity and 3D collisions." : "The learned coupled controller coordinates all 22 human motors and bounded 3D object guidance in one MuJoCo model.");
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
    if (sharedExercise() && (!exerciseContractReady || !exerciseStateReady || snapshot.terminated || snapshot.truncated || exerciseMetrics()?.failure || mode === "learned" && !learnedPolicyAvailable)) return;
    requestPending = true;
    stepPromise = (async () => {
    try {
      let payload = {action, steps: count, policy: usePolicy};
      if (sharedExercise()) {
        const vector = exerciseActions.names.map(name => mode === "manual" && name === selectedMotor ? (action === 1 ? .25 : action === 2 ? -.25 : 0) : 0);
        payload = mode === "learned" ? {steps: count, policy: true} : mode === "reference" ? {steps: count, controller: "reference"} : mode === "coast" ? {steps: count, controller: "coast"} : {steps: count, action: vector};
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
    if (sharedExercise() && (!exerciseContractReady || !exerciseStateReady || exerciseMetrics()?.failure || mode === "learned" && !learnedPolicyAvailable)) return;
    usePolicy = mode === "learned"; action = mode === "reference" ? 5 : 0;
    running = true; clearError(); updateControls(); tick(++playbackGeneration);
  }
  function selectMode(value) {
    if (controlBusy) return;
    if (sharedExercise() ? !["manual", "coast", "reference", "learned"].includes(value) || value === "learned" && !learnedPolicyAvailable : value === "coast") return;
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
  const pauseAll = () => { pause(); stopReference(); };
  window.addEventListener("blur", pauseAll);
  document.addEventListener("visibilitychange", () => { if (document.hidden) { pauseAll(); clearTimeout(statusTimer); } else scheduleStatus(0); });
  window.addEventListener("pagehide", () => { pauseAll(); clearTimeout(statusTimer); });

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
    if (sharedExercise()) configureExerciseTask(data);
    const noLearnedPolicy = sharedExercise() ? !configureExercisePolicy(data) : data.learned_policy_available === false || scene.learned_policy_available === false;
    learnedPolicyAvailable = !noLearnedPolicy;
    if (data.capabilities) updateCapabilities(data.capabilities);
    if (data.evidence_provenance?.layers) {
      const prov = data.evidence_provenance, layers = prov.layers, src = prov.source_evidence || {};
      const missing = src.missing_or_uncertain_summary ? ` · Gaps/uncertainty: ${src.missing_or_uncertain_summary}.` : "";
      const muscles = layers.muscle_display && layers.muscle_display !== "not_applicable" ? ` · Muscles: ${layers.muscle_display.replaceAll("_", " ")}.` : "";
      $("evidence-layers").hidden = false;
      text($("evidence-layers-text"), `Layer separation — Source: ${layers.recorded_source}${src.frame_count ? ` (${src.frame_count} frames)` : ""}.${missing} Kinematic projection: ${layers.kinematic_projection}. Controllers: ${(layers.controllers || []).join(", ")}. Physics: ${layers.physical_simulation}.${muscles}`);
    }
    if (!sharedExercise() && !data.snapshot?.human && learnedPolicyAvailable && !trainedActor) trainedActor = scene.controlled_id;
    const controllable = sharedExercise() ? [{id: "human", label: "Human · equipment stays passive"}] : sharedWorld()
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
      text($("evidence-boundary"), "Coupled shared-world-v1 run. Source slot conservation, coordinate scale bridge, and kinematic preflight are explicit; physical validation remains unverified.");
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
      text($("world-description"), "Rigid 3D objects share meter-scale coordinates, quaternions, contacts and MuJoCo dynamics. Depth, shape, materials, controls and task remain explicit assumptions; physical validation is unverified.");
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
      const extra = typeof score.mean_final_distance_m === "number" ? ` · mean return ${number(score.mean_return)} · final dist ${number(score.mean_final_distance_m)} m` : "";
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
  function applyInspection(data) {
    const source = inspection.source, tracking = inspection.tracking;
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
    renderObjectDetails(); files(data); showGeometryPanel(); updateControls();
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
        text($("inspection-status"), `Saved inspection summary unavailable: ${result.reason || "no supported metadata"}. Original files and simulation controls are unchanged.`);
        text($("inspection-tracking"), "No validated compact tracking summary is available. Use the saved source files; no completeness claim is made.");
        return;
      }
      inspection = result; applyInspection(data);
    } catch (_) {
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
        if (session.complete) for (const [label, file] of [["Interactive replay", "replay.html"], ["Video", "video.mp4"], ["Preview", "preview.gif"]]) li.append(link(label, `sessions/${session.name}/${file}`));
        $("recordings").append(li);
      }
    }
    exporting = data.export?.status === "running";
    if (data.export) { text($("export-status"), data.export.status === "ready" ? "Recording exported and verified. Download the offline replay or video below." : data.export.status === "failed" ? `Export failed: ${data.export.error}. The action journal is preserved.` : "Exporting recorded states to video and an offline replay…"); }
  }
  function scheduleStatus(delay = 900) { clearTimeout(statusTimer); if (!document.hidden) statusTimer = setTimeout(poll, delay); }
  function seekSource(time) {
    pendingSourceSeek = time;
    if ($("source").readyState >= 1) {
      try { $("source").pause?.(); $("source").currentTime = time; pendingSourceSeek = null; }
      catch (_) { text($("pose-issue"), "The source video could not seek to this frame. Download it from Files to inspect it."); }
    }
  }
  $("source").addEventListener("loadedmetadata", () => { if (pendingSourceSeek !== null) seekSource(pendingSourceSeek); });
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
