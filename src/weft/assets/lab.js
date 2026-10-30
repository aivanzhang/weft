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
  const wholeBody = () => scene?.profile === "exercise-imitation-v1";
  const spatialRigid = () => scene?.profile === "spatial-rigid-v1";
  const dynamicObjects = () => (scene?.objects || []).filter((obj) => obj.role !== "static");
  const number = (value, places = 3) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(places) : "unavailable";
  const held = new Set();
  const keyActions = {ArrowLeft: 1, a: 1, ArrowRight: 2, d: 2, ArrowUp: 3, w: 3, ArrowDown: 4, s: 4, " ": 5};
  const spatialKeys = {ArrowLeft: 1, a: 1, A: 1, ArrowRight: 2, d: 2, D: 2, ArrowUp: 3, w: 3, W: 3, ArrowDown: 4, s: 4, S: 4, q: 5, Q: 5, PageUp: 5, e: 6, E: 6, PageDown: 6, " ": 7};
  const keyAction = (key) => spatialRigid() ? spatialKeys[key] : keyActions[key] === undefined ? undefined : wholeBody() && key === " " ? 0 : current?.profile === "exercise" ? (key === " " ? 3 : ["ArrowLeft", "a", "ArrowUp", "w"].includes(key) ? 1 : 2) : keyActions[key];
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
  }
  for (const button of $("stages").querySelectorAll("button")) button.onclick = () => showPanel(button.dataset.panel);

  function draw(canvas, state) {
    if (!scene || !state) return;
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
  function updateSnapshot(state) {
    if (state.simulation_failure) {
      const message = `Simulation failed: ${state.simulation_failure.reason || "unknown dynamics failure"}. Showing the last valid state, not a fabricated terminal pose. Reset before continuing.`;
      error(message); simulationFailureMessage = message;
    } else if (simulationFailureMessage !== null) {
      if ($("error").textContent === simulationFailureMessage) clearError();
      simulationFailureMessage = null;
    }
    snapshot = state; draw($("live"), state);
    updateControls();
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
      const suffix = state.success ? "TASK SUCCESS — reset to continue" : state.truncated ? "TIME LIMIT — reset to continue" : state.terminated ? "TRIAL ENDED — reset to continue" : `action: ${usePolicy ? "learned policy" : actionNames[action]}`;
      text($("readout"), `t ${(state.time_s || 0).toFixed(3)}s · step ${state.steps} · ${body.id} pos [${body.position.map((v) => v.toFixed(3)).join(", ")}] m · speed ${Math.hypot(...body.velocity).toFixed(3)} m/s · quat [${body.quaternion.map((v) => v.toFixed(3)).join(", ")}] · spin [${body.angular_velocity.map((v) => v.toFixed(2)).join(", ")}] rad/s · contacts ${state.contact_count ?? 0} · substeps ${state.substeps ?? 0} · ${suffix}`);
      const objectLines = state.objects.map((o) => `${o.id} (${o.role}, ${o.shape}): pos [${o.position.map((v) => v.toFixed(3)).join(", ")}] · vel [${o.velocity.map((v) => v.toFixed(3)).join(", ")}] · quat [${o.quaternion.map((v) => v.toFixed(3)).join(", ")}] · spin [${o.angular_velocity.map((v) => v.toFixed(2)).join(", ")}]`);
      const contactLines = (state.contact_details || []).length
        ? state.contact_details.map((c) => `contact ${c.objects.join("↔")} @ [${c.position.map((v) => v.toFixed(3)).join(", ")}] dist ${c.distance.toFixed(4)}m Fn ${c.normal_force.toFixed(2)}N`)
        : ["contacts: none"];
      $("spatial-readout").hidden = false;
      text($("spatial-readout"), [`virtual force (${controlledId}): [${(state.virtual_force || [0, 0, 0]).map((v) => v.toFixed(2)).join(", ")}] N`, ...objectLines, ...contactLines].join("\n"));
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
    const ended = snapshot?.terminated || snapshot?.truncated;
    text($("toggle"), running ? "Pause" : ended ? "Start again" : "Start");
    text($("play-state"), !scene ? "Preparing environment" : running ? "Running" : snapshot?.success ? "Goal reached" : ended ? "Trial finished" : "Paused — ready to try");
    if (snapshot?.human) {
      const h = snapshot.human, degrees = value => (value * 180 / Math.PI).toFixed(0);
      text($("goal-summary"), wholeBody() ? "Track the known full-body reference using 19 joint motors and 3 virtual root-force controls. Tracking and floor-contact errors determine success; this is not learned balance." : mode === "reference" ? "Follow the video’s estimated joint movement. This controller is programmed, not learned." : `${h.selected_joint.replaceAll("_", " ")}: ${degrees(h.angles[h.selected_joint])}° now → ${degrees(h.target_angle)}° target. Reach the target and slow to a stop.`);
    } else if (spatialRigid()) text($("goal-summary"), `Move ${snapshot?.controlled_id || scene.controlled_id} into the 3D dashed target [${(snapshot?.target || scene.target).map((v) => v.toFixed(2)).join(", ")}] m and slow below ${scene.task.speed_tolerance} m/s.`);
    else if (scene) text($("goal-summary"), dynamicObjects().length > 1 ? `Move ${snapshot?.controlled_id || scene.controlled_id} into the dashed target and slow to a stop.` : "Move the highlighted object into the dashed target and slow to a stop.");
    text($("mode-description"), wholeBody() ? (mode === "manual" ? "Choose one of 22 motors. Apply +0.25 or −0.25 of its force limit; every other motor stays at zero. Release sets all motors to zero." : mode === "reference" ? "Authored feedback follows the known full-body reference. Every joint torque and virtual root force is explicit; this baseline is not learned." : "The learned full-body controller follows a known reference using 19 joint torques and 3 virtual root forces. It has not learned unassisted balance or human intent.") : mode === "manual" ? "You control the movement. Tap a direction for a small move, or start and hold a direction." : mode === "reference" ? "A programmed controller follows the video’s estimated poses. This is simulated movement, not learned exercise behavior." : current?.profile === "exercise" ? `The learned controller moves the ${(trainedJoint || "trained joint").replaceAll("_", " ")} toward one target angle. It has not learned the whole exercise.` : "The learned controller attempts the target task. Watch its choices, or pause to inspect a step.");
    for (const [id, value] of [["mode-manual", "manual"], ["reference-servo", "reference"], ["policy", "learned"]]) {
      $(id).setAttribute("aria-pressed", String(mode === value));
      $(id).disabled = !scene || controlBusy || (value === "learned" && (!learnedPolicyAvailable || (current?.profile === "exercise" && (wholeBody() ? !wholeBodyPolicyReady : !trainedJoint))));
    }
    for (const button of document.querySelectorAll("[data-action]")) {
      const code = Number(button.dataset.action);
      const isSpatialExtra = button.dataset.spatial === "true";
      const spatialMismatch = isSpatialExtra && !spatialRigid();
      const mobileOnly = !isSpatialExtra && code >= 6 && scene?.profile !== "exercise-v3";
      const incompatible = wholeBody() && ![0, 1, 2].includes(code);
      const exerciseServo = current?.profile === "exercise" && code === 5;
      button.disabled = !scene || controlBusy || mode !== "manual" || !!ended || spatialMismatch || mobileOnly || incompatible || (current?.profile === "rolling" && code > 2) || exerciseServo;
      button.hidden = spatialMismatch || mobileOnly || incompatible || exerciseServo;
    }
    $("joint").disabled = (recording && !wholeBody()) || controlBusy || mode !== "manual";
    $("actor").disabled = recording || controlBusy || mode !== "manual";
    $("record").disabled = recording || exporting || !scene || controlBusy;
    $("stop-record").disabled = !recording || controlBusy;
    for (const id of ["toggle", "reset", "step"]) $(id).disabled = !scene || controlBusy || (id === "step" && !!ended);
    text($("record-status"), recording ? `● Recording · ${(snapshot?.time_s || 0).toFixed(1)}s of simulation` : exporting ? "Saving recording…" : "Not recording");
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
    requestPending = true;
    stepPromise = (async () => {
    try {
      let payload = {action, steps: count, policy: usePolicy};
      if (wholeBody()) {
        const names = scene.action_contract.names;
        const vector = names.map(name => name === selectedMotor ? (action === 1 ? .25 : action === 2 ? -.25 : 0) : 0);
        payload = mode === "learned" ? {steps: count, policy: true} : mode === "reference" ? {steps: count, controller: "reference"} : {steps: count, action: vector};
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
    usePolicy = mode === "learned"; action = mode === "reference" ? 5 : 0;
    running = true; clearError(); updateControls(); tick(++playbackGeneration);
  }
  function selectMode(value) {
    if (controlBusy) return;
    pause(); mode = value; usePolicy = mode === "learned"; updateControls();
  }
  async function mutate(path, data = {}, prepareLearned = false) {
    if (controlBusy) return;
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
      if (prepareLearned && mode === "learned" && !snapshot?.human && trainedActor && (snapshot?.controlled_id || scene?.controlled_id) !== trainedActor) {
        if (recording) throw new Error("Stop & save this recording before switching to the trained object.");
        const switched = await request("/api/actor", {actor: trainedActor});
        scene = switched.scene; if (switched.capabilities) updateCapabilities(switched.capabilities);
        updateSnapshot(switched.snapshot); $("actor").value = trainedActor;
      }
      const result = await request(path, data);
      if (result.capabilities) updateCapabilities(result.capabilities);
      if (result.scene) {
        scene = result.scene;
        if (!snapshot?.human && $("actor").children.length) $("actor").value = result.scene.controlled_id;
        if (!referenceState && (result.snapshot || snapshot)) draw($("world-canvas"), result.snapshot || snapshot);
      }
      if (result.snapshot) updateSnapshot(result.snapshot);
      if ("recording" in result) setRecording(result.recording);
      if (result.export === "running") exporting = true;
      scheduleStatus(0); return result;
    } catch (e) {
      if (snapshot?.human && !wholeBody()) $("joint").value = snapshot.human.selected_joint;
      else if (!snapshot?.human && scene?.controlled_id && $("actor").children.length) $("actor").value = snapshot?.controlled_id || scene.controlled_id;
      error(e.message);
    }
    finally { requestPending = false; controlBusy = false; updateControls(); }
  }
  async function begin({fresh = false, record = false} = {}) {
    if (controlBusy) return;
    const needsJoint = mode === "learned" && snapshot?.human && !wholeBody() && snapshot.human.selected_joint !== trainedJoint;
    const needsActor = mode === "learned" && !snapshot?.human && Boolean(trainedActor) && (snapshot?.controlled_id || scene?.controlled_id) !== trainedActor;
    if (fresh || record || needsJoint || needsActor || snapshot?.terminated || snapshot?.truncated) {
      const pending = mutate(record ? "/api/record/start" : "/api/reset", {}, true);
      const generation = playbackGeneration;
      if (!await pending || generation !== playbackGeneration) return;
    }
    start();
  }
  $("toggle").onclick = () => running ? pause() : begin();
  $("mode-manual").onclick = () => selectMode("manual");
  $("policy").onclick = () => selectMode("learned");
  $("reference-servo").onclick = () => selectMode("reference");
  $("step").onclick = async () => {
    pause();
    const needsJoint = mode === "learned" && snapshot?.human && !wholeBody() && snapshot.human.selected_joint !== trainedJoint;
    const needsActor = mode === "learned" && !snapshot?.human && Boolean(trainedActor) && (snapshot?.controlled_id || scene?.controlled_id) !== trainedActor;
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
    if (!wholeBody()) return mutate("/api/joint", {joint: $("joint").value});
    if (controlBusy || mode !== "manual") { $("joint").value = selectedMotor; return; }
    pause(); selectedMotor = $("joint").value; chooseAction(0); updateControls();
  };
  $("actor").onchange = () => mutate("/api/actor", {actor: $("actor").value});
  $("start-manual").onclick = () => { showPanel("play"); selectMode("manual"); $("live").focus(); };
  $("start-learned").onclick = () => { showPanel("play"); selectMode("learned"); begin({fresh: true}); };
  $("start-reference").onclick = () => showPanel("world");
  function redrawViews() {
    if (snapshot) draw($("live"), snapshot);
    if (referenceState) draw($("world-canvas"), referenceState);
    else if (spatialRigid() && snapshot) draw($("world-canvas"), snapshot);
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
    if (mode !== "manual" || controlBusy || (wholeBody() && ![0, 1, 2].includes(value))) return;
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
    const fullBody = wholeBody();
    ctx.fillText(fullBody ? "Best training return by generation (not evaluation)" : "Training success · last 100–500 episodes (not evaluation)", 20, 23);
    if (!history.length) return;
    ctx.strokeStyle = "#D8C9BA"; ctx.beginPath(); ctx.moveTo(45, 40); ctx.lineTo(45, 205); ctx.lineTo(940, 205); ctx.stroke();
    ctx.strokeStyle = "#965644"; ctx.lineWidth = 2; ctx.beginPath();
    const max = fullBody ? history.at(-1).generation + 1 : history.at(-1).episode;
    history.forEach((row, i) => { const x = 45 + (fullBody ? row.generation + 1 : row.episode) / max * 890, y = 205 - (fullBody ? row.best_training_return : row.recent_training_success_rate) * 160; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
    ctx.fillText(fullBody ? "0" : "0%", 8, 205); ctx.fillText(fullBody ? "1" : "100%", 4, 45); ctx.fillText(`${max} ${fullBody ? "generations" : "episodes"}`, 820, 230);
  }
  async function readyDetails(data) {
    scene = data.scene;
    const noLearnedPolicy = data.learned_policy_available === false || scene.learned_policy_available === false;
    learnedPolicyAvailable = !noLearnedPolicy;
    if (data.capabilities) updateCapabilities(data.capabilities);
    if (!data.snapshot?.human && learnedPolicyAvailable && !trainedActor) trainedActor = scene.controlled_id;
    const controllable = dynamicObjects();
    $("actor-controls").hidden = Boolean(data.snapshot?.human) || controllable.length < 2;
    $("actor").replaceChildren();
    for (const obj of controllable) {
      const option = document.createElement("option");
      option.value = obj.id;
      text(option, `${obj.label || obj.id} (${obj.id})`);
      $("actor").append(option);
    }
    if (controllable.length) $("actor").value = data.snapshot?.controlled_id || scene.controlled_id;
    if (data.profile === "exercise" && (wholeBody() || noLearnedPolicy)) {
      const feasibility = data.reference_feasibility, summary = feasibility?.summary;
      $("reference-feasibility").hidden = false;
      const label = feasibility?.status === "infeasible" ? "Reference has floor conflicts; this sandbox is a diagnostic simulation." : feasibility?.status === "feasible" ? "Reference passed the configured floor checks; this sandbox is a diagnostic simulation, not validated physical reconstruction." : "Reference floor feasibility is unavailable; this sandbox is a diagnostic simulation.";
      const detail = summary ? ` Maximum reference penetration: ${number(summary.max_penetration_model_units)} model units. Conflicting source poses: ${summary.failed_source_poses}/${summary.source_pose_checks}; conflicting sampled control poses: ${summary.failed_control_grid_poses}/${summary.control_grid_checks}. ${noLearnedPolicy ? "No learned policy is available for this candidate." : "Selected RL window results remain separate."}` : "";
      const gap = data.ground_clearance?.summary;
      const clearance = gap ? ` Reference body–ground gap: minimum ${number(gap.minimum)}, median ${number(gap.median)}, maximum ${number(gap.maximum)} model units. Contact timing has not been inferred.` : "";
      text($("feasibility-notice"), label + detail + clearance);
    }
    updateSnapshot(data.snapshot); draw($("world-canvas"), data.snapshot);
    $("build-details").open = false; text($("build-status"), "· complete");
    text($("start-help"), "Ready. Start with the captured movement, or jump straight into the sandbox.");
    for (const id of ["start-reference", "start-manual", "start-learned", "mode-manual"]) $(id).disabled = id === "start-learned" && (!learnedPolicyAvailable || data.profile === "exercise");
    $("scene-json").dataset.json = data.profile === "rolling" ? "world.json" : "scene.json";
    $("reconstruction-json").dataset.json = data.profile === "rolling" ? "report.json" : "reconstruction.json";
    if (data.profile === "rolling") text($("control-help"), "Left/right arrows or A/D to push. Release to coast. This profile has horizontal control only.");
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
    $("objects").replaceChildren();
    for (const obj of scene.objects) { const row = document.createElement("tr"); cell(row, obj.label || obj.id); cell(row, obj.role); cell(row, obj.shape); cell(row, JSON.stringify(obj.provenance || (data.profile === "exercise" ? "Simplified body segment; inferred proportions and authored physics" : "Measured geometry; authored actuator"))); $("objects").append(row); }
    $("assumptions").replaceChildren(); for (const assumption of scene.assumptions || []) { const li = document.createElement("li"); text(li, assumption); $("assumptions").append(li); }
    const results = await Promise.allSettled([noLearnedPolicy ? Promise.resolve(null) : artifact("training.json"), noLearnedPolicy ? Promise.resolve(null) : artifact("rl-evaluation.json"), artifact("repair-ledger.json"), artifact(data.profile === "rolling" ? "report.json" : "reconstruction.json")]);
    if (results[0].status === "fulfilled" && results[0].value) { const training = results[0].value; curve(training); text($("training-caption"), wholeBody() ? `${training.completed_generations} completed generations · ${number(training.seconds, 2)} seconds${training.budget_exhausted ? " · stopped at local time budget" : ""}` : `${training.completed_episodes ?? training.episodes} training episodes · ${training.steps} steps · ${Number(training.seconds).toFixed(2)} seconds${training.budget_exhausted ? " · stopped at local time budget" : ""}`); }
    if (results[1].status === "fulfilled" && results[1].value) metrics($("scores"), Object.entries(results[1].value.policies).map(([name, score]) => {
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
    updateControls();
  }
  function files(data) {
    const entries = data.artifacts || [], signature = entries.map((item) => item.name).join("|");
    if (signature === lastFiles) return; lastFiles = signature;
    $("files").replaceChildren();
    for (const file of entries) {
      const row = document.createElement("tr"); const name = cell(row, ""); name.append(link(file.name, `run/${file.name}`));
      cell(row, `${(file.bytes / 1024).toFixed(1)} KB`); const inspectCell = cell(row, "");
      if (file.name.endsWith(".json")) { const button = document.createElement("button"); button.textContent = "Inspect"; button.onclick = () => inspect(file.name); inspectCell.append(button); }
      $("files").append(row);
    }
    const names = entries.map((item) => item.name);
    const sourceName = names.find((name) => /^source\.(mp4|webm|mov|m4v|avi|mkv)$/.test(name));
    if (sourceName && !$("source").getAttribute("src")) $("source").src = fileURL(`run/${sourceName}`);
    const preview = names.includes("tracks.gif") ? "tracks.gif" : names.includes("replay-slow.gif") ? "replay-slow.gif" : null;
    if (preview) { $("tracks").src = fileURL(`run/${preview}`); $("tracks").hidden = false; $("tracks-empty").hidden = true; }
    const replay = names.includes("reconstruction.gif") ? "reconstruction.gif" : names.includes("replay-slow.gif") ? "replay-slow.gif" : names.includes("tracks-contact-sheet.png") ? "tracks-contact-sheet.png" : null;
    if (replay) { $("reconstruction-preview").src = fileURL(`run/${replay}`); $("reconstruction-preview").hidden = false; }
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
      if (data.status === "failed" || data.status === "cancelled") {
        error(data.error?.message || "Pipeline stopped. Inspect the worker log and preserved artifacts.");
        text($("build-status"), "· stopped"); text($("start-help"), "Generation stopped before an interactive environment was ready. Inspect Video for tracking issues and Files for the saved evidence.");
        if ($("tracks").hidden) text($("tracks-empty"), "No tracking preview was saved. Inspect the original video, tracking issues and frame measurements instead.");
      }
      if (!["ready", "failed", "cancelled"].includes(data.status) || data.export?.status === "running" || poseCoveragePending) scheduleStatus();
    } catch (e) { error(`Cannot reach the local lab: ${e.message}. Keep the CLI process running.`); pause(); }
    finally { pollPending = false; if (pollAgain) { pollAgain = false; scheduleStatus(0); } }
  }
  poll();
})();
