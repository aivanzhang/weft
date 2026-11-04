((root) => {
  const $ = (id) => document.getElementById(id);
  let token = root.WeftDemoSession.token;
  const auth = { Authorization: "Bearer " + token };
  const signed = (path) => path + (path.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token);
  const MINE = "weft_demo_runs";

  async function api(path, options = {}) {
    const response = await fetch(path, { ...options, headers: { ...auth, ...(options.headers || {}) } });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      if ((response.status === 401 || response.status === 403) && path !== "/api/demo/unlock") {
        $("auth-warning").hidden = false;
        $("auth-warning").textContent = "Your session needs to reconnect. Enter your demo code below and select Unlock.";
        $("gate-panel").hidden = false;
        $("builder").hidden = true;
      }
      throw new Error(data.error || "That request did not complete.");
    }
    return data;
  }

  const seconds = (value) => {
    const total = Math.max(0, Math.round(value || 0));
    return total < 60 ? `${total}s` : `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, "0")}s`;
  };

  /* Plain-language meaning for every badge a judge will see. */
  const GLOSSARY = {
    "API calls": "How many paid provider calls this step made. Zero means it ran entirely on this machine.",
    "kept": "What the step retained as evidence instead of discarding once it had an answer.",
    "physics": "The simulator the world runs in. MuJoCo is the standard engine for robotics and RL research.",
    "scope": "Exactly what was modelled — and by omission, what was not.",
    "validation": "Whether the world has been checked against reality. Unverified means it is a conditional model.",
    "gates": "Success thresholds fixed before the run so a result cannot be made to pass after the fact.",
    "characters ready": "How many compatible characters are loaded and can be swapped onto the physics.",
    "control steps": "How many times the controller acted. Each step advances the simulation.",
    "controller": "What drove the body. 'reference' replays the authored controller derived from your clip.",
    "tracking error": "How far the simulated body drifted from the observed motion, in radians. Lower is better.",
    "swap": "The control step where the character was replaced while the recording continued.",
    "task success": "Whether every frozen gate passed for this rollout.",
    "failed gates": "Which thresholds were missed. 'none' means all gates passed.",
    "model": "The Meta model that reviewed this world.",
    "hypotheses": "Distinct testable claims Muse made about what the world might have got wrong.",
    "open questions": "What Muse said it could not settle from the evidence it was given.",
    "applied": "Whether the model's proposals changed the world. They never do automatically.",
    "API cost": "The actual recorded charge for this call, from an append-only ledger.",
    "imported": "The character asset that was packed and loaded into the chooser.",
    "live generation": "Whether text-to-character generation ran, and under which spending gate.",
    "outcome": "How the step ended when it did not complete normally.",
  };

  const chipRow = (entries) => {
    const row = document.createElement("div");
    row.className = "chips";
    for (const [key, value] of entries) {
      const chip = document.createElement("span");
      chip.className = "chip";
      const note = GLOSSARY[key];
      if (note) {
        chip.classList.add("has-note");
        chip.title = note;
      }
      if (key) chip.append(document.createTextNode(key + " "));
      const strong = document.createElement("b");
      strong.textContent = String(value);
      chip.append(strong);
      row.append(chip);
    }
    return row;
  };

  let META = {};
  async function loadMeta() {
    try {
      const data = await api("/api/demo/steps");
      META = Object.fromEntries((data.steps || []).map((step) => [step.id, step]));
    } catch (error) { META = {}; }
  }

  function renderSteps(list, steps, run) {
    if (list.children.length !== steps.length) {
      list.replaceChildren(...steps.map(() => document.createElement("li")));
    }
    steps.forEach((runtime, index) => {
      const step = { ...(META[runtime.id] || {}), ...runtime };
      const item = list.children[index];
      item.className = "step " + (step.status || "pending");
      item.replaceChildren();

      const head = document.createElement("div");
      head.className = "step-head";
      const title = document.createElement("h3");
      title.textContent = step.title || step.id;
      head.append(title);
      if (step.experimental) {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.title = "Shipped as experimental: it can refuse its own output, and you will see that refusal.";
        tag.textContent = "EXPERIMENTAL";
        head.append(tag);
      }
      const spent = step.elapsed_s ?? step.elapsed_seconds;
      const budget = step.estimate_s ?? step.estimate_seconds;
      const time = document.createElement("span");
      time.className = "step-time";
      time.textContent = spent ? seconds(spent)
        : step.status === "running" ? "running…" : budget ? `~${seconds(budget)}` : "";
      head.append(time);
      item.append(head);

      for (const [field, cls] of [["explainer", "explain"], ["why", "matters"], ["matters", "matters"],
                                  ["impact", "impact"], ["detail", "detail"]]) {
        if (!step[field]) continue;
        const p = document.createElement("p");
        p.className = cls;
        p.textContent = step[field];
        item.append(p);
      }
      const metrics = step.metrics && Object.entries(step.metrics);
      if (metrics && metrics.length) item.append(chipRow(metrics));
      if (step.artifact && step.artifact.name) {
        const link = document.createElement("a");
        link.className = "download";
        link.href = signed(`/api/demo/artifact/${step.id}` + (run ? `/${run}` : ""));
        link.download = step.artifact.name;
        const bytes = step.artifact.bytes || 0;
        link.textContent = `↓ ${step.artifact.name} · ` + (bytes > 1048576
          ? (bytes / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(bytes / 1024)) + " KB");
        item.append(link);
      }
    });
  }

  /* ---------------- gate ---------------- */
  function unlocked() {
    $("gate-panel").hidden = true;
    $("builder").hidden = false;
    renderMine();
  }

  $("gate-code").value = root.WeftDemoSession.accessCode;

  $("gate-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    $("gate-submit").disabled = true;
    $("gate-error").textContent = "";
    try {
      const session = await api("/api/demo/unlock", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: $("gate-code").value }),
      });
      token = session.token;
      root.WeftDemoSession.setToken(token);
      auth.Authorization = "Bearer " + token;
      $("auth-warning").hidden = true;
      try { sessionStorage.setItem("weft_demo_unlocked", "yes"); } catch { /* Storage is optional. */ }
      keepToken();
      unlocked();
      loadDemo();
    } catch (error) {
      $("gate-error").textContent = error.message;
    } finally {
      $("gate-submit").disabled = false;
    }
  });

  /* ---------------- my runs (this browser only) ---------------- */
  const readMine = () => {
    try { return JSON.parse(localStorage.getItem(MINE) || "[]"); } catch (error) { return []; }
  };
  const saveMine = (rows) => {
    try { localStorage.setItem(MINE, JSON.stringify(rows.slice(0, 24))); } catch (error) { /* private mode */ }
  };

  /* ---------------- finished-run carousel ---------------- */
  let demos = [];
  let featured = [];
  let current = 0;

  async function loadFeatured() {
    try {
      featured = (await api("/api/demo/featured")).featured || [];
    } catch (error) { featured = []; }
  }

  async function loadGallery(force) {
    try {
      const data = await api("/api/demo/gallery");
      const all = Array.isArray(data) ? data : (data.items || data.demos || []);
      // Seeded clips always show; a judge's own run shows once it finished with a video.
      const next = all.filter((row) => !String(row.id || "").startsWith("run-")
                                       || featured.includes(row.id));
      const same = next.length === demos.length && next.every((d, i) => d.id === demos[i]?.id);
      if (same && !force) return;
      demos = next;
      current = Math.min(current, Math.max(0, demos.length - 1));
      showDemo();
    } catch (error) {
      /* the carousel is optional; the walkthrough is the point */
    }
  }

  function showDemo() {
    const available = demos.length > 0;
    $("finished-content").hidden = !available;
    $("finished-nav").hidden = !available;
    $("finished-empty").hidden = available;
    if (!available) {
      $("finished-empty").textContent = "No finished runs yet. Build a world above to create one.";
      return;
    }
    const demo = demos[current];
    // Seek a fraction in: a paused video with no poster otherwise shows a black frame.
    $("demo-video").src = signed(demo.video_url || `/api/demo/gallery/${demo.id}/video`) + "#t=0.1";
    const mine = /^run-/.test(demo.id || "") || ["judge", "run", "yours", "studio"].includes(demo.kind);
    $("demo-source").textContent = (mine ? "your run" : "verified run").toUpperCase();
    $("demo-title").textContent = demo.title;
    $("demo-caption").textContent = demo.caption || "";
    $("demo-metrics").textContent = demo.metrics && typeof demo.metrics === "object"
      ? Object.entries(demo.metrics).map(([k, v]) => `${k} ${v}`).join(" · ")
      : demo.metrics || "";
    $("open-studio").hidden = false;
    $("open-studio").textContent = mine ? "Open run" : "Watch recording";
    $("open-studio").onclick = () => openStudio(demo.id);
    $("dots").replaceChildren(...demos.map((_, index) => {
      const dot = document.createElement("button");
      dot.className = "dot";
      dot.type = "button";
      dot.setAttribute("aria-label", `Run ${index + 1}`);
      dot.setAttribute("aria-current", String(index === current));
      dot.onclick = () => { current = index; showDemo(); };
      return dot;
    }));
  }

  $("prev").addEventListener("click", () => {
    if (demos.length) { current = (current - 1 + demos.length) % demos.length; showDemo(); }
  });
  $("next").addEventListener("click", () => {
    if (demos.length) { current = (current + 1) % demos.length; showDemo(); }
  });

  function renderMine() {
    const rows = readMine();
    $("mine").hidden = rows.length === 0;
    $("mine-list").replaceChildren(...rows.map((row) => {
      const item = document.createElement("li");
      const main = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = row.title || "Your run";
      const note = document.createElement("span");
      note.textContent = row.note || "";
      main.append(title, note);
      const open = document.createElement("button");
      open.type = "button";
      open.className = "ghost";
      open.textContent = "Open studio";
      open.onclick = () => openStudio(row.id);
      item.append(main, open);
      return item;
    }));
  }

  $("demo-video").addEventListener("play", () => {
    $("demo-video").pause();
    openStudio(demos[current]?.id, "recording");
  });

  /* ---------------- studio modal ---------------- */
  let world = null;
  const focusViewer = root.WeftRunFocus();
  let studioRequest = 0;
  function closeStudio() {
    studioRequest += 1;
    $("studio-modal").hidden = true;
    document.body.classList.remove("locked");
    focusViewer.close();
    world = null;
  }
  $("studio-modal").addEventListener("click", (event) => {
    if (event.target.hasAttribute("data-close")) closeStudio();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("studio-modal").hidden) closeStudio();
  });

  async function openStudio(identity, view) {
    if (!identity) return;
    const request = ++studioRequest;
    try {
      const selected = demos.find((item) => item.id === identity);
      const record = /^run-/.test(identity)
        ? await api(`/api/demo/studio/${encodeURIComponent(identity)}`)
        : { id: identity, title: selected?.title, caption: selected?.caption, metrics: selected?.metrics, steps: [] };
      if (request !== studioRequest) return;
      $("studio-error").hidden = true;
      $("studio-modal").hidden = false;
      document.body.classList.add("locked");
      $("studio-title").textContent = record.title || "Inside this run";
      $("studio-note").textContent = record.caption || "";
      $("studio-results").replaceChildren(chipRow(root.WeftRunResults(record)));
      const result = record.result || {};
      world = result.project || result.child || record.project || null;
      const entry = demos.find((item) => item.id === identity);
      focusViewer.open({ world, video: entry?.video_url ? signed(entry.video_url) : null, view });
      if (world) {
        $("play-full").href = `/labs/${world}/#token=${encodeURIComponent(token)}`;
        $("readout").replaceChildren();
        refreshWorld();
      }
      $("modal-body").scrollTo({ top: 0 });
    } catch (error) {
      if (request !== studioRequest) return;
      $("studio-modal").hidden = false;
      document.body.classList.add("locked");
      world = null;
      focusViewer.open({ world: null, video: null });
      $("studio-error").textContent = error.message;
      $("studio-error").hidden = false;
    }
  }

  const setBusy = (busy) => {
    for (const id of ["play-run", "play-coast", "play-reset"]) $(id).disabled = busy;
  };

  function readout(snapshot, note) {
    const metrics = (snapshot && snapshot.task_metrics) || {};
    const entries = [];
    if (note) entries.push(["", note]);
    if (metrics.angular_rmse_rad != null)
      entries.push(["tracking error", `${metrics.angular_rmse_rad.toFixed(3)} rad / 0.35 gate`]);
    if (metrics.root_rmse_model_units != null)
      entries.push(["root error", `${metrics.root_rmse_model_units.toFixed(3)} / 0.15 gate`]);
    if (metrics.success != null) entries.push(["task success", metrics.success ? "true" : "not yet"]);
    $("readout").replaceChildren(entries.length ? chipRow(entries) : document.createTextNode(""));
  }

  async function refreshWorld() {
    if (!world) return;
    try {
      const status = await api(`/labs/${world}/api/status`);
      readout(status.snapshot, status.status === "ready" ? "world ready" : status.status);
    } catch (error) {
      readout(null, "this world is no longer loaded");
    }
  }

  async function drive(controller, label) {
    if (!world) return;
    setBusy(true);
    readout(null, label);
    try {
      let snapshot = null;
      const target = world;
      for (let batch = 0; batch < 8 && world === target && focusViewer.active === "world" && !document.hidden; batch += 1) {
        const reply = await api(`/labs/${world}/api/step`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ steps: 6, controller }),
        });
        snapshot = reply.snapshot || snapshot;
        readout(snapshot, label);
        if (snapshot && (snapshot.terminated || snapshot.truncated)) break;
      }
    } catch (error) {
      readout(null, error.message);
    } finally {
      setBusy(false);
    }
  }

  $("play-run").addEventListener("click", () => drive("reference", "running the agent…"));
  $("play-coast").addEventListener("click", () => drive("coast", "no control — physics only…"));
  $("play-reset").addEventListener("click", async () => {
    if (!world) return;
    setBusy(true);
    try {
      const reply = await api(`/labs/${world}/api/reset`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
      });
      readout(reply.snapshot, "reset to the start");
    } catch (error) {
      readout(null, error.message);
    } finally {
      setBusy(false);
    }
  });

  /* ---------------- build ---------------- */
  let file = null;
  const drop = $("drop");
  $("video").addEventListener("change", (event) => select(event.target.files[0]));
  ["dragenter", "dragover"].forEach((name) =>
    drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((name) =>
    drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (event) => select(event.dataTransfer?.files?.[0]));

  function select(chosen) {
    if (!chosen) return;
    file = chosen;
    clip = null;
    markClips();
    $("file-label").textContent = chosen.name;
  }

  // Clips whose whole chain was measured: one click beats hunting for footage the reader refuses.
  let clip = null;
  function markClips() {
    document.querySelectorAll("#clip-row .clip").forEach((button) =>
      button.setAttribute("aria-pressed", String(button.dataset.clip === clip)));
  }

  async function loadClips() {
    let data;
    try { data = await api("/api/demo/clips"); } catch { return; }
    const row = $("clip-row");
    (data.clips || []).forEach((entry) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "clip";
      button.dataset.clip = entry.id;
      button.setAttribute("aria-pressed", "false");
      const title = document.createElement("strong");
      title.textContent = entry.title;
      const note = document.createElement("span");
      note.textContent = [entry.seconds ? `~${entry.seconds}s end to end` : null,
                          entry.learned ? `trained policy ${entry.learned} vs random ${entry.random}` : null]
        .filter(Boolean).join(" · ");
      button.append(title, note);
      // The walkthrough drives the authored reference controller; the policy numbers come
      // from a separate training study, so name which is which rather than implying one run.
      button.title = [entry.learned && "Trained-policy result from a separate study — this"
                      + " walkthrough drives the authored reference controller instead.",
                      entry.why, entry.creator && `${entry.creator} · ${entry.licence}`]
        .filter(Boolean).join("\n");
      button.addEventListener("click", () => {
        clip = clip === entry.id ? null : entry.id;
        if (clip) { file = null; $("video").value = ""; $("file-label").textContent = entry.title; }
        else { $("file-label").textContent = "Drop a video here, or choose a file"; }
        markClips();
      });
      row.append(button);
    });
    // CC BY-SA requires visible attribution, so credit the sources on the page itself
    // rather than only in a tooltip. Clips without a recorded licence are never offered.
    const credits = [...new Set((data.clips || [])
      .filter((c) => c.creator && c.licence)
      .map((c) => `${c.creator} (${c.licence})`))];
    $("clips-credit").textContent = credits.length ? `Source clips: ${credits.join("; ")}.` : "";
    if (row.children.length) $("clips").hidden = false;
  }

  $("run").addEventListener("click", async () => {
    $("run").disabled = true;
    $("run-note").textContent = file ? "Uploading your clip…"
      : clip ? "Starting with the clip you picked…" : "Starting with the sample clip…";
    try {
      const prompt = $("prompt").value.trim();
      const query = new URLSearchParams();
      if (file) query.set("name", file.name);
      if (!file && clip) query.set("clip", clip);
      if (prompt) query.set("prompt", prompt);
      const started = await api("/api/demo/start?" + query.toString(), { method: "POST", body: file || undefined });
      requestedRun = started.run;
      render(started);
      $("run-note").textContent = "Running. Each step appears below as it finishes.";
      poll();
    } catch (error) {
      $("run-note").textContent = error.message;
      $("run").disabled = false;
    }
  });

  let timer = null, requestedRun = null, estimatedSeconds = 0;
  const elapsedClock = root.WeftElapsedClock((elapsed, running) => {
    const text = running ? `${elapsed.toFixed(1)}s elapsed · about ${seconds(estimatedSeconds)} total`
      : elapsed ? `${seconds(elapsed)} total` : "";
    if ($("progress-time").textContent !== text) $("progress-time").textContent = text;
  });
  function render(state) {
    const steps = state.steps || [];
    const done = steps.filter((s) => ["done", "skipped", "failed"].includes(s.status)).length;
    const running = state.status === "running";
    const showSteps = running || (state.status === "failed" && state.run === requestedRun);
    $("steps").hidden = !showSteps;
    $("progress").hidden = !showSteps;
    if (showSteps) renderSteps($("steps"), steps, state.run);
    else $("steps").replaceChildren();
    $("bar").style.width = `${Math.round((done / Math.max(1, steps.length)) * 100)}%`;
    $("progress-label").textContent = running
      ? `Step ${Math.min(done + 1, steps.length)} of ${steps.length}`
      : state.status === "complete" ? "Your world is ready"
      : state.status === "failed" ? "Stopped — the reason is recorded below" : "Ready";
    estimatedSeconds = steps.reduce((sum, s) =>
      sum + (META[s.id]?.estimate_s || META[s.id]?.estimate_seconds || 0), 0);
    const spent = state.elapsed_s ?? state.elapsed_seconds;
    elapsedClock.update({ id: state.run, elapsed: spent, running });
    $("run").disabled = running;

    if (!running && state.status === "complete" && state.run) {
      const rows = readMine();
      if (!rows.some((row) => row.id === state.run)) {
        const metrics = (steps.find((s) => s.id === "avatar") || {}).metrics || {};
        rows.unshift({
          id: state.run,
          title: (state.source && state.source.name) ? `Run on ${state.source.name}` : "Your run",
          note: metrics["task success"] ? `task success ${metrics["task success"]}` : "complete",
        });
        saveMine(rows);
      }
      renderMine();
      // A finished run is published to the carousel; pick it up without a reload.
      loadFeatured().then(() => loadGallery(true));
      $("run-note").textContent = "Done — open your result below to watch the recording or drive the world.";
      $("run").disabled = false;
    } else if (!running && state.status === "failed") {
      $("run-note").textContent = state.error || "The run stopped; its reason is recorded above.";
      $("run").disabled = false;
    }
  }

  async function poll() {
    clearTimeout(timer);
    try {
      const state = await api("/api/demo/state");
      render(state);
      if (state.status === "running") timer = setTimeout(poll, 1200);
    } catch (error) {
      if ($("gate-panel").hidden) timer = setTimeout(poll, 3000);
    }
  }


  /* Internal links must carry the lab token, or the next page cannot call the API. */
  function keepToken() {
    if (!token) return;
    for (const link of document.querySelectorAll('a[href^="/"]')) {
      if (link.target === "_blank") continue;
      const url = new URL(link.getAttribute("href"), location.origin);
      if (url.pathname.startsWith("/api/")) continue;
      link.href = url.pathname + url.search + (root.WeftDemoSession.stored ? "" : `#token=${encodeURIComponent(token)}`);
    }
  }
  function loadDemo() {
    clearTimeout(timer);
    loadMeta().then(() => loadFeatured())
      .then(() => { renderMine(); loadClips(); loadGallery(true); poll(); });
  }
  keepToken();
  if (token) {
    try { if (sessionStorage.getItem("weft_demo_unlocked") === "yes") unlocked(); } catch { /* Storage is optional. */ }
    loadDemo();
  }
})(window);
