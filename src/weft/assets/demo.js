((root) => {
  const $ = (id) => document.getElementById(id);
  const token = root.WeftDemoSession.token;
  const auth = { Authorization: "Bearer " + token };
  const signed = (path) => path + (path.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token);

  async function api(path, options = {}) {
    const response = await fetch(path, { ...options, headers: { ...auth, ...(options.headers || {}) } });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) warnAuth();
      throw new Error(data.error || "That request did not complete.");
    }
    return data;
  }

  function warnAuth() {
    const banner = $("auth-warning");
    if (!banner || !banner.hidden) return;
    banner.hidden = false;
    banner.textContent = token
      ? "This link's access token is no longer valid — the portal was restarted. Open the current link printed by the portal."
      : "This page needs the portal's access link. Open the full http://127.0.0.1:…/demo#token=… link printed when the portal started.";
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
    "controller": "What drove the body. 'reference' replays the authored controller derived from the clip.",
    "tracking error": "How far the simulated body drifted from the observed motion, in radians. Lower is better.",
    "root error": "How far the body's base drifted from the observed path, in model units.",
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
      if (note) { chip.classList.add("has-note"); chip.title = note; }
      if (key) chip.append(document.createTextNode(key + " "));
      const strong = document.createElement("b");
      strong.textContent = String(value);
      chip.append(strong);
      row.append(chip);
    }
    return row;
  };

  /* ---------------- steps rendering (shared by live run and studio) ---------------- */
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
      title.textContent = step.title;
      head.append(title);
      if (step.experimental) {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.title = "Shipped as experimental: it can refuse its own output, and you will see that refusal.";
        tag.textContent = "EXPERIMENTAL";
        head.append(tag);
      }
      const time = document.createElement("span");
      time.className = "step-time";
      const spent = step.elapsed_s ?? step.elapsed_seconds;
      const budget = step.estimate_s ?? step.estimate_seconds;
      time.textContent = spent ? seconds(spent)
        : step.status === "running" ? "running…"
        : budget ? `~${seconds(budget)}` : "";
      head.append(time);
      item.append(head);

      for (const [field, cls] of [["explainer", "explain"], ["why", "matters"], ["impact", "impact"],
                                  ["detail", "detail"]]) {
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
        const size = bytes > 1048576 ? (bytes / 1048576).toFixed(1) + " MB"
          : Math.max(1, Math.round(bytes / 1024)) + " KB";
        link.textContent = `↓ ${step.artifact.name} · ${size}`;
        item.append(link);
      }
    });
  }

  /* ---------------- gallery ---------------- */
  let demos = [];
  let current = 0;

  let featured = [];
  async function loadFeatured() {
    try {
      const data = await api("/api/demo/featured");
      featured = data.featured || [];
    } catch (error) { featured = []; }
  }

  async function loadGallery(force) {
    try {
      const data = await api("/api/demo/gallery");
      const all = Array.isArray(data) ? data : (data.items || data.demos || []);
      // This page shows curated work only: seeded clips plus explicitly featured runs.
      // Anything a judge builds stays on their own builder page.
      const next = all.filter((row) => !String(row.id || "").startsWith("run-")
                                       || featured.includes(row.id));
      const same = next.length === demos.length && next.every((d, i) => d.id === demos[i]?.id);
      if (same && !force) return;
      demos = next;
      current = Math.min(current, Math.max(0, demos.length - 1));
      showDemo();
    } catch (error) {
      /* the gallery is optional; the walkthrough is the point */
    }
  }

  function showDemo() {
    $("explore").hidden = demos.length === 0;
    if (!demos.length) return;
    const demo = demos[current];
    $("demo-video").src = signed(demo.video_url || `/api/demo/gallery/${demo.id}/video`);
    const mine = /^run-/.test(demo.id || "") || ["judge", "run", "yours"].includes(demo.kind);
    $("demo-source").textContent = (mine ? "your run" : "verified run").toUpperCase();
    $("demo-title").textContent = demo.title;
    $("demo-caption").textContent = demo.caption || "";
    const metrics = demo.metrics && typeof demo.metrics === "object"
      ? Object.entries(demo.metrics).map(([k, v]) => `${k} ${v}`).join(" · ")
      : demo.metrics || "";
    $("demo-metrics").textContent = metrics;
    $("open-studio").hidden = false;
    $("open-studio").textContent = mine ? "Open run" : "Watch recording";
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

  $("prev").addEventListener("click", () => { current = (current - 1 + demos.length) % demos.length; showDemo(); });
  $("next").addEventListener("click", () => { current = (current + 1) % demos.length; showDemo(); });

  $("demo-video").addEventListener("play", () => {
    $("demo-video").pause();
    openStudio(demos[current]?.id, "recording");
  });

  /* ---------------- studio ---------------- */
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

  $("open-studio").addEventListener("click", () => openStudio(demos[current]?.id));
  // Backdrop and the close button share one dismissal path.
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
        : { id: identity, title: selected?.title, caption: selected?.caption, steps: [] };
      if (request !== studioRequest) return;
      $("studio-error").hidden = true;
      $("studio-modal").hidden = false;
      document.body.classList.add("locked");
      const galleryEntry = demos.find((d) => d.id === identity) || {};
      $("studio-title").textContent = record.title || galleryEntry.title || "Inside this run";
      $("studio-note").textContent = record.caption || galleryEntry.caption || "";
      renderSteps($("studio-steps"), record.steps || [], record.run || record.id || identity);
      const result = record.result || {};
      world = result.project || result.child || record.project || null;
      const entry = demos.find((item) => item.id === identity);
      focusViewer.open({ world, video: entry?.video_url ? signed(entry.video_url) : null, view });
      if (world) {
        $("play-full").href = `/labs/${world}/#token=${encodeURIComponent(token)}`;
        $("readout").replaceChildren();
        refreshWorld();
      }
      $("modal-body") && $("modal-body").scrollTo({ top: 0 });
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

  /* ---------------- live world controls ---------------- */
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
    const failed = metrics.failed_gates;
    if (failed) entries.push(["failed gates", failed.length ? failed.join(", ") : "none"]);
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
  keepToken();
  Promise.all([loadMeta(), loadFeatured()]).then(() => loadGallery(true));
})(window);
