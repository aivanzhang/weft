"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const token = new URLSearchParams(location.hash.slice(1)).get("token") || sessionStorage.getItem("weft-portal-token") || "";
  if (token) sessionStorage.setItem("weft-portal-token", token);
  history.replaceState(null, "", location.pathname);
  let selected = null, timer = null, pending = false, uploading = false, lastProjects = "", lastDetail = "";
  const terminal = new Set(["ready", "failed", "cancelled", "interrupted"]);
  const labels = {queued:"Queued",uploading:"Uploading",running:"Processing",ready:"Ready to explore",failed:"Needs attention",cancelled:"Cancelled",interrupted:"Interrupted"};
  const error = message => { $("error").textContent = message; $("error").hidden = false; };
  const fileURL = (id, name) => `/api/projects/${id}/files/${name.split("/").map(encodeURIComponent).join("/")}?token=${encodeURIComponent(token)}`;
  async function request(path, options = {}) {
    const response = await fetch(path, {...options, headers: {Authorization: `Bearer ${token}`, ...options.headers}, signal: options.signal || AbortSignal.timeout(15000)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
    return result;
  }
  function artifactLink(id, job, name, available) {
    $(id).hidden = !available;
    if (available) { $(id).href = fileURL(job.id, name); $(id).download = name.split("/").pop(); }
  }
  async function show(job) {
    if (!job) return;
    selected = job.id;
    const signature = JSON.stringify(job);
    if (lastDetail === signature) return;
    lastDetail = signature;
    $("project").hidden = false;
    $("project-heading").textContent = job.name;
    $("project-status").textContent = labels[job.status] || job.status;
    $("project-error").hidden = !job.error;
    $("project-error").textContent = job.error || "";
    $("cancel").hidden = terminal.has(job.status);
    const stage = {source:"reconstruct",observe:"reconstruct",world:"reconstruct",repair:"reconstruct",evaluate:"reconstruct",render:"train",starting:"analyze",queued:"upload"}[job.stage] || job.stage;
    const names = ["upload", "analyze", "reconstruct", "train", "ready"], current = names.indexOf(stage);
    document.querySelectorAll("#steps li").forEach(node => {
      const index = names.indexOf(node.dataset.stage);
      node.classList.toggle("current", index === current);
      node.classList.toggle("done", current >= 0 && index < current);
    });
    $("analysis").hidden = !job.plan;
    if (job.plan) {
      $("analysis-summary").textContent = job.plan.summary;
      $("analysis-warnings").replaceChildren(...(job.plan.warnings || []).map(value => { const li = document.createElement("li"); li.textContent = value; return li; }));
    }
    $("events").replaceChildren(...(job.events || []).slice(-60).map(event => { const li = document.createElement("li"); li.textContent = event.message; return li; }));
    $("sandbox").hidden = !job.sandbox;
    $("sandbox-help").hidden = !job.sandbox;
    if (job.sandbox) $("sandbox").href = `${job.sandbox}#token=${encodeURIComponent(token)}`;
    artifactLink("source-download", job, "source" + job.name.slice(job.name.lastIndexOf(".")).toLowerCase(), !!job.source_sha256);
    artifactLink("plan-download", job, "analysis/plan.json", !!job.plan);
    artifactLink("receipt-download", job, "analysis/receipt.json", !!job.receipt);
    artifactLink("failure-download", job, "failure.json", job.status === "failed");
    if (job.plan && $("frame-previews").dataset.job !== job.id) {
      $("frame-previews").dataset.job = job.id;
      $("frame-previews").replaceChildren();
      try {
        const response = await fetch(fileURL(job.id, "analysis/frames.json"), {signal: AbortSignal.timeout(10000)});
        if (!response.ok) return;
        const frames = await response.json();
        if (selected !== job.id) return;
        $("frame-previews").replaceChildren(...frames.frames.slice(0,6).map(frame => {
          const img = document.createElement("img"); img.src = fileURL(job.id, "analysis/" + frame.artifact);
          img.alt = `Frame ${frame.frame} used for scene analysis`; img.loading = "lazy"; img.decoding = "async"; return img;
        }));
      } catch { /* Analysis remains usable when thumbnail loading fails. */ }
    }
  }
  async function refresh() {
    if (pending || document.hidden) return;
    pending = true;
    clearTimeout(timer);
    try {
      const data = await request("/api/portal");
      $("connection").textContent = "Local portal · connected";
      $("ai-status").textContent = data.credential.configured ? "Project API key configured on the server" : "OpenAI key unavailable on the server";
      $("budget").textContent = data.budget.configured ? `$${data.budget.remaining_usd.toFixed(3)} remaining of $${data.budget.total_usd.toFixed(2)} · ${data.budget.calls} requests · ${data.budget.unsettled_calls} uncertain. Up to $${data.budget.reservation_usd.toFixed(4)} is reserved per request; known unused cost is released.` : "No OpenAI allowance configured. Local mode remains available with an explicit scene type.";
      const signature = JSON.stringify(data.jobs.map(job => [job.id,job.name,job.status,job.stage,selected]));
      if (signature !== lastProjects) {
        lastProjects = signature;
        $("empty").hidden = data.jobs.length > 0;
        $("projects").replaceChildren(...data.jobs.map(job => {
          const button = document.createElement("button"); button.type = "button"; button.className = "project-card";
          button.setAttribute("aria-pressed", String(selected === job.id));
          const name = document.createElement("strong"); name.textContent = job.name;
          const meta = document.createElement("small"); meta.textContent = job.ai ? "OpenAI + local reconstruction" : "Local reconstruction";
          const title = document.createElement("span"); title.append(name,meta);
          const state = document.createElement("span"); state.className = "state"; state.textContent = labels[job.status] || job.status;
          button.append(title,state); button.onclick = () => { selected = job.id; lastProjects = ""; show(job); refresh(); }; return button;
        }));
      }
      if (!selected && data.jobs.length) selected = data.jobs[0].id;
      await show(data.jobs.find(job => job.id === selected));
      $("create").disabled = uploading;
      const active = data.jobs.some(job => !terminal.has(job.status));
      timer = setTimeout(refresh, active ? 1500 : 12000);
    } catch (e) { error(e.message); $("connection").textContent = "Connection needs attention"; timer = setTimeout(refresh, 12000); }
    finally { pending = false; }
  }
  $("upload-form").onsubmit = async event => {
    event.preventDefault();
    if (uploading) return;
    $("error").hidden = true;
    const video = $("video").files[0], ai = $("use-ai").checked, mode = $("mode").value;
    if (!video || video.size > 100000000 || !$("rights").checked) return error("Select a video up to 100 MB and confirm permission to process it.");
    if (!ai && mode === "auto") return error("Choose exercise or objects when OpenAI analysis is disabled.");
    uploading = true; $("create").disabled = true; $("upload-status").textContent = "Uploading the original video…";
    const requestId = crypto.randomUUID().replaceAll("-", "");
    try {
      const query = new URLSearchParams({name:video.name,mode,ai:String(ai),request_id:requestId});
      const job = await request(`/api/projects?${query}`, {method:"POST",body:video,headers:{"Content-Type":"application/octet-stream"},signal:AbortSignal.timeout(120000)});
      selected = job.id; $("upload-status").textContent = "Video saved. Processing continues in the background."; await show(job);
    } catch (e) { error(`${e.message}. Refresh the project list before submitting again; an upload may already have been accepted.`); $("upload-status").textContent = "Upload did not finish normally."; }
    finally { uploading = false; $("create").disabled = false; refresh(); }
  };
  $("cancel").onclick = async () => {
    if (!selected) return;
    $("cancel").disabled = true;
    try { await request(`/api/projects/${selected}/cancel`, {method:"POST"}); await refresh(); }
    catch (e) { error(e.message); }
    finally { $("cancel").disabled = false; }
  };
  $("refresh").onclick = refresh;
  document.addEventListener("visibilitychange", () => { clearTimeout(timer); if (!document.hidden) refresh(); });
  window.addEventListener("pagehide", () => clearTimeout(timer));
  refresh();
})();
