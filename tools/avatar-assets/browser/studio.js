const STATUS = {
  created: "Request saved", queued: "Queued", launching: "Starting", planning: "Preparing plan", running: "Creating",
  awaiting_authorization: "Allowance needed", planned: "Plan ready", cancel_requested: "Stopping",
  cancelled: "Stopped", complete: "Ready", unsupported: "Could not complete", needs_inspection: "Needs review",
  failed: "Stopped with an error", pending: "Not started", active: "In progress", passed: "Passed",
  accepted: "Accepted", rejected: "Needs repair", repair: "Repairing", skipped: "Not needed",
};
const LIVE = new Set(["queued", "launching", "planning", "running", "cancel_requested"]);
const text = (value, limit = 8000) => typeof value === "string" ? value.slice(0, limit) : "";
const statusLabel = (value) => STATUS[value] || text(value, 80).replaceAll("_", " ") || "Status unavailable";
const finite = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;

export function parseActions(value, maximum = 8, characters = 600) {
  const actions = value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (actions.length > maximum) throw new Error(`Use up to ${maximum} actions, one per line.`);
  const index = actions.findIndex((action) => action.length > characters);
  if (index >= 0) throw new Error(`Action ${index + 1} is too long. Use up to ${characters} characters per action.`);
  return actions;
}

export function durationLabel(seconds) {
  if (!finite(seconds)) return "";
  const whole = Math.floor(seconds);
  if (whole < 60) return `${whole}s`;
  if (whole < 3600) return `${Math.floor(whole / 60)}m ${whole % 60}s`;
  return `${Math.floor(whole / 3600)}h ${Math.floor(whole % 3600 / 60)}m`;
}

export function localAssetURL(value, origin) {
  if (typeof value !== "string" || !value.startsWith("/") || value.startsWith("//") || value.length > 2048) return null;
  try {
    const url = new URL(value, origin);
    return url.origin === origin && !url.username && !url.password && /^https?:$/.test(url.protocol)
      ? `${url.pathname}${url.search}` : null;
  } catch { return null; }
}

export function canStart(job) {
  return ["planned", "awaiting_authorization"].includes(job?.status) &&
    job?.capabilities?.start !== false && job?.plan?.authorization?.can_start === true && Boolean(planID(job));
}

function planID(job) {
  return text(job?.plan?.id || job?.plan?.plan_id || job?.plan_id, 160);
}

export async function bootStudio() {
  const $ = (id) => document.getElementById(id);
  const state = { session: null, jobs: [], selected: null, job: null, busy: false, jobSignature: "", listSignature: "",
    resultSignature: "", result: null, mode: "portrait", clip: null, read: null, mutation: null, timer: null,
    pollBusy: false, reference: null, referenceURL: null, uploadedReference: null, acknowledgedPlan: null,
    preview: null, previewTimer: null, previewVersion: 0, modelReady: false, modelRuntime: null, raf: 0, lastPaint: 0,
    draftID: null, draftFingerprint: null };
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const model = $("result-model"), video = $("result-video"), portrait = $("result-portrait");
  const apiPath = (id, suffix = "") => `/api/studio/jobs/${encodeURIComponent(id)}${suffix}`;
  const assetURL = (value) => localAssetURL(value, location.origin);
  const node = (tag, content, className) => {
    const item = document.createElement(tag);
    if (content != null) item.textContent = String(content);
    if (className) item.className = className;
    return item;
  };
  const limit = (key, fallback) => {
    const value = state.session?.limits?.[key];
    return Number.isInteger(value) && value > 0 ? Math.min(value, fallback) : fallback;
  };

  function message(value = "") {
    $("message").textContent = value;
    $("message").hidden = !value;
  }
  function connection(error) {
    $("connection").textContent = error ? "Connection interrupted" : "Local workspace";
    $("connection").dataset.connected = String(!error);
    $("connection-error").hidden = !error;
    $("connection-error").textContent = error ? `${error} Saved work is not discarded. Use Refresh to check again.` : "";
  }
  async function request(path, { method = "GET", body, signal, requestID } = {}) {
    const controller = new AbortController();
    const forwardAbort = () => controller.abort();
    signal?.addEventListener("abort", forwardAbort, { once: true });
    if (signal?.aborted) controller.abort();
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 20000);
    try {
      const headers = { Accept: "application/json" };
      if (method !== "GET") {
        if (!state.session?.csrf_token) throw new Error("The local session is unavailable. Refresh before continuing.");
        headers["X-Avatar-Studio-Token"] = state.session.csrf_token;
        if (requestID) headers["X-Avatar-Request-Id"] = requestID;
        if (!(body instanceof FormData)) { headers["Content-Type"] = "application/json"; body = JSON.stringify(body || {}); }
      }
      const response = await fetch(path, { method, headers, body, signal: controller.signal, credentials: "same-origin", cache: "no-store", redirect: "error" });
      if (!response.headers.get("content-type")?.includes("application/json")) throw new Error("The local service returned an unreadable response.");
      const raw = await response.text();
      if (raw.length > 2 * 1024 * 1024) throw new Error("The local service returned too much data.");
      let data;
      try { data = JSON.parse(raw); } catch { throw new Error("The local service returned invalid data."); }
      if (!response.ok) {
        if (response.status === 401 || response.status === 403) state.session = null;
        throw new Error(text(data?.error?.message, 2000) || `Request failed (${response.status}).`);
      }
      return data;
    } catch (error) {
      if (timedOut) throw new Error("The request timed out. Check the job status before trying again; it may already have been accepted.");
      throw error;
    } finally { clearTimeout(timeout); signal?.removeEventListener("abort", forwardAbort); }
  }

  function actionInput() {
    try {
      const actions = parseActions($("actions").value, limit("actions", 8), limit("action_chars", 600));
      $("action-count").textContent = `${actions.length} of ${limit("actions", 8)} actions`;
      $("actions").removeAttribute("aria-invalid"); $("actions").setCustomValidity("");
      $("actions-error").hidden = true;
      return actions;
    } catch (error) {
      $("actions-error").textContent = error.message; $("actions-error").hidden = false;
      $("actions").setAttribute("aria-invalid", "true"); $("actions").setCustomValidity(error.message);
      $("action-count").textContent = "Check your actions";
      return null;
    }
  }
  function updateButtons() {
    const job = state.job;
    $("save-request").disabled = state.busy || !state.session;
    $("save-request").textContent = state.busy && !state.selected ? "Saving request…" : "Save request →";
    $("prepare-plan").disabled = state.busy || !state.session || job?.status !== "created" || job?.capabilities?.plan === false;
    $("start-job").disabled = state.busy || !state.session || !canStart(job) || !$("estimate-ack").checked;
    $("cancel-job").disabled = state.busy || job?.status === "cancel_requested" || job?.capabilities?.cancel === false;
    $("estimate-ack").disabled = state.busy || !canStart(job);
    $("refresh").disabled = state.busy || state.pollBusy;
    $("new-request").disabled = $("copy-request").disabled = state.busy;
    for (const id of ["prompt", "style", "actions", "preset", "remove-reference"]) $(id).disabled = state.busy || !state.session;
    $("reference").disabled = state.busy || state.session?.capabilities?.reference_upload !== true;
  }
  function presetDescription() {
    const preset = state.session?.presets?.find((item) => (typeof item === "string" ? item : item.id) === $("preset").value);
    $("preset-description").textContent = text(preset?.description, 1000) || "The plan will list the exact output sizes before creation starts.";
  }
  function populateSession(session) {
    if (!text(session.csrf_token, 500)) throw new Error("The local service did not provide a valid session.");
    state.session = session;
    $("prompt").maxLength = limit("prompt_chars", 4000); $("style").maxLength = limit("style_chars", 800);
    const previous = $("preset").value;
    $("preset").replaceChildren();
    for (const preset of (Array.isArray(session.presets) ? session.presets : [{ id: "preview", label: "Preview" }]).slice(0, 12)) {
      const id = typeof preset === "string" ? preset : preset.id;
      if (!id) continue;
      const option = node("option", typeof preset === "string" ? statusLabel(preset) : preset.label || preset.id);
      option.value = id; $("preset").append(option);
    }
    if ([...$("preset").options].some((item) => item.value === previous)) $("preset").value = previous;
    presetDescription(); updateButtons(); connection();
    if (session.capabilities?.reference_upload !== true) $("reference-help").textContent = "Image upload is not enabled in this local session. You can still create from text.";
  }
  function unwrapJob(payload) {
    const job = payload?.job || payload;
    if (!job || !text(job.id, 160) || typeof job.status !== "string") throw new Error("The service returned an invalid job.");
    return job;
  }
  function renderList() {
    const signature = JSON.stringify([state.selected, state.jobs.map((job) => [job.id, job.title, job.status])]);
    if (signature === state.listSignature) return;
    state.listSignature = signature;
    const fragment = document.createDocumentFragment();
    for (const job of state.jobs.slice(0, 100)) {
      const button = node("button", null, "job-card"); button.type = "button"; button.dataset.job = job.id;
      button.setAttribute("aria-pressed", String(job.id === state.selected));
      button.append(node("strong", text(job.title || job.request?.prompt, 160) || "Untitled creation"), node("small", statusLabel(job.status)));
      button.addEventListener("click", () => selectJob(job.id)); fragment.append(button);
    }
    $("job-list").replaceChildren(fragment); $("empty-jobs").hidden = state.jobs.length > 0;
  }
  function addFact(container, label, value) {
    if (value == null || value === "") return;
    const group = node("div"); group.append(node("dt", label), node("dd", value)); container.append(group);
  }
  function renderRequest(job) {
    const saved = job.request || {};
    $("request-prompt").textContent = text(saved.prompt);
    $("request-facts").replaceChildren();
    addFact($("request-facts"), "Look and materials", text(saved.style, 1000));
    addFact($("request-facts"), "Output", statusLabel(saved.preset));
    addFact($("request-facts"), "Reference", saved.reference_selected ? "Selected image" : "Text only");
    if (saved.reuse_model_selected) addFact($("request-facts"), "Geometry", "Existing generated model");
    if (job.created_at && Number.isFinite(Date.parse(job.created_at))) addFact($("request-facts"), "Saved", new Intl.DateTimeFormat("en-US", {
      timeZone: "America/Los_Angeles", month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short",
    }).format(new Date(job.created_at)));
    const actions = Array.isArray(saved.custom_actions) ? saved.custom_actions : [];
    $("requested-actions").replaceChildren(...actions.slice(0, 8).map((action) => node("li", text(typeof action === "string" ? action : action.description || action.action, 600))));
    $("requested-actions-title").hidden = actions.length === 0;
  }
  function money(amount, currency = "USD") {
    if (!finite(amount)) return null;
    try { return new Intl.NumberFormat("en-US", { style: "currency", currency }).format(amount); }
    catch { return null; }
  }
  function renderPlan(job) {
    const plan = job.plan, estimate = plan?.estimate;
    $("plan-description").textContent = text(plan?.summary) || (job.operation === "plan" || job.status === "planning"
      ? "Preparing the proposed assets and a bounded work estimate. No generation is started by this step."
      : "Your request is saved. Prepare a plan to see the proposed assets and estimated work.");
    $("planning-cost-note").textContent = text(state.session?.planning_cost_note, 2000) || "Planning uses the configured AI service. Account-specific dollar cost is unavailable. Model generation requires a separate reviewed estimate.";
    $("planning-cost-note").hidden = Boolean(plan);
    $("prepare-plan").hidden = job.status !== "created";
    $("plan-warnings").replaceChildren(...(Array.isArray(plan?.warnings) ? plan.warnings : []).slice(0, 16).map((warning) => node("li", text(warning))));
    $("plan-warnings").hidden = !$("plan-warnings").children.length;
    $("estimate").hidden = !estimate;
    if (!estimate) return;
    const credits = finite(estimate.credits) ? `${estimate.credits.toLocaleString()} credits` : "Not available";
    const cost = money(estimate.amount, estimate.currency);
    $("estimate-total").textContent = credits;
    $("cost-breakdown").replaceChildren();
    for (const item of (Array.isArray(estimate.breakdown) ? estimate.breakdown : []).slice(0, 12)) {
      addFact($("cost-breakdown"), text(item.label, 120), finite(item.credits) ? `${item.credits} credits` : money(item.amount, estimate.currency) || "Not priced");
    }
    addFact($("cost-breakdown"), "Estimated dollar cost", cost || "Not available for this account");
    $("estimate-note").textContent = text(estimate.note, 3000);
    const pinned = planID(job);
    if (state.acknowledgedPlan !== pinned) { $("estimate-ack").checked = false; state.acknowledgedPlan = pinned; }
    $("authorization").textContent = text(plan.authorization?.reason, 2000) || (canStart(job) ? "An existing spending allowance covers this plan." : "Starting is not authorized for this plan.");
    if (!pinned && ["planned", "awaiting_authorization"].includes(job.status)) $("authorization").textContent = "This estimate is not yet linked to a frozen plan. Starting is blocked.";
    $("authorization").dataset.allowed = String(canStart(job));
    const startableStage = ["planned", "awaiting_authorization"].includes(job.status);
    $("estimate-ack-label").hidden = !startableStage;
    $("start-job").hidden = !startableStage;
  }
  function renderTimeline(job) {
    $("job-elapsed").textContent = finite(job.elapsed_seconds) ? `${durationLabel(job.elapsed_seconds)} elapsed` : "";
    const summaries = {
      created: "Generation has not started.", planned: "Review the estimate before starting.", awaiting_authorization: "The plan is saved. Creation is blocked until its spending allowance is available.",
      queued: "Waiting for the local worker. This job will not start a second copy of an active operation.", launching: "Starting the local worker. Progress will update when it begins the saved operation.", planning: "Preparing the plan.", running: "Working through the saved plan. Stages update as results are recorded.",
      cancel_requested: "Stop requested. Waiting for the current operation to reach a safe stopping point.", cancelled: "Stopped. Completed work and cost records are preserved.",
      complete: "The finished output is available below. Review its saved limitations before use.", unsupported: "This request could not be completed faithfully. Saved attempts are preserved.",
      needs_inspection: "This job needs inspection before it can continue. It will not restart automatically.", failed: "Creation stopped. Saved work is preserved; no retry is started automatically.",
    };
    $("progress-summary").textContent = summaries[job.status] || "Waiting for a status update.";
    const fragment = document.createDocumentFragment();
    for (const stage of (Array.isArray(job.stages) ? job.stages : []).slice(0, 40)) {
      const item = node("li"); item.dataset.status = stage.status;
      const dot = node("span", ["complete", "passed"].includes(stage.status) ? "✓" : ["failed", "unsupported"].includes(stage.status) ? "!" : "", "stage-dot"); dot.setAttribute("aria-hidden", "true");
      const content = node("div"), title = node("div", null, "stage-title");
      title.append(node("strong", text(stage.label, 160) || statusLabel(stage.id)), node("span", `${statusLabel(stage.status)}${finite(stage.elapsed_seconds) ? ` · ${durationLabel(stage.elapsed_seconds)}` : ""}`));
      content.append(title);
      if (stage.detail) content.append(node("p", text(stage.detail, 2000), "stage-detail"));
      const progress = stage.progress;
      if (finite(progress?.current) && finite(progress?.total) && progress.total > 0 && progress.current <= progress.total) {
        const meter = node("progress", null, "stage-progress"); meter.max = progress.total; meter.value = progress.current;
        meter.setAttribute("aria-label", `${stage.label || "Stage"}: ${progress.current} of ${progress.total} ${text(progress.unit, 50)}`);
        content.append(meter, node("p", `${progress.current} / ${progress.total} ${text(progress.unit, 50)}`, "stage-detail"));
      }
      item.append(dot, content); fragment.append(item);
    }
    $("timeline").replaceChildren(fragment);
    $("job-problem").textContent = text(job.error?.message);
    $("job-problem").hidden = !job.error?.message;
    const cancellable = LIVE.has(job.status);
    $("cancel-job").hidden = $("cancel-note").hidden = !cancellable;
  }
  function renderReviews(job) {
    const reviews = Array.isArray(job.reviews) ? job.reviews.slice(0, 12) : [];
    $("reviews-panel").hidden = !reviews.length;
    $("reviews").replaceChildren(...reviews.map((review, i) => {
      const group = node("div", null, "review");
      group.append(node("strong", `Review ${review.attempt ?? i + 1} · ${statusLabel(review.status)}`), node("p", text(review.summary)));
      return group;
    }));
  }
  function renderJob(job) {
    state.job = job;
    const signature = JSON.stringify(job);
    if (state.jobSignature !== signature) {
      state.jobSignature = signature;
      $("job-title").textContent = text(job.title || job.request?.prompt, 160) || "Your creation";
      $("job-description").textContent = job.status === "complete" ? "Your assets are ready to inspect and download." : "Follow your request from plan to finished assets.";
      if ($("job-status").textContent !== statusLabel(job.status)) $("job-status").textContent = statusLabel(job.status);
      $("job-status").dataset.status = job.status;
      $("job-eyebrow").textContent = "YOUR CREATION";
      $("fixture-notice").hidden = job.provenance !== "analytic_fixture";
      renderRequest(job); renderPlan(job); renderTimeline(job); renderReviews(job); renderResult(job);
    }
    $("composer").hidden = true; $("job-detail").hidden = false; updateButtons();
  }

  function stopPlayback() {
    video.pause(); model.pause?.(); cancelAnimationFrame(state.raf); state.raf = 0;
    $("model-play").textContent = "Play once";
  }
  function clearPreview() {
    stopPlayback(); state.previewVersion++; state.preview?.abort(); clearTimeout(state.previewTimer);
    model.removeAttribute("src"); video.removeAttribute("src"); video.removeAttribute("poster"); video.load(); portrait.removeAttribute("src");
    state.modelReady = false; $("model-play").disabled = true;
  }
  function previewError(value) { $("preview-error").textContent = value; $("preview-error").hidden = !value; }
  function preparedResult(result) {
    if (!result || typeof result !== "object") return null;
    return { portrait: assetURL(result.portrait_url), model: assetURL(result.model_url),
      videos: (Array.isArray(result.videos) ? result.videos : []).slice(0, 8).map((clip) => ({ ...clip, url: assetURL(clip.url), poster: assetURL(clip.poster_url) })).filter((clip) => clip.url),
      downloads: (Array.isArray(result.downloads) ? result.downloads : []).slice(0, 32).map((item) => ({ ...item, url: assetURL(item.url) })).filter((item) => item.url),
      limitations: (Array.isArray(result.limitations) ? result.limitations : []).slice(0, 24) };
  }
  function renderResult(job) {
    let previews = job.previews;
    if (Array.isArray(previews)) {
      const portrait = [...previews].reverse().find((item) => ["portrait", "image"].includes(item.kind) && assetURL(item.url));
      const videos = previews.filter((item) => item.kind === "video" && assetURL(item.url));
      previews = portrait || videos.length ? { portrait_url: portrait?.url, videos,
        limitations: ["Work-in-progress preview. Quality review and final delivery are not complete."] } : null;
    }
    const raw = job.result || previews;
    const signature = JSON.stringify([job.id, raw]);
    if (signature === state.resultSignature) return;
    state.resultSignature = signature; clearPreview(); state.result = preparedResult(raw);
    $("result-panel").hidden = !state.result;
    if (!state.result) return;
    $("result-panel").setAttribute("aria-label", job.status === "complete" ? "Finished assets" : "Work in progress previews");
    $("job-eyebrow").textContent = job.status === "complete" ? "READY TO REVIEW" : "SAVED CANDIDATE";
    const links = [...state.result.downloads];
    if (state.result.portrait && !links.some((item) => item.url === state.result.portrait)) links.unshift({ label: "Portrait", url: state.result.portrait });
    if (state.result.model && !links.some((item) => item.url === state.result.model)) links.push({ label: "3D model", url: state.result.model });
    for (const clip of state.result.videos) if (!links.some((item) => item.url === clip.url)) links.push({ label: `${clip.label || statusLabel(clip.id)} video`, url: clip.url });
    $("downloads").replaceChildren(...links.map((item) => {
      const link = node("a", `${text(item.label, 120) || "Download"} ↓`); link.href = item.url;
      link.download = text(item.filename, 180).replace(/[\\/\x00-\x1f]/g, "_"); return link;
    }));
    $("result-limits").replaceChildren(...state.result.limitations.map((item) => node("li", text(item))));
    for (const button of document.querySelectorAll("[data-view]")) button.disabled = button.dataset.view === "model" ? !state.result.model : button.dataset.view === "video" ? !state.result.videos.length : !state.result.portrait;
    portrait.alt = `${text(job.title, 160) || "Avatar"} portrait`;
    state.clip = null; showPreview("portrait");
  }
  function setClips(clips) {
    const previous = state.clip;
    $("clip").replaceChildren(...clips.map((clip) => { const option = node("option", text(clip.label, 200) || statusLabel(clip.id)); option.value = clip.id; return option; }));
    if (clips.some((clip) => clip.id === previous)) $("clip").value = previous;
    state.clip = $("clip").value || null;
    $("clip-label").hidden = !clips.length || state.mode === "portrait";
  }
  function finishPreview(version) {
    if (version !== state.previewVersion) return;
    clearTimeout(state.previewTimer); $("preview-loading").hidden = true;
  }
  async function showPreview(mode, fallbackMessage = "") {
    if (!state.result) return;
    stopPlayback(); state.preview?.abort(); clearTimeout(state.previewTimer); state.previewTimer = null;
    const version = ++state.previewVersion; state.preview = new AbortController(); const signal = state.preview.signal;
    state.mode = mode; state.modelReady = false;
    previewError(fallbackMessage); $("preview-loading").hidden = true; $("preview-empty").hidden = true;
    model.hidden = mode !== "model"; video.hidden = mode !== "video"; portrait.hidden = mode !== "portrait";
    $("model-controls").hidden = mode !== "model"; $("model-play").disabled = true;
    $("clip-label").hidden = true;
    for (const button of document.querySelectorAll("[data-view]")) button.setAttribute("aria-pressed", String(button.dataset.view === mode));
    if (mode !== "video") { video.removeAttribute("src"); video.load(); }
    if (mode !== "model") model.removeAttribute("src");
    $("motion-note").textContent = mode === "portrait" ? (reducedMotion.matches ? "Reduced motion: static portrait selected. Playback is your choice." : "Static portrait. Choose a clip or the 3D model to inspect motion.") : "Playback starts paused. Playing or repeating a clip does not prove a seamless loop.";
    const fail = () => {
      if (version !== state.previewVersion) return;
      if (mode !== "portrait") showPreview("portrait", `${mode === "model" ? "3D" : "Video"} preview could not load. ${state.result.portrait ? "Showing the static portrait." : "Use the saved downloads instead."}`);
      else { finishPreview(version); portrait.hidden = true; $("preview-empty").hidden = false; previewError("The portrait could not load. Other saved downloads may still be available."); }
    };
    if (mode === "portrait") {
      if (!state.result.portrait) { portrait.hidden = true; $("preview-empty").hidden = false; return; }
      $("preview-loading").hidden = false;
      portrait.addEventListener("load", () => finishPreview(version), { once: true, signal }); portrait.addEventListener("error", fail, { once: true, signal });
      portrait.src = state.result.portrait;
      if (portrait.complete && portrait.naturalWidth) finishPreview(version);
    } else if (mode === "video") {
      if (!state.result.videos.length) return showPreview("portrait", "No rendered clip is available yet.");
      setClips(state.result.videos);
      const clip = state.result.videos.find((item) => item.id === state.clip) || state.result.videos[0];
      if (clip.poster || state.result.portrait) video.poster = clip.poster || state.result.portrait;
      $("preview-loading").hidden = false;
      video.addEventListener("loadedmetadata", () => finishPreview(version), { once: true, signal }); video.addEventListener("error", fail, { once: true, signal });
      video.loop = false; video.src = clip.url; video.load();
      $("motion-note").textContent = clip.loop_verified === true ? "Playback starts paused. This clip has a recorded loop-boundary check." : "Playback starts paused. A seamless loop has not been verified for this clip.";
    } else if (mode === "model") {
      if (!state.result.model) return showPreview("portrait", "No 3D model is available yet.");
      $("preview-loading").hidden = false;
      state.previewTimer = setTimeout(fail, 25000);
      try {
        state.modelRuntime ||= import("/vendor/model-viewer.min.js").then(() => customElements.whenDefined("model-viewer"));
        await state.modelRuntime;
        if (version !== state.previewVersion) return;
        customElements.get("model-viewer").modelCacheSize = 1;
        model.addEventListener("error", fail, { once: true, signal });
        model.addEventListener("load", async () => {
          if (version !== state.previewVersion) return;
          try {
            setClips((model.availableAnimations || []).slice(0, 8).map((id) => ({ id, label: state.result.videos.find((clip) => clip.id === id)?.label || statusLabel(id) })));
            await chooseModelClip(version);
            finishPreview(version);
          } catch { fail(); }
        }, { once: true, signal });
        model.src = state.result.model;
      } catch { fail(); }
    }
    if (version === state.previewVersion && !$("preview-loading").hidden && !state.previewTimer) state.previewTimer = setTimeout(fail, 25000);
  }
  async function chooseModelClip(version = state.previewVersion) {
    state.modelReady = false;
    if (state.clip && model.availableAnimations?.includes(state.clip)) {
      model.animationName = state.clip; await model.updateComplete;
      if (version !== state.previewVersion || state.mode !== "model") return;
      model.timeScale = 1; model.play({ repetitions: 1 }); model.pause(); model.currentTime = 0;
      state.modelReady = true;
    }
    $("model-play").disabled = !state.modelReady; paintModelTime();
  }
  function paintModelTime() {
    const duration = finite(model.duration) ? model.duration : 0, position = finite(model.currentTime) ? Math.min(model.currentTime, duration) : 0;
    $("model-time").textContent = state.modelReady ? `${position.toFixed(1)} / ${duration.toFixed(1)} s` : "No skeletal animation in this model.";
    $("model-play").textContent = state.modelReady && !model.paused ? "Pause" : "Play once";
  }
  function playbackTick(now) {
    if (now - state.lastPaint > 100) { paintModelTime(); state.lastPaint = now; }
    state.raf = state.mode === "model" && !model.paused && !document.hidden ? requestAnimationFrame(playbackTick) : 0;
  }

  function stopPolling() { clearTimeout(state.timer); state.timer = null; state.read?.abort(); }
  function schedulePoll() {
    clearTimeout(state.timer);
    if (document.hidden || state.busy || state.read || state.pollBusy || !state.session) return;
    if (state.jobs.some((job) => LIVE.has(job.status)) || LIVE.has(state.job?.status)) state.timer = setTimeout(refresh, 2500);
  }
  async function refresh() {
    if (document.hidden || state.pollBusy || state.busy) return;
    state.pollBusy = true; state.read?.abort(); const controller = new AbortController(); state.read = controller; updateButtons();
    const selected = state.selected;
    try {
      if (!state.session) populateSession(await request("/api/studio/session", { signal: controller.signal }));
      const payload = await request("/api/studio/jobs", { signal: controller.signal });
      if (controller.signal.aborted) return;
      if (!Array.isArray(payload.jobs)) throw new Error("The local service did not return a creation list.");
      state.jobs = payload.jobs.slice(0, 100); renderList();
      if (selected && selected === state.selected) {
        const job = unwrapJob(await request(apiPath(selected), { signal: controller.signal }));
        if (!controller.signal.aborted && selected === state.selected) renderJob(job);
      }
      connection();
    } catch (error) { if (error.name !== "AbortError") connection(error.message); }
    finally { if (state.read === controller) state.read = null; state.pollBusy = false; updateButtons(); schedulePoll(); }
  }
  async function selectJob(id) {
    if (typeof id !== "string" || !id || id.length > 160) return;
    stopPolling(); clearPreview(); message(); state.selected = id; state.job = null; state.jobSignature = ""; state.resultSignature = "";
    state.acknowledgedPlan = null; $("estimate-ack").checked = false;
    const controller = new AbortController(); state.read = controller; renderList();
    $("composer").hidden = true; $("job-detail").hidden = true; updateButtons();
    try {
      const job = unwrapJob(await request(apiPath(id), { signal: controller.signal }));
      if (!controller.signal.aborted && state.selected === id) { renderJob(job); connection(); }
    } catch (error) { if (error.name !== "AbortError" && state.selected === id) message(`Could not load this creation. ${error.message}`); }
    finally { if (state.read === controller) state.read = null; schedulePoll(); }
  }
  function newRequest(copy = null) {
    if (state.busy) return;
    stopPolling(); clearPreview(); state.selected = null; state.job = null; state.result = null;
    state.jobSignature = ""; state.resultSignature = ""; message();
    $("composer").hidden = false; $("job-detail").hidden = true;
    if (copy) {
      state.draftID = state.draftFingerprint = null;
      $("prompt").value = text(copy.prompt, 4000); $("style").value = text(copy.style, 800);
      $("actions").value = (Array.isArray(copy.custom_actions) ? copy.custom_actions : []).filter((action) => typeof action === "string").slice(0, 8).join("\n");
      if ([...$("preset").options].some((option) => option.value === copy.preset)) $("preset").value = copy.preset;
      removeReference(); presetDescription(); actionInput();
      if (copy.reference_selected || copy.reuse_model_selected) message("Text copied into a new request. Select the reference image again if needed; an existing model is not carried over automatically.");
    }
    renderList(); updateButtons(); $("prompt").focus(); schedulePoll();
  }
  async function mutate(operation) {
    if (state.busy) return;
    stopPolling(); state.busy = true; state.mutation = new AbortController(); message(); updateButtons();
    try { await operation(state.mutation.signal); }
    catch (error) { if (error.name !== "AbortError") message(error.message); }
    finally { state.busy = false; state.mutation = null; updateButtons(); refresh(); }
  }
  function removeReference() {
    if (state.referenceURL) URL.revokeObjectURL(state.referenceURL);
    state.reference = state.referenceURL = state.uploadedReference = null;
    state.draftID = state.draftFingerprint = null;
    $("reference").value = ""; $("reference-image").removeAttribute("src"); $("reference-preview").hidden = true;
  }
  $("reference").addEventListener("change", () => {
    const file = $("reference").files[0]; removeReference(); message(); if (!file) return;
    if (!/^image\/(png|jpeg|webp)$/.test(file.type) || !/\.(png|jpe?g|webp)$/i.test(file.name)) return message("Select a PNG, JPEG, or WebP image.");
    if (!file.size || file.size > limit("reference_bytes", 8 * 1024 * 1024)) return message("The reference must be a non-empty image no larger than 8 MiB.");
    // Decode only the server-validated image; a small compressed file can have huge dimensions.
    state.reference = file; $("reference-image").hidden = true;
    $("reference-name").textContent = `${file.name} · checked when saved`; $("reference-preview").hidden = false;
  });
  $("reference-image").addEventListener("error", () => { if (state.reference) { removeReference(); message("This image could not be previewed. Select another reference."); } });
  $("remove-reference").addEventListener("click", removeReference);
  $("request-form").addEventListener("submit", (event) => {
    event.preventDefault(); const customActions = actionInput(); if (!customActions || !$("request-form").reportValidity()) return;
    const prompt = $("prompt").value.trim(); if (!prompt) { $("prompt").focus(); return message("Describe the avatar you want to create."); }
    const payload = { prompt, style: $("style").value.trim(), custom_actions: customActions, preset: $("preset").value };
    const reference = state.reference;
    const fingerprint = JSON.stringify([payload, reference && [reference.name, reference.size, reference.lastModified]]);
    if (fingerprint !== state.draftFingerprint) { state.draftID = crypto.randomUUID(); state.draftFingerprint = fingerprint; }
    const requestID = state.draftID;
    mutate(async (signal) => {
      if (reference) {
        if (!state.uploadedReference) { const form = new FormData(); form.set("file", reference); state.uploadedReference = await request("/api/studio/references", { method: "POST", body: form, signal }); }
        if (!text(state.uploadedReference.id, 160)) throw new Error("The reference could not be attached to this request.");
        payload.reference_id = state.uploadedReference.id;
      }
      const job = unwrapJob(await request("/api/studio/jobs", { method: "POST", body: payload, signal, requestID }));
      state.draftID = state.draftFingerprint = null;
      state.selected = job.id; state.jobSignature = ""; state.resultSignature = ""; renderJob(job); renderList();
      $("job-title").scrollIntoView({ block: "nearest" });
    });
  });
  $("prepare-plan").addEventListener("click", () => {
    const id = state.selected; if (!id || state.job?.status !== "created") return;
    mutate(async (signal) => { const job = unwrapJob(await request(apiPath(id, "/plan"), { method: "POST", body: {}, signal })); if (state.selected === id) renderJob(job); });
  });
  $("start-job").addEventListener("click", () => {
    const job = state.job; if (!canStart(job) || !$("estimate-ack").checked || state.acknowledgedPlan !== planID(job)) return;
    const id = job.id, plan = planID(job);
    mutate(async (signal) => { const next = unwrapJob(await request(apiPath(id, "/start"), { method: "POST", body: { plan_id: plan }, signal })); if (state.selected === id) renderJob(next); });
  });
  $("cancel-job").addEventListener("click", () => {
    const id = state.selected; if (!id || !LIVE.has(state.job?.status)) return;
    mutate(async (signal) => { const job = unwrapJob(await request(apiPath(id, "/cancel"), { method: "POST", body: {}, signal })); if (state.selected === id) renderJob(job); });
  });
  for (const id of ["prompt", "style", "actions", "preset"]) $(id).addEventListener("input", () => { state.draftID = state.draftFingerprint = null; });
  $("actions").addEventListener("input", actionInput); $("preset").addEventListener("change", presetDescription);
  $("estimate-ack").addEventListener("change", updateButtons); $("refresh").addEventListener("click", refresh);
  $("new-request").addEventListener("click", () => newRequest()); $("copy-request").addEventListener("click", () => newRequest(state.job?.request));
  for (const button of document.querySelectorAll("[data-view]")) button.addEventListener("click", () => showPreview(button.dataset.view));
  $("clip").addEventListener("change", () => { stopPlayback(); state.clip = $("clip").value; if (state.mode === "model") chooseModelClip(); else showPreview("video"); });
  $("model-play").addEventListener("click", () => {
    if (!state.modelReady || state.mode !== "model") return;
    if (!model.paused) return stopPlayback();
    if (model.currentTime >= model.duration - .02) { model.play({ repetitions: 1 }); model.pause(); model.currentTime = 0; }
    model.play({ repetitions: 1 }); paintModelTime(); if (!state.raf) state.raf = requestAnimationFrame(playbackTick);
  });
  model.addEventListener("finished", () => { stopPlayback(); paintModelTime(); });
  document.addEventListener("visibilitychange", () => { if (document.hidden) { stopPolling(); stopPlayback(); } else refresh(); });
  reducedMotion.addEventListener("change", () => { if (reducedMotion.matches && state.result) showPreview("portrait"); });
  if (typeof IntersectionObserver !== "undefined") new IntersectionObserver(([entry]) => { if (!entry.isIntersecting) stopPlayback(); }).observe($("preview-stage"));
  window.addEventListener("pagehide", () => { stopPolling(); clearPreview(); state.mutation?.abort(); if (state.referenceURL) URL.revokeObjectURL(state.referenceURL); });
  actionInput(); await refresh();
}

if (typeof document !== "undefined") bootStudio().catch(() => {
  const error = document.getElementById("connection-error");
  error.textContent = "The creation screen could not initialize. Refresh to reconnect; no request is retried automatically."; error.hidden = false;
});
