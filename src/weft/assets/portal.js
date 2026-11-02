"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const token = new URLSearchParams(location.hash.slice(1)).get("token") || sessionStorage.getItem("weft-portal-token") || "";
  if (token) sessionStorage.setItem("weft-portal-token", token);
  history.replaceState(null, "", location.pathname);
  let selected = null, timer = null, pending = false, uploading = false, lastProjects = "", lastDetail = "";
  let museAvailable = false, reviewing = false;
  let checking = false, repairing = false;
  let creatingSupport = false, selectedJob = null;
  const reviewActive = job => ["queued", "running", "cancelling"].includes(job.muse?.status);
  const checkActive = job => ["queued", "running", "cancelling"].includes(job.muse?.checks?.status);
  const repairActive = job => ["queued", "running", "cancelling"].includes(job.muse?.repairs?.status);
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
    selectedJob = job;
    const signature = JSON.stringify([job, museAvailable, reviewing, checking, repairing]);
    if (lastDetail === signature) return;
    lastDetail = signature;
    $("project").hidden = false;
    $("project-heading").textContent = job.name;
    $("project-status").textContent = labels[job.status] || job.status;
    $("project-error").hidden = !job.error;
    $("project-error").textContent = job.error || "";
    const readiness = job.status === "failed" ? job.reconstruction_readiness : null;
    $("reconstruction-readiness").hidden = !readiness;
    if (readiness) {
      $("readiness-coverage").textContent = `${readiness.usable_samples} of ${readiness.sampled_frames} sampled poses usable (${(readiness.usable_fraction * 100).toFixed(1)}%); ${(readiness.minimum_usable_fraction * 100).toFixed(0)}% required. ${readiness.competing_samples} frames have competing pose hypotheses.`;
      const gap = readiness.longest_gap;
      const clock = readiness.time_basis === "native_pts_elapsed_time" ? "native video time" : "estimated frame-rate time, not native timestamps";
      $("readiness-gap").textContent = gap ? `Longest unclear interval: ${gap.first_time_s.toFixed(2)}–${gap.last_time_s.toFixed(2)} seconds (${clock}); ${gap.sample_count} sampled poses. Missing poses were not filled in.` : `No unusable pose interval found (${clock}); inspect the original construction error.`;
      $("readiness-retry").textContent = readiness.retry_reason;
    }
    artifactLink("readiness-download", job, "analysis/reconstruction-readiness.json", !!readiness);
    $("cancel").hidden = terminal.has(job.status) && !reviewActive(job) && !checkActive(job) && !repairActive(job);
    $("cancel").textContent = repairActive(job) ? "Cancel suggestion tests" : checkActive(job) ? "Cancel local checks" : reviewActive(job) ? "Cancel Muse review" : "Cancel processing";
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
    artifactLink("source-download", job, "source" + job.name.slice(job.name.lastIndexOf(".")).toLowerCase(), job.source_available ?? !!job.source_sha256);
    artifactLink("plan-download", job, "analysis/plan.json", !!job.plan);
    artifactLink("receipt-download", job, "analysis/receipt.json", !!job.receipt);
    artifactLink("failure-download", job, "failure.json", job.status === "failed");
    $("support-editor").hidden = !job.support_experiment_available;
    $("experiment-note").hidden = job.kind !== "support_experiment";
    $("experiment-note").textContent = job.kind === "support_experiment" ? "Authored support experiment: source-bound body, user-specified box and load task. Equipment/contact are not recovered. This is a new task, not a scored repair of the parent project." : "";
    artifactLink("support-priors-download", job, "support-priors.json", job.kind === "support_experiment");
    $("muse-review").hidden = job.status !== "ready" && !job.muse;
    $("review-muse").hidden = !!job.muse;
    $("review-muse").disabled = !museAvailable || reviewing;
    const legacyBodyReview = job.kind === "support_experiment" && job.muse?.status === "proposed" && job.muse.review_scope !== "coupled_world_assumptions";
    $("muse-scope").hidden = job.route !== "exercise" && job.kind !== "support_experiment";
    $("muse-scope").textContent = legacyBodyReview
      ? "This saved review covers reused body evidence only, not the authored boxes or load requirements."
      : "New exercise reviews include the saved world's geometry, supports and control assumptions. Authored settings remain separate from observed video evidence; scores and learned policies are not sent.";
    $("muse-status").textContent = job.muse ? (job.muse.message || job.muse.error || job.muse.status) : museAvailable ? "One bounded request, charged to the Meta allowance shown above." : "Meta key or remaining allowance is unavailable. Your simulation remains usable.";
    const advice = job.muse?.proposal?.advice;
    $("muse-summary").textContent = advice?.summary || "";
    $("muse-hypotheses").replaceChildren(...(advice?.hypotheses || []).map(hypothesis => {
      const section = document.createElement("section"), heading = document.createElement("h4"), claim = document.createElement("p"), evidence = document.createElement("p"), alternatives = document.createElement("p"), tests = document.createElement("ul");
      heading.textContent = `${hypothesis.concept} · ${hypothesis.status} · not applied`;
      claim.textContent = hypothesis.claim;
      evidence.textContent = hypothesis.evidence_ids.length ? `Evidence: ${hypothesis.evidence_ids.join(", ")}` : "Prior assumption; no observation cited.";
      evidence.className = "hint";
      alternatives.textContent = `Other explanations: ${hypothesis.alternatives.join("; ") || "None proposed"}`;
      tests.replaceChildren(...hypothesis.required_tests.map(value => { const li = document.createElement("li"); li.textContent = `Test needed: ${value}`; return li; }));
      section.append(heading, claim, evidence, alternatives, tests); return section;
    }));
    $("muse-questions").replaceChildren(...(advice?.unanswered_questions || []).map(value => { const li = document.createElement("li"); li.textContent = `Still unknown: ${value}`; return li; }));
    artifactLink("muse-proposal-download", job, "muse/advice/hypotheses.json", !!advice);
    artifactLink("muse-context-download", job, "muse/advice/context.json", !!job.muse?.receipt);
    artifactLink("muse-receipt-download", job, "muse/advice/advisor-call.json", !!job.muse?.receipt);
    const checks = job.muse?.checks, report = checks?.report;
    $("run-checks").hidden = job.muse?.status !== "proposed" || !!checks;
    $("run-checks").disabled = checking;
    $("checks-help").hidden = job.muse?.status !== "proposed";
    $("checks-status").textContent = checks?.message || checks?.status || "";
    const controls = {authored:"Authored controller",root_unforced:"Controller without root assistance",coast:"No control",positive_x:"Push right",negative_x:"Push left"};
    $("checks-results").replaceChildren(...(report?.trials || []).map(trial => {
      const li = document.createElement("li"); li.textContent = `${controls[trial.condition] || trial.condition}: ${trial.steps}/${trial.horizon_steps} steps · original task success: ${trial.success ? "yes" : "no"} · total reward: ${trial.reward_sum.toFixed(3)}`; return li;
    }));
    artifactLink("checks-report-download", job, "muse/checks/report.json", !!report);
    $("checks-traces").replaceChildren(...(report?.trials || []).map(trial => {
      const a = document.createElement("a"); a.textContent = `${controls[trial.condition] || trial.condition} actions and states`; a.className = "text-link";
      a.href = fileURL(job.id, "muse/checks/" + trial.trace); a.download = trial.trace; return a;
    }));
    const repairs = job.muse?.repairs, repairReport = repairs?.report, repairAvailability = job.muse?.repair_availability;
    $("repair-comparison").hidden = job.muse?.status !== "proposed" && !repairs;
    $("run-repairs").hidden = !repairAvailability?.available || !!repairs;
    $("run-repairs").disabled = repairing;
    $("repairs-status").textContent = repairs?.message || repairs?.error || repairs?.status || repairAvailability?.reason || "A supported evidence-linked control or articulation hypothesis is required.";
    $("repairs-outcome").textContent = repairReport ? (repairReport.accepted_in_simulation
      ? `Candidate ${repairReport.selected_candidate} improved the fixed simulator score. Nothing was applied.`
      : "No candidate improved enough. The original controller remains unchanged.") : "";
    const score = value => Number.isFinite(value) ? value.toFixed(4) : "not available";
    $("repairs-results").replaceChildren(...(repairReport?.candidates || []).slice(0, 3).map(candidate => {
      const li = document.createElement("li");
      li.textContent = `${candidate.id}: score ${score(candidate.candidate_score)} vs original ${score(candidate.baseline_score)} · original task gates: ${candidate.all_original_gates_passed ? "passed" : "not passed"} · ${candidate.accepted_in_simulation ? "improved in simulation, not applied" : "rejected"}. ${candidate.reason || ""}`;
      return li;
    }));
    artifactLink("repairs-report-download", job, "muse/repairs/run/report.json", !!repairReport);
    artifactLink("repairs-plan-download", job, "muse/repairs/run/plan.json", !!repairReport);
    artifactLink("repairs-runtime-download", job, "muse/repairs/run/runtime-source.json", !!repairReport);
    artifactLink("repairs-manifest-download", job, "muse/repairs/run/manifest.json", !!repairReport);
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
      museAvailable = !!data.muse?.available;
      $("muse-budget").textContent = data.muse?.budget_configured ? `Meta / Muse: $${data.muse.remaining_usd.toFixed(3)} remaining of $${data.muse.total_budget_usd.toFixed(2)} · ${data.muse.unsettled_count} uncertain. Estimated reservation $${data.muse.estimated_one_call_usd.toFixed(4)} per review; this is not a provider hard cap.` : "Meta / Muse allowance unavailable. No review will be sent.";
      const signature = JSON.stringify(data.jobs.map(job => [job.id,job.name,job.status,job.stage,selected]));
      if (signature !== lastProjects) {
        lastProjects = signature;
        $("empty").hidden = data.jobs.length > 0;
        $("projects").replaceChildren(...data.jobs.map(job => {
          const button = document.createElement("button"); button.type = "button"; button.className = "project-card";
          button.setAttribute("aria-pressed", String(selected === job.id));
          const name = document.createElement("strong"); name.textContent = job.name;
          const meta = document.createElement("small"); meta.textContent = job.kind === "support_experiment" ? "Authored support experiment · reused body" : job.ai ? "OpenAI + local reconstruction" : "Local reconstruction";
          const title = document.createElement("span"); title.append(name,meta);
          const state = document.createElement("span"); state.className = "state"; state.textContent = labels[job.status] || job.status;
          button.append(title,state); button.onclick = () => { selected = job.id; lastProjects = ""; show(job); refresh(); }; return button;
        }));
      }
      if (!selected && data.jobs.length) selected = data.jobs[0].id;
      await show(data.jobs.find(job => job.id === selected));
      $("create").disabled = uploading;
      const active = data.jobs.some(job => !terminal.has(job.status) || reviewActive(job) || checkActive(job) || repairActive(job));
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
  $("review-muse").onclick = async () => {
    if (!selected || reviewing || !museAvailable) return;
    reviewing = true; $("review-muse").disabled = true;
    const identity = selected;
    try {
      const job = await request(`/api/projects/${identity}/muse`, {method:"POST"});
      if (selected === identity) await show(job);
    } catch (e) { error(`${e.message}. Refresh to inspect the saved review; no automatic retry was made.`); }
    finally { reviewing = false; lastDetail = ""; refresh(); }
  };
  $("run-checks").onclick = async () => {
    if (!selected || checking) return;
    checking = true; $("run-checks").disabled = true;
    const identity = selected;
    try {
      const job = await request(`/api/projects/${identity}/muse/checks`, {method:"POST"});
      if (selected === identity) await show(job);
    } catch (e) { error(e.message); }
    finally { checking = false; lastDetail = ""; refresh(); }
  };
  $("run-repairs").onclick = async () => {
    if (!selected || repairing || !selectedJob?.muse?.repair_availability?.available || selectedJob?.muse?.repairs) return;
    repairing = true; $("run-repairs").disabled = true;
    const identity = selected;
    try {
      const job = await request(`/api/projects/${identity}/muse/repairs`, {method:"POST"});
      if (selected === identity) await show(job);
    } catch (e) { error(`${e.message}. Existing comparison evidence will not be retried or overwritten.`); }
    finally { repairing = false; lastDetail = ""; refresh(); }
  };
  $("support-form").onsubmit = async event => {
    event.preventDefault();
    if (creatingSupport || !selectedJob?.support_experiment_available) return;
    if (!$("support-confirm").checked) return error("Confirm that support placement and load are your simulation assumptions.");
    const number = id => {
      if (!$(id).value.trim() || !Number.isFinite(Number($(id).value))) throw new Error("Enter every position, dimension and load-window value.");
      return Number($(id).value);
    };
    let priors;
    try {
      priors = {schema_version:1,profile:"shared-static-support-priors-v1",source_sha256:selectedJob.source_sha256,scale:1,
        root_assistance:$("support-root").checked,objects:[{id:"support_box",label:"User-authored support box",role:"static",shape:"box",
          position:[number("support-x"),number("support-y"),number("support-z")],size:[number("support-width"),number("support-depth"),number("support-height")],
          quaternion:[1,0,0,0],velocity:[0,0,0],angular_velocity:[0,0,0],mass:10,friction:[.8,.01,.001],color:"#707070",
          provenance:{mode:"authored_addition",calibration_status:"estimated_prior",physical_validation:"unverified",source_contact_verified:false,
            assumptions:["User-specified position and size; axis-aligned static box and material defaults are authored, not measured."]}}],
        support_requirements:[{object_id:"support_box",human_segment:$("support-segment").value,direction_world:[0,0,1],
          minimum_load_body_weight_fraction:number("support-load")/100,start_step:number("support-start"),end_step:number("support-end")}]};
    } catch(e) { return error(e.message); }
    creatingSupport = true; $("support-create").disabled = true;
    const identity = selectedJob.id;
    try {
      const job = await request(`/api/projects/${identity}/supports`, {method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({request_id:crypto.randomUUID().replaceAll("-",""),priors,train:$("support-train").checked})});
      selected = job.id; $("support-status").textContent = "Separate experiment queued; the original project is unchanged."; await show(job);
    } catch(e) { error(`${e.message}. Refresh the project list before submitting again; no automatic retry was made.`); }
    finally { creatingSupport = false; $("support-create").disabled = false; refresh(); }
  };
  $("refresh").onclick = refresh;
  document.addEventListener("visibilitychange", () => { clearTimeout(timer); if (!document.hidden) refresh(); });
  window.addEventListener("pagehide", () => clearTimeout(timer));
  refresh();
})();
