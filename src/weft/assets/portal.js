"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const token = new URLSearchParams(location.hash.slice(1)).get("token") || sessionStorage.getItem("weft-portal-token") || "";
  if (token) sessionStorage.setItem("weft-portal-token", token);
  history.replaceState(null, "", location.pathname);
  let selected = null, timer = null, pending = false, uploading = false, lastProjects = "", lastDetail = "";
  let museAvailable = false, reviewing = false;
  let checking = false, repairing = false;
  let preparingEquipment = false, choosingEquipment = false;
  let preflighting = false, preparingPerception = false, consentKey = "", perceptionRequest = null;
  let bodyDetailBusy = false, bodyDetailKey = "", bodyDetailRequest = null;
  let movingBusy = false, movingKey = "", movingChoices = [], movingFields = [];
  const movingActive = job => ["queued", "running", "cancelling"].includes(job.moving_equipment?.status);
  let creatingSupport = false, selectedJob = null;
  const reviewActive = job => ["queued", "running", "cancelling"].includes(job.muse?.status);
  const checkActive = job => ["queued", "running", "cancelling"].includes(job.muse?.checks?.status);
  const repairActive = job => ["queued", "running", "cancelling"].includes(job.muse?.repairs?.status);
  const equipmentActive = job => ["queued", "running", "cancelling"].includes(job.equipment_candidates?.status);
  const perceptionActive = job => !!job.perception?.busy || ["queued_preflight", "preflighting", "queued", "analyzing", "preparing_tracking", "tracking", "exporting_evidence", "cancelling"].includes(job.perception?.status);
  const terminal = new Set(["ready", "preview", "needs_input", "failed", "cancelled", "interrupted"]);
  const labels = {queued:"Queued",uploading:"Uploading",running:"Processing",ready:"Ready to explore",preview:"Body evidence ready · no physics",needs_input:"Scene inputs needed",failed:"Needs attention",cancelled:"Cancelled",interrupted:"Interrupted"};
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
    const signature = JSON.stringify([job, museAvailable, reviewing, checking, repairing, preflighting, preparingPerception, bodyDetailBusy]);
    if (lastDetail === signature) return;
    lastDetail = signature;
    $("project").hidden = false;
    $("project-heading").textContent = job.name;
    $("project-status").textContent = labels[job.status] || job.status;
    $("project-error").hidden = !job.error;
    $("project-error").textContent = job.error || "";
    const inventory = job.scene_inventory, imported = job.kind === "imported_lab";
    const legacyScope = !inventory && !imported && !job.equipment_candidate_id && !job.moving_candidate_id && job.control_profile !== "shared_world_exploration" && job.kind !== "support_experiment" && job.status === "ready" && job.route === "exercise";
    $("scene-inventory").hidden = !inventory && !imported && !legacyScope && !job.equipment_candidate_id && !job.moving_candidate_id;
    $("inventory-summary").textContent = job.control_profile === "moving_equipment_task" ? "Authored placement task on a saved body-and-object simulation. Manual controls include virtual forces; this is not recovered exercise intent or grasp. Inspect the task and its body/floor limits in the sandbox."
      : job.moving_candidate_id ? `Selected moving simulation option ${job.moving_candidate_id}. Manual, unscored exploration using saved body/object evidence and your explicit identity/depth assumptions. Physical depth, contact, grasp and task success are not verified.` : job.equipment_candidate_id ? `Selected conditional equipment candidate ${job.equipment_candidate_id}. Prepared evidence was verified; geometry, contact and task success remain unverified.` : inventory?.message || (imported
      ? `Imported source-bound simulation candidate, not automatic equipment reconstruction. Task success and physical/contact accuracy are not established. ${job.review_unavailable_reason || ""}`
      : legacyScope ? "Legacy person-only simulation: this upload did not record an explicit equipment declaration. Its saved world is unchanged; do not treat it as a full-scene reconstruction or verified equipment absence." : "");
    $("inventory-next-steps").replaceChildren(...(inventory?.next_steps || []).map(value => {const li=document.createElement("li");li.textContent=value;return li;}));
    artifactLink("inventory-download", job, "analysis/scene-inventory.json", !!inventory);
    showMoving(job);
    const equipment = job.equipment_candidates, equipmentReport = equipment?.report, equipmentAvailable = job.equipment_availability;
    const recovery = equipment?.recovery, recovered = recovery?.status === "recorded";
    $("equipment-candidates").hidden = !equipment && !(job.status === "needs_input" && equipmentAvailable);
    const recoveryNote = recovered ? `Previous preparation failure retained. ${recovery.started ? "Saved evidence reused" : "Saved evidence queued for reuse"}; no new API call or pose inference. ` : recovery ? `${recovery.reason} ` : "";
    const assemblyReasons = (equipmentReport?.assembly_reasons || []).map(reason => typeof reason === "string" ? reason : reason?.message || reason?.code || "Unspecified construction limitation");
    const roleExplanation = assemblyReasons.some(reason => reason.startsWith("insufficient_role_margin:") || reason.startsWith("unresolved_body_correspondence:")) ? "Could not reliably separate the person from equipment in the saved tracks. " : "";
    const proxyExplanation = assemblyReasons.includes("no_supported_rigid_proxy_candidates") ? "No supported equipment proxy could be proposed. " : "";
    const outcomeLabel = equipmentReport?.assembly_status === "abstained" ? "No world constructed (abstained)" : equipmentReport?.assembly_status || "unavailable";
    const assemblyOutcome = recovered && equipmentReport ? ` Assembly outcome: ${outcomeLabel}; ${(equipmentReport.candidates || []).filter(candidate => candidate.selectable).length} playable candidates. ${roleExplanation}${proxyExplanation}${assemblyReasons.join("; ")} Object identity, contact and physical accuracy remain unverified.` : "";
    $("equipment-status").textContent = recoveryNote + (equipment?.message || equipmentAvailable?.reason || "") + assemblyOutcome;
    $("equipment-source-options").replaceChildren(...(!equipment ? equipmentAvailable?.matches || [] : []).map(match => {
      const button = document.createElement("button");button.type = "button";
      button.textContent = `Build candidates · ${match.label}`;button.disabled = preparingEquipment;
      button.onclick = () => equipmentAction("prepare", match.id);return button;
    }));
    const sourceProof = equipmentReport?.source_validation;
    $("equipment-coverage").textContent = sourceProof ? `${sourceProof.body_reference_count} body pose samples across ${sourceProof.source_slots} decoded source slots; ${sourceProof.raw_object_ids.length} raw object tracks retained. Unresolved: ${sourceProof.unresolved_reasons.join("; ") || "No role ambiguity reported; physical accuracy is still unverified."}` : "";
    $("equipment-alternatives").replaceChildren(...(equipmentReport?.candidates || []).map(candidate => {
      const card = document.createElement("section"), title = document.createElement("h4"), details = document.createElement("p"), reasons = document.createElement("p"), choose = document.createElement("button");
      title.textContent = `${candidate.id} · ${candidate.selectable ? "Playable, untrained candidate" : candidate.status}`;
      details.textContent = `Person correspondence: ${candidate.assignment?.body_correspondence || "unresolved"}. Rigid proxies: ${(candidate.assignment?.rigid_proxies || []).join(", ") || "none"}. This assignment is a hypothesis; candidates may represent different tasks.`;
      reasons.textContent = candidate.reasons.length ? `Constructor result: ${candidate.reasons.join("; ")}` : "No task-success or physical-contact test has passed merely because this candidate is playable.";
      choose.textContent = `Choose ${candidate.id} and open sandbox`;choose.type = "button";
      choose.hidden = !candidate.selectable || !!job.equipment_candidate_id;choose.disabled = choosingEquipment;
      choose.onclick = () => equipmentAction("select", candidate.id);card.append(title, details, reasons, choose);return card;
    }));
    $("equipment-selection-note").textContent = job.equipment_candidate_id ? `This project is locked to ${job.equipment_candidate_id}; all recordings use that world. Other alternatives remain preserved in the report.` : equipmentReport ? recovered && !(equipmentReport.candidates || []).some(candidate => candidate.selectable) ? "No candidate is available to select. Review the retained uncertainty and construction reasons; no sandbox was substituted." : "Choose one admitted candidate before opening the sandbox. Selection locks this project; no score or automatic ranking makes the choice." : "";
    artifactLink("equipment-report-download", job, "equipment/candidates.json", !!equipmentReport);
    const perception = job.perception, quote = perception?.quote;
    const nextConsentKey = `${job.id}:${perception?.quote_sha256 || ""}`;
    if (consentKey !== nextConsentKey) {
      consentKey = nextConsentKey; perceptionRequest = null;
      $("perception-openai").checked = false; $("perception-meta").checked = false;
    }
    $("perception-preparation").hidden = !perception || (!perception.available && perception.status === "not_started") || (perception.status === "not_started" && !!equipmentAvailable?.available);
    $("perception-status").textContent = perception?.error || perception?.message || "";
    $("perception-preflight").hidden = !perception?.available || perception.status !== "not_started";
    $("perception-preflight").disabled = preflighting || perceptionActive(job);
    const detail = job.body_detail;
    const existingDetail = typeof detail?.child_id === "string" && /^[a-f0-9]{32}$/.test(detail.child_id);
    const canDetail = detail?.available === true && detail.recipe === "full-dense-region-v1";
    $("body-detail-option").hidden = !canDetail && !existingDetail && !detail?.parent_id;
    $("body-detail-reason").textContent = detail?.reason || "";
    $("body-detail-start").hidden = !canDetail || existingDetail;
    $("body-detail-start").disabled = bodyDetailBusy || perceptionActive(job);
    $("body-detail-open").hidden = !existingDetail;
    $("body-detail-open").onclick = () => { if (existingDetail) { selected = detail.child_id; lastProjects = ""; lastDetail = ""; refresh(); } };
    if (bodyDetailKey !== job.id) { bodyDetailKey = job.id; bodyDetailRequest = null; }
    $("perception-consent").hidden = perception?.status !== "awaiting_consent" || !quote;
    $("perception-quote").textContent = quote ? `${quote.frame_count} source frames. OpenAI (${quote.openai_model}): reserve up to $${quote.openai_max_usd.toFixed(4)}. Meta (${quote.meta_model}): reserve up to $${quote.meta_max_usd.toFixed(4)}. ${quote.message}` : "";
    updatePerceptionConsent();
    $("perception-events").replaceChildren(...(perception?.events || []).slice(-40).map(event => { const li = document.createElement("li"); li.textContent = event.message; return li; }));
    const poseDiagnostic = perception?.status === "failed" ? perception.pose_diagnostic : null;
    $("perception-pose-diagnostic").hidden = !poseDiagnostic;
    $("perception-pose-counts").textContent = poseDiagnostic ? `${poseDiagnostic.usable_samples} of ${poseDiagnostic.sampled_frames} sampled poses usable; ${poseDiagnostic.identity_breaks} identity breaks. The body could not be completed.` : "";
    $("perception-pose-explanation").textContent = poseDiagnostic?.message || "";
    $("perception-pose-barriers").replaceChildren(...(poseDiagnostic?.listed_barriers || []).map(row => {
      const li = document.createElement("li");
      const decoder = row.decoder_timestamp_s === null ? "decoder time unavailable" : `decoder-reported time ${row.decoder_timestamp_s.toFixed(3)} s`;
      li.textContent = `Source frame ${row.frame}: ${row.hypothesis_count} competing body estimates; ${row.source_time_s.toFixed(3)} s by frame rate (${decoder}). Neither estimate was selected.`;
      return li;
    }));
    $("perception-pose-window-note").textContent = poseDiagnostic ? `Showing ${poseDiagnostic.listed_barriers.length} of ${poseDiagnostic.identity_breaks} identity breaks and ${poseDiagnostic.listed_windows.length} of ${poseDiagnostic.window_count} source windows. The complete report retains every sampled frame and both raw alternatives. These are diagnostic windows, not playable or selectable episodes. Times are unchanged source estimates, not a completed native-clock reconstruction.` : "";
    $("perception-pose-windows").replaceChildren(...(poseDiagnostic?.listed_windows || []).map(row => {
      const li = document.createElement("li");
      li.textContent = `Frames ${row.frame_interval[0]}–${row.frame_interval[1]}: ${row.usable_samples}/${row.sampled_frames} usable poses. ${row.screen_passed ? "Count, coverage and duration checks pass; other checks remain untested." : "Count, coverage or duration checks fail."}`;
      return li;
    }));
    $("perception-pose-scope").textContent = poseDiagnostic?.scope || "";
    const artifactLabels = {body_detail_request:"Detailed attempt: original source and request",body_detail_recipe:"Detailed attempt: all-estimate region, model and sampling",person_region:"Person tracking region and excluded landmarks",body_model:"Estimated body model (no physics validation)",body_evidence:"Original inferred pose evidence",raw_poses:"All raw body estimates, including competing alternatives",pose_diagnostic:"Complete identity-break report and retained source poses",pose_diagnostic_summary:"Identity-break summary",pose_diagnostic_manifest:"Diagnostic source and failure identities",pose_diagnostic_error:"Why a pose diagnostic was unavailable",body_clock:"Native video timestamps",body_preview:"Source pose overlay (not simulation)",preflight_result:"Local check results",quote:"Saved cost and media quote",request:"Saved consent request",analysis_receipt:"OpenAI usage receipt",bridge_result:"Tracking preparation result",prepared_source:"Prepared source identities",failure:"Preparation failure details",preflight_receipt:"Local check resource receipt",preparation_receipt:"AI preparation resource receipt"};
    if (recovered) artifactLabels.saved_recovery = "Saved-evidence recovery receipt · no new API call";
    const perceptionArtifacts = {...(perception?.artifacts || {}), ...(recovered ? {saved_recovery: recovery.artifact} : {})};
    $("perception-artifacts").replaceChildren(...Object.entries(perceptionArtifacts).filter(([key]) => artifactLabels[key]).map(([key,path]) => {
      const a = document.createElement("a"); a.textContent = artifactLabels[key]; a.className = "text-link"; a.href = fileURL(job.id,path); a.download = path.split("/").pop(); return a;
    }));
    const preview = job.status === "preview" ? job.body_preview : null;
    $("body-preview").hidden = !preview;
    $("body-preview-image").hidden = !preview?.image;
    $("body-preview-image-note").hidden = !preview || !!preview.image;
    if (preview?.image) $("body-preview-image").src = fileURL(job.id, preview.image);
    for (const [id,key] of [["model","model"],["data","body"],["evidence","evidence"],["clock","clock"],["manifest","manifest"]])
      artifactLink(`body-preview-${id}`, job, preview?.[key] || "", !!preview?.[key]);
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
    $("cancel").hidden = terminal.has(job.status) && !reviewActive(job) && !checkActive(job) && !repairActive(job) && !equipmentActive(job) && !perceptionActive(job) && !movingActive(job);
    $("cancel").textContent = movingActive(job) ? "Cancel moving preparation" : perceptionActive(job) ? "Cancel video preparation" : equipmentActive(job) ? "Cancel equipment candidates" : repairActive(job) ? "Cancel suggestion tests" : checkActive(job) ? "Cancel local checks" : reviewActive(job) ? "Cancel Muse review" : "Cancel processing";
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
    const importedReviewReady = imported && job.review_attachment?.available;
    const conditionalReview = !!job.equipment_candidate_id;
    const reviewOnly = imported || conditionalReview || job.control_profile === "shared_world_exploration";
    artifactLink("attachment-receipt-download", job, "review-evidence/receipt.json", importedReviewReady);
    artifactLink("attachment-context-download", job, "review-evidence/prepared/context.json", importedReviewReady && job.review_attachment.prepared);
    artifactLink("attachment-request-download", job, "review-evidence/prepared/request.json", importedReviewReady && job.review_attachment.prepared);
    $("muse-review").hidden = !!job.review_unavailable_reason || (imported && !importedReviewReady) || (job.status !== "ready" && !job.muse);
    $("review-muse").hidden = !!job.muse;
    $("review-muse").disabled = (imported && !importedReviewReady) || !museAvailable || reviewing;
    const legacyBodyReview = job.kind === "support_experiment" && job.muse?.status === "proposed" && job.muse.review_scope !== "coupled_world_assumptions";
    $("muse-scope").hidden = job.route !== "exercise" && job.kind !== "support_experiment";
    $("muse-scope").textContent = importedReviewReady
      ? "Verified attached body observations and the current equipment world's assumptions are reviewed together. Physical accuracy is unverified. Imported local checks and repairs remain unavailable."
      : conditionalReview
      ? "Muse rechecks the selected world and original body evidence before sending a request. Object identities and physical accuracy remain uncertain. Local checks and repairs are unavailable for this candidate."
      : legacyBodyReview
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
    $("run-checks").hidden = reviewOnly || job.muse?.status !== "proposed" || !!checks;
    $("run-checks").disabled = checking;
    $("checks-help").hidden = reviewOnly || job.muse?.status !== "proposed";
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
    $("repair-comparison").hidden = reviewOnly || (job.muse?.status !== "proposed" && !repairs);
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
      const active = data.jobs.some(job => !terminal.has(job.status) || reviewActive(job) || checkActive(job) || repairActive(job) || equipmentActive(job) || perceptionActive(job) || movingActive(job));
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
      if (mode !== "objects") {
        query.set("exercise_inventory", $("exercise-inventory").value);
        query.set("exercise_output", $("exercise-output").value);
      }
      const job = await request(`/api/projects?${query}`, {method:"POST",body:video,headers:{"Content-Type":"application/octet-stream"},signal:AbortSignal.timeout(120000)});
      selected = job.id; $("upload-status").textContent = "Video saved. Processing continues in the background."; await show(job);
    } catch (e) { error(`${e.message}. Refresh the project list before submitting again; an upload may already have been accepted.`); $("upload-status").textContent = "Upload did not finish normally."; }
    finally { uploading = false; $("create").disabled = false; refresh(); }
  };
  $("mode").onchange = () => { $("exercise-options").hidden = $("mode").value === "objects"; };
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
  function updatePerceptionConsent() {
    const p = selectedJob?.perception;
    $("perception-prepare").disabled = preparingPerception || p?.status !== "awaiting_consent" || !p?.quote || !p?.quote_sha256 || perceptionActive(selectedJob || {}) || !$("perception-openai").checked || !$("perception-meta").checked;
  }
  $("perception-openai").onchange = updatePerceptionConsent;
  $("perception-meta").onchange = updatePerceptionConsent;
  $("perception-preflight").onclick = async () => {
    if (!selected || preflighting || selectedJob?.perception?.status !== "not_started" || !selectedJob.perception.available) return;
    preflighting = true; $("perception-preflight").disabled = true;
    const identity = selected;
    try {
      const job = await request(`/api/projects/${identity}/perception/preflight`, {method:"POST"});
      if (selected === identity) await show(job);
    } catch(e) { error(`${e.message}. Refresh to inspect local progress; no automatic retry was made.`); }
    finally { preflighting = false; lastDetail = ""; refresh(); }
  };
  $("perception-prepare").onclick = async () => {
    updatePerceptionConsent();
    if (!selected || $("perception-prepare").disabled) return;
    const identity = selected, p = selectedJob.perception;
    perceptionRequest ||= {quote_sha256:p.quote_sha256, consent:{openai_sampled_images:true,meta_full_video_and_regions:true},
      openai_max_usd:p.quote.openai_max_usd,meta_max_usd:p.quote.meta_max_usd,request_id:crypto.randomUUID().replaceAll("-", "")};
    preparingPerception = true; updatePerceptionConsent();
    try {
      const job = await request(`/api/projects/${identity}/perception/prepare`, {method:"POST",body:JSON.stringify(perceptionRequest),headers:{"Content-Type":"application/json"}});
      if (selected === identity) await show(job);
    } catch(e) { error(`${e.message}. Inspect the saved request and receipts before taking another action; no automatic retry was made.`); }
    finally { preparingPerception = false; lastDetail = ""; refresh(); }
  };
  $("body-detail-start").onclick = async () => {
    if (!selected || bodyDetailBusy || perceptionActive(selectedJob || {}) || selectedJob?.body_detail?.available !== true
        || selectedJob.body_detail.recipe !== "full-dense-region-v1") return;
    const identity = selected;
    bodyDetailRequest ||= {recipe:"full-dense-region-v1",request_id:crypto.randomUUID().replaceAll("-", "")};
    bodyDetailBusy = true; $("body-detail-start").disabled = true;
    try {
      const child = await request(`/api/projects/${identity}/perception/body-detail`, {method:"POST",body:JSON.stringify(bodyDetailRequest),headers:{"Content-Type":"application/json"}});
      if (selected === identity) { lastProjects = ""; await show(child); }
    } catch (e) { error(`${e.message}. The original attempt is preserved. Refresh to inspect progress; no automatic retry was made.`); }
    finally { bodyDetailBusy = false; lastDetail = ""; refresh(); }
  };
  function showMoving(job) {
    const state = job.moving_equipment;
    $("moving-equipment").hidden = !state?.available && !state?.status;
    $("moving-status").textContent = state?.error || state?.message || "";
    $("moving-prepare-form").hidden = !state?.available;
    $("moving-prepare").disabled = movingBusy;
    artifactLink("moving-family-download", job, "moving-equipment/family.json", !!state?.family);
    artifactLink("moving-report-download", job, "moving-equipment/candidates.json", !!state?.report);
    const family = state?.family;
    $("moving-family").textContent = family ? `${family.factors.length} raw tracks; all ${family.identity_family.assignment_count} possible nonempty assignments of these tracks remain saved, along with every depth range. Choose up to three simulation options to build; no option is automatically ranked or dropped. ${family.settings ? `Your assumptions: camera distance ${family.settings.camera.distance_model_units} model units; depth domain [${family.settings.depth_domain_model_units.join(", ")}] model units; sphere from image-box size; mass ${family.settings.mass_kg} kg; friction [${family.settings.friction.join(", ")}]; first observed position extended backward with zero velocity.` : ""}` : "";
    $("moving-choice-editor").hidden = state?.status !== "awaiting_choices";
    const key = `${job.id}:${family?.family_sha256 || ""}`;
    if (key !== movingKey) {
      movingKey = key; movingChoices = []; movingFields = [];
      $("moving-factors").replaceChildren(...(family?.factors || []).map(factor => {
        const section=document.createElement("section"), title=document.createElement("h4"), explanation=document.createElement("p");
        title.textContent=`${factor.object_id} · ${{feasible_intervals:"Depth choices available",no_solution:"No allowed depth",boundary_only:"Only uncertain boundaries remain",unsupported_input:"Insufficient source support"}[factor.status] || factor.status}`;
        explanation.textContent=`${factor.certified_intervals.length} allowed depth ranges; ${factor.possible_intervals.length} possible ranges before boundary uncertainty. Allowed means initial model clearance only, not verified physical depth. ${factor.boundary_ambiguity ? "Uncertain boundaries cannot be selected. " : ""}${factor.reasons.join("; ")}`;
        const intervals=document.createElement("p"), previous=document.createElement("button"), next=document.createElement("button");
        let page=0;
        const render=()=>{const start=page*32;intervals.textContent=factor.certified_intervals.slice(start,start+32).map((range,i)=>`#${start+i+1}: [${range[0]}, ${range[1]}]`).join("; ") || "No allowed depth is available.";previous.disabled=page===0;next.disabled=start+32>=factor.certified_intervals.length;};
        previous.type=next.type="button";previous.textContent="Previous depth ranges";next.textContent="Next depth ranges";
        previous.onclick=()=>{if(page>0){page--;render();}};next.onclick=()=>{if((page+1)*32<factor.certified_intervals.length){page++;render();}};render();
        const include=document.createElement("input"), index=document.createElement("input"), depth=document.createElement("input");
        include.type="checkbox";include.disabled=!factor.certified_intervals.length;index.type=depth.type="number";index.min="1";index.step="1";depth.step="any";
        const label=(text,input)=>{const label=document.createElement("label");label.textContent=text;label.append(input);return label;};
        section.append(title,explanation,intervals,previous,next,label("Treat this raw track as equipment",include),label("Allowed depth range number (starts at 1)",index),label("Depth in model units (explicit)",depth));
        movingFields.push({factor,include,index,depth});return section;
      }));
      renderMovingChoices();
    }
    $("moving-build").disabled = movingBusy || !movingChoices.length || state?.status !== "awaiting_choices";
    $("moving-candidates").replaceChildren(...(state?.report?.candidates || []).map(row=>{
      const section=document.createElement("section"), text=document.createElement("p"), button=document.createElement("button");
      text.textContent=`${row.id}: ${row.status}. ${row.reasons.join("; ")} Manual exploration only; no task success or physical validation.`;
      button.type="button";button.textContent=`Open ${row.id} and lock this world`;button.hidden=!row.selectable || !!job.moving_candidate_id;button.disabled=movingBusy;
      button.onclick=()=>movingAction("select",{id:row.id});section.append(text,button);return section;
    }));
    if(job.moving_candidate_id) $("moving-status").textContent=`This project is locked to ${job.moving_candidate_id}. Other hypotheses and earlier failures remain saved. Manual controls and recording use only this world.`;
  }
  function renderMovingChoices() {
    $("moving-choices").replaceChildren(...movingChoices.map((choice,i)=>{const row=document.createElement("li");row.textContent=`Hypothesis ${i+1}: `+choice.equipment.map(e=>`${e.object_id}, range #${e.interval_index+1}, depth ${e.depth_model_units}`).join("; ");return row;}));
    $("moving-add-choice").disabled=movingChoices.length>=3;
    $("moving-remove-choice").disabled=!movingChoices.length;
    $("moving-build").disabled=!movingChoices.length || movingBusy;
  }
  async function movingAction(action,payload) {
    if(!selected || movingBusy)return;
    movingBusy=true;const identity=selected;
    try {const job=await request(`/api/projects/${identity}/moving-equipment/${action}`,{method:"POST",body:JSON.stringify(payload),headers:{"Content-Type":"application/json"}});if(selected===identity){lastDetail="";await show(job);}}
    catch(e){error(`${e.message}. Saved evidence and previous attempts were preserved; no automatic retry.`);}
    finally{movingBusy=false;lastDetail="";refresh();}
  }
  $("moving-prepare-form").onsubmit=event=>{
    event.preventDefault();
    if(!$("moving-confirm").checked)return error("Confirm the authored camera, material and prefix assumptions.");
    try {
      const number=id=>{const text=$(id).value.trim();if(!text || !Number.isFinite(Number(text)))throw new Error("Enter every moving assumption explicitly.");return Number(text);};
      movingAction("prepare",{schema_version:1,profile:"moving-depth-family-priors-v1",camera:{distance_model_units:number("moving-camera")},depth_domain_model_units:[number("moving-depth-low"),number("moving-depth-high")],shape:{method:"sphere_from_median_image_box_diagonal"},mass_kg:number("moving-mass"),friction:[number("moving-friction"),.005,.0001],initialization:{method:"first_initialized_box_zero_velocity"}});
    }catch(e){error(e.message);}
  };
  $("moving-add-choice").onclick=()=>{
    try {
      if(movingChoices.length>=3)throw new Error("At most three hypotheses can be constructed in this attempt; the all alternatives remain saved.");
      const equipment=movingFields.filter(row=>row.include.checked).map(({factor,index,depth})=>{
        if(!index.value.trim() || !depth.value.trim())throw new Error("Enter a range number and depth for each included track.");
        const i=Number(index.value)-1,d=Number(depth.value),range=factor.certified_intervals[i];
        if(!Number.isInteger(i)||!range||!Number.isFinite(d)||d<range[0]||d>range[1])throw new Error("Depth must lie inside its specified allowed range.");
        return {object_id:factor.object_id,interval_index:i,depth_model_units:d};
      }).sort((a,b)=>a.object_id<b.object_id?-1:a.object_id>b.object_id?1:0);
      if(!equipment.length)throw new Error("Include at least one raw equipment track; the complement stays unknown.");
      const choice={equipment};if(movingChoices.some(old=>JSON.stringify(old)===JSON.stringify(choice)))throw new Error("That explicit hypothesis is already listed.");
      movingChoices.push(choice);renderMovingChoices();
    }catch(e){error(e.message);}
  };
  $("moving-remove-choice").onclick=()=>{movingChoices.pop();renderMovingChoices();};
  $("moving-build").onclick=()=>movingAction("build",{schema_version:1,profile:"moving-depth-selection-v1",family_sha256:selectedJob?.moving_equipment?.family?.family_sha256,choices:movingChoices});
  async function equipmentAction(action, id) {
    if (!selected || preparingEquipment || choosingEquipment) return;
    preparingEquipment = action === "prepare";choosingEquipment = action === "select";
    const identity = selected;
    try {
      const job = await request(`/api/projects/${identity}/equipment/${action}`, {method:"POST", body:JSON.stringify({id}), headers:{"Content-Type":"application/json"}});
      if (selected === identity) await show(job);
    } catch (e) { error(`${e.message}. Existing source and candidates were preserved; no automatic retry.`); }
    finally { preparingEquipment = false;choosingEquipment = false;lastDetail = "";refresh(); }
  }
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
