import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {Element} from "./lab_ui_smoke.mjs";

const html = fs.readFileSync(new URL("../src/weft/assets/portal.html", import.meta.url), "utf8");
const code = fs.readFileSync(new URL("../src/weft/assets/portal.js", import.meta.url), "utf8");
const tick = async () => { for(let i=0;i<40;i++) await Promise.resolve(); };
function setup(jobs=[]) {
  const elements = new Map();
  for (const match of html.matchAll(/<([a-z][\w-]*)\b([^>]*)>/gi)) {
    const attrs = Object.fromEntries([...match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g)].map(m=>[m[1],m[2]??""]));
    if(attrs.id) elements.set(attrs.id,new Element(match[1],attrs));
  }
  elements.get("exercise-inventory").value="unknown"; elements.get("exercise-output").value="body_preview";
  const timers=new Map(), delays=new Map(), calls=[], listeners=new Map(); let next=1;
  const document={hidden:false,getElementById:id=>elements.get(id),createElement:tag=>new Element(tag),querySelectorAll:()=>[],addEventListener:(name,fn)=>listeners.set(name,fn)};
  const info={credential:{configured:true},budget:{configured:true,remaining_usd:5,total_usd:5,calls:0,unsettled_calls:0,reservation_usd:.3135},muse:{available:true,budget_configured:true,remaining_usd:4.7,total_budget_usd:5,unsettled_count:0,estimated_one_call_usd:.058368},jobs};
  const context={document,location:{hash:"#token=test-owner-token",pathname:"/"},history:{replaceState(){}},sessionStorage:{getItem(){return null;},setItem(){}},URLSearchParams,AbortSignal,crypto:{randomUUID:()=>"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"},
    setTimeout:(fn,delay)=>{const id=next++;timers.set(id,fn);delays.set(id,delay);return id;},clearTimeout:id=>{timers.delete(id);delays.delete(id);},window:{addEventListener(){}},
    fetch:async(path,options={})=>{calls.push({path,options});if(path==="/api/portal")return {ok:true,json:async()=>structuredClone(info)};
      if(path.startsWith("/api/projects?"))return {ok:true,json:async()=>({id:"b".repeat(32),name:"clip.webm",status:"queued",stage:"queued",events:[]})};
      if(path.endsWith("/muse")){info.jobs[0].muse={status:"queued",message:"Waiting"};return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      if(path.endsWith("/muse/checks")){info.jobs[0].muse.checks={status:"queued",message:"Waiting for local comparisons"};return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      if(path.endsWith("/muse/repairs")){info.jobs[0].muse.repairs={status:"queued",message:"Waiting for suggestion tests",applied:false};info.jobs[0].muse.repair_availability.available=false;return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      if(path.endsWith("/perception/preflight")){info.jobs[0].perception={...info.jobs[0].perception,status:"queued_preflight"};return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      if(path.endsWith("/perception/prepare")){info.jobs[0].perception={...info.jobs[0].perception,status:"queued",quote:null};return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      if(path.endsWith("/perception/body-detail")){const parent=info.jobs.find(j=>path.includes(j.id));const child={id:"d".repeat(32),name:parent.name,kind:"video",route:"exercise",status:"needs_input",events:[],body_detail:{available:false,parent_id:parent.id,reason:"Detailed local attempt"},perception:{available:true,status:"queued_preflight",artifacts:{}}};parent.body_detail={...parent.body_detail,available:false,child_id:child.id};info.jobs.push(child);return {ok:true,json:async()=>structuredClone(child)};}
      if(path.endsWith("/moving-equipment/prepare")){info.jobs[0].moving_equipment={status:"queued",phase:"prepare",message:"Saved evidence only"};return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      if(path.endsWith("/moving-equipment/build")){info.jobs[0].moving_equipment.status="queued";info.jobs[0].moving_equipment.phase="build";return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      if(path.endsWith("/moving-equipment/select")){info.jobs[0].status="ready";info.jobs[0].moving_candidate_id=JSON.parse(options.body).id;info.jobs[0].control_profile="shared_world_exploration";info.jobs[0].review_unavailable_reason="Manual exploration has no qualified Muse review";info.jobs[0].sandbox=`/labs/${info.jobs[0].id}/`;return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      if(path.endsWith("/equipment/prepare")){info.jobs[0].equipment_candidates={status:"queued",message:"Waiting for conditional construction"};return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      if(path.endsWith("/equipment/select")){info.jobs[0].status="ready";info.jobs[0].equipment_candidate_id=JSON.parse(options.body).id;info.jobs[0].sandbox=`/labs/${info.jobs[0].id}/`;return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      if(path.endsWith("/supports")){const child={id:"e".repeat(32),name:info.jobs[0].name,kind:"support_experiment",parent_id:info.jobs[0].id,status:"queued",stage:"queued",events:[],source_available:false};info.jobs.unshift(child);return {ok:true,json:async()=>structuredClone(child)};}
      return {ok:true,json:async()=>({frames:[]})};},console};
  vm.runInNewContext(code,context);
  return {elements,calls,timers,delays,document,listeners,info};
}

{
  const id="7".repeat(32),parent={id,name:"Failed pose.mp4",kind:"video",route:"exercise",status:"needs_input",events:[],
    body_detail:{available:true,recipe:"full-dense-region-v1",reason:"Original pose quality failed <script>"},
    perception:{available:true,status:"failed",error:"Preserved first failure",artifacts:{failure:"perception/failure.json"}}};
  const f=setup([parent]);await tick();const e=f.elements;
  assert.equal(e.get("body-detail-option").hidden,false);assert.equal(e.get("body-detail-start").hidden,false);
  assert.equal(e.get("body-detail-reason").textContent,parent.body_detail.reason);
  assert.equal(f.calls.filter(c=>c.path.includes("/perception/")).length,0);
  await Promise.all([e.get("body-detail-start").onclick(),e.get("body-detail-start").onclick()]);await tick();
  const calls=f.calls.filter(c=>c.path.endsWith("/perception/body-detail"));assert.equal(calls.length,1);
  assert.deepEqual(JSON.parse(calls[0].options.body),{recipe:"full-dense-region-v1",request_id:"a".repeat(32)});
  assert.equal(parent.perception.status,"failed");assert.equal(parent.perception.error,"Preserved first failure");
  assert.equal(e.get("body-detail-start").hidden,true);assert.equal(e.get("perception-consent").hidden,true);
  e.get("projects").children[0].onclick();await tick();assert.equal(e.get("body-detail-open").hidden,false);
  e.get("body-detail-open").onclick();await tick();assert.equal(e.get("body-detail-start").hidden,true);
  assert.ok(!f.calls.some(c=>c.path.endsWith("/perception/prepare")||c.path.includes("/files/")||c.path.endsWith("/muse")));
  console.log("PASS detailed body extraction is an explicit local child action, deduplicated and preserves failed parent");
}

{
  const id="3".repeat(32),f=setup([{id,name:"Recovered equipment.mp4",kind:"video",route:"exercise",status:"needs_input",events:[],
    perception:{available:false,status:"failed",error:"Original source-publication bound exceeded",artifacts:{failure:"perception/failure.json"}},
    equipment_candidates:{status:"complete",message:"No candidate passed the unchanged constructor.",
      recovery:{status:"recorded",started:true,artifact:"equipment/recovery.json",api_calls:0,inference_calls:0},
      report:{assembly_status:"abstained",assembly_reasons:["insufficient_role_margin:sam-1","no_supported_rigid_proxy_candidates",{code:"ambiguous_role",message:"sam-1 remains ambiguous <script>"}],candidates:[]}}}]);await tick();
  const e=f.elements;
  assert.match(e.get("equipment-status").textContent,/Previous preparation failure retained.*Saved evidence reused.*no new API call or pose inference/);
  assert.match(e.get("equipment-status").textContent,/No world constructed \(abstained\); 0 playable candidates.*sam-1 remains ambiguous <script>/);
  assert.match(e.get("equipment-status").textContent,/Could not reliably separate the person from equipment.*No supported equipment proxy/);
  assert.match(e.get("equipment-status").textContent,/identity, contact and physical accuracy remain unverified/);
  assert.match(e.get("equipment-selection-note").textContent,/No candidate is available.*no sandbox was substituted/);
  assert.equal(e.get("perception-status").textContent,"Original source-publication bound exceeded");
  assert.equal(e.get("perception-preflight").hidden,true);assert.equal(e.get("perception-consent").hidden,true);
  assert.equal(e.get("sandbox").hidden,true);assert.equal(e.get("equipment-alternatives").children.length,0);
  assert.ok(e.get("perception-artifacts").children.some(a=>a.href.includes("equipment/recovery.json")));
  assert.ok(e.get("perception-artifacts").children.some(a=>a.href.includes("perception/failure.json")));
  assert.equal(f.calls.length,1);
  f.info.jobs[0].equipment_candidates.recovery.started=false;f.info.jobs[0].equipment_candidates.status="queued";
  f.info.jobs[0].equipment_candidates.report=null;await e.get("refresh").onclick();await tick();
  assert.match(e.get("equipment-status").textContent,/Saved evidence queued for reuse/);
  f.info.jobs[0].equipment_candidates.recovery={status:"unavailable",reason:"Recovery metadata changed"};
  await e.get("refresh").onclick();await tick();
  assert.match(e.get("equipment-status").textContent,/Recovery metadata changed/);
  assert.ok(!e.get("perception-artifacts").children.some(a=>a.href.includes("equipment/recovery.json")));
  console.log("PASS recovered zero-candidate assembly preserves original failure, safe reasons and no paid retry");
}
{
  const id="8".repeat(32), readiness={usable_samples:353,sampled_frames:554,usable_fraction:353/554,minimum_usable_fraction:.65,competing_samples:2,
    time_basis:"native_pts_elapsed_time",longest_gap:{first_time_s:26.118,last_time_s:32.156,sample_count:91},retry_reason:"Five supported retries cannot reach the unchanged visibility requirement."};
  const f=setup([{id,name:"unfamiliar.webm",status:"failed",stage:"failed",error:"Original source rejection",events:[],reconstruction_readiness:readiness}]);await tick();
  const e=f.elements;assert.equal(e.get("reconstruction-readiness").hidden,false);
  assert.match(e.get("readiness-coverage").textContent,/353 of 554.*63.7%.*65% required.*2 frames/);
  assert.match(e.get("readiness-gap").textContent,/26.12–32.16.*native video time.*91/);
  assert.equal(e.get("readiness-retry").textContent,readiness.retry_reason);
  assert.equal(e.get("project-error").textContent,"Original source rejection");assert.equal(e.get("sandbox").hidden,true);
  assert.match(e.get("readiness-download").href,/analysis\/reconstruction-readiness.json\?token=test-owner-token$/);
  assert.ok(!f.calls.some(call=>call.path.includes("reconstruction-readiness.json")));
  f.info.jobs[0].reconstruction_readiness.time_basis="frame_index_divided_by_fps";
  f.info.jobs[0].reconstruction_readiness.retry_reason="<script>untrusted diagnostic text</script>";
  await e.get("refresh").onclick();await tick();
  assert.match(e.get("readiness-gap").textContent,/estimated frame-rate time, not native timestamps/);
  assert.equal(e.get("readiness-retry").textContent,"<script>untrusted diagnostic text</script>");
  f.info.jobs[0].status="ready";f.info.jobs[0].sandbox=`/labs/${id}/`;
  await e.get("refresh").onclick();await tick();
  assert.equal(e.get("reconstruction-readiness").hidden,true);assert.equal(e.get("readiness-download").hidden,true);
  console.log("PASS rejected reconstruction coverage, honest clocks, safe diagnostic text and report download without eager fetch");
}
{
  const id="f".repeat(32), f=setup([{id,name:"body.webm",status:"ready",stage:"ready",events:[],source_sha256:"a".repeat(64),support_experiment_available:true}]);await tick();
  const e=f.elements;assert.equal(e.get("support-editor").hidden,false);
  e.get("support-confirm").checked=false;await e.get("support-form").onsubmit({preventDefault(){}});
  assert.ok(!f.calls.some(c=>c.path.endsWith("/supports")));
  e.get("support-confirm").checked=true;e.get("support-root").checked=false;e.get("support-train").checked=true;e.get("support-segment").value="left_forearm";
  for(const [name,value] of Object.entries({x:"1",y:"0.4",z:"0.2",width:"0.2",depth:"0.2",height:"0.2",start:"0",end:"30",load:"5"}))e.get("support-"+name).value=value;
  await e.get("support-form").onsubmit({preventDefault(){}});await tick();
  const call=f.calls.find(c=>c.path.endsWith("/supports")),body=JSON.parse(call.options.body);
  assert.equal(body.train,true);assert.equal(body.priors.source_sha256,"a".repeat(64));
  assert.equal(body.priors.objects[0].provenance.source_contact_verified,false);assert.equal(body.priors.support_requirements[0].minimum_load_body_weight_fraction,.05);
  assert.equal(e.get("source-download").hidden,true);assert.equal(e.get("experiment-note").hidden,false);
  assert.equal(e.get("muse-scope").hidden,false);
  assert.match(e.get("muse-scope").textContent,/New exercise reviews include/);
  assert.match(e.get("experiment-note").textContent,/not a scored repair/);assert.equal(e.get("support-editor").hidden,true);
  console.log("PASS explicit authored support inputs create a distinct experiment without claiming recovered contact");
}
{
  const id="9".repeat(32), f=setup([{id,name:"support.webm",kind:"support_experiment",route:"exercise",status:"ready",events:[],muse:{status:"proposed"}}]);await tick();
  assert.match(f.elements.get("muse-scope").textContent,/saved review covers reused body evidence only/);
  f.info.jobs[0].muse.review_scope="coupled_world_assumptions";
  await f.elements.get("refresh").onclick();await tick();
  assert.match(f.elements.get("muse-scope").textContent,/geometry, supports and control assumptions/);
  console.log("PASS saved body-only review scope stays distinct from new coupled-world review");
}
{
  const id="d".repeat(32), f=setup([{id,name:"exercise.webm",status:"ready",stage:"ready",events:[],sandbox:`/labs/${id}/`}]);await tick();
  const e=f.elements;assert.equal(e.get("review-muse").disabled,false);assert.match(e.get("muse-budget").textContent,/Meta \/ Muse/);
  await e.get("review-muse").onclick();await tick();
  assert.equal(f.calls.filter(c=>c.path.endsWith("/muse")).length,1);assert.equal(e.get("cancel").textContent,"Cancel Muse review");
  assert.equal(e.get("sandbox").hidden,false);
  f.info.jobs[0].muse={status:"proposed",message:"Not applied",receipt:{cost_usd:.002},proposal:{accepted:false,advice:{summary:"Possible support",hypotheses:[{concept:"contact",status:"hypothesis",claim:"<script>untrusted</script>",evidence_ids:["source:frame:1"],alternatives:["Camera projection"],required_tests:["Measure the support"]}],unanswered_questions:["Scale?"]}}};
  await e.get("refresh").onclick();await tick();
  assert.equal(e.get("review-muse").hidden,true);assert.equal(e.get("muse-summary").textContent,"Possible support");
  assert.equal(e.get("muse-hypotheses").children[0].children[1].textContent,"<script>untrusted</script>");
  assert.equal(e.get("muse-proposal-download").hidden,false);assert.equal(e.get("cancel").hidden,true);
  assert.equal(e.get("run-checks").hidden,false);
  await e.get("run-checks").onclick();await tick();
  assert.equal(f.calls.filter(c=>c.path.endsWith("/muse/checks")).length,1);
  assert.equal(e.get("cancel").textContent,"Cancel local checks");assert.equal(e.get("sandbox").hidden,false);
  f.info.jobs[0].muse.checks={status:"complete",message:"Physical hypotheses remain unresolved",report:{trials:[{condition:"root_unforced",steps:6,horizon_steps:50,success:false,reward_sum:-1.25,trace:"root_unforced.jsonl"}]}};
  await e.get("refresh").onclick();await tick();
  assert.match(e.get("checks-results").children[0].textContent,/without root assistance.*6\/50.*success: no/);
  assert.equal(e.get("checks-report-download").hidden,false);assert.equal(e.get("checks-traces").children.length,1);
  console.log("PASS Muse request, separate budget, retained sandbox and untrusted proposal display");
}

{
  const f=setup();await tick();
  assert.equal(f.calls.length,1);assert.equal(f.calls[0].options.headers.Authorization,"Bearer test-owner-token");
  assert.equal(f.timers.size,1);assert.match(f.elements.get("budget").textContent,/5.00/);
  f.document.hidden=true;f.listeners.get("visibilitychange")();assert.equal(f.timers.size,0);
  const e=f.elements;e.get("video").files=[{name:"clip.webm",size:500}];e.get("rights").checked=true;e.get("use-ai").checked=true;e.get("mode").value="auto";
  await e.get("upload-form").onsubmit({preventDefault(){}});await tick();
  const call=f.calls.find(c=>c.path.startsWith("/api/projects?"));assert.ok(call);assert.equal(call.options.method,"POST");
  assert.ok(!JSON.stringify(call).includes("openai-api-key"));
  console.log("PASS authenticated upload, budget display, no key exposure and hidden-tab polling pause");
}
{
  const id="c".repeat(32), f=setup([{id,name:"<script>unsafe</script>.webm",status:"failed",stage:"failed",error:"Tracking failed",events:[{message:"Uncertain camera"}],plan:{summary:"Proposal only",warnings:["No contact proof"]}}]);await tick();
  assert.equal(f.elements.get("project-heading").textContent,"<script>unsafe</script>.webm");
  assert.equal(f.elements.get("sandbox").hidden,true);assert.equal(f.elements.get("project-error").textContent,"Tracking failed");
  f.info.jobs[0]={...f.info.jobs[0],status:"ready",stage:"ready",error:null,sandbox:`/labs/${id}/`};
  await f.elements.get("refresh").onclick();await tick();
  assert.equal(f.elements.get("sandbox").href,`/labs/${id}/#token=test-owner-token`);
  console.log("PASS failed runs stay inspectable and only ready projects open a sandbox");
}
{
  const f=setup();await tick(); const e=f.elements;
  e.get("video").files=[{name:"clip.mp4",size:100}];e.get("rights").checked=true;e.get("use-ai").checked=false;e.get("mode").value="auto";
  await e.get("upload-form").onsubmit({preventDefault(){}});
  assert.ok(!f.calls.some(c=>c.options.method==="POST"));assert.match(e.get("error").textContent,/Choose exercise or objects/);
  console.log("PASS local mode needs an explicit builder, not a silent AI fallback");
}
{
  const id="7".repeat(32), f=setup([{id,name:"exercise.webm",route:"exercise",status:"ready",events:[],sandbox:`/labs/${id}/`,
    muse:{status:"proposed",repair_availability:{available:true,reason:"Up to three local candidates; no API charge"}}}]);await tick();
  const e=f.elements;assert.equal(e.get("repair-comparison").hidden,false);assert.equal(e.get("run-repairs").hidden,false);
  await Promise.all([e.get("run-repairs").onclick(),e.get("run-repairs").onclick()]);await tick();
  const calls=f.calls.filter(call=>call.path.endsWith("/muse/repairs"));assert.equal(calls.length,1);
  assert.equal(calls[0].options.method,"POST");assert.equal(calls[0].options.body,undefined);
  assert.equal(calls[0].options.headers.Authorization,"Bearer test-owner-token");
  assert.equal(e.get("cancel").hidden,false);assert.equal(e.get("cancel").textContent,"Cancel suggestion tests");
  assert.equal(e.get("sandbox").hidden,false);assert.equal([...f.delays.values()].at(-1),1500);
  await e.get("run-repairs").onclick();assert.equal(f.calls.filter(call=>call.path.endsWith("/muse/repairs")).length,1);
  f.info.jobs[0].muse.repairs={status:"complete",message:"All candidates rejected",applied:false,report:{accepted_in_simulation:false,selected_candidate:null,
    candidates:[{id:"feedback-1",candidate_score:.9148,baseline_score:.9166,all_original_gates_passed:true,accepted_in_simulation:false,reason:"<script>untrusted reason</script>"}]}};
  await e.get("refresh").onclick();await tick();
  assert.match(e.get("repairs-outcome").textContent,/No candidate improved enough.*unchanged/);
  assert.match(e.get("repairs-results").children[0].textContent,/0.9148 vs original 0.9166.*gates: passed.*rejected/);
  assert.ok(e.get("repairs-results").children[0].textContent.includes("<script>untrusted reason</script>"));
  assert.equal(e.get("run-repairs").hidden,true);assert.equal(e.get("cancel").hidden,true);
  for(const [name,file] of [["report","report.json"],["plan","plan.json"],["runtime","runtime-source.json"],["manifest","manifest.json"]]) {
    const link=e.get(`repairs-${name}-download`);assert.equal(link.hidden,false);assert.equal(link.href,`/api/projects/${id}/files/muse/repairs/run/${file}?token=test-owner-token`);
  }
  assert.ok(!f.calls.some(call=>call.path.includes("/files/muse/repairs/")));
  f.info.jobs[0].muse.repairs.report.accepted_in_simulation=true;
  f.info.jobs[0].muse.repairs.report.selected_candidate="feedback-2";
  await e.get("refresh").onclick();await tick();
  assert.match(e.get("repairs-outcome").textContent,/improved the fixed simulator score.*Nothing was applied/);
  f.document.hidden=true;f.listeners.get("visibilitychange")();assert.equal(f.timers.size,0);
  console.log("PASS one authenticated repair request, active polling/cancellation, honest candidate scores, safe text and lazy artifact links");
}
{
  const id="6".repeat(32), f=setup([{id,name:"support.webm",route:"exercise",status:"ready",events:[],muse:{status:"proposed",
    repair_availability:{available:false,reason:"Physical measurements cannot be replaced by simulation."}}}]);await tick();
  const e=f.elements;assert.equal(e.get("run-repairs").hidden,true);
  assert.match(e.get("repairs-status").textContent,/Physical measurements/);
  await e.get("run-repairs").onclick();assert.ok(!f.calls.some(call=>call.path.endsWith("/muse/repairs")));
  console.log("PASS unsupported Muse hypotheses remain explicit and cannot launch candidate tests");
}
{
  const id="5".repeat(32), f=setup([{id,name:"equipment.webm",route:"exercise",status:"preview",events:[],
    scene_inventory:{message:"Body evidence only; equipment remains unresolved.",next_steps:[]},
    body_preview:{image:"build/body_source/tracks.gif",body:"build/body_source/human.json",model:"build/body_source/human.xml",evidence:"build/body_source/evidence.json",clock:"build/body_source/native-clock.json",manifest:"build/body_source/manifest.json"}}]);await tick();
  const e=f.elements;
  assert.equal(e.get("body-preview").hidden,false);assert.equal(e.get("body-preview-image").hidden,false);
  assert.match(e.get("body-preview-image").src,/tracks.gif\?token=test-owner-token$/);
  assert.match(e.get("body-preview-model").href,/human.xml\?token=test-owner-token$/);
  assert.match(e.get("project-status").textContent,/no physics/);
  assert.equal(e.get("sandbox").hidden,true);assert.equal(e.get("muse-review").hidden,true);assert.equal(e.get("cancel").hidden,true);
  assert.equal(e.get("inventory-summary").textContent,"Body evidence only; equipment remains unresolved.");
  assert.ok(!f.calls.some(c=>c.path.includes("/files/")));
  f.info.jobs[0]={id,name:"mixed.webm",status:"needs_input",route:"exercise",events:[],scene_inventory:{message:"Explicit equipment inputs needed",next_steps:["<script>unsafe</script>"]}};
  await e.get("refresh").onclick();await tick();
  assert.equal(e.get("body-preview").hidden,true);assert.equal(e.get("cancel").hidden,true);
  assert.equal(e.get("inventory-next-steps").children[0].textContent,"<script>unsafe</script>");
  console.log("PASS real pose-preview artifact links, no physics/sandbox claims and terminal mixed-scene input guidance");
}
{
  const f=setup();await tick();const e=f.elements;
  e.get("video").files=[{name:"clip.webm",size:500}];e.get("rights").checked=true;e.get("use-ai").checked=false;e.get("mode").value="exercise";
  e.get("exercise-inventory").value="equipment_or_obstacles";e.get("exercise-output").value="body_preview";
  await e.get("upload-form").onsubmit({preventDefault(){}});await tick();
  const call=f.calls.find(c=>c.path.startsWith("/api/projects?")),query=new URLSearchParams(call.path.split("?")[1]);
  assert.equal(query.get("exercise_inventory"),"equipment_or_obstacles");assert.equal(query.get("exercise_output"),"body_preview");
  e.get("mode").value="objects";e.get("mode").onchange();assert.equal(e.get("exercise-options").hidden,true);
  console.log("PASS uploader scene declaration and output choice survive the authenticated request");
}
{
  const id="4".repeat(32), f=setup([{id,name:"imported.webm",kind:"imported_lab",route:"exercise",status:"ready",events:[],sandbox:`/labs/${id}/`,
    inventory:{method:"saved_world"},review_unavailable_reason:"Muse review of imported labs needs a verified body-evidence attachment"}]);await tick();
  const e=f.elements;assert.equal(e.get("sandbox").hidden,false);assert.equal(e.get("muse-review").hidden,true);assert.equal(e.get("support-editor").hidden,true);
  assert.match(e.get("inventory-summary").textContent,/Imported source-bound simulation candidate/);
  assert.match(e.get("inventory-summary").textContent,/verified body-evidence attachment/);
  assert.ok(!e.get("inventory-summary").textContent.includes("Legacy person-only"));
  f.info.jobs[0]={id,name:"old.webm",route:"exercise",status:"ready",events:[],sandbox:`/labs/${id}/`};
  await e.get("refresh").onclick();await tick();
  assert.match(e.get("inventory-summary").textContent,/did not record an explicit equipment declaration/);
  assert.equal(e.get("sandbox").hidden,false);
  console.log("PASS imported coupled worlds stay distinct from legacy undeclared person-only labs");
}
{
  const id="5".repeat(32), f=setup([{id,name:"equipment.webm",kind:"imported_lab",route:"exercise",status:"ready",events:[],review_attachment:{available:true,prepared:true}}]);await tick();
  const e=f.elements;assert.equal(e.get("muse-review").hidden,false);
  assert.match(e.get("muse-scope").textContent,/current equipment world's assumptions/);
  assert.match(e.get("attachment-request-download").href,/review-evidence\/prepared\/request.json\?token=test-owner-token$/);
  assert.equal(e.get("attachment-receipt-download").hidden,false);
  f.info.jobs[0].muse={status:"proposed",repair_availability:{available:true}};
  await e.get("refresh").onclick();await tick();
  assert.equal(e.get("run-checks").hidden,true);assert.equal(e.get("repair-comparison").hidden,true);
  assert.ok(!f.calls.some(c=>c.path.includes("prepared/request")));
  console.log("PASS imported evidence enables review and inspectable preparation without enabling repairs or eager downloads");
}
{
  const id="6".repeat(32),f=setup([{id,name:"mixed.webm",status:"needs_input",route:"exercise",events:[],equipment_availability:{available:true,matches:[{id:"a".repeat(16),label:"Verified prepared source"}],reason:"Exact video match; not yet admitted"}}]);await tick();
  const e=f.elements;assert.equal(e.get("equipment-candidates").hidden,false);assert.equal(e.get("sandbox").hidden,true);
  const build=e.get("equipment-source-options").children[0];await Promise.all([build.onclick(),build.onclick()]);await tick();
  const requests=f.calls.filter(c=>c.path.endsWith("/equipment/prepare"));assert.equal(requests.length,1);assert.equal(requests[0].options.headers.Authorization,"Bearer test-owner-token");assert.deepEqual(JSON.parse(requests[0].options.body),{id:"a".repeat(16)});
  assert.equal(e.get("cancel").textContent,"Cancel equipment candidates");assert.equal([...f.delays.values()].at(-1),1500);
  f.info.jobs[0].equipment_candidates={status:"complete",message:"Choose one candidate",report:{source_validation:{body_reference_count:121,source_slots:257,raw_object_ids:["sam-1","sam-2","sam-3"],unresolved_reasons:["identity ambiguity"]},candidates:[{id:"alternative-1",status:"ready",selectable:true,assignment:{body_correspondence:"sam-1",rigid_proxies:["sam-2","sam-3"]},reasons:[]},{id:"alternative-2",status:"rejected_initial_geometry",selectable:false,assignment:{body_correspondence:"sam-2",rigid_proxies:["sam-1","sam-3"]},reasons:["<script>floor failure</script>"]}]}};
  await e.get("refresh").onclick();await tick();
  assert.equal(e.get("equipment-alternatives").children.length,2);assert.equal(e.get("equipment-alternatives").children[1].children[3].hidden,true);
  assert.equal(e.get("equipment-alternatives").children[1].children[2].textContent,"Constructor result: <script>floor failure</script>");
  assert.match(e.get("equipment-coverage").textContent,/121.*257.*3 raw object tracks/);
  await e.get("equipment-alternatives").children[0].children[3].onclick();await tick();
  assert.equal(e.get("sandbox").hidden,false);assert.match(e.get("equipment-selection-note").textContent,/locked to alternative-1/);
  assert.equal(e.get("equipment-alternatives").children[0].children[3].hidden,true);
  assert.ok(!f.calls.some(c=>c.path.includes("/files/equipment/")));
  console.log("PASS prepared equipment matching, authenticated deduplicated queue, retained failures and explicit frozen candidate selection");
}

{
  const id="9".repeat(32), f=setup([{id,name:"equipment.mp4",ai:true,status:"needs_input",events:[],perception:{available:true,status:"not_started",message:"Local checks first"}}]);await tick();
  const e=f.elements;
  assert.equal(e.get("perception-preflight").hidden,false);assert.equal(e.get("perception-consent").hidden,true);
  await Promise.all([e.get("perception-preflight").onclick(),e.get("perception-preflight").onclick()]);await tick();
  assert.equal(f.calls.filter(c=>c.path.endsWith("/perception/preflight")).length,1);
  assert.equal(e.get("cancel").textContent,"Cancel video preparation");assert.equal([...f.delays.values()].at(-1),1500);
  const p=f.info.jobs[0].perception;
  Object.assign(p,{status:"awaiting_consent",quote_sha256:"b".repeat(64),quote:{frame_count:68,openai_model:"gpt-4.1-mini",meta_model:"sam-3.1",openai_max_usd:.3135,meta_max_usd:.15,message:"Separate budgets"},artifacts:{quote:"perception/quote.json",body_evidence:"perception/preflight/body_source/evidence.json"}});
  await e.get("refresh").onclick();await tick();
  assert.equal(e.get("perception-openai").checked,false);assert.equal(e.get("perception-meta").checked,false);
  assert.match(e.get("perception-quote").textContent,/68 source frames.*0.3135.*0.1500/);
  assert.equal(e.get("perception-prepare").disabled,true);
  e.get("perception-openai").checked=true;e.get("perception-openai").onchange();
  assert.equal(e.get("perception-prepare").disabled,true);
  e.get("perception-meta").checked=true;e.get("perception-meta").onchange();
  assert.equal(e.get("perception-prepare").disabled,false);
  p.busy=true;await e.get("refresh").onclick();await tick();assert.equal(e.get("perception-prepare").disabled,true);
  p.busy=false;await e.get("refresh").onclick();await tick();
  await Promise.all([e.get("perception-prepare").onclick(),e.get("perception-prepare").onclick()]);await tick();
  const paid=f.calls.filter(c=>c.path.endsWith("/perception/prepare"));assert.equal(paid.length,1);
  assert.equal(paid[0].options.headers.Authorization,"Bearer test-owner-token");
  assert.deepEqual(JSON.parse(paid[0].options.body),{quote_sha256:"b".repeat(64),consent:{openai_sampled_images:true,meta_full_video_and_regions:true},openai_max_usd:.3135,meta_max_usd:.15,request_id:"a".repeat(32)});
  Object.assign(f.info.jobs[0].perception,{status:"failed",error:"<script>provider uncertainty</script>"});
  await e.get("refresh").onclick();await tick();
  assert.equal(e.get("perception-status").textContent,"<script>provider uncertainty</script>");
  assert.equal(e.get("perception-preflight").hidden,true);assert.equal(e.get("perception-consent").hidden,true);
  assert.equal(f.calls.filter(c=>c.path.endsWith("/perception/prepare")).length,1);
  assert.ok(!f.calls.some(c=>c.path.includes("/files/perception/")));
  console.log("PASS local-only preflight, distinct media consents, quote caps, worker fence, one authenticated paid action and no retry");
}
{
  const id="7".repeat(32),f=setup([{id,name:"new-equipment.mp4",kind:"video",route:"exercise",status:"ready",events:[],equipment_candidate_id:"alternative-1",review_attachment:{available:true,status:"pending_action_time_validation",repair_qualified:false},muse:{status:"proposed",repair_availability:{available:false}}}]);await tick();
  const e=f.elements;
  assert.equal(e.get("muse-review").hidden,false);
  assert.match(e.get("muse-scope").textContent,/rechecks the selected world.*identities.*uncertain/);
  assert.equal(e.get("run-checks").hidden,true);assert.equal(e.get("checks-help").hidden,true);
  assert.equal(e.get("repair-comparison").hidden,true);assert.equal(e.get("attachment-receipt-download").hidden,true);
  assert.ok(!f.calls.some(c=>c.path.includes("/files/")));
  console.log("PASS selected perception worlds enable source-revalidated Muse review without unsupported checks, repairs or eager evidence reads");
}
{
  const id="6".repeat(32), diagnostic={sampled_frames:153,usable_samples:143,identity_breaks:2,window_count:3,
    listed_barriers:[{frame:111,source_time_s:3.699838,decoder_timestamp_s:3.7,hypothesis_count:2},{frame:123,source_time_s:4.099821,decoder_timestamp_s:4.1,hypothesis_count:2}],
    listed_windows:[{frame_interval:[0,108],usable_samples:35,sampled_frames:37,screen_passed:true},{frame_interval:[114,120],usable_samples:2,sampled_frames:3,screen_passed:false}],
    message:"<script>Both estimates stay unknown</script>",scope:"Diagnostic only; no body completed."};
  const f=setup([{id,name:"competing.mp4",status:"needs_input",stage:"needs_input",events:[],perception:{available:true,status:"failed",error:"Original competing-pose refusal",pose_diagnostic:diagnostic,
    artifacts:{pose_diagnostic:"perception/pose-diagnostic/report.json",raw_poses:"perception/preflight/body_source/pose-landmarks.json"}}}]);await tick();
  const e=f.elements;assert.equal(e.get("perception-pose-diagnostic").hidden,false);
  assert.match(e.get("perception-pose-counts").textContent,/143 of 153.*2 identity breaks.*could not be completed/);
  assert.equal(e.get("perception-pose-explanation").textContent,diagnostic.message);
  assert.equal(e.get("perception-status").textContent,"Original competing-pose refusal");
  assert.match(e.get("perception-pose-barriers").children[0].textContent,/Source frame 111.*3.700 s by frame rate.*decoder-reported time 3.700.*Neither/);
  assert.match(e.get("perception-pose-window-note").textContent,/not playable or selectable episodes/);
  assert.match(e.get("perception-pose-windows").children[0].textContent,/other checks remain untested/);
  assert.match(e.get("perception-pose-windows").children[1].textContent,/checks fail/);
  assert.equal(e.get("sandbox").hidden,true);
  assert.ok(e.get("perception-artifacts").children.some(a=>a.href.includes("pose-diagnostic/report.json")));
  assert.ok(!f.calls.some(c=>c.path.includes("pose-diagnostic/") || c.path.includes("pose-landmarks.json")));
  f.info.jobs[0].perception={status:"awaiting_consent",available:true};
  await e.get("refresh").onclick();await tick();
  assert.equal(e.get("perception-pose-diagnostic").hidden,true);assert.equal(e.get("perception-pose-barriers").children.length,0);
  console.log("PASS failed perception explains identity barriers with safe text, unchanged failure, no playable windows or eager pose fetch");
}

{
  const id="d".repeat(32),familyPin="a".repeat(64),ranges=Array.from({length:40},(_,i)=>[1+i*.01,1+i*.01+.005]);
  const family={family_sha256:familyPin,settings:{camera:{distance_model_units:2},depth_domain_model_units:[.1,4],mass_kg:1,friction:[.5,.005,.0001]},identity_family:{assignment_count:7},factors:[{object_id:"raw-0",status:"feasible_intervals",certified_intervals:ranges,possible_intervals:ranges,boundary_ambiguity:true,reasons:[]},{object_id:"raw-1",status:"no_solution",certified_intervals:[],possible_intervals:[],reasons:["No safe depth <script>"],boundary_ambiguity:false}]};
  const f=setup([{id,name:"Saved moving clip.mp4",kind:"video",route:"exercise",status:"needs_input",events:[],perception:{status:"evidence_ready",available:false},moving_equipment:{status:"awaiting_choices",phase:"prepare",message:"No new AI request",family}}]);await tick();
  const e=f.elements,cards=e.get("moving-factors").children;
  assert.equal(e.get("moving-choice-editor").hidden,false);assert.equal(e.get("moving-prepare-form").hidden,true);
  assert.match(e.get("moving-family").textContent,/all 7 possible nonempty assignments.*remain saved/);
  assert.equal(cards.length,2);assert.equal(cards[1].children[5].children[0].disabled,true);
  assert.match(cards[1].children[1].textContent,/No safe depth <script>/);
  assert.match(cards[0].children[2].textContent,/#32/);assert.doesNotMatch(cards[0].children[2].textContent,/#33/);
  cards[0].children[4].onclick();assert.match(cards[0].children[2].textContent,/#33/);
  assert.match(e.get("moving-family").textContent,/Your assumptions: camera distance 2.*mass 1 kg.*zero velocity/);
  assert.match(cards[0].children[1].textContent,/initial model clearance only, not verified physical depth/);
  assert.equal(e.get("moving-build").disabled,true);assert.equal(f.calls.length,1);
  const include=cards[0].children[5].children[0],index=cards[0].children[6].children[0],depth=cards[0].children[7].children[0];
  include.checked=true;index.value="1";depth.value="1.002";e.get("moving-add-choice").onclick();
  assert.equal(e.get("moving-choices").children.length,1);assert.equal(e.get("moving-build").disabled,false);
  e.get("moving-add-choice").onclick();assert.match(e.get("error").textContent,/already listed/);
  depth.value="1.5";e.get("moving-add-choice").onclick();assert.match(e.get("error").textContent,/inside its specified/);
  await e.get("moving-build").onclick();await tick();
  const built=f.calls.find(c=>c.path.endsWith("/moving-equipment/build"));assert.ok(built);assert.equal(built.options.headers.Authorization,"Bearer test-owner-token");
  assert.deepEqual(JSON.parse(built.options.body),{schema_version:1,profile:"moving-depth-selection-v1",family_sha256:familyPin,choices:[{equipment:[{object_id:"raw-0",interval_index:0,depth_model_units:1.002}]}]});
  assert.equal(e.get("cancel").hidden,false);assert.equal(e.get("sandbox").hidden,true);
  f.info.jobs[0].moving_equipment={status:"complete",family,report:{candidates:[{id:"candidate-1",status:"ready",selectable:true,reasons:[]},{id:"candidate-2",status:"rejected",selectable:false,reasons:["Pairwise overlap"]}]}};
  await e.get("refresh").onclick();await tick();
  const choices=e.get("moving-candidates").children;assert.equal(choices[1].children[1].hidden,true);assert.equal(choices[0].children[1].hidden,false);
  await choices[0].children[1].onclick();await tick();assert.equal(e.get("sandbox").hidden,false);assert.equal(e.get("muse-review").hidden,true);
  assert.match(e.get("moving-status").textContent,/locked.*earlier failures remain saved/);
  assert.match(e.get("inventory-summary").textContent,/Selected moving simulation.*Manual, unscored/);
  assert.doesNotMatch(e.get("inventory-summary").textContent,/Legacy person-only/);
  assert.ok(!f.calls.some(c=>c.path.endsWith("/perception/prepare")||c.path.endsWith("/muse")));
  console.log("PASS complete factorized moving family, paged intervals, explicit authored choices, retained refusals and manual-only selected sandbox");
}
{
  const id="e".repeat(32),f=setup([{id,name:"Saved source.mp4",kind:"video",route:"exercise",status:"needs_input",events:[],moving_equipment:{available:true,message:"No new AI request"}}]);await tick();const e=f.elements;
  await e.get("moving-prepare-form").onsubmit({preventDefault(){}});assert.match(e.get("error").textContent,/Confirm the authored/);
  e.get("moving-confirm").checked=true;
  for(const [id,value] of [["moving-camera","2"],["moving-depth-low","0.1"],["moving-depth-high","4"],["moving-mass","1"],["moving-friction","0.5"]])e.get(id).value=value;
  await e.get("moving-prepare-form").onsubmit({preventDefault(){}});await tick();
  const call=f.calls.find(c=>c.path.endsWith("/moving-equipment/prepare"));assert.ok(call);assert.deepEqual(JSON.parse(call.options.body).depth_domain_model_units,[.1,4]);
  assert.equal(e.get("cancel").hidden,false);assert.equal(e.get("sandbox").hidden,true);
  console.log("PASS saved moving preparation requires explicit authored priors and queues no provider request");
}
