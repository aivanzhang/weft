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
      if(path.endsWith("/supports")){const child={id:"e".repeat(32),name:info.jobs[0].name,kind:"support_experiment",parent_id:info.jobs[0].id,status:"queued",stage:"queued",events:[],source_available:false};info.jobs.unshift(child);return {ok:true,json:async()=>structuredClone(child)};}
      return {ok:true,json:async()=>({frames:[]})};},console};
  vm.runInNewContext(code,context);
  return {elements,calls,timers,delays,document,listeners,info};
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
