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
  const timers=new Map(), calls=[], listeners=new Map(); let next=1;
  const document={hidden:false,getElementById:id=>elements.get(id),createElement:tag=>new Element(tag),querySelectorAll:()=>[],addEventListener:(name,fn)=>listeners.set(name,fn)};
  const info={credential:{configured:true},budget:{configured:true,remaining_usd:5,total_usd:5,calls:0,unsettled_calls:0,reservation_usd:.3135},muse:{available:true,budget_configured:true,remaining_usd:4.7,total_budget_usd:5,unsettled_count:0,estimated_one_call_usd:.058368},jobs};
  const context={document,location:{hash:"#token=test-owner-token",pathname:"/"},history:{replaceState(){}},sessionStorage:{getItem(){return null;},setItem(){}},URLSearchParams,AbortSignal,crypto:{randomUUID:()=>"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"},
    setTimeout:fn=>{const id=next++;timers.set(id,fn);return id;},clearTimeout:id=>timers.delete(id),window:{addEventListener(){}},
    fetch:async(path,options={})=>{calls.push({path,options});if(path==="/api/portal")return {ok:true,json:async()=>structuredClone(info)};
      if(path.startsWith("/api/projects?"))return {ok:true,json:async()=>({id:"b".repeat(32),name:"clip.webm",status:"queued",stage:"queued",events:[]})};
      if(path.endsWith("/muse")){info.jobs[0].muse={status:"queued",message:"Waiting"};return {ok:true,json:async()=>structuredClone(info.jobs[0])};}
      return {ok:true,json:async()=>({frames:[]})};},console};
  vm.runInNewContext(code,context);
  return {elements,calls,timers,document,listeners,info};
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
