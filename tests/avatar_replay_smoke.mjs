/** Offline replay lifecycle and exact-state checks with fake DOM/workers, not browser/GPU evidence. */
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {createRequire} from "node:module";
import {performance} from "node:perf_hooks";
const require=createRequire(import.meta.url);
const replay=require("../src/weft/assets/avatar_replay.js"),math=require("../src/weft/assets/avatar_retarget.js");
const code=fs.readFileSync(new URL("../src/weft/assets/avatar_replay.js",import.meta.url),"utf8");
const mathCode=fs.readFileSync(new URL("../src/weft/assets/avatar_retarget.js",import.meta.url),"utf8");
const A="a".repeat(64),B="b".repeat(64),MISSING="f".repeat(64);

function dataFixture(){
  const object={id:"weight",role:"passive",shape:"box",size:[.2,.3,.4],position:[1.,1.,.3],quaternion:[1,0,0,0],color:"#965644"};
  const scene={profile:"shared-world-v2",bounds:[3,2,2.5],dt:1/30,coordinate_frame:{scale_bridge:{body_height_to_world_scale:1,origin_world:[0,0,0]}},
    human_spec:{lengths:{torso:.5,shoulder_width:.4,hip_width:.3,upper_arm:.3,forearm:.3,thigh:.4,shin:.4},mobility:{base_height:1,ground_plane:{normal:[0,0,1]}}},objects:[object]};
  const events=["start","step","step","reset","step","step"].map((type,index)=>({type,index,seed:9,action:type==="step"?[.25,0,-.1]:undefined,
    controller:type==="step"?"manual":undefined,snapshot:{time_s:index<3?index/30:(index-3)/30,objects:[{...object,position:[1+index*.01,1,.3]}],
      human:{segments:[{id:"torso",a:[1,1,.4+index*.01],b:[1,1,1.1+index*.01]}],muscles:[]}},preserved:{original_index:index,extra:"unchanged"}}));
  const changes=[[0,null],[1,A],[1,B],[3,null],[4,MISSING]].map(([after_event_index,asset],index)=>({index,after_event_index,after_event_sha256:"c".repeat(64),
    settings:{asset,overlay:index%2===0,show_muscles:true,camera:{yaw:.1*index,zoom:1+.1*index,follow:index%2===1}}}));
  const asset=name=>({display_name:name,render_sha256:"d".repeat(64),render_base64:Buffer.from('{"profile":"synthetic-viewer-fixture"}').toString("base64"),textures_base64:[]});
  return JSON.parse(JSON.stringify({profile:"weft-offline-avatar-replay-v1",scene:{profile:"shared-exercise-recording-v1",engine_scene:scene},events,changes,
    assets:{[A]:asset("Panda"),[B]:asset("Robot")},worker_base64:Buffer.from('globalThis.avatarWorkerStarted=true;if(!globalThis.WeftAvatarRetarget)importScripts("/avatar_retarget.js");').toString("base64")}));
}

function mountFixture({fallback=false,data=dataFixture()}={}){
  const elements=new Map(),blobs=new Map(),workers=[],viewers=[],timers=new Map(),listeners={},draws=[];let next=0,peakQueue=0,peakTimers=0;
  const context=new Proxy({},{get:(_,name)=>(...args)=>draws.push({name,args}),set:()=>true});
  const element=id=>{
    if(!elements.has(id))elements.set(id,{id,tagName:"DIV",value:id==="zoom"?"1":id==="yaw"?".5":"0",checked:false,disabled:true,hidden:false,open:false,textContent:"",width:1080,height:420,listeners:{},attributes:{},
      addEventListener(type,fn){this.listeners[type]=fn;},setAttribute(key,value){this.attributes[key]=value;},getContext:()=>context});
    return elements.get(id);
  };
  element("avatar-replay-data").textContent=JSON.stringify(data);element("avatar-replay-math").textContent=mathCode;element("avatar-replay-controller").textContent=code;
  const document={hidden:false,getElementById:element,addEventListener(type,fn){listeners[type]=fn;}};
  class FakeBlob{constructor(parts){this.source=parts.join("");}}
  class FakeWorker{
    constructor(url){this.queue=[];this.terminated=false;this.scope=vm.createContext({postMessage:message=>this.onmessage({data:message}),atob:value=>Buffer.from(value,"base64").toString("binary"),TextDecoder,Uint8Array,ArrayBuffer,Float32Array,Map,Set,console});vm.runInContext(blobs.get(url).source,this.scope);workers.push(this);}
    postMessage(message){assert(!this.terminated);this.queue.push(message);peakQueue=Math.max(peakQueue,this.queue.length);}
    flush(){assert(this.queue.length);this.scope.onmessage({data:this.queue.shift()});}
    terminate(){this.terminated=true;this.queue=[];}
  }
  const api={create(canvas,{workerURL,onStatus}){
    const viewer={canvas,workerURL,updates:[],loads:[],visibility:[],disposed:false,configure(scene){this.scene=scene;},
      load(bytes){this.loads.push(bytes);onStatus({status:"loading",reason:"Loading exact asset"});const result=fallback?{status:"fallback",reason:"WebGL2 unavailable"}:{status:"ready"};onStatus(result);return Promise.resolve(result);},
      update(frame){this.updates.push(frame);},setVisible(value){this.visibility.push(value);},dispose(){this.disposed=true;}};
    viewers.push(viewer);if(fallback)onStatus({status:"fallback",reason:"WebGL2 unavailable"});return viewer;
  }};
  const scope={document,WeftAvatarViewer:api,Blob:FakeBlob,Worker:FakeWorker,URL:{createObjectURL(blob){const key=`blob:test-${++next}`;blobs.set(key,blob);return key;},revokeObjectURL(key){blobs.delete(key);}},
    addEventListener(type,fn){listeners[type]=fn;},setTimeout(fn,delay){const id=++next;timers.set(id,{fn,delay});peakTimers=Math.max(peakTimers,timers.size);return id;},clearTimeout(id){timers.delete(id);},console};
  vm.runInNewContext(code,scope);assert.equal(workers.length,1);workers[0].flush();workers[0].flush();
  const click=id=>element(id).listeners.click();
  const select=index=>{element("scrub").value=String(index);element("scrub").listeners.input();};
  return {element,document,workers,viewers,timers,blobs,listeners,draws,data,click,select,stats:()=>({peakQueue,peakTimers}),
    async settle(){while(workers[0].queue.length)workers[0].flush();await Promise.resolve();await Promise.resolve();},
    tick(){assert.equal(timers.size,1);const [id,timer]=timers.entries().next().value;timers.delete(id);timer.fn();},
    event(){element("event-details").open=true;element("event-details").listeners.toggle();return JSON.parse(element("event-json").textContent);}};
}

async function everyPhysicalEventAndLastAnchoredAppearanceAreRetained(){
  const ui=mountFixture(),before=JSON.stringify(ui.data);
  assert.match(ui.element("appearance-status").textContent,/no avatar selected/);
  for(let i=0;i<ui.data.events.length;i++){
    if(i){ui.select(i);await ui.settle();}
    assert.deepEqual(ui.event().event,ui.data.events[i]);
    assert.deepEqual(ui.event().appearance_change,replay.settingsAt(ui.data.changes,i));
    assert.match(ui.element("event-label").textContent,new RegExp(`Physical event ${i} / 5`));
  }
  assert.equal(ui.viewers.length,1);assert.equal(ui.viewers[0].loads.length,1,"Same-frame selection uses Robot, never the earlier Panda");
  assert.match(ui.element("appearance-status").textContent,/Missing recorded avatar.*unavailable.*original recorded body/);
  assert.equal(JSON.stringify(ui.data),before);assert.equal(ui.timers.size,0);
}

async function seeksCoalesceAndOldResponsesCannotOverwriteTheLatestEvent(){
  const ui=mountFixture();for(const i of [1,2,4,5])ui.select(i);
  assert.equal(ui.workers[0].queue.length,1);ui.workers[0].flush();assert.equal(ui.workers[0].queue.length,1);
  assert.equal(ui.event().event.index,0,"Stale event response is not displayed");
  await ui.settle();assert.equal(ui.event().event.index,5);assert.equal(ui.stats().peakQueue,1);
  assert.equal(ui.viewers.length,0,"Superseded avatar bytes do not create a viewer");
}

async function playbackPausesWhenHiddenAndDisposesItsBoundedResources(){
  const ui=mountFixture();ui.click("play");assert.equal(ui.timers.size,1);ui.tick();await ui.settle();
  assert.equal(ui.event().event.index,1);assert.equal(ui.timers.size,1);ui.tick();
  ui.document.hidden=true;ui.listeners.visibilitychange();await ui.settle();
  assert.equal(ui.event().event.index,1);assert.equal(ui.timers.size,0);assert.equal(ui.element("play").textContent,"Play");
  ui.document.hidden=false;ui.listeners.visibilitychange();await ui.settle();
  assert.equal(ui.event().event.index,2);assert.equal(ui.timers.size,0,"Visibility does not silently restart playback");
  ui.listeners.pagehide({persisted:false});assert(ui.workers[0].terminated);assert(ui.viewers[0].disposed);
  assert.equal(ui.blobs.size,0);assert.equal(ui.timers.size,0);assert.equal(ui.stats().peakTimers,1);
}

async function fallbackAndInspectionCameraAreExplicitWithoutChangingTheEvent(){
  const ui=mountFixture({fallback:true});ui.select(1);await ui.settle();
  assert.match(ui.element("appearance-status").textContent,/WebGL2 unavailable.*showing original recorded body/);
  assert(ui.element("avatar-canvas").hidden&&!ui.element("fallback-canvas").hidden);
  const original=ui.event();ui.element("inspect-camera").checked=true;ui.element("inspect-camera").listeners.change();
  ui.element("yaw").value="1.2";ui.element("yaw").listeners.input();await ui.settle();
  assert.match(ui.element("camera-label").textContent,/Inspection camera override/);assert.deepEqual(ui.event(),original);
  assert.equal(ui.viewers[0].updates.at(-1).camera.yaw,1.2);
  ui.click("recorded-camera");await ui.settle();assert.match(ui.element("camera-label").textContent,/Recorded camera/);
  assert.equal(ui.viewers[0].updates.at(-1).camera.yaw,original.appearance_change.settings.camera.yaw);
  ui.click("next-reset");await ui.settle();assert.equal(ui.event().event.index,3);assert.match(ui.element("event-label").textContent,/Seeded reset/);
}

async function blobWorkersHavePinnedMathAndCannotImportNetworkCode(){
  const ui=mountFixture();assert.equal(ui.blobs.size,2);
  for(const blob of ui.blobs.values()){
    const scope=vm.createContext({postMessage(){},TextDecoder,Uint8Array,ArrayBuffer,atob:value=>Buffer.from(value,"base64").toString("binary")});
    vm.runInContext(blob.source,scope);assert(scope.WeftAvatarRetarget);assert.throws(()=>scope.importScripts("https://example.invalid/code.js"),/forbids imported scripts/);
  }
  ui.listeners.keydown({key:"End",target:{tagName:"DIV"},preventDefault(){}});await ui.settle();assert.equal(ui.event().event.index,5);
  ui.listeners.keydown({key:"Home",target:{tagName:"DIV"},preventDefault(){}});await ui.settle();assert.equal(ui.event().event.index,0);
}

async function fallbackUsesRecordedEndpointsAndSharedCameraForAllThreeViews(){
  const data=dataFixture(),scene=replay.compactScene(data.scene),config=math.compileScene(scene),outlines=replay.prepareOutlines(math,config),state=data.events[2].snapshot;
  const before=JSON.stringify({scene,state}),camera={yaw:.9,zoom:1.4,follow:true};
  const views=replay.fallbackFrame(math,config,outlines,state,camera,false),expected=math.cameraViews(config,state,camera,1080,420);
  assert.equal(views.length,3);
  for(let i=0;i<3;i++){
    const [x,,w,h]=expected[i].viewport;
    const points=[state.human.segments[0].a,state.human.segments[0].b].map(p=>{const q=math.point(expected[i].matrix,p);return [x+(q[0]+1)*w/2,(1-q[1])*h/2];});
    assert.deepEqual(views[i].paths.at(-1).points,points);
  }
  assert.equal(JSON.stringify({scene,state}),before);
  const event={...data.events[2],snapshot:{profile:"shared-exercise-state-v1",engine_state:state,engine_state_status:"last_valid_control_boundary",
    task_metrics:{simulated_duration_s:.075,completed_interval:false,success:false,failure:"recorded failure",failed_gates:["body_tracking"]}}};
  assert.match(replay.describe(event).outcome,/0\.075 s.*not passed.*last valid geometry/);
}

const checks=[everyPhysicalEventAndLastAnchoredAppearanceAreRetained,seeksCoalesceAndOldResponsesCannotOverwriteTheLatestEvent,
  playbackPausesWhenHiddenAndDisposesItsBoundedResources,fallbackAndInspectionCameraAreExplicitWithoutChangingTheEvent,
  blobWorkersHavePinnedMathAndCannotImportNetworkCode,fallbackUsesRecordedEndpointsAndSharedCameraForAllThreeViews];
let failed=0;const started=performance.now();
for(const check of checks){try{await check();console.log(`PASS ${check.name}`);}catch(error){failed++;console.error(`FAIL ${check.name}: ${error.stack}`);}}
console.log(JSON.stringify({checks:checks.length,failed,seconds:(performance.now()-started)/1000,scope:"Fake DOM/worker lifecycle and pure recorded-state geometry; no browser, GPU, physics, media export or source study"}));
if(failed)process.exitCode=1;
