/** Analytic compound projections only: no browser, server, model or physics execution. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {performance} from 'node:perf_hooks';
import {fixture} from './lab_ui_smoke.mjs';

const html=fs.readFileSync(new URL('../src/weft/assets/shared-world-replay.html',import.meta.url),'utf8');
const replayScript=[...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
const color='#3571A2', q=Math.SQRT1_2;
const compound={id:'block',label:'Analytic compound',role:'passive',shape:'compound',position:[1.5,1,1],
  quaternion:[q,0,0,q],velocity:[0,0,0],angular_velocity:[0,0,0],mass:1,friction:.5,color,
  provenance:{origin:'authored_hypothesis'},geometry:{profile:'rigid-compound-v1',origin:'authored_hypothesis',
    assumptions:['Known analytic fixture, not reconstructed equipment'],inertia:{mode:'authored_centered_diagonal',diagonal:[.1,.1,.1]},
    parts:[{id:'body',shape:'box',position:[.2,0,0],quaternion:[q,q,0,0],size:[.4,.2,.6]},
      {id:'knob',shape:'sphere',position:[-.4,0,0],quaternion:[1,0,0,0],radius:.08},
      {id:'handle',shape:'capsule',position:[0,.3,.3],quaternion:[q,0,q,0],radius:.05,length:.6}]} };
const ball={id:'companion',label:'Primitive sphere',role:'passive',shape:'sphere',position:[2,.7,.5],
  quaternion:[1,0,0,0],velocity:[0,0,0],angular_velocity:[0,0,0],radius:.035,color:'#C7795D'};
const floor={id:'floor',label:'Primitive floor',role:'static',shape:'box',position:[1.5,1,.05],
  quaternion:[1,0,0,0],velocity:[0,0,0],angular_velocity:[0,0,0],size:[3,2,.1],color:'#D8C9BA'};
const objects=()=>structuredClone([compound,ball,floor]);
const near=(a,b,label)=>assert(Math.abs(a-b)<1e-9,`${label}: ${a} != ${b}`);

function paths(calls,operation='fill',style=color) {
  let path=[];
  const result=[];
  for(const call of calls) {
    if(call.name==='beginPath')path=[];
    else if(['moveTo','lineTo','arc','closePath'].includes(call.name))path.push({name:call.name,args:[...call.args]});
    else if(call.name===operation&&call.fillStyle===style)result.push(structuredClone(path));
  }
  return result;
}
function finiteCalls(calls) {
  for(const call of calls)if(['moveTo','lineTo','arc','rect'].includes(call.name))assert(call.args.every(Number.isFinite));
}
function project([x,y,z],pane) {
  const scale=100.8;
  if(pane===0)return [180+(x-1.5)*scale,226.8-(z-1.25)*scale];
  if(pane===1)return [540+(x-1.5)*scale,226.8-(y-1)*scale];
  return [900+((x-1.5)*Math.cos(.5)-(y-1)*Math.sin(.5))*scale,
    226.8+((x-1.5)*Math.sin(.5)**2+(y-1)*Math.cos(.5)*Math.sin(.5)-(z-1.25)*Math.cos(.5))*scale];
}
function assertProjection(calls,operation='fill') {
  finiteCalls(calls);
  const rendered=paths(calls,operation);
  assert.equal(rendered.length,24,'Each pane has six box faces, one sphere and one capsule; no enclosing box');
  for(let pane=0;pane<3;pane++) {
    const partPaths=rendered.slice(pane*8,pane*8+8);
    const faces=partPaths.filter(path=>!path.some(call=>call.name==='arc'));
    assert.equal(faces.length,6);
    // Rz(90) * Rx(90) permutes box axes; reversing composition swaps these extents.
    const expected=[];
    for(const x of [1.2,1.8])for(const y of [1,1.4])for(const z of [.9,1.1])expected.push(project([x,y,z],pane));
    for(const vertex of faces.flat().filter(call=>['moveTo','lineTo'].includes(call.name))) {
      assert(expected.some(point=>point.every((v,i)=>Math.abs(v-vertex.args[i])<1e-9)),'Box vertex must use both rotations');
    }
    const arcs=partPaths.flat().filter(call=>call.name==='arc');
    const sphere=arcs.find(call=>Math.abs(call.args[2]-.08*100.8)<1e-9);
    assert(sphere);project([1.5,.6,1],pane).forEach((v,i)=>near(sphere.args[i],v,'Sphere parent-rotated offset'));
    const ends=arcs.filter(call=>Math.abs(call.args[2]-.05*100.8)<1e-9);
    assert.equal(ends.length,pane===0?1:2,'End-on capsule must remain a circle');
    const endpoints=pane===0?[[1.2,1,1.3]]:[[1.2,.7,1.3],[1.2,1.3,1.3]];
    for(const point of endpoints)assert(ends.some(call=>project(point,pane).every((v,i)=>Math.abs(call.args[i]-v)<1e-9)),
      'Capsule endpoint uses parent × local rotation, not the enclosing box or local Z alone');
  }
}

function replay(objects) {
  const event={type:'step',snapshot:{objects,target:[2,1,.5],time_s:0,substeps:8,contact_count:0,contact_details:[],
    human:{segments:[],muscles:[],mobility:{floor_height:.1}},contact_summary:{human_object:0},coupled_metrics:{object_distance_world:.2}}};
  const raw=JSON.stringify({scene:{profile:'shared-world-v1',bounds:[3,2,2.5],dt:1/30},events:[event]});
  const elements=new Map(),calls=[];let source='',worker;
  const ctx=new Proxy({}, {get:(obj,name)=>obj[name]||((...args)=>{
    if(name==='clearRect')calls.length=0;
    calls.push({name,args:[...args],fillStyle:obj.fillStyle,strokeStyle:obj.strokeStyle,lineWidth:obj.lineWidth});
  }),set:(obj,name,value)=>(obj[name]=value,true)});
  function element(id) {
    if(!elements.has(id))elements.set(id,{value:id==='zoom'?'1':id==='yaw'?'.5':'0',checked:false,width:1080,height:420,
      open:false,textContent:id==='data'?raw:'',listeners:{},getContext:()=>ctx,
      addEventListener(type,fn){this.listeners[type]=fn;},remove(){}});
    return elements.get(id);
  }
  class Worker {
    constructor(){worker=this;this.queue=[];this.context=vm.createContext({postMessage:data=>this.onmessage({data})});vm.runInContext(source,this.context);}
    postMessage(data){this.queue.push(data);}
    flush(){this.context.onmessage({data:this.queue.shift()});}
    terminate(){}
  }
  vm.runInNewContext(replayScript,{document:{getElementById:element,addEventListener(){}},window:{addEventListener(){}},
    Blob:class{constructor(parts){source=parts.join('');}},URL:{createObjectURL:()=>'',revokeObjectURL(){}},Worker,
    setTimeout(){throw Error('Inspection must not start playback');},clearTimeout(){},location:{reload(){}}});
  worker.flush();worker.flush();assert.equal(worker.queue.length,0);
  return {calls,element,selected(){element('details').open=true;element('details').listeners.toggle();return JSON.parse(element('detail').textContent);}};
}

async function localAndReplayUseExactPartTransforms() {
  const input=objects(),original=JSON.stringify(input),ui=fixture('shared-world',{rigidObjects:input});await ui.ready();
  assertProjection(ui.ids.get('live').drawCalls);
  assertProjection(ui.ids.get('world-canvas').drawCalls);
  assert.equal(ui.ids.get('objects').children.length,3);
  assert.match(ui.ids.get('objects').children[0].children[2].textContent,/3 authored parts.*body: box.*knob: sphere.*handle: capsule/);
  assert.deepEqual(ui.ids.get('actor').children.map(option=>option.value),['human','block','companion'],'Parts never become independent controls');
  const saved=replay(input);assertProjection(saved.calls);
  assert.deepEqual(paths(ui.ids.get('live').drawCalls),paths(saved.calls),'Live and offline geometry agree');
  assert.deepEqual(saved.selected().snapshot.objects,input,'Inspection preserves complete local geometry and parent state');
  const before=ui.backendState(),count=ui.requests.length;
  ui.ids.get('spatial-yaw').value='-1.2';ui.ids.get('spatial-yaw').dispatch('input');
  ui.ids.get('spatial-zoom').value='2.3';ui.ids.get('spatial-zoom').dispatch('input');
  finiteCalls(ui.ids.get('live').drawCalls);assert.equal(ui.requests.length,count);assert.deepEqual(ui.backendState(),before);
  assert.equal(JSON.stringify(input),original);
}

async function staticPartsRemainWireframes() {
  const input=objects();input[0].role='static';
  const ui=fixture('shared-world',{rigidObjects:input});await ui.ready();
  assert.equal(paths(ui.ids.get('live').drawCalls).length,0,'Static parts must not occlude the human with solid fills');
  assertProjection(ui.ids.get('live').drawCalls,'stroke');
  const saved=replay(input);assert.equal(paths(saved.calls).length,0);assertProjection(saved.calls,'stroke');
  assert.deepEqual(ui.ids.get('actor').children.map(option=>option.value),['human','companion']);
}

async function smallCompoundPartsUseTheirActualRadius() {
  const input=objects();input[0].geometry.parts=[{...input[0].geometry.parts[1],radius:.01}];input[1].radius=.01;input[1].color='#7235A2';
  const ui=fixture('shared-world',{rigidObjects:input});await ui.ready();
  for(const calls of [ui.ids.get('live').drawCalls,replay(input).calls]) {
    const compoundArcs=paths(calls).flat().filter(call=>call.name==='arc');assert.equal(compoundArcs.length,3);
    for(const arc of compoundArcs)near(arc.args[2],1.008,'Compound sphere keeps its actual projected radius');
    const legacyArcs=paths(calls,'fill',input[1].color).flat().filter(call=>call.name==='arc');assert.equal(legacyArcs.length,3);
    for(const arc of legacyArcs)assert.equal(arc.args[2],2,'Legacy primitive display remains unchanged');
  }
}

function capacityObjects() {
  return Array.from({length:4},(_,i)=>({...structuredClone(compound),id:['block','companion','third','fourth'][i],
    position:[.6+i*.5,1,.6],quaternion:[1,0,0,0],geometry:{...structuredClone(compound.geometry),parts:Array.from({length:8},(_,j)=>({
      id:`part-${j}`,shape:j%2?'capsule':'sphere',position:[(j%4)*.09-.135,Math.floor(j/4)*.12-.06,0],
      quaternion:[q,0,q,0],radius:.025,...(j%2?{length:.1}:{})}))}}));
}
async function malformedOrOversizedCompoundsNeverBecomeBoxes() {
  const invalid=[];
  for(const edit of [o=>o.geometry.profile='unknown',o=>o.geometry.parts.push(...Array(6).fill(o.geometry.parts[0])),
      o=>o.geometry.parts[2].length=NaN,o=>o.geometry.parts[0].quaternion=[1,1,0,0],o=>o.geometry.parts[0]=null]) {
    const input=objects();edit(input[0]);invalid.push(input);
  }
  invalid.push([...capacityObjects(),structuredClone(ball)]);
  for(const input of invalid) {
    const ui=fixture('shared-world',{rigidObjects:input});await ui.ready();
    for(const calls of [ui.ids.get('live').drawCalls,replay(input).calls]) {
      assert.equal(calls.filter(call=>call.name==='fillText'&&String(call.args[0]).includes('Rigid geometry unavailable')).length,3);
      assert.equal(paths(calls).length,0,'Invalid compound must not silently become a bounding box or truncated part list');
      finiteCalls(calls);
    }
  }
}
async function maximumPartSceneRemainsBounded() {
  const input=capacityObjects(),original=JSON.stringify(input),ui=fixture('shared-world',{rigidObjects:input});await ui.ready();
  for(const calls of [ui.ids.get('live').drawCalls,replay(input).calls]) {
    assert.equal(paths(calls).length,96,'All 32 parts appear in each of three views');finiteCalls(calls);
  }
  const timings=[],before=ui.requests.length;
  for(let i=0;i<60;i++) {
    ui.ids.get('spatial-yaw').value=String(-1.5+i*.05);
    const start=performance.now();ui.ids.get('spatial-yaw').dispatch('input');timings.push(performance.now()-start);
    assert.equal(paths(ui.ids.get('live').drawCalls).length,96);
  }
  assert.equal(ui.requests.length,before);assert.equal(JSON.stringify(input),original);
  timings.sort((a,b)=>a-b);
  console.log(JSON.stringify({scope:'32-part analytic mocked-canvas redraw of both live/world views; excludes browser layout/paint and physics',
    redraws:timings.length,median_ms:timings[30],p95_ms:timings[56],max_ms:timings.at(-1)}));
}

const checks=[localAndReplayUseExactPartTransforms,staticPartsRemainWireframes,smallCompoundPartsUseTheirActualRadius,
  malformedOrOversizedCompoundsNeverBecomeBoxes,maximumPartSceneRemainsBounded];
const requested=new Set(process.argv.slice(2)),selected=requested.size?checks.filter(check=>requested.has(check.name)):checks;
assert(!requested.size||selected.length===requested.size,'Unknown compound check');
for(const check of selected) {
  await check();console.log(`PASS ${check.name}`);
}
