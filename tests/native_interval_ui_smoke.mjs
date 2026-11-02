/** Native interval presentation and actual worker/timer scheduling, without a browser. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {fixture, inspectionFixture} from './lab_ui_smoke.mjs';

const marker={profile:'native-terminal-control-interval-v1',duration_s:.075,terminal_duration_s:.075-2/30};
const timeline={status:'available',reference_span_s:[0,.075],reference_poses:4,task_window_s:[0,.075],
  task_duration_s:.075,task_steps:3,native_source_clock:'declared_native_presentation_binding',
  terminal_control_duration_s:marker.terminal_duration_s};
const ui=fixture('shared-world',{sharedExercise:true,sharedProfile:'shared-world-v3',
  exerciseInterval:{start_offset_s:0,horizon_steps:3,native_interval:marker},
  inspection:{...inspectionFixture(),timeline}});
await ui.ready();
assert.match(ui.ids.get('exercise-task-scope').textContent,/0\.07 seconds \(3 control steps\)/);
assert.match(ui.ids.get('window-task').textContent,/0\.000–0\.075s.*0\.075s/);
assert.match(ui.ids.get('window-clocks').textContent,/native presentation.*0\.008333s.*capture timing is unverified/);

const html=fs.readFileSync(new URL('../src/weft/assets/shared-world-replay.html',import.meta.url),'utf8');
const script=[...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
const scene={profile:'shared-world-v3',bounds:[3,2,2.5],dt:1/30,objects:[],target:null,
  task:{native_interval:marker,object_goal:null}};
const times=[0,1/30,2/30,.075];
const events=times.map((time_s,index)=>({type:index?'step':'start',snapshot:{time_s,source_time_s:time_s,
  human:{segments:[],muscles:[],mobility:{floor_height:0}},objects:[],target:null,
  contact_details:[],contact_count:0,contact_summary:{},coupled_metrics:{},substeps:16}}));
const raw=JSON.stringify({scene,events}),elements=new Map(),timers=new Map();
let worker,blobSource='',timerId=0;
const context=new Proxy({}, {get:(object,key)=>object[key]||(()=>{}),set:(object,key,value)=>(object[key]=value,true)});
function element(id){
  if(!elements.has(id))elements.set(id,{id,value:id==='zoom'?'1':'0',checked:false,disabled:true,hidden:false,
    open:false,width:1080,height:420,textContent:id==='data'?raw:'',listeners:{},
    addEventListener(name,fn){this.listeners[name]=fn;},setAttribute(){},remove(){},getContext(){return context;}});
  return elements.get(id);
}
class FakeWorker{
  constructor(){worker=this;this.queue=[];this.context=vm.createContext({postMessage:data=>this.onmessage({data})});vm.runInContext(blobSource,this.context);}
  postMessage(data){this.queue.push(data);}
  flush(){assert(this.queue.length);this.context.onmessage({data:this.queue.shift()});}
  terminate(){}
}
vm.runInNewContext(script,{document:{hidden:false,getElementById:element,addEventListener(){}},
  window:{addEventListener(){}},location:{reload(){}},Blob:class{constructor(parts){blobSource=parts.join('');}},
  URL:{createObjectURL(){return 'blob:native';},revokeObjectURL(){}},Worker:FakeWorker,
  setTimeout(fn,delay){timers.set(++timerId,{fn,delay});return timerId;},clearTimeout(id){timers.delete(id);}});
worker.flush();worker.flush();
element('play').listeners.click();
for(let index=1;index<times.length;index++){
  assert.equal(timers.size,1);
  const [id,timer]=[...timers][0];
  assert(Math.abs(timer.delay-1000*(times[index]-times[index-1]))<1e-9);
  timers.delete(id);timer.fn();
  assert.equal(timers.size,0,'Wait for the requested state before scheduling its next interval');
  assert.equal(worker.queue.length,1);
  worker.flush();
  assert.equal(element('frame').textContent,`${index+1} / 4`);
}
assert.equal(timers.size,0);assert.equal(element('play').textContent,'Play');
assert.equal(JSON.stringify({scene,events}),raw);
const delay=vm.runInNewContext(`(${script.match(/function nativeReplayDelay\(bundle,index\)\{[\s\S]*?\n\}/)[0]})`);
const failed={scene,events:[
  {snapshot:{time_s:2/30},exercise_metrics:{simulated_duration_s:2/30}},
  {snapshot:{time_s:2/30},exercise_metrics:{simulated_duration_s:2/30+.001},engine_state_status:'last_valid_control_boundary'},
]};
assert(Math.abs(delay(failed,0)-1)<1e-9,'Use observed task time after partial failure, not stale geometry');
assert.equal(delay({...failed,scene:{...scene,task:{}}},0),1000/30,'Legacy timing stays unchanged');
console.log('PASS native interval labels, exact terminal/failure playback delay, one-frame queue and final stop');
