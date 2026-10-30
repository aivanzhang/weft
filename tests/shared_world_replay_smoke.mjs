/** Recorded mixed-world inspection only: no browser, server or live physics. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html=fs.readFileSync(new URL('../src/weft/assets/shared-world-replay.html',import.meta.url),'utf8');
const script=[...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)].at(-1)[1];
assert.match(html,/offline replay, not live controls/);
assert.match(html,/not measured activation; physical validation is unverified/);
const block={id:'block',role:'passive',shape:'box',size:[.1,.1,.1],position:[.8,.6,.4],quaternion:[1,0,0,0],color:'#965644'};
const ball={id:'ball',role:'passive',shape:'sphere',radius:.04,position:[1.2,.6,.4],quaternion:[1,0,0,0],color:'#C7795D'};
const floor={...block,id:'floor',role:'static',position:[1.5,1,.05],size:[3,2,.1],color:'#D8C9BA'};
const scene={profile:'shared-world-v1',bounds:[3,2,2.5],dt:1/30,task:{object_goal:{position_tolerance:.04}}};
const events=[0,1,2,3].map(index=>({type:index?'step':'start',action:Array(28).fill(0),snapshot:{
  objects:[{...block,position:[.8+.05*index,.6,.4]},ball,floor],target:[1.8,.6,.4],
  human:{segments:[{id:'torso',a:[.8+.05*index,.6,.5],b:[.8+.05*index,.6,1.1],radius:.1}],
    muscles:[{name:'elbow_flexor',a:[.8,.6,.5],b:[.85,.6,.9],activation:.2+.2*index,status:'illustrative'}],
    mobility:{floor_height:.1}},
  contact_details:[{kind:'human_object',objects:['human:hand','block'],position:[.8,.6,.45],normal_force:2},
    {kind:'object_object',objects:['block','floor'],position:[.8,.6,.1],normal_force:1}],
  contact_count:2,contact_summary:{human_object:1},coupled_metrics:{object_distance_world:.2},
  time_s:index/30,substeps:8}}));
const original=JSON.stringify({scene,events});
const elements=new Map(),timers=new Map(),documentListeners={},windowListeners={},revoked=[];
let calls=[],timerId=0,blobSource='',worker,reloads=0;
const ctx=new Proxy({}, {get:(object,name)=>object[name]||((...args)=>{
  if(name==='clearRect') calls=[];
  if(['arc','moveTo','lineTo','rect'].includes(name)) assert(args.every(Number.isFinite));
  calls.push({name,args,fillStyle:object.fillStyle,strokeStyle:object.strokeStyle,lineWidth:object.lineWidth});
}),set:(object,name,value)=>(object[name]=value,true)});
function element(id) {
  if(!elements.has(id)) elements.set(id,{id,value:id==='zoom'?'1':id==='yaw'?'.5':'0',checked:id==='muscles',
    disabled:true,open:false,width:1080,height:420,textContent:id==='data'?original:'',listeners:{},
    addEventListener(name,fn){this.listeners[name]=fn;},remove(){this.removed=true;},getContext(){return ctx;}});
  return elements.get(id);
}
class FakeWorker {
  constructor(){this.queue=[];worker=this;this.context=vm.createContext({postMessage:data=>this.onmessage({data})});vm.runInContext(blobSource,this.context);}
  postMessage(data){assert(!this.terminated);this.queue.push(data);}
  flush(){assert(this.queue.length);this.context.onmessage({data:this.queue.shift()});}
  terminate(){this.terminated=true;}
}
const document={hidden:false,getElementById:element,addEventListener(name,fn){documentListeners[name]=fn;}};
const sandbox=vm.createContext({document,window:{addEventListener(name,fn){windowListeners[name]=fn;}},
  location:{reload(){reloads++;}},Blob:class{constructor(parts){blobSource=parts.join('');}},
  URL:{createObjectURL(){return 'blob:shared-replay-test';},revokeObjectURL(url){revoked.push(url);}},Worker:FakeWorker,
  setTimeout(fn,delay){timers.set(++timerId,{fn,delay});return timerId;},clearTimeout(id){timers.delete(id);}});
vm.runInContext(script,sandbox);
assert.equal(worker.queue[0].type,'init');assert.equal(element('data').removed,true);
assert.equal(element('play').disabled,true);
worker.flush();assert.equal(revoked.length,1);assert.equal(worker.queue.length,1);worker.flush();
assert.equal(element('play').disabled,false);assert.equal(element('frame').textContent,'1 / 4');
assert.equal(element('detail').textContent,'');
assert.deepEqual(calls.filter(c=>c.name==='fillText').map(c=>c.args[0]),['Front: x / height','Top: x / depth','3D: coupled world']);
assert.equal(calls.filter(c=>c.name==='clip').length,3);
assert.equal(calls.filter(c=>c.name==='stroke'&&c.strokeStyle==='#9BB3B0').length,3,'Body capsule is rendered in every pane');
assert.equal(calls.filter(c=>c.name==='fill'&&c.fillStyle==='#965644').length,21,'Dynamic block faces and ordinary contact are rendered');
assert.equal(calls.filter(c=>c.name==='stroke'&&c.strokeStyle?.startsWith('rgb(')).length,3,'Illustrative muscles are rendered in every pane');
assert.equal(calls.filter(c=>c.name==='arc'&&c.args[2]===4.5&&c.fillStyle==='#C7795D').length,3);
assert.equal(calls.filter(c=>c.name==='arc'&&c.args[2]===2.5).length,3);
assert.match(element('status').textContent,/2 contacts \(1 human↔object\).*object dist 0\.200.*substeps 8/);

for(const value of ['1','2','3']){element('scrub').value=value;element('scrub').listeners.input();}
assert.equal(worker.queue.length,1,'Only one requested frame can be in flight');
const previousCalls=calls;worker.flush();
assert.equal(calls,previousCalls,'A stale frame cannot replace the latest requested view');
assert.equal(worker.queue.length,1);assert.equal(worker.queue[0].index,3);worker.flush();
assert.equal(element('frame').textContent,'4 / 4');assert.equal(worker.queue.length,0);
element('details').open=true;element('details').listeners.toggle();
assert.equal(JSON.parse(element('detail').textContent).snapshot.time_s,.1);
assert.equal(JSON.parse(element('detail').textContent).snapshot.contact_details[0].kind,'human_object');
const frozenDetails=element('detail').textContent;
element('yaw').value='-1.2';element('yaw').listeners.input();
element('zoom').value='2.5';element('zoom').listeners.input();
element('muscles').checked=false;element('muscles').listeners.change();
assert(!calls.some(c=>c.name==='stroke'&&c.strokeStyle?.startsWith('rgb(')));
assert.equal(element('detail').textContent,frozenDetails);assert.equal(worker.queue.length,0);assert.equal(timers.size,0);
assert.equal(JSON.stringify({scene,events}),original,'Camera and display settings must not change recorded evidence');
element('details').open=false;element('details').listeners.toggle();assert.equal(element('detail').textContent,'');

element('play').listeners.click();assert.equal(timers.size,1);assert.equal([...timers.values()][0].delay,1000/30);
worker.flush();
const [id,timer]=[...timers][0];timers.delete(id);timer.fn();
assert.equal(worker.queue.length,1);worker.flush();assert.equal(element('frame').textContent,'2 / 4');
document.hidden=true;documentListeners.visibilitychange();assert.equal(timers.size,0);assert.equal(element('play').textContent,'Play');
document.hidden=false;documentListeners.visibilitychange();assert.equal(timers.size,0,'Becoming visible must not silently resume playback');
windowListeners.pagehide();assert.equal(worker.terminated,true);assert.equal(timers.size,0);assert.equal(revoked.length,2);
const label=element('frame').textContent;
worker.onmessage({data:{type:'frame',index:0,event:events[0]}});assert.equal(element('frame').textContent,label,'Late worker results cannot redraw a closed page');
windowListeners.pageshow({persisted:true});assert.equal(reloads,1);
console.log('Shared-world replay checks passed: mixed 3-pane body/objects, illustrative muscles, distinct simulated contacts, worker parsing, bounded latest-frame queue, deferred details, camera-only controls, simulation-clock playback, hidden pause, worker cleanup and cached-page recovery. No browser was used.');
