import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import vm from "node:vm";
import {createHash,webcrypto} from "node:crypto";
import {createRequire} from "node:module";
const require=createRequire(import.meta.url),math=require("../src/weft/assets/avatar_retarget.js");
const viewerSource=readFileSync(new URL("../src/weft/assets/avatar_viewer.js",import.meta.url),"utf8");
const workerSource=readFileSync(new URL("../src/weft/assets/avatar_worker.js",import.meta.url),"utf8");
const copyBuffer=bytes=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);
const hash=bytes=>createHash("sha256").update(new Uint8Array(bytes)).digest("hex");
const texture=copyBuffer(Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l4sAAAAASUVORK5CYII=","base64"));
const document={profile:"weft-avatar-render-v1",nodes:[{index:0,parent:null,local_matrix:math.identity()}],skins:[{joints:[0],inverse_bind_matrices:[math.identity()]}],
  primitives:[{node:0,skin:0,positions:[-.1,0,.3,.1,0,.3,0,0,.5],normals:[0,-1,0,0,-1,0,0,-1,0],texcoords:[0,0,1,0,.5,1],joints:new Array(12).fill(0),weights:[1,0,0,0,1,0,0,0,1,0,0,0],indices:[0,1,2],material:0}],
  materials:[{base_color_factor:[.2,.5,.8,1],base_color_texture:0,unlit:true,alpha_mode:"MASK",alpha_cutoff:.4,double_sided:true}],
  textures:[{path:`textures/${hash(texture)}.png`,sha256:hash(texture),mime_type:"image/png",width:1,height:1,wrap_s:10497,wrap_t:33071,mag_filter:9728,min_filter:9987}],
  retargeting:{profile:"weft-avatar-retarget-v1",root_node:0,scale:1,orientation_xyzw:[0,0,0,1],bindings:[{node:0,body:"mobile_root",translation:[0,0,0],rotation_xyzw:[0,0,0,1]}]}};
const renderBytes=()=>new TextEncoder().encode(JSON.stringify(document)).buffer;
const packet=()=>{const bytes=renderBytes();return {renderBytes:bytes,renderSha256:hash(bytes),textures:[texture.slice(0)]};};
const scene={profile:"shared-world-v2",bounds:[3,3,3],coordinate_frame:{scale_bridge:{body_height_to_world_scale:1,origin_world:[1,1,0]}},human_spec:{lengths:{torso:.5,shoulder_width:.36,hip_width:.24,upper_arm:.28,forearm:.25,thigh:.42,shin:.4},mobility:{base_height:1},reference:["must not reach viewer"]},objects:[{id:"box",shape:"box",role:"passive",size:[.1,.1,.1],position:[1,1,.1],quaternion:[1,0,0,0],color:"#965644"}]};
const angles={torso_twist:0,torso_lateral:0,torso_flex:0},joints={pelvis:[1,1,1],neck:[1,1,1.5],head:[1,1,1.68]};
for(const [side,sign] of [["left",-1],["right",1]])for(const [proximal,distal,tip,width,z,length1,length2] of [["shoulder","elbow","wrist",.36,1.5,.28,.25],["hip","knee","ankle",.24,1,.42,.4]]){
  for(const suffix of ["","_twist","_lateral"])angles[`${side}_${proximal}${suffix}`]=0;angles[`${side}_${distal}`]=0;
  joints[`${side}_${proximal}`]=[1+sign*width/2,1,z];joints[`${side}_${distal}`]=[1+sign*width/2,1,z-length1];joints[`${side}_${tip}`]=[1+sign*width/2,1,z-length1-length2];
}
const snapshot={human:{angles,joints,mobility:{translation:[0,0,0]},segments:[{a:joints.pelvis,b:joints.neck}],muscles:[{a:joints.left_shoulder,b:joints.left_elbow,activation:.7}]},objects:scene.objects};

function controllerHarness({supported=true}={}){
  const workers=[],timers=new Map(),listeners=new Map(),statuses=[];let nextTimer=0;
  class Worker{constructor(){this.messages=[];this.terminated=false;workers.push(this);}postMessage(data){this.messages.push(data);}terminate(){this.terminated=true;}emit(data){this.onmessage({data});}}
  const doc={hidden:false,addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name)};
  const context={ArrayBuffer,Worker:supported?Worker:undefined,document:doc,setTimeout:fn=>{timers.set(++nextTimer,fn);return nextTimer;},clearTimeout:id=>timers.delete(id),addEventListener:(name,fn)=>listeners.set(name,fn),removeEventListener:name=>listeners.delete(name)};
  context.globalThis=context;vm.runInNewContext(viewerSource,context);
  const canvas={width:1080,height:420,transferControlToOffscreen:()=>({})};
  const view=context.WeftAvatarViewer.create(canvas,{onStatus:value=>statuses.push(value)});
  return {view,workers,timers,listeners,statuses,doc};
}
const h=controllerHarness(),worker=h.workers[0];h.view.configure(scene);
assert.equal(worker.messages.find(row=>row.type==="configure").scene.human_spec.reference,undefined);
const first=h.view.load(packet()),second=h.view.load(packet()),third=h.view.load(packet());
assert.equal((await second).status,"cancelled");assert.equal(worker.messages.filter(row=>row.type==="load").length,1);
worker.emit({type:"loadCancelled",generation:1});assert.equal((await first).status,"cancelled");
assert.equal(worker.messages.filter(row=>row.type==="load").length,2);
worker.emit({type:"loaded",generation:3,stats:{vertices:3}});assert.equal((await third).status,"ready");
for(let i=0;i<1000;i++)h.view.update({snapshot,camera:{yaw:i/1000},overlay:true,showMuscles:true});
assert.equal(worker.messages.filter(row=>row.type==="frame").length,1);worker.emit({type:"frame",sequence:1});
assert.equal(worker.messages.filter(row=>row.type==="frame").length,2);assert.equal(worker.messages.at(-1).sequence,1000);
h.view.setVisible(false);worker.emit({type:"frame",sequence:1000});h.view.update({snapshot});
h.doc.hidden=true;h.listeners.get("visibilitychange")();h.doc.hidden=false;h.listeners.get("visibilitychange")();
assert.equal(worker.messages.filter(row=>row.type==="frame").length,2,"Returning to browser must not unhide inactive panel");
h.view.setVisible(true);assert.equal(worker.messages.filter(row=>row.type==="frame").length,3);worker.emit({type:"frame",sequence:1001});
assert.equal(h.statuses.filter(row=>row.status==="ready").length,1,"No per-frame status/render churn");
const invalid=packet();invalid.renderSha256="bad";assert.equal((await h.view.load(invalid)).status,"fallback");h.view.update({snapshot});assert.equal(worker.messages.filter(row=>row.type==="frame").length,3);
h.view.dispose();worker.emit({type:"disposed"});assert.equal(worker.terminated,true);assert.equal(h.timers.size,0);assert.equal(h.listeners.size,0);
const unavailable=controllerHarness({supported:false});assert.equal(unavailable.statuses.at(-1).status,"fallback");assert.equal((await unavailable.view.load(packet())).status,"fallback");
const timeout=controllerHarness();timeout.view.configure(scene);const late=timeout.view.load(packet());[...timeout.timers.values()][0]();assert.equal((await late).status,"fallback");assert.equal(timeout.workers[0].terminated,true);timeout.view.dispose();
const mismatch=controllerHarness();mismatch.view.configure(scene);const mismatchLoad=mismatch.view.load(packet());mismatch.workers[0].emit({type:"loaded",generation:1});await mismatchLoad;
mismatch.view.update({snapshot});mismatch.workers[0].emit({type:"frame",sequence:1,error:"Avatar FK disagrees with recorded left_wrist"});
for(let i=0;i<10;i++)mismatch.view.update({snapshot});
assert.equal(mismatch.workers[0].messages.filter(row=>row.type==="frame").length,1,"Failed FK stays suspended until a new selection");
assert.equal(mismatch.statuses.at(-1).status,"fallback");mismatch.view.dispose();mismatch.workers[0].emit({type:"disposed"});

// Call traces test worker ownership and shader inputs, not actual GPU execution.
function fakeGl(){
  const calls=[],constants=new Map(),objects=new Map();let next=1;
  const target={calls,objects};
  return new Proxy(target,{get(object,key){
    if(key in object)return object[key];
    if(/^[A-Z0-9_]+$/.test(key)){if(!constants.has(key))constants.set(key,key==="NO_ERROR"?0:constants.size+1);return constants.get(key);}
    if(key==="getParameter")return value=>value===constants.get("MAX_TEXTURE_SIZE")?4096:16;
    if(key==="getShaderParameter"||key==="getProgramParameter")return ()=>true;
    if(key==="getError")return ()=>0;
    if(key==="getUniformLocation")return (_handle,name)=>name;
    if(key.startsWith("create"))return ()=>{const handle={id:next++,type:key};objects.set(handle.id,handle);return handle;};
    if(key.startsWith("delete"))return handle=>{if(handle)objects.delete(handle.id);calls.push([key,handle]);};
    return (...args)=>{calls.push([key,...args]);};
  }});
}
const gl=fakeGl(),messages=[],bitmaps=[];let imported=false,closed=false;
const context={ArrayBuffer,Uint8Array,Float32Array,Uint32Array,TextDecoder,Blob,crypto:webcrypto,WeftAvatarRetarget:math,
  postMessage:message=>messages.push(message),close:()=>{closed=true;},importScripts:()=>{imported=true;throw new Error("No network expected");},
  createImageBitmap:async(_blob,options)=>{assert.equal(options.premultiplyAlpha,"none");const bitmap={width:1,height:1,closed:false,close(){this.closed=true;}};bitmaps.push(bitmap);return bitmap;}};
context.self=context;vm.runInNewContext(workerSource,context);
const send=data=>context.onmessage({data});
send({type:"init",canvas:{getContext:name=>{assert.equal(name,"webgl2");return gl;},addEventListener(){}},width:1080,height:420});
send({type:"configure",scene});
async function settle(predicate){for(let i=0;i<100;i++){if(predicate())return;await new Promise(resolve=>setTimeout(resolve,2));}assert.fail("Worker did not reach bounded test result");}
send({type:"load",generation:1,...packet()});await settle(()=>messages.some(row=>row.type==="loaded"));
assert.equal(imported,false);assert.equal(bitmaps.length,1);assert.equal(bitmaps[0].closed,true);
assert.ok(gl.calls.some(row=>row[0]==="generateMipmap"));assert.ok(gl.calls.some(row=>row[0]==="texParameteri"&&row[3]===33071));
assert.ok(gl.calls.some(row=>row[0]==="shaderSource"&&row[2].includes("texelFetch")),"GPU palette texture avoids 64-matrix uniform overflow");
send({type:"frame",sequence:1,snapshot,camera:{yaw:.5,zoom:1},overlay:true,showMuscles:true});
assert.equal(messages.at(-1).error,undefined);assert.equal(messages.at(-1).sequence,1);
assert.equal(gl.calls.filter(row=>row[0]==="drawElements").length,9,"Floor, equipment and skinned mesh in all three views");
assert.ok(gl.calls.some(row=>row[0]==="uniform1i"&&row[1]==="uMask"&&row[2]===true));
assert.ok(gl.calls.some(row=>row[0]==="uniform1i"&&row[1]==="uDoubleSided"&&row[2]===true));
const drawCount=gl.calls.filter(row=>row[0]==="drawElements").length;send({type:"visible",visible:false});send({type:"frame",sequence:2,snapshot,camera:{}});assert.equal(gl.calls.filter(row=>row[0]==="drawElements").length,drawCount);
const bad=packet();bad.textures=[new ArrayBuffer(4)];send({type:"load",generation:2,...bad});await settle(()=>messages.some(row=>row.type==="loadError"&&row.generation===2));assert.match(messages.at(-1).reason,/texture bytes changed/);
send({type:"dispose"});assert.equal(closed,true);assert.equal(gl.objects.size,0,"GL buffers, textures, programs and shaders are released");
console.log(JSON.stringify({status:"passed",queued_updates:1000,submitted_frames:3,worker_draw_calls:drawCount,scope:"Bounded controller/worker lifecycle, hash/texture/UV/material inputs and fallback; mocked WebGL, no browser/GPU performance claim"}));
