import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {spawnSync} from 'node:child_process';
import {readFileSync,writeFileSync} from 'node:fs';
const math=createRequire(import.meta.url)('../src/weft/assets/avatar_retarget.js');
const near=(a,b,e=2e-6)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<=e,`${i}: ${v} != ${b[i]}`));};
const py=String.raw`
import json,time
from pathlib import Path
from weft.avatar_gltf import normalize_avatar_gltf
out=[]
for name in ('weft-muse-panda-v1','weft-humanoid-robot-v1'):
 p=Path('tools/avatar-assets/outputs')/name;m=json.loads((p/'mapping.json').read_bytes())['retargeting'];m['profile']='weft-avatar-retarget-v2'
 m['bindings']=[{'node':r['node'],'body':r['body'].replace('left','TEMP').replace('right','left').replace('TEMP','right')} for r in m['bindings']]
 out.append({'name':name,'document':normalize_avatar_gltf('model.glb',{'model.glb':(p/'model.glb').read_bytes()},m,deadline=time.monotonic()+10)[0]})
print(json.dumps(out))
`;
const p=spawnSync('.venv-lab/bin/python',['-c',py],{encoding:'utf8',timeout:20000,maxBuffer:10*1024*1024});assert.equal(p.status,0,p.stderr);
const assets=JSON.parse(p.stdout);
function generic(){
 const I=math.identity(),nodes=[{index:0,parent:null,local_matrix:I}],bindings=[{node:0,body:'mobile_root'}];
 const add=(body,parent,translation,rotation=math.identity())=>{let index=nodes.length;nodes.push({index,parent,local_matrix:math.multiply(math.transform(translation),rotation)});bindings.push({node:index,body});return index;};
 const torso=add('torso',0,[0,0,1]);
 for(const [side,sign] of [['left',-1],['right',1]]){
  const shoulder=add(`${side}_shoulder_body`,torso,[sign*.5,0,.7],math.axisRotation([0,1,0],sign*.3));add(`${side}_elbow_body`,shoulder,[0,0,-.6]);
  const hip=add(`${side}_hip_body`,0,[sign*.3,0,.9]);add(`${side}_knee_body`,hip,[0,0,-.7]);
 }
 nodes.push({index:10,parent:torso,local_matrix:math.transform([.1,.1,.9])});
 const rest=[];for(const n of nodes)rest[n.index]=math.multiply(n.parent===null?I:rest[n.parent],n.local_matrix);
 const positions=[],joints=[],weights=[],indices=[];
 for(let i=1;i<10;i++){const c=rest[i].slice(12,15),v=positions.length/3;positions.push(...c,...c.map((x,j)=>x+(j===0?.05:0)),...c.map((x,j)=>x+(j===2?.05:0)));indices.push(v,v+1,v+2);for(let k=0;k<3;k++){joints.push(i-1,0,0,0);weights.push(1,0,0,0);}}
 return {profile:'weft-avatar-render-v1',nodes,skins:[{joints:bindings.slice(1).map(b=>b.node),inverse_bind_matrices:rest.slice(1,10).map(math.inverse)}],primitives:[{node:0,skin:0,positions,normals:Array.from({length:positions.length},(_,i)=>Number(i%3===1)),texcoords:null,joints,weights,indices,material:0}],materials:[{base_color_factor:[.3,.6,.8,1],base_color_texture:null,unlit:true,double_sided:false,alpha_mode:'OPAQUE',alpha_cutoff:.5}],textures:[],retargeting:{profile:'weft-avatar-retarget-v2',root_node:0,scale:1.7,orientation_xyzw:[0,0,0,1],bindings}};
}
assets.push({name:'generic-rotated-rest-long-limbs',document:generic()});
function body(index){
 const torso=math.multiply(math.transform([2,3,1.2]),math.multiply(math.axisRotation([0,0,1],index*.09),math.axisRotation([1,0,0],index*.04)));
 const frames={torso,mobile_root:math.transform([2,3,1.2])};
 for(const side of ['left','right'])for(const [first,second] of [['shoulder','elbow'],['hip','knee']]){
  const a=`${side}_${first}_body`,b=`${side}_${second}_body`,sign=side==='left'?-1:1;
  frames[a]=math.multiply(torso,math.multiply(math.axisRotation([0,1,0],index*.06*sign),math.axisRotation([1,0,0],index*.03)));
  frames[b]=math.multiply(frames[a],math.axisRotation([1,0,0],index*.08*(second==='knee'?-1:1)));
  // Deliberately different simulation proportions must not stretch avatar bones.
  frames[a][12]+=sign*(.2+index*.02);frames[b][14]-=.3+index*.05;
 }
 return {frames};
}
const proof=[];
for(const item of assets){
 const before=JSON.stringify(item.document),asset=math.prepareAsset(item.document),neutral=body(0),neutralPose=math.retarget(asset,neutral),torsoRest=asset.rest.get(asset.roles.get('torso'));
 const common=math.multiply(neutral.frames.torso,math.transform(torsoRest.slice(12,15).map(v=>-v)));
 for(const [id,m] of asset.rest)near(neutralPose.nodes.get(id),math.multiply(common,m));
 for(let i=0;i<12;i++){
  const input=body(i),saved=JSON.stringify(input),pose=math.retarget(asset,input);
  near(pose.nodes.get(asset.roles.get('torso')).slice(12,15),input.frames.torso.slice(12,15));
  for(const role of math.BODY_FRAMES.slice(1)){
   const id=asset.roles.get(role),m=pose.nodes.get(id),rest=asset.rest.get(id),rotation=input.frames[role].slice();rotation[12]=rotation[13]=rotation[14]=0;
   const expected=math.multiply(rotation,rest);for(const k of [0,1,2,4,5,6,8,9,10])near([m[k]],[expected[k]]);
   if(role!=='torso'){
    const parent=role.includes('elbow')?role.replace('elbow','shoulder'):role.includes('knee')?role.replace('knee','hip'):'torso',pid=asset.roles.get(parent);
    const length=(a,b)=>Math.hypot(...[12,13,14].map(k=>a[k]-b[k]));near([length(m,pose.nodes.get(pid))],[length(rest,asset.rest.get(pid))]);
   }
  }
  const moved=structuredClone(input);for(const m of Object.values(moved.frames)){m[12]+=.7;m[13]-=.3;m[14]+=.2;}
  const translated=math.retarget(asset,moved);for(const id of asset.order)near(translated.nodes.get(id).slice(12,15),pose.nodes.get(id).slice(12,15).map((v,k)=>v+[.7,-.3,.2][k]));
  const meshes=math.posedGeometry(asset,pose);assert.ok(meshes.every(m=>Array.from(m.positions).every(Number.isFinite)));
  assert.equal(JSON.stringify(input),saved);
  if([0,6,11].includes(i))proof.push({name:item.name,pose:i,meshes:meshes.map(m=>({positions:Array.from(m.positions),indices:Array.from(m.indices),color:asset.materials[m.material].base_color_factor})),pivot:input.frames.torso.slice(12,15)});
 }
 assert.equal(JSON.stringify(item.document),before);
}
for(const mutate of [d=>d.retargeting.extra=true,d=>d.retargeting.bindings.pop(),d=>d.retargeting.bindings[2].body='right_shoulder_body',d=>d.nodes[3].parent=0,d=>d.nodes[1].local_matrix[0]=2,d=>d.skins[0].inverse_bind_matrices[0][12]+=.1,d=>d.retargeting.bindings[0].translation=[0,0,0]]){const doc=generic();mutate(doc);assert.throws(()=>math.prepareAsset(doc));}
const actual=JSON.parse(readFileSync('tests/fixtures/avatar_forward.json','utf8')).case;
const physical=JSON.stringify(actual),live=math.bodyFrames(math.compileScene(actual.scene),actual.snapshot);
for(const item of assets){const a=math.prepareAsset(item.document);math.posedGeometry(a,math.retarget(a,live));}
assert.equal(JSON.stringify(actual),physical);
const personOnly=structuredClone(actual);personOnly.scene.profile='shared-world-v3';personOnly.scene.objects=[];personOnly.snapshot.objects=[];
const personConfig=math.compileScene(personOnly.scene),personPose=math.bodyFrames(personConfig,personOnly.snapshot);
for(const role of math.BODY_FRAMES)near(personPose.frames[role],live.frames[role]);
assert.equal(math.prepareWorld(personConfig).meshes.length,1);
for(const item of assets){const a=math.prepareAsset(item.document);assert.ok(math.posedGeometry(a,math.retarget(a,personPose)).every(m=>Array.from(m.positions).every(Number.isFinite)));}
if(process.argv[2])writeFileSync(process.argv[2],JSON.stringify(proof),{flag:'wx'});
console.log(JSON.stringify({status:'passed',assets:3,poses_per_asset:12,exact_rl_snapshot_cases:1,scope:'CPU pose/skin verification; rest lengths, exact rotations/root, no baked clips or simulator changes. Not browser/GPU or anatomical calibration.'}));
