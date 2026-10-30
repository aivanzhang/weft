import assert from "node:assert/strict";
import {createRequire} from "node:module";
import {spawnSync} from "node:child_process";
import {writeFileSync} from "node:fs";
import {performance} from "node:perf_hooks";
const require=createRequire(import.meta.url);
const math=require("../src/weft/assets/avatar_retarget.js");
const near=(actual,expected,tolerance=1e-9)=>{assert.equal(actual.length,expected.length);actual.forEach((value,i)=>assert.ok(Math.abs(value-expected[i])<=tolerance,`${i}: ${value} != ${expected[i]}`));};

// Forward kinematics only: no simulator actions, source studies, media or server.
const python=String.raw`
import json, math
import numpy as np
import mujoco
from weft.shared_world import _build_world_xml, HUMAN_SEGMENT_DEFS
from weft.shared_world_lab import authored_shared_world_scene
from weft.mobile_human import ROOT_AXES
from weft.spatial_pose import SPATIAL_JOINT_NAMES
rows=[]
for index in range(12):
    scene=authored_shared_world_scene()
    scale=[.45,1.,1.7][index%3]
    scene['coordinate_frame']['scale_bridge']['body_height_to_world_scale']=scale
    scene['coordinate_frame']['scale_bridge']['origin_world']=[1.2,.8,.1]
    yaw=.17*index; tilt=.03*index
    rz=np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
    ry=np.array([[math.cos(tilt),0,math.sin(tilt)],[0,1,0],[-math.sin(tilt),0,math.cos(tilt)]])
    scene['human_spec']['body_frame']={'rotation':(rz@ry).tolist()}
    if index%2: scene['human_spec']['mobility']['ground_plane']={'normal':[.1,.2,.97]}
    model=mujoco.MjModel.from_xml_string(_build_world_xml(scene)); data=mujoco.MjData(model)
    angles={name:float(.4*math.sin(index*.31+i*.29)) for i,name in enumerate(SPATIAL_JOINT_NAMES)}
    translation=[.07*math.sin(index),-.03*math.cos(index),.01*index]
    for name,value in angles.items():data.qpos[model.joint(name).qposadr[0]]=value
    for name,value in zip(ROOT_AXES,translation):data.qpos[model.joint(name).qposadr[0]]=value*scale
    mujoco.mj_forward(model,data)
    names={point for _,a,b,_ in HUMAN_SEGMENT_DEFS for point in [a,b]}
    joints={name:data.site_xpos[model.site(name+'_site').id].tolist() for name in names}
    frames={}
    for name in ['mobile_root','torso']+[f'{side}_{joint}_body' for side in ['left','right'] for joint in ['shoulder','elbow','hip','knee']]:
        bid=model.body(name).id; m=np.eye(4); m[:3,:3]=data.xmat[bid].reshape(3,3);m[:3,3]=data.xpos[bid]
        frames[name]=m.T.reshape(-1).tolist()
    human=scene['human_spec'];scene['human_spec']={key:human[key] for key in ['lengths','mobility','body_frame']}
    segments=[{'id':sid,'a':joints[a],'b':joints[b],'radius':radius*scale} for sid,a,b,radius in HUMAN_SEGMENT_DEFS]
    objects=[{'id':obj['id'],'role':obj['role'],'position':data.xpos[model.body(obj['id']).id].tolist(),'quaternion':data.xquat[model.body(obj['id']).id].tolist()} for obj in scene['objects']]
    rows.append({'scene':scene,'snapshot':{'human':{'angles':angles,'mobility':{'translation':translation},'joints':joints,'segments':segments,'muscles':[]},'objects':objects},'frames':frames})
print(json.dumps(rows))
`;
const result=spawnSync(".venv-lab/bin/python",["-c",python],{encoding:"utf8",timeout:20000,maxBuffer:2*1024*1024});
assert.equal(result.status,0,result.stderr);
const rows=JSON.parse(result.stdout);
for(const row of rows){const before=JSON.stringify(row),config=math.compileScene(row.scene),pose=math.bodyFrames(config,row.snapshot);for(const name of math.BODY_FRAMES)near(pose.frames[name],row.frames[name],2e-12);assert.equal(JSON.stringify(row),before);
  const world=math.prepareWorld(config),matrices=math.worldMatrices(world,row.snapshot);assert.equal(world.meshes.length,row.snapshot.objects.length+1);assert.equal(matrices.length,world.meshes.length);
  const views=math.cameraViews(config,row.snapshot,{yaw:.7,zoom:2});assert.equal(views.length,3);for(const view of views)near(math.point(view.matrix,config.bounds.map(v=>v/2)).slice(0,2),[0,-.08],1e-12);
}
const compound=structuredClone(rows[0]);compound.scene.objects[0]={id:compound.scene.objects[0].id,role:"passive",shape:"compound",color:"#965644",geometry:{parts:[{shape:"box",size:[.2,.1,.3],position:[.1,0,0],quaternion:[1,0,0,0]},{shape:"capsule",radius:.03,length:.1,position:[0,0,.3],quaternion:[1,0,0,0]}]}};
const compoundWorld=math.prepareWorld(math.compileScene(compound.scene));assert.equal(compoundWorld.meshes.length,compound.scene.objects.length+1);assert.ok(compoundWorld.meshes[0].indices.length>36);
const changed=structuredClone(rows[0]);changed.snapshot.human.joints.left_wrist[0]+=.001;
assert.throws(()=>math.bodyFrames(math.compileScene(changed.scene),changed.snapshot),/disagrees.*left_wrist/);
const missing=structuredClone(rows[0]);delete missing.snapshot.human.angles.left_shoulder_twist;
assert.throws(()=>math.bodyFrames(math.compileScene(missing.scene),missing.snapshot),/joint left_shoulder_twist/);
const task={profile:"shared-exercise-state-v1",engine_state:rows[0].snapshot};
near(math.bodyFrames(math.compileScene(rows[0].scene),task).frames.torso,rows[0].frames.torso,2e-12);

function fixture(){
  return {profile:"weft-avatar-render-v1",nodes:[{index:0,parent:null,local_matrix:math.identity()},{index:1,parent:0,local_matrix:math.transform([1,0,0])},{index:2,parent:1,local_matrix:math.transform([0,1,0])}],
    skins:[{joints:[0,1],inverse_bind_matrices:[math.identity(),math.inverse(math.transform([1,0,0]))]}],
    primitives:[{node:0,skin:0,positions:[0,0,0,2,0,0,1,1,0],normals:[0,0,1,0,0,1,0,0,1],texcoords:[0,0,1,0,.5,1],joints:[0,0,0,0,1,0,0,0,0,1,0,0],weights:[1,0,0,0,1,0,0,0,.5,.5,0,0],indices:[0,1,2],material:0},
      {node:2,skin:null,positions:[0,0,0,.1,0,0,0,.1,0],normals:[0,0,1,0,0,1,0,0,1],texcoords:null,joints:null,weights:null,indices:[0,1,2],material:0}],
    materials:[{base_color_factor:[.2,.7,.3,1],base_color_texture:null,unlit:true,alpha_mode:"OPAQUE",alpha_cutoff:.5,double_sided:false}],textures:[],
    retargeting:{profile:"weft-avatar-retarget-v1",root_node:0,scale:1,orientation_xyzw:[0,0,0,1],bindings:[{node:0,body:"mobile_root",translation:[0,0,0],rotation_xyzw:[0,0,0,1]},{node:1,body:"left_shoulder_body",translation:[0,0,0],rotation_xyzw:[0,0,0,1]}]}};
}
const raw=fixture(),before=JSON.stringify(raw),asset=math.prepareAsset(raw);
const atRest=math.retarget(asset,{frames:{mobile_root:math.identity(),left_shoulder_body:math.transform([1,0,0])}});
near(math.posedGeometry(asset,atRest)[0].positions,raw.primitives[0].positions,1e-7);
const moved=math.retarget(asset,{frames:{mobile_root:math.identity(),left_shoulder_body:math.multiply(math.transform([1,0,0]),math.axisRotation([0,0,1],Math.PI/2))}});
const posed=math.posedGeometry(asset,moved);
near(posed[0].positions,[0,0,0,1,1,0,.5,.5,0],1e-6);
near(posed[1].positions.slice(0,3),[0,0,0],1e-6);
assert.equal(JSON.stringify(raw),before);
assert.deepEqual([...posed[0].texcoords],[0,0,1,0,.5,1]);
assert.equal(posed[0].material,0);
const shifted=fixture();shifted.retargeting.scale=2;shifted.retargeting.bindings[0].translation=[0,0,.5];
const shiftedPose=math.retarget(math.prepareAsset(shifted),{frames:{mobile_root:math.transform([3,4,5]),left_shoulder_body:math.identity()}});
near(math.point(shiftedPose.nodes.get(0),[1,0,0]),[5,4,5.5]);
const nearUnit=fixture();nearUnit.primitives[0].weights[0]=.99995;
const nearUnitAsset=math.prepareAsset(nearUnit),nearUnitPose=math.retarget(nearUnitAsset,{frames:{mobile_root:math.transform([3,4,5]),left_shoulder_body:math.transform([1,0,0])}});
near(math.posedGeometry(nearUnitAsset,nearUnitPose)[0].positions.slice(0,3),[3,4,5],1e-6);
const stretched=math.identity();stretched[0]=2;stretched[10]=.5;
const normalLength=Math.hypot(.5,1,2);near(math.normalVector(stretched,[1,1,1]),[.5,1,2].map(value=>value/normalLength));
const collapsed=math.identity();collapsed[0]=0;near(math.normalVector(collapsed,[0,0,1]),[0,0,1]);
for(const mutate of [
  x=>{x.nodes[0].parent=1;},x=>{x.retargeting.bindings[0].body="guessed_head";},x=>{x.skins[0].joints.push(120);},
  x=>{x.primitives[0].weights[0]=.7;},x=>{x.primitives[0].joints[0]=2;},x=>{x.materials[0].alpha_mode="BLEND";},
  x=>{x.primitives[0].positions[0]=NaN;},x=>{x.nodes[2].local_matrix[0]=0;},x=>{x.retargeting.root_node=2;},
  x=>{x.materials[0].base_color_texture=0;},x=>{x.retargeting.orientation_xyzw=[0,0,0,2];},
]){const invalid=fixture();mutate(invalid);assert.throws(()=>math.prepareAsset(invalid));}
const large=fixture();large.primitives=large.primitives.slice(0,1);const grid=large.primitives[0];
grid.positions=[];grid.normals=[];grid.texcoords=[];grid.joints=[];grid.weights=[];grid.indices=[];
for(let y=0;y<100;y++)for(let x=0;x<200;x++){grid.positions.push(x/200,y/100,0);grid.normals.push(0,0,1);grid.texcoords.push(x/200,y/100);grid.joints.push(0,0,0,0);grid.weights.push(1,0,0,0);if(x<199&&y<99){const a=y*200+x;grid.indices.push(a,a+1,a+200,a+1,a+201,a+200);}}
const begin=performance.now(),largeAsset=math.prepareAsset(large),prepareMilliseconds=performance.now()-begin;
assert.equal(largeAsset.stats.vertices,20000);assert.equal(largeAsset.stats.triangles,39402);
const poseBegin=performance.now(),largePose=math.retarget(largeAsset,{frames:{mobile_root:math.identity(),left_shoulder_body:math.transform([1,0,0])}});
const largeGeometry=math.posedGeometry(largeAsset,largePose),poseMilliseconds=performance.now()-poseBegin;
assert.equal(largeGeometry[0].positions.length,60000);assert.ok(Array.from(largeGeometry[0].positions).every(Number.isFinite));
const oversized=structuredClone(large);oversized.primitives.push(fixture().primitives[0]);assert.throws(()=>math.prepareAsset(oversized),/vertex bound/);
console.log(JSON.stringify({status:"passed",mujoco_forward_only_cases:rows.length,body_matrices:rows.length*math.BODY_FRAMES.length,large_geometry:{vertices:20000,triangles:39402,prepare_ms:prepareMilliseconds,cpu_pose_ms:poseMilliseconds},scope:"Exact analytic FK, explicit retarget/weights/attachment/UV preservation and rejection guards; one Node timing, no browser/GPU/sustained-performance claim"}));
if(process.argv[2])writeFileSync(process.argv[2],JSON.stringify({profile:"weft-avatar-analytic-forward-fixtures-v1",scope:"Analytic forward kinematics only; no simulation steps or authored character art",cases:rows}),{flag:"wx"});
