import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {writeFileSync} from 'node:fs';
import {performance} from 'node:perf_hooks';
const require=createRequire(import.meta.url),math=require('../src/weft/assets/avatar_retarget.js');
const id=math.identity;
function fixture(count=3){
  const positions=[],normals=[],joints=[],weights=[],indices=[];
  for(let i=0;i<count;i++){positions.push((i%31)/10-1.5,Math.floor(i/31)%19/10-.9,(i%7)/10);normals.push(0,0,1);joints.push(0,1,0,0);weights.push(.3,.7,0,0);}
  for(let i=0;i+2<count;i+=3)indices.push(i,i+1,i+2);
  return {profile:'weft-avatar-render-v1',nodes:[{index:0,parent:null,local_matrix:id()},{index:1,parent:0,local_matrix:id()},{index:2,parent:0,local_matrix:math.transform([2,0,0])}],
    skins:[{joints:[0,1],inverse_bind_matrices:[id(),id()]}],
    primitives:[{node:0,skin:0,positions,normals,joints,weights,indices,texcoords:null,material:0},
      {node:2,skin:null,positions:[0,0,0,.1,0,0,0,.1,0],normals:[0,0,1,0,0,1,0,0,1],indices:[0,1,2],joints:null,weights:null,texcoords:null,material:0}],
    materials:[{base_color_factor:[.2,.7,.3,1],base_color_texture:null,unlit:true,alpha_mode:'OPAQUE',alpha_cutoff:.5,double_sided:false}],textures:[],
    retargeting:{profile:'weft-avatar-retarget-v1',root_node:0,scale:1,orientation_xyzw:[0,0,0,1],bindings:[{node:0,body:'mobile_root',translation:[0,0,0],rotation_xyzw:[0,0,0,1]},{node:1,body:'left_shoulder_body',translation:[0,0,0],rotation_xyzw:[0,0,0,1]}]}};
}
const corners=b=>Array.from({length:8},(_,m)=>[0,1,2].map(i=>m&(1<<i)?b.max[i]:b.min[i]));
const contained=(b,p)=>p.forEach((v,i)=>assert.ok(v>=b.min[i]-1e-5&&v<=b.max[i]+1e-5,JSON.stringify({b,p})));
const config={bounds:[2,2,2]},snapshot={objects:[]};
const original=fixture(120),before=JSON.stringify(original),asset=math.prepareAsset(original),cached=asset.framing;
let checked=0;
for(const scale of [.2,1,3])for(let k=0;k<9;k++){
  const frames={mobile_root:math.multiply(math.transform([k-4,.3*k,scale],undefined,scale),math.axisRotation([1,0,0],k*.3)),left_shoulder_body:math.multiply(math.transform([k-2,-.2*k,3]),math.axisRotation([0,1,0],k*.4))};
  const pose=math.retarget(asset,{frames}),bounds=math.posedBounds(asset,pose);
  assert.equal(asset.framing,cached);assert.equal(bounds.corners,24);
  for(const primitive of math.posedGeometry(asset,pose))for(let i=0;i<primitive.positions.length;i+=3){contained(bounds,primitive.positions.subarray(i,i+3));checked++;}
  for(const [width,height] of [[1080,420],[303,600]])for(const follow of [false,true]){
    const camera={yaw:k*.4,zoom:1,follow};const views=math.cameraViews(config,snapshot,camera,width,height,bounds);
    for(const view of views)for(const corner of corners(bounds)){const clip=math.point(view.matrix,corner);assert.ok(Math.abs(clip[0])<=.841&&Math.abs(clip[1])<=.841&&Math.abs(clip[2])<1);}
    const zoomed=math.cameraViews(config,snapshot,{...camera,zoom:2},width,height,bounds);
    for(let i=0;i<3;i++)assert.ok(Math.abs(zoomed[i].matrix[0]-2*views[i].matrix[0])<1e-9);
  }
}
assert.equal(JSON.stringify(original),before);
assert.deepEqual(math.cameraViews(config,snapshot),math.cameraViews(config,snapshot,{},1080,420,null));
assert.throws(()=>math.cameraViews(config,snapshot,{},1080,420,{min:[0,0,NaN],max:[1,1,1]}));
assert.throws(()=>math.cameraViews(config,snapshot,{},1080,420,{min:[1,0,0],max:[0,1,1]}));
const world={meshes:[{id:'equipment',framing:corners({min:[-2,-1,-.5],max:[2,1,.5]})},{id:null}]};
const pose=math.retarget(asset,{frames:{mobile_root:id(),left_shoulder_body:id()}});
const withWorld=math.posedBounds(asset,pose,world,[math.transform([10,0,0]),id()]);contained(withWorld,[12,1,.5]);
assert.throws(()=>math.posedBounds(asset,pose,world,[]));
const bodyBounds=math.posedBounds(asset,pose,null,null,{human:{segments:[{a:[-10,0,0],b:[-8,0,0],radius:.1}]}});contained(bodyBounds,[-10.1,0,0]);
const badPose={...pose,palette:[...pose.palette]};badPose.palette[0]=id();badPose.palette[0][15]=-1;assert.throws(()=>math.posedBounds(asset,badPose),/projective/);
const largeRaw=fixture(19995),t0=performance.now(),large=math.prepareAsset(largeRaw),prepareMs=performance.now()-t0;
const largePose=math.retarget(large,{frames:{mobile_root:math.transform([4,2,3]),left_shoulder_body:math.axisRotation([0,0,1],.7)}});
const timed=(count,fn)=>{const t=performance.now();for(let i=0;i<count;i++)fn();return (performance.now()-t)/count;};
for(let i=0;i<10;i++)math.posedBounds(large,largePose);
const oldCameraMs=timed(1000,()=>math.cameraViews(config,snapshot));
const boundedCameraMs=timed(1000,()=>math.cameraViews(config,snapshot,{},1080,420,math.posedBounds(large,largePose)));
const fullSkinMs=timed(12,()=>math.posedGeometry(large,largePose));
const result={status:'passed',groups:6,analytic_pose_cases:27,actual_vertices_checked:checked,large_vertices:large.stats.vertices,per_pose_bound_corners:math.posedBounds(large,largePose).corners,
  timing:{prepare_ms:prepareMs,old_camera_ms:oldCameraMs,bounded_camera_ms:boundedCameraMs,full_cpu_skin_ms:fullSkinMs},
  scope:'One bounded Node measurement, not browser/GPU latency. Geometry/pose preserved; fitting is camera-only and zoom remains deliberate.'};
console.log(JSON.stringify(result));if(process.argv[2])writeFileSync(process.argv[2],JSON.stringify(result,null,2)+'\n',{flag:'wx'});
