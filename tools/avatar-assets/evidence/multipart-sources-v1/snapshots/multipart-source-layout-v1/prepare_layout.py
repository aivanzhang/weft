"""Author a disclosed local test layout from two pinned source bounds; no generation."""
from pathlib import Path
import hashlib,json,math,time,sys
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'outputs/multipart-source-layout-v1';started=time.monotonic()
sys.path.insert(0,str(OUT/'sources'))
from avatar_assets.three_d import _GLB
from avatar_assets.normalize_generated import _array
items=[('robot','workshop-robot-meshy-v1','5f00602dd727429277ae53c0e8940ba9162ea119e643ba554f081c00cc1b7e8c',1.2,[-.8,0,0],0.),('dragon','emerald-dragon-meshy-v1','a5e034586961d2999846d185321e6b1b4289e8138ce4b11a4a19746700109efb',.9,[.85,0,0],-20.)]
components=[];records=[]
for id,folder,digest,height,base,angle in items:
 source=ROOT/'outputs'/folder/'stages/texture/downloads/model.glb';glb=_GLB(source);assert glb.hash==digest
 primitive=glb.doc['meshes'][0]['primitives'][0]
 points=_array(glb,primitive['attributes']['POSITION'],3,5126,100000).astype(float)
 low,high=points.min(0),points.max(0);scale=height/(high[1]-low[1]);theta=math.radians(angle)
 quaternion=[0.,math.sin(theta/2),0.,math.cos(theta/2)]
 rotation=np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
 pivot=(low+high)*.5;pivot[1]=low[1]
 translation=np.asarray(base)-scale*(rotation@pivot)
 transformed=scale*(points@rotation.T)+translation
 components.append({'id':id,'source':'../'+folder+'/stages/texture/downloads/model.glb','source_sha256':digest,'source_meters_per_unit':float(scale),'placement':{'translation_m':translation.tolist(),'rotation_xyzw':quaternion,'uniform_scale':1.}})
 records.append({'id':id,'source_path':str(source.relative_to(ROOT)),'source_sha256':digest,'source_bounds':{'min':low.tolist(),'max':high.tolist()},'authored_height_m':height,'authored_base_center_m':base,'authored_yaw_degrees':angle,'transformed_bounds':{'min':transformed.min(0).tolist(),'max':transformed.max(0).tolist()},'units_measured':False,'placement_is_automatic_identity_inference':False})
 del points,transformed,glb
prompt='Local multipart pipeline development: place the preserved workshop robot and standalone emerald dragon side by side as two complete independently identifiable sources. Keep their existing geometry, textures and materials. Apply only the recorded authored uniform scaling, yaw and translations. This is not the panda shoulder companion, new neural generation, a rigged output or an animation request.'
config={'schema_version':1,'id':'robot-dragon-source-assembly-v1','prompt':prompt,'rights':'Reuse of previously user-authorized Meshy text-generated sources. Original provider prompts, downloads and account terms remain applicable; commercial rights are not independently verified. Hidden geometry and these metric placements are authored assumptions, not measured anatomy or physics. No provider upload or new generation is part of this composition.','space':'gltf_y_up_right_handed','requested_rig_height_m':1.2,'components':components}
for name,data in [('compose.json',config),('layout.json',{'schema_version':1,'status':'prepared_not_executed','components':records,'elapsed_seconds':time.monotonic()-started,'new_provider_calls':0,'new_pipeline_role_calls':0,'native_calls':0,'geometry_modified':False,'description':'Uniform authored layout only; original source bytes are unchanged.'})]:
 with (OUT/name).open('x') as f:json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':'layout_prepared','components':len(records),'seconds':time.monotonic()-started}))
