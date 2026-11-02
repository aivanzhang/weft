import contextlib, hashlib, importlib.util, io, json, math, platform, sys, warnings
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets.three_d import _GLB, MAX_TRIANGLE_INDICES
from avatar_assets.safety import write_json, sha256
prepared=json.loads((OUT/'prepared.json').read_text())
assert all(sha256(ROOT/r['path'])==r['sha256'] for r in prepared['inputs'])
JOB=ROOT/prepared['job']
cfg=json.loads((JOB/'model-1/finish-config.json').read_text())['authored_rig']
inspection=json.loads((JOB/'inspection/native-inspection.json').read_text())
g=_GLB(JOB/'inputs/source-model.glb')
assert len(g.doc['nodes'])==1 and len(g.doc['meshes'])==1 and not g.doc.get('skins')
node=g.doc['nodes'][0]
assert node=={'mesh':0,'matrix':[1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.]}
assert len(g.doc['meshes'][0]['primitives'])==1
p=g.doc['meshes'][0]['primitives'][0]
assert p.get('mode',4)==4
positions=np.asarray(g.accessor(p['attributes']['POSITION'],'VEC3',(5126,)),dtype=np.float64)
indices=np.asarray(g.accessor(p['indices'],'SCALAR',(5121,5123,5125),count_limit=MAX_TRIANGLE_INDICES),dtype=np.int64).reshape(-1)
used=np.unique(indices)
assert np.array_equal(used,np.arange(len(positions)))
# Installed importer takes sorted used indices without welding by default.
points=positions[:,[0,2,1]].copy();points[:,1]*=-1
low,high=points.min(axis=0),points.max(axis=0)
assert np.array_equal(low,inspection['bounds_blender_z_up']['min'])
assert np.array_equal(high,inspection['bounds_blender_z_up']['max'])
assert len(points)==inspection['meshes'][0]['vertices']==26957
assert len(indices)//3==inspection['meshes'][0]['faces']
factor=cfg['height_meters']/(high[2]-low[2]);offset=np.array([(low[0]+high[0])/2,(low[1]+high[1])/2,low[2]])
points=(points-offset)*factor
lo,hi=points.min(axis=0),points.max(axis=0);span=hi-lo
normalized=(points-lo)/span
layers=[]
for i,layer in enumerate(cfg['layers']):
 edge=np.minimum(normalized-np.asarray(layer['min']),np.asarray(layer['max'])-normalized).min(axis=1)
 alpha=np.clip(edge/layer['feather'],0,1) if layer['feather'] else (edge>=0).astype(float)
 alpha=alpha*alpha*(3-2*alpha)
 selected=np.flatnonzero(alpha>0)
 near=np.count_nonzero(np.abs(edge)<=1e-7)
 layers.append({'index':i,'bones':layer['bones'],'selected_vertices':int(len(selected)),'required_min_vertices':layer['min_vertices'],'passes_minimum':len(selected)>=layer['min_vertices'],'shortfall':max(0,layer['min_vertices']-len(selected)),'selected_indices_sha256':hashlib.sha256(selected.astype('<u4').tobytes()).hexdigest(),'edge_near_zero_within_1e_7_vertices':int(near),'min_edge':float(edge.min()),'max_edge':float(edge.max())})
spec=importlib.util.spec_from_file_location('saved_creature_math',OUT/'sources/creature_rig_math.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with warnings.catch_warnings(record=True) as caught:
 warnings.simplefilter('always',RuntimeWarning)
 try:
  m.skin_weights(points,lo,hi,cfg)
  failure=None
 except ValueError as error:
  failure=str(error)
original_warnings=[{'category':w.category.__name__,'message':str(w.message),'line':w.lineno} for w in caught]
heads=np.array([b['head'] for b in cfg['bones']])*span+lo
tails=np.array([b['tail'] for b in cfg['bones']])*span+lo
bone_metrics=[]
for bone,head,tail in zip(cfg['bones'],heads,tails):
 delta=points-head;direction=tail-head
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always',RuntimeWarning)
  denominator=float(direction@direction)
  numerator=delta@direction
  fraction=np.clip(numerator/denominator,0,1)
  distance=np.linalg.norm(delta-fraction[:,None]*direction,axis=1)
 scalar_denominator=math.fsum(float(x)*float(x) for x in direction)
 scalar_numerator=delta[:,0]*direction[0]+delta[:,1]*direction[1]+delta[:,2]*direction[2]
 scalar_fraction=np.clip(scalar_numerator/scalar_denominator,0,1)
 scalar_delta=delta-scalar_fraction[:,None]*direction
 scalar_distance=np.sqrt(np.sum(scalar_delta*scalar_delta,axis=1))
 assert np.isfinite(distance).all() and np.isfinite(scalar_distance).all()
 bone_metrics.append({'bone':bone['name'],'direction_length_squared_m2':denominator,'scalar_denominator_m2':scalar_denominator,'max_abs_dot_error':float(np.max(np.abs(numerator-scalar_numerator))),'max_abs_distance_error_m':float(np.max(np.abs(distance-scalar_distance))),'warnings':[str(w.message) for w in ws]})
config_text=io.StringIO()
with contextlib.redirect_stdout(config_text):np.show_config()
(OUT/'numpy-configuration.txt').write_text(config_text.getvalue())
assert failure=='Authored weight layer missed expected geometry: hip_l,leg_l_upper,leg_l_lower,foot_l'
assert layers[4]['passes_minimum'] is False
unchanged=all(sha256(ROOT/r['path'])==r['sha256'] for r in prepared['inputs'])
assert unchanged
report={'schema_version':1,'status':'diagnosed','source_sha256':g.hash,'job':prepared['job'],'scope':'CPU reproduction of exact saved source vertices and unchanged authored rig; no native or texture decode','correspondence':{'single_identity_node':True,'all_source_position_rows_indexed':True,'merge_vertices_default':False,'native_count_and_exact_bounds_match':True,'vertices':len(points),'triangles':len(indices)//3,'axis_conversion':'glTF (x,y,z) to Blender (x,-z,y)','calibration_matches_saved_create_rig':True,'native_point_snapshot_available':False,'point_array_sha256':hashlib.sha256(points.astype('<f8').tobytes()).hexdigest()},'layers':layers,'saved_kernel_error':failure,'saved_kernel_warnings':original_warnings,'distance_math':{'all_finite':True,'bones':bone_metrics,'comparison':'same float64 points, BLAS matmul versus explicit elementwise sum; no configuration thresholds changed'},'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),'native_log_version':inspection['blender_version'],'native_numpy_backend_not_measured':True},'inputs_unchanged':unchanged,'conclusions':['The observed stopping condition is an actual authored box coverage shortfall, independent of bone-distance matmul.','A CPU distance comparison does not measure Blender-specific floating-point flags or establish that the native warnings were harmless; original native outputs remain failed.','No replacement rig, modified thresholds or successful animation is produced.']}
write_json(OUT/'diagnosis.json',report)
print(json.dumps({'status':report['status'],'layers':[{'index':r['index'],'count':r['selected_vertices'],'minimum':r['required_min_vertices']} for r in layers],'warnings':len(original_warnings),'max_distance_error_m':max(r['max_abs_distance_error_m'] for r in bone_metrics),'min_denominator_m2':min(r['direction_length_squared_m2'] for r in bone_metrics),'inputs_unchanged':unchanged}))
