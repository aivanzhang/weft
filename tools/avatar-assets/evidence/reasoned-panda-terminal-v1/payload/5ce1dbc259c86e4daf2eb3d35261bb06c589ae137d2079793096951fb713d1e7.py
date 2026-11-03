"""Two diagnostic renders of one saved exported pose; only coat visibility changes."""
from pathlib import Path
import hashlib,json,sys,time
import bpy
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'sources'))
from blender_verify import hash_file,imported_bindings,bind_action,set_frame,write_json
from blender_provider_preview import configure_exported_preview_engine
prepared=json.loads((OUT/'prepared.json').read_text())
source=Path(prepared['scene']['path']);model=Path(prepared['model']['path'])
assert hash_file(source)==prepared['scene']['sha256'] and hash_file(model)==prepared['model']['sha256']
start=time.monotonic()
bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
scene=bpy.context.scene
policy=configure_exported_preview_engine(scene,8)
assert scene.render.resolution_x==scene.render.resolution_y==720 and scene.render.resolution_percentage==100
scene.render.threads_mode='FIXED';scene.render.threads=4
rigs=[x for x in scene.objects if x.type=='ARMATURE'];assert len(rigs)==1
for rig in rigs:rig.data.pose_position='POSE'
actions=list(bpy.data.actions);bindings,owners=imported_bindings(list(scene.objects),actions)
selected=[a for a in actions if a.name==prepared['state']];assert len(selected)==1
bind_action(selected[0],bindings,owners,rigs);set_frame(prepared['second']*24)
coats=[scene.objects.get(name) for name in prepared['coat_nodes']]
assert all(x is not None and x.type=='MESH' and not x.hide_render for x in coats)
body=scene.objects.get(prepared['source_node']);assert body is not None and body.type=='MESH' and not body.hide_render
assert len(body.data.vertices)+sum(len(x.data.vertices) for x in coats)<=100000

def state_signature():
 deps=bpy.context.evaluated_depsgraph_get();evaluated=body.evaluated_get(deps)
 points=np.empty(len(evaluated.data.vertices)*3,dtype='<f4');evaluated.data.vertices.foreach_get('co',points)
 normals=np.empty(len(evaluated.data.loops)*3,dtype='<f4')
 for i,n in enumerate(evaluated.data.corner_normals):normals[i*3:i*3+3]=n.vector
 assert np.isfinite(points).all() and np.isfinite(normals).all()
 return {'body_vertices':len(evaluated.data.vertices),'body_faces':len(evaluated.data.polygons),
         'posed_body_positions_sha256':hashlib.sha256(points.tobytes()).hexdigest(),
         'posed_body_corner_normals_sha256':hashlib.sha256(normals.tobytes()).hexdigest(),
         'camera_matrix':[list(row) for row in scene.camera.matrix_world],
         'camera_ortho_scale':scene.camera.data.ortho_scale,
         'lights':[{'name':x.name,'matrix':[list(row) for row in x.matrix_world],'energy':x.data.energy,'size':x.data.size}
                   for x in sorted(scene.objects,key=lambda x:x.name) if x.type=='LIGHT']}

before=state_signature();renders=[]
for name,hidden in (('with-coat',False),('body-only',True)):
 for obj in coats:obj.hide_render=hidden
 scene.render.filepath=str(OUT/(name+'.png'))
 t=time.monotonic();bpy.ops.render.render(write_still=True)
 assert time.monotonic()-t<=60
 assert state_signature()==before,'Body/camera/light state changed between renders'
 renders.append({'path':name+'.png','sha256':hash_file(OUT/(name+'.png')),'seconds':time.monotonic()-t,
                 'hidden_coat_nodes':prepared['coat_nodes'] if hidden else []})
assert hash_file(source)==prepared['scene']['sha256'] and hash_file(model)==prepared['model']['sha256']
write_json(OUT/'native-result.json',{'schema_version':1,'status':'completed_diagnostic','state':prepared['state'],
 'second':prepared['second'],'frame':scene.frame_current+scene.frame_subframe,'render_policy':policy,
 'same_body_normals_positions_camera_lights':True,'body_state':before,'renders':renders,
 'source_files_unchanged':True,'elapsed_seconds':time.monotonic()-start,
 'scope':'Visibility ablation at one exported pose, not an accepted delivery or root-cause proof of skin versus normal maps'})
