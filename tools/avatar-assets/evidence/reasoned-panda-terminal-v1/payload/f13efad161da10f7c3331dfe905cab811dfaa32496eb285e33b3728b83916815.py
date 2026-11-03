"""One shading-only control; body geometry, custom normals, UVs and skin remain untouched."""
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
start=time.monotonic();bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
scene=bpy.context.scene;policy=configure_exported_preview_engine(scene,8)
assert scene.render.resolution_x==scene.render.resolution_y==720 and scene.render.resolution_percentage==100
scene.render.threads_mode='FIXED';scene.render.threads=4
rigs=[x for x in scene.objects if x.type=='ARMATURE'];assert len(rigs)==1
for rig in rigs:rig.data.pose_position='POSE'
actions=list(bpy.data.actions);bindings,owners=imported_bindings(list(scene.objects),actions)
selected=[a for a in actions if a.name==prepared['state']];assert len(selected)==1
bind_action(selected[0],bindings,owners,rigs);set_frame(prepared['second']*24)
coats=[scene.objects.get(name) for name in prepared['coat_nodes']]
assert all(x is not None and x.type=='MESH' for x in coats)
for coat in coats:coat.hide_render=True
body=scene.objects.get(prepared['source_node']);assert body is not None and body.type=='MESH' and not body.hide_render

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

def links(material):
 return sorted([link.from_node.name,link.from_socket.identifier,link.to_node.name,link.to_socket.identifier]
               for link in material.node_tree.links)

def socket_values(material):
 result=[]
 for node in material.node_tree.nodes:
  for socket in node.inputs:
   if not hasattr(socket,'default_value'):continue
   value=socket.default_value
   if hasattr(value,'__len__') and not isinstance(value,str):value=list(value)
   if isinstance(value,(int,float,str,bool,list)):result.append([node.name,socket.identifier,value])
 return result

before=state_signature();reference=json.loads(Path(prepared['prior_native_result']['path']).read_text())
assert hash_file(Path(prepared['prior_native_result']['path']))==prepared['prior_native_result']['sha256']
assert before==reference['body_state'],'Saved body/camera/light state no longer matches baseline'
materials=list(dict.fromkeys(slot.material for slot in body.material_slots));assert 1<=len(materials)<=8 and all(materials)
removed=[];snapshots=[]
for material in materials:
 assert material.use_nodes and len(material.node_tree.nodes)<=128
 bsdfs=[n for n in material.node_tree.nodes if n.bl_idname=='ShaderNodeBsdfPrincipled']
 assert len(bsdfs)==1,'Unsupported body shader graph'
 normal=bsdfs[0].inputs['Normal'];incoming=list(normal.links)
 assert len(incoming)==1 and incoming[0].from_node.bl_idname=='ShaderNodeNormalMap','Unsupported normal-texture route'
 row=[incoming[0].from_node.name,incoming[0].from_socket.identifier,incoming[0].to_node.name,incoming[0].to_socket.identifier]
 old_links=links(material);old_values=socket_values(material)
 material.node_tree.links.remove(incoming[0])
 expected=list(old_links);expected.remove(row)
 assert links(material)==expected and socket_values(material)==old_values,'Non-normal shader data changed'
 removed.append({'material':material.name,'removed_link':row,'normal_map_strength':bpy.data.materials[material.name].node_tree.nodes[row[0]].inputs['Strength'].default_value})
 snapshots.append({'material':material.name,'all_socket_values_unchanged':True,
                   'other_links_unchanged':True,'before_links':old_links,'after_links':expected})
scene.render.filepath=str(OUT/'body-no-normal-map.png')
bpy.ops.render.render(write_still=True)
assert state_signature()==before,'Body/custom normals/camera/lights changed'
assert hash_file(source)==prepared['scene']['sha256'] and hash_file(model)==prepared['model']['sha256']
write_json(OUT/'native-result.json',{'schema_version':1,'status':'completed_diagnostic','state':prepared['state'],'second':prepared['second'],
 'frame':scene.frame_current+scene.frame_subframe,'render_policy':policy,'body_state':before,'removed_links':removed,
 'material_preservation':snapshots,'body_normals_geometry_camera_lights_equal_to_baseline':True,'source_files_unchanged':True,
 'render':{'path':'body-no-normal-map.png','sha256':hash_file(OUT/'body-no-normal-map.png')},
 'elapsed_seconds':time.monotonic()-start,'scope':'Single posed-body normal-map shading ablation; not material or animation acceptance'})
