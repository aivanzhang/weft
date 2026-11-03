"""Static posed-geometry normal control; no source, position, topology, UV or material edits."""
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
def mesh_arrays(mesh):
 assert len(mesh.vertices)<=100000 and len(mesh.polygons)<=100000
 assert all(len(face.vertices)==3 for face in mesh.polygons),'Non-triangle posed body unsupported'
 positions=np.empty(len(mesh.vertices)*3,dtype='<f4');mesh.vertices.foreach_get('co',positions);positions=positions.reshape((-1,3))
 triangles=np.asarray([tuple(face.vertices) for face in mesh.polygons],dtype='<i4')
 uv={layer.name:np.asarray([tuple(v.uv) for v in layer.data],dtype='<f4').tobytes() for layer in mesh.uv_layers}
 return positions,triangles,uv

deps=bpy.context.evaluated_depsgraph_get();evaluated=body.evaluated_get(deps)
posed,triangles,uv_before=mesh_arrays(evaluated.data)
material_names=[material.name if material else None for material in evaluated.data.materials]
material_indices=[face.material_index for face in evaluated.data.polygons]
static=bpy.data.meshes.new_from_object(evaluated,preserve_all_data_layers=True,depsgraph=deps)
static.name='DiagnosticPosedBodyFreshNormals'
new_positions,new_triangles,new_uv=mesh_arrays(static)
assert np.array_equal(new_positions,posed) and np.array_equal(new_triangles,triangles) and new_uv==uv_before
assert [material.name if material else None for material in static.materials]==material_names
assert [face.material_index for face in static.polygons]==material_indices
removed_attributes=[]
for name in ('custom_normal','sharp_edge','sharp_face'):
 attribute=static.attributes.get(name)
 if attribute is not None:static.attributes.remove(attribute);removed_attributes.append(name)
assert not static.has_custom_normals,'Unable to clear inherited custom normals'
for face in static.polygons:face.use_smooth=True
coords=posed.astype(np.float64);corners=coords[triangles]
cross=np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0])
double_area=np.linalg.norm(cross,axis=1)
assert np.isfinite(cross).all() and np.all(double_area>1e-15),'Degenerate posed triangle prevents this diagnostic'
face_normals=cross/double_area[:,None];weighted=np.zeros_like(coords)
for corner in range(3):
 a=corners[:,(corner+1)%3]-corners[:,corner];b=corners[:,(corner+2)%3]-corners[:,corner]
 angle=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b))
 np.add.at(weighted,triangles[:,corner],face_normals*(double_area*angle)[:,None])
length=np.linalg.norm(weighted,axis=1)
assert np.isfinite(weighted).all() and np.all(length>1e-15),'Cancelling/unreferenced posed normal is unsupported'
fresh=(weighted/length[:,None]).astype('<f4')
attribute=static.attributes.new(name='custom_normal',type='FLOAT_VECTOR',domain='POINT')
attribute.data.foreach_set('vector',fresh.ravel());static.update()
body.modifiers.clear();body.data=static;bpy.context.view_layer.update()
after=state_signature()
assert {k:v for k,v in after.items() if k!='posed_body_corner_normals_sha256'}=={k:v for k,v in before.items() if k!='posed_body_corner_normals_sha256'}
final_positions,final_triangles,final_uv=mesh_arrays(body.evaluated_get(bpy.context.evaluated_depsgraph_get()).data)
assert final_positions.tobytes()==posed.tobytes() and final_triangles.tobytes()==triangles.tobytes() and final_uv==uv_before
actual=np.asarray([tuple(n.vector) for n in static.corner_normals],dtype=np.float64)
loop_ids=np.asarray([loop.vertex_index for loop in static.loops],dtype=np.int64)
assert np.max(np.abs(actual-fresh[loop_ids]))<=1e-6,'Fresh normal storage differs'
normal_control={'method':'Per-indexed-vertex sum of unit face normal times face area times corner angle; normalized',
 'coordinate_space':'Evaluated posed body mesh local frame at2seconds','position_sha256':hashlib.sha256(posed.tobytes()).hexdigest(),
 'oriented_triangle_sha256':hashlib.sha256(triangles.tobytes()).hexdigest(),'uv_layer_bytes_preserved':True,
 'all_faces_smooth':True,'source_normal_overrides_removed':removed_attributes,'vertices':len(posed),'triangles':len(triangles),
 'fresh_normals_sha256':hashlib.sha256(fresh.tobytes()).hexdigest(),'source_vertex_correspondence':'Exact index; no welding or cross-seam position matching',
 'static_pose_only':True,'source_skin_or_animation_modified':False}
scene.render.filepath=str(OUT/'posed-smooth-normals.png')
bpy.ops.render.render(write_still=True)
assert state_signature()==after,'Static posed state changed during render' 
assert hash_file(source)==prepared['scene']['sha256'] and hash_file(model)==prepared['model']['sha256']
write_json(OUT/'native-result.json',{'schema_version':1,'status':'completed_diagnostic','state':prepared['state'],'second':prepared['second'],
 'frame':scene.frame_current+scene.frame_subframe,'render_policy':policy,'body_state':after,'original_body_state':before,'normal_control':normal_control,'removed_links':removed,
 'material_preservation':snapshots,'posed_positions_triangles_uvs_materials_camera_lights_equal_to_baseline':True,'source_files_unchanged':True,
 'render':{'path':'posed-smooth-normals.png','sha256':hash_file(OUT/'posed-smooth-normals.png')},
 'elapsed_seconds':time.monotonic()-start,'scope':'One static baked pose with recalculated normals; no exported animation repair or acceptance'})
