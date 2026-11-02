"""Restore only source-bound cheek corner normals for a matched gray diagnostic."""
import hashlib,json,sys
from pathlib import Path
import bpy
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'sources'))
from blender_verify import require,hash_file,write_json,imported_bindings,bind_action,set_frame
args=sys.argv[sys.argv.index('--')+1:];require(len(args)==2,'Require frozen config and output')
cfg=json.loads(Path(args[0]).read_text());out=Path(args[1]);source=Path(cfg['scene']);payload_path=Path(cfg['payload'])
require(hash_file(source)==cfg['scene_sha256'] and hash_file(payload_path)==cfg['payload_sha256'],'Frozen inputs changed')
payload=json.loads(payload_path.read_text());require(payload['face_count']==198 and len(payload['faces'])==198,'Source face count differs')
bpy.ops.wm.open_mainfile(filepath=str(source),use_scripts=False);scene=bpy.context.scene
require(scene.render.engine=='BLENDER_EEVEE_NEXT' and scene.eevee.taa_render_samples==16 and scene.render.resolution_x==scene.render.resolution_y==720,'Render settings differ')
rigs=[o for o in scene.objects if o.type=='ARMATURE'];actions={a.name:a for a in bpy.data.actions};bindings,owners=imported_bindings(list(scene.objects),list(actions.values()))
bind_action(actions[cfg['action']],bindings,owners,rigs)
for rig in rigs:rig.data.pose_position='POSE'
set_frame(cfg['second']*24)
obj=bpy.data.objects.get(cfg['body_object']);require(obj is not None and obj.type=='MESH','Body missing');mesh=obj.data
require(len(mesh.vertices)<=100000 and len(mesh.loops)<=300000,'Mesh exceeds bounds')
mesh.calc_loop_triangles();triangles=[t for t in mesh.loop_triangles if obj.data.materials[t.material_index].name==cfg['target_material']]
positions=np.array([[tuple(mesh.vertices[i].co)for i in t.vertices]for t in triangles],dtype=np.float64)
uvs=np.array([[tuple(mesh.uv_layers[0].data[i].uv)for i in t.loops]for t in triangles],dtype=np.float64)
old=np.asarray([tuple(n.vector)for n in mesh.corner_normals],dtype=np.float32);changed=old.copy();selected=set();mapping=[]
for face in payload['faces']:
 p=np.asarray(face['rest_positions_blender_xyz_m']);uv=np.asarray(face['uv0_blender_v_up']);matches=[]
 for rotation in range(3):
  errors=np.max(np.abs(np.roll(positions,-rotation,axis=1)-p),axis=(1,2));uv_errors=np.max(np.abs(np.roll(uvs,-rotation,axis=1)-uv),axis=(1,2))
  matches.extend((int(i),rotation)for i in np.flatnonzero((errors<=1e-6)&(uv_errors<=2e-6)))
 require(len(matches)==1,'Missing/ambiguous exact native face correspondence')
 index,rotation=matches[0];loops=np.roll(np.asarray(triangles[index].loops),-rotation)
 require(not selected.intersection(map(int,loops)),'Overlapping target corners');selected.update(map(int,loops))
 values=np.asarray(face['original_v7_normal_blender_xyz'],dtype=np.float32);require(values.shape==(3,3)and np.isfinite(values).all(),'Invalid source normals')
 changed[loops]=values;mapping.append({'target_primitive':face['target_primitive'],'target_triangle':face['target_triangle'],'native_polygon':triangles[index].polygon_index,'native_loops':loops.tolist()})
require(len(selected)==594,'Corner selection differs')

def geometry_hash():
 points=np.empty((len(mesh.vertices),3),dtype=np.float32);mesh.vertices.foreach_get('co',points.ravel())
 layer=np.empty((len(mesh.loops),2),dtype=np.float32);mesh.uv_layers[0].data.foreach_get('uv',layer.ravel())
 return {'positions':hashlib.sha256(points.tobytes()).hexdigest(),'uv0':hashlib.sha256(layer.tobytes()).hexdigest()}

before=geometry_hash();gray=bpy.data.materials.new('DiagnosticConstantGray');gray.use_nodes=True;shader=gray.node_tree.nodes.get('Principled BSDF')
shader.inputs['Base Color'].default_value=(.35,.35,.35,1);shader.inputs['Roughness'].default_value=1;shader.inputs['Metallic'].default_value=0
for slot in obj.material_slots:
 if slot.material.name in cfg['body_materials']:slot.material=gray
bpy.data.objects[cfg['coat_object']].hide_render=True
scene.render.filepath=str(out/'baseline_gray.png');bpy.ops.render.render(write_still=True)
attribute=mesh.attributes.get('custom_normal');require(attribute is not None,'Imported custom normals missing')
prior_type=attribute.data_type;mesh.attributes.remove(attribute)
attribute=mesh.attributes.new(name='custom_normal',type='FLOAT_VECTOR',domain='CORNER');attribute.data.foreach_set('vector',changed.ravel());mesh.update();bpy.context.view_layer.update()
stored=np.asarray([tuple(n.vector)for n in mesh.corner_normals],dtype=np.float32);mask=np.ones(len(old),dtype=bool);mask[list(selected)]=False
outside=float(np.abs(stored[mask]-old[mask]).max());target=float(np.abs(stored[list(selected)]-changed[list(selected)]).max())
require(outside<=2e-6 and target<=2e-6 and geometry_hash()==before,'Normal-only mutation proof failed')
scene.render.filepath=str(out/'source_normals_gray.png');bpy.ops.render.render(write_still=True)
require(hash_file(source)==cfg['scene_sha256'] and hash_file(payload_path)==cfg['payload_sha256'],'Sources changed')
write_json(out/'normal-ablation.json',{'schema_version':1,'status':'diagnostic_not_quality_approved','source_scene_sha256':cfg['scene_sha256'],'payload_sha256':cfg['payload_sha256'],'selected_faces':198,'selected_corners':594,'mapping':mapping,'prior_storage':prior_type,'new_storage':'FLOAT_VECTOR CORNER','nonselected_corner_normal_max_error':outside,'selected_source_normal_max_error':target,'geometry_uv_before_after_exact':before,'source_files_unchanged':True,'images':{name:hash_file(out/name)for name in ('baseline_gray.png','source_normals_gray.png')},'scope':'Plain gray on selected body materials; only198source-boundfacecornernormalschanged. Geometry,UV,skin,pose,camera/lightsuntouched. Not a fullmaterialrepair or sourceGLBexport.'})
