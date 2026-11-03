"""One data-only GLB reduction; original files and armature rest pose remain untouched."""
import json
from pathlib import Path
import sys
import time
import bpy
import bmesh
sys.path.insert(0,str(Path(__file__).parent))
from weft_delivery_mesh import component_plan

source,out,budget=sys.argv[sys.argv.index('--')+1:]
source,out,budget=Path(source),Path(out),int(budget)
started=time.monotonic()
def check():
    if time.monotonic()-started>100:raise TimeoutError('Delivery native work exceeded100seconds')
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(source),import_pack_images=True)
rigs=[o for o in bpy.context.scene.objects if o.type=='ARMATURE']
assert len(rigs)==1,'Require one retained semantic armature'
rig=rigs[0];helpers={b.custom_shape for b in rig.pose.bones if b.custom_shape is not None}
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH' and o not in helpers]
assert 1<=len(meshes)<=16 and all(len(o.vertex_groups)>0 for o in meshes),'Original source meshes must retain skin'
rig.data.pose_position='REST'
for o in bpy.context.scene.objects:
    o.animation_data_clear()
for bone in rig.pose.bones:bone.matrix_basis.identity()
counts=[]
for obj in meshes:
    check();obj.data.calc_loop_triangles();counts.append(len(obj.data.loop_triangles))
assert sum(counts)<=100000 and len(rig.data.bones)<=64
rows=[]
for obj,count in zip(meshes,counts):
    check();target=max(4,int(budget*count/sum(counts)))
    bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj
    plan=component_plan(len(obj.data.vertices),[list(p.vertices) for p in obj.data.polygons],target)
    if count>target and plan['method']=='whole_component_thinning':
        keep=set(plan.pop('keep_vertices'));bm=bmesh.new();bm.from_mesh(obj.data);bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm,geom=[v for v in bm.verts if v.index not in keep],context='VERTS')
        bm.to_mesh(obj.data);bm.free();obj.data.update()
    elif count>target:
        modifier=obj.modifiers.new('WeftDeliveryExplicitDecimation','DECIMATE');modifier.ratio=target/count;modifier.use_collapse_triangulate=True
        # Collapse original rest geometry, never the evaluated armature deformation.
        obj.modifiers.move(len(obj.modifiers)-1,0)
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    plan.pop('keep_vertices',None)
    obj.data.calc_loop_triangles();after=len(obj.data.loop_triangles)
    assert after<=target+2,'Native decimation exceeded declared per-mesh bound'
    rows.append({'name':obj.name,'original_triangles':count,'target_triangles':target,'derived_triangles':after,'vertices':len(obj.data.vertices),'reduction':plan})
    assert len(obj.vertex_groups)>0,'Skin weights must survive reduction'
check();assert sum(r['derived_triangles'] for r in rows)<=6500
# Imported bone-display shapes are not source geometry and must not enter the delivery.
bpy.ops.object.select_all(action='DESELECT')
for obj in [rig,*meshes]:obj.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(out/'decimated.glb'),export_format='GLB',export_animations=False,
    use_selection=True,export_skins=True,export_all_influences=False,export_def_bones=False,export_tangents=False,
    export_attributes=False,export_yup=True,export_image_format='AUTO',export_extras=False)
check()
(out/'native-report.json').write_text(json.dumps({'profile':'weft-delivery-native-v1','meshes':rows,'joint_names':[b.name for b in rig.data.bones],
    'excluded_importer_bone_display_helpers':sorted(o.name for o in helpers),'source_animations_exported':False,'armature_pose':'REST','source_blend_executed':False,'seconds':time.monotonic()-started},indent=2)+'\n')
