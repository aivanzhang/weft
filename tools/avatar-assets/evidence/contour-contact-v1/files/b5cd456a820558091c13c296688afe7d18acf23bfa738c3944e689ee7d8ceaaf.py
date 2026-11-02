"""Identify actual visible surface hits for frozen diagnostic pixels; no render or edits."""
import json,sys
from pathlib import Path
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).parent/'sources'))
from blender_verify import require,hash_file,write_json,imported_bindings,bind_action,set_frame
args=sys.argv[sys.argv.index('--')+1:];require(len(args)==2,'Require config andoutput')
cfg=json.loads(Path(args[0]).read_text());out=Path(args[1]);scene_path=Path(cfg['scene'])
require(hash_file(scene_path)==cfg['scene_sha256']and scene_path.stat().st_size<128*1024**2,'Scene changed')
bpy.ops.wm.open_mainfile(filepath=str(scene_path),use_scripts=False)
scene=bpy.context.scene;rigs=[o for o in scene.objects if o.type=='ARMATURE'];actions={a.name:a for a in bpy.data.actions}
bindings,owners=imported_bindings(list(scene.objects),list(actions.values()));bind_action(actions[cfg['action']],bindings,owners,rigs)
for rig in rigs:rig.data.pose_position='POSE'
set_frame(cfg['second']*24)
identity={'camera':[[float(x)for x in row]for row in scene.camera.matrix_world],'ortho_scale':scene.camera.data.ortho_scale,'frame':scene.frame_current+scene.frame_subframe,'bones':{b.name:[float(x)for row in b.matrix for x in row]for b in rigs[0].pose.bones}}
require(identity==cfg['pose_camera'],'Diagnostic pose/camera differs')
obj=bpy.data.objects.get(cfg['body_object']);require(obj and obj.type=='MESH','Body mesh missing')
obj.data.calc_loop_triangles();tris=list(obj.data.loop_triangles)
require(len(tris)<=100000 and len(obj.data.vertices)<=100000 and len(obj.data.uv_layers)>=1,'Body bounds/UV missing')
evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
try:
 require(len(mesh.vertices)==len(obj.data.vertices),'Evaluated topology differs')
 points=[evaluated.matrix_world@v.co for v in mesh.vertices];indices=[tuple(t.vertices)for t in tris]
 tree=BVHTree.FromPolygons(points,indices,all_triangles=True)
 camera=scene.camera;basis=camera.matrix_world.to_3x3();forward=basis@Vector((0,0,-1));right=basis@Vector((1,0,0));up=basis@Vector((0,1,0))
 rows=[]
 for sample in cfg['pixels']:
  x,y=sample['pixel'];origin=camera.matrix_world.translation+right*((x+.5)/720-.5)*camera.data.ortho_scale+up*(.5-(y+.5)/720)*camera.data.ortho_scale
  hit,normal,index,distance=tree.ray_cast(origin,forward)
  row={**sample,'hit':index is not None}
  if index is not None:
   tri=tris[index];ids=tuple(tri.vertices);a,b,c=[points[i]for i in ids];v0=b-a;v1=c-a;v2=hit-a
   d00=v0.dot(v0);d01=v0.dot(v1);d11=v1.dot(v1);d20=v2.dot(v0);d21=v2.dot(v1);den=d00*d11-d01*d01
   require(abs(den)>1e-20,'Degenerate hit triangle');v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den
   rest=[obj.matrix_world@obj.data.vertices[i].co for i in ids]
   uv=[list(obj.data.uv_layers[0].data[i].uv)for i in tri.loops]
   row.update(object=obj.name,triangle_index=index,polygon_index=tri.polygon_index,material_slot=tri.material_index,material=obj.data.materials[tri.material_index].name,
    barycentric=[1-v-w,v,w],world_hit=list(hit),distance=distance,normal=list(normal),native_vertex_ids=list(ids),uv0_corners=uv,
    rest_positions_blender_xyz=[list(p)for p in rest],rest_positions_gltf_xyz=[[p.x,p.z,-p.y]for p in rest])
  rows.append(row)
 write_json(out/'attribution.json',{'schema_version':1,'source_scene_sha256':cfg['scene_sha256'],'source_model_sha256':cfg['model_sha256'],'pose_camera':identity,'pixels':rows,'geometry_edited':False,'scope':'Closest visible base-mesh triangle excludingthehiddenfringemesh; body+companionmaterialspartitionremaininthebaseobject. Pixelcenter rays do not modelantialiasing/mipmapcoverage.'})
finally:evaluated.to_mesh_clear()
require(hash_file(scene_path)==cfg['scene_sha256'],'Source changed')
