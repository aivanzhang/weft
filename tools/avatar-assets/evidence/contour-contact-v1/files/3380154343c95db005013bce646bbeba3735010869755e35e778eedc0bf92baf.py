"""Four controlled surface views of a hash-bound authored scene; no model edits."""
import hashlib,json,sys
from pathlib import Path
import bpy
from mathutils import Matrix
sys.path.insert(0,str(Path(__file__).parent/'sources'))
from blender_verify import bind_action,clear_actions,hash_file,imported_bindings,set_frame,write_json,require

args=sys.argv[sys.argv.index('--')+1:]
require(len(args)==2,'Require frozen config and new output')
config_path,out=Path(args[0]),Path(args[1]);cfg=json.loads(config_path.read_text())
scene_path=Path(cfg['scene']);require(scene_path.is_file() and not scene_path.is_symlink()
 and scene_path.stat().st_size<=128*1024**2 and hash_file(scene_path)==cfg['scene_sha256'],'Scene changed or oversized')
require(out.is_dir() and not out.is_symlink(),'Output directory missing')
bpy.ops.wm.open_mainfile(filepath=str(scene_path),use_scripts=False)
scene=bpy.context.scene
require(scene.render.engine=='BLENDER_EEVEE_NEXT' and scene.render.resolution_x==scene.render.resolution_y==720
 and scene.render.resolution_percentage==100 and scene.eevee.taa_render_samples==16,'Render controls differ')
require(len(bpy.data.objects)<=128 and len(bpy.data.materials)<=64 and len(bpy.data.actions)<=8,'Scene inventory exceeds bounds')
rigs=[o for o in scene.objects if o.type=='ARMATURE'];require(len(rigs)==1,'One rig required')
actions={a.name:a for a in bpy.data.actions};require(set(actions)=={cfg['action']},'Action differs')
bindings,owners=imported_bindings(list(scene.objects),list(actions.values()))
bind_action(actions[cfg['action']],bindings,owners,rigs);rigs[0].data.pose_position='POSE';set_frame(cfg['second']*24)
coat=bpy.data.objects.get(cfg['coat_object']);require(coat is not None and coat.type=='MESH'
 and [s.material.name for s in coat.material_slots]==[cfg['coat_material']],'Coat selector differs')
selected=set(cfg['body_materials']);require(1<=len(selected)<=16 and len(selected)==len(cfg['body_materials']),'Invalid body selector')
slots=[];found=set()
for obj in scene.objects:
 if obj.type!='MESH' or obj is coat:continue
 for slot in obj.material_slots:
  if slot.material and slot.material.name in selected:
   slots.append((slot,slot.material));found.add(slot.material.name)
require(found==selected and 1<=len(slots)<=32,'Body selector incomplete')
require(not ({cfg['coat_material']}|set(cfg['preserved_materials']))&selected,'Selectors overlap')
for name in cfg['preserved_materials']:require(bpy.data.materials.get(name) is not None,'Preserved material absent')

def identity():
 return {'camera':[[float(x)for x in row]for row in scene.camera.matrix_world],
  'ortho_scale':scene.camera.data.ortho_scale,'frame':scene.frame_current+scene.frame_subframe,
  'bones':{b.name:[float(x)for row in b.matrix for x in row]for b in rigs[0].pose.bones}}

initial=identity();gray=bpy.data.materials.new('DiagnosticConstantGray');gray.use_nodes=True
shader=gray.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(.35,.35,.35,1)
shader.inputs['Roughness'].default_value=1;shader.inputs['Metallic'].default_value=0
unlit={}
for _,material in slots:
 if material.name in unlit:continue
 copy=material.copy();copy.name='DiagnosticUnlit_'+material.name
 shaders=[n for n in copy.node_tree.nodes if n.type=='BSDF_PRINCIPLED'];require(len(shaders)==1,'Need one original body shader')
 outputs=[n for n in copy.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output]
 require(len(outputs)==1,'Need one active body material output')
 color=shaders[0].inputs['Base Color'];emission=copy.node_tree.nodes.new('ShaderNodeEmission')
 emission.inputs['Color'].default_value=color.default_value
 if color.links:
  require(len(color.links)==1,'Ambiguous basecolor route');copy.node_tree.links.new(color.links[0].from_socket,emission.inputs['Color'])
 emission.inputs['Strength'].default_value=1
 copy.node_tree.links.new(emission.outputs['Emission'],outputs[0].inputs['Surface']);unlit[material.name]=copy
rows=[]
for mode in ('full','coat_hidden','constant_gray','unlit_basecolor'):
 coat.hide_render=mode!='full'
 for slot,original in slots:slot.material=gray if mode=='constant_gray' else unlit[original.name] if mode=='unlit_basecolor' else original
 bpy.context.view_layer.update();require(identity()==initial,'Pose/camera changed during ablation')
 path=out/(mode+'.png');require(not path.exists() and not path.is_symlink(),'Do not overwrite images')
 scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
 rows.append({'mode':mode,'path':path.name,'sha256':hash_file(path),'coat_visible':not coat.hide_render,
  'scope':'Gray retains custom normals/shadows; it is not geometry-only' if mode=='constant_gray' else
          'Original basecolor route without lighting/normal maps on selected body materials' if mode=='unlit_basecolor' else mode})
for slot,original in slots:slot.material=original
coat.hide_render=False
require(hash_file(scene_path)==cfg['scene_sha256'],'Source scene changed')
write_json(out/'diagnostics.json',{'schema_version':1,'status':'controlled_surface_diagnostics_not_acceptance',
 'scene_sha256':cfg['scene_sha256'],'config_sha256':hash_file(config_path),'pose_camera':initial,'renders':rows,
 'body_materials':cfg['body_materials'],'preserved_materials':cfg['preserved_materials'],'geometry_edited':False,
 'source_unchanged':True,'resolution':[720,720],'samples':16,'render_engine':scene.render.engine,'quality_approved':False})
