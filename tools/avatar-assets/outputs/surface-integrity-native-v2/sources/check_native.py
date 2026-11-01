import bpy, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from blender_finish_generated import set_surface_positions
from blender_creature_rig import create_rig
from surface_smoothing import refine_positions
from surface_integrity import SurfaceChangeError
root=Path(sys.argv[sys.argv.index('--')+1])
results=[]
for cfg in json.loads((root/'configs.json').read_text()):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=cfg['source'])
    obj=next(o for o in bpy.context.scene.objects if o.type=='MESH')
    if cfg.get('authored_rig'):
        create_rig(obj,cfg['authored_rig'])
    for rig in [o for o in bpy.context.scene.objects if o.type=='ARMATURE']:
        rig.data.pose_position='REST'
    world=obj.matrix_world.copy()
    obj.parent=None
    obj.matrix_world=world
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active=obj
    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    original=np.array([tuple(v.co) for v in obj.data.vertices])
    edges=np.array([tuple(e.vertices) for e in obj.data.edges])
    refined,smoothing=refine_positions(original,edges,cfg['surface_smoothing'])
    dest=root/cfg['id'];dest.mkdir()
    try:
        report=set_surface_positions(obj,original,refined,dest)
        status='passed'
    except SurfaceChangeError as error:
        report=error.report
        status='rejected_before_render'
    results.append({'id':cfg['id'],'status':status,'source_sha256':cfg['source_sha256'],'report':report})
(root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
