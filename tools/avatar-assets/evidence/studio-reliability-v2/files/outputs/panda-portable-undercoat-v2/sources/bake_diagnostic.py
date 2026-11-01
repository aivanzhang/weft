"""One diagnostic bake; source GLB geometry is never exported from Blender."""
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

sys.path.insert(0,str(Path(__file__).parent))
from blender_fur_undercoat import apply_undercoat
from blender_verify import source_header, hash_file, write_json
from groom_flow import prepare_flow


def checked(path, expected, limit=512*1024*1024):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit or hash_file(path)!=expected:
        raise ValueError('Diagnostic input hash/size mismatch')
    return path


def run(config_path, out):
    cfg=json.loads(Path(config_path).read_text());out=Path(out)
    model=checked(cfg['source_model'],cfg['source_model_sha256'],64*1024*1024)
    blend=checked(cfg['mask_source_blend'],cfg['mask_source_blend_sha256'])
    report_path=checked(cfg['mask_source_report'],cfg['mask_source_report_sha256'],1024*1024)
    design_path=checked(cfg['design'],cfg['design_sha256'],1024*1024)
    source_report=json.loads(report_path.read_text());design=json.loads(design_path.read_text())
    source_header(model)
    bpy.ops.wm.open_mainfile(filepath=str(blend),load_ui=False,use_scripts=False)
    for rig in (obj for obj in bpy.context.scene.objects if obj.type=='ARMATURE'):
        rig.data.pose_position='REST'
    bpy.context.view_layer.update()
    candidates=[obj for obj in bpy.context.scene.objects if obj.type=='MESH' and
                obj.vertex_groups.get('AuthoredFurDensity') and len(obj.particle_systems)==1]
    if len(candidates)!=1:
        raise ValueError('Exactly one measured fur-density witness is required')
    witness=candidates[0]
    if len(witness.data.vertices)>100000 or len(witness.data.loops)>300000:
        raise ValueError('Measured mask witness exceeds bounds')
    group=witness.vertex_groups['AuthoredFurDensity'].index
    flags=np.asarray([any(g.group==group and g.weight>.5 for g in vertex.groups)
                      for vertex in witness.data.vertices],dtype=bool)
    if int(flags.sum())!=source_report['groom']['density_selected_vertices']:
        raise ValueError('Measured density differs from the accepted finish report')
    points=np.asarray([tuple(witness.matrix_world@vertex.co) for vertex in witness.data.vertices],dtype=np.float64)
    tree=KDTree(len(points))
    for index,point in enumerate(points): tree.insert(point,index)
    tree.balance()
    if witness.data.uv_layers.active is None:
        raise ValueError('Measured witness has no original UV0')
    corner_witness=[[] for _ in points]
    for face in witness.data.polygons:
        material=witness.data.materials[face.material_index].name
        for loop_index in face.loop_indices:
            loop=witness.data.loops[loop_index]
            uv=tuple(witness.data.uv_layers.active.data[loop_index].uv)
            if (material,uv) not in corner_witness[loop.vertex_index]:
                corner_witness[loop.vertex_index].append((material,uv))
    mask_hash=hashlib.sha256(np.packbits(flags,bitorder='little').tobytes()).hexdigest()
    witness_count=len(flags)
    # Only compact measured arrays survive; saved scripts/handlers are never executed.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(model),merge_vertices=False)
    meshes=[obj for obj in bpy.context.scene.objects if obj.type=='MESH' and obj.data.name in cfg['mesh_names']]
    bodies=[obj for obj in meshes if obj.data.name==cfg['body_mesh_name']]
    if len(meshes)!=len(cfg['mesh_names']) or len(bodies)!=1:
        raise ValueError('Imported mesh identities differ from the exact GLB')
    obj=bodies[0]
    for rig in (obj for obj in bpy.context.scene.objects if obj.type=='ARMATURE'):
        rig.data.pose_position='REST'
    bpy.context.view_layer.update()
    world=obj.matrix_world.copy();obj.parent=None;obj.matrix_world=world
    bpy.ops.object.select_all(action='DESELECT');obj.select_set(True);bpy.context.view_layer.objects.active=obj
    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    bpy.context.view_layer.update()
    body_receipt=json.loads((out/'source-body-receipt.json').read_text())
    if body_receipt['source_model_sha256']!=cfg['source_model_sha256']:
        raise ValueError('Source-body correspondence belongs to another model')
    expected=np.load(checked(out/'source-body.npz',body_receipt['sha256'],8*1024*1024),allow_pickle=False)
    actual_points=np.asarray([tuple(vertex.co) for vertex in obj.data.vertices],dtype=np.float32)
    actual_faces=np.asarray([tuple(face.vertices) for face in obj.data.polygons],dtype=np.int32)
    actual_uvs=np.asarray([tuple(obj.data.uv_layers.active.data[loop].uv)
                           for face in obj.data.polygons for loop in face.loop_indices],dtype=np.float32).reshape(-1,3,2)
    shapes = {'points': actual_points.shape == expected['points'].shape,
              'faces': actual_faces.shape == expected['faces'].shape,
              'uvs': actual_uvs.shape == expected['uvs'].shape}
    correspondence = {'shape_matches': shapes,
        'actual_shapes': {'points': list(actual_points.shape), 'faces': list(actual_faces.shape), 'uvs': list(actual_uvs.shape)},
        'expected_shapes': {name: list(expected[name].shape) for name in shapes},
        'position_tolerance_m': 1e-6, 'uv_tolerance': 2e-7,
        'source_glb_sha256': cfg['source_model_sha256'], 'indices_must_match_exactly': True}
    if shapes['points']:
        distances = np.linalg.norm(actual_points.astype(np.float64)-expected['points'],axis=1)
        bad = np.flatnonzero(distances > 1e-6)
        correspondence.update(position_max_distance_m=float(distances.max(initial=0)),
            positions_beyond_1um=int(len(bad)), positions_beyond_old_0_2um=int((distances>2e-7).sum()),
            worst_position_rows=[{'vertex':int(i),'distance_m':float(distances[i]),
                'actual':actual_points[i].tolist(),'expected':expected['points'][i].tolist()}
                for i in np.argsort(distances)[-8:][::-1]])
    if shapes['faces']:
        correspondence['face_index_mismatch_count'] = int(np.count_nonzero(np.any(actual_faces != expected['faces'],axis=1)))
    if shapes['uvs']:
        correspondence['uv_max_component_difference'] = float(np.max(np.abs(actual_uvs-expected['uvs']),initial=0))
        correspondence['uv_exact_array_match'] = bool(np.array_equal(actual_uvs,expected['uvs']))
    write_json(out/'import-correspondence.json',correspondence)
    if (not all(shapes.values()) or correspondence.get('positions_beyond_1um',1) or
            correspondence.get('face_index_mismatch_count',1) or correspondence.get('uv_max_component_difference',1)>2e-7):
        raise ValueError('Native import/source GLB correspondence failed; inspect exact saved mismatch metrics')
    mask=np.zeros(len(actual_points),dtype=bool);seen=np.zeros(len(mask),dtype=bool)
    max_distance,max_uv_distance=0.,0.
    cache={}
    for face in obj.data.polygons:
        material=obj.data.materials[face.material_index].name
        for loop_index in face.loop_indices:
            vertex=obj.data.loops[loop_index].vertex_index
            uv=np.asarray(obj.data.uv_layers.active.data[loop_index].uv)
            key=(vertex,material,float(uv[0]),float(uv[1]))
            if key in cache:
                found=cache[key]
            else:
                candidates=[]
                for point,index,distance in tree.find_range(Vector(actual_points[vertex]),2e-6):
                    for witness_material,witness_uv in corner_witness[index]:
                        uv_distance=float(np.max(np.abs(uv-witness_uv)))
                        if witness_material==material and uv_distance<=2e-6:
                            candidates.append(bool(flags[index]))
                            max_distance=max(max_distance,float(distance));max_uv_distance=max(max_uv_distance,uv_distance)
                if not candidates or len(set(candidates))!=1:
                    raise ValueError('Missing or conflicting measured fur-mask corner witness')
                found=candidates[0];cache[key]=found
            if seen[vertex] and mask[vertex]!=found:
                raise ValueError('Imported vertex combines conflicting measured fur-mask corners')
            seen[vertex]=True;mask[vertex]=found
    if not seen.all(): raise ValueError('Unreferenced imported body vertices lack a measured mask')
    selected=mask[actual_faces].all(axis=1)
    np.save(out/'selected-triangles.npy',selected,allow_pickle=False)
    prepared=prepare_flow(actual_points.min(axis=0),np.ptp(actual_points,axis=0),
                          design['groom']['direction'],design['groom']['comb'],design['groom_flow'])
    result=apply_undercoat(obj,mask,prepared,cfg['settings'],out)
    result.update(kind='undercoat_only_native_bake',status='baked_not_visually_accepted',
        source_glb_sha256=cfg['source_model_sha256'],source_blend_sha256=cfg['mask_source_blend_sha256'],
        design_sha256=cfg['design_sha256'],mask_witness_sha256=mask_hash,
        mask_source_vertices=witness_count,mask_source_selected_vertices=int(flags.sum()),
        imported_mask_vertices=len(mask),imported_mask_selected_vertices=int(mask.sum()),
        mask_max_position_difference_m=max_distance,mask_max_uv_difference=max_uv_distance,
        selected_triangle_mask_sha256=hash_file(out/'selected-triangles.npy'),
        native_triangle_order_verified=True,source_basecolor_uv_skin_animation_changed=False,
        original_portable_fibers_changed=False,blender_version=bpy.app.version_string)
    for key,expected_hash in [('source_model',cfg['source_model_sha256']),('mask_source_blend',cfg['mask_source_blend_sha256'])]:
        if hash_file(Path(cfg[key]))!=expected_hash: raise ValueError('Original diagnostic source changed')
    write_json(out/'native-bake-report.json',result)


if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:]
    run(args[0],args[1])
