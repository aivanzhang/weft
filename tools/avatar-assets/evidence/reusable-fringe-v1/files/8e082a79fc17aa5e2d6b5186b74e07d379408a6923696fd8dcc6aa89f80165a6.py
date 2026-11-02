"""Prepared one-shot actual adapter build; no native groom or rendering."""
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources/scripts'))
from blender_edge_fringe import attach_edge_fringe
from blender_finish_generated import exact_root_texture_contract
from blender_verify import clear_actions, imported_bindings

CONFIG = json.loads((HERE/'config.json').read_text())
OUT = ROOT/CONFIG['output']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def mesh_snapshot(obj):
    mesh = obj.data
    mesh.calc_loop_triangles()
    positions = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get('co', positions.ravel())
    uv_names = [layer.name for layer in mesh.uv_layers]
    # Reacquire by name; CustomData allocation can invalidate retained RNA layer handles.
    layers = {name: np.array([tuple(v.uv) for v in mesh.uv_layers[name].data], dtype='<f4') for name in uv_names}
    influences = [[(obj.vertex_groups[g.group].name, float(g.weight)) for g in vertex.groups] for vertex in mesh.vertices]
    return {'vertices': len(mesh.vertices), 'triangles': len(mesh.loop_triangles),
            'position_sha256': hashlib.sha256(positions.tobytes()).hexdigest(),
            'triangle_sha256': digest([tuple(t.vertices) for t in mesh.loop_triangles]),
            'corner_normal_sha256': digest([tuple(n.vector) for n in mesh.corner_normals]),
            'uv_sha256': {name: hashlib.sha256(array.tobytes()).hexdigest() for name, array in layers.items()},
            'weights_sha256': digest(influences),
            'material_slots': [material.name for material in mesh.materials],
            'face_material_sha256': digest([p.material_index for p in mesh.polygons]),
            'active_uv': mesh.uv_layers.active.name,
            'render_uv': [layer.name for layer in mesh.uv_layers if layer.active_render],
            'matrix_world': [list(row) for row in obj.matrix_world]}


def rig_snapshot(rig, actions):
    bones = [{'name': bone.name, 'parent': bone.parent.name if bone.parent else None,
              'matrix': [list(row) for row in bone.matrix_local], 'head': list(bone.head_local),
              'tail': list(bone.tail_local)} for bone in rig.data.bones]
    records = []
    for action in actions:
        curves = []
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        curves.append({'path': curve.data_path, 'index': curve.array_index,
                                       'points': [[*point.co, *point.handle_left, *point.handle_right,
                                                   point.interpolation, point.easing] for point in curve.keyframe_points]})
        require(curves and sum(len(c['points']) for c in curves) <= 100000, 'Bounded imported action channels required')
        records.append({'name': action.name, 'curves': curves})
    return {'bone_names': [b['name'] for b in bones], 'bones_sha256': digest(bones),
            'action_names': [a['name'] for a in records], 'actions_sha256': digest(records),
            'curve_counts': {a['name']: len(a['curves']) for a in records}}


def main():
    started = time.monotonic()
    proof = json.loads((HERE/'baseline-proof.json').read_text())
    source = HERE/CONFIG['baseline']
    require(sha(source) == proof['baseline_sha256'], 'Baseline hash changed')
    require(OUT.is_dir() and not list(OUT.iterdir()), 'Native output must be an empty new directory')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.threads_mode = 'FIXED'
    bpy.context.scene.render.threads = 4
    save('runtime.json', {'blender': bpy.app.version_string, 'build_hash': bpy.app.build_hash.decode(),
                         'numpy': np.__version__, 'threads': 4, 'rendering_performed': False})
    bpy.ops.import_scene.gltf(filepath=str(source), merge_vertices=False)
    rigs = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']
    helpers = {bone.custom_shape for rig in rigs for bone in rig.pose.bones if bone.custom_shape}
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and obj not in helpers]
    def inventory_object(obj):
        return {'name': obj.name, 'type': obj.type, 'bone_display_shape': obj in helpers,
                'collections': [collection.name for collection in obj.users_collection],
                'hide_render': obj.hide_render,
                'armature_modifiers': [modifier.object.name if modifier.object else None
                                       for modifier in obj.modifiers if modifier.type == 'ARMATURE']}
    save('import-inventory.json', {'context_scene': bpy.context.scene.name,
         'context_scene_objects': [inventory_object(obj) for obj in bpy.context.scene.objects],
         'all_data_objects': [inventory_object(obj) for obj in bpy.data.objects],
         'all_scenes': [{'name': scene.name, 'objects': [obj.name for obj in scene.objects]} for scene in bpy.data.scenes],
         'rigs': [obj.name for obj in rigs], 'bone_display_shapes': sorted(obj.name for obj in helpers),
         'real_mesh_candidates': [obj.name for obj in meshes]})
    require(len(meshes) == len(rigs) == 1, 'Expected one original mesh and one rig')
    obj, rig = meshes[0], rigs[0]
    require(any(modifier.type == 'ARMATURE' and modifier.object == rig for modifier in obj.modifiers),
            'Original emitter must retain its actual source armature modifier')
    actions = sorted(bpy.data.actions, key=lambda action: action.name)
    require([action.name for action in actions] == sorted(CONFIG['expected_actions']), 'Source actions changed on import')
    bindings, owners = imported_bindings(list(bpy.context.scene.objects), actions)
    require(all(bindings[action.name] for action in actions), 'An original action has no actual imported binding')
    clear_actions(owners, [rig])
    rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    names = [material.name for material in obj.data.materials]
    require(set(names) == set(CONFIG['selected_materials']+CONFIG['excluded_retained_materials']), 'Unexpected imported material scope')
    selected_slots = [index for index, name in enumerate(names) if name in CONFIG['selected_materials']]
    images, contracts = {}, []
    for slot in selected_slots:
        shaders = [node for node in obj.data.materials[slot].node_tree.nodes if node.type == 'BSDF_PRINCIPLED']
        require(len(shaders) == 1 and len(shaders[0].inputs['Base Color'].links) == 1, 'Selected source material is not direct texture')
        node = shaders[0].inputs['Base Color'].links[0].from_node
        require(node.type == 'TEX_IMAGE' and node.image, 'Selected source image absent')
        images[slot] = node.image
        contracts.append(exact_root_texture_contract(obj, node.image, [slot]))
    mask = np.zeros(len(obj.data.vertices), dtype=bool)
    for polygon in obj.data.polygons:
        if polygon.material_index in selected_slots:
            mask[list(polygon.vertices)] = True
    before_mesh, before_rig = mesh_snapshot(obj), rig_snapshot(rig, actions)
    require(before_mesh['vertices'] == CONFIG['source_budget']['source_export_vertices']
            and before_mesh['triangles'] == CONFIG['source_budget']['source_export_triangles'], 'Imported base count changed')
    require(len(rig.data.bones) == CONFIG['expected_bones'], 'Imported bone count changed')
    save('native-source-before.json', {'mesh': before_mesh, 'rig': before_rig, 'texture_contracts': contracts,
                                     'final_mask_vertices': int(mask.sum()), 'mask_origin': 'Declared positive source material names only'})
    coats, reports = attach_edge_fringe(obj, rig, mask, images, CONFIG['portable_fringe'],
        CONFIG['native_groom_controls_for_flow_only'], CONFIG['groom_flow'], OUT, CONFIG['source_budget'],
        root_contract=exact_root_texture_contract)
    after_mesh, after_rig = mesh_snapshot(obj), rig_snapshot(rig, actions)
    save('native-source-after.json', {'mesh': after_mesh, 'rig': after_rig})
    require(before_mesh == after_mesh and before_rig == after_rig, 'Adapter changed source geometry, UV, weights, rig or action data')
    for action in actions:
        action.use_fake_user = True
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True); rig.select_set(True)
    for coat in coats:
        coat.select_set(True)
    bpy.context.view_layer.objects.active = rig
    rig.data.pose_position = 'POSE'
    bpy.ops.export_scene.gltf(filepath=str(OUT/'raw-export.glb'), export_format='GLB', use_selection=True,
        export_animations=True, export_animation_mode='ACTIONS', export_anim_single_armature=True,
        export_force_sampling=True, export_frame_range=False, export_yup=True, export_skins=True,
        export_normals=True, export_tangents=True, export_image_format='AUTO', export_morph=False,
        export_extras=False, export_materials='EXPORT', export_current_frame=False)
    rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    final_mesh, final_rig = mesh_snapshot(obj), rig_snapshot(rig, actions)
    save('native-source-after-export.json', {'mesh': final_mesh, 'rig': final_rig})
    require(before_mesh == final_mesh and before_rig == final_rig, 'Exporter changed original native geometry or action data')
    for image in bpy.data.images:
        if image.size[0]:
            image.pack()
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'source.blend'), compress=True)
    save('native-result.json', {'status': 'actual_component_exported_pending_independent_verification',
         'source': CONFIG['source'], 'source_sha256': CONFIG['source_sha256'], 'baseline_sha256': proof['baseline_sha256'],
         'raw_model_sha256': sha(OUT/'raw-export.glb'), 'blend_sha256': sha(OUT/'source.blend'),
         'component_reports': reports, 'native_source_unchanged': True,
         'actual_action_names': [action.name for action in actions], 'actual_bones': len(rig.data.bones),
         'elapsed_seconds': time.monotonic()-started, 'quality_approved': False,
         'native_groom_generated': False, 'rendered_images': 0, 'new_provider_or_ai_role_calls': 0,
         'limits': ['Actual Blender export is reserialization, not an append-spliced preservation claim.',
                    'raw-export.glb is retained before the declared unused non-normalmapped fringe TANGENT omission.',
                    'Exported geometry, weighted poses, skin/clip validity and normals still require the prepared CPU audit.',
                    'Existing authored fox actions are retained; no new action meaning or fresh AI avatar acceptance is claimed.']})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.is_dir() and not (OUT/'native-failure.json').exists():
            save('native-failure.json', {'status': 'failed_preserved_no_retry', 'error': str(error), 'type': type(error).__name__})
        raise
