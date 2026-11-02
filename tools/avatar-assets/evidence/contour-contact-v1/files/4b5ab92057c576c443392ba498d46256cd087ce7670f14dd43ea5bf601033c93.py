"""Matched color-only review of an actual UV repair on the saved posed panda."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'sources'))
from blender_verify import bind_action, hash_file, imported_bindings, require, set_frame, write_json

args = sys.argv[sys.argv.index('--')+1:]
require(len(args) == 1, 'Require fresh output')
out = Path(args[0]); cfg = json.loads((HERE/'config.json').read_text())
scene_cfg = json.loads((HERE/'scene-config.json').read_text())
payload_path = Path(cfg['payload'])
require(hash_file(payload_path) == cfg['payload_sha256'], 'Repair payload changed')
payload = json.loads(payload_path.read_text())
require(len(payload['faces']) == cfg['expected_faces'] <= 1000, 'Repair face count differs')
source = Path(scene_cfg['scene'])
require(hash_file(source) == scene_cfg['scene_sha256'], 'Source scene changed')
bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False)
scene = bpy.context.scene
require(scene.render.engine == 'BLENDER_EEVEE_NEXT' and scene.render.resolution_x == scene.render.resolution_y == 720
        and scene.render.resolution_percentage == 100 and scene.eevee.taa_render_samples == 16, 'Render settings differ')
rigs = [o for o in scene.objects if o.type == 'ARMATURE']
actions = list(bpy.data.actions)
require(len(rigs) == 1 and len(actions) == 1 and actions[0].name == scene_cfg['action'], 'Action ownership differs')
bindings, owners = imported_bindings(list(scene.objects), actions)
bind_action(actions[0], bindings, owners, rigs)
rigs[0].data.pose_position = 'POSE'; set_frame(scene_cfg['second']*24)
coat = bpy.data.objects[scene_cfg['coat_object']]; coat.hide_render = True
obj = bpy.data.objects[cfg['body_object']]; mesh = obj.data
selected_materials = set(scene_cfg['body_materials'])
require(obj.type == 'MESH' and len(mesh.vertices) <= 100000 and len(mesh.loops) <= 300000, 'Body bounds differ')
mesh.calc_loop_triangles()
triangles = [t for t in mesh.loop_triangles if mesh.materials[t.material_index].name in selected_materials]
positions = np.asarray([[tuple(mesh.vertices[v].co) for v in t.vertices] for t in triangles], dtype=float)
uv = np.asarray([[tuple(mesh.uv_layers[0].data[l].uv) for l in t.loops] for t in triangles], dtype=float)
buckets = {}
for index, triangle in enumerate(positions):
    for rotation, point in enumerate(triangle):
        key = tuple(np.floor(point*100000).astype(int))
        buckets.setdefault(key, []).append((index, rotation))
before_uv = np.empty((len(mesh.loops), 2), dtype=np.float32)
mesh.uv_layers[0].data.foreach_get('uv', before_uv.ravel())
changed_uv = before_uv.copy(); selected = set(); mapping = []
for face in payload['faces']:
    points = np.asarray(face['rest_positions_blender_xyz_m'], dtype=float)
    original_uv = np.asarray(face['uv0_blender_v_up'], dtype=float)
    repaired_uv = np.asarray(face['repaired_uv0_blender_v_up'], dtype=float)
    require(points.shape == (3, 3) and original_uv.shape == repaired_uv.shape == (3, 2)
            and np.isfinite(points).all() and np.isfinite(original_uv).all() and np.isfinite(repaired_uv).all(), 'Invalid repair rows')
    key = np.floor(points[0]*100000).astype(int); candidates = []
    for delta in itertools.product((-1, 0, 1), repeat=3):
        for index, rotation in buckets.get(tuple(key+delta), ()):
            if (np.max(np.abs(np.roll(positions[index], -rotation, axis=0)-points)) <= 1e-6
                    and np.max(np.abs(np.roll(uv[index], -rotation, axis=0)-original_uv)) <= 2e-6):
                candidates.append((index, rotation))
    require(len(candidates) == 1, 'Missing/ambiguous position-plus-UV face identity')
    index, rotation = candidates[0]
    loops = np.roll(np.asarray(triangles[index].loops), -rotation)
    require(not selected.intersection(map(int, loops)), 'Overlapping repair corners')
    selected.update(map(int, loops)); changed_uv[loops] = repaired_uv
    mapping.append({'source_primitive': face['source_primitive'], 'source_triangle': face['source_triangle'],
        'native_polygon': triangles[index].polygon_index, 'native_loops': loops.tolist()})
require(len(selected) == 3*cfg['expected_faces'], 'Corner coverage differs')

def state():
    points = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get('co', points.ravel())
    normals = np.asarray([tuple(n.vector) for n in mesh.corner_normals], dtype=np.float32)
    layers = {}
    for layer in list(mesh.uv_layers)[1:]:
        values = np.empty((len(mesh.loops), 2), dtype=np.float32); layer.data.foreach_get('uv', values.ravel())
        layers[layer.name] = hashlib.sha256(values.tobytes()).hexdigest()
    weights = [[(int(g.group), float(g.weight)) for g in v.groups] for v in mesh.vertices]
    return {'positions': hashlib.sha256(points.tobytes()).hexdigest(),
        'normals': hashlib.sha256(normals.tobytes()).hexdigest(), 'other_uv_layers': layers,
        'weights': hashlib.sha256(json.dumps(weights).encode()).hexdigest(),
        'camera': [[float(v) for v in row] for row in scene.camera.matrix_world],
        'ortho_scale': scene.camera.data.ortho_scale, 'frame': scene.frame_current+scene.frame_subframe,
        'bones': {b.name: [float(v) for row in b.matrix for v in row] for b in rigs[0].pose.bones}}

protected = state(); unlit = {}
for slot in obj.material_slots:
    original = slot.material
    if original.name not in selected_materials:
        continue
    if original.name not in unlit:
        material = original.copy(); material.name = 'DiagnosticUnlit_'+original.name
        shaders = [n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']
        outputs = [n for n in material.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output]
        require(len(shaders) == len(outputs) == 1, 'Ambiguous source material')
        color = shaders[0].inputs['Base Color']; emission = material.node_tree.nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = color.default_value; emission.inputs['Strength'].default_value = 1
        if color.links:
            require(len(color.links) == 1, 'Ambiguous original color route')
            material.node_tree.links.new(color.links[0].from_socket, emission.inputs['Color'])
        material.node_tree.links.new(emission.outputs['Emission'], outputs[0].inputs['Surface'])
        unlit[original.name] = material
    slot.material = unlit[original.name]
scene.render.filepath = str(out/'baseline_unlit.png'); bpy.ops.render.render(write_still=True)
mesh.uv_layers[0].data.foreach_set('uv', changed_uv.ravel()); mesh.update(); bpy.context.view_layer.update()
require(state() == protected, 'Protected geometry, shading, skin, UV1 or pose changed')
stored_uv = np.empty_like(changed_uv); mesh.uv_layers[0].data.foreach_get('uv', stored_uv.ravel())
require(np.array_equal(stored_uv, changed_uv), 'Stored UVs differ')
mask = np.ones(len(mesh.loops), bool); mask[list(selected)] = False
require(np.array_equal(stored_uv[mask], before_uv[mask]), 'Unselected UVs changed')
scene.render.filepath = str(out/'repaired_unlit.png'); bpy.ops.render.render(write_still=True)
require(hash_file(source) == scene_cfg['scene_sha256'] and hash_file(payload_path) == cfg['payload_sha256'], 'Inputs changed')
write_json(out/'color-ablation.json', {'schema_version': 1, 'status': 'color_only_review_pending',
    'source_scene_sha256': scene_cfg['scene_sha256'], 'payload_sha256': cfg['payload_sha256'],
    'faces': len(mapping), 'corners': len(selected), 'mapping': mapping, 'protected': protected,
    'unselected_uv0_byte_exact': True, 'protected_state_byte_exact': True, 'source_files_unchanged': True,
    'quality_approved': False, 'model_exported': False, 'images': {name: hash_file(out/name)
        for name in ('baseline_unlit.png', 'repaired_unlit.png')},
    'scope': 'Transient posed-scene UV0 changes with coat hidden and unlit basecolor. No normal-map, fur, geometry, animation or overall appearance acceptance.'})
