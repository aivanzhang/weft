"""Render one hash-bound native task in a fresh process; never execute embedded scripts."""

import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).parent))
from blender_verify import bind_action, clear_actions, hash_file, imported_bindings, set_frame, write_json
from native_render_tasks import require, validate_task


def render(plan_path, expected_hash, task_index, out):
    started = time.monotonic()
    plan_path, out = Path(plan_path), Path(out)
    require(plan_path.is_file() and not plan_path.is_symlink() and plan_path.stat().st_size <= 2*1024**2,
            'Prepared render plan exceeds bounds')
    require(hash_file(plan_path) == expected_hash, 'Prepared render plan changed')
    plan = json.loads(plan_path.read_text())
    require(plan.get('schema_version') == 1 and plan.get('status') == 'prepared_not_rendered'
            and plan.get('scene') == 'prepared-source.blend', 'Not a prepared native scene')
    tasks = plan.get('tasks')
    require(isinstance(tasks, list) and 1 <= len(tasks) <= 80
            and type(task_index) is int and 0 <= task_index < len(tasks), 'Invalid native task selection')
    task = validate_task(tasks[task_index])
    source = plan_path.parent/plan['scene']
    require(source.is_file() and not source.is_symlink() and source.stat().st_size <= 384*1024**2
            and hash_file(source) == plan['scene_sha256'], 'Prepared native scene changed or exceeds384MiB')
    require(out.is_dir() and not out.is_symlink()
            and all(not (out/name).exists() and not (out/name).is_symlink()
                    for name in ('image.png', 'before-render.json', 'render-report.json')),
            'Native task outputs must be new regular files')
    bpy.ops.wm.open_mainfile(filepath=str(source), use_scripts=False)
    scene = bpy.context.scene
    require(scene.name == task['scene'] and len(bpy.data.objects) <= 1024
            and len(bpy.data.actions) <= 16, 'Prepared scene inventory changed')
    rig, camera = bpy.data.objects.get(task['rig']), bpy.data.objects.get(task['camera']['name'])
    require(rig is not None and rig.type == 'ARMATURE' and camera is not None
            and camera.type == 'CAMERA' and camera.data.type == 'ORTHO', 'Prepared rig/camera is missing')
    collections = {'Object': bpy.data.objects, 'Mesh': bpy.data.meshes,
                   'Armature': bpy.data.armatures, 'Key': bpy.data.shape_keys}
    _, owners = imported_bindings(list(scene.objects), list(bpy.data.actions))
    armatures = [obj for obj in scene.objects if obj.type == 'ARMATURE']
    action = bpy.data.actions.get(task['action']) if task['action'] is not None else None
    if task['action'] is not None:
        require(action is not None, 'Prepared action is missing')
        bindings = []
        for row in task['bindings']:
            owner = collections[row['kind']].get(row['name'])
            require(owner is not None, 'Prepared action owner is missing')
            slots = [slot for slot in action.slots if slot.identifier == row['slot']] if row['slot'] else [None]
            require(len(slots) == 1, 'Prepared action slot is ambiguous or missing')
            bindings.append((owner, slots[0]))
        bind_action(action, {action.name: bindings}, owners, armatures)
    else:
        clear_actions(owners, armatures)
    rig.data.pose_position = task['pose_position']
    set_frame(task['frame'])
    require(set(task['bone_matrices']) == {bone.name for bone in rig.pose.bones}, 'Prepared bones changed')
    actual = {'rig_world': [value for row in rig.matrix_world for value in row],
              'bone_matrices': {bone.name: [value for row in bone.matrix for value in row] for bone in rig.pose.bones}}
    differences = [abs(a-b) for a, b in zip(actual['rig_world'], task['rig_world'])]
    for name, values in actual['bone_matrices'].items():
        differences.extend(abs(a-b) for a, b in zip(values, task['bone_matrices'][name]))
    maximum = max(differences)
    require(maximum <= 2e-6, 'Prepared action transforms changed after native reload')
    camera.matrix_world = Matrix(task['camera']['matrix_world'])
    for key in ('ortho_scale', 'clip_start', 'clip_end'):
        setattr(camera.data, key, task['camera'][key])
    scene.camera = camera
    scene.render.engine = 'CYCLES'
    scene.cycles.device, scene.cycles.samples = 'CPU', task['samples']
    scene.cycles.use_denoising = task['denoise']
    scene.render.threads_mode, scene.render.threads = 'FIXED', 4
    scene.render.resolution_x, scene.render.resolution_y = task['resolution']
    scene.render.resolution_percentage, scene.render.film_transparent = 100, False
    scene.render.image_settings.file_format, scene.render.image_settings.color_mode = 'PNG', 'RGB'
    require([scene.render.fps, scene.render.fps_base] == task['fps']
            and scene.view_settings.view_transform == task['view_transform'], 'Prepared timing/color settings changed')
    scene.render.filepath = str(out/'image.png')
    write_json(out/'before-render.json', {'schema_version': 1, 'task': task, 'max_transform_error': maximum,
        'plan_sha256': expected_hash, 'scene_sha256': plan['scene_sha256'], 'embedded_scripts_enabled': False})
    bpy.ops.render.render(write_still=True)
    require(hash_file(source) == plan['scene_sha256'] and hash_file(plan_path) == expected_hash,
            'Prepared source changed during rendering')
    write_json(out/'render-report.json', {'schema_version': 1, 'status': 'native_still_rendered_not_quality_accepted',
        'task_index': task_index, 'target_relative': task['path'], 'image_sha256': hash_file(out/'image.png'),
        'plan_sha256': expected_hash, 'scene_sha256': plan['scene_sha256'], 'max_transform_error': maximum,
        'samples': task['samples'], 'denoise': task['denoise'], 'resolution': task['resolution'],
        'elapsed_seconds': time.monotonic()-started, 'quality_approved': False})


if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--')+1:]
    require(len(args) == 4, 'Require plan, exact hash, task index and new output directory')
    render(args[0], args[1], int(args[2]), args[3])
