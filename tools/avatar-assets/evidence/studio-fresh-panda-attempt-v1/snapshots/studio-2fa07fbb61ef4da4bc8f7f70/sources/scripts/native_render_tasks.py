"""Capture bounded native still tasks without treating a prepared scene as rendered output."""

import hashlib
import math
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def _number(value, low, high):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            'Invalid native render number')


def validate_task(task):
    fields = {'path', 'scene', 'camera', 'rig', 'pose_position', 'frame', 'action', 'bindings',
              'resolution', 'samples', 'denoise', 'fps', 'view_transform', 'rig_world', 'bone_matrices'}
    require(isinstance(task, dict) and set(task) == fields, 'Invalid native render task')
    require(isinstance(task['path'], str), 'Invalid native image path')
    path = Path(task['path'])
    require(not path.is_absolute() and '..' not in path.parts and path.suffix == '.png'
            and 1 <= len(path.parts) <= 4 and len(str(path)) <= 256, 'Unsafe native image path')
    for key in ('scene', 'rig', 'view_transform'):
        require(isinstance(task[key], str) and 0 < len(task[key]) <= 128, 'Invalid native '+key)
    require(task['pose_position'] in ('REST', 'POSE'), 'Invalid native pose mode')
    _number(task['frame'], -600, 3600)
    require(task['action'] is None or isinstance(task['action'], str) and 0 < len(task['action']) <= 128,
            'Invalid native action')
    require(isinstance(task['bindings'], list) and len(task['bindings']) <= 64
            and bool(task['bindings']) == (task['action'] is not None), 'Native action bindings are incomplete')
    for row in task['bindings']:
        require(isinstance(row, dict) and set(row) == {'kind', 'name', 'slot'}
                and row['kind'] in ('Object', 'Mesh', 'Armature', 'Key')
                and isinstance(row['name'], str) and 0 < len(row['name']) <= 128
                and (row['slot'] is None or isinstance(row['slot'], str) and 0 < len(row['slot']) <= 128),
                'Unsupported native action owner')
    size = task['resolution']
    require(isinstance(size, list) and len(size) == 2
            and all(type(v) is int and 128 <= v <= 1600 for v in size), 'Invalid native image size')
    require(type(task['samples']) is int and 1 <= task['samples'] <= 64 and type(task['denoise']) is bool,
            'Invalid native sample settings')
    require(isinstance(task['fps'], list) and len(task['fps']) == 2, 'Invalid native frame rate')
    _number(task['fps'][0], 1, 120)
    _number(task['fps'][1], .001, 1001)
    camera = task['camera']
    require(isinstance(camera, dict) and set(camera) == {'name', 'matrix_world', 'ortho_scale', 'clip_start', 'clip_end'}
            and isinstance(camera['name'], str) and 0 < len(camera['name']) <= 128, 'Invalid native camera')
    matrix = camera['matrix_world']
    require(isinstance(matrix, list) and len(matrix) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in matrix), 'Invalid native camera matrix')
    for row in matrix:
        for value in row:
            _number(value, -10000, 10000)
    require(matrix[3] == [0, 0, 0, 1], 'Native camera matrix must be affine')
    for key in ('ortho_scale', 'clip_start', 'clip_end'):
        _number(camera[key], .00001, 100000)
    require(camera['clip_start'] < camera['clip_end'], 'Invalid native camera clipping')
    bones = task['bone_matrices']
    require(isinstance(bones, dict) and 1 <= len(bones) <= 64
            and all(isinstance(name, str) and 0 < len(name) <= 128 for name in bones), 'Invalid native bone proof')
    for values in [task['rig_world'], *bones.values()]:
        require(isinstance(values, list) and len(values) == 16, 'Invalid native transform proof')
        for value in values:
            _number(value, -10000, 10000)
    return task


def render_or_record(scene, rig, action, bindings, tasks, out, *, hash_output=False):
    import bpy
    if tasks is None:
        bpy.ops.render.render(write_still=True)
        if hash_output:
            with Path(scene.render.filepath).open('rb') as handle:
                return hashlib.file_digest(handle, 'sha256').hexdigest()
        return None
    require(isinstance(tasks, list) and len(tasks) < 80, 'Native render plan exceeds80tasks')
    bpy.context.view_layer.update()
    require(scene.render.engine == 'CYCLES' and scene.cycles.device == 'CPU'
            and scene.render.resolution_percentage == 100 and not scene.render.film_transparent
            and scene.render.image_settings.file_format == 'PNG'
            and scene.render.image_settings.color_mode == 'RGB'
            and scene.camera.data.type == 'ORTHO', 'Unsupported native render state')
    path = Path(scene.render.filepath).resolve().relative_to(Path(out).resolve()).as_posix()
    require(path not in {item['path'] for item in tasks}, 'Duplicate native image task')
    camera = scene.camera
    task = {'path': path, 'scene': scene.name, 'rig': rig.name,
            'pose_position': rig.data.pose_position, 'frame': float(scene.frame_current+scene.frame_subframe),
            'action': action.name if action else None,
            'bindings': [{'kind': owner.bl_rna.identifier, 'name': owner.name,
                          'slot': slot.identifier if slot else None} for owner, slot in bindings] if action else [],
            'camera': {'name': camera.name, 'matrix_world': [list(row) for row in camera.matrix_world],
                       'ortho_scale': camera.data.ortho_scale, 'clip_start': camera.data.clip_start,
                       'clip_end': camera.data.clip_end},
            'resolution': [scene.render.resolution_x, scene.render.resolution_y],
            'samples': scene.cycles.samples, 'denoise': scene.cycles.use_denoising,
            'fps': [scene.render.fps, scene.render.fps_base], 'view_transform': scene.view_settings.view_transform,
            'rig_world': [value for row in rig.matrix_world for value in row],
            'bone_matrices': {bone.name: [value for row in bone.matrix for value in row] for bone in rig.pose.bones}}
    tasks.append(validate_task(task))
    return None
