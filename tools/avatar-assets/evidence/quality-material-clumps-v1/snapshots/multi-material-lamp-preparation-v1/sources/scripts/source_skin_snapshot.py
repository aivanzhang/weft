"""Freeze exact native skin rows and stamp export-safe source vertex identities."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


POLICY = 'source-skin-id-v1'
ATTRIBUTE = '_AUTHOR_VERTEX_ID'
MAX_VERTICES = 100_000
MAX_TRIANGLES = 100_000
MAX_OBJECTS = 32
MAX_BONES = 64
MAX_GROUP_REFERENCES = 1_000_000
MAX_NPZ_BYTES = 16 * 1024**2
FRAME_TOLERANCE = 1e-7
WEIGHT_SUM_TOLERANCE = 1e-6


def _require(condition, message):
    if not condition:
        raise ValueError('Source skin snapshot: ' + message)


def _name(value):
    _require(isinstance(value, str) and 0 < len(value.encode('utf-8')) <= 255
             and not any(ord(c) < 32 for c in value), 'invalid native name')
    return value


def _pointer(value):
    method = getattr(value, 'as_pointer', None)
    return method() if callable(method) else id(value)


def _identity(value, label):
    matrix = np.asarray(value, dtype=np.float64)
    _require(matrix.shape == (4, 4) and np.isfinite(matrix).all()
             and np.max(abs(matrix - np.eye(4))) <= FRAME_TOLERANCE,
             label + ' must have an applied identity frame')


def _hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _number(value, label):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number))
             and math.isfinite(float(value)), label + ' must be a finite number')
    return float(value)


def snapshot_skin_sources(objects, rig, out):
    """Snapshot all selected body/fringe skins; caller exports their temporary ID attribute."""
    start = time.monotonic()

    def deadline():
        _require(time.monotonic() - start <= 30, '30 second preparation deadline exceeded')

    _require(isinstance(objects, (list, tuple)) and 1 <= len(objects) <= MAX_OBJECTS,
             'requires 1..32 explicitly selected mesh objects')
    out = Path(out)
    _require(out.is_dir() and not out.is_symlink(), 'output must be an existing local directory')
    names = ('source-skin.npz', 'source-skin.json', 'source-skin.npz.tmp', 'source-skin.json.tmp')
    _require(all(not (out/name).exists() and not (out/name).is_symlink() for name in names),
             'snapshot output already exists')
    _require(getattr(rig, 'type', None) == 'ARMATURE', 'one intended armature is required')
    _require(getattr(rig.data, 'pose_position', None) == 'REST', 'rig must already be in REST')
    _require(getattr(rig, 'mode', 'OBJECT') != 'EDIT', 'rig cannot be in edit mode')
    _identity(rig.matrix_world, 'rig world')
    _identity(rig.matrix_basis, 'rig local')
    _require(1 <= len(rig.data.bones) <= MAX_BONES, 'rig requires 1..64 bones')
    bone_names = [_name(b.name) for b in rig.data.bones]
    _require(len(set(bone_names)) == len(bone_names),
             'rig requires 1..64 unique named bones')
    bone_slots = {name: index for index, name in enumerate(bone_names)}
    rig_name, rig_data_name = _name(rig.name), _name(rig.data.name)
    owners, object_ids, mesh_ids, object_names = [], set(), set(), set()
    total_vertices = total_triangles = 0
    for obj in objects:
        deadline()
        _require(getattr(obj, 'type', None) == 'MESH' and getattr(obj, 'mode', 'OBJECT') == 'OBJECT',
                 'selected owners must be object-mode meshes')
        mesh = obj.data
        obj_id, mesh_id, obj_name = _pointer(obj), _pointer(mesh), _name(obj.name)
        _require(obj_id not in object_ids and mesh_id not in mesh_ids and obj_name not in object_names,
                 'duplicate object, shared mesh ownership, or object name')
        object_ids.add(obj_id); mesh_ids.add(mesh_id); object_names.add(obj_name)
        _identity(obj.matrix_world, 'mesh world')
        _identity(obj.matrix_basis, 'mesh local')
        _require(len(obj.modifiers) == 1, 'each mesh must have exactly one armature modifier')
        modifier = obj.modifiers[0]
        _require(modifier.type == 'ARMATURE' and _pointer(modifier.object) == _pointer(rig),
                 'mesh modifier must use the intended rig')
        _require(modifier.show_viewport and modifier.show_render and modifier.use_vertex_groups
                 and not modifier.use_bone_envelopes and not modifier.use_deform_preserve_volume,
                 'only active linear vertex-group armature deformation is supported')
        _require(len(mesh.attributes) <= 128, 'native attribute count exceeds bound')
        _require(not any(a.name.startswith('_') for a in mesh.attributes),
                 'existing underscore attributes would collide or leak into export')
        vertex_count, polygon_count = len(mesh.vertices), len(mesh.polygons)
        _require(0 < vertex_count <= MAX_VERTICES and 0 < polygon_count <= MAX_TRIANGLES,
                 'empty or oversized owner geometry')
        _require(len(mesh.loops) <= 3*MAX_TRIANGLES, 'mesh corner count exceeds bound')
        loop_totals = np.empty(polygon_count, dtype=np.int32)
        mesh.polygons.foreach_get('loop_total', loop_totals)
        _require(np.all(loop_totals >= 3), 'owner has a non-surface polygon')
        triangle_count = int(np.sum(loop_totals.astype(np.int64) - 2))
        _require(0 < triangle_count <= MAX_TRIANGLES, 'triangulated owner exceeds bound')
        _require(total_vertices + vertex_count <= MAX_VERTICES
                 and total_triangles + triangle_count <= MAX_TRIANGLES,
                 'aggregate vertex or triangle count exceeds bound')
        _require(1 <= len(obj.material_slots) <= 128, 'owner requires 1..128 material slots')
        _require(all(slot.material is not None for slot in obj.material_slots), 'empty material slot')
        materials = [_name(slot.material.name) for slot in obj.material_slots]
        _require(len(obj.vertex_groups) <= 128, 'too many native vertex groups')
        groups = {}
        group_names = set()
        for group in obj.vertex_groups:
            name = _name(group.name)
            _require(type(group.index) is int and 0 <= group.index < 128
                     and group.index not in groups and name not in group_names, 'invalid native vertex group')
            groups[group.index] = bone_slots.get(name)
            group_names.add(name)
        owners.append({'index': len(owners), 'name': obj_name, 'mesh_name': _name(mesh.name),
            'vertex_start': total_vertices, 'vertex_count': vertex_count,
            'triangle_start': total_triangles, 'triangle_count': triangle_count,
            'materials': materials, 'ignored_non_skin_groups': sorted(group_names-set(bone_names)),
            '_groups': groups})
        total_vertices += vertex_count; total_triangles += triangle_count

    positions = np.empty((total_vertices, 3), dtype='<f4')
    joints = np.zeros((total_vertices, 4), dtype='<u2')
    weights = np.zeros((total_vertices, 4), dtype='<f4')
    triangles = np.empty((total_triangles, 3), dtype='<u4')
    triangle_materials = np.empty(total_triangles, dtype='<u2')
    group_references, maximum_sum_error = 0, 0.
    for obj, owner in zip(objects, owners):
        deadline()
        mesh = obj.data
        v0, count = owner['vertex_start'], owner['vertex_count']
        t0, face_count = owner['triangle_start'], owner['triangle_count']
        mesh.vertices.foreach_get('co', positions[v0:v0+count].reshape(-1))
        _require(np.isfinite(positions[v0:v0+count]).all()
                 and np.max(abs(positions[v0:v0+count])) <= 10000, 'invalid bounded rest coordinates')
        for index, vertex in enumerate(mesh.vertices):
            if index % 256 == 0:
                deadline()
            _require(vertex.index == index and len(vertex.groups) <= 32, 'invalid vertex index or group count')
            group_references += len(vertex.groups)
            _require(group_references <= MAX_GROUP_REFERENCES, 'group-reference budget exceeded')
            active, seen_groups = [], set()
            for member in vertex.groups:
                _require(type(member.group) is int and member.group in owner['_groups']
                         and member.group not in seen_groups, 'invalid or duplicate group membership')
                seen_groups.add(member.group)
                weight = _number(member.weight, 'skin weight')
                _require(0 <= weight <= 1, 'skin weight is outside 0..1')
                joint = owner['_groups'][member.group]
                if joint is not None and weight > 0:
                    active.append((joint, np.float32(weight)))
            _require(1 <= len(active) <= 4 and all(weight > 0 for _, weight in active),
                     'every vertex requires 1..4 positive float32 bone influences')
            active.sort(key=lambda item: (-float(item[1]), item[0]))
            total = sum(float(weight) for _, weight in active)
            error = abs(total-1)
            _require(error <= WEIGHT_SUM_TOLERANCE, 'source bone weights must already be normalized')
            maximum_sum_error = max(maximum_sum_error, error)
            for column, (joint, weight) in enumerate(active):
                joints[v0+index, column], weights[v0+index, column] = joint, weight
        mesh.calc_loop_triangles()
        _require(len(mesh.loop_triangles) == face_count, 'native triangle count differs from bounded polygon count')
        local = np.empty((face_count, 3), dtype=np.int32)
        materials = np.empty(face_count, dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices', local.reshape(-1))
        mesh.loop_triangles.foreach_get('material_index', materials)
        _require(np.all((local >= 0) & (local < count))
                 and np.all((materials >= 0) & (materials < len(owner['materials']))),
                 'invalid triangle indices or material slots')
        _require(np.all(local[:,0] != local[:,1]) and np.all(local[:,1] != local[:,2])
                 and np.all(local[:,2] != local[:,0]), 'collapsed index triangle')
        _require(np.array_equal(np.unique(local), np.arange(count)), 'loose or unreferenced source vertices are unsupported')
        triangles[t0:t0+face_count] = local + v0
        triangle_materials[t0:t0+face_count] = materials
        del owner['_groups']

    arrays = {'positions': positions, 'joints': joints, 'weights': weights,
              'triangles': triangles, 'triangle_materials': triangle_materials}
    deadline()
    payload_bytes = sum(value.nbytes for value in arrays.values())
    _require(payload_bytes <= MAX_NPZ_BYTES, 'decoded snapshot array budget exceeded')
    with (out/'source-skin.npz.tmp').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    _require((out/'source-skin.npz.tmp').stat().st_size <= MAX_NPZ_BYTES, 'encoded snapshot exceeds bound')
    report = {'schema_version': 1, 'policy': POLICY, 'attribute': ATTRIBUTE,
        'space': 'applied_rest_Blender_XYZ_meters', 'vertex_count': total_vertices,
        'triangle_count': total_triangles, 'bone_names': bone_names,
        'rig': {'name': rig_name, 'data_name': rig_data_name, 'pose_position': 'REST'},
        'owners': owners, 'npz_file': 'source-skin.npz', 'npz_sha256': _hash(out/'source-skin.npz.tmp'),
        'npz_bytes': (out/'source-skin.npz.tmp').stat().st_size,
        'arrays': {name: {'dtype': value.dtype.str, 'shape': list(value.shape),
                         'sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}
                   for name, value in arrays.items()},
        'id_encoding': {'native_domain': 'POINT', 'native_type': 'INT',
                        'export_component_type': 5126, 'export_accessor_type': 'SCALAR',
                        'minimum': 0, 'maximum': total_vertices-1},
        'maximum_source_weight_sum_error': maximum_sum_error,
        'weight_policy': 'Every positive native bone influence is retained as float32; no pruning or renormalization',
        'limits': {'vertices': MAX_VERTICES, 'triangles': MAX_TRIANGLES, 'objects': MAX_OBJECTS,
                   'bones': MAX_BONES, 'npz_bytes': MAX_NPZ_BYTES, 'preparation_seconds': 30,
                   'identity_tolerance': FRAME_TOLERANCE, 'weight_sum_tolerance': WEIGHT_SUM_TOLERANCE},
        'normal_snapshot_included': False,
        'export_instructions': {'export_attributes': True, 'export_apply': False,
                                'preserve_all_positive_skin_weights_after_export': 'Required; Blender prunes <=0.0001 before serialization'},
        'limitations': ['Snapshot/staging IDs are not exported-skin verification or avatar quality approval.',
                       'Caller must retain the existing external native-job resource guard.']}
    added = []
    try:
        for obj, owner in zip(objects, owners):
            deadline()
            layer = obj.data.attributes.new(name=ATTRIBUTE, type='INT', domain='POINT')
            added.append((obj.data, layer))
            _require(layer.name == ATTRIBUTE and layer.data_type == 'INT' and layer.domain == 'POINT',
                     'source ID attribute was renamed or changed')
            ids = np.arange(owner['vertex_start'], owner['vertex_start']+owner['vertex_count'], dtype=np.int32)
            layer.data.foreach_set('value', ids)
            actual = np.empty(len(ids), dtype=np.int32)
            layer.data.foreach_get('value', actual)
            _require(np.array_equal(ids, actual), 'source ID write/readback differs')
        deadline()
        with (out/'source-skin.json.tmp').open('x') as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False); handle.write('\n')
        (out/'source-skin.npz.tmp').rename(out/'source-skin.npz')
        (out/'source-skin.json.tmp').rename(out/'source-skin.json')
    except Exception:
        # Roll back only attributes this call created; retain file evidence for inspection.
        for mesh, layer in reversed(added):
            mesh.attributes.remove(layer)
        raise
    return {**report, 'report_file': 'source-skin.json', 'report_sha256': _hash(out/'source-skin.json')}
