"""Apply bounded normal metadata to the current mesh; never move or repaint it."""

import hashlib
import math
import struct

import numpy as np

try:
    from normal_continuity import (FACE_ATTRIBUTE, MAX_CORNERS, MAX_FACES, MAX_VERTICES,
        POINT_ATTRIBUTE, array_hash, plan_normal_continuity, validate_normal_continuity)
except ModuleNotFoundError as error:
    if error.name != 'normal_continuity':
        raise
    from avatar_assets.normal_continuity import (FACE_ATTRIBUTE, MAX_CORNERS, MAX_FACES, MAX_VERTICES,
        POINT_ATTRIBUTE, array_hash, plan_normal_continuity, validate_normal_continuity)


MAX_WEIGHT_ENTRIES = 400000
MAX_UV_LAYERS = 8


def _protected_hashes(obj):
    mesh = obj.data
    positions = np.asarray([tuple(vertex.co) for vertex in mesh.vertices], dtype=np.float64)
    corners = np.asarray([loop.vertex_index for loop in mesh.loops], dtype=np.int64)
    materials = np.asarray([face.material_index for face in mesh.polygons], dtype=np.int64)
    uv_hash = hashlib.sha256()
    for layer in mesh.uv_layers:
        if len(layer.data) != len(mesh.loops):
            raise ValueError('Normal continuity UV corner count mismatch')
        values = np.empty((len(mesh.loops), 2), dtype=np.float32)
        layer.data.foreach_get('uv', values.reshape(-1))
        if not np.isfinite(values).all():
            raise ValueError('Normal continuity UV values must be finite')
        name = layer.name.encode('utf-8')
        uv_hash.update(struct.pack('<I', len(name))+name+values.astype('<f4').tobytes())
    weight_hash, entries = hashlib.sha256(), 0
    groups = {group.index: group.name for group in obj.vertex_groups}
    for index, name in sorted(groups.items()):
        if type(index) is not int or not 0 <= index < 64 or not isinstance(name, str) or len(name) > 128:
            raise ValueError('Normal continuity requires bounded skin group identities')
        encoded = name.encode('utf-8')
        weight_hash.update(struct.pack('<II', index, len(encoded))+encoded)
    for index, vertex in enumerate(mesh.vertices):
        if len(vertex.groups) > 64:
            raise ValueError('Normal continuity vertex has more than 64 weight groups')
        entries += len(vertex.groups)
        if entries > MAX_WEIGHT_ENTRIES:
            raise ValueError('Normal continuity exceeds 400000 skin weight entries')
        weight_hash.update(struct.pack('<II', index, len(vertex.groups)))
        for group in vertex.groups:
            if group.group not in groups or not math.isfinite(group.weight) or not 0 <= group.weight <= 1:
                raise ValueError('Normal continuity contains an invalid skin weight')
            weight_hash.update(struct.pack('<Id', group.group, group.weight))
    return {'positions': array_hash(positions, '<f8'), 'face_corner_indices': array_hash(corners, '<i8'),
            'face_material_indices': array_hash(materials, '<i8'), 'uv_layers': uv_hash.hexdigest(),
            'skin_weights_and_groups': weight_hash.hexdigest()}


def apply_normal_continuity(obj, configs):
    """Return a JSON report; both named attributes are object-local, before modifiers."""
    configs = validate_normal_continuity(configs)
    if not configs:
        return {'schema_version': 1, 'status': 'not_requested', 'corrections': []}
    if obj.type != 'MESH' or not 1 <= len(obj.data.vertices) <= MAX_VERTICES:
        raise ValueError('Normal continuity requires one bounded mesh')
    mesh = obj.data
    if (not 1 <= len(mesh.polygons) <= MAX_FACES or not 3 <= len(mesh.loops) <= MAX_CORNERS
            or not 1 <= len(mesh.materials) <= 64 or len(mesh.uv_layers) > MAX_UV_LAYERS
            or len(obj.vertex_groups) > 64):
        raise ValueError('Normal continuity mesh exceeds face/corner/material/UV/skin bounds')
    matrix = np.asarray(obj.matrix_world, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[:3, :3], np.eye(3), rtol=0, atol=1e-8)
            or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-8)):
        raise ValueError('Normal continuity requires applied rotation and scale in meters')
    if mesh.attributes.get(POINT_ATTRIBUTE) is not None or mesh.attributes.get(FACE_ATTRIBUTE) is not None:
        raise ValueError('Normal continuity will not overwrite existing authored-normal attributes')
    if any(material is None for material in mesh.materials):
        raise ValueError('Normal continuity requires named nonempty material slots')
    faces = [tuple(face.vertices) for face in mesh.polygons]
    if [i for face in faces for i in face] != [loop.vertex_index for loop in mesh.loops]:
        raise ValueError('Normal continuity polygon/corner ordering mismatch')
    plan = plan_normal_continuity(
        [tuple(vertex.co) for vertex in mesh.vertices], faces,
        [face.material_index for face in mesh.polygons], [material.name for material in mesh.materials],
        [tuple(normal.vector) for normal in mesh.corner_normals], configs)
    before = _protected_hashes(obj)
    # All input and count checks finish before the first mesh mutation.
    normal_attribute = mesh.attributes.new(name=POINT_ATTRIBUTE, type='FLOAT_VECTOR', domain='POINT')
    repair_attribute = mesh.attributes.new(name=FACE_ATTRIBUTE, type='BOOLEAN', domain='FACE')
    normal_attribute.data.foreach_set('vector', plan['vertex_normals'].astype(np.float32).reshape(-1))
    repair_attribute.data.foreach_set('value', plan['repaired_faces'])
    for index in np.flatnonzero(plan['repaired_faces']):
        mesh.polygons[int(index)].use_smooth = True
    mesh.normals_split_custom_set(plan['corner_normals'].tolist())
    mesh.update()
    after = _protected_hashes(obj)
    if after != before:
        raise ValueError('Normal continuity changed protected geometry, UVs, materials or skin weights')
    stored = np.asarray([tuple(normal.vector) for normal in mesh.corner_normals], dtype=np.float64)
    if stored.shape != plan['corner_normals'].shape or not np.isfinite(stored).all():
        raise ValueError('Normal continuity stored corner normals are invalid')
    report = plan['report']
    report.update(status='applied', geometry_uv_skin_unchanged=True, protected_before=before, protected_after=after,
        stored_corner_normals_sha256=array_hash(stored, '<f8'),
        stored_custom_normal_max_component_error=float(np.abs(stored-plan['corner_normals']).max()),
        attributes={'surface_normal': {'name': POINT_ATTRIBUTE, 'domain': 'POINT', 'type': 'FLOAT_VECTOR'},
                    'repair_face': {'name': FACE_ATTRIBUTE, 'domain': 'FACE', 'type': 'BOOLEAN'}},
        native_particle_note='Particles may use geometric normals; the caller must orient repaired-face strands from AuthoredSurfaceNormal explicitly.',
        authored_not_recovered=True)
    return report
