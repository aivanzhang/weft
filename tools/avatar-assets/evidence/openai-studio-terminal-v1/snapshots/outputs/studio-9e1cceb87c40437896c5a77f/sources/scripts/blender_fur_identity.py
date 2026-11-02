"""Stamp owned point identities so exported seams cannot obscure fur attachment."""

import numpy as np


ATTRIBUTE = '_AVATAR_FUR_VERTEX_ID'
POLICY = 'portable-fur-source-ids-v1'
MAX_VERTICES = 100000
MAX_TRIANGLES = 100000


def _require(condition, message):
    if not condition:
        raise ValueError('Portable fur identity: '+message)


def _name(value):
    _require(type(value) is str and 0 < len(value.encode('utf-8')) <= 255
             and not any(ord(c) < 32 or ord(c) == 127 for c in value), 'invalid object name')
    return value


def _identity(value):
    array = np.asarray(value, dtype=np.float64)
    _require(array.shape == (4, 4) and np.isfinite(array).all()
             and np.max(abs(array-np.eye(4))) <= 1e-7, 'requires applied identity metre frames')


def _pointer(value):
    method = getattr(value, 'as_pointer', None)
    return method() if callable(method) else id(value)


def stamp_fur_identities(body, coats, rig, reports, *, allow_source_skin_ids=False):
    """Add only local point IDs; native source-skin experiments retain their separate IDs."""
    _require(type(allow_source_skin_ids) is bool, 'source-skin allowance must be boolean')
    _require(type(coats) is list and type(reports) is list and 1 <= len(coats) == len(reports) <= 128,
             'requires one report per bounded coat object')
    _require(getattr(rig, 'type', None) == 'ARMATURE' and rig.data.pose_position == 'REST'
             and 1 <= len(rig.data.bones) <= 64, 'requires one rest armature with 1..64 bones')
    _identity(rig.matrix_world)
    _identity(rig.matrix_basis)
    names, meshes, objects = set(), set(), set()
    total_vertices = total_triangles = 0
    owners = [body, *coats]
    allowed = {'_AUTHOR_VERTEX_ID'} if allow_source_skin_ids else set()
    for obj in owners:
        _require(getattr(obj, 'type', None) == 'MESH' and getattr(obj, 'mode', 'OBJECT') == 'OBJECT',
                 'owners must be object-mode meshes')
        name, mesh = _name(obj.name), obj.data
        _require(name not in names and _pointer(obj) not in objects and _pointer(mesh) not in meshes,
                 'duplicate object, name or shared mesh ownership')
        names.add(name); objects.add(_pointer(obj)); meshes.add(_pointer(mesh))
        _identity(obj.matrix_world)
        _identity(obj.matrix_basis)
        _require(len(obj.modifiers) == 1, 'each owner needs one linear armature modifier')
        modifier = obj.modifiers[0]
        _require(modifier.type == 'ARMATURE' and _pointer(modifier.object) == _pointer(rig)
                 and modifier.show_render and modifier.show_viewport and modifier.use_vertex_groups
                 and not modifier.use_bone_envelopes and not modifier.use_deform_preserve_volume,
                 'unsupported or inactive skin deformation')
        _require(len(mesh.attributes) <= 128
                 and all(not a.name.startswith('_') or a.name in allowed for a in mesh.attributes),
                 'existing custom export attributes collide or are unowned')
        _require(0 < len(mesh.vertices) <= MAX_VERTICES and 0 < len(mesh.polygons) <= MAX_TRIANGLES,
                 'owner vertex/face bounds exceeded')
        _require(all(v.index == index for index, v in enumerate(mesh.vertices)), 'noncanonical native vertex IDs')
        mesh.calc_loop_triangles()
        _require(0 < len(mesh.loop_triangles) <= MAX_TRIANGLES, 'owner triangle bounds exceeded')
        total_vertices += len(mesh.vertices)
        total_triangles += len(mesh.loop_triangles)
        _require(total_vertices <= MAX_VERTICES and total_triangles <= MAX_TRIANGLES,
                 'aggregate geometry bounds exceeded')
    root_count = 0
    for coat, report in zip(coats, reports):
        _require(type(report) is dict and report.get('geometry_profile') in ('legacy', 'combed-v1', 'tuft-v1'),
                 'unknown six-vertex card profile')
        count = report.get('cards')
        _require(type(count) is int and 1 <= count <= 5000, 'invalid root count')
        root_count += count
        _require(root_count <= 5000 and len(coat.data.vertices) == 6*count
                 and len(coat.data.loop_triangles) == 4*count, 'card topology differs from recorded roots')
        summary = report.get('root_summary')
        receipt = summary.get('binding_archive') if isinstance(summary, dict) else None
        _require(type(receipt) is dict and receipt.get('schema_version') == 2 and receipt.get('roots') == count,
                 'new export requires v2 root correspondence')
        expected = {'policy': POLICY, 'attribute': ATTRIBUTE, 'source_node': body.name, 'coat_node': coat.name,
                    'source_vertex_count': len(body.data.vertices), 'source_triangle_count': len(body.data.loop_triangles),
                    'coat_vertex_count': 6*count}
        _require(receipt.get('export_correspondence') == expected, 'root receipt belongs to different objects')

    # Validate the complete selection before adding any exportable attribute.
    added = []
    try:
        for obj in owners:
            layer = obj.data.attributes.new(name=ATTRIBUTE, type='INT', domain='POINT')
            added.append((obj.data, layer))
            _require(layer.name == ATTRIBUTE and layer.domain == 'POINT' and layer.data_type == 'INT',
                     'identity attribute was renamed or changed')
            values = np.arange(len(obj.data.vertices), dtype=np.int32)
            layer.data.foreach_set('value', values)
            actual = np.empty_like(values)
            layer.data.foreach_get('value', actual)
            _require(np.array_equal(actual, values), 'identity write/readback differs')
    except Exception:
        for mesh, layer in reversed(added):
            mesh.attributes.remove(layer)
        raise
    return {'schema_version': 1, 'policy': POLICY, 'attribute': ATTRIBUTE,
            'space': 'applied_rest_Blender_XYZ_meters', 'source_node': body.name,
            'source_vertex_count': len(body.data.vertices), 'source_triangle_count': len(body.data.loop_triangles),
            'coats': [{'node': coat.name, 'vertices': len(coat.data.vertices), 'roots': report['cards']}
                      for coat, report in zip(coats, reports)],
            'recorded_roots': root_count, 'export_attributes': True,
            'native_attribute_type': 'INT', 'native_attribute_domain': 'POINT',
            'export_attribute_type': 'SCALAR', 'export_component_type': 5126,
            'identity_scope': 'IDs are local to each named mesh; bottom card endpoints are 6*r and 6*r+1',
            'attachment_verified': False, 'source_skin_restoration_performed': False}
