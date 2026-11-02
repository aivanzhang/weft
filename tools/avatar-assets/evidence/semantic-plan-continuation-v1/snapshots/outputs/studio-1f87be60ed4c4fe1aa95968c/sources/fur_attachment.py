"""Measure portable roots against the final exported body, without repairing either."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from avatar_assets import fur_bindings, safety, three_d
from avatar_assets.safety import require


POLICY = 'portable-fur-attachment-v1'
IDENTITY_POLICY = 'portable-fur-source-ids-v1'
ATTRIBUTE = '_AVATAR_FUR_VERTEX_ID'
MAX_JSON = 1024 * 1024
MAX_ROOTS = 5000
MAX_WITNESSES = 32
GATES = {'named_weight': 1e-6, 'rest_midpoint_m': 1e-6,
         'animated_attachment_m': 1e-5, 'authored_inset_m': .0005}
LIMITATIONS = ['Attachment to the final exported body, not native-source skin fidelity.',
               'Discrete 24 fps samples only; no continuous-motion, loop, physics or appearance approval.']
ERROR_ENCODING = 'little-endian float64 max-absolute-component error per root; part order, state/time then part order'


class UnsupportedAttachment(ValueError):
    """Valid evidence outside this auditor's declared mapping/animation contract."""


def _hash(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def _intent(value):
    require(type(value) is dict and set(value) == {
        'schema_version', 'source_sha256', 'design_sha256', 'model_sha256', 'state_ids'},
        'Invalid attachment intent fields')
    require(type(value['schema_version']) is int and value['schema_version'] == 1, 'Invalid intent version')
    for key in ('source_sha256', 'design_sha256', 'model_sha256'):
        _hash(value[key])
    ids = value['state_ids']
    require(type(ids) is list and 1 <= len(ids) <= 8, 'Expected 1..8 states')
    require([safety.slug(x) for x in ids] == ids and len(set(ids)) == len(ids), 'Duplicate states')
    return json.loads(json.dumps(value))


def _file(root, name, maximum):
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts,
            'Unsafe attachment input path')
    path = root / name
    for entry in (path, *path.parents):
        if entry == root:
            break
        require(not entry.is_symlink(), 'Symlink in attachment input')
    path = safety.local_file(path, maximum)
    require(path.is_relative_to(root), 'Attachment input escaped pack')
    return path


def _bound_file(root, name, maximum, artifacts, inventory):
    path = _file(root, name, maximum)
    row = {'path': name, 'bytes': path.stat().st_size, 'sha256': safety.sha256(path)}
    matches = [entry for entry in artifacts if entry.get('path') == name]
    require(len(matches) == 1 and all(matches[0].get(k) == v for k, v in row.items()),
            'Missing or changed attachment input: '+name)
    inventory.append(row)
    return path


def _context(pack, intent, *, arrays):
    root = Path(pack)
    require(root.is_dir() and not root.is_symlink(), 'Invalid finish pack')
    root = root.resolve()
    manifest_path = _file(root, 'manifest.json', MAX_JSON)
    manifest = safety.read_json(manifest_path)
    require(type(manifest) is dict and manifest.get('schema_version') == 1
            and manifest.get('status') == 'needs_visual_review', 'Finish pack is not complete')
    artifacts = manifest.get('artifacts')
    require(type(artifacts) is list and 1 <= len(artifacts) <= 4096
            and all(type(row) is dict for row in artifacts), 'Invalid artifact inventory')
    inventory = [{'path': 'manifest.json', 'bytes': manifest_path.stat().st_size,
                  'sha256': safety.sha256(manifest_path)}]
    report_path = _bound_file(root, 'finish-report.json', MAX_JSON, artifacts, inventory)
    design_path = _bound_file(root, 'sources/finish-design.json', MAX_JSON, artifacts, inventory)
    model_path = _bound_file(root, 'model.glb', three_d.MAX_FILE_BYTES, artifacts, inventory)
    report, design = safety.read_json(report_path), safety.read_json(design_path)
    require(all(type(x) is dict and x.get('schema_version') == 1 for x in (report, design)),
            'Invalid finish/design schema')
    require(all(x.get('source_sha256') == intent['source_sha256'] for x in (manifest, report, design))
            and report.get('source_unchanged') is True, 'Source binding changed')
    require(safety.sha256(design_path) == intent['design_sha256']
            and inventory[-1]['sha256'] == intent['model_sha256'], 'Design/model binding changed')
    require(manifest.get('finish') == report and manifest.get('id') == design.get('id')
            and type(design.get('id')) is str, 'Embedded finish/design differs')
    for value in (manifest, report, design):
        states = value.get('states')
        require(type(states) is list and all(type(s) is dict for s in states)
                and [s.get('id') for s in states] == intent['state_ids'], 'Finish states differ from intent')
    parts = report.get('portable_fur')
    if parts is None or parts == []:
        require(design.get('portable_fur') is None and report.get('portable_fur_identity') is None,
                'Requested portable fur has no recorded parts')
        return root, model_path, [], inventory, None
    require(type(parts) is list and 1 <= len(parts) <= 128 and type(design.get('portable_fur')) is dict,
            'Invalid portable parts')
    bands = design['portable_fur'].get('bands')
    require(type(bands) is list and 1 <= len(bands) <= 4
            and all(type(band) is dict and type(band.get('count')) is int
                    and 1 <= band['count'] <= MAX_ROOTS for band in bands), 'Invalid requested fur bands/counts')
    expected_roots = sum(band['count'] for band in bands)
    require(expected_roots <= MAX_ROOTS, 'Requested aggregate roots exceed 5000')
    loaded, roots, paths, owners = [], 0, set(), set()
    historical = False
    for part in parts:
        require(type(part) is dict and type(part.get('cards')) is int and 1 <= part['cards'] <= MAX_ROOTS,
                'Invalid portable root count')
        roots += part['cards']
        require(roots <= MAX_ROOTS, 'Aggregate roots exceed 5000')
        require(part.get('geometry_profile', 'legacy') in ('legacy', 'combed-v1', 'tuft-v1'),
                'Unknown portable geometry profile')
        summary = part.get('root_summary')
        receipt = summary.get('binding_archive') if type(summary) is dict else None
        if receipt is None:
            historical = True
            loaded.append((part, None, None))
            continue
        require(type(receipt) is dict and receipt.get('roots') == part['cards'], 'Binding count differs')
        name = receipt.get('path')
        require(type(name) is str and name not in paths, 'Duplicate binding archive')
        paths.add(name)
        path = _bound_file(root, name, 512*1024, artifacts, inventory)
        require(inventory[-1]['bytes'] == receipt.get('bytes')
                and inventory[-1]['sha256'] == receipt.get('sha256'), 'Binding receipt changed')
        if receipt.get('schema_version') == 1:
            historical = True
        else:
            require(receipt.get('schema_version') == 2, 'Invalid binding version')
        # The bounded loader validates receipt/NPY metadata even for historical evidence.
        values = fur_bindings.load_bindings(root, receipt) if arrays else None
        if receipt.get('schema_version') == 2:
            owner = receipt.get('export_correspondence')
            require(type(owner) is dict and owner.get('coat_node') not in owners, 'Duplicate coat owner')
            owners.add(owner['coat_node'])
        loaded.append((part, receipt, values))
    require(sum(row['bytes'] for row in inventory if row['path'] in paths) <= MAX_JSON,
            'Binding archives exceed 1 MiB')
    require(roots == expected_roots, 'Recorded roots differ from the exact requested band total')
    identity = report.get('portable_fur_identity')
    if historical:
        require(identity is None and all(receipt is None or receipt.get('schema_version') == 1
                                       for _, receipt, _ in loaded),
                'Advertised v2 export identity has missing or mixed-version bindings')
    else:
        _identity_receipt(identity, loaded, roots)
    return root, model_path, loaded, inventory, 'historical' if historical else identity


def _identity_receipt(identity, parts, roots):
    first = parts[0][1]['export_correspondence']
    expected = {'schema_version': 1, 'policy': IDENTITY_POLICY, 'attribute': ATTRIBUTE,
                'space': 'applied_rest_Blender_XYZ_meters', 'source_node': first['source_node'],
                'source_vertex_count': first['source_vertex_count'],
                'source_triangle_count': first['source_triangle_count'],
                'coats': [], 'recorded_roots': roots, 'export_attributes': True,
                'native_attribute_type': 'INT', 'native_attribute_domain': 'POINT',
                'export_attribute_type': 'SCALAR', 'export_component_type': 5126,
                'identity_scope': 'IDs are local to each named mesh; bottom card endpoints are 6*r and 6*r+1',
                'attachment_verified': False, 'source_skin_restoration_performed': False}
    for part, receipt, _ in parts:
        owner = receipt['export_correspondence']
        require(owner['policy'] == IDENTITY_POLICY and owner['attribute'] == ATTRIBUTE
                and all(owner[key] == first[key] for key in ('source_node', 'source_vertex_count', 'source_triangle_count'))
                and owner['coat_vertex_count'] == 6*part['cards'], 'Inconsistent identity owners')
        expected['coats'].append({'node': owner['coat_node'], 'vertices': 6*part['cards'], 'roots': part['cards']})
    require(identity == expected, 'Native identity receipt differs from bindings')


def _implementations():
    return {name: safety.sha256(Path(module.__file__)) for name, module in (
        ('fur_attachment.py', __import__(__name__, fromlist=['audit_finish_pack'])),
        ('fur_bindings.py', fur_bindings), ('three_d.py', three_d), ('safety.py', safety))}


def _matrix(values):
    return np.asarray(values, dtype=np.float64).reshape(4, 4, order='F')


def _hierarchy(glb):
    nodes = glb.doc.get('nodes', [])
    require(nodes, 'No exported nodes')
    parents, names = {}, {}
    for i, node in enumerate(nodes):
        three_d._local_matrix(node)
        name = node.get('name')
        if name is not None:
            require(type(name) is str and name and name not in names, 'Ambiguous exported node name')
            names[name] = i
        children = node.get('children', [])
        require(type(children) is list and len(children) <= 256, 'Invalid children')
        for child in children:
            three_d._index(child, len(nodes), 'child')
            require(child not in parents, 'Multiple node parents')
            parents[child] = i
    for i in range(len(nodes)):
        seen, current = set(), i
        while current in parents:
            require(current not in seen, 'Cyclic hierarchy')
            seen.add(current)
            current = parents[current]
    scenes = glb.doc.get('scenes', [])
    scene = scenes[three_d._index(glb.doc.get('scene', 0), len(scenes), 'scene')]
    roots = scene.get('nodes')
    require(type(roots) is list and 1 <= len(roots) <= 256, 'Invalid scene roots')
    for root in roots:
        three_d._index(root, len(nodes), 'scene root')
        require(root not in parents, 'Scene root has parent')
    reachable, pending = set(), list(roots)
    while pending:
        i = pending.pop()
        if i not in reachable:
            reachable.add(i)
            pending.extend(nodes[i].get('children', []))
    joint_set = set()
    for skin in glb.doc.get('skins', []):
        joints = skin.get('joints')
        require(type(joints) is list and 1 <= len(joints) <= 64
                and all(type(i) is int for i in joints) and len(set(joints)) == len(joints), 'Invalid skin joints')
        for i in joints:
            three_d._index(i, len(nodes), 'joint')
            require(i in reachable and nodes[i].get('name') in names, 'Unnamed/unreachable joint')
        joint_set.update(joints)
    require(1 <= len(joint_set) <= 64, 'Require 1..64 actual joints')
    ordered = sorted(joint_set, key=lambda i: nodes[i]['name'])
    return nodes, parents, names, reachable, ordered


def _oriented(faces):
    return np.take_along_axis(faces, (faces.argmin(axis=1)[:, None]+np.arange(3)) % 3, axis=1)


def _admit_owners(glb, owners):
    """Count primitive occurrences before cached accessors can expand into dense copies."""
    doc = glb.doc
    accessors, meshes, skins = (doc.get(key, []) for key in ('accessors', 'meshes', 'skins'))

    def metadata(index, kind, components, limit, *, normalized=False):
        value = accessors[three_d._index(index, len(accessors), 'attachment accessor')]
        count = value.get('count')
        require(value.get('type') == kind and value.get('componentType') in components
                and not value.get('sparse') and type(count) is int and 1 <= count <= limit,
                'Invalid attachment accessor metadata')
        require(value.get('normalized', False) is normalized, 'Invalid attachment accessor normalization')
        return count

    aggregate_vertices = aggregate_triangles = 0
    for node_index, native_vertices, expected_triangles in owners:
        require(type(native_vertices) is int and 1 <= native_vertices <= 100000
                and type(expected_triangles) is int and 1 <= expected_triangles <= 100000,
                'Invalid declared owner geometry bounds')
        node = doc['nodes'][node_index]
        skin = skins[three_d._index(node.get('skin'), len(skins), 'owner skin')]
        require(metadata(skin.get('inverseBindMatrices'), 'MAT4', (5126,), 64) == len(skin['joints']),
                'Inverse bind count differs')
        mesh = meshes[three_d._index(node.get('mesh'), len(meshes), 'owner mesh')]
        entries = mesh.get('primitives')
        require(type(entries) is list and 1 <= len(entries) <= 32, 'Invalid primitive count')
        vertices = triangles = 0
        for primitive in entries:
            require(primitive.get('mode', 4) == 4 and not primitive.get('targets'), 'Unsupported owner topology')
            attrs = primitive.get('attributes')
            require(type(attrs) is dict and ATTRIBUTE in attrs, 'Missing exported fur vertex IDs')
            require(not any(k.startswith(('JOINTS_', 'WEIGHTS_')) and k not in ('JOINTS_0', 'WEIGHTS_0')
                            for k in attrs), 'More than four exported skin influences is unsupported')
            count = metadata(attrs.get('POSITION'), 'VEC3', (5126,), 100000)
            id_count = metadata(attrs[ATTRIBUTE], 'SCALAR', (5126,), 100000)
            joint_count = metadata(attrs.get('JOINTS_0'), 'VEC4', (5121, 5123), 100000)
            weight = accessors[three_d._index(attrs.get('WEIGHTS_0'), len(accessors), 'weights accessor')]
            weight_count = metadata(attrs['WEIGHTS_0'], 'VEC4', (5121, 5123, 5126), 100000,
                                    normalized=weight.get('componentType') != 5126)
            require(count == id_count == joint_count == weight_count, 'Owner attribute counts differ')
            index_count = (metadata(primitive['indices'], 'SCALAR', (5121, 5123, 5125), 300000)
                           if 'indices' in primitive else count)
            require(index_count % 3 == 0, 'Triangle index count is not divisible by three')
            vertices += count
            triangles += index_count//3
            aggregate_vertices += count
            aggregate_triangles += index_count//3
            require(vertices <= 100000 and triangles <= 100000, 'Owner metadata geometry budget exceeded')
            require(aggregate_vertices <= 100000 and aggregate_triangles <= 100000,
                    'Aggregate metadata geometry budget exceeded')
        require(vertices >= native_vertices and triangles == expected_triangles,
                'Owner metadata geometry coverage differs')


def _owner(glb, node_index, count, triangles_expected, nodes, world, bone_order):
    node, doc = nodes[node_index], glb.doc
    if (np.max(abs(_matrix(three_d._local_matrix(node))-np.eye(4))) > 1e-7
            or np.max(abs(world[node_index]-np.eye(4))) > 1e-7):
        raise UnsupportedAttachment('Portable owners require applied identity meter frames')
    skin = doc.get('skins', [])[three_d._index(node.get('skin'), len(doc.get('skins', [])), 'owner skin')]
    palette = np.asarray(skin['joints'], dtype=np.int32)
    inverse = np.asarray(glb.accessor(skin.get('inverseBindMatrices'), 'MAT4', (5126,)), dtype=np.float64)
    require(len(inverse) == len(palette), 'Inverse bind count differs')
    inverse = inverse.reshape(-1, 4, 4).transpose(0, 2, 1)
    require(np.max(abs(inverse[:, 3]-[0, 0, 0, 1])) <= 1e-6
            and np.all(abs(np.linalg.det(inverse[:, :3, :3])) > 1e-12), 'Invalid inverse bind matrices')
    mesh_index = three_d._index(node.get('mesh'), len(doc.get('meshes', [])), 'owner mesh')
    mesh = doc['meshes'][mesh_index]
    entries = mesh.get('primitives')
    require(type(entries) is list and 1 <= len(entries) <= 32, 'Invalid primitive count')
    xyzs, ids_list, joints_list, weights_list, dense_list, faces_list = [], [], [], [], [], []
    bone_lookup = {node: i for i, node in enumerate(bone_order)}
    named_palette = np.asarray([bone_lookup[i] for i in palette], dtype=np.int32)
    total = 0
    for primitive in entries:
        require(primitive.get('mode', 4) == 4 and not primitive.get('targets'), 'Unsupported owner topology')
        attrs = primitive.get('attributes', {})
        require(type(attrs) is dict and ATTRIBUTE in attrs, 'Missing exported fur vertex IDs')
        require(not any(k.startswith(('JOINTS_', 'WEIGHTS_')) and k not in ('JOINTS_0', 'WEIGHTS_0') for k in attrs),
                'More than four exported skin influences is unsupported')
        xyz = np.asarray(glb.accessor(attrs.get('POSITION'), 'VEC3', (5126,)), dtype=np.float64)
        ids = np.asarray(glb.accessor(attrs[ATTRIBUTE], 'SCALAR', (5126,)), dtype=np.float64).ravel()
        require(not doc['accessors'][attrs[ATTRIBUTE]].get('normalized') and np.all(ids == np.floor(ids))
                and np.all((ids >= 0) & (ids < count)), 'Invalid exported point IDs')
        ids = ids.astype(np.int32)
        joints = np.asarray(glb.accessor(attrs.get('JOINTS_0'), 'VEC4', (5121, 5123)), dtype=np.int32)
        require(not doc['accessors'][attrs['JOINTS_0']].get('normalized')
                and np.all((joints >= 0) & (joints < len(palette))), 'Invalid joint indices')
        weights = np.asarray(glb.accessor(attrs.get('WEIGHTS_0'), 'VEC4', (5121, 5123, 5126)), dtype=np.float64)
        weight_meta = doc['accessors'][attrs['WEIGHTS_0']]
        require(weight_meta['componentType'] == 5126 or weight_meta.get('normalized') is True,
                'Integer weights must be normalized')
        require(len(xyz) == len(ids) == len(joints) == len(weights) and np.all(weights >= 0)
                and np.all(abs(weights.sum(axis=1)-1) <= .001), 'Invalid skin coverage')
        weights /= weights.sum(axis=1, keepdims=True)
        dense = np.zeros((len(xyz), len(bone_order)), dtype=np.float64)
        for column in range(4):
            np.add.at(dense, (np.arange(len(xyz)), named_palette[joints[:, column]]), weights[:, column])
        if 'indices' in primitive:
            indices = np.asarray(glb.accessor(primitive['indices'], 'SCALAR', (5121, 5123, 5125),
                                             count_limit=three_d.MAX_TRIANGLE_INDICES), dtype=np.int32).ravel()
            require(not doc['accessors'][primitive['indices']].get('normalized'), 'Normalized triangle indices')
        else:
            indices = np.arange(len(xyz), dtype=np.int32)
        require(len(indices) % 3 == 0 and np.all((indices >= 0) & (indices < len(xyz))), 'Invalid triangle indices')
        require(len(np.unique(indices)) == len(xyz), 'Unreferenced exported owner rows')
        total += len(xyz)
        require(total <= 100000, 'Owner vertex budget exceeded')
        faces_list.append(ids[indices].reshape(-1, 3))
        xyzs.append(xyz); ids_list.append(ids); joints_list.append(joints)
        weights_list.append(weights); dense_list.append(dense)
    xyz, ids, joints, weights, dense = (np.concatenate(rows) for rows in (
        xyzs, ids_list, joints_list, weights_list, dense_list))
    faces = np.concatenate(faces_list)
    require(len(faces) == triangles_expected and len(faces) <= 100000, 'Owner triangle coverage differs')
    require(np.all(np.diff(np.sort(faces, axis=1), axis=1) > 0), 'Repeated source ID within triangle')
    unique, first = np.unique(ids, return_index=True)
    require(np.array_equal(unique, np.arange(count)), 'Missing source IDs')
    require(np.array_equal(xyz, xyz[first][ids]) and np.array_equal(dense, dense[first][ids]),
            'Ambiguous seam duplicate position or named weights')
    return {'positions': xyz[first], 'joints': joints[first], 'weights': weights[first],
            'named': dense[first], 'palette': palette, 'inverse': inverse, 'faces': faces,
            'exported_vertices': total, 'mesh': mesh_index, 'node': node_index}


def _skin(owner, ids, world, *, vectors=None):
    positions = owner['positions'][ids] if vectors is None else vectors
    transforms = world[owner['palette']] @ owner['inverse']
    output = np.zeros_like(positions)
    for column in range(4):
        matrices = transforms[owner['joints'][ids, column]]
        value = np.einsum('nij,nj->ni', matrices[:, :3, :3], positions)
        if vectors is None:
            value += matrices[:, :3, 3]
        output += owner['weights'][ids, column, None] * value
    require(np.isfinite(output).all() and np.max(abs(output), initial=0) <= 100,
            'Skinned points exceed finite world bounds')
    return output


def _states(glb, nodes, joints, expected):
    if any(sampler.get('interpolation', 'LINEAR') not in ('LINEAR', 'STEP')
           for clip in glb.doc.get('animations', []) for sampler in clip.get('samplers', [])):
        raise UnsupportedAttachment('Only LINEAR/STEP animation interpolation is supported')
    parsed = three_d._animation_tracks(glb, nodes, set(joints))
    require(len(parsed) == len(expected) and len({s[0] for s in parsed}) == len(parsed)
            and {s[0] for s in parsed} == set(expected), 'Actual clips differ from expected states')
    result = []
    for name in expected:
        tracks = next(row[1] for row in parsed if row[0] == name)
        start, end = min(t[3][0] for t in tracks), max(t[3][-1] for t in tracks)
        if not 0 < end-start <= 4:
            raise UnsupportedAttachment('Attachment sampling requires clips longer than zero and at most four seconds')
        times = [start+i/24 for i in range(math.floor((end-start)*24)+1)]
        if times[-1] != end:
            times.append(end)
        require(len(times) <= 97, 'Sample count exceeds 97 per state')
        result.append(({'id': name, 'times': times}, tracks))
    return result


def _empty_metrics():
    return {'max_named_weight_error': 0., 'named_weight_failures': 0,
            'max_rest_midpoint_error_m': 0., 'rest_midpoint_failures': 0,
            'max_attachment_error_m': 0., 'attachment_failures': 0,
            'unique_attachment_failed_roots': 0, 'root_pose_checks': 0,
            'max_inset_m': 0., 'body_vertices': 0, 'coat_vertices': 0,
            'exported_vertices': 0, 'triangles': 0}


def _measured(glb, parts, identity, intent, started):
    nodes, parents, names, reachable, bones = _hierarchy(glb)
    owner_names = [identity['source_node'], *[c['node'] for c in identity['coats']]]
    require(len(set(owner_names)) == len(owner_names), 'Source and coat ownership overlap')
    for name in owner_names:
        require(name in names and names[name] in reachable, 'Missing or unreachable named owner')
    mesh_nodes = {i for i, node in enumerate(nodes) if 'mesh' in node}
    require(mesh_nodes == {names[name] for name in owner_names}, 'Unrecorded or missing mesh owner')
    mesh_ids = [nodes[names[name]]['mesh'] for name in owner_names]
    require(len(set(mesh_ids)) == len(mesh_ids), 'Shared mesh ownership is ambiguous')
    owners = [(names[identity['source_node']], identity['source_vertex_count'], identity['source_triangle_count'])]
    owners.extend((names[c['node']], c['vertices'], 4*c['roots']) for c in identity['coats'])
    _admit_owners(glb, owners)
    world = np.asarray([_matrix(x) for x in three_d._world_matrices(nodes, parents)])
    body = _owner(glb, names[identity['source_node']], identity['source_vertex_count'],
                  identity['source_triangle_count'], nodes, world, bones)
    triangle_keys = {tuple(x) for x in _oriented(body['faces']).tolist()}
    states = _states(glb, nodes, bones, intent['state_ids'])
    metrics, prepared = _empty_metrics(), []
    hashes = {name: hashlib.sha256() for name in ('named_weight', 'rest_midpoint', 'animated_attachment')}
    witnesses = {name: [] for name in hashes}
    metrics.update(body_vertices=len(body['positions']), exported_vertices=body['exported_vertices'],
                   triangles=len(body['faces']))

    def failures(kind, errors, threshold, part_index, values, *, state=None, second=None):
        failed = np.flatnonzero(errors > threshold)
        if len(failed):
            worst = int(failed[np.argmax(errors[failed])])
            candidate = {'kind': kind, 'part': part_index, 'root': worst,
                         'source_vertex_ids': values['source_vertex_ids'][worst].tolist(),
                         'error': float(errors[worst]), 'threshold': threshold,
                         'state': state, 'second': second}
            previous = next((w for w in witnesses[kind] if (w['part'], w['root']) == (part_index, worst)), None)
            if previous is None or candidate['error'] > previous['error']:
                rows = [w for w in witnesses[kind] if w is not previous]+[candidate]
                witnesses[kind] = sorted(rows, key=lambda w: -w['error'])[:10]
        hashes[kind].update(np.asarray(errors, dtype='<f8').tobytes())
        return len(failed)

    for part_index, (part, receipt, values) in enumerate(parts):
        require(time.monotonic()-started < 55, 'Attachment audit exceeded wall-time bound')
        info = receipt['export_correspondence']
        coat = _owner(glb, names[info['coat_node']], info['coat_vertex_count'], 4*part['cards'],
                      nodes, world, bones)
        metrics['coat_vertices'] += len(coat['positions'])
        metrics['exported_vertices'] += coat['exported_vertices']
        metrics['triangles'] += len(coat['faces'])
        require(metrics['exported_vertices'] <= 100000 and metrics['triangles'] <= 100000,
                'Aggregate exported geometry exceeds bounds')
        source_ids, bary = values['source_vertex_ids'], values['barycentric']
        require(all(tuple(row) in triangle_keys for row in _oriented(source_ids).tolist()),
                'Declared oriented source triangle is absent from final body')
        roots = np.arange(part['cards'], dtype=np.int32)*6
        inset = values['root_inset'][:, [0, 2, 1]].copy()
        inset[:, 2] *= -1
        metrics['max_inset_m'] = max(metrics['max_inset_m'], float(np.linalg.norm(inset, axis=1).max()))
        expected_weights = np.einsum('ri,rij->rj', bary, body['named'][source_ids])
        weight_errors = np.maximum(abs(coat['named'][roots]-expected_weights).max(axis=1),
                                   abs(coat['named'][roots+1]-expected_weights).max(axis=1))
        metrics['max_named_weight_error'] = max(metrics['max_named_weight_error'], float(weight_errors.max()))
        metrics['named_weight_failures'] += failures('named_weight', weight_errors, GATES['named_weight'], part_index, values)
        expected = np.einsum('ri,rij->rj', bary, body['positions'][source_ids])-inset
        midpoint = (coat['positions'][roots]+coat['positions'][roots+1])*.5
        rest_errors = abs(midpoint-expected).max(axis=1)
        metrics['max_rest_midpoint_error_m'] = max(metrics['max_rest_midpoint_error_m'], float(rest_errors.max()))
        metrics['rest_midpoint_failures'] += failures('rest_midpoint', rest_errors, GATES['rest_midpoint_m'], part_index, values)
        prepared.append((coat, roots, source_ids, bary, inset, values, np.zeros(part['cards'], dtype=bool)))
    for state, tracks in states:
        for second in state['times']:
            require(time.monotonic()-started < 55, 'Attachment audit exceeded wall-time bound')
            posed_world = np.asarray([_matrix(x) for x in three_d._world_matrices(nodes, parents, three_d._pose(tracks, second))])
            for part_index, (coat, roots, source_ids, bary, inset, values, failed_roots) in enumerate(prepared):
                corners = _skin(body, source_ids.ravel(), posed_world).reshape(-1, 3, 3)
                surface = np.einsum('ri,rij->rj', bary, corners)
                midpoint = (_skin(coat, roots, posed_world)+_skin(coat, roots+1, posed_world))*.5
                # Each endpoint's actual skin transforms its half of the intentional inset.
                correction = (_skin(coat, roots, posed_world, vectors=inset)
                              + _skin(coat, roots+1, posed_world, vectors=inset))*.5
                errors = abs(midpoint+correction-surface).max(axis=1)
                failed_roots |= errors > GATES['animated_attachment_m']
                metrics['root_pose_checks'] += len(roots)
                metrics['max_attachment_error_m'] = max(metrics['max_attachment_error_m'], float(errors.max()))
                metrics['attachment_failures'] += failures('animated_attachment', errors, GATES['animated_attachment_m'],
                                                          part_index, values, state=state['id'], second=second)
    metrics['unique_attachment_failed_roots'] = sum(int(row[-1].sum()) for row in prepared)
    selected = [w for rows in witnesses.values() for w in rows]
    return metrics, [s[0] for s in states], selected, {k: h.hexdigest() for k, h in hashes.items()}


def _pass(metrics):
    return (metrics['max_named_weight_error'] <= GATES['named_weight']
            and metrics['max_rest_midpoint_error_m'] <= GATES['rest_midpoint_m']
            and metrics['max_attachment_error_m'] <= GATES['animated_attachment_m']
            and metrics['named_weight_failures'] == metrics['rest_midpoint_failures'] == metrics['attachment_failures'] == 0)


def audit_finish_pack(pack, intent):
    """Malformed custody raises; valid unsupported or measured failures never pass."""
    started, intent = time.monotonic(), _intent(intent)
    _, model_path, parts, inventory, identity = _context(pack, intent, arrays=True)
    report = {'schema_version': 1, 'policy': POLICY, 'status': 'not_applicable', 'passed': True,
              'reason': 'No portable fur requested or reported', 'intent': intent, 'inputs': inventory,
              'implementations': _implementations(), 'roots': sum(p[0]['cards'] for p in parts),
              'expected_state_ids': intent['state_ids'], 'states': [], 'gates': dict(GATES),
              'metrics': _empty_metrics(), 'witnesses': [], 'error_hashes': {},
              'error_encoding': ERROR_ENCODING,
              'limitations': list(LIMITATIONS), 'elapsed_seconds': 0.}
    if parts:
        try:
            if identity == 'historical':
                raise UnsupportedAttachment('Historical portable bindings lack exact export correspondence')
            glb = three_d._GLB(model_path)
            require(glb.hash == intent['model_sha256'], 'Model changed during audit')
            metrics, states, witnesses, error_hashes = _measured(glb, parts, identity, intent, started)
            report.update(status='measured', passed=_pass(metrics), reason='Attachment measured against the final exported body',
                          metrics=metrics, states=states, witnesses=witnesses, error_hashes=error_hashes)
        except UnsupportedAttachment as exc:
            report.update(status='unsupported', passed=False, reason=str(exc))
    else:
        glb = three_d._GLB(model_path)
        require(not any(ATTRIBUTE in primitive.get('attributes', {})
                        for mesh in glb.doc.get('meshes', []) for primitive in mesh.get('primitives', [])),
                'Unreported portable identity attributes')
    report['elapsed_seconds'] = time.monotonic()-started
    # Rebind files after measurement; a concurrent replacement cannot acquire a valid receipt.
    require(_context(pack, intent, arrays=False)[3] == inventory, 'Attachment inputs changed during measurement')
    return validate_report(pack, intent, report)


def validate_report(pack, intent, report):
    """Validate cached custody and gate arithmetic without evaluating poses again."""
    intent = _intent(intent)
    require(type(report) is dict and len(json.dumps(report, allow_nan=False).encode()) <= MAX_JSON,
            'Attachment report exceeds JSON bound')
    fields = {'schema_version', 'policy', 'status', 'passed', 'reason', 'intent', 'inputs', 'implementations',
              'roots', 'expected_state_ids', 'states', 'gates', 'metrics', 'witnesses', 'error_hashes',
              'error_encoding', 'limitations', 'elapsed_seconds'}
    require(set(report) == fields and report['schema_version'] == 1 and report['policy'] == POLICY
            and type(report['passed']) is bool and report['intent'] == intent
            and report['expected_state_ids'] == intent['state_ids'] and report['gates'] == GATES,
            'Attachment report schema/intent/gates differ')
    require(type(report['reason']) is str and 0 < len(report['reason']) <= 512
            and report['limitations'] == LIMITATIONS and report['error_encoding'] == ERROR_ENCODING,
            'Invalid attachment scope')
    require(type(report['elapsed_seconds']) in (int, float) and 0 <= report['elapsed_seconds'] <= 60,
            'Invalid attachment elapsed time')
    _, _, parts, inventory, identity = _context(pack, intent, arrays=False)
    roots = sum(p[0]['cards'] for p in parts)
    require(type(report['roots']) is int and report['roots'] == roots and report['inputs'] == inventory
            and report['implementations'] == _implementations(), 'Attachment report custody changed')
    status = report['status']
    require(status in ('measured', 'not_applicable', 'unsupported'), 'Invalid attachment status')
    if status != 'measured':
        require(report['passed'] is (status == 'not_applicable')
                and (not parts if status == 'not_applicable' else bool(parts))
                and report['metrics'] == _empty_metrics() and report['states'] == []
                and report['witnesses'] == [] and report['error_hashes'] == {}, 'Invalid nonmeasured result')
        return report
    require(parts and identity != 'historical', 'Measured result lacks v2 correspondence')
    states = report['states']
    require(type(states) is list and len(states) == len(intent['state_ids']), 'Invalid sampled states')
    samples = 0
    for state, name in zip(states, intent['state_ids']):
        require(type(state) is dict and set(state) == {'id', 'times'} and state['id'] == name, 'Invalid sampled state')
        times = state['times']
        require(type(times) is list and 2 <= len(times) <= 97
                and all(type(t) in (int, float) and math.isfinite(t) and 0 <= t <= 120 for t in times)
                and all(a < b for a, b in zip(times, times[1:])) and 0 < times[-1]-times[0] <= 4,
                'Invalid sample schedule')
        expected = [times[0]+i/24 for i in range(math.floor((times[-1]-times[0])*24)+1)]
        if expected[-1] != times[-1]:
            expected.append(times[-1])
        require(times == expected, 'Sampling is not exact 24 fps plus endpoint')
        samples += len(times)
    metrics = report['metrics']
    require(type(metrics) is dict and set(metrics) == set(_empty_metrics()), 'Invalid metric fields')
    for key, value in metrics.items():
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1e8, 'Invalid numeric metric')
        if not key.startswith('max_'):
            require(type(value) is int, 'Invalid count metric')
    require(metrics['root_pose_checks'] == roots*samples and samples <= 776
            and 1 <= metrics['body_vertices'] <= 100000 and metrics['coat_vertices'] == 6*roots
            and metrics['body_vertices']+metrics['coat_vertices'] <= metrics['exported_vertices'] <= 100000
            and 1 <= metrics['triangles'] <= 100000 and metrics['max_inset_m'] <= .0005+1e-12,
            'Metric coverage/bounds differ')
    pairs = [('max_named_weight_error', 'named_weight_failures', GATES['named_weight'], roots),
             ('max_rest_midpoint_error_m', 'rest_midpoint_failures', GATES['rest_midpoint_m'], roots),
             ('max_attachment_error_m', 'attachment_failures', GATES['animated_attachment_m'], roots*samples)]
    for maximum, count, gate, limit in pairs:
        require(metrics[count] <= limit and (metrics[maximum] > gate) == (metrics[count] > 0),
                'Failure count differs from measured maximum')
    require(0 <= metrics['unique_attachment_failed_roots'] <= min(roots, metrics['attachment_failures'])
            and bool(metrics['unique_attachment_failed_roots']) == bool(metrics['attachment_failures']),
            'Invalid failed-root count')
    require(report['passed'] == _pass(metrics), 'Cached passed flag contradicts numeric failures')
    require(type(report['error_hashes']) is dict and set(report['error_hashes']) == {
        'named_weight', 'rest_midpoint', 'animated_attachment'}, 'Missing measured-error hashes')
    for value in report['error_hashes'].values():
        _hash(value)
    witnesses = report['witnesses']
    require(type(witnesses) is list and len(witnesses) <= MAX_WITNESSES, 'Invalid bounded witnesses')
    kind_gates = {'named_weight': GATES['named_weight'], 'rest_midpoint': GATES['rest_midpoint_m'],
                  'animated_attachment': GATES['animated_attachment_m']}
    for witness in witnesses:
        require(type(witness) is dict and set(witness) == {'kind', 'part', 'root', 'source_vertex_ids', 'error',
                'threshold', 'state', 'second'} and witness.get('kind') in kind_gates, 'Invalid witness fields')
        part, root_index = witness['part'], witness['root']
        require(type(part) is int and 0 <= part < len(parts) and type(root_index) is int
                and 0 <= root_index < parts[part][0]['cards'], 'Invalid witness root')
        ids = witness['source_vertex_ids']
        require(type(ids) is list and len(ids) == 3 and len(set(ids)) == 3
                and all(type(i) is int and 0 <= i < identity['source_vertex_count'] for i in ids), 'Invalid witness source IDs')
        require(witness['threshold'] == kind_gates[witness['kind']]
                and type(witness['error']) in (int, float) and math.isfinite(witness['error'])
                and witness['error'] > witness['threshold'], 'Invalid witness error')
        if witness['kind'] == 'animated_attachment':
            require(any(state['id'] == witness['state'] and witness['second'] in state['times'] for state in states),
                    'Witness is outside sampled poses')
        else:
            require(witness['state'] is None and witness['second'] is None, 'Rest witness has a pose')
    for kind, (_, count, _, _) in zip(kind_gates, pairs):
        require(bool(metrics[count]) == any(w['kind'] == kind for w in witnesses), 'Failure lacks a witness')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pack', type=Path)
    parser.add_argument('intent', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = audit_finish_pack(args.pack, safety.read_json(args.intent))
    safety.write_json(args.report, report)
    print(json.dumps({'status': report['status'], 'passed': report['passed'], 'roots': report['roots']}))


if __name__ == '__main__':
    main()
