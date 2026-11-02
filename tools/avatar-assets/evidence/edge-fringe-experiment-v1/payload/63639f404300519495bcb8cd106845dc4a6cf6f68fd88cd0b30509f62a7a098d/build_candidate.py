"""One bounded, data-only edge-fringe derivative; no quality approval or defaults."""

from copy import deepcopy
import gc
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE/'sources'), str(ROOT)]
from edge_fringe import build_edge_fringe, two_tip_atlas
from groom_flow import prepare_flow, evaluate_prepared_flow
from avatar_assets.normalize_generated import _array
from avatar_assets.provider_model import preflight
from avatar_assets.safety import require, sha256, write_json
from avatar_assets.three_d import _GLB, _world_matrices, _animation_tracks, _pose, validate_glb, validate_render_attributes


def read_array(glb, index, width, kind, maximum=100000):
    spec = glb.doc['accessors'][index]
    require(not spec.get('normalized'), 'This prototype requires unnormalized stored attributes')
    require(spec.get('componentType') in kind, 'Unexpected attribute component type')
    return _array(glb, index, width, spec['componentType'], maximum)


def skin(points, joints, weights, matrices):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    result = np.zeros((len(points), 3), dtype=np.float64)
    for slot in range(4):
        result += np.einsum('nij,nj->ni', matrices[joints[:, slot]], homogeneous)[:, :3]*weights[:, slot, None]
    return result


def main():
    started = time.monotonic()
    prepared = json.loads((HERE/'prepared.json').read_text())
    for row in prepared['inputs']:
        require(sha256(ROOT/row['path']) == row['sha256'], 'Pinned input changed: '+row['path'])
    config = json.loads((HERE/'candidate.json').read_text())
    source, out = ROOT/config['source'], ROOT/config['output']
    require(out.is_dir() and not any(out.iterdir()), 'Candidate output must be a fresh empty monitored directory')
    glb = _GLB(source)
    original = glb.doc
    require(glb.hash == config['source_sha256'], 'Source model changed')
    finish = json.loads((ROOT/config['finish_config']).read_text())
    report = json.loads((ROOT/config['finish_report']).read_text())
    parents = {child: i for i, node in enumerate(original['nodes']) for child in node.get('children', [])}
    world = _world_matrices(original['nodes'], parents)
    closures = {name for row in finish.get('normal_continuity', []) for name in row['materials']}
    selected_materials = {row['material']: row for row in report['fur_undercoat']['materials']
                          if row['source_material'] not in closures}
    require(selected_materials and len(selected_materials) <= 4, 'No bounded recorded fur selection')
    selected = []
    for ni, node in enumerate(original['nodes']):
        if 'mesh' not in node:
            continue
        for pi, primitive in enumerate(original['meshes'][node['mesh']]['primitives']):
            name = original['materials'][primitive['material']]['name']
            if name in selected_materials:
                selected.append((ni, node['mesh'], pi, primitive, name))
    require(len(selected) == 1, 'Prototype supports one exact recorded fur primitive; no implicit merging')
    ni, mi, pi, primitive, material_name = selected[0]
    node = original['nodes'][ni]
    require('skin' in node and np.allclose(np.asarray(world[ni]).reshape(4, 4).T, np.eye(4), atol=1e-7, rtol=0),
            'Prototype requires a skinned identity-world emitter in meters')
    contract = json.loads((ROOT/config['rig_contract']).read_text())
    require(contract['units'] == 'meters' and contract['axes'] == {'forward': '+Z', 'up': '+Y', 'handedness': 'right'},
            'Require explicit meter-scale glTF +Y-up/+Z-forward contract')
    attrs = primitive['attributes']
    points = read_array(glb, attrs['POSITION'], 3, (5126,))
    normals = read_array(glb, attrs['NORMAL'], 3, (5126,)).astype(np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    require(np.max(np.abs(lengths-1)) < .001, 'Source normals are not unit length')
    normals /= lengths[:, None]
    uv = read_array(glb, attrs['TEXCOORD_0'], 2, (5126,))
    faces = read_array(glb, primitive['indices'], 1, (5121, 5123, 5125), 300000).reshape(-1, 3)
    joints = read_array(glb, attrs['JOINTS_0'], 4, (5121, 5123))
    weights = read_array(glb, attrs['WEIGHTS_0'], 4, (5126,))
    require(len(faces) == selected_materials[material_name]['selected_faces'], 'Recorded fur face count changed')
    with np.load(ROOT/config['safe_edges'], allow_pickle=False) as data:
        safe_edges = data[f'primitive_{pi}_edges']
    require(safe_edges.ndim == 2 and safe_edges.shape[1] == 2 and len(safe_edges) <= 150000,
            'Safe-edge census exceeds bounds')
    safe_codes = np.sort(safe_edges[:, 0].astype(np.int64)*len(points)+safe_edges[:, 1])
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    codes = edges[:, 0].astype(np.int64)*len(points)+edges[:, 1]
    unique, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    unsafe_shared = (counts[inverse] == 2)&~np.isin(codes, safe_codes, assume_unique=False)
    excluded_faces = unsafe_shared.reshape(-1, 3).any(axis=1)
    source_material = original['materials'][primitive['material']]
    pbr = source_material['pbrMetallicRoughness']
    texture = pbr['baseColorTexture']
    require(texture.get('texCoord', 0) == 0 and source_material.get('alphaMode', 'OPAQUE') == 'OPAQUE',
            'Root color needs opaque explicit UV0')
    image_record = original['images'][original['textures'][texture['index']]['source']]
    view = original['bufferViews'][image_record['bufferView']]
    raw = glb.binary[view.get('byteOffset', 0):view.get('byteOffset', 0)+view['byteLength']]
    with Image.open(io.BytesIO(raw)) as image:
        require(image.width*image.height <= 16777216 and image.size[0] <= 4096 and image.size[1] <= 4096
                and getattr(image, 'n_frames', 1) == 1, 'Root image exceeds bounds')
        pixels = np.asarray(image.convert('RGB')).copy()
    texture_hash = hashlib.sha256(raw).hexdigest()
    del raw
    base_points = []
    for part in original['meshes'][mi]['primitives']:
        base_points.append(read_array(glb, part['attributes']['POSITION'], 3, (5126,)))
    all_points = np.concatenate(base_points)
    native_points = all_points[:, [0, 2, 1]]*np.array([1., -1., 1.])
    low, high = native_points.min(axis=0), native_points.max(axis=0)
    flow = prepare_flow(low, high-low, finish['groom']['direction'], finish['groom']['comb'], finish['groom_flow'])
    selected_native = points[:, [0, 2, 1]]*np.array([1., -1., 1.])
    native_directions = np.asarray([evaluate_prepared_flow(point, flow)[0] for point in selected_native])
    directions = native_directions[:, [0, 2, 1]]*np.array([1., 1., -1.])
    base_vertices = sum(len(value) for value in base_points)
    base_triangles = sum(original['accessors'][part['indices']]['count']//3 for part in original['meshes'][mi]['primitives'])
    del all_points, native_points, base_points, selected_native, native_directions
    result = build_edge_fringe(points, faces, normals[faces], uv[faces], np.zeros(len(faces), dtype=np.int32),
        np.zeros(len(faces), dtype=np.int32), np.ones(len(faces), dtype=bool),
        excluded_faces, np.zeros(len(faces), dtype=bool), joints, weights,
        directions, {0: {'pixels': pixels, 'factor': np.asarray(pbr.get('baseColorFactor', [1, 1, 1, 1])[:3], dtype=float)}},
        np.asarray(config['design_views_gltf'], dtype=float), config['settings'], source_export_vertices=base_vertices)
    del pixels, directions
    gc.collect()
    root_codes = result['root_vertex_ids'][:, 0].astype(np.int64)*len(points)+result['root_vertex_ids'][:, 1]
    require(np.isin(root_codes, safe_codes).all(), 'Chosen roots violate the independently measured safe-edge census')

    skin_spec = original['skins'][node['skin']]
    require(1 <= len(skin_spec['joints']) <= 64 and joints.max() < len(skin_spec['joints']), 'Invalid skeleton binding')
    inverse_bind = np.asarray(glb.accessor(skin_spec['inverseBindMatrices'], 'MAT4', (5126,))).reshape(-1, 4, 4).transpose(0, 2, 1)
    tracks = _animation_tracks(glb, original['nodes'], set(skin_spec['joints']))
    require(len(tracks) == 1 and tracks[0][0] == config['action_id'], 'Unexpected source action')
    ids = result['root_vertex_ids'].reshape(-1)
    fin_count = result['report']['selected_fins']
    root_indices = (np.arange(fin_count)[:, None]*4+np.array([0, 1])).reshape(-1)
    posed_reports, maximum_root_error = [], 0.
    neutral = None
    for second in config['seconds']:
        transforms = _world_matrices(original['nodes'], parents, _pose(tracks[0][1], second))
        matrices = np.asarray([transforms[i] for i in skin_spec['joints']]).reshape(-1, 4, 4).transpose(0, 2, 1)@inverse_bind
        expected = skin(points[ids], joints[ids], weights[ids], matrices)
        posed = skin(result['positions'], result['joints'], result['weights'], matrices)
        error = float(np.max(np.linalg.norm(posed[root_indices]-expected, axis=1)))
        maximum_root_error = max(maximum_root_error, error)
        require(error < 1e-5, 'Fin roots detach from their exact source endpoints')
        quads = posed.reshape(-1, 4, 3)
        extension = np.linalg.norm(quads[:, 2:]-quads[:, :2], axis=2)
        require(float(extension.max()) <= config['settings']['length_m']*1.001, 'Skinned fin extension exceeds rigid-bind bound')
        if neutral is None:
            neutral = posed.copy()
        posed_reports.append({'second': second, 'maximum_root_error_m': error,
                              'maximum_extension_m': float(extension.max()),
                              'maximum_motion_from_first_pose_m': float(np.max(np.linalg.norm(posed-neutral, axis=1)))})
    require(max(row['maximum_motion_from_first_pose_m'] for row in posed_reports) > .001, 'No actual weighted fin motion')

    doc = deepcopy(original)
    old_materials = {row['material']['material'] for row in report['portable_fur']}
    removed_meshes = [i for i, mesh in enumerate(original['meshes']) if mesh['primitives'] and
                      all(original['materials'][p['material']]['name'] in old_materials for p in mesh['primitives'])]
    require(removed_meshes == list(range(mi+1, len(original['meshes']))) and len(removed_meshes) == len(old_materials),
            'Only an unreferenced suffix of exact recorded coat meshes may be removed')
    removed_nodes = []
    for i, record in enumerate(doc['nodes']):
        if record.get('mesh') in removed_meshes:
            require(not record.get('children') and i not in skin_spec['joints'], 'Old coat node has structural dependents')
            require(not any(channel['target']['node'] == i for animation in original['animations'] for channel in animation['channels']),
                    'Old coat node is directly animated')
            record.pop('mesh')
            record.pop('skin', None)
            removed_nodes.append(i)
    doc['meshes'] = doc['meshes'][:mi+1]
    binary = bytearray(glb.binary)

    def append(raw, target=None):
        binary.extend(b'\0'*(-len(binary)%4))
        record = {'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(raw)}
        if target is not None:
            record['target'] = target
        binary.extend(raw)
        doc['bufferViews'].append(record)
        return len(doc['bufferViews'])-1

    def accessor(value, kind, component, target):
        value = np.asarray(value, dtype={5126: '<f4', 5123: '<u2', 5125: '<u4'}[component])
        record = {'bufferView': append(value.tobytes(), target), 'componentType': component, 'count': len(value), 'type': kind}
        if kind == 'VEC3':
            record.update(min=value.min(axis=0).tolist(), max=value.max(axis=0).tolist())
        doc['accessors'].append(record)
        return len(doc['accessors'])-1

    new_attrs = {name: accessor(result[key], kind, component, 34962) for name, key, kind, component in (
        ('POSITION', 'positions', 'VEC3', 5126), ('NORMAL', 'normals', 'VEC3', 5126),
        ('TEXCOORD_0', 'uv', 'VEC2', 5126), ('COLOR_0', 'colors', 'VEC3', 5126),
        ('JOINTS_0', 'joints', 'VEC4', 5123), ('WEIGHTS_0', 'weights', 'VEC4', 5126))}
    new_indices = accessor(result['triangles'].reshape(-1, 1), 'SCALAR', 5125, 34963)
    atlas = two_tip_atlas(config['settings']['seed'])
    with (out/'two-tip-atlas.png').open('xb') as file:
        Image.fromarray(atlas).save(file, format='PNG')
    doc['images'].append({'bufferView': append((out/'two-tip-atlas.png').read_bytes()), 'mimeType': 'image/png', 'name': 'EdgeFringeTwoTips'})
    doc.setdefault('samplers', []).append({'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071})
    doc['textures'].append({'source': len(doc['images'])-1, 'sampler': len(doc['samplers'])-1})
    doc['materials'].append({'name': 'ExperimentalEdgeFringe', 'doubleSided': True, 'alphaMode': 'MASK', 'alphaCutoff': .18,
        'pbrMetallicRoughness': {'baseColorFactor': [1, 1, 1, 1], 'baseColorTexture': {'index': len(doc['textures'])-1, 'texCoord': 0},
                               'metallicFactor': 0, 'roughnessFactor': .92}})
    new_primitive = {'attributes': new_attrs, 'indices': new_indices, 'mode': 4, 'material': len(doc['materials'])-1}
    doc['meshes'][mi]['primitives'].append(new_primitive)
    doc['buffers'][0]['byteLength'] = len(binary)
    doc['asset']['generator'] = 'Local experimental edge-fringe derivative; source body, rig and motion unchanged'
    encoded = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' '*(-len(encoded)%4)
    binary.extend(b'\0'*(-len(binary)%4))
    total = 28+len(encoded)+len(binary)
    require(total <= 64*1024*1024 and base_vertices+len(result['positions']) <= 100000
            and base_triangles+len(result['triangles']) <= 100000, 'Candidate exceeds unchanged HD bounds')
    with (out/'model.glb').open('xb') as file:
        file.write(struct.pack('<4sII', b'glTF', 2, total))
        file.write(struct.pack('<II', len(encoded), 0x4E4F534A))
        file.write(encoded)
        file.write(struct.pack('<II', len(binary), 0x004E4942))
        file.write(binary)
    require(binary[:len(glb.binary)] == glb.binary, 'Original embedded bytes changed')
    del binary
    require(doc['meshes'][mi]['primitives'][:-1] == original['meshes'][mi]['primitives'], 'Base primitive descriptions changed')
    for name in ('accessors', 'bufferViews', 'images', 'textures', 'materials', 'samplers'):
        require(doc[name][:len(original.get(name, []))] == original.get(name, []), 'Original descriptors changed: '+name)
    for name in ('skins', 'animations', 'scenes', 'scene'):
        require(doc.get(name) == original.get(name), 'Original rig/action/scene changed: '+name)
    for i, record in enumerate(original['nodes']):
        expected = deepcopy(record)
        if i in removed_nodes:
            expected.pop('mesh')
            expected.pop('skin', None)
        require(doc['nodes'][i] == expected, 'Unexpected node mutation')
    arrays = {name: result[name] for name in ('root_vertex_ids', 'root_face_ids', 'root_corner_ids', 'root_source_uv',
        'root_centers', 'outward_normals', 'fin_directions')}
    np.savez_compressed(out/'source-correspondence.npz', **arrays)
    write_json(out/'rig-contract.json', contract)
    write_json(out/'candidate.json', config)
    gc.collect()
    write_json(out/'preflight.json', preflight(out/'model.glb'))
    write_json(out/'render-attributes.json', validate_render_attributes(out/'model.glb'))
    verified_rig = validate_glb(out/'model.glb', contract)
    require(verified_rig['motion']['animation_verified'], 'Independent GLB validator did not observe weighted motion')
    write_json(out/'rig-verification.json', verified_rig)
    write_json(out/'build-report.json', {'status': 'cpu_candidate_pending_independent_reload', 'quality_approved': False,
        'source_glb_sha256': glb.hash, 'candidate_sha256': sha256(out/'model.glb'), 'model_bytes': total,
        'source_selection': {'node': ni, 'mesh': mi, 'primitive': pi, 'material': material_name,
                             'recorded_closure_source_materials_excluded': sorted(closures), 'exact_root_texture_sha256': texture_hash,
                             'conservatively_excluded_faces_near_unsafe_shared_edges': int(excluded_faces.sum()),
                             'all_chosen_edges_in_independent_census': True},
        'fin_report': result['report'], 'base_vertices': base_vertices, 'base_triangles': base_triangles,
        'exported_vertices': base_vertices+len(result['positions']), 'exported_triangles': base_triangles+len(result['triangles']),
        'retained_original_binary_bytes': len(glb.binary), 'original_binary_sha256': hashlib.sha256(glb.binary).hexdigest(),
        'original_base_and_rig_and_actions_exact': True, 'detached_coat_nodes': removed_nodes, 'removed_coat_meshes': removed_meshes,
        'weighted_motion': {'maximum_root_error_m': maximum_root_error, 'poses': posed_reports, 'all_fins_tested': fin_count},
        'flow': {'original_normalized_bounds_blender_z_up': {'min': low.tolist(), 'max': high.tolist()},
                 'regional_direction_reused': True, 'length_and_lean': 'Fixed experimental 8 mm / 0.15, not native regional lengths/comb'},
        'limitations': ['Analytic endpoint attachment is not appearance or full collision testing.',
            'Existing hidden-surface/contact defects are preserved, not repaired or concealed by a quality claim.',
            'HD normal maps/UV1 and vertex budget remain unsupported by the current Weft consumer.',
            'No native groom, hair physics, fresh generation, dense-fur equivalence or seamless-loop certification.'],
        'elapsed_seconds': time.monotonic()-started, 'new_generation_calls': 0})
    for row in prepared['inputs']:
        require(sha256(ROOT/row['path']) == row['sha256'], 'Pinned input changed after build: '+row['path'])
    print(json.dumps({'status': 'cpu_candidate_pending_independent_reload', 'fins': fin_count,
                      'model_bytes': total, 'maximum_root_error_m': maximum_root_error}))


if __name__ == '__main__':
    main()
