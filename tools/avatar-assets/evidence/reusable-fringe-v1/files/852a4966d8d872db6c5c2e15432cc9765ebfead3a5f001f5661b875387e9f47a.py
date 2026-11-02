"""Numeric original-to-native-export correspondence and weighted-pose audit; no GLB edits."""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources/host'))
from avatar_assets.provider_model import preflight
from avatar_assets.three_d import (_GLB, _animation_tracks, _pose, _world_matrices,
                                  validate_glb, validate_render_attributes)

CONFIG = json.loads((HERE/'config.json').read_text())
OUT = ROOT/CONFIG['output']
AUDIT = OUT/'audit'
START = time.monotonic()


def require(condition, message):
    if not condition:
        raise ValueError(message)
    if time.monotonic()-START > 55:
        raise ValueError('Export audit internal55second deadline')


def save(name, value):
    with (AUDIT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def parents(doc):
    return {child: index for index, node in enumerate(doc['nodes']) for child in node.get('children', [])}


def array(glb, index, shape, types, indices=False):
    return np.asarray(glb.accessor(index, shape, types, count_limit=300000 if indices else 100000))


def records(glb, bone_names):
    result = {}
    for node in glb.doc['nodes']:
        if 'mesh' not in node:
            continue
        skin = glb.doc['skins'][node['skin']]
        names = [glb.doc['nodes'][index]['name'] for index in skin['joints']]
        require(set(names) == set(bone_names), 'Exported skin changed original bone names')
        for primitive in glb.doc['meshes'][node['mesh']]['primitives']:
            material = glb.doc['materials'][primitive['material']]['name']
            require(material not in result, 'Ambiguous repeated material primitive')
            attrs = primitive['attributes']
            position = array(glb, attrs['POSITION'], 'VEC3', (5126,))
            normal = array(glb, attrs['NORMAL'], 'VEC3', (5126,))
            uv = array(glb, attrs['TEXCOORD_0'], 'VEC2', (5126,))
            ids = array(glb, attrs['JOINTS_0'], 'VEC4', (5121, 5123)).astype(np.int32)
            weights = array(glb, attrs['WEIGHTS_0'], 'VEC4', (5121, 5123, 5126))
            dense = np.zeros((len(position), len(bone_names)))
            for native, name in enumerate(names):
                dense[:, bone_names.index(name)] = np.where(ids == native, weights, 0).sum(axis=1)
            result[material] = {'positions': position, 'normals': normal, 'uv': uv, 'dense_weights': dense,
                'joints': ids, 'weights': weights, 'skin': skin['joints'],
                'inverse': array(glb, skin['inverseBindMatrices'], 'MAT4', (5126,)).reshape(-1, 4, 4).transpose(0, 2, 1),
                'faces': array(glb, primitive['indices'], 'SCALAR', (5121, 5123, 5125), True).reshape(-1, 3).astype(np.int32),
                'attributes': sorted(attrs), 'primitive': primitive}
    return result


OFFSETS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)


def correspondence(source, target):
    buckets = defaultdict(list)
    grid = np.floor(source['positions']/1e-5).astype(np.int64)
    for index, cell in enumerate(grid):
        buckets[tuple(cell)].append(index)
    tolerance = CONFIG['numeric_tolerances']
    fields = [('positions', tolerance['source_position_m']), ('uv', tolerance['source_uv']),
              ('normals', tolerance['source_normal_component']), ('dense_weights', tolerance['source_weight'])]
    maxima = {name: 0. for name, _ in fields}

    def match(values, measure):
        matches = np.empty(len(values['positions']), dtype=np.int32)
        for index, point in enumerate(values['positions']):
            if index % 256 == 0:
                require(True, 'deadline')
            base = np.floor(point/1e-5).astype(np.int64)
            candidate = [i for offset in OFFSETS for i in buckets.get(tuple(base+offset), ())]
            require(0 < len(candidate) <= 256, 'Missing or oversized position correspondence bucket')
            candidate = np.asarray(candidate, dtype=np.int32)
            for name, bound in fields:
                distance = np.max(abs(source[name][candidate]-values[name][index]), axis=1)
                candidate = candidate[distance <= bound]
            require(len(candidate) > 0, 'Base position/UV/normal/skin correspondence failed')
            chosen = int(candidate.min())
            matches[index] = chosen
            if measure:
                for name, _ in fields:
                    maxima[name] = max(maxima[name], float(np.max(abs(source[name][chosen]-values[name][index]))))
        return matches

    source_ids, target_ids = match(source, False), match(target, True)
    require(set(source_ids) == set(target_ids), 'A source vertex attribute group was omitted')

    def triangles(record, ids):
        result = Counter()
        for face in ids[record['faces']]:
            a, b, c = map(int, face)
            result[min((a, b, c), (b, c, a), (c, a, b))] += 1
        return result

    require(triangles(source, source_ids) == triangles(target, target_ids), 'Oriented base triangle multiset changed')
    return target_ids, maxima


def skin_positions(record, world):
    matrix = np.array([np.asarray(world[index]).reshape(4, 4).T for index in record['skin']]) @ record['inverse']
    points = np.column_stack((record['positions'], np.ones(len(record['positions']))))
    result = np.zeros((len(points), 3))
    for column in range(4):
        moved = np.einsum('nij,nj->ni', matrix[record['joints'][:, column], :3], points)
        result += moved*record['weights'][:, column, None]
    return result


def helper_record(source, bone_names):
    path = OUT/'fringe-correspondence.npz'
    require(path.stat().st_size <= 8*1024**2, 'Fringe correspondence archive exceeds bound')
    import zipfile
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) <= 20 and sum(row.file_size for row in archive.infolist()) <= 16*1024**2,
                'Fringe correspondence decoded-array bound')
    with np.load(path, allow_pickle=False) as saved:
        positions, normals = saved['positions'].copy(), saved['normals'].copy()
        triangles = saved['triangles'].copy()
        native_map = saved['native_vertex_to_helper_vertex'].copy()
        uv, ids, weights = saved['uv_gltf'].copy(), saved['joints'].copy(), saved['weights'].copy()
    require(1000*4 <= len(positions) <= 4800*4 and len(positions) % 4 == 0,
            'Saved fin geometry count differs from fixed request')
    require(np.array_equal(native_map, triangles.reshape(-1)) and len(native_map) == len(positions)*3,
            'Independent native corners do not exactly map to helper triangles')
    report = json.loads((OUT/'fringe-report.json').read_text())
    helper_names = report['bone_order']
    skin = source.doc['skins'][0]
    source_names = [source.doc['nodes'][index]['name'] for index in skin['joints']]
    require(set(helper_names) == set(source_names), 'Helper bone map differs from source skin')
    remap = np.array([source_names.index(name) for name in helper_names])
    dense = np.zeros((len(positions), len(bone_names)))
    for index, name in enumerate(helper_names):
        dense[:, bone_names.index(name)] = np.where(ids == index, weights, 0).sum(axis=1)
    # Helper uses applied Blender axes; the actual exporter converts to glTF Y-up.
    positions, normals = positions[:, [0, 2, 1]].copy(), normals[:, [0, 2, 1]].copy()
    positions[:, 2] *= -1; normals[:, 2] *= -1
    return {'positions': positions, 'normals': normals, 'uv': uv, 'dense_weights': dense,
            'joints': remap[ids], 'weights': weights, 'skin': skin['joints'], 'faces': triangles,
            'inverse': array(source, skin['inverseBindMatrices'], 'MAT4', (5126,)).reshape(-1, 4, 4).transpose(0, 2, 1)}, {
        'helper_vertices': len(positions), 'native_independent_corners': len(native_map),
        'exact_native_corner_map': True, 'native_map_sha256': hashlib.sha256(native_map.tobytes()).hexdigest()}


def fringe_material(glb, record):
    primitive = record['primitive']
    attrs = primitive['attributes']
    material = glb.doc['materials'][primitive['material']]
    pbr = material['pbrMetallicRoughness']
    require(not any(name.startswith('COLOR_') for name in attrs), 'Source atlas was multiplied by a vertex color')
    require('TANGENT' not in attrs and 'normalTexture' not in material,
            'Only unused non-normalmapped fringe tangents may be omitted')
    require(material.get('alphaMode', 'OPAQUE') == 'MASK' and abs(material.get('alphaCutoff', .5)-.18) < 1e-6
            and material.get('doubleSided', False) is False, 'Actual fringe mask/culling differs from policy')
    require(pbr.get('baseColorFactor', [1, 1, 1, 1]) == [1, 1, 1, 1]
            and pbr.get('metallicFactor', 1) == 0 and abs(pbr.get('roughnessFactor', 1)-.92) < 1e-6,
            'Actual fringe color/roughness factor changed')
    texture = pbr['baseColorTexture']
    require(texture.get('texCoord', 0) == 0 and not texture.get('extensions'), 'Unexpected fringe texture UV route')
    texture_def = glb.doc['textures'][texture['index']]
    sampler = glb.doc['samplers'][texture_def['sampler']]
    require(sampler == {'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071},
            'Actual fringe atlas filtering/addressing differs from policy')
    return {'material': material['name'], 'alphaMode': 'MASK', 'alphaCutoff': .18, 'doubleSided': False,
            'normalTexture': False, 'unused_TANGENT_binding': False, 'vertex_color_multiplier': False,
            'sampler': sampler, 'scope': 'Actual exported material; not inferred from native node settings'}


def main():
    require(AUDIT.is_dir() and not list(AUDIT.iterdir()), 'Audit output must be an empty new directory')
    source = _GLB(HERE/CONFIG['baseline'])
    target = _GLB(OUT/'model.glb')
    source_skin = source.doc['skins'][0]
    bone_names = [source.doc['nodes'][index]['name'] for index in source_skin['joints']]
    original, exported = records(source, bone_names), records(target, bone_names)
    names = CONFIG['selected_materials']+CONFIG['excluded_retained_materials']
    require(set(original) == set(names) and set(exported) == set(names)|{'SourceColoredFringe'}, 'Unexpected exported material set')
    save('fringe-material.json', fringe_material(target, exported['SourceColoredFringe']))
    mapping, comparison = {}, []
    for name in names:
        mapping[name], errors = correspondence(original[name], exported[name])
        comparison.append({'material': name, 'source_vertices': len(original[name]['positions']),
                           'exported_vertices': len(exported[name]['positions']), 'max_errors': errors,
                           'oriented_triangle_multiset_equal': True})
    save('base-correspondence.json', {'status': 'passed_numeric_correspondence', 'tolerances': CONFIG['numeric_tolerances'],
         'source_sha256': source.hash, 'model_sha256': target.hash, 'materials': comparison,
         'limits': ['Attribute-equivalent source aliases share canonical IDs; exporter index order is not claimed identical.']})
    np.savez_compressed(AUDIT/'base-correspondence.npz', **mapping)
    original['SourceColoredFringe'], corner_proof = helper_record(source, bone_names)
    mapping['SourceColoredFringe'], fin_errors = correspondence(original['SourceColoredFringe'], exported['SourceColoredFringe'])
    save('fringe-export-correspondence.json', {**corner_proof, 'max_errors': fin_errors,
        'exported_vertices': len(exported['SourceColoredFringe']['positions']),
        'oriented_helper_triangle_multiset_equal': True,
        'scope': 'Actual native12-corner topology and final exporter data checked against preserved helper arrays'})
    np.savez_compressed(AUDIT/'fringe-export-correspondence.npz', exported_vertex_to_helper_vertex=mapping['SourceColoredFringe'])
    src_tracks = {name: tracks for name, tracks, _ in _animation_tracks(source, source.doc['nodes'], set(source_skin['joints']))}
    dst_joints = {index for skin in target.doc['skins'] for index in skin['joints']}
    dst_tracks = {name: tracks for name, tracks, _ in _animation_tracks(target, target.doc['nodes'], dst_joints)}
    require(set(src_tracks) == set(dst_tracks) == set(CONFIG['expected_actions']), 'Original four clips not preserved')
    motion = []
    for name in sorted(src_tracks):
        source_range = [min(track[3][0] for track in src_tracks[name]), max(track[3][-1] for track in src_tracks[name])]
        target_range = [min(track[3][0] for track in dst_tracks[name]), max(track[3][-1] for track in dst_tracks[name])]
        require(np.max(abs(np.array(source_range)-target_range)) <= 1e-6, 'Clip timing changed')
        times = np.linspace(*source_range, 9)
        maximum = 0.
        for seconds in times:
            src_world = _world_matrices(source.doc['nodes'], parents(source.doc), _pose(src_tracks[name], float(seconds)))
            dst_world = _world_matrices(target.doc['nodes'], parents(target.doc), _pose(dst_tracks[name], float(seconds)))
            for material in names+['SourceColoredFringe']:
                before = skin_positions(original[material], src_world)[mapping[material]]
                after = skin_positions(exported[material], dst_world)
                maximum = max(maximum, float(np.linalg.norm(before-after, axis=1).max()))
                require(maximum <= CONFIG['numeric_tolerances']['weighted_pose_m'], 'Original weighted pose changed beyond tolerance')
        motion.append({'action': name, 'seconds': times.tolist(), 'max_weighted_error_m': maximum,
                       'all_retained_exported_vertices_checked': sum(len(exported[m]['positions']) for m in names),
                       'all_exported_fringe_vertices_checked': len(exported['SourceColoredFringe']['positions'])})
    save('weighted-pose-comparison.json', {'status': 'passed_actual_reserialized_pose_comparison', 'states': motion,
         'tolerance_m': CONFIG['numeric_tolerances']['weighted_pose_m'],
         'limits': ['Nine samples per existing clip verify sampled numerical preservation, not new visual meaning or all-time collision freedom.']})
    # Reuse the bounded production validators after the independent correspondence checks.
    del original, exported, source, target
    import gc
    gc.collect()
    contract = json.loads((ROOT/'outputs/fox-hd-build-v2/rig-contract.json').read_text())
    save('preflight.json', preflight(OUT/'model.glb'))
    save('render-attributes.json', validate_render_attributes(OUT/'model.glb'))
    rig = validate_glb(OUT/'model.glb', contract)
    save('rig-verification.json', rig)
    require(rig['motion']['animation_verified'], 'Actual exported skin motion was not verified')
    save('export-audit.json', {'status': 'passed_component_numerics_pending_visual_review',
        'source_sha256': CONFIG['source_sha256'], 'model_sha256': rig['sha256'],
        'elapsed_seconds': time.monotonic()-START, 'quality_approved': False, 'new_provider_or_ai_role_calls': 0,
        'baseline_exact_byte_preservation': 'inputs/base-no-legacy-cards.glb only',
        'final_export': 'Actual Blender glTF reserialization; numerical correspondences and weighted poses verified, not byte identity',
        'limits': ['No native groom, render, Muse equivalence, fresh prompt acceptance, or Weft compatibility claim.']})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if AUDIT.is_dir() and not (AUDIT/'export-audit-failure.json').exists():
            save('export-audit-failure.json', {'status': 'failed_preserved_no_retry', 'error': str(error), 'type': type(error).__name__})
        raise
