"""Read-only panda helper/export attribute diagnosis; no rendering or acceptance."""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources/host'))
from avatar_assets.three_d import _GLB
START = time.monotonic()


def save(name, value):
    with (HERE/('position-uv-'+name)).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def deadline():
    if time.monotonic()-START > 27:
        raise ValueError('Panda localization27second internal deadline')


def array(glb, index, shape, types, indices=False):
    return np.asarray(glb.accessor(index, shape, types, count_limit=300000 if indices else 100000))


def main():
    inputs = json.loads((HERE/'inputs.json').read_text())
    for row in inputs['inputs']:
        path = ROOT/row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Frozen input changed')
    target = _GLB(ROOT/inputs['inputs'][0]['path'])
    matches = []
    for ni, node in enumerate(target.doc['nodes']):
        if 'mesh' not in node:
            continue
        for pi, prim in enumerate(target.doc['meshes'][node['mesh']]['primitives']):
            if target.doc['materials'][prim['material']].get('name') == 'SourceColoredFringe':
                matches.append((ni, node, pi, prim))
    require(len(matches) == 1, 'Need exactly one actual fringe primitive')
    node_id, node, primitive_id, prim = matches[0]
    require(prim.get('mode', 4) == 4 and 'skin' in node, 'Expected skinned triangles')
    skin = target.doc['skins'][node['skin']]
    bones = [target.doc['nodes'][i]['name'] for i in skin['joints']]
    require(0 < len(bones) <= 64 and len(set(bones)) == len(bones), 'Invalid actual exported skin names')
    attrs = prim['attributes']
    positions = array(target, attrs['POSITION'], 'VEC3', (5126,))
    normals = array(target, attrs['NORMAL'], 'VEC3', (5126,))
    uv = array(target, attrs['TEXCOORD_0'], 'VEC2', (5126,))
    joints = array(target, attrs['JOINTS_0'], 'VEC4', (5121, 5123)).astype(np.int32)
    weights = array(target, attrs['WEIGHTS_0'], 'VEC4', (5121, 5123, 5126))
    faces = array(target, prim['indices'], 'SCALAR', (5121, 5123, 5125), True).reshape(-1, 3).astype(np.int32)
    require(len(positions) <= 57600 and np.all((joints >= 0) & (joints < len(bones))), 'Invalid fringe bounds')
    dense = np.zeros((len(positions), len(bones)))
    for i in range(len(bones)):
        dense[:, i] = np.where(joints == i, weights, 0).sum(axis=1)
    after = {'SourceColoredFringe': {'positions': positions, 'normals': normals, 'uv': uv,
        'dense_weights': dense, 'faces': faces}}
    path = ROOT/inputs['inputs'][1]['path']
    require(path.stat().st_size <= 8*1024**2, 'Fringe archive file bound')
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) <= 20 and sum(row.file_size for row in archive.infolist()) <= 16*1024**2,
                'Fringe decoded-array bound')
    with np.load(path, allow_pickle=False) as saved:
        points, normal = saved['positions'].copy(), saved['normals'].copy()
        triangles, native_map = saved['triangles'].copy(), saved['native_vertex_to_helper_vertex'].copy()
        tex, ids, influence = saved['uv_gltf'].copy(), saved['joints'].copy(), saved['weights'].copy()
    require(1000*4 <= len(points) <= 4800*4 and len(points) % 4 == 0, 'Invalid fixed fin count')
    require(np.array_equal(native_map, triangles.reshape(-1)) and len(native_map) == len(points)*3,
            'Independent native corner map differs from helper triangle order')
    report = json.loads((ROOT/inputs['inputs'][2]['path']).read_text())
    helper_bones = report['bone_order']
    require(len(helper_bones) == len(set(helper_bones)) and set(helper_bones) == set(bones), 'Actual helper/GLB skin names differ')
    require(report['cards']*4 == len(points) and report['vertices'] == len(native_map), 'Actual report count mismatch')
    require(points.shape == normal.shape == (len(points),3) and tex.shape == (len(points),2)
            and ids.shape == influence.shape == (len(points),4), 'Malformed actual helper arrays')
    require(np.all((ids >= 0) & (ids < len(helper_bones))) and np.all((triangles >= 0) & (triangles < len(points))),
            'Actual helper indices outside bound')
    for value in (points, normal, tex, influence):
        require(np.all(np.isfinite(value)), 'Nonfinite actual helper arrays')
    helper_dense = np.zeros((len(points),len(bones)))
    for i, name in enumerate(helper_bones):
        helper_dense[:, bones.index(name)] = np.where(ids == i, influence, 0).sum(axis=1)
    # Helper rest arrays use Blender Z-up; exported geometry uses glTF Y-up.
    points, normal = points[:,[0,2,1]].copy(), normal[:,[0,2,1]].copy()
    points[:,2] *= -1; normal[:,2] *= -1
    before = {'SourceColoredFringe': {'positions': points, 'normals': normal, 'uv': tex,
        'dense_weights': helper_dense, 'faces': triangles}}
    module = {'OFFSETS': np.array(list(itertools.product((-1,0,1),repeat=3)),dtype=np.int64)}
    corner_proof = {'helper_vertices': len(points), 'native_independent_corners': len(native_map),
        'exact_native_corner_map': True, 'native_map_sha256': hashlib.sha256(native_map.tobytes()).hexdigest(),
        'helper_bone_order': helper_bones, 'actual_exported_bone_order': bones,
        'helper_to_exported_bone_indices': [bones.index(n) for n in helper_bones],
        'actual_node_index': node_id, 'actual_skin_index': node['skin'], 'actual_primitive_index': primitive_id,
        'target_attributes_present_not_all_validated': sorted(attrs)}
    rows = []
    thresholds = [('positions', 1e-6), ('uv', 2e-6)]
    for ordinal, material in enumerate(before):
        deadline()
        src, dst = before[material], after[material]
        buckets = defaultdict(list)
        for index, cell in enumerate(np.floor(src['positions']/1e-5).astype(np.int64)):
            buckets[tuple(cell)].append(index)
        metrics = {name: 0. for name, _ in thresholds}
        normal_errors, normal_angles, ambiguity_spreads, worst = [], [], [], []
        weight_errors, weight_worst, weight_ambiguities = [], [], []

        def match(values, measure):
            ids = np.full(len(values['positions']), -1, dtype=np.int32)
            histogram = Counter()
            missing = []
            for index, point in enumerate(values['positions']):
                if index % 256 == 0:
                    deadline()
                base = np.floor(point/1e-5).astype(np.int64)
                candidate = np.array([i for offset in module['OFFSETS'] for i in buckets.get(tuple(base+offset), ())], dtype=np.int32)
                if len(candidate) > 256:
                    raise ValueError('Correspondence bucket exceeds256vertices')
                for name, bound in thresholds:
                    if not len(candidate):
                        break
                    distance = np.max(abs(src[name][candidate]-values[name][index]), axis=1)
                    candidate = candidate[distance <= bound]
                histogram[len(candidate)] += 1
                if not len(candidate):
                    if len(missing) < 8:
                        missing.append({'vertex': index, 'position': point.tolist()})
                    continue
                ids[index] = int(candidate.min())
                if measure:
                    difference = np.max(abs(src['dense_weights'][candidate]-values['dense_weights'][index]), axis=1)
                    closest_weight = int(np.argmin(difference))
                    weight_error = float(difference[closest_weight])
                    weight_errors.append(weight_error)
                    weight_ambiguities.append(float(np.ptp(src['dense_weights'][candidate],axis=0).max()) if len(candidate)>1 else 0.)
                    if weight_error > 1e-6:
                        weight_worst.append({'target_vertex':index,'closest_source_vertex':int(candidate[closest_weight]),
                            'maximum_component_error':weight_error,'source_dense_weights':src['dense_weights'][candidate[closest_weight]].tolist(),
                            'target_dense_weights':values['dense_weights'][index].tolist()})
                    for name, _ in thresholds:
                        metrics[name] = max(metrics[name], float(np.max(abs(src[name][ids[index]]-values[name][index]))))
                    component = np.max(abs(src['normals'][candidate]-values['normals'][index]), axis=1)
                    closest = int(np.argmin(component))
                    value = float(component[closest])
                    normal = src['normals'][candidate[closest]]
                    other = values['normals'][index]
                    angle = float(np.degrees(np.arccos(np.clip(normal@other/(np.linalg.norm(normal)*np.linalg.norm(other)), -1, 1))))
                    normal_errors.append(value); normal_angles.append(angle)
                    spread = float(np.ptp(src['normals'][candidate], axis=0).max()) if len(candidate) > 1 else 0.
                    ambiguity_spreads.append(spread)
                    if value > .0002:
                        worst.append({'target_vertex': index, 'closest_source_vertex': int(candidate[closest]),
                                      'eligible_correspondences': len(candidate), 'component_error': value,
                                      'angular_error_degrees': angle, 'source_normal': normal.tolist(),
                                      'target_normal': other.tolist()})
            return ids, {'candidate_count_histogram': dict(sorted(histogram.items())),
                         'missing_count': histogram.get(0, 0), 'first_missing': missing,
                         'ambiguous_vertices': sum(count for size, count in histogram.items() if size > 1)}

        source_ids, source_matching = match(src, False)
        target_ids, target_matching = match(dst, True)
        complete = bool(np.all(source_ids >= 0) and np.all(target_ids >= 0))

        def oriented(record, ids):
            counter = Counter()
            for a, b, c in ids[record['faces']].tolist():
                counter[min((a, b, c), (b, c, a), (c, a, b))] += 1
            return counter

        topology_equal = bool(complete and oriented(src, source_ids) == oriented(dst, target_ids))
        normal_errors = np.asarray(normal_errors)
        normal_angles = np.asarray(normal_angles)
        row = {'material': material, 'reference_kind': 'actual_helper_arrays' if material == 'SourceColoredFringe' else 'original_GLB_base',
               'source_vertices': len(src['positions']), 'exported_vertices': len(dst['positions']),
               'matching_without_normals': {'source': source_matching, 'target': target_matching, 'complete': complete,
                   'thresholds': dict(thresholds), 'maximum_selected_errors': metrics,
                   'oriented_triangle_multiset_equal': topology_equal,
                   'source_attribute_groups_covered': bool(complete and set(source_ids) == set(target_ids))},
               'normal_comparison': {'samples': len(normal_errors),
                   'maximum_component_error': float(normal_errors.max()) if len(normal_errors) else None,
                   'p99_component_error': float(np.quantile(normal_errors, .99)) if len(normal_errors) else None,
                   'maximum_angular_error_degrees': float(normal_angles.max()) if len(normal_angles) else None,
                   'original_component_limit': .0002, 'over_original_limit': int(np.count_nonzero(normal_errors > .0002)),
                   'ambiguous_matches_with_different_source_normals': int(np.count_nonzero(np.asarray(ambiguity_spreads) > 0)),
                   'maximum_source_normal_ambiguity_component_spread': max(ambiguity_spreads, default=0.),
                   'policy': 'Normal is excluded from correspondence. Report closest candidate normal as a lower bound where position/UV/weight aliases are ambiguous.',
                   'worst_violations': sorted(worst, key=lambda item: -item['component_error'])[:3]},
               'weight_comparison': {'samples': len(weight_errors), 'original_component_limit': 1e-6,
                    'maximum_component_error': max(weight_errors,default=0.),
                    'over_original_limit': sum(v>1e-6 for v in weight_errors),
                    'ambiguous_matches_with_different_source_weights':sum(v>0 for v in weight_ambiguities),
                    'worst_violations':sorted(weight_worst,key=lambda r:-r['maximum_component_error'])[:3],
                    'policy':'Weights excluded only to localize original-gate failures; closest candidate is a lower bound for aliases.'},
               'acceptance': False}
        rows.append(row)
        save(f'material-{ordinal:02d}.json', row)
    result = {'status': 'supplemental_position_uv_only_localization_original_gates_unchanged', 'inputs': inputs['inputs'],
        'model_sha256': target.hash, 'native_corner_mapping': corner_proof, 'materials': rows,
        'elapsed_seconds': time.monotonic()-START, 'normal_threshold_changed': False, 'weight_threshold_changed': False, 'matching_excludes_weights_for_diagnosis': True,
        'normals_patched': False, 'native_rerun': False, 'accepted': False,
        'limits': ['Raw retained derivative is from a memory-rejected native build; this diagnosis is not delivery acceptance.',
            'The original0.0002component gate is unchanged; matching without normals is diagnosis only.',
            'No texture decode, tangent inspection or omission, render, weighted-pose audit, or appearance review was performed.',
            'Only actual saved helper rest arrays and the raw exported fringe primitive are compared; original body is outside scope.']}
    save('diagnosis.json', result)
    print(json.dumps({'status': result['status'], 'materials': len(rows),
        'normal_violations': sum(r['normal_comparison']['over_original_limit'] for r in rows),
        'all_independent_geometry_matches': all(r['matching_without_normals']['complete'] and r['matching_without_normals']['oriented_triangle_multiset_equal'] for r in rows),
        'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('failure.json', {'type': type(error).__name__, 'message': str(error), 'accepted': False})
        raise
