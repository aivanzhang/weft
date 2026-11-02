"""Full CPU-only fidelity localization; the original normal gate stays failed."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import runpy
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREP = ROOT/'outputs/fox-fringe-component-preparation-v2'
MODEL = ROOT/'outputs/fox-fringe-component-v2'
START = time.monotonic()


def save(name, value):
    with (HERE/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def deadline():
    if time.monotonic()-START > 27:
        raise ValueError('Full localization27second internal deadline')


def main():
    module = runpy.run_path(str(PREP/'audit_export.py'))
    source = module['_GLB'](PREP/'inputs/base-no-legacy-cards.glb')
    target = module['_GLB'](MODEL/'model.glb')
    bones = [source.doc['nodes'][index]['name'] for index in source.doc['skins'][0]['joints']]
    before, after = module['records'](source, bones), module['records'](target, bones)
    before['SourceColoredFringe'], corner_proof = module['helper_record'](source, bones)
    rows = []
    thresholds = [('positions', 1e-6), ('uv', 2e-6), ('dense_weights', 1e-6)]
    for ordinal, material in enumerate(before):
        deadline()
        src, dst = before[material], after[material]
        buckets = defaultdict(list)
        for index, cell in enumerate(np.floor(src['positions']/1e-5).astype(np.int64)):
            buckets[tuple(cell)].append(index)
        metrics = {name: 0. for name, _ in thresholds}
        normal_errors, normal_angles, ambiguity_spreads, worst = [], [], [], []

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
               'acceptance': False}
        rows.append(row)
        save(f'material-{ordinal:02d}.json', row)
    native = {name: json.loads((MODEL/name).read_text()) for name in
              ('native-source-before.json', 'native-source-after.json', 'native-source-after-export.json')}
    first, second, third = native.values()
    assert first['mesh'] == second['mesh'] == third['mesh'] and first['rig'] == second['rig'] == third['rig']
    result = {'status': 'full_read_only_localization_original_gate_failed', 'source_original_sha256': module['CONFIG']['source_sha256'],
              'baseline_sha256': source.hash, 'model_sha256': target.hash, 'native_before_after_sha256': {name: sha(MODEL/name) for name in native},
              'raw_export_sha256': sha(MODEL/'raw-export.glb'), 'native_mesh_rig_actions_identical_before_after': True,
              'native_corner_mapping': corner_proof, 'materials': rows, 'elapsed_seconds': time.monotonic()-START,
              'normal_threshold_changed': False, 'normals_patched': False, 'native_rerun': False, 'accepted': False,
              'limits': ['Identical native pre/post attributes do not imply source GLB and reserialized GLB normals are identical.',
                         'The original0.0002component gate remains failed; diagnostic matching without normals is not acceptance.',
                         'No new source texture decode, render, weighted-pose audit, or appearance review was performed.']}
    save('diagnosis.json', result)
    print(json.dumps({'status': result['status'], 'materials': len(rows), 'normal_violations': sum(r['normal_comparison']['over_original_limit'] for r in rows),
                      'all_independent_geometry_matches': all(r['matching_without_normals']['complete'] and r['matching_without_normals']['oriented_triangle_multiset_equal'] for r in rows),
                      'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
