"""Read-only first correspondence failure localization; never changes audit thresholds."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import runpy
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREP = ROOT/'outputs/fox-fringe-component-preparation-v2'
START = time.monotonic()
module = runpy.run_path(str(PREP/'audit_export.py'))
source = module['_GLB'](PREP/'inputs/base-no-legacy-cards.glb')
target = module['_GLB'](ROOT/'outputs/fox-fringe-component-v2/model.glb')
bone_names = [source.doc['nodes'][index]['name'] for index in source.doc['skins'][0]['joints']]
before, after = module['records'](source, bone_names), module['records'](target, bone_names)
diagnoses = []
for material in ('coat', 'cream', 'dark'):
    src, dst = before[material], after[material]
    buckets = defaultdict(list)
    for index, cell in enumerate(np.floor(src['positions']/1e-5).astype(np.int64)):
        buckets[tuple(cell)].append(index)
    failures = []
    fields = [('positions', 1e-6), ('uv', 2e-6), ('normals', 2e-4), ('dense_weights', 1e-6)]
    for index, point in enumerate(dst['positions']):
        assert time.monotonic()-START < 25
        base = np.floor(point/1e-5).astype(np.int64)
        candidates = np.array([i for offset in module['OFFSETS'] for i in buckets.get(tuple(base+offset), ())])
        stages = []
        for field, bound in fields:
            if not len(candidates):
                stages.append({'field': field, 'candidates': 0})
                break
            distance = np.max(abs(src[field][candidates]-dst[field][index]), axis=1)
            best = int(candidates[int(distance.argmin())])
            stages.append({'field': field, 'bound': bound, 'candidates_before': len(candidates),
                           'minimum_component_error': float(distance.min()), 'best_source_vertex': best,
                           'source_values': src[field][best].tolist(), 'target_values': dst[field][index].tolist()})
            candidates = candidates[distance <= bound]
        if not len(candidates):
            failures.append({'target_vertex': index, 'position': point.tolist(), 'stages': stages})
            if len(failures) == 3:
                break
    diagnoses.append({'material': material, 'source_vertices': len(src['positions']), 'target_vertices': len(dst['positions']),
                      'source_bounds': [src['positions'].min(axis=0).tolist(), src['positions'].max(axis=0).tolist()],
                      'target_bounds': [dst['positions'].min(axis=0).tolist(), dst['positions'].max(axis=0).tolist()],
                      'first_failures': failures})
result = {'status': 'read_only_failure_localized', 'source_sha256': source.hash, 'model_sha256': target.hash,
          'audit_script_sha256': hashlib.sha256((PREP/'audit_export.py').read_bytes()).hexdigest(),
          'diagnoses': diagnoses, 'elapsed_seconds': time.monotonic()-START,
          'thresholds_changed': False, 'native_provider_ai_role_calls': 0, 'quality_approved': False}
with (HERE/'diagnosis.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
print(json.dumps(result))
