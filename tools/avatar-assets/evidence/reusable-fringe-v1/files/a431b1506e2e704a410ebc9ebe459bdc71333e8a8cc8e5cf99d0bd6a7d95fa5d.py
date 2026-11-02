"""Small analytic checks of the prepared correspondence audit, not an actual export."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
for name in ('prepare_baseline.py', 'build_component.py', 'audit_export.py', 'execute.py'):
    ast.parse((HERE/name).read_text())
module = runpy.run_path(str(HERE/'audit_export.py'))
source = {'positions': np.array([[0., 0, 0], [.02, 0, 0], [0, .02, 0]]),
          'normals': np.tile([0., 0, 1], (3, 1)), 'uv': np.array([[0., 0], [1., 0], [0., 1]]),
          'dense_weights': np.ones((3, 1)), 'faces': np.array([[0, 1, 2]])}
order = np.array([2, 0, 1])
target = {key: values[order].copy() if key != 'faces' else np.array([[1, 2, 0]]) for key, values in source.items()}
mapping, errors = module['correspondence'](source, target)
assert np.array_equal(mapping, order) and all(value == 0 for value in errors.values())
rejected = []
for name, replacement in (('uv', np.full((3, 2), .4)), ('normals', -target['normals']),
                          ('dense_weights', np.full((3, 1), .5)), ('faces', target['faces'][:, ::-1])):
    wrong = copy.deepcopy(target); wrong[name] = replacement
    try:
        module['correspondence'](source, wrong)
    except ValueError:
        rejected.append(name)
    else:
        raise AssertionError('Audit admitted '+name+' corruption')
point = {'positions': np.array([[1., 2, 3]]), 'joints': np.zeros((1, 4), dtype=int),
         'weights': np.array([[1., 0, 0, 0]]), 'skin': [0], 'inverse': np.eye(4)[None]}
world = [tuple(np.array([[1., 0, 0, .1], [0, 1., 0, .2], [0, 0, 1., .3], [0, 0, 0, 1.]]).T.ravel())]
assert np.allclose(module['skin_positions'](point, world), [[1.1, 2.2, 3.3]], atol=1e-14, rtol=0)
report = {'status': 'analytic_preparation_checks_passed', 'native_executed': False,
          'permuted_vertex_correspondence': True, 'corruptions_rejected': rejected,
          'column_major_weighted_transform_oracle': True, 'source_sha256': {
              name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
              for name in ('prepare_baseline.py', 'build_component.py', 'audit_export.py', 'execute.py', 'check_preparation.py')},
          'limits': ['Prepared checker semantics only; no actual native import/export, source weighted pose or appearance acceptance.']}
with (HERE/'preparation-check.json').open('x') as stream:
    json.dump(report, stream, indent=2, sort_keys=True); stream.write('\n')
print(json.dumps(report))
