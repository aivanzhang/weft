import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'tests'), str(ROOT)]
prepared = json.loads((HERE/'prepared.json').read_text())
for row in prepared['sources']:
    assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest() == row['sha256']
suite = unittest.defaultTestLoader.loadTestsFromNames(prepared['modules'])
result = unittest.TextTestRunner(verbosity=2).run(suite)
for row in prepared['sources']:
    assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest() == row['sha256']
with (HERE/'result.json').open('x') as stream:
    json.dump({'status': 'passed' if result.wasSuccessful() else 'failed', 'tests': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors),
               'analytic_only': True, 'source_pins_unchanged': True}, stream, indent=2)
    stream.write('\n')
sys.exit(not result.wasSuccessful())
