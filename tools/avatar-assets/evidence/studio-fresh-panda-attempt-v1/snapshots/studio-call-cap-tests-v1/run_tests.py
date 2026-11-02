import json
from pathlib import Path
import resource
import sys
import time
import unittest

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
started = time.monotonic()
suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_studio_jobs')
result = unittest.TextTestRunner(verbosity=2).run(suite)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
receipt = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
           'passed': result.wasSuccessful(), 'elapsed_seconds': time.monotonic()-started,
           'self_peak_rss_bytes': peak if sys.platform == 'darwin' else peak*1024,
           'scope': 'Analytic temporary Studio lifecycle fixtures; no AI, native, provider, server or browser calls.'}
(OUT/'test-result.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
