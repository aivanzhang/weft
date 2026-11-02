import json
from pathlib import Path
import resource
import sys
import time
import unittest

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
started=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_fur_bindings')
result=unittest.TextTestRunner(verbosity=2).run(suite)
value={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),
       'errors':len(result.errors),'skipped':len(result.skipped),'elapsed_seconds':time.monotonic()-started,
       'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
       'scope':'Numeric NPZ and analytic filesystem custody only; no native/model/provider calls.'}
with (OUT/'test-result.json').open('x') as stream:
    json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
