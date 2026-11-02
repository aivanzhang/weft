"""Five focused analytic/mock modules only; no native jobs or generated assets."""
import json
from pathlib import Path
import sys
import time
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
TESTS = ['test_fur_root_sampling.py', 'test_root_sampling_contract.py', 'test_finish_quality_contracts.py', 'test_combed_fur_cards.py', 'test_tuft_fur_cards.py']
started = time.monotonic()
loader = unittest.TestLoader()
suite = unittest.TestSuite(loader.discover(str(ROOT/'tests'), pattern=name) for name in TESTS)
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {'schema_version':1,'kind':'analytic_and_mocked_tests_not_generated_assets',
    'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'skipped':len(result.skipped),'successful':result.wasSuccessful(),
    'elapsed_seconds':time.monotonic()-started,'test_modules':TESTS,
    'native_provider_ai_role_calls':0,'quality_approved':False}
with (HERE/'test-result.json').open('x') as handle:
    json.dump(report,handle,indent=2,sort_keys=True);handle.write('\n')
sys.exit(0 if result.wasSuccessful() else 1)
