import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ('test_action_intent','test_action_audit','test_action_plan_semantics','test_stationary_language'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}))
raise SystemExit(0 if result.wasSuccessful() else 1)
