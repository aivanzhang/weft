from pathlib import Path
import json, sys, time, unittest
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root),str(root/'tests')]
started=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromName('test_fur_skin_roots')
result=unittest.TextTestRunner(verbosity=2).run(suite)
receipt={'kind':'analytic_software_tests','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),'elapsed_seconds':time.monotonic()-started,'actual_avatar_attachment_verified':False,'quality_verified':False,'native_calls':0,'provider_calls':0,'role_calls':0}
with Path(__file__).with_name('result.json').open('x') as h:json.dump(receipt,h,sort_keys=True,indent=2);h.write('\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
