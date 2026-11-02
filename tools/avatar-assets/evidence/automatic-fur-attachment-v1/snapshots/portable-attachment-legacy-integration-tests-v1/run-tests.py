from pathlib import Path
import json, sys, time, unittest
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root),str(root/'tests')]
started=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromNames(['test_material_native_reuse', 'test_skin_root_integration', 'test_generated_workflow', 'test_pipeline_progress'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
receipt={'kind':'analytic_software_tests','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),'elapsed_seconds':time.monotonic()-started,'native_calls':0,'provider_calls':0,'role_calls':0,'actual_avatar_attachment_verified':False,'quality_verified':False}
with Path(__file__).with_name('result.json').open('x') as handle:json.dump(receipt,handle,sort_keys=True,indent=2);handle.write('\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
