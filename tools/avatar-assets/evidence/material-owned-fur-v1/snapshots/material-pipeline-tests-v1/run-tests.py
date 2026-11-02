from pathlib import Path
import json, sys, time, unittest
root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root),str(root/'tests')]
started=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromNames(['test_material_pipeline', 'test_pipeline_prompts', 'test_quality_policy', 'test_pipeline.PipelineTests.test_prepare_is_free_bounded_and_preserves_request', 'test_pipeline.PipelineTests.test_request_and_backend_tampering_block_resume'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
receipt={'kind':'analytic_software_tests','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),'elapsed_seconds':time.monotonic()-started,'generated_asset_quality_verified':False,'provider_calls':0,'native_calls':0,'role_calls':0}
with Path(__file__).with_name('result.json').open('x') as h:json.dump(receipt,h,sort_keys=True,indent=2);h.write('\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
