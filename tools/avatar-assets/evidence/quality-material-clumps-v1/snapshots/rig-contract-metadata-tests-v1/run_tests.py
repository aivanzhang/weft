import json,resource,sys,time,unittest
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
start=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_rig_contract_metadata')
result=unittest.TextTestRunner(verbosity=2).run(suite)
receipt={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),'elapsed_seconds':time.monotonic()-start,'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_scope':'Direct child getrusage high-water RSS on macOS, distinct from bounded monitor samples','native_jobs':0,'provider_calls':0}
with(OUT/'test-result.json').open('x')as stream:json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
sys.exit(0 if result.wasSuccessful()else 1)
