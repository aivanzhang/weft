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
names=['test_custody_failure_stops_without_repair_call','test_new_delivery_and_critic_bind_the_validated_audit',
       'test_numeric_failure_repairs_once_with_witnesses_before_media_or_critic',
       'test_pass_completes_gate_before_native_reload','test_unsupported_mapping_stops_without_repair_call']
suite=unittest.defaultTestLoader.loadTestsFromNames([
    'tests.test_portable_attachment_pipeline.AttachmentPipelineTests.'+name for name in names])
result=unittest.TextTestRunner(verbosity=2).run(suite)
value={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),
       'errors':len(result.errors),'skipped':len(result.skipped),'elapsed_seconds':time.monotonic()-started,
       'self_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
       'scope':'Analytic orchestration, custody and Studio projection. No provider, agent-role, numeric asset or native job.'}
value['selected_tests']=names
with (OUT/'test-result.json').open('x') as stream:
    json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
