import json
from pathlib import Path
import re
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts,require,sha256,write_json

prepared=json.loads((OUT/'prepared.json').read_text())
for row in prepared['inputs']:
    require(sha256(ROOT/row['path'])==row['sha256']==sha256(OUT/row['snapshot']),'Changed frozen test source')
job=run_bounded(prepared['command'],OUT/'tests.log',OUT,timeout=30,memory_mb=256,output_mb=16)
write_json(OUT/'job.json',job)
unchanged=all(sha256(ROOT/row['path'])==row['sha256'] for row in prepared['inputs'])
log=(OUT/'tests.log').read_text();match=re.search(r'Ran (\d+) tests? in ([0-9.]+)s',log)
receipt={'schema_version':1,'status':'passed' if job['exit_code']==0 and not job['failure'] and unchanged else 'failed',
    'tests':int(match[1]) if match else None,'unittest_seconds':float(match[2]) if match else None,
    'job':job,'sources_unchanged':unchanged,'test_set':prepared['test_set'],
    'analytic_fixture_only':True,'native_jobs':0,'paid_calls':0,'guard_tests_rerun':False,
    'quality_approved':False,'production_edits_in_this_lane':False}
write_json(OUT/'receipt.json',receipt)
write_json(OUT/'manifest.json',{'schema_version':1,'quality_approved':False,'artifacts':artifacts(OUT)})
print(json.dumps({k:receipt[k] for k in ('status','tests','unittest_seconds','sources_unchanged')}))
print(json.dumps(job,sort_keys=True))
raise SystemExit(0 if receipt['status']=='passed' and receipt['tests']==prepared['expected_test_methods'] else 1)
