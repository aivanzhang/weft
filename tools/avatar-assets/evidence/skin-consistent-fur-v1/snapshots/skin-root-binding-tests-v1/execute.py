from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts,sha256,write_json

inputs=[]
for name in ('avatar_assets/fur_bindings.py','tests/test_fur_bindings.py',
             'avatar_assets/runner.py','avatar_assets/safety.py','avatar_assets/__init__.py'):
    path=ROOT/name
    target=OUT/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists();shutil.copyfile(path,target)
    inputs.append({'path':name,'sha256':sha256(path),'bytes':path.stat().st_size})
for name in ('run_tests.py','execute.py'):
    compile((OUT/name).read_text(),str(OUT/name),'exec')
(OUT/'tmp').mkdir()
command=['/usr/bin/env','PYTHONDONTWRITEBYTECODE=1','OPENBLAS_NUM_THREADS=1',
         'OMP_NUM_THREADS=1','TMPDIR='+str(OUT/'tmp'),sys.executable,'-B',str(OUT/'run_tests.py')]
write_json(OUT/'prepared.json',{'schema_version':1,
    'at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),'command':command,
    'equivalent_test_command':'.venv/bin/python -m unittest -v tests.test_fur_bindings',
    'inputs':inputs,'scripts':{n:sha256(OUT/n) for n in ('run_tests.py','execute.py')},
    'limits':{'seconds':30,'memory_mb':512,'output_mb':16,'native_jobs':0,'provider_calls':0},
    'scope':'Writer numeric contracts and byte-only copy custody, no native or semantic model approval.'})
job=run_bounded(command,OUT/'cpu.log',OUT,timeout=30,memory_mb=512,output_mb=16)
write_json(OUT/'cpu-job.json',job)
unchanged=[{'path':x['path'],'unchanged':(ROOT/x['path']).stat().st_size==x['bytes'] and sha256(ROOT/x['path'])==x['sha256']} for x in inputs]
result=json.loads((OUT/'test-result.json').read_text()) if (OUT/'test-result.json').is_file() else None
passed=job.get('exit_code')==0 and not job.get('failure') and job.get('process_exit_verified') and result and result['passed'] and all(x['unchanged'] for x in unchanged)
write_json(OUT/'receipt.json',{'schema_version':1,'status':'passed' if passed else 'failed','job':job,
    'test_result':result,'source_unchanged':unchanged,'native_jobs':0,'provider_calls':0,'role_calls':0,
    'copy_semantic_validation':False,'limitations':['Copy checks byte/hash custody only; the independent bounded loader must validate array semantics.']})
write_json(OUT/'manifest.json',{'schema_version':1,'kind':'skin_root_binding_tests',
    'status':'passed' if passed else 'failed','artifacts':artifacts(OUT)})
print(json.dumps({'status':'passed' if passed else 'failed','tests':result,'job':job,
                  'manifest_sha256':sha256(OUT/'manifest.json')},sort_keys=True))
raise SystemExit(0 if passed else 1)
