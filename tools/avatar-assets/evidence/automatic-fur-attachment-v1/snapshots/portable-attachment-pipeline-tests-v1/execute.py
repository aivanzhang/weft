from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets import pipeline
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts,sha256,write_json

paths={str(path.relative_to(ROOT)) for path in pipeline._pipeline_sources().values()}
paths.update(['tests/test_portable_attachment_pipeline.py','avatar_assets/studio_jobs.py',
              'avatar_assets/runner.py','avatar_assets/safety.py','avatar_assets/cli.py',
              'avatar_assets/gallery.py','avatar_assets/__init__.py'])
inputs=[]
for name in sorted(paths):
    source=ROOT/name
    target=OUT/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
    assert source.is_file() and not source.is_symlink() and not target.exists()
    shutil.copyfile(source,target)
    inputs.append({'path':name,'bytes':source.stat().st_size,'sha256':sha256(source)})
for name in ('run_tests.py','execute.py'):compile((OUT/name).read_text(),str(OUT/name),'exec')
(OUT/'tmp').mkdir()
command=['/usr/bin/env','PYTHONDONTWRITEBYTECODE=1','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
         'TMPDIR='+str(OUT/'tmp'),sys.executable,'-B',str(OUT/'run_tests.py')]
write_json(OUT/'prepared.json',{'schema_version':1,'at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
    'command':command,'equivalent_test_command':'.venv/bin/python -m unittest -v tests.test_portable_attachment_pipeline',
    'inputs':inputs,'scripts':{name:sha256(OUT/name) for name in ('run_tests.py','execute.py')},
    'limits':{'seconds':30,'memory_mb':512,'output_mb':16,'native_jobs':0,'provider_calls':0,'agent_calls':0},
    'scope':'Control-flow and hash-custody fixtures. Numeric validator/worker seam is explicitly analytic; independent real auditor tests remain separate.'})
job=run_bounded(command,OUT/'cpu.log',OUT,timeout=30,memory_mb=512,output_mb=16)
write_json(OUT/'cpu-job.json',job)
unchanged=[{'path':x['path'],'unchanged':(ROOT/x['path']).stat().st_size==x['bytes'] and sha256(ROOT/x['path'])==x['sha256']} for x in inputs]
result=json.loads((OUT/'test-result.json').read_text()) if (OUT/'test-result.json').is_file() else None
passed=job.get('exit_code')==0 and not job.get('failure') and job.get('process_exit_verified') and result and result['passed'] and all(x['unchanged'] for x in unchanged)
write_json(OUT/'receipt.json',{'schema_version':1,'status':'passed' if passed else 'failed','job':job,
    'test_result':result,'source_unchanged':unchanged,'native_jobs':0,'provider_calls':0,'agent_calls':0,
    'limitation':'No real attachment, renderer or avatar quality verdict is produced by these orchestration fixtures.'})
write_json(OUT/'manifest.json',{'schema_version':1,'kind':'portable_attachment_pipeline_tests',
    'status':'passed' if passed else 'failed','artifacts':artifacts(OUT)})
print(json.dumps({'status':'passed' if passed else 'failed','tests':result,'job':job,
                  'manifest_sha256':sha256(OUT/'manifest.json')},sort_keys=True))
raise SystemExit(0 if passed else 1)
