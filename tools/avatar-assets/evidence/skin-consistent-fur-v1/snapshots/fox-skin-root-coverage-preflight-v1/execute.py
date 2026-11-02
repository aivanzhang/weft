from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, sha256, write_json

prior='outputs/fox-vertex-root-coverage-preflight-v1/'
source='outputs/fox-material-fur-finish-v1/'
paths=[
    'outputs/fox-fringe-component-preparation-v2/inputs/base-no-legacy-cards.glb',
    source+'finish-config.json',source+'material-groom/report.json',
    source+'material-groom/correspondence.npz',source+'material-coat.json',
    prior+'source-binding.json',prior+'report.json',prior+'release-manifest.json',prior+'census.py',
    'avatar_assets/runner.py','avatar_assets/safety.py']
inputs=[]
for name in paths:
    p=ROOT/name
    assert p.is_file() and not p.is_symlink()
    inputs.append({'path':name,'bytes':p.stat().st_size,'sha256':sha256(p)})
    if p.suffix=='.py':
        target=OUT/'sources'/('reference_census.py' if name==prior+'census.py' else name)
        target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists()
        shutil.copyfile(p,target)
assert inputs[0]['sha256']=='f34aa798252355fc6e44a03f78cdc7f31eab682a20b591a93c9dcb35ed82f8a3'
assert sha256(OUT/'sources/reference_census.py')=='370ebc05c1a6d4cbd4c68cbd5b12f1d3f1c2afdc542eaf20dfb16fa9f9b27b3c'
for name in ['census.py','execute.py']:
    compile((OUT/name).read_text(),str(OUT/name),'exec')
command=[sys.executable,'-B',str(OUT/'census.py')]
write_json(OUT/'prepared.json',{'schema_version':1,
    'at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
    'inputs':inputs,'scripts':{name:sha256(OUT/name) for name in ['census.py','execute.py']},
    'command':command,'limits':{'attempts':1,'seconds':30,'memory_mb':512,'output_mb':16},
    'hypothesis':'Uniform named skin faces keep area-distributed interior roots; only varying faces require source vertex anchors.',
    'normalization':'Sort positive named influences, math.fsum normalization, exact tuple equality, no cutoff/tolerance.',
    'allocation':'Fixed material 2017/527/456; Hamilton domain quotas, uniform-first ties, one-root transfer only if a nonempty domain would be zero.',
    'no_selector_or_model_execution':True,'original_body_proof_reused_not_rerun':True})
job=run_bounded(command,OUT/'cpu.log',OUT,timeout=30,memory_mb=512,output_mb=16)
write_json(OUT/'cpu-job.json',job)
unchanged=[{'path':x['path'],'unchanged':(ROOT/x['path']).stat().st_size==x['bytes'] and sha256(ROOT/x['path'])==x['sha256']} for x in inputs]
result=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').is_file() else None
passed=job.get('exit_code')==0 and not job.get('failure') and job.get('process_exit_verified') and result is not None and all(x['unchanged'] for x in unchanged)
write_json(OUT/'receipt.json',{'schema_version':1,'status':'measured' if passed else 'failed','job':job,
    'source_unchanged':unchanged,'report_sha256':sha256(OUT/'report.json') if result else None,
    'native_jobs':0,'provider_calls':0,'role_calls':0,'quality_approved':False})
write_json(OUT/'manifest.json',{'schema_version':1,'kind':'skin_consistent_root_coverage_preflight',
    'status':'measured' if passed else 'failed','artifacts':artifacts(OUT)})
print(json.dumps({'status':'measured' if passed else 'failed','job':job,'manifest_sha256':sha256(OUT/'manifest.json')},sort_keys=True))
raise SystemExit(0 if passed else 1)
