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

candidate='outputs/fox-skin-consistent-fur-finish-v1/'
old='outputs/fox-material-fur-finish-v1/'
prior='outputs/fox-skin-root-coverage-preflight-v1/'
original='outputs/fox-vertex-root-coverage-preflight-v1/'
report=json.loads((ROOT/candidate/'finish-report.json').read_text())
archives=[candidate+p['root_summary']['binding_archive']['path'] for p in report['portable_fur']]
paths=[candidate+x for x in ['model.glb','finish-report.json','finish-config.json','material-coat.json',
    'material-groom/report.json','material-groom/correspondence.npz',
    'sources/blender_fur_cards.py','sources/blender_material_coat.py']]+archives+[
    'outputs/fox-fringe-component-preparation-v2/inputs/base-no-legacy-cards.glb',
    old+'material-groom/report.json',old+'material-coat.json',
    prior+'report.json',prior+'classification.npz',prior+'release-manifest.json',
    original+'source-binding.json',original+'spatial-grid.json',original+'census.py',
    'avatar_assets/runner.py','avatar_assets/safety.py']
inputs=[]
for name in paths:
    p=ROOT/name
    assert p.is_file() and not p.is_symlink()
    inputs.append({'path':name,'bytes':p.stat().st_size,'sha256':sha256(p)})
    if p.suffix=='.py':
        target=OUT/'sources'/('reference_census.py' if name==original+'census.py' else name)
        target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists();shutil.copyfile(p,target)
model_hash=inputs[0]['sha256']
assert model_hash=='e08bd6e0e548e1e89c6a46b811ab9d5fa11b8c9e15844d50f5ec480455171d80'
for name in ['measure.py','execute.py']:compile((OUT/name).read_text(),str(OUT/name),'exec')
command=[sys.executable,'-B',str(OUT/'measure.py')]
write_json(OUT/'prepared.json',{'schema_version':1,'at':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
    'model_sha256':model_hash,'inputs':inputs,
    'scripts':{name:sha256(OUT/name) for name in ['measure.py','execute.py']},
    'command':command,'limits':{'attempts':1,'seconds':30,'memory_mb':512,'output_mb':16},
    'scope':'Selected binding root counts, exact coincident aliases and component/grid allocation. No repeated exported body/skin/color audit.',
    'settings':{'grid':[6,6,6],'root_count':3000,'expected_uniform':2518,'expected_vertex':482,
                'texture_decode':False,'final_glb_attribute_decode':False,'native_jobs':0,'provider_calls':0}})
job=run_bounded(command,OUT/'cpu.log',OUT,timeout=30,memory_mb=512,output_mb=16)
write_json(OUT/'cpu-job.json',job)
unchanged=[{'path':x['path'],'unchanged':(ROOT/x['path']).stat().st_size==x['bytes'] and sha256(ROOT/x['path'])==x['sha256']} for x in inputs]
result=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').is_file() else None
passed=job.get('exit_code')==0 and not job.get('failure') and job.get('process_exit_verified') and result is not None and all(x['unchanged'] for x in unchanged)
write_json(OUT/'receipt.json',{'schema_version':1,'status':'measured' if passed else 'failed',
    'job':job,'source_unchanged':unchanged,'report_sha256':sha256(OUT/'report.json') if result else None,
    'native_jobs':0,'provider_calls':0,'quality_approved':False})
write_json(OUT/'manifest.json',{'schema_version':1,'kind':'selected_skin_consistent_root_coverage',
    'status':'measured' if passed else 'failed','artifacts':artifacts(OUT)})
print(json.dumps({'status':'measured' if passed else 'failed','job':job,
                  'manifest_sha256':sha256(OUT/'manifest.json')},sort_keys=True))
raise SystemExit(0 if passed else 1)
