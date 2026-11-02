import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sources'))
from avatar_assets.runner import run_bounded
def sha(p):
 with Path(p).open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def write(name,value):
 with (HERE/name).open('x') as h:json.dump(value,h,indent=2,sort_keys=True,allow_nan=False);h.write('\n')
p=json.loads((HERE/'prepared.json').read_text())
for r in p['inputs']+p['sources']:
 if sha(r['path'])!=r['sha256']:raise ValueError('Pinned source changed: '+r['path'])
command=[p['python'],'-m','unittest','discover','-s',str(HERE/'sources/tests'),'-t',str(HERE/'sources'),'-p','test_portable_fringe_contracts.py','-v']
job=run_bounded(command,HERE/'tests.log',HERE,timeout=40,memory_mb=512,output_mb=16)
write('job.json',job)
unchanged=all(sha(r['path'])==r['sha256'] for r in p['inputs']+p['sources'])
passed=job['exit_code']==0 and not job['failure'] and unchanged
artifacts=[{'path':str(f.relative_to(HERE)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(HERE.rglob('*')) if f.is_file()]
write('manifest.json',{'schema_version':1,'passed':passed,'inputs_unchanged':unchanged,'artifacts':artifacts,'native_calls':0,'provider_calls':0,'quality_approved':False})
print(json.dumps({'passed':passed,'job':job,'manifest_sha256':sha(HERE/'manifest.json')}))
sys.exit(0 if passed else 1)
