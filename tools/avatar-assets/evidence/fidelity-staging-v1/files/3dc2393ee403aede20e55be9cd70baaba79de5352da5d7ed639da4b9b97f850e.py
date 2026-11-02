import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'sources/host'))
from avatar_assets.runner import run_bounded
p=json.loads((HERE/'prepared.json').read_text())
for row in p['source_pins']:
    f=HERE/row['path']
    with f.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==row['sha256'] and f.stat().st_size==row['bytes']
for row in p['installed_pins']:
    f=ROOT/row['path']
    with f.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256']
with (HERE/'started.json').open('x') as f:json.dump({'prepared_sha256':hashlib.sha256((HERE/'prepared.json').read_bytes()).hexdigest(),'one_attempt':True,'authorization':p['authorization']},f,indent=2);f.write('\n')
r=run_bounded(p['native_command'],HERE/'native.log',HERE,timeout=60,memory_mb=1024,output_mb=64)
with (HERE/'native-job.json').open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'stage':'native','job':r}),flush=True)
if r.get('exit_code')!=0 or r.get('failure'):sys.exit(1)
c=run_bounded(p['checker_command'],HERE/'checker.log',HERE,timeout=15,memory_mb=512,output_mb=64)
with (HERE/'checker-job.json').open('x') as f:json.dump(c,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'stage':'checker','job':c}),flush=True)
sys.exit(0 if c.get('exit_code')==0 and not c.get('failure') else 1)
